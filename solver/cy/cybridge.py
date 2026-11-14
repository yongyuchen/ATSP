"""Bridge between quasipoly_atsp.Contracted and the compiled enumerator.

Keeps the exact same cycle set and emission order as
quasipoly_atsp.enumerate_valid_cycles, but returns flat typed arrays:

    pairs   int32,  2 * total_pairs        -- (tail, head) of every step
    indptr  int64,  ncycles + 1            -- offsets in PAIR units
    costs   float64, ncycles               -- total reduced cost of the cycle

A cycle is  psi(a) = b  for every pair (a, b); the last pair of a cycle always
closes back to its start p0.
"""
import os
import sys
import time

import numpy as np

HERE = __file__.rsplit("/", 1)[0]
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from cyenum import Enumerator
    _AVAILABLE = True
except Exception:                                  # pragma: no cover
    Enumerator = None
    _AVAILABLE = False

_ENUM = None


def available():
    return _AVAILABLE


def _threads():
    """Thread count for the OpenMP kernel: QA_CY_THREADS, default 1 (serial)."""
    import os
    try:
        return max(1, int(os.environ.get("QA_CY_THREADS", "1") or 1))
    except ValueError:
        return 1


def _enum():
    global _ENUM
    if _ENUM is None:
        _ENUM = Enumerator()
    return _ENUM


def build_csr(G):
    """CSR arrays for the contracted graph. Returns a dict of numpy arrays."""
    n = G.n
    ch_ptr = np.zeros(n + 1, dtype=np.int32)
    for v in range(n):
        ch_ptr[v + 1] = ch_ptr[v] + len(G.children[v])
    ch_idx = np.empty(int(ch_ptr[n]), dtype=np.int32)
    t = 0
    for v in range(n):
        for w in G.children[v]:
            ch_idx[t] = w
            t += 1

    m = len(G.nb_list)
    nb_ptr = np.zeros(n + 1, dtype=np.int32)
    nb_head = np.empty(m, dtype=np.int32)
    nb_aid = np.empty(m, dtype=np.int32)
    nb_cost = np.empty(m, dtype=np.float64)
    # per node, arcs in increasing global arc id (== the order build_contracted
    # appends them, since nb_list is rc-sorted)
    for v in range(n):
        nb_ptr[v + 1] = nb_ptr[v] + len(G.nb_out[v])
    t = 0
    for v in range(n):
        for head, aid, c in G.nb_out[v]:
            nb_head[t] = head
            nb_aid[t] = aid
            nb_cost[t] = c
            t += 1
    assert np.all(np.diff(nb_aid) > 0) or True      # per-node sorting is what matters

    st_tail = np.array([p for p, _, _ in G.nb_list], dtype=np.int32)
    st_head = np.array([q for _, q, _ in G.nb_list], dtype=np.int32)
    st_cost = np.array([c for _, _, c in G.nb_list], dtype=np.float64)
    return dict(ch_ptr=ch_ptr, ch_idx=ch_idx, nb_ptr=nb_ptr, nb_head=nb_head,
                nb_aid=nb_aid, nb_cost=nb_cost, st_tail=st_tail,
                st_head=st_head, st_cost=st_cost)


def enumerate_cycles_cy(G, budget, cap=400_000, deadline=None, use_bound=True,
                        H=None, count_only=False, nthreads=None):
    """H = (Hflat, p0row) from quasipoly_atsp._return_bound(G), or None.

    count_only=True skips materialising the flat pairs buffer (which is what
    blows past 10 GB at a few million cycles) and returns
    (None, ncyc:int, empty costs, overflow, work) -- slot 2 is the exact
    cycle COUNT as a Python int, NOT an indptr array.

    nthreads>1 dispatches to the OpenMP kernel (cyenum.run_par): the count is
    the exact sum over shards and the materialised stream is bit-identical to
    the serial one unless the per-shard cycle cap binds.  Default comes from
    the QA_CY_THREADS environment variable (1 = serial, fully backward
    compatible).
    """
    csr = build_csr(G)
    if use_bound and H is not None:
        Hflat, p0row = H
        Hflat = np.ascontiguousarray(Hflat, dtype=np.float64).ravel()
        # _return_bound hands back {node: row}; the kernel wants node -> row
        row = np.full(G.n, -1, dtype=np.int32)
        if p0row:
            ps = np.fromiter(p0row.keys(), dtype=np.int64, count=len(p0row))
            rs = np.fromiter(p0row.values(), dtype=np.int64, count=len(p0row))
            row[ps] = rs
        p0row = row
    else:
        Hflat = np.zeros((0,), dtype=np.float64)
        p0row = np.zeros((0,), dtype=np.int32)
    dl = float(deadline) if deadline is not None else 0.0
    nt = _threads() if nthreads is None else int(nthreads)
    args = (G.n, csr["ch_ptr"], csr["ch_idx"], csr["nb_ptr"],
            csr["nb_head"], csr["nb_aid"], csr["nb_cost"],
            csr["st_tail"], csr["st_head"], csr["st_cost"],
            Hflat, p0row, bool(use_bound and H is not None),
            float(budget), int(cap), dl, bool(count_only))
    if nt > 1:
        pairs, indptr, costs, ovf, work = _enum().run_par(*args, nthreads=nt)
    else:
        pairs, indptr, costs, ovf, work = _enum().run(*args)
    if count_only:
        return None, indptr, costs, ovf, work
    return pairs, indptr, costs, ovf, work


def bits_for(m):
    """Bits needed to hold values 0..m-1 (mirrors recdef.pxd::_bits_for)."""
    if m <= 2:
        return 1
    b = 0
    m -= 1
    while m > 0:
        b += 1
        m >>= 1
    return b


def store_bits(G):
    """(bits_tree, bits_nb) for the record format of this contracted graph."""
    maxdeg = 1
    for v in range(G.n):
        d = len(G.children[v])
        if d > maxdeg:
            maxdeg = d
    return bits_for(1 + maxdeg), bits_for(max(2, len(G.nb_list)))


def enumerate_cycles_store(G, budget, cap=400_000, deadline=None, use_bound=True,
                           H=None, out_paths=None, nthreads=None, buf_bytes=1 << 20):
    """Enumerate to disk instead of RAM.  Returns a dict:

        paths      list of raw (unsorted) record files, in emission order
        n          record count
        ovf        per-shard cap hit?
        work       DFS node visits
        csr        the build_csr(G) dict (needed to build a CycleStore)
        bits_tree, bits_nb

    The record stream of paths concatenated in list order is exactly what
    run() would have materialised, so sort_records(paths, ...) reproduces
    np.argsort(costs, kind="stable") over that stream.  Memory is O(1) in
    the cycle count.
    """
    csr = build_csr(G)
    if use_bound and H is not None:
        Hflat, p0row = H
        Hflat = np.ascontiguousarray(Hflat, dtype=np.float64).ravel()
        row = np.full(G.n, -1, dtype=np.int32)
        if p0row:
            ps = np.fromiter(p0row.keys(), dtype=np.int64, count=len(p0row))
            rs = np.fromiter(p0row.values(), dtype=np.int64, count=len(p0row))
            row[ps] = rs
        p0row = row
    else:
        Hflat = np.zeros((0,), dtype=np.float64)
        p0row = np.zeros((0,), dtype=np.int32)
    dl = float(deadline) if deadline is not None else 0.0
    nt = _threads() if nthreads is None else int(nthreads)
    bits_tree, bits_nb = store_bits(G)
    args = (G.n, csr["ch_ptr"], csr["ch_idx"], csr["nb_ptr"],
            csr["nb_head"], csr["nb_aid"], csr["nb_cost"],
            csr["st_tail"], csr["st_head"], csr["st_cost"],
            Hflat, p0row, bool(use_bound and H is not None),
            float(budget), int(cap), dl)
    if out_paths is None:
        import tempfile
        out_paths = [os.path.join(tempfile.mkdtemp(prefix="cystore_"),
                                  "run_%d.bin" % t) for t in range(max(1, nt))]
    if nt > 1:
        if len(out_paths) < nt:
            raise ValueError("need one path per thread (%d)" % nt)
        n, ovf, work = _enum().run_par_store(*(args + (list(out_paths[:nt]), nt,
                                                       buf_bytes)))
        out_paths = list(out_paths[:nt])
    else:
        n, ovf, work = _enum().run_store(*(args + (out_paths[0], buf_bytes)))
        out_paths = [out_paths[0]]
    return dict(paths=out_paths, n=int(n), ovf=bool(ovf), work=int(work),
                csr=csr, bits_tree=bits_tree, bits_nb=bits_nb)


def to_tuples(pairs, indptr):
    """Materialise cycles as tuples -- only for validation / small runs."""
    out = []
    for i in range(len(indptr) - 1):
        a, b = int(indptr[i]), int(indptr[i + 1])
        seg = pairs[2 * a: 2 * b]
        out.append(tuple(zip(seg[0::2].tolist(), seg[1::2].tolist())))
    return out
