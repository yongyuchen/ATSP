"""Element-by-element validation of the compiled enumerator against the Python one.

The compiled DFS must produce the SAME cycle list, in the SAME order, with the
SAME costs -- otherwise it is not a drop-in and every published number would
have to be regenerated.

Memory note (why this does not OOM any more)
--------------------------------------------
The Python enumerator materialises one tuple-of-tuples per cycle, ~3.2 KB each,
so 10^6 cycles is ~3 GB, and `to_tuples` on the C side doubles that.  We
therefore

  * cap the enumeration (default 120 k cycles),
  * flatten the Python result into an int32 array IMMEDIATELY and drop the
    tuples, so at most one copy is alive at a time,
  * gc explicitly between cases.

The comparison is still element-by-element and ORDER-SENSITIVE: we compare the
flattened pair stream and the cost vector index by index.
"""
import gc
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
sys.path.insert(0, os.path.join(_ROOT, "solver", "cy"))
import quasipoly_atsp as qa
import cybridge

CAP = 120_000


def make(n, seed, kind="uniform"):
    rng = np.random.default_rng(1000 * n + seed)
    if kind == "uniform":
        C = rng.random((n, n))
    elif kind == "exponential":
        C = rng.exponential(1.0, (n, n))
    elif kind == "ties":
        C = rng.integers(0, 5, (n, n)).astype(float)
    elif kind == "metric-ish":
        P = rng.random((n, 2))
        D = np.abs(P[:, None, :] - P[None, :, :]).sum(-1)
        C = D + rng.random((n, n)) * 0.05
    else:
        raise ValueError(kind)
    np.fill_diagonal(C, np.inf)
    return C


def contracted_for(C, k_frac=0.03):
    n = C.shape[0]
    phi, uh, vh = qa.hungarian_with_duals(C)
    u, v = qa.optimal_duals(C, phi, hung0=(uh, vh))
    rc = np.clip(np.where(np.isfinite(C), C - u[:, None] - v[None, :], np.inf),
                 0.0, None)
    invphi = np.empty(n, dtype=np.int64)
    invphi[phi] = np.arange(n)
    extra, _ = qa.build_basis_tree(rc, phi)
    rc_nb = rc.copy()
    rc_nb[np.arange(n), phi] = np.inf
    flat = np.sort(rc_nb[np.isfinite(rc_nb)])
    b = float(flat[int(k_frac * flat.size)])
    G = qa.build_contracted(n, extra, invphi, rc, phi, b)
    return G, b


def flatten_py(cycles):
    """tuple-of-(i,j)-pairs -> (flat int32 pairs, indptr in PAIR units)."""
    indptr = np.zeros(len(cycles) + 1, dtype=np.int64)
    for i, cyc in enumerate(cycles):
        indptr[i + 1] = indptr[i] + len(cyc)
    flat = np.empty(int(indptr[-1]) * 2, dtype=np.int32)
    p = 0
    for cyc in cycles:
        for (i, j) in cyc:
            flat[p] = i
            flat[p + 1] = j
            p += 2
    return flat, indptr


def main(sizes=(60, 100, 150, 200),
         kinds=("uniform", "exponential", "ties", "metric-ish")):
    print(f"compiled enumerator available: {cybridge.available()}   cap={CAP}")
    print(f"{'n':>5} {'kind':>11} {'arcs':>6} | {'py cyc':>8} {'cy cyc':>8} "
          f"{'stream':>7} {'cost':>6} {'ovf':>5} | {'py(s)':>8} {'cy(s)':>8} "
          f"{'speed':>8}")
    bad = 0
    for n in sizes:
        for kind in kinds:
            C = make(n, 1, kind)
            G, b = contracted_for(C)
            H = qa._return_bound(G)

            t = time.perf_counter()
            c1, k1, ovf1 = qa.enumerate_valid_cycles(
                G, b, cap=CAP, deadline=None, use_bound=True)
            t1 = time.perf_counter() - t
            f1, i1 = flatten_py(c1)
            del c1
            gc.collect()

            t = time.perf_counter()
            pairs, indptr, costs, ovf2, work = cybridge.enumerate_cycles_cy(
                G, b, cap=CAP, deadline=None, use_bound=True, H=H)
            t2 = time.perf_counter() - t

            nc1 = len(i1) - 1
            nc2 = len(indptr) - 1
            same_stream = (nc1 == nc2 and np.array_equal(i1, indptr)
                           and np.array_equal(f1, pairs))
            same_cost = (len(k1) == len(costs) and np.allclose(
                np.asarray(k1, dtype=np.float64),
                np.asarray(costs, dtype=np.float64), rtol=0, atol=1e-12))
            same_ovf = (bool(ovf1) == bool(ovf2))
            ok = same_stream and same_cost and same_ovf
            if not ok:
                bad += 1
            print(f"{n:5d} {kind:>11} {len(G.nb_list):6d} | {nc1:8d} {nc2:8d} "
                  f"{str(same_stream):>7} {str(same_cost):>6} "
                  f"{str(same_ovf):>5} | {t1:8.3f} {t2:8.3f} "
                  f"{t1 / max(t2, 1e-9):7.1f}x", flush=True)
            del f1, i1, pairs, indptr, costs, G, H, C, k1
            gc.collect()
    print(f"\nmismatches: {bad}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
