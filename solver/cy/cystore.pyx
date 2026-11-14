# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = -O3
"""Disk-backed cycle store for the Bell-Frieze enumeration.

The enumeration kernel already counts 10^10 cycles in O(1) memory
(count_only mode); what it cannot do is hand them to the packing search,
which needs them in cost-sorted order together with a vertex mask per cycle.
In RAM that is ~0.5 kB per cycle, so a round of 10^7 cycles needs ~5 GB and
the rung that actually proves an instance (measured: >= 4.3e9 cycles) is out
of reach by two orders of magnitude.

This module moves the cycle set to disk in a compact self-describing form and
streams it back one block at a time, so the packing search sees exactly the
same data with O(block) memory.

Record format (variable length, self-delimiting):

    [0:8]    cost      float64, raw
    [8:12]   a0        int32,   index of the start non-basic arc
    [12:14]  nmoves    uint16,  moves after the start arc
    [14:...]  moves     bit packed

A move is one arc of the cycle after the start arc; the close is a move too
(a tree move into p0, or a non-basic arc whose head is p0):

    tree move:  bit 0  + bits_tree bits  (index into children[v])
    nb   move:  bit 1  + bits_nb   bits  (global non-basic arc id)

Replaying the moves walks the folded basis structure from q0 and ends exactly
at p0, so the record determines the cycle completely.  Note the basis
structure is NOT a tree: build_basis_tree Kruskals a bipartite graph and folds
it onto n nodes, so a node can have several parents and the path between two
anchors is not unique -- which is why the encoding stores the actual walk
(child indices) rather than the anchors.

Measured: 20-45 bytes per cycle against 328 bytes for the flat (tail, head)
pair array the RAM path uses (8 bytes per arc, ~41 arcs per cycle).

Pipeline: kernel --(records, emission order)--> run files --(sort by
(cost, emission index))--> rec.bin + blk_off --> CycleStore (block cache).
"""
import os
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free, calloc, qsort
from libc.string cimport memcpy, memset, memmove
from libc.stdio cimport FILE, fopen, fread, fwrite, fclose, fflush, fseek, ftell, SEEK_SET, SEEK_END
# libc.unistd.read carries "except -1", which forces the GIL back on inside
# nogil helpers -- declare both without error propagation.
cdef extern from "unistd.h":
    ssize_t read "read"(int fd, void *buf, size_t count) noexcept nogil
    int close "close"(int fd) noexcept nogil
    ssize_t pread "pread"(int fd, void *buf, size_t count, long long off) noexcept nogil
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t

cnp.import_array()

DEF HDR = 14                      # byte-aligned record header
DEF VER_PAD = 64                  # slack so bw_get never reads past the block
DEF HUGE_OFF = 1 << 40             # hard bound: never rejects on the worst case


cdef int _rss_mb() noexcept:
    """Current RSS in MB (diagnostics only)."""
    cdef int v = 0
    with open("/proc/self/statm") as fh:
        v = int(fh.read().split()[1]) * 4096 // 1000000
    return v


include "recdef.pxd"


# ══════════════════════════════════════════════════════════════════════
# external sort: chunked runs + k-way merge, key (cost, emission index)
# ══════════════════════════════════════════════════════════════════════
cdef struct SortItem:
    double cost
    long long seq
    long long off
    int32_t length


cdef int _cmp_item(const void* a, const void* b) noexcept nogil:
    cdef const SortItem* x = <const SortItem*> a
    cdef const SortItem* y = <const SortItem*> b
    if x.cost < y.cost:
        return -1
    if x.cost > y.cost:
        return 1
    if x.seq < y.seq:
        return -1
    if x.seq > y.seq:
        return 1
    return 0


cdef struct RunReader:
    int fd
    uint8_t* buf
    long long cap
    long long len
    long long pos
    long long base            # emission index of this run's first record
    long long cur             # emission index of the record in hand
    long long nread           # total bytes pulled from the file
    bint eof
    bint done
    const uint8_t* rec
    long long reclen
    double cost
    int bits_tree
    int bits_nb


cdef int _rr_next(RunReader* r, bint verbose=False) noexcept nogil:
    """Load the next record of run r.  0 ok, 1 EOF, -1 error."""
    cdef long long avail, need, got
    while True:
        avail = r.len - r.pos
        if avail >= HDR:
            need = rec_skip(r.buf, r.pos, r.len, r.cap, r.bits_tree, r.bits_nb)
            if need > 0:
                r.rec = r.buf + r.pos
                r.reclen = need
                memcpy(&r.cost, r.rec, 8)
                r.cur = r.base
                r.base += 1
                r.pos += need
                return 0
        if r.eof:
            return 1 if avail == 0 else -1
        if r.pos > 0:
            memmove(r.buf, r.buf + r.pos, avail)
            r.len = avail
            r.pos = 0
        if r.len == r.cap:
            return -1
        got = read(r.fd, r.buf + r.len, r.cap - r.len)
        if got < 0:
            return -1
        if got == 0:
            r.eof = True
        else:
            r.len += got
            r.nread += got


cdef inline int _rr_less(RunReader* rs, int a, int b) noexcept nogil:
    """Heap order: (cost, emission index), then run id for determinism."""
    if rs[a].cost < rs[b].cost:
        return 1
    if rs[a].cost > rs[b].cost:
        return 0
    if rs[a].cur < rs[b].cur:
        return 1
    if rs[a].cur > rs[b].cur:
        return 0
    return a < b


def sort_records(list in_paths, str out_path, int bits_tree, int bits_nb,
                 int chunk_recs=2_000_000, int buf_bytes=1 << 23,
                 int blk=4096, str tmpdir=None, bint keep=False,
                 int rr_buf=1 << 20, bint verbose=False):
    """Sort record files by (cost, emission index); returns (n, blk_off).

    The emission index is the position in the concatenation of in_paths in
    the given order -- the order the kernel would have emitted serially --
    so breaking cost ties by it reproduces np.argsort(kind="stable") exactly.
    blk_off has nblocks+1 entries (byte offsets of DEF_BLK boundaries).
    """
    cdef int k, nruns = len(in_paths)
    cdef long long total = 0, seq = 0
    cdef str tmp = tmpdir or os.path.dirname(os.path.abspath(out_path)) or "."
    # run 文件集中放在 tmp/runs/ 下: 调用方要清理时 rename+rmtree 整个
    # 子目录即可, 不会碰到 out_path 本身, 也不会逐个 unlink 几百个文件.
    try:
        os.makedirs(os.path.join(tmp, "runs"), exist_ok=True)
        tmp = os.path.join(tmp, "runs")
    except OSError:
        pass
    cdef list run_paths = []
    cdef list run_counts = []
    cdef SortItem* items = <SortItem*> malloc(chunk_recs * sizeof(SortItem))
    cdef uint8_t* cbuf = <uint8_t*> malloc(buf_bytes)
    cdef uint8_t* obuf = <uint8_t*> malloc(buf_bytes)
    cdef FILE* fout = NULL
    cdef long long clen, pos, ntake, t, got, olen, wrote, nrec, nblk, j
    cdef uint16_t nm
    cdef long long need
    cdef str rp
    cdef cnp.ndarray[cnp.int64_t, ndim=1] blk_arr
    cdef int64_t* bp
    if items == NULL or cbuf == NULL or obuf == NULL:
        free(items); free(cbuf); free(obuf)
        raise MemoryError("sort buffers")
    try:
        for k in range(nruns):
            fd = os.open(in_paths[k], os.O_RDONLY)
            try:
                clen = 0
                while True:
                    while clen < buf_bytes:
                        got = read(fd, cbuf + clen, buf_bytes - clen)
                        if got < 0:
                            raise OSError("read failed on " + in_paths[k])
                        if got == 0:
                            break
                        clen += got
                    if clen == 0:
                        break
                    pos = 0
                    ntake = 0
                    while pos + HDR <= clen and ntake < chunk_recs:
                        need = rec_skip(cbuf, pos, clen, buf_bytes,
                                        bits_tree, bits_nb)
                        if need < 0:
                            break                   # record crosses the buffer
                        memcpy(&items[ntake].cost, cbuf + pos, 8)
                        items[ntake].seq = seq
                        items[ntake].off = pos
                        items[ntake].length = <int32_t> need
                        seq += 1
                        pos += need
                        ntake += 1
                    if ntake == 0:
                        raise RuntimeError("record larger than the sort buffer")
                    if verbose:
                        print("[cystore] chunk clen=%d ntake=%d pos=%d seq=%d"
                              % (clen, ntake, pos, seq))
                    # The run file must be written BEFORE the buffer is
                    # compacted: the records live at cbuf[items.off ...], and
                    # the memmove below would overwrite the first ones.
                    qsort(items, <size_t> ntake, sizeof(SortItem), _cmp_item)
                    rp = os.path.join(tmp, "cst_run_%d_%d.bin"
                                      % (os.getpid(), len(run_paths)))
                    run_paths.append(rp)
                    fout = fopen(rp.encode(), "wb")
                    if fout == NULL:
                        raise OSError("cannot open run file " + rp)
                    olen = 0
                    for t in range(ntake):
                        if olen + items[t].length > buf_bytes:
                            if fwrite(obuf, 1, <size_t> olen, fout) != <size_t> olen:
                                raise OSError("short write to " + rp)
                            olen = 0
                        memcpy(obuf + olen, cbuf + items[t].off,
                               <size_t> items[t].length)
                        olen += items[t].length
                    if olen:
                        if fwrite(obuf, 1, <size_t> olen, fout) != <size_t> olen:
                            raise OSError("short write to " + rp)
                    fclose(fout)
                    fout = NULL
                    total += ntake
                    run_counts.append(ntake)
                    if pos < clen:
                        memmove(cbuf, cbuf + pos, clen - pos)
                        clen -= pos
                    else:
                        clen = 0
                    if verbose and (total % (20 * chunk_recs)) < ntake:
                        print("[cystore] sorted %d records into %d runs"
                              % (total, len(run_paths)))
            finally:
                os.close(fd)
                # 流式语义: 输入读完就可以删, 否则峰值磁盘是
                # raw + runs + sorted 三份 (1e9 圈要 ~90 GB, 本机放不下).
                if keep:
                    pass
    finally:
        free(items); free(cbuf); free(obuf)
    if verbose:
        print("[cystore] phase1 done total=%d nruns=%d rss=%d MB"
              % (total, len(run_paths), _rss_mb()))

    if total == 0:
        open(out_path, "wb").close()
        return 0, np.zeros(1, dtype=np.int64)

    # ---------------- phase 2: k-way merge ----------------
    cdef int nr = len(run_paths)
    cdef RunReader* rs = <RunReader*> calloc(nr, sizeof(RunReader))
    cdef int* heap = <int*> malloc(nr * sizeof(int))
    cdef long long* boff = <long long*> malloc(
        (total // blk + 2) * sizeof(long long))
    cdef uint8_t* wbuf = <uint8_t*> malloc(buf_bytes)
    cdef long long outpos = 0, wpos = 0, nout = 0, base_seq = 0
    cdef long long* dbg = <long long*> calloc(nr + 1, sizeof(long long))
    cdef int hn = 0, parent, child, tmpi, rc, kk
    cdef FILE* fout2 = NULL
    if rs == NULL or heap == NULL or boff == NULL or wbuf == NULL:
        free(rs); free(heap); free(boff); free(wbuf); free(dbg)
        raise MemoryError("merge structures")
    try:
        for k in range(nr):
            rs[k].fd = os.open(run_paths[k], os.O_RDONLY)
            # 读窗口固定 1 MB: run 数 = 总记录/chunk_recs (159 个时曾把
            # 8 MB/个的缓冲吃成 1.27 GB), 与 buf_bytes 解耦.
            rs[k].buf = <uint8_t*> malloc(rr_buf)
            rs[k].cap = rr_buf
            rs[k].bits_tree = bits_tree
            rs[k].bits_nb = bits_nb
            if rs[k].buf == NULL:
                raise MemoryError("run buffer")
        if verbose:
            print("[cystore] merge start nr=%d buf_bytes=%d rss=%d MB"
                  % (nr, buf_bytes, _rss_mb()))
        # run k covers the contiguous emission-index range starting at base_seq
        for k in range(nr):
            rs[k].base = base_seq
            base_seq += run_counts[k]
        boff[0] = 0
        fout2 = fopen(out_path.encode(), "wb")
        if fout2 == NULL:
            raise OSError("cannot open " + out_path)
        for k in range(nr):
            rc = _rr_next(&rs[k], verbose)
            if rc != 0:
                raise RuntimeError("run file %s is empty" % run_paths[k])
            heap[hn] = k
            hn += 1
            child = hn - 1
            while child > 0:
                parent = (child - 1) >> 1
                if _rr_less(rs, heap[child], heap[parent]):
                    tmpi = heap[child]; heap[child] = heap[parent]; heap[parent] = tmpi
                    child = parent
                else:
                    break
        while hn > 0:
            kk = heap[0]
            if wpos + rs[kk].reclen > buf_bytes:
                if fwrite(wbuf, 1, <size_t> wpos, fout2) != <size_t> wpos:
                    raise OSError("short write to " + out_path)
                outpos += wpos
                wpos = 0
            memcpy(wbuf + wpos, rs[kk].rec, <size_t> rs[kk].reclen)
            wpos += rs[kk].reclen
            nout += 1
            if (nout % blk) == 0:
                boff[nout // blk] = outpos + wpos
            rc = _rr_next(&rs[kk], verbose)
            if rc == 1:
                if verbose:
                    print("[cystore] run %d EOF after %d records: nread=%d "
                          "fsize=%d pos=%d len=%d"
                          % (kk, rs[kk].base, rs[kk].nread,
                             os.fstat(rs[kk].fd).st_size, rs[kk].pos,
                             rs[kk].len))
                hn -= 1
                heap[0] = heap[hn]
            elif rc != 0:
                raise RuntimeError("run file %s truncated" % run_paths[kk])
            if verbose:
                dbg[kk] += 1
            parent = 0
            while hn > 0:
                child = 2 * parent + 1
                if child >= hn:
                    break
                if child + 1 < hn and _rr_less(rs, heap[child + 1], heap[child]):
                    child += 1
                if _rr_less(rs, heap[child], heap[parent]):
                    tmpi = heap[child]; heap[child] = heap[parent]; heap[parent] = tmpi
                    parent = child
                else:
                    break
        if verbose:
            for k in range(nr):
                print("[cystore] run %d: expected %d, merged %d"
                      % (k, run_counts[k], dbg[k]))
        if wpos:
            if fwrite(wbuf, 1, <size_t> wpos, fout2) != <size_t> wpos:
                raise OSError("short write to " + out_path)
            outpos += wpos
        fflush(fout2)
        nblk = (total + blk - 1) // blk
        boff[nblk] = outpos
        blk_arr = np.empty(nblk + 1, dtype=np.int64)
        bp = <int64_t*> blk_arr.data
        for j in range(nblk + 1):
            bp[j] = boff[j]
        fclose(fout2)
        fout2 = NULL
    finally:
        if fout2 != NULL:
            fclose(fout2)
        for k in range(nr):
            if rs[k].fd >= 0:
                os.close(rs[k].fd)
            if rs[k].buf != NULL:
                free(rs[k].buf)
        free(rs); free(heap); free(boff); free(wbuf); free(dbg)
    # run 文件不在这里删: 一次 1e9 圈的排序会产生几百个 run, 逐个 unlink
    # 会撞上本机沙箱的单回合 50 次删除配额 (进程直接被 kill, 连 traceback
    # 都没有). 改由调用方把整个 tmp 目录 rename 到 trash 再一次 rmtree --
    # 那样 safe-delete 按目录计一次. QA_STORE_KEEP=1 时连输入也不删.
    if keep:
        pass
    if nout != total:
        raise RuntimeError("merge lost records: %d != %d" % (nout, total))
    return total, blk_arr


# ══════════════════════════════════════════════════════════════════════
# block decoding
# ══════════════════════════════════════════════════════════════════════
cdef int _scan_block(const uint8_t* raw, long long nbytes, int nrec,
                     int bits_tree, int bits_nb,
                     cnp.int32_t* ch_ptr, cnp.int32_t* ch_idx,
                     cnp.int32_t* st_tail, cnp.int32_t* st_head,
                     double* cost, int64_t* roff,
                     uint64_t* mask_w, uint64_t* cmask_w,
                     cnp.int64_t* fcid, int W, int Wc) noexcept nogil:
    """Parse nrec records out of raw; fill cost/roff/mask_w/cmask_w.

    Returns 0 on success, -1 on a malformed record.  roff has nrec+1 entries
    (the end offset of the last record included) so a single record can be
    re-parsed later for its arc list.
    """
    cdef long long pos = 0, bit
    cdef int j, t, s, i2, w, nb, nbtake
    cdef double c
    cdef int32_t a0, cur, nxt
    cdef uint16_t nm
    cdef uint32_t v
    cdef int64_t tail
    for j in range(nrec):
        if pos + HDR > nbytes:
            return -1
        memcpy(&c, raw + pos, 8)
        memcpy(&a0, raw + pos + 8, 4)
        memcpy(&nm, raw + pos + 12, 2)
        roff[j] = pos
        cost[j] = c
        bit = (pos + HDR) * 8
        cur = st_head[a0]
        tail = st_tail[a0]
        for w in range(W):
            mask_w[j * W + w] = 0
        for w in range(Wc):
            cmask_w[j * Wc + w] = 0
        mask_w[j * W + (tail >> 6)] |= <uint64_t> 1 << (tail & 63)
        tail = fcid[tail]
        cmask_w[j * Wc + (tail >> 6)] |= <uint64_t> 1 << (tail & 63)
        for t in range(nm):
            v = bw_get(raw, &bit, 1)
            if (v & 1) == 0:
                nb = bw_get(raw, &bit, bits_tree)
                if nb >= ch_ptr[cur + 1] - ch_ptr[cur]:
                    return -1
                nxt = ch_idx[ch_ptr[cur] + nb]
            else:
                nb = bw_get(raw, &bit, bits_nb)
                nxt = st_head[nb]
            # arc (cur -> nxt): tail = cur
            mask_w[j * W + (cur >> 6)] |= <uint64_t> 1 << (cur & 63)
            tail = fcid[cur]
            cmask_w[j * Wc + (tail >> 6)] |= <uint64_t> 1 << (tail & 63)
            cur = nxt
        pos = (bit + 7) >> 3
    roff[nrec] = pos
    return 0


cdef int _decode_rec(const uint8_t* p, long long nbytes, int bits_tree,
                     int bits_nb, cnp.int32_t* ch_ptr, cnp.int32_t* ch_idx,
                     cnp.int32_t* st_tail, cnp.int32_t* st_head,
                     cnp.int32_t* out_a, cnp.int32_t* out_b, int cap) noexcept nogil:
    """Decode one record into (tail, head) pairs.  Returns the arc count,
    or -1 if it does not fit in cap."""
    cdef long long bit
    cdef int32_t a0, cur, nxt
    cdef uint16_t nm
    cdef uint32_t v, nb
    cdef int t = 0
    if nbytes < HDR:
        return -1
    memcpy(&a0, p + 8, 4)
    memcpy(&nm, p + 12, 2)
    if 1 + nm > cap:
        return -1
    out_a[0] = st_tail[a0]
    out_b[0] = st_head[a0]
    t = 1
    cur = st_head[a0]
    bit = (HDR) * 8
    for _ in range(nm):
        v = bw_get(p, &bit, 1)
        if (v & 1) == 0:
            nb = bw_get(p, &bit, bits_tree)
            nxt = ch_idx[ch_ptr[cur] + nb]
        else:
            nb = bw_get(p, &bit, bits_nb)
            nxt = st_head[nb]
        out_a[t] = cur
        out_b[t] = nxt
        t += 1
        cur = nxt
    return t

cdef int _max_children(cnp.ndarray[cnp.int32_t, ndim=1] ch_ptr) noexcept:
    cdef int i, m = 0
    for i in range(ch_ptr.shape[0] - 1):
        if ch_ptr[i + 1] - ch_ptr[i] > m:
            m = ch_ptr[i + 1] - ch_ptr[i]
    return m


cdef int _verify_sorted(int fd, int64_t* blk_off, long long nblk,
                        int bits_tree, int bits_nb,
                        long long* bad_out, long long* nrec_out) noexcept nogil:
    """0 ok, 1 发现逆序, 2 记录损坏(rec_skip 失败), -1 内存, -2 读失败.

    hard 边界 = ln + VER_PAD: rec_skip 的最坏长度检查用硬边界, 而记录本身
    紧实打包, 所以块尾最后一条记录的最坏长度可能越过 ln (它只需要更少
    字节). 多读 VER_PAD 字节垫在块后, 让位读取永不越界.
    """
    cdef long long b, ln, pos, nxt, nrec = 0
    cdef double prev = -1e308, c
    cdef uint8_t* buf = NULL
    cdef long long cap = 0
    cdef ssize_t got
    for b in range(nblk):
        ln = blk_off[b + 1] - blk_off[b]
        if ln + VER_PAD > cap:
            free(buf)
            buf = <uint8_t*> calloc(1, <size_t> (ln + VER_PAD))
            if buf == NULL:
                free(buf)
                return -1
            cap = ln + VER_PAD
        got = pread(fd, buf, <size_t> ln, blk_off[b])
        if got != ln:
            free(buf)
            return -2
        pos = 0
        while pos < ln:
            nxt = rec_skip(buf, pos, ln, HUGE_OFF, bits_tree, bits_nb)
            if nxt < 0:
                free(buf)
                return 2
            memcpy(&c, buf + pos, 8)
            if c < prev:
                bad_out[0] = nrec
                free(buf)
                return 1
            prev = c
            nrec += 1
            pos += nxt          # rec_skip returns the LENGTH, not the offset
    bad_out[0] = -1
    nrec_out[0] = nrec
    free(buf)
    return 0


cdef class CycleStore:
    """Cost-sorted cycle set on disk, streamed one block at a time.

    Same data the packing search gets from RAM, same order:

        n                 record count
        cost(i)           float      reduced cost
        mask(i)           int        vertex bitmask (bit a = arc tail a)
        cmask(i)          int        phi-subtour bitmask
        arcs(i)           [(a, b)]   the cycle's arcs (psi: a -> b)
        suffix_or(i)      int        OR of cmask[i:]
        cover_first(s, i) (idx, cost) or None -- first cycle >= i meeting s

    Memory is O(block + nblocks), independent of n: the block cache holds
    ncache decoded blocks and the auxiliary arrays are block-granular.
    """
    cdef:
        readonly long long n
        readonly int nfc
        readonly int bits_tree
        readonly int bits_nb
        readonly int B
        readonly long long nblocks
        int n_, m_nb, W, Wc
        object _keep
        cnp.int32_t *ch_ptr, *ch_idx, *st_tail, *st_head
        cnp.int64_t *fcid
        int fd
        cnp.int64_t *blk_off
        # auxiliary arrays (block granularity)
        object _suf_or          # uint64 (nblocks+1, Wc)
        object _suf_first       # int32  (nfc, nblocks)  suffix-min first index
        object _sufmin          # float64 (nfc, nblocks) suffix-min cost
        uint64_t* suf_or
        int32_t* suf_first
        double* sufmin
        # block cache (LRU)
        dict _cache
        list _order
        int ncache
        cnp.int32_t* _oa
        cnp.int32_t* _ob
        object _oa_keep, _ob_keep

    def __init__(self, str rec_path, object blk_off, int n_, int m_nb, int nfc,
                 object ch_ptr, object ch_idx, object st_tail, object st_head,
                 object fcid, long long n, int block=4096, int cache=4):
        cdef cnp.ndarray[cnp.int32_t, ndim=1] cp = np.ascontiguousarray(ch_ptr, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] ci = np.ascontiguousarray(ch_idx, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] stl = np.ascontiguousarray(st_tail, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] sth = np.ascontiguousarray(st_head, dtype=np.int32)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] fc = np.ascontiguousarray(fcid, dtype=np.int64)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] bo = np.ascontiguousarray(blk_off, dtype=np.int64)
        self._keep = (cp, ci, stl, sth, fc, bo)
        self.ch_ptr = <cnp.int32_t*> cp.data
        self.ch_idx = <cnp.int32_t*> ci.data
        self.st_tail = <cnp.int32_t*> stl.data
        self.st_head = <cnp.int32_t*> sth.data
        self.fcid = <cnp.int64_t*> fc.data
        self.n_ = n_
        self.m_nb = m_nb
        self.nfc = nfc
        self.W = (n_ + 63) >> 6
        self.Wc = ((nfc + 63) >> 6) or 1
        self.bits_tree = _bits_for(1 + _max_children(cp))
        self.bits_nb = _bits_for(m_nb if m_nb > 1 else 2)
        self.blk_off = <cnp.int64_t*> bo.data
        self.n = n
        self.B = block
        self.nblocks = (n + block - 1) // block if n else 0
        if bo.shape[0] != self.nblocks + 1:
            raise ValueError("blk_off has %d entries, expected %d for n=%d B=%d"
                             % (bo.shape[0], self.nblocks + 1, n, block))
        self.fd = os.open(rec_path, os.O_RDONLY)
        self._cache = {}
        self._order = []
        self.ncache = max(1, cache)
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] soa = np.zeros(
            (self.nblocks + 1) * self.Wc, dtype=np.uint64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] sfa = np.zeros(
            max(1, self.nfc) * max(1, self.nblocks), dtype=np.int32)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] sma = np.full(
            max(1, self.nfc) * max(1, self.nblocks), 1e308, dtype=np.float64)
        self._suf_or = soa
        self._suf_first = sfa
        self._sufmin = sma
        self.suf_or = <uint64_t*> soa.data
        self.suf_first = <int32_t*> sfa.data
        self.sufmin = <double*> sma.data
        cdef cnp.ndarray[cnp.int32_t, ndim=1] oaa = np.empty(1 << 16, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] oba = np.empty(1 << 16, dtype=np.int32)
        self._oa_keep = oaa
        self._ob_keep = oba
        self._oa = <cnp.int32_t*> oaa.data
        self._ob = <cnp.int32_t*> oba.data

    def __dealloc__(self):
        if self.fd >= 0:
            os.close(self.fd)

    # ------------------------------------------------------------------
    cdef object _blk(self, long long i):
        cdef long long b = i // self.B
        cdef long long b0 = b * self.B
        cdef int cnt = <int> (self.n - b0 if self.n - b0 < self.B else self.B)
        if cnt <= 0:
            raise IndexError("cycle index %d out of range (n=%d)" % (i, self.n))
        if b in self._cache:
            self._order.remove(b)
            self._order.append(b)
            return self._cache[b]
        cdef long long start = self.blk_off[b]
        cdef long long end = self.blk_off[b + 1]
        raw = os.pread(self.fd, end - start, start)
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] rawa = np.frombuffer(raw, dtype=np.uint8)
        cdef const uint8_t* rp = <const uint8_t*> rawa.data
        cdef cnp.ndarray[cnp.float64_t, ndim=1] cost = np.empty(cnt, dtype=np.float64)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] roff = np.empty(cnt + 1, dtype=np.int64)
        # 2-D scratch: indexed as rows from Python, touched as a flat buffer
        # from C, so keep a Python handle plus a typed 2-D view for .data
        mw = np.zeros((cnt, self.W), dtype=np.uint64)
        cw = np.zeros((cnt, self.Wc), dtype=np.uint64)
        cdef cnp.ndarray[cnp.uint64_t, ndim=2] mwv = mw
        cdef cnp.ndarray[cnp.uint64_t, ndim=2] cwv = cw
        cdef int rc = _scan_block(rp, len(raw), cnt, self.bits_tree,
                                  self.bits_nb, self.ch_ptr, self.ch_idx,
                                  self.st_tail, self.st_head,
                                  <double*> cost.data, <int64_t*> roff.data,
                                  <uint64_t*> mwv.data, <uint64_t*> cwv.data,
                                  self.fcid, self.W, self.Wc)
        if rc != 0:
            raise RuntimeError("corrupt record in block %d" % b)
        # intra-block suffix union of cmask
        ior = np.zeros((cnt + 1, self.Wc), dtype=np.uint64)
        cdef cnp.ndarray[cnp.uint64_t, ndim=2] iorv = ior
        cdef uint64_t* ip = <uint64_t*> iorv.data
        cdef int j, w, s
        for j in range(cnt - 1, -1, -1):
            for w in range(self.Wc):
                ip[j * self.Wc + w] = ip[(j + 1) * self.Wc + w] | cw[j * self.Wc + w]
        intra_or = [int.from_bytes(ior[j].tobytes(), "little")
                    for j in range(cnt + 1)]
        # per-subtour cover lists inside this block
        cdef list cov_j = [[] for _ in range(self.nfc)]
        cdef list cov_c = [[] for _ in range(self.nfc)]
        cdef uint64_t bits
        for j in range(cnt):
            bits = cw[j * self.Wc]
            while bits:
                s = (bits & -bits).bit_length() - 1
                cov_j[s].append(b0 + j)
                cov_c[s].append(float(cost[j]))
                bits &= bits - 1
        blk = dict(b0=b0, cnt=cnt, raw=raw, roff=roff, cost=cost.tolist(),
                   mask=None, cmask=None, intra_or=intra_or,
                   cov_j=cov_j, cov_c=cov_c, mw=mw, cw=cw)
        self._cache[b] = blk
        self._order.append(b)
        while len(self._order) > self.ncache:
            self._cache.pop(self._order.pop(0), None)
        return blk

    # ------------------------------------------------------------------
    def cost(self, long long i):
        return self._blk(i)["cost"][i % self.B]

    def mask(self, long long i):
        blk = self._blk(i)
        if blk["mask"] is None:
            mw = blk["mw"]
            blk["mask"] = [int.from_bytes(mw[j].tobytes(), "little")
                           for j in range(blk["cnt"])]
        return blk["mask"][i % self.B]

    def cmask(self, long long i):
        blk = self._blk(i)
        if blk["cmask"] is None:
            cw = blk["cw"]
            blk["cmask"] = [int.from_bytes(cw[j].tobytes(), "little")
                            for j in range(blk["cnt"])]
        return blk["cmask"][i % self.B]

    def arcs(self, long long i):
        """The cycle as a list of (tail, head) pairs -- identical to the RAM path."""
        blk = self._blk(i)
        cdef long long b = i // self.B
        cdef int j = <int> (i - b * self.B)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] roff = blk["roff"]
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] rawa = np.frombuffer(blk["raw"], dtype=np.uint8)
        cdef const uint8_t* rp = <const uint8_t*> rawa.data
        cdef long long a = int(roff[j])
        cdef long long z = int(roff[j + 1])
        cdef int na = _decode_rec(rp + a, z - a, self.bits_tree, self.bits_nb,
                                  self.ch_ptr, self.ch_idx, self.st_tail,
                                  self.st_head, self._oa, self._ob, 1 << 16)
        if na < 0:
            raise RuntimeError("cycle %d does not decode" % i)
        return [(int(self._oa[t]), int(self._ob[t])) for t in range(na)]

    def suffix_or(self, long long i):
        """OR of cmask[i:]: block suffix unions plus the intra-block part."""
        cdef long long b = i // self.B
        blk = self._blk(i)
        cdef int j = <int> (i - b * self.B)
        cdef uint64_t* so = self.suf_or + (b + 1) * self.Wc
        cdef int w
        acc = 0
        for w in range(self.Wc):
            if so[w]:
                acc |= int(so[w]) << (64 * w)
        return acc | blk["intra_or"][j]

    def cover_first(self, int s, long long i):
        """First cycle with index >= i whose cmask touches subtour s.

        Exact: the intra-block candidate is the first entry of the block's
        cover list at or after i; otherwise the answer is the suffix-minimum
        first index over the later blocks.  Indices inside block b are all
        smaller than indices in later blocks, so the intra candidate wins
        whenever it exists.
        """
        cdef long long b = i // self.B
        cdef long long nb = self.nblocks
        cdef int32_t fi
        blk = self._blk(i)
        cdef list js = blk["cov_j"][s]
        if js:
            lo, hi = 0, len(js)
            while lo < hi:
                mid = (lo + hi) >> 1
                if js[mid] < i:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(js):
                return (js[lo], blk["cov_c"][s][lo])
        if b + 1 < nb:
            fi = self.suf_first[s * nb + (b + 1)]
            if fi >= 0:
                return (int(fi), float(self.sufmin[s * nb + (b + 1)]))
        return None

    # ------------------------------------------------------------------
    def build_aux(self, bint verbose=False):
        """One streaming pass over the sorted file.

        suf_or[b]      OR of cmask over blocks >= b
        suf_first[s][b]  smallest cycle index >= block b start meeting s
        sufmin[s][b]     cost of that cycle
        """
        cdef long long b, j, start, end
        cdef int s, w, cnt, rc
        cdef cnp.ndarray[cnp.float64_t, ndim=1] blkmin = np.full(
            self.nfc * self.nblocks, 1e308, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] blkfirst = np.full(
            self.nfc * self.nblocks, -1, dtype=np.int32)
        cdef double* bmp = <double*> blkmin.data
        cdef int32_t* bfp = <int32_t*> blkfirst.data
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] bor = np.zeros(
            self.nblocks * self.Wc, dtype=np.uint64)
        cdef uint64_t* borp = <uint64_t*> bor.data
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] rawa
        cdef cnp.ndarray[cnp.float64_t, ndim=1] cost = np.empty(self.B, dtype=np.float64)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] roff = np.empty(self.B + 1, dtype=np.int64)
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] mwd = np.empty(
            self.B * self.W, dtype=np.uint64)
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] cw = np.empty(
            self.B * self.Wc, dtype=np.uint64)
        cdef double* cp = <double*> cost.data
        cdef uint64_t* cpp = <uint64_t*> cw.data
        cdef const uint8_t* rp
        cdef uint64_t bits
        for b in range(self.nblocks):
            cnt = <int> (self.n - b * self.B if self.n - b * self.B < self.B else self.B)
            if cnt <= 0:
                break
            start = self.blk_off[b]
            end = self.blk_off[b + 1]
            raw = os.pread(self.fd, end - start, start)
            rawa = np.frombuffer(raw, dtype=np.uint8)
            rp = <const uint8_t*> rawa.data
            rc = _scan_block(rp, len(raw), cnt, self.bits_tree, self.bits_nb,
                             self.ch_ptr, self.ch_idx, self.st_tail,
                             self.st_head, cp, <int64_t*> roff.data,
                             <uint64_t*> mwd.data, cpp, self.fcid,
                             self.W, self.Wc)
            if rc != 0:
                raise RuntimeError("corrupt record in block %d" % b)
            for j in range(cnt):
                for w in range(self.Wc):
                    borp[b * self.Wc + w] |= cpp[j * self.Wc + w]
            for s in range(self.nfc):
                bmp[s * self.nblocks + b] = 1e308
                bfp[s * self.nblocks + b] = -1
            for j in range(cnt):
                bits = cpp[j * self.Wc]
                while bits:
                    s = (bits & -bits).bit_length() - 1
                    if bfp[s * self.nblocks + b] < 0:
                        bfp[s * self.nblocks + b] = <int32_t> (b * self.B + j)
                        bmp[s * self.nblocks + b] = cp[j]
                    bits &= bits - 1
            if verbose and (b & 8191) == 0 and b:
                print("[cystore] aux block %d/%d" % (b, self.nblocks))
        # suffix minima over blocks (backwards)
        cdef long long nb = self.nblocks
        cdef cnp.ndarray[cnp.int32_t, ndim=1] sfa = self._suf_first
        cdef cnp.ndarray[cnp.float64_t, ndim=1] sma = self._sufmin
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] soa = self._suf_or
        cdef int32_t* sfp = <int32_t*> sfa.data
        cdef double* smp = <double*> sma.data
        cdef uint64_t* sop = <uint64_t*> soa.data
        cdef int32_t f1, f2
        cdef double c1, c2
        for b in range(nb - 1, -1, -1):
            for w in range(self.Wc):
                sop[b * self.Wc + w] = sop[(b + 1) * self.Wc + w] | borp[b * self.Wc + w]
            if b + 1 < nb:
                for s in range(self.nfc):
                    f1 = bfp[s * nb + b]
                    f2 = sfp[s * nb + (b + 1)]
                    c1 = bmp[s * nb + b]
                    c2 = smp[s * nb + (b + 1)]
                    if f1 < 0 or (f2 >= 0 and f2 < f1):
                        sfp[s * nb + b] = f2
                        smp[s * nb + b] = c2
                    else:
                        sfp[s * nb + b] = f1
                        smp[s * nb + b] = c1
            else:
                for s in range(self.nfc):
                    sfp[s * nb + b] = bfp[s * nb + b]
                    smp[s * nb + b] = bmp[s * nb + b]

    # ------------------------------------------------------------------
    def stats(self):
        cdef long long tot = 0
        cdef long long b
        for b in range(self.nblocks):
            tot += self.blk_off[b + 1] - self.blk_off[b]
        return dict(n=self.n, nblocks=self.nblocks, B=self.B,
                    bytes_file=tot, bytes_per_cycle=tot / max(1, self.n),
                    bits_tree=self.bits_tree, bits_nb=self.bits_nb,
                    nfc=self.nfc, W=self.W, Wc=self.Wc)

    def verify_sorted(self):
        """全文件扫一遍, 确认 cost 非递减 (即外排序在整个文件上成立).

        逐块 pread + rec_skip 走位, 只读 8 字节 cost, 所以是磁盘顺序读,
        4e8 条约几分钟. 返回 (ok, n_scanned, first_bad_index).
        """
        cdef long long bad = -1, nrec = 0
        cdef int rc = _verify_sorted(self.fd, self.blk_off, self.nblocks,
                                     self.bits_tree, self.bits_nb,
                                     &bad, &nrec)
        if rc == -1:
            raise MemoryError("verify_sorted buffer")
        if rc == -2:
            raise OSError("verify_sorted read failed")
        if rc == 2:
            raise RuntimeError("verify_sorted: corrupt record at index %d"
                               % nrec)
        return (rc == 0), nrec, bad
