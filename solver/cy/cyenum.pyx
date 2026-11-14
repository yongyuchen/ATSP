# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = -O3 -fopenmp
# distutils: extra_link_args = -fopenmp
"""Compiled replacement for quasipoly_atsp.enumerate_valid_cycles.

Same cycle set, same order, same semantics -- including the exact admissible
lower-bound prune h[p0][v] = min reduced cost of any path v -> p0 in the
contracted graph.  Cycles are emitted into flat typed arrays instead of Python
tuples, because the rung that actually proves an instance at n >= 1500 holds
10^5 - 10^7 cycles and a tuple per cycle costs ~3.2 kB (measured: 123 792
cycles already take 0.39 GB).

Two things the Python version does that are improved here without changing the
result:

  * per node v the non-basic arcs are kept in increasing global arc id (that is
    what build_contracted produces: nb_list is rc-sorted and appended in that
    order), so the `aid > a0` test becomes a binary search instead of a scan of
    the whole adjacency list;
  * all state lives in raw C arrays, so a DFS step costs a handful of ns
    instead of a Python frame plus several list/tuple allocations.

Parallel paths (OpenMP).  The enumeration is a loop over start arcs a0, each
running a self-contained DFS rooted at that arc; two start arcs share nothing
but the read-only graph.  Two parallel entry points exploit this, both
bit-compatible with the serial one:

  * run_par(count_only=True): pure integer counting, sharded over start arcs
    with dynamic scheduling, one counter per thread.  The sum equals the serial
    count exactly (integer addition; no ordering or rounding effects).  O(1)
    memory per thread.

  * run_par(count_only=False): materialisation, sharded over CONTIGUOUS ranges
    of start arcs, one buffer set per thread, concatenated in shard order.
    Because the serial loop visits start arcs in increasing order, the
    concatenation reproduces the serial stream exactly -- same pairs, same
    indptr, same costs, bit for bit.  The cycle cap is applied per shard, so a
    capped run emits at most cap + nthreads cycles in total; when no shard
    reaches its cap the output is identical to the serial output.

The GIL is released for the whole parallel region: every operation inside is on
raw C arrays, so no Python object is touched.  Enable via QA_CY_THREADS (>1) in
the environment, or nthreads=... in cybridge.enumerate_cycles_cy.
"""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free, calloc
from libc.string cimport memset, memcpy
from libc.time cimport time as ctime
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
from libc.stdio cimport FILE, fopen, fwrite, fclose, fflush, setvbuf, _IOFBF

include "recdef.pxd"

from cython.parallel cimport parallel, prange
from openmp cimport omp_get_thread_num, omp_get_max_threads

cnp.import_array()

DEF _PAIR_CHUNK = 1 << 22
DEF _CYC_CHUNK = 1 << 18
DEF _LLMAX = 0x7FFFFFFFFFFFFFFF


ctypedef struct Ctx:
    # ---- shared, read-only graph ----
    int n
    cnp.int32_t* ch_ptr
    cnp.int32_t* ch_idx
    cnp.int32_t* nb_ptr
    cnp.int32_t* nb_head
    cnp.int32_t* nb_aid
    double* nb_cost
    cnp.int32_t* st_tail
    cnp.int32_t* st_head
    double* st_cost
    double* H
    cnp.int32_t* p0row
    bint use_bound
    double budget
    double blim
    long long cap
    double deadline
    bint count_only
    # ---- store (disk) mode ----
    bint store_mode
    int bits_tree
    int bits_nb
    long long rec_cap
    FILE* out
    uint8_t* rbuf
    cnp.int32_t* pa_aid      # aid of each path arc, -1 for a tree arc
    cnp.int32_t* pa_ci       # child index of a tree arc (0 otherwise)
    # ---- per-shard mutable state ----
    unsigned char* vis
    cnp.int32_t* pa
    cnp.int32_t* pb
    int plen
    int p0
    int a0
    double* Hrow
    long long work
    long long ncyc
    bint overflow
    bint memfail
    # ---- per-shard materialisation buffers ----
    cnp.int32_t* o_pairs
    long long o_cap_pairs
    long long o_n_slots
    long long* o_indptr
    double* o_cost
    long long o_cap_cyc


cdef bint _ctx_grow_pairs(Ctx* c, long long need) nogil:
    cdef long long newcap = c.o_cap_pairs
    cdef cnp.int32_t* np_
    if newcap <= 0:
        newcap = _PAIR_CHUNK
    while newcap < need:
        newcap *= 2
    np_ = <cnp.int32_t*> realloc(c.o_pairs, newcap * sizeof(cnp.int32_t))
    if np_ == NULL:
        c.memfail = True
        c.overflow = True
        return False
    c.o_pairs = np_
    c.o_cap_pairs = newcap
    return True


cdef bint _ctx_grow_cyc(Ctx* c) nogil:
    cdef long long newcap = c.o_cap_cyc
    cdef bint fresh = newcap <= 0
    cdef long long* ni
    cdef double* nc
    if fresh:
        newcap = _CYC_CHUNK
    else:
        newcap *= 2
    ni = <long long*> realloc(c.o_indptr, (newcap + 1) * sizeof(long long))
    nc = <double*> realloc(c.o_cost, newcap * sizeof(double))
    if ni == NULL or nc == NULL:
        c.memfail = True
        c.overflow = True
        return False
    c.o_indptr = ni
    c.o_cost = nc
    c.o_cap_cyc = newcap
    if fresh:
        c.o_indptr[0] = 0          # realloc does not zero; the serial path
                                   # presets this once after malloc
    return True


cdef bint _emit_store(Ctx* c, double cost, int close_is_nb,
                      int32_t close_id) nogil:
    """Write one compact record.  Moves are path slots 1..plen-1 plus the
    close.  Returns True on write failure (caller stops the run)."""
    cdef long long n = rec_write(c.rbuf, c.rec_cap, cost, c.pa_aid[0],
                                 c.plen - 1, c.pa_aid + 1, c.pa_ci + 1,
                                 close_is_nb, close_id,
                                 c.bits_tree, c.bits_nb)
    if n < 0:
        c.memfail = True
        return True
    if fwrite(c.rbuf, 1, <size_t> n, c.out) != <size_t> n:
        c.memfail = True
        return True
    c.ncyc += 1
    if c.ncyc >= c.cap:
        c.overflow = True
        return True
    return False


cdef bint _emit(Ctx* c, int v, double cost) nogil:
    cdef long long need, base
    cdef int i
    cdef int lo2, hi2, m2, nb2, ne2
    if c.store_mode:
        # the close arc is v -> p0; a tree close needs the child index of p0
        lo2 = c.ch_ptr[v]
        hi2 = c.ch_ptr[v + 1]
        for m2 in range(lo2, hi2):
            if c.ch_idx[m2] == c.p0:
                return _emit_store(c, cost, 0, m2 - lo2)
        # not a tree close => a non-basic arc v -> p0; find its id
        nb2 = c.nb_ptr[v]
        ne2 = c.nb_ptr[v + 1]
        for m2 in range(nb2, ne2):
            if c.nb_head[m2] == c.p0:
                return _emit_store(c, cost, 1, c.nb_aid[m2])
        return False                      # cannot happen for a real close
    if c.count_only:
        # pure integer count: no pairs, no indptr, no cost array touched,
        # so the run stays O(1) memory even at 10^10+ cycles.
        c.ncyc += 1
        if c.ncyc >= c.cap:
            c.overflow = True
            return True
        return False
    if c.o_cap_cyc == 0 or c.ncyc >= c.o_cap_cyc:
        if not _ctx_grow_cyc(c):
            return True
    need = c.o_n_slots + (<long long> c.plen + 1) * 2
    if need > c.o_cap_pairs:
        if not _ctx_grow_pairs(c, need):
            return True
    base = c.o_n_slots
    if not c.count_only:
        for i in range(c.plen):
            c.o_pairs[base + 2 * i] = c.pa[i]
            c.o_pairs[base + 2 * i + 1] = c.pb[i]
        c.o_pairs[base + 2 * c.plen] = v
        c.o_pairs[base + 2 * c.plen + 1] = c.p0
        c.o_n_slots = need
    c.o_cost[c.ncyc] = cost
    c.ncyc += 1
    c.o_indptr[c.ncyc] = need >> 1
    if c.ncyc >= c.cap:
        c.overflow = True
        return True
    return False


cdef bint _dfs(Ctx* c, int v, double cost) nogil:
    cdef int i, w, head, lo, hi, mid, s, e, st
    cdef double nc, cc
    c.work += 1
    if c.deadline > 0.0 and (c.work & 8191) == 0:
        if <double> ctime(NULL) > c.deadline:
            c.overflow = True
            return True

    # (A) close: a tree arc v -> p0, or a later non-basic arc straight to p0
    for i in range(c.ch_ptr[v], c.ch_ptr[v + 1]):
        if c.ch_idx[i] == c.p0:
            if _emit(c, v, cost):
                return True
            break
    s = c.nb_ptr[v]
    e = c.nb_ptr[v + 1]
    cdef int st2 = s
    lo, hi = s, e
    while lo < hi:                                # first index with aid > a0
        mid = (lo + hi) >> 1
        if c.nb_aid[mid] > c.a0:
            hi = mid
        else:
            lo = mid + 1
    st = lo
    for i in range(st, e):
        if c.nb_head[i] == c.p0 and cost + c.nb_cost[i] <= c.budget:
            if _emit(c, v, cost + c.nb_cost[i]):
                return True

    # exact prune: no completion of this branch can get back to p0 in budget
    if c.use_bound:
        cc = c.Hrow[v]
        if cost + cc > c.blim:
            return False

    # (B) descend along zero-cost basis-tree arcs
    for i in range(c.ch_ptr[v], c.ch_ptr[v + 1]):
        w = c.ch_idx[i]
        if c.vis[w]:
            continue
        if c.use_bound and cost + c.Hrow[w] > c.blim:
            continue
        c.vis[w] = 1
        c.pa[c.plen] = v
        c.pb[c.plen] = w
        c.pa_aid[c.plen] = -1
        c.pa_ci[c.plen] = i - c.ch_ptr[v]
        c.plen += 1
        if _dfs(c, w, cost):
            return True
        c.plen -= 1
        c.vis[w] = 0

    # (C) extend along later non-basic arcs (each cycle counted once)
    for i in range(st, e):
        head = c.nb_head[i]
        if c.vis[head] or head == c.p0:
            continue
        nc = cost + c.nb_cost[i]
        if nc > c.budget:
            continue
        if c.use_bound and nc + c.Hrow[head] > c.blim:
            continue
        c.vis[head] = 1
        c.pa[c.plen] = v
        c.pb[c.plen] = head
        c.pa_aid[c.plen] = c.nb_aid[i]
        c.pa_ci[c.plen] = 0
        c.plen += 1
        if _dfs(c, head, nc):
            return True
        c.plen -= 1
        c.vis[head] = 0
    return False


cdef bint _run_arc(Ctx* c, int a0, bint* stop) nogil:
    """One start arc: self-contained DFS rooted at that arc.  Returns True if
    the run should stop (cap or deadline hit); also raises *stop if given."""
    cdef double c0
    cdef int p0, q0
    c0 = c.st_cost[a0]
    if c0 > c.budget:
        return False
    p0 = c.st_tail[a0]
    q0 = c.st_head[a0]
    memset(c.vis, 0, c.n)
    c.vis[p0] = 1
    c.vis[q0] = 1
    c.pa[0] = p0
    c.pb[0] = q0
    c.pa_aid[0] = a0
    c.pa_ci[0] = 0
    c.plen = 1
    c.p0 = p0
    c.a0 = a0
    if c.use_bound:
        c.Hrow = c.H + (<long long> c.p0row[p0]) * c.n
    if _dfs(c, q0, c0):
        if stop != NULL:
            stop[0] = True
        return True
    return False


cdef void _ctx_set_graph(Ctx* c,
                         int n,
                         cnp.int32_t* ch_ptr, cnp.int32_t* ch_idx,
                         cnp.int32_t* nb_ptr, cnp.int32_t* nb_head,
                         cnp.int32_t* nb_aid, double* nb_cost,
                         cnp.int32_t* st_tail, cnp.int32_t* st_head,
                         double* st_cost,
                         double* H, cnp.int32_t* p0row,
                         bint use_bound, double budget, long long cap,
                         double deadline, bint count_only) nogil:
    c.n = n
    c.store_mode = False
    c.bits_tree = 1
    c.bits_nb = 1
    c.rec_cap = 0
    c.out = NULL
    c.rbuf = NULL
    c.pa_aid = NULL
    c.pa_ci = NULL
    c.ch_ptr = ch_ptr
    c.ch_idx = ch_idx
    c.nb_ptr = nb_ptr
    c.nb_head = nb_head
    c.nb_aid = nb_aid
    c.nb_cost = nb_cost
    c.st_tail = st_tail
    c.st_head = st_head
    c.st_cost = st_cost
    c.H = H
    c.p0row = p0row
    c.use_bound = use_bound
    c.budget = budget
    c.blim = budget + 1e-12
    c.cap = cap
    c.deadline = deadline
    c.count_only = count_only
    c.plen = 0
    c.p0 = 0
    c.a0 = 0
    c.Hrow = NULL
    c.work = 0
    c.ncyc = 0
    c.overflow = False
    c.memfail = False
    c.o_pairs = NULL
    c.o_cap_pairs = 0
    c.o_n_slots = 0
    c.o_indptr = NULL
    c.o_cost = NULL
    c.o_cap_cyc = 0
    c.pa_aid = NULL
    c.pa_ci = NULL


cdef int _ctx_alloc_scratch(Ctx* c, int n) nogil:
    """Allocate the per-shard DFS scratch (vis/pa/pb/pa_aid/pa_ci).  0 on success."""
    c.vis = <unsigned char*> calloc(n, 1)
    c.pa = <cnp.int32_t*> malloc(n * sizeof(cnp.int32_t))
    c.pb = <cnp.int32_t*> malloc(n * sizeof(cnp.int32_t))
    c.pa_aid = <cnp.int32_t*> malloc(n * sizeof(cnp.int32_t))
    c.pa_ci = <cnp.int32_t*> malloc(n * sizeof(cnp.int32_t))
    if c.vis == NULL or c.pa == NULL or c.pb == NULL or c.pa_aid == NULL or c.pa_ci == NULL:
        return 1
    return 0


cdef void _ctx_free_shard(Ctx* c) nogil:
    if c.vis != NULL:
        free(c.vis)
    if c.pa != NULL:
        free(c.pa)
    if c.pb != NULL:
        free(c.pb)
    if c.pa_aid != NULL:
        free(c.pa_aid)
    if c.pa_ci != NULL:
        free(c.pa_ci)
    if c.o_pairs != NULL:
        free(c.o_pairs)
    if c.o_indptr != NULL:
        free(c.o_indptr)
    if c.o_cost != NULL:
        free(c.o_cost)
    c.vis = NULL
    c.pa = NULL
    c.pb = NULL
    c.pa_aid = NULL
    c.pa_ci = NULL
    c.o_pairs = NULL
    c.o_indptr = NULL
    c.o_cost = NULL


cdef bint _emit_closes(Ctx* c, int a0) nogil:
    """(A) closes at the top level of start arc a0 -- verbatim the first block
    of _dfs, which emits the direct return to p0 before any descent.  Returns
    True if the run should stop."""
    cdef int i, s, e, lo, hi, mid
    cdef int p0 = c.st_tail[a0], q0 = c.st_head[a0]
    cdef double c0 = c.st_cost[a0]
    for i in range(c.ch_ptr[q0], c.ch_ptr[q0 + 1]):
        if c.ch_idx[i] == p0:
            if _emit(c, q0, c0):
                return True
            break
    s = c.nb_ptr[q0]
    e = c.nb_ptr[q0 + 1]
    lo, hi = s, e
    while lo < hi:
        mid = (lo + hi) >> 1
        if c.nb_aid[mid] > a0:
            hi = mid
        else:
            lo = mid + 1
    for i in range(lo, e):
        if c.nb_head[i] == p0 and c0 + c.nb_cost[i] <= c.budget:
            if _emit(c, q0, c0 + c.nb_cost[i]):
                return True
    return False


cdef long long _expand(Ctx* c, int a0,
                       cnp.int32_t* it_a0, cnp.int32_t* it_w,
                       cnp.int32_t* it_ci, double* it_cost,
                       long long pos, bint fill) nogil:
    """Work items of start arc a0: the close item (w = -1) first, then one item
    per descendable first-level child, in the order _dfs visits them.  With
    fill=False only counts (and needs no buffers); with fill=True appends at
    pos and returns the new pos.  The filters are exactly the (B)/(C) loops of
    _dfs evaluated at the top level, where vis = {p0, q0}."""
    cdef int i, w, head, s, e, lo, hi, mid, cnt = 1
    cdef int p0 = c.st_tail[a0], q0 = c.st_head[a0]
    cdef double c0 = c.st_cost[a0], nc
    if c0 > c.budget:
        return 0
    if fill:
        it_a0[pos] = a0
        it_w[pos] = -1
        it_ci[pos] = 0
        pos += 1
    for i in range(c.ch_ptr[q0], c.ch_ptr[q0 + 1]):
        w = c.ch_idx[i]
        if w == p0 or w == q0:
            continue
        if c.use_bound and c0 + c.Hrow[w] > c.blim:
            continue
        if fill:
            it_a0[pos] = a0
            it_w[pos] = w
            it_ci[pos] = i - c.ch_ptr[q0]
            it_cost[pos] = c0
            pos += 1
        cnt += 1
    s = c.nb_ptr[q0]
    e = c.nb_ptr[q0 + 1]
    lo, hi = s, e
    while lo < hi:
        mid = (lo + hi) >> 1
        if c.nb_aid[mid] > a0:
            hi = mid
        else:
            lo = mid + 1
    for i in range(lo, e):
        head = c.nb_head[i]
        if head == p0 or head == q0:
            continue
        nc = c0 + c.nb_cost[i]
        if nc > c.budget:
            continue
        if c.use_bound and nc + c.Hrow[head] > c.blim:
            continue
        if fill:
            it_a0[pos] = a0
            it_w[pos] = head
            it_ci[pos] = -1 - i        # negative: non-basic step, arc = nb_aid[i]
            it_cost[pos] = nc
            pos += 1
        cnt += 1
    return cnt if not fill else pos


cdef bint _run_item(Ctx* c, int a0, int w, int ci, double cost, bint* stop) nogil:
    """One work item.  w < 0: the (A) closes of arc a0.  Otherwise: the
    sub-DFS rooted at the first-level child w of arc a0, with the path
    p0 -> q0 -> w already in place -- the state _dfs(q0, c0) would be in after
    taking that first step.  ci is the child index of a tree step, or
    -(arc+1) for a non-basic first step."""
    cdef int w_is_nb = ci < 0
    cdef int w_aid = -1 - ci if w_is_nb else 0
    cdef int p0, q0
    if w < 0:
        p0 = c.st_tail[a0]
        q0 = c.st_head[a0]
        c.pa[0] = p0
        c.pb[0] = q0
        c.pa_aid[0] = a0
        c.pa_ci[0] = 0
        c.plen = 1
        c.p0 = p0
        c.a0 = a0
        if c.use_bound:
            c.Hrow = c.H + (<long long> c.p0row[p0]) * c.n
        if _emit_closes(c, a0):
            if stop != NULL:
                stop[0] = True
            return True
        return False
    p0 = c.st_tail[a0]
    q0 = c.st_head[a0]
    memset(c.vis, 0, c.n)
    c.vis[p0] = 1
    c.vis[q0] = 1
    c.vis[w] = 1
    c.pa[0] = p0
    c.pb[0] = q0
    c.pa[1] = q0
    c.pb[1] = w
    c.pa_aid[0] = a0
    c.pa_ci[0] = 0
    if w_is_nb:
        c.pa_aid[1] = w_aid
        c.pa_ci[1] = 0
    else:
        c.pa_aid[1] = -1
        c.pa_ci[1] = ci
    c.plen = 2
    c.p0 = p0
    c.a0 = a0
    if c.use_bound:
        c.Hrow = c.H + (<long long> c.p0row[p0]) * c.n
    if _dfs(c, w, cost):
        if stop != NULL:
            stop[0] = True
        return True
    return False


cdef class Enumerator:
    cdef:
        int alloc_n
        # serial-path persistent scratch (reused across calls)
        unsigned char* s_vis
        cnp.int32_t* s_pa
        cnp.int32_t* s_pb
        cnp.int32_t* s_pa_aid
        cnp.int32_t* s_pa_ci
        cnp.int32_t* s_pairs
        long long s_cap_pairs
        long long* s_indptr
        double* s_cost
        long long s_cap_cyc
        object _keep

    def __cinit__(self):
        self.s_pairs = NULL
        self.s_indptr = NULL
        self.s_cost = NULL
        self.s_vis = NULL
        self.s_pa = NULL
        self.s_pb = NULL
        self.s_pa_aid = NULL
        self.s_pa_ci = NULL

    def __dealloc__(self):
        if self.s_pairs != NULL:
            free(self.s_pairs)
        if self.s_indptr != NULL:
            free(self.s_indptr)
        if self.s_cost != NULL:
            free(self.s_cost)
        if self.s_vis != NULL:
            free(self.s_vis)
        if self.s_pa != NULL:
            free(self.s_pa)
        if self.s_pb != NULL:
            free(self.s_pb)
        if self.s_pa_aid != NULL:
            free(self.s_pa_aid)
        if self.s_pa_ci != NULL:
            free(self.s_pa_ci)

    def run(self,
            int n,
            cnp.ndarray[cnp.int32_t, ndim=1] ch_ptr,
            cnp.ndarray[cnp.int32_t, ndim=1] ch_idx,
            cnp.ndarray[cnp.int32_t, ndim=1] nb_ptr,
            cnp.ndarray[cnp.int32_t, ndim=1] nb_head,
            cnp.ndarray[cnp.int32_t, ndim=1] nb_aid,
            cnp.ndarray[cnp.float64_t, ndim=1] nb_cost,
            cnp.ndarray[cnp.int32_t, ndim=1] st_tail,
            cnp.ndarray[cnp.int32_t, ndim=1] st_head,
            cnp.ndarray[cnp.float64_t, ndim=1] st_cost,
            cnp.ndarray[cnp.float64_t, ndim=1] Hflat,
            cnp.ndarray[cnp.int32_t, ndim=1] p0row,
            bint use_bound,
            double budget,
            long long cap,
            double deadline,
            bint count_only=False):
        """Hflat is (nrow, n) row-major; row = p0row[p0].  Returns
        (pairs int32, indptr int64 over pairs, costs float64, overflow, work).
        When count_only is True the pairs buffer is not materialised, so the
        enumeration can run to an exact cycle count without the 10+ GB blowup."""
        self._keep = (ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid, nb_cost,
                      st_tail, st_head, st_cost, Hflat, p0row)
        cdef Ctx c
        memset(&c, 0, sizeof(Ctx))
        cdef double* Hp
        cdef cnp.int32_t* PR
        if Hflat.shape[0] > 0:
            Hp = &Hflat[0]
        else:
            Hp = <double*> NULL
        if p0row.shape[0] > 0:
            PR = &p0row[0]
        else:
            PR = <cnp.int32_t*> NULL
        _ctx_set_graph(&c, n,
                       &ch_ptr[0], &ch_idx[0], &nb_ptr[0], &nb_head[0],
                       &nb_aid[0], &nb_cost[0], &st_tail[0], &st_head[0],
                       &st_cost[0], Hp, PR,
                       use_bound and Hflat.shape[0] > 0,
                       budget, cap, deadline, count_only)

        # The DFS scratch buffers must be sized for the CURRENT n, not for the
        # n of whichever call happened to run first -- reusing a smaller buffer
        # corrupts the heap (observed: segfault going from n=60 to n=100).
        if self.s_vis == NULL or n > self.alloc_n:
            self.s_vis = <unsigned char*> realloc(self.s_vis, n)
            self.s_pa = <cnp.int32_t*> realloc(self.s_pa, n * sizeof(int))
            self.s_pb = <cnp.int32_t*> realloc(self.s_pb, n * sizeof(int))
            self.s_pa_aid = <cnp.int32_t*> realloc(self.s_pa_aid, n * sizeof(int))
            self.s_pa_ci = <cnp.int32_t*> realloc(self.s_pa_ci, n * sizeof(int))
            if (self.s_vis == NULL or self.s_pa == NULL or self.s_pb == NULL
                    or self.s_pa_aid == NULL or self.s_pa_ci == NULL):
                raise MemoryError("dfs buffers")
            self.alloc_n = n
        c.vis = self.s_vis
        c.pa = self.s_pa
        c.pb = self.s_pb
        c.pa_aid = self.s_pa_aid
        c.pa_ci = self.s_pa_ci
        if not count_only:
            if self.s_pairs == NULL:
                self.s_pairs = <cnp.int32_t*> malloc(_PAIR_CHUNK * sizeof(int))
                self.s_cap_pairs = _PAIR_CHUNK
            c.o_pairs = self.s_pairs
            c.o_cap_pairs = self.s_cap_pairs
            if self.s_indptr == NULL:
                self.s_indptr = <long long*> malloc((_CYC_CHUNK + 1) * sizeof(long long))
                self.s_cost = <double*> malloc(_CYC_CHUNK * sizeof(double))
                self.s_cap_cyc = _CYC_CHUNK
            c.o_indptr = self.s_indptr
            c.o_cost = self.s_cost
            c.o_cap_cyc = self.s_cap_cyc
            c.o_indptr[0] = 0

        cdef int a0, nar = <int> st_tail.shape[0]
        for a0 in range(nar):
            if _run_arc(&c, a0, NULL):
                break

        if count_only:
            # 2nd slot is the exact cycle COUNT as a Python int (not an array)
            return None, c.ncyc, np.empty(0, dtype=np.float64), c.overflow, c.work

        # write grown buffers back so the next call reuses them
        self.s_pairs = c.o_pairs
        self.s_cap_pairs = c.o_cap_pairs
        self.s_indptr = c.o_indptr
        self.s_cost = c.o_cost
        self.s_cap_cyc = c.o_cap_cyc
        if c.memfail:
            raise MemoryError("cycle buffer")

        pairs = np.empty(c.o_n_slots, dtype=np.int32)
        indptr = np.empty(c.ncyc + 1, dtype=np.int64)
        costs = np.empty(c.ncyc, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] pairs_mv = pairs
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indptr_mv = indptr
        cdef cnp.ndarray[cnp.float64_t, ndim=1] costs_mv = costs
        cdef long long i
        for i in range(c.o_n_slots):
            pairs_mv[i] = c.o_pairs[i]
        for i in range(c.ncyc + 1):
            indptr_mv[i] = c.o_indptr[i]
        for i in range(c.ncyc):
            costs_mv[i] = c.o_cost[i]
        return pairs, indptr, costs, c.overflow, c.work

    # ------------------------------------------------------------------
    # store (disk) mode: write compact records instead of RAM buffers
    # ------------------------------------------------------------------
    def run_store(self,
                  int n,
                  cnp.ndarray[cnp.int32_t, ndim=1] ch_ptr,
                  cnp.ndarray[cnp.int32_t, ndim=1] ch_idx,
                  cnp.ndarray[cnp.int32_t, ndim=1] nb_ptr,
                  cnp.ndarray[cnp.int32_t, ndim=1] nb_head,
                  cnp.ndarray[cnp.int32_t, ndim=1] nb_aid,
                  cnp.ndarray[cnp.float64_t, ndim=1] nb_cost,
                  cnp.ndarray[cnp.int32_t, ndim=1] st_tail,
                  cnp.ndarray[cnp.int32_t, ndim=1] st_head,
                  cnp.ndarray[cnp.float64_t, ndim=1] st_cost,
                  cnp.ndarray[cnp.float64_t, ndim=1] Hflat,
                  cnp.ndarray[cnp.int32_t, ndim=1] p0row,
                  bint use_bound,
                  double budget,
                  long long cap,
                  double deadline,
                  str out_path,
                  int buf_bytes=1 << 20):
        """Enumerate and stream compact records to out_path (serial).

        Returns (n_records, overflow, work).  Memory is O(1) in the cycle
        count: nothing but the DFS scratch and one record buffer is held.
        The record format and its cost order are identical to run(); the
        caller sorts the stream (cystore.sort_records).
        """
        self._keep = (ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid, nb_cost,
                      st_tail, st_head, st_cost, Hflat, p0row)
        cdef Ctx c
        memset(&c, 0, sizeof(Ctx))
        cdef double* Hp
        cdef cnp.int32_t* PR
        if Hflat.shape[0] > 0:
            Hp = &Hflat[0]
        else:
            Hp = <double*> NULL
        if p0row.shape[0] > 0:
            PR = &p0row[0]
        else:
            PR = <cnp.int32_t*> NULL
        cdef int maxdeg = 1
        cdef int i
        for i in range(n):
            if ch_ptr[i + 1] - ch_ptr[i] > maxdeg:
                maxdeg = ch_ptr[i + 1] - ch_ptr[i]
        cdef int m_nb = <int> st_tail.shape[0]
        _ctx_set_graph(&c, n,
                       &ch_ptr[0], &ch_idx[0], &nb_ptr[0], &nb_head[0],
                       &nb_aid[0], &nb_cost[0], &st_tail[0], &st_head[0],
                       &st_cost[0], Hp, PR,
                       use_bound and Hflat.shape[0] > 0,
                       budget, cap, deadline, False)
        if self.s_vis == NULL or n > self.alloc_n:
            self.s_vis = <unsigned char*> realloc(self.s_vis, n)
            self.s_pa = <cnp.int32_t*> realloc(self.s_pa, n * sizeof(int))
            self.s_pb = <cnp.int32_t*> realloc(self.s_pb, n * sizeof(int))
            self.s_pa_aid = <cnp.int32_t*> realloc(self.s_pa_aid, n * sizeof(int))
            self.s_pa_ci = <cnp.int32_t*> realloc(self.s_pa_ci, n * sizeof(int))
            if (self.s_vis == NULL or self.s_pa == NULL or self.s_pb == NULL
                    or self.s_pa_aid == NULL or self.s_pa_ci == NULL):
                raise MemoryError("dfs buffers")
            self.alloc_n = n
        c.vis = self.s_vis
        c.pa = self.s_pa
        c.pb = self.s_pb
        c.pa_aid = self.s_pa_aid
        c.pa_ci = self.s_pa_ci
        c.store_mode = True
        c.bits_tree = _bits_for(1 + maxdeg)
        c.bits_nb = _bits_for(m_nb if m_nb > 1 else 2)
        c.rec_cap = RHDR + 8 * n + 8
        c.rbuf = <uint8_t*> malloc(<size_t> c.rec_cap)
        if c.rbuf == NULL:
            raise MemoryError("record buffer")
        c.out = fopen(out_path.encode(), "wb")
        if c.out == NULL:
            free(c.rbuf)
            raise OSError("cannot open " + out_path)
        setvbuf(c.out, NULL, _IOFBF, buf_bytes)
        cdef int a0, nar = <int> st_tail.shape[0]
        try:
            for a0 in range(nar):
                if _run_arc(&c, a0, NULL):
                    break
            fflush(c.out)
        finally:
            fclose(c.out)
            free(c.rbuf)
            c.rbuf = NULL
            c.out = NULL
        return c.ncyc, c.overflow, c.work

    def run_par_store(self,
                      int n,
                      cnp.ndarray[cnp.int32_t, ndim=1] ch_ptr,
                      cnp.ndarray[cnp.int32_t, ndim=1] ch_idx,
                      cnp.ndarray[cnp.int32_t, ndim=1] nb_ptr,
                      cnp.ndarray[cnp.int32_t, ndim=1] nb_head,
                      cnp.ndarray[cnp.int32_t, ndim=1] nb_aid,
                      cnp.ndarray[cnp.float64_t, ndim=1] nb_cost,
                      cnp.ndarray[cnp.int32_t, ndim=1] st_tail,
                      cnp.ndarray[cnp.int32_t, ndim=1] st_head,
                      cnp.ndarray[cnp.float64_t, ndim=1] st_cost,
                      cnp.ndarray[cnp.float64_t, ndim=1] Hflat,
                      cnp.ndarray[cnp.int32_t, ndim=1] p0row,
                      bint use_bound,
                      double budget,
                      long long cap,
                      double deadline,
                      list out_paths,
                      int nthreads=0,
                      int buf_bytes=1 << 20):
        """OpenMP version of run_store: contiguous start-arc ranges per
        thread, one file per thread.  Concatenating the files in thread order
        reproduces the serial emission order, so the (cost, emission index)
        sort of the concatenation matches the serial one exactly.

        Returns (n_records, overflow, work).
        """
        if nthreads <= 0:
            nthreads = omp_get_max_threads()
        if nthreads < 1:
            nthreads = 1
        if nthreads == 1:
            return self.run_store(n, ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid,
                                  nb_cost, st_tail, st_head, st_cost, Hflat,
                                  p0row, use_bound, budget, cap, deadline,
                                  out_paths[0], buf_bytes)
        self._keep = (ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid, nb_cost,
                      st_tail, st_head, st_cost, Hflat, p0row)
        cdef int nar = <int> st_tail.shape[0]
        cdef bint ub = use_bound and Hflat.shape[0] > 0
        cdef double* Hp = &Hflat[0] if Hflat.shape[0] > 0 else <double*> NULL
        cdef cnp.int32_t* PR = &p0row[0] if p0row.shape[0] > 0 else <cnp.int32_t*> NULL
        cdef long long shcap = cap // nthreads + 1
        cdef int maxdeg = 1
        cdef int i
        for i in range(n):
            if ch_ptr[i + 1] - ch_ptr[i] > maxdeg:
                maxdeg = ch_ptr[i + 1] - ch_ptr[i]
        cdef int m_nb = <int> st_tail.shape[0]
        cdef int bits_tree = _bits_for(1 + maxdeg)
        cdef int bits_nb = _bits_for(m_nb if m_nb > 1 else 2)
        cdef long long rec_cap = RHDR + 8 * n + 8

        cdef Ctx* tctx = <Ctx*> calloc(nthreads, sizeof(Ctx))
        if tctx == NULL:
            raise MemoryError("thread contexts")
        cdef int t, rc = 0, lo, hi, a0
        cdef Ctx* ctx
        cdef long long tot = 0, totw = 0
        cdef bint ovf = False
        cdef long long* bnds = <long long*> malloc((nthreads + 1) * sizeof(long long))
        if bnds == NULL:
            free(tctx)
            raise MemoryError("shard bounds")
        try:
            for t in range(nthreads):
                _ctx_set_graph(&tctx[t], n,
                               &ch_ptr[0], &ch_idx[0], &nb_ptr[0], &nb_head[0],
                               &nb_aid[0], &nb_cost[0], &st_tail[0], &st_head[0],
                               &st_cost[0], Hp, PR, ub, budget, shcap, deadline,
                               False)
                rc = _ctx_alloc_scratch(&tctx[t], n)
                if rc != 0:
                    break
                tctx[t].store_mode = True
                tctx[t].bits_tree = bits_tree
                tctx[t].bits_nb = bits_nb
                tctx[t].rec_cap = rec_cap
                tctx[t].rbuf = <uint8_t*> malloc(<size_t> rec_cap)
                tctx[t].out = fopen(out_paths[t].encode(), "wb")
                if tctx[t].rbuf == NULL or tctx[t].out == NULL:
                    rc = 2
                    break
                setvbuf(tctx[t].out, NULL, _IOFBF, buf_bytes)
            if rc != 0:
                raise MemoryError("thread store setup")
            for t in range(nthreads):
                bnds[t] = (<long long> t) * nar // nthreads
            bnds[nthreads] = nar
            with nogil, parallel(num_threads=nthreads):
                t = omp_get_thread_num()
                lo = <int> bnds[t]
                hi = <int> bnds[t + 1]
                ctx = &tctx[t]
                for a0 in range(lo, hi):
                    if _run_arc(ctx, a0, NULL):
                        break
        finally:
            tot = 0
            totw = 0
            ovf = False
            for t in range(nthreads):
                tot += tctx[t].ncyc
                totw += tctx[t].work
                if tctx[t].overflow:
                    ovf = True
                if tctx[t].memfail:
                    rc = 3
            for t in range(nthreads):
                if tctx[t].out != NULL:
                    fflush(tctx[t].out)
                    fclose(tctx[t].out)
                if tctx[t].rbuf != NULL:
                    free(tctx[t].rbuf)
                _ctx_free_shard(&tctx[t])
            free(tctx)
            free(bnds)
        if rc == 3:
            raise MemoryError("record write failed")
        return tot, ovf, totw

    def run_par(self,
                int n,
                cnp.ndarray[cnp.int32_t, ndim=1] ch_ptr,
                cnp.ndarray[cnp.int32_t, ndim=1] ch_idx,
                cnp.ndarray[cnp.int32_t, ndim=1] nb_ptr,
                cnp.ndarray[cnp.int32_t, ndim=1] nb_head,
                cnp.ndarray[cnp.int32_t, ndim=1] nb_aid,
                cnp.ndarray[cnp.float64_t, ndim=1] nb_cost,
                cnp.ndarray[cnp.int32_t, ndim=1] st_tail,
                cnp.ndarray[cnp.int32_t, ndim=1] st_head,
                cnp.ndarray[cnp.float64_t, ndim=1] st_cost,
                cnp.ndarray[cnp.float64_t, ndim=1] Hflat,
                cnp.ndarray[cnp.int32_t, ndim=1] p0row,
                bint use_bound,
                double budget,
                long long cap,
                double deadline,
                bint count_only=False,
                int nthreads=0,
                bint split=True,
                cnp.ndarray[cnp.int64_t, ndim=1] work_out=None):
        """OpenMP version of run(); same returns, same order, same values.

        count_only with split=True (the default): the enumeration is expanded
        into first-level work items -- one close item plus one item per
        descendable child of each start arc -- and the items are scheduled
        dynamically.  This breaks the subtree-size imbalance that caps
        arc-granular sharding at ~2.6x (measured: the largest start arc is 38%
        of all node visits at n=1500).  The item stream covers exactly the
        serial cycle set, so the count is their exact sum.
        count_only with split=False: arc-granular dynamic sharding (the
        reference arm; also the only arm that fills work_out).
        materialising: contiguous start-arc ranges per thread, per-thread
        buffers concatenated in thread order -- bit-identical to run() whenever
        no shard reaches its (per-shard) cap.

        work_out (diagnostics only, split=False count path): if given, an int64
        array of length nar receiving the DFS node-visit count of each start
        arc.  Every arc is owned by exactly one thread, so the writes are
        race-free.
        """
        if nthreads <= 0:
            nthreads = omp_get_max_threads()
        if nthreads < 1:
            nthreads = 1
        if nthreads == 1:
            return self.run(n, ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid,
                            nb_cost, st_tail, st_head, st_cost, Hflat, p0row,
                            use_bound, budget, cap, deadline, count_only)
        self._keep = (ch_ptr, ch_idx, nb_ptr, nb_head, nb_aid, nb_cost,
                      st_tail, st_head, st_cost, Hflat, p0row)
        cdef int nar = <int> st_tail.shape[0]
        cdef bint ub = use_bound and Hflat.shape[0] > 0
        cdef double* Hp = &Hflat[0] if Hflat.shape[0] > 0 else <double*> NULL
        cdef cnp.int32_t* PR = &p0row[0] if p0row.shape[0] > 0 else <cnp.int32_t*> NULL
        # cap is per shard, so a capped run emits at most cap + nthreads
        cdef long long shcap = cap // nthreads + 1

        cdef Ctx* tctx = <Ctx*> calloc(nthreads, sizeof(Ctx))
        if tctx == NULL:
            raise MemoryError("thread contexts")
        cdef int t, rc = 0
        for t in range(nthreads):
            _ctx_set_graph(&tctx[t], n,
                           &ch_ptr[0], &ch_idx[0], &nb_ptr[0], &nb_head[0],
                           &nb_aid[0], &nb_cost[0], &st_tail[0], &st_head[0],
                           &st_cost[0], Hp, PR, ub, budget, shcap, deadline,
                           count_only)
            rc = _ctx_alloc_scratch(&tctx[t], n)
            if rc != 0:
                break
        if rc != 0:
            for t in range(nthreads):
                _ctx_free_shard(&tctx[t])
            free(tctx)
            raise MemoryError("thread scratch")

        cdef bint stopbuf[1]
        stopbuf[0] = False
        cdef bint* stop = &stopbuf[0]
        cdef int tid, a0, lo, hi
        cdef Ctx* ctx
        cdef Ctx* c1 = &tctx[0]
        cdef long long tot_cyc = 0, tot_work = 0, slots_off = 0, cyc_off = 0, i
        cdef long long w0, n_items = 0, pos = 0, it
        cdef cnp.int32_t* it_a0 = NULL
        cdef cnp.int32_t* it_w = NULL
        cdef cnp.int32_t* it_ci = NULL
        cdef double* it_cost = NULL
        cdef bint oom = False
        cdef bint any_ovf = False, any_memfail = False
        cdef cnp.int64_t* wbuf = NULL
        if work_out is not None and work_out.shape[0] >= nar:
            wbuf = &work_out[0]
        # contiguous shard bounds for the materialising path
        cdef long long* bnds = <long long*> malloc((nthreads + 1) * sizeof(long long))
        if bnds == NULL:
            for t in range(nthreads):
                _ctx_free_shard(&tctx[t])
            free(tctx)
            raise MemoryError("shard bounds")
        for t in range(nthreads):
            bnds[t] = (<long long> t) * nar // nthreads
        bnds[nthreads] = nar

        if count_only and split:
            # ---- phase 1: expand start arcs into first-level work items ----
            # Splitting at the child level is what breaks the subtree-size
            # imbalance: measured at n=1500 the single largest start arc is
            # 38% of all DFS node visits and the top ten arcs are 88%, so
            # arc-granular sharding cannot exceed ~2.6x at any thread count.
            # The close item (w = -1) keeps the (A) emissions exactly once per
            # arc, so the item stream covers the same cycle set as the serial
            # loop -- each cycle still generated by exactly one item.
            for a0 in range(nar):
                c1.p0 = c1.st_tail[a0]
                c1.a0 = a0
                if c1.use_bound:
                    c1.Hrow = c1.H + (<long long> c1.p0row[c1.p0]) * c1.n
                n_items += _expand(c1, a0, NULL, NULL, NULL, NULL, 0, False)
            if n_items > 0:
                it_a0 = <cnp.int32_t*> malloc(n_items * sizeof(cnp.int32_t))
                it_w = <cnp.int32_t*> malloc(n_items * sizeof(cnp.int32_t))
                it_ci = <cnp.int32_t*> malloc(n_items * sizeof(cnp.int32_t))
                it_cost = <double*> malloc(n_items * sizeof(double))
                if it_a0 == NULL or it_w == NULL or it_ci == NULL or it_cost == NULL:
                    oom = True
                else:
                    for a0 in range(nar):
                        c1.p0 = c1.st_tail[a0]
                        c1.a0 = a0
                        if c1.use_bound:
                            c1.Hrow = c1.H + (<long long> c1.p0row[c1.p0]) * c1.n
                        pos = _expand(c1, a0, it_a0, it_w, it_ci,
                                      it_cost, pos, True)
            if not oom:
                with nogil, parallel(num_threads=nthreads):
                    tid = omp_get_thread_num()
                    ctx = &tctx[tid]
                    for it in prange(n_items, schedule='dynamic', chunksize=1):
                        if not stop[0]:
                            _run_item(ctx, it_a0[it], it_w[it],
                                      it_ci[it], it_cost[it], stop)
            if it_a0 != NULL:
                free(it_a0)
            if it_w != NULL:
                free(it_w)
            if it_ci != NULL:
                free(it_ci)
            if it_cost != NULL:
                free(it_cost)
            if oom:
                for t in range(nthreads):
                    _ctx_free_shard(&tctx[t])
                free(tctx)
                free(bnds)
                raise MemoryError("item list")
        elif count_only:
            with nogil, parallel(num_threads=nthreads):
                tid = omp_get_thread_num()
                ctx = &tctx[tid]
                for a0 in prange(nar, schedule='dynamic', chunksize=1):
                    if not stop[0]:
                        w0 = ctx.work
                        _run_arc(ctx, a0, stop)
                        if wbuf != NULL:
                            wbuf[a0] = ctx.work - w0
        else:
            with nogil, parallel(num_threads=nthreads):
                tid = omp_get_thread_num()
                lo = <int> bnds[tid]
                hi = <int> bnds[tid + 1]
                ctx = &tctx[tid]
                for a0 in range(lo, hi):
                    if _run_arc(ctx, a0, NULL):
                        break

        for t in range(nthreads):
            tot_cyc += tctx[t].ncyc
            tot_work += tctx[t].work
            if tctx[t].overflow:
                any_ovf = True
            if tctx[t].memfail:
                any_memfail = True
        if any_memfail:
            for t in range(nthreads):
                _ctx_free_shard(&tctx[t])
            free(tctx)
            free(bnds)
            raise MemoryError("cycle buffer")

        if count_only:
            for t in range(nthreads):
                _ctx_free_shard(&tctx[t])
            free(tctx)
            free(bnds)
            return None, tot_cyc, np.empty(0, dtype=np.float64), any_ovf, tot_work

        # concatenate shard streams in thread order == start-arc order
        cdef long long tot_slots = 0
        for t in range(nthreads):
            tot_slots += tctx[t].o_n_slots
        pairs = np.empty(tot_slots, dtype=np.int32)
        indptr = np.empty(tot_cyc + 1, dtype=np.int64)
        costs = np.empty(tot_cyc, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] pairs_mv = pairs
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indptr_mv = indptr
        cdef cnp.ndarray[cnp.float64_t, ndim=1] costs_mv = costs
        indptr_mv[0] = 0
        for t in range(nthreads):
            if tctx[t].ncyc == 0:
                continue
            for i in range(tctx[t].o_n_slots):
                pairs_mv[slots_off + i] = tctx[t].o_pairs[i]
            for i in range(tctx[t].ncyc):
                indptr_mv[cyc_off + i + 1] = (slots_off >> 1) + tctx[t].o_indptr[i + 1]
                costs_mv[cyc_off + i] = tctx[t].o_cost[i]
            slots_off += tctx[t].o_n_slots
            cyc_off += tctx[t].ncyc
        for t in range(nthreads):
            _ctx_free_shard(&tctx[t])
        free(tctx)
        free(bnds)
        return pairs, indptr, costs, any_ovf, tot_work
