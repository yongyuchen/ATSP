# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Compiled preprocessing for the packing stage.

`quasipoly_atsp.solve` turns every enumerated cycle into two Python big ints

    mask  = OR over the cycle's arcs of  (1 << tail)      -- n bits
    cmask = OR over the cycle's arcs of  (1 << fcid[tail])-- nfc bits

and then ORs the cmasks into a suffix array.  At n = 1500 a single mask is
already ~190 bytes and the loop is pure Python, so once the enumerator can
hand over 10^6 cycles this preprocessing becomes the second bottleneck.

It does not touch the combinatorial content: masks are accumulated into
uint64 words with the same bit order as the Python `|=` version and only then
materialised as Python ints (little-endian, identical value).
"""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cnp.import_array()


def build_masks(cnp.ndarray[cnp.int32_t, ndim=1] pairs,
                cnp.ndarray[cnp.int64_t, ndim=1] indptr,
                cnp.ndarray[cnp.int64_t, ndim=1] fcid,
                int n, int nfc,
                cnp.ndarray[cnp.int64_t, ndim=1] order=None):
    """Returns (mask, cmask, suffix_or, cover_list) exactly as solve() builds
    them from the tuple representation, given the flat cycle stream.

    If `order` is given, output slot i describes stream cycle order[i] -- this
    is the cost-sorted permutation, applied inside the kernel so that the
    (10^6-entry, tens-of-MB) pair array never has to be physically permuted.

    mask[i], cmask[i]      Python ints for cycle i
    suffix_or[i]           OR of cmask[i:], with suffix_or[nc] = 0
    cover_list[s]          indices i (ascending) whose cmask touches phi-cycle s
    """
    cdef long long nc = indptr.shape[0] - 1
    cdef cnp.int64_t* ordp = NULL
    if order is not None:
        if order.shape[0] != nc:
            raise ValueError("order length mismatch")
        ordp = &order[0]
    cdef int W = (n + 63) >> 6
    cdef int Wc = (nfc + 63) >> 6
    cdef int nbytes = W * 8
    cdef int ncbytes = Wc * 8

    cdef unsigned long long* mw = <unsigned long long*> malloc(
        <size_t> (nbytes if nbytes else 8))
    cdef unsigned long long* cw = <unsigned long long*> malloc(
        <size_t> (ncbytes if ncbytes else 8))
    cdef unsigned long long* suf = <unsigned long long*> malloc(
        <size_t> ((nc + 1) * Wc * 8 if (nc + 1) * Wc else 8))
    cdef long long i, p, s, si
    cdef int a, w
    cdef unsigned long long bit
    if mw == NULL or cw == NULL or suf == NULL:
        raise MemoryError("build_masks")
    try:
        mask = [0] * nc
        cmask = [0] * nc

        for i in range(nc):
            for w in range(W):
                mw[w] = 0
            for w in range(Wc):
                cw[w] = 0
            if ordp != NULL:
                si = ordp[i]
            else:
                si = i
            for p in range(indptr[si], indptr[si + 1]):
                a = pairs[2 * p]
                bit = <unsigned long long> 1
                mw[a >> 6] |= bit << (a & 63)
                a = <int> fcid[a]
                cw[a >> 6] |= bit << (a & 63)
            mask[i] = int.from_bytes(
                (<char*> mw)[:nbytes], "little")
            cmask[i] = int.from_bytes(
                (<char*> cw)[:ncbytes], "little")
            # suffix_or is filled backwards below, so stash the words
            for w in range(Wc):
                suf[i * Wc + w] = cw[w]
        for w in range(Wc):
            suf[nc * Wc + w] = 0
        suffix_or = [0] * (nc + 1)
        for i in range(nc - 1, -1, -1):
            for w in range(Wc):
                suf[i * Wc + w] = suf[i * Wc + w] | suf[(i + 1) * Wc + w]
            suffix_or[i] = int.from_bytes(
                (<char*> (suf + i * Wc))[:ncbytes], "little")

        cover_list = [[] for _ in range(nfc)]
        for i in range(nc):
            cm = cmask[i]
            bits = cm
            while bits:
                low = bits & -bits
                cover_list[low.bit_length() - 1].append(i)
                bits ^= low
        return mask, cmask, suffix_or, cover_list
    finally:
        free(mw)
        free(cw)
        free(suf)
