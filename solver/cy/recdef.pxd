# Shared record format for the disk-backed cycle store.
# Included by both cyenum.pyx (writer) and cystore.pyx (reader) so the two
# can never drift apart.
#
# Record layout (variable length, self-delimiting):
#
#     [0:8]    cost      float64, raw
#     [8:12]   a0        int32,   index of the start non-basic arc
#     [12:14]  nmoves    uint16,  moves after the start arc
#     [14:...]  moves     bit packed
#
# A move is one arc of the cycle after the start arc; the close is a move too:
#
#     tree move:  bit 0  + bits_tree bits  (index into children[v])
#     nb   move:  bit 1  + bits_nb   bits  (global non-basic arc id)
#
# Replaying the moves walks the folded basis structure from q0 and ends
# exactly at p0.  The basis structure is NOT a tree (build_basis_tree Kruskals
# a bipartite graph and folds it onto n nodes), so the path between two
# anchors is not unique and the encoding must store the actual walk.

DEF RHDR = 14

cdef struct BitW:
    uint8_t* buf
    long long nbit


cdef void bw_put(BitW* w, uint32_t v, int nb) noexcept nogil:
    """Append the low nb bits of v.  Buffer bytes must be zeroed first."""
    cdef long long byte = w.nbit >> 3
    cdef int off = <int> (w.nbit & 7)
    cdef uint64_t acc = <uint64_t> v << off
    cdef int nbytes = (off + nb + 7) >> 3
    cdef int i
    for i in range(nbytes):
        w.buf[byte + i] |= <uint8_t> ((acc >> (8 * i)) & 0xFF)
    w.nbit += nb


cdef uint32_t bw_get(const uint8_t* buf, long long* pos,
                      int nb) noexcept nogil:
    cdef long long byte = pos[0] >> 3
    cdef int off = <int> (pos[0] & 7)
    cdef uint64_t acc = <uint64_t> 0
    cdef int nbytes = (off + nb + 7) >> 3
    cdef int i
    for i in range(nbytes):
        acc |= (<uint64_t> buf[byte + i]) << (8 * i)
    cdef uint32_t mask
    if nb >= 32:
        mask = <uint32_t> 0xFFFFFFFF
    else:
        mask = (<uint32_t> 1 << nb) - 1
    pos[0] += nb
    return <uint32_t> ((acc >> off) & mask)


cdef int _bits_for(int m) noexcept nogil:
    """Bits needed to hold values 0..m-1 (at least 1)."""
    cdef int b = 0
    if m <= 2:
        return 1
    m -= 1
    while m > 0:
        b += 1
        m >>= 1
    return b


cdef long long _rec_move_bytes(int nmoves, int bits_tree,
                            int bits_nb) noexcept nogil:
    """Worst-case byte length of the move field (used to size buffers)."""
    cdef int w = bits_tree if bits_tree > bits_nb else bits_nb
    return (nmoves * (1 + w) + 7) >> 3


cdef long long rec_skip(const uint8_t* buf, long long pos, long long end,
                        long long hard, int bits_tree,
                        int bits_nb) noexcept nogil:
    """Byte offset just past the record starting at pos.

    Records are packed at their ACTUAL length (rec_write returns it), which
    depends on the tree/nb mix of the moves -- so the length cannot be
    recovered from the header alone; the move bits have to be walked.

    end   = end of the VALID data for this record
    hard  = end of the allocated buffer (the bit reader may touch up to
            pos + RHDR + nm*(1+w) bits' worth of bytes, so hard must cover
            that even when the record itself is shorter)
    Returns -1 if the record does not fit in [pos, end).
    """
    cdef uint16_t nm
    cdef long long bit, worst
    cdef uint32_t v
    cdef int t, w
    if pos + RHDR > end:
        return -1
    memcpy(&nm, buf + pos + 12, 2)
    w = bits_tree if bits_tree > bits_nb else bits_nb
    worst = RHDR + (((<long long> nm) * (1 + w) + 7) >> 3)
    if pos + worst > hard:
        return -1
    bit = (pos + RHDR) * 8
    for t in range(nm):
        v = bw_get(buf, &bit, 1)
        if (v & 1) == 0:
            bw_get(buf, &bit, bits_tree)
        else:
            bw_get(buf, &bit, bits_nb)
    worst = (bit + 7) >> 3          # absolute end of the record
    if worst > end:
        return -1
    return worst - pos


cdef long long rec_write(uint8_t* out, long long cap, double cost,
                                int32_t a0, int npath,
                                const int32_t* mv_aid, const int32_t* mv_ci,
                                int close_is_nb, int32_t close_id,
                                int bits_tree, int bits_nb) noexcept nogil:
    """Serialise one record.  Returns its byte length, or -1 if > cap.

    mv_aid[k] >= 0  -> non-basic move with that arc id
    mv_aid[k] <  0  -> tree move, child index mv_ci[k]
    The close move comes last: (close_is_nb, close_id) with close_id an arc
    id when close_is_nb, else a child index.
    """
    cdef BitW w
    cdef int k, nmoves = npath + 1
    cdef long long need = RHDR + _rec_move_bytes(nmoves, bits_tree, bits_nb)
    if nmoves > 65535 or need > cap:
        return -1
    memset(out, 0, <size_t> need)
    memcpy(out, &cost, 8)
    memcpy(out + 8, &a0, 4)
    memcpy(out + 12, &nmoves, 2)
    w.buf = out + RHDR
    w.nbit = 0
    for k in range(npath):
        if mv_aid[k] >= 0:
            bw_put(&w, <uint32_t> 1, 1)
            bw_put(&w, <uint32_t> mv_aid[k], bits_nb)
        else:
            bw_put(&w, <uint32_t> 0, 1)
            bw_put(&w, <uint32_t> mv_ci[k], bits_tree)
    if close_is_nb:
        bw_put(&w, <uint32_t> 1, 1)
        bw_put(&w, <uint32_t> close_id, bits_nb)
    else:
        bw_put(&w, <uint32_t> 0, 1)
        bw_put(&w, <uint32_t> close_id, bits_tree)
    return RHDR + ((w.nbit + 7) >> 3)
