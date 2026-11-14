"""Validate cypack.build_masks against the Python packing preprocessing.

Every one of the four outputs (mask, cmask, suffix_or, cover_list) must be
bit-for-bit identical, because solve()'s packing search consumes them directly.
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
import cypack

CAP = 60_000


def build(n, seed=1, kind="uniform", k_frac=0.03):
    rng = np.random.default_rng(1000 * n + seed)
    if kind == "uniform":
        C = rng.random((n, n))
    elif kind == "exponential":
        C = rng.exponential(1.0, (n, n))
    elif kind == "ties":
        C = rng.integers(0, 5, (n, n)).astype(float)
    else:
        raise ValueError(kind)
    np.fill_diagonal(C, np.inf)
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
    return C, phi, G, b


def main():
    print(f"{'n':>5} {'kind':>11} {'cyc':>8} | {'mask':>6} {'cmask':>6} "
          f"{'suffix':>7} {'cover':>6} | {'py(s)':>8} {'cy(s)':>8} {'speed':>8}")
    bad = 0
    for n in (60, 100, 150, 200):
        for kind in ("uniform", "exponential", "ties"):
            C, phi, G, b = build(n, 1, kind)
            H = qa._return_bound(G)
            cycles, costs, ovf = qa.enumerate_valid_cycles(
                G, b, cap=CAP, deadline=None, use_bound=True)
            if not cycles:
                print(f"{n:5d} {kind:>11} {0:8d} | (no cycles)")
                continue
            pairs, indptr, ccosts, covf, work = cybridge.enumerate_cycles_cy(
                G, b, cap=CAP, deadline=None, use_bound=True, H=H)

            fcid, fpos, nfc = qa.phi_cycle_labels(phi, n)

            # ---- reference: exactly what solve() does today ----
            t = time.perf_counter()
            order = sorted(range(len(cycles)), key=lambda t_: costs[t_])
            cyc_sorted = [cycles[t_] for t_ in order]
            mask_sorted = []
            cmask_sorted = []
            for edges in cyc_sorted:
                m = 0
                cm = 0
                for a, _ in edges:
                    m |= 1 << a
                    cm |= 1 << fcid[a]
                mask_sorted.append(m)
                cmask_sorted.append(cm)
            suffix_or = [0] * (len(cmask_sorted) + 1)
            for i in range(len(cmask_sorted) - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | cmask_sorted[i]
            cover_list = [[] for _ in range(nfc)]
            for idx, cmm in enumerate(cmask_sorted):
                bits = cmm
                while bits:
                    low = bits & -bits
                    cover_list[low.bit_length() - 1].append(idx)
                    bits ^= low
            t1 = time.perf_counter() - t

            # ---- compiled: same thing from the flat stream ----
            # the flat stream is in emission order, so apply the same
            # cost-sorted permutation first
            c_order = np.argsort(np.asarray(ccosts, dtype=np.float64),
                                 kind="stable")
            t = time.perf_counter()
            m2, cm2, s2, cv2 = cypack.build_masks(
                pairs, indptr, np.asarray(fcid, dtype=np.int64), n, nfc,
                order=c_order)
            t2 = time.perf_counter() - t

            ok = (len(m2) == len(mask_sorted)
                  and m2 == mask_sorted and cm2 == cmask_sorted
                  and s2 == suffix_or and cv2 == cover_list)
            if not ok:
                bad += 1
            print(f"{n:5d} {kind:>11} {len(cycles):8d} | "
                  f"{str(m2 == mask_sorted):>6} {str(cm2 == cmask_sorted):>6} "
                  f"{str(s2 == suffix_or):>7} {str(cv2 == cover_list):>6} | "
                  f"{t1:8.3f} {t2:8.3f} {t1 / max(t2, 1e-9):7.1f}x", flush=True)
            del cycles, costs, pairs, indptr, ccosts, m2, cm2, s2, cv2
            del mask_sorted, cmask_sorted, suffix_or, cover_list
            gc.collect()
    print(f"\nmismatches: {bad}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
