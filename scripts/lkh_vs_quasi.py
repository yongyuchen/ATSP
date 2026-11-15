"""LKH-3 vs our exact Bell-Frieze solver.
   Same instances (i.i.d. U[0,1] asymmetric costs -- the Bell-Frieze regime).
   LKH is run twice per instance: RUNS=1 (minimum effort) and RUNS=10.
   Costs are written to TSPLIB EXPLICIT FULL_MATRIX scaled by 1e6 and rounded to int.

   *** TYPE must be ATSP, not TSP ***  (verified with dbg_lkh_type.py: with TYPE: TSP
   LKH treats the explicit matrix as symmetric and returns tours 100-240% above the
   Held-Karp optimum; with TYPE: ATSP it hits the optimum even at RUNS=1.)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
import numpy as np
import quasipoly_atsp as qa

SCALE = 1_000_000
SIZES = [100, 200, 300, 500, 800, 1000]
SEEDS = [1, 2, 3]
OURS_TL = 60.0

# /tmp is a 10 MB tmpfs on this machine; a 1500x1500 FULL_MATRIX file does not
# fit.  Put LKH's scratch files under /home instead.
LKH_TMP = os.environ.get("LKH_TMP", os.path.join(_ROOT, "_work", "lkh"))
os.makedirs(LKH_TMP, exist_ok=True)


def lkh(C, runs, tmpdir=None):
    """returns (tour_cost_in_original_units, wall_seconds); C has inf diagonal"""
    n = C.shape[0]
    big = int(SCALE * 10)                      # substitute for inf diagonal
    Cf = np.where(np.isfinite(C), C, big / SCALE)
    Ci = np.rint(Cf * SCALE).astype(np.int64)
    np.fill_diagonal(Ci, big)
    with tempfile.TemporaryDirectory(dir=tmpdir or LKH_TMP) as td:
        tsp, par, out = (os.path.join(td, f) for f in ("p.tsp", "p.par", "p.tour"))
        with open(tsp, "w") as f:
            f.write(f"NAME: p\nTYPE: ATSP\nDIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n):
                f.write(" ".join(map(str, Ci[i])) + "\n")
            f.write("EOF\n")
        with open(par, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp}\nTOUR_FILE = {out}\n"
                    f"RUNS = {runs}\nTRACE_LEVEL = 0\nSEED = 1\n")
        t0 = time.perf_counter()
        subprocess.run([os.environ.get("LKH_BIN", "LKH"), par], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        el = time.perf_counter() - t0
        tour, reading = [], False
        for line in open(out):
            if line.startswith("TOUR_SECTION"):
                reading = True; continue
            if reading:
                v = int(line.strip())
                if v == -1:
                    break
                tour.append(v - 1)
    assert len(tour) == n and len(set(tour)) == n, "LKH returned an invalid tour"
    cost = sum(C[tour[k], tour[(k + 1) % n]] for k in range(n))
    return float(cost), el


rows = []
print(f"{'n':>5} {'sd':>2} | {'ours cost':>10} {'proved':>6} {'ours t':>8} | "
      f"{'LKH1 cost':>10} {'LKH1 t':>8} | {'LKH10 cost':>11} {'LKH10 t':>8} | "
      f"{'ours/LKH10':>10} {'speedup1':>9} {'speedup10':>10}")
print("-" * 118)
for n in SIZES:
    for s in SEEDS:
        rng = np.random.default_rng(1000 * n + s)
        C = rng.random((n, n))
        np.fill_diagonal(C, np.inf)

        t0 = time.perf_counter()
        r = qa.solve(C, time_limit=OURS_TL)
        our_t = time.perf_counter() - t0
        our_c, proved = float(r.cost), bool(r.proved)

        l1c, l1t = lkh(C, 1)
        l10c, l10t = lkh(C, 10)

        rows.append(dict(n=n, seed=s, ours_cost=our_c, proved=proved, ours_t=our_t,
                         lkh1_cost=l1c, lkh1_t=l1t, lkh10_cost=l10c, lkh10_t=l10t))
        print(f"{n:5d} {s:2d} | {our_c:10.6f} {str(proved):>6} {our_t:8.3f} | "
              f"{l1c:10.6f} {l1t:8.3f} | {l10c:11.6f} {l10t:8.3f} | "
              f"{our_c/l10c:10.6f} {l1t/our_t:9.2f} {l10t/our_t:10.2f}", flush=True)

json.dump(rows, open("lkh_vs_quasi.json", "w"), indent=1)

print()
for tag in ("lkh1", "lkh10"):
    eq = sum(1 for r in rows if abs(r["ours_cost"] - r[f"{tag}_cost"]) < 1e-9)
    better = sum(1 for r in rows if r["ours_cost"] < r[f"{tag}_cost"] - 1e-9)
    worse = sum(1 for r in rows if r["ours_cost"] > r[f"{tag}_cost"] + 1e-9)
    sp = [r[f"{tag}_t"] / r["ours_t"] for r in rows]
    print(f"{tag}: identical {eq}/{len(rows)}   ours better {better}   ours worse {worse}   "
          f"speedup median {np.median(sp):.2f}x  range {min(sp):.2f}-{max(sp):.2f}")
print(f"ours proved: {sum(1 for r in rows if r['proved'])}/{len(rows)}")
