"""R4: sensitivity of the empirical scope marker |A~|/n^2 to the incumbent.

The statistic |A~|/n^2 = #{non-basic arcs with reduced cost <= Delta} / n^2
uses Delta = UB - Z_AP, so it is a *joint* statistic of the instance and the
initial incumbent.  This script recomputes it under four incumbents:

    A0  bare Karp patching tour (no local search)
    A   shipped incumbent = patching + Or-opt (the paper's Table 13 UB0)
    B   LKH-3 tour (RUNS=10, TIME_LIMIT=20 s, SEED=1)
    C   published optimum, used as an oracle lower envelope

on the 27 ATSPLIB instances, and under A/B on the 15 random-uniform reference
instances (n = 100, 200, 300, 500, 800; seeds 1..3; default_rng(1000*n+seed)).
It then reproduces the paper's table: certified / uncertified ranges, the
separation margin, AUC, rank correlations, bootstrap CIs, a label-permutation
test, and a held-out check (threshold fixed on the 19 distributed ATSPLIB
files, tested on the 8 reconstructed ftv prefixes).

The upper bounds are shipped in data/r4_incumbent/incumbents.json, so the whole
analysis is reproducible without LKH-3.  Pass --run-lkh to recompute arm B
(the LKH-3 binary must be on PATH); the other arms need only numpy.

Usage:
    python scripts/r4_incumbent_sensitivity.py
    python scripts/r4_incumbent_sensitivity.py --run-lkh
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
import atsplib                      # noqa: E402
import quasipoly_atsp as qa         # noqa: E402

DATA = os.path.join(_ROOT, "data", "r4_incumbent")
INSTANCES = os.path.join(_ROOT, "instances")
CERTIFIED = {"rbg323", "rbg358", "rbg403", "rbg443"}
HELD_OUT = {"ftv90", "ftv100", "ftv110", "ftv120", "ftv130", "ftv140", "ftv150", "ftv160"}
OPT = {
    "br17": 39, "ftv33": 1286, "ftv35": 1473, "ftv38": 1530, "p43": 5620,
    "ftv44": 1613, "ftv47": 1776, "ry48p": 14422, "ft53": 6905, "ftv55": 1608,
    "ftv64": 1839, "ft70": 38673, "ftv70": 1950, "kro124p": 36230,
    "ftv90": 1579, "ftv100": 1788, "ftv110": 1958, "ftv120": 2166,
    "ftv130": 2307, "ftv140": 2420, "ftv150": 2611, "ftv160": 2683,
    "ftv170": 2755, "rbg323": 1326, "rbg358": 1163, "rbg403": 2465,
    "rbg443": 2720,
}


def core(C):
    """Optimal assignment, optimal duals, reduced costs, non-basic mask."""
    n = C.shape[0]
    phi, uh, vh = qa.hungarian_with_duals(C)
    u, v = qa.optimal_duals(C, phi, hung0=(uh, vh))
    rc = np.clip(np.where(np.isfinite(C), C - u[:, None] - v[None, :], np.inf), 0.0, None)
    Z_AP = float(C[np.arange(n), phi].sum())
    tree_arcs, _ = qa.build_basis_tree(rc, phi)
    nonbasic = np.ones((n, n), dtype=bool)
    nonbasic[np.arange(n), phi] = False       # M*
    for i, j in tree_arcs:
        nonbasic[i, j] = False                # T* \ M*
    return Z_AP, rc, nonbasic, phi


def dens(Z_AP, rc, nonbasic, UB):
    d = float(UB) - Z_AP
    return 100.0 * int((nonbasic & (rc <= d)).sum()) / rc.size


def lkh_incumbent(C, runs=10, time_limit=20.0):
    """LKH-3 tour cost in the original units; integer matrices stay unscaled."""
    n = C.shape[0]
    fin = C[np.isfinite(C)]
    scale = 1 if np.allclose(fin, np.rint(fin)) else 1_000_000
    big = 10_000_000
    Ci = np.rint(np.where(np.isfinite(C), C * scale, big)).astype(np.int64)
    np.fill_diagonal(Ci, big)

    td = os.path.join(os.environ.get("LKH_TMP", os.path.join(_ROOT, "_work")), "r4")
    os.makedirs(td, exist_ok=True)
    tsp, par, out = (os.path.join(td, f) for f in ("r4.tsp", "r4.par", "r4.tour"))
    with open(tsp, "w") as f:
        f.write("NAME: r4\nTYPE: ATSP\nDIMENSION: %d\n" % n)
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        f.write("\n".join(" ".join(map(str, Ci[i])) for i in range(n)))
        f.write("\nEOF\n")

    def once(runs_, tl_):
        with open(par, "w") as f:
            f.write("PROBLEM_FILE = %s\nTOUR_FILE = %s\nRUNS = %d\n" % (tsp, out, runs_))
            f.write("TRACE_LEVEL = 0\nSEED = 1\n")
            if tl_ is not None:
                f.write("TIME_LIMIT = %f\n" % tl_)
        if os.path.exists(out):
            os.remove(out)
        t0 = time.perf_counter()
        try:
            subprocess.run(["LKH", par], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True,
                           timeout=None if tl_ is None else tl_ + 60)
        except subprocess.TimeoutExpired:
            pass
        if not os.path.exists(out):
            return None, time.perf_counter() - t0
        tour, reading = [], False
        for line in open(out):
            if line.startswith("TOUR_SECTION"):
                reading = True
                continue
            if reading:
                v = int(line.strip())
                if v == -1:
                    break
                tour.append(v - 1)
        if len(tour) != n or len(set(tour)) != n:
            return None, time.perf_counter() - t0
        arcs = [C[tour[k], tour[(k + 1) % n]] for k in range(n)]
        if not all(np.isfinite(a) for a in arcs):
            return None, time.perf_counter() - t0
        return float(sum(arcs)), time.perf_counter() - t0

    cost, el = once(runs, time_limit)
    if cost is None:
        cost, el2 = once(1, None)
        el += el2
    if cost is None:
        raise RuntimeError("LKH-3 produced no valid tour")
    return cost, el


def ranks(x):
    return np.argsort(np.argsort(np.asarray(x, dtype=float)))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def kendall(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    con = dis = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            con += s > 0
            dis += s < 0
    return float((con - dis) / (con + dis))


def auc(neg, pos):
    return sum(1.0 if p > x else 0.5 if p == x else 0.0
               for p in pos for x in neg) / (len(neg) * len(pos))


def quartiles(x):
    a = np.asarray(x, float)
    return dict(min=float(a.min()), q1=float(np.percentile(a, 25)),
                median=float(np.median(a)), q3=float(np.percentile(a, 75)),
                max=float(a.max()), mean=float(a.mean()))


def analyse(rows, arms=("A0", "A", "B", "C")):
    y = np.array([0 if r["certified"] else 1 for r in rows])
    rng = np.random.default_rng(20261115)
    res = {}
    base = np.array([r["A"] for r in rows])
    rb = ranks(base)
    for arm in arms:
        d = np.array([r[arm] for r in rows])
        c, u = d[y == 0], d[y == 1]
        ms, as_ = [], []
        for _ in range(10000):
            idx = rng.integers(0, len(rows), len(rows))
            cc, uu = d[idx][y[idx] == 0], d[idx][y[idx] == 1]
            if cc.size and uu.size:
                ms.append(uu.min() - cc.max())
                as_.append(auc(cc, uu))
        res[arm] = dict(
            certified=quartiles(c), uncertified=quartiles(u),
            margin=float(u.min() - c.max()), threshold=float((u.min() + c.max()) / 2),
            auc=float(auc(c, u)),
            spearman_vs_A=pearson(rb, ranks(d)),
            kendall_vs_A=kendall(base, d),
            rank_max_shift=int(np.abs(ranks(d) - rb).max()),
            mean_abs_change_pp=float(np.abs(d - base).mean()),
            margin_boot95=[float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
            auc_boot95=[float(np.percentile(as_, 2.5)), float(np.percentile(as_, 97.5))],
            margin_boot_positive_fraction=float(np.mean(np.array(ms) > 0)))
    obs = base[y == 1].min() - base[y == 0].max()
    hits = 0
    rng_p = np.random.default_rng(20261116)
    for _ in range(20000):
        yp = rng_p.permutation(y)
        if base[yp == 1].min() - base[yp == 0].max() >= obs:
            hits += 1
    res["permutation_p"] = (hits + 1) / 20001.0
    disc = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]
    for arm in arms:
        tau = (min(r[arm] for r in disc if not r["certified"]) +
               max(r[arm] for r in disc if r["certified"])) / 2
        res[arm]["held_out"] = dict(
            threshold=float(tau), n=len(held),
            correct=sum(r[arm] >= tau for r in held),
            min_held=min(r[arm] for r in held), max_held=max(r[arm] for r in held))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-lkh", action="store_true", help="recompute arm B with LKH-3")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "r4_analysis.json"))
    args = ap.parse_args()

    ubs = json.load(open(os.path.join(DATA, "incumbents.json")))
    stored = {r["instance"]: r for r in
              json.load(open(os.path.join(DATA, "atsplib_results.json")))}
    rows = []
    for name in OPT:
        C = atsplib.read_atsp(os.path.join(INSTANCES, name + ".atsp"))[1]
        Z, rc, nb, phi = core(C)
        rec = dict(ubs["atsplib"][name])
        if args.run_lkh:
            rec["B"], rec["lkh_seconds"] = lkh_incumbent(C)
        if abs(rec.get("C", OPT[name]) - OPT[name]) > 1e-9:
            raise ValueError("%s: oracle arm is not the published optimum" % name)
        rec["C"] = OPT[name]
        row = dict(instance=name, n=C.shape[0], certified=name in CERTIFIED,
                   held_out=name in HELD_OUT)
        for arm in ("A0", "A", "B", "C"):
            row[arm] = dens(Z, rc, nb, rec[arm])
            ref = stored[name][arm]["dens"]
            if abs(row[arm] - ref) > 1e-8 and not args.run_lkh:
                raise ValueError("%s arm %s: %.6f != stored %.6f" %
                                 (name, arm, row[arm], ref))
        rows.append(row)

    rnd = []
    for n in (100, 200, 300, 500, 800):
        for sd in (1, 2, 3):
            name = "rnd_n%d_s%d" % (n, sd)
            C = np.random.default_rng(1000 * n + sd).random((n, n))
            np.fill_diagonal(C, np.inf)
            Z, rc, nb, phi = core(C)
            rec = dict(ubs["random15"][name])
            if args.run_lkh:
                rec["B"], _ = lkh_incumbent(C)
            rnd.append(dict(instance=name, n=n,
                            A=dens(Z, rc, nb, rec["A"]), B=dens(Z, rc, nb, rec["B"])))

    res = analyse(rows)
    for arm in ("A", "B"):
        res.setdefault("random15", {})[arm] = quartiles([r[arm] for r in rnd])
    res["random15"]["spearman_A_B"] = pearson(ranks([r["A"] for r in rnd]),
                                              ranks([r["B"] for r in rnd]))
    res["random15"]["n300_s2"] = {a: next(r[a] for r in rnd
                                          if r["instance"] == "rnd_n300_s2")
                                  for a in ("A", "B")}
    res["n_instances"] = len(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    print("R4: |A~|/n^2 (%) on 27 ATSPLIB instances, four incumbents")
    print("%-4s %-14s %-16s %8s %6s %8s %10s" %
          ("arm", "certified", "uncertified", "margin", "AUC", "rho(A)", "held-out"))
    for arm in ("A0", "A", "B", "C"):
        a = res[arm]
        print("%-4s %5.1f--%-6.1f %5.1f--%-8.1f %8.1f %6.3f %8.3f %6d/%-2d" %
              (arm, a["certified"]["min"], a["certified"]["max"],
               a["uncertified"]["min"], a["uncertified"]["max"], a["margin"],
               a["auc"], a["spearman_vs_A"], a["held_out"]["correct"], a["held_out"]["n"]))
    print("permutation p = %.3g" % res["permutation_p"])
    print("random15 A median %.2f%% (%.2f--%.2f), B median %.2f%% (%.2f--%.2f)" %
          (res["random15"]["A"]["median"], res["random15"]["A"]["min"],
           res["random15"]["A"]["max"], res["random15"]["B"]["median"],
           res["random15"]["B"]["min"], res["random15"]["B"]["max"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
