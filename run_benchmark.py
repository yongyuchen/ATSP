"""Unified four-way ATSP benchmark: ours / CPLEX / SCIP / Gurobi.

Cross-platform rewrite of the paper's four-engine drivers.  Design points:

  * no hard-coded absolute paths -- every path is derived from this
    file's location;
  * our solver runs in-process with whatever interpreter launched this script
    (pure Python; Cython is optional and NOT required -- see docs/IMPLEMENTATION_NOTES.md);
  * each external solver is a separate subprocess, and a solver whose import or
    licence fails is recorded as "unavailable" instead of aborting the run;
  * a per-solver interpreter can be given explicitly, defaulting to the one
    running this script.

Usage (from the package root):
    python run_benchmark.py --group random   --tl 60
    python run_benchmark.py --group atsplib  --tl 60
    python run_benchmark.py --group random   --tl 60 --solver ours,scip
    python run_benchmark.py --group atsplib  --tl 60 --out results/win.json

Results are appended to the output JSON after EVERY instance, so an interrupted
run can simply be restarted: instances already present are skipped.

Protocol (identical to the paper):
  * one thread, wall-clock limit TL, gap 0
  * DFJ = assignment + subtour elimination via lazy cuts (branch-and-cut)
  * MTZ = assignment + Miller-Tucker-Zemlin (only for n <= --mtz-max)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = os.path.join(HERE, "workers")
SOLVER = os.path.join(HERE, "solver")
DATA = os.path.join(HERE, "data")
INST = os.path.join(HERE, "instances")
WORK = os.path.join(HERE, "_work")

sys.path.insert(0, SOLVER)
import quasipoly_atsp as qa          # noqa: E402
import atsplib                       # noqa: E402

PLAN_SMALL = [(20, (1, 2, 3)), (30, (1, 2, 3)), (50, (1, 2, 3)),
              (80, (1, 2, 3)), (100, (1, 2, 3)), (150, (1, 2)),
              (200, (1, 2)), (300, (1, 2))]
PLAN_LARGE = [(500, (1,)), (800, (1,))]
# the 19 ATSPLIB instances used in the paper's tab:atsplib (random100 excluded)
ATSPLIB_NAMES = None      # filled from data/expected_atsplib.json when present


def random_inst(n, seed):
    """U[0,1] i.i.d. costs, +inf diagonal. Same generator as the paper."""
    g = np.random.default_rng(seed * 7919 + n)
    D = g.random((n, n))
    np.fill_diagonal(D, np.inf)
    return D


def expand(plan):
    return [(n, s) for n, seeds in plan for s in seeds]


def load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def fin(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


def gap_pct(cost, ref):
    if not fin(cost) or not ref:
        return None
    return 100.0 * (cost - ref) / ref


# --------------------------------------------------------------- subprocesses
def run_worker(python, script, args, tl, extra_env=None):
    """Run one worker; return (parsed_json | None, note)."""
    path = os.path.join(WORKERS, script)
    if not os.path.exists(path):
        return None, "worker missing"
    res_path = args[1]
    cmd = [python, path] + [str(a) for a in args]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # keep scratch off tiny /tmp filesystems
    env.setdefault("TMPDIR", WORK)
    env.setdefault("TEMP", WORK)
    env.setdefault("TMP", WORK)
    log_path = res_path + ".log"
    if os.path.exists(res_path):
        os.remove(res_path)
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            subprocess.run(cmd, env=env, timeout=tl + 150, check=False,
                           stdout=log, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        return None, "interpreter not found: %s" % python
    except Exception as e:                              # noqa: BLE001
        return None, "error: %s" % e
    if not os.path.exists(res_path):
        tail = ""
        try:
            tail = open(log_path, encoding="utf-8",
                        errors="replace").read()[-400:]
        except Exception:
            pass
        return None, "no result; log tail: " + tail.replace("\n", " ")[-200:]
    try:
        return json.load(open(res_path)), "ok"
    except Exception as e:                              # noqa: BLE001
        return None, "bad json: %s" % e


def arm(res, form, opt):
    """Normalise one worker arm into the flat row format."""
    if not res or not res.get(form):
        return {"cost": None, "proved": False, "t": None, "gap_pct": None,
                "nodes": None, "note": "missing"}
    d = res[form]
    c = d.get("cost")
    if not fin(c):
        c = None
    return {"cost": c, "proved": bool(d.get("proved", False)),
            "t": d.get("t"), "gap_pct": gap_pct(c, opt),
            "nodes": d.get("nodes"), "note": str(d.get("status_str", ""))}


def discover_cplex_env():
    """Find the CPLEX Python API and shared library, if installed locally.

    The CPLEX Python API is normally used via PYTHONPATH pointing at
    <studio>/cplex/python/<ver>/<platform>, and the .so/.dll via
    LD_LIBRARY_PATH / PATH pointing at <studio>/cplex/bin/<platform>.
    We locate a CPLEX Studio installation and export both so the worker just
    works without the user pre-configuring their shell.
    """
    env = {}
    roots = []
    # explicit override wins
    for k in ("CPLEX_STUDIO_DIR", "CPLEX_STUDIO_DIR221", "CPLEX_HOME"):
        if os.environ.get(k):
            roots.append(os.environ[k])
    candidates = [
        os.environ.get("CPLEX_STUDIO_DIR", ""),
        r"C:\Program Files\IBM\ILOG\CPLEX_Studio221",
        r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211",
        r"C:\Program Files\IBM\ILOG\CPLEX_Studio222",
        r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211",
        "/opt/ibm/ILOG/CPLEX_Studio221",
        "/opt/ibm/ILOG/CPLEX_Studio2211",
        os.path.expanduser("~/ILOG/CPLEX_Studio221"),
        os.path.expanduser("~/ILOG/CPLEX_Studio2211"),
    ]
    roots += [c for c in candidates if c and os.path.isdir(c)]
    if not roots:
        return env
    for root in roots:
        pydir = os.path.join(root, "cplex", "python")
        if not os.path.isdir(pydir):
            continue
        # highest supported version folder that actually contains the package
        found = None
        for ver in ("3.10", "3.11", "3.9", "3.12", "3.8"):
            for plat in ("x86-64_linux", "x64_win64", "x86-64_osx", "arm64_osx"):
                p = os.path.join(pydir, ver, plat)
                if os.path.isdir(os.path.join(p, "cplex")):
                    found = p
                    break
            if found:
                break
        if not found:
            continue
        env["PYTHONPATH"] = found + (
            os.pathsep + os.environ["PYTHONPATH"]
            if os.environ.get("PYTHONPATH") else "")
        for plat in ("x86-64_linux", "x64_win64", "x86-64_osx", "arm64_osx"):
            b = os.path.join(root, "cplex", "bin", plat)
            if os.path.isdir(b):
                env["LD_LIBRARY_PATH"] = b + (
                    os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                    if os.environ.get("LD_LIBRARY_PATH") else "")
                env["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                env["CPLEX_BIN"] = b
                break
        env["CPLEX_STUDIO_DIR"] = root
        break
    return env


# ------------------------------------------------------------------- one row
def solve_row(name, key, C, OPT, tl, solvers, cfg):
    """Run every requested arm on one instance. `key` identifies the row."""
    n = C.shape[0]
    row = {"instance": name, "key": key, "n": n, "OPT": OPT}
    try:
        r, c = linear_sum_assignment(C)
        row["Z_AP"] = float(C[r, c].sum())
    except Exception:
        row["Z_AP"] = None

    os.makedirs(WORK, exist_ok=True)

    # --- ours: in-process, exact same protocol as the paper
    if "ours" in solvers:
        try:
            res = qa.solve(C, time_limit=tl)
            cost = float(res.cost) if res.cost is not None else None
            row["ours"] = {
                "cost": cost, "proved": bool(res.proved),
                "t": float(res.time_sec), "gap_pct": gap_pct(cost, OPT),
                "Z_AP": float(res.Z_AP), "n_cycles": int(res.n_cycles),
                "heuristic_cost": float(res.heuristic_cost),
                "lb_cert": (float(res.lb_cert)
                            if res.lb_cert is not None else None),
                "rung_ok": int(res.rung_ok),
                "search_nodes": int(res.search_nodes),
                "note": ""}
        except Exception as e:                          # noqa: BLE001
            row["ours"] = {"cost": None, "proved": False, "t": None,
                           "gap_pct": None, "note": "error: %s" % e}

    # --- instance file for the external workers (diagonal 0; workers skip it)
    D = C.copy()
    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        row["warning"] = "non-finite off-diagonal entry; external arms skipped"
        return row
    ipath = os.path.join(WORK, "inst_%s.json" % key)
    json.dump(D.tolist(), open(ipath, "w"))
    res_dir = os.path.join(WORK, "res")
    os.makedirs(res_dir, exist_ok=True)

    # --- CPLEX
    if "cplex" in solvers and cfg["cplex_py"]:
        p = os.path.join(res_dir, "c_%s.json" % key)
        r, note = run_worker(cfg["cplex_py"], "cplex_worker.py",
                             [ipath, p, tl, cfg["mtz_max"]], tl,
                             cfg.get("cplex_env"))
        row["cplex_dfj"] = arm(r, "dfj", OPT)
        row["cplex_mtz"] = arm(r, "mtz", OPT)
        row["cplex_note"] = note

    # --- SCIP
    if "scip" in solvers and cfg["scip_py"]:
        p = os.path.join(res_dir, "s_%s.json" % key)
        r, note = run_worker(cfg["scip_py"], "scip_worker.py",
                             [ipath, p, tl, cfg["mtz_max"]], tl)
        row["scip_dfj"] = arm(r, "dfj", OPT)
        row["scip_mtz"] = arm(r, "mtz", OPT)
        row["scip_note"] = note

    # --- Gurobi
    if "gurobi" in solvers and cfg["gurobi_py"]:
        p = os.path.join(res_dir, "g_%s.json" % key)
        r, note = run_worker(cfg["gurobi_py"], "gurobi_worker.py",
                             [ipath, p, tl, cfg["mtz_max"]], tl,
                             cfg.get("gurobi_env"))
        row["gurobi_dfj"] = arm(r, "dfj", OPT)
        row["gurobi_mtz"] = arm(r, "mtz", OPT)
        row["gurobi_note"] = note

    # --- cross-check: every arm that proved optimality must agree on the cost
    bad, costs = [], []
    for a in ("ours", "cplex_dfj", "cplex_mtz", "scip_dfj", "scip_mtz",
              "gurobi_dfj", "gurobi_mtz"):
        d = row.get(a)
        if d and d.get("proved") and fin(d.get("cost")):
            costs.append(d["cost"])
            if OPT and abs(d["cost"] - OPT) > 1e-6:
                bad.append(a)
    if costs:
        ref = min(costs)
        bad += [a for a in
                ("ours", "cplex_dfj", "cplex_mtz", "scip_dfj", "scip_mtz",
                 "gurobi_dfj", "gurobi_mtz")
                if (row.get(a) or {}).get("proved")
                and fin((row.get(a) or {}).get("cost"))
                and abs(row[a]["cost"] - ref) > 1e-6 and a not in bad]
    row["match"] = not bad
    row["mismatch"] = sorted(set(bad))
    return row


# ------------------------------------------------------------------- drivers
def atsplib_names():
    global ATSPLIB_NAMES
    if ATSPLIB_NAMES is None:
        exp = load_json(os.path.join(DATA, "expected_atsplib.json"), None)
        if exp:
            ATSPLIB_NAMES = [e["instance"] for e in exp]
        else:
            ATSPLIB_NAMES = sorted(
                f[:-5] for f in os.listdir(INST) if f.endswith(".atsp")
                and f[:-5] != "random100")
    return ATSPLIB_NAMES


def short(x):
    return "%8.2f" % x if fin(x) else "     n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="random",
                    choices=["random", "atsplib", "all"])
    ap.add_argument("--tl", type=float, default=60.0, help="wall-clock limit (s)")
    ap.add_argument("--out", default=None, help="result JSON path")
    ap.add_argument("--solver", default="ours,cplex,scip,gurobi",
                    help="comma list: ours,cplex,scip,gurobi")
    ap.add_argument("--mtz-max", type=int, default=300,
                    help="run MTZ only when n <= this")
    ap.add_argument("--ours-tl", type=float, default=None,
                    help="time limit for our solver (default: --tl)")
    ap.add_argument("--cplex-py", default=None, help="python for CPLEX worker")
    ap.add_argument("--scip-py", default=None, help="python for SCIP worker")
    ap.add_argument("--gurobi-py", default=None, help="python for Gurobi worker")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    solvers = [s.strip().lower() for s in args.solver.split(",") if s.strip()]
    out = args.out or os.path.join(HERE, "results", "compare_%s.json" % args.group)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    cfg = {
        "cplex_py": args.cplex_py or sys.executable,
        "scip_py": args.scip_py or sys.executable,
        "gurobi_py": args.gurobi_py or sys.executable,
        "mtz_max": args.mtz_max,
        "cplex_env": discover_cplex_env(),
        "gurobi_env": {k: v for k, v in os.environ.items()
                       if k.startswith("GRB_")},
    }
    if cfg["cplex_env"]:
        print("  CPLEX   : %s" % cfg["cplex_env"].get("CPLEX_STUDIO_DIR", "?"))
    else:
        print("  CPLEX   : not auto-detected (set CPLEX_STUDIO_DIR or "
              "pass --cplex-py)")

    # warn when an external solver falls back to the interpreter running us but
    # that interpreter may not have the package -- this is the #1 source of
    # "solver shows n/a" confusion. Skip the warning if the user pinned an
    # explicit interpreter for that solver.
    mods = {"cplex": ("cplex", "cplex_py"),
            "scip": ("pyscipopt", "scip_py"),
            "gurobi": ("gurobipy", "gurobi_py")}
    explicit = {"cplex": bool(args.cplex_py), "scip": bool(args.scip_py),
                "gurobi": bool(args.gurobi_py)}
    for s in solvers:
        if s not in mods or explicit[s]:
            continue
        mod, _ = mods[s]
        try:
            __import__(mod)
        except Exception:
            print("  WARNING : '%s' is not importable by %s -- that solver will "
                  "report n/a.\n            Use --%s-py <python that has %s>."
                  % (s, sys.executable, s, mod))
    # our solver may need its own limit (the paper gives it more headroom)
    if args.ours_tl is not None:
        qa_tl = args.ours_tl
    else:
        qa_tl = args.tl

    rows = load_json(out, [])
    done = {r.get("key") for r in rows}

    # build the work list
    jobs = []
    if args.group in ("random", "all"):
        expected = {(e["n"], e["seed"]): e for e in
                    load_json(os.path.join(DATA, "expected_random.json"), [])}
        for n, seed in expand(PLAN_SMALL) + expand(PLAN_LARGE):
            jobs.append(("r%d_%d" % (n, seed), n, seed, expected))
    n_before = len(jobs)
    if args.group in ("atsplib", "all"):
        opt_tab = atsplib.known_optima()
        for name in atsplib_names():
            jobs.append(("a_" + name, None, name, opt_tab))
    jobs = jobs[args.start: (args.end or len(jobs))]

    print("four-way ATSP benchmark")
    print("  package : %s" % HERE)
    print("  python  : %s" % sys.executable)
    print("  solvers : %s" % ",".join(solvers))
    print("  Cython  : %s" % ("available" if qa.cy_backend() else
                              "not used (pure Python)"))
    print("  TL      : %.0fs   ours TL: %.0fs   rows already done: %d"
          % (args.tl, qa_tl, len(done)))
    print()

    hdr = ("%-9s %4s | %-10s | %-24s | %-24s | %-24s | ok")
    print(hdr % ("inst", "n", "OPT", "ours t/pr", "CPLEX-DFJ t/pr",
                 "SCIP-DFJ t/pr"), flush=True)
    print("-" * 122, flush=True)

    for key, n, tag, table in jobs:
        if key in done:
            print("%-9s (done, skip)" % key, flush=True)
            continue
        if n is None:                       # ATSPLIB instance
            name = tag
            nm, C = atsplib.read_atsp(os.path.join(INST, name + ".atsp"))
            C = np.asarray(C, dtype=float)
            OPT = table.get(nm, table.get(nm.rstrip("p")))
            label = name
        else:
            name = "n%d_s%d" % (n, tag)
            C = random_inst(n, tag)
            e = table.get((n, tag))
            if e and fin(e.get("OPT")):
                OPT = e["OPT"]
            else:
                rr = qa.solve(C, time_limit=qa_tl)
                OPT = float(rr.cost)
            label = "%d/%d" % (n, tag)

        row = solve_row(name, key, C, OPT, qa_tl, solvers, cfg)
        rows.append(row)
        json.dump(rows, open(out, "w"), indent=1)

        o = row.get("ours") or {}
        c = row.get("cplex_dfj") or {}
        s = row.get("scip_dfj") or {}
        g = row.get("gurobi_dfj") or {}

        def pr(d):
            return ("Y" if d.get("proved") else "n")
        print("%-9s %4d | %-10s | %s %s | %s %s | %s %s | %s"
              % (label, row["n"],
                 ("%.6f" % OPT) if fin(OPT) else "n/a",
                 short(o.get("t")), pr(o),
                 short(c.get("t")), pr(c),
                 short(s.get("t")), pr(s),
                 "OK" if row["match"] else ("BAD:" + ",".join(row["mismatch"]))),
              flush=True)
        if g:
            print("%-9s      |            | gurobi-DFJ %s %s"
                  % ("", short(g.get("t")), pr(g)), flush=True)

    print("-" * 122, flush=True)
    print("rows: %d   written to %s" % (len(rows), out))
    report(rows)


def report(rows):
    """Median speedups over the arms that proved optimality."""
    import statistics as st
    arms = [("cplex_dfj", "CPLEX-DFJ"), ("scip_dfj", "SCIP-DFJ"),
            ("gurobi_dfj", "Gurobi-DFJ")]
    ours = [r for r in rows if (r.get("ours") or {}).get("proved")
            and fin(r["ours"].get("t")) and r["ours"]["t"] > 0]
    print("\nspeedups (median over instances where BOTH proved):")
    for k, lab in arms:
        v = [r[k]["t"] / r["ours"]["t"] for r in ours
             if (r.get(k) or {}).get("proved") and fin(r[k].get("t"))]
        if v:
            print("  ours vs %-11s n=%2d  median %6.1fx  range %.1f..%.1f"
                  % (lab, len(v), st.median(v), min(v), max(v)))
        else:
            print("  ours vs %-11s no paired results" % lab)
    bad = [r["key"] for r in rows if not r.get("match", True)]
    print("\ncost agreement: %d/%d rows consistent"
          % (len(rows) - len(bad), len(rows)))
    if bad:
        print("  MISMATCH rows: %s" % ", ".join(bad))


if __name__ == "__main__":
    main()
