"""Gurobi 12.0.3 exact ATSP baselines (run with Python 3.10).

Two formulations, matching cplex_worker.py / scip_dfj.py:
  * dfj : assignment + DFJ subtour cuts via a MIP-solution callback
          (branch-and-cut; exact).
  * mtz : assignment + Miller-Tucker-Zemlin ordering variables (static ILP).

Usage:  python gurobi_worker.py INSTANCE.json RESULT.json [time_limit] [max_n_mtz] [MIPSTART.json]

The instance JSON is a list-of-lists cost matrix; diagonal ignored.
The result JSON contains one dict with keys dfj_* / mtz_*.

A Gurobi licence is required to solve (academic WLS or a gurobi.lic file).
Without one the model construction succeeds but optimize() raises GurobiError,
which we report as a non-fatal error row so the driver can continue.
"""
import json
import sys
import time

import gurobipy as grb

# Configurable OOM guard.  Defaults to the 8 GB data-segment limit used for
# the runs reported in the paper; set QA_WORKER_RLIMIT_GB to another value,
# or to 0 to disable it (e.g. on a machine with more RAM).
try:
    import os
    import resource
    _gb = float(os.environ.get("QA_WORKER_RLIMIT_GB", "8"))
    if _gb > 0:
        _lim = int(_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_DATA, (_lim, _lim))
except Exception:                     # noqa: BLE001  (Windows / restricted)
    pass


def _components(n, x):
    """Weakly-connected components of the support digraph (x_ij > 0.5).

    Returns [] unless every row sums to 1 (genuine assignment point); this
    mirrors the conservative guard used by the CPLEX/SCIP baselines so that
    pseudo/partial solutions never trigger a cut.
    """
    parent = list(range(n))
    rowsum = [0.0] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ncomp = n
    for (i, j), v in x.items():
        rowsum[i] += v
        if v > 0.5:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
                ncomp -= 1
    if ncomp == 1:
        return []
    for rs in rowsum:
        if abs(rs - 1.0) > 1e-4:
            return []
    groups = {}
    for v in range(n):
        groups.setdefault(find(v), []).append(v)
    out = []
    allv = set(range(n))
    for comp in groups.values():
        if len(comp) >= n:
            continue
        if 2 * len(comp) <= n:
            S = comp
        else:
            cs = set(comp)
            S = [v for v in allv - cs]
        out.append(S)
    return out


def gurobi_dfj(C, time_limit=90.0, tour=None, threads=1):
    n = len(C)
    t0 = time.time()
    model = grb.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", threads)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.0)
    model.setParam("LazyConstraints", 1)   # enable lazy cuts from the callback
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    var = {}
    for (i, j) in arcs:
        var[(i, j)] = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY,
                                   name=f"x_{i}_{j}")
    for i in range(n):
        model.addConstr(grb.quicksum(var[(i, j)] for j in range(n) if j != i) == 1)
        model.addConstr(grb.quicksum(var[(j, i)] for j in range(n) if j != i) == 1)
    model.setObjective(grb.quicksum(C[i][j] * var[(i, j)] for (i, j) in arcs),
                       grb.GRB.MINIMIZE)
    if tour is not None:
        for k in range(n):
            var[(tour[k], tour[(k + 1) % n])].start = 1.0
    var_list = [var[e] for e in arcs]

    def subtour_cb(model, where):
        if where == grb.GRB.Callback.MIPSOL:
            sol = model.cbGetSolution(var_list)
            x = {arcs[k]: sol[k] for k in range(len(arcs))}
            for S in _components(n, x):
                if len(S) < 2:
                    continue
                model.cbLazy(grb.quicksum(var[(i, j)]
                                         for i in S for j in S if i != j)
                             <= len(S) - 1)

    try:
        model.optimize(subtour_cb)
    except grb.GurobiError as e:
        return {"cost": float("nan"), "proved": False, "status": -1,
                "status_str": f"GurobiError:{e}", "t": time.time() - t0,
                "nodes": -1, "gap": float("nan"), "cuts": 0}
    st = model.Status
    proved = (st == grb.GRB.OPTIMAL)
    try:
        cost = float(model.ObjVal)
    except Exception:
        cost = float("nan")
    try:
        nodes = int(model.NodeCount)
    except Exception:
        nodes = -1
    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = float("nan")
    model.dispose()
    return {"cost": cost, "proved": bool(proved), "status": int(st),
            "status_str": str(st), "t": time.time() - t0, "nodes": nodes,
            "gap": gap, "cuts": 0}


def gurobi_mtz(C, time_limit=90.0):
    n = len(C)
    t0 = time.time()
    model = grb.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.0)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    var = {}
    for (i, j) in arcs:
        var[(i, j)] = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY,
                                   name=f"x_{i}_{j}")
    for i in range(n):
        model.addConstr(grb.quicksum(var[(i, j)] for j in range(n) if j != i) == 1)
        model.addConstr(grb.quicksum(var[(j, i)] for j in range(n) if j != i) == 1)
    u = {}
    for i in range(n):
        lb = 0.0 if i == 0 else 1.0
        ub = 0.0 if i == 0 else float(n - 1)
        u[i] = model.addVar(lb=lb, ub=ub, vtype=grb.GRB.CONTINUOUS,
                            name=f"u_{i}")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * var[(i, j)] <= n - 1)
    model.setObjective(grb.quicksum(C[i][j] * var[(i, j)] for (i, j) in arcs),
                       grb.GRB.MINIMIZE)
    try:
        model.optimize()
    except grb.GurobiError as e:
        return {"cost": float("nan"), "proved": False, "status": -1,
                "status_str": f"GurobiError:{e}", "t": time.time() - t0,
                "nodes": -1, "gap": float("nan"), "cuts": 0}
    st = model.Status
    proved = (st == grb.GRB.OPTIMAL)
    try:
        cost = float(model.ObjVal)
    except Exception:
        cost = float("nan")
    try:
        nodes = int(model.NodeCount)
    except Exception:
        nodes = -1
    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = float("nan")
    model.dispose()
    return {"cost": cost, "proved": bool(proved), "status": int(st),
            "status_str": str(st), "t": time.time() - t0, "nodes": nodes,
            "gap": gap, "cuts": 0}


def main():
    inst_path, res_path = sys.argv[1], sys.argv[2]
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    max_n_mtz = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    tour = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    C = json.load(open(inst_path))
    out = {"dfj": gurobi_dfj(C, tl, tour)}
    if len(C) <= max_n_mtz:
        out["mtz"] = gurobi_mtz(C, tl)
    else:
        out["mtz"] = None
    json.dump(out, open(res_path, "w"), indent=2)


if __name__ == "__main__":
    main()
