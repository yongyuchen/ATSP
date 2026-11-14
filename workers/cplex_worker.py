"""CPLEX exact ATSP baselines (run with the CPLEX-supported Python, 3.9/3.10).

Two formulations, matching the SCIP baselines in scip_dfj.py:
  * dfj : assignment + DFJ subtour cuts via a lazy-constraint callback
          (branch-and-cut; exact).
  * mtz : assignment + Miller-Tucker-Zemlin ordering variables (static ILP).

Usage:  python cplex_worker.py INSTANCE.json RESULT.json [time_limit] [max_n_mtz] [MIPSTART.json]

An optional MIP start (a tour as a list of node indices) lets CPLEX start from
the same LKH incumbent that our solver uses; this is the fair "warm start"
comparison.

The instance JSON is a list-of-lists cost matrix; diagonal entries are ignored.
The result JSON contains one dict with keys dfj_* / mtz_*.
"""
import json
import sys
import time

import cplex
from cplex.callbacks import LazyConstraintCallback

# Keep a runaway model from taking the machine down (OOM safety).
# `resource` is unavailable on Windows; ignore it there.
try:
    import resource
    resource.setrlimit(resource.RLIMIT_DATA, (8 << 30, 8 << 30))
except Exception:                     # noqa: BLE001
    pass

EPS = 1e-6


class SubtourLazy(LazyConstraintCallback):
    """Reject any integer assignment that decomposes into >1 subtour.

    SEC:  x(E(S)) <= |S| - 1  for every proper nonempty S.
    We separate connected components of the support digraph of the candidate
    integer solution; each component (and its complement) violates an SEC.
    """

    def __init__(self, env):
        super().__init__(env)

    def __call__(self):
        n = self.n
        arcs = self.arcs
        vals = self.get_values(self.xidx)
        rowsum = [0.0] * n
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ncomp = n
        for k, (i, j) in enumerate(arcs):
            v = vals[k]
            rowsum[i] += v
            if v > 0.5:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
                    ncomp -= 1
        if ncomp == 1:
            return
        # Conservative guard: only cut genuine assignment points.
        for rs in rowsum:
            if abs(rs - 1.0) > 1e-4:
                return
        groups = {}
        for v in range(n):
            groups.setdefault(find(v), []).append(v)
        allv = set(range(n))
        for comp in groups.values():
            if len(comp) >= n:
                continue
            if 2 * len(comp) <= n:
                S = comp
            else:
                cs = set(comp)
                S = [v for v in allv - cs]
            S = sorted(S)
            if len(S) < 2:
                continue
            ind = [self.var[(i, j)] for i in S for j in S if i != j]
            if not ind:
                continue
            self.add(constraint=cplex.SparsePair(ind=ind, val=[1.0] * len(ind)),
                     sense="L", rhs=float(len(S) - 1))
            self.ncuts += 1


def _new_model(time_limit, threads=1):
    m = cplex.Cplex()
    m.set_log_stream(None)
    m.set_error_stream(None)
    m.set_warning_stream(None)
    m.set_results_stream(None)
    m.parameters.threads.set(threads)
    m.parameters.timelimit.set(time_limit)
    m.parameters.mip.display.set(0)
    m.objective.set_sense(m.objective.sense.minimize)
    return m


def _assignment(m, C):
    n = len(C)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    xidx = list(m.variables.add(obj=[float(C[i][j]) for (i, j) in arcs],
                                lb=[0.0] * len(arcs),
                                ub=[1.0] * len(arcs),
                                types=["B"] * len(arcs)))
    var = {(i, j): xidx[k] for k, (i, j) in enumerate(arcs)}
    rows = []
    for i in range(n):
        rows.append(cplex.SparsePair(ind=[var[(i, j)] for j in range(n) if j != i],
                                    val=[1.0] * (n - 1)))
    for j in range(n):
        rows.append(cplex.SparsePair(ind=[var[(i, j)] for i in range(n) if i != j],
                                    val=[1.0] * (n - 1)))
    m.linear_constraints.add(lin_expr=rows, senses=["E"] * (2 * n),
                             rhs=[1.0] * (2 * n))
    return arcs, xidx, var


def _finish(m, t0):
    st = m.solution.get_status()
    proved = st in (101, 102)          # MIP optimal / optimal with tolerance
    try:
        cost = float(m.solution.get_objective_value())
    except Exception:
        cost = float("nan")
    try:
        nodes = int(m.solution.progress.get_num_nodes_processed())
    except Exception:
        nodes = -1
    try:
        gap = float(m.solution.MIP.get_mip_relative_gap())
    except Exception:
        gap = float("nan")
    return {"cost": cost, "proved": bool(proved), "status": int(st),
            "status_str": m.solution.get_status_string(), "t": time.time() - t0,
            "nodes": nodes, "gap": gap}


def cplex_dfj(C, time_limit=90.0, tour=None):
    n = len(C)
    t0 = time.time()
    m = _new_model(time_limit)
    arcs, xidx, var = _assignment(m, C)
    cb = m.register_callback(SubtourLazy)
    cb.n, cb.arcs, cb.xidx, cb.var = n, arcs, xidx, var
    cb.ncuts = 0
    if tour is not None:
        ind = [var[(tour[k], tour[(k + 1) % n])] for k in range(n)]
        m.MIP_starts.add(cplex.SparsePair(ind=ind, val=[1.0] * n),
                         m.MIP_starts.effort_level.check_feasibility, "lkh")
    m.solve()
    out = _finish(m, t0)
    out["cuts"] = cb.ncuts
    del cb, m
    return out


def cplex_mtz(C, time_limit=90.0):
    n = len(C)
    t0 = time.time()
    m = _new_model(time_limit)
    arcs, xidx, var = _assignment(m, C)
    uidx = list(m.variables.add(
        obj=[0.0] * n,
        lb=[0.0] + [1.0] * (n - 1),
        ub=[0.0] + [float(n - 1)] * (n - 1),
        types=["C"] * n))
    rows = []
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                rows.append(cplex.SparsePair(
                    ind=[uidx[i], uidx[j], var[(i, j)]],
                    val=[1.0, -1.0, float(n)]))
    m.linear_constraints.add(lin_expr=rows,
                             senses=["L"] * len(rows),
                             rhs=[float(n - 1)] * len(rows))
    m.solve()
    out = _finish(m, t0)
    out["cuts"] = 0
    del m
    return out


def main():
    inst_path, res_path = sys.argv[1], sys.argv[2]
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    max_n_mtz = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    tour = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    C = json.load(open(inst_path))
    out = {"dfj": cplex_dfj(C, tl, tour)}
    if len(C) <= max_n_mtz:
        out["mtz"] = cplex_mtz(C, tl)
    else:
        out["mtz"] = None
    json.dump(out, open(res_path, "w"), indent=2)


if __name__ == "__main__":
    main()
