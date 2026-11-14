"""Classical AP-based branch & bound for the ATSP (Carpaneto--Toth, 1980).

Why this file exists
--------------------
The paper's central empirical claim is that the Bell--Frieze pipeline is fast.
But "fast" has two candidate explanations:

  A. the assignment relaxation is extremely tight on i.i.d. random ATSP (gap
     <= 0.45% for n >= 150, k = 4--7 subtours), so the residual search is
     trivial for ANY method built on the AP bound;
  B. the Bell--Frieze machinery (basis spanning tree + budget-pruned alternating
     cycle DFS) is what makes the search small.

Nothing in the paper separates them, because every baseline it compares against
(SCIP-MTZ, SCIP-lazy-DFJ, Concorde, LKH-3) rests on a *different* relaxation.
The right control is a classical AP-based branch and bound -- the Miller--Pekny
(1991) / Carpaneto--Dell'Amico--Toth (1995) line -- which uses exactly the same
relaxation and the same reduced-cost sparsification, and none of the
Bell--Frieze structure.

What is implemented
-------------------
Node    = one AP solve on a cost matrix with a set of forbidden arcs.
Bound   = the AP value (a valid lower bound for the restricted problem).
Branch  = Carpaneto--Toth rule: take the shortest subtour of the current AP
          solution that does NOT contain the root vertex, and create one child
          per arc of that subtour, each forbidding that arc.
Search  = depth first.

Three variants, differing only in reduced-cost fixing:
  fix="none"  no fixing at all
  fix="root"  fix at the root only -- this is the Miller--Pekny / Carpaneto
              sparsification as described in the literature
  fix="node"  fix at every node (stronger than the classical version)

The fixing rule is the one the literature describes and the one the paper calls
"budget pruning": with duals (u,v) and LB = sum(u)+sum(v), any tour using arc
(i,j) costs at least LB + (c_ij - u_i - v_j), so if that exceeds UB the arc can
be deleted. This is why the comparison matters -- if the rule is the same in
both solvers, whatever difference remains is due to the *structure*, not to the
pruning idea.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))
import quasipoly_atsp as qa


# --------------------------------------------------------------------- helpers

def duals_from_matching(C, phi, max_rounds=None):
    """Optimal duals (u, v) for the assignment, from an optimal matching.

    Given the optimal matching phi, the constraints u_i + v_j <= C_ij with
    equality on matching arcs reduce to a system of difference constraints on
    the columns,

        v_j - v_k <= D[k, j] := C[phi^{-1}(k), j] - C[phi^{-1}(k), k],

    which is feasible exactly when phi is optimal (a negative cycle would be an
    improving alternating cycle). One Bellman--Ford relaxation from a virtual
    source joined to every column by a zero-weight arc gives a feasible v; then
    u_i = C[i, phi(i)] - v_{phi(i)}. Costs O(n^2) per round and converges in a
    handful of rounds, which is what makes fixing cheap at every node.
    """
    n = C.shape[0]
    inv = np.empty(n, dtype=np.int64)
    inv[phi] = np.arange(n)
    Ci = C[inv]                                   # Ci[k, j] = C[inv[k], j]
    D = Ci - Ci[np.arange(n), np.arange(n)][:, None]
    v = np.zeros(n)
    rounds = max_rounds or n
    for _ in range(rounds):
        # (D + v[:, None])[k, j] = D[k, j] + v[k]  -- the row index is the
        # "from" column k; broadcasting v[None, :] here would add v[j] instead.
        nv = np.minimum(v, (D + v[:, None]).min(axis=0))
        if np.array_equal(nv, v):
            break
        v = nv
    u = C[np.arange(n), phi] - v[phi]
    return u, v


def subtours_of(phi):
    """Cycles of the permutation phi (phi[i] = column assigned to row i)."""
    n = len(phi)
    seen = np.zeros(n, dtype=bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        cyc, x = [], s
        while not seen[x]:
            seen[x] = True
            cyc.append(int(x))
            x = int(phi[x])
        out.append(cyc)
    return out


class CTResult:
    __slots__ = ("cost", "tour", "nodes", "fixed_arcs", "time_sec",
                 "proved", "Z_AP_root", "capped")

    def __init__(self, cost, tour, nodes, fixed_arcs, time_sec, proved,
                 Z_AP_root, capped):
        self.cost = cost
        self.tour = tour
        self.nodes = nodes
        self.fixed_arcs = fixed_arcs
        self.time_sec = time_sec
        self.proved = proved
        self.Z_AP_root = Z_AP_root
        self.capped = capped


# -------------------------------------------------------------------- the B&B

def ct_solve(C0, UB0, fix="none", node_cap=2_000_000, time_limit=None,
             root=0, eps=1e-9, patch_every=0):
    """Carpaneto--Toth branch and bound.

    C0   : cost matrix, C0[i,i] = +inf (or any non-finite entry) forbidden
    UB0  : initial upper bound -- pass the SAME value the Bell--Frieze solver
           used, otherwise the comparison measures the incumbent, not the method
    fix  : "none" | "root" | "node"
    patch_every : run Karp patching + Or-opt at every k-th node to improve the
           incumbent (0 = never). Miller--Pekny apply patching "at the root node
           and to the other nodes with decreasing frequency as the search
           progresses", so a fair classical baseline must offer it; without it
           the B&B almost never lands on a full tour and its incumbent stagnates.
    """
    C0 = np.ascontiguousarray(np.asarray(C0, dtype=float))
    n = C0.shape[0]
    fin0 = np.isfinite(C0)
    top = float(C0[fin0].max()) if fin0.any() else 1.0
    BIG = max(1.0, top) * (n + 1) * 10.0 + 1.0      # > any tour cost, by far
    Cw = np.where(fin0, C0, BIG).copy()             # working matrix
    base = Cw.copy()                                # pristine values for undo

    UB = float(UB0)
    best_tour = None
    nodes = 0
    fixed_total = 0
    capped = False
    t_start = time.perf_counter()
    deadline = t_start + time_limit if time_limit is not None else None
    tol = eps * max(1.0, abs(UB))

    Z_AP_root = float("nan")

    def forbid(rows, cols):
        Cw[rows, cols] = BIG

    def undo(rows, cols):
        Cw[rows, cols] = base[rows, cols]

    def rec(depth):
        """One node. The caller has already applied this node's branching arc."""
        nonlocal UB, best_tour, nodes, fixed_total, capped, Z_AP_root

        nodes += 1
        if nodes > node_cap or (deadline is not None
                                and time.perf_counter() > deadline):
            capped = True
            return

        rid, cid = linear_sum_assignment(Cw)
        phi = np.asarray(cid, dtype=np.int64)
        Z = float(Cw[rid, cid].sum())
        if depth == 0:
            Z_AP_root = Z

        if Z >= UB - tol:                     # bound cut (also catches infeasible)
            return

        cyc = subtours_of(phi)
        if len(cyc) == 1:                     # a Hamiltonian tour
            UB = Z
            best_tour = [int(x) for x in cyc[0]]
            return

        # ---- incumbent repair (Miller--Pekny patching inside the tree) ----
        if patch_every and (nodes - 1) % patch_every == 0:
            t = qa.karp_patching(C0, phi)          # on the ORIGINAL costs: any
            tc = qa.tour_cost(C0, t)               # tour is a valid global UB
            t2, c2 = qa.or_opt(C0, t)
            if c2 < tc:
                t, tc = t2, c2
            if tc < UB - tol:
                UB = tc
                best_tour = [int(x) for x in t]
            if Z >= UB - tol:                 # the new bound may prune this node
                return

        # ---- reduced-cost fixing (valid for this whole subtree) ----
        fix_rows = fix_cols = None
        if fix == "node" or (fix == "root" and depth == 0):
            u, v = duals_from_matching(Cw, phi)
            rc = Cw - u[:, None] - v[None, :]
            gap = UB - Z
            allowed = Cw < BIG / 2.0
            mask = allowed & (rc > gap + 1e-12)
            if mask.any():
                fix_rows, fix_cols = np.nonzero(mask)
                forbid(fix_rows, fix_cols)
                fixed_total += int(fix_rows.size)

        # ---- Carpaneto--Toth branching ----
        cand = [c for c in cyc if root not in c]
        if not cand:                          # cannot happen: len(cyc) > 1
            return
        S = min(cand, key=len)
        try:
            for i in S:
                j = int(phi[i])
                # apply the branching arc, recurse, restore it
                Cw[i, j] = BIG
                rec(depth + 1)
                Cw[i, j] = base[i, j]
                if capped:
                    return
        finally:
            if fix_rows is not None:
                undo(fix_rows, fix_cols)

    rec(0)

    if best_tour is None:
        # UB0 was already optimal, or nothing better was found: UB0 stands.
        best_tour = None

    return CTResult(UB, best_tour, nodes, fixed_total,
                    time.perf_counter() - t_start, not capped, Z_AP_root, capped)


# ------------------------------------------------------------------- self test

def _self_test():
    sys.setrecursionlimit(200000)
    # the dual routine is the one piece that could be silently wrong, so check
    # feasibility and optimality of the potentials it returns
    rng = np.random.default_rng(11)
    for n in (30, 80):
        C = rng.random((n, n))
        np.fill_diagonal(C, 1e6)
        phi, _, _ = qa.hungarian_with_duals(C)
        u, v = duals_from_matching(C, phi)
        slack = C - u[:, None] - v[None, :]
        assert slack.min() >= -1e-9, f"dual infeasible, min {slack.min():.3e}"
        Z = float(C[np.arange(n), phi].sum())
        assert abs(u.sum() + v.sum() - Z) < 1e-9, "dual objective != Z_AP"
        on_match = slack[np.arange(n), phi]
        assert abs(on_match).max() < 1e-9, "complementary slackness violated"
        print(f"[duals] n={n}: feasible, sum(u)+sum(v)=Z_AP={Z:.6f}  OK")

    rng = np.random.default_rng(7)
    for n in (20, 40, 60):
        C = rng.random((n, n))
        np.fill_diagonal(C, np.inf)
        ref = qa.solve(C)
        for fx in ("none", "root", "node"):
            r = ct_solve(C, ref.heuristic_cost, fix=fx)
            ok = abs(r.cost - ref.cost) < 1e-9
            print(f"n={n:3d} fix={fx:5s} nodes={r.nodes:6d} "
                  f"fixed={r.fixed_arcs:6d} cost={r.cost:.6f} "
                  f"ref={ref.cost:.6f} {'OK' if ok else 'MISMATCH'}")
            assert ok, (n, fx, r.cost, ref.cost)
            assert r.proved


if __name__ == "__main__":
    _self_test()
