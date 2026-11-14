"""ATSP -> symmetric TSP (2n-node) transformation and a Concorde driver.

Concorde is a SYMMETRIC TSP solver and works with 32-bit integer weights.
To apply it to an asymmetric instance we use the classical 2n-node expansion
and a big-M that is *derived from a proven bound on this instance*, so that
correctness is checkable rather than assumed.

Construction (nodes a_1..a_n, b_1..b_n; index a_i = 2i, b_i = 2i+1):

    w(a_i, b_i) = w(b_i, a_i) = 0                      (pair edges)
    w(b_i, a_j) = w(a_j, b_i) = C_ij + M   (i != j)    (inter edges)
    w(a_i, a_j) = w(b_i, b_j) = 2M         (i != j)    (forbidden)
    diagonal = 0

Derivation, because the constant matters.  Start from the signed form
pair = -M, inter = C_ij, forbidden = G, and let a tour use s pair edges,
r inter edges and f forbidden edges (s + r + f = 2n, s <= n):

    cost = -s*M + sum_inter C + f*G.

  * The all-pair tour a_i1 b_i1 a_i2 b_i2 ... has s = n, r = n, f = 0 and
    cost = T - nM, T the ATSP optimum.
  * f >= 1:  cost >= -nM + G, so this loses iff G > T.
  * f = 0 and s <= n-1:  r >= n+1 and cost >= -(n-1)M, which loses iff M > T.
Adding M to every edge adds 2n*M to every tour and so preserves argmin;
this clears the negative weights Concorde cannot take and gives the table
above with G = 2M.  Both requirements are therefore just M > T, and the
largest weight in the expanded instance is 2M ~ 2T --- INDEPENDENT of n.

That last point is why the constant is worth deriving.  If one shifts the
pair and inter edges but forgets the forbidden edges, the requirement
becomes G > nM + T and the largest weight grows like n^2*S in the scale
S; combined with Concorde's 32-bit integer weights and its numerical
behaviour under large coefficients, that is enough to make the solver
crash (we reproduced SIGSEGV reproducibly at max weight ~1.7e8 while the
same size solved at ~8e7).  The formulation above keeps the weights at
~3.4*S for every n and removes the failure mode.

M = UB + 1 where UB is a proven upper bound on the integer instance, so
M > T is guaranteed rather than hoped for.  Every returned tour is
additionally checked structurally -- exactly n pair edges, zero forbidden
edges, and a decoded ATSP tour whose cost reproduces the claimed optimum --
so a violation would be detected rather than silently absorbed.
"""
import os
import subprocess
import tempfile
import time

import numpy as np

CONCORDE_PKG = os.environ.get("PYCONCORDE", "")


# --------------------------------------------------------------------------
# TSPLIB writer (tsplib95 is not installed; the format is trivial)
# --------------------------------------------------------------------------
def write_tsplib_full_matrix(path, W, name="instance"):
    """Write a symmetric integer matrix W as a TSPLIB FULL_MATRIX file."""
    m = W.shape[0]
    with open(path, "w") as fp:
        fp.write("NAME: {}\n".format(name))
        fp.write("TYPE: TSP\n")
        fp.write("COMMENT: 2n expansion of an ATSP instance\n")
        fp.write("DIMENSION: {}\n".format(m))
        fp.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        fp.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        fp.write("EDGE_WEIGHT_SECTION\n")
        for i in range(m):
            row = W[i]
            fp.write(" ".join(str(int(v)) for v in row))
            fp.write("\n")
        fp.write("EOF\n")


# --------------------------------------------------------------------------
# 2n expansion
# --------------------------------------------------------------------------
def expand(C_int, M):
    """Build the 2n x 2n symmetric integer matrix.  C_int is integer, diag 0.

    Requires M > T (optimal ATSP cost on C_int).  Largest entry is 2M.
    """
    C_int = np.asarray(C_int)
    n = C_int.shape[0]
    F = 2 * M
    m = 2 * n
    W = np.full((m, m), F, dtype=np.int64)
    np.fill_diagonal(W, 0)

    a = np.arange(0, 2 * n, 2)      # a_i
    b = np.arange(1, 2 * n, 2)      # b_i

    # pair edges (a_i, b_i)
    W[a, b] = 0
    W[b, a] = 0

    # inter edges (b_i, a_j) = C_ij + M for i != j
    off = ~np.eye(n, dtype=bool)
    W[np.ix_(b, a)] = np.where(off, C_int + M, 0)
    W[np.ix_(a, b)] = np.where(off, C_int.T + M, 0)
    # NOTE: W[a_i, b_j] must equal C_ji + M; C_int.T gives C_int[j, i].  Correct.
    return W, F


def decode_tour(tour, n):
    """Decode a 2n-node tour into the ATSP arc set.

    Returns (perm, n_pair, n_forb) where perm is the city order of the
    decoded Hamiltonian cycle, or None if the tour is not well formed.

    The direction is read off the arcs, never off the traversal order:
    the edge {b_i, a_j} (i != j) carries weight C_ij + M and therefore
    IS the arc i -> j.  Reading cities off the traversal order instead
    silently reverses the tour when Concorde emits b_i a_i rather than
    a_i b_i, and on an asymmetric matrix the reversed tour has a
    different cost.
    """
    tour = list(map(int, tour))
    m = 2 * n
    if len(tour) != m or len(set(tour)) != m:
        return None

    n_pair = 0
    n_forb = 0
    succ = {}
    for t in range(m):
        u, v = tour[t], tour[(t + 1) % m]
        iu, bu = u // 2, u % 2
        iv, bv = v // 2, v % 2
        if bu == bv:                  # A-A or B-B -> forbidden
            n_forb += 1
        elif iu == iv:                # (a_i, b_i) or (b_i, a_i) -> pair edge
            n_pair += 1
        elif bu == 1 and bv == 0:     # b_i -> a_j  ==  arc i -> j
            succ[iu] = iv
        else:                         # a_p -> b_q  ==  arc q -> p
            succ[iv] = iu

    if len(succ) != n or n_forb != 0 or n_pair != n:
        return None
    # follow the cycle from city 0
    perm, x = [], 0
    for _ in range(n):
        perm.append(x)
        x = succ.get(x, -1)
        if x < 0:
            return None
    if x != 0 or len(set(perm)) != n:
        return None
    return perm, n_pair, n_forb


# --------------------------------------------------------------------------
# Concorde driver (runs in a subprocess so that a crash cannot take us down)
# --------------------------------------------------------------------------
_SOLVE_SRC = r'''
import sys, os, json, time
sys.path.insert(0, os.environ.get("PYCONCORDE", ""))
from concorde.tsp import TSPSolver

fname = sys.argv[1]
tb = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0

# Concorde is a C library and prints its progress straight to fd 1, which
# would corrupt the JSON we return on stdout.  Redirect fd 1 at the OS level
# (Python-level sys.stdout redirection does NOT catch this).
sys.stdout.flush()
saved = os.dup(1)
dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(dn, 1)

payload = None
try:
    t0 = time.perf_counter()
    solver = TSPSolver.from_tspfile(fname)
    res = solver.solve(time_bound=tb, verbose=False, random_seed=0)
    t1 = time.perf_counter()
    payload = {
        "ok": True,
        "tour": [int(x) for x in res.tour],
        "opt": float(res.optimal_value),
        "success": bool(res.success),
        "found_tour": bool(res.found_tour),
        "hit_timebound": bool(res.hit_timebound),
        "t": t1 - t0,
        "ncount": int(solver._ncount),
    }
except BaseException as e:
    payload = {"ok": False, "err": "{}: {}".format(type(e).__name__, e)}
finally:
    os.dup2(saved, 1)
    os.close(saved)
    os.close(dn)

print(json.dumps(payload))
sys.stdout.flush()
'''


def solve_with_concorde(W, name="instance", tmpdir=None, timeout=None,
                        time_bound=-1.0):
    """Write W to a TSPLIB file and solve it with Concorde in a subprocess."""
    import json
    d = tempfile.mkdtemp(dir=tmpdir)
    f = os.path.join(d, "inst.tsp")
    write_tsplib_full_matrix(f, W, name=name)
    src = os.path.join(d, "_solve.py")
    with open(src, "w") as fp:
        fp.write(_SOLVE_SRC)
    p = subprocess.run(
        [sys.executable, src, f, str(time_bound)],
        capture_output=True, text=True, timeout=timeout,
    )
    out = p.stdout.strip().splitlines()
    if not out:
        return {"ok": False, "err": "no stdout; stderr=" + p.stderr[-500:]}
    try:
        return json.loads(out[-1])
    except Exception:
        return {"ok": False, "err": "unparsable stdout: " + out[-1][:300]}
