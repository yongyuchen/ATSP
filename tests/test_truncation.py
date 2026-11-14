"""Regression tests for the certificate-truncation invariant (Theorem 2 of the paper).

The soundness theorem says `proved = True` is returned only when the final round
of the budget ladder ran to completion: every vertex-disjoint packing within
budget was enumerated and none of the resource caps --- per-round cycle cap,
search-node cap, wall-clock deadline --- was hit.

These tests force each cap to bind on a small instance that the solver *can*
certify under generous limits, and assert that the solver then reports
`proved = False` rather than silently issuing a certificate for a tour it did
not actually prove optimal.

Run:
    python tests/test_truncation.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))

import quasipoly_atsp as qa  # noqa: E402

# The n=20, seed=1 i.i.d. U(0,1) instance used across the paper's benchmark
# (generator = seed * 7919 + n), whose certified optimum is known exactly.
N, SEED = 20, 1
OPT = 1.6475735650387524
TOL = 1e-9


def instance():
    C = np.random.default_rng(SEED * 7919 + N).random((N, N))
    np.fill_diagonal(C, np.inf)
    return C


def main():
    C = instance()

    # 0) baseline: generous limits certify the known optimum.
    base = qa.solve(C, time_limit=120)
    assert base.proved, "baseline should certify n=20 seed=1"
    assert abs(base.cost - OPT) < TOL, (base.cost, OPT)
    print("baseline: proved=True, cost=%.12f" % base.cost)

    checks = [
        ("node cap (search_cap)", dict(search_cap=50), dict(time_limit=120)),
        ("cycle cap (cycle_cap)", dict(cycle_cap=5), dict(time_limit=120)),
        ("wall clock (time_limit)", dict(time_limit=0.0), {}),
    ]
    for label, kw, outer in checks:
        r = qa.solve(C, **outer, **kw)
        assert not r.proved, "%s must not report proved=True" % label
        # The truncated search returns the incumbent (suboptimal), never a
        # certificate.  Assert it is indeed suboptimal, so a false positive
        # proved=True would have been caught by the optimum check above.
        assert r.cost > OPT + TOL, "%s: truncated tour should be suboptimal" % label
        print("%s: proved=%s  overflow=%s  cost=%.6f  (OPT=%.6f)" %
              (label, r.proved, r.overflow, r.cost, OPT))

    # 1) sanity: the result object exposes the overflow flag for truncation.
    r = qa.solve(C, time_limit=120, search_cap=50)
    assert r.overflow, "overflow flag should be set when the node cap binds"

    print("all truncation invariants hold")


if __name__ == "__main__":
    main()
