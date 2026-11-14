"""Structural statistics of the budget test, for the 19 ATSPLIB instances.

Reproduces the two structural columns of the paper's ATSPLIB table:

    Delta / Z_AP          the initial relative budget,
    |A~| / n^2            the fraction of non-basic arcs that survive it,

where Delta = UB0 - Z_AP and UB0 is the Karp-patching + Or-opt incumbent, i.e.
exactly the quantity the ladder of ``quasipoly_atsp.solve`` starts from.

|A~| = #{ (i,j) : (i,j) not in M*, (i,j) not in T*, rc_ij <= Delta }.

The script also prints the worst reduced cost of an arc of T* (rho), which the
solver requires to be <= 1e-9: it is the machine-checkable precondition that
lets the enumeration price every tree arc at zero.

Usage:
    python scripts/atsplib_structure.py [--out results/atsplib_structure.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "solver"))

import atsplib                      # noqa: E402
import quasipoly_atsp as qa         # noqa: E402


def instance_stats(path):
    name, C = atsplib.read_atsp(path)
    n = C.shape[0]

    phi, uh, vh = qa.hungarian_with_duals(C)
    u, v = qa.optimal_duals(C, phi, hung0=(uh, vh))
    rc = np.where(np.isfinite(C), C - u[:, None] - v[None, :], np.inf)
    rc = np.clip(rc, 0.0, None)
    Z_AP = float(C[np.arange(n), phi].sum())

    tree_arcs, rho = qa.build_basis_tree(rc, phi)
    in_tree = np.zeros((n, n), dtype=bool)
    for i, j in tree_arcs:
        in_tree[i, j] = True

    nonbasic = np.ones((n, n), dtype=bool)
    nonbasic[np.arange(n), phi] = False          # M*
    nonbasic &= ~in_tree                          # T* \ M*

    incumbent = qa.karp_patching(C, phi)
    incumbent, UB0 = qa.or_opt(C, incumbent)
    UB0 = float(UB0)

    delta = UB0 - Z_AP
    surviving = int((nonbasic & (rc <= delta)).sum())

    return {
        "instance": name,
        "n": n,
        "Z_AP": Z_AP,
        "UB0": UB0,
        "Delta": delta,
        "Delta_over_Z_AP": (delta / Z_AP) if Z_AP else None,
        "absA": surviving,
        "absA_over_n2": surviving / (n * n),
        "rho": float(rho),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=os.path.join(_ROOT, "instances"))
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "results", "atsplib_structure.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.instances, "*.atsp")))
    if not paths:
        sys.exit("no .atsp files in %s" % args.instances)

    rows = []
    print(f"{'instance':9s} {'n':>4s} {'Z_AP':>10s} {'UB0':>10s} {'Delta':>10s} "
          f"{'Delta/Z':>8s} {'|A~|':>8s} {'|A~|/n^2':>9s} {'rho':>7s}")
    for path in paths:
        r = instance_stats(path)
        rows.append(r)
        rel = "     n/a" if r["Delta_over_Z_AP"] is None else \
            "%7.3f%%" % (100 * r["Delta_over_Z_AP"])
        print(f"{r['instance']:9s} {r['n']:4d} {r['Z_AP']:10.4f} {r['UB0']:10.4f} "
              f"{r['Delta']:10.4f} {rel:>8s} {r['absA']:8d} "
              f"{100*r['absA_over_n2']:8.2f}% {r['rho']:7.1e}")

    rows.sort(key=lambda r: r["n"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote %s (%d instances)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
