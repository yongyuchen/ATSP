"""Minimal TSPLIB ATSP reader + instance statistics.

Only FULL_MATRIX / EXPLICIT ATSP files are handled, which is the whole ATSPLIB
set.  The TSPLIB convention is that the diagonal of an explicit ATSP matrix is a
placeholder (0, -1, 9999, 9999999 ... depending on the file) and must be treated
as forbidden; we always overwrite it with +inf.
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = 9_999_999  # TSPLIB's "infinite" placeholder in ft*/ftv*/kro124p


def read_atsp(path):
    """Return (name, C) with C float array, diagonal +inf, non-finite kept as inf."""
    with open(path) as fh:
        text = fh.read()
    head, _, rest = text.partition("EDGE_WEIGHT_SECTION")
    meta = dict()
    for line in head.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().upper()] = v.strip()
    if meta.get("TYPE", "").upper() != "ATSP":
        raise ValueError(f"{path}: TYPE is {meta.get('TYPE')!r}, expected ATSP")
    if meta.get("EDGE_WEIGHT_FORMAT", "").upper().strip() not in ("FULL_MATRIX",):
        raise ValueError(f"{path}: unsupported format {meta.get('EDGE_WEIGHT_FORMAT')!r}")
    n = int(re.search(r"\d+", meta["DIMENSION"]).group())
    body = rest.split("EOF")[0].split("DISPLAY_DATA_SECTION")[0]
    vals = np.array([float(x) for x in body.split()], dtype=float)
    if vals.size != n * n:
        raise ValueError(f"{path}: got {vals.size} weights, expected {n*n}")
    C = vals.reshape(n, n).copy()
    C[C >= BIG] = np.inf
    np.fill_diagonal(C, np.inf)
    return meta.get("NAME", os.path.basename(path)), C


def known_optima():
    """Best-known ATSPLIB values, parsed from the mirrored atsp-sol.html."""
    out = {}
    p = os.path.join(HERE, "atsp-sol.html")
    if os.path.exists(p):
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"<LI>\s*([A-Za-z0-9_]+)\s*:\s*(\d+)", txt):
            out[m.group(1)] = int(m.group(2))
    return out


def available():
    return sorted(f for f in os.listdir(HERE) if f.endswith(".atsp"))


def describe(name, C):
    n = C.shape[0]
    fin = np.isfinite(C)
    off = ~np.eye(n, dtype=bool)
    vals = C[fin & off]
    sym = np.allclose(C, C.T, atol=0, equal_nan=True)
    return dict(name=name, n=n,
                n_infeasible=int((~fin & off).sum()),
                finite_min=float(vals.min()), finite_max=float(vals.max()),
                n_zero=int((vals == 0).sum()),
                symmetric=bool(sym))


if __name__ == "__main__":
    opt = known_optima()
    from scipy.optimize import linear_sum_assignment
    print(f"{'name':10s} {'n':>5s} {'infeas':>7s} {'min':>6s} {'max':>9s} "
          f"{'zeros':>7s} {'Z_AP':>10s} {'opt':>8s} {'gap%':>7s}")
    for f in available():
        nm, C = read_atsp(os.path.join(HERE, f))
        d = describe(nm, C)
        r, c = linear_sum_assignment(C)
        z = float(C[r, c].sum())
        key = nm if nm in opt else nm.rstrip("p")
        o = opt.get(key)
        g = "" if (o is None or z == 0) else f"{100*(o-z)/z:7.3f}"
        print(f"{nm:10s} {d['n']:5d} {d['n_infeasible']:7d} {d['finite_min']:6.0f} "
              f"{d['finite_max']:9.0f} {d['n_zero']:7d} {z:10.0f} "
              f"{'' if o is None else o:>8} {g}")
