"""SCIP exact ATSP baselines (run with the interpreter that has PySCIPOpt).

Reuses scip_dfj.py: DFJ branch-and-cut (constraint handler) and static MTZ.
Usage: python scip_worker.py INSTANCE.json RESULT.json [time_limit] [max_n_mtz]
"""
import json
import os
import sys
import time

import numpy as np

# --- portability: `resource` does not exist on Windows -------------------
try:
    import resource
    resource.setrlimit(resource.RLIMIT_DATA, (8 << 30, 8 << 30))
except Exception:                     # noqa: BLE001  (Windows / restricted)
    pass

# --- portability: locate scip_dfj.py next to this file -------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scip_dfj import scip_atsp_dfj, scip_atsp_mtz  # noqa: E402


def wrap(C, fn, tl):
    t0 = time.time()
    try:
        cost, proved, dt, cuts = fn(C, time_limit=tl)
    except Exception as e:
        return {"cost": float("nan"), "proved": False, "t": time.time() - t0,
                "cuts": -1, "error": "%s: %s" % (type(e).__name__, e)}
    return {"cost": float(cost), "proved": bool(proved), "t": float(dt),
            "cuts": int(cuts)}


def main():
    inst_path, res_path = sys.argv[1], sys.argv[2]
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    max_n_mtz = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    C = np.asarray(json.load(open(inst_path)), dtype=float)
    np.fill_diagonal(C, np.inf)
    out = {"dfj": wrap(C, scip_atsp_dfj, tl)}
    if C.shape[0] <= max_n_mtz:
        out["mtz"] = wrap(C, scip_atsp_mtz, tl)
    else:
        out["mtz"] = None
    json.dump(out, open(res_path, "w"), indent=2)


if __name__ == "__main__":
    main()
