# ATSP

A certified solver for the random Asymmetric Traveling Salesman Problem, based on
the quasi-polynomial algorithm of Bell and Frieze
([arXiv:2308.02946](https://arxiv.org/abs/2308.02946)).

This repository is the artifact for the paper *A Certifying Implementation of the
Bell–Frieze Quasi-Polynomial Algorithm for the Asymmetric Traveling Salesman
Problem*. It contains the solver, the optimality-certificate machinery, the
benchmark drivers, the 27 ATSPLIB instances used in the paper, and the expected
values used to check a run.

The solver returns a tour **together with an optimality certificate**: `proved=True`
means the budget-defined search space was exhausted, so the tour is optimal for the
instance as given. The certificate does not rely on the random-cost model; the model
only governs how quickly the search terminates. When a cap or a deadline truncates
the search, the flag is `False` and no optimality is claimed.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`numpy` and `scipy` are the only hard requirements. CPLEX, SCIP, Gurobi, LKH-3 and
Concorde are optional and only needed for the corresponding baselines; the driver
skips any solver whose import or licence fails and records the reason.

The Cython kernel is optional. The pure-Python path is bit-identical in every output
it feeds; the kernel is only faster.

```bash
cd solver/cy && python setup.py build_ext --inplace && cd ../..
```

---

## 2. Quick start

```python
import numpy as np
import sys
sys.path.insert(0, "solver")
import quasipoly_atsp as qa

rng = np.random.default_rng(0)
C = rng.random((300, 300))
np.fill_diagonal(C, np.inf)

r = qa.solve(C, time_limit=60)
print(r.cost, r.proved, r.Z_AP, r.search_nodes)
```

Command line, one instance of the paper's benchmark set:

```bash
python run_benchmark.py --group random --tl 60 --start 0 --end 1 --solver ours
```

---

## 3. Reproducing the paper

All commands are run from the repository root. `results/` is created automatically.

| Paper item | Command | Notes |
|---|---|---|
| Four-engine random benchmark (23 instances) | `python run_benchmark.py --group random --tl 60 --out results/random.json` | needs CPLEX / SCIP / Gurobi for the external columns |
| ATSPLIB, our solver only (27 instances) | `python run_benchmark.py --group atsplib --tl 60 --solver ours --out results/atsplib.json` | no external solver needed |
| ATSPLIB with CPLEX / SCIP | `python run_benchmark.py --group atsplib --tl 60 --solver ours,cplex,scip --out results/atsplib_cplex.json` | |
| Structural columns of the ATSPLIB table ($\Delta/Z_{AP}$, $|\tilde A|/n^2$) | `python scripts/atsplib_structure.py` | writes `results/atsplib_structure.json` |
| Runtime distribution and scaling fit | `python scripts/fit_runtime.py` | 30 instances, best-of-5 below 1 s |
| Certificate audit (2 880 instances) | `python scripts/audit_certificate.py` | n = 6..9, six cost models, brute-force ground truth |
| LKH-3 comparison | `LKH_BIN=/path/to/LKH python scripts/lkh_vs_quasi.py` | LKH-3 is not bundled |
| Concorde comparison | `PYCONCORDE=/path/to/pyconcorde python scripts/concorde_atsp.py` | pyconcorde + Concorde binary required |
| AP-based branch and bound baseline | `python scripts/ct_baseline.py` | baseline implementation + self-test; Table 11 combines `ct_solve` with the same nine instances |

Every benchmark writes one JSON row per instance **after that instance finishes**, so
an interrupted run can be restarted: rows already present are skipped.

### Instance generation

Two seed conventions appear in the paper and mixing them silently changes the
instances. Both are reproduced here.

* **Four-engine / MILP benchmark, 23 instances** — `numpy.random.default_rng(seed * 7919 + n)`,
  with the sizes and seeds listed in `docs/PROTOCOL.md`. This is what
  `run_benchmark.py --group random` uses, and it is the convention behind
  `data/expected_random.json`.
* **Scaling set, 30 instances** — `numpy.random.default_rng(1000 * n + seed)`,
  three seeds at each of `n = 20, 30, 50, 80, 100, 150, 200, 300, 500, 800`.
  This is what `scripts/fit_runtime.py` uses.

In both cases the matrix is i.i.d. `U(0,1)` off the diagonal and `+inf` on it.
`data/expected_random.json` and `data/expected_atsplib.json` give the reference
optimum for every instance; the driver checks a run against them as it goes. The
ATSPLIB job list is taken from `instances/` (27 files), not from the expected-value
file, so a stale reference table cannot silently shrink a run.

**One deliberate exception**, documented in the paper: the `n500_s1` entry of
`expected_random.json` holds our un-warm-started heuristic upper bound
`1.7193316948`, not the true optimum `1.6453830212`. A solver returning the true
optimum on that row is therefore flagged `BAD` by the cost-agreement check; this
is why the paper reports 22/23 agreement on the random set.

### What "proved" means

`proved=True` is issued only on a *complete* ladder round, i.e. when the admitted arc
set contains every arc an improving tour could use, and only if no cap truncated the
packing search. `Result.lb_cert` and `Result.rung_ok` expose the partial certificate
of the last completed round when the full one is out of reach. See
`docs/IMPLEMENTATION_NOTES.md` for the three implementation choices that this
discipline depends on.

---

## 4. Layout

```
.
├── run_benchmark.py        four-engine driver (ours / CPLEX / SCIP / Gurobi)
├── solver/
│   ├── quasipoly_atsp.py   solver, certificate, Karp patching, Or-opt
│   ├── atsplib.py          TSPLIB ATSP reader
│   ├── atsp-sol.html       published ATSPLIB optima (scoring only)
│   └── cy/                 optional Cython kernel + element-wise validators
├── workers/                one subprocess per external solver
├── scripts/                experiments that are not the four-engine driver
├── instances/              27 ATSPLIB instances used in the paper (19 distributed + 8 reconstructed ftv prefixes)
├── data/                   expected values and the JSON behind the paper tables
├── docs/                   protocol, implementation notes, reproducibility
└── results/                outputs (created at run time)
```

`instances/random100.atsp` is deliberately not included: it is not one of the 27
ATSPLIB instances used in the paper.

Inline comments in `solver/quasipoly_atsp.py` are partly in Chinese; the English
summary of every non-obvious choice is `docs/IMPLEMENTATION_NOTES.md`.

---

## 5. External solvers

| Solver | How to get it | Notes |
|---|---|---|
| CPLEX 22.1 | IBM ILOG CPLEX Optimization Studio | Python API supports 3.9/3.10; set `CPLEX_STUDIO_DIR` or pass `--cplex-py` |
| SCIP | `pip install pyscipopt` | bundled SCIP binary |
| Gurobi | `pip install gurobipy` + licence | `MIPGap=0` is used, so a full licence is needed beyond the size-limited one |
| LKH-3 | <http://webhotel4.ruc.dk/~keld/research/LKH-3/> | set `LKH_BIN`; the driver writes `TYPE: ATSP`, never `TYPE: TSP` |
| Concorde | <https://www.math.uwaterloo.ca/tsp/concorde.html> + `pyconcorde` | set `PYCONCORDE`; ATSP is passed through the $2n$-node expansion described in the paper |

The CPLEX and Gurobi arms are optional. Without them the driver still reproduces
every "ours" column and the SCIP columns.

---

## 6. Data and licences

The 8 reconstructed `ftv90`–`ftv160` prefix instances and their cross-solver results are in `data/ftv_extra/` (see `docs/FTV27_RESULTS.md`). The ATSPLIB instances in `instances/` are redistributed under the terms of
[TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/); the published
optima in `solver/atsp-sol.html` are used only to score a run and are never given to
any solver.

The code in this repository is released under the MIT licence; see `LICENSE`.

## 7. Citation

If you use this code, please cite the paper (see `CITATION.cff`).

---

## Repository history

The `Untitled*.ipynb` files at the root are the original exploratory notebooks and
are kept for provenance. They are not needed to run anything in this repository.
