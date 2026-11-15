# Experimental protocol

This file records the constants that make a new run comparable with the numbers in
the paper. Changing any of them changes the instances or the comparison.

## Cost model

| Item | Value |
|---|---|
| Distribution | i.i.d. `U(0,1)` off the diagonal |
| Diagonal | `+inf` (self-loops are infeasible, not merely expensive) |
| Matrix | dense, asymmetric, `C[i,j]` and `C[j,i]` independent |

## Random instances

Two generators are used, deliberately, and they must not be mixed.

**Benchmark set (23 instances)** — `numpy.random.default_rng(seed * 7919 + n)`:

| n | seeds |
|---|---|
| 20 | 1, 2, 3 |
| 30 | 1, 2, 3 |
| 50 | 1, 2, 3 |
| 80 | 1, 2, 3 |
| 100 | 1, 2, 3 |
| 150 | 1, 2 |
| 200 | 1, 2 |
| 300 | 1, 2 |
| 500 | 1 |
| 800 | 1 |

21 of these have `n <= 300`; that is the set behind the MILP and AP-B&B comparisons.

**Scaling set (30 instances)** — `numpy.random.default_rng(1000 * n + seed)`, three
seeds at each of `n = 20, 30, 50, 80, 100, 150, 200, 300, 500, 800`. This is the set
behind the runtime distribution and the scaling fit.

## ATSPLIB

Twenty-seven instances, the ones in the paper's ATSPLIB table:

```
br17  ft53  ft70  ftv33  ftv35  ftv38  ftv44  ftv47  ftv55  ftv64
ftv70 ftv90 ftv100 ftv110 ftv120 ftv130 ftv140 ftv150 ftv160 ftv170
kro124p p43 rbg323 rbg358 rbg403 rbg443 ry48p
```

TSPLIB distributes 19 of these as files. The eight `ftv90`–`ftv160` instances are,
per TSPLIB95 §3.3, prefix subproblems of `ftv170`; they are reconstructed here by
taking the leading `X+1` rows and columns of `ftv170` and are included in
`instances/`. Their published optima were verified independently by Concorde,
CPLEX-DFJ and SCIP-DFJ (8/8 each); see `docs/FTV27_RESULTS.md`.

Matrices are read as explicit full matrices. Every entry at or above the TSPLIB
sentinel `9 999 999` becomes `+inf`, and the diagonal becomes `+inf` regardless of
what the file carries.

## Solver settings

| Item | Value |
|---|---|
| Threads | 1 (all solvers) |
| Time limit | 60 s for the external arms; the our-solver limit is `--ours-tl` |
| MIP gap | 0 (only proven optima are accepted) |
| DFJ | assignment + subtour elimination by lazy cuts |
| MTZ | assignment + Miller–Tucker–Zemlin, only for `n <= 300` |
| Warm start | none, unless a table says otherwise |
| Memory guard | 8 GB `RLIMIT_DATA` per worker; override with `QA_WORKER_RLIMIT_GB` (`0` disables) |

CPLEX statuses 101 (*integer optimal*) and 102 (*integer optimal, tolerance*) both
count as certificates, at the solver's default MIP tolerance. Every reported optimum
is cross-checked between at least two independent solvers before a ratio is computed.

## Timing

Wall-clock time, one process per solver, single-threaded. For instances that finish
in well under a second the reported value is the minimum of five repetitions;
otherwise it is a single run. The runtime is heavy-tailed, so the paper reports
medians and per-instance values rather than means with confidence intervals, and
absolute times from different batches are never pooled. Speedup ratios are
dimensionless and are computed only within one batch.
