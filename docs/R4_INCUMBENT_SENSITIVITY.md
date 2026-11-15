# R4 — Sensitivity of the scope marker to the initial incumbent

Reviewer point (second-pass report, R4): `|A~|/n^2` depends on the budget
`Delta = UB - Z_AP`, hence on the initial incumbent. The certified / uncertified
ranges reported in the paper could therefore be an artefact of our own
patching incumbent rather than a property of the instances; the same batch was
used to observe and to report the separation. Recommended experiment:
recompute the statistic under the in-house incumbent and under an LKH-3
incumbent, and check whether the ranges and the ordering survive.

This note records the completed experiment. It is reproduced by

```bash
python scripts/r4_incumbent_sensitivity.py            # from the shipped upper bounds
python scripts/r4_incumbent_sensitivity.py --run-lkh  # recompute LKH-3 arm (needs LKH-3)
```

Only `numpy` is required for the first form; all four upper bounds per instance
are shipped in `data/r4_incumbent/incumbents.json`.

## Protocol

Statistic (identical to Table 13 of the paper and to
`scripts/atsplib_structure.py`):

```
Z_AP, u*, v*  = optimal assignment and optimal duals
rc_ij         = max(C_ij - u*_i - v*_j, 0)
|A~|          = #{ (i,j) not in M* and not in T* : rc_ij <= UB - Z_AP }
statistic     = 100 * |A~| / n^2   (percent)
```

Four incumbents, all evaluated on the same matrices and the same duals; only
`UB` changes:

| arm | incumbent |
|---|---|
| A0 | bare Karp patching tour, no local search |
| A  | shipped incumbent = patching + Or-opt (Table 13 `UB0`) |
| B  | LKH-3, `RUNS=10`, `TIME_LIMIT=20`, `SEED=1`; cost recomputed on the original matrix |
| C  | published optimum, used as an oracle lower envelope |

Instances: the 27 ATSPLIB matrices in `instances/` (4 certified, 23 not) plus
the 15 random-uniform reference instances (`n = 100,200,300,500,800`,
seeds 1–3, `default_rng(1000*n+seed)`). Integer TSPLIB matrices are passed to
LKH-3 unscaled; float matrices are scaled by `1e6`; the diagonal and
infeasible arcs use the same `1e7` sentinel as `scripts/lkh_vs_quasi.py`.

LKH-3 reproduces the published optimum on 26 of the 27 instances (on `p43` it
is one unit above), so arms B and C give identical `|A~|/n^2` on all 27. The
separation observed under A therefore already cannot be strengthened away by
a better incumbent: C is the tightest attainable budget.

## Results on 27 ATSPLIB instances

| arm | certified (%) | uncertified (%) | margin (pp) | AUC | Spearman vs A | Kendall vs A | held-out |
|---|---|---|---|---|---|---|---|
| A0 | 10.8–23.6 | 58.8–97.0 | 35.2 | 1.000 | 0.972 | 0.886 | 8/8 |
| A  | 10.8–23.6 | 43.2–97.0 | 19.6 | 1.000 | 1.000 | 1.000 | 8/8 |
| B  | 10.8–23.6 | 32.6–92.9 | 9.0  | 1.000 | 0.869 | 0.692 | 8/8 |
| C  | 10.8–23.6 | 32.6–92.9 | 9.0  | 1.000 | 0.869 | 0.692 | 8/8 |

* The certified group is the four `rbg` instances; it is unchanged in every arm
  because `Z_AP = OPT` there (`Delta = 0`).
* AUC is 1.000 in all four arms: every uncertified instance ranks above every
  certified one. A label-permutation test on arm A gives `p = 5e-5`
  (20,000 draws, no permutation attains the observed separation; this is the
  resolution limit of the test).
* Bootstrap (`10,000` resamples of the 27 instances, fixed seed): the 95% CI of
  the separation margin is `[35.2, 48.0]` for A0, `[19.6, 36.2]` for A and
  `[9.0, 23.2]` for B/C; 100% of resamples keep a positive margin, and the
  AUC 95% CI is `[1.000, 1.000]` in every arm.
* The margin shrinks monotonically as the incumbent improves (35.2 → 19.6 →
  9.0 pp), and the mean absolute change from A to B is 13.9 pp with a maximum
  rank shift of 12 of 27 places. The *ordering* is therefore stable enough to
  preserve complete separation, but not stable enough to justify a fixed
  numerical threshold.
* Held-out check: fix the separating value on the 19 distributed ATSPLIB
  files only, `(min(uncertified) + max(certified))/2`, and classify the 8
  reconstructed `ftv90`–`ftv160` prefixes that were added later. All four arms
  classify 8/8 correctly, but the threshold itself moves 41.2% → 36.2% →
  31.2% across A0/A/B. A threshold calibrated for one incumbent does not
  transfer to another.

## Random-uniform reference (15 instances)

| arm | min | median | max |
|---|---|---|---|
| A (patching) | 1.98% | 6.36% | 23.25% |
| B (LKH-3)    | 0.01% | 0.14% | 1.01% |

Arm A reproduces the paper's `2.0%–23.2%`, median `6.4%`. Under LKH-3 the
whole reference set drops below 1.1%, so the gap to the uncertified ATSPLIB
instances widens rather than closes. Spearman between the two arms is 0.44:
at these very low densities the ordering is genuinely sensitive to the
incumbent, which is another reason not to read `|A~|/n^2` as a calibrated
score.

The known one-way counterexample survives: the `n = 300`, seed 2 instance of
Section 6.6.4 has 3.39% under A and 0.93% under B and is still not certified,
because that failure is in the packing search, not in the arc budget. A low
value is therefore not a sufficient condition for certification.

## What this establishes

`|A~|/n^2` is a joint statistic of the instance and the incumbent. The
qualitative scope condition — certified instances admit few non-basic arcs,
uncertified structured instances admit many — is robust to replacing the
shipped incumbent with a bare patching tour, an LKH-3 tour, or the optimum
itself, and it passes both a bootstrap check and a genuinely held-out set of
eight instances. The numerical ranges and any threshold derived from them are
not portable across incumbents, and a low value is not sufficient for a
certificate. The paper therefore keeps the wording *empirical scope marker*
and does not present the statistic as a routing rule or a classifier.

## Files

| path | content |
|---|---|
| `scripts/r4_incumbent_sensitivity.py` | reproducible analysis (and optional LKH-3 re-run) |
| `data/r4_incumbent/incumbents.json` | the four upper bounds per instance + protocol |
| `data/r4_incumbent/atsplib_results.json` | per-instance per-arm full result (27 rows) |
| `data/r4_incumbent/random15_results.json` | per-instance per-arm full result (15 rows) |
| `data/r4_incumbent/analysis.json` | ranges, AUC, rank statistics, bootstrap, held-out |
