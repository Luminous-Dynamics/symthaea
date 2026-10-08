# RH-006 Conditioning × Ridge × Estimand-Policy × Size/Power — v1

## Status

Research diagnostic only. Not approved for formal inference.

`applicability = not-approved-for-execution`

`selection = stop-assumption-failure`

`formal p-value = disabled`

`formal confidence interval = disabled`

This artifact is an independent estimator-mechanics reimplementation of the RH-006 rolling evaluator. It is not a Symthaea Rust workflow result.

## Parent source

The experiment was frozen against branch head:

`451f02a297c2122aae99d34a89835c920f303ad3`

The branch ref was independently checked before execution. The PR metadata cache exposed a stale head, so the branch ref was treated as authoritative.

## Purpose

This experiment attacks the remaining methodological question directly:

`origin-local conditioning × ridge × near-singularity × full-space equal-risk null × identifiable-subspace null × null calibration × local power`

The goal is not to choose a preferred estimator. It is to determine whether numerical stabilization or dimension reduction creates an apparent inferential improvement that disappears once the scientific estimand is kept explicit.

## Frozen policies

### 1. Full-space hard-fail

- full three-channel relational feature space;
- ordinary least squares;
- origin-local singular-value-ratio threshold `1e-3`;
- any required origin below the threshold makes the complete diagnostic dataset inadmissible.

### 2. Full-space ridge stabilization

- full three-channel relational feature space;
- fixed ridge, with separate runs at `1e-8` and `1e-4`;
- no conditioning stop;
- origin-local conditioning remains diagnostic and is retained in the result.

### 3. Identifiable-subspace estimand

- per-origin SVD of the standardized three-channel relational design;
- retain singular directions with ratio at least `1e-3`;
- fixed ridge `1e-8`;
- fit the baseline context plus the retained projected relational coordinates;
- the resulting estimand is explicitly **not** the original three-channel coefficient-space hypothesis.

The projection rule is based on the design alone, not on the outcome, but the retained subspace can still vary by origin and by sample realization.

## Frozen directions

Two coefficient-space directions were predeclared:

- `strong = [1, 1, 0.5] / ||[1,1,0.5]||`, aligned with the approximately dominant near-collinear feature direction;
- `weak = [0, 1, -2] / ||[0,1,-2]||`, approximately orthogonal to the dominant direction and therefore probing the weakly identified contrast.

This weak direction is the important adversarial case. Under the near-singular construction, the observable variation in this contrast collapses while the outcome perturbation remains defined.

## Frozen null and alternative

The experiment does **not** substitute the structural null

`beta_relational = 0`

for finite-sample equal forecast risk.

For every feature path and policy/direction pair, the diagnostic solves the finite-sample equal-risk quadratic under the actual AR(0.5) + heteroskedastic error covariance used to generate the outcome. The positive root is the policy-specific equal-risk point.

The local alternative is then an outcome-scale perturbation equal to

`0.25 × nominal innovation scale = 0.02`

in RMS effect, converted to coefficient scale using the realized standard deviation of the chosen relational direction.

This keeps the alternative local on the outcome scale even when the coefficient required to express a weak direction becomes very large.

## Statistic

Each origin produces a target-window squared-loss differential:

`D[o] = MSE_context[o] - MSE_relational[o]`

The diagnostic statistic is the origin-level mean with a fixed Bartlett-lag-3 studentizer:

`T = sqrt(O) * mean(D[o]) / sqrt(Bartlett_LRV(D[o]))`

A fixed `N(0,1)` cutoff of `1.645` is used only as a cross-condition stress-screening threshold. It is **not** a repository-approved rejection rule and is not a formal p-value procedure.

## Geometry

- train = 48
- gap = 4
- test = 16
- origins = 24
- step = 16
- horizon = 1
- outcome errors = AR(0.5)
- heteroskedastic innovation scale driven by the common-driver channel
- Monte Carlo paths = 200 per conditioning level
- conditioning stresses = `1e-2`, `1e-3`, `1e-4`, exact `0`

## Executed result

### Origin-local admissibility

The hard-fail policy is already enough to make the local boundary explicit:

| epsilon | hard-fail datasets rejected |
|---:|---:|
| `1e-2` | 0 / 200 |
| `1e-3` | 200 / 200 |
| `1e-4` | 200 / 200 |
| `0` | 200 / 200 |

Thus the previously observed pooled-vs-local distinction is operationally decisive: once the actual estimator unit is the origin, near-singularity is not something the aggregate statistic can safely average away.

### Weak-direction equal-risk and power behavior

| epsilon | policy | equal-risk root availability | null rejection | local-power rejection |
|---:|---|---:|---:|---:|
| `1e-2` | full hard-fail | 100% | 10.5% | 80.5% |
| `1e-2` | ridge `1e-8` | 100% | 9.0% | 81.0% |
| `1e-2` | ridge `1e-4` | 100% | 8.5% | 80.5% |
| `1e-2` | subspace `1e-3` | 100% | 7.5% | 81.0% |
| `1e-3` | full hard-fail | 0% | — | — |
| `1e-3` | ridge `1e-8` | 100% | 5.5% | 81.5% |
| `1e-3` | ridge `1e-4` | 100% | 7.5% | 79.0% |
| `1e-3` | subspace `1e-3` | 96.5% | 4.7% | 12.4% |
| `1e-4` | full hard-fail | 0% | — | — |
| `1e-4` | ridge `1e-8` | 100% | 8.0% | 82.5% |
| `1e-4` | ridge `1e-4` | 41.5% | 2.4% | 7.2% |
| `1e-4` | subspace `1e-3` | 5.0% | 0.0% | 0.0% |
| `0` | all policies | 0% | — | — |

The strong direction behaves very differently: across the same pilot it retains roughly 74–83% rejection under the local alternative for admissible policies, including severe conditioning. That divergence between strong and weak directions is exactly the point of the experiment.

## Main findings

### 1. Hard-fail converts the problem into an explicit scientific boundary

At `epsilon = 1e-3` and below, the origin-local `1e-3` gate rejects every feature path in the 200-replicate diagnostic. This is not a loss of statistical power; it is a declaration that the original full-space estimator is not admissible for those geometries.

### 2. Ridge preserves execution but can leave the inferential geometry pathological

At `epsilon = 1e-4`, the `1e-8` ridge lane remains computationally executable and retains an equal-risk root for the weak direction in all 200 paths. But the median weak-direction root is about `1.43e3`, compared with about `0.092` in the strong direction. This is an enormous coefficient-scale representation of the same fixed outcome-scale perturbation.

Increasing ridge to `1e-4` makes the weak-direction null geometry worse rather than uniformly better: only 41.5% of paths retain a positive equal-risk root, and the median root among available paths rises above `9e3`.

Thus “the solver is stable” is not interchangeable with “the risk surface is scientifically identified.”

### 3. Projection does not rescue the original weak-direction hypothesis

The identifiable-subspace policy is the clearest adversarial case. At `epsilon = 1e-3`, it still has a weak-direction equal-risk root on 96.5% of paths, but local rejection falls from about 80% for full-space ridge to only 12.4%.

At `epsilon = 1e-4`, only 5% of paths retain a weak-direction equal-risk root and neither null nor alternative rejection is observed in the available cells.

This is not evidence that SVD projection is “bad.” It is evidence that its predictive/inferential target is different. Once the weak direction is discarded, strong performance for retained directions cannot be advertised as evidence for the original three-channel hypothesis.

### 4. The exact-singular case exposes the semantic endpoint

At exact rank-one relational structure, the weak-direction equal-risk root disappears for every tested policy. Positive ridge still makes the linear system executable, but there is no finite weak-direction coefficient representation that recovers the original equal-risk geometry in this diagnostic.

That is the cleanest demonstration yet of:

`ridge solves the normal equations`

`≠ feature geometry is identified`

`≠ original equal-risk surface is identified`.

## Statistical interpretation boundary

The rejection rates above are deliberately not called formal size/power estimates. They are fixed-cutoff Monte Carlo diagnostics under a synthetic DGP whose covariance structure is fully known to the simulation.

The procedure does not yet establish:

- a valid real-data null DGP with estimated nuisance parameters;
- a valid bootstrap or asymptotic critical value for the actual RH-006 estimator;
- uniform size control over the null surface;
- coverage for a confidence interval;
- compatibility of the Rust implementation with the independent mechanics mirror;
- validity after data-dependent subspace selection.

The inferential selector therefore remains closed.

## Literature bridge

Giacomini & White explicitly preserve finite-sample estimator effects and allow rolling-window forecasts and general estimation methods, which is why their framework remains conceptually relevant. But Zhu & Timmermann show that a conditional expected-accuracy null can fail under rolling estimation, so the RH-006 implementation cannot inherit validity from the citation alone.

Clark & McCracken are especially relevant to the present finite-sample null distinction: equal forecast accuracy can occur with nonzero extra coefficients because of an estimation-error/bias-variance tradeoff. Doko Tchatoka & Haque further support a full-refit bootstrap direction when nesting and serial dependence matter.

The subspace result also reinforces the broader post-selection literature: once an analysis target is data-dependent, the inferential object needs its own explicit definition rather than being silently treated as inference for the pre-selection parameterization.

## Reproducibility receipt

- command: `python scripts/research/rh006_conditioning_ridge_estimand_policy_power.py --mc 200 --seed 20261021`
- committed script Git blob SHA-1: `f9cad5c47ffffd694c509d4772379f15a3381afc`
- script SHA-256: `8028bd791fc91af7b6702f1a34aa8a63ded3d7c1ba41d9dd415b621dcec0bb95`
- source parent head: `451f02a297c2122aae99d34a89835c920f303ad3`
- executed payload SHA-256 after receipt binding: `3c11ff0c6e7d4264633fc38481128ebad363a90aeafe35e4c7c8cd56c9aead1e`

## Next boundary

The next meaningful attack is not more ridge values in isolation. It is an **estimated-nuisance, predeclared null-surface calibration** in which:

1. the null point is estimated rather than oracle-known;
2. the covariance/error-process nuisance is estimated inside each resample;
3. the full-space and projected policies are both evaluated under their own explicitly stated estimands;
4. subspace selection is treated as part of the statistical procedure rather than a preprocessing footnote;
5. origin-local conditioning is retained as a pre-inference gate;
6. the resampling procedure reproduces rolling refit, training-only standardization, overlap, and the fixed gap.

Until that bridge is shown to behave prospectively across null directions and near-singular regimes, the formal path stays closed.
