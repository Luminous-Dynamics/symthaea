# Random-Projection Feature-Shift Scoring v0.1

**Status:** stacked draft research work on top of the sequential adequacy monitor. The implementation is committed; exact-head Rust execution is not yet verified. No runtime speedup, field performance, or state-of-the-art claim is established.

## Why this increment exists

The exact fixed-reference moment/correlation scorer in `feature_shift.rs` stores every pairwise correlation and recomputes all pairs for each audit. For (d) input features and (n) rows per batch, this is (O(nd^2)) scoring work and (O(d^2)) stored correlation values. This becomes a practical bottleneck for large sensor banks or learned feature vectors.

`random_projection_shift.rs` adds a fixed-seed Rademacher sketch with (k<d) projections. It computes the same three types of summary in a frozen (k)-dimensional space: mean changes, log-scale changes, and pairwise correlation changes. Its dominant scoring cost is (O(ndk + nk^2)); the stored reference state is (O(d+k^2)), excluding the audit batch and temporary projected rows. It recomputes projection signs from a deterministic mixer instead of retaining a dense (d\times k) matrix.

The first unit test uses 127 input features and 16 projected dimensions: the exact summary has 8,001 pairwise correlations while the sketch has 120. This is a complexity fixture, not a runtime benchmark. A diffuse mean-shift fixture checks that the compressed score responds to one declared alternative.

## Statistical contract

- The projection seed, count, feature order, scale floor, reference rows, batch protocol, and aggregate score are frozen before calibration.
- The fit reference is used to estimate original-feature means/scales and the projected reference profile. Independent clean calibration batches must then be scored against that frozen profile.
- Audit scores are calibrated only against those clean scores; the projected scorer returns a nonconformity statistic, never a p-value on its own.
- Each audit must be independent under the stated calibration contract. The public wrapper cannot verify provenance or independence on behalf of its caller.
- Feature shift is not synonymous with predictive error, harm, concept drift, or a false scientific law. A no-alarm result is not proof of adequacy.
- Random projection can lose information. Changes confined to weak or nearly orthogonal directions may be missed; a fixed seed is not a universal detector. Refit or change the projection only in a new epoch with fresh calibration.

This is an engineering baseline using a standard sketching idea, not a novel projection theorem or a new distribution-free guarantee.

## Confirmatory benchmark protocol

### Comparison set

Under a fixed calibration budget, batch size, target alpha, and audit budget, compare:

1. exact all-pairs moment/correlation scorer;
2. random-projection scorer at (k\in\{16,32,64,128\}), only where (k<d);
3. two-sample baselines such as MMD (including random Fourier feature approximation), energy distance, and a cross-validated classifier two-sample test;
4. deployment/sequential comparators such as WATCH and conditional conformal test martingales only where their assumptions and input protocol are correctly reproduced.

Do not compare only nominal alarm rates: a detector can achieve low false alarms by having low power. Use paired seeds, equal clean calibration volume, equal audit count, and held-out worlds. Tune (k), seeds, thresholds, and any method hyperparameters on development worlds only.

### Structural worlds

Include at least six distinct world families and keep confirmatory worlds unseen during tuning: diffuse mean shift; scale/heteroscedastic shift; sparse single-feature shift; low-rank covariance/correlation shift; nonlinear dependence with unchanged first and second moments; and benign covariate change with unchanged conditional predictions. Add out-of-support/extrapolation, missingness, subgroup shifts, abrupt changes, and gradual ramps before making deployment claims. Use real feature data with controlled perturbations as well as synthetic data.

### Metrics and provisional gates

Report empirical false-alarm rate with Wilson or paired-bootstrap intervals; power by alternative strength; detection delay and normalized detection time; missed-detection rate; repeated-test familywise false alarms over a declared horizon; runtime (p50/p95), peak memory, feature count, batch size, projection count, and bytes of frozen state. Publish all per-world results, not just aggregate wins.

Proposed engineering goals, **not current results**:
- for a clean null at nominal α=0.05, 10,000+ Monte Carlo streams per confirmatory null world and a confidence interval compatible with the intended bound;
- at least 80% power on a predeclared medium diffuse shift and 95% on a predeclared large shift at equal false-alarm budget;
- at least 5x p95 scoring speedup over the exact scorer for (d\geq1,024), (k\leq64), and batch size 64, without an unacceptable loss in held-out power;
- report a Pareto curve over (k), not a single cherry-picked projection dimension;
- preregister acceptable missed-detection rates for sparse and projection-orthogonal shifts; if those fail, document fallback to exact or alternative detectors rather than widening claims.

Targets should be revised only with a documented reason before confirmatory runs, never in response to hidden test results.

## Literature and benchmark rationale

- Cerqueira et al., KDD 2026, [A Framework for Evaluating and Benchmarking Concept Drift Detection Methods](https://doi.org/10.1145/3770855.3819070): emphasizes structural diversity, timing-aware metrics, and leave-one-dataset-out tuning.
- Prinster et al., ICML 2025, [WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales](https://proceedings.mlr.press/v267/prinster25a.html): a comparator for covariate shift and harmful conditional shift when its assumptions apply.
- Bhattacharyya and Ramdas, September 2026, [Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods](https://arxiv.org/abs/2609.27179): a current comparator for false-alarm control and detection delay.
- Liu, Prinster, and Orabona, October 2026, [Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales](https://arxiv.org/abs/2610.17343): a comparator for unknown change times.

These methods have different assumptions and guarantees. They are targets for correct reproduction, not boxes to check with superficially similar code.

## Industry baseline and what would count as better

Operational data-monitoring platforms already combine feature-wise statistical tests or distance metrics with dataset-level aggregation and visual investigation. For example, Evidently documents tests/distances selected by feature type and data volume, and a configurable rule over the share of drifting columns; NannyML documents univariate methods plus PCA reconstruction error for multivariate drift; Arize Phoenix exposes embedding drift trends using distance between reference and current vectors. These are useful baselines, not straw men:
- [Evidently drift algorithm](https://docs.evidentlyai.com/): column-wise tests/distances, reference datasets, and configurable dataset-level rules.
- [NannyML documentation](https://docs.nannyml.com/): univariate drift and PCA-based multivariate reconstruction error.
- [Arize Phoenix embedding analysis](https://arize.com/docs/phoenix/inferences/use-cases-inferences/embeddings-analysis): distance-based vector-drift investigation.

This Rust scorer is not yet a replacement for those complete workflows: it provides a small, typed, calibratable statistic and deterministic CPU implementation. The potential distinction is *auditable repeated monitoring with explicit false-alarm contracts plus a resource-bounded sketch*, rather than richer dashboards. To beat operational baselines, benchmark the full decision pathway: calibration validity, alert quality, detection delay, diagnosis, memory and p95 latency. Also measure the cost of delayed labels and follow-up investigations; a statistically sensitive alert that cannot distinguish benign covariate shift from harmful conditional change is not sufficient.

## Current evidence boundary

The implementation adds unit tests for (a) score compression from 8,001 to 120 correlation comparisons, (b) a diffuse mean-shift response, (c) malformed inputs, and (d) typed conformal assessment. At this writing, Rust tests and rustfmt have not been observed to pass on the exact head. The tests are committed; their outcomes remain unverified until the focused workflow reports success for that exact SHA.
