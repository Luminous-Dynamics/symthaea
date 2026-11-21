# RCAF synthetic null and injection–recovery calibration V0

**Status:** Implemented synthetic engineering benchmark; it does not use biological recordings.  
**Date:** 2026-11-21

## 1. Purpose

A similarity metric should be tested against known ground truth before it is used to interpret empirical connectomes. This protocol tests two complementary properties:

1. **Null calibration:** when two condition-specific segments are generated from the same underlying network model, does the participant-level paired-gap test reject too often?
2. **Injection recovery:** when a known network perturbation is introduced, does the metric detect it, and how does recovery depend on its size and measurement noise?

The generator uses a Gaussian two-factor model to create positive-semidefinite covariance structures, samples finite time series, estimates sample correlation matrices, and evaluates the existing RCAF absolute Weighted-Jaccard metric. This provides a reproducible code-path calibration, not a full neurovascular, EEG/MEG sensor, source-localization or preprocessing simulator.

## 2. Generative model

For participant (p), region (r), and time point (t):

[
X_{prt}=a_{pr}G_{pt}+b_{pr}M_{pt}
+sqrt{1-a_{pr}^2-b_{pr}^2},epsilon_{prt},
]

where (G), (M), and (epsilon) are independent standard-normal variables. The loading vectors vary slightly by participant.

The injected state changes the module loading (b_{pr}) for a predefined contiguous module of regions:

[
b^{(1)}_{pr}=b^{(0)}_{pr}+delta
]

for module regions, while loadings outside the module remain unchanged. The residual variance is adjusted so each region retains unit marginal variance. The covariance is therefore positive semidefinite by construction.

Four independent segments are sampled per participant:
- repeat A: baseline;
- repeat B: baseline;
- null C: baseline;
- injected C: perturbed baseline.

For the null track, compare A–B within-state similarity against A–C null similarity. For the injection track, compare A–B against A–C injected similarity. Both tracks therefore share the same participant-level repeatability reference.

## 3. Estimand and test

For each participant:

[
g_p=S(A_p,B_p)-S(A_p,C_p).
]

Positive (g_p) means the injected/null comparison is less similar to the baseline repeat than the within-state repeat is.

The replicate-level inferential test is the existing two-sided exact participant sign test on (g_p). Participants are independently generated within each Monte-Carlo replicate; repeated segments are never treated as extra participants.

- Null false-positive rate: fraction of Monte-Carlo replicates with exact sign-test (p<alpha).
- Injection detection power: fraction of injection replicates with exact sign-test (p<alpha).
- Both rates include 95% Wilson Monte-Carlo confidence intervals.
- Median paired gap and positive-gap fraction are summarized across replicates.

This calibrates the RCAF metric plus the specific paired sign-test procedure under the declared synthetic generator. It does not estimate statistical power for an actual dataset or any biological effect.

## 4. Initial engineering fixture

Default fixture:
- 24 independent participants per replicate;
- 12 regions, 160 time points per segment;
- module size 5;
- global factor loading 0.20;
- module loading 0.18;
- participant loading jitter 0.06;
- injected module-loading increment 0.40;
- 100 Monte-Carlo replicates;
- two-sided (alpha=0.05);
- fixed seed recorded in output.

The unit tests use 80 replicates and require:
- null false-positive estimate no greater than 0.15;
- absolute median null gap below 0.025;
- injection detection power at least 0.80;
- injected median gap at least 0.035 above the null median gap.

These are deliberately broad smoke-test bounds, not performance claims or acceptance thresholds for neuroscience studies. The Wilson intervals must be inspected; 80–100 Monte-Carlo replicates are insufficient to characterize a 5% tail precisely.

## 5. Required expansion before empirical interpretation

The initial implementation now includes both a fixed default calibration point and a five-point common-random-number sweep over factor-loading increments [0.0, 0.10, 0.20, 0.30, 0.40]. The paired sweep is designed to isolate effect-size changes from Monte-Carlo seed changes. The next benchmark surface should vary:
- injection size (delta), including zero, weak, moderate and strong effects;
- participants per replicate;
- samples per segment;
- number and size of altered modules;
- global/module loading, sign structure and heterogeneous participant response;
- missing edges and ROI-order errors;
- motion-like structured nuisance and global-signal-like shared components;
- temporal autocorrelation and nonstationary segments;
- estimator choice: Pearson, Spearman, shrinkage covariance/precision, pyspi candidates, RCAF signed and unsigned variants;
- neutral-edge threshold and minimum common-edge floor.

Report full type-I and power curves with Monte-Carlo intervals. The null must be tested across the full nuisance grid, not just one convenient seed or one Gaussian model. Calibration parameters must be locked before comparing estimators.

## 6. Proposed benchmark targets

Engineering targets—not achieved claims:
- Null false-positive rate compatible with nominal alpha across predeclared nuisance scenarios; report uncertainty and do not accept only a single point estimate.
- Injection-recovery curves monotonic in the controlled effect over a useful region.
- Recovery stability across participant count and segment duration.
- Correct rejection of invalid ROI order and insufficient common-edge support.
- Explicit separation of network-change sensitivity from measurement-quality degradation.
- Reproducibility from fixed config, code revision, seed and output digest.

A method that is highly sensitive to injections but fails null calibration is not acceptable. A method that controls the null but cannot recover plausible injected changes is also not useful. Reliability cannot be optimized at the expense of validity.

## 7. Dose-response sweep

The Rust API `run_injection_recovery_sweep` and example `rcaf_injection_recovery_sweep` run a strictly increasing grid of non-negative loading increments. The same PRNG seed and draw order are reused at every increment, so the baseline loadings and random draws are paired across the curve. The first point at zero is the null-dose control.

The sweep reports each point's null false-positive estimate, injection-detection power, Wilson intervals and median paired gap. It also reports the fraction of adjacent effect-size increments for which the median paired gap increases. A non-monotonic response is retained and reported; it must not be hidden by selecting a favourable subset of points.

## 8. Run

```sh
cargo test --manifest-path research/rcaf/Cargo.toml
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_injection_recovery
```

The example emits JSON conforming to `injection-recovery.schema.json`.

## 9. Claim ceiling

This benchmark proves only that a particular software implementation behaves in a known synthetic world under stated assumptions. It does not prove:
- the factor model represents the brain, BOLD, EEG or MEG;
- the selected similarity measure is optimal;
- empirical data are free of artifacts or confounds;
- a real participant has changed conscious state;
- RCAF is clinically validated or superior to existing neuroscience estimators.


## 10. Temporal-dependence stress check

The Rust test suite now runs an additional exchangeable-null calibration with AR(1) coefficient `rho=0.65`. It requires the Monte-Carlo null rejection estimate to remain no greater than 0.15 and the absolute median paired null gap to remain below 0.04. This is a second synthetic operating point, not proof of general null calibration. A full autocorrelation grid with Monte-Carlo intervals remains a required benchmark before applying this metric to real time series.


## 11. Estimator semantics and matched-input comparison

The simulation harness exposes a matched-input estimator tournament:
1. **Absolute Weighted Jaccard:** compares absolute edge magnitudes, deliberately discarding sign.
2. **Signed-shift Weighted Jaccard:** shifts every correlation from [-1,1] into [0,2]; sign disagreement lowers similarity.
3. **Normalized mean absolute edge error:** (1-\operatorname{mean}(|r_i-s_i|)/2) over the upper triangle.

All three consume the same simulated matrices using the same random seed. A known sign-flip fixture returns absolute WJ=1, signed-shift WJ=1/3, and normalized mean-absolute similarity=0.5. This demonstrates that the estimators are not interchangeable; any "winner" must first be tied to a declared construct and then evaluated for null calibration and injection sensitivity.

Run `cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_estimator_tournament`. The output schema is `research/rcaf/estimator-tournament.schema.json`.


## 12. Execution receipts and archived outputs

The focused research workflow records all three simulation JSON outputs and creates a content-addressed execution receipt:
- exact Git commit;
- Rust compiler, Cargo and Python versions;
- hash of every tracked file under the RCAF research package;
- file digests and sizes for the single-point, dose-response and estimator-tournament reports.

The artifact name includes the exact checked-out commit SHA. This provides an independently inspectable CI record; it does not prove independent replay, validate a biological pipeline or upgrade a synthetic result to empirical evidence.


## 13. Estimator-by-effect-size response surface

The next implementation extends the single-metric dose-response curve into a matched-input estimator × dose matrix. It evaluates:
- absolute Weighted Jaccard;
- signed-shift Weighted Jaccard;
- normalized mean absolute edge error;

at the same predeclared dose grid. Every estimator uses the same seed and the same random-draw order at every dose. The JSON output records a full sweep and Monte-Carlo uncertainty for each estimator, not a single saturated point. The executable is `rcaf_estimator_dose_response_tournament` and its contract is `estimator-dose-response-tournament.schema.json`.

This avoids treating one strong injection as a universal estimator test. Compare weak/moderate sensitivity only after confirming null calibration and construct semantics. Do not average the different estimands or select a universal winner from this synthetic grid.
