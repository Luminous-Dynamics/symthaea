# SPECTRA event-detection metric calibration

A deterministic synthetic harness for qualifying the *metrics* used by the
SPECTRA-CHARGE-001 event-detection task. It is not a simulator of an
electromagnetic enclosure, an observer's brain, or a physical information path.

## Why this exists

Signal-detection metrics are measurement instruments. Before relying on them
for a scientific conclusion, test whether they recover known sensitivity and
response criterion under controlled conditions.

This harness composes the shared `symthaea-statistics` API. It does not add a
second d-prime, ROC, precision-recall, randomization-inference, or sequential
testing implementation.

## Mathematical reference world

The synthetic latent evidence is generated from:

- noise: (X_0 \sim N(-d'/2, 1));
- signal: (X_1 \sim N(+d'/2, 1));
- response: signal when (X \ge c).

The equal-variance model gives analytic expectations:

- hit probability: standard-normal CDF at `d-prime / 2 - c`.
- false-alarm probability: standard-normal CDF at `-d-prime / 2 - c`.
- ROC AUC: (Phi(d'/\sqrt{2})).

The model allows sensitivity (d') and response criterion (c) to vary
independently. It is a deliberately simple reference family; its assumptions
are not imposed on real-world observations.

## What the harness measures

Each generated campaign produces separate artifacts:

- candidate view: opaque trial ordinal, graded evidence score, and binary call;
- evaluator sidecar: hidden signal/noise labels, seed, and injected model values.

The sidecar must not be passed to the candidate. The type split is a software
boundary, not cryptographic custody: a process receiving both objects or
inspecting the source can recover the truth.

The evaluator reports:

- raw hits, misses, false alarms, and correct rejections;
- shared d-prime and response-criterion estimates;
- tie-correct ROC/AUC and non-interpolated average precision from continuous
  scores;
- prevalence and the random-ranking average-precision baseline;
- accuracy and balanced accuracy as diagnostic comparisons;
- analytic hit/false-alarm, AUC, accuracy, and balanced-accuracy expectations.

ROC/PR are computed from the graded score, never reconstructed from the single
binary call. An evidence score is not a calibrated probability; Brier score,
log loss, and probability calibration are intentionally absent.

## Reproducible sweep

From this directory:

    cargo test
    cargo run --release --bin run_sweep -- 256 > metric-calibration.csv

The optional integer argument is the number of independent campaign seeds per
cell (default 256). The grid contains d-prime values
`[0, 0.25, 0.5, 1, 1.5, 2]`, criterion values `[-1, 0, 1]`, and balanced and
5%-prevalence tracks. Every cell reuses the same seed set; independent signal,
noise, and shuffle streams provide common random numbers across the sensitivity
and criterion comparisons.

The CSV contains one row per seed and cell. The campaign/seed is the replicate
for metric-calibration summaries. Do not pool trial windows and pretend they
are independent replicates.

## Qualification properties

Tests cover:

1. exact replay under an identical model/design/seed;
2. criterion shifts changing binary calls and criterion estimates without
   changing continuous-score ranking or AUC;
3. monotone AUC and recovered d-prime across the sensitivity ladder;
4. agreement with analytic Gaussian operating characteristics;
5. prevalence-aware average-precision baselines in the rare-event track;
6. criterion/prevalence sensitivity of accuracy;
7. candidate/evaluator identity mismatch and non-finite-score rejection;
8. invalid parameter, empty-class, and reversed-ranking controls.

## Interpreting calibration

The dynamic-range-fraction idea from Miller et al. (Nature Biotechnology,
1 October 2026, DOI 10.1038/s41587-026-03307-w) is useful when a metric has
meaningful positive/negative controls and a finite ideal range. Do **not** apply
that formula mechanically to d-prime: its perfect-discrimination ceiling is
infinite. For d-prime, evaluate recovery bias/RMSE, calibration slope/intercept,
monotonicity, and criterion-invariance. For bounded AUC, compare negative
controls, known-signal positive controls, and the analytic AUC reference.

## Limitations and claim ceiling

- Equal-variance Gaussianity is the generator assumption, not a finding about
  any participant or historical event.
- The implementation qualifies software behavior only under this synthetic
  family. It does not qualify a human-study statistical model.
- The PRNG is deterministic and intended for replay, not cryptographic
  unpredictability or hidden target custody.
- Box-Muller uses floating-point transcendental functions. Replay is specified
  for the declared implementation/runtime; cross-platform bitwise identity is
  not claimed.
- No human-subject, physical, electrical, RF, acoustic, or enclosure data are
  generated or consumed.
- No historical 1952 hit totals are used as priors, calibration controls, or
  acceptance thresholds.
- The harness does not estimate type-I error for an inferential procedure,
  provide cluster-robust uncertainty, or authorize causal claims.

Real event-detection data may violate independence, equal variance, stationarity,
or the Gaussian evidence model. Such a mismatch must trigger model criticism
or a not-evaluable disposition rather than automatic acceptance of d-prime.

## Research basis

- Miller et al. (2026), *Deep learning perturbation models can outperform
  baselines on calibrated metrics*, Nature Biotechnology, published
  1 October 2026: https://doi.org/10.1038/s41587-026-03307-w
- Alexeev & Morton (2026), *Cluster Trials Inference With CARE*, Statistics in
  Medicine, published 4 June 2026: https://doi.org/10.1002/sim.70610
- Banzi et al. (2026), *An international consensus on core reproducibility
  items in research*, PLOS Biology, published 16 April 2026:
  https://doi.org/10.1371/journal.pbio.3003726

## Validation status

Rust tests and the sweep are authored but have **not** been compiled or run in
this environment. The host used for this authoring session has no Cargo or
rustc. No exact-head CI result, numerical qualification pass, or performance
claim is made.
