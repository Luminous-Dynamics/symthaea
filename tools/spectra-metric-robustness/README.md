# SPECTRA signal-detection metric robustness

A deterministic, synthetic-only stress harness for the applicability envelope of
signal-detection summaries. It extends (rather than duplicates) the shared
`symthaea-statistics` metrics and the equal-variance calibration work in
[#10131](https://github.com/Luminous-Dynamics/symthaea/pull/10131).

## Paradigm shift: qualify the assumptions, not just the formula

A metric can be implemented correctly and still be misapplied. The Gaussian
known-truth calibration covers the equal-variance IID reference family. This
harness asks whether diagnostics expose departures from that family before an
analyst interprets a scalar d-prime as if its model assumptions were known to
hold.

The benchmark has a strict separation:

- **Candidate view:** ordered synthetic observations, optional observed score
  and binary response, opaque participant/session identifiers, and an auxiliary
  channel. It contains no target labels, scenario name, seed, or injected
  sensitivity.
- **Evaluator sidecar:** target truth, injected sensitivity and missingness
  state. It is not passed to the candidate-facing analysis.
- **Report:** shared d-prime/criterion, shared ROC/AUC and average precision,
  transparent descriptive diagnostics, and a conservative interpretation
  disposition.

The type split prevents accidental API mixing; it is not cryptographic custody.

**Measurement boundary:** the candidate-visible `evidence_score` is a synthetic,
graded observer-output surrogate supplied by the benchmark generator. It is
not a latent brain signal and it is not a field that a person can be assumed to
report. Variance, tail-shape and score-autocorrelation diagnostics in this
benchmark are directly applicable only when a real task has a separately
validated graded observation channel. SPECTRA-OBS-001
[#10179](https://github.com/Luminous-Dynamics/symthaea/issues/10179) owns the
future confidence / response-time measurement profile. A binary yes/no-only
record does not identify the unequal-variance ROC shape from one operating
point; do not infer a participant's internal evidence distribution from
hit/miss counts alone.


## Frozen stress worlds

| World | Synthetic departure | Required diagnostic |
|---|---|---|
| GaussianReference | Equal-variance Gaussian, independent trials | No configured concern should be flagged at the reference seed |
| UnequalVariance | Signal SD differs from noise SD | Signal/noise variance ratio |
| HeavyTailed | Standardized Student-t(3) evidence | Skew / excess-kurtosis diagnostic |
| ContaminatedMixture | Rare high-variance contamination in both classes | Tail / kurtosis diagnostic |
| SerialDependence | Within-participant AR(1) residuals | Within-session lag-1 residual correlation |
| SensitivityDrift | Sensitivity changes across sessions | Early-versus-late d-prime contrast |
| CriterionDrift | Decision threshold changes across sessions | Early-versus-late response-criterion contrast |
| ParticipantHeterogeneity | Participant sensitivity varies | Distribution of participant-level d-prime estimates |
| MissingCompletelyAtRandom | Uniform 15% random dropout | Negative control: no class/session differential flag |
| MissingAtRandomBySession | Dropout depends on visible session covariate | Across-session missingness-rate range |
| OutcomeDependentMissingness | Positive signal responses are selectively missing | Missingness rates by known class and NotEvaluable gate |
| CovariateLeakage | Main evidence is null; auxiliary channel encodes target | Auxiliary-channel AUC |

The generator is a falsification fixture, not a biological or physical model.
The auxiliary leakage channel is an intentionally injected synthetic
assignment side-channel, not a claim about any real apparatus.

## Diagnostics and their limits

Configured *screening thresholds* are deliberately visible in source and are
not inferential tests:

- variance ratio outside [0.67, 1.50];
- absolute class-conditional skew above 0.75 or excess kurtosis above 2.0;
- absolute lag-1 residual correlation above 0.25;
- early/late d-prime or criterion shift above 0.35;
- between-participant d-prime SD above 0.40;
- absolute signal/noise missingness-rate gap above 0.10;
- across-session missingness-rate range above 0.15;
- auxiliary-channel AUC outside [0.30, 0.70].

These thresholds are benchmark configuration, not universal cut-offs. Their
false-flag and miss rates must be estimated over independent campaigns and
thresholds must be frozen before any real data are examined. A flag means
"inspect this assumption"; absence of a flag does **not** prove the assumption.
MCAR is a negative control for class-differential and session-varying flags.
MAR is injected through the visible session covariate and tested by the
across-session missingness-rate range. MNAR is injected through hidden
target/response state. These worlds test different dropout mechanisms; the
known-truth synthetic sidecar measures diagnostic detection limits, not whether
observed data identify a missingness mechanism.

The report never turns a flagged d-prime into an adjusted estimate. It returns
one of:

- `NotEvaluable`: too few observed trials in either class or excessive overall
  missingness;
- `AssumptionConcern`: at least one configured diagnostic fired;
- `NoConfiguredConcern`: no configured diagnostic fired (not proof of model
  validity).

The harness intentionally does not implement p-values, confidence intervals,
cluster-robust inference, a permutation engine, or sequential testing. Use the
existing assignment-matched inference and cluster/statistics owners only after
their assumptions and exact releases are qualified. Repeated trials are not
independent replicates; the campaign seed, not each trial row, is the unit for
sweep-level diagnostic rates.

## Run

From this directory:

    cargo test
    cargo run --release --bin run_sweep -- 256 > metric-robustness.csv

The optional argument is the number of independent campaign seeds per world.
Seed-level CSV is written to stdout. A second per-world campaign summary is
written to stderr and reports both per-diagnostic trigger counts/rates and the
aggregate any-flag rate. This is essential: nine diagnostics each having a low
individual false-flag rate does not imply the complete battery has the same
familywise false-alert rate. The summary reports raw counts and descriptive
rates only; it does not manufacture confidence intervals or p-values. The
generator uses deterministic SplitMix64 streams and Box-Muller normals for
replay only; it is not cryptographically unpredictable. Floating-point
transcendental functions mean bitwise replay is specified only within the
declared implementation/runtime.

## Benchmarks to claim improvement

At equal trial counts and compute budgets, compare:

1. shared equal-variance d-prime/criterion and ROC/PR as the conventional
   baseline;
2. simple threshold diagnostics in this harness;
3. a separately preregistered unequal-variance SDT fit, only after its estimator
   and confidence model have independent tests;
4. a design-based / assignment-matched baseline at the actual randomization
   unit;
5. a hierarchical observer/session model as a separately validated comparator.

Report, by world and across independent seeds: per-diagnostic false-flag rates
and the aggregate any-flag false-alert rate in the Gaussian reference;
detection rate for each injected violation; confusion matrix between world
family and diagnostic flags; uncertainty over campaign-level rates;
d-prime/criterion bias only where the target is meaningful; AUC/AP recovery;
missingness-mechanism detection and MCAR false-flag rate;
abstention and false-abstention rates; runtime and memory at equal data budgets.
Do not select thresholds on the same seeds used for the reported score.

Proposed qualification targets (not achieved):

- deterministic replay for identical code, design and seed;
- familywise false-alert rate no greater than 5% across the complete diagnostic
  battery in held-out Gaussian-reference campaigns, with intervals reported;
- per-diagnostic false-flag rates reported separately, not substituted for the
  familywise target;
- at least 90% detection for each deliberately injected, sufficiently strong
  violation family at its declared benchmark severity;
- no silent re-use of equal-variance d-prime interpretation under declared
  variance inequality;
- no "pass" claim from missing, unexecuted, or non-independent campaigns;
- 100% rejection of malformed, misaligned, or non-finite candidate/evaluator
  records;
- every diagnostic threshold and scenario generator versioned in the report.

The 5% / 90% targets are engineering goals, not established test properties.
They must be evaluated with enough independent campaign seeds and a frozen
analysis plan.

## Current validation boundary

Rust unit tests and a sweep are authored but have not been compiled or run in
this environment because Cargo/rustc are unavailable. Any independent reference
replay is disclosed separately from Rust execution and CI. No participant,
human-subject, hardware, enclosure, electrical, RF, acoustic, or biological
data are included. No ESP or physical mechanism claim is made.

## Research references

- 2026 unequal-variance SDT paper: *Correcting for unequal variance in signal
  detection models using response time*, iScience (11 February 2026),
  https://doi.org/10.1016/j.isci.2026.114998. The paper reports that conventional
  d-prime can overestimate detection performance under unequal-variance
  scenarios and compares RT-informed unequal-variance estimates with
  confidence-based estimates.
- National Academies, *The Prevention and Treatment of Missing Data in
  Clinical Trials*, chapter on sensitivity analysis:
  https://www.ncbi.nlm.nih.gov/books/NBK209895/. Missingness mechanisms cannot
  in general be selected by observed-data fit alone; sensitivity analysis across
  plausible mechanisms is required.
- Talts et al., *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration*, https://arxiv.org/abs/1804.06788. Simulation is useful for
  verifying inference pipelines, but the stress benchmark here is a
  frequentist/measurement-assumption exercise, not posterior SBC.
- Andrew Gelman, *Simulation-based calibration: Some challenges and directions
  for future research* (2021), https://statmodeling.stat.columbia.edu/2021/09/03/simulation-based-calibration-some-challenges-and-directions-for-future-research/.
