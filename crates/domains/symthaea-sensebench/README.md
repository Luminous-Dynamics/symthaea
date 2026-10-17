# Symthaea SenseBench

SenseBench is a deterministic, hardware-free starter harness for the Symthaea Sense research stream. It includes a transparent anomaly baseline and a model-agnostic, cost-aware Bayesian experiment-ranking primitive.

## Run

    cargo test -p symthaea-sensebench
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite anomaly --scenario all --seed 42 --samples 512
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite anomaly --scenario null --seed 2026 --samples 4096
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite causal --trials 512 --budget 8 --seed 42
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite robustness
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite blind-worlds --trials 512 --budget 8 --seed 42

The CLI requires `SENSEBENCH_CODE_REVISION` and emits a JSON report carrying the suite, generator version, exact revision string, full scenario manifest or causal-run configuration, seed, sample count/trials, and model-specific metrics. Run from a clean worktree so the Git commit identifies the exact code. `--allow-unversioned` permits exploratory output with a missing revision and must not be used for benchmark claims. Identical generator version, scenario, seed, sample count and interval produce identical observations. The report includes point metrics, average precision, event recall, missing sample count, contiguous false-alarm episodes per sensor-hour, and detection delay. Point-level false-positive sample counts remain available separately.

## Controlled causal-selection benchmark

Run:

    cargo run -p symthaea-sensebench -- --suite causal --trials 512 --budget 8 --seed 42

The synthetic evaluator compares three strategies on balanced (even-count) trials between two candidate explanations: a direct-cause model and a shared-cause model. Outcome generation uses common random numbers across policies to reduce noise in comparisons; random action selection uses a separate random stream. A passive observation and negative control have identical outcome likelihoods under both hypotheses; an intervention has different likelihoods. Policies are active expected-information-gain selection, uniform random selection, and passive-only selection. The policy sees priors and declared likelihoods, not the evaluator's hidden true-hypothesis index.

JSON output reports mean posterior probability assigned to the true hypothesis and a 95% normal-approximation interval across trials, multiclass Brier score and interval, top-hypothesis accuracy with a Wilson interval, threshold-reaching fraction with a Wilson interval, experiment count among successful threshold-reaching trials, and informative experiments used. Intervals quantify Monte Carlo variation conditional on this fixed synthetic model; they do not cover model misspecification or real-world generalization.

This is intentionally a toy model with correctly specified likelihoods. It demonstrates the expected direction of results in a controlled environment; it does not show real-world causal-discovery capability, robust performance under model misspecification, or general superiority over a strong fixed experimental design. The next issue adds blinded causal graph families, model-misspecification conditions, uncertainty intervals across seeds, and stronger baselines.

## Model-misspecification regret fixture

Run:

    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite robustness

This deterministic fixture compares nominal single-model expected gain, nominal predictive variance, model-weighted expected gain, worst-case robust ranking, one fixed comparator, and the exact expected score of uniform random selection. Selector policies receive only two declared likelihood-model scenarios; a separate evaluator scores their choices against two corresponding toy worlds and one held-out likelihood world. Reports include the oracle candidate, expected information gain under evaluator-only hidden-world likelihoods per cost, regret, mean/worst-case regret, and maximum-regret reduction relative to the nominal selector.

The fixture intentionally includes a strong fixed-design control. In this small constructed example, the fixed comparator matches the robust policy's worst-case regret; this is not evidence that active selection beats a good fixed design. The held-out world is separated in code, not in a cryptographic sandbox. All probabilities are hand-authored, synthetic mathematical inputs, not empirically calibrated instrument likelihoods. This suite tests that the evaluator can expose model brittleness and compare decision regret; it does not establish generalization to real experiments.

## Outcome-generating blind-world benchmark

Run:

    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite blind-worlds --trials 512 --budget 8 --seed 42

This simulation compares nominal expected information gain, predictive variance, model-weighted gain, worst-case robust ranking, a fixed design selected from development regimes, uniform random selection, and passive-only observation. Outcomes are sampled from evaluator-only likelihood tables across six regime templates: two development regimes and four held-out regimes covering intervention artifacts, sensor degradation, effect reversal, and a null negative control. Four deterministic, seed-keyed likelihood variants per template produce 24 parameterized worlds (eight development, 16 held out). All policies share the same nominal Bayesian update rule and synthetic resource budget.

Reports include Brier score, log loss, top-hypothesis accuracy, calibration error, cost to a confidence threshold, declared-model versus evaluator-world expected information gain, myopic per-selection regret against the hidden-world best affordable action, realized entropy reduction, and the rate of high-confidence causal claims on no-direct-cause trials. Negative realized entropy reduction is preserved. The root seed controls both world-parameter perturbations and outcome streams. The regret metric is a local one-step comparator, not a globally optimal multi-step policy oracle. This is an outcome-generating synthetic benchmark, not an independent graph-topology generator; test-world definitions are source-visible, so held-out regimes are not cryptographically sealed and results are not evidence of physical causal discovery. See [the full protocol](../../../docs/research/SENSEBENCH_BLIND_WORLD_PROTOCOL.md).

## Domain-specific hypothesis fixture

`geophysics` includes a synthetic Earth-interior model-discrimination fixture with two abstract hypotheses and three candidate measurements. Its likelihoods and costs are hand-authored solely to exercise the shared ranking API; they are not derived from geophysical data, calibrated forward models, or evidence about Earth's core. Treat the candidate ordering as a math/test fixture only.

## Active experiment ranking

The Rust library exposes `rank_experiments` and `rank_experiments_robust` in `experiment_design`. The first accepts hypothesis priors and candidate-specific outcome likelihoods, computes expected entropy reduction in bits, and ranks by information gain per declared cost. The robust variant accepts a weighted set of plausible likelihood-model scenarios. Candidate IDs, costs, and outcome-index semantics must agree across scenarios; the ranking prioritizes worst-case expected information gain per cost and also reports weighted expected gain and model-sensitivity standard deviation.

This remains a ranking primitive, not a hypothesis generator. Robustness is limited to the supplied scenario set; it does not guarantee robustness to omitted models. Poor or overconfident likelihood inputs can still lead to poor rankings. The caller must source and calibrate likelihoods, record their provenance, and respect safety constraints before acting. The selector does not yet connect to Symthaea's HDC/LTC reasoning loop or control real instruments.

## Scope and limits

This first slice contains a simple fixed-initial-window z-score baseline, not a novel detector. The scenarios are synthetic: null, step change, transient spike, gradual bias drift, and dropout. Dropout is handled as missingness/integrity, never coerced into a zero measurement or numeric anomaly score.

The evaluator retains labels and seed in the scenario manifest; a real blind evaluation must keep that evaluator object away from the detector. This baseline runner enforces separation at the function boundary but is not a cryptographically isolated evaluator.

Synthetic test success validates software mechanics only. It does not show that Symthaea can detect unknown phenomena in real instruments. Future increments should add clock faults, stuck-at and saturation quality signals, multichannel confounding scenarios, random/fixed/information-gain sampling comparisons, uncertainty calibration, and external real-world datasets such as the Numenta Anomaly Benchmark. NAB is a complementary public streaming benchmark; it does not replace domain-specific metrology and causal controls.


## Earth-interior toy fixture

The optional `geophysics` module exercises the shared ranking primitive with two abstract competing explanations and three hand-authored likelihood tables. The expected-gain ranking and robust multi-model ranking have closed-form reference expectations. The robust fixture deliberately makes the targeted magnetic candidate perfect under one toy model and uninformative under another, so the robust selector should prefer the consistently informative coarse comparator. This is deliberately a software/reference fixture, not a physical model, calibrated geophysical likelihood, measured observation, or claim about Earth's core. A qualified forward model and its evidence manifest must replace these toy likelihoods before they can inform scientific decisions.
