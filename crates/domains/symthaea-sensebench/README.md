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
    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite scm-graph --trials 32 --budget 8 --seed 42

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

Reports include Brier score, log loss, top-hypothesis accuracy, calibration error, cost to a confidence threshold, declared-model versus evaluator-world expected information gain, myopic per-selection regret against the hidden-world best affordable action, realized entropy reduction, and the rate of high-confidence causal claims on no-direct-cause trials. Negative realized entropy reduction is preserved. The version 0.2 report also computes prequential predictive log loss under the declared model ensemble and calibrates a per-policy adequacy cutoff at the 99th empirical percentile using independent-seed simulations from declared model scenarios. Trials above the cutoff are flagged for claim abstention; the JSON reports both null-world claim suppression and the fraction of trials retained after this gate. The root seed controls both world-parameter perturbations and outcome streams. The regret metric is a local one-step comparator, not a globally optimal multi-step policy oracle. The adequacy cutoff is only a synthetic, policy/budget-specific diagnostic: an alarm means the trajectory is surprising under the declared ensemble, while a lack of alarm does not prove the model is correct. It is not a formally calibrated test for real scientific data. This is an outcome-generating synthetic benchmark, not an independent graph-topology generator; test-world definitions are source-visible, so held-out regimes are not cryptographically sealed and results are not evidence of physical causal discovery. See [the full protocol](../../../docs/research/SENSEBENCH_BLIND_WORLD_PROTOCOL.md).

## Generated structural causal graph benchmark

Run a smaller exploratory configuration (the structural evaluator scores 25 candidate DAGs per action, so begin with 32 trials per world):

    SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
      cargo run -p symthaea-sensebench -- --suite scm-graph --trials 32 --budget 8 --seed 42

This is a separate evaluator from `blind-worlds`. It enumerates all 25 labeled DAGs on three observed variables. Seven graph-family templates generate four graph variants each; every exact graph variant is then evaluated under all four separate mechanism regimes, yielding 28 sampled topologies and 112 graph × mechanism worlds. Separate seeded RNG streams generate topology and parameters; each graph-variant × mechanism-regime parameter vector is paired unchanged across all seven graph families to avoid family-specific coefficient signatures. Sixteen worlds are development-only (two development graph families × four graph variants × two positive-effect development mechanisms); 96 are held out because either topology family or mechanism regime is held out. It compares eleven policies: nominal expected information gain, predictive outcome entropy, model-weighted gain, worst-case robust selection, a fixed-design action, uniform random action selection, passive observation, an optimistic known-mechanism oracle, a structure-blind random graph guess, an asymptotic observational PC baseline, and an exact-conditional observational PC baseline.

Each measurement observes the full three-variable binary state. An intervention fixes one variable to zero or one; the outcome distribution is enumerated exactly from a binary structural causal model, and observations are sampled from the hidden world's graph and mechanism parameters. The policies do not receive the hidden graph or those hidden parameters. The Bayesian acquisition policies share a nominal posterior update so the comparison primarily concerns acquisition choice. Both PC baselines use passive observations and submit CPDAGs while abstaining from exact full-DAG claims. The asymptotic variant uses 5% chi-square cutoffs; the exact-conditional variant enumerates the G-squared null distribution conditional on observed binary-table margins. The latter is a finite-sample reference for the tiny binary setting, not a solution for arbitrary conditional-independence testing. The known-mechanism oracle is an explicitly optimistic diagnostic, not a deployable policy.

The report scores graph recovery—not just confidence—with skeleton F1, directed-edge F1, structural Hamming distance (one edit for an added/removed adjacency or reversed direction), exact graph recovery, posterior mass on the true graph, graph-posterior Brier score, and measurement/intervention cost to threshold. It also emits paired policy deltas versus nominal EIG, development-fixed design, uniform-random action selection, and both asymptotic and exact-conditional PC baselines. Deltas compare policies on the same world/trial configuration, report topology-clustered intervals, and separate directed-edge F1, CPDAG skeleton F1, SHD, measurement/intervention count and false graph-claim rate by topology-only, mechanism-only, joint and any-held-out shifts. These are paired synthetic estimates, not a measured win: adopt an acquisition policy only if the preregistered paired interval supports improvement over the strong fixed baseline without worsening claim risk or hiding a loss of coverage. Schema v6 reports 112 worlds (28 topology draws crossed with four mechanism regimes), a factorized 16-development / 96-held-out split, and separate topology-only, mechanism-only and joint-shift aggregates. Held-out confidence intervals average the paired mechanism regimes within each exact topology first, then calculate uncertainty across topology instances; the same graph under four mechanisms is not counted as four independent topologies. It also groups the DAG posterior into the 11 Markov-equivalence classes for three variables, emits the highest-posterior-mass CPDAG, and reports CPDAG skeleton F1, compelled-direction F1, edge-mark distance, equivalence-class recovery, posterior mass on the true class, and separate full-DAG versus CPDAG claim coverage/false-claim rates. This avoids penalizing unresolved arrows that observational data cannot identify and avoids producing a potentially invalid PDAG by thresholding edges independently. It retains per-world summaries and aggregates held-out performance with worlds as units. For both PC baselines, v6 also reports the mean number of edge marks changed by projection onto a valid CPDAG and the fraction of trials where any mark changes; this makes the post-inference repair step visible rather than silently equating projection validity with statistical validity.

**Important limits:** three variables only; all three-variable DAGs are in the candidate catalogue; graph families/mechanisms remain synthetic and source-visible. Only two topology families and two positive-effect mechanism regimes are development-only. The paired factorial split leaves 16 development worlds and 96 held-out worlds across 28 sampled topologies. Crossing topology with mechanism avoids a direct one-to-one family-to-coefficient shortcut, but seven hand-authored graph families, four graph draws per family and four mechanism regimes are not a broad random sample of real causal systems; the 112 worlds contain only 28 unique topology draws reused across mechanism regimes. The oracle is given the hidden mechanism parameters. The aggregate intervals describe this fixed parameterized set and do not by themselves establish generalization to real-world graph families. This suite is an executable substrate for the next benchmark tranche, not evidence of scientific discovery. See [the protocol](../../../docs/research/SENSEBENCH_SCM_GRAPH_PROTOCOL.md).

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
