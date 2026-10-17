# SenseBench Blind-World Protocol v0.2

**Status:** implementation added to the `feat/symthaea-sense-observation-envelope` branch; Rust tests are authored but have not been executed in the current environment.

## Question being tested

Does active measurement selection improve identification and decision quality when the true outcome-generating process differs from the declared likelihood models, or does it merely optimize the assumptions used by its own scorer?

The benchmark is an outcome-generating simulation, not a physical-science claim. Its source is intentionally inspectable, so "held out" means excluded from policy-model construction—not cryptographically sealed.

## Design

The `blind-worlds` CLI suite compares:

- nominal expected information gain;
- nominal predictive variance;
- model-weighted expected information gain;
- worst-case robust information-gain ranking;
- a fixed design selected using development regimes only;
- uniform random selection;
- passive-only observation.

All policies receive the same three-hypothesis prior, declared candidate-model set, nominal posterior-update rule, and synthetic resource budget. Candidate likelihoods are used by selectors; hidden-world likelihoods independently generate each observed binary outcome. Policies do not receive the hidden world ID, regime label, truth table, or true hypothesis index.

Three hypotheses are represented: direct cause, shared-cause/confounder, and no direct causal effect. Six regime templates cover two development conditions and four held-out conditions: intervention artifact, sensor degradation, effect reversal, and a null negative-control regime. Each template generates four deterministic likelihood perturbations keyed by the root seed, regime index, and variant index, yielding 24 parameterized worlds (eight development worlds and 16 held-out worlds). The generator preserves each regime's defining control: null likelihoods remain equal across hypotheses, the intervention-artifact candidate remains uninformative, and effect reversal remains reversed. Each candidate has a declared synthetic cost. The fixed-design comparator is selected by mean expected information gain per cost over the eight development worlds; it does not inspect held-out worlds to choose its action.

World likelihood perturbations and outcome streams are both deterministic functions of the root seed. The same world/trial outcome stream is paired across policies, while action selection has a separate deterministic stream. Ground-truth hypotheses cycle across the three hypotheses so each world’s trial set remains approximately balanced. Each policy spends from the same total synthetic resource budget; it stops when it reaches the predeclared posterior threshold or no affordable candidate remains.

## Reported outcomes

Per world and policy, JSON reports:

- multiclass Brier score and its trial-level 95% normal-approximation interval;
- multiclass log loss and its trial-level 95% normal-approximation interval;
- prequential predictive log loss: the negative log probability of each observed outcome under the current posterior and declared model ensemble, scored before the posterior update;
- a per-policy model-adequacy cutoff set at the 99th empirical percentile of independent-seed simulations generated from the declared likelihood scenarios;
- model-adequacy alarm rate and Wilson interval, fraction of trials retained after the claim gate, and null-world causal-claim rate both before and after gating;
- top-hypothesis accuracy with a Wilson interval;
- threshold-reaching fraction with a Wilson interval;
- mean experiments and synthetic cost spent;
- experiments/cost to the threshold among trials that reach it;
- 10-bin expected calibration error;
- declared-model expected information gain versus evaluator-world expected information gain per selected action;
- per-selection acquisition regret versus the highest expected-information-gain-per-cost candidate available under the evaluator-world likelihoods;
- realized posterior entropy reduction per cost (can be negative);
- unsupported positive causal-claim rate when the true hypothesis is the no-direct-cause hypothesis, with a Wilson interval.

An individual realized entropy reduction may be negative. It is not clamped to make a policy look better. Acquisition regret is a local, myopic per-selection comparison against the best currently affordable candidate under the hidden world; it does not compare globally optimal multi-step experiment plans. The reported realized quantity measures entropy change under the benchmark's declared posterior update rule after outcomes sampled from the hidden-world distribution; it is not proof of true-world posterior calibration. The adequacy gate is a post-trial claim filter: when a trial's mean prequential predictive loss exceeds its policy-specific cutoff, the result is treated as a reason to abstain from a causal claim. The cutoff is calibrated on independent seeds using the declared two-model ensemble, never the held-out evaluation-world tables. It is a fixture-specific diagnostic, not a universal or formally guaranteed goodness-of-fit test. Report both suppressed false claims and retained-claim coverage; a gate that rejects everything is not considered successful.

## Reproduction

Run from the repository's Nix devshell or other supported Rust environment:

```sh
SENSEBENCH_CODE_REVISION=$(git rev-parse HEAD) \
  cargo run -p symthaea-sensebench -- --suite blind-worlds --trials 512 --budget 8 --seed 42

SENSEBENCH_CODE_REVISION=$(git rev-parse HEAD) \
  cargo test -p symthaea-sensebench
```

Use the exact commit SHA for every recorded report. Do not attribute a report to a parent commit or treat skipped workflow checks as successful validation.

## Claims this protocol does not support

- It does not prove causal discovery in real systems or validated physical measurement.
- The likelihood tables are synthetic and source-visible.
- Trial-level intervals quantify Monte Carlo uncertainty conditional on each parameterized world. The 24 worlds are variants of six hand-authored templates, not a random sample of all causal graph topologies; they do not establish broad generalization. The 99th-percentile adequacy cutoff is calibrated from finite synthetic simulations and must be re-qualified when the candidate library, policy, budget, posterior threshold, or generator changes.
- A model-weighted or worst-case policy may fail to outperform a competent fixed design. Preserve such a result.
- Deterministic seeds and report manifests improve replayability; they do not establish independent data custody or prevent source-level leakage.

## Model-adequacy / abstention gate

This version adds a prequential model-fit diagnostic. Before each Bayesian posterior update, compute the observed outcome's probability under the current hypothesis posterior, marginalized over the declared likelihood-model ensemble; accumulate the negative log probability across the trial. For each policy, calibrate a cutoff using a disjoint deterministic seed stream and outcomes generated from each declared model scenario. Use the empirical 99th percentile of these mean trial losses. The evaluation-world outcomes are never used to set the cutoff.

An alarm means **the observed trajectory is surprising under this declared ensemble**, not that the current hypotheses are certainly false. Conversely, no alarm does not prove model adequacy. The system should abstain from a causal claim on alarm, retain the evidence, and report the lost coverage. Future work should use sequentially valid tests/e-values and independent calibration data rather than treating this simple quantile rule as a formal guarantee.

## Next qualification gates

These are proposed acceptance gates, not measured capabilities:

1. **Correctness controls:** exact known-answer ranking for a perfectly discriminating 1-bit action versus a 0-bit null action; malformed model/cost/outcome inputs fail closed.
2. **Paired effectiveness:** at least 15% lower median synthetic resource cost to a predeclared identification threshold than both uniform random and development-selected fixed design on held-out regimes, with a paired 95% interval excluding zero improvement.
3. **Calibration:** report Brier, log loss and calibration error together; no success claim based on threshold-reaching rate alone. Report overconfidence separately from identification accuracy.
4. **Null discipline:** target zero positive causal claims above the declared confidence threshold on null controls. Any violation is a benchmark result to investigate, not a reason to weaken the test.
5. **Robustness:** compare worst-case regret and performance on the correctly specified control; quantify any conservatism cost.
6. **World-level inference:** the current four variants per template are parameter jitters, not independently generated causal graph topologies. Add a graph/measurement-family generator and aggregate uncertainty by resampling both worlds and trial seeds; retain per-world results so a good average cannot hide a catastrophic regime.
7. **Stronger custody:** for a serious prospective benchmark, split generator/scorer into separate processes, publish a protocol commitment before scoring, hash the challenge corpus, and protect test labels/tables outside policy-accessible artifacts.
8. **Real-stream bridge:** only after the synthetic suite is reproducible, replay an actual sensor stream with injected faults and compare to a trusted independent reference. Do not allow this suite to reconfigure or control physical hardware.

The separate [SCM graph recovery suite](SENSEBENCH_SCM_GRAPH_PROTOCOL.md) provides a graph-level complement with CPDAG-aware scoring and a factorized split across graph topology and mechanism regimes. Use it for graph-structure claims; this three-hypothesis likelihood-table suite remains complementary rather than a substitute. Its v0.3 design replays 28 graph draws under four mechanism regimes (112 worlds), reports topology-only, mechanism-only and joint shift separately, and clusters uncertainty by topology so repeated mechanisms on one graph are not treated as independent graph draws.

## Research basis

- NIST AI 800-3 distinguishes performance on a fixed benchmark from generalized performance and emphasizes explicit estimands and defensible uncertainty estimates: https://www.nist.gov/news-events/news/2026/02/new-report-expanding-ai-evaluation-toolbox-statistical-models
- Robust Bayesian active learning research analyzes how model misspecification can amplify generalization error and motivates testing more than nominal information gain: https://proceedings.mlr.press/v300/tang26d.html
- Bayesian adaptive experimental design can itself suffer active-learning bias under misspecification: https://arxiv.org/abs/2205.13698
- The Numenta Anomaly Benchmark is a later complementary stream-detection adapter target; its streaming score does not establish sensor calibration or causal validity: https://github.com/numenta/NAB
