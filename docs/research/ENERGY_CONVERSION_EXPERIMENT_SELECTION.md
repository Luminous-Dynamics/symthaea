# Energy-Conversion Experiment Selection Protocol

**Status:** research protocol; a deterministic Phase A ranking baseline is implemented in `symthaea-energy-bench`; expected-information-gain and active-learning methods are not implemented  
**Related PR:** [Symthaea energy-conversion discovery #7480](https://github.com/Luminous-Dynamics/symthaea/pull/7480)  
**Predecessors:** [Benchmark contract](ENERGY_CONVERSION_BENCHMARK_CONTRACT.md), [discovery plan](ENERGY_CONVERSION_DISCOVERY.md)

## The paradigm shift

A conventional design loop asks, “Which candidate has the best predicted score?” A more valuable engineering research loop asks, “Which safe, affordable experiment is most likely to tell us which design decision is correct?”

The second question can change the order of work. A cheap test that distinguishes two competing loss mechanisms can be more valuable than running another expensive simulation of a candidate that is already unlikely to be built. Experiment selection must therefore be treated as a registered optimization problem in its own right.

Research supports multi-fidelity information-gain estimation for costly physics models. Coons and Huan's multifidelity expected-information-gain study uses approximate control variates to reduce the cost of estimating information gain in Bayesian optimal experimental design. The 2026 survey of multi-fidelity optimization also discusses how low- and high-fidelity model evaluations can be combined under computational budgets:
- https://doi.org/10.1137/25M1731812
- https://doi.org/10.1145/3801959

These methods are research inputs, not evidence that Symthaea already implements Bayesian optimization or expected information gain.

## Experiment proposal record

Every proposed simulation, measurement, or prototype experiment should register:

- **Experiment identity:** immutable proposal ID, parent benchmark/policy digest, code/model revision, experiment owner, and proposal timestamp.
- **Decision question:** which decision could change because of the result; competing hypotheses or candidate families; and the registered decision margin.
- **Predictions:** predicted observable quantities under each hypothesis, their uncertainty/distributions, and a declaration of shared nuisance parameters such as sensor calibration, ambient conditions, or material-property estimates.
- **Discrimination:** which result patterns distinguish the hypotheses, including outcomes that would falsify all current candidate models.
- **Cost and risk:** estimated compute/rig time, direct cost, turnaround, safety category, operating envelope, and independent review requirements.
- **Fidelity:** analytical / reduced-order / calibrated simulation / high-fidelity simulation / bench / integrated-system / independent replication, with the evidence tier stated before execution.
- **Decision rule:** stop, continue, reject, recalibrate, or promote-to-higher-fidelity conditions decided before observing the result.
- **Artifacts:** complete input hash, random seed where applicable, raw output location/hash, solver/runtime identity, and replay instructions.

A proposal lacking a measurable observable, falsifiable competing hypotheses, or a preregistered decision rule is not yet an executable experiment.

## Selection algorithm: implement in increasing order of sophistication

### Phase A — transparent baseline

Start with a deterministic ranked queue rather than immediately claiming Bayesian optimality. Enforce safety and hard constraints first. Remove experiments with missing measurements, invalid boundaries, insufficient provenance, or no decision relevance. Rank the remainder by a published tuple:

1. Expected decision relevance: can the outcome change feasibility, the selected design, or the next fidelity tier?
2. Expected uncertainty reduction in a named quantity or uncertainty reduction around a specific constraint.
3. Information gained per unit of compute time, rig time, or cost, with each quantity reported separately so weighting remains visible.
4. Reproducibility and external validity: can an independent reviewer repeat or cross-check the result?
5. Risk/operational burden as a hard gate or explicitly registered penalty, never silently traded against a safety limit.

Do not collapse the tuple to a single score unless the benchmark policy states the weights and includes sensitivity analysis. Keep a simple, reproducible random/space-filling design as the comparison baseline.

#### Phase A implementation status

The `symthaea-energy-bench::experiment_selection::rank_experiments` API implements the transparent baseline described above. Its queue policy must be marked as preregistered before queue generation. Each proposal must name its decision question and observable, include at least two distinct competing hypotheses, declare a fractional uncertainty-reduction estimate and positive cost in the policy's exact registered unit, and have a prior registration, approved safety review, registered operating envelope and abort criteria, and complete provenance plan. Invalid proposals are retained in a rejection list with explicit reasons; duplicate IDs and incomparable cost units are rejected. Eligible items receive deterministic lexicographic ordering with a stable ID tie-break.

This Phase A score is a **heuristic screening statistic**, not expected information gain. Its uncertainty-reduction estimate is supplied by the proposer; it is not independently calibrated by this module. The implementation does not choose physical operating conditions, perform simulations, authorize experiments, infer risk, or prove that a registered statement was actually committed before observation. These limits must remain visible in results and documentation. Unit tests have been authored but require exact-head CI or a local Rust toolchain to execute.

## Phase B — model-based expected information gain

Once predictive distributions have been calibrated, implement expected information gain or expected value of sample information. Use candidate experiments to predict observation distributions under competing model parameters or design hypotheses; select experiments expected to reduce uncertainty about quantities that can change the decision.

- Account for common-mode calibration, ambient, and solver discrepancy terms. Do not assume that candidate and baseline uncertainty is independent when shared sources exist.
- Compare low-fidelity estimators to a high-fidelity reference subset, measuring estimator bias/variance and total runtime, not just optimizer score.
- Record every model evaluation, failed run, rejected hypothesis, and all samples/seeds. Do not report only the best candidate or best experiment.
- Allocate high-fidelity simulations or physical tests only when their expected decision value justifies their cost and risk.
- Maintain a held-out validation set that candidate generation and surrogate calibration cannot inspect.

### Phase C — independent falsification and real-world escalation

A preferred experiment must try to distinguish the favored design from its best alternative and attempt to falsify the mechanism behind the predicted improvement. If all compared models predict the same result, the experiment may have low decision value even when the predicted response is large.

Physical experiments require an approved protocol, trained operators, calibrated instrumentation, safety interlocks, abort criteria, raw-data retention, and independent review. Model-based experiment ranking must never bypass ordinary engineering controls.

## Known-answer harness

The `symthaea-energy-bench::experiment_benchmark` module provides a deterministic oracle-backed testbed comparing Phase A, seeded random, cheapest-first, one-dimensional space-filling, and an exhaustive oracle upper bound under the same per-scenario resource budget and admission gates. It generates deceptive-proxy scenarios by construction and reports utility, regret, resolution, costs, selected IDs, and rejection records. Heuristic strategies stop once the preregistered synthetic resolution threshold is reached; exact subset enumeration also estimates the minimum cost to reach that threshold. A 1,000-scenario regression test and a runnable example are authored. Run `cargo run -p symthaea-energy-bench --example selector_benchmark > energy-selector-benchmark.json` for a JSON receipt comparing Phase A, cheapest-first, space-filling, the oracle upper bound, and 30 seeded random baselines. The JSON receipt embeds all inputs and includes policy/scenario/suite/output SHA-256 digests; `parse_and_verify_benchmark_report_json` checks the digests and replays each strategy. Store the exact code revision separately because input digests do not identify executable code. False-feasible and false-superiority candidate claims are not yet modeled by this harness and require a separate decision-oracle test suite.

Its oracle is intentionally narrow: independent per-experiment probabilities combine as `1 - product(1 - gain)`. It tests selector behavior against a known answer under a declared toy model. It does not calculate physical energy, realistic thermodynamic information gain, or validate model assumptions. The exhaustive oracle is only an upper bound for this toy synthetic objective.

## Separate candidate-decision negative controls

Experiment-choice regret is not the same as risk in an energy-superiority claim. The `candidate_decision_benchmark` module separately tests the efficiency comparator with synthetic cases where oracle truth is independent of the recorded observed delta. It varies observed-vs-true delta, uncertainty scale, unknown correlation declarations, and the registered worst-case differential-bias bound (0, 0.02, or 0.05 efficiency fraction); the bound is subtracted linearly after expanded measurement uncertainty, not treated as a random standard uncertainty. The expanded 720-case grid reports false-superiority, false-negative, and indeterminate rates. A documented `differential_bias_bound_basis` is required but is not proof of calibration quality; absent or weak physical evidence must not be converted into a zero-bias assumption. Run `cargo run -p symthaea-energy-bench --example candidate_decision_benchmark > candidate-decision-benchmark.json`.

A nonzero false-superiority rate under deliberately biased observations is a useful negative-control result: it demonstrates that a mathematically conservative interval is not protective against systematic errors that the uncertainty model omits. The test grid is not a calibrated estimate of real-world risk and cannot replace measurement-system characterization or independent reference validation.

## Acceptance targets and fair comparison

Use the staged acceptance targets in [ENERGY_CONVERSION_BENCHMARK_TARGETS.md](ENERGY_CONVERSION_BENCHMARK_TARGETS.md). In particular, compare Phase A against random and deterministic space-filling baselines under equal resource budgets and paired held-out scenarios. Report decision regret, budget to a correct registered decision, false-feasible and false-superiority rates, calibration when predictive distributions exist, selection robustness, and replay/audit completeness. The proposed 20%/25% cost-regret goals are targets to test, not established results or universal standards.

## Evaluation of the experiment selector

Before using Symthaea's selector to guide real R&D, compare it against the deterministic queue and random/space-filling baselines under identical budgets. Report:
- cost and number of model evaluations needed to resolve a registered decision;
- calibration/error of predicted observation distributions on held-out cases;
- decision regret and false-feasible/false-superiority rates;
- information gained per unit cost and how often the selected experiment changes the next decision;
- reproducibility across seeds and revisions;
- all failed experiments and safety constraint rejections.

The selector is useful only if it improves the quality or speed of engineering decisions without raising false claims or bypassing constraints. An estimate of expected information gain is not itself measured physical improvement.

## Measurement-uncertainty foundation

Use documented uncertainty propagation, with covariances included when justified by the measurement design. NIST TN 1297 describes combining uncertainty components and covariance through the law of propagation of uncertainty; JCGM 101 specifies Monte Carlo propagation of distributions:
- https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty
- https://www.bipm.org/en/doi/10.59161/jcgm101-2008

JCGM published a 2026 amendment to the GUM concerning nonlinearity in measurement models, reinforcing that a first-order approximation is not appropriate in every regime. Check the applicable current edition and applicability before designating the estimator suitable for a benchmark:
- https://doi.org/10.59161/PPDI3267

## Implementation boundary today

The current `symthaea-energy-bench` comparator implements a first-order delta approximation for *charge-sustaining net cycle efficiency* only when primitive input/output, interval, and candidate/baseline uncertainties are explicitly assumed independent. It does not yet implement expected information gain, covariance-matrix propagation, Monte Carlo propagation, or a multi-fidelity controller. The transparent Phase A proposal queue is implemented, but does not perform active learning. This protocol specifies the research direction and current implementation boundary.
