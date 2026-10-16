# Energy-Conversion Benchmark Targets and Evaluation Protocol

**Status:** proposed acceptance targets, not demonstrated performance claims  
**Related implementation:** [Symthaea PR #7480](https://github.com/Luminous-Dynamics/symthaea/pull/7480)  
**Related protocol:** [Experiment Selection Protocol](ENERGY_CONVERSION_EXPERIMENT_SELECTION.md)

## Why the benchmark must test decisions, not just scores

A design optimizer can report a higher predicted efficiency while selecting a worse real design, exploiting a simulator defect, or ignoring an unmodeled cost. Symthaea therefore needs three distinct benchmark layers:

1. **Accounting and evidence correctness:** is each run valid, reproducible, and honest about its evidence class?
2. **Experiment-selection quality:** does the selector reach the correct preregistered decision at lower cost than simple baselines?
3. **Physical validity and usefulness:** do the modeled gains survive calibrated reference data and independent measurement?

Passing one layer does not imply passing the next. Synthetic tests prove code behavior under specified assumptions; they do not establish real-world efficiency gains.

Recent multifidelity benchmarking literature recommends analytically specified test functions with known answers, varied difficulties (including multimodality, discontinuities, and noise), repeated initial conditions, and both optimization-effectiveness and approximation-quality metrics. See Mainini et al., *Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods* (2025/2026), https://doi.org/10.1007/s11831-025-10392-8. The 2026 survey by Li and Li provides broader context: https://doi.org/10.1145/3801959.

For physical comparisons, measurement methods and operating conditions must be defined for the product or technology; there is no universal efficiency scalar that fairly compares engines, heat pumps, fuel cells, propulsion systems, and storage. DOE test-procedure guidance provides an example of the need for representative-use procedures: https://www.energy.gov/cmei/buildings/test-procedure-development-and-revision.

## Benchmark ladder

### Gate 0 — deterministic correctness and fail-closed behavior

These are release gates, not optimization targets.

- **100% deterministic replay:** identical policy, proposal bytes, and software revision produce identical eligible ordering, rejection ordering, and reason codes.
- **100% hard-gate enforcement:** every tested violation of registration, safety review, operating-envelope declaration, abort criteria, provenance, competing hypotheses, finite positive cost, or common resource unit is rejected.
- **Zero silent drops:** every input proposal appears exactly once as eligible or rejected; duplicate IDs each receive an explicit rejection.
- **Zero synthetic promotion:** synthetic examples remain not-for-decision and cannot support physical-superiority claims.
- **100% known-answer accounting cases:** hand-calculated, contract-conforming first-law fixtures close within the registered tolerance; deliberately corrupted intervals, time gaps, missing phases, and storage discharge cannot be hidden by aggregate cancellation.
- **Fail-closed uncertainty:** unsupported metrics, unknown correlations, and missing covariance evidence do not produce affirmative superiority decisions.
- **Reproducible provenance:** each evaluation records immutable policy/manifest hashes, source revision, solver and fidelity identity, seeds, raw artifacts, and a replay path.

The percentages above refer only to enumerated conformance tests. They are not a claim that every possible failure mode has been covered.

### Current harness implementation

The Rust crate now contains `experiment_benchmark`, a known-answer synthetic benchmark API. It compares the Phase A queue with seeded random selection, cheapest-first selection, a one-dimensional space-filling baseline, and an exhaustive oracle upper bound. The deterministic generator produces six-proposal scenarios and includes deliberately deceptive proposer estimates. Reports preserve chosen experiment IDs, rejected candidates, resource budgets and use, achieved utility, oracle utility, regret, decision-resolution rates, and summary statistics. A JSON receipt embeds all scenario inputs and carries SHA-256 digests for the policy, each scenario, and suite configuration; record the exact source commit separately because input hashes do not identify executable code.

The oracle assigns each synthetic experiment an independent probability of resolving the hidden decision. Combined utility is defined as `1 - product(1 - gain_i)`. This convenient closed-form assumption makes exact subset enumeration possible; it is not a general information-gain estimator and does not model correlated, redundant, or mutually dependent experiments. The oracle is only an upper bound for this toy objective.

A 1,000-scenario Rust regression test and `examples/selector_benchmark.rs` runner have been authored. Run the receipt-producing example with `cargo run -p symthaea-energy-bench --example selector_benchmark > energy-selector-benchmark.json`; the example compares Phase A, cheapest-first, space-filling, the exhaustive oracle upper bound, and 30 seeded random strategies over the same generated scenarios. Heuristic strategies stop selecting once their registered synthetic resolution threshold is reached, allowing resource-to-resolution to be measured; the exhaustive oracle also calculates the minimum cost of any eligible subset that reaches the threshold. The current selector harness does **not** model candidate feasibility/superiority declarations, so the false-feasible and false-superiority metrics above require a separate candidate-decision oracle suite. Until tests execute on an identified revision, they must not be described as passing or used to claim that Phase A meets the proposed targets.

### Gate 1 — oracle-backed synthetic experiment selection

Build an oracle-backed suite where the true decision and experiment outcomes are known. Keep the generator seed and scenario definitions versioned. Include at least these scenario families:

1. **Decision reversal:** the initially favored design is wrong; only some experiments reveal the decisive loss or constraint.
2. **Deceptive proxy:** the proposal with the largest declared uncertainty reduction is not the one that resolves the decision.
3. **Budget pressure:** compute time, rig time, and direct cost produce different choices; never compare incompatible units.
4. **Noisy and biased fidelity:** cheap models are fast but systematically wrong in a controlled region.
5. **Common-mode uncertainty:** candidate and baseline share calibration or ambient errors; independence-based comparison should be refused or shown to be inadequate.
6. **Constraint boundary:** tiny changes in a hard requirement switch feasibility; optimize decision correctness, not average response error alone.
7. **Distribution shift:** training/reference cases differ from held-out conditions or operating envelopes.
8. **Adversarial records:** duplicate identifiers, missing hypotheses, post-observation registration, invalid numeric values, incomplete artifacts, and rejected/pending safety review.

Start with at least **1,000 generated scenarios across registered seeds** and at least **30 paired seeds per stochastic selector/baseline comparison**. These are proposed starting sample sizes; use power analysis or interval width to increase them when the failure-rate confidence bounds remain too wide.

Compare at equal total resource budgets against:
- random selection;
- deterministic space-filling selection;
- the current Phase A lexicographic queue;
- later, calibrated model-based expected-information-gain selection.

Every baseline must have access to the same admissible proposals and total budget. Report cost separately by compute time, rig time, and money where applicable. Do not grant one method extra high-fidelity evaluations without accounting for their cost.

#### Primary metrics

- **Decision regret:** loss from the decision chosen by the selector versus the oracle-optimal decision under the same scenario and budget; normalize only with a documented nonzero reference scale.
- **Budget to resolve:** total resource consumed until a preregistered decision rule is correctly triggered.
- **False-feasible rate:** fraction of truly infeasible candidates incorrectly declared feasible.
- **False-superiority rate:** fraction of cases where an affirmative superiority claim is issued but the oracle says the registered margin is not met.
- **Coverage and calibration:** empirical coverage of predictive intervals and a proper scoring rule once calibrated predictive distributions exist.
- **Selection robustness:** frequency of a material decision change under small perturbations in declared costs, predicted uncertainty reduction, and plausible operating conditions.
- **Auditability:** proportion of proposals, rejections, model calls, seeds, and output artifacts completely captured and replayable.

Always report counts and uncertainty intervals, not only point estimates. Use paired comparisons on the same scenarios; report per-family performance so easy cases cannot mask failure on adversarial ones.

#### Proposed Phase A graduation targets

These are **engineering acceptance goals to test**, not field-wide standards or measured current performance:

- all Gate 0 conformance tests pass;
- no false-superiority claims in the initial 1,000-case synthetic oracle suite, with the upper confidence bound reported (zero observed failures alone is not proof of zero risk);
- median normalized decision regret at least **20% lower than random selection** on scenario families where the declared proposal attributes contain genuinely decision-relevant signal;
- median budget-to-correct-decision at least **20% lower than the best simple baseline** on those same families, with the paired 95% confidence interval supporting improvement;
- no scenario family regresses materially without the limitation being documented and an explicit reason for acceptance;
- input-order permutation and repeated-run tests preserve exact ranking and rejection output.

If Phase A fails these targets, first improve the proposals, estimates, or ranking policy; do not immediately add a more complex optimizer.

### Gate 2 — calibrated multifidelity selection

Only graduate to expected information gain (EIG) or expected value of sample information when there are predictive distributions and credible reference observations.

Proposed goals:

- evaluate held-out predictive calibration, bias, variance, and runtime by fidelity level;
- for nominal 90% and 95% predictive intervals, report empirical coverage by scenario family and investigate material departures rather than optimizing one aggregate calibration number;
- demonstrate at least **25% lower median total cost to resolve the same decision** versus the best Phase A/random/space-filling baseline on held-out decision tasks, with paired confidence intervals;
- show that the result is robust to reasonable prior, model discrepancy, correlation, and fidelity-cost assumptions;
- include ablations for each major component and compare against simple methods at equal cost;
- preserve and report failed runs, rejected hypotheses, negative controls, and cases where the selector chooses poorly.

The thresholds above are proposed go/no-go targets. They should be preregistered and versioned before benchmark results are viewed, and relaxed only through a documented policy revision with old results retained.

### Gate 3 — reference cases and independent measurement

Use technology-specific benchmark contracts, not one metric across every technology. Begin with independently traceable, measured or accepted reference cases for distinct categories, for example a charge-sustaining engine/generator, a heat pump, and an electric drive. Use each technology's proper metric: net electrical/shaft efficiency, COP under specified conditions, or mission-specific propulsion/performance metrics. Do not compare raw COP with engine efficiency.

For each reference case, declare the boundary, representative duty cycle, instrumentation/calibration, ambient conditions, uncertainty and covariance assumptions, solver/model revision, and acceptance tolerance before comparing results. Report:
- conservation/closure error relative to the preregistered tolerance and measurement uncertainty;
- prediction error by operating point and duty-cycle phase;
- uncertainty coverage and residual structure;
- performance across off-design operation, transients, aging/degradation, start/stop, and auxiliary loads where relevant;
- whether candidate superiority survives propagated uncertainty and the full system boundary;
- independent replication or a documented reason why it is not yet available.

There is no defensible universal target such as “20% better energy efficiency” before the mission, boundary, technology metric, and feasibility constraints are registered. The goal is a measured improvement that survives fair comparison and uncertainty—not an impressive-looking number detached from a real use case.

## Benchmark governance

Before each benchmark campaign, freeze the protocol digest, scenarios, metrics, budgets, baseline implementations, and decision thresholds. Hold the evaluation scenarios out from selector design and surrogate calibration. Retain raw data, failed cases, software revisions, seeds, and prior results; never silently delete a failure or change a threshold after seeing an unfavorable result.

A green synthetic benchmark is permission to proceed to the next evidence tier, not permission to claim physical superiority or bypass engineering safety approval.
