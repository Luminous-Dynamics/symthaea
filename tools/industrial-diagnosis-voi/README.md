# Active Industrial Diagnosis and Value-of-Information Benchmark v0.1

**Owner:** Symthaea INDUSTRIAL-DIAG-VOI-001 #12403  
**Status:** synthetic finite-hypothesis decision-support benchmarks: one-step VoI and exact adaptive depth-two test selection.

## Research question

A diagnosis system should not only classify a fault from supplied telemetry. It should decide which measurement is worth taking next, given the expected consequences of a wrong decision, the uncertainty in the measurement model, the cost and delay of inspection, and the evidence/authority needed to run the test.

Information gain and decision value are different objectives. This benchmark includes a small counterexample where the most informative test is not the test with the greatest expected reduction in decision loss after measurement cost.

## Frozen synthetic pump/valve case

Hypotheses:
- dangerous process degradation: prior 0.10;
- metrology calibration drift: prior 0.30;
- normal/innocuous variation: prior 0.60.

Available decisions: continue, stop and quarantine, recalibrate and retest. The synthetic loss matrix is in the Rust fixture and intentionally assigns high loss to continuing under dangerous process degradation.

Candidate tests:
- calibration reference artifact: cost 1.0 units; P(positive | hypotheses) = [0.10, 0.95, 0.05];
- independent process safety test: cost 2.0 units; likelihoods [0.90, 0.15, 0.10];
- general vibration screen: cost 0.5 units; likelihoods [0.55, 0.65, 0.35].

Results from the reference case:
- baseline minimum expected decision loss = 11.4;
- calibration reference artifact expected information gain ≈ 0.605079 bits, net VoI = 0.89;
- independent process safety test expected information gain ≈ 0.200566 bits, net VoI = 1.63;
- general vibration screen net VoI = -0.50.

Thus maximum information gain chooses the calibration reference artifact, while maximum positive net decision value chooses the independent process safety test. The results are dimensionless synthetic fixture values, not measured diagnostic performance, real downtime savings, or a production safety recommendation.

## Evidence and hostile controls

A measurement is blocked unless its calibration, process applicability and authority are all current, its likelihood model is known and valid, and its evidence origin is not already represented by an existing observation. The hostile suite covers stale calibration, duplicated/correlated evidence, out-of-envelope applicability, unknown sensitivity and missing inspection authority.

The current version evaluates one measurement and its two finite outcomes. Probabilities and losses use f64; "exact" means exhaustive evaluation of the supplied discrete hypotheses and both outcomes, not symbolic rational arithmetic or a validated Bayesian model of real equipment.

## Adaptive depth-two extension

The crate now includes an exact two-stage policy-tree evaluator in `sequential.rs`. It enumerates each admissible first measurement, updates the posterior for each non-zero-probability outcome, then chooses a second test or stops on that observed branch. It never sees a hidden scenario label or a future test outcome.

Frozen pump/valve result with a measurement budget of 4.0 synthetic cost units:

- No-test baseline expected loss: 11.4.
- Greedy one-step net-VoI selects the independent process-safety test; expected total loss including cost: 9.77.
- Exact depth-two policy selects the calibration reference artifact first. After a positive result it stops and chooses recalibrate/retest; after a negative result it conducts the independent process-safety test.
- Adaptive depth-two expected terminal decision loss: 4.248.
- Expected measurement cost: 2.35 (first test cost 1.0 plus second test cost 2.0 on the 0.675-probability negative branch).
- Expected total loss: 6.598.
- Worst-case path total loss in the two-branch policy: 7.204444.

These values are finite-model outputs in dimensionless synthetic units. They demonstrate the value of adaptive sequencing relative to a greedy one-step policy on this constructed fixture; they are not real diagnostic performance, cost savings or safety advice.

The depth-two evaluator assumes conditional independence between different evidence origins given the supplied hypotheses. It blocks reuse of the same measurement ID and same evidence origin, charges first and branch-specific second-test costs once, respects a total path budget, and abstains from zero-probability branches. Cross-origin correlated measurement models, timing/downtime, likelihood uncertainty and high-consequence robust constraints remain future work.

## Metrics

Expected information gain is reported separately from expected posterior decision loss, measurement cost, gross value of information and net value of information. A test with negative net VoI is not selected by the decision-value policy. If no admissible test has positive net VoI, the policy returns NO_TEST.

## Verification

    cargo fmt --manifest-path tools/industrial-diagnosis-voi/Cargo.toml -- --check
    cargo clippy --manifest-path tools/industrial-diagnosis-voi/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/industrial-diagnosis-voi/Cargo.toml --all-targets
    cargo run --manifest-path tools/industrial-diagnosis-voi/Cargo.toml --bin voi-report
    cargo run --manifest-path tools/industrial-diagnosis-voi/Cargo.toml --bin voi-sequential-report

## Next generation

Add multi-step adaptive measurement trees, correlated measurement error models, high-consequence loss constraints, robust Bayesian uncertainty over test sensitivity/specificity, exact policy-tree search for tiny cases, and schedule/reconfiguration consequence coupling. Then compare against fixed decision trees, greedy information gain, EVSI, cost-aware diagnosis and human-led inspection under identical observation and measurement budgets. Physical claims require an instrumented benign demonstrator and independent verification.
