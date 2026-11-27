# RITE-Bench

**Research Integrity, Trust & Efficiency** is a deterministic evaluator contract for research agents. It complements CEDAR-Bench (evidence dependence), METHODE-Bench (method fidelity), and REPAIR-Bench (correction propagation) by making the overall qualification profile measurable without hiding failures inside one aggregate.

## Why another benchmark layer?

A final-answer score cannot distinguish a correct answer supported by independent evidence from a lucky answer backed by duplicated sources. A report-quality score cannot establish that its code executes correctly. Mean task accuracy does not disclose how often a system is near-perfect, how often it abstains, or whether it changes its answer when decisive evidence is removed while remaining stable under placebo changes.

Recent benchmarks are moving toward richer evaluation:
- SCI-AGENT-EXAM evaluates 120 tasks across 11 fields and seven workflow types with both executable and textual artifacts. Its reported leading average was 80.9, while 26.0% of runs crossed a 0.95 success threshold. This is a useful warning against relying on means alone.
- Autonomous Science Bench 0.1 evaluates six real laboratory workflows; its pilot leaderboard reports a 72.2% resolution rate for the leading configuration, with substantial cost and time. This is promising but still a small pilot set.

RITE-Bench does not claim to replace those task suites. It provides a common qualification layer that can be applied to task-level evaluator outputs.

## Scored dimensions

All metrics are emitted separately, with numerator and denominator preserved. Rates with no eligible observations are **undefined**, not zero.

| Dimension | Metric | Contract |
|---|---|---|
| Outcome | Decision accuracy | Exact agreement with the evaluator's decision oracle |
| Abstention | Answer coverage | Fraction not returning InsufficientEvidence |
| Selective reliability | Selective risk | Error rate among non-abstaining decisions |
| Calibration | Brier score | Mean squared error of P(submitted decision is correct) |
| Calibration | 10-bin ECE | Weighted absolute calibration gap, fixed equal-width bins |
| Citations | Citation precision | Eligible claim-specific citations / all cited sources |
| Independence | Support-origin adequacy | Required tasks meeting the oracle's distinct eligible-origin minimum |
| Causal robustness | Counterfactual sensitivity | Decision changed when the benchmark's decisive fact changed |
| Causal robustness | Source-removal sensitivity | Decision changed when a required supporting source was removed |
| Causal robustness | Placebo invariance | Decision remained stable under a non-decisive placebo change |
| Reproducibility | Replay pass rate | Independent replay, exact contract digest, and exact output-manifest match |
| Resource discipline | Budget compliance | All declared resource caps respected |
| Economics | Cost per correct decision | Total measured spend / correct decisions, reported in integer micro-USD |

The evaluator does **not** emit an overall weighted score. High accuracy cannot compensate for failed replay, poor calibration, weak evidence independence, or budget violations. Consumers should compare systems on a Pareto profile and publish each denominator.

## Record semantics

A TaskRecord is constructed by the benchmark harness and independent evaluator, not by the agent being scored.

- expected_decision is the task oracle's label.
- submitted_decision is the agent's final label. InsufficientEvidence is the abstention action.
- confidence_correct means the agent's stated probability that its submitted decision is correct.
- Citation eligibility and support relation are evaluator-adjudicated against the exact claim.
- origin_id groups mirrored URLs or re-hosted copies under one underlying origin. Distinct URLs do not imply independent evidence.
- Counterfactual and source-removal interventions are expected to change the decision; placebo interventions are expected not to.
- A required replay passes only when an independent replay succeeds and both the contract digest and output manifest match exactly.
- Resource limits are explicit. Missing limits are not silently interpreted as unlimited budget observations.

The evaluator rejects empty/duplicate task IDs, invalid confidence, empty citation identities, empty budget declarations, and replay evidence on tasks where replay was not required. A missing required replay is scored as a replay failure.

## Execution

This crate is a standalone Rust 2024 package with no external dependencies:

~~~sh
cargo fmt --manifest-path tools/rite-bench/Cargo.toml -- --check
cargo test --manifest-path tools/rite-bench/Cargo.toml --locked
cargo clippy --manifest-path tools/rite-bench/Cargo.toml --locked --all-targets -- -D warnings
~~~

## Qualification and claims boundary

The unit tests are synthetic contract tests. They prove that the evaluator computes its defined metrics and rejects selected malformed records; they do **not** establish model performance, benchmark validity on real scientific work, or superiority over any commercial system.

A public comparative result requires:
1. Expert-authored tasks and decision oracles, with private/sealed evaluation tasks.
2. Pre-registered run budgets, tool access, model versions, prompts, and harness commits.
3. At least three independent seeds per task for instability analysis, with task-level confidence intervals.
4. A frozen candidate stream and independently held auditor stream.
5. Human validation of claim-specific citation and support-origin judgments.
6. Exact run manifests, source hashes, code artifacts, environment lockfiles, and independently replayed outputs.
7. Full reporting of task outcomes, abstention, reliability, costs, and all denominators.

No leaderboard claim should be made from the contract fixtures alone.
