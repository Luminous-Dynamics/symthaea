# Decision-Relevant Predictability Metrics

Dependency-free Rust metric contracts and a deterministic paired-world evaluator fixture for [issue #12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546).

The package includes no candidate forecasting model. It currently provides evaluator-side primitives:

- Qualified Lyapunov-time normalization that rejects any reference exponent not marked `Reliable`.
- Observed forecast-horizon extraction with finite-value and monotonic-time validation.
- Information-set Bayes risk, candidate policy risk, a separately labelled perfect-information lower bound, and regret.
- Expected value of information from expected posterior Bayes risk and acquisition cost in task-loss units.
- Componentwise Pareto dominance over a declared resource vector.
- A paired-world fixture in which a physical hazard and a sensor-offset-only world generate exactly the same passive primary-sensor observations.
- Separate candidate-visible episode/observation types and evaluator-only mechanism/loss records.
- Known-answer tests for passive observational equivalence, irreducible abstention risk, a decision-changing reference probe, and a one-bit nuisance observation with zero decision value.

The decision evaluator distinguishes the best action available under the candidate's information set from the evaluator-only perfect-information lower bound. This prevents irreducible uncertainty from being charged to the candidate as avoidable model error.

## Run

```bash
cargo fmt --manifest-path research/decision-predictability-benchmark/Cargo.toml -- --check
cargo clippy --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets
```

The crate uses only the Rust standard library and is an isolated Cargo workspace. The focused workflow is pinned to Rust 1.96.0.

## Paired-world evaluator boundary

Candidate-facing code should receive only `CandidateEpisode` and authorized `PrimarySensorObservation` values. Evaluator-only `EvaluatorWorld` records contain the latent mechanism and action losses. The Rust type split helps make the benchmark interface explicit, but this package is **not a sandbox or process-security boundary**. Confirmatory evaluations must keep evaluator truth in a separate process or permissioned environment and release only authorized observations to candidates.

The canonical fixture currently tests a single pair of worlds and exact deterministic probe outcomes. It does not yet implement a general sensor planner, noisy/missing/delayed probe channels, deadline enforcement, or a broad world generator.

## Scope and limits

These are metric/oracle primitives and finite synthetic controls, not a complete benchmark tournament, forecasting system, physical sensor system, or safety controller. Remaining work includes broader paired-world generators, observation permission and clock/budget enforcement, noise and missingness controls, candidate/evaluator process isolation, and matched external predictor baselines.

Resource vectors are comparable only when units, measurement boundaries, hardware and workloads match. The crate does not measure physical energy or hardware performance. Lyapunov normalization is rejected unless the reference exponent is marked `Reliable`; a candidate's own unqualified estimate must not define its score.

No output from this crate establishes SOTA, physical reliability, consciousness or causal reconstruction.
