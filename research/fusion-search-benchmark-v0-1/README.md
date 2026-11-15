# Fusion reduced-space search benchmark v0.1

A dependency-free, synthetic-only benchmark for the candidate-screening problem tracked in [FUSION-CANDIDATE-FUNNEL-001 #12212](https://github.com/Luminous-Dynamics/symthaea/issues/12212).

This is not a plasma model, reactor systems code, PROCESS adapter, or physical benchmark. Its closed-form reference oracle exists only to test search accounting and decision metrics before connecting specialist solvers.

## Why this benchmark exists

Candidate-count recall alone can conceal a catastrophic search failure. A policy may retain 99% of feasible candidates while screening out the one candidate on the true Pareto frontier. The benchmark therefore measures both:

- feasible-candidate recall;
- Pareto-frontier recall and best-objective regret.

It also keeps every generated candidate in the denominator and records solver outcomes instead of silently dropping failed or non-converged evaluations.

## Frozen experiment

- Candidate universe: the full 64 × 64 integer grid, exactly 4,096 stable candidate identities.
- Objectives: two dimensionless synthetic service scores, both maximized.
- Scenarios: a smooth feasible region and the same region with one isolated, high-value feasible island.
- Policies: exhaustive evaluation, a low-fidelity threshold screen, and a threshold screen with a declared uncertainty guard.
- High-fidelity outcomes: deterministic converged, failed, non-converged, unsupported, and inconclusive statuses. These statuses are synthetic controls, not solver behavior estimates.
- No stochastic seed or external dependency is required.

The uncertainty guard is deliberately constructed to cover the remote corner containing the rare feasible island. This is a controlled positive case, not evidence that a real model's uncertainty envelope is calibrated.

## Run

From this directory:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path Cargo.toml --bin run_fusion_search_benchmark
```

The executable prints the scorecard for every scenario-policy pair.

## Metrics and interpretation

- Candidate-universe coverage: fraction of the frozen identities represented exactly once.
- Candidate recall: fraction of reference-feasible candidates admitted by the analytical screen.
- Pareto recall: fraction of reference Pareto-frontier candidates admitted by the screen.
- Usable Pareto recall: frontier candidates admitted and returned by a converged high-fidelity run.
- Best-objective regret: reference-best sum of the two synthetic objectives minus the best feasible, converged result found by the policy.
- High-fidelity attempts and explicit failure/non-convergence/unsupported/inconclusive counts.

Exact recall, frontier, and regret metrics are withheld when the candidate universe or reference labels are incomplete. This mirrors the fail-closed requirement in FECOMP-0.

## Acceptance targets for this synthetic harness

- 100% candidate-universe and disposition coverage.
- Zero silent drops.
- Failed, non-converged, unsupported, and inconclusive evaluations remain visible.
- The rare-island negative control demonstrates that count recall can remain above 99% while Pareto recall is zero.
- The guarded policy recovers the rare Pareto candidate while using fewer high-fidelity evaluations than exhaustive evaluation.
- Incomplete candidate coverage never emits exact recall or regret.

These are software benchmark properties, not fusion engineering performance claims.

## Next qualification stage

Replace the closed-form oracle only after a pinned reference lane is reproducible. The next stage is an enumerable reduced design space evaluated exhaustively with a pinned PROCESS release, followed by D0FUS on common supported outputs. Keep the synthetic harness as a negative-control suite. A systems-code result remains model evidence, not physical truth; measured validation and held-out-domain evidence remain separate.
