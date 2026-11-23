# Decision-value Compute × Observation Interaction Reference

Owner: Symthaea issue #13279. Benchmark protocol: #13216 and DECISION_VALUE_COMPUTE_OBSERVATION_TOURNAMENT_V0_1.md.

This standalone Rust crate evaluates the signed interaction in evaluator-measured decision regret across exactly four paired runs:

| Arm | Additional compute | Additional observation |
|---|---:|---:|
| Baseline | 0 | 0 |
| Compute only | 1 | 0 |
| Observation only | 0 | 1 |
| Joint | 1 | 1 |

For regrets R00, R10, R01 and R11, all in one frozen task-loss scale and measured at each arm's actual application time:

- Compute benefit: R00 − R10
- Observation benefit: R00 − R01
- Joint benefit: R00 − R11
- Interaction: R10 + R01 − R00 − R11

Positive interaction denotes complementarity. Zero denotes additive/no interaction. Negative interaction denotes redundancy or antagonism. Treatment benefits remain signed; harmful compute or observation is never clipped to zero.

## Pairing requirements

The four runs must share episode ID, hidden-world ID and digest, initial-state digest, task/loss scale, evaluator profile, execution-environment digest, action-domain digest, resource/deadline-budget digest, candidate model/configuration, policy digest and deterministic seed. Application timestamps are allowed to differ and are returned in the certificate, because delay is a treatment outcome.

Each arm remains in the denominator. If it applies an action, score action regret at actual application time. If it does not apply an action, emit an explicit NotApplied disposition (deadline miss, no qualified action, consumer rejection or execution failure) and use the terminal no-output regret penalty defined by the frozen task contract. The evaluator does not invent the penalty; absence is never scored as zero. Application and terminal timestamps are preserved separately.

Each score must be evaluator-supplied and lie within [0, 1,000,000] PPM. Missing/duplicate arms, identity mismatch, empty digests and invalid regret are rejected.

## Relationship to existing owners

- #13176 continues to own exact factorial attribution for wrong-belief, abstention, missing-output exposure and twelve resource-cost dimensions.
- This crate adds only the paired decision-regret contrast, which is not present in the current #13176 outcome schema.
- #13216 owns the compute benchmark protocol; #9563 owns observation policy; #12567/#12551 own expected posterior VOI; #13202 owns robust observation-support regret.
- #9437 still requires evaluator process isolation. This crate is not a security boundary.

This crate does not implement a candidate controller, scheduler, sensor policy, belief model, task-loss oracle, resource allocator or action authority.

## Run

    cargo fmt --manifest-path research/decision-value-interaction-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/decision-value-interaction-001/Cargo.toml
    cargo clippy --locked --manifest-path research/decision-value-interaction-001/Cargo.toml --all-targets -- -D warnings

## Verification boundary

Fourteen deterministic Rust unit tests are authored and passed on exact source head 0e5f80ca51541711752869d8d6bfe0304df1496d (workflow run 45086902847, Rust 1.96.0): changed-crate formatting passed, 14/14 tests passed, strict Clippy passed. This qualifies only bounded arithmetic, arm pairing, identity checks and signed contrast semantics for supplied records. It does not validate the evaluator's loss oracle, causal treatment design, support completeness, statistical significance, physical latency, energy, or deployment readiness.
