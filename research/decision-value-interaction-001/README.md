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

Twenty-four Rust tests are authored and passed at exact source head `99a1ed303d8fc9d7539de8505349a1c1ea5f41f2` (workflow run `45119544918`, Rust 1.96.0):
- rustfmt: passed;
- tests: 24 passed, 0 failed;
- strict Clippy with warnings denied: passed.

The 24 tests comprise the original paired interaction checks plus ten cluster-adapter tests. They qualify deterministic four-arm arithmetic, 13-field pairing validation, terminal no-application scoring, hierarchy-preserving cluster aggregation and fail-closed input rejection. They do not validate the task-loss oracle, causal treatment design, sampling representativeness, statistical assumptions, support completeness, physical latency, energy or deployment readiness.

## Cluster-level inference handoff

The evaluator retains the complete validated `PairedRunIdentity` in each certificate and exposes `cluster_adapter::export_cluster_deltas()`.

It emits the exact `VIS-DVPC-CLUSTER-DELTAS-001-v1` format consumed by [#8578 — VIS-DVPC-014](https://github.com/Luminous-Dynamics/symthaea/pull/8578), using this hierarchy:
1. Recompute the four signed effects from the four arm regrets and verify the interaction sign.
2. Average hidden-world outcomes within an episode.
3. Average episode means within each independently sampled scene/sequence/session.
4. Emit one delta per independent cluster for compute benefit, observation benefit, joint benefit and compute × observation interaction.

The adapter rejects duplicate cluster/episode/world records, episode-to-multiple-cluster assignments, changes to task/loss/evaluator/environment/action/resource/model/policy context, initial-state/seed drift within an episode and hidden-world cohort drift within a cluster. Five clusters is only a mechanics floor, not a scientific sample-size recommendation.

#8578 alone owns cluster bootstrap, sign-flip diagnostics and multiplicity adjustment. The analyzer PR contains the standalone implementation and a synthetic six-scene handoff fixture; its exact-head analyzer workflow result is not verified here. No cluster statistical significance or SOTA claim is inferred from the adapter tests.
