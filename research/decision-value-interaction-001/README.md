# Decision-value Compute × Observation Interaction Reference

Owner: Symthaea issues #13279 and #13375. Benchmark protocol: #13216 and DECISION_VALUE_COMPUTE_OBSERVATION_TOURNAMENT_V0_1.md.

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

## Causal trajectory pairing (#13375)

The `causal_pairing` module validates the four arms' trajectory lineage after execution. It reuses the generic replay-regime distinction in #8764 rather than defining a second replay engine.

- **Open-loop action-invariant:** the arms may share a recorded future world trajectory. The validator requires matching world-state and exogenous-input commitments at every tick; observation-delivery commitments may differ by treatment.
- **Closed-loop action-coupled:** every arm must carry its own driver-executed trajectory and independent replay-audit receipt. The arms share the initial state, dynamics/simulator/clock roots, exogenous schedule and RNG-domain commitments, while state and observation traces may legitimately diverge as actions change the world.
- **Offline logged-policy evaluation:** rejected by this simulator-pairing validator because it needs a separate support and estimand contract.

The validator also checks contiguous integer ticks, a continuous pre/post state chain, common exogenous event commitments on every overlapping tick, separated policy/world/sensor/evaluator RNG domains, and deterministic transition consistency when two arms present the same tick/state/action/exogenous inputs.

This is an evidence-lineage and pairing check only. It does not validate simulator physics, prove exogenous schedule completeness, establish offline counterfactual identification, or authorize actions. In particular, a shared exogenous digest is not proof that the environment model is true.

Protocol: [Symthaea #13375](https://github.com/Luminous-Dynamics/symthaea/issues/13375). Generic replay regimes: [#8764](https://github.com/Luminous-Dynamics/symthaea/issues/8764). Driver-owned decision-time evidence: [#8732](https://github.com/Luminous-Dynamics/symthaea/issues/8732).

The pairing certificate also validates the arm-local decision-time information filtration:

- each decision binds its driver-clock decision time and actual action-application time;
- each delivered event carries its stable event identity, availability time and delivery time;
- consumed event IDs must be present in that arm's delivered prefix by the decision time;
- earlier delivered evidence may be reused, but future, unavailable, late, duplicated or empty event references are rejected;
- decision clocks may not regress, and actions may not apply before their proposals.

This is a DVPC semantic adapter over driver-supplied event references. Generic canonical event encoding, SHA-256 hash-chain receipts, oracle isolation and signing remain with #9437/#8715; the driver-owned information ledger remains with #8732. The validator does not prove policy non-anticipativity without an actual driver replay and future-suffix mutation test.

Each arm carries `stochastic_draw_bundle_sha256`, `stochastic_draw_bundle_verifier_sha256`, `stochastic_draw_bundle_verification_receipt_sha256` and `stochastic_draw_bundle_semantically_verified`. The causal certificate retains all four bundle, verifier-profile and verifier-receipt digests. The validator rejects missing bundle/verifier/receipt identities and a false semantic-verification disposition. Symtropy #3690 owns semantic address projection and Philox draw receipts; #3743 independently reconstructs the correspondence/address projection from complete event inventories and scopes. This Rust module does not implement correspondence matching, causal-invariance inference, Philox or verification of the external receipt bytes. Downstream admission must retrieve and independently validate the exact #3743 receipt and referenced bundle.

The evaluator also exposes `validate_causal_pairing_with_interaction`, which joins the four causal trajectory receipts to the #13281 regret outcomes. Applied outcomes must match the exact world tick, action digest and consumer application timestamp. Non-applied outcomes must match the terminal timestamp and reason. The combined certificate carries the exact same thirteen-field `PairedRunIdentity` as the regret certificate, so individually valid but unrelated trajectory and regret payloads cannot be joined.

## Run

    cargo fmt --manifest-path research/decision-value-interaction-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/decision-value-interaction-001/Cargo.toml
    cargo clippy --locked --manifest-path research/decision-value-interaction-001/Cargo.toml --all-targets -- -D warnings

## Verification boundary

Thirty Rust tests are authored and passed at exact source head `8f6d2e9b605e3ef07af5c593b1f209bc9a2c410e` (workflow run `45133813434`, Rust 1.96.0):
- rustfmt: passed;
- tests: 30 passed, 0 failed;
- strict Clippy with warnings denied: passed.

The 30 tests comprise the original paired interaction checks plus sixteen cluster-adapter tests. They qualify deterministic four-arm arithmetic, 13-field pairing validation, terminal no-application scoring, hierarchy-preserving cluster aggregation and fail-closed input rejection. They do not validate the task-loss oracle, causal treatment design, sampling representativeness, statistical assumptions, support completeness, physical latency, energy or deployment readiness.

## Cluster-level inference handoff

The evaluator retains the complete validated `PairedRunIdentity` in each certificate and exposes `cluster_adapter::export_cluster_deltas()`.

It emits the exact `VIS-DVPC-CLUSTER-DELTAS-001-v1` format consumed by [#8578 — VIS-DVPC-014](https://github.com/Luminous-Dynamics/symthaea/pull/8578), using this hierarchy:
1. Recompute the four signed effects from the four arm regrets and verify the interaction sign.
2. Average hidden-world outcomes within an episode.
3. Average episode means within each independently sampled scene/sequence/session.
4. Emit one delta per independent cluster for compute benefit, observation benefit, joint benefit and compute × observation interaction.

The adapter rejects duplicate cluster/episode/world records, episode-to-multiple-cluster assignments, changes to task/loss/evaluator/environment/action/resource/model/policy context, initial-state/seed drift within an episode and hidden-world cohort drift within a cluster. Five clusters is only a mechanics floor, not a scientific sample-size recommendation.

#8578 alone owns cluster bootstrap, sign-flip diagnostics and multiplicity adjustment. The analyzer PR contains the standalone implementation and a synthetic six-scene handoff fixture; its exact-head analyzer workflow result is not verified here. No cluster statistical significance or SOTA claim is inferred from the adapter tests.
