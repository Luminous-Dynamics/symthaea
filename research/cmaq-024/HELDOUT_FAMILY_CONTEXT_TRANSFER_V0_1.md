# CMAQ-024 — Held-out environment-family transfer through observable context v0.1

**Status:** deterministic integration test added; exact-head formatting/tests/Clippy pending.  
**Path:** actual NixWard family-label-free HDC contextual delta → frozen scalar query readout → CMAQ-020 state-response abstraction score.  
**Claim ceiling:** transfer across held-out family *identifiers* where the observable context encodes the response map. This is not general causal mechanism discovery.

## Why the previous fixture was insufficient

CMAQ-022's original four-state integration test held out evaluation clusters, but the evaluation family IDs also appeared in training. It tested response-equivalent compression within known families, not family-identity transfer.

CMAQ-024 makes the training family IDs and evaluation family IDs disjoint, and also gives every evaluation state a distinct held-out cluster identity.

## Frozen known-answer construction

Train the real `NixWorldModel` on two family identities:

- training family A: context HDC A, per-action HDC delta +0.1 on coordinate 0;
- training family B: context HDC B, per-action HDC delta +0.5 on coordinate 0.

Each family contributes three independent training clusters. All episodes use the same realized action and action-parameter HDC, while the context HDC differs.

Evaluate on two new family identities C/D:

- held-out family C has observable context A and the same response map as training family A;
- held-out family D has observable context B and the same response map as training family B.

Each held-out family contributes two independent environment clusters, with nuisance variation in a separate state coordinate. Evaluation states all start at coordinate 0 = 0.1. The registered intervention/query/horizon reads coordinate 0 after the action, yielding expected integer responses 200 for C and 600 for D.

The candidate calls `predict_state_without_family_label`. It receives state, action, action-parameter HDC and observable context HDC only. Environment-family IDs are attached to evaluator records after candidate predictions are formed; they are not arguments to the candidate method.

## Matched controls

All four candidate partitions use the same held-out states and the same CMAQ response contract:

1. family-label-free contextual HDC prediction with a frozen coordinate-0 query readout;
2. pooled per-action mean-delta HDC baseline;
3. identity/no-op;
4. full predicted HDC vector partition.

The predictor requires at least three distinct matching training clusters before returning a contextual forecast. Here each training context has three independent clusters. Repeated samples and duplicate cluster IDs across family buckets do not inflate support; the family-blind predictor returns `InsufficientContextSupportUnknown` when de-duplicated support is below three. This is a deterministic engineering guardrail, not a calibrated confidence threshold.

The test asserts:
- contextual HDC gets 4/4 exact changed-query responses, 0 false merges, 0 false splits and 2 abstract classes;
- pooled global delta gets 0/4 exact responses and 4 false merges;
- identity/no-op gets 0/4 exact responses and 4 false merges;
- full-state memorization creates 4 classes and 2 false splits among response-equivalent nuisance variants.

These are expected assertions in a deterministic toy fixture, not reported benchmark results until the exact-head run passes.

## What transfer is actually demonstrated

The held-out family IDs and state clusters are new, but their observable context vectors are intentionally aligned with two training context/response modes. The test therefore demonstrates that the model can retrieve a known context-conditioned response map without receiving the family label. It does not show that the model can infer an entirely novel response law, extrapolate to arbitrary unseen contexts, or identify a latent causal mechanism.

The existing ambiguity test is the complementary boundary: when equally matched training buckets share the same observable context/action features but imply incompatible deltas, the family-label-free model must return `AmbiguousContextUnknown`. However, a truly novel hidden mechanism that changes the outcome while leaving all candidate-visible inputs indistinguishable from one known context cannot be detected from these inputs alone. No predictor can identify information it has not been given; an honest UNKNOWN policy needs an independent novelty signal, an additional permitted probe, or more informative observation.

## Next validation requirements

1. Expand from the synthetic coordinate readout to multiple semantically grounded intervention × query × horizon contracts.
2. Include action failure, partial completion, rollback, delayed state changes and outcome-unknown reconciliation.
3. Hold out whole mechanism families with context shifts and nuisance shifts; report the weakest family and query cells rather than only the average.
4. Include blind probe acquisition and evaluator-hinted CEGAR as separate matched-budget arms.
5. Compare HDC-only, context-only, action-parameter-only, combined HDC, symbolic/SCM, full-state, low-rank and recurrent baselines.
6. Use independent mechanism/environment clusters for inference under #8578, not repeated transitions.
7. Measure changed-response accuracy, pair-level false merges/splits, response regret, UNKNOWN/abstention, all-in memory, update/inference compute, latency and downstream closed-loop utility.

No global causal sufficiency, arbitrary held-out mechanism generalization, SOTA superiority or physical-safety claim follows from this finite known-answer test.
