# VIS-005 observation-grounded JEPA adapter contract — 2026-11-01

**Status:** additive API boundary in symthaea-jepa. No cognitive-loop wiring, camera/simulator adapter, calibrated stochastic predictor or real-world performance claim is included.

## Why this increment exists

The existing JEPA latent predictor already computes (z_t, a_t) → ẑ_{t+1} and accepts an action index. But the cognitive-loop caller currently forms its next-state target from the CfC multi-scale prediction approximation. This is a valid self-distillation/auxiliary prediction path, but it does not establish that JEPA has learned to predict an independently observed future visual state.

The new API gives the visual/simulator adapter a path for a different experiment: score/train the same action-conditioned JEPA against an independently acquired later observation, with explicit temporal and provenance metadata.

## New APIs

### ObservedTransitionEvidence

The struct carries:
- an explicit v0.1 schema identifier;
- source-history digest;
- source and target raw-observation digests;
- source and target embedding digests;
- canonical action-manifest digest;
- a JEPA-recomputed action-binding digest over the manifest digest, stable action ID and discrete action index;
- episode, branch and forecast IDs;
- distinct source/target observation IDs and monotonic sequence numbers;
- stable action ID and its discrete JEPA action index;
- source observation, forecast issue, action start/end, requested target and actual target timestamps;
- maximum permitted target-observation lateness.

train_observed_transition and evaluate_observed_transition reject:
- unsupported evidence schema versions;
- malformed lowercase SHA-256 identifiers or an action ID/index binding digest mismatch;
- empty or repeated observation IDs;
- non-increasing observation sequence;
- non-finite or temporally misordered timestamps;
- missing episode/branch/forecast IDs or a target observation later than its declared lateness budget;
- action indices outside the configured discrete action space;
- input dimensions different from JepaConfig.input_dim;
- non-finite embeddings;
- embedding digest mismatch.

The crate recomputes the source/target embedding SHA-256 over a domain-separated canonical byte encoding: domain tag, little-endian u64 dimension, then each exact f32 IEEE-754 bit pattern in little-endian order. This binds the values supplied to JEPA. It does not prove that the HDC embeddings were honestly derived from the raw observation bytes; the adapter must bind those bytes and digests independently.

The transition clock requires source observation ≤ forecast issuance ≤ action start ≤ action end ≤ requested target ≤ actual target. The actual target may be late only within the explicitly declared maximum lateness. The actual target may be late only within the explicitly declared maximum lateness. This preserves a precise requested horizon while allowing bounded sensor timing jitter.

The learning method additionally rejects zero, negative, non-finite or greater-than-one learning rates. The evaluator path does not update model weights.

### predict_action_set_one_step

This API:
1. validates one source embedding and the source-history/action-manifest digest syntax;
2. requires at least two candidates;
3. rejects duplicate action IDs, duplicate discrete action indices and out-of-range action indices;
4. encodes the source observation exactly once;
5. evaluates every candidate action against the same encoded context;
6. returns action-indexed latent point predictions and a canonical candidate-set digest.

The candidate-set digest is invariant to row order and binds the full action-manifest digest plus sorted (action_id, action_index) pairs. For a single observed transition, a separate action-binding digest is recomputed from the manifest digest, stable action ID and discrete JEPA action index. The manifest digest must itself be generated and authenticated by the upstream adapter from canonical action descriptors, units and action-space version.

These outputs are deterministic point predictions. Do not promote repeated calls, candidate rows or latent dimensions into a posterior sample bank. This API is a point-prediction comparator / explicit-action-head interface. CMAQ-018 remains the separate uncertainty and paired-effect evaluator.

## Relation to the existing cognitive loop

The low-level JepaEngine::train_step and current cognitive-loop caller remain unchanged, preserving compatibility. The new methods are opt-in and require explicit observation evidence.

The correct future integration is not to silently replace the existing target with whatever vector happens to be available next cycle. Instead, the adapter must establish that:
- the source vector came from a real acquired observation;
- the action started after that source observation;
- the target vector came from a distinct later acquired observation after action completion;
- the action and target belong to the same episode/branch;
- the training target was sealed after the source/action prediction;
- hidden state, unchosen counterfactual outcomes and evaluator labels were not accessible to the predictor.

If an internal cognitive cycle can occur without a new observation, it must not be treated as an observed transition. The observation adapter, not the model's cycle clock, owns the observation sequence and timestamps.

## Relationship to CMAQ-018

predict_action_set_one_step returns deterministic per-action latent means. It is not directly a JointActionSetSamples producer because it has no posterior or shared-noise draws. A future stochastic extension must specify how one common model/posterior draw and valid exogenous disturbance are reused across candidate branches. Only then can CMAQ-018 compute paired effects with a meaningful joint draw axis.

Until then:
- the new point predictions can be scored for action sensitivity and effect sign/rank against exact Symtropy outcomes;
- the latent prediction is not a calibrated confidence interval;
- no paired-effect interval may be fabricated from marginal per-action uncertainty;
- no SOTA claim follows from a successful API test.

## Tests and focused CI

The crate now tests:
- accepted future-observation transition training;
- rejection of malformed digests and evidence schema;
- rejection of invalid sequence, cross-branch identity omissions, forecast/action ordering, target-time ordering and excessive observation lateness;
- rejection of out-of-range action indices and invalid learning rates;
- rejection when the exact source/target embedding bytes do not match the supplied embedding digests;
- rejection when the stable action ID is substituted without updating its manifest/index binding digest;
- common-context action-set prediction and identity preservation;
- row-order invariance of candidate-set digest and named-action outputs;
- duplicate action identity/index rejection;
- SHA-256 known-answer behavior.

A new focused workflow runs cargo test --locked -p symthaea-jepa, formatting and Clippy for this crate. It has not yet completed on the PR's exact head, so no compilation/test pass is claimed here.

## Research qualification targets

On a sealed simulator panel, compare:
- A0 persistence / last-frame baseline;
- A1 current autonomous CfC delayed visual forecast;
- A2 action-conditioned JEPA trained on future observed embeddings via this API;
- A3 explicit action-effect head;
- A4 a future calibrated joint-draw model integrated with CMAQ-018.

For A2, report one-step and multi-step only where the implementation actually supports them; the current JEPA API is one-step. Report action-swap, zero-action, descriptor-swap and null-effect controls; effect-sign accuracy; latent cosine error; task-relevant structured-state error; and per-action sample counts. For an uncertainty-capable A4, additionally report simultaneous action-set coverage, interval score/width, selected regret-bound coverage, false-dominance rate and abstention utility.

Run all policies on identical frozen histories and candidate action manifests. Keep evaluator truth sealed. Report results by action family, horizon and hidden regime. Charge encoder, predictor, action-set evaluation and acquisition costs. Use paired episode-level confidence intervals.

## Claim ceiling

This increment proves that the typed API validates declared observation/action chronology and exact HDC embedding payload digests before calling JEPA, and that candidate action point predictions share one encoded context. It does not authenticate raw capture provenance, prove physical causality, demonstrate a production visual adapter, calibrate uncertainty, establish same-state counterfactual execution, or show improved decisions.
