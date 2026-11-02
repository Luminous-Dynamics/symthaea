# Action-conditioned visual forecast evidence contract

**Status:** typed producer interface, content-addressed action/draw/forecast-bank contracts, exact-target branch scorer, and synthetic tests. No production CfC/JEPA action-conditioned predictor is implemented by this module.

## Why this module is separate from the autonomous horizon evaluator

The existing VisionManifold DelayedHorizonEvaluator is an important baseline: it issues autonomous CfC forecasts and scores them against subsequent real visual observations with a persistence comparison. It does not accept a hypothetical action set or bind each future to a named simulator branch.

This module defines the adjacent VIS-005 path:

    immutable observed visual belief + frozen action manifest
        -> action-conditioned prediction bank
        -> exact branch/frame observation
        -> visual prediction score

The old delayed-horizon evaluator remains unchanged and should continue as the autonomous A1 control. CrossManifoldPredictor remains vision-to-cognition mapping. The current JEPA path remains a separate self-distillation/latent comparison until a future-observation target is demonstrated.

## Producer contract

ActionConditionedVisualPredictor::predict_validated accepts an ActionForecastRequest with only:

- the observed belief vector, source frame sequence, capture time and history digest;
- canonical action descriptors and the action-manifest digest;
- model/checkpoint digest;
- horizon, target frame sequence and target timestamp;
- joint draw IDs and the declared coupling mode/draw-manifest digest.

It has no parameter for simulator hidden state, unchosen outcomes, target frames, evaluator labels or success/contact truth. It is &self and is specified as observational; concrete implementations must preserve that purity in their own tests.

The action manifest hash is recomputed from action IDs and canonical serialized descriptors sorted by action ID. The draw manifest hash is recomputed from sorted draw IDs plus the coupling enum. Action and draw row order is deliberately non-semantic.

The returned ActionForecastBank must have exactly one sample for every (action_id, joint_draw_id) pair. Every sample has a unique branch ID and a finite, non-zero belief vector of the same dimension as the frozen persistence baseline. The bank SHA-256 binds history, both manifests, model digest, source/target identities, times, coupling, exact f32 bit patterns, baseline and every branch sample. Action/draw/sample rows are sorted by semantic keys for hashing, so reordering rows does not change the bank digest.

The JSON representation is specified in schemas/action-forecast-bank.v0.1.schema.json; the synthetic known-answer payload is in fixtures/action-forecast-bank.example.json. The Rust validator enforces cross-field invariants that JSON Schema cannot express.

## Exact-target branch scoring

ActionConditionedForecastEvaluator admits a complete bank atomically into a bounded pending queue. Each real branch observation must match:

- history digest;
- action-manifest digest;
- joint-draw-manifest digest;
- action ID, joint draw ID and unique branch ID;
- exact target frame sequence;
- target timestamp within the configured tolerance.

A frame before the target leaves the forecast pending. A skipped target frame expires the forecast. A frame at the target sequence but after the permitted timestamp window expires it. A frame that claims the target sequence but has an earlier timestamp is rejected without consuming the pending forecast. No later frame is silently substituted for the target.

A matured score contains prediction cosine error, persistence cosine error, timing error, action/draw/branch identity and relative skill versus persistence. Relative skill is undefined when the persistence baseline has zero error. The evaluator never inserts the predicted vector into observation memory.

At episode close, finish_history explicitly expires every unobserved branch forecast for that source history and returns the missing action/draw identities. This prevents missing outcomes from disappearing into a pending queue or being carried into another episode.

## Coupling and provenance limits

The enum ActionDrawCoupling supports shared posterior/world noise, paired exogenous disturbances, and deterministic shared latent indices. The draw-manifest digest binds IDs and the declared coupling; it does not prove that the producer actually reused the same latent/noise draw.

Before qualification, a Symtropy/evaluator adapter must add and independently verify branch-execution receipts binding each branch to:
- an exact clone/reset of the same initial hidden state;
- the action payload and action-space version;
- the exogenous disturbance tape/draw identity;
- the exact target-frame sequence and captured timestamp.

Those evaluator-side receipts must not be passed into the predictor. This module currently verifies the forecast bank and the identity fields supplied with an observed branch; it does not authenticate the simulator process, branch reset, camera, profiler or source of draw coupling.

## Integration sequence

1. Keep existing autonomous delayed-horizon CfC forecasts as A1.
2. Implement an actual predictor behind ActionConditionedVisualPredictor, using an observed visual belief and hypothetical action as inputs.
3. Supply a real future-observation target; do not train against the predictor's own output and call it grounded.
4. In Symtropy, clone the identical initial state for each action/draw branch and retain independent branch receipts.
5. Run predictions before revealing future frames or evaluator labels; then score exact branch targets.
6. Export channel-specific outcome samples into CMAQ-018 joint action-bank evaluation for paired effects, simultaneous coverage, selected-action regret, false-dominance and net decision value.
7. Keep the physical authority gate independent of all model predictions and interval certificates.

## Required first qualification controls

- action swap and action reversal: changing the hypothetical action changes the predicted future for the correct causal reason;
- null-action/no-op: predicted branch remains aligned with the autonomous/persistence controls;
- row permutation: action/draw semantics and hashes remain invariant under payload row order;
- incomplete and duplicate action/draw pairs reject;
- stale/mutated action manifest, draw manifest and sample-bank bytes reject;
- branch ID, action ID, draw ID, history digest or manifest mismatch never consumes a pending forecast;
- skipped/late target and episode-close missing observation are explicitly expired;
- future observations never enter the predictor before issuance is sealed;
- all action branches begin from the same independently attested hidden initial state;
- compare A0 persistence, A1 autonomous CfC, A2 action-conditioned predictor, A3 explicit-effect head and A4 uncertainty-aware selection under equal total compute.

## Claim ceiling

This implementation is a producer-neutral contract, integrity checker and exact-target scorer. It is not a learned action-conditioned visual model, not a real Symtropy branch producer, not a proof of honest coupling, and not a safety or SOTA claim. The next measurable milestone is a real VIS-005 predictor that passes the synthetic controls and produces replayable exact-target branch receipts.
