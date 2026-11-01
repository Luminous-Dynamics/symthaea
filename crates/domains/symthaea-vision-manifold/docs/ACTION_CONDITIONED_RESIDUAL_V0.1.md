# VIS-005 action-conditioned visual residual control — v0.1

**Status:** executable research control in the vision-manifold crate. The implementation is not yet qualified by the exact-head Rust workflow and has no measured benchmark result.

## Why this is a separate model arm

The current visual CfC delayed-horizon evaluator provides a valuable autonomous forecast baseline: it predicts from the latest real visual manifold state and later scores against a future real encoded frame. The current JEPA path consumes a previous FEP action but presently uses the CfC prediction approximation as its target; it is not yet an independently supervised action-conditioned future-visual model. The cross-manifold predictor is vision-to-cognition, not future visual prediction.

This module introduces a deliberately small A2 control between the A1 autonomous CfC and a future higher-capacity action-conditioned world model:

    A2(h, a, H) = A1(h, H) + e(a) ⊙ w

- h: immutable starting visual history, named by SHA-256;
- a: canonical action identity, descriptor digest and action embedding;
- H: explicit positive forecast horizon;
- A1: autonomous CfC horizon forecast from the same starting belief;
- e(a): action embedding from a caller-owned frozen action vocabulary/adapter;
- w: learned diagonal residual weights, initialized to zero.

Zero initialization means the A2 forecast is exactly the A1 autonomous forecast before supervision. The first measured differences are therefore attributable to updates driven by later observations, not a separate random initialization.

This is intentionally not a full nonlinear action-state transition model, pixel generator or calibrated stochastic predictor. It is a falsifiable first control for whether an explicit action residual adds predictive information over autonomous CfC.

## API and evidence flow

### Autonomous baseline

VisionManifold::predict_autonomous_horizon_checked(horizon) exposes the current real-frame CfC forecast as a checked, read-only operation. It rejects nonpositive/non-finite horizons, missing real frames and non-finite state. A regression test verifies repeated calls are deterministic and do not mutate the saved manifold or advance the observed-frame sequence.

### Forecast one action

ActionConditionedVisualResidualPredictor::issue_forecast accepts:

- exact observed-history digest;
- source real-frame sequence;
- frozen action-manifest digest;
- exact action-descriptor digest and action ID;
- model-checkpoint digest supplied by the caller;
- horizon in seconds;
- autonomous CfC baseline vector;
- action embedding.

It returns a typed ActionConditionedForecastReceipt; the forecast remains a prediction and is not written into observation memory.

### Forecast a complete candidate action set

issue_action_set takes two or more candidate descriptors/embeddings and creates all forecasts atomically from the same history, autonomous baseline, checkpoint identity, horizon and model version. Candidate IDs are unique and output receipts are sorted by action ID. Each receipt shares an action-set ID, exact action-set descriptor digest and action-set size.

The action-set digest binds sorted stable action IDs to their descriptor SHA-256 digests. The caller's action-manifest digest must itself be the digest of the canonical manifest (including action-space version, units and parameters); this module cannot verify a manifest whose bytes the caller does not provide.

### Evaluate held-out branches without learning

score_action_set is the qualification path. It requires the complete set of forecast IDs and later real-observation outcomes. It verifies that all receipts share the same history, manifest, action-set ID/digest/size, source sequence, horizon, model-checkpoint identity and forecast model version. It rejects duplicate/partial/mixed sets.

All branch scores are computed before any state mutation. The method consumes the pending forecast set and returns one score receipt per action, but does not change action weights or training version. Use this path for sealed held-out counterfactual qualification.

### Learn from authorized training outcomes

score_and_learn is a single-action supervised-training path only. It rejects forecasts that belong to multi-action sets.

score_action_set_and_learn is a batch supervised-training path only. It requires the complete set, computes every score from the shared frozen model version, averages the per-action diagonal gradients, and applies exactly one update after all inputs validate. This avoids candidate-order effects from scoring the first branch, changing weights, then scoring the next branch.

Never send held-out qualification outcomes to either learning method. Keep the evaluator truth store outside the predictor and use score_action_set for evaluation. A training/online-learning experiment must use a separate authorized training split and report the exact source/model version used.

### Timing and one-shot semantics

A scored future observation must have a sequence greater than the source real-frame sequence and delay at least as large as the requested horizon. Lateness is measured explicitly; a set whose branch outcome is outside the allowed lateness window is expired as a whole and is not trained. A forecast ID is one-shot and cannot be rescored.

The delay is supplied by the caller. The adapter must derive it from the same monotonic modality clock used by the real observation stream. Do not infer exact target alignment from frame sequence alone.

## Content-bound receipts

The forecast receipt binds:
- schema/version and unique forecast ID;
- observed-history digest and source frame sequence;
- canonical action-manifest digest and per-action descriptor digest;
- action ID and complete action-set identity/size;
- caller-supplied model-checkpoint digest and internal residual-weight digest;
- horizon and model training version;
- exact digests for autonomous base, action embedding and predicted future HV.

The pending checkpoint record stores the exact baseline, action embedding and residual weights used at issue time. Restore recomputes those vector hashes and the predicted vector, checks the canonical forecast digest, and rejects incomplete action sets or altered pending inputs.

The score receipt binds:
- source/manifest/action/checkpoint/forecast identities;
- exact digest of the later observed visual HV;
- source and observed frame sequence;
- requested delay and lateness;
- action-conditioned prediction error, autonomous baseline error, improvement and MSE;
- forecast model version and scoring/update versions.

SHA-256 protects the stored byte identity; it does not authenticate who produced a digest or prove that an external simulator used the declared action or common disturbance.

## Qualification protocol

Keep A1 and A2 as separate named arms. Use paired episodes with identical:
- observed starting history and frame sequence;
- action manifest and candidate IDs;
- target horizon and real branch reset;
- observation encoding and outcome scorer;
- training-data and total compute budget.

For every held-out history:
1. compute the A1 autonomous baseline once;
2. issue A2 forecasts for the full frozen action bank;
3. seal forecast receipts before any future observation/evaluator outcome is exposed;
4. execute each candidate in an independent clone of the same hidden initial simulator state; use shared exogenous disturbance IDs only when the simulator semantics truly permit it;
5. pass only the later encoded observations for those exact branches into the evaluator;
6. call score_action_set (not a learning method);
7. join its per-action score receipts to the CMAQ-018 joint-draw/action-bank evidence.

The caller must not expose hidden simulator state, unchosen branch outcomes, future frames or evaluator labels to the forecast producer. A real-camera deployment usually cannot observe all unchosen counterfactuals; in that case, use randomized matched interventions or mark unavailable counterfactuals as unavailable rather than inventing them.

### Required ablations and controls

- A0: last-frame persistence baseline;
- A1: autonomous CfC future visual HV;
- A2: zero-init action-conditioned diagonal residual (this module);
- A3: explicit action-effect head;
- A4: learned nonlinear action/state transition predictor;
- negative controls: zero action, shuffled action labels, swapped descriptors, shuffled histories and independent/unpaired disturbance draws.

### Metrics

Report by horizon and mechanism family:
- mean squared HV prediction error and cosine error;
- improvement over A1 (positive means A2 improves over autonomous CfC);
- action-effect sign and magnitude error;
- rank Kendall tau-b and top-action regret;
- action reversal/null-effect controls;
- terminal/contact false-safe and OOD strata;
- simultaneous action-set coverage, selected regret and false-dominance if a calibrated probabilistic head is later added;
- net decision value versus D0/D1 after inference, sensing, recovery and failure cost.

Do not infer calibration from the deterministic point residual. It emits no predictive distribution. Do not claim action-set coverage until a calibrated joint distribution and independent calibration panel exist.

## Claim ceiling

This module is a deterministic diagonal residual baseline with observation-supervised updates, full-action-set forecast issuance, frozen-model scoring, checkpointable pending receipts and content hashes. It does not prove causal correctness, cross-episode generalization, calibrated uncertainty, robust action selection, benchmark/SOTA improvement, physical safety or independent hardware/simulator provenance. The exact-head Rust tests and CI must pass before code qualification; real matched-branch results are a separate gate.
