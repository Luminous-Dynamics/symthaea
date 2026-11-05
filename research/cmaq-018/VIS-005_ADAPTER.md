# CMAQ-018 ↔ VIS-005 action-conditioned predictor adapter

**Status:** interface contract only. No production VIS-005 adapter or actual Symthaea prediction result is included in this synthetic benchmark PR.

## Architectural owner

Vision Mind program: [#3934](https://github.com/Luminous-Dynamics/symthaea/issues/3934), specifically VIS-005 (action-conditioned visual prediction), VIS-006 (active vision) and VIS-008 (Symtropy hidden-ground-truth crucible).

VIS-005 specifies:

current belief + hypothetical action -> predicted future visual belief

CMAQ-018 tests the decision-critical aspect of this interface: from one same-history belief, do candidate action-conditioned forecasts preserve the true differences and ranking among alternative actions?

## Candidate contract

The Rust candidate boundary is:

- CandidateBranchPredictor::predict(&ObservedHistory, Intervention, ForecastKind) -> Option<CandidateForecast>
- CandidateEffectPredictor::predict_effect(&ObservedHistory, left_action, right_action) -> Option<f64>

The candidate interface receives observed history and a canonical hypothetical action. It does **not** receive evaluator history IDs or random seeds, scenario labels, hidden mechanism/regime state, evaluator disturbance values, realized outcomes for any branch, future counterfactuals for unchosen actions, or observation-memory mutation handles/evaluator access.

None is an explicit abstention: the candidate declines to forecast an unsupported intervention. It must not be rewritten as zero error or a safe outcome.

### Forecast provenance

ForecastKind::Predicted and ForecastKind::Counterfactual are distinct tags. The current same-history branch-bank experiment calls the interface with Counterfactual, because the actions are hypothetical alternatives from the same observed history. A future online controller may label a forecast of its committed next action as Predicted.

These tags describe epistemic provenance, not confidence or truth. Neither is an observation. Predicted/counterfactual outputs must never be appended to observation memory as though they were sensor evidence.

### Typed visual belief adaptation

The synthetic harness uses ObservedHistory { x, v } and a two-control action Intervention { force, grip }. This is a minimal test fixture, not the production visual-belief schema.

A real VIS-005 adapter should map:

1. A frozen, typed current visual belief containing observed entity/geometry state, uncertainty, coordinate frame, timestamp and evidence lineage.
2. A canonical hypothetical action defined by the applicable embodiment/action schema.
3. A predicted future visual belief for a declared horizon, preserving source-belief and action identity.
4. Explicit provenance Predicted or Counterfactual, with no mutation of the observation store.

The adapter must not silently convert image coordinates into world coordinates, silently promote inferred/predicted data to observed data, or omit uncertainty/coordinate-frame identity. The belief representation and action schema must be frozen and digested in the shared CMAQ manifest before an external comparison.

## Evaluator boundary

Symtropy owns hidden physical truth. The candidate receives rendered/observed evidence only. The evaluator separately holds scenario/world identity, state transitions, branch outcomes and safety/termination truth.

For real sensor traces, do not invent an unexecuted real-world branch outcome. Score only genuinely executed outcomes or a simulator's exact hidden-state interventions, and label the source accordingly. Observational logs alone cannot provide ground truth for an action that was never performed.

## Required qualification ladder

1. **Synthetic interface:** current CMAQ-018 known-answer fixtures, action-order metamorphic test, hidden-variable information boundary and metric denominators.
2. **VIS-005 control:** adapt the existing CfC/vision dynamics predictor as the first actual model baseline; retain it as a control, not a presumed winner.
3. **Learned predictor ablation:** optional JEPA-style model only after source/checkpoint identity and output provenance are frozen. Compare against the existing predictor, no-model/direct baseline and explicit-effect head.
4. **Horizon calibration:** score future belief errors by prediction horizon, action, entity, occlusion/contact event and in-/out-of-support strata.
5. **VIS-008 crucible:** Symtropy retains hidden world state and emits only sensory observations through the candidate path; include occlusion, object swaps, shadows/lighting, ego-motion, reflection, partial visibility and causal interactions.
6. **Closed-loop bridge:** connect only after offline forecasts qualify; execute actual actions through an authorized, safety-gated actuator path and score realized decision value separately from visual forecast fidelity.

## Graduation measures

Keep distinct:

- state/geometry forecast error by horizon;
- action-following error;
- effect-sign and effect-magnitude error;
- ranking agreement and regret across same-history branches;
- calibration and coverage of uncertainty/abstention;
- false-safe and termination underprediction;
- observation-memory contamination count (must remain zero);
- closed-loop utility lift over direct controls, net of measured inference cost.

A successful synthetic contract is not evidence that the trained Symthaea world model is accurate. A low visual forecast error is not evidence that action ranking or decisions improve. A positive counterfactual ranking result is not a physical safety proof.

## Reproduction

Run the CMAQ-018 crate's focused locked workflow and preserve its report/run receipt. Any actual VIS-005 adapter must bind its own candidate, model weights, observation/action schema, horizon configuration and evaluator artifacts through the shared frozen-run manifest schema.
