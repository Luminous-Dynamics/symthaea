# VIS-005 action-conditioned visual prediction: current-path audit — 2026-11-01

**Scope:** source-backed gap analysis of the existing CfC, JEPA, cross-manifold and delayed-horizon paths. This is an architecture audit and qualification protocol, not a claim that the current system already has an action-conditioned visual world model.

## Executive finding

Symthaea already has a useful delayed visual prediction evaluator. It compares predictions with later encoded visual observations and includes a persistence baseline, multiple horizons, lateness accounting, queue limits, expiry accounting and serialized pending work.

But the current path does not yet satisfy VIS-005's required interface:

    observed visual belief + hypothetical action -> predicted future visual belief

There are three distinct mechanisms today, and they must not be conflated:

1. **Vision manifold delayed forecasts:** autonomous CfC prediction from current visual input/state, scored later against the next real frame. No candidate action or action-manifest identity is bound to the forecast.
2. **CrossManifoldPredictor:** maps a current vision hypervector into a cognitive hypervector using a learned elementwise binding/tanh mapping. It is vision-to-cognition, not future visual-state prediction.
3. **JEPA training path:** consumes the previous FEP action index, but the current target vector is built from the CfC's own multi-scale prediction approximation. That is not an independently acquired future visual target. It can be a useful latent/self-distillation experiment, but it does not by itself establish action-conditioned prediction fidelity.

The joint action-bank API from this PR provides the evaluation-side uncertainty structure. A real action-conditioned visual predictor and same-history branch producer are still required.

## Source trace

### 1. Existing delayed-horizon visual forecast

In crates/domains/symthaea-vision-manifold/src/manifold.rs:

- observe_frame_impl updates the live CfC state, sets last_frame_hv, predicts the next frame, then calls DelayedHorizonEvaluator::observe(self, dt) once per accepted positive-duration frame (around lines 1470–1530).
- DelayedHorizonEvaluator::observe advances its modality clock, matures pending forecasts, scores each against the latest real last_frame_hv, compares against a persistence forecast, records error dispersion/lateness/sample counts, then issues new autonomous forecasts (around lines 4535–4740).
- Each new forecast is currently generated as manifold.predict_horizon(actual, horizon). That method computes a closed-form CfC jump from the current state/current input and horizon (around lines 1866–1873).
- The evaluator retains bounded pending queues, drops issue attempts on queue pressure, and expires forecasts whose actual observation is later than the configured maximum lateness. The default maximum lateness is four times the requested horizon.

This gives genuine delayed observation-grounded prediction evidence. It should remain a permanent baseline.

What is absent from the pending forecast record:
- a source-history/frame digest;
- a candidate action ID and canonical descriptor digest;
- a model/checkpoint digest and prediction-adapter version;
- a branch/fork identity;
- a common disturbance-tape identity;
- a replayable per-forecast issuance receipt.

The evaluator aggregates by horizon. It does not preserve a per-action branch identity or pair action alternatives from the same initial state.

### 2. Existing cross-manifold mapping is not VIS-005

In crates/domains/symthaea-vision-manifold/src/bridge.rs, CrossManifoldPredictor::predict_cognitive_checked accepts a current vision state and returns a predicted cognitive state. observe_cognitive_checked then receives the actual cognitive state and updates the mapping weights.

That is a useful cross-modal bridge, but it does not predict the next visual belief after a hypothetical action. Its name should not be used as evidence that VIS-005 is implemented.

### 3. Current JEPA action conditioning is not a future-observation target

In src/cognitive_loop/cycle_phase_dynamics/planning.rs around lines 523–550:

- current_hv is formed from the CfC input array;
- next_hv is formed from prediction.clone(), where prediction came from get_multi_scale_prediction;
- action is self.fep.last_action_idx, i.e. the previous cycle's action because CfC planning runs before the current FEP step;
- jepa.train_step(current_hv, next_hv, action, lr) updates the predictor.

This establishes an action-conditioned JEPA training path, but the visible target in this code is the model's predicted state approximation rather than a future frame encoded independently by the vision manifold. It must therefore not be presented as an externally grounded action-conditioned visual rollout until the target path is changed and independently evaluated.

## Timing and label-integrity issue

The delayed-horizon evaluator scores a matured forecast against the latest encoded frame once its due time has passed, subject to a lateness threshold. It records mean lateness but does not produce exact-time interpolation or a per-forecast actual-frame binding in the returned aggregate.

For a VIS-005 qualification panel, preserve the actual observation timestamp/frame sequence used to score each forecast. Do not silently score multiple requested horizons against one late observation and call them exact-horizon predictions. The existing lateness/expiry controls should remain; stricter exact-horizon evaluation should add a separate protocol rather than weakening current tolerance behavior.

## Minimal production architecture for VIS-005

### A. Separate the predictor contract from the evaluator contract

A future producer API should accept:

- an immutable observed belief snapshot and its digest;
- a canonical action manifest and action-space version;
- the hypothetical action ID and exact action descriptor;
- forecast horizon / target time;
- predictor/checkpoint/config identity;
- a specified posterior/noise draw identity if stochastic prediction is used.

It should return a typed PredictedVisualBelief, never an Observed frame. The predicted belief should expose the task-relevant state variables the model actually predicts (object hypotheses, pose/geometry, velocity, contact, visibility/occlusion, confidence/uncertainty), and may additionally carry a rendered latent or image forecast. Do not require an image decoder if the task model predicts structured belief more faithfully.

### B. Use Symtropy to produce real same-history alternative branches

For each frozen history:
1. clone the exact hidden simulator state into a branch bank;
2. freeze the candidate action manifest and disturbance tape;
3. ask Symthaea for a predicted future visual belief for each candidate action;
4. seal prediction outputs before revealing any future frame or evaluator label;
5. execute each action in an independent clone, using the declared shared exogenous disturbance policy where structurally valid;
6. capture the realized future visual observation/state in evaluator-only storage;
7. score predicted-versus-realized branch outcomes and pairwise action effects.

A real camera observation cannot reveal all unchosen counterfactual branches. Use genuinely randomized matched interventions or keep unchosen outcomes marked unavailable. Do not infer them from observational logs alone.

### C. Preserve the epistemic boundary

The predictor receives only the starting observed history, action, model state and allowed draw specification. It never receives:
- hidden simulator truth;
- unchosen branch outcomes;
- future camera frames;
- contact/success/terminal evaluator labels;
- target-state data from the scoring step.

A predicted frame must never be written into observation memory as though it had been captured. Future observations enter evaluator scoring or later training only after the forecast has been sealed and issued.

### D. Keep the current delayed evaluator as the autonomous control

Retain:
- current autonomous CfC forecast;
- persistence baseline;
- current delayed horizon handling;
- current JEPA path as its own separately named comparison until a future-observation target is verified.

Add explicit arms:
- A0: persistence / last-frame baseline;
- A1: current autonomous CfC visual forecast;
- A2: action-conditioned CfC with actual action input and future-observation target;
- A3: explicit action-effect head;
- A4: joint-draw action-bank predictor / uncertainty-aware selection;
- A5: JEPA using a sealed future-observation target (only after target/temporal alignment is proven);
- controls: shuffled action labels, swapped action descriptors, no-action, independently sampled/no-coupling draws, and evaluator oracle.

### E. Join to this PR's joint action bank

The existing JointActionSetSamples provides the per-channel shared-draw structure for candidate action sets. For visual belief, use a declared bank per structured scalar channel or a typed vector-valued extension with explicit metric semantics; do not collapse object geometry, contact and utility into an arbitrary scalar.

The content digest binds exact sample values and action/draw IDs. It does not establish that the predictor honestly used the declared common latent/disturbance coupling. That requires independent retained producer artifacts and a verifier.

## Required qualification gates

1. **Action sensitivity:** action-swap and zero-action controls demonstrate that predictions change for causal reasons, not from action labels alone.
2. **Future-observation supervision:** training targets are encodings of actual later observations, not the predictor's own output.
3. **Temporal alignment:** every forecast receipt identifies the exact future observation used for its score and measured lateness.
4. **Same-history counterfactual fidelity:** each alternative action begins from one identical state/history digest.
5. **Joint uncertainty:** candidate branches preserve valid shared draw identity; marginal intervals are not subtracted to create effects.
6. **Selection reliability:** report simultaneous coverage, selected regret-bound coverage and false-dominance certificates after action selection.
7. **Decision value:** W1/W2/W3 must beat D0 and D1 on matched episodes after all inference/acquisition/failure costs.
8. **Provenance:** retain exact predictor checkpoint, source commit, action manifest, branch/reset receipt, disturbance tape and per-forecast outcome receipt.
9. **Distribution shift:** report action reversal, hidden regime, contact/termination seam and unsupported action strata separately.
10. **No actuation authority:** the predictor and all certificates remain proposals; any physical actuation still requires the independent authority/safety gate.

## Paradigm-shift thesis

The key shift is from "predict what the world will do" to **"predict how the world changes under each possible intervention, with a calibrated distribution over those differences, then verify whether choosing from that distribution improves outcomes at equal total cost."**

That is a falsifiable scientific contribution. It connects visual representation, action-conditioned prediction, counterfactual truth, selection-aware uncertainty, abstention and actual decision value without confusing one with another.

## Claim ceiling

The existing delayed-horizon evaluator establishes observation-grounded autonomous forecast metrics. The current JEPA path uses a prior action but its target is the CfC prediction approximation. The current cross-manifold predictor maps vision to cognition. The joint action-bank PR supplies uncertainty/paired-effect metrics and selection/value evaluators. None of these facts alone establishes a production action-conditioned visual world model, calibrated counterfactual predictions, SOTA, or physical safety.


## Structured belief output contract update

The joint action-bank interface now supports more than a scalar utility. The structured counterpart preserves independently named geometry, motion, contact, visibility, terminal-risk and task-utility channels under one frozen history/action/draw identity. This is the appropriate seam for future VIS-005 outputs such as tracked object position/velocity, contact likelihood and visibility belief. Each channel retains its own unit and paired-effect interval; heterogeneous channels are never pooled until a separate, versioned task-utility mapping is applied. The content digest binds the exact channel descriptors and f64 samples but still does not prove the predictor honestly used common noise.
