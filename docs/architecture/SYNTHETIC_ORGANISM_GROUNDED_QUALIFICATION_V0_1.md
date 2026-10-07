# Synthetic Organism — Grounded World-Model Qualification v0.1

## Purpose

This experiment tests whether the existing cognitive-loop world model can participate in a causal
prediction/action/consequence loop inside a deterministic environment.

It is intentionally narrower than a consciousness or artificial-life claim.

The qualification target is:

**predict → act → observe consequence → quantify error → regulate/select → learn → repeat**

The experiment is only considered useful when each edge is executable and independently scoreable.

## Model under test

The model under test is the current FepModule::world_model implementation.

The qualification harness clones that exact world-model state before training. It never resets or
re-trains the production world model in place.

The clone uses the existing action-conditioned WorldModelBridge implementation and is evaluated
against the existing deterministic viability_micro_world transition oracle.

The native FEP ODE trajectory planner is deliberately not replaced or silently coupled to this
bridge in v0.1. That separation is itself an experimental variable.

## Experimental phases

### A. Adaptation

Train the copied action-conditioned world model on the nominal scenario.

For each cycle:

1. observe the deterministic state;
2. predict the next state for the scheduled action;
3. execute the oracle transition;
4. observe the realized next state;
5. update the action-conditioned model.

The training score is one-step mean absolute error (MAE).

### B. Frozen transfer

Evaluate the trained model on the stressed scenario without calling the learning hook.

The stressed scenario contains deterministic threat and integrity perturbations.

The following must remain frozen during scoring:

- model parameters;
- confidence state;
- action schedule;
- scenario definition.

This prevents held-out transfer from becoming another online-learning score.

### C. Closed-loop resilience

Use the same trained model as the predictor for the existing horizon-aware homeostatic policy.

The environment executes the selected action, including deterministic perturbations.

Online model updates are permitted in this phase only.

This phase measures whether prediction actually helps a policy survive and recover rather than merely
improving a forecast statistic.

### D. Comparator

Run the same closed-loop policy with the persistence predictor.

The comparator uses the same actuator boundary and the same termination semantics, so an apparent
gain cannot come from giving one predictor a more permissive execution path.

This is a deliberately weak control. It is not an optimal controller and should not be described as
one.

A meaningful model contribution should improve at least one policy-level quantity without silently
changing the policy's hard-coded viability objective.

## Metrics

### Prediction accuracy

For each transition:

MAE = mean(|predicted_channel - observed_channel|)

The state channels are bounded to [0,1], so MAE is itself bounded.

### Confidence calibration

The bridge's scalar confidence is compared with realized continuous forecast accuracy:

realized_accuracy = 1 - MAE

The harness reports:

- sample count;
- ten-bin expected calibration error (ECE);
- mean squared confidence/accuracy error;
- horizon-adjusted confidence inside counterfactual planning.

Confidence must not be treated as constant across a long imagined trajectory. Recent world-model
evaluation work likewise recommends separating one-step and multi-step rollout accuracy and treating
uncertainty calibration as decision-relevant rather than decorative telemetry.

These are operational calibration metrics for the forecast-confidence signal. They are not evidence
that the confidence is a probability of a discrete event.

### Survival

A run survives only when the final environment state remains viable **and** the policy loop did not
terminate because the deterministic actuator rejected an action:

energy > 0.08 && integrity > 0.08 && no_execution_failure_termination

The minimum actually observed viability margin is also retained.

Rejected actions are closed as explicit prediction cancellations with execution-failure evidence;
they are never converted into successful outcomes.

### Recovery

For every perturbation actually reached, recovery is the number of cycles required to regain the
exact pre-perturbation viability margin.

None means that margin was not recovered before the run ended.

This prevents a system from receiving a recovery score merely for remaining above the absolute
death boundary.

### Continuous temporal extrapolation

The shared transition-model interface can be extended through the existing Dormand-Prince engine.

The qualification harness rolls a frozen action-conditioned model for 0.5 seconds with tau = 0.1,
then compares its terminal state with five repeated deterministic oracle transitions from the same
starting state.

This metric answers a different question from one-step MAE:

**does a model that predicts one consequence also compose that prediction coherently over time?**

A large gap between one-step MAE and continuous-rollout MAE is evidence of temporal model mismatch,
even when the one-step predictor looks strong.

### Multi-horizon profile

The qualification harness evaluates a frozen model at 1, 2, 4, and 8 environment steps.

Each horizon reports:

- mean one-step error from the same visited states;
- terminal error from repeatedly applying the frozen **discrete** predictor;
- terminal error from the continuous ODE rollout;
- the continuous terminal-to-one-step error ratio.

The discrete terminal score is the direct multi-step world-model test. The continuous score is a
separate test of whether the transition law admits a coherent continuous relaxation. The two are
not treated as mathematically interchangeable. In particular, the ODE adapter's relaxation does
not claim to reproduce a discrete transition exactly at one time constant.

### Policy-induced distribution shift

A second frozen evaluation lets the model choose its own actions with the horizon-aware policy.

The environment advances using those chosen actions, but the model is not updated during scoring.
The resulting predictor MAE is compared with the fixed-schedule held-out MAE.

A rising ratio indicates that the model becomes less reliable on states induced by its own policy.
This directly tests a control-specific failure mode: a planner can move the environment into states that
were rare or absent during training.

The same run separately calibrates confidence only on planner-selected actions. This is intentionally
stricter than aggregate held-out calibration because selection can preferentially expose overconfident
model errors.

### Prediction-error adaptation response

The qualification layer now includes an isolated adaptation-response experiment on the stressed scenario.

For each deterministic perturbation shock:

1. capture the perturbed state and scheduled action;
2. score the shock transition before learning;
3. score a nearby state/action probe before learning;
4. apply exactly one observed transition update to an isolated model clone;
5. re-score the identical shock transition;
6. re-score the nearby probe without any further update.

The report distinguishes:

- one auditable receipt per perturbation, including cycle, action, and exact shock-state digest;
- mean shock MAE before and after the update;
- same-transition improvement and improvement rate;
- neighboring-probe MAE before and after the update;
- neighboring-probe improvement and improvement rate;
- previously learned anchor MAE before and after the update;
- anchor regression and regression rate.

This separates three increasingly strong claims:

**error detected** → **the exact error can be corrected** → **the correction transfers to a nearby state**.

It also tests a fourth property:

**adaptation without damaging previously learned dynamics**.

The nearby probe is deliberately not part of the update. Improvement there is therefore stronger evidence of learned
local dynamics than simply replaying the exact transition used for the update. The anchor is also not part of the
update; worsening on it is evidence of adaptation-induced forgetting.

A negative or zero improvement is valid evidence too. The benchmark must not assume that adaptation succeeds.

### Procedurally generated held-out transfer

The qualification layer also evaluates a deterministic procedural family that is never used for
adaptation.

Eight explicit seeds generate scenarios by varying:

- initial internal/external state;
- six-action schedules;
- two perturbation events, including event type, timing, and magnitude.

The transition law itself is held fixed. Therefore this test measures **generalization across
unseen state/schedule/perturbation configurations**, not zero-shot discovery of a new environment
dynamics law.

Every generated scenario has its own manifest digest, and the family has an aggregate manifest
digest. Each scenario is scored with a fresh clone of the trained model so that no generated
evaluation can mutate another fold.

The report retains:

- generated scenario count;
- mean and worst improvement over the persistence baseline;
- fraction of scenarios beating persistence;
- survival rate;
- the complete procedural-family manifest digest.

A strong result here means the learned transition structure transfers beyond the four authored
scenarios. It does not establish generalization to arbitrary environments or unseen physics.

### Environment-level counterfactual queries

The frozen qualification layer now includes a deterministic query bank inspired by
environment-level world-model evaluation.

For each scenario, it creates four probe states:

- the exact scenario initial state;
- a threat-spike intervention;
- an integrity-damage intervention;
- an energy-drain intervention.

From each probe state it evaluates every length-three action sequence (4 × 6³ = 864 queries per
scenario). The model must answer the entire sequence by repeatedly composing its frozen transition
prediction; the deterministic oracle answers the same query through the real transition function.

The report measures:

- total, valid, and invalid query counts plus sequence length;
- mean path MAE across all intermediate query steps, over valid answers only;
- mean changed-channel F1 across intermediate steps, identifying whether the model predicts the
  variables whose values actually change;
- mean terminal-state MAE over valid answers only;
- mean minimum-viability-margin error over valid answers only;
- survival-decision agreement across all queries.

The query bank is only scoreable when at least one valid answer exists. An all-invalid query population
therefore cannot masquerade as a valid zero-error result.

Invalid/non-finite model answers are explicit disagreements, remain in the total query count, and
are never converted into synthetic error values. This prevents invalid forecasts from both
inflating and deflating the numerical error metrics.

This is intentionally different from scheduled-trajectory replay: it asks the model to answer
counterfactual intervention queries from states it did not receive as its normal scheduled path.
That makes the benchmark a stronger test of whether the learned transition model supports a usable
environment model rather than only memorizing observed action/state pairs.

The changed-channel F1 is a shortcut-resistance metric. A predictor that largely copies persistent
state can obtain deceptively good full-state overlap when only a few fields change; it should score
poorly when it fails to identify the fields whose values changed because of the intervention.

### Planning quality

The horizon-aware policy already computes deterministic oracle horizon regret.

Lower regret means the selected action was closer to the best available action under the benchmark's
explicit utility function.

### Action ranking and exploitation

The frozen held-out evaluator compares the model's one-step action ordering against the deterministic
oracle ordering using both top-1 agreement and pairwise agreement.

It also reports an **exploitation gap**: the mean model-assigned advantage on strict pairwise preference
inversions. A high value means the model is not merely inaccurate; it is assigning strong preference
to an action whose true consequence is worse than the alternative.

This matters because model-based planners can actively search for precisely those model errors. The
metric is therefore a negative-control/evidence metric, not a claim that exploitation is eliminated.

### Held-out transfer

The primary transfer comparison is:

trained_predictor_MAE < persistence_MAE

A model that only improves during online adaptation but does not transfer to a frozen scenario has
not yet demonstrated robust predictive structure.

### Leave-one-scenario-out transfer

The four-scenario benchmark is no longer treated as a single train/test split.

The qualification layer performs a complete leave-one-scenario-out matrix:

- start every train → held-out pair from its own clone of the same exact pre-qualification world-model state;
- adapt on exactly one scenario;
- freeze the model;
- score every other scenario without learning;
- record the training and held-out scenario manifest digests for every fold.

For four scenarios this yields 12 train → held-out transfer folds.

The report retains:

- mean improvement over persistence across all folds;
- worst-fold improvement;
- fraction of held-out folds that beat persistence;
- the stable digest of the complete scenario family.

This matters because a single favorable held-out scenario can be explained by scenario-specific familiarity, initialization, or perturbation alignment. The leave-one-out matrix tests whether predictive structure transfers across the benchmark family rather than merely across one chosen split.

## Interpretation matrix

| Observation | Interpretation |
|---|---|
| Training MAE falls, held-out MAE does not | adaptation may be memorizing scenario-specific dynamics |
| One-step MAE improves, continuous-rollout MAE remains high | the learned transition does not compose coherently over the temporal horizon |
| Multi-horizon terminal error grows rapidly with horizon | the model has short-range predictive skill without reliable long-horizon dynamics |
| Policy-induced error ratio rises above fixed-schedule error | the model is vulnerable to its own policy-induced distribution shift |
| Planner-selected calibration is worse than aggregate calibration | the model becomes overconfident on the states/actions its policy prefers |
| Action ranking improves but exploitation gap remains high | the model's score ordering contains strong, exploitable preference inversions |
| Held-out MAE improves, confidence ECE worsens | prediction improves but confidence is not calibrated |
| Held-out MAE improves, survival does not | predictive knowledge is not causally reaching useful action selection |
| Survival improves, oracle regret does not | hard-coded homeostatic biases may dominate the benefit |
| Regret improves, recovery worsens | planner may discover brittle short-term solutions |
| Recovery improves with held-out transfer | strongest current evidence that learned prediction is functionally coupled to regulation |
| Confidence rises on repeated bad predictions | confidence model has regressed; evidence quantity is overpowering accuracy |
| Prediction/observation identity checks fail | evidence chain is invalid; results must not qualify the architecture |
| Leave-one-out transfer varies sharply by held-out scenario | generalization is scenario-dependent; report the worst fold rather than only the mean |
| Mean transfer improves but one or more folds regress | the learned model has useful structure with a remaining environment-specific blind spot |
| All leave-one-out folds beat persistence | stronger evidence of cross-scenario predictive structure, still limited to the benchmark family |
| Counterfactual query error is high while scheduled-trajectory error is low | the model may fit observed trajectories without supporting broader environment-level reasoning |
| Counterfactual survival agreement is poor | the model's composed predictions are not reliable enough to support viability judgments away from the observed path |
| Counterfactual path error is high while terminal error is modest | intermediate model dynamics may be wrong even when endpoint error partly cancels |
| Changed-channel F1 is low while terminal overlap is high | the model may be copying persistent state while missing the causal variables altered by the action |
| The query bank has zero valid answers | the world-model interface is not scoreable; do not interpret zero numerical error as success |
| Procedural held-out transfer regresses while authored transfer passes | the model may be specialized to the authored scenario family |
| Procedural held-out transfer passes across all seeds | stronger evidence of generalization across sampled state/schedule/perturbation configurations, still bounded to this generator |
| Same-shock learning improves but neighboring probes do not | adaptation may be memorizing observed transitions rather than learning transferable local dynamics |
| Neighboring-probe improvement follows a shock update | stronger evidence that prediction error changes the model in a locally useful way |
| Anchor error worsens after shock adaptation | adaptation is causing measurable regression/forgetting of previously learned dynamics |
| Shock and neighbor improve while anchor remains stable | strongest current local evidence for useful, non-destructive adaptation |
| Shock error does not improve after the update | the observed error is not yet producing effective model correction |

No single metric is a synthetic-organism detector.

## Required evidence discipline

A future green qualification should report all of:

1. exact repository commit;
2. exact scenario definitions and perturbation schedule;
3. frozen benchmark-manifest digest;
4. training/frozen boundary;
4. model and policy configuration;
5. one-step and multi-horizon temporal prediction accuracy;
6. policy-induced distribution-shift error;
7. aggregate and planner-selected confidence calibration;
8. action-ranking agreement and exploitation gap;
9. survival and minimum viability margin;
10. perturbation recovery;
11. oracle horizon regret;
12. persistence comparator;
13. held-out transfer;
14. leave-one-scenario-out transfer matrix and worst-fold result;
15. environment-level counterfactual query-bank results, including path error and changed-channel F1;
16. procedurally generated held-out transfer results and procedural-family manifest digest;
17. prediction-error adaptation response, including per-shock receipts, same-transition correction,
neighboring-probe transfer, and anchor regression;
18. trace/invariant verification where action evidence is recorded.

Queued CI is not a pass.

## Current architectural gate

The branch now includes an experimental common transition-model interface shared by:

- the action-conditioned WorldModelBridge;
- the FEP generative transition model;
- the existing Dormand-Prince ODE engine through a continuous adapter.

The continuous extension is explicit:

ds/dt = (F(s,a) - s) / tau

For the WorldModelBridge delta model this becomes a constant action-specific velocity. It is an
experimental numerical extension, not a claim that this is the unique or biologically correct
continuous-time realization.

Counterfactual confidence is also horizon-aware: each deeper simulated step receives the prior
step confidence multiplied by a fixed 0.85 decay factor. This is a conservative engineering prior,
not a learned probability law.

The current qualification gate is empirical execution of the complete protocol:

**learned discrete prediction → continuous trajectory rollout → actual deterministic consequence**

with separate measurements for one-step error, discrete multi-step error, continuous extrapolation
error, policy-induced distribution-shift error, confidence calibration, planning regret, survival,
recovery, the completed leave-one-scenario-out transfer matrix, and procedurally generated held-out
transfer.

A dedicated CI job runs the qualification module directly. The branch should not be considered
qualified until that job has completed successfully for the exact candidate commit; queued or
pending execution is not evidence.

Only after that comparison is stable should the shared transition abstraction be considered for
runtime policy coupling or viability-driven modulation.
