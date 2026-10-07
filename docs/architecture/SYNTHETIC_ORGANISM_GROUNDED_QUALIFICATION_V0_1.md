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

### Planning quality

The horizon-aware policy already computes deterministic oracle horizon regret.

Lower regret means the selected action was closer to the best available action under the benchmark's
explicit utility function.

### Held-out transfer

The primary transfer comparison is:

trained_predictor_MAE < persistence_MAE

A model that only improves during online adaptation but does not transfer to a frozen scenario has
not yet demonstrated robust predictive structure.

## Interpretation matrix

| Observation | Interpretation |
|---|---|
| Training MAE falls, held-out MAE does not | adaptation may be memorizing scenario-specific dynamics |
| One-step MAE improves, continuous-rollout MAE remains high | the learned transition does not compose coherently over the temporal horizon |
| Held-out MAE improves, confidence ECE worsens | prediction improves but confidence is not calibrated |
| Held-out MAE improves, survival does not | predictive knowledge is not causally reaching useful action selection |
| Survival improves, oracle regret does not | hard-coded homeostatic biases may dominate the benefit |
| Regret improves, recovery worsens | planner may discover brittle short-term solutions |
| Recovery improves with held-out transfer | strongest current evidence that learned prediction is functionally coupled to regulation |
| Confidence rises on repeated bad predictions | confidence model has regressed; evidence quantity is overpowering accuracy |
| Prediction/observation identity checks fail | evidence chain is invalid; results must not qualify the architecture |

No single metric is a synthetic-organism detector.

## Required evidence discipline

A future green qualification should report all of:

1. exact repository commit;
2. exact scenario definitions and perturbation schedule;
3. training/frozen boundary;
4. model and policy configuration;
5. prediction accuracy;
6. confidence calibration;
7. survival and minimum viability margin;
8. perturbation recovery;
9. oracle horizon regret;
10. persistence comparator;
11. held-out transfer;
12. trace/invariant verification where action evidence is recorded.

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

The remaining qualification task is to measure:

**learned discrete prediction → continuous trajectory rollout → actual deterministic consequence**

with separate measurements for one-step model error, continuous extrapolation error, confidence
calibration, planning regret, survival, and recovery.

Only after that comparison is stable should the shared transition abstraction be considered for
runtime policy coupling or viability-driven modulation.
