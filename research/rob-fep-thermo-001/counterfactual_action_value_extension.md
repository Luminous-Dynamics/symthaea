# Counterfactual action prediction and decision value for thermodynamic robotics

This addendum composes existing owners; it does not create a new world model, simulator, causal-inference engine or action-selection benchmark.

Existing owners:
- #598 — object-factorized action-conditioned rigid dynamics;
- #664 — causal intervention/effect qualification;
- #9010 — matched world-model value-add against direct policies;
- #10625 — repeated held-out continuations and denoised selectable headroom;
- #8764 — action-coupled closed-loop evaluation;
- #11122 — decision-focused EVSI;
- #11496 — FEP/thermodynamic robotics;
- #11561 — real-time cognition;
- #11581 — adaptive commitment horizon.

## Why this matters

A model can predict the most likely future image while failing to identify which candidate action actually causes the best outcome. Ordinary one-step prediction error does not establish:
- correct action-conditioned transition modeling;
- counterfactual action ranking;
- expected action value;
- decision improvement after compute/latency costs;
- safe/recoverable execution.

OpenWAM (October 2026) provides a useful composable comparison: a shared causal robot-video foundation supports video-then-action, action-then-video, joint and decoupled generation, plus independently trained local inverse/forward dynamics. Its LIBERO results report 98.6% mean success for VTA across four suites; bimanual real-world success averages 92.1% for VTA and 91.9% for Joint across its three tasks. More directly relevant, counterfactual supervision reduces forward-dynamics RGB error by 34.5% and increases correct outcome identification among 16 same-state action outcomes from 21.1% to 71.3% in its reported evaluation. https://openwam.stanford.edu/

These are OpenWAM's benchmark-specific results, not evidence that Symthaea has a comparable video model or that pixel prediction accuracy is decision value.

## Four different qualification questions

1. **Prediction:** Does the model forecast a future observation/state under a supplied action sequence?
2. **Action-outcome identification:** Given K candidate actions from the same initial context, can it identify which predicted/observed outcome corresponds to each action?
3. **Decision ranking:** Does the model rank candidate actions by their expected downstream task value, not merely by visual similarity?
4. **Closed-loop decision value:** Does access to the model improve held-out task outcomes after charging inference, observation, action-application delay, energy and safety constraints?

Keep these metrics separate. A predictor can pass #1 and fail #2–#4.

## Same-history, action-coupled experiment

For each frozen pre-action state/history:
- generate a candidate action bank before seeing evaluation outcomes;
- fork every action from the same exact initial-state receipt;
- preserve action-specific state/observation evolution; do not force all policies to share one realized observation stream;
- use common exogenous random numbers only where the causal model justifies invariance across the candidate actions;
- repeat stochastic continuations for every candidate;
- use separate continuation sets for candidate selection, selector development/calibration and held-out evaluation;
- prevent the candidate/model from receiving hidden simulator state, future counterfactual observations or answer keys.

This composes #8764 and #10625. One lucky rollout per candidate must not be mistaken for expected action value.

## Baseline and ablation matrix

Where data and task domains permit, compare:
- direct policy without a world model;
- direct policy with the same memory and observation budget;
- demo-only action-conditioned dynamics;
- counterfactual-only dynamics;
- mixed demonstration/counterfactual training;
- OpenWAM-style video-then-action, action-then-video, joint and decoupled interaction programs;
- independently trained local inverse and forward dynamics;
- Symthaea FEP and corrected action-conditioned predictive models;
- shuffled-action, shuffled-outcome, stale-model and observation-only negative controls;
- evaluator-only oracle action values, never deployable.

Use exact same task, initial state, candidate action bank and evaluation continuation set within paired comparisons. Keep incompatible datasets and embodied platforms in separate result strata.

## Required metrics

Report:
- video/state prediction error;
- action-conditioned transition calibration;
- top-1 and top-k action-outcome identification among K same-state branches;
- pairwise ranking accuracy and rank correlation;
- expected-value calibration by candidate action;
- one-shot best-of-K estimate versus repeated-continuation held-out value;
- held-out action-selection regret;
- decision lift versus direct policy at matched model-query/compute budgets;
- latency, memory, inference energy, observation age and action expiry;
- safe-success, hard constraint violations, adverse outcomes and future recovery options.

The episode/context is the statistical unit. Candidate branches and control ticks from one initial state are not independent samples.

## Integration with FEP and adaptive commitment horizon

Symthaea's generative model should expose an action-conditioned distribution over outcomes for candidate actions, with uncertainty and support. FEP/EVSI can use those distributions to value information or compare candidate policies. The action-horizon selector can propose a commitment prefix, but #11581 still clips it against freshness, deadline, thermal, actuator, dynamics, viability, queue and continuity caps.

A prediction that is accurate at computation start must be evaluated at actual action-application time. If additional counterfactual rollout makes the proposal stale or consumes too much thermal/battery reserve, its net decision value can be negative.

## Proposed target (not achieved)

A model earns a role in the robot only if it improves held-out action ranking and closed-loop decision value over matched direct-policy and model-corruption baselines, after charging inference/latency/resource costs, with no adverse safety or recovery regression. Report cases where counterfactual modeling provides no value or increases regret.
