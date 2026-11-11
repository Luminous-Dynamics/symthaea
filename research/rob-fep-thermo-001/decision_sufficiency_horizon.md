# Decision-sufficiency horizon: task-conditioned freshness for action commitment

Owner/composition: Symthaea #11581 (adaptive commitment horizon), #11573 (distributed inference), #11122 (EVSI), #9010 (world-model value-add), #11561 (real-time cognition), and Symtropy #4117/#4382/#4588 (certificate freshness, projection and observation-constrained future mobility).

This is an evaluation/reference definition. It does not create another belief engine, EVSI selector, clock authority, viability solver, scheduler or controller.

## Why age alone is not enough

Age of Information answers how long ago the source observation was generated. It does not answer whether acting on that observation is still a good decision.

- A 600 ms old observation of a static wall may still support exactly the same safe action.
- A 25 ms old observation of a fast crossing object may already be decision-insufficient.
- A fresh observation may have zero EVSI if every possible result leads to the same action.
- A sensor/model uncertainty increase can make the action's safety status Unknown even when the timestamp is very recent.

Therefore a fixed maximum observation age is a necessary engineering guard in some systems, but it is not a sufficient task-level measure of information usefulness.

## Definition

At observation cutoff t0, a policy selects an action or committed prefix a0. At a future application time t, let:
- b_t be the predicted belief over the task state given the information actually available at t;
- L_task(a,s) be the frozen task-specific loss;
- A_adm(t) be the independently admissible action set for the current state/evidence;
- C_t be the calibrated uncertainty/confidence set for the belief and loss prediction.

Define stale-action decision regret:

`R_dec(t) = E_{s~b_t}[L_task(a0,s)] - min_{a in A_adm(t)} E_{s~b_t}[L_task(a,s)]`

A conservative upper bound is computed over the declared confidence set:

`R_UCB(t) = sup_{(b,L) in C_t} R_dec(t)`

The stale decision remains *decision-sufficient* through t only if all of the following hold:

1. **Loss sufficiency:** `R_UCB(t) <= epsilon_task`, where epsilon_task is fixed in the declared task-loss units.
2. **Robust action persistence:** a0 remains in the independently defined robust admissible action set across the declared plausible state set. If the action is no longer common-admissible, reject it even if expected task regret is small.
3. **Evidence validity:** source identity, clock relation, model coverage and required certificate validity remain current at the actual application time.
4. **No hidden hard-bound violation:** the proposal still satisfies action-buffer, deadline, thermal, actuator and recovery constraints.

Define the decision-sufficiency horizon as:

`H_dec = sup { h >= 0 : conditions 1-4 hold for every t in [t0, t0+h] }`

Use a finite, qualified bound. If the predictive model is out-of-support, its uncertainty set is uncalibrated, action admissibility is Unknown, or a required upstream certificate is stale, H_dec is Unknown—not infinity and not a permissive default.

## Distinction from adjacent metrics

- **AoI:** elapsed time from observation generation to point of use.
- **AoA / PoMA:** actuation timeliness and probability/cost of missing a declared actuation window.
- **EVSI:** expected downstream task-loss reduction from acquiring a new observation now.
- **Decision-sufficiency horizon:** how long the current observation-conditioned action remains adequately good and robustly admissible under predicted evolution.
- **Command-to-motion delay:** physical response after the command is latched by the actuator.

These quantities are related but not interchangeable and must be reported separately.

A high AoI can coexist with a long H_dec in a stable scene. A low AoI can coexist with a short H_dec near a contact, collision, slip or thermal transition. A zero EVSI probe can coexist with a positive H_dec when every possible observation preserves the current decision.

## Composition with adaptive commitment horizon

Add H_dec as a distinct bound to the min-of-qualified-bounds commitment rule in #11581:

`N_commit = min(N_proposal, N_freshness, N_deadline, N_decision, N_thermal, N_actuator, N_dynamics, N_viability, N_queue, N_continuity)`

N_decision is derived from H_dec and the control period. If it is Unknown or unqualified, do not lengthen the commitment. The independent projection/RTA/controller path still decides whether any action may execute.

## Calibration and evaluation protocol

For each task/stratum:
- freeze the task-loss function and epsilon_task before test runs;
- calibrate R_UCB on training/validation data, then evaluate coverage and regret on held-out seeds and OOD strata;
- record the actual stale action, current best admissible action, realized task loss and whether the old action remained robustly admissible;
- report risk-coverage curves and the rate at which predicted-sufficient actions later become decision-insufficient;
- report per-task loss units; do not sum unrelated task losses without a validated common utility scale.

A nominal high confidence level is not a guarantee until empirical coverage is measured under the claimed deployment distribution. When coverage fails or the environment shifts, H_dec must contract or become Unknown.

## Synthetic known-answer corpus

- `fixtures/decision_sufficiency_v0_1.json`
- `fixtures/decision_sufficiency_v0_1.schema.json`

Cases include old-but-sufficient static scene, fresh-but-insufficient fast crossing hazard, expected-loss small but robust action no longer admissible, and OOD/un-calibrated prediction that must return Unknown.

## Proposed targets (not achieved)

- Zero commitment extension when H_dec is Unknown or model coverage is unqualified.
- Zero stale-action use when the old action leaves the robust common-admissible set.
- Empirical coverage of the declared R_UCB confidence level on held-out in-distribution strata, reported with confidence intervals.
- Report OOD/shifted strata separately and allow abstention/shortening rather than claiming calibrated coverage outside the envelope.
- Improve service/inference efficiency by avoiding unnecessary observations in stable decision-sufficient periods without worsening safe-success, recovery or hard-bound outcomes.

## Claim boundary

H_dec is a task- and model-conditioned predictive bound. It is not thermodynamic entropy, a clock-age measurement, EVSI, a formal control barrier certificate or actuation permission. It can only shorten/qualify an action commitment after its assumptions and calibration are tested; it cannot override #4117, #4382 or #2836.
