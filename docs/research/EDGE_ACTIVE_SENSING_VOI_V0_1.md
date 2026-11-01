# Decision-focused active sensing and value-of-information planner v0.1

Tracking: #9817  
Implementation branch: `research/edge-active-sensing-voi-v0.1`  
Parent: [real-data provenance PR #9815](https://github.com/Luminous-Dynamics/symthaea/pull/9815), stacked on [edge temporal bridge PR #9808](https://github.com/Luminous-Dynamics/symthaea/pull/9808).

## Objective

Most fixed-stream TinyML benchmarks ask how well a model predicts from the observations it receives. A genuinely active embedded intelligence system also chooses **when to acquire the next observation**. Its sensing policy can save sensor wake energy and compute, but can miss short-lived events if it sleeps too long.

This module adds a small, deterministic decision-focused value-of-information (VoI) planner. It evaluates the expected reduction in task decision loss caused by a possible next observation. It does not use entropy reduction as the objective.

## Core distinction: information versus decision value

For action (a), the planner receives a finite distribution over possible observation outcomes. Each outcome has:
- its probability in parts per million;
- the candidate's estimated optimal task decision loss after incorporating that outcome.

Expected post-observation loss:

`L_after(a) = Σ_o P(o | a) × L*(o)`

Decision value:

`VoI_task(a) = L_before - L_after(a)`

A high-entropy observation can still have zero task VoI. For example, learning the exact colour of an irrelevant background object might reduce state entropy but not change the alert/actuation decision. Such an action is rejected if its expected task loss is not reduced.

This is a model-based planning estimate. The task loss, observation probability distribution and posterior losses must themselves be calibrated and evaluated against held-out outcomes. This v0.1 module does not prove those estimates are statistically calibrated.

## Resource model

Each action declares a six-dimensional incremental cost vector:

- sensor energy (nJ);
- sensor wake energy (nJ);
- compute cycles;
- latency (µs);
- transmitted bytes;
- memory-write bytes.

The action is rejected if **any** component exceeds the remaining budget. The planner does not exchange joules for cycles, bytes for latency or memory writes for radio energy.

Among actions with positive expected task-loss reduction and component-wise feasible cost, it returns a non-dominated frontier in:

`expected task-loss reduction × complete resource-cost vector`

The application can select an action from that frontier using its own preregistered utility/safety policy. If a scalar utility is needed, the conversion weights and thresholds must be explicit application parameters, not hidden global benchmark weights.

## Action types

The initial enum covers:
- sample now;
- activate another modality;
- increase sampling rate;
- decrease sampling rate;
- transmit a summary;
- wait or sleep.

A policy can submit an action estimate for each feasible next step. Wait/sleep may consume idle energy and is not assumed to be a zero-cost action.

## Receipt and reproducibility

The content-addressed plan receipt binds:
- current expected task loss;
- complete remaining resource budget;
- sorted non-dominated action set;
- each action kind, cost vector and claimed entropy reduction (diagnostic only);
- probability/posterior-loss distribution used to calculate expected loss;
- reproduced expected loss before/after and expected task-loss reduction;
- rejected action IDs/reasons;
- canonical plan digest.

Receipt validation recomputes each frontier action's expected task loss from the stored outcome distribution and checks resource feasibility, sorted/unique action IDs, no duplicate IDs across accepted/rejected lists, absence of dominated frontier actions and canonical digest.

The receipt is a planning artifact, not proof that an observation was actually acquired. Actual sample timestamps, sensor state, input bytes, action execution, power trace, event quality and device session need separate candidate/system receipts.

## Tests authored

Six synthetic tests cover:
1. a high-entropy but decision-irrelevant observation is rejected (zero task VoI);
2. a decision-relevant observation enters the frontier with an exact expected-loss reduction;
3. incomparable resource-cost trade-offs are both retained;
4. an action exceeding one resource component is rejected;
5. malformed outcome probabilities and unsorted actions fail closed;
6. tampering with a plan receipt is rejected.

Tests are authored but have not been compiled or executed. No hardware action or physical energy measurement has been run.

## Next experiment

Use one exact hidden-state task with both rare, short-lived events and long negative idle intervals. Compare:
- fixed sampling at 1/5/10/25/50 Hz;
- variance-triggered sampling;
- threshold/hysteresis with cooldown;
- uncertainty-triggered sampling;
- entropy-maximizing sampling;
- decision-focused VoI sampling;
- HDC-only, CfC-only and HDC-LTC stateful policies.

Give every policy the same initial sensor state, same observations available at requested times, same task deadlines, quality target and six-component budget. Report event recall/precision, false alarms per hour, worst-case detection delay, samples/hour, wake count, update energy, transmission bytes, and measured complete-event energy on real hardware.

A critical counterfactual boundary applies: replaying a high-rate recording does not magically reveal sensor values at times the policy did not sample. Offline simulation must use either:
- a frozen high-rate reference trace as a source of available measurements at the policy-requested sample times, explicitly labelled an emulated sensor/acquisition model; or
- a new physical run where the actual sensor follows the policy and records every acquired value.

Do not interpolate hidden values and present them as observed measurements.

## Claim ceiling

- This is a deterministic expected-task-loss planner, not a Bayesian posterior guarantee.
- The input outcome distribution/posterior losses are model estimates and need calibration.
- A Pareto frontier is not a selected action until an application utility/safety rule chooses one.
- Synthetic action-cost estimates are not measured energy.
- No sampling reduction, improved detection, battery-life or SOTA result is claimed.
