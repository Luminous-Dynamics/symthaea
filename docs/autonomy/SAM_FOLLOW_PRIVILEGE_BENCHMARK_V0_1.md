# SAM Follow Privilege-Separation Benchmark v0.1

Status: **specification only** — no campaign result is claimed.  
Owner: Symthaea SAM measurement / benchmark layer  
Tracking issue: [#11258](https://github.com/Luminous-Dynamics/symthaea/issues/11258)  
Composes: #11149, #11109, #10237, #10256, #10257, #10320, #10012; Symtropy #3387, #3471, #3532.

## 1. Purpose

A Follow policy can appear "teacher-free" while still consuming simulator-derived peer channels. A teacher action can be computed without blending or supervision. A zero-blend training step can still use teacher output as a BPTT target. A frozen weight matrix can still have recurrent hidden-state and FEP state evolution.

These are different causal paths. The benchmark measures them independently rather than compressing them into one label.

## 2. Privilege dimensions

| Dimension | Question | Required observable |
|---|---|---|
| Policy-observation privilege | What did the student actually observe before choosing? | Observation profile, source roles, exact policy-input digest |
| Action-blending privilege | Was a teacher action mixed into the command? | Blend flag, coefficient, raw and post-blend action digests |
| Learning/supervision privilege | Did teacher output influence gradients, BPTT, replay or distillation? | Supervision flag, update flag, replay and parameter digests |
| Evaluator privilege | What truth became available only after action commitment? | Evaluator-only source list, delivery/commit tick and access trace |
| Inference-state evolution | What changed despite no trainable-weight update? | Recurrent-state, normalization and FEP modulation flags/digests |

"Teacher-free" must never be used without naming which of these dimensions is disabled.

## 3. Experimental arms

- **A — legacy privileged:** existing simulator-derived observation channels and explicitly declared teacher blend/supervision configuration.
- **B — masked-observation teacher:** peer/oracle channels are masked from the student representation, while teacher blend and teacher supervision are independently toggled as separate arms.
- **C — no-teacher frozen:** no teacher action in policy inputs, action blending or learning targets; no optimizer update, replay update or exploration noise. Evaluator truth may be exposed only after the student action is committed for scoring or a shadow-reference calculation. Recurrent hidden-state evolution, state normalization and FEP state-dependent tau modulation remain active if the deployed inference path normally uses them; they must be reported, not silently disabled.
- **D — local-perception Follow:** eligible only after a qualified local metric tracked-actor adapter satisfies SAM scene admission #11109. Not available from simulator-only evidence.
- **E — cooperative advisory:** V2X remains a distinct remote claim. It cannot be treated as a local track or actuator authority.

Arms A–C are the initial simulation-only campaign. D/E are interface/qualification targets, not current experimental results.

## 4. Per-step evidence record

Each row must bind exact scenario, episode, actor-policy, controller, teacher, evaluator, dynamics, map, observation profile and configuration identities.

Required fields:

- policy observation profile, policy-input digest and delivered source references;
- source evidence role (local observation, simulator oracle, remote claim, map prior, derived estimate, unknown);
- teacher action computed, action blended and blend coefficient;
- teacher target used for BPTT/replay/distillation;
- evaluator-only and reference-only classification;
- optimizer/gradient update, replay insert/sample, exploration applied;
- controller parameter digest before/after;
- recurrent hidden-state digest/evolution, normalization evolution and FEP tau modulation;
- student raw proposal, teacher output, post-blend action and executed command as separate values;
- action commit tick, evaluator-truth delivery tick, scenario clock and source age;
- target availability, stale/missing/rejected/unknown disposition and exact denominators.

A teacher action computed but used neither in blending nor supervision is reference-only. A no-gradient-update inference path is not described as a static model if its recurrent or FEP state evolves.

## 5. Metrics

### Privilege and leakage

- **Oracle observation leakage:** among paired cases where only a simulator-oracle sidecar changes, fraction of local-only policy-input digests or student actions that change.
- Teacher blend rate and coefficient distribution.
- Teacher supervision rate, reported separately from blend rate.
- Evaluator-to-policy contamination: evaluator-only facts appearing in pre-commit inputs, action selection, replay or gradients.
- Frozen-weight invariance: exact parameter digest equality across the evaluation episode.
- Inference-state evolution: recurrent, normalization and FEP modulation changes, reported separately.
- Target availability and freshness: current, stale, missing, rejected and unknown counts/rates.

### Driving outcome

- Following-gap absolute error: median, p90, p95 and worst family.
- Minimum separation and TTC only under a declared simulator measurand / event definition.
- Contact events separately from model-relative avoidability classifications.
- Acceleration, braking, steering, jerk, comfort, lane deviation and route progress.
- Interventions, no-admissible-target events, abstentions and false interventions.
- p50/p95/p99 policy + FEP + RTA runtime, deadline misses and compute/memory.

Do not collapse these values into a universal safety score. Preserve family-level distributions, denominators and censored/incomplete trials.

## 6. Anti-leakage protocol

1. Freeze scenario, initial state, exogenous stream, actor policy, model, vehicle dynamics, maps and action budget.
2. Separate world/exogenous, actor-policy, student-policy, teacher, training and evaluator random streams.
3. Commit each student command before evaluator-only ground truth or shadow teacher output is exposed.
4. On local-only profiles, mutate simulator-oracle sidecars while holding local observations constant; policy-input digest and action must remain invariant.
5. Keep train/dev/holdout disjoint by scenario family and actor-policy lineage. Frames and sibling variants are not independent episodes.
6. Retain timeouts, unsupported runs, unavailable targets and censored outcomes. Never silently convert them to no-event or zero error.
7. Report matched-compute and best-attainable-compute tracks separately.
8. No claim of local perception is admitted from simulator-oracle or synthetic mesh observations.

## 7. Negative-control matrix

| Mutation | Required outcome |
|---|---|
| Simulator-oracle sidecar only changes under local-only profile | Same policy-input digest and action |
| Raw V2X claim only changes | No local-track or policy-input mutation before explicit qualified admission |
| Shadow teacher action is computed after action commit | Committed action unchanged |
| Teacher blend is zero but BPTT target is used | Supervision flag remains true |
| Teacher is computed but neither blended nor supervised | Reference-only, not teacher-supervised |
| Weights are frozen but recurrent/FEP state evolves | Weight digest stable; inference-state evolution recorded |
| All teacher, update, replay and exploration paths disabled | Weight and replay digests unchanged |
| Lead target missing or stale | No raw peer-distance fallback |
| Oracle target wrong-frame, wrong-scenario, future or stale | Reject |
| CSV header/row width differs | Fail the run/report |

## 8. Prospective acceptance targets

These are design targets, not achieved results:

- 100% pass of declared negative-control vectors.
- Zero evaluator-to-policy leakage on the paired intervention suite.
- Zero undeclared teacher-blend or teacher-supervision steps in Arm C.
- Exact trainable parameter digest equality before and after Arm C; recurrent/FEP state evolution reported separately.
- No Arm D claims before scene admission and local sensor evidence qualification.
- Every performance claim matched on scenario, action, compute and evidence privilege, then repeated on a sealed actor/failure-family slice.

## 9. Industry and benchmark interpretation

This is a SAM-developed internal experiment, not an external leaderboard. It complements rather than replaces NAVSIM v2 (pseudo-simulation planning), ReactSim-Bench (ego-deviation-conditioned reactive actor response), reciprocal Symtropy #3532, and field safety evidence. It specifically tests information privilege and training/evaluation leakage that aggregate planning scores do not fully expose.

## 10. Claim ceiling

A passing campaign establishes only that the named policy/teacher/evaluator pathways obeyed the declared simulation experiment protocol for the exact artifacts and scenarios. It does not establish local sensor perception quality, human behavioural realism, V2X security, physical vehicle safety, roadworthiness, public-road readiness or SAE Level 4/5 capability.
