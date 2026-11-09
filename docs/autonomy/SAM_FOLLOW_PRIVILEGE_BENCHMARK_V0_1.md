# SAM Follow Privilege-Separation Contract v0.1

Status: **specification only; no trial results**.  
Owner: Symthaea SAM measurement / benchmark layer.  
Tracking issue: [#11258](https://github.com/Luminous-Dynamics/symthaea/issues/11258).  
Implementation context: [#11186](https://github.com/Luminous-Dynamics/symthaea/pull/11186), [#11213](https://github.com/Luminous-Dynamics/symthaea/pull/11213), scene admission [#11109](https://github.com/Luminous-Dynamics/symthaea/issues/11109).

## 1. Why this exists

A Follow policy can appear "teacher-free" while still consuming simulator-derived peer channels. A teacher action can be computed without blending or supervision. A zero-blend training step can still use teacher output as a BPTT target. A frozen weight matrix can still have recurrent hidden-state and FEP state evolution.

These are distinct causal paths. The benchmark must measure them independently instead of compressing them into one `teacher_free`, `oracle_free`, `frozen` or `sensor_only` label.

## 2. Orthogonal privilege axes

| Axis | Question | Required evidence |
|---|---|---|
| Policy observation | What did the student receive before deciding? | Observation profile, source roles and exact input digest |
| Action blending | Did a teacher action influence the command? | Computed/blended flags, coefficient and action digests |
| Learning/supervision | Did teacher output influence BPTT, gradients or replay? | Supervision, update and replay flags; parameter digests |
| Evaluator truth | When did ground truth become available? | Action commit tick, evaluator delivery tick and source IDs |
| Inference-state evolution | What changes without weight updates? | Recurrent-state digests, normalization and FEP modulation |

The no-teacher/frozen arm C deliberately uses `simulation_20d_with_sim_peer_channels` in this specification: it disables teacher action paths and learning updates, but it does **not** pretend that simulated input channels are local sensor data. A future local-perception arm D remains blocked pending scene-admission qualification.

## 3. Required arm set

- **A — legacy privileged:** existing synthetic Follow curriculum, with each privilege factor measured separately.
- **B — masked-observation teacher:** simulated peer sidecar is masked from the student; teacher blending and supervision remain independent factors.
- **C — no-teacher frozen:** no teacher action in policy or gradients; no optimizer/replay/exploration updates. Recurrent/FEP state may evolve as in normal inference. Evaluator truth is post-commit only.
- **D — local-perception Follow:** not eligible until a local metric tracked-actor adapter is admitted by #11109.
- **E — cooperative advisory:** remote V2X is a distinct claim, never local perception or actuator authority.

## 4. Machine-checkable trial record

The companion JSON Schema defines:
- exact trial/episode/scenario/actor-policy/arm IDs;
- observation profile, source-role list and policy-input digest;
- teacher action computed, blended, supervised, evaluation-only and reference-only flags;
- optimizer, replay and exploration flags;
- parameter digest before/after and explicit weight-change flag;
- recurrent-state digest before/after, plus normalization/FEP evolution;
- student raw, post-blend and executed action digests;
- action observation/commit ticks;
- evaluator-truth availability, delivery tick and source IDs;
- Follow target availability, source role, age, frame and disposition.

Every trial also binds a controller digest and a paired-case identity. Paired reports have one baseline and one intervention with the same episode, scenario, actor policy, controller, observation tick and action-commit tick. The supported paired interventions are:

- `oracle_sidecar_mutation` on masked-observation Arm B: simulator-oracle evidence changes while policy-input and raw-student-action digests remain identical.
- `raw_v2x_mutation` on Arm E: the remote claim changes while the local tracked Follow target remains byte-identical.
- `post_commit_shadow_teacher` on Arm C: an evaluator/reference-only teacher is added after action commitment without changing the executed-action digest.

Standalone trials use `paired_role=standalone`, a null paired-case ID and `intervention_kind=none`.

The canonical fixture intentionally has `benchmark_status=specification_only` and `trials=[]`. It is not a campaign result.

## 5. Semantic validation requirements

Beyond JSON Schema, the validator checks:

1. Each trial names a declared arm; arm IDs are unique.
2. No source appears twice in one observation list.
3. Simulator-oracle or remote cooperative claims are not policy-visible under a local-only observation profile.
4. A local-perception Follow target cites a local sensor source; a simulator benchmark target cites simulator-oracle source. A remote V2X claim can never satisfy either.
5. Teacher blending requires a computed action and a positive blend coefficient; no blending requires coefficient zero.
6. Teacher supervision requires a computed teacher action.
7. A reference-only teacher was computed, but was neither blended nor supervised.
8. An evaluation-only teacher is reference-only and cannot influence the committed student action.
9. Evaluator truth marked available must be delivered strictly after the student action commit and must not be policy-visible.
10. Frozen/no-teacher arm C has no blend, supervision, optimizer, replay or exploration update; parameter digests remain equal. Recurrent-state evolution is allowed and separately reported.
11. Weight-change and recurrent-evolution flags agree with before/after digest equality.
12. Missing/stale/rejected Follow target evidence remains explicit; no simulator or V2X source is silently relabeled as local perception.

## 6. Outcome metrics

Report privilege/leakage metrics separately from driving metrics:
- oracle-observation leakage rate;
- teacher blend and supervision rates separately;
- evaluator-to-policy contamination rate;
- exact frozen-weight digest invariance;
- recurrent-state evolution and FEP-state ablations;
- valid/stale/missing/rejected target rates and target age;
- gap error p50/p90/p95 and worst-family error;
- minimum separation/TTC only under a declared measurand contract;
- contact and avoidability as different outcomes;
- comfort, route progress, interventions and no-admissible-target events;
- p50/p95/p99 latency and deadline misses.

Do not combine these into a universal safety score. Preserve trial denominators and incomplete/censored outcomes.

## 7. Negative-control campaign

1. Change only simulator-oracle sidecars under local-only profile: policy-input digest and action stay identical.
2. Change only raw V2X claims: local track state remains unchanged before explicit admission.
3. Compute shadow teacher after action commitment: committed action cannot change.
4. Zero blend but use BPTT: supervision remains true.
5. Compute unused teacher: reference-only true.
6. Freeze weights while recurrent/FEP state evolves: parameter digest unchanged; state evolution recorded.
7. Disable teacher, updates, replay and exploration: parameter/replay digests unchanged.
8. Missing/stale target: no raw peer-distance fallback.
9. Wrong frame/scenario, future or stale oracle: reject.
10. CSV header/row width must match exactly.

## 8. Claim ceiling

This contract measures declared privilege and information-flow semantics in a named simulation experiment. It does not establish local perception quality, human behavioral realism, V2X security, physical vehicle safety, roadworthiness, public-road readiness or SAE Level 4/5 capability.
