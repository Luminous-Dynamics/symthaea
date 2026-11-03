# SAM Counterfactual Crossing Oracle v0.1

**Protocol ID:** `SAM-CROSSING-ORACLE-001`  
**Status:** analytic research fixture; no simulator/physical validation claimed  
**Date:** 2026-11-03  
**Owners:** Symthaea model/decision evaluation; Symtropy scenario execution #3387 and interactive actors #3471  
**Composes:** Symthaea #10257, #10266, #10278, #10043, #10012, #6283; Symtropy #3385, #3387, #3471

## 1. Objective

Create the smallest end-to-end falsification experiment that demonstrates:

1. A world model can be wrong in a decision-relevant way.
2. An optimizer can select the action favored by that error.
3. An independently specified reference oracle can show a different outcome.
4. The evaluator detects ranking reversal, regret and model optimism.
5. Missing, censored or altered reference evidence cannot silently preserve the apparent result.
6. The counterexample can be replayed and promoted into a permanent regression.

This is an analytic one-dimensional crossing problem. It is intentionally smaller than a road simulator so its reference outcome can be calculated independently and exactly. It is an R0 analytic oracle under NIST-style Tier 3 counterfactual evaluation, not a physical collision model or a human-behavior model.

## 2. Coordinate system and initial conditions

Use SI units and a single monotonic simulation clock.

| Quantity | Symbol | Value |
|---|---:|---:|
| Ego initial longitudinal position | (x_0) | 0 m |
| Ego conflict-point coordinate | (x_c) | 20 m |
| Ego initial speed | (v_0) | 10 m/s |
| Ego vehicle length | (L_e) | 4 m |
| Pedestrian crossing width | (W_p) | 4 m |
| Pedestrian lateral speed (reference) | (v_p) | 1.4 m/s |
| Ego braking magnitude | (a_b) | 3 m/s² |
| Ego restart acceleration | (a_a) | 2 m/s² |
| Pedestrian crossing start | (t_0) | 0 s |

The ego path and pedestrian path intersect at one conflict point. The pedestrian occupies the conflict region from (t=0) until the full crossing width has been traversed. The ego occupies the conflict region while its center is within (L_e/2) of (x_c).

The reference pedestrian clearance time is:

`T_p = W_p / v_p = 4 / 1.4 = 20/7 = 2.857142857... s`

This constant-speed actor is a deliberately narrow analytic reference. It is not a population model, a behavioral prior or a claim about pedestrian reaction.

## 3. Candidate ego actions and reference outcomes

### Action A — continue

The ego maintains 10 m/s.

`T_continue = (x_c - x_0) / v_0 = 2.0 s`

Its conflict-occupancy interval is:

`[T_continue - L_e/(2v_0), T_continue + L_e/(2v_0)] = [1.8, 2.2] s`

This overlaps the pedestrian occupancy interval `[0, 2.857142857...]`. The reference oracle therefore labels the action as conflict-overlap.

### Action B — yield

The ego brakes at 3 m/s² until stationary.

`T_stop = v_0/a_b = 10/3 = 3.333333... s`

`D_stop = v_0²/(2a_b) = 50/3 = 16.666666... m`

The ego stops 3.333333... m before the conflict point. The pedestrian has already cleared at 2.857142857... s. The ego then accelerates from rest at 2 m/s²:

`D_remaining = 20 - 50/3 = 10/3 m`

`T_accel = sqrt(2 D_remaining/a_a) = sqrt(10/3) = 1.825741858... s`

`T_yield = T_stop + T_accel = 5.159075191... s`

At the conflict point its speed is approximately 3.65148 m/s. Its occupancy interval using the 4 m vehicle length is approximately `[4.61135, 5.70680] s`, after pedestrian clearance. The reference oracle labels this action as no conflict-overlap.

This action profile is a prescribed control primitive in the analytic experiment; it is not a validated braking controller.

## 4. Deliberately flawed predictor and action-selection exploit

The predictor is intentionally wrong about pedestrian crossing speed:

`v_p,predicted = 8/3 = 2.666666... m/s`

It therefore predicts pedestrian clearance at 1.5 s. The ego continue occupancy starts at 1.8 s, so the predictor forecasts no overlap for Continue even though the reference oracle detects overlap.

To isolate decision-ranking exploitation from any unimplemented planner, this fixture injects the following frozen planner outputs:

| Action | Predicted utility | Reference utility |
|---|---:|---:|
| Continue | 10 | 2 |
| Yield | 7 | 8 |

The reference utility is deliberately transparent:

`U = 10 - λ max(0, T_arrival - 2) - 8 I(overlap)`

where `I(overlap)` is 1 when the ego/pedestrian occupancy intervals overlap and 0 otherwise, and:

`λ = 2 / (T_yield - T_continue) ≈ 0.6330966750`

Thus:
- Continue reference utility: `10 - 8 = 2`.
- Yield reference utility: `10 - λ(T_yield - 2) = 8`.

The planner selects Continue because its predicted utility is 10 rather than 7. The independent reference prefers Yield.

Expected decision audit:

| Output | Expected value |
|---|---:|
| Predicted-best action | Continue |
| Reference-best action | Yield |
| Ranking reversal | true |
| Selected action | Continue |
| Selected-action regret | 6 utility units |
| Predicted selected-action advantage over Yield | +3 |
| Reference selected-action advantage over Yield | -6 |
| Optimism gap | 9 utility units |

The values are analytic fixture outputs, not observed vehicle performance. Utility units are synthetic and must not be relabeled as safety, collision probability, severity or physical risk.

## 5. Machine-readable golden fixture

The corresponding JSON fixture is [`fixtures/SAM_CROSSING_ORACLE_V0_1.json`](fixtures/SAM_CROSSING_ORACLE_V0_1.json). It records the SI-unit inputs, predictor bias, reference intervals, synthetic utilities, expected decision audit and digest-helper policy. The equations in this document remain the normative derivation; the JSON is the machine-readable expected-output companion.

## 6. Expected evaluator input contract

Use `symthaea_vehicle::model_discrepancy` from PR #10276.

Bind:
- Predictor model/config ID and artifact digest.
- Independent analytic reference ID and implementation digest.
- Evaluator ID and artifact digest.
- Planner ID and artifact digest.
- Sampling profile and protocol digest.
- Frozen expected residual slots and `expected_schedule_artifact_digest`.
- One observed residual row for each declared expected slot.
- One decision with both candidate actions and unique command/execution/evaluator receipts.
- `selection_commit_sequence < reference_outcome_reveal_sequence`.
- Exact scenario, environment, dynamics, action-history and observation-availability digests.

Use the public helpers for expected-schedule, action-set and decision-context digests. Do not duplicate their serialization logic in the fixture generator.

The decision's candidate set is complete for this finite two-action analytic problem. This makes regret exact over the declared set, not over every physically possible control trajectory.

## 7. Falsification and negative controls

Each control must change the expected report in the specified way:

1. **Perfect predictor:** use the reference crossing speed in the predictor. The continue prediction must show overlap; no claim of model discrepancy remains.
2. **No-overlap counterfactual:** use a pedestrian speed that clears before the ego's front enters the conflict region. The reference outcome must change consistently with the declared occupancy rule.
3. **Action-label permutation:** swap labels but retain command digests and outcomes; ranking follows the actual bound actions, not array order.
4. **Action-set mutation:** change one command digest without rebuilding the action-set digest; reject.
5. **Reference mutation:** change one realized utility or execution digest while retaining the old report/evidence identity; reject or produce a distinct report identity.
6. **Outcome leakage:** set reference reveal sequence equal to or earlier than selection commit; reject.
7. **Missing candidate reference:** set one candidate to Missing/Censored/TimedOut/Unsupported with no fabricated utility or receipt; decision becomes IncompleteReference and has no ranking/regret.
8. **Missing expected horizon:** omit one scheduled residual outcome; reject the report.
9. **Unexpected horizon:** add a row not present in the frozen schedule; reject the report.
10. **Missing intermediate horizon:** retain an explicit missing slot between two observed horizons; do not calculate a bridged paired-growth result.
11. **Context mutation:** change initial state, environment, dynamics, actor policy, observation availability or action history in one row; reject the pair.
12. **Order permutation:** reorder inputs and actions without changing identity; canonical report digest remains invariant.
13. **Numeric extremes:** reject non-finite residuals and overflowed utility ranges.

The test corpus must retain both positive controls and negative controls. A test that only proves rejection is insufficient: legitimate complete analytic inputs must produce the exact expected report.

## 8. Extension to ego-action-conditioned actors

The analytic oracle above uses a prescribed crossing occupancy. The next layer is Symtropy #3471's synthetic interactive actor adapter.

The actor policy must receive only:
- A timestamped allowed observation view.
- Its own current actor state and declared policy/config.
- A declared exogenous input stream.

It must not receive hidden ego controller state, future commands, privileged simulator labels or future world state.

For the first causal actor extension, compare two ego histories (Continue and Yield) against the same actor policy, initial state and exogenous stream. Actor response may differ only after the ego-action difference is observable and after the declared reaction delay. Keep the fixed crossing script as a separately labeled control.

The actor adapter should output a bounded kinematic target applied by the existing Symtropy Kinematic body path; it must not create a new physics engine or duplicate vehicle dynamics.

## 9. Research and industry benchmark positioning

This oracle is not a substitute for public benchmarks. It is the analytic gate before larger benchmark execution.

**Leaderboard comparability cautions:** the surfaced AV2, WOSAC, Waymo E2E and Bench2Drive rows above are third-party leaderboard snapshots, not all official live challenge results. The Bench2Drive community leaderboard explicitly says its v0.0.3 results are self-reported from publications, mix different data/training/inference budgets and include privileged ground-truth-perception methods in a separate section. For Waymo, its official site says 2026 formal challenges are not being hosted, while public leaderboards remain active. Pin exact task, dataset, evaluator, sensor privilege, compute budget and submission protocol before any comparison.

| Benchmark | Current external reference | What SAM should measure beyond headline score |
|---|---|---|
| Argoverse 2 multi-agent motion forecasting test | SOTA2 surfaced IMR 0.92 minADE K=1 and QCNeXt 0.50 minADE K=6 (board updated about 3 months ago) | minADE/minFDE, miss rate, Brier score, calibration and downstream action regret |
| Waymo Open Sim Agents Challenge 2025 | SMART-R1 realism 78.58; kinematics 49.44; interactivity 81.10; map adherence 92.01 (third-party board updated about 7 months ago) | Keep realism dimensions separate; verify exact official evaluator/version before claiming comparability |
| Fail2Drive | 200 paired routes, 17 shift classes; paper reports 22.8% average success-rate degradation for evaluated models | Paired ID/OOD degradation, family worst-case, failure taxonomy and confidence intervals |
| Waymo E2E Driving Challenge | SOTA2 surfaced OpenEMMA 2026.05 at 6.3162 overall RFS; construction/intersection/pedestrian/cyclist RFS dimensions are separately listed; board updated about 4 months ago | RFS direction and evaluator semantics must be frozen from the official protocol; do not infer operational safety from the score |
| Bench2Drive 220-route official track | SOTA2 surfaced LVDrive 2026.05 at 80.71 Overall Success Rate (DS), with SR 58.26; board updated about 4 months ago | Use the exact 220-route task and scoring; compare route completion, success, efficiency, comfort and maneuver-specific outcomes separately |
| CARLA Leaderboard 2.1 | Linear infraction penalty; early-stop reward exploit explicitly addressed | Route completion and each infraction family; frozen route/route-phase denominator; no 2.0/2.1 score mixing |
| NIST Measurement Science for Automated Vehicles | Four-tier progression: surrogate metrics, outcome prediction, counterfactual decisions, cross-scenario diagnosis | Use as SAM's decision-measurement architecture, not as a claim of NIST endorsement or certification |

Waymo states that it is not hosting formal challenges in 2026 while its public leaderboards remain active. Treat the WOSAC 2025 board as a dated reference snapshot, not as a verified live November 2026 challenge result.

Sources:
- https://www.sota2.com/research/sota/multi-agent-motion-forecasting-on-argoverse-2-av2-test
- https://www.sota2.com/research/sota/motion-simulation-on-waymo-open-sim-agents-challenge-2025
- https://waymo.com/open/challenges/
- https://arxiv.org/abs/2604.08535
- https://leaderboard.carla.org/
- https://www.sota2.com/research/sota/end-to-end-driving-on-waymo-e2e-driving-challenge-leaderboard
- https://www.sota2.com/research/sota/end-to-end-autonomous-driving-on-bench2drive-220-routes-official
- https://github.com/autonomousvision/Bench2Drive-Leaderboard
- https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0

## 10. Milestones and acceptance

### M0 — analytic oracle
- Independent implementation of the equations above.
- Exact expected occupancy intervals and utilities.
- All negative controls in section 6.
- Independent recomputation of the report from raw fixture rows.

### M1 — discrepancy evaluator
- #10276 exact-head compile/test.
- Exact expected residual schedule and digest.
- Decision ranking reversal, regret 6 and optimism gap 9.
- Missing/censored/out-of-schedule controls.

### M2 — deterministic execution
- Symtropy #3387 executes both candidate profiles under one frozen scenario manifest.
- Exact same-runtime replay reproduces actor and ego traces.
- Stable scenario/actor IDs are independent of runtime BodyHandle allocation.

### M3 — interactive actor
- Symtropy #3471's future-command, hidden-state, occlusion, stale-observation and reaction-delay controls.
- Two synthetic actor policies with causal, replayable response differences.
- No human-behavior or physical-fidelity claim.

### M4 — benchmark bridge
- Promote minimized analytic counterexamples to permanent regressions.
- Run Fail2Drive paired shift profiles and CARLA 2.1 only with exact rights/version/protocol.
- Run WOSAC-style agent realism only after implementing its actual task contract.
- Run AV2 only for a comparable forecasting task and declared input budget.

### Prospective targets (not results)
- 100% pass on the named analytic positive/negative controls.
- Zero accepted missing expected slots, duplicate receipts or outcome leakage.
- Zero cases where missing/censored/unsupported outcomes are emitted as zero or success.
- At least 25% relative reduction in paired OOD success degradation versus a frozen, faithfully reproduced baseline, with no more than 1 percentage-point ID success regression.
- At least 10% lower median and p95 finite-set regret versus a frozen baseline at matched compute, with no increase in infeasibility or deadline misses.

These are research targets, not current capabilities, safety thresholds or certification criteria.

### Industry architecture reference: NVIDIA Alpamayo

NVIDIA Alpamayo 2 Super is an important industry reference, but not a directly comparable benchmark entry. NVIDIA describes a 34B reasoning VLA, the AlpaSim closed-loop simulator, AlpaGym closed-loop RL, causal reasoning traces and open driving datasets. NVIDIA-reported results include 0.911 m minADE6 at 6.4 s on 1,434 challenging samples, reasoning score 0.433, LingoQA 79.2, and AlpaSim score 1.50 ± 0.13 on 913 reconstructed scenes. These numbers belong to different tasks/data/protocols and must not be compared as one score.

Strategic implication: SAM should not try to beat a 34B camera-language-action foundation model at raw perception with a small analytic/HDC controller. Its differentiated research target is evidence qualification: expose model error, detect action-ranking exploitation, preserve uncertainty/missingness and produce independently recomputable counterfactual decision audits. A later teacher/student or external-VLA integration can compare decision quality at matched observations, action spaces and compute.

Reference: https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/

## 11. Claim ceiling

A passing M0/M1 result establishes that the specified analytic example and evaluator detect this particular model-induced ranking reversal under the declared assumptions. M2/M3 adds deterministic synthetic execution and causal actor response only. M4 adds benchmark-specific results only.

No stage by itself establishes physical collision risk, human behavioral fidelity, roadworthiness, operational crash reduction, ISO 26262 / ISO 21448 qualification, public-road driving authority or SAE Level 4/5 capability.
