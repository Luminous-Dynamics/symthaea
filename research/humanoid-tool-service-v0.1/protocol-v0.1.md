# HUM-TOOL-SERVICE-001 — Benchmark protocol v0.1

**Issue:** #12657  
**Evidence status:** prospective protocol; no task execution claimed  
**Initial scope:** simulation / synthetic fixtures / passive or low-energy bench articles

## 1. Research questions

Primary: can a compositional, evidence-aware architecture improve tool-use and service-task outcomes under held-out tool, task, environment and embodiment conditions?

Secondary:
- Does representing a held tool as part of the robot's temporary body schema improve reach, contact and recovery?
- Does a functional/causal tool model transfer better than object-label or trajectory retrieval?
- Do tactile/contact forecasts improve contact-state detection and recovery over matched vision-only and raw-tactile baselines?
- Does active probing reduce wrong-tool or wrong-contact errors enough to justify its latency and energy?
- Can a robot complete a service loop and independently prove the repaired system meets its declared acceptance test?
- Does the same skill contract transfer across different embodiments once each adapter is separately qualified?

## 2. External anchor: HumanoidToolBench

Paper: Jang et al., *HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution*, 1 October 2026. https://arxiv.org/abs/2610.02089

### 2.1 Exact upstream and rights snapshot

- Official evaluation repository: https://github.com/SNU-PI/HumanoidToolBench
- Pinned source snapshot for this protocol: `SNU-PI/HumanoidToolBench@3fec9bd36933a1648ff7f95468c983f3452243a0` (observed 17 November 2026).
- Evaluation-toolkit code: MIT, per the upstream `LICENSE`.
- ToolBook recordings: CC BY-NC 4.0. Upstream publishes separate simulation and physical-recording revisions:
  - Simulation: `8b2cd31e107b64cb13f812ea217a63a20845c78a`.
  - Real G1: `47eca9322bb53fa1c685363271a87d2e414cb0e8`.
- Third-party models, weights, meshes and assets retain their own terms. The MIT code licence does not relicense these assets or the ToolBook dataset.

**Commercial-use boundary:** ToolBook is non-commercial research data. Do not use its recordings or derived training artefacts in a commercial product or commercial training pipeline without a separately reviewed permission/licence. The MIT evaluator can be studied or adapted subject to preserving its licence and auditing bundled third-party components. Luminous commercial deployment must be kept on a separate rights-cleared data/model path.

### 2.2 Benchmark matrix and task semantics

The published matrix contains:
- Scenarios: BallMove (long stick pushes a ball), BallRetrieve (hook pulls a ball), IceBreak (hammer impact breaks two ice blocks).
- Levels: L0 correct tool selection and pickup; L1 stationary use; L2 mobile use after repositioning along the table.
- Modes: S (standard) and R (decoy). The paper uses D for the decoy mode; upstream environment IDs use R.
- Total: 3 × 3 × 2 = 18 conditions.
- ToolBook: approximately 3.1k simulation and real Unitree G1 demonstrations.
- Success is held for 50 consecutive controller steps (1 second) at 50 Hz. L0 requires lifting the correct tool at least 0.08 m. L1/L2 require the scenario-specific physical task predicate; simply picking up or touching a tool is not task completion.

Official environment IDs follow `humanoidtoolbench/G1<Scenario>-L<level>-<S|R>`, e.g. `G1BallMove-L2-R`. The case manifest records exact IDs and the S/R code token separately from its human-readable `standard`/`decoy` label.

### 2.3 Exact evaluator interface: not a torque-vector API

The official public evaluator simulates a **Unitree G1** in MuJoCo and uses a supplied whole-body controller. It runs a policy through an HTTP interface. The public policy contract is:
- Observation: 32-value state plus one egocentric RGB image and task instruction.
- Action: 36-value `decoupled_v1` action chunk, one row per 50 Hz control step.
- Action slots: 14 hand targets, 14 arm targets, 3 torso targets, base height, then 4 navigation values.
- One action chunk is executed before the next policy query (unless the episode terminates).

The 32-value policy state is not the raw 43-value recorded G1 observation. The real-hardware recordings also have a separate 35-value supervised action contract; the documented real-hardware control semantics differ from the simulation contract, including waist targets and navigation yaw. Never pad, truncate or reorder arrays and call the result an adapter.

This is materially different from Symthaea's current DMC21 / Dexterous53 morphology and its inspected 21-joint PCA9685 HAL path. A valid comparison needs a named, typed G1-to-Symthaea observation/action adapter with explicit semantic joint mapping, units, frames, base/navigation semantics, timing, reset/history behaviour and failure handling. Until that adapter is qualified, Symthaea's G1 benchmark result is **Unavailable / Incompatible**, not zero and not a matched direct comparison.

### 2.4 Official evaluation budget and compute reality

The upstream protocol specifies:
- 100 episodes per condition;
- seeds 10000–10099;
- up to 3000 policy-control steps per episode;
- 50 Hz policy control;
- MuJoCo 3.11.0 and CPython 3.10 in the pinned toolkit.

Maintainers report approximately 4 GB setup download, 9 GB disk use, an additional ~2 GB first policy-checkpoint download, about two hours for one 100-episode condition with the shipped ACT checkpoint on one workstation GPU, and roughly a day and a half / 15 GB for all 18 conditions sequentially. These are upstream estimates, not measurements on Luminous hardware.

Use staged execution:
1. Install/asset verification and a one-episode/100-step diagnostic.
2. Check camera/EGL correctness and validate the policy adapter.
3. Run one fully reportable 100-episode condition.
4. Expand to all 18 only after the first condition and resource budget are sound.

The toolkit supports Mesa software-rendered smoke tests, but these are diagnostics, not benchmark scores. If the system cannot support the pinned runtime, record the blocker rather than silently substituting a different simulator, asset set, controller or action interface.

### 2.5 Paper-reported simulation comparison

Scenario order in each cell is BallMove / BallRetrieve / IceBreak. These percentages are paper-reported, not locally reproduced:

| Policy | Level | Standard | Decoy |
|---|---|---|---|
| GR00T N1.7 | L0 | 77 / 73 / 88 | 67 / 59 / 75 |
| GR00T N1.7 | L1 | 64 / 36 / 66 | 58 / 30 / 63 |
| GR00T N1.7 | L2 | 10 / 9 / 31 | 30 / 5 / 48 |
| FastWAM | L0 | 64 / 62 / 98 | 74 / 45 / 93 |
| FastWAM | L1 | 76 / 57 / 86 | 75 / 45 / 82 |
| FastWAM | L2 | 9 / 20 / 69 | 8 / 11 / 61 |

These values are not a single overall score and must not be pooled across unlike task conditions. They show that success at tool selection or stationary use does not imply success at mobile tool use.

The paper's real-robot evaluation is a narrow selected subset: three policies, selected L1 tasks, and ten trials per condition. It also reports one training run per policy, so training-seed variability is not estimated. A Symthaea research reproduction must disclose this limitation and, where computationally feasible, use at least five independent training seeds.

Official source documentation:
- Repository README and licence: https://github.com/SNU-PI/HumanoidToolBench
- Environment/task definitions: https://github.com/SNU-PI/HumanoidToolBench/blob/main/docs/ENVIRONMENTS.md
- Public policy interface and execution budget: https://github.com/SNU-PI/HumanoidToolBench/blob/main/docs/PUBLIC_EVALUATION.md
- ToolBook data, revisions and rights: https://github.com/SNU-PI/HumanoidToolBench/blob/main/data/README.md

## 3. Current industry reference points

| System / result | Publicly reported evidence | Proper comparison |
|---|---|---|
| Figure Helix 2.5 | Figure reports 56% whole-task success vs 9% from scratch across three behaviours in 30 unseen homes. | Household whole-body generalization; company report, not a tool-use score. |
| Gemini Robotics 2.5 + Apollo 2 | Google reports 200+ evaluated tasks and a 96% automotive kitting pilot with Mercedes-Benz. | Task-specific industrial kitting; vendor report, not universal dexterity. |
| GR00T N1.7 | HumanoidToolBench includes GR00T N1.7 in its tool-use evaluation. | Direct paper comparator where exact checkpoint/runtime compatibility permits. |
| BEHAVIOR-1K | 1,000 household activities and 50 scenes. | Long-horizon household activity track; do not mix with tool benchmark percentages. |

Figure source: https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization  
Google DeepMind source: https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/  
HumanoidToolBench: https://arxiv.org/abs/2610.02089  
BEHAVIOR: https://behavior.stanford.edu/

Vendor results are vendor-reported. Exact checkpoints, evaluation splits, task criteria and trial counts must be rechecked when freezing an external comparator. No proprietary Figure or Gemini checkpoint availability is assumed.

## 3A. Complementary dexterity and tactile-world-model baselines

HumanoidToolBench is the humanoid-specific tool-use anchor, but it does not fully isolate functional transfer or tactile prediction. Include these complementary tracks where task and hardware semantics match:

| Research | Reported scope/result | What Symthaea should test |
|---|---|---|
| Evo-DexBench (15 Oct 2026) | 30 tasks across six functional families; 1,248 multimodal demonstrations using simulated Shadow Hands with 123 taxels; best reported method averages 38.5% success. | Core-to-Gen transfer: train on related functional interactions, then evaluate a novel task in the same family with few target demonstrations. Measure functional transfer rather than trajectory memorization. |
| DexTouch-WM (17 Sep 2026) | Keeps five hours of robot supervision fixed while scaling compatible human tactile interaction from 0 to 100 hours; reports improved held-out visual, geometric and contact prediction. | Test action-conditioned tactile forecasting and sensor-topology transfer. Treat the shared tactile layout and action retargeting as explicit prerequisites, not a free transfer assumption. |
| PhysCanvas (20 Oct 2026) | On four real manipulation tasks, reports world-model/real-success correlation increasing from 0.602 to 0.885 with state-reward supervision; reports π0.5 real success increasing from 59.1% SFT to 70.8% after world-model post-training. | Require state-consistent predictions: predicted contact, object motion, tactile state and task outcome must agree. Evaluate whether simulated rollouts predict real policy ranking, not just visually plausible frames. |
| Tabero (ICML 2026) | Multimodal gentle manipulation benchmark; reports over 70% reduction in average grip force under gentle instructions while maintaining task success. | Add physical-interaction-quality measures alongside task success: force/pressure envelope, deformation, slip, stability and correct force modulation. Do not equate this result with humanoid tool performance. |
| World Models for Robotic Manipulation survey (31 Aug 2026) | Organizes world models by predicted future representation, action conditioning and use in the learning/control pipeline; highlights contact modeling, hallucination control, action alignment and closed-loop evaluation as open challenges. | Audit prediction fidelity and downstream decision value separately. Do not use generated rollouts as ground truth or allow a reward model to certify its own predictions. |

Sources:
- Evo-DexBench: https://arxiv.org/abs/2610.21180
- DexTouch-WM: https://arxiv.org/abs/2609.20649
- PhysCanvas: https://arxiv.org/abs/2610.28613
- Tabero (PMLR 306, ICML 2026): https://proceedings.mlr.press/v306/wu26cb.html
- World Models for Robotic Manipulation: https://doi.org/10.1002/smb2.70053

These are paper-reported reference results, not Symthaea results and not numerically comparable across task suites. Confirm each paper's released source, checkpoint, dataset rights and exact evaluation before attempting reproduction.

### Resulting benchmark portfolio

Keep five distinct scorecards:
1. Humanoid tool selection → stationary use → mobile use.
2. Dexterous functional transfer (Core-to-Gen).
3. Tactile/contact prediction and post-contact recovery.
4. Gentle/quality-aware manipulation.
5. Verified service restoration and operational utility.

A generalist claim requires evidence across the relevant scorecards; success on one cannot compensate for failure on another.

## 3B. Agentic manipulation, persistent memory and motion-latent comparators

A major 6–17 November 2026 development is the emergence of **agentic manipulation policies (AMPs)**: a generalist model stays in the perception/action loop and revises its plan from outcomes, instead of selecting only from a fixed library of pretrained skills.

### AMP1 — direct agent-as-policy comparator

Chen et al., *On Agentic Manipulation Policies*, arXiv:2611.09892 (first submitted 6 Nov, revised 9 Nov 2026):
- AMP1 reports 85.6% zero-shot success on RoboLab-120 versus 49.2% and 42.9% for cited VLA/WAM comparators.
- Its real-world test uses a 7-DoF Franka Research 3, Robotiq 2F-85 and three cameras—not a humanoid. It reports 104/108 success (96.3%) over 12 tasks × 9 trials, compared with 49 for FLUX 3 Action and 30 for π0.5.
- Without its recovery-from-error functions, it reports 94/108 successes (87.0%). Physical failures were often collision/planning-related, including table/wall/self-collision, calibration sensitivity, and simplified robot geometry.
- The memory-augmented AMP1+ reports RoboDojo bimanual success rising from 26.4% zero-shot to 57.6% after one pass through 42 tasks. The authors report an approximately 4.6× reduction in failure rate on those bimanual tasks versus zero-shot deployment.
- AMP1 can build reusable procedures, exploit active perception, use objects as tools, infer hidden geometry from failed interactions, and revise plans. Its harness gives the model a persistent Python interpreter and a robot SDK.

Paper: https://arxiv.org/abs/2611.09892  
Project / trial details: https://apple-aiml-research.github.io/ml-amp/

**Important caveats:** the paper reports that model identity/reasoning effort matters much more than modest harness changes; it also documents benchmark/verifier artefacts. For example, some nominal failures were caused by objects outside the gripper envelope or a flaky verifier. On the 95 RoboLab tasks without the identified benchmark artefacts, Gemini 4 Argon + AMP1 achieved 96.2% in that paper's protocol. These are paper-reported research results, not an independent replication or humanoid result.

### RoboDojo — broader sim/real and memory benchmark

RoboDojo (42 simulation tasks and 18 real tasks across three non-humanoid robot platforms) is a complementary benchmark for general manipulation, long horizon, precision, generalization and memory. It must remain a separate scorecard from HumanoidToolBench.

Public leaderboard snapshot observed 17 November 2026:
- Standard simulation policy board: WALL-OSS-1.0-lite listed at 42.59 average score / 34.97 success rate.
- Separate Agent track: NVIDIA SOLVE* listed at 79.50 / 74.96. The asterisk means this entry was tested before the Agent Track rules published 14 November 2026 and is marked non-conforming with at least one updated rule.
- Public layouts are supplemented with hidden verification layouts in the official protocol.

Leaderboard: https://robodojo-benchmark.com/leaderboard?bench=sim  
Integrity protocol: https://robodojo-benchmark.com/leaderboard/protocol?lang=en

Do not compare the Agent track's score directly with standard VLA/WAM policy rankings, HumanoidToolBench success rates, or physical trial success.

### VioLA — shared body/hand motion-latent outputs

VioLA (8 Oct 2026; https://arxiv.org/abs/2610.12435) predicts shared body and hand motion latents rather than raw joint commands; pretrained body/hand controllers track those latents. It reports 140.6M training frames with 93.2% human recordings, 100% zero-shot locomotion success on its specific real-humanoid test, and 88.6% manipulation success without task-specific fine-tuning.

This motivates a distinct action-space ablation:
- raw joint/actuator targets;
- typed task-space goals;
- controller-tracked body/hand motion latents.

Human motion is not automatically a robot command. Cross-domain action labels require exact frame, morphology, controller, contact and execution semantics. VioLA code/checkpoints were stated as forthcoming in the paper; mark the comparator unavailable until the exact artifacts and licence can be inspected.

### Symthaea's bounded agentic hypothesis

We should test an agentic generalist model without copying unrestricted live Python actuation:
1. Planner receives visual observations, typed proprioception, task instruction and explicit remaining time/budget.
2. It requests narrow typed capabilities: inspect, estimate affordance, check reachability, propose grasp/pose, request motion, read contact state, verify, retreat, regrasp, abstain or hand off.
3. Skill executive composes versioned procedures and deterministic local controllers.
4. Every physical effect remains behind the existing UEC/action projection, runtime safety, interlock and qualified backend.
5. Persistent memory separates embodiment/configuration, semantic tool knowledge, procedural skill, episodes and failure history. Only independently verified and applicability-matched experiences may promote reusable procedures.
6. Candidate code/parameters remain isolated and never rewrite a live safety-critical controller.

The point is not to forbid all generated code. It is to prevent generated code from becoming physical authority. A compiled/validated candidate skill may be considered only through the existing evidence and authority promotion path.

### Memory promotion and poisoning tests

Memory entries must bind:
- exact robot profile, as-built generation, tool/fixture identities and calibration applicability;
- source episode and raw observation/action evidence;
- independent verifier and accepted postcondition;
- preconditions, parameter envelope, failure modes and recovery;
- confidence/coverage and known unknowns;
- version, validity window and invalidation dependencies.

Negative controls include poisoned demonstrations, false task-success reports, wrong embodiment, changed tool geometry, stale camera transform, revoked calibration, contradictory observations, scene aliasing, skill-family regression and memory replay after morphology change.

Evaluate three memory types separately:
- episodic memory: retrieval of exact prior events;
- semantic memory: tested tool/embodiment facts;
- procedural memory: validated reusable skill contracts.

Count retrieval errors, stale-memory use, harmful transfers, forgotten earlier task families and unsafe actions. A memory-induced performance gain is not acceptable if it silently lowers evidence quality or increases unsafe continuation.

## 4. Matched baseline ladder

Evaluate the following arms only when the exact artifact and interface are available:
1. Scripted or geometry-based baseline.
2. Simple reactive / no-memory baseline.
3. Original benchmark-native reference policies.
4. ACT / Diffusion Policy or other task-native imitation baseline.
5. GR00T N1.7 and another compatible open VLA baseline (for example pi0.5).
6. VioLA-style body/hand motion-latent policy, if exact code/checkpoint and morphology-compatible controllers become available.
7. AMP1-style agent-as-policy baseline, including an open-weight agent model where available (the AMP1 paper reports Qwen3.8-Flash-Next at 64.9% on RoboLab-120 in its protocol).
8. Agentic model + typed capability API + deterministic action admission (Symthaea-bound harness).
9. Arm 8 plus episodic memory only.
10. Arm 8 plus validated semantic/procedural skill memory.
11. Symthaea perception + proprioception policy.
12. Symthaea + functional tool model.
13. Symthaea + tactile evidence, with calibrated raw-sensor comparator.
14. Symthaea + action-conditioned tactile/contact prediction.
15. Symthaea + uncertainty-aware probing and event-based recovery.
16. Hybrid agentic planner + Symthaea skill/controller substrate + independent evidence/safety planes.

Unavailable artifacts are “unavailable”, not a zero score. Pin licence, source SHA, checkpoint digest, model/provider/version, reasoning-effort setting, prompt, tool API and route policy, pre/post-processing, robot morphology, actuation semantics, sensor set, training data, memory content/provenance, adaptation budget, inference hardware and evaluation runtime.

Matched comparisons require equal task/reset distribution, demonstration/adaptation opportunities, action chunk/horizon, observation access, episode timeout and declared compute/time budgets. For agentic comparisons additionally report model identity/reasoning effort, model-call count, tool-call count, wall-clock reasoning time, actuation time, token/API cost and failures/timeouts. The AMP1 paper finds model choice has a larger effect than modest harness changes; therefore separate model effects from harness/architecture effects rather than claiming a harness win from different frontier models. If budgets or robot semantics differ, report the mismatch and do not claim a controlled comparison.

## 5. Symthaea architecture hypotheses

### 5.1 Tool-as-body-schema

Create a temporary embodied model of robot + grasp + tool:
- tool identity and geometry;
- grasp transform and confidence;
- tool-tip / working-interface frame;
- added reach and collision envelope;
- mass/inertia estimate with provenance;
- compliance / stiffness / contact properties with unknowns explicit;
- force/torque transmission assumptions;
- permitted use envelope and task suitability.

Invalidate/re-estimate when grip, tool, attachment, calibration or embodiment changes. This is a cognitive/dynamics model, not an actuation authorization.

### 5.2 Functional affordance inference

Represent tools by task-relevant physical function, interface and constraints, not merely category names. Distinguish observed geometry from inferred material, strength or serviceability. Use low-risk tests only where the test is itself within the approved experiment envelope. Prefer abstention to unsupported extrapolation.

### 5.3 Contact prediction and active sensing

Predict action-conditioned contact outcomes and time-to-failure boundaries. Score forecast error against sensor-native observations. Compare against:
- vision-only;
- raw tactile;
- fixed engineered contact thresholds;
- frozen tactile representation;
- tactile world-model variant;
- ablations removing tactile, vision, prediction or active probing.

Charge model-query and sensing latency to the policy. Track freshness, calibration, saturation, dropout, clock alignment and modality contradiction. Learned latent similarity is not force metrology.

### 5.4 Event-based recovery

Represent distinct milestones: tool selected, secured, first contact, correct contact, desired physical effect, postcondition met. Recovery actions include pause, retreat, release, regrasp, reposition, re-localize, alternative tool, retry within budget, abstain and handoff. No “success” inferred from issued commands alone.

### 5.5 Verified service composition

Use skill contracts with preconditions, observations, permitted actions, expected consequences, stop conditions, recovery, postconditions and independent verifier. The sequence is:

```
inspect -> isolate -> diagnose -> select replacement/tool
 -> disassemble -> repair/replace -> reassemble
 -> calibrate -> independently test -> accept/quarantine
```

A repair proposal and a repair execution are different records. Return-to-service requires a current, independent acceptance result and exact configuration lineage.

## 6. Tool-use evaluation protocol

### 6.1 Published anchor

Reproduce all 18 HumanoidToolBench cells when feasible. Preserve the benchmark's native scorer and report all cells. Do not silently alter the official tasks.

### 6.2 Extension splits

Add held-out sets, separately named and versioned:
- unseen instances within known tool families;
- entirely held-out tool families;
- new tool × task combinations;
- changed target pose / distance / height / reach requirement;
- held-out rooms, clutter and work surfaces;
- changed friction, payload and compliant objects;
- worn, damaged, bent, missing or incorrectly labelled tools;
- lighting, occlusion, camera and tactile degradation;
- delays, dropped frames, stale calibration and clock skew;
- task changes mid-episode and interruption/resumption;
- natural-language paraphrase, irrelevant, contradictory and adversarial instruction.

Split by provenance groups. A new texture on a familiar geometry is not a new tool family.

### 6.3 Negative controls

Mandatory controls:
- no tool present;
- obviously wrong tool;
- plausible decoy tool;
- shuffled tool name/geometry labels;
- irrelevant instruction;
- instruction conflicts with observed task;
- task target moves after planning;
- stale / mismatched tool calibration;
- tactile sensor missing, stale, saturated or contradictory;
- altered physical tool geometry without profile update;
- tool is damaged or outside declared limits;
- evaluator/private ground-truth feature injected into a probe policy;
- simple scripted and low-capacity shortcut probes;
- no-op / frozen policy.

A candidate that succeeds only because of scene layout, task order, privileged state or evaluator leakage must be identified as shortcut-solvable.

## 7. Verified service-loop corpus

V0 contains eight passive/low-energy cases:
1. Replace a non-structural cable guide or protective cover.
2. Replace an accessible sensor mount / passive bracket.
3. Install and remove a non-load-bearing clamp or fixture.
4. Replace a passive end-effector finger pad.
5. Assemble and disassemble a fastener-and-bracket fixture.
6. Diagnose and correct a deliberately misassembled passive mechanism.
7. Complete a bounded dimensional / fit inspection after reassembly.
8. Detect an incorrect or damaged replacement and refuse return-to-service.

Each case must bind:
- exact task instruction and permitted task scope;
- as-designed and as-built fixture identity;
- tool and replacement part IDs/revisions;
- robot embodiment / tool interface / calibration profile;
- expected preconditions and isolation proof where relevant;
- allowed action envelope and stop conditions;
- independent dimensional or functional acceptance test;
- postcondition/failure taxonomy;
- evidence class, source digest and evaluator identity.

No live mains, high voltage, stored-pressure systems, powered cutting, hazardous substances, load-bearing structural repairs, clinical procedures or unsupervised human treatment in V0.

## 7A. Human + Symthaea repair-workbench track

Robot-only maintenance and human-supported maintenance are different capability questions. Do not wait for a physically qualified humanoid to test whether Symthaea can already help people repair, inspect and restore equipment.

Reuse the existing [HUMAN-TOOL-001 #7983](https://github.com/Luminous-Dynamics/symthaea/issues/7983) protocol and draft evaluator [PR #7997](https://github.com/Luminous-Dynamics/symthaea/pull/7997). Do not create another repair ledger, diagnostic model, instrument adapter, outcome evaluator or human-factors framework.

The existing human-workbench comparison already defines:
- A0: human-only with current manuals/procedures;
- A1: structured work instructions/checklist and same instruments;
- A2: human + local Symthaea advisory with evidence-linked ranked hypotheses, contradicting evidence, missing/stale data, next measurement or abstention, and rationale.

For HUM-TOOL-SERVICE integration, treat this as a **parallel human-augmentation track**, not as humanoid tool-use performance. It asks whether the same verified-service loop and evidence semantics can improve human repair work before the humanoid has equivalent physical competence.

### Required integration

- The same low-risk fault/task family and starting conditions across arms.
- Same instruments, tool availability, manuals/reference access, participant time and safety boundaries.
- Ground truth held from participants; independent adjudication of diagnosis and restored function.
- Primary endpoint: elapsed time from task start to **independently verified useful-function restoration**, including diagnostic time, failed attempts, rework, waiting and safe referral.
- Retain all started cases in denominators: successful, not restored, abandoned, censored, referred, safety-stopped and unknown.
- The advisory has no actuation authority and cannot override protection/interlocks.
- The model must distinguish hypotheses from measured facts and recommend the next measurement only when it has decision value.
- Preserve participant consent, data minimization, human-factors outcomes and applicable review before any participant study.

### Metrics to compose

Reuse #7983's outcomes: root-cause determination, next-test value, unnecessary swaps, destructive disassembly, repeat work, false reassurance, missed critical indicators, safe referral, recurrence, P50/P90 restoration time, labor/tool/parts/energy cost, workload/usability/trust calibration, measurement uncertainty and evidence conflict.

Add a transfer dimension only after the base study protocol is stable: compare advice on fault families with and without equivalent prior task exposure. This must not change the registered primary endpoint or make the participant study a test of robot physical dexterity.

### Why this matters strategically

Symthaea should be useful before it can physically perform every repair. A reliable advisory system can help people select measurements, reason about unfamiliar faults, find correct tools/parts and verify outcomes. This is a complementary route to the full humanoid—not a substitute for eventual physical competence and not evidence that Symthaea itself executed the repair.

## 8. Statistics and reporting

- Freeze code, model, config, task set, split, seed, environment, robot, tool and evaluator identities before evaluation.
- Target at least five independent training seeds where affordable; use paired evaluation seeds and report seed-level variation.
- Start with 100 simulation episodes per cell/policy when affordable to align with the published benchmark. Increase sample counts from a preregistered precision/power analysis, not after inspecting favourable results.
- Physical trial counts need a separate risk and statistical plan. Thirty successful trials do not certify general safety.
- Report task-cell rates, numerator/denominator, uncertainty intervals, effect sizes and failures; retain cluster structure for homes/tools/physical articles.
- Report expected, completed, invalid, missing and infrastructure-indeterminate episodes. Do not silently remove timeouts, safe refusals or failures.
- Publish training-data exposure, fine-tuning, annotation/human-operator, compute, wall-clock, energy and hardware budgets.
- For agentic arms, decompose total wall time into perception, reasoning/model wait, action planning, actuator execution, verification and recovery; report p50/p95 for each, total model calls, tool calls, tokens/API cost and timeout rate.
- For memory arms, freeze a training/practice task sequence and disjoint evaluation sequence; report transfer to unvisited tasks, retention on earlier tasks, retrieval precision, stale/poisoned-memory use and memory update/revocation events.
- Use paired differences where valid; report median, p90/p95, lower tail and confidence intervals. Do not assume all task cells or physical articles are IID.
- Preserve benchmark artifact failures separately from policy failures and verifier disagreement. A benchmark false positive/negative must not be attributed to the robot policy.
- Maintain one result vector per capability; no aggregate score may hide a hard constraint failure.

## 9. Metrics

### Tool-use outcome vector
- suitable tool selection;
- secure grasp and retention;
- correct first contact / tool-target engagement;
- task effect achieved;
- independently verified completion;
- stationary versus mobile gap;
- seen-tool versus unseen-instance versus unseen-family transfer;
- decoy and instruction sensitivity;
- contact/slip/seating detection;
- recovery success and time;
- risk-coverage, calibrated abstention and handoff correctness;
- p50/p95 completion time, planner/reasoning latency, action latency, energy, compute, model/tool calls and provider failures;
- number and quality of active observations, scene-model corrections and plan revisions;
- skill synthesis/reuse rate, cross-task transfer, memory retrieval precision and stale/poisoned-memory rejection;
- task-family retention after memory updates and harmful-transfer rate;
- collisions, object/tool damage, unsafe forces and invalid contact.

### Service outcome vector
- fault localization and diagnostic time;
- correct replacement/tool selection;
- isolation/precondition compliance;
- first-time fix and independently verified repair rates;
- reassembly/conformance;
- functional post-test pass;
- calibration currency and lineage completeness;
- return-to-service time;
- human intervention / handoff;
- rework, scrap, damage, consumables and energy;
- false acceptance, unsafe continuation and lost traceability.

### Operational utility
Where a workcell is sufficiently specified, reuse #11607:
- accepted quality-qualified outputs per available robot-hour;
- first-pass yield;
- autonomous completion share;
- recovery/intervention burden;
- availability and downtime taxonomy;
- rework/scrap;
- energy/consumables/wear per accepted output;
- traceability completeness.

## 10. Prospective targets (not results)

These are stretch hypotheses and must be preregistered:
- ≥10 percentage-point paired improvement over the strongest reproducible baseline on a held-out / decoy / mobile composite, with uncertainty interval excluding zero improvement.
- ≥80% success on held-out instances within a declared known tool family; report unseen-family transfer separately.
- ≥20% relative reduction in recovery/intervention burden without reducing verified success.
- As a stretch memory target, ≥4× relative failure-rate reduction on a matched multi-task sequence, with no material regression on previously learned task families and no poisoned/stale-memory acceptance.
- ≥90% verified success on the eight-case low-risk synthetic service corpus.
- Zero unsafe continuation and zero false return-to-service in the frozen negative-control corpus.
- Demonstrate the value of each architecture component via removal ablations, while reporting extra data, compute, latency and energy.
- For the agentic track, demonstrate a positive accepted-work-per-robot-hour improvement after charging reasoning latency, model/API cost, retries, interventions and recovery.

A zero-event synthetic corpus result is not a general safety proof. Targets must be revised if task prevalence, feasibility or measurement precision make them unsound.

## 11. Evidence classes and gates

Maintain separate labels:
- protocol / synthetic fixture;
- offline dataset;
- simulator;
- hardware-in-the-loop;
- instrumented physical bench;
- supervised field;
- independently adjudicated deployment.

```
benchmark score != capability qualification
tool selected != tool suitable
tool grasped != tool correctly engaged
command issued != physical effect observed
repair performed != service restored
simulation != HIL != physical
calibration file != current qualified calibration
self-diagnostic != independent verification
```

Physical progression must consume the existing embodiment topology, calibration and commissioning owners (#4619/#4620/#4623), experiment/instrumentation owners (#4800/#4866), JointLab (#4814), HIL/physical gates (#1399/#1400), and external safety authority. No new actuator path is permitted.

Human-contact capability remains separate and consent-bound under #4680. V0 excludes goal-directed human contact.

## 12. Immediate decision sequence

1. Pin official HumanoidToolBench source, checkpoint, data, assets and license; confirm whether the protocol is executable in our supported environment.
2. Establish a robot/control/observation compatibility matrix. If no honest mapping exists, use the published benchmark's robot as an external comparator only; do not relabel Symthaea's morphology.
3. Implement exact 18-cell protocol import and baseline runner only after source/reproducibility audit.
4. Run no-op/scripted probes before costly model training.
5. Run matched Symthaea ablations in simulation.
6. Add eight service-loop cases and an independent postcondition verifier.
7. Select one physical task only after HIL, topology, calibration, instrumentation and physical qualification prerequisites are met.

## 13. Claim ceiling

This protocol defines a prospective evaluation. It does not establish that Symthaea has tool competence, service capability, self-maintenance, wilderness capability, physical safety, SOTA performance or household generality. No benchmark pass creates actuator authority, human-contact consent, safety certification or production release.
