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

## 3C. Current agentic SOTA audit and reproduction constraints (17 November 2026)

This section supersedes the older single-number framing of AMP1 and RoboDojo. Keep paper results, live leaderboard snapshots, after-practice results and real-robot trials in separate rows.

### AMP1 — paper results and current public comparison

Primary paper: https://arxiv.org/abs/2611.09892 (first posted 6 November 2026). Official project: https://apple-aiml-research.github.io/ml-amp/. Official source repository: https://github.com/apple-aiml-research/ml-amp.

Current project page and paper report:
- RoboLab-120: 85.6% zero-shot for Gemini 4 Argon on the paper protocol; 82.8% with Claude Opus 5.5. The current NVIDIA RoboLab leaderboard reports AMP1 at 86.0% ± 2.5 percentage points, 2,579/3,000 episodes, 25 episodes/task. The paper and leaderboard snapshots should not be collapsed into one exact figure or denominator.
- CaP-Gym: 95.2% on LIBERO-PRO (reported as best zero-shot), 100% on the selected robosuite uni-/bimanual suite, and 96.0% on the two BEHAVIOR-1K mobile-manipulation tasks.
- RoboDojo: 84.1% is explicitly labelled by AMP as the best public number **after practice**, not a zero-shot score. AMP1+ uses memory/practice before evaluation on held-out layouts. The paper separately reports 4.6× lower failure rate on RoboDojo bimanual tasks with carried memory, and an improvement from 26.4% zero-shot to 57.6% after one pass through 42 tasks in its related bimanual analysis.
- Real-world: 104/108 successful trials across 12 tasks (nine trials each) on a 7-DoF Franka Research 3 with a Robotiq 2F-85 and three cameras. This is an arm benchmark, not a humanoid result. The paper reports 104/108 with recovery; 94/108 without its recovery-from-controller-error functions. Collision, camera-calibration and geometry approximation remain meaningful failure modes.

Sources:
- Paper: https://arxiv.org/abs/2611.09892
- Results, harness description and hardware walkthrough: https://apple-aiml-research.github.io/ml-amp/
- Current NVIDIA RoboLab leaderboard: https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html?family=all
- Current RoboDojo simulation leaderboard / Agent Track: https://robodojo-benchmark.com/leaderboard?bench=sim

### 3C.1 Current RoboLab-120 leaderboard context (17 November 2026)

The live public leaderboard currently lists the following overall entries:

| Policy | Type | Evaluated episodes | Success rate | Reported model size / VRAM |
|---|---|---:|---:|---|
| AMP1 | Agentic LM | 2,579 / 3,000; 25 eps/task | 86.0% | not listed |
| SenseNova-RoboRSI | Agentic LM | 685 / 1,200; 10 eps/task | 57.1% | not listed |
| Lakeside | LWM | 1,531 / 3,000; 25 eps/task | 51.0% | 5B / 27.2 GB |
| BiMind v0.5 | VLA | 590 / 1,200; 10 eps/task | 49.2% | 8B / 30 GiB |
| Praxis-1 | WAM | 555 / 1,200; 10 eps/task | 46.2% | 27.3B / ~70 GB |
| FLUX 3 Action | WAM | 515 / 1,200; 10 eps/task | 42.9% | 7B / 69 GiB |
| Cosmic | TAMP + FM | 508 / 1,200; 10 eps/task | 42.3% | ~350M / 10 GB |

Source: NVIDIA's live [RoboLab-120 leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html), observed 17 November 2026.

The table shows a meaningful gap between AMP1 and the next listed policies in this snapshot, but denominators differ: AMP1/Lakeside have 25 episodes per task; most other rows have 10 episodes per task. Results are not all paired or equally sampled. Report the actual episode count and uncertainty, do not treat the row order as a significance test, and do not interpret these as humanoid hardware results. RoboLab is a generalist tabletop/interaction simulation suite, not HumanoidToolBench or physical G1 qualification. Its published score should be a model/harness reference only.

The page exposes action-generation latency and replanning delay columns, but current AMP1 entries are not populated there. This makes end-to-end reasoning latency, motion overlap, action execution and cost especially important to reproduce locally rather than infer from headline success.

### Agent Track governance and score comparability

RoboDojo published revised Agent Track rules on 14 November 2026 and marks previously evaluated entries with an asterisk when they do not conform to one or more updated provisions. The 17 November snapshot shows a separate Agent Track, including SOLVE* at 79.50 average / 74.96 success and Q0-Preview* at 73.20 / 67.34, while the standard policy board has a different scale and top standard entry. Do not rank starred pre-rule results as compliant submissions, and do not compare Agent Track averages with standard policy averages, RoboLab percentages or HumanoidToolBench success rates.

The Agent Track's current protocol explicitly separates Generalization, Precision, Long-Horizon and Memory. This decomposition is worth adopting in Symthaea's scorecard rather than using one aggregate completion percentage.

### AMP1 harness mechanisms worth testing—not copying blindly

The official project page describes:
- a persistent Python interpreter that carries state and generated routines across model turns;
- active sensing and selectable RGB/depth/point-cloud observations;
- an editable scene model;
- timed end-effector waypoints and gripper commands;
- a `preview` reachability/speed/joint-limit check, collision checks and an execute call that may refuse a motion;
- generalist-model calls interleaved with environment observations.

The project walkthrough notes the simulation pauses while the model reasons, so only executed motion counts against the depicted motion-time limit. The paper also identifies slow model reasoning and inability to reconsider a trajectory while it is executing as limitations; it proposes faster local feedback controllers and overlapping planning with motion as future directions.

**Important Symthaea opportunity:** use an agentic slow planner to decide *what* to do, but use a qualified local fast loop to monitor contact, slip, collision margin and state transitions continuously. If the physical scene changes while a motion is executing, the local controller must stop/hold/recover without waiting for the next LLM turn. Compare both benchmark-motion time and end-to-end wall-clock time; charge model wait, API cost, retries, compute, energy and intervention to operational utility even when a benchmark pauses simulation during reasoning.

### AMP1 reproduction, licence and compute boundary

The official source repo's `LICENSE` is an Apple custom copyright licence granting a personal, non-exclusive licence; it is not a conventional MIT/Apache-2.0 licence. It contains third-party materials with separate terms. Do not copy AMP source into Symthaea, redistribute a fork, or incorporate it into a Luminous product until the exact licence and third-party terms have been reviewed. The public paper, benchmark protocol and reported results can still be studied and independently implemented without copying protected source.

The official reproduction guide pins third-party source revisions and requires x86_64 Linux, NVIDIA GPU, a provider API key or compatible model server, and the full benchmark stack. Its RoboLab quickstart lists approximately 9 GB VRAM, 15 GB RAM and 40 GB disk; paper simulation results used H100 80 GB GPUs with driver 580.159.03. It reports Python/uv, Isaac Sim and substantial assets. These are not requirements for the small Rust mapping audit, but they are relevant to a full AMP1 reproduction. The known RTX 2070 Laptop 8 GB configuration is below the guide's stated VRAM figure; do not assume it can run the official stack without a separate tested lower-resource path.

Sources:
- AMP source and licence: https://github.com/apple-aiml-research/ml-amp
- Reproduction pins: https://github.com/apple-aiml-research/ml-amp/blob/main/docs/reproducibility.md
- Hardware/setup guide: https://github.com/apple-aiml-research/ml-amp/blob/main/docs/quickstart.md
- Model/provider guide: https://github.com/apple-aiml-research/ml-amp/blob/main/docs/models.md

### Active tool affordance discovery — an important benchmark gap

New child programme: [HUM-TOOL-DISCOVERY-001 #12701](https://github.com/Luminous-Dynamics/symthaea/issues/12701).

HumanoidToolBench is the immediate published humanoid tool-use anchor: 55 tool assets and approximately 3.1k demonstrations, with selection, stationary tool use, mobile execution and decoys. But selecting among known tools is not the same as identifying what an unfamiliar physical tool can do or learning its operating envelope.

The new discovery track asks whether an agent can:
1. distinguish visual appearance from causal physical affordance;
2. choose bounded probes that separate plausible tool-function hypotheses;
3. infer effective reach, working-end geometry, engagement, compliance and operating limits with uncertainty;
4. transfer the discovered affordance to a new target/task/tool instance;
5. refuse when material strength, safe load, interface compatibility or other critical properties remain unknown.

The first synthetic cases cover lever, hook, wedge, rotary driver, clamp and mallet families, with held-out same-family variants, task-transfer variants and visually similar decoys. The 24-case manifest is:
- `tool-discovery-cases-v0.1.json`
- `tool-discovery-cases-v0.1.schema.json`

The full file is evaluator-owned: policy context receives only candidate-visible observations and probe budget. Hidden physical parameters, split labels and postcondition oracle must remain sealed from the candidate.

No powered tools, blades, high force, mains voltage, hazardous materials, human-contact tasks or destructive probes are included in v0.1.


## 3D. ULTRA — a direct humanoid whole-body loco-manipulation comparator

ULTRA (He et al., IROS 2026; arXiv:2603.03279) is a particularly relevant comparator because it reports a unified multimodal controller evaluated in simulation and on a physical Unitree G1. It supports dense reference tracking when motion references exist and sparse, goal-conditioned whole-body loco-manipulation when they do not. Its released project page demonstrates long-horizon goal following under motion-capture state and egocentric depth, plus sim-to-sim transfer with point-cloud perturbations. Its approach combines physics-driven neural motion retargeting, teacher/student distillation, compact motion-skill latents and RL fine-tuning.

Sources:
- Project, demonstrations and paper: https://ultrahumanoid.github.io/
- Paper: https://arxiv.org/abs/2603.03279
- Public code: https://github.com/Sirui-Xu/ULTRA

**Why it matters:** ULTRA is closer to our target embodiment/control problem than AMP1's Franka real-robot trials. It is a direct baseline for the question “can one controller convert sparse intent and egocentric sensing into coordinated G1 whole-body behavior?” AMP1 remains the separate comparator for “does a generalist agent reason and revise manipulation plans during the task?” HumanoidToolBench remains the separate benchmark for “can a G1 select and use tools, including after mobile repositioning and with decoys?” These test different capabilities; their scores must not be pooled.

**ULTRA limits and reproduction boundary:** the project page establishes real-G1 evaluation and qualitative long-horizon demonstrations, but does not present one directly comparable aggregate success percentage for HumanoidToolBench or a shared service task. Do not invent a ULTRA-to-HumanoidToolBench conversion. The public repository lists ULTRA code as Apache-2.0 and an InterMimic-derived stack under MIT; assets, motion data and third-party dependencies retain separate terms. Its reproduction instructions use Python, IsaacGym Preview 4 and MuJoCo. Before reproduction or reuse, pin the exact commit/checkpoint, establish compatible robot and task semantics, inspect asset/data licences, and measure the actual compute requirement.

**Matched baseline addition:** include ULTRA's released G1 controller, where checkpoint, data rights and runtime can be qualified, as an independent learned whole-body baseline. Compare it against the existing Symthaea hierarchy on matched G1 tasks and sensors, then compare both with and without the same HLB/agentic skill executive. Keep controller policy, planner, sensor access and safety stack factors separable. If full reproduction is not feasible, record ULTRA as a paper/project reference and mark the runtime baseline unavailable rather than substituting a different policy.



## 3E. Physical humanoid frontier and the capability-transition insight (17 November 2026)

### Industry and research frontier

| System | Public evidence as of 17 Nov 2026 | Proper comparison and caveat |
|---|---|---|
| Figure Helix 2.5 | Figure reports one Index-pretrained foundation model adapted to three whole-body behaviors (living-room tidy, towel folding, bed making) and evaluated with a fixed checkpoint in 30 unseen Bay Area homes. The company reports 56% zero-shot task success versus 9% from scratch with the same task-specific data. | A strong real-humanoid generalization reference. "Zero-shot" means no data/fine-tuning in evaluation homes or manipulated objects; the behaviors were specified with task data collected elsewhere. It is a vendor report, not a HumanoidToolBench tool-use score. |
| Gemini Robotics 2.5 + Apptronik Apollo 2 | Google DeepMind reports evaluation across 200+ distinct tasks and a 96% success rate in a Mercedes-Benz automotive-kitting proof-of-concept using Apollo 2. Its described system pairs Gemini Robotics ER for workflow reasoning with Gemini Robotics 2.5 for hand/arm actions. | A valuable industrial reliability reference, but the 96% is for one partner pilot/task definition, not universal humanoid capability or an independent benchmark. Vendor/partner-reported. |
| ULTRA (IROS 2026) | Unified multimodal whole-body controller evaluated in simulation and on a real Unitree G1, supporting dense reference tracking and sparse goal-conditioned loco-manipulation from egocentric inputs. | Direct G1 controller baseline; no comparable HumanoidToolBench aggregate score is published in its project/paper summary. |
| HumanoidToolBench | 18 tasks, 55 tool assets, 3,094 trajectories (3,003 simulation and 91 real-G1 demonstrations); seven policies in simulation and three on G1. | Direct tool selection/use benchmark. Separate selection, stationary use, mobile use, standard/decoy conditions and event markers. |
| AMP1 / RoboLab-120 | Live NVIDIA snapshot lists AMP1 at 86.0% ±2.5 pp (2,579/3,000, 25 episodes/task); next listed SenseNova-RoboRSI is 57.1% ±5.7 pp (685/1,200, 10 episodes/task). | Agent-as-policy benchmark reference, not a humanoid or physical G1 tool-use result. Episode denominators differ; do not treat row rank as a paired significance test. |

Sources: [Figure Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization), [Google DeepMind Gemini Robotics 2.5](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/), [ULTRA](https://ultrahumanoid.github.io/), [ULTRA paper](https://arxiv.org/abs/2603.03279), [HumanoidToolBench](https://arxiv.org/abs/2610.02089), [NVIDIA RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html).

All headline numbers above refer to different task distributions, checkpoints, embodiments, trial counts and evaluation protocols. They are a portfolio of capability references, not a single cross-system ranking. The commercial reports are vendor-reported; independently reproduce exact protocols before making superiority claims.

### HumanoidToolBench exposes a deeper failure than tool recognition

The benchmark explicitly logs five events: candidate contact, correct first contact, correct-tool lift, tool-target contact and terminal task success. It separates L0 selection/pickup, L1 stationary use and L2 mobile use, and includes decoy sets. The robot is a Unitree G1 with 29 body joints, Dex3 hands and a head camera; lower-body locomotion components are executed by pretrained GR00T Balance/Walk controllers. Consequently, this benchmark measures a coupled policy-plus-controller system, not an isolated policy that learns all locomotion from scratch.

A useful public result from the benchmark analysis: FastWAM's standard BallMove success falls from 76% in L1 to 9% in L2. The paper reports an L1-to-L2 decline for every policy with non-zero L1 success, but L2 also changes target layout; this does not isolate locomotion as the sole cause. Secondary extraction of the paper's event table reports that, averaged over L1/L2 and the three scenarios in standard mode, FastWAM contacts a candidate in 99.8% of episodes, lifts the suitable tool in 89.2%, contacts the target in 88.2%, but completes only 52.8%; GR00T N1.7 reports 99.7%, 90.0%, 79.8% and 36.0% for those same stages. These aggregate values are not a strict sequential funnel, and the averages hide task-cell differences.

Sources: [primary paper](https://arxiv.org/abs/2610.02089), [paper table/event analysis](https://www.alphaxiv.org/abs/2610.02089v1), [secondary event-table extraction](https://mekawire.com/papers/humanoidtoolbench-benchmarking-humanoid-tool-use-from-selection-to-mob-2610-02089).

### Paradigm shift: evaluate the capability transition graph, not only terminal success

A single binary success score cannot distinguish a perception failure from a grasp failure, tool-orientation error, failed physical interaction, mobility problem or failure to verify the actual task outcome. Preserve the five event timestamps/occurrence markers and derive a per-episode transition graph:

```text
candidate contact
       |
correct first contact
       |
correct-tool lift
       |
tool-target contact
       |
verified task completion
```

This is an analysis graph, not an assumption that events always form a strict funnel. Retain event ordering, repeated contacts, missing instrumentation and out-of-order recovery actions. Compute, per task cell and split:

- event incidence among episodes where that event was actually instrumented;
- event-to-event conversion conditional on both events being captured;
- time from event to next event and total task time;
- stage-specific failure / recovery cause;
- intervention, regrasp, reorientation, reposition and retry cost;
- decoy penalty and held-out tool-property generalization;
- L1/L2 paired difference, with layout and route-distance changes exposed as confounders.

The new controller hypothesis is stage-conditioned intervention: HLB should infer *which transition is failing* and select the least-cost, risk-bounded next action. Wrong first contact should trigger reinspection or affordance probing; correct lift without target contact should trigger grasp/tool-frame reorientation or whole-body repositioning; target contact without completion should trigger contact/load prediction and a bounded adjustment; L1-to-L2 degradation should trigger explicit carrying, balance, route and tool-frame reasoning. The low-level whole-body and safety layers remain independent.

### New prospective targets — not current results

1. Reduce the largest preregistered stage-transition failure rate by at least 30% relative to the strongest matched baseline, without degrading earlier event quality or safety.
2. Close at least 30% of the paired L1-to-L2 success gap on the same task families, using a controlled extension that separates locomotion demand from target-layout changes.
3. Achieve at least 2× the strongest baseline in the weakest held-out mobile/decoy task cell, with confidence intervals and all 18 cells reported.
4. Preserve zero false success/false task-completion acceptance in frozen negative controls; task completion must be independently verified rather than inferred from tool-target contact alone.

These goals are prospective. They should be frozen only after exact upstream policy/checkpoint, episode seeds, controller version, task geometry, event instrumentation and statistical power are pinned.

## 4. Matched baseline ladder

Evaluate the following arms only when the exact artifact and interface are available:
1. Scripted or geometry-based baseline.
2. Simple reactive / no-memory baseline.
3. Original benchmark-native reference policies.
4. ACT / Diffusion Policy or other task-native imitation baseline.
5. GR00T N1.7 and another compatible open VLA baseline (for example pi0.5).
6. VioLA-style body/hand motion-latent policy, if exact code/checkpoint and morphology-compatible controllers become available.
6a. ULTRA unified multimodal G1 loco-manipulation controller, where released checkpoint/data/robot semantics and runtime can be qualified; treat as a distinct real-humanoid baseline, not as a tool-use score.
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
17. Active tool-affordance discovery: passive VLM / geometry heuristic / supervised tool labels / agentic probing / causal tool model / tactile-enhanced causal model / oracle-property upper bound. Use HUM-TOOL-DISCOVERY-001 #12701; report probe budget, information gain, held-out task transfer and safe abstention separately.

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

## 5A. HLB in the humanoid: architecture and falsifiable ablations

### Decision and responsibility boundary

Use Symthaea's Holographic Liquid Brain (HLB) as the persistent embodied-cognition and temporal-state substrate, **not** as the hard-real-time servo controller, the sole whole-body controller, or the physical-safety authority. The HLB hypothesis is that compositional HDC state, liquid temporal dynamics, active inference, prediction, uncertainty and validated memory improve long-horizon adaptation, tool-use transfer and recovery. It is not assumed to beat simpler recurrent models.

The current modern `symthaea-humanoid` trainer already uses `HdcLtcUnifiedNetwork`: the 16,384-dimensional state is projected through a 64-dimensional bottleneck to a morphology-sized learned residual. The current trainer feeds observation-encoded state to that network every physics step, composes its residual with task-specific PD baselines through `HierarchicalHumanoidController`, then applies fall protection, deterministic safety projection and backend adaptation before stepping the simulator. Current defaults are 40 Hz physics and 10 Hz cognitive/FEP updates. These are software configuration defaults, not measured physical hardware rates.

The distinction in learning maturity is important:
- Head-only output learning is the default checkpointable path.
- An optional path updates HDC-LTC recurrent neurons, but is disabled by default.
- Full recurrent learner checkpoint/resume is not currently supported; current checkpoints explicitly mark recurrent state absent and reject a claim that it has been restored.
- The legacy `HumanoidEmbodiment::step(thought_hv, dt, phi)` bridge is a separate path, not the same observation-encoded hierarchical trainer.

The recurrent-state gap belongs to #1269; clock semantics belong to #1284; action/observation semantics belong to UEC v2 #1274/#1288; plant safety must remain independent from cognitive readiness under #1276. Do not create parallel contracts or infer authority from HLB confidence, Φ/Ψ or capability discovery.

### Composed architecture

```text
typed observations + provenance + clocks
                |
       state estimation / freshness
                |
      HLB embodied belief and prediction
 (body + grasp + tool + environment + task)
                |
 bounded skill executive / active sensing
                |
    task-space goal / motion-latent proposal
                |
 existing whole-body + contact-aware control
                |
 independent safety admission + local watchdog
                |
 measured plant transition + independent verifier
                |
 episodic receipt -> candidate memory/skill
                |
 replay, counterfactual, transfer, retention gate
```

HLB owns persistent context, predictions, uncertainty, active observation selection, skill retrieval and candidate learning. The skill executive owns bounded preconditions, state transitions, postconditions, retry limits and handoff. The existing whole-body stack keeps balance, contact, inverse dynamics, grasp and trajectory tracking. Servo/current loops, watchdog, e-stop, safety admission and qualified local fallback remain independent and must function when HLB is late, stale, reset or absent.

### Matched ablation matrix

Hold the morphology, observations, simulator, seeds, action envelope, training/evaluation budget and verifier fixed. Run:
A. deterministic baseline + hierarchy, no learned residual;
B. capacity-matched MLP residual;
C. capacity-matched GRU residual;
D. HDC encoder + static readout (isolates representation);
E. current HDC-LTC residual path (current Symthaea baseline);
F. checkpointable recurrent HDC-LTC learning (only after #1269);
G. HLB + episodic memory, no procedural skill promotion;
H. HLB + validated procedural memory + active sensing;
I. AMP-style agent planner + the same local whole-body controller;
J. same agent planner + HLB state/memory + the same local controller.

Separate HDC encoding, liquid dynamics, FEP/active inference, episodic memory, procedural memory, active sensing and planner identity with individual ablations. Include parameter-matched and compute-budget-matched recurrent baselines. Do not attribute a gain from a stronger proprietary planner to HLB.

### Metrics and prospective targets

Report verified task success by task cell, held-out tool/task/environment transfer, contact prediction and calibration, correct abstention, recovery success and burden, repeated-failure rate, protected-skill retention, poisoned/stale-memory rejection, false verifier acceptance, HLB and controller p50/p95/p99 latency, deadline misses, model-reasoning wall time, plant time, energy, compute, human intervention and accepted work per robot-hour. Unsafe action admission, loss of local fallback and false return-to-service are non-compensable failures.

Prospective targets (not current results):
- RoboLab-120: stretch target >=90% under the official 3,000-episode protocol, versus the current 86.0% leaderboard point estimate. This is not a humanoid claim and not directly comparable to HumanoidToolBench.
- HumanoidToolBench: >=10 percentage-point paired improvement on a preregistered held-out mobile/decoy composite over the strongest locally reproducible baseline, after G1 adapter runtime conformance; publish all 18 cells separately.
- Own held-out transfer suite: >=80% verified success on same-family and task-composition transfer.
- >=20% relative reduction in recovery/intervention burden without reducing verified success; separately test a 4x reduction in repeated failure after memory, with protected-skill and poisoning controls.
- >=90% verified completion on the low-risk service suite, with zero false restoration acceptances in its frozen negative controls.
- HLB p99 update latency must fit a measured profile-specific budget without consuming local controller/watchdog deadlines. Do not invent a universal update frequency.
- All frozen timeout/stale/reset tests must retain the declared local fallback or enter the declared restrictive state. This finite result is not a universal safety guarantee.

The full experimental contract, ablation arms, metrics, gates and no-claim list live in [hlb-humanoid-ablation-v0.1.md](hlb-humanoid-ablation-v0.1.md). No HLB ablation has been run and no SOTA improvement is claimed.

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

## 7B. Failure-to-skill learning, independent promotion and retention

New child programme: [HUM-FAILURE-TO-SKILL-001 #12706](https://github.com/Luminous-Dynamics/symthaea/issues/12706), complementary to agentic policy/memory #12685 and capability retention #11866.

The core distinction:
- episodic memory remembers what happened;
- semantic memory records what was learned about tools/embodiments;
- procedural memory stores a reusable skill contract;
- only independently qualified evidence may promote a candidate procedure.

A failed episode must never overwrite a previously qualified skill. It creates an evidence record and a *candidate* correction. The correction must pass exact failure replay, counterfactual control, held-out transfer where claimed, prior-skill retention, exact applicability binding and an independent verifier before promotion.

The synthetic challenge corpus has 30 cases across six skill families (contact approach, grasp retention, tool orientation, force-limited insertion, mobile reposition and perception/calibration drift). Each family includes:
1. valid causal correction;
2. irrelevant confounding change;
3. poisoned memory record;
4. wrong embodiment / stale applicability;
5. protected-skill regression.

Artifacts:
- `failure-to-skill-cases-v0.1.json`
- `failure-to-skill-cases-v0.1.schema.json`

The evaluator retains hidden causal labels, counterfactual outcomes and expected dispositions; the candidate sees only the episode evidence and budget. These are invented synthetic traces, not measured robot failures.

Primary measurements include repeated-failure rate, recovery after informative observations, candidate correction precision/recall, false promotion/revocation, independent replay, held-out transfer, harmful transfer, protected-skill retention, poisoned/stale memory rejection and time/probe/compute/energy costs.

Prospective research targets (not results): reduce repeated failure by at least 50% under matched replay; ≥80% independent replay success for promoted corrections in their declared envelope; ≥60% held-out same-family transfer for skills claiming reuse; non-inferior protected prior skills; and zero false promotions on frozen poisoned, wrong-embodiment, stale-calibration and verifier-spoofing controls. Finite synthetic zero-event results do not certify physical safety.

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

### Event-level capability transition reporting

Use the dependency-free evaluator in [capability-transition-ledger](capability-transition-ledger/), tracked by [HUM-CAPABILITY-TRANSITION-001 #12741](https://github.com/Luminous-Dynamics/symthaea/issues/12741). Every episode carries five event records: candidate contact, correct first contact, correct-tool lift, tool-target contact and independently verified task success. Each record is explicitly `NotInstrumented`, `InstrumentedAbsent` or `OccurredAt(source_timestamp)`.

Report, per policy/split/scenario/L0-L1-L2/standard-decoy cell:
- event incidence among episodes where the event was instrumented;
- jointly instrumented episode count for each adjacent event pair;
- source-event occurrences in that eligible subset;
- co-occurrence regardless of temporal order;
- temporally ordered conversion and mean latency (target time ≥ source time);
- capture coverage and explicit uninstrumented counts.

This is **not** a mandatory funnel. Regrasp, repeated contact and recovery may produce noncanonical event order. Keep both co-occurrence and temporal conversion; do not silently drop out-of-order episodes or turn missing instrumentation into an absent event. The evaluator rejects duplicate episode IDs and empty provenance and keeps policy/task/split/level/mode strata isolated. Eight Rust test cases are authored. Instrumented task-success evidence requires a separately identified verifier source distinct from the policy and event recorder. The JSON Schema and six synthetic fixtures are under capability-transition-ledger/. A static readback check found no fixture-shape/verifier-provenance violations; the Rust tests have not been executed locally.

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
- repeated-failure rate, recovery success, failure-to-new-skill conversion and negative transfer after memory update;
- active tool-discovery accuracy, property calibration, information gained per probe, probe cost and held-out task transfer;
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

## 12. Immediate decision sequence and current status

1. **Pin upstream source, assets, checkpoint, license and resources — completed as a source audit.** The exact HumanoidToolBench commit and dataset revisions are recorded in the manifest. AMP1 source, licence and H100/VRAM constraints are also audited. This does not mean the upstream runtime has been installed.
2. **Establish G1/control/observation compatibility — raw simulation qpos order resolved; UEC v2 runtime adapter pending.** The 43-value chain is traced through `SonicLocoManip._get_obs`, `MujocoEngine.get_robot_qpos` and `G1Sonic.get_robot_qpos`; the mapping manifest records the three producer files/blob IDs. A pure, non-actuating Rust mapping audit crate and nine authored tests are now in the branch. UEC v2 platform integration and runtime conformance remain blocked on its canonical APIs and qualification.
3. **Freeze the 18-cell HumanoidToolBench protocol and baseline ledger — specification complete.** Event-level capability-transition evaluator is now implemented in a pure-Rust crate (tests authored, not run); see #12741. No local benchmark is run or claimed yet. First runtime work should be a one-episode diagnostic / one-condition 100-episode run after environment and compute qualification.
4. **Build active tool discovery — protocol/corpus defined.** The 24-case synthetic corpus spans six families with development, held-out instance, transfer and decoy splits. The candidate view must exclude hidden physical parameters and evaluator oracles. No physics episodes have run.
5. **Run pure mapping audit locally/CI — authored, not yet executed.** The crate is dependency-free and has a focused workflow. The current editing environment has no `rustc` or `cargo`; no test/format/CI pass is claimed.
6. **Run no-op/scripted probes before costly model work.** Use matched budget and sealed-oracle controls; do not use benchmark success flags as the only postcondition evidence.
7. **Run matched agentic/VLA/Symthaea simulation ablations.** Separate model/provider, reasoning effort, harness, action interface, reasoning wall time, action-motion time and compute/API costs. Do not copy AMP source without licence review.
8. **Advance human-augmentation evaluation in parallel.** Compose HUMAN-TOOL-001 #7983 / PR #7997; this is a human + advisory outcome, not humanoid dexterity.
9. **Progress to physical work only after HIL, topology, calibration, instrumentation and independent physical qualification prerequisites are met.** Select one bounded, low-risk tool task; no physical actuator authority is implied by software mapping or synthetic corpus passes.

## 13. Claim ceiling

This protocol defines a prospective evaluation. It does not establish that Symthaea has tool competence, service capability, self-maintenance, wilderness capability, physical safety, SOTA performance or household generality. No benchmark pass creates actuator authority, human-contact consent, safety certification or production release.
