# HUM-TOOL-SERVICE-001 — v0.1

**Status:** protocol and case-specification research only  
**Issue:** [#12657](https://github.com/Luminous-Dynamics/symthaea/issues/12657)  
**Base:** `main@d67bb3ea4824fde5e95c721dd28dc9cc2b064bd9`  
**Branch:** `research/humanoid-tool-service-v0-1`

## Objective

Determine whether Symthaea can acquire transferable tool competence and complete bounded maintenance tasks with independently verified outcomes.

The unit of capability is not “recognizes a wrench” or “can grasp an object”. It is a complete, inspectable chain:

```
interpret goal
  -> identify affordances and constraints
  -> select tool and part
  -> approach / grasp / secure
  -> predict contact consequences
  -> use tool (stationary or mobile)
  -> detect effect, error and uncertainty
  -> recover, abstain or hand off
  -> independently verify postcondition
```

For maintenance, extend the chain:

```
inspect -> isolate -> diagnose -> plan -> disassemble
  -> repair / replace -> reassemble -> recalibrate
  -> independently test -> accept or quarantine
```

A robot's own completion flag is not the verifier.

## Why this programme is different

We will test seven hypotheses rather than assume them:

1. **Tool-as-body-schema:** a held tool changes effective reach, collision envelope, inertia, contact geometry and affordances. The robot should reason over the combined embodiment + grasp + tool, not just the object's class.
2. **Functional transfer:** infer what a tool can do from its physical/functional properties, not from memorized tool names or trajectories.
3. **Active physical inquiry:** choose bounded, low-risk probes when resolving an unknown property would change the action.
4. **Contact prediction:** predict contact transition, slip, engagement, load and task effect; compare forecasts to native observations.
5. **Compositional service:** combine reusable skills into verifiable diagnosis and restoration workflows.
6. **Cross-embodiment portability:** share skill semantics while keeping morphology, tool interface, sensor calibration, actuation and safety envelopes embodiment-specific.
7. **Calibrated abstention:** recognize insufficient evidence, damaged tools, incompatible parts, missing isolation or stale calibration and stop/handoff correctly.

## Current Symthaea starting point

The public humanoid crate already contains HDC-LTC control, FEP regulation, morphology profiles, state estimation, terrain/contact/whole-body control, deterministic safety projection and fall recovery. The morphology profiles include 21-, 27-, 53- and 64-actuator variants. The current task enum is limited to Stand, Walk, Run, Reach and Grasp.

The 53-DOF DAgger example currently configures the Stand task; its OOD correction segment samples a simple humanoid simulator and PD standing targets. It does not demonstrate learned tool use. The inspected HAL servo path is a 21-joint PCA9685 implementation, not a dexterous 53-DOF hand interface.

The benchmark's Unitree G1/Dex3 embodiment is a separate physical/configuration subject. **No direct morphology, actuator ordering, units, observation space or control-mode equivalence is assumed.** A compatibility adapter must be specified and tested before any comparison is called matched.

## HLB in the humanoid: the recommended role

**Use Symthaea's Holographic Liquid Brain as the persistent embodied-cognition substrate, not as the hard-real-time servo controller or safety authority.** The HLB hypothesis is that HDC composition, liquid temporal state, prediction, uncertainty, memory and active inference can make the robot better at long-horizon adaptation, tool use and recovery. It must earn that role through matched ablations; high-dimensional representations are not presumed superior.

The current modern trainer already contains an HDC-LTC path: a 16,384-dimensional network feeds a 64-dimensional bottleneck and a morphology-sized learned residual. That residual is composed with task-specific PD baselines and the contact-aware hierarchical controller, then passes fall protection, deterministic safety projection and backend actuation adaptation. Current default trainer configuration is 40 Hz physics and 10 Hz FEP/cognitive cadence. Recurrent learning exists as an optional training path, but is disabled by default; recurrent parameter/state serialization and resume are explicitly blocked, and checkpoints truthfully mark recurrent state as absent.

This is not the same as the legacy `HumanoidEmbodiment::step(thought_hv, dt, phi)` bridge, which consumes an external thought hypervector and uses the older Phi-derived motor tier. Do not conflate those paths or infer physical capability from the architecture.

Recommended composition:

- HLB: embodied belief, body+tool schema, temporal/contact prediction, calibrated uncertainty, active sensing, episodic/semantic/procedural memory and candidate skill learning.
- Skill executive: typed bounded task-space goals, preconditions, progress, retries, recovery and independent postconditions.
- Existing whole-body/contact controllers: balance, inverse dynamics, grasp and trajectory tracking.
- Independent safety/local control: admission, watchdog, e-stop, force/collision limits and qualified fallback; cognition can restrict goal-directed authority but cannot establish plant safety.

The proposed matched baseline matrix (deterministic baseline, MLP, GRU, static HDC, current HDC-LTC, checkpointable recurrent HDC-LTC, HLB memory variants, agentic planner, and planner+HLB hybrid), metrics and prospective targets are specified in [HLB humanoid ablation protocol v0.1](hlb-humanoid-ablation-v0.1.md). The experiment is prospective; it has not established an HLB performance gain.

**Hard dependencies:** recurrent snapshot/restore #1269; cognitive/plant/transport time separation #1284; UEC v2 #1274 and qualification #1288; independent plant-safety/readiness split #1276. Reuse those owners rather than introducing a second recurrent-state, clock or safety contract.


### HLB component maturity: important integration caveat

The full HLB architecture is not yet a turnkey multimodal robot world model. The current `LiquidHolocell` is a first-order leaky hypervector state; its dimensionality dilation tiles/permutates existing coordinates and has not established additional independent information. The `PredictiveHdcEncoder` currently accepts text strings and externally supplied predictions; it is not a native camera/tactile/force/joint-state predictive encoder. The legacy ROS2 bridge has assumed/stubbed state fields and Φ-derived effort scaling, and is not the UEC v2 qualified G1 adapter. See §2.5 of [the HLB ablation protocol](hlb-humanoid-ablation-v0.1.md) for source paths, exact behavior and migration boundaries.

Therefore, the first experiment should reuse the modern humanoid trainer's existing observed-state HDC-LTC residual path. Typed multimodal encoders, action-conditioned contact prediction and HLB memory should be introduced only as measured, separately ablated modules. Do not treat vector-size growth or a consciousness/readiness scalar as physical evidence, predictive accuracy or actuation authority.

## Ownership map

This programme composes existing owners and must not duplicate them.

| Concern | Canonical owner |
|---|---|
| Tool, platform, role and workcell specialization | [ROB-SPEC #6033](https://github.com/Luminous-Dynamics/symthaea/issues/6033) |
| Repair parts and productive self-maintenance closure | [ROB-MFG #5775](https://github.com/Luminous-Dynamics/symthaea/issues/5775), [SELF-BOOT #6229](https://github.com/Luminous-Dynamics/symthaea/issues/6229) |
| Degradation, reliability and maintainability | [ROB-LIFE #4863](https://github.com/Luminous-Dynamics/symthaea/issues/4863) |
| Capability observatory and external benchmark import | [HUM-OBS #4780](https://github.com/Luminous-Dynamics/symthaea/issues/4780), [#4781](https://github.com/Luminous-Dynamics/symthaea/issues/4781) |
| Benchmark validity and transfer claims | [ROB-BENCH-AUDIT #11231](https://github.com/Luminous-Dynamics/symthaea/issues/11231) |
| Tactile co-design / representation / native metrology / prediction | [#5821](https://github.com/Luminous-Dynamics/symthaea/issues/5821), [#12477](https://github.com/Luminous-Dynamics/symthaea/issues/12477), [#12484](https://github.com/Luminous-Dynamics/symthaea/issues/12484), [#12500](https://github.com/Luminous-Dynamics/symthaea/issues/12500), [#12548](https://github.com/Luminous-Dynamics/symthaea/issues/12548) |
| Contact-rich recovery | [#11606](https://github.com/Luminous-Dynamics/symthaea/issues/11606) |
| Design, experiment, instrumentation and physical article | [ROB-DESIGN #4795](https://github.com/Luminous-Dynamics/symthaea/issues/4795), [ROB-EXP #4800](https://github.com/Luminous-Dynamics/symthaea/issues/4800), [ROB-INSTR #4866](https://github.com/Luminous-Dynamics/symthaea/issues/4866), [JOINTLAB #4814](https://github.com/Luminous-Dynamics/symthaea/issues/4814) |
| Physical grasp and human-contact consent | [#1400](https://github.com/Luminous-Dynamics/symthaea/issues/1400), [#4680](https://github.com/Luminous-Dynamics/symthaea/issues/4680) |
| Accepted productive work accounting | [#11607](https://github.com/Luminous-Dynamics/symthaea/issues/11607) |

This programme owns benchmark protocol, tool/service skill composition questions, and the gap analysis only.

## Research tracks

- **T0 — Paper-faithful tool-use reproduction:** reproduce HumanoidToolBench as published when rights, source, checkpoint and runtime permit.
- **T1 — Transfer and shortcut resistance:** held-out tool instances/families, unseen environments, tool/task recombination, decoys, irrelevant instructions and sensor faults.
- **T2 — Verified service loops:** simulation and passive/low-energy bench fixtures for repair, replacement, fit-check and return-to-service.
- **T3 — Field readiness:** outdoor navigation, communications, power budgeting and low-risk field-kit maintenance only after T0–T2 show credible capability.

No single aggregate humanoid score. Every task result retains robot, tool, environment, task, model, evidence class, uncertainty and failure identity.

## HumanoidToolBench compatibility, rights and resource gate

The official evaluator source was inspected at `SNU-PI/HumanoidToolBench@3fec9bd36933a1648ff7f95468c983f3452243a0` on 17 November 2026.

- Evaluator code is MIT-licensed.
- ToolBook simulation and real-G1 recordings are CC BY-NC 4.0. This is non-commercial research data; do not use its recordings or derived artefacts in a commercial product/training pipeline without separately reviewed permission. Third-party models and assets retain their own terms.
- The simulator policy contract is MuJoCo 3.11.0 / Python 3.10, with 32 observation values and 36 `decoupled_v1` action values. The real-G1 recordings have a distinct 35-value action contract. Neither is interchangeable with Symthaea's 21/53 actuator command arrays by padding or truncation.
- Upstream standard budget: 100 episodes per condition, seeds 10000–10099, 3000 policy steps at 50 Hz. Maintainers estimate roughly 4 GB setup download, 9 GB disk, another 2 GB for an initial checkpoint, around two hours per condition and about 1.5 sequential days for all 18 on one GPU. These are upstream estimates, not local measurements.
- The evaluator provides an HTTP policy interface, so a future Rust policy server can be written without making the Symthaea controller a Python production dependency. First, however, a semantics-preserving G1 observation/action adapter must exist and be qualified.

For Luminous commercial work, the benchmark's task design and MIT evaluator may be informative, but any training data and redistributed assets must have rights that permit the intended use. The CC BY-NC ToolBook path is research-only absent additional permission.

## Complementary research anchors

HumanoidToolBench is the direct humanoid tool-use anchor. We will also audit:
- **Evo-DexBench** (arXiv:2610.21180): functional transfer across 30 tasks / six related-task families; best reported average success is 38.5%. This lets us test whether a learned interaction primitive transfers to a new task with few demonstrations.
- **DexTouch-WM** (arXiv:2609.20649): action-conditioned visual/tactile prediction trained with compatible human and robot tactile interfaces; five hours of robot supervision are held fixed while human tactile experience scales to 100 hours.
- **PhysCanvas** (arXiv:2610.28613): state-consistent visuo-tactile world modeling; reported world-model/real-success correlation rises from 0.602 to 0.885 with state-reward supervision.
- **Tabero** (ICML 2026): closed-loop force feedback and a multidimensional gentle-manipulation benchmark; reports over 70% lower average grip force under gentle instructions while maintaining task success.

These results suggest that Symthaea should be judged on *functional transfer, contact prediction, interaction quality and verified physical outcomes*, not only task completion. All values above are paper-reported and task-specific; they are not directly comparable or Symthaea results. Reproduction requires exact source/checkpoint/data/licence and protocol audit.


**Direct humanoid whole-body comparator — ULTRA (IROS 2026):** ULTRA is a unified multimodal Unitree G1 controller that supports both dense reference tracking and sparse goal-conditioned loco-manipulation from egocentric perception, with physics-driven motion retargeting, student distillation and RL fine-tuning. Its project page demonstrates long-horizon goal following and real G1 deployment; it is therefore a closer control/embodiment comparator for Symthaea's humanoid hierarchy than AMP's Franka real-robot trials. It is not a tool-selection benchmark or a persistent cognitive-memory system. The public repository states ULTRA code is Apache-2.0 and the InterMimic-derived stack MIT, while robot assets, motion datasets and third-party dependencies keep their own terms. Its released training stack uses Python / IsaacGym Preview 4 and MuJoCo, so reproduction/runtime cost and data rights need separate qualification. Sources: [ULTRA project and real-robot demonstrations](https://ultrahumanoid.github.io/), [paper](https://arxiv.org/abs/2603.03279), [public code and licence notes](https://github.com/Sirui-Xu/ULTRA).

## Current SOTA shift: agentic policies and durable skill memory

The benchmark portfolio now includes **agentic manipulation policies**, not only VLA/WAM action-chunk policies.

AMP1 (arXiv:2611.09892, 6 Nov 2026) reports:
- 85.6% zero-shot RoboLab-120 success in the paper (Gemini 4 Argon); the current NVIDIA RoboLab leaderboard lists 86.0% ±2.5 percentage points from 2,579/3,000 episodes. Keep the paper result and live leaderboard snapshot separate.
- 95.2% on LIBERO-PRO, 100% on the selected robosuite suite, and 96.0% on two BEHAVIOR-1K mobile tasks on the AMP project page.
- 84.1% on RoboDojo **after practice**, not zero-shot. The paper's carried-memory analysis reports a 4.6× lower failure rate on RoboDojo bimanual tasks in its setup.
- 104/108 real-world trials across 12 tasks on a 7-DoF Franka Research 3, not a humanoid.

Sources: [AMP paper](https://arxiv.org/abs/2611.09892), [AMP results](https://apple-aiml-research.github.io/ml-amp/), [NVIDIA RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html?family=all), [RoboDojo Agent Track](https://robodojo-benchmark.com/leaderboard?bench=sim).


The 17 November NVIDIA RoboLab snapshot also lists SenseNova-RoboRSI at 57.1% (685/1,200 episodes), Lakeside at 51.0% (1,531/3,000), BiMind v0.5 at 49.2% (590/1,200), Praxis-1 at 46.2%, FLUX 3 Action at 42.9%, and Cosmic at 42.3%. Several comparators use 10 episodes/task while AMP1 uses 25, so report denominators and uncertainty and do not interpret this as a paired significance test. RoboLab is a separate simulation suite, not a G1 tool-use or physical humanoid qualification benchmark. The full row-by-row snapshot is pinned in §3C.1 of [protocol-v0.1.md](protocol-v0.1.md).

RoboDojo's 17 Nov public snapshot distinguishes its standard policy leaderboard from the Agent track. The 14 Nov rules mark some earlier entries with an asterisk because they were tested before the updated Agent Track rules; do not treat these as compliant submissions or compare Agent Track averages directly with standard policy results or HumanoidToolBench percentages.

The AMP agent harness uses a persistent Python interpreter, active perception and checked/refused motion calls. Two limitations are particularly relevant to Symthaea: the authors note slow model reasoning and inability to replan an executing trajectory; the RoboLab walkthrough pauses simulation while the model reasons. Our architecture should combine a slow planner with a qualified local fast loop that can stop/hold/recover during contact changes without waiting for another model turn. Report both benchmark-motion time and end-to-end wall-clock time, charging API cost, compute, retries, energy and human intervention.

**Reproduction and licence boundary:** the AMP source repository uses an Apple custom personal, non-exclusive licence, not a conventional MIT/Apache-2.0 licence; third-party components retain separate terms. Do not copy the AMP source into Symthaea or redistribute a fork without specific licence review. The official reproduction guide lists roughly 9 GB VRAM, 15 GB RAM and 40 GB disk for RoboLab; the paper's simulation runs used H100 80 GB GPUs. The known RTX 2070 Laptop 8 GB configuration is below the quickstart's stated VRAM figure, so do not assume the full official stack will run there. A separate lower-resource route would need its own measured qualification.

VioLA (8 Oct 2026) motivates a second output-space baseline: predict transferable body/hand motion latents and let a qualified whole-body controller track them, rather than predicting raw joint commands. Its reported training set includes 140.6M frames, 93.2% human motion; human motion is not automatically robot-compatible without a valid morphology/frame/action contract.

Symthaea's hypothesis is to combine:
- a generalist agent that remains in the task loop and performs active perception / plan revision;
- typed, bounded capabilities instead of unrestricted generated code with direct live hardware access;
- HDC-LTC / FEP multi-timescale cognition and local fast controllers;
- validated, provenance-bearing procedural skill memory;
- terminal deterministic safety, explicit authority and independent physical postcondition verification.

Related programme: [HUM-AGENTIC-POLICY-001 #12685](https://github.com/Luminous-Dynamics/symthaea/issues/12685).


### Physical humanoid frontier: where we should compete

The public industry picture has moved beyond isolated dexterity demos. Figure reports Helix 2.5 at 56% zero-shot success across three whole-body behaviors in 30 unseen homes, versus 9% from scratch with the same task-specific data; Google DeepMind reports Gemini Robotics 2.5 across 200+ tasks and 96% in a Mercedes-Benz automotive-kitting proof-of-concept with Apptronik Apollo 2. These are vendor-reported, task-specific results and not a common leaderboard. ULTRA remains the closer real-G1 whole-body controller comparator; AMP1 is the agentic planner comparator; HumanoidToolBench is the direct G1 tool selection/use benchmark.

A particularly revealing HumanoidToolBench result is the L1-to-L2 mobile gap: FastWAM's standard BallMove success falls from 76% stationary to 9% mobile, while the benchmark records candidate contact, correct first contact, correct-tool lift, tool-target contact and task completion as separate events. A secondary extraction of the paper's event table shows high interaction rates can coexist with much lower verified task success. That is the key opportunity for Symthaea: do not optimize one terminal score while remaining blind to why the task failed.

Our next benchmark improvement is a **capability-transition ledger**: event timestamps, capture completeness, transition incidence, conditional conversion, time-to-transition, stage-specific recovery/intervention cost and verified terminal outcome. HLB should use it to choose targeted reinspection, affordance probing, regrasp/reorientation, contact adjustment or mobile repositioning. The five-event graph must not be treated as a mandatory funnel; keep out-of-order events and repeated attempts. See section 3E of [the benchmark protocol](protocol-v0.1.md).

## Active tool affordance discovery

[HUM-TOOL-DISCOVERY-001 #12701](https://github.com/Luminous-Dynamics/symthaea/issues/12701) asks whether Symthaea can identify what an unfamiliar tool can physically do, not merely classify its name or choose among known objects.

The first synthetic corpus has 24 cases across lever, hook, wedge, rotary driver, clamp and mallet families: six development cases, six held-out same-family cases, six held-out task-transfer cases and six visually similar decoys. Hidden parameters, split labels and postcondition oracles are evaluator-only.

Artifacts: [24-case manifest](tool-discovery-cases-v0.1.json) · [schema](tool-discovery-cases-v0.1.schema.json).

This is protocol/synthetic data only. It contains no measured physical tool properties, material-strength, safe-load or serviceability claims. V0.1 excludes powered tools, blades, high forces, mains voltage, hazardous substances and human contact.

## G1 interface compatibility artifact

The exact HumanoidToolBench G1 action/state mapping is now pinned separately:
- [G1 decoupled_v1 mapping manifest](g1-decoupled-v1-mapping.json)
- [G1 mapping JSON Schema](g1-decoupled-v1-mapping.schema.json)
- [HUM-G1-UEC-ADAPTER-001 #12681](https://github.com/Luminous-Dynamics/symthaea/issues/12681)

The mapping preserves all 43 raw named G1 joint positions, the derived 32-value policy-state projection, the 36-value action schema, the left-hand thumb/middle/index permutation, the waist order change, the base-height command-history field, RGB image modality, and navigation turn-flag semantics.

**The raw 43-value qpos ordering is now source-traced**, not assumed. The producer chain is `SonicLocoManip._get_obs → MujocoEngine.get_robot_qpos → G1Sonic.get_robot_qpos`; the final dictionary was populated by iterating the pinned `G1Sonic.joint_names`/`WHOLE_BODY_JOINTS` list, and Python preserves insertion order. The producer files and blob IDs are recorded in the mapping manifest.

A dependency-free Rust audit crate is also present at [g1-mapping-audit](g1-mapping-audit/README.md). It includes named and pinned-order state projection, action decode/round-trip, nine authored golden/negative-control tests, a minimal lockfile and a focused workflow. Its tests have not been executed because this editing environment does not have `rustc`/`cargo`.

This remains a **non-actuating software mapping specification**, not a qualified UEC v2 platform adapter. It does not map G1 joint targets onto Symthaea DMC21/Dexterous53 torques, does not connect to the current 21-joint PCA9685 HAL, and creates no TransportAccepted/Applied/Measured receipt or actuation authority. UEC v2 integration and runtime conformance remain separate gates.

## Active tool affordance discovery

A new child programme, [HUM-TOOL-DISCOVERY-001 #12701](https://github.com/Luminous-Dynamics/symthaea/issues/12701), tests whether Symthaea can identify what an unfamiliar tool can physically do—not merely recognize a label or choose among familiar assets.

The first synthetic corpus contains 24 cases across six families: lever, hook, wedge, rotary driver, clamp and mallet. Each family has a development instance, held-out same-family instance, held-out task-transfer instance and a visually similar decoy. The evaluator retains hidden parameters and postconditions; the candidate sees only its declared observations and bounded probe budget.

Artifacts:
- [Tool discovery cases](tool-discovery-cases-v0.1.json)
- [Tool discovery schema](tool-discovery-cases-v0.1.schema.json)

This is synthetic protocol data only. It contains no measured physical tools, material-strength claims, safe-load claims or physical task results.

## Executable G1 mapping audit

A small dependency-free Rust crate now implements the G1 policy-state projection and `decoupled_v1` action decode/encode:
- [Rust crate README](g1-mapping-audit/README.md)
- [Rust source](g1-mapping-audit/src/lib.rs)
- [Focused workflow](../../.github/workflows/g1-mapping-audit.yml)

It contains nine authored golden/negative-control tests and a minimal lockfile. The raw 43-value `joint_qpos` order is now traced at the pinned upstream commit through:
`SonicLocoManip._get_obs → MujocoEngine.get_robot_qpos → G1Sonic.get_robot_qpos`.
The producer chain and blob IDs are recorded in the mapping manifest. The crate provides both explicit name/value projection and a pinned-order projection helper.

**Important distinction:** this resolves the raw-qpos ordering blocker for the pinned simulation source. It does not yet provide a UEC v2 platform implementation, runtime adapter conformance, real-G1 hardware binding, physical calibration, safety qualification or actuator authority. The Rust tests are authored but have not been executed in this environment.

## Failure-to-skill learning and protected capability retention

A second child programme, [HUM-FAILURE-TO-SKILL-001 #12706](https://github.com/Luminous-Dynamics/symthaea/issues/12706), measures whether a failed interaction can produce a reusable correction without promoting an unverified episode or degrading older skills.

Its synthetic corpus includes 30 cases across six skill families. Each family contains a causal-correction case, irrelevant confound, poisoned memory record, wrong-embodiment/stale applicability case and protected-skill regression case.

Artifacts:
- [30-case failure-to-skill corpus](failure-to-skill-cases-v0.1.json)
- [Corpus schema](failure-to-skill-cases-v0.1.schema.json)

Promotion is candidate-only until independent replay, counterfactual testing, held-out transfer, protected-skill retention and verifier evidence all pass. No live controller or skill is modified by this research tranche.

## Human-augmented repair is a parallel delivery path

Symthaea should be useful before its humanoid is physically qualified to perform every repair itself.

Compose the existing [HUMAN-TOOL-001 #7983](https://github.com/Luminous-Dynamics/symthaea/issues/7983) and draft [PR #7997](https://github.com/Luminous-Dynamics/symthaea/pull/7997) rather than duplicating its advisory diagnostic contract or repair-outcome evaluator.

That programme compares:
- A0 — human-only with current manuals/procedures;
- A1 — human with structured checklist and the same instruments;
- A2 — human + Symthaea evidence-linked advisory, which may rank hypotheses, surface contradictions/missing evidence, recommend a bounded next measurement, abstain or refer.

The outcome is independently verified useful-function restoration, with failed/abandoned/referred/unknown cases retained. The advisory has no actuation authority and must not override safety interlocks. This is a **human-augmentation result**, not a humanoid dexterity result.

Longer term, the two tracks can share validated tool semantics, fault knowledge and repair skill contracts, but neither may promote an unverified diagnosis or human demonstration directly into autonomous robot authority.

## Deliverables

- `protocol-v0.1.md`: preregisterable comparison and claim-boundary protocol, including agentic SOTA, human assistance and active tool discovery.
- `hlb-humanoid-ablation-v0.1.md`: source-audited HLB architecture decision, MLP/GRU/HDC-LTC/agent/ULTRA ablation matrix, benchmark targets, dependencies and claim ceilings; tracked by [HUM-HLB-ABLATION-001 #12723](https://github.com/Luminous-Dynamics/symthaea/issues/12723).
- `cases-v0.1.json` / `schema-v0.1.json`: exact 18-cell benchmark anchor, eight service-case definitions and negative controls.
- `g1-decoupled-v1-mapping.json` / `g1-decoupled-v1-mapping.schema.json`: exact G1 state/action mapping and source-traced raw-qpos producer chain.
- `g1-mapping-audit/`: dependency-free Rust mapping primitives, golden/negative-control tests and focused workflow.
- `tool-discovery-cases-v0.1.json` / `tool-discovery-cases-v0.1.schema.json`: 24 synthetic active-affordance discovery cases across six tool families.
- `failure-to-skill-cases-v0.1.json` / `failure-to-skill-cases-v0.1.schema.json`: 30 synthetic cases for causal skill correction, poisoned memory, applicability and protected-skill retention.
- `capability-transition-ledger/`: dependency-free Rust evaluator, receipt schema and six synthetic fixtures for five HumanoidToolBench events, missing-vs-absent evidence, co-occurrence, temporal conversion, stage latency and independent task-success verifier provenance; tracked by [HUM-CAPABILITY-TRANSITION-001 #12741](https://github.com/Luminous-Dynamics/symthaea/issues/12741).

These artifacts define a prospective benchmark. They do not indicate that benchmark episodes were run or that any task has been solved.

## Initial success criterion

A first research win is not “Symthaea beats Figure”. It is a credible, reproducible, matched experiment showing that one or more proposed mechanisms (functional tool model, tactile/contact prediction, active probing, compositional recovery) improve **held-out verified task success** over a strong baseline at a declared data/compute/latency budget, without increasing unsafe action, false acceptance or unreported human intervention.

## Claim ceiling

Protocol consistency is not robot capability. A simulation pass is not HIL. HIL is not physical qualification. Physical task qualification is not unrestricted deployment, human-contact safety, clinical competence or legal/product certification. No actuator authority is introduced here.
