# HLB-HUMANOID-ABLATION-001 — v0.1

**Status:** prospective experiment contract; no new benchmark execution or superiority claim  
**Parent:** [HUM-TOOL-SERVICE-001 #12657](https://github.com/Luminous-Dynamics/symthaea/issues/12657)  
**Policy owner:** [HUM-AGENTIC-POLICY-001 #12685](https://github.com/Luminous-Dynamics/symthaea/issues/12685)  
**Dependencies:** UEC v2 #1274 / qualification #1288; recurrent snapshot/restore #1269; plant-time contract #1284; safety/authority split #1276; benchmark validity #11231

## 1. Decision

Use the Holographic Liquid Brain (HLB) as the humanoid's **persistent embodied cognition and temporal state substrate**, not as the hard-real-time servo loop, universal safety authority, or unrestricted direct torque generator.

The research proposition is:

> A temporally persistent, hyperdimensional embodied state model with explicit uncertainty, causal predictions and validated procedural memory can improve long-horizon humanoid task completion and recovery over capacity-matched recurrent baselines, while leaving hard-real-time balance, contact and actuator protection to qualified local controllers.

This is a falsifiable proposal. A matched GRU/MLP or VLA/WAM baseline may win. HLB earns a larger role only if the ablations demonstrate better held-out transfer, recovery, memory retention, calibration or total operational cost without increasing unsafe action admission.

## 2. Source audit: what is already real

Source snapshot inspected on 17 November 2026. The following distinctions are important.

### 2.1 Current trainer path already consumes HDC-LTC

- `crates/domains/symthaea-humanoid/src/controller.rs` constructs a `HdcLtcUnifiedNetwork` at `HDC_DIMENSION`, uses a 64-dimensional bottleneck, and projects the result to a morphology-sized learned residual command.
- `HumanoidController::forward` evolves the HDC-LTC state from the current sensor hypervector and emits the residual.
- `crates/domains/symthaea-humanoid/src/training.rs` obtains a typed/state-estimated observation, encodes it with `HumanoidHdcEncoder`, calls the HDC-LTC controller, combines that residual with a task-specific PD baseline through `HierarchicalHumanoidController`, applies fall protection and the deterministic safety projector, adapts the command to the backend, and only then steps the simulator.
- `crates/domains/symthaea-humanoid/src/hierarchical.rs` owns contact-aware balance, recovery, terrain/footstep planning and whole-body/inverse-dynamics layers. This deterministic substrate should remain; HLB should not rediscover basic stability reflexes.
- `crates/domains/symthaea-humanoid/src/types.rs` defaults to 40 Hz physics, 10 Hz cognitive/FEP ticks and training every two physics steps. These are defaults of this trainer configuration, not evidence of a physical robot's measured rates.

### 2.2 Learning and checkpoint limits are explicit

- `enable_recurrent_learning` defaults to `false`.
- The default training path can update the output head using replay while retaining the initialized recurrent network dynamics.
- The optional recurrent-learning path updates the HDC-LTC neurons, but current checkpoint/resume code rejects it because recurrent parameters/state cannot yet be serialized and restored as a complete learner.
- `ControllerCheckpoint` writes `recurrent_state_included = false`; it must not be described as a full recurrent learner checkpoint.
- Core `HdcLtcUnifiedNetwork` supports irregular timestamp stepping and serializable Rust structures, but that does not by itself provide the humanoid controller with a versioned, tested recurrent snapshot/restore contract. Reuse #1269 rather than creating a second latent-state format.

### 2.3 There is a legacy bridge path to keep distinct

`crates/domains/symthaea-humanoid/src/embodiment.rs` contains a simpler `HumanoidEmbodiment::step(thought_hv, dt, phi)` path. It consumes an external thought hypervector, derives a legacy motor tier from Phi, steps its simple simulator, and encodes post-step perception for telemetry. That is not the same evidence chain as the modern trainer's observation-encoded, hierarchical, safety-projected loop.

Do not silently treat the legacy bridge, the training simulator, the 21-actuator PCA9685 HAL path, a 53-DOF morphology, and Unitree G1 as interchangeable embodiments. The G1 decoupled adapter remains a separate UEC/runtime-conformance gate.

### 2.4 Explicit dependency gates

- #1269: checkpoint and restore model parameters, mutable recurrent state, caches and timing state with deterministic continuation evidence.
- #1284: separate HLB internal integration time, cognitive scheduler cadence, plant time and transport latency. HLB must never redefine physical elapsed time through its internal `tau`.
- #1274 / #1288: expose typed observation, requested/projected/applied action, identity, clock and calibration at the shared embodiment boundary.
- #1276: cognitive readiness may restrict goal-directed action, but cannot positively establish plant safety; local fallback must remain independent of HLB.
- #11231: audit benchmark validity, split leakage and external comparator protocol.


### 2.5 HLB primitive maturity and ROS2 integration boundary

The robot should reuse HLB components where they have measured, typed semantics, but the existence of a named HLB primitive is not evidence that a full multimodal embodied model is already integrated.

**Liquid Holocell** (`crates/core/symthaea-core/src/hdc/liquid_holocell.rs`) currently stores a `ContinuousHV`, scalar `tau`, dimensionality tier and pressure. Its `step` is a first-order leaky-state update:
`H_new = H_old * exp(-dt/tau) + input * (1-exp(-dt/tau))`.
Its current `dilate` path expands by tiling permuted copies of the existing vector and contracts by averaging chunks. This changes vector length but does not, by itself, create new independent information or prove increased representational capacity. Treat dilation as a representation-size transform until orthogonality, retrieval, task accuracy and information-retention experiments establish a stronger claim. The `simulate` helper advances a cloned cell with hard-coded `dt=0.1` and maps state dissimilarity to a [0,1] surprise/load score; it is not a physical power meter, calibrated energy forecast or contact-dynamics simulator.

**Predictive HDC encoder** (`crates/core/symthaea-core/src/hdc/predictive_encoder.rs`) currently accepts `&str`, uses a text encoder and primitive-name detection, compares against a prediction supplied externally via `set_prediction`, and adjusts primitive attention from prediction error. This is useful as a language/semantic predictive primitive. It is not yet a camera, tactile, force/torque or joint-state encoder, nor a full action-conditioned physical world model. Build typed modality-specific encoders and align them by timestamp, frame, calibration and sensor provenance; do not stringify physical readings as the sole multimodal interface.

**ROS2 bridge** (`crates/bridges/symthaea-ros-bridge/src/lib.rs`) is a legacy integration path, not the UEC v2 qualified G1 adapter. The source describes the node as a placeholder; the convenience state conversion injects assumed root pose / heights and zero extremity data, while the current ROS2 state path estimates linear velocity as IMU acceleration multiplied by a fixed 0.025 factor. The command conversion maps humanoid torques to ROS trajectory effort. Its publishing path uses a peak-effort interlock and, under the ROS2 feature, scales efforts using a Phi-derived gain scheduler. These are not substitutes for typed fresh sensor evidence, morphology/calibration-specific constraints, independent `PlantSafetyDecision`, UEC v2 requested/projected/applied receipts, watchdog guarantees or platform-specific fallback qualification. Keep direct actuator publication outside this research branch and do not treat this bridge as production-ready humanoid actuation.

**Integration consequence:** the near-term HLB experiment should first reuse the modern humanoid trainer's existing state encoder + HDC-LTC residual + hierarchical controller, while measuring each contribution separately. The next integration tranche should introduce a typed multimodal observation encoder and predictive state model over measured robot/task variables; it should not attempt to wire the whole HLB monograph or legacy ROS2 bridge directly into a motor loop.

## 3. Proposed architecture: cognition over control, not cognition instead of control

```text
camera / proprioception / tactile / force / tool / environment
                         |
              typed UEC v2 observations
                         |
          state estimation + freshness + provenance
                         |
           HLB embodied belief / temporal state
     (HDC composition + liquid dynamics + prediction)
                         |
     task state / tool affordance / uncertainty / memory
                         |
             bounded skill executive
     (preconditions, parameters, progress, recovery)
                         |
       task-space goal / motion latent / contact intent
                         |
      existing whole-body + contact-aware controller
    (balance, IK, MPC, impedance, inverse dynamics)
                         |
      independent safety admission + local watchdog
                         |
             plant / measured transition
                         |
       execution receipt + independent verifier
                         |
        episodic memory / candidate skill update
                         |
       held-out replay + retention before promotion
```

### HLB responsibilities

1. **Embodied state:** compose body, tool, grasp, environment and task state with morphology, frame, calibration and embodiment identity attached.
2. **Temporal prediction:** forecast contact transitions, slip, load changes, task progress and likely consequences of a proposed skill.
3. **Uncertainty:** represent missing, stale, conflicting or weakly grounded observations explicitly; abstain when a decision depends on unresolved uncertainty.
4. **Active perception:** request the next camera/tactile/force observation or a bounded low-risk probe only when its expected decision value justifies cost and risk.
5. **Memory:** retrieve episodes and validated procedural skills with exact applicability, provenance, version, calibration and expiry conditions.
6. **Failure diagnosis:** propose a correction, then test it against the failure replay, a counterfactual control, held-out transfer and protected-skill retention before promotion.
7. **Cross-task composition:** maintain semantic continuity when a task spans locomotion, tool use, manipulation, inspection and verification.

### Responsibilities that remain outside HLB

- Servo-current, torque, PWM and hard-real-time motor loops.
- Independent e-stop, watchdog, collision/force limits and local fallback.
- Final whole-body feasibility, inverse dynamics and action projection.
- Authorization, consent and physical safety admission.
- Independent verification of service completion.

HLB may propose a task-space target or bounded residual. It does not get to declare a command safe, an action executed, or a repair successful.

## 4. Multi-rate timing hypothesis

Do not prescribe a single universal frequency. Keep the plant's measured clock authoritative and report every cadence separately.

| Layer | Initial experimental role | Timing rule |
|---|---|---|
| Local servo / reflex | balance, impedance, force limits, watchdog, emergency fallback | platform/controller-owned hard-real-time cadence |
| Whole-body controller | contact, IK, MPC, inverse dynamics, trajectory tracking | platform-qualified update/deadline |
| Skill executive | progress, subgoal transitions, contact skill state, recovery selection | event-triggered or bounded periodic |
| HLB temporal state | observation fusion, prediction, belief update, memory retrieval | timestamped; may skip/defer on overload |
| Deliberative agent | language/task decomposition, new tool strategy, long-horizon replanning | asynchronous; never blocks local safety/control |

Use 40 Hz physics and 10 Hz FEP only as the current DMC21 trainer default baseline. The experiment must also sweep slower/faster cognitive cadence and event-triggered updates while holding simulated plant time fixed. Report wall-clock inference latency separately from simulated plant seconds.

A strong result would be **graceful degradation**: when HLB is late, stale, reset or unavailable, the robot retains an already-qualified local controller/fallback rather than applying stale intent or becoming unstable.

## 5. Falsifiable matched ablation matrix

The first stage uses the existing simulator and a frozen morphology/configuration. Every arm receives the same observation stream, task definitions, seed schedule, action envelope, simulator version, train/evaluation budget and verifier.

| Arm | Representation / temporal model | Higher-level cognition | Purpose |
|---|---|---|---|
| A | No learned residual | none; deterministic baseline + hierarchy | minimum capable controller |
| B | Capacity-matched MLP | none | non-recurrent learned residual control |
| C | Capacity-matched GRU | none | conventional recurrent baseline |
| D | HDC encoder + non-liquid readout | none | isolates hyperdimensional representation |
| E | Current HDC-LTC residual path | current FEP schedule | current Symthaea baseline |
| F | HDC-LTC with checkpointable recurrent learning | current FEP schedule | tests learned liquid dynamics; only after #1269 |
| G | HLB + episodic memory, no procedural promotion | bounded skill executive | tests useful history vs validated skill |
| H | HLB + validated procedural memory + active sensing | bounded skill executive | tests reusable knowledge and causal inquiry |
| I | matched AMP-style agent planner + same local controller | agentic planner | strong generalist-planner comparator |
| J | agent planner + HLB skill/memory layer + same local controller | hybrid | tests whether HLB adds value beyond the planner |
| K | released ULTRA G1 unified multimodal controller, where its exact checkpoint/runtime and task/sensor compatibility can be qualified | whole-body loco-manipulation baseline | tests Symthaea against a real-G1 sparse-goal/egocentric controller; not a tool-use score |

Ablations must separate these factors rather than attribute every gain to the full stack:

- HDC representation on/off.
- Liquid temporal dynamics vs GRU and feed-forward controls.
- FEP/active-inference decision layer on/off.
- Episodic memory on/off.
- Validated procedural memory on/off.
- Active sensing on/off.
- Local-controller and safety stack held constant.

Capacity comparisons must disclose parameter count, update count, compute, memory, architecture and training data. Use both a parameter-matched and a compute-budget-matched comparison when feasible. Report results per task and per difficulty; do not hide a weak arm inside a pooled average.

## 6. Evaluation ladder

### Stage 0 — software semantics and replay

- Validate observation/action identity and missingness.
- Freeze seeds, splits, simulator, morphology and calibration.
- Test deterministic replay and exact recurrent snapshot/restore.
- Test reset, irregular timestamp, stale observation and delayed HLB outputs.
- Verify that HLB off, timeout or malformed state leaves the local controller in its declared behavior.

### Stage 1 — balance, disturbance and recovery

Use existing stand/walk/run, contact, terrain, perturbation and fall-recovery scenarios. Stratify push direction/magnitude, timing, support phase, friction, payload, sensor delay/dropout, calibration drift and actuator degradation. Include both known disturbances and held-out combinations.

### Stage 2 — tool use

Use all 18 HumanoidToolBench G1 conditions only after the G1 observation/action adapter passes UEC v2 runtime conformance. The conditions distinguish tool selection, stationary use, mobile use and decoys. Until that gate is met, the correct Symthaea result is **Unavailable / Incompatible**, not zero and not a direct comparison.

### Stage 3 — active discovery and service

Use the 24-case active tool-discovery and 30-case failure-to-skill synthetic corpora already in this programme. Add independent task-effect verification and low-risk physical fixtures only after the corresponding fixture, instrumentation and safety gates are approved.

### Stage 4 — cross-embodiment and real robot

Only after humanoid adapter qualification and a second qualified embodiment. Do not turn shared HDC role labels into a claim of transferable motor control. Hardware trials require a named robot, exact profile, calibration, local controller, watchdog and independent physical experiment record.

## 7. Metrics that matter


### Capability-transition instrumentation

The event statistics are implemented as a dependency-free Rust evaluator in [capability-transition-ledger](capability-transition-ledger/) and tracked by #12741. This gives the HLB ablation a common per-episode event receipt rather than requiring each model arm to invent its own progress metric. Instrumented task-success evidence must name a verifier source distinct from the policy and event recorder; this identity check does not itself prove organizational independence. It does not measure task success unless the independent event recorder supplies that evidence.

Do not evaluate HLB's contribution only through terminal task success. HumanoidToolBench reports separate interaction milestones: candidate contact, correct first contact, correct-tool lift, tool-target contact and task success. Its published FastWAM standard BallMove result falls from 76% stationary L1 to 9% mobile L2; the benchmark also notes that L2 changes target layout, so locomotion is not isolated as the sole cause. Figure Helix 2.5's vendor-reported 56% zero-shot result across 30 unseen homes and Google's vendor-reported 96% Apollo 2 automotive-kitting pilot establish a higher bar for broad transfer and industrial repeatability, but the task suites differ and must not be pooled.

For every HLB arm, collect an event/transition ledger with event occurrence timestamp, source clock, capture completeness, attempt/recovery lineage, failure cause and independently verified terminal postcondition. Preserve non-occurrence versus not-instrumented as separate states. Compute per task cell:
- event incidence and correct-first-contact rate;
- transition conversion conditioned on both events being instrumented;
- event-to-event latency and retries;
- stage-specific recovery/intervention burden;
- decoy and held-out tool-property generalization;
- paired L1/L2 change, with layout, route distance and target placement confounds explicit.

These milestones are not assumed to be a strict funnel: regrasping, repeated contacts and out-of-order recovery are valid behaviors. HLB's hypothesis is that it predicts the next failing transition and chooses a bounded intervention targeted to that stage rather than blindly retrying the whole plan.

Prospective target: close at least 30% of the largest preregistered stage-transition failure gap relative to the strongest matched baseline, while reducing the paired L1-to-L2 gap by at least 30% in a layout-controlled extension; keep false success acceptance at zero in frozen negative controls. These targets are not results. Protocol details and external sources are in section 3E of the parent benchmark protocol.

### Capability and generalization
- Complete task success and independently verified postcondition success.
- Tool-selection, grasp, engagement, manipulation and mobile execution success separately.
- Held-out tool instance, tool family, task composition, environment and embodiment transfer.
- Performance by horizon: task steps and elapsed plant seconds.
- Skill reuse rate, reuse success and time/episodes to adapt to a new task.

### Reliability and recovery
- Falls, collisions, unsafe contacts, joint-limit events and force-limit events.
- Failure localization and time to detect divergence.
- Recovery success conditioned on a real/injected failure.
- Recovery time, extra motion, intervention count and reset/restart cost.
- Repeated-failure rate after memory update.
- Protected-skill retention and harmful-transfer incidence.

### Epistemic quality
- Contact/slip/load/task-effect prediction error and calibration.
- Confidence calibration (Brier/log score or appropriate continuous calibration metric).
- Correct abstention/refusal/handoff on impossible, unsafe, stale or unverified cases.
- False completion acceptance and verifier disagreement.
- Active-probe information gain, decision value, risk and wall-clock cost.

### Runtime and operations
- HLB update p50/p95/p99 latency, deadline misses and stale-output rejection.
- Whole-body solver p50/p95/p99 latency and fallback rate.
- Model/API latency separately from plant time.
- CPU/GPU memory, allocations, compute, bandwidth, energy and cost per verified task.
- Human intervention, rework and accepted productive work per robot-hour.

Safety metrics are **non-compensable**: a higher task-success score cannot offset unsafe action admission, false restoration acceptance or loss of a qualified fallback.

## 8. External comparison and target ladder

The current external comparison is not one leaderboard. AMP1 is a strong agentic-manipulation reference: the live NVIDIA RoboLab-120 leaderboard lists 86.0% ±2.5 percentage points from 2,579/3,000 episodes. Its project page reports 104/108 real-world successes across 12 tasks on a 7-DoF Franka Research 3, not a humanoid. Those numbers are strong but do not prove humanoid tool-use, whole-body recovery or service competence. The November 2026 long-horizon manipulation survey specifically argues that binary success hides partial progress, failure propagation, recovery cost and horizon degradation.

ULTRA is a **more direct whole-body humanoid comparator**: its IROS 2026 project reports a unified multimodal controller evaluated in simulation and on a physical Unitree G1, supporting dense reference tracking and sparse goal-conditioned loco-manipulation from egocentric perception. The project demonstrates long-horizon goal following, depth-based sensing and sim-to-sim transfer. It does not report a directly comparable HumanoidToolBench or service-task aggregate on the project page, so do not invent one. ULTRA's public code is Apache-2.0, the InterMimic-derived stack is MIT, and data/assets/third-party components retain their own terms; the released training recipe uses Python, IsaacGym Preview 4 and MuJoCo. Include it as Arm K only after exact checkpoint, sensor/task semantics, robot morphology, licenses and runtime are pinned and qualified.

This creates a cleaner comparison taxonomy: ULTRA for a learned unified G1 whole-body controller; AMP1 for a deliberative agentic manipulation policy (real-arm results, not humanoid); HumanoidToolBench for tool-selection and tool-use tasks on G1; and the Symthaea ablations for whether HLB provides an incremental advantage over capacity-matched MLP/GRU, current HDC-LTC, and controller/agent baselines.

Relevant sources:
- ULTRA project and demos: https://ultrahumanoid.github.io/
- ULTRA paper: https://arxiv.org/abs/2603.03279
- ULTRA code and licensing notes: https://github.com/Sirui-Xu/ULTRA
- Whole-body control review (13 November 2026): https://www.mdpi.com/2227-7390/14/22/4173
- Long-horizon manipulation survey (16 November 2026): https://link.springer.com/article/10.1007/s44336-026-00048-x

Sources:
- https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html
- https://apple-aiml-research.github.io/ml-amp/
- https://arxiv.org/abs/2611.09892
- https://link.springer.com/article/10.1007/s44336-026-00048-x
- https://arxiv.org/abs/2610.02089 (HumanoidToolBench)

### Prospective targets — not results

1. **RoboLab-120:** target at least 90% success under the official 3,000-episode protocol, with model/tool/runtime/cost fully disclosed. This is a stretch target above the current 86% leaderboard point estimate, not a prediction that Symthaea will achieve it. Do not claim parity until the exact protocol and submission requirements are met.
2. **HumanoidToolBench:** achieve at least +10 percentage points paired improvement over the strongest locally reproducible baseline on a preregistered held-out mobile/decoy composite, after G1 adapter conformance. Publish all 18 cells separately.
3. **Held-out transfer:** at least 80% success on same-family and task-composition transfer in the programme's own fixture suite, with split leakage controls and uncertainty.
4. **Recovery:** at least 20% relative reduction in recovery/intervention burden without reducing verified success; separately test whether memory produces a fourfold reduction in repeated failures on matched sequences, inspired by—but not assumed to reproduce—the AMP memory result.
5. **Skill retention:** no regression on the frozen protected-skill set beyond the preregistered non-inferiority margin after candidate updates; zero accepted poisoned-memory corrections in the negative-control corpus.
6. **Service verification:** at least 90% verified completion on the low-risk service suite and zero false returns-to-service in its frozen negative controls.
7. **Control runtime:** HLB p99 execution must fit the budget allocated by the platform profile while leaving the qualified local controller and watchdog within their own deadlines. Report the actual measured budget; do not invent a universal HLB frequency.
8. **Graceful degradation:** 100% of frozen timeout/stale-state/reset fault injections must retain the declared local fallback or enter the declared restrictive state; a finite test result is not a universal safety guarantee.

A benchmark lead is meaningful only when the confidence interval, test count, model/data rights, inference budget, action frequency, hardware, latency and task semantics are comparable.

## 9. Immediate implementation order

1. **Source and integration audit:** keep the modern trainer path distinct from the legacy bridge; map its exact semantic observation/action and control-stage boundary to UEC v2 rather than adding a second humanoid controller.
2. **Finish #1269 before recurrent-learning promotion:** capture static parameters, mutable neuron state, cached outputs, timing state and exact continuation identity; test replay from snapshot.
3. **Close #1284 time semantics:** HLB tau and cognitive cadence must not change the physical plant dt.
4. **Build a benchmark harness around existing code:** freeze per-arm configs/seeds, run the same episodes, emit per-step receipts and explicit valid/invalid/infra-indeterminate outcomes. Reuse the current telemetry and benchmark audit owners.
5. **Run the cheap ablations first:** baseline vs current HDC-LTC vs capacity-matched MLP/GRU on stand/walk/perturbation. No frontier API required for this phase.
6. **Add HLB memory and active discovery:** only after baseline instrumentation, independent verification and replay are in place.
7. **Integrate an agentic planner last:** compare the same planner with/without HLB memory and local control to isolate architecture contribution.
8. **Move to G1 and physical hardware only after typed adapter, safety, calibration, transport, HIL and rights gates are closed.**

## 10. Claims we will not make

This protocol does not claim that HLB currently beats Transformers, GRUs, VLAs, WAMs or AMP1; that a 16,384D representation is inherently superior; that recurrent learning is checkpoint-safe today; that simulator success is physical competence; that current G1 mapping is runtime-conformant; or that Phi/Ψ establishes motor safety. It defines the experiments required to earn those claims.
