# MANIP-HLB-BIO M1: deterministic contract fixtures v0.1

**Status:** research/contract scaffold. A crate-local integration test source validates fixture-set structure and selected fail-closed invariants, but it has not been executed. This does not establish runtime behavior, simulator, HIL, or physical qualification.

**Owner:** [Symthaea #12735](https://github.com/Luminous-Dynamics/symthaea/issues/12735)

**Architecture dependencies:** [#1274 UEC v2](https://github.com/Luminous-Dynamics/symthaea/issues/1274), [#1276 cognitive readiness vs plant safety](https://github.com/Luminous-Dynamics/symthaea/issues/1276), [#1281 metric semantics](https://github.com/Luminous-Dynamics/symthaea/issues/1281), [#1284 cognitive vs plant time](https://github.com/Luminous-Dynamics/symthaea/issues/1284), [#1272 typed prediction residuals](https://github.com/Luminous-Dynamics/symthaea/issues/1272), and [#1269 recurrent snapshot/restore](https://github.com/Luminous-Dynamics/symthaea/issues/1269).

## 1. Why this tranche exists

M1 starts with deterministic semantic and authority-boundary cases, not a new controller. It tests whether the next HLB experiment can distinguish:

- physical contact from missing contact data;
- an observed physical state from a latent HDC representation;
- a fresh observation from a stale or out-of-order sample;
- a local reflex proposal from a global skill/mission proposal;
- a prediction hypothesis from a measured outcome;
- a compatible experience from a superficially similar but incompatible one;
- cognitive readiness from physical plant-safety evidence;
- cognitive time from physical plant time;
- a requested action from a projected, applied, observed, and independently accepted result.

The core biological hypothesis is **distributed sensing with hierarchical coordination**. Local sensing and bounded reflexes may reduce response latency and communication load; a slower supervisory layer may coordinate task-level behavior. We do not assume that an octopus, insect, or vertebrate circuit can be copied directly into a robot.

A separate physical-mechanism owner is [Symtropy #5991 — MORPH-PHYS-COMPUTE-001](https://github.com/Luminous-Dynamics/symtropy/issues/5991). It tests whether compliance, material geometry, distributed sensing and mechanical coupling perform task-useful filtering/routing/switching or physical-reservoir computation. This supplies a body-mechanism ablation to the HLB study; it does not add another manipulator controller, tactile calibration pipeline or world model.

## 2. Source audit: exact facts to preserve

Source was fetched from the public `main` branch on 18 November 2026. These SHAs identify the inspected blobs, not a claim that the runtime is qualified.

| Source | Blob SHA | Audited fact |
|---|---|---|
| `crates/domains/symthaea-manipulator/src/sensorimotor.rs` | `40f8a4fd4fbdfb868b9cd4840d6d7be929f6bf0c` | `ControllerInputSchema::default()` is `LegacyDirectV0`; role-bound thought/body fusion is an explicit schema. |
| `crates/domains/symthaea-manipulator/src/controller.rs` | `eb7a81daa75aa1298de1c7e2139992dc79844980` | HDC-LTC output is projected to seven normalized joint torque outputs and one gripper output. |
| `crates/domains/symthaea-manipulator/src/predictor.rs` | `c9943f13f12e46442543b0d434688f3efa2de29f` | Current state-prediction comparison is a fixed-scale normalized RMS scalar; it is not a calibrated predictive distribution. |
| `crates/domains/symthaea-manipulator/src/embodiment.rs` | `c56a5fc5ad8ff83f02534aaec10db15ef8c84ce3` | The current bridge fuses thought/body HVs, obtains a learned command, applies admittance, and routes it through the stateful safety supervisor before stepping the simulator. The legacy `phi` argument is mapped to a `MotorSafetyLevel` and max-composed with other restrictions. |
| `crates/domains/symthaea-manipulator/src/safety_supervisor.rs` | `da7c25b79d662a13674853d5746132d6e1229702` | The manipulator supervisor declares final authority over nominal commands before the physics/hardware backend. |
| `src/cognitive_loop/accessors/consciousness.rs` | `5558d956e6831c05d939a591e37fc8ca1200afe5` | `safety_gate_phi()` returns a stored compatibility scalar with two unrelated writer formulas and records writer/cycle provenance separately. The accessor documentation warns that interleaving entry points makes the value order-dependent and unattributed. |

The manipulator is therefore a useful research bridge, but it is not yet the proposed HLB-only cognitive proposal architecture. The existing HDC-LTC controller participates in the control path, albeit behind the platform's independent supervisor. New high-level HLB planning must not be added as another direct motor-command path.

The `ControllerWeights` compatibility detail is also intentional: `ManipulatorController::export_weights()` labels the default schema `LegacyDirectV0`, while the trainer's export wrapper explicitly changes it to `SensorimotorBoundV1`; `ManipulatorEmbodiment::install_weights()` accepts only the latter. Future manifests must bind the schema actually used.

## 3. Biological evidence: useful mechanism, bounded claim

Del Dottore et al., *Peripheral control enabled by distributed sensing in an octopus-inspired soft robotic arm for autonomous underwater grasping*, Nature Machine Intelligence 8, 708–721 (published 12 May 2026), DOI [10.1038/s42256-026-01230-y](https://doi.org/10.1038/s42256-026-01230-y), is a concrete engineering reference.

The paper reports a single 410 mm soft arm with ten sensorized suction cups. Each cup integrates optical mechanosensing; the reported mean force-signal slope is about 394.9 ± 21 mV/N, directional error is below 18 degrees, and the tested 0–2 N range gives approximately 0.1 N accuracy. Local contact circuits trigger suction reflexes, while an arm-level peripheral controller coordinates grasp strategy. The reported task demonstrations use a restricted object set in still water; the prototype does not implement a central nervous system or inter-arm coordination.

**Do not compare those sensor numbers directly with a rigid seven-joint manipulator's force sensor or claim equivalent task performance.** Use the paper to motivate a mechanism ablation: local reflex vs centralized-only processing vs hybrid local/global coordination. For a fair comparison, match sensing budget, plant, contact scenario, communication assumptions, and physical task.

## 4. Fixture envelope

Every fixture binds, or explicitly declares unavailable:

- embodiment/profile and instance identity;
- morphology, tool/gripper, actuator map, and calibration identity;
- physical role, unit, frame kind, realized frame/transform identity;
- sample source, sequence, source timestamp, clock domain, and freshness policy;
- model/checkpoint identity and prediction horizon;
- action request, projection, admission, application, observation, and acceptance stages;
- memory provenance and the exact compatibility predicate;
- independent safety policy and required evidence set.

Missing, stale, invalid, uncalibrated, wrong-frame, or wrong-configuration evidence is never silently converted to a measured zero, a healthy value, a current value, or a successful action.

## 5. Known-answer cases

The normative machine-readable cases are in [`crates/domains/symthaea-manipulator/tests/fixtures/hlb_bio_contract_fixtures_v0_1.json`](../../crates/domains/symthaea-manipulator/tests/fixtures/hlb_bio_contract_fixtures_v0_1.json). The structural test source is [`crates/domains/symthaea-manipulator/tests/hlb_bio_contract_fixtures.rs`](../../crates/domains/symthaea-manipulator/tests/hlb_bio_contract_fixtures.rs); it checks schema identity, unique fixture IDs, required fields, declared invariants and selected negative controls. It does not yet test the actual runtime semantics. The first implementation should cover:

1. **Missing vs zero:** absent tactile coverage is not a measured zero-force/no-contact sample.
2. **Stale observation:** expired tactile/proprioceptive data cannot support a current contact or grasp assertion.
3. **Frame mismatch:** equal numeric vectors in different realized frames are not directly comparable.
4. **Duplicate/out-of-order samples:** replay or stale sequence identity cannot manufacture a new observation.
5. **Wrong-configuration memory:** an episode from another gripper/tool/calibration profile is diagnostic-only unless an explicit transfer predicate passes.
6. **Slip + local reflex:** local bounded stabilization proposal is emitted from measured slip/contact evidence; it does not grant mission-level authority.
7. **Prediction mismatch:** repeated typed residuals may invalidate a model/trigger re-observation; a heuristic scalar cannot become calibrated uncertainty or physical safety.
8. **HLB timeout:** stale/late cognition is discarded while the already-qualified local controller/fallback continues.
9. **Independent veto:** a physical hard constraint vetoes a cognitive proposal regardless of cognitive readiness, HDC similarity, or predicted reward.
10. **Partial sensor/network loss:** missing local nodes are explicit; the global layer must not infer full coverage from the remaining sensor subset.
11. **Plant/cognition time separation:** changing cognitive cadence cannot change physical elapsed time or plant integration.
12. **Replay determinism:** same frozen inputs, identities and event ordering reproduce the same fixture disposition and canonical digest.

## 6. Proposed M1 decision vocabulary

These are fixture-level dispositions, not production API names:

- `ACCEPT_FOR_EVALUATION`: complete, fresh, compatible evidence may be evaluated further.
- `RESTRICTED`: a declared restriction applies; only the independently qualified restricted path remains.
- `ABSTAIN`: required evidence is unavailable or ambiguous.
- `QUARANTINE`: a sample, memory, or model artifact is retained for diagnosis but cannot support action.
- `REJECT_STALE`: expiry/freshness condition failed.
- `REJECT_IDENTITY`: embodiment, tool, calibration, model, frame, or contract identity mismatched.
- `VETO_PHYSICAL`: independent plant-safety policy rejected the proposed physical action.
- `FALLBACK_LOCAL`: the platform's separately qualified local fallback owns the immediate response.

A fixture disposition is not an actuator command. The test harness must separately assert that no proposal or evidence disposition can bypass action projection, plant admission, and platform safety supervision.

## 7. Biology-to-engineering ablation plan

After M1 fixtures are implemented, run a minimal, matched manipulation study with the same robot, sensors, task family, initial states, compute and physical limits:

| Arm | Mechanism |
|---|---|
| A | Centralized tactile processing; no local reflex |
| B | Local contact reflex only; no HLB integration |
| C | Local reflex + centralized task coordination |
| D | C + typed HDC semantic representation |
| E | D + CfC/LTC action-conditioned prediction |
| F | E + episodic memory with configuration-bound retrieval |
| G | F + active sensing / information-seeking proposals |

Keep a strong deterministic/impedance baseline and a matched recurrent baseline. Add a tactile modality/availability ablation and a communication-delay/loss stratum. Compare the cognitive components only after the local controller and physical plant are fixed and independently measured.

Measure separately:

- contact detection and direction/force error against independent reference instrumentation;
- sensor-to-local-reflex latency and full sensor-to-controller latency;
- communication bytes, dropouts, and local operation under network partition;
- grasp retention, slip, damage, task success and partial progress;
- recovery latency, retries, human intervention and preserved recovery options;
- prediction error and interval coverage by physical role/regime;
- memory retrieval precision, false analogue rate and invalidation after configuration changes;
- CPU/GPU, memory, sensor, actuator and network energy with explicit system boundaries;
- p50/p95/p99/worst latency and hard deadline misses.

Do not collapse these into one weighted score. Report paired mission/episode results and worst-condition strata.

## 8. Promotion gates

- M1 fixture implementation and known-answer tests pass independently of HLB.
- Existing controller/supervisor behavior is unchanged by this research scaffold.
- All HLB proposals remain shadow-only during the first comparative experiment.
- #1276 cognitive-readiness/plant-safety separation and #1284 plant-time semantics are respected; no new path treats the legacy `phi` argument as a physical measurement.
- #1272 typed residual semantics and #1281 metric-kind distinctions are used where their prerequisites exist; otherwise record explicit unavailability.
- Memory/model promotion remains quarantined until independent held-out evaluation and rollback evidence are present.
- No HLB advantage is claimed unless matched ablations show positive held-out task/decision benefit without increasing hard physical-envelope violations.

## 9. External benchmark comparators and SOTA strategy

These suites answer different questions. Their scores must remain separate.

| Comparator | Published 2026 reference point | What it tests | How to use it here |
|---|---|---|---|
| RLBench 18-task subset | Stanford AI Index 2026 reports EquAct at 89.4% average success as of January 2026. | Standardized, comparatively short-horizon simulated manipulation across 18 tasks. | Use as a short-horizon manipulation baseline only when task, demos, observation, compute and evaluator match. Add 3D rotation and object-condition holdouts. |
| BEHAVIOR-1K Challenge 2026 | 100 full-length household tasks, 7 scenes, RGB + depth + proprioception, 20,000 demonstrations / 1,950 hours; official scoring uses average BDDL predicate partial credit. The first-place HiPhys-RSI system reports Q=0.683 and 48.2% full-task success on hidden instances. | Long-horizon household workflows, mobile manipulation, bimanual dexterity, subgoal dependencies and task completion. | Treat as the most relevant external long-horizon comparator. Include a procedure/skill-graph-only baseline, a VLA baseline, HLB without procedures, and HLB + procedures. Only compare under the official task split, observations, data, compute and evaluator. |
| LIBERO-RECOVER | September 2026 preprint: 1,000+ recovery scenarios across action retry, action adaptation, object-state recovery and environmental recovery. | Recovery after execution failure rather than success from ideal initial states. | Use its four recovery levels as a separate challenge family; preserve natural and injected failures in separate strata. |
| Morphology-conditioned QWM | CoRL 2026 morphology-conditioned quadruped world model with zero-shot transfer bounded to the studied quadruped family. | Dynamics transfer conditioned on explicit morphology. | Compare quadruped prediction and locomotion on held-out morphology combinations and fault × morphology strata; do not describe this as universal transfer. |
| Octopus-inspired distributed soft arm | Nature Machine Intelligence, 12 May 2026: ten sensorized suction cups, local reflexes and hierarchical global grasp coordination; reported force/direction results are specific to its sensor and still-water experiments. | Distributed tactile observation and peripheral control. | Use as a mechanism comparator for local-vs-central-vs-hybrid sensing/control, not as a direct rigid-manipulator leaderboard score. |

### Evaluation coverage gap: prediction must lead to executed, measured decisions

The final snapshot of *A Survey of World Model Benchmarks* (Preprints.org v2, 14 September 2026; corpus checked 18 October and coding finalized 8 November) catalogs 130 representative benchmarks and separates evaluation target, protocol, metric family and data source. Its taxonomy reports 98 open-loop evaluation tracks and 13 closed-loop interaction tracks; 99 tracks use prediction-level metrics and 14 include downstream-outcome metrics. For control fidelity/interactive dynamics, only 6 of 74 tracks include closed-loop interaction and 4 include outcome-level metrics. For interactive action fidelity, the corresponding counts are 3 of 22 and 2 of 22. Functional utility is more often closed-loop: 13 of 15 tracks include closed-loop interaction and outcome metrics. Source: https://world-model-benchmarks.github.io/World-Model-Benchmarks/.

A separate survey, *Do World Models Make Better Robots?* (arXiv:2609.29669, 30 August 2026), uses a different 160-benchmark corpus and reports 11 explicit VLA-vs-world-model contrasts and four prediction-to-action benchmark suites. **These are separate surveys and denominators; do not merge their counts or present them as one statistic.**

This creates a concrete Luminous benchmark opportunity: predict candidate action consequences, select a candidate, execute it through the independently controlled plant, measure the outcome, then compare the predicted ranking with realized outcomes. Prediction accuracy, counterfactual fidelity, policy-ranking agreement, net decision lift, model exploitability and physical-envelope violations remain separate result planes.

Compose with Symthaea #9010 (general matched world-model decision value), #12548 (tactile action-conditioned futures), and #11880 (anticipatory reflex/time-to-boundary). #12735 supplies the contact-rich manipulator profile; do not fork their world model or reflex owners.

### Industry frontier: vendor-reported November 2026 systems

These are important external systems to admit as comparators where their checkpoints, task adapters, license and sensor/action interfaces permit. Their press-release figures are vendor-reported and use distinct evaluation protocols; they are not one shared leaderboard.

| System | Vendor-reported 2026 result | What it changes in our comparison |
|---|---|---|
| Google DeepMind Gemini Robotics 2.5 (10 Nov 2026) | Evaluated across 200 distinct tasks; Google reports 30–100% higher success than prior baselines depending on task and 20–30% faster completion. A real-world proof-of-concept with Apptronik Apollo 2 on Mercedes-Benz production lines reports 96% pick-and-pack success. | A strong generalist/VLA comparator for task breadth, dexterity, and production-cell workflows. Require exact task, episode count, reset, operator intervention, confidence interval and hardware details before treating 96% as a reproducible benchmark result. |
| NVIDIA GR00T 2 (CoRL 2026, Nov 2026) | NVIDIA reports at least one completion on 40/50 evaluated tasks without task-specific demonstrations; its one-demo and five-demo settings report 71% and 95% average success across the 50-task suite, respectively. | A direct baseline for few-shot adaptation and cross-embodiment transfer. Compare equal demonstrations, adaptation wall-clock, reset budget, hardware and held-out embodiment/task combinations. |
| Figure Helix 2.5 (17 Sep 2026) | Figure reports zero-shot whole-body autonomy for three long-horizon behaviors across 30 previously unseen homes, with no data collected, fine-tuning or adaptation in those environments or manipulated objects; its report gives 56% success after Index human-data pretraining vs 9% without it. | A strong unseen-environment and human-data-transfer comparator. Compare home/task splits, human-data exposure, success denominator, safety interventions and generalization beyond the three reported behaviors. |
| Figure Helix 02 (27 Jan 2026) | Figure reports a continuous four-minute dishwasher unload/reload task integrating whole-body locomotion, manipulation and balance without resets or human intervention. | A whole-body long-horizon systems comparator; do not reduce the claim to isolated grasp accuracy. |

Sources:
- Gemini Robotics 2.5: https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/
- GR00T 2 / CoRL: https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/
- Helix 2.5: https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization
- Helix 02: https://www.figure.ai/news/helix-02

**Strategic implication:** do not try to out-VLA the foundation-model vendors on generic task breadth alone. Our defensible research wedge is measurable incremental value from (a) persistent configuration-bound experience, (b) action-conditioned and counterfactual prediction that improves executed decisions, (c) distributed tactile/local responses, (d) mechanically embodied computation and (e) independent outcome/authority evidence. Each component must win its own ablation before the combined stack receives credit.

### Concrete competitive targets

1. **Reproduce before claiming:** implement the official task/evaluator adapters and first reproduce the published reference point with the admitted baseline or report exactly why the reference is not reproducible in our environment.
2. **Long-horizon stretch:** on a comparable future BEHAVIOR evaluation, target Q > 0.683 and full-task success > 48.2%, while separately reporting partial predicate completion, worst task families, recovery cost, elapsed simulation time and intervention burden. This is an aspiration, not a forecast.
3. **Short-horizon stretch:** target at least 90% mean success on the exact RLBench 18-task subset with the same observation and demonstration budget, then test 3D orientation and object-condition generalization. A 90% result on a different task or easier split does not beat EquAct.
4. **Recovery superiority:** beat the strongest matched retry/adaptation/state-recovery/environment-recovery baseline on paired recovery episodes, not merely on nominal task success.
5. **HLB attribution:** compare procedure graph alone vs HLB alone vs procedure graph + HLB. The HiPhys-RSI result shows structured executable procedures are already a strong baseline; memory retrieval or a larger latent state earns credit only if its incremental effect is measured.
6. **World-model utility:** report role-conditioned prediction error and interval coverage, but also policy-ranking correlation, downstream decision improvement, exploitability under closed-loop rollouts, and sim-to-real discrepancy. Plausible imagined rollouts are not sufficient.
7. **Distributed-sensing value:** compare local reflex, centralized processing and hybrid local/global coordination at matched sensor hardware, contact task, compute, bandwidth and physical envelope. Report latency, bytes, power, calibration drift and performance under node/network loss.

The current benchmark literature explicitly warns that binary success obscures partial progress, error propagation, recovery cost and horizon-dependent degradation. The 16 November 2026 long-horizon manipulation survey is a useful current framing; the world-model survey likewise emphasizes action alignment, rollout exploitation and closed-loop utility. These motivate our scorecard but do not replace official benchmark protocols.

References:
- Stanford AI Index 2026, robotics / RLBench: https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_2_technical.pdf
- 2026 BEHAVIOR Challenge and rules: https://behavior.stanford.edu/challenge/index.html and https://behavior.stanford.edu/challenge/evaluation.html
- 2026 BEHAVIOR winner report: https://behavior1k-champion.com/
- LIBERO-RECOVER: https://arxiv.org/abs/2609.05178
- QWM: https://arxiv.org/abs/2604.08780
- Octopus-inspired distributed sensing: https://doi.org/10.1038/s42256-026-01230-y
- Long-horizon manipulation survey (16 Nov 2026): https://doi.org/10.1007/s44336-026-00048-x
- Manipulation world-model survey: https://doi.org/10.1002/smb2.70053

## 10. Claim ceiling

This contract scaffold is not an executed Rust test pass, an implemented sensor/reflex network, a calibrated world model, a biological reproduction, a physical qualification, or evidence of SOTA superiority. The added Rust test source is a structural validator only and remains unrun. Even an eventual fixture-suite pass would prove only the fixture implementation for its declared cases. In-simulation results remain in-simulation results; HIL and physical evidence require separate instrumentation and qualification.
