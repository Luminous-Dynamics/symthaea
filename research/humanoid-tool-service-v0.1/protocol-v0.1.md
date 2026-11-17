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

The published matrix contains:
- Scenarios: BallMove, BallRetrieve, IceBreak.
- Levels: L0 tool selection/pickup; L1 stationary tool use; L2 mobile tool use.
- Modes: standard and decoy tool sets.
- Total: 3 × 3 × 2 = 18 task cells.
- ToolBook: approximately 3.1k demonstration trajectories across simulation and real Unitree G1 collection.

Published simulation comparison (scenario order BallMove / BallRetrieve / IceBreak; percentages are paper-reported):

| Policy | Level | Standard | Decoy |
|---|---|---|---|
| GR00T N1.7 | L0 | 77 / 73 / 88 | 67 / 59 / 75 |
| GR00T N1.7 | L1 | 64 / 36 / 66 | 58 / 30 / 63 |
| GR00T N1.7 | L2 | 10 / 9 / 31 | 30 / 5 / 48 |
| FastWAM | L0 | 64 / 62 / 98 | 74 / 45 / 93 |
| FastWAM | L1 | 76 / 57 / 86 | 75 / 45 / 82 |
| FastWAM | L2 | 9 / 20 / 69 | 8 / 11 / 61 |

These values are not a single overall score and must not be averaged across unlike tasks to manufacture one. They illustrate the level-dependent gap and are not locally reproduced results.

The paper reports real-robot results for a narrow selected subset: three policies, L1 scenarios, and ten trials per condition. It also reports only one training run per policy, so policy-training variability is not estimated. A Symthaea reproduction must disclose this and, where computationally feasible, use at least five independent training seeds.

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

## 4. Matched baseline ladder

Evaluate the following arms only when the exact artifact and interface are available:
1. Scripted or geometry-based baseline.
2. Simple reactive / no-memory baseline.
3. Original benchmark-native reference policies.
4. ACT / Diffusion Policy or other task-native imitation baseline.
5. GR00T N1.7 and another compatible open VLA baseline (for example pi0.5).
6. Symthaea perception + proprioception policy.
7. Symthaea + functional tool model.
8. Symthaea + tactile evidence, with calibrated raw-sensor comparator.
9. Symthaea + action-conditioned tactile/contact prediction.
10. Symthaea + uncertainty-aware probing and event-based recovery.

Unavailable artifacts are “unavailable”, not a zero score. Pin license, source SHA, checkpoint digest, pre/post-processing, robot morphology, actuation semantics, sensor set, training data, adaptation budget, inference hardware and evaluation runtime.

Matched comparisons require equal task/reset distribution, demonstrations, fine-tuning opportunities, action chunk/horizon, wall-clock budget, compute where measurable, episode timeout and observation access. If not equal, report the mismatch and do not claim a controlled comparison.

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

## 8. Statistics and reporting

- Freeze code, model, config, task set, split, seed, environment, robot, tool and evaluator identities before evaluation.
- Target at least five independent training seeds where affordable; use paired evaluation seeds and report seed-level variation.
- Start with 100 simulation episodes per cell/policy when affordable to align with the published benchmark. Increase sample counts from a preregistered precision/power analysis, not after inspecting favourable results.
- Physical trial counts need a separate risk and statistical plan. Thirty successful trials do not certify general safety.
- Report task-cell rates, numerator/denominator, uncertainty intervals, effect sizes and failures; retain cluster structure for homes/tools/physical articles.
- Report expected, completed, invalid, missing and infrastructure-indeterminate episodes. Do not silently remove timeouts, safe refusals or failures.
- Publish training-data exposure, fine-tuning, annotation/human-operator, compute, wall-clock, energy and hardware budgets.
- Use paired differences where valid; report median, p90/p95, lower tail and confidence intervals. Do not assume all task cells or physical articles are IID.
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
- p50/p95 completion time, latency, energy and compute;
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
- ≥90% verified success on the eight-case low-risk synthetic service corpus.
- Zero unsafe continuation and zero false return-to-service in the frozen negative-control corpus.
- Demonstrate the value of each architecture component via removal ablations, while reporting extra data, compute, latency and energy.

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
