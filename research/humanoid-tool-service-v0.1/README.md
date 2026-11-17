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

## Current SOTA shift: agentic policies and durable skill memory

The benchmark portfolio now explicitly includes **agentic manipulation policies**, not only VLA/WAM action-chunk policies.

AMP1 (arXiv:2611.09892, 6 Nov 2026) reports 85.6% on RoboLab-120 and 104/108 trials on a Franka-based real-world evaluation, with an AMP1+ persistent-memory experiment reporting 4.6× lower failure rate on RoboDojo bimanual tasks versus zero-shot. This is an arm-based result, not a humanoid result. The paper also documents collision/calibration failures and benchmark/verifier artefacts; it states model choice matters much more than modest changes to its harness.

RoboDojo's 17 Nov 2026 public snapshot distinguishes its standard policy leaderboard from the Agent track. Its SOLVE* agent entry is marked as evaluated before the 14 Nov Agent Track rules and is therefore flagged as non-conforming with updated rules. The scores must not be compared directly with the standard policy board or HumanoidToolBench.

VioLA (8 Oct 2026) motivates a second output-space baseline: predict transferable body/hand motion latents and let a qualified whole-body controller track them, rather than predicting raw joint commands. Its reported 140.6M frames include 93.2% human motion data, but human motion must be converted through a valid morphology/frame/action contract before it is usable as robot supervision.

Symthaea's hypothesis is to combine:
- a generalist agent that remains in the task loop and performs active perception / plan revision;
- typed, bounded capabilities instead of unrestricted generated code with direct live hardware access;
- HDC-LTC / FEP multi-timescale cognition and local fast controllers;
- a validated, provenance-bearing procedural skill memory;
- terminal deterministic safety, explicit authority and independent physical postcondition verification.

Related programme: [HUM-AGENTIC-POLICY-001 #12685](https://github.com/Luminous-Dynamics/symthaea/issues/12685).

## G1 interface compatibility artifact

The exact HumanoidToolBench G1 action/state mapping is now pinned separately:
- [G1 decoupled_v1 mapping manifest](g1-decoupled-v1-mapping.json)
- [G1 mapping JSON Schema](g1-decoupled-v1-mapping.schema.json)
- [HUM-G1-UEC-ADAPTER-001 #12681](https://github.com/Luminous-Dynamics/symthaea/issues/12681)

The mapping preserves all 43 raw named G1 joint positions, the derived 32-value policy-state projection, the 36-value action schema, the left-hand thumb/middle/index permutation, the waist order change, the base-height command-history field, RGB image modality, and the navigation turn-flag semantics.

This is a **non-actuating software compatibility specification**. It does not map G1 joint targets onto Symthaea DMC21/Dexterous53 torques, does not connect to the current 21-joint PCA9685 HAL, and creates no UEC Applied/Measured receipt or actuation authority. The typed Rust adapter is deliberately dependent on the canonical UEC v2 APIs and their current qualification state.

## Deliverables

- `protocol-v0.1.md`: preregisterable comparison and claim-boundary protocol.
- `cases-v0.1.json`: exact 18-cell benchmark anchor, eight service-case definitions and negative controls.
- `schema-v0.1.json`: strict structural contract for the case manifest.

These artifacts define a prospective benchmark. They do not indicate that benchmark episodes were run or that any task has been solved.

## Initial success criterion

A first research win is not “Symthaea beats Figure”. It is a credible, reproducible, matched experiment showing that one or more proposed mechanisms (functional tool model, tactile/contact prediction, active probing, compositional recovery) improve **held-out verified task success** over a strong baseline at a declared data/compute/latency budget, without increasing unsafe action, false acceptance or unreported human intervention.

## Claim ceiling

Protocol consistency is not robot capability. A simulation pass is not HIL. HIL is not physical qualification. Physical task qualification is not unrestricted deployment, human-contact safety, clinical competence or legal/product certification. No actuator authority is introduced here.
