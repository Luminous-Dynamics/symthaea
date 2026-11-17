# Subterra Robotic Loops Architecture v0.1

**Status:** research architecture proposal  
**Date:** 2026-11-17  
**Parent work:** [#12612 — SUBTERRA-ROBOTIC-LOOPS-001](https://github.com/Luminous-Dynamics/symthaea/issues/12612)  
**Composes:** [#12606 — verified construction](https://github.com/Luminous-Dynamics/symthaea/issues/12606), [#12609 — asset observability](https://github.com/Luminous-Dynamics/symthaea/issues/12609), [#12569 — robust survival policy referee](https://github.com/Luminous-Dynamics/symthaea/issues/12569)

## 1. Thesis

Subterra should be designed as a **federation of bounded feedback loops**, not one giant autonomous controller and not a collection of robots that merely exchange waypoints.

Each loop owns a distinct uncertainty or control responsibility. Every hand-off carries typed state, freshness, authority, provenance and an explicit failure disposition. A loop may propose work; it cannot silently inherit another loop's authority.

The system-level invariant is:

> A physical mutation is not complete because it was planned, commanded, acknowledged or reported by the machine that performed it. It is complete only when the intended physical postcondition has been independently observed and accepted against the still-valid work package. An ambiguous outcome blocks further mutation until reconciled.

This generalizes across excavation, ground support, pipe/cable inspection, drawpoint assessment, maintenance and repair.

## 2. What already exists — reuse before adding controllers

The public `symthaea-subterranean` crate already contains substantial lower-level machinery:

| Existing capability | Current module(s) | Role in this architecture |
|---|---|---|
| Hazard and reflex response | `safety`, `reflex`, `invariant_monitor`, `actuator_isolation` | Local hard safety boundary |
| Mission/work scheduling | `mission`, `mission_executive`, `work_orders`, `logistics` | Work selection and resource admission |
| Plan invalidation | `plan_freshness`, `temporal_runtime` | Reject stale plans after safety/world changes |
| Causal event history | `temporal_event`, `temporal_assurance`, `audit_chain` | Ordering, event lineage and integrity |
| Checkpoint/recovery | `recovery_journal`, `recovery_planner`, `partition_recovery` | Restore and reconcile after disruption |
| Maintenance/health | `maintenance`, `maintenance_window`, `degradation_forecast` | Capability derating and lifecycle planning |
| Multi-agent coordination | `team`, `team_operations`, `relay_mesh`, `occupancy` | Coordination and right-of-way |

This proposal does **not** call for a second mission executive, another hazard supervisor, or a parallel event ledger. The new work is composition: define how the existing pieces and new construction/inspection adapters exchange state and evidence.

## 3. The six loops

Loop rates are deliberately not hard-coded here. Each implementation must declare a deadline, measure its worst-case execution time on the target, and demonstrate that a slower loop cannot suppress a faster safety response.

| Loop | Trigger / cadence | Primary owner | Output | Authority ceiling |
|---|---|---|---|---|
| **L0 — Reflex and safety** | Fastest local control cycle; hazard/event triggered | Symtropy plant boundary + deterministic Symthaea safety supervisor | Continue, constrain, hold, isolate, retreat or emergency response | Can preempt work. Cannot expand work scope or authorize a new mutation |
| **L1 — State and observability** | Sensor frames, motion, map changes, uncertainty or inspection demand | Symtropy sensing/geometry + Symthaea belief and active-sensing policy | Observed/inferred/occluded/unknown state, quality and next useful observation | Can request sensing or abstain. Cannot label unobserved space as clear |
| **L2 — Mission and resource** | Work completion, resource update, topology change or periodic replanning | Symthaea mission executive + logistics | Admissible work selection and route/resource envelope | Selects from approved work. Cannot change engineering design or authority |
| **L3 — Bounded intervention** | One authorized work-package operation | Machine-specific controller behind a versioned capability adapter | One bounded command attempt and measured execution record | May mutate only inside the approved package and current safety envelope |
| **L4 — Independent verification** | After each consequential mutation and at defined inspection gates | Separate verifier and sensor/evidence path | Accepted, rejected or insufficient-evidence postcondition | Can accept/reject evidence. Cannot command the robot or rewrite the design |
| **L5 — Fleet and recovery** | Assignment, handoff, link degradation, robot fault, reboot or partition | Symthaea team/relay + Mycelix provenance and coordination | Capability-matched assignment, lease, recovery and evidence handoff | Coordinates approved work. Cannot override local safety or invent authority |

A further **asset-lifecycle cycle** composes L1 and L4 over hours, shifts and maintenance intervals: inspect → classify condition → obtain engineering/owner disposition → authorize maintenance → intervene → independently verify → monitor recurrence. It should initially be a workflow over the same loops, not a new privileged controller.

### 3.1 L0 — local safety must remain local

L0 must not depend on cloud availability, fleet consensus, a learned model response, a trust score or successful evidence upload. A communications partition may degrade mission capability, but it must not disable local limits.

The local supervisor owns:
- hard actuator and operating-envelope limits;
- stop/hold/isolate/retreat transitions;
- freshness and plausibility checks for safety-critical inputs;
- a latched fault state where a fault cannot be cleared by merely receiving one nominal sample;
- bounded recovery actions whose feasibility is independently checked.

A learned controller may propose a nominal command. It is never the final authority over the physical envelope.

### 3.2 L1 — map observability, not just geometry

Represent these as different states:

- **Observed:** supported by a qualified measurement with source, calibration, frame and coverage.
- **Inferred:** model-derived estimate with explicit hypothesis and uncertainty.
- **Occluded:** known region not adequately observed from the current sensor configuration.
- **Unknown:** no qualified basis for a claim.

The active-sensing policy should maximize *decision-relevant information*, not raw point count. Candidate observations are ranked by expected reduction in consequential decision error, divided by inspection time, energy, exposure and resource cost. The estimate must include common-cause dependencies: two sensors sharing a clock, mount, power rail, calibration target or perception stack are not automatically independent.

L1 is where the pipe/cable corridor and drawpoint volume tasks from #12609 belong. Their defect labels, geometry metrics and acceptance limits remain task-specific.

### 3.3 L2 — schedule only evidence-admissible work

L2 should reuse the existing work-order and mission executive. A work item is eligible only when:
- its authority and design revision are current;
- prerequisites are complete;
- the robot capability profile satisfies the required task;
- outbound and return routes are feasible under current uncertainty;
- energy, consumables, waste/spoil capacity, thermal margin and tool condition are within budget;
- the safety supervisor has not preempted the mission.

Work priority must not bypass physical admission. A high-priority task that lacks an admissible return route or required verification resource remains blocked.

### 3.4 L3 — one operation, one bounded state transition

The intervention loop handles one mutation attempt at a time. It is not a free-running planner-to-actuator pipe.

Before command issuance, freeze and bind:
- work-package ID and revision;
- design and coordinate-frame digests;
- operation type, geometry and allowed state delta;
- permitted machine capability and tool profile;
- calibration and sensor revisions;
- allowed energy, material, time and wear envelope;
- authority evidence and expiry;
- precondition observation set.

After execution, preserve separate events for command issuance, device acknowledgement, actuator telemetry and measured physical change. These are not interchangeable.

### 3.5 L4 — independent postcondition verification

The verifier receives the approved specification and qualified post-action observations. It must not receive planner-private state, candidate confidence caches or hidden synthetic ground truth during inference.

Its result is three-valued:

- **Accepted:** evidence covers the required region and satisfies all declared postconditions.
- **Rejected:** evidence demonstrates a failed or out-of-tolerance postcondition.
- **Insufficient:** evidence is missing, stale, occluded, uncalibrated, frame-inconsistent or otherwise unable to decide.

Insufficient is not accepted. Rejected does not imply that an automatic retry is safe; correction requires a still-valid or newly approved work package. A separate sensor is not necessarily an independent sensor: independence must be assessed across modality, calibration, mounting, power, timebase, firmware and evidence pipeline.

### 3.6 L5 — fleet capability and recoverability

Assign a task to the platform with the right *capability evidence*, not merely the nearest available robot. Capability profiles should describe:
- traversability envelope and localization limits;
- payload, reach, contact force and tool interface;
- sensing modalities, calibration state and inspection resolution;
- energy and consumables;
- communication and autonomy-under-partition capability;
- health, maintenance and recovery constraints.

A robot's failure must not erase completed evidence, create duplicate execution, or leave a second robot believing a leased work package is unowned. A handoff carries the last accepted event sequence, operation state, unresolved-outcome lock, evidence digests and lease generation.

## 4. Cross-loop operation state machine

The first reference implementation should make these states explicit:

```text
Approved
   |
   v
PreflightAccepted
   |
   v
CommandIssued -----> OutcomeUnknown
   |                       |
   v                       +----> ReconciledNotExecuted
Acknowledged               |          |
   |                       |          +--> new attempt with new attempt ID
   v                       |
ActuationObserved           +----> ReconciledExecutedAwaitingInspection
   |                                  |
   v                                  v
PhysicalDeltaObserved <---------------+
   |
   v
IndependentVerification
   |             |                 |
   v             v                 v
Accepted       Rejected        Insufficient
   |             |                 |
   v             v                 v
Closed        ReworkReview   AwaitAdditionalEvidence
                                  |
                                  +--> new qualified observations
                                       then independent verification
                 |
                 +--> requires valid authority for a new mutation
```

Important semantics:

1. An acknowledgement proves only that a device/interface reported receipt.
2. Actuator telemetry proves only the measured actuator response within its own sensor and calibration limits.
3. Physical-delta observation proves only the measured physical change and its coverage.
4. Independent acceptance proves only the postconditions explicitly checked by the verifier.
5. A timeout after command issuance creates **OutcomeUnknown**, not failure and not success.
6. While OutcomeUnknown is unresolved, no subsequent mutation may be issued against the affected work scope.
7. ReconciledNotExecuted permits a new attempt only with a new attempt ID and current preflight.
8. ReconciledExecutedAwaitingInspection requires observation and verification before any dependent work can proceed.
9. Missing evidence, stale evidence, frame mismatch or insufficient coverage cannot become a pass through planner confidence or majority vote.

This state machine is the common composition point for #12606 construction and #12609 asset maintenance. The payloads and task-specific acceptance logic remain separate adapters.

## 5. Robot roles: capability bundles, not a fixed robot zoo

We should design around **roles that can be implemented by different morphologies**. The role boundary survives hardware changes.

| Role | Typical embodiment | Primary work | Explicit exclusions |
|---|---|---|---|
| Scout / survey | Wheeled, tracked, legged or aerial where permitted | Reconnaissance, mapping, gas/environment sensing, route and coverage assessment | No structural or utility-clearance certification |
| Asset inspector | Tracked crawler, legged platform, rail/pipe crawler or manipulator | Close-range multimodal pipe/cable and support inspection; drawpoint observation from approved stand-off | No unapproved contact, entry or dislodging |
| Intervention platform | Bolter, drill, excavator, manipulator or specialized construction machine | Bounded excavation, support installation, lining, repair or material placement | No self-approval of design changes or closeout |
| Relay / guardian | Fixed node or mobile relay platform | Communications, localization anchors, environmental monitoring and emergency status relay | No independent mission expansion |
| Verification platform | Independent scan rig, second robot or separately calibrated metrology system | As-built measurement and acceptance evidence | No access to candidate-private state or permission to intervene |

A single robot may implement multiple roles only when the capability profile and evidence independence requirements are satisfied. The same camera used by the planner and verifier is not independent merely because it is queried twice.

## 6. System boundaries

| System | Owns | Must not own |
|---|---|---|
| **Symthaea** | Belief/hypothesis management, active sensing, planning, work selection, coordination proposals and evidence selection | Engineering sign-off, hidden-truth access, physical safety override |
| **Symtropy** | Geometry, geology/material hypotheses, equipment dynamics, sensor/actuator physics, excavation/support simulation and digital/physical-twin comparison | Human authority or self-certified field truth |
| **Mycelix** | Identity, signed work-package authority, provenance, leases, event ordering and reconciliation records | Physical permission derived from trust/reputation scores |
| **Sol-Atlas** | Rights-aware spatial context, asset/utility locations, coordinate lineage and source provenance | Geotechnical or structural acceptance |
| **Independent verifier** | Postcondition checks against approved criteria and qualified measurements | Candidate controller state, actuator command authority or design alteration |
| **Human / responsible engineer** | Approval, engineering judgement, exception disposition and release of newly scoped work | Unreviewed acceptance based solely on model confidence |

## 7. Benchmark composition

Do not collapse the programme into one scalar autonomy score. Publish independent hard gates and task/operational metrics.

| Track | Matched baselines | Primary metrics | Hard negatives |
|---|---|---|---|
| **B1 Observability** | Single modality, geometry-only, late fusion, human inspection, active inspection | Defect recall at fixed false-alarm rate, 3-D localization, coverage, unknown fraction, calibration, energy/time | Occlusion falsely marked clear; stale calibration; frame mismatch; correlated sensor fault |
| **B2 Intervention** | Scripted/open-loop, local feedback, risk-aware MPC, robust shared policy | As-built tolerance error, installation completeness, false acceptance, resource reconciliation, rework | Commanded-but-absent installation; partial/incorrect work; utility incursion; expired package |
| **B3 Verification** | Candidate self-report, same-sensor inspection, independent modality/metrology, human check | False-accept rate, sensitivity, evidence coverage, verifier latency and independence audit | Planner-private data leakage; incomplete coverage; duplicate evidence; common-cause failure |
| **B4 Robustness/recovery** | Nominal controller, fault-aware controller, frozen shared contingent policy | Worst-model violations, recovery time, safe hold/retreat, unresolved-work duration, evidence retention | Timeout, reboot, partition, stale state, duplicate/reordered receipts, robot immobilization |
| **B5 Fleet mission** | Single robot, greedy allocation, charging-aware planning, capability-aware fleet, verification-aware fleet | Accepted work per robot-hour, total energy, inspection completeness, task regret, fleet recovery and evidence continuity | Leased work duplicated; infeasible handoff; relay loss; shared-resource deadlock |
| **B6 Integrated closeout** | Human plan, commercial baseline where comparable, scripted and proposed loop composition | Accepted as-built output, total mission cost, rework, inspection effort, hazard exposure and closeout latency | Any hard-gate violation or unknown state incorrectly closed |

### 7.1 Existing quantitative hypotheses — retain, do not inflate

From #12606 and #12609:
- target at least **15% less rework or inspection effort** than matched scripted/open-loop construction baselines at equal physical quality;
- target at least **20% lower inspection time or energy** at non-inferior defect recall and false-alarm rate for asset-observability tasks;
- target **100% rejection** of the finite frozen stale-design, stale-calibration, frame/unit mismatch, no-contact and false-closeout negative controls;
- target **100% correct handling** of the frozen ambiguous-outcome cases.

These are proposed targets, not measured results. Report confidence intervals and per-scenario outcomes. Zero failures in a finite suite does not imply zero real-world risk.

### 7.2 Additional systems metrics

Report, separately:
- value of verification: consequential defects found per additional minute, joule and exposure unit;
- unknown-state debt: volume/asset area still unverified and age of unresolved claims;
- evidence latency: physical event to independently reviewable receipt;
- evidence continuity: fraction of accepted work whose full lineage survives reboot, partition and handoff;
- mutation containment: number and scope of commands admitted after the first unresolved outcome;
- common-cause sensitivity: false acceptance under shared power, timebase, mount, calibration, firmware and perception faults;
- intervention reversibility: bounded recovery cost and residual hazard after abort or partial execution.

## 8. Industry comparison and where Subterra should compete

| Reference | Demonstrated focus | What it does not establish for our problem | Subterra comparison |
|---|---|---|---|
| DARPA SubT / CoSTAR / CERBERUS | Extreme-environment exploration, mapping, artifact search and multi-robot autonomy | Construction acceptance, utility condition, support installation or verified repair | Preserve exploration baselines; add asset-state and intervention closeout |
| Sandvik AutoMine / AutoMine Aura | Commercial underground equipment automation, remote supervision, 3-D perception and productivity | Independent as-built verification for arbitrary third-party work packages | Compete on evidence-backed physical outcomes and cross-vendor closeout, not generic autonomy |
| Komatsu + AIM + Smart Construction | Autonomous bulldozer/excavator workflows tied to construction planning and terrain data | Independent verification semantics across geotechnical/asset constraints | Add typed tolerances, measurement lineage, unresolved outcomes and independent acceptance |
| 2026 charging-aware mine inspection study | Two-stage task allocation/path planning with road and charging factors; simulation reports 7.20–11.16% lower objective cost versus selected heuristics | Physical deployment, defect calibration, intervention correctness or evidence integrity | Reproduce the objective and scenario first; then evaluate value-of-verification and robust recovery |
| UC San Diego Pipe Network Inspection Robot (2026 project update) | Self-centering 2–6 inch in-pipe robots; 6-inch steel-network tests include elbows and T-joints; autonomous network traversal is an active next phase | A component mobility/joint-recognition demonstration, not mine-wide condition assessment or verified repair | Reproduce confined-pipe mobility and joint/pose baselines; add defect observability, condition acceptance and evidence continuity |
| 2025 gecko-inspired cable inspection robot study | Quadruped cable-inspection concept and image-based detection of insulation ageing, micro-cracks and contamination | Does not establish independent defect acceptance across sites or mine-wide operations | Compare matched sensor/defect tasks; add calibration, coverage, unknowns and independent corroboration |
| 2026 acoustic in-pipe inspection research | Dual-resolution acoustic sensing for autonomous in-pipe inspection | Modality-specific result; not a complete underground asset workflow | Include acoustic sensing as a matched modality baseline, with calibrated applicability and common-cause analysis |
| 2026 underground mine infrastructure review | Finds mine-specific pipe/cable inspection and drawpoint hang-up volumetric assessment comparatively under-addressed relative to re-entry and ground-condition assessment | A review of mine-infrastructure maturity, not proof that no adjacent commercial or research pipe/cable robots exist | Prioritize mine-specific condition assessment and drawpoint volume, while benchmarking against adjacent pipe/cable prototypes |

### 8.1 Standards are interface and safety inputs, not a certification claim

- **ISO 17757:2019** remains the current reviewed standard for safety of autonomous/semi-autonomous earth-moving and mining machine systems; it was reviewed and confirmed in 2024. Map its applicable requirements into the hazard analysis for each target machine and operational domain.
- **ISO 21423:2026** specifies communications and interoperability among industrial mobile robots, fleet managers and related equipment/applications. Its published scope explicitly excludes safety and navigation requirements. Use it as an interoperability reference, never as evidence of safety.

This architecture is not certification to either standard.

## 9. Staged qualification

1. **Synthetic reference:** deterministic plant/sensor models, sealed truth, frozen fault catalog, independent evaluator. Exact inputs and outputs are reproducible.
2. **Software-in-the-loop / HIL:** real controller and interface adapters against simulated plants, actuator feedback, clocks, comms faults and power cycles.
3. **Instrumented test cell:** physically bounded mock tunnel/utility corridor/drawpoint geometry, independent metrology, hardwired stop/isolation and specialist review. Validate only the tested material, geometry, tools and operating envelope.
4. **Supervised site pilot:** only after mine-owner authorization, task-specific hazard analysis, geotechnical/electrical/structural sign-off, emergency and rescue plan, regulatory review and independent field measurements.

Passing one stage authorizes evaluation at the next stage; it does not automatically qualify a new machine, site, material or task.

## 10. Immediate implementation order

1. Inventory and map the existing `mission_executive`, `work_orders`, `plan_freshness`, `temporal_event`, `recovery_journal`, safety/reflex and team APIs. Do not add duplicates.
2. Define a versioned `OperationAttempt` contract in a standalone research package: package/design/frame/calibration digests, attempt ID, idempotency key, sequence, resource envelope and explicit state.
3. Implement independent transition validation for command → acknowledgement → actuator observation → physical observation → verification.
4. Add an `OutcomeUnknown` reconciliation lock and adversarial tests for timeout, restart, duplicate receipt, stale package and partial execution.
5. Add a sealed synthetic construction fixture from #12606 and an asset-inspection fixture from #12609; keep their task-specific metrics distinct.
6. Add a composition runner that records per-loop deadlines, input freshness, preemption, evidence lineage and recovery outcomes.
7. Reproduce the 2026 charging-aware planning paper's declared simulation before claiming improvement on fleet routing.
8. Build the decision-conditioned inspection adapters by reusing the existing finite-horizon Subterra measurement planner/referee and the new [#12628 — inspection EVSI benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/12628); do not create a second generic planner.

## 11. References

1. DARPA, [Subterranean Challenge](https://www.darpa.mil/research/programs/darpa-subterranean-challenge) and [Final Event winners](https://www.darpa.mil/news/2021/subterranean-challenge-winners).
2. Sandvik, [AutoMine Aura launch, May 2026](https://www.mining.sandvik/en/news-and-media/news-archive/2026/05/sandvik-launches-automine-aura-a-first-of-its-kind-automation-platform-for-the-future-of-mining/). Vendor-reported performance.
3. Komatsu, [Komatsu / AIM autonomous construction partnership, 31 July 2026](https://www.komatsu.jp/en/newsroom/2026/20260731).
4. Xu et al., [Task allocation and path planning for multi-robot inspection in mines considering charging](https://www.nature.com/articles/s41598-026-65759-z), *Scientific Reports*, 6 August 2026. Simulation result; reproduce before comparison.
5. [UC San Diego Pipe Network Inspection Robot](https://robotics.ucsd.edu/projects/pipe-robot/). Project-reported 2–6 inch pipe prototypes; distinguish demonstrated steel test-network behaviors from the autonomous traversal phase underway in late 2026.
6. Guan & Shakibaei Asli, [Gecko-Inspired Robots for Underground Cable Inspection](https://www.mdpi.com/2079-9292/14/15/3142), *Electronics*, 5 August 2025.
7. [Sun et al., dual-resolution acoustic-sensing robot for autonomous in-pipe inspection](https://doi.org/10.1016/j.tust.2025.107133), 2026.
8. Naheem, Frimpong & Antwi, [Underground Mine of the Future: A Review of Robotic Technology Intervention for Infrastructure Resilience](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2026.2017682/abstract), 2026.
9. [ISO 17757:2019](https://www.iso.org/cms/%20render/live/en/sites/isoorg/contents/data/standard/07/61/76126.html).
10. [ISO 21423:2026](https://www.iso.org/standard/86749.html).

## Claim ceiling

This is a research architecture and benchmark proposal. It does not establish improved autonomy, lower risk, geotechnical adequacy, structural integrity, equipment certification, mine safety, field productivity, regulatory compliance or permission to intervene. Those claims require the relevant matched experiments, independent physical evidence and responsible engineering authority.
