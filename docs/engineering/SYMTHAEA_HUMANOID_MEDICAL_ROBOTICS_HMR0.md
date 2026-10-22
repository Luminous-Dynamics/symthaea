# Symthaea Humanoid Medical Robotics Platform — Concept and Engineering Contract v0.1

**Working program:** HMR-0 (Humanoid Medical Robotics, research generation 0)  
**Status:** concept/design research. Not a device specification, clinical device classification, safety case, procurement specification, or authorization to operate.

## Executive decision

Yes: Symthaea should be used to co-design a humanoid medical-robotics platform and its task-specific tools. But the engineering target is not a humanoid that can freely perform clinical work. The target is a **toolable, evidence-bound, independently constrained robotics platform** that starts with non-clinical hospital logistics and progressively earns additional capabilities through distinct validation stages.

The first design question is not "how humanlike can it be?" It is "which tasks justify a humanoid body rather than a wheeled mobile manipulator or fixed automation?" Design and benchmark all three on common tasks. The full biped is a candidate, not an assumption. If a less complex morphology outperforms it on task completion, energy, cost, recovery, accessibility and total safety burden, use the simpler morphology for that task.

## 1. Intended initial use

### Initial research scope
- Move sealed, labelled *mock* supply totes between fixed staging points in simulation.
- Recognize inventory labels and report discrepancies for a human to resolve.
- Present nonclinical supplies on a passive exchange tray.
- Test access to mock drawers/doors only in simulation, with measured fixtures and explicit force limits.
- Validate task completion, tool attachment identity, collision-aware motion, privacy, interruption recovery and auditability.

### Explicitly out of scope for HMR-0
- Diagnosis, triage, clinical recommendations or autonomous interpretation of patient observations.
- Medication selection, preparation, dispensing or administration.
- Invasive, surgical, rehabilitation or therapeutic actions.
- Patient lifting, transfers, restraint, physical examination or direct patient contact.
- Entering or interacting with sterile fields or claiming sterility/disinfection.
- Autonomous clinical prioritization or overriding a clinician.
- Physical actuation from this initial benchmark suite.

This exclusion is a design boundary, not a finding that these tasks could never be developed. Each would require a separate intended-use, hazard, regulatory and clinical evidence package.

## 2. Reference architecture

Use five separated planes. Symthaea must not be the sole safety controller or sole verifier of its own designs.

1. **Cognition and mission planning — Symthaea.** Generates candidate tasks and plans; tracks uncertainty; asks for missing evidence; explains assumptions; proposes tool concepts and design revisions. It has no direct torque, motor-current, brake-release or medical-function authority.
2. **World model and simulation — Symtropy.** Maintains versioned environment state, robot/tool digital twins, task episodes, contact models, resource use and recovery scenarios. Every plan binds to the exact world, robot, tool and calibration revisions.
3. **Capability gateway.** Checks signed plan identity, task allowlist, tool manifest, revision freshness, operator authorization, environment constraints and evidence class before a plan can be queued for simulation or bench qualification.
4. **Independent safety controller.** Separate deterministic/safety-rated hardware and software boundary for joint/velocity/workspace limits, torque/current limits where relevant, watchdogs, emergency stop, protective stops, fall/power-loss response and communication-loss behavior. The cognition layer cannot bypass this boundary. Actual safety functions require hazard analysis and appropriate functional-safety engineering; this concept does not prescribe a SIL/PL target.
5. **Physical interface and evidence ledger.** Tool changer, sensors, power/data, brakes and mechanical structures have explicit interface IDs. All design outputs, tests, revisions and disposition receipts carry hashes and evidence class. A passing simulation does not promote itself to physical qualification.

## 3. Morphology portfolio: a real baseline, not a marketing comparison

Keep three builds under the same task contract:
- **M1 — fixed/bench manipulator:** the lower-complexity baseline for repeatable staging or drawer-fixture tasks.
- **M2 — wheeled mobile manipulator with humanoid upper body:** likely the first practical hospital-aisle research platform; avoids treating bipedal balance as a prerequisite for useful service.
- **M3 — full biped humanoid:** research candidate for environments where stairs, step-over obstacles or human-height interaction demonstrably justify the balance, fall, energy and maintenance costs.

Run common tasks, resource budgets and fault injections. Compare completed qualified tasks per Wh and per wall-clock hour, human interventions, fall/near-collision severity, recovery, maintenance burden, fit to real facility geometry and full lifecycle cost. Do not combine safety hard gates into a weighted score that a high productivity result can compensate for.

## 4. Tool Forge: Symthaea designs tools as versioned, testable products

A tool proposal is not an image or a plausible CAD mesh. It is an auditable package with:
- tool ID, revision, intended task and interaction domain;
- geometry/design digest and simulation-model digest;
- robot mount/interface and verified attachment state;
- mass, center of mass, payload range and compatible object/fixture classes;
- force/torque/speed envelope with units and the task/fixture source for those limits;
- collision geometry and keep-out envelope;
- required sensors, calibration profile and tool-state detection;
- energy/power/data interface, if applicable;
- cleanability/material compatibility and explicit contamination/sterility status;
- failure modes, tool-detach detection and safe stop/release behavior;
- verification cases, test artifacts, open risks and design owner/reviewer evidence.

The initial parameterized concept definitions are stored in `docs/engineering/hmr0-tool-cad-concepts-v0.1.json`. This is a design-input contract with parameter ranges and test proposals, not generated geometry or manufacturing data. The separate Symtropy manifest gate is in [PR #2699](https://github.com/Luminous-Dynamics/symtropy/pull/2699).

### First tool concepts to model
- **SealedTote Gripper:** compliant, geometry-adaptable parallel grasp surfaces for sealed mock supply bins; no cutting, piercing, powered heat or patient contact; load/center-of-mass and slip detection are mandatory model parameters.
- **PassiveTray Dock:** alignment funnels and mechanical retention for a passive exchange tray; tool cannot autonomously dispense medication or hand objects directly to a patient.

Add a **Drawer/Lever Adapter** only as a simulation-only concept after force measurement, pinch-point geometry and failure release are represented. The initial physical-tool authorization remains false for every design.

### Tool qualification rule
A design must pass in sequence: schema/provenance review → geometry/collision checks → simulation across nominal and adversarial cases → bench fixture measurements → independent design/safety review → any separate applicable conformity assessment. A candidate may not issue its own approval. A generated CAD asset never counts as a physical test result.

## 5. Development gates

| Gate | Work allowed | Evidence to exit |
|---|---|---|
| H0 — concept | Requirements, trade studies, CAD/digital mock-ups | Assumption ledger, risk register, comparative morphology study |
| H1 — simulation-only | Synthetic objects, nonclinical tasks, no hardware actuation | Reproducible episodes, fail-closed tool manifests, exact-head receipts |
| H2 — benchtop, non-human | Instrumented fixtures, e-stop, physical mock totes | Measured load/force, cycle reliability, attachment detection, stop-distance and failure tests |
| H3 — hardware-in-the-loop | Safe hardware boundary with physical outputs disabled or isolated | Timing, watchdog, communication-loss, sensor corruption and safety-controller independence evidence |
| H4 — facility pilot | Nonclinical logistics only, under site governance and risk assessment | Facility-specific approval, human factors, cybersecurity, privacy, maintenance and operational evidence |
| H5 — clinical/patient interaction | Not authorized by this program tranche | A new intended-use and regulatory strategy, qualified clinical/engineering review, risk management and applicable conformity/ethics approvals |

No gate automatically confers the next one. H0–H3 are engineering evidence levels, not a regulatory classification.

## 6. Design and test method

Use requirements-to-evidence traceability. For each requirement, store: source and version, owner, hazard/risk rationale, acceptance criterion, verification method, test artifact, result, reviewer and design revision. Record contradictions and missing evidence as blockers rather than resolving them through plausible generation.

Symthaea may propose alternatives and critique prior work; an independent checker must validate structure and constraints, external solver results must be distinguished from dry-run stubs, and a qualified human reviewer must control acceptance. Where simulation fidelity is unknown, model discrepancy is an explicit uncertainty term.

## 7. Initial benchmark targets

Targets below are provisional acceptance criteria for the declared *synthetic* suite, not achieved results or claims of medical safety.

- 100% of tool manifests with unknown/missing required fields are rejected before scoring.
- 100% of proposed designs carry stable geometry, model, tool and robot revision identifiers.
- 0 simulation-eligible tool proposals may enable physical actuation, patient contact, medication delivery, diagnosis, therapy, surgery or sterile-field access.
- 100% detection of the frozen invalid-manifest and stale-tool/revision mutations.
- 0 false-eligible outcomes in the registered synthetic contract suite.
- 100% of simulated tool changes invalidate any plan derived from the previous tool attachment/revision.
- For every task, report completion rate, partial/failure categories, time, Wh, human interventions, collision/proximity violations and recovery success with denominators and uncertainty.
- No single aggregate score across tasks with incompatible units; report the Pareto frontier for M1/M2/M3.

Before hardware testing, define sample sizes and confidence bounds for cycle reliability and rare failures with a qualified safety engineer. "Zero failures in N trials" is not equivalent to zero risk; state N and the sampling/model limitations.

## 8. Standards and regulatory watchlist — applicability not yet determined

This is a mapping for qualified review, not a compliance checklist:

- **IEC 80601-2-78:2019 + AMD1:2024** concerns medical robots that physically interact with a patient with impairment to support movement-function rehabilitation, assessment, compensation or alleviation. It may become relevant for an intended use in that scope. It does not automatically cover a non-contact supply logistics robot.
- **IEC/DIS 80601-2-78 Edition 2** is a draft shown by ISO as under development as of 2026-10-22; track it rather than treating the draft as the current published edition.
- **ISO 14971:2019** is the medical-device risk-management process standard; ISO lists it as current after review/confirmation in 2025.
- **IEC 62304** medical-device software lifecycle, **IEC 62366-1** usability engineering, **IEC 60601 series** medical electrical equipment, and **ISO 13485** quality management may be relevant depending on intended use and jurisdiction.
- **ISO/FDIS 13482 Edition 2** explicitly excludes medical applications in its scope; do not use a service-robot standard as a substitute for medical-robot safety requirements.
- For humanoid general-purpose performance, use the six broad categories in Fraunhofer IPA's May 2026 benchmark as a comparison set: basic capabilities, complex tasks, cleanability/cleanroom suitability, functional safety, cybersecurity and energy. Add healthcare-specific contamination control, tool accountability, human workflow and downtime/recovery.

A qualified regulatory specialist must determine applicability for the specific product, intended use and target market before any clinical deployment.

## 9. Research basis and links checked 2026-10-22

- Liang et al., *In vivo feasibility study of humanoid robots in surgery*, Nature 657 (2026), DOI: https://doi.org/10.1038/s41586-026-10796-x. This reports a humanoid laparoscopic teleoperation framework, benchtop characterisation, dry-lab user studies and in-vivo porcine work. It is a concrete comparator for specialist surgical teleoperation, not evidence that general-purpose autonomous humanoids are ready for clinical work.
- Fraunhofer IPA, *Benchmark for humanoid robots*, 27 May 2026: https://www.ipa.fraunhofer.de/en/press-media/press_releases/benchmark-for-humanoid-robots.html
- ISO, IEC 80601-2-78:2019: https://www.iso.org/standard/68474.html
- ISO, draft IEC/DIS 80601-2-78 Edition 2: https://www.iso.org/standard/90453.html
- ISO, ISO 14971:2019: https://www.iso.org/standard/72704.html
- NIST, Performance Assessment Framework for Robotic Systems: https://www.nist.gov/programs-projects/performance-assessment-framework-robotic-systems
- NIST, Grasping, Manipulation, and Contact Safety Performance: https://www.nist.gov/programs-projects/grasping-manipulation-and-contact-safety-performance-robotic-systems

## 10. Immediate deliverable

HMR-0 begins with the architecture and tool-manifest contract, a machine-readable catalogue of tool candidates, and a standalone adversarial tool-qualification benchmark in Symtropy. This tranche outputs only simulation-review dispositions and evidence receipts. It does not generate device-ready manufacturing files or authorize physical actuation.
