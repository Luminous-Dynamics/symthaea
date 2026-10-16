# Aircraft Factory Design and Practical Skills Training

**Date:** 2026-10-16  
**Status:** research and integration contract; no factory or aircraft production process is qualified by this document.  
**Tracking issue:** [AIRCRAFT-FACTORY-TRAINING-001 #7652](https://github.com/Luminous-Dynamics/symthaea/issues/7652)

## Executive thesis

Symthaea could help design aircraft factories by co-optimizing the aircraft product, manufacturing processes, layout, equipment, utilities, workforce skills, inspection capacity and lifecycle economics. It should not act as a standalone plant-layout generator. It should compose existing engineering, factory realization, manufacturing-process, materials, robotics, facilities and qualification owners.

Praxis could be used to train designers, manufacturing engineers, mechanics, technicians, machinists, inspectors and maintainers through interactive simulations tied to the factory's actual processes. Praxis should be an instructional and competency-assessment interface—not an authority that declares someone licensed, an aircraft airworthy, or a process qualified.

**Praxis discovery caveat:** searches of the accessible public Luminous-Dynamics repositories did not identify a repository or issue clearly named Praxis. This document therefore defines a versioned integration boundary and does not pretend a Praxis API exists. Once the exact project repository and owner are identified, the adapter can be mapped to its real interfaces.

## Existing owners to compose

This should extend existing owners, not introduce parallel factory or manufacturing subsystems:

- [BUILT-ENV-001 #6032](https://github.com/Luminous-Dynamics/symthaea/issues/6032): evidence-bound realization of buildings, factories and supporting site infrastructure.
- [ENG-DESIGN-001 #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005): requirements, design decisions, verification/validation, design-for-X and change impact.
- [ROB-DESIGN-000 #4795](https://github.com/Luminous-Dynamics/symthaea/issues/4795): robotic design proposals, simulation, fabrication, commissioning and physical evidence.
- [AIRCRAFT-ENERGY-001 #7646](https://github.com/Luminous-Dynamics/symthaea/issues/7646) and [PERSONAL-SOARING-001 #7648](https://github.com/Luminous-Dynamics/symthaea/issues/7648): initial civilian aircraft product and mission scope.
- Mycelix [MFG-EVID-001 #3084](https://github.com/Luminous-Dynamics/mycelix/issues/3084): physical product revisions, manufactured-unit identity and qualification evidence.
- Mycelix [MFG-PROC-001 #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085): distributed manufacturing process/resources/routing.

Symthaea owns engineering and process meaning; Mycelix owns the distributed work/resource/provenance record; Praxis, once identified, would own learner interaction and training delivery. Qualified personnel and the applicable aviation authority retain real-world approval and licensing roles.

## First factory: deliberately modest

The first model should cover a low-volume, civilian light-aircraft or glider assembly/maintenance line. It should not begin with a transport-category airliner, a high-rate composite plant, or an assumed clean-sheet aircraft.

Compare several routes only where the chosen aircraft design and its approvals permit them:

- conventional metal subassemblies, machining, fastening and inspection;
- tube-and-fabric or other established light-aircraft construction, where applicable;
- composite layup/infusion, only with material- and process-specific qualification, controlled environment, cure evidence and qualified inspection;
- final assembly, electrical/harness installation, propulsion installation, weighing, control rigging and final inspection as separate controlled process families.

The objective for the first iteration is a reproducible facility/process model and realistic workforce training scenarios—not a claim that an aircraft can be manufactured or released from this model.

## Paradigm shift: co-design product, process, factory and workforce

A conventional workflow often hands the design to manufacturing and then discovers that a component is difficult to fixture, inspect or assemble. The proposed system keeps these constraints in the same design loop.

### 1. Product/process co-design

Symthaea compares design variants on manufacturability, part count, tolerances, inspection access, materials/tool availability, assembly sequence, repairability and approved configuration control. An apparent aerodynamic or weight improvement should be evaluated together with production cost, assembly error risk and inspection burden.

### 2. Factory digital twin

Generate and compare candidate cell layouts and flow plans. Represent physical constraints such as material movement, machine/fixture capability, operator skills, lifting/handling, utilities, metrology, environmental controls, safety zones, maintenance access, and inspection/hold-point capacity.

Keep these states distinct:

- designed building vs verified site and utility capacity;
- installed machine vs commissioned and calibrated resource;
- planned operation vs executed work;
- simulated production rate vs measured production capability;
- completed assembly vs inspected and accepted unit.

### 3. Skill-aware production planning

A schedule is infeasible if it assumes an unavailable qualified technician, expired calibration, unavailable fixture, unapproved material lot, missing inspection capacity, or an operator lacking the required authorization. Throughput optimization must not average away these hard constraints.

### 4. Human–robot teaming

Use automation for tasks where repeatability, ergonomics, access or measurement justify it, and where a process-specific safety case exists. Do not pursue maximum automation as an abstract target. NASA has studied human–robot teaming to assist skilled aviation manufacturing workers with manual-task bottlenecks and ergonomics; this is a useful precedent for augmenting skilled work rather than assuming wholesale replacement.

### 5. Continuous learning from production evidence

Compare predicted cycle time, yield, rework and failure modes with real measurements. Symthaea identifies bottlenecks and proposes experiments. Praxis uses approved process revisions and observed failure modes to make training more realistic. Changes are released only after the relevant qualified process review, practical validation and required approvals.

## Praxis: proposed training architecture

Until the Praxis codebase/API is identified, define a provider-neutral versioned training-event contract. It should support:

- lesson/module ID and revision;
- competency ID, prerequisites and the authoritative framework it maps to;
- authorized source procedure/manual/standard revision and rights;
- simulator, workstation, tool and process-profile identity;
- scenario ID, allowed actions, constraints, injected fault and expected critical behaviors;
- learner/session identifier and privacy policy;
- raw event/action trace and measurement outputs;
- rubric version, scoring method, assessor identity and independent verification status;
- distinction between formative learning, internal training, supervised on-the-job evidence and formal external qualification;
- a learner portfolio export with revision/currentness and source provenance;
- revocation/supersession when procedures or competency standards change.

Do not send sensitive learner or employment data to a public provenance graph. Do not score competence by time-on-platform, quiz scores alone, or generative-AI confidence. Practical skills require task-aligned evidence, observation and appropriate assessment.

### Initial curriculum

1. **Engineering literacy:** drawings, units and tolerances, materials, measurement, configuration and requirements.
2. **Manufacturing basics:** process plans, work instructions, fixtures, fasteners, torque, material traceability, cleanliness/FOD and controlled nonconformance handling.
3. **Assembly:** alignment, control-system installation/rigging verification, harness routing, connectors/grounding, propulsion installation and mass/balance awareness.
4. **Quality and inspection:** calibration, metrology, acceptance criteria, defect classification, root-cause analysis, independent checks and audit records.
5. **Maintenance:** interpreting approved maintenance data, troubleshooting, part eligibility, logbooks, service bulletins and escalation/return-to-service boundaries.
6. **Engineering practice:** requirements decomposition, loads/material evidence, tolerance stack-up, simulation limitations, risk analysis, V&V and change impact.
7. **Human factors:** fatigue, shift handovers, communication, stop-work behavior, error-chain recognition and incident reporting.
8. **Factory operations:** bottleneck analysis, scheduling, resource qualification, calibration currentness and safe maintenance of production equipment.

Use public or appropriately licensed technical material. Praxis must not reproduce proprietary maintenance instructions without permission.

## Evidence pipeline

~~~mermaid
flowchart TD
    A["Approved requirements and aircraft design revision"] --> B["Symthaea: process alternatives and factory layout"]
    B --> C["Frozen process plan, tooling and acceptance criteria"]
    C --> D["Mycelix: work order, resource and material/component provenance"]
    C --> E["Praxis adapter: training scenarios tied to the exact revision"]
    E --> F["Scenario events, rubric and practical assessment"]
    D --> G["Manufacturing execution and inspection"]
    F --> H["Training evidence: formative vs independently assessed"]
    G --> I["As-built serial/configuration and inspection evidence"]
    H --> J["Qualified instructor / assessor review"]
    I --> K["Qualified engineering and applicable authority decisions"]
    J --> K
    K --> L["Approved production/maintenance process and controlled change"]
    L --> M["Compare predictions with measured outcomes"]
    M --> B
~~~

Training records and as-built manufacturing records are linked but not interchangeable. A learner's successful simulation does not establish a qualified repair, a released component, an aircraft inspection sign-off, a licence or airworthiness.

## Benchmarks and adversarial controls

Freeze the reference process, facility model, training rubric, source revisions and material/component identity before scoring. The proposed goals below are hypotheses, not existing results.

### Factory and production

- **100% process traceability:** every released operation references a versioned process, requirements, required skills/authorizations, tools/fixtures, input material identity, acceptance criteria and required inspection hold points.
- **100% pilot-unit lineage:** each serialised subassembly/product binds to its design revision, process revision, relevant component/material lots, calibrated measurement, inspection and nonconformance/rework history.
- Measure first-pass yield, escape defects, rework hours, scrap/material loss, labor hours, lead time, bottleneck utilization and unplanned downtime separately.
- No schedule may pass if required skills, material, tooling/calibration, inspection capacity or facility utility constraints are unknown or violated.
- Stretch target: at least **10% lower rework hours or end-to-end lead time** than a frozen baseline, without increasing escape defects, weakening acceptance criteria, skipping inspection or worsening worker safety.

### Praxis and skill acquisition

- Every assessed competency maps to a specific source revision, rubric and task context.
- Evaluate on held-out scenarios using an independent verifier, not solely the same generator that authored the training lesson.
- Measure correct diagnosis, process selection, measurement accuracy, stop/escalation behavior, error detection, delayed retention and transfer to supervised physical practice.
- **Critical-error gate:** every known-critical unsafe action in the frozen challenge set must result in an explicit task failure or safe stop. A high aggregate score cannot compensate for one critical error.
- Stretch target: at least **20% less time-to-proficiency or 20% fewer repeated errors** on a preregistered task set versus the existing training baseline, while maintaining or improving independent practical assessment outcomes.
- Report sample size, confidence intervals, assessor agreement, false-pass and false-fail rates, retention delay and transfer to real tasks.

### Integration and adversarial cases

The test corpus should include wrong or unknown material lot, expired calibration, changed procedure revision, wrong fixture, incomplete inspection, out-of-tolerance part, misrouted harness, incorrect fastener/torque selection, missing required competence, changed simulator build, unauthorized substitution and a training event incorrectly represented as approval.

For all critical cases, preserve exact source revision and expected outcome before testing. Unknown APIs, expired procedures, missing assessors, incompatible schemas, and missing evidence must produce indeterminate/rejected states, never a green pass.

The verifier must recompute results from recorded events and the frozen rubric, not trust a UI score supplied by the training system.

## Regulatory boundary

For a first South African pilot:

- SACAA describes aviation training for licensing/rating privileges as subject to an approved Aviation Training Organisation and its approved training specifications. See [SACAA Training](https://www.caa.co.za/industry-information/personnel-licensing/training/) and its [ATO verification portal](https://www.caa.co.za/ato-home/).
- SACAA lists manufacturing-organisation approval guidance and forms on its [Airworthiness page](https://www.caa.co.za/industry-information/airworthiness/). A simulated plant does not qualify a real manufacturing organisation.
- SACAA advises prospective aircraft builders to contact its Engineering Office, with guidance in its [FAQ](https://www.caa.co.za/information-for-the-public/frequently-asked-questions/).
- The FAA provides a reference example of formal aircraft mechanic experience paths and written/oral/practical examinations, but those US requirements do not replace South African rules: [FAA experience and training paths](https://www.faa.gov/mechanics/become/experience), [practical testing standards](https://www.faa.gov/mechanics/become/testing/practical).
- NASA's [human–robot teaming for aviation manufacturing project](https://techport.nasa.gov/projects/96115) is a precedent for process-centered human assistance and ergonomic improvement.

Recommended gates: product/category and jurisdiction selection; frozen requirements/layout/skills model; simulated line and learner scenarios; qualified review; training on non-flight articles and supervised practice; physical process and measurement-system qualification; required authority oversight; and controlled production with ongoing configuration/change management.

## Responsibilities

- **Symthaea:** engineering analysis, factory/layout alternatives, process meaning, simulation, uncertainty, optimization, predicted-vs-observed analysis and experiment selection.
- **Praxis (pending actual codebase identification):** lesson and scenario delivery, learner interaction, training assessment, and export of versioned event/portfolio evidence.
- **Mycelix:** distributed resource/work-order/routing references, manufacturing identity, component and material provenance, inspection/test/maintenance lineage and external attestations; reuse MFG-EVID/MFG-PROC and AeroCommons owners.
- **Qualified personnel and authorities:** real engineering approval, practical assessment, inspection acceptance, certification, licensing, maintenance release and manufacturing/airworthiness decisions.

## Claim ceiling

This document does not claim that an aircraft factory has been designed, that a process route is production-ready, that Praxis already exposes an identified API, or that a training module confers competence or formal qualification. It defines a reproducible route to evaluate factory alternatives and a safe adapter for training evidence. Factory operation, production release, maintenance authority and aircraft airworthiness remain separate engineering and regulatory gates.
