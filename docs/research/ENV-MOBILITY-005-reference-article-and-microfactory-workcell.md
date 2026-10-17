# ENV-MOBILITY-005: Freeze a first repairable survey-robot manufacturing reference article

**Status:** Engineering work package and illustrative routing fixture. No hardware has been built, no parts have been purchased, and no cost, calibration, or manufacturing result is claimed.  
**Parent proposal:** [PR #7870 — environmental survey and repairability benchmark kit](https://github.com/Luminous-Dynamics/symthaea/pull/7870).  
**Reuse, do not duplicate:** [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608), [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686), [ROB-MFG-001 #5775](https://github.com/Luminous-Dynamics/symthaea/issues/5775), [FACTORY-OPS-001 #7677 / PR #7678](https://github.com/Luminous-Dynamics/symthaea/pull/7678), [AIRCRAFT-FACTORY-TRAINING #7652](https://github.com/Luminous-Dynamics/symthaea/issues/7652), and [benign manufacturing cell pilot #6040](https://github.com/Luminous-Dynamics/symthaea/issues/6040).

## Decision: start with a sensor module, not a complete vehicle

The first physical reference article should be a **low-voltage, field-replaceable environmental sensor/logger module with a simple mounting adapter**, called SPM-0. Initially, use an ordinary dry-side sensor profile such as temperature / relative humidity / barometric pressure, a low-power data logger and a removable harness. Freeze exact part candidates only after checking datasheets, local availability, interface semantics, and calibration resources.

This is intentionally smaller than a complete drone or outpost. It exercises design release, editable CAD, BOM and substitutions, simple fabrication, wiring, inspection, calibration, data lineage, service access and repair evidence before introducing flight or marine hazards. It also creates a sensor payload that may later fit more than one vehicle family through **separate, qualified mounting adapters**.

The first article does **not** claim a water-quality sensor, waterproofing, pressure resistance, flightworthiness, airspace authorization, marine readiness, or environmental instrument accreditation. Do not use the dry-side enclosure underwater. Underwater deployments need separate wet-side design, ingress protection, connector, pressure and qualification evidence.

## Paradigm: establish the local repair closure boundary

Do not ask “can the factory make the entire drone?” as a yes/no. Produce a capability closure map:

- **Level 0 — replace from supplier:** exact qualified module is sourced and traceable.
- **Level 1 — local field repair:** a trained operator replaces the module with common tools and performs specified post-repair checks.
- **Level 2 — local passive-part production:** fixtures, covers, adapter plates and non-safety-critical brackets are fabricated locally and dimensionally inspected.
- **Level 3 — local assembly:** approved harnesses, mounting kits and sensor/logger subassemblies are built locally from documented qualified parts/processes.
- **Level 4 — local measurement renewal:** reference tools/instruments, calibration status and measurement uncertainty can be maintained through a defined external or in-house service path.
- **Not assumed — self-manufactured critical electronics, propulsion, batteries or pressure boundaries:** these remain explicit supply and qualification dependencies unless a separate evidence-backed program closes them.

Each level must have its own evidence. “Can print a replacement” is not equivalent to “replacement conforms”; “can assemble a battery” is not equivalent to “battery is qualified.” This composes Symthaea's robotics repair-closure owner (#5775) rather than inventing a new self-manufacturing score.

## Design requirements to freeze before CAD release

The design owner should create a signed-off input sheet for SPM-0 containing:

- the intended measurand(s), units, range, accuracy/uncertainty need, update rate and environmental envelope;
- exact sensor/logger part IDs, datasheet revisions, electrical interface and allowed supply range;
- connector pinout, polarity, wire specification, strain relief and replaceable harness interface;
- mounting datum, hole/slot tolerances, maximum allowable mass and envelope for the intended carrier;
- serviceability target and list of parts permitted to be changed in the field;
- source files and neutral CAD/drawing exports, BOM with approved and prohibited substitutions;
- firmware/configuration identity, acquisition profile, time basis and data schema;
- calibration/check method, reference-instrument identity and acceptance tolerance derived from the declared application requirement;
- inspection steps, failure states, repair records and specific retest obligations after any change.

Do not invent an accuracy threshold before the measurement task and selected instruments are specified. Calibrate against an appropriate reference and report measurement-system uncertainty. A module identity does not make its calibration current.

## Illustrative manufacturing routing

The companion file **env-mobility-microfactory-routing-v1.json** is an explicit **planning fixture**, with illustrative durations and required skills/resources. It is not claimed to be directly consumable by the scheduler API until the schema is mapped against [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677) / PR #7678.

A separate effects-disabled fixture audit is now available in source as `cargo run -p xtask -- env-mobility-route-audit`, with route mutation tests under `cargo test -p xtask env_mobility_route`. It validates schema consistency, required worker/resource candidates, dependency cycles, mandatory operations and key test/manifest/acceptance ordering. It is **not** a scheduler run and it does not calculate makespan, worker assignment, bottleneck utilization, or real manufacturing throughput. Cargo/test status remains unverified until it is compiled and run on this exact revision.

The target routing is:

1. **Release design package** — verify requirement, CAD/drawing revisions, BOM, wiring table, firmware/configuration and calibration plan.
2. **Incoming inspection** — match exact parts/labels and approved BOM substitutions; record lot/part evidence and physical condition.
3. **Fabricate adapter / print non-critical fixture** — prefer a purchased plate or locally fabricated non-structural adapter based on tolerance, durability and total cost. The print fixture may be used to drill/check; it is not a flight-critical part.
4. **Prepare harness** — cut, crimp and label an approved harness; check pinout, polarity and continuity with a repeatable fixture.
5. **Inspect adapter and interfaces** — measure datums/fit, verify connector clearance, access, strain relief and ability to remove the module without destructive disassembly.
6. **Assemble module** — mount sensor/logger, connect the harness and secure connectors using the released work instruction.
7. **Electrical smoke test** — use an appropriately current-limited bench supply; log input voltage/current and function; no carrier, propulsion or actuation connection is required at this stage.
8. **Measurement check** — compare readings with a traceable/identified reference under the declared check conditions; retain raw readings, reference identity, calibration status and residuals.
9. **Bind build manifest** — exact article serial, components, substitutions, source/configuration digest, tool/fixture references, calibration and test artifacts.
10. **Independent acceptance review** — verify all required gates; reject missing/failed evidence; preserve deviations and rework history instead of silently marking the unit passed.

Expected durations in the JSON are initial **planning estimates only**, not measured cycle times. After five baseline units, replace them with observed time distributions separated into setup, touch labor, queue, rework, inspection and calibration.

## Low-capital starter workcell

Start with tools that directly support the routing and are serviceable.

### Acquire or borrow for the initial dry-side bench
- quality hand tools, calipers and gauges appropriate to drawing tolerances;
- an ESD-aware electronics bench and suitable solder/crimp tools if the selected connector/process requires them;
- current-limited bench supply, calibrated multimeter and a harness continuity/polarity fixture;
- connector-specific tooling for the chosen exact contacts—do not assume a generic crimper is valid;
- fixture printer or access to a local print service, with approved material/process for the non-critical adapter or gauge;
- computer/workflow for editable source, CAD, versioned BOM, work traveler and data capture;
- suitable storage, labeling, lighting, ventilation and PPE for the chosen processes.

### Use a supplier or qualified lab initially
- sensor/logger modules, connectors and cable until a measured business case justifies a custom board;
- external reference instruments/calibration where in-house calibration capability does not exist;
- precision machining where required tolerances exceed the local process capability;
- all rated underwater, battery-pack, flight-critical and pressure-boundary parts/tests.

Do not buy a CNC machine, automated assembly robot, environmental chamber or automated test station simply because it looks like a factory. Add equipment when repeated measured demand, tolerance capability, supplier lead-time or true lifecycle-cost data justify it. If a supplier provides a cheaper conforming adapter with lower lead time and adequate provenance, outsourcing that operation may be the correct factory design.

## Proposed pilot and benchmarks

### Batch A — establish a truthful baseline
Build five SPM-0 units using the same released revision. Record all labor, setup, waiting, inspection, scrap, rework, calibration and missing-part events. Keep each serial's configuration and evidence distinct.

### Batch B — test one improvement at a time
Change one concrete issue—fixture design, harness length, connector access, fastener commonality or work-step order—while preserving the product and acceptance criteria. Compare the following batch against the frozen baseline and independently review whether the result improved.

### Initial target hypotheses, not results
- **Repair:** replace a routine field-replaceable sensor/logger module in ≤10 minutes; report diagnosis and post-repair verification separately and also report end-to-end return-to-service time.
- **Tools:** ≤3 routine tool types for 90% of declared routine repairs, unless interface/electrical or structural requirements justify a specialist tool.
- **Access:** zero destructive disassembly for parts declared field-replaceable.
- **Configuration integrity:** 100% of built units bind exact serial, BOM/substitutions, released source/configuration and required test/calibration records.
- **Test integrity:** zero release-to-acceptance when required polarity, continuity, function, configuration or calibration checks are missing or failed.
- **Assembly learning curve:** after recording a five-unit baseline, target ≥30% lower assembly touch labor at 25 cumulative units without reducing first-pass yield or measurement quality.
- **Manufacturing yield:** target ≥95% first-pass yield only after the process is stable and the definition of a unit passing all release checks is frozen. Any failed critical check blocks acceptance regardless of aggregate yield.
- **Cost:** calculate fully burdened cost per valid calibrated observation, separately reporting module BOM, tooling amortization, labor, scrap, external services, failed checks and maintenance. Do not claim ≥30% lifecycle cost improvement until compared with a frozen baseline at equivalent measurement quality and mission profile.

These are software/bench/process targets. SPM-0 does not qualify a vehicle for flight, surface navigation, underwater use or field deployment.

## Data and tooling contracts

The digital thread should bind the design revision → routing → actual components/materials → exact physical article → measured dimensions and electrical state → calibration → acceptance or rework → repair. Keep a separate identity for the physical article versus the model and test fixture. A component substitution or sensor remount may invalidate its former evidence.

Use the existing manufacturing process, metrology, engineering V&V, scheduler and Mycelix production/provenance owners. Where interoperability helps, map information to established standards such as STEP AP242 for product definition and QIF for quality measurement instead of creating competing identifiers or exchange formats. NIST's digital-thread research emphasizes the loop between design, manufacturing and inspection; the proposed workcell should preserve that loop even when work is outsourced. Sources: [NIST Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing) and [NIST Digital Thread for Smart Manufacturing](https://www.nist.gov/programs-projects/digital-thread-smart-manufacturing).

## Stop conditions

Stop release and return to design/evidence review if the released file doesn't match the physical article, a substitute lacks approval, a routine module cannot be removed as specified, a required instrument reference is missing, calibration/check fails, a measurement is represented above its evidence class, or a critical post-repair test is skipped.

**Goal:** prove a small, useful product can be designed, assembled, inspected, calibrated and repaired consistently with an inexpensive workcell. Only then expand the same digital-thread process to a full multirotor, surface vehicle and eventually separate underwater systems.
