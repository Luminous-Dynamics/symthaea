# ENV-MOBILITY-002: Repairable Survey Robots and a Distributed Microfactory

**Status:** Design proposal. No bill of materials, manufacturing process, price, hardware qualification or production capability is claimed as completed.  
**Parent proposal:** [ENV-MOBILITY-001 — Aerial–Aquatic Survey Fleets and Regional Outposts](https://github.com/Luminous-Dynamics/symthaea/pull/7870).  
**Related Symthaea owners:** [robotics domain ownership #7650](https://github.com/Luminous-Dynamics/symthaea/issues/7650), [robot platform specialization #6033](https://github.com/Luminous-Dynamics/symthaea/issues/6033), [transport engineering #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244), [physical validation ladder #6254](https://github.com/Luminous-Dynamics/symthaea/issues/6254), [persistent platforms #7343](https://github.com/Luminous-Dynamics/symthaea/issues/7343).  
**Related Mycelix owners:** [AeroCommons #3638](https://github.com/Luminous-Dynamics/mycelix/issues/3638), [Mobility Commons #3708](https://github.com/Luminous-Dynamics/mycelix/issues/3708), manufacturing lifecycle and provenance owners to be mapped before implementation.

## 1. Design thesis: optimize lifetime service, not minimum purchase price

Build robots that are easy to understand, fabricate, inspect, repair, update, and recycle. Minimize *unnecessary* moving mechanisms, proprietary components, custom parts, connector diversity, destructive assembly and inaccessible service points. Do not minimize a count mechanically if doing so makes recovery, stability, reliability or safe stopping worse.

The economic target is **cost per valid observation over service life**, not the cheapest bill of materials:

\[
C_{valid} =
\frac{C_{parts}+C_{fabrication}+C_{labor}+C_{energy}+C_{maintenance}+C_{failed\ missions}+C_{support}}
{N_{valid\ observations}}
\]

Keep cost components separate and publish the numerator and denominator. Count observations only when the quality gate accepts them against the mission's stated measurement requirements. No universal currency target is defensible before a real BOM, geography, duty cycle, source quotations, labor assumptions and reference configuration are frozen.

The factory itself must follow the same rule: prioritize inexpensive fixtures, standard interfaces and excellent quality checks before buying expensive automation that creates a maintenance burden of its own.

## 2. Paradigm shifts for repairability and simplicity

### 2.1 Reduce mechanisms before optimizing the control algorithm

Prefer direct drive where feasible, fixed geometry, passive alignment features, simple linkages, standard bearings and replaceable modules over gearboxes, custom transmissions, multiple servos, complex tilt mechanisms and proprietary gimbals. Every removed mechanism must be checked against mission capability, failure modes, controllability and recovery requirements.

Examples to test, not assume:
- A fixed-wing aircraft with one propulsion motor may be better for broad-area transects when its launch/recovery corridor exists; a multirotor is better for local hovering and constrained recovery. Neither is the universal winner.
- A fixed sensor or passive mounting may beat a powered gimbal when payload orientation and mission profile allow it.
- A surface craft using differential thrust may avoid a separate rudder actuator, but its extra motor, efficiency, redundancy and degraded-mode trade-offs need analysis.
- A simple tethered ROV can be a more accessible first underwater platform than an untethered AUV because it avoids some onboard navigation, acoustic telemetry and recovery complexity. It introduces tether drag, snag and operator burdens instead; evaluate total service cost.
- Do not treat pressure housings, structural joints, battery systems or safety interlocks as places to remove design margin.

**Rule:** count external articulated joints, independent actuators, gears/belts, seals, special tools, connector types and inaccessible fasteners by function. Publish those counts alongside mass, power, payload and qualification evidence.

### 2.2 Standardize interfaces, not whole vehicles

Use an explicit family of interface contracts for:
- mechanical mounting and load paths;
- electrical power rails, protection, grounding/isolation and connector pinout;
- sensor/payload mounting and coordinate frame;
- data transport and message schema;
- thermal/environmental envelope;
- calibration, configuration identity and maintenance events.

A shared interface does not make different vehicles physically equivalent. Aerial, surface, ROV and AUV profiles retain separate load cases, environmental limits, controls and qualification.

Where practical, the same serviceable compute, logging, clock, sensor-acquisition, wiring and payload modules can be used across more than one vehicle. Measure actual commonality and lifecycle benefit; do not invent an elaborate universal connector or force every component into every platform.

### 2.3 Make the likely-to-fail part the easiest part to replace

Draw the service sequence before the CAD design is frozen. Motors, propellers, arms, exposed guards, landing feet, connectors, cable penetrators, sacrificial bumpers, battery modules and sensor pods should each have a deliberate repair strategy appropriate to their platform.

Preferred principles:
- minimal number of fastener families chosen for the real loads and environment;
- captive fasteners where loss is likely;
- direct access to service/check points;
- replaceable wiring harnesses instead of buried solder joints;
- replaceable electronics tray or potted module where a whole sealed module can economically be swapped and correctly tested;
- labeled polarity, ports and connectors with physical keying where feasible;
- no structural adhesive or hidden snap-fit where routine non-destructive access is needed;
- no need to strip the entire vehicle to replace a common wear part;
- downloadable as-built configuration, wiring drawing, test fixture, repair guide and known-failure list for every revision.

A module swap is not a successful repair until post-repair checks establish identity, configuration, correct function and required calibration. “Replace and hope” is not maintenance evidence.

### 2.4 Separate high-cost, high-value, and consumable parts

Use disassembly and lifecycle analysis to decide which components should be:
- repaired to component level;
- replaced as a standardized module;
- refurbished or remanufactured;
- treated as a consumable;
- quarantined or retired.

Do not automatically replace every low-cost part, and do not spend technician hours repairing a part when the full cost/risk of repair exceeds replacement. Preserve recovered modules for diagnosis and qualified refurbishment rather than feeding unknown parts back into service.

### 2.5 Make local repair and open manufacturing first-class

Publish, in preferred editable formats and with appropriate licensing:
- mechanical CAD and drawing exports with tolerances and datums;
- schematics, PCB sources and editable harness drawings;
- bill of materials including acceptable substitutions and prohibited substitutions;
- assembly and inspection work instructions;
- calibration and functional test procedures;
- fixtures, gauges, firmware/configuration manifests and data formats;
- revision history, failures, maintenance intervals and measured production metrics.

A rendered CAD picture, incomplete BOM or nominally “open” source without edit-ready manufacturing files is insufficient. Follow established open-hardware principles and use local/common components where it does not create new performance, safety or supply-chain weaknesses.

## 3. Platform family strategy: start with the simplest sufficient fleet

Do not attempt to make one chassis fly, float and dive. Instead, share well-bounded modules and manufacturing methods.

### A. Repairable close-survey multirotor

Design objective: easy arm, motor, propeller, guard, landing-gear, battery and payload replacement; standard fasteners; clear wiring; accessible flight controller; no powered camera gimbal unless the measured survey task requires it.

Trade-offs to test include motor/prop selection, arm geometry, vibration and image quality, wind envelope, energy use, physical guarding, redundancy and repair stock. Use existing, well-documented flight-controller ecosystems and only claim the safety functions actually configured and tested. A replacement arm or motor must trigger the declared inspections and any required calibration checks.

### B. Efficient broad-area aerial survey

Evaluate conventional fixed-wing before adding VTOL complexity. It may be a better fit where energy per mapped area dominates and launch/recovery conditions allow it. If vertical launch is an actual site requirement, compare the full VTOL mechanism/mass, transition modes, additional actuators, test burden and service cost against a multirotor or a separate launch/recovery aid.

### C. Surface environmental vehicle

Start with a serviceable, stable hull and protected, accessible propulsion modules, not a complex boat full of bespoke mechanisms. Compare single- versus multi-thruster arrangements using maneuverability, energy, redundancy, weeds/entanglement exposure, repair time and safe recovery. The hull, propulsors, sensor pole, battery and compute tray should be independently inspectable.

### D. Underwater platform ladder

Start with a supervised, tethered ROV where it can answer the first science question. Use replaceable thruster modules, accessible electronics, a well-defined wet/dry boundary, strain-relieved cable entries, appropriate corrosion control and documented leak/functional checks. Advance to AUV capabilities only when the mission needs untethered operation and the team can justify added navigation, battery, communications and recovery complexity.

Pressure-bearing parts, seals, underwater connectors and batteries require qualified design and appropriately rated test methods. Do not substitute an unvalidated 3D-printed enclosure or homemade pressure vessel for rated pressure equipment.

### E. Reusable payloads and reference instruments

Build sensor payloads as field-replaceable, independently identified modules when scientific and electrical requirements permit. Each payload needs explicit mass, power, mounting datum, acquisition interface, environmental envelope, calibration state and data schema. Sharing a housing does not establish sensor accuracy or calibration equivalence.

## 4. Manufacturing architecture: a distributed microfactory, not an expensive fully automated plant

The first factory should be a **small, fixture-led, human-operated production cell**. It should be capable of building, inspecting and repairing low volumes consistently, while outsourcing specialized work until demand and quality requirements justify equipment ownership.

### Cell 0 — design and manufacturing engineering

- CAD with editable native files and neutral exports.
- Versioned BOM, drawings, tolerances, approved substitutions and configuration manifest.
- Design-for-assembly and disassembly map.
- Manufacturing router: ordered steps, tools, fixtures, inspection points, and safety requirements.
- Cost model separating purchased parts, fabrication, setup, scrap, labor and inspection.
- Revision/change-impact review before a new configuration is released.

### Cell 1 — frame and mechanical fabrication

Use the lowest-cost process that meets verified geometry, strength, finish and repeatability requirements:
- supplier-cut sheet, tube and plate for early prototypes;
- suitable hand tools, drill press and simple jigs for low-risk operations;
- bench vise, deburring and finishing tools for prepared parts;
- desktop FDM printing for fixtures, drill guides, cable guides and qualified non-critical covers;
- CNC routing/milling where dimensional need and repeated volume justify it;
- turning or specialist machining outsourced for shaft, bearing, precision and load-critical components as appropriate.

Do not purchase every machine up front. For each process, compare the supplier's total cost/lead time with in-house machine cost, operator skill, tooling, dust/fume management, metrology, maintenance and utilization.

Printed jigs may improve repeatability; printed parts are not automatically strong, watertight, fatigue-resistant, UV-resistant or suitable for a pressure boundary. Load-bearing and fluid/pressure parts must be qualified for their actual material/process and loads.

### Cell 2 — electrical and harness assembly

- ESD-aware bench and controlled parts storage appropriate to the electronics.
- Current-limited power supplies, calibrated multimeter, crimp tools and connector-specific gauges.
- Standard harness lengths, connector part numbers, wiring tables and labeled/testable breakouts.
- Repeatable low-voltage continuity, polarity, insulation/isolation and load checks as required by the actual circuit.
- Separate wet-side and dry-side equipment and explicitly controlled grounding/isolation boundaries.

Buy proven controllers, radios, motors, ESCs, batteries, connectors and compute modules initially when they satisfy documented needs. Designing custom PCBs only makes sense after interfaces, volumes, service needs and lifecycle economics are understood. Battery protection, cell chemistry, charging and storage must not be improvised; use suitable validated equipment and qualified procedures.

### Cell 3 — propulsion and actuator test

- Commercially appropriate, mechanically contained motor/propeller test fixture with remote controls and instrumentation.
- Measure electrical input, thrust/force, temperature, vibration, noise where relevant and speed under a documented protocol.
- Use guarded, secured and exclusion-controlled testing; never run an exposed propeller on an improvised open bench.
- Create serial-linked acceptance records so changed motors, propellers, firmware or ESCs are not silently treated as equivalent.

### Cell 4 — sensor, navigation and payload metrology

- Reproducible calibration/check targets and geometry references.
- Known reference instruments appropriate to the measurement.
- Clock and GNSS/time-source checks; declared coordinate datum and sensor-to-body mounting transforms.
- Dark/flat, color/radiometric, water-quality or other sensor-specific checks as appropriate to the measurement.
- Calibration interval, residuals, uncertainty and failed-check disposition recorded explicitly.

### Cell 5 — environmental and ingress screening

- A controlled bench for splash/drip checks, cleaning and drying workflow where appropriate.
- Instrumented temperature, vibration and connector-cycle fixtures only to the level required by the declared operating envelope.
- Rated and professionally operated leak/pressure-test equipment for underwater components. Do not design a DIY high-pressure test chamber into the microfactory plan.
- Defined pass/fail criteria, sensor state before and after tests, and safe quarantine/recovery route.

### Cell 6 — final assembly, acceptance and repair

Every vehicle receives:
1. exact configuration and serial identity;
2. component and substitution check against the approved BOM;
3. fastener, harness and connector inspection;
4. power-on and functional checks;
5. propulsion-disabled interface checks before guarded propulsion tests;
6. sensor calibration/status verification;
7. platform-specific checks under approved procedures;
8. a recorded acceptance result and unresolved limitations;
9. a repair and service history linked to that exact physical article.

A schedule entry or green light on an unvalidated dashboard is not evidence that the inspection occurred or passed.

### Cell 7 — materials and circularity

Separate reusable modules, repairable subassemblies, consumables, damaged batteries, electronic waste, contaminated wet-side parts and retired pressure parts. Track scrap and rejected units. Prefer repair/reuse where quality can be verified; do not reintroduce a safety-critical module whose history or condition is unknown.

## 5. Factory digital thread: design → build → test → repair

Build a minimal versioned digital thread, not a factory-wide AI platform as the first step.

For each revision, bind:
- requirement and design revision;
- editable CAD, drawing and BOM hashes;
- material/process and supplier-lot records where relevant;
- component substitutions and justification;
- tool/fixture identity and calibration status;
- build traveler and operator-recorded steps;
- as-built dimensions and inspection results;
- propulsion, electrical and payload test artifacts;
- software, firmware and parameter versions;
- accepted deviations, unresolved defects and repair history.

A product design, a produced unit and the unit's present operational status are separate identities. A new motor, frame, battery, pressure seal, payload or control configuration may invalidate part of the prior evidence and require explicit retest. Do not overwrite prior failed records or silently inherit qualification from a similar-looking version.

Use established manufacturing interoperability where useful, including STEP AP242 for product definition, QIF for quality measurement, and MTConnect for eligible equipment telemetry. NIST's smart-manufacturing work provides reference patterns for connecting design, fabrication, inspection and lifecycle data. The microfactory need not implement every standard on day one; preserve a documented mapping rather than inventing incompatible identifiers.

Symthaea's role:
- analyze design/mission trade-offs;
- propose fixture or process changes;
- help search the design space;
- compare predicted and measured deviations;
- flag evidence gaps and maintenance patterns.

The authority boundary:
- deterministic checkers decide whether required fields/constraints are satisfied;
- qualified operators approve the manufacturing router and quality disposition;
- calibrated measurement systems provide physical readings;
- independent review and applicable regulatory or professional authorities retain their role;
- no generated recommendation automatically changes a machine, energizes equipment or releases a vehicle.

## 6. Factory cost and scale strategy

### Stage 1 — distributed prototype network

Buy or borrow basic hand tools and measurement equipment; buy commodity modules; outsource laser-cut/CNC/turned parts and specialist PCBs; print fixtures locally where suitable. Build a few units by hand to discover assembly sequence, failures, tool needs and repair times.

### Stage 2 — repeatable bench production

Freeze interfaces and a baseline BOM; introduce modular jigs, harness fixtures, calibration targets, guided work instructions and test data capture. Track labor, defects and rework for each unit. Acquire only the equipment that removes a demonstrated bottleneck or repeated supplier delay.

### Stage 3 — local parts and repair cells

Bring simple repeated fabrication processes in-house after cost and competence review. Maintain a compatibility-qualified supplier/substitution list, critical spares, repair stock and a path for independently manufactured parts to meet published dimensions and tests.

### Stage 4 — semi-automated microfactory

Only automate proven, stable, repetitive work such as dispensing, drilling, harness preparation, test sequencing, inspection capture or material handling when measured return justifies the automation and its safety/maintenance burden. Keep humans able to inspect, override, service and reproduce the process with ordinary tools where practicable.

Do not begin with an expensive autonomous lights-out factory. A low-volume, modular fleet is more likely to benefit first from standardization, process discipline, repairable modules and rigorous test fixtures.

## 7. Benchmark plan and proposed targets

These are **targets to test**, not measured results. Freeze an appropriate existing platform/configuration as the baseline, publish the BOM and actual unit economics, and compare like-for-like performance and safety envelope.

| Dimension | Measurement | Initial target / gate |
|---|---|---|
| Routine serviceability | Time from work order to verified return to service by one trained technician | Replace a common field module in ≤10 minutes; include diagnosis, post-repair checks and tool change separately |
| Tool simplicity | Distinct tools required for 90% of routine repairs | ≤3 common tool types for the selected service profile; safety-critical service may require specialist tools |
| Fastener simplicity | Fastener families in the field-service envelope | ≤2 routine metric fastener families where load/environment calculations allow; don't compromise joint integrity |
| Destructive access | Repairs requiring cut/glue/destructive disassembly | 0 for declared routine-replaceable parts |
| Repair documentation | Service procedures, parts and configuration traceability | 100% of field-replaceable modules have an editable guide, part ID, compatible revision and post-repair test |
| Parts availability | Repairable-module value and lead time | Track local sourcing and the fraction of BOM value serviceable from two qualified sources; set per-market thresholds after supplier census |
| Assembly labor | Recorded person-minutes from kitted components to acceptance-ready unit | Establish 5-unit baseline first; preregister ≥30% reduction at 25 units without lowering acceptance yield |
| First-pass yield | Share passing all defined manufacturing checks without rework | Baseline first; target ≥95% on a frozen process after a stable pilot batch, then investigate every failure |
| Repair success | Repairs returning a unit to valid acceptance without repeat failure in a declared interval | Establish by subsystem and exposure; no single composite reliability claim across air/water platforms |
| Part commonality | Share of parts/subassemblies that are genuinely interchangeable across the chosen fleet | Track count and BOM value; aim for >50% of the *eligible support/data interface modules*, excluding mode-specific physics, propulsion and safety-critical structure unless proven compatible |
| Cost | Full lifecycle cost per valid observation | Target ≥30% improvement over a frozen baseline at equal measurement quality, declared coverage, environment and safety envelope; require real quotes and labor data |
| Manufacturing footprint | Tools, purchased services, setup time, operator skill and floor-space per unit | Publish all five; avoid claiming “local manufacturable” if a single critical proprietary part remains a long-lead dependency |
| Energy productivity | Validated observations per Wh | Compare within each platform/mode and scientific task, not across incomparable missions |
| Waste/circularity | Repair, reuse, scrap and end-of-life mass/value fractions | Mass-balance rejected parts, wear items, electronics and battery disposal; publish unresolved hazardous streams |

### Important interpretation rules

- A 95% first-pass yield is a proposed pilot target, not acceptable evidence to waive critical safety checks. Any failed critical check blocks release.
- The ≤10-minute module-replacement target includes the physical replacement task only if diagnosis or post-repair verification is reported separately; for end-to-end serviceability, record total elapsed time too.
- A <30% cost reduction with higher quality, fewer failures, or much lower lifecycle risk can still be worthwhile. Do not game cost by omitting labor, failed missions, calibration or spares.
- Report each platform family separately; never hide poor underwater performance inside an aerial average.
- Preserve uncertainty and failure distributions; do not claim a reliability percentage without enough trials and a defined exposure model.

## 8. Frozen manufacturing and repair test corpus

Start with a small, human-reviewed baseline: three articles per family for an initial process-learning batch, then expand as failure data warrants. This is a planning quantity, not enough by itself to establish reliability.

Include destructive-looking but non-hazardous seeded defects and repair scenarios:
- motor module replaced but ESC mapping/configuration remains stale;
- left/right or front/rear arm substituted incorrectly;
- harness pinout changed while connector shell remains identical;
- battery chemistry, voltage, connector or rated-current profile differs from baseline;
- sensor payload mounted with changed orientation but old transform retained;
- calibration was due or a check failed;
- printed cover fits nominally but obstructs service or cooling;
- water ingress indicator trips after cleaning/drying;
- pressure boundary or connector has unknown service history and must be quarantined;
- thread, insert or fastener torque check fails;
- supplier substitution changes material, tolerance or process;
- damaged module is reintroduced without acceptance checks;
- design file changes but released router/BOM does not;
- test evidence belongs to a prior hardware or firmware generation;
- an apparently interchangeable part violates a real load/current/environment envelope.

For all critical negative controls, desired outcome is explicit rejection/quarantine, not a low score. A correct repair must trigger only the affected retest obligations based on documented change-impact rules, while preserving prior history.

## 9. Practical first microfactory bill of capabilities

This is a **capability list**, not a purchase order. Selection should follow an actual prototype and supplier quotation exercise.

### Buy/borrow first
- quality hand tools, torque tools where specified, calipers, basic gauges and a documented inspection kit;
- ESD bench tools, current-limited power supply, calibrated multimeter, connector-specific crimp tooling and continuity fixtures;
- safe battery test/charging and storage equipment appropriate to the selected cells;
- a guarded commercial propulsion test setup and force/current/temperature instrumentation;
- work surfaces, labeled storage, kitted parts bins, ventilation and suitable PPE;
- a dedicated computer/workflow for version-controlled CAD, BOM, assembly travelers and measurement capture.

### Add after a demonstrated need
- FDM printer for fixtures/covers where material/process is suitable;
- small drill press, vise, deburring and finishing tools with appropriate guarding/dust controls;
- CNC router/mill or other fabrication equipment after comparing outsourcing, required tolerances, run volume and operator capability;
- small inspection/vision fixtures and controlled lighting where measurement repeatability benefits;
- environmental/vibration test fixtures scoped to the declared envelope;
- local harness-board fixtures and semi-automated continuity testing for repetitive harnesses.

### Initially outsource or use qualified facilities
- high-precision shafts, bearings and difficult structural interfaces;
- safety-critical battery packs and custom electronics that require design/manufacturing controls not yet available;
- pressure-rated housings, underwater pressure testing and specialized connectors;
- accredited or regulated testing and any work needing specialized certification.

A machine is not a capability by itself. Factory readiness is the ability to produce a conforming part repeatedly, measure it, trace it to its design revision and respond when it fails.

## 10. Verification and progression

1. **Source and owner audit:** trace each design claim to the existing physics/manufacturing/sensing/V&V owner. No parallel robotics core.
2. **Digital baseline:** freeze editable CAD, BOM, interface drawings and service analysis for one benign platform and one outpost/service module.
3. **Hand-calculated model and fit checks:** verify quantities, stack-ups, load-path assumptions and interface constraints with explicit units and uncertainty.
4. **Bench prototype:** build one serviceable article; time assembly/disassembly/repair tasks; record all failed fits, rework and missing instructions.
5. **Manufacturing repeatability:** produce a small batch from the same router and compare dimensions, mass, energy, vibration and test outcomes.
6. **Independent checks:** separately written validator checks BOM/identity compatibility, required test completion and prohibited substitutions; include mutation tests.
7. **Supervised physical qualification:** proceed only through the existing physical-validation process with the relevant test site, operators and approvals.
8. **Lifecycle comparison:** compare verified repairs and valid observations across comparable duty cycles and retain failed/aborted runs.

A passing model is not a passing manufactured unit. A passing bench article is not a flight/sea-qualified system. A manufacturing process is not qualified solely because it produced one working prototype.

## 11. Research anchors

These sources support the design principles, not a claim that this proposed factory or vehicle has been tested:

- Open Source Hardware Association, [Open Source Hardware Definition](https://oshwa.org/resources/open-source-hardware-definition/): calls for publicly available source in the preferred modification format and encourages readily available components, standard processes and open tools.
- NASA, [Hardware and Equipment — maintainability](https://www.nasa.gov/reference/9-0-hardware-and-equipment-vol-2/): discusses serviceable modules, access, fewer fastener types and minimum fasteners subject to structural integrity.
- K. Lu et al., [Design for Disassembly: An analytical method for assessing target components for remanufacturing](https://doi.org/10.1016/j.clpl.2025.100113), *Cleaner Production Letters* 9 (2025), 100113: combines product architecture, manual disassembly and component-level life-cycle assessment; highlights minimizing steps and standardizing fasteners.
- NIST, [Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed): provides an example joining CAD/CAM, fabrication, inspection, validation and manufacturing information.
- NIST, [Digital Thread for Smart Manufacturing](https://www.nist.gov/programs-projects/digital-thread-smart-manufacturing): describes lifecycle information integration using standards and feedback from inspection to design.
- Blue Robotics, [BlueROV2 assembly guide](https://bluerobotics.com/learn/bluerov2-assembly/) and [technical reference](https://docs.bluerobotics.com/downloads/): provide accessible assembly/product documentation examples and illustrate the value of revision-specific instructions. These sources do not establish a lowest-cost or universally best ROV.
- European Commission, [Right to repair: new consumer rights](https://commission.europa.eu/news-and-media/news/right-repair-new-consumer-rights-easy-and-attractive-repairs-2026-07-31_en): a current policy signal for accessible repair information and reasonably available parts in product categories covered by EU rules; it is not a claim that these specific drone designs are automatically subject to those rules.

## 12. Exit criteria

Before claiming that a design is “cheap, repairable, or locally manufacturable,” publish:
- a revision-specific editable design package and full BOM with substitution rules;
- a task-based disassembly map and parts-serviceability classification;
- measured assembly, repair, calibration, inspection and rework times;
- supplier quotes and local sourcing assumptions;
- production cost including labor, scrap, tooling and external services;
- first-pass yield and failed-unit dispositions;
- independently verified subsystem and final acceptance test results;
- the exact evidence maturity and operating-envelope limits.

The intended paradigm is a repairable product family plus a distributed, low-capital manufacturing process: local fabrication where it is economical and measurable, commodity modules where they reduce cost and risk, standardized interfaces where compatibility is demonstrated, and no claim of production capability until repeated builds and inspections prove it.
