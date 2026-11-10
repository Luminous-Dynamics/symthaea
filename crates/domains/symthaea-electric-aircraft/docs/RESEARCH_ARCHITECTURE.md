# Luminous Electric Aviation — research architecture

**Status:** proposed engineering architecture; not an aircraft design approval, certification basis, or claim of airworthiness.

## Thesis: compile evidence into aircraft engineering

The differentiator should not be an AI that draws an aircraft or an optimizer that returns a low-drag shape. It should be a traceable engineering system that connects mission intent, geometry, models, components, manufacturing, physical tests, operations, maintenance, and independent verification.

Every design decision should carry:
- the requirement and mission assumptions it addresses;
- the configuration and source data used;
- units, coordinate frame, operating condition, and valid range;
- model/solver identity and version;
- input and output digests;
- uncertainty and sensitivity;
- verification level and unresolved assumptions;
- independent review/approval where required.

A missing source, unknown validity domain, stale configuration, or failed check must remain visible as unknown/incomplete/failed. It must not be silently promoted to a design fact.

## System responsibilities

### Symthaea — design-space intelligence

Use Symthaea for bounded design exploration, surrogate modelling, sensitivity analysis, system identification, anomaly detection, test selection, and maintenance forecasting. All proposed changes must be replayable and evaluated against frozen baselines. Learned models may advise or identify candidates; they do not set flight-critical authority or bypass validated limits.

### Symtropy — physical and systems simulation

Use Symtropy as the simulation orchestration and deterministic replay layer. Couple appropriate, independently validated specialist tools for:
- external aerodynamics and propeller interaction (CFD / wind-tunnel correlation);
- structural loads, fatigue, and aeroelasticity (FEA);
- electrical networks, switching, protection, and power electronics;
- battery electro-thermal behaviour;
- rigid-body flight dynamics and control-system hardware-in-loop.

A general-purpose rigid-body engine or reduced-order aircraft kernel is not a substitute for validated CFD, FEA, battery, or power-electronics models.

### Sol-Atlas — spatial and configuration intelligence

Use Sol-Atlas for aircraft geometry, coordinate frames, meshes, spatial relationships, mass-property stations, component envelopes, routing clearances, airfield/terrain constraints, and spatial provenance. Geometry revisions must map to the exact analysis and manufacturing configurations that used them.

### Mycelix — cross-organisation trust and lifecycle

Use Mycelix for supplier identity and qualification, part/material provenance, serialised BOMs, engineering change records, manufacturing inspection evidence, maintenance histories, and evidence exchange among organisations. Mycelix records and verifies evidence; it does not make an engineering authority decision or self-authorise a journal, release, or flight.

## Digital-thread lifecycle

1. **Mission definition:** payload, route, range, airspeed, altitude, weather, runway, reserves, turnaround, charging availability, dispatch rate, noise, and operator constraints.
2. **Requirements and hazards:** measurable requirements, operating envelope, failure conditions, safety objectives, and verification methods.
3. **Architecture trades:** conventional single-prop, distributed propulsion, hybrid-electric range extender, and (later) VTOL architectures. Compare all against the same mission and assumptions.
4. **Geometry and mass properties:** versioned CAD, material definitions, centre of gravity, inertia, component access, wire/cooling routing, and maintainability.
5. **Multi-fidelity analysis:** analytical screening, specialist simulation, uncertainty propagation, and independent cross-checks.
6. **Design freeze:** configuration-controlled airframe, propulsion, energy storage, electrical distribution, software, and supplier parts.
7. **Verification:** bench tests, environmental tests, component qualification, structural tests, hardware-in-loop, ground tests, and staged flight tests.
8. **Release and continued airworthiness:** approved configuration, operating limitations, maintenance programme, service bulletins, life limits, incident evidence, and controlled design changes.

A result must identify which exact design revision it applies to. No test report should be attached to a mutable or ambiguous configuration.

## First development target

The first product target should be a conventional fixed-wing electric cargo / utility demonstrator, with a mission selected from a real operator or logistics use case. Start with a bench-validated propulsion-energy module and a subscale unmanned integration demonstrator before deciding whether a full-size aircraft is justified.

The subscale aircraft is useful for integration, control, telemetry, propulsion and operational-learning questions. Its results cannot simply be scaled into full-size structural, aerodynamic, battery-safety, or certification claims; Reynolds number, thermal scaling, energy density, and failure consequences change with scale.

Do not freeze a nominal aircraft mass, range, payload, or battery chemistry until the mission and supplier evidence justify them. The 640 kg test scenario in v0.1 is synthetic software test data, not a selected design or competitor comparison.

## Architecture candidates to compare

| Candidate | Potential advantage | Main proof burden |
|---|---|---|
| Conventional single-prop fixed-wing | Lowest architectural complexity; strong baseline for comparison | Propeller efficiency, cooling, battery mass, single-point propulsion failure and glide/landing envelope |
| Distributed electric propulsion | Potential integration of propulsors with wing aerodynamics; flexible placement | Net aircraft-level benefit after wiring, inverters, mass, drag, thermal management, control and common-cause faults |
| Hybrid-electric fixed-wing | Extends mission flexibility where battery-only energy is insufficient | Generator mass/efficiency, fuel system, thermal/acoustic effects, dual-propulsion failures, lifecycle emissions and certification scope |
| eVTOL | Runway-independent access and vertical operations | Hover power, rotor/propulsor-out cases, transition, thermal transients, redundancy, noise and more complex certification |

Distributed propulsion, hybrid-electric systems, and VTOL are hypotheses to evaluate, not preferred answers in advance.

## Paradigm shifts worth testing

1. **Mission-first co-design:** optimise airframe, propulsor, battery, thermal system, payload, reserves, and charging together. Do not optimise airframe drag while leaving battery and mission assumptions fixed.
2. **Evidence-compiled engineering:** turn every requirement into a model/test/evidence obligation and every release into a reproducible, independently reviewable package.
3. **Uncertainty-aware design:** optimise not just the nominal result but robust performance across weather, altitude, payload, battery ageing, manufacturing tolerances, sensor error, and component degradation.
4. **Safety as an architectural input:** isolate battery strings and propulsion channels where justified by fault analysis; design containment, venting, isolation, detection, and safe landing from the start. Redundancy must be evaluated for independence and common-cause failure, not counted by component quantity.
5. **Lifetime economics:** optimise energy per payload-kilometre and cost per payload-kilometre including battery degradation/replacement, charging losses, maintenance, turnaround, availability, infrastructure, and end-of-life handling.
6. **Closed-loop fleet learning:** turn field observations into versioned model updates, test campaigns, inspection requirements, and controlled design changes. A fleet anomaly must not automatically mutate aircraft software or hardware.
7. **Interoperable infrastructure:** design chargers, energy management, telemetry, and maintenance interfaces for more than one aircraft vendor while keeping each operator's data and authority under its control.

## Competitive position and route beyond current practice

The field is not waiting for electric propulsion. Luminous must not claim to be ahead of companies that already have flight-tested systems, certified components, production infrastructure, and regulatory programmes.

### Current reference positions

- **Pipistrel / Textron eAviation:** the Velis Electro is an EASA type-certified light-electric trainer. Its published configuration provides a disciplined certification and operations baseline: 24.8 kWh nominal battery system, two liquid-cooled packs, charging times, 50-minute endurance plus VFR reserve, a published initial 500-hour battery limit, and 2,000-hour motor TBO. It shows that certification and local-flight training operations are achievable, while battery throughput and turn time remain important economic constraints.
- **BETA Technologies:** a strong integrated-aircraft-and-infrastructure competitor. Its reported strategy links battery packs, motors, avionics, aircraft, charging products and airport deployment. It uses five common modular packs on CX300; the company reports its packs may need replacement every 12–24 months depending on use. Its shared CX300/A250 components simplify some manufacturing and evidence reuse, but IEEE notes that some shared geometry adds weight and drag to the fixed-wing aircraft. That is a genuine commonality-versus-mission-optimality trade, not a design rule to copy.
- **Safran:** a useful example of electric propulsion moving through the established engine certification pathway. The ENGINeUS 100 B1's EASA type certificate shows why the Luminous programme should consider certified supplier components before attempting to design every component internally.
- **NASA / OpenMDAO research:** current multidisciplinary propulsion studies join thermodynamic component models and aircraft-level optimization across conventional, turboelectric and turbogenerator architectures. We should compete through rigorous mission-level coupling and validation, not by claiming a new optimizer is superior without matched tests.

### How Luminous should differentiate

1. **Out-validate before out-performing.** Publish the test protocol, input data, uncertainty, failed cases and configuration hash. A smaller aircraft with a reproducible, correlated model is more credible than a spectacular uncorrelated range claim.
2. **Optimize installed-system and lifecycle outcomes.** Include battery packs, cooling, structural reinforcement, wiring, protection, reserves, charging infrastructure, replacement cadence, maintenance and downtime. Compare Wh/payload-km and cost/payload-km under identical missions.
3. **Use common modules where they create net value.** Share batteries, avionics, software interfaces, charger interfaces and manufacturing processes across aircraft families when the quantified benefit exceeds the aerodynamic, structural, mass, drag and certification penalty. Do not force one airframe to serve CTOL and VTOL if commonality damages either mission.
4. **Make uncertainty and evidence executable.** A design point outside a measured map domain is incomplete; a missing source is unknown; a failed test remains failed. The new v0.2 characterization API begins this approach but does not yet verify artifact bytes or model multidimensional performance.
5. **Exploit operating conditions as a design input.** The reference mission should cover hot ambient conditions, high density altitude, runway quality, local grid capacity, remote logistics, charging access and maintenance supply chains—not only ISA sea-level flight.
6. **Build a component ecosystem before an aircraft factory.** Start with validated battery/propulsion modules, test services, engineering software and partner integration. A full airframe programme should follow demonstrated operator demand, supplier access, safety evidence, certification resources and funding.

### Quantitative comparison protocol

For each public competitor or candidate, store values as separate evidence classes: certified type data, physical measurement, manufacturer claim, published research, simulation, or Luminous research target. Every value must also carry the configuration, mass boundary, mission, operating conditions, source timestamp and uncertainty. Missing values remain missing. Do not infer usable battery energy from nominal capacity, continuous motor power from peak power, or battery life from a single calendar-time estimate.

## First implementation sequence

### A0 — software kernel and analytical oracle
- Maintain a small deterministic mission-energy kernel.
- Add a separately implemented calculation oracle and property tests.
- Freeze synthetic mission inputs and model versions.
- Record every unsupported assumption and invalid operating region.

### A1 — evidence-backed reference aircraft
- Select one published, type-certificated light-electric reference configuration.
- Obtain traceable aircraft, propulsion, and energy-storage data rather than inferring missing battery details.
- Reconcile mission energy, power, mass, reserves, and charging data.
- Mark all unavailable source values as unknown.

### A2 — propulsion bench
- Characterise motor, inverter, propeller, cooling, sensors, wiring, and protection over a controlled operating map.
- Replace the v0.2 one-dimensional curves with measured multidimensional maps and explicit validity conditions.
- Compare predicted and measured efficiency, temperature, vibration, torque, thrust and power.
- Calibrate uncertainty on a held-out set; reject out-of-domain operating points.

### A3 — battery bench and safety
- Measure capacity, OCV, DC resistance, thermal response, ageing, and power capability over SOC and temperature.
- Test cell and pack faults, propagation, isolation, venting, contactors, insulation monitoring, and emergency shutdown with appropriate qualified facilities.
- Do not use the v0.1 lumped thermal node as a battery safety argument.

### A4 — integrated subscale vehicle
- Integrate the selected propulsion and energy subsystem with a controlled unmanned demonstrator.
- Use a separate deterministic flight-control/safety architecture and independent flight-test approvals.
- Correlate simulation against measured flight data; preserve failed runs and uncertainty.

### A5 — full-size concept decision
Proceed only when an operator mission, supplier chain, energy model, safety concept, certification path, manufacturing plan, and funding case are all evidenced. A promising simulation alone is insufficient.

## Benchmark principles

- Freeze the reference configuration and mission before comparison.
- Separate certified, measured, company-reported, simulated, and aspirational values.
- Compare equivalent payload, range, speed, reserve, weather, altitude, state of health, turnaround, and operating environment.
- Report specific energy at both cell and pack level; never substitute cell values for pack values.
- Report Wh/payload-km, cost/payload-km, power margins, noise, dispatch reliability, component life, charge time, and maintenance burden.
- Use multiple scenarios and independent holdout data. Publish bias, RMSE, calibration, coverage, failures, and incomplete cases.
- Do not claim superiority over state of the art until the same benchmark has been independently reproduced.

## Safety and certification boundary

Engage the South African Civil Aviation Authority early on the proposed aircraft category and intended operation. SACAA's Type Acceptance Certificate pathway is for acceptance of a foreign type certificate and is applied for by that foreign type-certificate holder; it is not a shortcut for certifying a new Luminous design. A novel type requires an authority-agreed certification basis, design organisation / production pathway as applicable, and a compliance plan. International FAA/EASA material is valuable for technical comparison but does not replace SACAA requirements or authority decisions.

No AI component, digital twin, simulation score, cryptographic record, or successful software test establishes airworthiness. Flight authority remains with qualified pilots/approved deterministic systems and the independent safety architecture.
