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
- Compare predicted and measured efficiency, temperature, vibration, and power.
- Calibrate uncertainty on a held-out set.

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
