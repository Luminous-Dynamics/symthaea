# Symthaea Multiphysics Reasoning and Robotics Twin Validation Plan V1

**Status:** Research/design proposal; documentation only  
**Date:** 2026-10-13  
**Companion Symtropy contract:** ROBOTICS_MULTIPHYSICS_DIGITAL_TWIN_WORLD_CONTRACT_V0_1.md  
**Companion Symtropy PR:** https://github.com/Luminous-Dynamics/symtropy/pull/1586  
**Rock-mechanics roadmap:** https://github.com/Luminous-Dynamics/symthaea/pull/7438

## 1. Purpose

Symthaea should be able to evaluate every supported robotics platform in a shared Symtropy digital-twin world without pretending all robots, sensors, materials or physics have the same dynamics.

Symthaea's job is to reason over evidence and uncertainty, decide what to measure next, compare candidate tasks, select a policy, track discrepancies and preserve safety/authority boundaries. Symtropy's job is to host the canonical replayable world and represented robot/tool physics. Domain-specific solvers own the physical equations they actually implement and validate.

This plan defines an extension path for hard-rock methods (mechanical, acoustic/ultrasonic, thermal, microwave, electrical-discharge/plasma and hydraulic), but does not promote any method to a physical actuator or claim current hardware capability. These methods differ physically and require separate input/output contracts and validation.

## 2. Current starting point

The current public Symthaea subterranean domain already contains:
- a stratified geology model and lookahead;
- a deterministic reduced-order boring/scout plant;
- actuator and hazard arbitration separate from the learned controller;
- causal command/response attribution that can report ambiguous and contradicted effects;
- sensor quality, delayed observation, team mapping, return-route and temporal assurance concepts;
- draft work identifying external sensor/plant interfaces, scenario curriculum and independent measurements.

The current geology/material profile is useful for fast scenarios but normalized hardness/abrasiveness values are not dimensional material mechanics. Its granite label does not establish a real granite constitutive model, and basalt is not currently a defined material class in the inspected geology source. The current simulation therefore remains the low-cost reference tier until further calibration.

Preserve the existing deterministic plant for controller and fault campaigns. Introduce new physics and high-fidelity world coupling additively behind versioned interfaces.

## 3. Evidence planes for the full digital twin

Every interaction must preserve these distinct objects:

1. **Scenario declaration:** synthetic geometry/materials, boundary conditions, seed and planned disturbances.
2. **Solver state:** a numerical result produced by an explicit solver/model/profile.
3. **Sensor output:** simulated sensor response from the solver/world plus declared latency/noise/failure model.
4. **Symthaea belief:** inferred material, environment, hazard and robot state, including uncertainty and contradictions.
5. **Command and prediction:** the chosen candidate action and its expected result distribution.
6. **Physical observation:** calibrated, timestamped data from a real instrument or controlled trial.
7. **Evidence/qualification claim:** what specific observation or test actually establishes for one subject, property, model generation and operating envelope.

A synthetic solver frame cannot be upgraded into independent physical evidence just because Symthaea consumes it. A sensor simulation and the state simulator may share a model error; their shared lineage must be explicit to prevent circular validation.

## 4. Physics-modality registry for hard-rock engineering

A modality registry should describe what an action or sensor can actually do, with a typed profile and current evidence state.

| Modality | Distinct use cases | Required model inputs/outputs | First use in Symthaea | Promotion boundary |
|---|---|---|---|---|
| Mechanical rotary/percussive | Tool/rock removal and robot motion | Tool geometry, torque, thrust, feed, impact, wear, loads, removed volume | Calibrated reduced-order cutting response and plan comparison | Held-out cutting/drilling measurement |
| Acoustic emission (AE) | Observe transient activity potentially associated with cracking | Source/receiver geometry, wave speed, attenuation, sensor coupling, bandwidth, noise, sample timing | Diagnose simulated observation quality and choose probes | Independent specimen measurements; avoid using generated AE as proof of its own fracture model |
| Active ultrasound | Pulse-echo, interface/crack/elastic-wave measurement | Wave propagation, frequency, attenuation, boundary/reflection model, transducer response | Active measurement selection with an uncertainty report | Reference specimens and geometry-specific calibration |
| Ultrasonic-assisted/percussive actuation | Excite the tool-rock interaction | Frequency, amplitude, phase, tool/contact model, energy, material condition | Offline experiment profile / reduced-order ablation | Controlled, instrumented cutting tests |
| Thermal spallation / flame or plasma jet | Heat-induced stress, spalling and possibly melt removal | Heat flux, temperature-dependent properties, conductivity/capacity, stress/fracture response, water/cooling state | Offline thermal scenario and energy/cut trade-off | Matched thermal/rock removal experiment and heat-affected-zone measurements |
| Microwave-assisted heating | Electromagnetic deposition and thermal damage | Frequency, field distribution, dielectric properties, moisture/mineralogy, exposure, heat transport | Offline hypothesis/benchmark comparison | Measured field/temperature/material response; no generic strength-reduction constant |
| Electrical discharge / electro-pulse | Breakdown channel and dynamic electro-thermo-mechanical damage | Electric field/conductivity, breakdown assumptions, pulse waveform, energy, coupled mechanical response | Offline research and published-benchmark reproduction | Specialized solver and lab/containment review before physical actuator authorization |
| Hydraulic/fluid assistance | Circulation, cooling, pressure/erosion or fluid-driven fracture | Pressure/flow, fluid properties, boundaries, saturation, permeability, cavitation if represented | Compare support/flush/energy trade-offs | Independent flow/load and material-removal validation |
| Camera/lidar/IMU/force/torque/thermal/pressure/gas | Observe geometry, motion, loads and environment | Calibration, frame, timing, noise, occlusion, saturation, bandwidth and failure modes | Common multi-sensor observation interface | Sensor-specific calibration plus independent physical reference |

### Rule

Do not treat AE sensing as equivalent to ultrasonic actuation; do not treat plasma jet heating as equivalent to electric-pulse breakdown; do not treat microwave energy deposition as a generic temperature input. Every modality must state which physical state it owns, which fields it reads, which outputs it computes, and what is omitted.

## 5. Multi-modal active inference without physical overclaim

Symthaea can use active inference and other planning mechanisms to select among candidate actions such as inspect, probe, rotate, advance, cool, flush, retreat, or request an alternate solver result—but only when the platform profile declares those actions available and the action passes independent constraints.

For an observation/action choice, score a candidate using a declared objective that may include:
- expected uncertainty reduction on a decision-relevant property;
- mission progress and route coverage;
- energy and remaining return reserve;
- tool wear and consumable budgets;
- thermal, flood, gas and structural hazard state;
- communication/recovery constraints;
- observation latency and solver uncertainty;
- consequences of incorrect inference.

Information gain is not a sufficient reason to take a dangerous measurement. The physical safety/authority arbiter remains external to the cognition score. If a critical sensor or solver is missing, stale, contradictory or out-of-domain, Symthaea must preserve that epistemic state and restrict affected plans rather than hallucinate a nominal physical result.

## 6. Canonical world and platform profiles

Every robotics platform should be testable against one shared Symtropy scenario contract, with project-specific profiles. A Symthaea platform profile should reference:
- exact robot/tool geometry, kinematic graph, payload and actuator envelope;
- supported actions, sensors, calibration and known failure modes;
- power source, energy/thermal limits and return reserve;
- environment classes, coordinate frame and map representation;
- supported physics modules and fidelity tiers for each claim;
- controller frequency, estimator, command interface and safety boundary;
- unsupported conditions and explicit no-claim states;
- scenarios, expected invariants, metrics, exact model/source/data digests and evidence tier.

Shared scenarios should cover nominal operation, tool wear/jam, actuator saturation, sensor bias/dropout/latency, terrain/tunnel uncertainty, abrupt material transitions, thermal load, water/gas hazards, communication partition, localization drift, stale solver results, solver timeout/non-convergence, and recovery/emergency stop.

Symthaea should not implement a second divergent scenario truth when Symtropy already owns the world. It should consume the exact scenario identity and output provenance-bound beliefs, policy proposals and evaluation records.

## 7. Cross-project contracts

### Symtropy world interface

Consumes: validated platform profile, scenario/world manifest, initial world generation, solver profiles, observation stream and world clock.

Produces: exact world snapshot/version, simulated state, sensor outputs with lineage, contact/action records, and explicit solver failures.

### Symthaea reasoning interface

Consumes: observations plus timing, calibration, source lineage, uncertainty, world/model generation and scenario identity.

Produces: typed hypotheses/beliefs, candidate actions, reasons/expected outcomes, requested extra observations, expected value/uncertainty trade-off, and restricted/unknown states.

### Safety/actuation interface

Consumes: current physical measurements, reviewed operating envelope, current restrictions, operator authorization, candidate command and relevant model validity.

Produces: accepted/restricted/rejected disposition and the command actually passed downstream. Cognition may propose, not mint physical authority.

### Evaluation interface

Consumes: prediction, action, exact world/model generation and independent outcome.

Produces: per-observable residuals, uncertainty coverage, attribution/ambiguity, safety interventions, model drift and a qualification decision limited to the tested profile. It cannot convert simulated-state agreement into physical validation.

## 8. Validation and falsification plan

### Phase M0 — Contract only

Write schemas and model claim records. Ensure dimensions/units, calibration, frame, time, provenance and physical-state ownership are explicit. No new physical modality is promoted.

### Phase M1 — Shared deterministic world

Run platform profiles through the same Symtropy scenario IDs and seed manifests. Prove stable replay within the declared same-environment numerical contract, clear unsupported-state reporting, and preservation of negative tests.

### Phase M2 — Acoustic sensing MVP

Implement a low-order AE/ultrasound observation model with explicit geometry, sampling, attenuation, noise, latency, clipping and dropouts. Freeze synthetic expectations. The output is a simulated observation, not a fracture truth label.

### Phase M3 — Mechanical + thermal baseline

Add calibrated tool/rock response and an energy/temperature balance. Compare conventional versus thermally assisted action only on supported input ranges and label unmeasured properties unknown. Do not include plasma or microwave as real hardware action types until the required actuator and solver are implemented.

### Phase M4 — One published high-fidelity benchmark

Select one exact reference case for thermal mechanics or fracture. Pin solver build, data, parameters, boundary conditions, numerical settings and output comparison. A benchmark PASS means only that the declared case was reproduced within its predeclared tolerance.

### Phase M5 — Experimental modality lane

Assess microwave, electro-pulse, plasma jet and ultrasonic assistance separately. For each, require a feasible model, accessible/reproducible reference data, validated observable, equipment and environmental constraints, a documented energy budget, and reviewed safety case. Do not merge their performance claims.

### Phase M6 — Closed-loop composition and external metrology

Connect Symtropy robot/tool dynamics, Symthaea planning, sensor models and one validated specialist response. Use independent force/torque, temperature, removed-volume and acoustic measurements. Freeze train/calibration versus held-out specimens/sites before reporting results.

### Phase M7 — All-robot qualification

Run cross-platform scenario packs. Report every platform's supported, restricted, unsupported and failed cases. A missing profile, missing physical state or failed solver must not disappear from the denominator.

## 9. Required measures

Report per exact platform/material/tool/environment:
- penetration and removed-volume error;
- force/torque and energy residual;
- temperature error and heat-affected-zone uncertainty where measured;
- wear estimate error and interval coverage;
- AE/ultrasound event timing and detection metrics against independent references;
- pose and mapping error where surveyed truth is available;
- mission completion, return-reserve violations, collision/contact events and recovery success;
- sensor/solver failures detected, false alarms and missed hazards;
- proportion of cases unknown/out-of-domain;
- exact runtime, source/model/parameter/input/output digests.

Use predeclared tolerances with a justified source. Preserve per-case tails and failure examples. Mean accuracy alone is insufficient for safety-sensitive robotics.

## 10. Safety and physical authority

- Learned policy, HDC confidence, FEP free energy, Phi or another cognition metric does not establish physical correctness.
- Model outputs never bypass emergency stop, operator restrictions, consent requirements or hazard interlocks.
- High-energy tools, plasma, microwave, electrical discharge and thermal treatment stay in offline simulation/research profiles until actuation hardware, containment, electrical/thermal safety, sensor coverage, independent stop behavior and qualified review exist.
- Out-of-domain or non-converged solver output cannot increase the permitted operating envelope.
- Independent measured outcomes must not be replaced by a replay of the same simulator.

## 11. Research references

- Ultrasonic-assisted rock-breaking review, *Energies* (2022): https://doi.org/10.3390/en15228394
- Ultrasonic percussive drilling dynamics in hard rocks, *Journal of Sound and Vibration* (2005): https://doi.org/10.1016/j.jsv.2003.12.045
- Plasma-jet rock-breaking thermal/removal mechanism in granite, *Petroleum Science* (2025): https://doi.org/10.1016/j.petsci.2024.12.017
- Simulating electropulse fracture of granitic rock, *International Journal of Rock Mechanics and Mining Sciences* (2020): https://doi.org/10.1016/j.ijrmms.2020.104238
- Combined high-voltage electropulse and mechanical drilling model, *Geoenergy Science and Engineering* (2026): https://doi.org/10.1016/j.geoen.2025.214128
- Microwave-assisted underground excavation tests, including granite and basalt: https://doi.org/10.1016/j.jrmge.2015.10.004
- Thermal-spallation hard-rock field test: https://doi.org/10.1016/j.petrol.2020.107005
- OpenGeoSys THMC and geomechanics: https://www.opengeosys.org/stable/
- OpenGeoSys phase-field fracture benchmarks: https://www.opengeosys.org/stable/docs/benchmarks/phase-field/
- Symtropy multi-physics digital-twin world contract (proposal): https://github.com/Luminous-Dynamics/symtropy/blob/research/subterranean-rock-physics-contract-v1/docs/canon/ROBOTICS_MULTIPHYSICS_DIGITAL_TWIN_WORLD_CONTRACT_V0_1.md

The literature demonstrates research activity and selected experiments, not universal benefits or an operationally validated model. Data/license access and the match between each reference and our target use case must be evaluated independently.

## 12. Non-claims

This plan does not add a physics solver, implement real plasma/acoustic/microwave tool actuation, validate current geology properties, or certify Symthaea/Symtropy as field-ready. It defines how those capabilities can be introduced incrementally, measured, rejected when unsupported, and composed without blurring cognition, simulated physics, sensor evidence and physical authority.
