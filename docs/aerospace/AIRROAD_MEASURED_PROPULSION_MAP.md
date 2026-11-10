# AIRROAD measured propulsion map and evidence contract

**Programme:** AIRROAD-MEASURED-PROPULSION-001  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11529  
**Status:** research contract and proposed targets; no physical test data or flight qualification is claimed  
**Scope:** fixed-wing and rotorcraft propulsion measurement interfaces; vehicle-specific physics remain with their current owners

This document is the versioned repository form of the engineering contract. The tracking issue owns implementation sequencing and evidence collection. All thresholds below are research targets conditional on measurement uncertainty, not achieved performance or certification requirements.

---

## Parent and composition

- Electric rotorcraft architecture / mission programme: [AIRROAD-ROTOR-001 #11490](https://github.com/Luminous-Dynamics/symthaea/issues/11490).
- Fixed-wing propulsor co-design: [PROP-OPT-001 #7809](https://github.com/Luminous-Dynamics/symthaea/issues/7809).
- Independent rotor-shaft-to-DC-bus reference: [AIRROAD-DRIVE-001 #11509](https://github.com/Luminous-Dynamics/symthaea/issues/11509), draft implementation [PR #11511](https://github.com/Luminous-Dynamics/symthaea/pull/11511).
- Mission electro-thermal feedback: [#11411](https://github.com/Luminous-Dynamics/symthaea/issues/11411); TMS architecture and policy: [#11428](https://github.com/Luminous-Dynamics/symthaea/issues/11428), [#11467](https://github.com/Luminous-Dynamics/symthaea/issues/11467).
- Cross-fidelity aircraft electro-thermal validation: [#11445](https://github.com/Luminous-Dynamics/symthaea/issues/11445).
- Aircraft energy architecture: [#7646](https://github.com/Luminous-Dynamics/symthaea/issues/7646).

## Purpose

Create a versioned, evidence-qualified measurement and map contract connecting propulsor aerodynamic output to rotor/propeller shaft power, transmission, motor, inverter and DC-bus input. The contract must work across explicitly separate fixed-wing and rotorcraft configurations while preserving each domain's physics and existing owners.

This issue owns **measurement semantics, test-data provenance, map identification, uncertainty, held-out validation and integration interfaces**. It does not create another aircraft solver, battery model, thermal solver, rotor aerodynamic solver, motor electromagnetic solver, flight controller or certification authority.

## Critical correction exposed by the v0.1 drive oracle

The current independent drive oracle closes a scalar power chain from rotor torque/RPM through explicitly supplied efficiencies. That is a sound reference for its declared boundary, but it does not yet determine a real motor operating point behind a reduction gearbox.

A gearbox changes speed and torque as well as dissipating power. Define and record the ratio explicitly, for example:

- g = omega_motor / omega_rotor for the declared motoring direction;
- omega_motor = g * omega_rotor;
- P_rotor = Q_rotor * omega_rotor;
- P_motor_shaft_in = P_rotor / eta_transmission;
- Q_motor = P_motor_shaft_in / omega_motor.

The ratio convention, direction, sign convention and measurement locations must be frozen. A scalar efficiency without kinematics is insufficient to check motor torque-speed limits, field weakening, current demand or map validity. Efficiency must be identified across operating conditions; it is not a universal nominal constant.

## Paradigm shift

Move from **component nameplate specifications plus assumed efficiencies** to **configuration-specific, experimentally identified propulsion maps with uncertainty and explicit applicability domains**.

Then move from **testing every point uniformly** to **decision-directed experimental design**: prioritize the next safe, authorized measurement where it is most likely to change a feasible architecture ranking or reduce consequential uncertainty. The test-selection algorithm recommends; a qualified test authority approves. This is not autonomous rig control.

The evidence chain is:

  frozen aircraft / propulsion configuration
  -> calibrated, synchronized raw observations
  -> independent physical-boundary reconstruction
  -> identified component and integrated maps
  -> uncertainty + domain / OOD assessment
  -> held-out validation
  -> mission and architecture decision
  -> versioned evidence manifest and review

## Required physical boundaries

Keep every measurement boundary explicit and non-overlapping:

1. **Aerodynamic output:** thrust (and force/moment where applicable), airspeed, density, rotor/propeller RPM, blade pitch/control state, torque, vibration and acoustic observations. Compute shaft power independently as Q × omega; do not substitute electrical input for shaft power.
2. **Rotor / propeller shaft:** torque and speed at a named physical shaft plane. If there are multiple rotors, identify each channel and aggregation rule.
3. **Transmission:** input/output shaft torque and speed where accessible, gear ratio, direction, temperature, lubricant state and measured loss. If only one side is metered, label inferred quantities and propagate uncertainty.
4. **Motor:** shaft torque/speed and three-phase terminal voltage/current or a validated power-analyser boundary; winding, stator, rotor and coolant temperatures where instrumented. Record continuous/peak limits and map domain.
5. **Inverter / drive:** DC input voltage/current and motor-side electrical output at declared terminals, switching/control configuration, cooling state and component temperatures. Keep inverter and motor electrical boundaries separate so their losses are not merged or double-counted.
6. **DC distribution:** busbar, cable, contactor and connector voltage/current or resistance boundaries where included. Keep these distinct from inverter conversion loss.
7. **Battery / source:** battery-terminal voltage/current, SOC/SOH and temperature are supplied by the battery owner; do not infer battery electrochemistry from the drive map.
8. **Thermal management:** pump/fan/compressor/heater electrical loads, coolant flow, inlet/outlet temperature, pressure drop and heat rejection are separate measured loads and thermal paths. TMS power is included once in the declared mission bus ledger; it is not automatically heat deposited in the battery.
9. **Installed system:** motor, inverter, transmission, cables, cooling hardware, mounts, sensors and required containment mass/volume are all reported separately and as an explicitly bounded installed package.

The protocol must state which boundaries are directly measured, inferred, estimated or UNKNOWN. No energy loss may be assigned twice or hidden in a residual.

## Data and provenance contract

Every test run and derived map must be reproducible from:

- immutable aircraft, propulsor, gearbox, motor, inverter, cooling and software/firmware configuration identifiers;
- physical asset serial/lot identity where applicable and configuration/revision digests;
- sensor/channel ID, unit, range, sampling rate, clock source/synchronization uncertainty and measurement location;
- calibration identity, traceability, calibration date/status and uncertainty (including covariance where material);
- raw-data digest, acquisition format, test-procedure revision, operator/authority and environmental conditions;
- operating point, test direction, warm-up/soak state, duration, transient/steady classification and any excluded samples with reason;
- derived-data transformation version, source-data digests, fitting method, parameter covariance and applicability domain;
- model, software build, schema and benchmark corpus revisions.

Preserve raw observations. Do not overwrite raw data with filtered values. Every filtering, resampling, alignment, correction and outlier exclusion must be versioned and replayable.

Use SI units internally. Missing calibration, unsynchronized channels, stale configuration, impossible sign conventions, unsupported temperatures or map extrapolation produce UNKNOWN/OOD or non-PASS, never a guessed nominal value.

## Map identification and validation

- Map motor torque, speed, electrical frequency, bus voltage, current, temperature and relevant control state over the actual intended operating envelope.
- Map inverter losses at its declared DC and motor terminals; do not derive inverter efficiency by subtracting motor loss from an unmetered combined measurement.
- Map transmission losses against input/output torque, speed, direction, temperature and gear ratio where the design allows.
- Map propulsor thrust, torque, RPM and acoustic/vibration behavior against airspeed/advance ratio, density, pitch/control state and configuration. Static hover results must not be reused as cruise efficiency.
- Distinguish steady-state maps from transient capability, thermal-soak limits, peak-duration limits and repeated-sortie duty cycles.
- Include motoring and regenerative quadrants only when the hardware supports them and the test campaign separately characterizes reverse power flow.
- Freeze calibration/training conditions and independent holdout conditions before fitting. Split by asset/configuration and operating campaign, not adjacent samples from the same run.
- Report interpolation method, local data density, uncertainty and validity boundary. Never extrapolate outside characterized support and label the result feasible.
- Validate both component maps and the integrated chain against independent metering. Report signed bias, RMSE, peak error, repeatability, prediction-interval coverage and energy-closure residuals.
- Test altitude/pressure, hot/cold soak, humidity, thermal ageing and insulation/partial-discharge conditions in separate authorized campaigns where relevant; do not claim that an efficiency map qualifies insulation or HV safety.

## Experimental-design paradigm

Build a decision-value test selector after the deterministic benchmark exists. Its input is a frozen candidate set, current evidence and uncertainty, approved test envelope, cost/time and the mission decisions under consideration. Its output is a ranked **proposal** of test points and expected information/decision value, with reasons and uncertainty. It must:

- prefer tests that could change the feasible set, architecture ranking or safety margin;
- avoid redundant points when they add little information;
- never recommend points outside an independently approved rig/test envelope;
- expose the assumptions used to estimate information gain and decision impact;
- retain exploration/holdout separation and prevent test-selection leakage into validation;
- abstain when evidence is too weak to rank candidates.

This is a research decision-support tool only. It does not actuate a rig, bypass interlocks or authorize physical tests.

## Benchmark families

1. **Metrology and analytical closure:** torque/speed, voltage/current and known-load references; channel synchronization; calibration uncertainty; energy and power closure.
2. **Component maps:** motor, inverter and transmission loss/efficiency, continuous/peak capability, temperature dependence and map-domain coverage.
3. **Propulsor maps:** thrust/torque/RPM/power, advance ratio, disk loading where relevant, rotor interaction, vibration and acoustic spectra.
4. **Integrated chain:** DC-bus terminal to useful thrust/propulsive work, including declared wiring, conversion, transmission and cooling boundaries.
5. **Mission transfer:** same configuration maps evaluated on frozen hover, vertical climb, transition, cruise, descent, reserve and go-around profiles as applicable.
6. **Operating-day/lifecycle:** repeated sorties, charging/preconditioning, thermal regeneration, ageing, turnaround, maintainability and dispatch availability.

Compare battery-electric, series-hybrid and other justified candidates only on matched mission, payload, reserve, environment, installed-mass and evidence boundaries. Do not assume battery-electric or hybrid is categorically superior.

## Proposed acceptance targets

These are internal research targets, not achieved measurements, safety limits, certification requirements or universal industry thresholds.

### Tier 0 — data and numerical integrity
- 100% of derived values trace to raw channels, calibration records, configuration and transformations.
- 100% known-answer fixtures agree with independent analytical references within preregistered tolerances.
- Zero silent extrapolation, stale-config acceptance, unit ambiguity or missing-calibration promotion.
- Zero false-feasible outcomes in the frozen negative-control corpus.
- Deterministic replay from exact raw-data digest, configuration digest, schema and transformation revision.

### Tier 1 — component correlation
- Report uncertainty-normalized bias/RMSE for each component map and each operating region.
- Target component efficiency/loss prediction error <= 2% of reading where instrument uncertainty and the test standard support it; otherwise report the achievable uncertainty-limited threshold rather than claim a pass.
- Empirical prediction-interval coverage close to its preregistered nominal coverage on configuration-held-out data.
- Separate continuous capability, peak duration, thermal-soak and repeated-duty limits.

### Tier 2 — integrated chain correlation
- Target independent energy-accounting closure <= 0.5% where the combined metering uncertainty supports it.
- Target integrated mission energy bias within ±3% and RMSE < 5% on a preregistered holdout, contingent on measurement uncertainty and matched operating envelope.
- Zero observed false-safe/false-feasible points in the declared test corpus, reported as 0/N with confidence bounds and domain coverage.
- Publish false-derate rate and throughput cost alongside false-feasible outcomes.

### Tier 3 — mission and architecture advantage
- Stretch goal: at least 10% improvement in a preregistered useful-mission metric (payload-distance per operating-day, mission energy per payload-distance, or lifecycle cost per delivered payload-distance) against a matched baseline, with all hard constraints satisfied.
- Preserve the Pareto vector: mission energy, reserve, payload, installed mass/volume, thermal margin, acoustic exposure, turnaround, availability and lifecycle cost.
- Require held-out mission/configuration evidence and uncertainty-aware ranking; otherwise report no winner.

No 0/N result proves zero real-world risk. Every reported error must include sample count, coverage, confidence interval, instrumentation uncertainty and model applicability.

## Physical test ladder and safety boundary

Start with calibrated component metrology and independently instrumented motor/inverter/transmission benches; then a propulsor thrust stand or wind-tunnel campaign; then an integrated drive/thermal rig. Use approved facility procedures, qualified operators, containment, interlocks and independent emergency shutdowns. The software is offline engineering evidence and cannot command hardware or replace BMS, inverter protection, contactors, fuses, test-facility authority or regulatory approval. No flight test is implied by this issue.

## Industry reference points

- NASA's Electric Aircraft Testbed (NEAT) demonstrates the value of reconfigurable, integrated ground testing under realistic electrical and thermal conditions, including altitude chambers, bidirectional DC capability, batteries, gearboxes and support cooling: https://www.nasa.gov/eap-labs-and-testbeds/neat/.
- NASA's Electric Aircraft Systems Technology Development programme explicitly included integrated stand measurements of thrust, torque, RPM, efficiency, noise, motor/controller efficiency and total energy: https://techport.nasa.gov/projects/4475.
- NASA's HEMM programme lists 1.4 MW, 16 kW/kg and 99% as development targets for a partially superconducting motor, not achieved product or rotorcraft mission metrics: https://www.nasa.gov/eap-technology/electric-machines/hemm/.
- NASA's full-scale eVTOL acoustic project identifies full-scale rotor-rotor and rotor-structure noise validation as an active need: https://techport.nasa.gov/projects/125528.
- SAE AIR5744 (reaffirmed 2026-01-21) encourages platform-level thermal management rather than independent optimization of air vehicle, propulsion and avionics: https://saemobilus.sae.org/standards/air5744-aircraft-thermal-management-system-engineering.
- EUROCAE ED-344 (August 2026) addresses HV insulation ageing under temperature, reduced pressure, humidity and other environmental stresses: https://www.eurocae.net/product/ed-344-process-specification-test-guidelines-for-assessment-of-electrical-insulation-material-ageing-in-a-high-voltage-system/.
- SAE AIR6127 is an information report on higher aerospace voltages and partial discharge/component interactions; it is not a personnel-safety standard or a complete propulsion-system compliance basis: https://saemobilus.sae.org/standards/air6127-managing-higher-voltages-aerospace-electrical-systems.

The differentiator is not a novel motor-efficiency formula or an unsupported claim of beating NASA's hardware. It is a reproducible, uncertainty-aware bridge from physical test evidence to propulsion maps, complete mission decisions and an auditable engineering digital thread.

---

## Immediate implementation order

1. Freeze a canonical channel and measurement-boundary schema with SI units, timestamp semantics, calibration identity, uncertainty and raw-data digests.
2. Add gearbox ratio and both-side shaft operating-point semantics to the independent drive-chain reference before treating a map as a motor operating point.
3. Define a small, fully synthetic analytical fixture that proves the kinematic mapping, component loss ownership and energy closure; do not present it as measured hardware data.
4. Add map-domain validation that rejects stale configuration, missing calibration, unsupported interpolation and extrapolation.
5. Publish a first instrumented test campaign plan only after a qualified facility, test article, instrumentation and responsible test authority are identified.
6. Connect qualified maps to the mission electro-thermal owners; keep measured map fitting, independent holdout validation and aircraft-level mission decisions as separate evidence stages.

## Promotion rule

A document/schema milestone is not a physical characterization milestone. A passing analytical fixture is not a component test. A component map is not integrated aircraft validation. Integrated ground evidence is not flight qualification. Any consumer that cannot establish the exact configuration, map domain, calibration state and evidence digest must return UNKNOWN/OOD or non-PASS instead of silently using a nominal value.
