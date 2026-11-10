# Thermal-management policy benchmark contract

**Programme:** AIRROAD-TMS-POLICY-001 (#11467)  
**Dependencies:** coupled solver (#11411), architecture co-design (#11428), mission kernel (#11348), thermal overlay (#11409), independent oracle (#11427)  
**Status:** research contract. Targets below are not achieved performance, validated controls, airworthiness or certification evidence.

## 1. Purpose and ownership

This document defines the mission-policy and operating-schedule benchmark. It does not create another battery model, thermal network, aircraft solver or physical controller.

- The aircraft mission kernel owns mission demand and propulsion/battery mission accounting.
- The coupled solver owns time-resolved electrical/thermal plant analysis.
- The independent oracle owns its deliberately small reference problem.
- Architecture profiles own installed cooling hardware, flow paths, heat rejection and physical boundaries.
- This contract owns the choice and comparison of thermal operating policies across flight and ground phases.

Every candidate sees the same frozen mission, plant, architecture, initial state, environment and hard constraints. A policy may not make a mission feasible by silently reducing thrust, payload, speed, climb, reserve or required auxiliary demand.

## 2. Energy and mass boundaries

Never add quantities with incompatible units. In particular, a raw sum of kg and kWh is not a valid objective.

| Quantity | Definition | Accounting rule |
|---|---|---|
| Propulsion terminal energy (flight) | Propulsion electrical energy at the declared battery/bus terminal boundary during flight | Included in flight battery-terminal energy |
| TMS terminal energy (flight) | Onboard pump, fan, compressor, heater, valve/control and other declared TMS loads during flight | Included once in flight battery-terminal energy |
| Cabin/ECS terminal energy (flight) | Cabin/environmental-control electrical loads during flight | Included once in flight battery-terminal energy |
| Battery-terminal energy (flight) | Sum of all declared onboard electrical loads at the selected battery-terminal boundary during flight | Do not add its subcomponents again |
| Battery chemical energy | Chemical energy withdrawn from the battery model | Includes battery internal loss according to the selected model |
| TMS terminal energy (ground) | TMS energy drawn from the battery while on the ground | Included in ground battery-terminal energy |
| Shore/grid energy | Grid-side energy for preconditioning, charging and ground thermal reset | Kept separate; charger conversion losses are separately identified |
| Operating-day energy | Declared battery-terminal and shore-energy ledger across the operating day | Publish the boundary and conversion efficiencies |

A TMS share must name its denominator, such as TMS terminal energy during flight divided by total battery-terminal energy during flight. This is not interchangeable with a whole-operating-day share.

TMS electrical consumption is not automatically heat deposited in the battery or any other particular thermal node. Allocate pump, fan, compressor, heater, motor and power-electronics losses to their physical nodes and rejection paths. Battery I²R heat is computed from total battery demand in the coupled model; do not add TMS energy to battery ohmic heat a second time.

Keep installed mass, volume, centre-of-gravity effect, packaging/service envelope, cooling drag, pressure drop, coolant/air inventory, structural provisions, payload displacement and lifecycle burden as separately reported metrics. Any scalar mission consequence must define its physical/economic conversion, units, weights and sensitivity.

## 3. Thermal headroom representation

Do not compress a multi-node thermal system into an unexplained scalar reserve.

For each node and scenario, report:
- current temperature and evidence-supported operating domain;
- remaining temperature margin to warning and hard-limit boundaries;
- predicted time-to-warning and time-to-limit for each declared future load/ambient scenario;
- uncertainty interval, model applicability and OOD status;
- heat capacity and available rejection path, if characterized;
- relevant thermal gradients and sensor/model resolution.

For phase-change material, report usable latent-heat state or melt fraction, remaining buffer capacity, regeneration requirement and time-to-regenerate. For a coolant loop, report flow, inlet/outlet temperature, pressure drop, pump operating point and heat-exchanger boundary when available.

An aggregate thermal-headroom metric is permitted only when its horizon, scenario set, node weighting, model fidelity and uncertainty treatment are explicit. It is a decision-support quantity, not a safety limit or proof of future safety.

## 4. Policy families and reference baselines

Compare policy families in this order:

1. **Passive/minimum-active baseline:** physically admissible passive cooling or the minimum required active system.
2. **Fixed thermostat:** fixed setpoint and deadband, including declared hysteresis and actuator limits.
3. **Rule-based phase schedule:** predeclared modes across ground, start, climb, cruise, descent, landing, dwell and charging.
4. **Mission feed-forward schedule:** uses a frozen mission and forecast, with no hidden online adaptation.
5. **Deterministic MPC/NMPC:** constrained finite-horizon policy using the coupled plant.
6. **Robust/uncertainty-aware MPC:** only after deterministic plant and uncertainty boundaries have been independently qualified.

An offline optimizer may establish a candidate lower bound for a frozen model, but it is not a real-time controller and is not proof of a global optimum. Learned/RL policies are deferred until transparent baselines, hard-constraint negative controls and independent comparisons exist.

For each policy, freeze and report decision variables, sampling interval, actuator bounds/delays/efficiency, fault states, setpoints/deadbands/hysteresis, fallback rules, prediction horizon, terminal cost/constraints, solver tolerances and iteration cap, runtime, deterministic replay, mission/forecast inputs available at decision time, and every override or non-PASS outcome.

## 5. Mission and operating-day corpus

Use common profiles across all architectures and policies:

- fixed-wing CTOL climb, cruise, descent and ground turnaround;
- repeated short-hop sorties with limited cooling/PCM regeneration time;
- high-power climb followed by long cruise and hot soak;
- hot ambient, high altitude and reduced heat-rejection capability;
- cold-soak starts with battery warm-up and restricted initial capability;
- aged battery with capacity fade and increased resistance;
- charging concurrent with cabin conditioning and thermal reset;
- degraded fan/pump/compressor, restricted flow or unavailable cooling branches where modelled;
- forecast perturbations for ambient temperature, density, wind, mission duration and load;
- eVTOL hover, transition and cruise as a separate class with its own load histories.

Each case freezes payload, aircraft configuration, mission, speed/altitude, initial SOC/SOH and component temperatures, ambient/wind, reserve, cooling topology, forecast information, ground-energy availability and turnaround schedule. Holdout data must be split by physical asset/configuration/mission, not random adjacent time samples.

## 6. Objectives and hard constraints

Use a Pareto vector, not an opaque weighted sum.

Report separately:
- battery-terminal energy per flight and sortie;
- shore/grid energy per operating day;
- installed TMS mass and volume;
- cooling drag and propulsion-energy consequence, where supported;
- payload delivered per sortie and per operating day;
- turnaround time and dispatch availability;
- battery ageing exposure under a declared degradation model;
- maintainability, inspection and service burden;
- lifecycle cost per successful payload-km or payload-hour, only where inputs are defensible.

Hard constraints are non-compensatory: required payload and frozen mission performance; aircraft mass, centre-of-gravity and packaging; battery voltage/current/power/SOC/reserve and component limits; temperatures and gradients; coolant/air flow and pressure boundaries; and declared degraded/fault-state constraints.

A candidate violating a hard constraint is infeasible regardless of energy or cost benefit. No weighted score may trade a hard-limit violation for a lower energy result. If a one-number score is needed, use an interpretable common unit such as lifecycle cost or payload-km delivered, disclose conversion assumptions and weights, and publish sensitivity beside the underlying Pareto vector.

## 7. Benchmark targets

These are internal research goals, not standards, achieved results, certification criteria or universal aircraft requirements.

### Tier 0 — numerical and accounting correctness
- 100% agreement with independent deterministic analytical/reference cases within preregistered tolerances.
- Zero false-feasible outcomes in the frozen negative-control corpus; publish counts and test-domain coverage.
- Every OOD, non-convergent, voltage/current/power/reserve/thermal-limit case yields an explicit non-PASS result with diagnostics.
- Deterministic replay for identical frozen configuration, model build, inputs and evidence manifest.
- Energy and thermal conservation residuals reported per substep and mission.

### Tier 1 — physical component correlation

Once independently instrumented data exist and measurement uncertainty supports the targets:
- component-temperature RMSE below 2 K and peak absolute error below 5 K;
- empirical 95% prediction-interval coverage on an asset/configuration-held-out test set;
- component-loss reconstruction error below 3% for independently metered battery, inverter and motor boundaries;
- separate peak-temperature time, time-to-warning/limit, gradient and cooldown reporting.

### Tier 2 — integrated mission correlation

Once an independently instrumented integrated powertrain/TMS rig exists:
- TMS and total bus-energy accounting closure within 0.5% against independent meter boundaries;
- mission-energy bias within ±3% and RMSE below 5% on a preregistered holdout, if measurement uncertainty supports these limits;
- zero observed false-safe points in the declared test envelope, reported as 0/N with confidence bounds and coverage;
- false-derate rate and its mission/throughput cost reported beside false-safe outcomes;
- at least 100× real-time offline simulation for a named representative mission on a declared commodity CPU, preserving accuracy and case coverage.

### Tier 3 — policy and lifecycle benefit

Against frozen matched thermostat and rule-based baselines:
- stretch goal of at least 10% reduction in a preregistered mission TMS-energy metric, or at least 10% improvement in payload throughput/turnaround, without degrading hard constraints, payload or reserve;
- policy ranking stable across held-out hot/cold, aged-battery, repeated-sortie and degraded-cooling cases, or explicit abstention where ranking is not robust;
- report battery ageing, dispatch availability and lifecycle cost only when supported by qualified physical data and a declared model.

A result of 0/N observed false-safe events is not proof of zero real-world risk. State sample count, envelope, instrumentation uncertainty and confidence bound.

## 8. Evidence and model qualification

Every measurement/model input records source, evidence class, physical asset/configuration, serial/lot where applicable, calibration identity, units, thermal boundary, valid operating domain, uncertainty, raw-data digest and model applicability.

Keep physical measurement, manufacturer-reported values, published research, simulation, synthetic fixture and target values as separate evidence classes. Do not promote simulated or synthetic results to physical validation.

Missing values, stale configuration, unsupported operating points and absent calibration remain UNKNOWN/OOD; they are not silently assigned nominal values.

## 9. Safety and authority boundary

This benchmark is an offline engineering and design-decision tool. It is not an aircraft thermal controller, does not command hardware and cannot override BMS, inverter protection, contactors, fuses, thermal cutouts or other independently engineered safety protections. Physical rig operation, HIL, ground prototype and flight testing require their own approved procedures, responsible authority and safety controls.

A Tier 0 pass establishes only the stated software/numerical contract. Higher tiers require independent physical measurements and review. No benchmark pass establishes battery safety, thermal-runaway containment, aircraft performance, airworthiness or certification.

## 10. References and competitive context

- [Nakayama et al. (2026), mission-level integrated electrical/thermal eVTOL simulation](https://doi.org/10.1016/j.applthermaleng.2026.132253): TMS energy exceeded 20% of mission energy in the investigated cases; study-specific.
- [2026 whole-eVTOL thermal modelling framework](https://link.springer.com/article/10.1007/s13272-026-01032-3): reports a best-case simulated TMS-energy reduction; not a flight-demonstrated aircraft result.
- [Lu et al. (2026), topology-optimized PCM battery cooling](https://doi.org/10.1016/j.applthermaleng.2026.129829): transient, mission-specific thermal buffering; results remain study-specific.
- [Xu et al. (2026), energy-efficient MPC for BEV powertrains](https://doi.org/10.1016/j.applthermaleng.2026.129897): useful automotive policy-method baseline, not an aircraft comparator.
- [Koeln et al. (2020), hierarchical control of aircraft electro-thermal systems](https://doi.org/10.1109/TCST.2019.2905221): aircraft electro-thermal MPC and hardware-in-the-loop work predates this programme; MPC itself is not our novelty.
- [NASA TFAWS 2026 battery-module thermal-fluid work](https://ntrs.nasa.gov/citations/20260008248): useful instrumentation/correlation direction; preserve the distinction between preliminary validated operating points and higher target discharge rates.
- [SAE AIR5744](https://saemobilus.sae.org/standards/air5744-aircraft-thermal-management-system-engineering): information report reaffirmed in 2026, supporting platform-level thermal-management engineering.

The differentiator sought here is the reproducible, independently checked chain from physical evidence and a coupled plant model to policy × architecture × mission decisions and lifecycle outcomes—not the isolated use of MPC, PCM or a thermal network.
