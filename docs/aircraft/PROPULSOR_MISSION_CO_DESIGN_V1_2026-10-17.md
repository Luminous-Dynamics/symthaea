# Mission-Matched Electric Propulsor Co-Design

**Date:** 2026-10-17  
**Status:** proposed engineering benchmark contract; no candidate has been shown optimal  
**Tracking:** [PROP-OPT-001 #7809](https://github.com/Luminous-Dynamics/symthaea/issues/7809)  
**Compose with:** [fixed-wing aerodynamics and qualification #6249](https://github.com/Luminous-Dynamics/symthaea/issues/6249), [aircraft energy architectures #7646](https://github.com/Luminous-Dynamics/symthaea/issues/7646), [mass-coupled mission model #7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729), [electric mission-energy benchmark #7805](https://github.com/Luminous-Dynamics/symthaea/issues/7805), and [decision-value benchmark #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803).

## 1. Objective: design the best propulsor for a declared mission

There is no universal "best thruster." For a fixed-wing aircraft, open propellers, ducted fans, distributed propulsors and other architectures occupy different operating envelopes. Their ranking depends on flight speed, takeoff distance, climb, cruise, altitude, acoustic constraints, diameter/ground clearance, installed mass, thermal limits, failure tolerance and the airframe's aerodynamic interaction.

This specification defines how Symthaea should compare propulsors on a common mission contract. It does not implement CFD, blade design, a motor electromagnetic solver, flight controls, or a production design.

The system-level question is not "which one gives maximum static thrust?" It is:

> Which feasible propulsion configuration satisfies the complete mission and reserve requirements with the best evidence-supported trade-off among mission energy, installed mass, noise, robustness, maintainability and cost?

## 2. Research-grounded paradigm shifts

### 2.1 Optimize the air-moving system, not the isolated rotor

Propulsor efficiency cannot be separated from its operating point and installed geometry. Diameter, rotational speed, blade pitch, advance ratio, flow distortion and tip Mach matter. The most efficient static-thrust device may not be best in cruise.

NASA's distributed-electric-propulsion review describes potential benefits from boundary-layer ingestion, wake filling, blown-wing lift and alternative propulsor placement, while also documenting integration challenges. Its benefits are aircraft-dependent, not universal.  
Source: https://ntrs.nasa.gov/citations/20180004729

NASA's SCEPTOR work studied an integrated arrangement with small high-lift propellers and larger wingtip cruise propellers. NASA reported a projected wingtip propulsive benefit for that particular configuration; it is a useful design precedent, not a constant efficiency bonus to apply to arbitrary aircraft.  
Source: https://ntrs.nasa.gov/citations/20160010157

### 2.2 Keep propulsive, electrical and mission efficiency distinct

For each flight phase, keep separate quantities for:
- mechanical shaft power;
- motor/controller electrical input and losses;
- thrust and forward speed;
- propulsor/airframe installation losses;
- energy consumed during that phase;
- thermal and transient power constraints.

In forward flight, a commonly useful propulsive-efficiency definition is useful thrust power divided by shaft power, with thrust direction, aircraft velocity and reference plane stated consistently. Electrical-to-useful-power efficiency must include motor/controller losses. Neither metric alone captures the entire mission.

For static operation, forward speed is zero, so the expression thrust multiplied by forward speed cannot be used as a positive static useful-power measure. Use static thrust per electrical input power under specified conditions instead, and report it separately from cruise performance.

### 2.3 Use disk-area bounds as a sanity check, not a design answer

For an ideal actuator disk in static, incompressible, uniform flow, momentum theory gives induced velocity

    v_i = sqrt(T / (2 rho A))

and ideal induced power

    P_i = T * v_i = T^(3/2) / sqrt(2 rho A)

where T is thrust, rho is air density, and A is total actuator-disk area. This is an ideal lower bound on induced air power for the idealized static problem. It excludes blade profile and swirl losses, motor/controller losses, installation effects, compressibility, and structural constraints. It is not a cruise propeller formula.

It immediately demonstrates why "maximum thrust per kW" without diameter is an incomplete target: at fixed static thrust, ideal induced power decreases as disk area increases. Real designs eventually encounter diameter, mass, clearance, RPM, tip-speed, noise, interaction and manufacturability constraints.

### 2.4 Make distributed propulsion earn its complexity

Distributed propulsors can improve a coupled design through wing blowing, vortex interactions or noise placement. They also add motors, inverters, cables, mounts, cooling, failure interfaces and maintenance burden. The benchmark must charge all these to the aircraft-level mass, drag, energy, cost and reliability accounting.

The same rule applies to a ducted fan: the duct and supporting structure are part of the design. Never compare a ducted fan's thrust to an open propeller while omitting duct mass/drag from the system-level result.

## 3. Candidate families

The same mission family and requirements should be used to compare:

1. **Large, slower-turning open propeller**, including fixed or variable pitch where the controls/model support it. A plausible baseline for efficient low-subsonic fixed-wing operation when diameter is available.
2. **Compact/ducted electric fan**, when diameter, guarding, acoustic directionality or higher-speed integration justifies the duct cost.
3. **Distributed propulsors**, only where aircraft/propulsor coupling is explicitly represented; compare against a same-airframe baseline.
4. **Separate takeoff/high-lift and cruise propulsors**, with retraction/folding/drag penalties accounted for if used.
5. **Hybrid-driven propulsor architectures**, compared using the energy-system owner in #7646. Do not assume that a small turbine, piston engine, fuel cell, generator and battery are free sources of electrical power; include their installed mass, thermal rejection, conversion losses, startup/transient behavior and maintenance.

These are study families, not recommendations for an aircraft until its speed regime, dimensions, mission and constraints are frozen.

## 4. Input and evidence contract

Every candidate must identify:

- configuration ID, propulsor type, geometry/version, diameter, blade count, pitch/RPM schedule and operating envelope;
- total actuator disk area and whether disks are overlapping/interacting;
- motor, controller, gearbox and generator efficiency maps when applicable, each with revision/digest and validity domain;
- phase-specific density, temperature, forward speed, required thrust/power and duration;
- torque, RPM, power, tip Mach, peak duration and thermal constraints;
- aircraft installation location, clearance, mounting and geometry references;
- acoustic data or an explicit missing-evidence state, including measurement distance, operating state and protocol;
- mass boundary covering rotor/blades, duct, motor, controller, inverter, gearbox, wiring, pylon, cooling and support structure as applicable;
- uncertainty semantics, calibration status, model limitations and provenance;
- failure modes / failed-propulsor assumptions where redundancy is claimed;
- exact input/config/model digests and terminal evaluation disposition.

Digest consistency establishes artifact identity, not truth. A simulation map does not establish physical accuracy outside its validity domain. Missing efficiency maps, coefficients or acoustic measurements must not silently become favorable defaults.

## 5. Qualification sequence

### P0 — independent analytical oracle
Implement deterministic static actuator-disk test cases with independently calculated results. Test units, monotonicity in thrust/density/area, invalid inputs, floating-point overflow and the exact operating-domain label. A production caller must be unable to mistake the returned ideal induced-power bound for measured electrical power.

### P1 — component map comparison
Compare shaft/electrical power, thrust, torque, RPM, efficiency and temperature against independent dynamometer or wind-tunnel points. Separate calibration data from held-out validation cases.

NASA's Electric Aircraft Systems Technology Development project explicitly targets a database of propeller/motor/controller configurations and acoustic/energy use, with measurement of thrust, torque, RPM, efficiency and noise. This is a useful benchmark-measurement precedent.  
Source: https://techport.nasa.gov/projects/4475

### P2 — whole-mission comparison
Run all candidate types across the same declared mission family. Include takeoff, climb, cruise, descent and go-around/reserve as applicable. Generate candidate-specific airframe demand, and charge every installation's mass and drag.

### P3 — coupling to energy architecture
Feed each candidate's electrical/shaft demand into the existing mass-coupled sizing and energy-architecture paths. Compare battery-electric, liquid-fuel hybrid and hydrogen-electric architectures only with declared system boundaries, efficiencies, tanks, thermal systems, fuel-cell/generator masses and lifecycle-fuel assumptions.

### P4 — held-out and physical correlation
Use independent test conditions and document discrepancies, uncertainty and model limits. Numerical reproduction, model calibration, held-out validation and physical qualification are separate evidence classes.

## 6. Report a vector of metrics

Do not hide trade-offs in a single aggregate score. Report:

- static thrust per electrical kW for clearly stated static conditions;
- thrust and power across speed / advance-ratio envelope;
- phase and total mission electrical/shaft energy;
- installed propulsion-system mass and system specific power;
- propulsive/electrical efficiency by phase;
- peak/continuous power and torque/RPM/tip-Mach margins;
- specified acoustic levels/spectra and uncertainty;
- temperature, vibration and maintenance/service evidence where available;
- sensitivity to density, temperature, payload, manufacturing variation and model bias;
- failed/incomplete evaluations and feasibility outcomes;
- prediction error, held-out error, cost and run time.

A ranking counts as decisive only if its advantage exceeds the uncertainty and numerical/test resolution of the comparison.

## 7. Proposed acceptance targets (not measured)

- 100% agreement with independent analytical answers on the P0 static-oracle corpus.
- Zero use of static actuator-disk power as a cruise performance prediction.
- 100% accounting of required mission phases, constraints, missing evidence and terminal states.
- Exact determinism for identical pinned deterministic inputs.
- Zero silent favorable defaults for unprovided geometry, maps, mass, efficiency or noise.
- Detect every preregistered case where the added mass/drag of a duct, mount, extra motor/controller or distributed architecture reverses a nominal component-level ranking.
- Compare variable-pitch/RPM schedules against fixed schedules at equal constraints and evaluation budgets; claimed gains must exceed declared uncertainty.
- Report mission-energy, mass, acoustic and robustness Pareto results separately.
- Any distributed-propulsion benefit must survive coupled airframe analysis and held-out data; no generic percentage benefit is assumed.
- No "best propulsor" claim without a frozen mission, system boundary, hard constraints and explicit stakeholder preferences.

## 8. Energy carrier is a separate coupled choice

Do not conflate fuel specific energy with usable aircraft-system energy:

- **Battery-electric:** high conversion efficiency and good transient power, but battery-pack specific energy, reserve, cooling, protection, aging and installed mass constrain mission/range.
- **Renewable liquid hydrocarbon fuel:** potentially compatible with existing thermal engines if the particular fuel and engine are approved; retains combustion and lifecycle emissions. "SAF" pathways have very different life-cycle carbon intensities. Jet SAF is not automatically a valid fuel for a piston/avgas engine.
- **Hydrogen fuel cell + buffer battery:** hydrogen has high energy per unit fuel mass, but the tank, valves, insulation/compression, fuel cell, cooling, buffer battery, ventilation/leak detection and integration must all be counted. Liquid hydrogen's volumetric storage challenge is substantial, and hydrogen is not inherently safer than batteries or conventional fuel.
- **Combustion hybrid:** may extend endurance and buffer peak loads, but duplicates conversion paths and can add engine/generator mass, maintenance and emissions. A parallel hybrid may preserve a mechanical efficient cruise path; a series hybrid offers architectural flexibility but converts power through more stages. Compare both only within well-defined architectures.

For a practical near-term light-aircraft study, carry forward at least three fair baselines: battery-electric for a short mission, an approved renewable/drop-in liquid-fuel-compatible architecture for missions that need high onboard energy, and a hydrogen fuel-cell/battery concept as a longer-horizon research alternative. The requirements and approved powertrain determine applicability; this is not a blanket safety endorsement.

## 9. Scope boundary

This contract defines benchmarks and evidence contracts only. It is not a propeller/blade design release, CFD result, motor design, flight-control law, aircraft performance certification or airworthiness determination. Reuse #6249, #7646 and #7729; do not create a second universal aircraft solver.
