# Electric aircraft architecture trade study

**Status:** decision framework and research hypotheses; no aircraft configuration selected.  
**Rule:** do not assign scores until a named mission, frozen assumptions, comparable subsystem data and a reviewed scoring method exist.

## Why architecture selection is the first optimization

A fixed-wing CTOL aircraft, distributed-electric-propulsion aircraft, hybrid-electric aircraft and eVTOL may all use batteries, motors and power electronics, but their energy and safety envelopes are not interchangeable. A common comparison must preserve each architecture's distinct operating phases.

The optimisation target is not maximum motor efficiency or battery Wh/kg. It is feasible mission delivery at the lowest lifecycle cost and risk while satisfying every aircraft, reserve, environmental, structural, safety and certification constraint.

## Required mission manifest

Before any candidate is ranked, freeze:

- mission purpose and operator;
- useful payload mass, volume, geometry and centre-of-gravity range;
- crew, passenger count or cargo handling;
- route distance, diversion/alternate policy and reserves;
- takeoff/landing runway, surface, obstacle and abort constraints;
- cruise speed, altitude, climb/descent profile and time-on-ground;
- ambient temperature, pressure, density altitude, wind and icing conditions;
- mission frequency, annual utilisation, dispatch target and turnaround;
- charging power, grid availability, site footprint and operating energy price;
- maintenance capability, local supply chain, parts lead time and permitted downtime;
- noise, emissions and community constraints;
- intended aircraft category, operating rules and certification authority.

Missing values remain explicit unknowns. Do not replace them with a default that silently favours one architecture.

## Candidates

| Architecture | Primary hypothesis | Dominant proof obligations | Current recommendation |
|---|---|---|---|
| Conventional single-propeller fixed-wing electric | Simplest aircraft-level energy and integration baseline; fewer conversion and control channels | Range/payload/reserves, motor/propeller map, pack mass, glide/landing, thermal management, single propulsion failure | Baseline candidate |
| Distributed electric propulsion fixed-wing | Propulsor placement may improve lift distribution, control authority, high-lift performance or airframe integration | Net benefit after extra motors/inverters/wiring/mounts/cooling/drag/mass, aero-propulsive coupling, common-cause faults and control complexity | Research challenger, not assumed winner |
| Hybrid-electric fixed-wing | Liquid-fuel energy storage can extend mission range while electric machines provide dispatch/flexibility benefits | Complete fuel/generator/motor/battery mass, conversion losses, thermal rejection, emissions, acoustic impact, engine-out and electrical failure cases | Long-range challenger if battery-only mission fails |
| eVTOL | Removes runway dependency and may open point-to-point missions | Hover energy, transition corridor, rotor/propulsor failure cases, crashworthiness, noise, thermal peaks, landing-site infrastructure, more complex certification | Defer until propulsion and energy modules are qualified |

### Distributed propulsion: prove net aircraft benefit

A distributed architecture should not win because its local wing section or propulsor looks efficient. It must improve the whole aircraft under the same payload and mission after accounting for:
- added motor, inverter and mounting mass;
- wiring mass, voltage drop, insulation and EMI;
- nacelle/propulsor drag and interference;
- cooling flow and thermal rejection drag;
- control electronics and fault-isolation equipment;
- blade/propulsor failure, bird/foreign-object damage and common-cause loss;
- manufacturing complexity, maintainability and certification evidence.

Only a measured or validated net aircraft-level improvement justifies additional complexity.

### Hybrid-electric: compare the full energy chain

Compare fuel lower-heating-value energy, generator output, electrical conversion losses, motor/propeller efficiency, thermal management, battery buffering, fuel and battery reserves, emissions and servicing. Do not compare a hybrid's fuel-tank energy with a battery's rated energy without conversion efficiency, usable-energy and installed-mass boundaries.

### eVTOL: treat hover as its own mission class

Hover power and transition are not small extensions to fixed-wing cruise. The concept must model vertical lift, transition, reserve hover capability, propulsor-out response, thermal transients, acoustic exposure and landing-site constraints. Do not borrow fixed-wing range assumptions.

## Objective functions

At minimum, evaluate the following as separate outputs rather than collapsing them into one opaque score:

1. **Mission feasibility:** payload delivered, route and reserve completion, runway/site constraints, and operating envelope.
2. **Energy efficiency:** grid-to-battery, battery-to-shaft, shaft-to-useful-thrust, and end-to-end Wh per payload-kilometre.
3. **Mass efficiency:** useful payload fraction and installed energy-system mass fraction.
4. **Economics:** lifecycle cost per payload-kilometre including battery replacement, charger/grid cost, maintenance labour, downtime and financing.
5. **Operational capability:** dispatch reliability, turnaround, maintenance hours per flight hour and route availability.
6. **Environmental performance:** noise at defined observer locations, lifecycle emissions boundary and material/repairability profile.
7. **Safety and evidence:** hazard closure, independence/common-cause evidence, traceability and certification compliance status.

A Pareto frontier is preferable to one weighted scalar because aircraft operators have different mission priorities. Keep hard safety, reserve, MTOW and certification constraints outside any weighted performance score.

## Decision gates

### Gate 0 — Mission validity
An operator need, payload and route are specified. The mission is repeatable and the reserve policy is explicit.

### Gate 1 — Energy feasibility
A conservative mass and energy loop converges with credible pack/system properties, measured or source-bound maps, environmental conditions and reserves. The design must retain margin after uncertainty and battery ageing.

### Gate 2 — Component feasibility
Motor/inverter/propeller and battery/thermal modules are tested over required operating domains. Data includes calibrated sensors, raw traces, uncertainty and configuration identity.

### Gate 3 — Aircraft integration feasibility
CAD mass properties, structural analysis, thermal rejection, wiring, maintenance access and failure isolation are reconciled against the actual design configuration.

### Gate 4 — Subscale demonstrator
Ground integration and approved test flying establish subsystem integration and data correlation. Subscale tests do not establish full-scale strength, battery safety or certification.

### Gate 5 — Full-scale programme decision
Proceed only with a credible safety architecture, certification plan, supply chain, manufacturing path, operator commitment, test capability and funding case.

## SOTA comparison rules

- Velis Electro is a type-certified two-seat light-electric trainer. Use it as a certification and operational reference, not as a direct cargo/regional benchmark.
- BETA ALIA CX300 is a useful fixed-wing utility reference with company-reported 225 kWh rated battery energy and a 336 nm proof-of-concept flight. Do not equate that demonstrator flight with certification-intent payload, reserve and dispatch performance.
- Heart ES-30 is a hybrid-electric regional programme. Its published ranges and charging times are programme targets, not certified results.
- NASA HEMM/HEATherR/NEAT are technology research and testbed references, not complete aircraft products.
- ARPA-E JOULES-1K is an advanced storage R&D horizon targeting 1,000 Wh/kg and 1,000 Wh/L at end-of-life net-energy-system level for selected projects. It is not qualified manned-aircraft battery availability.

All public values must carry evidence class, date, source, exact denominator, aircraft or component configuration, mission, operating conditions and uncertainty. See METRIC_BOUNDARIES.md.

## Luminous research hypothesis

The most defensible initial sequence is:
1. A conventional fixed-wing CTOL mission as the baseline.
2. A bounded distributed-propulsion variant as an explicit challenger.
3. A hybrid-electric variant only if the operator mission cannot close within credible battery-only constraints.
4. eVTOL only as a separate later programme after the energy, propulsion and safety modules have been characterized.

This order is a risk-reduction strategy, not a prediction that CTOL will always be best. A mission with no usable runway could legitimately reverse it.

## Why the platform itself can differentiate

Symthaea should explore the Pareto design space and select high-value experiments. Symtropy should orchestrate the appropriate specialist solvers and replay test evidence. Sol-Atlas should hold geometry, coordinate frames, mass stations and installation clearances. Mycelix should preserve supplier, material, inspection and maintenance provenance.

The differentiator is a reproducible decision record: why one architecture was selected, which assumptions drove it, what evidence supports those assumptions, what alternatives were rejected, and what new evidence would change the decision.
