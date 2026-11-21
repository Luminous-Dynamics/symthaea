# MARINE-DESIGN-001 — mission-first yachts and cruise ships

Status: concept-stage design hypotheses and machine-readable records. No hull form, propulsion sizing, resistance result, energy balance, stability result, safety case or class approval is yet established.

Research issue: [MARINE-DESIGN-001 #7703](https://github.com/Luminous-Dynamics/symthaea/issues/7703)

## Executive design thesis

Design the service the ship provides—not only its hull. A yacht's value may be range, quiet anchorage, expedition capability, comfort and maintainability. A cruise ship's value includes passenger-nights, itinerary reliability, food/service, air quality, water, waste handling and accessible evacuation. Optimising only hull resistance can miss major hotel loads, energy conversion losses, fuel pathway emissions and port-infrastructure constraints.

The new family keeps three scales separate: a 60 m expedition-yacht hypothesis, a 150 m coastal expedition-cruise hypothesis, and a 280 m ocean-cruise stress case. They share an evidence and energy-accounting method but not a presumed hull, duty cycle, battery size or propulsion design.

## What current research says—and doesn't say

Hurtigruten's Sea Zero project is an important public benchmark anchor. Its company project page says its 2026 R&D phase found a route-specific design could reduce energy use by 44%, and describes a roughly 143.5 m coastal vessel for around 500 passengers with a planned 73 MWh battery system, retractable wing sails with solar integration, contra-rotating propellers, underwater air lubrication and improved heat-pump/HVAC systems. The claim is specific to that concept and its route assumptions; it is not independent evidence of a fleet-wide 44% saving, not a full life-cycle GHG figure, and not a reason to copy its battery size into a different vessel. Source: [Sea Zero](https://www.hurtigruten.com/en/why-hurtigruten/sustainability/sea-zero).

The International Maritime Organization's January 2026 SDC 12 workplan identifies future safety-rule work involving wind propulsion, lithium-ion batteries and swappable traction battery containers. That means the concept should preserve route- and technology-specific safety reviews as first-class design inputs rather than assume every novel arrangement is already accepted. Source: [IMO SDC 12](https://www.imo.org/en/mediacentre/meetingsummaries/pages/sdc-12.aspx).

Passenger ships must meet applicable IMO safety and pollution rules. For international voyages, SOLAS applies to passenger ships carrying more than 12 passengers, and garbage/sewage controls are central cruise-ship concerns; the exact legal matrix depends on flag, route, ship category, tonnage and current regulations. Source: [IMO passenger-ship safety and environmental standards](https://www.imo.org/en/mediacentre/hottopics/pages/passengership-default.aspx).

IMO's existing efficiency framework includes ship-design and operational measures such as EEDI, EEXI, SEEMP and CII. Fuel and regulatory pathways remain a sensitivity case rather than a single fixed assumption; the emissions comparison must disclose well-to-wake boundaries and the origin of shore electricity and fuel. Source: [IMO energy efficiency measures](https://www.imo.org/en/ourwork/environment/pages/improving%20the%20energy%20efficiency%20of%20ships.aspx).

Hydrodynamics must be validated with relevant evidence. The ITTC practical CFD guidance describes verification/validation as part of the numerical workflow and recognises that uncertainty and modelling gaps vary across resistance, propulsion, manoeuvring, seakeeping and sail aerodynamics. A converged computation is not proof of physical accuracy. Source: [ITTC Practical Guidelines for Ship CFD Applications](https://ittc.info/media/11958/75-03-02-03.pdf).

## Design archetypes

### A. Bluewater Explorer 60 — expedition yacht

**Mission hypothesis:** 60 m displacement explorer; 12 guests; a provisional 20-person crew assumption; long-distance passages, quiet anchorage, remote serviceability, tender/science equipment and low-noise hotel modes. The first study uses 10 kn and 5,000 nmi as provisional mission targets, not estimated achieved performance.

**Architecture candidates:** efficient displacement hulls and propulsor arrangements; a segregated ship-wide electric architecture; battery buffering for transients and quiet/port modes; onboard dispatchable generation; shore power when available; lifecycle-reviewed fuel options; and wind assistance only if rig mass, heeling, visibility, air draft, fatigue, tender handling and emergency access close. Solar contribution is calculated from actual area and route irradiance, not rated peak power.

**Core challenge:** range, battery mass, fuel storage, payload, stability, quiet mode, and redundancy compete. Do not choose battery MWh or propulsion kW before the route power/time series, hotel-load profile, reserve rule, hull and mass budget exist.

### B. Coastal Expedition 150 — frequent-stop coastal cruise

**Mission hypothesis:** approximately 150 m and 500 passengers, with a 12-day, frequent-stop route stress test. The initial 34-stop assumption is for scenario generation only, not an adopted itinerary. Optional community logistics or vehicle/cargo capacity should be included only when a real mission requires it.

**Architecture candidates:** low-resistance hull/propulsor integration; modular electrical distribution; shore power and charging scheduled together with port calls; batteries sized from measured/validated route loads and port dwell; retractable wind-solar rigs; HVAC zoning, heat pumps, thermal storage and waste-heat recovery; fresh-water and waste-treatment optimisation; and fuel alternatives evaluated with safety and lifecycle accounting.

**Core challenge:** the system is a moving hotel as well as a ship. Passenger comfort, HVAC, galley, refrigeration, laundry, hot water, desalination and waste treatment need time-resolved load models. If the timetable forces high speed or a port cannot supply the required power, a battery-dominant concept may not close.

### C. Ocean Voyager 280 — large cruise ship stress case

**Mission hypothesis:** 280 m, a 4,000-passenger design-capacity tier, and a provisional 7–14 day itinerary family. These are research-envelope assumptions used to force explicit treatment of energy, water, crowd movement, ventilation, machinery redundancy, evacuation and shore-power constraints. They are not a chosen arrangement or approved capacity.

**Architecture candidates:** segregated medium-voltage/electric zones and essential-service buses; hybrid-electric power and energy storage only where mission/mass/safety closure supports it; future-fuel and shore-power comparisons; heat pumps, thermal storage, heat recovery and demand-aware HVAC; and wind assistance only as a constrained variant after arrangement, stability, air draft, lifeboat, deck/egress and structural loads are considered.

**Core challenge:** large-ship decarbonisation is not a battery-sizing contest. It is a coupled optimisation of hull/propulsor, hotel energy, route and schedule, power conversion, thermal systems, fuel pathway, port network, service reliability, and life-safety architecture.

## Common architecture to investigate

1. **Mission and requirements layer:** routes, weather/sea state, speed distribution, guest/crew capacity, service and comfort levels, port calls, resupply, schedule, reserve policy and degraded modes.
2. **Hull/propulsor layer:** compare displacement and other appropriate hull families, appendages, propeller/rudder or azimuthing options, trim, fouling and sea-state effects. No single hull is assumed optimal for all archetypes.
3. **Electric and energy layer:** mode-resolved loads; segmented distribution; generator/fuel-cell/battery options; charging and reserve; protection/selectivity; black-start; thermal-runaway/fire zones; redundancy and maintainability.
4. **Thermal/hotel layer:** HVAC, refrigeration, galley, laundry, freshwater production, hot/cold storage, heat recovery, insulation, ventilation and passenger-setpoint control.
5. **Renewable-assist layer:** wind and solar measured as route-dependent contributions with interactions and constraints; never credit an idealised rated-power value as continuous output.
6. **Circular-service layer:** freshwater demand, grey/black water, garbage, food waste, bilge and maintenance residues; quantify retention/treatment/discharge under route regulations rather than assume zero waste.
7. **Operational intelligence:** route/weather/speed/trim optimisation, energy dispatch and predictive maintenance, always subordinate to collision avoidance, weather/route limits, human procedures and hard safety constraints.
8. **Evidence layer:** exact references, geometry/conditions, model version, uncertainty classes, dataset rights, held-out cases, provenance and the precise claim ceiling.

## Benchmarks: concepts to engineering evidence

These are proposed gates and ambition targets, not current achievements.

| Stage | Measure | Proposed target or gate |
|---|---|---|
| Input quality | Unit/source/status completeness | 100% of critical inputs have units, provenance or explicit assumption/unknown status; zero silent defaults |
| Mission coverage | Route/weather/port/occupancy variants | At least 10,000 deterministic scenarios per archetype after the model is defined, with held-out and boundary-adversarial strata |
| Energy/carbon | Well-to-wake GHG per guest-night or passenger-nautical-mile | Aim for ≥30% reduction against a functionally matched baseline; stretch ≥45%. This is an aspiration and must be checked against fuel, electricity and lifecycle assumptions |
| Port energy | Onboard direct combustion while berthed | Aim for zero where verified shore-power/energy supply and operating conditions make it feasible; report upstream electricity emissions separately |
| Hydrodynamics | Resistance prediction versus relevant model-test/reference data | After selecting lawful/eligible validation cases, target ≤5% error on the declared validation envelope and publish off-design error separately; concept estimates alone cannot pass this gate |
| Route energy | Predicted versus measured voyage energy | Once traceable voyage data exists, target ≤5% median absolute percentage error on the declared validation domain; report tails and prediction-interval coverage, not just the median |
| Reliability | Schedule and essential-service availability | Report route-specific reliability and degraded-mode performance; never mask a failed hard gate with a high weighted score |
| Safety | False-feasible hard decisions | Zero false-feasible decisions in prespecified negative controls. A concept remains unqualified until the competent reviews and evidence are complete |
| Trade-space quality | Multi-objective frontier | Publish nondominated candidates across GHG, energy, operating cost, capex, range, comfort, maintainability and schedule reliability; no hidden scalar winner |

Compare each design against at least three matched baselines: a conventional reference, an improved hybrid reference, and a constrained low/zero-direct-emission-at-berth variant. Hold the service level, passenger capacity, route, schedule and reserve policies equivalent. Keep yacht and cruise metrics distinct where appropriate—guest-night for yacht service and passenger-night/passenger-nautical-mile for cruise service.

## Why a Pareto frontier is important

A ship with lower energy use but poor range, unsafe stability, unacceptable comfort or unavailable charging is not a better ship. A concept that is cheap under one fuel-price assumption but has high life-cycle GHG may be dominated under a different pathway. Display the trade-offs and sensitivity instead of hiding them in one arbitrary weighted score.

Hard constraints remain outside the Pareto trade: stability, structure/fatigue, collision/grounding operating envelope, fire/explosion, evacuation/lifesaving, machinery redundancy, environmental discharges, route draft/air draft and class/flag requirements must be passed or remain explicitly unresolved.

## Repository artifacts

- `crates/domains/symthaea-marine-bench/schemas/marine_vessel_concept_v1.schema.json` — strict concept-input and assumption contract.
- `crates/domains/symthaea-marine-bench/fixtures/design-concepts/ld-yacht-explorer-060-v0.json` — expedition-yacht hypothesis.
- `crates/domains/symthaea-marine-bench/fixtures/design-concepts/ld-coastal-cruise-150-v0.json` — coastal cruise hypothesis.
- `crates/domains/symthaea-marine-bench/fixtures/design-concepts/ld-ocean-cruise-280-v0.json` — ocean-cruise stress case.

Every energy load, battery capacity, beam, draft and displacement is deliberately unknown until it can be derived from a defensible design process. Performance reductions are targets for later matched comparison, not predicted results.

## Implementation sequence

1. Validate the records against the strict schema and freeze an input manifest.
2. Define units, reference frames, load-state conventions, baseline selection and uncertainty propagation in Rust.
3. Build hourly or 15-minute operating profiles before sizing power, storage and fuel tanks.
4. Add a deterministic mission/energy-balance harness with explicit conversion losses, reserve policies, shore-power limits, load diversity and fuel-pathway emission factors.
5. Add a Pareto search under hard constraints; keep uncertainty and failed/unknown evidence visible.
6. Tie hydrodynamic and energy models to rights-reviewed references and appropriate model tests; only then report quantitative performance.
7. Move to general-arrangement/layout, stability, structural, fire, evacuation and class/flag reviews with qualified naval architects and marine engineers.

## Claim ceiling

A PASS in a concept-input or arithmetic benchmark demonstrates only that the declared contract and calculations behave as specified. It is not evidence of seaworthiness, safety, class approval, statutory compliance, build readiness or actual operating performance.