# Long-Duration Liveaboard Vessel — Mission Basis v0.1

**Status:** research/design-basis proposal; not a vessel design, specification for construction, or certification submission.  
**Parent:** [MARINE-LIVEABOARD-001 #10926](https://github.com/Luminous-Dynamics/symthaea/issues/10926)  
**Branch:** `research/marine-liveaboard-mission-basis-v0.1`

## 1. Objective

Define a repeatable 16–20 m liveaboard / ocean-habitat vessel family. Use 18 m as the initial comparison point, not a presumed optimum. Compare monohull, catamaran and (where useful) trimaran configurations on the same mission and useful-service contract.

The vessel must be evaluated as a coupled home, vehicle, energy system, water/waste system, workshop and maintainable asset. Do not optimize calm-water resistance or luxury volume in isolation.

## 2. Provisional mission profiles

These are alternative requirements to evaluate, not frozen customer requirements:

| Profile | Occupancy case | Resupply interval | Operating mission |
|---|---:|---:|---|
| Coastal residence | 2 permanent | 7 / 14 days | Coastal and near-offshore living |
| Island-hopping home | 2 permanent + 2 guests | 14 / 30 days | Repeated passages with port access |
| Extended offshore residence | 2–4 permanent | 30 / 45 days | Extended offshore cruising |
| Ocean-crossing | Defined crew/guest manifest | Route-specific | Separate high-assurance case; not implied by offshore residence |

Autonomy is a vector, never one scalar: energy, potable water, food, waste capacity, consumables, critical spares, medical/emergency provisions and safe return reserves must each have a duration or explicit resupply condition.

## 3. Competing reference architectures

1. Catamaran: accommodation volume and deck/solar area are attractive, but beam, bridge-deck loads/slamming, windage, structural weight, marina constraints and motion response must be priced into the result.
2. Monohull: a serious baseline with mature offshore design practices and potentially lower beam/structural complexity; compare usable volume, stability, motion and energy fairly.
3. Trimaran: evaluate only where its speed, payload, stability or accommodation trade-offs justify added structure and beam.

No architecture is the default winner. For each candidate, use the same payload, occupants, range/reserve, route/weather envelope, service level, build scope and cost boundary.

## 4. Energy architecture hypotheses

- Conventional diesel propulsion and generation: baseline, not endorsement.
- Battery-electric: coastal/short-range and low-speed cases where verified usable energy, charging and reserves close.
- Renewable-first hybrid-electric: primary long-duration candidate; solar/wind contributions must be time-resolved and weather-/latitude-/shade-dependent. Generator/range-extender energy remains explicit.
- Sailing or wind assistance: candidate only when rig, weather, crew workload, deck plan and safety requirements are compatible.
- Hydrogen, methanol, ammonia and other fuels: separate route- and supply-specific studies with tank, containment, conversion, auxiliary energy, payload and lifecycle penalties.
- SMR: research only for future commercial vessels or shore/floating infrastructure; explicitly excluded as a dependency for the first private liveaboard demonstrator.

Battery nameplate capacity is not usable mission energy. Solar kWp is not daily energy. "Electric mode" time share is not the same as zero fuel, zero lifecycle GHG or total energy autonomy.

## 5. Habitat service model

Account for:
- Potable/hygiene/technical water separately, including desalination output, specific energy, intake assumptions, filter life, recovery ratio, brine and maintenance.
- Food by mass, volume, nutritional coverage, storage life, cold-chain power and provisioning days. Growing food onboard is a supplement unless a complete verified production model closes.
- Blackwater, greywater, bilge and solid waste by stream, storage/treatment capacity and legal discharge boundary.
- Ventilation, air quality, humidity, condensation, mold prevention, HVAC, hot-water and waste-heat rejection.
- Sleep, privacy, safe movement, accessibility, acoustic/vibration comfort, watchkeeping fatigue, workshop and medical/emergency provisions.

Compose with the existing Symtropy habitat-metabolism work where applicable. Do not create a second generic life-support simulator.

## 6. Functional modularity and repair

Make propulsion, inverters, batteries, watermakers, pumps, HVAC, filters, navigation/comms and interior assemblies serviceable as functional modules where justified.

Each module needs an interface-control definition covering:
- Mechanical mounts, loads, fasteners, tolerances and lift/removal envelope.
- Electrical voltage/current, connectors, isolation and protection.
- Fluid/cooling, ventilation, drainage and leak containment.
- Data interfaces, cybersecurity and safe local fallback.
- Fire boundaries, access, human reach, inspection and escape.
- Replacement procedure, tooling, skill, spare availability and requalification impact.

Keep primary hull and cross-deck load paths structurally continuous unless a qualified design explicitly substantiates a joint for strength, fatigue, damage, inspectability and applicable approval. Modularity does not excuse structural shortcuts.

## 7. Benchmark and evidence contract

Compare in matched missions, not by length or marketing headline.

| Dimension | Primary metric | Comparison controls |
|---|---|---|
| Propulsion | kWh/nmi or fuel energy/nmi | Same speed, payload, hull condition, route and sea state |
| Total energy | kWh/nmi; kWh/occupied berth-day | Include auxiliaries/hotel loads and reserve |
| Renewable contribution | Delivered renewable kWh / total input | Time, weather, latitude, shading and curtailment |
| Range | Verified range at reserve | Ageing, temperature, loading, weather and speed |
| Water | L/person/day; kWh/m³ | Water quality and use profile |
| Food | Person-days by mass/volume | Nutritional and storage assumptions |
| Waste | Capacity/treatment per person-day | Waste stream and legal boundary |
| Comfort | Accelerations, roll/pitch, noise, vibration | Sea state, heading, speed, occupant locations |
| Safety | Stability, structure, fire, flood, redundancy | Non-compensable hard gates |
| Maintainability | Task time, tools, skill, access, MTTR | Defined fault and spare availability |
| Manufacturing | Labour hours, scrap, rework, first-pass yield | Production volume and quality scope |
| Economics | Acquisition and lifecycle cost | Currency/year, scope, utilisation, service life |
| Environment | Lifecycle GHG and material impacts | Upstream energy/fuel and replacement boundary |

Never use a composite score to hide failure of a hard constraint. Preserve unknowns, failures, censored trials, evidence origin, uncertainty and out-of-domain status.

## 8. Public comparator anchors — claims are not independently validated

- **Silent 60 Series / SY62:** the closest published size-class solar-electric comparator. The current Silent 60 Series page lists 18.86 m LOA, 8.99 m beam, 1.26 m draft, 46.65 t light displacement, 16.8 kWp solar, 354 kWh batteries, 2 × 164 kW continuous electric motors and a 130 kW range extender. A separate Silent operating article describes a 286 kWh / 16.8 kWp configuration with a 140 kW range extender and publishes power-demand examples (20 kW at 5 kn, 30 kW at 6 kn, 40 kW at 7 kn and 280 kW at 12 kn). These figures must remain separate until the exact vessel/option/revision is reconciled; do not silently average 286 and 354 kWh or 130 and 140 kW. The same article estimates 80 kWh/day solar from Mediterranean May–June 2024 data and uses 5 kW average domestic load (120 kWh/day), implying a 40 kWh/day nominal deficit under that scenario before additional losses; the manufacturer says careful load management can reduce this. The Tech Corner separately reports the 2026 Atlantic voyage (~3,800 nmi, ~5,500 L fuel, batteries for 72% of journey and ~40% lower fuel than a similar conventional 60-foot motor cat). All are manufacturer reports, not independently reproduced measurements. Sources: https://silent-yachts.com/en/yacht/60-series-2/ ; https://silent-yachts.com/living-on-a-silent-62-solar-electric-yacht/ ; https://silent-yachts.com/en/tech-corner/

- **M/Y Seawolf X:** 42.75 m full-aluminium hybrid-electric catamaran; Rossinavi reports 10 kn cruise, 12 kn top speed and 5,400 nmi at 8 kn. Its published operating scenarios report electric-mode time shares of 100% for one-day trips, 90% for multi-day trips and 80% for a transatlantic trip. These are manufacturer-reported figures for a much larger, custom superyacht; not a size-equivalent cost or performance baseline. Sources: https://rossinavi.press.camronglobal.com/rossinavi-unveils-my-seawolf-x and https://rossinavi.press.camronglobal.com/rossinavi-launches-my-seawolf-x
- **Sunreef 60 Eco:** an 18.4 m composite sailing-catamaran comparator. Sunreef describes integrated solar surfaces in hull sides, superstructure and bimini with a custom battery bank. Exact vessel/configuration-specific output and operating traces remain to be sourced. Source: https://sunreef-yachts.com/en/launched/sunreef-60-eco/

The companion `vessel_benchmark_register.csv` records only sourced public values and leaves unreported values blank. Manufacturer statements are not independent measurements. These vessels have different configurations, missions and service levels; no ranking is yet defensible.

## 9. Existing Luminous ownership

- Symthaea: requirements, mission identity, engineering interpretation, evidence contracts and candidate proposal.
- Existing marine-bench owners: catamaran candidates, energy budget, voyage traces, matched trials, operating-envelope completeness, validation and uncertainty.
- Symtropy: spatial configuration, interior/access/service clearances, assembly/resource scheduling and deterministic scenario replay; not a substitute for validated CFD or structural FEA.
- Mycelix: component provenance, custody, manufacturing and inspection records; not engineering approval.
- Sol-Atlas: source-aware route, site, climate, solar and port infrastructure projections.
- Robotics/HAL: bounded inspection/assembly/maintenance adapters; physical safety remains local and independently assured.

Primary references:
- Symthaea #10926: https://github.com/Luminous-Dynamics/symthaea/issues/10926
- Catamaran design #7719: https://github.com/Luminous-Dynamics/symthaea/issues/7719
- Catamaran search #7737: https://github.com/Luminous-Dynamics/symthaea/issues/7737
- Hydro co-design #8082: https://github.com/Luminous-Dynamics/symthaea/issues/8082
- Mission energy verifier #8092: https://github.com/Luminous-Dynamics/symthaea/issues/8092
- Shipbuilding design-for-manufacture #2048: https://github.com/Luminous-Dynamics/symtropy/issues/2048

## 10. First engineering gates

1. Every critical input has SI units, source/status, exact revision, uncertainty and applicability.
2. Mission profile and service envelope are frozen before candidate comparison.
3. Independent arithmetic fixtures verify energy, water, provisioning and payload/person-day calculations.
4. No false-safe results in frozen structural, stability, fire, flooding, battery, propulsion-loss and out-of-envelope controls.
5. Hydrodynamic models are validated per observable and condition against rights-reviewed data before they inform ranking.
6. Energy-model thresholds are chosen only after a traceable meter chain and mission-matched voyage data exist.
7. Maintainability and cost claims require timed service trials and source-pinned, scope-complete quotations.
8. No concept/simulation/digital-twin PASS is called seaworthy, approved, safe for offshore use, or production-ready.

## 11. Standards watch

Applicability depends on vessel use, length, tonnage, passenger capacity, flag and waters. Relevant public references include ISO 16315:2026 for electrical propulsion on applicable small craft up to 24 m; ISO 12217 stability/buoyancy; ISO 12215 structural requirements; applicable COLREGs, MARPOL/SOLAS, flag-state and classification rules. Track current editions and approval pathways rather than copying restricted standards text.

This document is a research design basis only. It is not a construction package or professional design approval.
