# Planetary Energy Service Envelope (PESE) — v0.1

**Status:** research contract / benchmark design  
**Issue:** [#11756](https://github.com/Luminous-Dynamics/symthaea/issues/11756)  
**Branch:** `research/planetary-energy-service-envelope-v0.1`  
**Evidence ceiling:** design specification plus analytic screening cases only. No physical energy-system result is claimed.

## 1. Thesis

PESE evaluates an energy architecture by the **service it delivers over a mission**, not by the efficiency of one component.

For a declared site, mission horizon, environmental history, load profile and evidence set, PESE asks:

> Which generation, storage, thermal, distribution, maintenance and local-manufacturing architecture can meet the required service with the lowest defensible lifecycle burden?

The output is a vector of feasibility, service, mass, lifecycle, resilience, uncertainty and evidence metrics. There is no universal weighted scalar called "best energy".

Electricity, heat at a specified temperature/pressure/flow, cooling/heat rejection, chemical carriers, mechanical work and exergy remain distinct service quantities. A kWh-thermal at 80 °C is not interchangeable with a kWh-electric or a kWh-thermal at 800 °C.

## 2. Research hypothesis and paradigm shifts

The programme tests six hypotheses:

1. **Service-first co-design:** optimizing the complete service envelope will expose failures hidden by cell/module efficiency, nameplate kWh or nominal W/kg.
2. **Geometry-to-power evidence chain:** site geometry and solar visibility can improve resource characterization, but only a separate irradiance / device / system model may derive electrical output.
3. **Electricity–heat co-design:** PV, process heat, thermal storage, heat recovery and radiative rejection should be optimized together while preserving energy quality and accounting boundaries.
4. **Repair-first infrastructure:** replaceable modules, cleaning, inspection, repair time, spares and local fabrication should be design variables from the beginning, not afterthoughts.
5. **Industrial closure over local-mass percentage:** the system should identify the critical imported components and equipment dependencies that prevent local productive capacity from expanding.
6. **Decision-value experimentation:** choose the next measurement because it could change a design decision, not merely because it reduces a model's uncertainty.

These are research hypotheses, not established advantages.

## 3. System ownership and composition

| System / workstream | PESE interface | Must not be duplicated |
|---|---|---|
| Symthaea #8496 | Diverse architecture candidates, hard-constraint reasoning, experiment proposals and Pareto analysis | A new universal energy architecture generator |
| Symthaea #8381 / #7248 | Source-qualified energy evidence and scheduling inputs | A competing evidence catalogue or scheduler |
| PIE #1604 / #1606 | Industrial process graph, material/mass closure and local productive-capability dependencies | A second ISRU process ontology or mass-balance engine |
| Lunar site #1603 / #1716 / #1719 | Site/terrain/horizon/illumination and common-site viability references | A second terrain or solar-visibility solver |
| Thermal / energy #11718 / #11734 / #11751 | Matched useful-heat services, energy-flow accounting and source registers | A second thermal-store or general energy-accounting engine |
| Battery design #11671 | Mission-conditioned storage and battery-design evidence | A battery electrochemistry or lifetime solver |
| Symtropy #3400 / #3418 | Time-coupled synthetic power-system execution, resilience, islanding and black-start scenarios | A new electrical-network or power-electronics solver |
| Symtropy #1467 | Exact deterministic twin projection, replay and synthetic trajectory boundary | An alternate twin-of-record or physical observation authority |
| Sol-Atlas | Spatial geometry, terrain, asset layout, routes, frames and as-built spatial relations | PV conversion or thermal qualification |
| Mycelix #4020 and existing manufacturing/supply-chain owners | Component/lot identity, material and custody provenance, maintenance and bootstrap milestones | Engineering design authority or physical qualification |

**Integration theorem:**

```text
site geometry
!= incident irradiance
!= PV DC output
!= inverter AC output
!= stored energy
!= delivered service
!= independently qualified plant
```

Every transition requires its own model, boundary, evidence and uncertainty.

## 4. Environmental reference cases

The first benchmark release contains seven analytic orbital-distance cases and one site-specific lunar layer:

| Case | Distance used | Nominal normal-incidence top-of-atmosphere flux | Role |
|---|---:|---:|---|
| Earth reference | 1.0 AU | 1,361 W/m² | Terrestrial baseline before atmosphere/weather |
| Moon orbital reference | 1.0 AU | 1,361 W/m² | Solar resource baseline; local terrain and polar geometry dominate site variation |
| Mars mean-distance screen | 1.524 AU | 585.987 W/m² | Solar/storage/fission hybrid stress case |
| Main belt inner study point | 2.2 AU | 281.198 W/m² | Array-area and mass scaling |
| Main belt reference study point | 2.7 AU | 186.694 W/m² | Reference asteroid-belt trade |
| Main belt outer study point | 3.2 AU | 132.910 W/m² | Lower-flux belt sensitivity |
| Jupiter-distance control | 5.2 AU | 50.333 W/m² | Low-solar-flux / firm-power comparison |
| Saturn-distance control | 9.5 AU | 15.080 W/m² | Very-low-solar-flux / firm-power comparison |

These are **analytic screening values**, calculated from the explicitly declared approximation:

[
S(r)=rac{1361}{r^2};mathrm{W/m^2}
]

where (r) is distance in astronomical units. They are not site measurements, ephemerides, panel-plane irradiance, annual energy, delivered electricity or mission power predictions. Real analyses must use time-resolved ephemerides and exact mission epochs. Earth and Mars additionally require atmosphere/weather/dust models. The lunar surface requires terrain horizon, solar-disc visibility, local slope and panel orientation. An asteroid's heliocentric distance and composition are specific to the target object and epoch.

The machine-readable seed cases are in [`benchmarks/planetary-energy-service/v0.1/orbit-reference-cases.json`](../../benchmarks/planetary-energy-service/v0.1/orbit-reference-cases.json).

### Lunar polar site layer

The Moon is not a uniformly illuminated 1-AU solar platform. For each candidate site retain:

- body-fixed coordinate frame, site identity and epoch lineage;
- horizon coverage and angular sampling gaps;
- Sun centre and full-disc visibility separately;
- local panel-plane incidence geometry and orientation;
- longest contiguous darkness interval, with sampling-derived brackets;
- terrain and visibility uncertainty;
- surface temperature and thermal environment evidence;
- radiation, dust/abrasion and site-preparation assumptions.

A visibility record from #1719 remains geometry evidence. It does not establish irradiance at a tilted module, electrical generation, useful thermal output, or site qualification. Missing horizon sectors and time gaps must remain unresolved.

### Mars and small-body layer

For Mars, separate atmospheric optical depth and airborne dust from deposited dust/soiling on equipment. Keep storm duration, seasonal timing, array cleaning energy, panel abrasion and recovery time explicit. A global storm scenario cannot be represented by a single permanent derating factor without a stated model and applicability range.

For asteroid cases, do not assign a generic resource composition. Resource abundance, mineralogy, physical form, contaminants, extraction yield and process energy must be linked to a target-specific evidence record. The inverse-square irradiance screen is not an asteroid mission model.

## 5. Mission service contract

Every candidate must be evaluated against the same immutable mission/service contract. At minimum it binds:

- mission/site/body identity, frame, epoch, duration and scenario-set digest;
- critical and flexible electrical loads as time series, with bus voltage, AC/DC form, peak/continuous power and reserve policy;
- useful thermal loads by temperature band, pressure, flow, duty cycle and timing;
- cooling / heat-rejection demand and permitted component temperature envelopes;
- start-up, standby, idle, maintenance and black-start loads;
- storage initial state, reserve, charging opportunities and end-of-mission requirements;
- mission interruptions and unserved-service policy;
- equipment configuration, deployment geometry and exact BOM revision;
- environment history and all applicable evidence/model identities.

A candidate may not obtain a favourable result by changing the mission, omitting an auxiliary load, assuming an unmetered heat sink, or counting a source-side output as delivered service.

## 6. Calculation and accounting chain

### 6.1 Resource-to-electricity

The irradiance screening stage may use:

[
G_{POA}(t)=S(r,t),max(0,mathbf ncdotmathbf s(t)),V(t),	au(t)
]

where:

- (S(r,t)) is solar irradiance at the declared distance/epoch;
- (mathbf n) is the panel-plane normal;
- (mathbf s(t)) is the direction toward the Sun in the same frame;
- (V(t)) represents a qualified visibility/occlusion factor;
- (	au(t)) represents a declared atmospheric transmission model when applicable.

This is an interface equation, not a complete irradiance transposition or PV device model. Partial solar-disc visibility, terrain uncertainty and temporal sampling gaps must be preserved. A simple binary (V) is acceptable only where the geometry contract supports it.

The PV conversion stage must separately bind module technology, I–V model or performance source, cell temperature, spectral/angle assumptions, radiation exposure, soiling, mismatch, wiring, bypass, converter and inverter losses. No universal efficiency or degradation scalar may be applied to all technologies and bodies.

### 6.2 Storage and dispatch

Storage is a time-dependent state, not a nameplate capacity:

[
E_{t+Delta t}=E_t+eta_c P_cDelta t-rac{P_dDelta t}{eta_d}-L(E,T,Delta t)
]

The exact storage model owns its own state limits, efficiency, temperature, degradation, self-discharge, power limits and reserve semantics. The expression above is an accounting sketch, not a universal battery model.

For each mission interval report:
- available energy and power envelope;
- state of charge / state of health or technology-appropriate state;
- charge and discharge losses;
- reserve and end-of-mission constraints;
- unavailable intervals and required recovery;
- replacement and recycling assumptions.

### 6.3 Thermal service and rejection

Thermal outputs are reported at the service boundary and temperature grade. Report supplied heat, recovered heat, stored heat, losses, auxiliary electricity and heat actually delivered to the specified process.

For vacuum radiators, a simplified radiation term is:

[
dot Q_{rad}=epsilonsigma A F_{view}(T_{rad}^{4}-T_{sink}^{4})
]

This is only a radiative submodel term. A qualified system model also needs geometry, view factors, solar and planetary absorbed flux, surface properties, radiator orientation, operating temperatures, plume/contamination constraints and uncertainty. It cannot be used alone to claim a radiator design is feasible.

### 6.4 Industrial / ISRU accounting

The process graph must preserve:

```text
resource occurrence
-> acquisition
-> beneficiation
-> refining
-> material grade
-> fabrication
-> use
-> maintenance
-> reuse / recycling / waste
```

Resource occurrence is not usable feedstock. Silicon-bearing material is not semiconductor-grade silicon. A material output is not a qualified component until composition, purity, defects, geometry, process history and intended-use requirements have been verified.

Mass, elemental, energy, heat, exergy, time, equipment capacity, consumables and waste/by-product closure remain separate ledgers. Local-mass percentage must be reported alongside critical-path imports and imported equipment dependencies.

## 7. Architectures and baseline tournament

All candidates use the same mission, site evidence, load trace, time horizon, uncertainty set, cost basis and information/compute budget.

### Earth
- transparent PV + inverter + grid/battery reference;
- PV + battery + demand management;
- tracking/bifacial/tandem candidates only when technology evidence is admitted;
- photovoltaic-thermal and solar-thermal storage;
- hybrid with firm generation where the service contract requires it.

Use NREL SAM/PVWatts and other public/qualified tools for matched use cases. Compare field performance-loss data with the exact distinction between module degradation, soiling and system performance loss.

### Moon
- PV + battery;
- PV + regenerative fuel-cell/chemical storage;
- concentrated solar thermal + thermal storage + power conversion;
- nuclear/fission firm reference;
- solar + nuclear hybrid.

Use NASA's 40-kWe-class Fission Surface Power as a reference architecture for continuous lunar power. Do not present it as an equivalent cell-efficiency benchmark. Include long darkness, low-Sun geometry, thermal extremes, regolith handling, dust, radiation, radiator mass and deployment operations.

### Mars
- solar + storage;
- fission firm generation;
- hybrid solar/fission;
- dust-storm and cleaning/recovery variants.

All candidates must deliver the same critical loads across the same storm and night traces.

### Main belt and outer system
Scale the solar collector from a common service contract at multiple heliocentric distances. Include area, structure, pointing, deployment, conversion, thermal management, storage, maintenance and transport mass. Compare against firm power where solar collector and storage burdens dominate.

### Search-policy baselines
Compare hand-authored engineering designs, seeded random, space-filling, simple sensitivity/uncertainty sampling, a qualified BO/MOO policy where assumptions are defined, and the Symthaea evidence-/decision-aware architecture and experiment search. Include the complete cost of required high-fidelity and physical verification; a cheap selected design that cannot be validated is not a cost saving.

## 8. Metrics

Do not hide metrics in one weighted score.

| Dimension | Required outputs |
|---|---|
| Critical electric service | availability by interval, unserved kWh, peak/continuous power shortfall, longest outage, reserve deficit |
| Thermal service | useful heat by temperature/pressure/flow/duration; recovered and stored heat; heat rejection and component temperatures |
| Specific power/mass | cell, module, array, converter, storage, radiator, installed plant and maintained plant, with explicit boundaries |
| Lifetime | delivered electrical kWh/kg and useful heat by grade/kg; degradation, replacement, service life and maintenance |
| Resilience | failure detection, isolation, recovery time, load shedding, black-start, islanding and communications-loss behavior |
| ISRU | feedstock-to-qualified-component yield, energy/mass/time per output, local/imported mass, critical-path imported dependencies |
| Circularity | repairability, disassembly, recycling yield, material loss, by-product/waste destinations |
| Model/evidence | prediction error, interval coverage, OOD detection, abstention, model discrepancy, convergence and repeatability |
| Search value | feasible design discovery rate, decision regret, false-feasible rate, experiment cost, wall-clock/compute/energy cost |
| Reproducibility | exact input/source/model/configuration digests, deterministic replay, complete attempted/failed/censored run ledger |

### Boundary discipline

Specific power must be published at multiple boundaries: cell, packaged module, deployed array, power subsystem and complete installed/maintained plant. Do not compare an un-packaged cell's W/kg with a deployed plant's W/kg.

Lifetime energy per kilogram must likewise state whether the denominator is active material, module, array, installed system, imported mass or total mass. Report more than one denominator where meaningful.

Electric and thermal output must not be added into a fictitious single energy efficiency. If a combined service metric is required, use a declared exergy or economic-value basis with its own assumptions.

## 9. Proposed targets and gates

These are internal research targets, not standards or measured results.

### Contract and evaluator
- 100% exact agreement with independent analytic fixtures for inverse-square flux and closed-form bookkeeping within declared tolerances.
- Zero false-feasible results in the frozen negative-control suite.
- 100% accounting for successful, failed, missing, censored, out-of-domain and abstained runs.
- 100% of promoted claims carry exact source, model, configuration, site, epoch, unit, service-boundary and uncertainty identities.
- Zero promotion of geometry-only visibility, synthetic trajectories, manufacturer claims or literature reports into physical measurements.

### Decision quality
- On held-out sites and temporal scenarios, demonstrate a statistically supported reduction in the pre-registered primary decision metric versus transparent, random, space-filling and uncertainty-only baselines at equal information/compute/validation cost.
- Stretch goal: at least 10% lower worst-case unserved critical energy OR at least 10% lower lifecycle burden for equivalent service in at least three held-out scenario families, with no regression in hard constraints. Freeze exact primary objective, variance and statistical protocol before the confirmatory run.
- Report counterexamples, effect sizes, uncertainty and subgroup outcomes. A failure to outperform is a valid result.

### Physical qualification progression
1. Synthetic known-truth semantics and analytic invariants.
2. Reproduction of source-defined public reference cases.
3. Independently implemented differential model comparisons.
4. Instrumented terrestrial component bench with calibrated metrology.
5. Environmental testing (thermal cycling, dust/soiling, vibration, vacuum/radiation as applicable) at qualified facilities.
6. Regolith-simulant processing demonstrator for one bounded material product.
7. Integrated field or mission-representative prototype with independent review.

A software PASS does not skip a rung.

## 10. Initial falsification corpus

At minimum, include:
1. Sun below terrain horizon while a naive flat-horizon calculation reports full sun.
2. Sun centre visible but part of solar disc occulted.
3. Missing azimuth sector or temporal gap that must block promoted darkness statistics.
4. Same orbit radius but changed epoch/frame/panel orientation; outputs must not be reused.
5. Atmosphere/dust missing for Mars; do not substitute zero attenuation.
6. Deposited dust vs atmospheric opacity represented as distinct effects.
7. Long darkness with insufficient storage energy despite enough daily generation.
8. Adequate kWh but inadequate discharge power/current.
9. Battery/thermal-store degradation causing late-mission reserve failure.
10. Thermal rejection capacity lower than simultaneous process and conversion heat.
11. Radiator with favorable emissivity but blocked view factor / excessive solar load.
12. Auxiliary/pump/heater/standby load unknown; superiority blocked.
13. Nested total and branch meters causing double counting.
14. Electricity and high-grade heat combined as if identical.
15. Local raw regolith mass counted as qualified product mass.
16. High local-mass fraction but one imported critical machine prevents production.
17. A failed/censored physical test omitted from denominator or campaign cost.
18. Same friendly design name with changed BOM/configuration; exact identity must change.
19. Simulation trajectory presented as physical measurement.
20. Candidate wins only because its comparator has a different mission, duty cycle, output temperature or accounting boundary.

## 11. Immediate implementation sequence

- **P0 — contract:** this document and the source/evidence register.
- **P1 — reference corpus:** orbit-distance screening cases, units, analytic oracle, synthetic environment/site profiles.
- **P2 — evidence and service-envelope evaluator:** boundary completeness, time-grid coverage, critical-service arithmetic, dispositions and deterministic reports; reuse existing quantity/energy owners.
- **P3 — composition:** #1719/#1716 site evidence; #8496 architecture search; #11734/#11751 thermal/accounting; #11671 storage; Symtropy #3418 resilience; Mycelix #4020 lineage.
- **P4 — Earth bench:** PV/thermal measurement with separate electrical and thermal meters, calibrated temperature and environment, dust/soiling control and baseline module.
- **P5 — ISRU coupon:** one explicit regolith-simulant product (e.g. sintered structural coupon or thermal-storage specimen) with mass/energy/time balance and independent product-quality tests.
- **P6 — integrated demonstrator:** couple energy service, local processing, repair and provenance only after each subsystem qualifies.

Runner-independent work should begin with P1 fixtures and P2 pure accounting/schema checks. Do not create a second solver while waiting for hardware or solver environments.

## 12. Industry and research references

- NASA, Fission Surface Power: <https://www.nasa.gov/space-technology-mission-directorate/fission-surface-power/>
- NASA, lunar surface technology call, 8 September 2026: <https://www.nasa.gov/space-technology-mission-directorate/technology-demonstrations/lunar-surface-technology/>
- NASA, Blue Alchemist / ISRU: <https://www.nasa.gov/space-technology-mission-directorate/technology-demonstrations/blue-alchemist/>
- NASA SmallSat Institute, State-of-the-Art: Power Systems: <https://www.nasa.gov/smallsat-institute/sst-soa/power/>
- NREL, PV Fleet Performance Data Initiative: <https://www.nrel.gov/pv/lifetime.html>
- NREL, System Advisor Model: <https://sam.nrel.gov/>
- NREL, REopt: <https://reopt.nrel.gov/>
- PyPSA: <https://pypsa.org/>
- GenX: <https://github.com/GenXProject/GenX>
- Nature Reviews Clean Technology (2026), perovskite PV commercialization perspective: <https://www.nature.com/articles/s44359-026-00173-2>

Source URLs are discovery pointers. Each future admitted case must pin the exact publication/product version, date, system boundary, evidence class, applicability, data rights and source bytes/digest. Links alone are not qualification evidence.


## 13. Industry comparison snapshot — 12 November 2026

These references establish what an external comparator can currently support. They are not one interchangeable leaderboard.

### Space-array specific power

NASA's 2026 SmallSat Institute survey describes empirical spacecraft solar-array specific power as strongly clustered around approximately 30 W/kg, with the cited mission dataset bounded by an observed maximum around 200 W/kg. This is a spacecraft-array reference distribution, not a lunar surface plant target: surface foundations, dust mitigation, deployment, cabling, storage, radiators, power conversion and maintenance can dominate a surface installation.

Benchmark requirement: report W/kg at cell, module, deployed-array, complete generation subsystem and complete maintained-plant boundaries. Compare the same environmental distance, mission, output voltage, deployment state and included mass.

Source: https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/

### Terrestrial lifetime performance

The NLR PV Fleet Performance Data Initiative reports a median fleet performance loss of 0.75%/year across 4,915 observations, with a P90 loss around 1.9%/year in the current published snapshot. A 2025 meta-analysis of 80 studies and 610 field degradation observations reports a 0.94%/year median module degradation rate. These figures are not contradictory: fleet-level performance loss and module degradation use different system boundaries and populations; soiling, inverter/plant faults, cohort selection and methodology matter.

Benchmark requirement: preserve raw energy yield, plane-of-array irradiance, meteorology, sensor calibration, availability, soiling/cleaning events and module/inverter identities. Report degradation and recoverable loss separately. A high initial efficiency does not win a lifetime-service competition by itself.

Sources:
- https://nrel.sitefinity.cloud/pv/fleet-performance-data-initiative
- https://doi.org/10.1016/j.rser.2025.115697
- https://doi.org/10.2172/3020423

### Lunar power and ISRU

NASA's September 2026 NextSTEP-3 lunar enabling infrastructure call explicitly includes vertical solar arrays with generation, management, distribution and storage; radioisotope Stirling power; regolith oxygen production; advanced manufacturing and nanomaterial production. These are strong indicators that mission architecture must integrate energy with manufacturing and local resource conversion.

NASA's February 2026 CaRD update reports integrated prototype tests using concentrated solar energy and simulated lunar regolith that confirmed carbon-monoxide production. This is a meaningful process-integration milestone, but it is not proof of an operational lunar plant, sustained oxygen production, or locally manufactured photovoltaic modules.

NASA's 40-kWe-class Fission Surface Power programme is a continuous-power reference architecture targeting a lunar demonstration in the early 2030s. Compare it as a firm-power alternative under identical service requirements, not as a solar-cell competitor.

Sources:
- https://www.nasa.gov/news-release/nasa-calls-for-proposals-to-accelerate-lunar-surface-technologies/
- https://www.nasa.gov/centers-and-facilities/johnson/sunlight-extracts-oxygen-from-regolith-using-solar-chemistry/
- https://www.nasa.gov/exploration-systems-development-mission-directorate/fission-surface-power/

### Lunar thermal and long-duration storage

NASA's 2026 Thermal & Fluids Analysis Workshop proceedings include a concept for a radiant heat-pump system with a radiation-only condenser/radiator and preliminary analysis of roughly 15 kW peak habitat heat load. The work is described as concept/model development with future representative testing; it is not a demonstrated lunar product.

A 2026 review of lunar chemical storage notes that regenerative fuel-cell concepts still require long-duration stability work in lunar conditions and that water-ice extraction/beneficiation/electrolysis are key dependencies for an ISRU-supplied loop.

Benchmark requirement: compare storage by delivered service, full round-trip chain, lifetime, reserve, replacement, heat rejection, process dependencies and local consumables. Include a declared firm-power competitor and the long-duration darkness scenario.

Sources:
- https://tfaws.nasa.gov/tfaws-2026-2/paper-proceedings/
- https://www.nature.com/articles/s44453-026-00053-w

### Competitive positioning

The most defensible path beyond the current field is not to claim a universal panel-efficiency record. It is to demonstrate, on common frozen missions and independently checkable boundaries:

1. lower worst-case unserved critical energy at equal installed mass and lifecycle assumptions;
2. lower lifetime burden per delivered electrical and temperature-graded thermal service;
3. more reliable service under dust, degradation, thermal extremes, long darkness and component faults;
4. a higher verified local-production capability with fewer critical imported dependencies;
5. fewer physical experiments to reach a decision of equal or better quality, with all validation costs counted.

A candidate is competitive only when it beats strong baselines on held-out scenarios and retains the hard physical/evidence constraints. A synthetic fixture or a source audit cannot establish these outcomes.

## 14. Runner-independent audit commands

### 14.1 Orbit reference audit

The orbit command checks the analytic inverse-square screening corpus:

```bash
cargo run -p xtask -- planetary-energy-audit
```

To select another input or write the machine-readable report:

```bash
cargo run -p xtask -- planetary-energy-audit \
  --input benchmarks/planetary-energy-service/v0.1/orbit-reference-cases.json \
  --out reports/planetary-energy/orbit-reference-audit.json
```

It refuses malformed schemas, unknown fields, invalid distances/units, duplicate identities, evidence-class promotion and values outside the declared rounding tolerance.

### 14.2 Time-indexed service-envelope audit

The service command evaluates synthetic upstream capability receipts against a frozen mission contract:

```bash
cargo run -p xtask -- planetary-service-audit
```

To write its machine-readable report:

```bash
cargo run -p xtask -- planetary-service-audit \
  --input benchmarks/planetary-energy-service/v0.1/service-envelope-cases.json \
  --out reports/planetary-energy/service-envelope-audit.json
```

The service evaluator checks complete time coverage, interval identity, electric energy after protected reserve, peak electric power, useful thermal energy, peak thermal power and required delivery temperature. It distinguishes:
- `PASS_SYNTHETIC_SERVICE`: all declared interval services meet the synthetic contract;
- `INFEASIBLE`: at least one known delivery quantity is insufficient;
- `BLOCKED_UNKNOWN`: required capability or time coverage is missing;
- `INVALID`: malformed identity, time, value or evidence classification.

It checks declared upstream capability receipts; it does not validate the upstream generation/storage/thermal models or derive their values. This is a join/eligibility checker, not an energy dispatch or thermal solver.

### Tests and claim ceiling

Focused tests are authored under `xtask/src/planetary_energy.rs` and can be run with:

```bash
cargo test -p xtask
```

The orbit and service commands use the existing `serde`, `serde_json`, `sha2` and `anyhow` dependencies. A corpus-level PASS means its expected synthetic/reference dispositions were reproduced; it does not mean all included candidates are feasible. Neither command validates a solar array, power system, battery, radiator, site, material process or ISRU plant.

The service receipt values in the synthetic fixture are deliberately synthetic. Production use must first bind exact source model, configuration, timebase, evidence class, units, uncertainty and capability receipt identity from the upstream owner. A reference to a capability receipt is not proof that the receipt itself is qualified.
