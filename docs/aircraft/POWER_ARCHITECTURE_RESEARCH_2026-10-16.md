# Aircraft Electrification and Future Nuclear Power: Architecture Research

**Status:** public research specification; no aircraft design or propulsion performance has been qualified by this document.  
**Date reviewed:** 2026-10-16  
**Tracking issue:** [AIRCRAFT-ENERGY-001 #7646](https://github.com/Luminous-Dynamics/symthaea/issues/7646)  
**Parent architecture:** [TRANSPORT-ENG-001 #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244)  
**Fixed-wing domain:** [TRANSPORT-FIXED-WING-001 #6249](https://github.com/Luminous-Dynamics/symthaea/issues/6249)

## Executive decision

Symthaea should pursue a staged aircraft-energy program with distinct evidence levels:

1. **Battery-electric:** evaluate now for small uncrewed and short-range utility missions where the whole-aircraft mass, usable energy, power, reserve, and turnaround close.
2. **Hybrid-electric and turboelectric:** evaluate as the strongest nearer-term paths for larger aircraft and longer missions, using mission-specific comparisons rather than assuming that more electrification always saves energy or cost.
3. **Offboard nuclear/fusion electricity:** model as a source of grid electricity for charging or synthetic-fuel production. This is a separate question from carrying a reactor on the aircraft.
4. **Onboard fission / small modular reactor (SMR):** retain only as a system-level research feasibility screen unless a credible complete installed plant closes the mass, specific-power, thermal, accident, maintenance, and regulatory budgets.
5. **Onboard fusion:** keep as speculative, long-horizon research. Fusion fuel energy density does not establish a flightworthy reactor, net electric output at aircraft scale, low installed mass, manageable heat rejection, or safe operation.

The program should not begin by selecting a propulsion technology. It should begin with a frozen mission, derive the required power and energy profiles, compare feasible architectures under the same accounting boundaries, and maintain a Pareto frontier across payload, range, cost, emissions, maintainability, infrastructure, and evidence maturity.

## 1. The fundamental engineering distinction: energy versus power

Aircraft energy systems have to satisfy several different constraints simultaneously.

- **Energy** is the amount of work available over the mission, commonly expressed in Wh or J.
- **Power** is the rate of delivering that work, expressed in W. Takeoff and climb can require much greater power than steady cruise even when cruise consumes most of the mission energy.
- **Specific energy** is energy per unit mass, expressed in Wh/kg. Cell-level and complete-pack values are not interchangeable.
- **Specific power** is power per unit mass, expressed in W/kg. A high-energy source that cannot provide the required peak and continuous power is not a viable aircraft source.
- **Installed-system performance** includes converters, motors, generators, wiring, cooling, mounts, protections, shielding or containment where applicable, and structural changes. Component data sheets do not establish aircraft-level performance.

A first-order model must calculate energy and power margins by mission segment, then account for reserve policy, temperature, degradation, conversion efficiency, and system mass. It must not infer feasibility from nominal energy capacity alone.

For each candidate, preserve at least:

- payload and useful volume;
- empty and maximum takeoff mass, including mass-growth allowance;
- center-of-gravity, inertia, and loading envelope;
- peak, continuous, and transient propulsion power;
- energy required by each mission segment and by auxiliaries;
- usable energy after reserve, temperature, aging, and conversion losses;
- cooling/heat-rejection hardware and any associated drag;
- energy turnaround, charging or fueling infrastructure, and dispatch constraints;
- maintenance, replacement, lifecycle cost, and upstream energy assumptions.

## 2. Architecture comparison

| Architecture | Best initial question | Main potential advantage | Dominant constraints |
|---|---|---|---|
| Battery-electric | Can this specific mission close with safe pack-level energy and sufficient peak power? | High motor efficiency, mechanical simplicity, local zero-emission operation | Pack mass/volume, reserve, aging, thermal safety, charging and turnaround |
| Parallel hybrid-electric | Can a fuel engine and electric motor share shaft power more efficiently across the duty cycle? | Electric boost with less battery energy than all-electric | Extra components, control complexity, mass, operating-point interactions |
| Series hybrid-electric | Can engine/generator sizing and electric distribution improve mission-level efficiency or control? | Electric motors drive propulsors; power generation can be decoupled from propulsor placement | Generator/conversion losses, cooling, mass, generator failure modes |
| Turboelectric / distributed electric | Does airframe and propulsor co-design reduce total fuel or improve other mission objectives? | Flexible propulsor placement and possible integration with boundary-layer ingestion or distributed propulsion | Motor/generator specific power, electrical losses, thermal systems, integration risk |
| Offboard nuclear/fusion electricity | Does low-carbon, reliable electricity improve the charging/fuel-production scenario? | Decouples grid energy source from aircraft certification and onboard powertrain | Grid availability, cost, upstream lifecycle impacts, airport infrastructure |
| Onboard fission-electric | Can a complete reactor plant meet the aircraft's mass/specific-power and safety budgets? | Extremely high fuel-level energy density | Shielding, containment, conversion equipment, cooling, accident consequences, maintenance and regulation |
| Onboard fusion-electric | Is there credible plant-level evidence for net useful electricity at flight-compatible installed mass? | Large theoretical fuel-energy potential | Fusion power-plant readiness, confinement, magnets or other hardware, fuel cycle, conversion, heat rejection, mass and safety |

NASA describes all-electric, parallel/series hybrid, and turboelectric configurations in its Electrified Aircraft Propulsion program. The architecture labels are useful starting categories, not proof that one system is generally superior.

## 3. What the current evidence does and does not say

### Electrified propulsion is an active engineering field

NASA reported on 2026-07-20 that a megawatt-class hybrid-electric propulsion demonstrator developed with GE Aerospace flew on a modified Saab 340B, including flight above 30,000 ft. That materially reduces uncertainty about whether this class of integrated propulsion system can be tested in flight. It does **not** establish certification, dispatch reliability, commercial economics, or fuel savings for any aircraft design we propose.

Source: [NASA, “NASA, GE Aerospace Work Enables Hybrid-Electric Flight Demonstration” (2026-07-20)](https://www.nasa.gov/aeronautics/electrified-aircraft-propulsion/nasa-ge-hybrid-electric-flight/).

NASA's pack-level battery requirement charts use different aircraft and mission assumptions. They show values around 300 Wh/kg for some 30–50-seat hybrid concepts and above 400 Wh/kg for certain single-aisle, takeoff-assist concepts. These numbers are context-specific reference points, not universal cutoffs and not evidence that any battery meeting one number makes an aircraft feasible.

Source: [NASA Glenn electrified-aircraft introductory material, battery-specific-energy requirement chart](https://www1.grc.nasa.gov/wp-content/uploads/1.-NASA-Intro.pdf).

NASA's battery projection work also distinguishes cell-level estimates from aircraft-usable system performance. A projection for a cell must not be copied directly into a model as a pack-level aircraft value.

Source: [NASA Technical Reports Server, “Battery Key Performance Projections Based on Historical Trends and Chemistries”](https://ntrs.nasa.gov/citations/20220005588).

### Thermal management is an aircraft-level constraint, not an accessory

NASA's HEATheR project explicitly targets megawatt-scale electrified propulsion thermal management: conversion losses become heat, while pumps, heat exchangers and ducting add system mass and drag. NASA investigates reducing conversion losses and using local/passive heat rejection to reduce those penalties. Therefore, a motor/generator efficiency gain cannot be counted as a net aircraft gain without accounting for the installed heat-rejection architecture, its mass, and its aerodynamic effect.

Source: [NASA, High-Efficiency Electrified Aircraft Thermal Research (HEATheR)](https://www.nasa.gov/eap-technology/heather/).

FAA guidance also treats electric power systems and propulsion systems as distinct technical disciplines, including electric and hybrid propulsion, energy storage, generation, safety, performance, and certification considerations. The screening kernel must preserve these as connected but non-interchangeable evidence owners.

Sources: [FAA, Electric Power Systems](https://www.faa.gov/aircraft/air_cert/step/disciplines/electric_power); [FAA, Propulsion Systems](https://www.faa.gov/aircraft/air_cert/step/disciplines/propulsion_systems).

### Fusion energy is not yet onboard aircraft propulsion

The US Department of Energy's finalized Fusion Science and Technology Roadmap, released 2026-06-09, describes an intended pathway toward fusion pilot plants and commercial power in the mid-2030s, contingent on closing outstanding science, materials, engineering, infrastructure, workforce, and commercialization gaps. This is a strategic target for stationary power plants, not an aircraft-propulsion readiness estimate.

Source: [DOE, “Energy Department Releases Finalized Fusion Science and Technology Roadmap” (2026-06-09)](https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate).

A fusion aircraft screen must use the mass and net electrical output of a complete plant, not fuel-level energy density or plasma/thermal output alone. Unless credible plant-level evidence exists for sustained net electrical output, complete mass, power conversion, heat rejection, confinement, fuel cycle, maintainability, and safety, its output should remain **research-only / indeterminate**.

### Historical nuclear-aircraft work is a warning about system closure

The historical US Aircraft Nuclear Propulsion program examined reactor concepts, shielding, material limits, containment, and aircraft integration but did not yield an operational nuclear-powered aircraft. This history is useful as a record of the engineering and safety problems; it is not a modern mass model or proof that every future concept is impossible.

Source: [Air Force Materiel Command, “History in Two: Manned Nuclear Aircraft Program”](https://www.afmc.af.mil/News/Article-Display/Article/2664365/history-in-two-manned-nuclear-aircraft-program/).

For present-day assessment, an SMR label must not be interpreted as “light enough to fly.” A reactor intended for stationary or maritime service may have entirely unsuitable system mass, shielding, containment, cooling, maintenance, or accident characteristics for an aircraft. No design should advance on core mass alone.

## 4. Paradigm shifts worth pursuing

### A. Mission-first co-design, not propulsion-first design

Freeze a useful mission and optimize the whole aircraft-energy system together. Wing loading, aerodynamic efficiency, payload layout, propulsor placement, thermal systems, battery mass, reserve policy, runway/launch requirements, and turnaround infrastructure interact. Their trade-offs must be visible in the same evaluation.

A lighter battery may make a larger wing or different propulsor advantageous; that change may alter drag and structural mass. Hybridization may reduce battery capacity while adding generator, fuel, and cooling mass. These effects should be evaluated together rather than added as isolated percentage bonuses.

### B. Treat the charger, depot, and duty cycle as part of the aircraft system

A mission that is infeasible with current battery technology might be viable with shorter legs, payload exchanges, standardized packs, predictable charging windows, or a different dispatch schedule. These are operational architecture options, not free performance improvements: they need their own inventory, turnaround, infrastructure, energy, availability, cost, and safety accounting.

Compare the same service delivered using different combinations of aircraft, charge infrastructure, turnaround, and schedule. Do not hide ground-system cost outside the boundary merely to make an aircraft look cheaper.

### C. Optimize peak power and total energy separately

Battery sizing must satisfy both the usable-energy requirement and the peak-power profile. A hybrid architecture may use a battery for short-duration peak power while a generator supplies most cruise energy. Conversely, a generator that is appropriately sized for cruise may be insufficient for climb or go-around unless the design includes a valid peak-power path.

The tool should reject candidates that satisfy total mission energy but fail any required peak/continuous power condition.

### D. Use multi-fidelity analysis and active experiment selection

Use fast analytical calculations for candidate screening, then higher-fidelity aerodynamic, structural, propulsion, electrical, and thermal analysis on promising candidates. Calibrate those models against authoritative references and measurements before treating rankings as meaningful.

Symthaea can prioritize the next test by expected information gain: which wind-tunnel, component, thermal, hardware-in-loop, or flight-test observation would best distinguish competing designs or reduce a consequential uncertainty? This is a test-selection hypothesis to benchmark, not an assumed Symthaea capability.

Never let a surrogate model silently replace its domain solver. Preserve model version, applicability envelope, input provenance, uncertainty, and prediction-versus-observation residuals.

### E. Optimize for manufacture, inspection, repair, and availability

Report installed mass, unit cost, manufacturing process constraints, inspectability, component replacement, degradation, and maintenance downtime alongside aerodynamic performance. A nominally efficient craft that requires rare materials, complex assembly, or frequent specialist repair may be worse for the intended service than a simpler design.

### F. Make the engineering record part of the architecture

Use Symthaea for candidate generation, analysis orchestration, uncertainty, experiment choice, and trade-space exploration. Use Mycelix to preserve versioned configuration identity, material and component provenance, test receipts, inspections, maintenance history, and externally issued approvals.

A provenance receipt demonstrates what was recorded and by whom; it does not prove that a calculation is valid. A model result does not establish physical qualification. Neither system may manufacture airworthiness, reactor safety, or permission to operate.

## 5. First benchmark program

The first demonstrator should be a **small, fixed-wing, civilian uncrewed utility/cargo mission**, chosen for low consequence and measurable outcomes. It is a starting benchmark, not a declaration that the repository already has a qualified aircraft design. Freeze the exact mission, source dataset, configuration, assumptions, and accounting boundary before scoring architectures.

### Required inputs

The case manifest must state payload, range, speed, climb/takeoff profile, environment, reserve/diversion rule, duty cycle, mass/CG envelope, energy/power limits, battery assumptions, charging/refueling availability, maintenance and replacement assumptions, lifecycle-cost boundary, and evidence quality for each input. Unknown values must stay unknown; they must not silently become zero or favorable assumptions.

### Required metrics

1. **Mission feasibility:** each segment's energy and peak/continuous power margins, including required reserves and degradation.
2. **Complete-system mass and volume:** energy storage/source, conversion, cooling, electrical distribution, protections, and mounting/structural changes.
3. **Useful mission output:** payload-range, endurance, turnaround, and successfully completed missions under the declared conditions.
4. **Economics:** lifecycle cost per successful mission and cost per payload-distance, including maintenance, replacements, energy, charging/fueling infrastructure, downtime, and financing assumptions where applicable.
5. **Environmental accounting:** operational fuel/emissions and lifecycle emissions as separate results with upstream energy-source provenance.
6. **Robustness:** sensitivity to payload, temperature, headwind, reserve, degradation, component efficiency, and power availability.
7. **Model validity:** prediction errors against held-out reference or test data, with uncertainty intervals and applicability limits.
8. **Evidence quality:** source provenance, versioned inputs, model identity, reproducibility, and unresolved gaps.

Do not compress these independent quantities into a single “aircraft intelligence” or “best propulsion” score. Preserve a Pareto frontier and reject candidates that violate hard constraints regardless of their weighted economic score.

### Proposed development targets

These are aspirational project targets, not standards and not measured results:

- Canonical, deterministic input manifest and reproducible screening outputs.
- Independent arithmetic oracle for energy, power, reserve, and mass accounting.
- Zero hard-constraint violations among candidates labelled feasible in the qualified test corpus.
- 100% traceability for every material input and every published comparison.
- After calibration, target no more than 5% absolute prediction error for mission energy on a held-out reference set only where data quality and model scope justify that tolerance. Otherwise publish metric-specific bounds instead of forcing the target.
- Stretch objective: at least 10% lower independently evaluated lifecycle cost per successful mission, or at least 10% improved payload-range at comparable cost, with the same declared mission, reserve, performance, and hard constraints.

The 10% target is a hypothesis for the design search to attempt to falsify. It must not be claimed until an independent, like-for-like evaluation reproduces it.

## 6. Mandatory hostile controls

At minimum, the benchmark must include:

1. Total energy passes but takeoff/climb peak power fails.
2. Energy demand exceeds usable energy after reserve, temperature, or end-of-life derating.
3. Battery nameplate/cell specific energy is substituted for qualified pack-level specific energy.
4. Reactor thermal output is misrepresented as net electrical power at the propulsion bus.
5. Onboard fission/fusion plant mass omits shielding, containment, conversion, cooling, controls, and electrical distribution.
6. An offboard nuclear/fusion electricity scenario is incorrectly described as an onboard-reactor aircraft.
7. A nuclear/fusion candidate lacks credible evidence for complete-plant net electric output or installed mass but is still labelled feasible.
8. A missing efficiency, mass, weather input, or source is silently filled with zero or a favorable assumption.
9. Lower direct energy cost hides payload loss, extra charger/depot cost, or excessive turnaround.
10. A weighted score hides a violated hard constraint.
11. The candidate is outside the aerodynamic or structural model's declared applicability envelope.
12. A simulation pass is incorrectly promoted to physical qualification, certification, or operating authority.
13. Identical display names hide changed battery, motor, generator, controller, geometry, firmware, or material-batch generations.
14. Reported improvement disappears under a fair conventional baseline or held-out case.

Expected outcomes must be frozen before full evaluation. The independent verifier must not simply call the production reducer or optimizer being tested.

## 7. Stage gates

**Gate 0 — reference mission and evidence freeze.** Select one low-consequence mission; document sources, units, assumptions, uncertainty, and explicit nonclaims.

**Gate 1 — deterministic screening.** Implement only transparent calculations with defined units and independently checked energy, power, reserve, mass, and cost accounting.

**Gate 2 — benchmark qualification.** Run positive controls, hostile inputs, held-out cases, and comparison against conventional methods. Preserve results, negative controls, exact revisions, and corpus counts.

**Gate 3 — calibrated simulation.** Compare with qualified analysis tools and reference/experimental data. No design-quality claim until applicable model error and uncertainty are reported.

**Gate 4 — physical evidence.** Conduct appropriately reviewed component, ground, hardware-in-loop, and prototype tests. Any flight activity requires qualified engineering and applicable operating approvals.

**Gate 5 — scale decision.** Proceed to larger payload/range classes only if physical evidence and a consistent lifecycle comparison justify the move.

A queued CI run is not a pass. An absent test or uncalibrated model is not evidence. Physical testing and required external approvals remain independent gates.

## 8. Mycelix evidence projection

The cross-system record should keep separate, linked evidence objects for:

- mission specification and requirements revision;
- aircraft geometry/mass/configuration generation;
- energy source, pack, motor, inverter, generator and thermal-system generations;
- material/component provenance and inspection;
- analysis model, version, input manifest, solver results, applicability envelope and uncertainty;
- test fixture, instrumentation/calibration, environmental conditions, measurements and residuals;
- maintenance, replacement, degradation and requalification;
- external engineering review and approvals from the actual responsible authority.

Use scoped disclosure for commercially sensitive component and supplier data and for aircraft telemetry that could expose people, cargo, or operational locations. A public provenance graph is not permission to publish all source records.

## 9. Research references

1. NASA, [Electrified Aircraft Propulsion](https://www.nasa.gov/mission/eap/) — propulsion architectures, technology components, testing and flight demonstrations.
2. NASA, [Aircraft Configurations](https://www.nasa.gov/eap-aircraft-concepts/aircraft-configurations/) — all-electric, parallel/series hybrid, partial hybrid and turboelectric definitions.
3. NASA, [NASA/GE Aerospace Hybrid-Electric Flight Demonstration (2026-07-20)](https://www.nasa.gov/aeronautics/electrified-aircraft-propulsion/nasa-ge-hybrid-electric-flight/).
4. NASA Technical Reports Server, [Electrified Powertrain Flight Demonstration executive summary](https://ntrs.nasa.gov/citations/20250008745).
5. NASA Glenn, [Battery specific-energy requirements for electrified aircraft](https://www1.grc.nasa.gov/wp-content/uploads/1.-NASA-Intro.pdf) — scenario-dependent pack-level references.
6. NASA Technical Reports Server, [Battery Key Performance Projections Based on Historical Trends and Chemistries](https://ntrs.nasa.gov/citations/20220005588).
7. US Department of Energy, [Finalized Fusion Science and Technology Roadmap (2026-06-09)](https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate).
8. Air Force Materiel Command, [History in Two: Manned Nuclear Aircraft Program](https://www.afmc.af.mil/News/Article-Display/Article/2664365/history-in-two-manned-nuclear-aircraft-program/).
9. NASA, [High-Efficiency Electrified Aircraft Thermal Research (HEATheR)](https://www.nasa.gov/eap-technology/heather/) — thermal-management mass, drag, and power-loss coupling.
10. FAA, [Electric Power Systems](https://www.faa.gov/aircraft/air_cert/step/disciplines/electric_power) — energy storage/generation and certification discipline.
11. FAA, [Propulsion Systems](https://www.faa.gov/aircraft/air_cert/step/disciplines/propulsion_systems) — electric/hybrid propulsion and lifecycle safety/certification discipline.
12. Mycelix, [AeroCommons engineering and manufacturing commons](https://github.com/Luminous-Dynamics/mycelix/issues/3638) and [physical-evidence reproducibility challenge](https://github.com/Luminous-Dynamics/mycelix/issues/3643) — provenance boundary for as-built configurations and test evidence.

## Deterministic design-space comparison status (2026-10-17)

The aircraft-energy research branch now includes four distinct layers:

1. **Frozen-mission envelope screening**: check energy after reserve, continuous and peak power, peak duration, installed-system mass, and explicit missing/invalid inputs.
2. **Same-profile candidate Pareto comparison**: compare candidate energy-system envelopes on one supplied mission profile without collapsing mass/margin trade-offs into a weighted score. Direct comparisons fail if valid candidates use different reserve policies.
3. **Bounded battery sensitivity grid**: enumerate pack specific energy, installed-system/pack mass ratio, temperature capacity fraction, end-of-life state of health, reserve fraction, and power derating, with a hard maximum of 2,000 generated combinations because the current exact Pareto comparison is quadratic. Larger experiments need reviewed batches and a verified frontier-merging procedure. Invalid configurations stay visible and never enter a frontier. The sweep creates a separate Pareto frontier for each reserve fraction; it does not claim a global frontier across different reserve policies.
4. **Candidate-specific profile comparison and evidence contract**: one API compares separately supplied mission profiles under a shared caller-declared use-case family, with a matching family digest, ordered phase IDs/duty classes, and reserve policy. Both direct pairwise comparison APIs cap one comparison at 2,000 candidates because the current exact frontier algorithm is quadratic; larger studies require a separately verified frontier-merging algorithm. The manifest validator checks equality bindings for configuration/energy/route/mission-family/mass/environment/phase-demand artifact digests, artifact metadata, mutually exclusive mass closure, maximum mass limits, CG datum, environment frames, reserve/mass consistency and phase uncertainty intervals before returning the mission profile alongside its evidence manifest. This checks record consistency, not aerodynamic or propulsion model accuracy.

**Important limitation:** the fixed-profile sensitivity grid does not recompute aircraft lift, drag, climb, transition, or required bus power after changing system mass. The candidate-specific API only consumes upstream mission profiles; it does not generate or validate them. A common family identifier is bookkeeping, not proof that route, payload, weather, reserves, acoustic limits, charging losses, or certification assumptions are equivalent. Do not interpret these screening outputs as an aircraft-level optimum, energy estimate, performance guarantee, safety case, or design validation.

Next gates are an independently reviewed, candidate-specific reduced-order mission model with explicit mass/payload/aerodynamic coupling; a frozen dataset and units/provenance manifest; sensitivity and adverse-case tests; and comparison against a conventional road vehicle plus conventional aircraft baseline. Until such a model exists, treat the sweep as an engineering bookkeeping and early rejection tool only.

## Claims this document does not make

This document does not claim that Symthaea has designed a superior aircraft, that any proposed propulsion architecture is feasible for a selected mission, that onboard SMR/fission or fusion propulsion is near-term, or that any aircraft is safe, certifiable, economical, or ready to build. Those claims require qualified models, independent evaluation, physical evidence, and relevant external approvals.
