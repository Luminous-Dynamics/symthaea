# MARINE-WAPS-CO-DESIGN-001
## Deep-sea cargo wind-assisted hybrid propulsion co-design

**Status:** research contract v0.1.0  
**Created:** 2026-11-10  
**Owner:** Symthaea research coordination; physical solvers and qualification stay with their existing owners  
**Parent issue:** [#11406](https://github.com/Luminous-Dynamics/symthaea/issues/11406)  
**Case schema:** [marine-propulsion-case.schema.json](./marine-propulsion-case.schema.json)  
**Reference corpus:** [marine-propulsion-reference-baselines-2026-11-10.json](./marine-propulsion-reference-baselines-2026-11-10.json)

> This document defines a research and evidence programme. It does not describe a completed ship design, validated CFD result, physical test, class approval, or construction-ready specification.

## 1. Research objective

Design and evaluate deep-sea cargo-vessel architectures that deliver the same cargo work and service quality with lower **whole-vessel lifecycle energy, cost and environmental burden**.

The object under optimization is the ship-service system:

`cargo demand → hull and loading state → route and metocean conditions → wind and hydrodynamics → propulsors and powertrain → fuel/storage/shore interface → voyage policy → port turnaround → maintenance and availability`

The comparison unit must preserve the service being delivered:

- **Mass cargo:** delivered tonne-nautical-mile.
- **Container service:** delivered TEU-nautical-mile, with TEU definition and utilization stated.
- **Fleet/service outcome:** annual delivered cargo work, missed calls, on-time performance and vessel availability.

A candidate that reduces energy per nautical mile by carrying less cargo, sailing slower than the service contract permits, omitting auxiliary loads, or accepting materially lower availability has not demonstrated a like-for-like improvement.

## 2. Evidence snapshot and research implications

### 2.1 Wind propulsion is deployable, but its net benefit remains route- and vessel-dependent

A 2026 open-access review assembled an inventory of 96 wind-equipped ships by the end of 2025, 90 of which were still operating. It compares Flettner rotors, suction wings, rigid wings, soft sails and kite sails, while identifying integration, reliability, safety, operating and economic questions.

Source: Wang et al., *Wind-propelled ships for maritime decarbonization: Technologies, global deployment, and engineering challenges*, International Journal of Naval Architecture and Ocean Engineering 18 (2026), 100785. https://doi.org/10.1016/j.ijnaoe.2026.100785

Implication: include real deployment and maintenance evidence in the reference corpus. Do not treat wind assistance as purely theoretical, but do not import a generic percentage saving into a new vessel.

### 2.2 Whole-ship aero/hydro/energy co-optimization is already a serious SOTA capability

DLR describes its Odyssa framework as analyzing aero- and hydrodynamics for the overall ship system, generating statistically relevant load, power and movement profiles from the transport task and historical weather, then evaluating and optimizing the energy and propulsion system. Its published description also identifies expansion into stability, safety, manoeuvrability, new energy systems and weather routing.

Sources:
- DLR, *Design and Optimisation of Future-Oriented Ships*: https://www.dlr.de/en/ms/research-transfer/topics/design-and-optimization-of-future-oriented-ships
- DLR, *Setting sail again: how wind power is driving more efficient cargo ships*, 18 June 2026: https://www.dlr.de/en/latest/news/2026/setting-sail-again-how-wind-power-is-driving-more-efficient-cargo-ships

Implication: Luminous must not claim novelty merely for connecting sails, hull, weather and powertrain. Odyssa is a relevant benchmark and possible interoperability target. The opportunity is to make the entire design thread evidence-carrying and externally reproducible: requirements, configuration, solver inputs/outputs, uncertainty, lifecycle economics, manufacturing, inspection, trial and maintenance evidence remain linked through change.

### 2.3 Hybrid-electric optimization has a meaningful route-specific published baseline

Du et al. describe a coastal general-cargo case using multi-year AIS/logbook-derived profiles, diesel generator sets, a DC bus, propulsion motors and battery storage. The article reports, for its Dalian–Yantai study case, 16–21% fuel reduction, 17–20% CO₂ reduction and 8–14% lifecycle-cost reduction for its optimized hybrid configuration relative to its mechanical baseline. The authors also identify limitations including simplified resistance/propulsion modelling, incomplete treatment of fouling/extreme weather/manoeuvring, and perfect-foresight dynamic programming in the offline controller.

Source: Du et al., *Integrated Design and Dynamic Performance Optimisation of Hybrid Electric Propulsion Systems for Coastal Cargo Vessels Under Real-World Operational Profiles*, Applied Sciences 16(10), 4940 (15 May 2026). https://doi.org/10.3390/app16104940

Implication: this is a strong coastal hybrid design/control reproduction target. It is **not** a deep-sea wind-assisted performance estimate, and its percentages must not be carried into another vessel class.

### 2.4 Deep-sea containership economic/service anchors: UCL and Lloyd's Register

The adjacent nuclear merchant-shipping assessment work has already admitted two useful, but non-equivalent, reference cases into the Symthaea marine economics programme: [NUCL-TEA-001 #11343](https://github.com/Luminous-Dynamics/symthaea/issues/11343), [reference verifier PR #11346](https://github.com/Luminous-Dynamics/symthaea/pull/11346).

- **UCL Energy Institute (published 5 November 2026):** large containership class 14,500–19,999 TEU; 25-year lifecycle; a 14.85-knot reference and a 25-knot high-speed case; multiple fuel/policy scenarios; TCO normalized by transport work. Its central conclusion is conditional: without the IMO Net-Zero Framework, no nuclear option is cheaper than conventional ships even at low assumed reactor costs; under the framework, competitiveness appears only with substantial reactor-cost reductions that have not yet been demonstrated. https://www.ucl.ac.uk/bartlett/publications/2026/nov/nuclear-reactors-merchant-shipping-total-cost-ownership-assessment-low-carbon-pathway
- **Lloyd's Register Advisory / LucidCatalyst for Seaspan (public case study, 14 July 2026):** 15,000 TEU at 25 knots; public summary models up to 38% more annual cargo capacity from 6.3 rather than 5 annual round voyages and up to 5% extra container space from removal of fuel tanks. Public summary inputs are insufficient to reconstruct the full model. https://www.lr.org/en/knowledge/case-studies/power-shift-nuclear-power-pathways-for-future-containership-fleets/

These cases must remain separate. UCL's TCO analysis and LR's high-speed cargo-throughput case do not share a fully disclosed common speed, route, capital-cost, service, utilization, fuel-policy or financing boundary. They are useful for testing our ability to preserve distinct assumptions and compare throughput alongside TCO; they are not directly comparable performance points and neither is a wind-assisted benchmark. Reproduce the source-specific cases before creating any bridge scenario. Unknown LR inputs remain unknown; do not infer hidden values.

### 2.5 CFD progress does not remove the need for vessel-specific validation

A 2026 systematic review reports ranges from heterogeneous studies for hull resistance reduction, surface drag reduction and propeller/wake improvements. Those reported ranges are not additive, are not a universal design allowance, and should not be treated as the expected performance of a proposed hull or device.

Source: Ma et al., *CFD-based approaches for low-carbon shipping: A systematic review*, Marine Pollution Bulletin 233 (2026), 120155; Epub 18 July 2026. https://doi.org/10.1016/j.marpolbul.2026.120155

Implication: the benchmark should report error and validation by observable and operating condition, not only one aggregate CFD score.

### 2.6 Commercial deployment is real; public savings claims still need normalization

The industry landscape has moved beyond prototypes into commercial installations and integrated systems:

- **Sea-Cargo / Norsepower, 17 April 2026:** Trans Hav and Trans Sol entered full operations after retrofits with three rotor sails per vessel, alongside solar, battery storage, power management and optimized propulsion. Norsepower reports installation/commissioning without off-hire and mechanical installation in as little as two working days per vessel. Sea-Cargo's own figures, quoted in the supplier announcement, report up to 35% lower CO₂ per tonne transported for the rebuilt vessels and up to 50% total emissions reduction across its upgraded fleet. The same announcement says rotor thrust can be equivalent to about 7,500 kW under favourable wind. These are **operator/vendor-reported combined-programme figures**, not independently audited rotor-only savings. https://www.norsepower.com/post/norsepower-rotor-sails-support-the-launch-of-sea-cargos-trans-hav-and-trans-sol-into-full-operations-following-major-efficiency-upgrade
- **LD Armateurs / Norsepower, 15 October 2026:** six 35 m × 5 m rotor sails were installed on the new RoRo cargo vessel Spirit of Toulouse, first of three low-emission RoRo ships for Airbus transport. The installation is paired with dual-fuel main engines and dual-fuel generator sets capable of using e-methanol, bio-methanol or marine diesel oil. This is an important integration example: wind devices, alternative fuels and ship service must be costed as a system, not as independent savings. https://www.lda.fr/en/ld-armateurs-worlds-largest-wind-propelled-cargo-vessel-fitted-with-norsepower-rotor-sails/
- **ABB / Econowind, 2026:** ABB reports a 2025 supply contract for about 150 VentoFoil units across 50 ships and says SPOS weather routing is used on more than 4,500 vessels. ABB describes VentoFoil fuel reductions of 6–10%. Preserve these as supplier claims and adoption statistics, not independently validated results or evidence that 4,500 ships carry VentoFoils. The commercial opportunity is the joined system: apparent-wind/device operation plus voyage routing and measurable counterfactual performance. https://www.abb.com/global/en/news/137533/how-abb-and-econowind-optimize-the-route-to-wind-assisted-shipping
- **BAR Technologies WindWings:** the vendor lists a 37.5 m × 20 m, 750 m² rigid wing for vessels above 40,000 DWT, with 1.5 tonnes/day average and 2.1 tonnes/day achievable fuel-saving claims; the compact 20–24 m variant is designed to tilt down to deck level for pilotage/dockside operations. Those values are product-page vendor claims; they are not transferable without ship, route, weather, baseline consumption and service normalization. https://www.bartechnologies.uk/commercial-ships/windwings/

These cases tell us where to compete: not by merely proposing a rotor, suction wing or wing sail, but by publishing a **better integrated, independently reproducible, service-matched performance record**. At minimum the evidence package should contain installation downtime, route and apparent-wind distribution, load/payload and speed schedule, device state/availability, parasitic power, engine/shaft/fuel traces, calibration, uncertainty, environmental corrections, maintenance events and a declared emissions boundary.

### 2.7 Safety and regulation are moving, but a work plan is not an operating code

The Maritime Technologies Forum (MTF), which includes flag States and classification societies, published WAPS safety-management guidance on 30 March 2026. It addresses wind-assisted ships using rotor, suction and wing sails; the scope is wind-assisted rather than primary wind-powered vessels. Its relevance is operational: manoeuvrability, visibility, maintenance, emergency preparedness, risk assessment, crew procedures and learning from near-misses. https://www.maritimetechnologiesforum.com/en/news-publications/maritime-technologies-forum-announces-new-guidelines-to-strengthen-safety-management-for-ships-using-wind-assisted-propulsion-systems

Lloyd's Register's LR-GN-044 (2 January 2024) also explicitly addresses WAPS foundations, hull support structure, deck plating and under-deck reinforcement. https://www.lr.org/en/knowledge/lloyds-register-rules/guidance-notes/guidance-notes-on-wind-assisted-propulsion-systems/

The IMO MSC 111 (13–22 May 2026) endorsed an SDC safety-framework work plan covering wind propulsion and wind-assisted power, batteries and swappable traction battery containers, alongside nuclear power. This is a regulatory development programme, **not** a completed harmonized code or specific vessel approval. https://www.imo.org/en/mediacentre/meetingsummaries/pages/temporary-msc-111.aspx

## 3. Paradigm shifts and hypotheses

### P1. Wind propulsion becomes a controllable energy subsystem

For each device, capture:
- aerodynamic forces and moments over apparent-wind direction/speed and control state;
- heel/roll, yaw, leeway and lateral-force consequences;
- rudder, appendage and propulsor penalties;
- parasitic electrical demand (including rotor drive or suction fans);
- cut-in/cut-out, storm, visibility, bridge/port and operational restrictions;
- retraction/deployment time, degraded states, availability and maintenance;
- deck/cargo interference, air draft, structural reinforcement, mass and stability effects.

**Core rule:** aerodynamic thrust is not the same as net propulsion saving. The comparison must include the entire ship and all associated loads.

### P2. Co-design the hull, wind device, rudder and propulsor

A large wind device changes side force, yaw moment, course-keeping demand and potentially the propeller inflow. These interactions can change the optimal device placement, rudder sizing, appendage layout, propeller operating point and control law.

Research the integrated force/moment and energy balance. Do not sum isolated component-efficiency percentages.

### P3. Use mechanical/electric hybridization selectively

Compare on the same route and load profile:
- direct low-speed mechanical drive;
- mechanical shaft with PTO and optional PTI boost;
- modular diesel-electric or other generator-electric layouts;
- hybrid shaft/electric arrangements with battery buffering;
- full-electric powertrain only where route, duration, recharge and service requirements justify it.

Full electric is not presumed to be better for long-haul deep-sea cargo. Conversion losses, machinery mass, redundancy, thermal management, storage volume, battery replacement, serviceability and cargo displacement all count.

### P4. Match storage to the time scale

Batteries may buffer power mismatch, transient loads, port/hotel services and short operating windows. Supercapacitors are a candidate only for demonstrably short high-power events. Neither should be promoted to the role of deep-sea primary energy without a route-specific mass/volume/recharge/economic proof.

### P5. Treat waste heat as an exergy and integration problem

For ORC or other recovery systems, include:
- the actual source temperature, mass flow and duty cycle;
- conversion efficiency at the operating point;
- parasitic pumps, cooling, control and pressure losses;
- mass, volume, back-pressure, fouling and maintenance;
- time correlation with the ship's useful demand.

Recovered heat, electrical output and shaft output are separate quantities. Count each energy stream once.

### P6. Make route policy part of the physical architecture

Co-optimize speed, heading, wind-device trim, weather routing, current, waves, fuel/SOC reserve, arrival windows, port congestion and shore-power access. Compare the same delivery service and report schedule slack. Do not give an optimizer navigation, safety or physical-actuation authority.

### P7. Search a robust Pareto frontier, not one weighted score

Preserve nondominated alternatives over:
- lifecycle cost and energy;
- well-to-wake GHG and local pollutants;
- delivered cargo throughput and service reliability;
- range, speed and arrival compliance;
- payload, cargo volume and stability margin;
- maintenance, availability, crew burden and repairability;
- model/evidence maturity and uncertainty.

Hard feasibility gates are non-compensable. A weighted score must never hide a stability, structural, route, fire, class/flag, port or operating failure.

### P8. Explore surplus-wind energy capture without calling it free energy

A further hypothesis worth testing is **wind-surplus mode**. Under favourable conditions, a wind device may generate more thrust than is needed to maintain the contracted speed. A controllable propulsor/hydrogenerator could then extract a portion of the surplus for electric loads or storage, if the resulting hydrodynamic drag and conversion losses are smaller than the excess wind work available.

This is not a presumed net gain. Compare it against:
1. using the surplus to reduce engine/fuel demand;
2. reducing speed while preserving arrival service;
3. harvesting through a hydrogenerator while maintaining the same speed;
4. retracting or feathering the device.

The candidate must satisfy the same route, arrival and safety contract. Model added drag, conversion losses and operating restrictions; do not double-count wind thrust and recovered electricity.

## 4. Architecture set to benchmark

| ID | Architecture | Purpose |
|---|---|---|
| A0 | Existing conventional slow-speed mechanical drive | Reference baseline |
| A1 | Best-practice hydrodynamic/mechanical design and trim/speed operation | Separates ordinary design improvements from new technologies |
| A2 | A1 + Flettner rotor(s) | Rotor-assisted comparison |
| A3 | A1 + suction wing(s) | Actively controlled wing comparison |
| A4 | A1 + rigid/soft wing or kite | Device-family comparison, subject to vessel and route suitability |
| A5 | Wind-assisted mechanical + PTO/PTI + battery buffer | Hybrid powertrain co-design |
| A6 | A5 + degradation-aware energy management and route/arrival policy | Whole-voyage integration |
| A7 | A6 + waste-heat recovery | Exergy-aware lifecycle comparison |
| A8 | Alternative-fuel variants | Separate fuel/storage/safety/lifecycle branches |
| A9 | Nuclear-electric interface | Separate conditional branch; external plant and hard-gate evidence required |
| AX | Experimental systems (superconducting machines, advanced boundary-layer control, wave-energy harvesting, etc.) | Research-only until independently supported |

The set is a comparison menu, not a proposal to fit every technology to one hull. Run ablations to identify each measure's marginal contribution and interaction effects.

## 5. Fair-comparison accounting

### 5.1 Service fingerprint

Every case pair must match or explicitly explain deviations in:
- cargo class, payload and utilization;
- route, distance, port calls and annual voyages;
- speed and arrival-window policy;
- loading/draft and environmental distributions;
- annual delivered work and availability requirement;
- measurement boundaries and lifecycle horizon.

A nominally equal ship speed is not a sufficient service match. A candidate may be slower or faster yet still deliver the same annual cargo work if voyage scheduling, port time and fleet size differ. Report that distinction explicitly.

### 5.2 Functional units

For a voyage segment:

`W_m = m_{cargo,delivered} \times d_{nm}`

`W_{TEU} = TEU_{delivered} \times d_{nm}`

Do not merge mass and TEU functional units. For a fleet/service alternative, include annual delivered work and vessel count.

### 5.3 Whole-system energy

At minimum preserve distinct meters/boundaries for:
- fuel input by fuel and pathway;
- fuel energy (declared LHV basis);
- prime-mover shaft output;
- PTO/PTI and electrical generation;
- converter/motor/shaft losses;
- propulsor delivered power;
- auxiliary and hotel loads;
- storage charge/discharge and round-trip losses;
- waste-heat recovery net of parasitic consumption;
- shore electricity and its upstream generation pathway.

The primary whole-service comparison is the total input energy and lifecycle impact required to deliver the matched transport service. Avoid counting wind-device thrust as a second energy source if its effect is already reflected in reduced measured/modelled shaft power.

### 5.4 Wind-system net effect

The most defensible system result is a matched counterfactual difference:

`\Delta E_{primary} = E_{primary,A1}(service, metocean) - E_{primary,candidate}(same\ service, metocean)`

The counterfactual must use the same service and weather/sea-state realization, and the candidate energy must include wind-device parasitics, changed rudder/propeller demand, additional machinery, and operating limits. If the physical model cannot represent these interactions, label the net effect unknown.

### 5.5 Lifecycle economics and emissions

Report CAPEX, OPEX, replacement, maintenance, fuel/electricity, port infrastructure, financing/discount assumptions, residual value and decommissioning with an explicit price year and lifecycle horizon. Report tank-to-wake and well-to-wake emissions separately. Fuel production pathway and electricity mix are part of the case, not footnotes.

### 5.6 Trial-evidence and measurement-chain contract

A whole-vessel comparison must be traceable from a design case to the physical signals used to judge it. The case schema therefore supports an optional `trial_evidence` block with:

- exact trial and protocol ID/revision;
- same service fingerprint, paired baseline/candidate case IDs and declared trial stage (bench, model test, sea trial, in-service, published-only, synthetic-only, unknown);
- each channel's physical quantity, unit, instrument identity/revision, measurement location and sample rate;
- calibration record and calibration disposition;
- timebase ID, synchronization method, time-alignment uncertainty, sample window, missing-sample count and raw trace SHA-256;
- channel uncertainty, provenance, limitations and derived-channel lineage;
- raw and processed dataset digests, environmental correction method/revision, and explicit excluded intervals/reasons.

**Critical boundary:** a digest proves identity of bytes presented to the workflow, not the sensor's authenticity. Calibration status does not by itself prove calibration traceability. A schema-valid record is not a valid sea trial. The trial protocol must establish measurement locations, sensor range/accuracy, sampling adequacy, synchronization, uncertainty propagation, environmental corrections, exclusion policy and pre-registered analysis. Report missing/invalid channels rather than aggregating around them.

For the first wind-assist validation campaign, prioritize synchronized:
1. apparent and true wind speed/direction with declared measurement height/location;
2. rotor/wing/suction device state, RPM/suction power, angle/camber, retraction state and controller commands;
3. shaft torque/RPM and main-engine fuel mass flow or independently calibrated fuel-energy measurement;
4. electrical generation, battery SOC/current/voltage/temperature and converter losses;
5. speed through water and over ground, heading, rudder angle, draft/trim/loading;
6. wave/current/weather reference channels and route/arrival/cargo-service record;
7. auxiliary and hotel-load power, plus all wind-device parasitic demand.

The paired candidate and baseline records must be comparable for service, route, loading, environment, sensor calibration and power/fuel boundary. When an exact paired baseline is impossible, the correction/counterfactual model becomes an explicit source of uncertainty and must be independently evaluated. Never present an uncorrected before/after comparison as causal proof.

## 6. SOTA and industry crosswalk

| Capability | Current benchmark / industry reference | Luminous strategy |
|---|---|---|
| Integrated windship aero/hydrodynamics and energy-system design | DLR Odyssa | Do not claim to replace it. Interoperate with specialist solvers and add reproducible evidence/configuration lineage and service economics |
| Hull/propulsor CFD and optimization | SHIPFLOW and CAESES workflows | Consume validated results and publish exact model, mesh, conditions, uncertainty and applicability; no new generic CFD solver |
| Speed-power sea-trial comparison | ISO 15016:2025 and relevant ITTC procedures | Store trial protocol, raw trace digests, corrected observations and uncertainty; qualified methods remain with their owners |
| In-service hull/propeller performance tracking | ISO 19030-1/-2/-3:2016, within their stated scope | Integrate the same-ship time-series indicators with revision, maintenance and service evidence |
| Hybrid electric route sizing and control | Du et al. 2026 coastal general-cargo study | Reproduce its declared case as a coastal benchmark; improve robustness/held-out validation before extending to deep sea |
| Commercial rotor-sail integration | Norsepower / Sea-Cargo Trans Hav and Trans Sol (2026) | Benchmark installation downtime, retrofit integration and multi-measure whole-system evidence; treat emissions percentages as operator/vendor reports |
| Commercial suction wing + weather routing | ABB / Econowind VentoFoil (2025-26) | Compare device operation and routing as a coupled control system; vendor 6-10% claim is not an independent baseline |
| Commercial rigid folding wing | BAR Technologies WindWings | Compare installed geometry, folding, deck/port restrictions and route-normalized net fuel; vendor t/day claims require matched-case evidence |
| Safety-management practice | Maritime Technologies Forum 2026 WAPS SMS guidance; LR-GN-044 | Bind hazard controls, manoeuvrability, visibility, crew training, maintenance, structural foundations and emergency procedures to design configuration |
| Whole-ship efficiency measure taxonomy | IMO GreenVoyage2050 technology portal | Use its categories as a coverage checklist; compare combined measures and interactions on matched service |

Public references:
- DLR Odyssa: https://www.dlr.de/en/ms/research-transfer/topics/design-and-optimization-of-future-oriented-ships
- SHIPFLOW: https://shipflow.se/
- ISO 15016:2025: https://www.iso.org/standard/82080.html
- ISO 19030-1:2016: https://www.iso.org/standard/63774.html
- ISO 19030-2:2016: https://www.iso.org/standard/63775.html
- IMO GreenVoyage2050: https://greenvoyage2050.imo.org/energy-efficiency-technologies-information-portal/
- 2026 wind-propulsion deployment review: https://doi.org/10.1016/j.ijnaoe.2026.100785
- 2026 coastal hybrid-electric study: https://doi.org/10.3390/app16104940

**Competitive claim discipline:** DLR's framework is already an integrated design-optimization competitor. Luminous can outperform a baseline only in explicitly tested dimensions—e.g. better evidence completeness, reproducibility, uncertainty handling, cross-tool traceability, or matched service economics. It cannot claim better hydrodynamic prediction or windship design until blind, independent tests on eligible held-out reference/physical data establish that.

## 7. Benchmark ladder

| Level | Evidence | Permitted claim |
|---|---|---|
| L0 — Contract | Schema validation, unit checks, known-answer arithmetic, adversarial negative controls | Data contract and arithmetic work for fixtures |
| L1 — Published reproduction | Exact source transcription, boundary recreation, sensitivity and independent calculation | Reproduces the declared published case to stated tolerance |
| L2 — Solver verification/validation | Qualified CFD/energy solver, mesh/time-step studies, model-test/reference comparison | Model performance only in the tested domain |
| L3 — Controlled physical trial | Instrumented model/basin/bench or sea trial with pre-registered matched baseline | Measured effect for exact configuration and conditions |
| L4 — In-service evidence | Longitudinal operational traces, weather/loading correction, maintenance/availability | Performance over the covered operating period and fleet/sample |
| L5 — Fleet transfer | Held-out ships/routes/seasons and external review | Generalization only across declared transfer domain |

The levels are not interchangeable. A high-fidelity CFD result is not a sea trial; a sea-trial result on one hull is not fleet reliability.

## 8. Proposed benchmark targets

These are **goals for the programme**, not achieved results or predicted design performance.

### Evidence and correctness
- 100% of comparisons carry a service fingerprint, units, provenance class, model/configuration identity, uncertainty status and limitations.
- Zero silent defaults, missing-as-zero conversions, duplicate identities or unreported exclusions.
- 100% of required route-condition-observable slices are reported.
- Zero false-feasible outcomes in declared route, stability, payload and equipment negative controls.
- 100% arithmetic agreement on hand-computable known-answer fixtures within frozen numerical tolerances.
- Deterministic report serialization/replay where that property is promised.

### Prediction and physical validation
- Freeze a reference dataset and validation envelope before scoring a candidate.
- Initial aspiration: median absolute percentage error no greater than 5% for voyage shaft energy and no greater than 5% for net energy per delivered cargo work **on the declared held-out validation domain**; also report p95 absolute error, signed bias, uncertainty-interval coverage and each route/condition stratum. Do not score this target before a suitable admitted reference dataset exists.
- A claimed improvement must exceed combined metrology/model uncertainty and repeat across independent matched voyages or reference cases.
- No aggregate pass can mask a failed condition, high tail error or out-of-domain prediction.

### Design ambition
- Investigate at least 10% lower whole-service lifecycle primary energy than the best-practice, service-matched mechanical baseline without reducing delivered cargo work, availability or arrival performance.
- Stretch research target: 20%.
- Treat the percentages as search targets, not forecasts. Reject them if the Pareto frontier, uncertainty or physical evidence does not support them.
- Report lifecycle cost and well-to-wake GHG alongside energy. A lower-energy design can still be more expensive or have worse emissions depending on fuel pathway and capital/maintenance.

## 9. Minimum experiment matrix

For each chosen vessel archetype, build matched slices across:
1. wind direction and apparent wind-speed bands;
2. sea-state and wave-heading bands;
3. ballast, partial-load and full-load conditions;
4. route speed and arrival slack;
5. fouled and clean hull/propeller states where reference evidence supports them;
6. each wind-device control/retraction state;
7. normal, degraded and failed wind-device availability;
8. powertrain part-load, transient and reserve states;
9. port/berth/shore-power availability;
10. annual and seasonal route/weather distributions.

Do not synthesize high-fidelity hydrodynamic truth. Synthetic cases are for software contract and stress testing only.

## 10. Owner boundaries across Luminous

- **Symthaea:** requirements, evidence admission, architecture alternatives, source/model identity, service matching, uncertainty reasoning, falsification and decision support.
- **Symtropy:** deterministic physical execution/replay where models are owned and validated; energy flow and resource schedules within their documented domains; manufacturing/process/inspection/test evidence.
- **Sol-Atlas:** geometry and spatial context—hull, device placement, deck/cargo interference, route/port spatial constraints and revision-bound as-designed/as-built relationships.
- **Mycelix:** supplier/component provenance, manufacturing and inspection records, rights/attestations, warranty and lifecycle obligation/commitment evidence.

No new general CFD solver, weather router, grid solver, optimizer, ledger or approval authority is created here. Integrate specialist software and existing domain owners.

## 11. Execution sequence

1. Use the case schema as the first contract; validate and adversarially test it before populating real cases.
2. Freeze the reference baseline register as a source-attributed corpus. Preserve per-source metrics, scope, evidence class, eligible uses and inference limits; never average unlike studies.
3. Use the existing Symthaea nuclear merchant-shipping reference registry to keep UCL and Lloyd's Register cases separate; link those source records into the marine propulsion comparison without conflating economics.
4. Freeze the service fingerprints and architecture matrix.
5. Implement independent arithmetic for service work, primary energy boundaries and net counterfactual differences.
6. Reproduce published references before attempting design optimization.
7. Add per-condition held-out validation and matched-trial eligibility.
8. Only then compare new candidate designs using a specialist qualified solver or research partner.
9. Plan physical trials with qualified naval architects, hydrodynamicists, operators, electrical engineers, shipyards and class/flag stakeholders.

## 12. Claim ceiling

A schema-valid case establishes only that the case conforms to a declared data structure. A deterministic arithmetic pass establishes only the specified arithmetic. A published-source reproduction establishes only the reproduced study boundary. A solver validation score establishes only the exact observables and tested domain. None independently establishes seaworthiness, structural integrity, stability, fire safety, class/flag approval, insurance, environmental compliance, physical reliability, or permission to operate.

Every public result must state the exact vessel/configuration, mission, environment, model revision, data rights, evidence stage, uncertainty and limitations.

## 13. Artifacts and current execution state

- [Machine-readable case schema v0.1.0](./marine-propulsion-case.schema.json) — retained unchanged for compatibility.
- [Machine-readable case schema v0.2.0](./marine-propulsion-case-v0.2.0.schema.json) — adds calibration/timebase registries, observed/expected sample counts and sample-completeness policy.
- [Synthetic v0.1.0 fixture](./fixtures/synthetic-waps-case-0001.json)
- [Synthetic v0.2.0 fixture](./fixtures/synthetic-waps-case-v0.2.0.json)
- [Source-attributed reference baseline registry](./marine-propulsion-reference-baselines-2026-11-10.json)
- [Rust semantic validator](../../tools/marine-propulsion-validator/README.md)
- [Route/service robustness specification](./MARINE-SERVICE-ROBUSTNESS-001.md)
- [Operational prediction benchmark](./MARINE-OPERATIONAL-PREDICTION-BENCHMARK-001.md)
- [Hydrodynamic validation specification](./MARINE-HYDRODYNAMIC-VALIDATION-001.md)
- [Canonical service fingerprint profile](./MARINE-SERVICE-FINGERPRINT-PROFILE-001.md)
- [Canonical service fingerprint vector](./fixtures/service-fingerprint-v1-vector-0001.json)

The registry now has 23 source records. The v0.1.0 schema remains unchanged; v0.2.0 introduces resolvable calibration/timebase registries, explicit sample completeness, a versioned canonical service fingerprint and a distinct not-applicable quantity state. The validator recomputes the service fingerprint from the declared service/route/loading payload using Symthaea's existing `symthaea-canonical-json-sha256-v1` profile (not RFC 8785); weather/metocean realization remains a separate counterfactual boundary. The v0.2.0 fixture carries a known-answer digest. A standalone Rust validator and integration/adversarial tests are committed. JSON artifacts have been retrieved and structurally inspected, but full JSON Schema meta-schema validation and Rust compilation/test execution have not been run in this environment. The isolated Rust crate still needs a generated Cargo.lock. Fixtures contain synthetic data and placeholder evidence digests; they are not vessel configurations or physical measurements.

