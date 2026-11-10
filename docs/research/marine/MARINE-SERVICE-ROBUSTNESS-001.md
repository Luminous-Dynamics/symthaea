# MARINE-SERVICE-ROBUSTNESS-001
## Route-adaptive cargo service, uncertainty and degraded-operation co-design

**Status:** research specification v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Validator dependency:** MARINE-WAPS-CASE-VALIDATOR-001A (#11412)

> Research specification only. This is not a ship design, validated route optimizer, CFD result, fuel-saving forecast, safety case or class/flag approval.

## 1. Paradigm shift: design for a service distribution, not an average voyage

The first programme correctly changes the optimization target from component efficiency to delivered cargo work. The next step is to stop optimizing against a single nominal route or mean weather year.

A commercial vessel experiences distributions of weather, loading, port delay, equipment availability, fuel and shore-energy access, forecast error and maintenance interruption. A design that wins on mean fuel consumption but fails in a tail condition may require extra reserve, miss a berth window, lose a voyage or consume its savings through recovery operations.

Design against a declared route-and-service portfolio:

- annual cargo demand, voyage schedule and arrival-window contract;
- distributions of route, load, metocean and port conditions;
- forecast uncertainty, forecast degradation and observation loss;
- normal, degraded, unavailable and recovery modes;
- lifecycle cost, primary energy and well-to-wake emissions;
- evidence quality and uncertainty for every scored outcome.

The winning design is not necessarily the one with the lowest mean voyage energy. It is the one that delivers the contracted cargo service with the best defensible lifecycle frontier while maintaining hard safety and operational constraints.

## 2. Three coupled decision horizons

### 2.1 Design horizon — months to decades

Design variables include hull and appendage configuration, propulsor, wind-device family and placement, structural foundations, stowage/retraction, prime movers, PTO/PTI machinery, DC-bus topology, storage, fuel tanks, cargo-space allocation, shore connection and maintainability.

Count installed mass and volume, deck obstruction, air draft, bridge/port restrictions, cargo interference, stability, structural loads, crew workload, inspection access and outage time. Evaluate against the route portfolio, not a single representative operating point.

### 2.2 Voyage horizon — days to weeks

Voyage decisions include route, departure time, speed schedule, heading, wind-device operating state, reserve policy, powertrain dispatch, battery state-of-charge trajectory and port/shore-energy choices.

Use receding-horizon planning or another qualified approach that updates as forecasts and observations change. A clairvoyant perfect-foresight controller may be an upper-bound reference, but must never be presented as an operational baseline.

### 2.3 Supervisory/control horizon — seconds to minutes

Supervisory decisions may include device trim, rotor/suction state, power split, generator loading, battery charge/discharge and fallback transitions. The system must detect stale inputs, model-domain violations and unavailable equipment.

This programme does not give a research optimizer direct navigation, propulsion, electrical-switching or safety authority. Any eventual control integration requires independent hazard analysis, class/flag and owner review, bounded actuation interfaces, human override and configuration-specific physical qualification.

## 3. Falsifiable research hypotheses

**H1 — Forecast value depends on architecture.** Better weather forecasts may change the value of wind devices, battery size, fuel reserve and powertrain topology. Measure incremental value by controlled degradation of forecast horizon, bias, variance and missingness.

**H2 — Wind hardware and operating policy are inseparable.** Net value depends on heading, route, heel/yaw/leeway, rudder demand, propulsor inflow, device trim/retraction, parasitic power and schedule slack. Device thrust alone is not a ship-level outcome.

**H3 — The best architecture may be a portfolio of modes.** A wind device may be useful on some route/season segments and stowed or feathered in others. A hybrid battery may be valuable as a transient buffer without being sized to replace deep-sea primary energy.

**H4 — Degraded-mode resilience has economic value.** Wind-device outage, battery derating, converter failure, forecast loss, unavailable shore power or port restriction can change annual delivered work. Measure recovery penalty and service impact, not only nominal fuel.

**H5 — Payload and turnaround can reverse an energy win.** Added mass, volume, structural foundation, stowage, inspection access and port restrictions can reduce payload or availability. Count cargo displaced and annual voyages lost.

**H6 — Surplus-wind energy recovery is not free energy.** Compare reducing engine power, reducing speed while preserving contracted arrival, and hydrogeneration after including added drag and conversion losses. DLR's Hyro work is a precedent for investigation, not proof that recovery benefits another hull or route.

## 4. Risk-aware objective and reporting

Keep the objective multi-dimensional. Report:

- lifecycle cost per delivered tonne-nautical-mile or TEU-nautical-mile;
- primary energy per delivered cargo work;
- well-to-wake GHG intensity and fuel/electricity pathway;
- annual delivered cargo work and vessel count;
- on-time probability and p50/p90/p95 lateness;
- availability, unplanned downtime and WAPS availability;
- payload, cargo volume and cargo utilization;
- degradation, replacement and maintenance burden;
- range, reserve margin and fallback capability;
- stability, structural, manoeuvring, fire, visibility and operational hard gates;
- prediction error, uncertainty coverage and evidence completeness.

For a declared service contract, report expected performance and tail risk such as CVaR at 95% for energy, cost or delay penalty. Freeze the loss function, risk measure and confidence level before candidate scoring.

A candidate research formulation is:

minimize Pareto(
  lifecycle cost / delivered work,
  primary energy / delivered work,
  well-to-wake GHG / delivered work,
  CVaR_95(service penalty)
)

subject to matched cargo work, arrival policy, route safety, stability, structure, range, reserve, port and regulatory constraints.

A chance constraint such as P(on-time arrival) >= target is meaningful only when the service owner specifies the target and the weather/port dataset represents the intended operating domain. Do not invent a universal punctuality threshold.

## 5. Matched experiment ladder

Use one vessel archetype and one service fingerprint at a time. Match cargo class, declared payload/utilization, route and port calls, annual delivered work, arrival policy, loading distribution, weather realization or declared counterfactual correction, and lifecycle boundary.

| ID | Design and policy | What it isolates |
|---|---|---|
| B0 | Conventional mechanical propulsion, current operating policy | Existing reference |
| B1 | Best-practice hull/propeller, trim, fouling management and speed policy | Ordinary efficiency opportunity |
| B2 | B1 + weather routing | Route-policy value without new wind hardware |
| B3 | B1 + WAPS with fixed/standard operating policy | Hardware contribution |
| B4 | B1 + WAPS with weather-aware trim/retraction/heading | Wind-control interaction |
| B5 | B1 + hybrid/PTO-PTI/battery with rule-based dispatch | Powertrain contribution |
| B6 | B1 + WAPS + hybrid with forecast-informed dispatch | Combined architecture value |
| B7 | B6 with degradation-aware storage sizing and replacement | Lifecycle battery trade-off |
| B8 | B6 with declared degraded/failure modes and reserve policies | Resilience and service penalty |
| B9 | B6/B8 with waste-heat recovery net of parasitics | Exergy recovery contribution |
| BX | Experimental energy harvesters, advanced materials, nuclear or novel propulsors | Separate research-only branches |

This matrix is not a requirement to install every measure on one hull. Use ablations and interaction analysis. Do not retain a measure simply because it is technologically interesting.

## 6. Mandatory operating-condition matrix

At minimum score eligible architectures across:

1. head, beam, following and quartering apparent/true wind bands;
2. wind-speed, gust and forecast-error strata;
3. sea-state, wave-heading, current and water-depth strata;
4. ballast, partial, design and full-load conditions;
5. clean and degraded/fouled hull/propeller states where data support them;
6. WAPS active, feathered, retracted, unavailable and faulted states;
7. nominal, derated and failed storage/converter/generator states;
8. part-load and transient propulsion demand;
9. port access, berth windows, shore-power availability and charging limits;
10. planned maintenance, inspection and unplanned outage;
11. weather/port closures and recovery/re-routing events;
12. seasons or years not used for model fitting.

Split held-out data by voyage, season or route block. Randomly splitting adjacent time samples from one voyage is leakage, not independent generalization.

Never synthesize high-fidelity physical truth. Synthetic scenarios are software stress tests, not ship-performance evidence.

## 7. Core metrics and proposed targets

All figures are programme targets, not achieved results, standards or performance forecasts.

| Dimension | Initial target | Required safeguards |
|---|---:|---|
| Contract/evidence completeness | 100% required-field/provenance coverage | Unknown stays unknown; preserve source rights and model identity |
| False feasibility | 0 in declared negative controls | Safety/service hard gates are non-compensable |
| Pair eligibility | 100% known-answer agreement | Preserve every mismatch and reason |
| Voyage shaft-energy prediction | Median absolute percentage error <=5% aspiration | Also report p95, signed bias and route/load/wind strata |
| Tail prediction | p95 absolute percentage error <=10% aspiration | Score only admitted held-out evidence |
| Prediction uncertainty | Nominal 90% interval coverage in an 85–95% empirical band | Report interval width and coverage per operating slice |
| Net energy improvement | >=10% research target; 20% stretch | Best-practice matched baseline; unchanged service and availability |
| Robustness | No eligible hard-gate failure in declared envelope | Explicit out-of-domain disposition and fallback evidence |
| Reproducibility | Deterministic replay of declared report | Exact input, model, code and protocol revisions |

The 5%/10% model-error figures are aspirations, not guaranteed capabilities. Do not score before the dataset, operating envelope, independent labels, uncertainty treatment and holdout protocol are frozen.

ISO 15016:2025 describes an accepted speed/power-trial assessment procedure for displacement ships with Lpp 50–500 m. Its stated target accuracy is for the speed-trial process (within 2% shaft power and 0.1 knot speed), not a universal benchmark for full-voyage energy prediction or unlike ship designs. Use it for eligible trial protocol and correction, then separately assess prediction error.

## 8. Published reproduction target: coastal hybrid-electric case

Du et al. (Applied Sciences, 15 May 2026, 16(10), 4940; DOI 10.3390/app16104940) describe an 8,000-DWT coastal general-cargo case with 14-knot service speed, 110 m LOA, 104 m Lpp, 18 m beam and 7 m design draft.

Their optimized diesel-generator/DC-bus/motor/lithium-ion configuration reports approximately 19% fuel reduction, 20% CO2 reduction and 14% lifecycle-cost reduction relative to that study's mechanical baseline. The reported route-pattern range is 16–21% fuel reduction, 17–20% CO2 reduction and 8–14% lifecycle-cost reduction. These numbers are specific to the coastal vessel, route profiles, components and model boundary.

The study uses 24 months of AIS records from 2021–2022, reports more than 120 complete voyages and an average original sampling interval of 3 minutes, then extracts five statistically representative profiles for optimization. Gaps longer than 15 minutes are repaired using linear interpolation or speed extrapolation. The paper states that its supporting data and code are available from the corresponding author upon reasonable request, rather than providing a public raw-data download.

Its lower-level DP controller assumes perfect knowledge of each voyage's future power profile. That is an offline design upper bound, not a deployable controller. The paper also identifies simplified resistance/propulsion modelling, incomplete hull-fouling/extreme-weather/manoeuvring representation, and laboratory-derived stationary battery cycle-life surfaces with pack-level scaling assumptions. These limitations define the next benchmark opportunity: test realistic forecast-based operation, held-out voyages/seasons, data-gap handling, power-model uncertainty and degradation transfer without claiming the source study is invalid.

Treat this as a separate coastal replication case:
1. transcribe published vessel, service, cost, component and policy assumptions with table/equation references;
2. recreate the mechanical and rule-based hybrid baselines before the optimized case;
3. independently reproduce rounded tables and route-specific comparisons;
4. disclose unavailable AIS/logbook inputs and undocumented parameters;
5. test fuel price, battery cost, ageing, replacement and forecast-knowledge sensitivity;
6. do not claim exact replication if source time series, component maps or controller inputs are unavailable;
7. score any online controller against perfect-foresight DP as an upper bound, not as its operational baseline.

The published 16–21% range belongs to this vessel, route and method. It is not a prediction for a deep-sea cargo ship or WAPS-equipped hull.

## 9. Deep-sea wind-assisted test plan

Use a deep-sea cargo archetype only after the coastal reproduction establishes the accounting and experiment pipeline.

- Select one conventional reference configuration and freeze the service contract.
- Add one WAPS family at a time (rotor, suction wing, rigid wing, soft wing or kite) using source-attributed device maps and integration constraints.
- Vary placement and count only within a naval-architect-approved candidate envelope.
- Measure net thrust/moment and resulting hull, rudder, propeller and engine interaction.
- Include parasitic energy, foundation mass, stowage/retraction, inspection, maintenance, cargo/deck interaction and availability.
- Test mechanical, PTO/PTI and hybrid variants as independent architectures.
- Test wind-surplus alternatives without double-counting: engine reduction, speed adjustment, then hydrogeneration with added drag.
- Assess forecast-informed policy on held-out voyage/weather periods and degraded forecast conditions.

The first sea-trial campaign should synchronize apparent/true wind, device state and power, shaft torque/RPM, fuel mass flow, electrical bus, battery current/SOC/temperature, vessel speed/heading/rudder/draft/trim, waves/current, auxiliary load and contracted service record.

## 10. Industry comparison and defensible advantage

| Existing capability | What it already establishes | Luminous response |
|---|---|---|
| DLR Odyssa | Integrated windship aero/hydrodynamic and energy-system concept assessment; developing stability, safety, manoeuvrability, energy and routing | Interoperate; do not claim to replace its physics or ship optimizer |
| SHIPFLOW / specialist CFD | Established commercial flow and hydrodynamic analysis workflows | Preserve model/mesh/conditions and validation domain; use qualified solver outputs |
| Wind-device suppliers and operators | Commercial installations and vessel-specific operating claims | Separate vendor, operator, modelled and independently measured evidence |
| Hybrid-electric research | Route-specific sizing and energy-management optimization, including battery degradation | Reproduce published cases, then improve holdout/forecast/failure evidence |
| IMO / class / flag guidance | Evolving safety frameworks for WAPS, batteries and alternative fuels | Trace requirements and evidence; never represent research output as approval |
| ISO/ITTC trial methods | Defined measurement/correction procedures within specific scopes | Preserve raw traces, calibration, timebase, correction method and uncertainty |

The proposed differentiation is not another ship solver. It is the open, reproducible evidence thread connecting declared service, vessel configuration, specialist solver inputs/results, uncertainty, production/inspection revisions, operational trial and lifecycle outcome.

To demonstrate advantage over SOTA, publish blind, independent comparisons on held-out cases. Candidate dimensions include fewer false-feasible/incomparable comparisons, higher evidence completeness, better calibrated transfer to held-out route/season/load strata, lower measured service-level lifecycle burden, and deterministic reproduction across tool/version/configuration changes.

A better document, schema or optimizer architecture is not proof of better ship design. A numerical win must exceed combined model/metrology uncertainty and recur on independent matched evidence.

## 11. Standards and regulatory alignment

- **ISO 15016:2025:** speed/power trial analysis and corrections within the standard's scope; not an absolute ranking standard.
- **ISO 19030:** measurement of hull and propeller performance changes on the same ship over time; not an absolute ranking across unlike hulls.
- **IMO 2024 LCA Guidelines (MEPC.391(81)):** well-to-tank, tank-to-wake and well-to-wake accounting for CO2, CH4 and N2O; upstream fuel/electricity pathways belong in each case.
- **MTF WAPS Safety Management System guidance (March 2026):** practical operational safety management for wind-assisted ships, including rotor, suction and wing systems.
- **IMO MSC 111 (13–22 May 2026):** endorsed the safety-framework work plan for wind propulsion, batteries and other technologies; approved interim hydrogen-fuel and ammonia-cargo-as-fuel guidelines. This is developing framework work, not a completed harmonized WAPS code or vessel approval.
- **IMO/SDC roadmap:** published work plan targets interim wind-propulsion safety guidelines in 2029. Treat milestones as planned work, not existing approval.

## 12. Data-contract gaps before L2/L3

The current v0.1.0 case schema is a useful beginning, but these records are not fully resolvable yet:

1. **Calibration records:** channel IDs refer to calibration records, but there is no normalized registry for valid dates, range, uncertainty, traceability and disposition.
2. **Timebases:** channels name a timebase, but the contract does not resolve clock source, synchronization method, offset/drift estimate, uncertainty and validity period per timebase.
3. **Sample completeness:** missing-sample count exists, but total expected/observed sample count and channel coverage are not fully declared.
4. **Voyage service realization:** annual delivered work exists, but voyage cargo quantities, loading/unloading timestamps, missed calls, schedule slack and port delay need a rights-aware event record.
5. **Forecast provenance:** weather profiles need provider/source, forecast initialization/valid time, ensemble member, revision, resolution, interpolation, missingness and observed-versus-forecast distinction.
6. **Failure/maintenance states:** equipment availability and planned/unplanned outage need event identity, affected configuration, duration, cause class and repair evidence.
7. **Fuel pathway:** well-to-wake factor needs pathway revision, boundary, data quality, geography, electricity mix and source/verification status.
8. **Comparability profiles:** approved differences between baseline and candidate need explicit profile ID/revision and reviewed rationale; otherwise return INCOMPARABLE.

Do not add these concepts as undocumented JSON extensions. Evolve the schema with versioned migrations and adversarial fixtures. A validator pass must state which registries and comparison rules were checked.

## 13. Execution sequence

1. Keep the Rust validator isolated and fail-closed; add exact-head tests and a lockfile before calling it executable.
2. Resolve calibration/timebase/sample-completeness registry gaps in a schema version increment.
3. Add paired baseline/candidate fixtures, including mismatched routes, forecasts, boundaries, lifecycle horizons, unknown gates and stale calibration.
4. Reproduce the Du et al. coastal hybrid case at the published-information level; preserve the unresolved Table 5 (2 × 2,250 kW DG) vs Table 8 (2,300 kW P_eng,max) interpretation conflict until author clarification.
5. Define one deep-sea WAPS service contract and source minimum operator/device data.
6. Connect specialist ship/CFD/propulsor/energy solvers through versioned adapters.
7. Freeze route/season holdouts and forecast-degradation experiments before optimization.
8. Plan model/bench/sea trials with qualified naval architect, operator, test facility and class/flag stakeholders.
9. Publish only the claim level supported by evidence: contract, published reproduction, solver V&V, controlled physical trial, in-service evidence or fleet transfer.

## 14. Fleet-level, reversible and cyber-physical design extensions

### 14.1 Optimize the fleet's contracted service

A single-vessel design can be an energy winner and still make the shipping service worse. Extend the unit of optimization from one hull to the fleet delivering a fixed annual service. Co-optimize vessel assignment, route rotations, speed schedules, port calls, berth windows, reserve capacity, maintenance slots and cargo allocation.

Compare alternatives against the same annual delivered tonne-nautical-mile or TEU-nautical-mile, service windows and reliability. Include fleet size, repositioning, idle/anchorage time, cargo utilization, reserve vessels, port constraints and total fleet lifecycle cost. A slower ship that requires an extra vessel is not automatically more efficient. A faster or larger ship that cannot use available berths or cargo demand is not automatically better.

A fleet-level result must retain vessel-level failure distributions and must not hide a weak ship behind a fleet average. Report both annual fleet service and per-vessel/route slices.

### 14.2 Design a reversible energy platform, not a single-fuel bet

Deep-sea vessel lives are long, while fuel availability, carbon pricing, port infrastructure, energy technology and regulations can change. Investigate modular, upgradeable interfaces for wind devices, shaft/PTO/PTI equipment, DC distribution, batteries and fuel systems.

Candidate design options include standardized mechanical/electrical interfaces, accessible modular converter/generator bays, provisions for replaceable storage modules, serviceable WAPS foundations and stowage, and reserved mass/volume/structural margins. The design should permit staged retrofits without assuming that every future technology will be installed.

Evaluate the option value against the actual cost of readiness: structural reinforcement, unused volume, mass, capital, maintenance complexity, shipyard time, downtime and lost cargo work. "Future-ready" provisions receive no free credit unless their future avoided cost and deployment probability are explicit.

### 14.3 Make causal counterfactuals part of the ship's evidence thread

A before/after fuel comparison is especially weak when route, loading, sea state, wind, speed, fouling, trim and machinery condition all change. Pre-register a matched-voyage protocol and record the service fingerprint and configuration revision for every segment.

Where safe and operationally permitted, compare WAPS enabled/feathered states or controller policies under matched conditions. Otherwise use a declared counterfactual model with independently assessed error and uncertainty. Preserve raw traces, exclusions, calibration, clock alignment and environmental corrections. Separate measured effects from model-adjusted estimates; hold out voyages, seasons or route blocks before scoring.

The counterfactual engine should emit the reason a segment is ineligible rather than fill gaps with nearest timestamps or silent interpolation. This becomes a key differentiator from a generic digital twin: the result is not merely a simulated alternative but a reproducible, bounded causal claim.

### 14.4 Treat cyber resilience as an operational design constraint

WAPS controls, battery-management systems, propulsion/power controls, route and forecast feeds, onboard sensors, telemetry and remote maintenance are coupled cyber-physical surfaces. Cyber failure can become loss of propulsion availability, unsafe device state, invalid measurement, loss of reserve policy or degraded manoeuvring.

Evaluate local safe operation without cloud/shore connectivity, segmentation of bridge/machinery/energy/WAPS/corporate networks, least privilege, authenticated command paths, signed software/configuration updates, rollback, local interlocks, deterministic fallback and manual crew override. A research optimizer or shore-side fleet service must remain advisory; it must not bypass local safety interlocks or issue unqualified actuator commands.

Include stale/forged wind or shaft data, GNSS/AIS anomalies, timebase drift, telemetry replay, forecast compromise, converter/BMS communication loss, remote-maintenance compromise, network partition and update rollback in the degraded-mode test matrix. Link cyber faults to the same service-availability and evidence dispositions as mechanical faults.

Reference the current IMO Guidelines on Maritime Cyber Risk Management (MSC-FAL.1/Circ.3/Rev.4) and applicable IACS UR E26/E27 requirements. Applicability depends on vessel type, size, build contract, class, flag and equipment; this research document is not a compliance assessment or approval.

### 14.7 WAPS wind-energy MRV evidence preparation

COM(2026)620 (procedure 2026/0210(COD)) is a pending proposal, not current law. It would align MRV/FuelEU monitoring of GHG emissions and onboard energy, include WAPS among substitute sources, and leave wind-energy quantity determination to future standards/certification in Commission implementing acts.

The new [MARINE-WIND-ENERGY-MRV-001](./MARINE-WIND-ENERGY-MRV-001.md) workstream (#11501) prepares a versioned evidence contract. It separates standard-derived wind energy, research estimates and counterfactual avoided propulsion energy. No current regulatory wind-energy formula is assumed.

### 14.5 Regulatory route and emissions exposure

The route-robustness benchmark must also preserve a versioned regulatory exposure map. Physical service, whole-vessel energy, well-to-wake emissions, statutory CII/EEXI/EEDI, EU ETS covered emissions, FuelEU intensity and ECA fuel constraints are separate outputs. Do not collapse them into one score.

MEPC 84 initiated phase-two review of EEXI/SEEMP/CII, including wind-propulsion accounting work; this is a review, not an adopted WAPS credit formula. EU ETS, FuelEU and the North-East Atlantic ECA add voyage- and date-dependent exposure. The IMO Net-Zero Framework remains a proposal/negotiation scenario as of 10 November 2026, not an effective rule.

The versioned rule engine and route segmentation are specified in [MARINE-REGULATORY-ROUTE-001](./MARINE-REGULATORY-ROUTE-001.md) and tracked in [#11469](https://github.com/Luminous-Dynamics/symthaea/issues/11469).

### 14.6 Deployable energy-management regret

Separate the offline perfect-foresight dynamic-programming result from deployable forecast-aware policy. The former is a clairvoyant reference, not an operational controller. The benchmark should measure the regret of rule-based, greedy, receding-horizon and robust/stochastic policies under forecasts available at decision time, forecast degradation, missing data, battery degradation and equipment failures.

The 2026 JMSE physics-informed daily-fuel paper provides a leakage-resistant methodology comparator (time-block R² 0.943, random-split optimism gap +0.032) but its three measured vessels are cruise/offshore-support types and its target is total daily onboard fuel, not cargo-vessel propulsion-only fuel. FuelCast licensing also limits reuse. It cannot support a WAPS cargo-performance claim.

A WAPS-specific predictor should test conditional physical constraints against longitudinal aerodynamic assistance/resistance and declared device state rather than impose a blanket monotonic rule on raw wind speed. The corresponding experiment is specified in [MARINE-EMS-FORECAST-REGRET-001](./MARINE-EMS-FORECAST-REGRET-001.md) and tracked in [#11473](https://github.com/Luminous-Dynamics/symthaea/issues/11473).

### 14.7 Couple wind force to leeway, rudder and propeller response

New public evidence shows why longitudinal-only WAPS optimization is insufficient:
- The January 2026 Kamsarmax weather-routing study reports 1-DOF versus 3-DOF differences of 2–7% in predicted savings, up to 7% MCR in main-engine power estimates, and a 1.25% improvement when route optimization itself uses 3-DOF rather than optimizing on 1-DOF and then evaluating at 3-DOF.
- September 2026 single-screw self-propulsion tests report strong leeway effects on thrust deduction/wake and rudder-induced propulsive-efficiency changes across tested angles.
- SINTEF's SOBC-1 campaign provides nearly 100 physical cyber-physical test cases including variable sail side-force/application point and transient wind.

These are case-specific benchmarks, not transferable correction factors. The new [#11510 WAPS 3-DOF coupling benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/11510) specifies a fidelity ladder from 1-DOF screening to steady 3/4-DOF, 6-DOF route/motion analysis, propeller-hull-rudder self-propulsion and transient aerodynamic coupling. Use the lowest fidelity shown adequate for each output, and validate every observable separately.

### 14.8 Optimize for validated Pareto improvement

Multi-fidelity optimization is established in naval architecture and aerospace; it is not itself the novelty. MARIN has used low-cost potential-flow/coarse-grid models with high-fidelity RANS in ship afterbody optimization, and 2026 work explicitly models the deferred validation cost of low-fidelity optima.

The benchmark should compare fidelity-selection/acquisition strategies by the **total cost to an independently validated Pareto improvement**, counting the final validation query, hard constraints, model discrepancy and rank reversals. Low-fidelity surrogate gains are not accepted as final outcomes.

See [#11513 — validation-aware multi-fidelity design](https://github.com/Luminous-Dynamics/symthaea/issues/11513) and [MARINE-VALIDATED-MULTIFIDELITY-DESIGN-001](./MARINE-VALIDATED-MULTIFIDELITY-DESIGN-001.md).

### 14.9 Dynamic manoeuvrability is a hard service constraint

Steady leeway/rudder equilibrium does not prove that a WAPS vessel can keep course or manoeuvre through gusts, turns, device failures and low-speed harbour operation. The 2025 WAPS MMG/VCT study for wPCC and Optiwise highlights that virtual-captive identification can be corrupted by assumptions about waves and roll, and that inverse-dynamics comparison with free-running tests is important.

Track dynamic manoeuvrability separately from propulsion efficiency: turning circle, zig-zag, stopping, course-keeping, rudder authority, yaw/roll excursions, gust rejection, WAPS stow/feather transitions and degraded recovery. MSC.137(76) is a conventional reference where applicable; WAPS-specific criteria require Administration/class/owner review.

See [#11521 — WAPS manoeuvrability qualification](https://github.com/Luminous-Dynamics/symthaea/issues/11521) and [MARINE-WAPS-MANOEUVRABILITY-001](./MARINE-WAPS-MANOEUVRABILITY-001.md).

## 15. Sources

- Liu et al., 2026 wind-propelled ship deployment and technology review, International Journal of Naval Architecture and Ocean Engineering 18, 100785. https://doi.org/10.1016/j.ijnaoe.2026.100785
- DLR, Odyssa / future-oriented ship design. https://www.dlr.de/en/ms/research-transfer/topics/design-and-optimization-of-future-oriented-ships
- DLR, “Setting sail again: how wind power is driving more efficient cargo ships”, 18 June 2026. https://www.dlr.de/en/latest/news/2026/setting-sail-again-how-wind-power-is-driving-more-efficient-cargo-ships
- Du et al., “Integrated Design and Dynamic Performance Optimisation of Hybrid Electric Propulsion Systems for Coastal Cargo Vessels Under Real-World Operational Profiles”, Applied Sciences 16(10), 4940 (15 May 2026). https://doi.org/10.3390/app16104940
- ISO 15016:2025, https://www.iso.org/standard/82080.html
- IMO lifecycle GHG intensity of marine fuels, https://www.imo.org/en/ourwork/environment/pages/lifecycle-ghg---carbon-intensity-guidelines.aspx
- Maritime Technologies Forum WAPS Safety Management System guidelines, 30 March 2026. https://www.maritimetechnologiesforum.com/en/news-publications/maritime-technologies-forum-announces-new-guidelines-to-strengthen-safety-management-for-ships-using-wind-assisted-propulsion-systems
- IMO MSC 111, 13–22 May 2026. https://www.imo.org/en/mediacentre/meetingsummaries/pages/temporary-msc-111.aspx
- IMO SDC 12 safety-framework work plan, 19–23 January 2026. https://www.imo.org/en/mediacentre/meetingsummaries/pages/sdc-12.aspx

## 17. SOTA update: leakage-resistant prediction and uncertainty-qualified hydrodynamics

Two further benchmarks materially sharpen the programme.

### 17.1 Operational fuel prediction

Park and Song's 23 September 2026 study, "Physics-Informed Machine Learning with Monotonic Constraints for Daily Fuel Consumption and CII Prediction of Ships" (JMSE 14(19), 1774; DOI 10.3390/jmse14191774), reports random-split R² 0.975 ± 0.001 versus purged time-block R² 0.943 ± 0.007 for FuelCast Poseidon (+0.032 random-split optimism bias). It reports elimination of non-physical ALE reversals in its three measured-ship cases using monotonic constraints and a CII MAPE of 6.80%.

FuelCast contains daily measured data from two cruise ships and one offshore support vessel—not a mainstream cargo vessel. The fuel target is total onboard daily fuel, not propulsion-only shaft fuel. Its records do not capture cargo loading variation adequately. Its reported license is CC BY-NC-ND 4.0; commercial use, derivative analysis and redistribution require a rights review. EU MRV annual aggregates are useful for fleet/CII diagnostics, but must not be disaggregated into daily training labels.

Our next protocol is therefore: reproduce the purged 20%-contiguous time block with 2% purge on both sides and five rotations; add whole-voyage, route/season and leave-one-ship-out holdouts; audit arithmetic target leakage; compare physics baselines and constrained/unconstrained RF/XGBoost; and report uncertainty calibration, worst-slice error and out-of-domain abstention.

Crucially, monotonic constraints must be conditional. A global "more wind speed means more fuel" rule is physically inappropriate for WAPS: relative direction and wind-device state can make favourable wind reduce engine demand. Current direction, SOG/STW, load/draft/trim and device state must remain explicit.

Specification: [MARINE-OPERATIONAL-PREDICTION-BENCHMARK-001](./MARINE-OPERATIONAL-PREDICTION-BENCHMARK-001.md).

### 17.2 Hydrodynamic verification/validation

ITTC procedure 7.5-03-02-01 Rev.02 (2024) gives a Series 60 cargo/container RANS example at Fr=0.316 and Re=4.3e6 with separate grid and uncertainty analysis. The example reports total-resistance comparison error of 7.2% against combined validation uncertainty of 3.1%, so total resistance is not validated in that case. Its wave-profile result is a separate near-validated observable. This is an important standard for claim discipline: no generic "CFD validated" badge is allowed when an observable has failed its uncertainty comparison.

Specification: [MARINE-HYDRODYNAMIC-VALIDATION-001](./MARINE-HYDRODYNAMIC-VALIDATION-001.md).

### 17.3 New execution workstreams

- [#11441 — operational prediction benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/11441)
- [#11442 — ITTC uncertainty-qualified hydrodynamic validation](https://github.com/Luminous-Dynamics/symthaea/issues/11442)
- [#11443 — canonical service fingerprint and comparison boundary](https://github.com/Luminous-Dynamics/symthaea/issues/11443)

Schema v0.2.0 and the validator now implement `marine-service-payload-v1+symthaea-canonical-json-sha256-v1`: the validator projects declared service, loading and ordered route/port/speed fields into a versioned payload and recomputes the SHA-256 using Symthaea's existing named canonical-JSON encoding profile (not RFC 8785). The synthetic v0.2.0 fixture carries a known-answer vector and mutation/key-order tests. Environment/metocean realization remains a separate counterfactual fingerprint/boundary. Explicit zero, unknown, not-applicable and absent quantity states remain distinct. #11443 tracks completion of the cross-language vectors, comparator integration and adversarial pair tests; the Rust suite remains unexecuted here.

## 16. Claim ceiling

A route-policy experiment is not proof of safe autonomous navigation. A model reproduction is not a vessel trial. A schema or validator pass is not sensor authentication or physical qualification. A one-ship result does not establish fleet transfer. Every conclusion must identify vessel, configuration, service contract, route, season, operating envelope, dataset, models, source rights, uncertainty, trial stage and excluded conditions.
