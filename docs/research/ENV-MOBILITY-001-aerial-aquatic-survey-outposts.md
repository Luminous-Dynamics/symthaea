# ENV-MOBILITY-001: Aerial–Aquatic Survey Fleets and Regional Outposts

**Status:** Research/design proposal. No vehicle, outpost, simulator, hardware, or field capability is claimed by this document.  
**Scope:** Civilian environmental observation, water stewardship, infrastructure inspection, mapping, biodiversity monitoring, and disaster assessment.  
**Parent architecture:** Symthaea [PERSISTENT-PLATFORMS-001 #7343](https://github.com/Luminous-Dynamics/symthaea/issues/7343), [ROB-DOMAIN-001 #7650](https://github.com/Luminous-Dynamics/symthaea/issues/7650), [external transport benchmark coverage #6266](https://github.com/Luminous-Dynamics/symthaea/issues/6266), [flight-controller provider edge #1291](https://github.com/Luminous-Dynamics/symthaea/issues/1291), and [maritime substrate #1259](https://github.com/Luminous-Dynamics/symthaea/issues/1259).  
**Commons/evidence integration:** Mycelix [AeroCommons #3638](https://github.com/Luminous-Dynamics/mycelix/issues/3638), [physical-evidence reproducibility challenge #3643](https://github.com/Luminous-Dynamics/mycelix/issues/3643), [airspace coordination #5023](https://github.com/Luminous-Dynamics/mycelix/issues/5023), and [Mobility Commons #3708](https://github.com/Luminous-Dynamics/mycelix/issues/3708).

## 1. Thesis: design the observing service, not a universal drone

Symthaea could help design better aerial and aquatic robots by optimizing the entire observation service: mission objective, environment, sensor package, vehicle morphology, route, energy, launch/recovery, regional support infrastructure, data quality, maintenance, and evidence. The most promising paradigm change is to make the **service and its verified observations** the unit of design—not a single airframe, leaderboard score, or autonomy demonstration.

Aerial, surface, and underwater systems share some information and lifecycle contracts, but they do not share one dynamics model or one safety case. Reuse the established Symthaea platform/domain owners; do not create a universal drone solver or duplicate the embodiment, flight-controller, marine, energy, or evidence systems.

The initial target is a repeatable system for lawful, non-covert scientific and public-benefit observation. Every mission must state its purpose, site, permitted operating envelope, data policy, responsible operator, and stopping conditions.

## 2. Mission-first design space

Select platforms from the scientific question and site constraints. “More autonomous,” “longer range,” and “more area covered” are not universal objectives.

| Platform family | Strong candidate uses | Distinct constraints that remain first-class |
|---|---|---|
| Multirotor aircraft | Local inspection, repeatable low-altitude imaging, hovering or vertical take-off in constrained sites | Rotor energy cost, wind/gust margin, obstacle clearance, landing area, propeller hazard, legal flight envelope |
| Fixed-wing aircraft | Broad-area mapping and efficient long transects where safe launch/recovery corridors exist | Stall/turn radius, landing corridor, launch/recovery mechanics, wind, camera geometry |
| VTOL fixed-wing | Sites that need broad-area cruise but lack a conventional runway | Transition envelope, extra mechanisms/mass, hover-energy reserve, failure modes during transition |
| Tethered or perching aerial sensor | Persistent local observation where tether/perch geometry is suitable | Tether snag/entanglement, weather exposure, support structure, fixed spatial coverage |
| Surface USV | Water-quality transects, shorelines, bathymetry support, surface weather/current observations, relay for a submerged vehicle | Stability, waves/current, collision avoidance, navigation and communication links, recovery, waterway constraints |
| AUV | Subsurface water-column sampling, mapping, habitat or submerged-asset inspection | Pressure/depth envelope, hydrodynamics, underwater localization, acoustic bandwidth, battery reserve, recovery and docking |
| ROV | Shorter-duration close inspection with a surface tether and direct operator supervision | Tether management, entanglement, operator workload, visibility/current, umbilical and launch/recovery |
| Underwater glider | Long-duration, low-speed sampling where slow response and its trajectory are acceptable | Limited maneuverability, mission latency, currents, surfacing windows and recovery |

These are candidate families, not blanket recommendations. AUV, ROV, surface-vessel and aerial controllers remain distinct. A shared mission envelope can describe constraints; it must not erase domain physics.

### Initial mission archetypes

1. **Watershed / lake:** aerial terrain and shoreline context; surface water-quality transects; underwater sampling or bathymetry when justified; fixed weather/water-level sensors as low-cost baselines.
2. **Coast / estuary:** aerial shoreline change and post-storm mapping; USV transects; AUV/ROV only for identified subsurface questions; tidal, wave, current and environmental restrictions explicitly represented.
3. **Agriculture / catchment:** aerial crop and drainage observations paired with soil/water reference instruments and ground truth; prioritize interventions that conserve water and avoid unnecessary flights.
4. **Wildfire, flood or storm aftermath:** lawful, authorized damage assessment from declared safe operating zones; favor rapid map products and uncertainty bounds; abstain where smoke, wind, visibility, traffic or emergency airspace make the mission infeasible.
5. **Biodiversity / habitat:** repeatable observation schedules, declared species/site sensitivity, and access rules; include a “do not disturb / do not sample” outcome when disturbance risk exceeds the information value.
6. **Remote infrastructure / conservation area:** long-distance survey with sparse communications; store-and-forward evidence and explicit abort/recovery plans rather than assuming continuous connectivity.

A static sensor, satellite product, manual survey, existing boat, or **no new robot** must remain valid baselines. The system should recommend the smallest sufficient sensing system, not maximize fleet size.

## 3. Regional outpost: modular service node, not an unattended robot garage

An outpost is a site-specific combination of landing/recovery infrastructure, energy, environmental reference sensing, communication, maintenance, safety, and data-quality services. Do not assume that one physical station is suitable for aerial, surface and underwater robots. Share the digital contract while keeping air-side and wet-side engineering separate.

### Reference modules

- **Site and weather module:** surveyed coordinates/datum, terrain and access envelope, wind and precipitation measurements, lightning/severe-weather source where available, solar exposure, flood/storm exposure, environmental restrictions, and currentness of each observation.
- **Power module:** metered renewable and/or grid input, storage, actual load profiles, reserve policy, degradation/temperature derating, and a safe low-power state. Compute seasonal energy balance; never assume nominal solar production equals usable mission energy.
- **Air-side operations module:** approved landing/launch pad, physical access control, rotor stop/arm state observability, weather-based no-go rules, battery inspection and controlled storage, and a safe manual recovery path.
- **Surface/wet-side module:** a compatible launch/recovery arrangement, corrosion control, rinse/dry and inspection workflow where appropriate, water ingress detection, and physically separated wet/dry electrical zones. Wet-side electrical and charging safety require qualified engineering review.
- **Subsurface docking module (optional, separate qualification):** dock geometry, approach envelope, visibility/turbidity, acoustic/optical localization assumptions, currents, pressure/depth, corrosion, alignment tolerance, charging/data-transfer method, missed-dock recovery and retrieval plan. Treat this as its own difficult engineering branch—not a simple extension of an aerial landing pad.
- **Metrology and calibration module:** reference targets or check points, sensor calibration records, known reference instruments, clock/geolocation health and a way to mark a payload out of service.
- **Local compute and communications module:** local-first mission execution, signed/immutable run manifests, health and telemetry buffering, intermittent-link operation, store-and-forward data, a minimal remote control plane, and a physically available local stop/disable mechanism.
- **Maintenance and spares module:** inspect/clean/replace workflows, consumables and parts inventory, battery-cycle history, module compatibility, repairability, waste handling, and an escalation/retrieval contact.
- **Public-interest data module:** observation metadata, data rights/license, retention and release policy, access controls for sensitive ecological locations, and a path to publish non-sensitive observations using open formats.
- **Human operations module:** named accountable operator, checklists, local access boundaries, dispatch/abort approval rules, incident reporting and a way to leave the station safely in a controlled offline state.

### Outpost sizing model

Size from a **declared duty cycle**, not a perpetual-autonomy assumption. The first model should explicitly account for:

- daily and seasonal energy generation versus measured mission, standby, communications, heating/cooling and processing loads;
- usable battery energy after reserve, temperature and end-of-life derating;
- simultaneous charging demand and the number/type of compatible charge interfaces;
- weather no-go periods and resulting service availability;
- turnaround time: recovery, inspection, cleaning/drying, data transfer, recharge, calibration check and redeployment;
- maintenance intervals, consumables, spare battery/module pools, logistics lead time and technician/operator time;
- backhaul availability, local storage capacity, storage retention and recovery after communication outages;
- common-cause failures (shared power, shared clock, shared weather feed, common radio/backhaul, fire/flood or physical site damage).

The first result is a transparent capacity and availability estimate with uncertainty and reserve requirements—not a certified installation or a claim of continuous operation. A multi-outpost network can provide service continuity through redundancy, but the fleet’s physical assets still need inspection, repair, replacement and retrieval.

## 4. A safer systems architecture

Keep the following stages distinct and traceable:

1. **Public-benefit question:** which observation or change-detection result is needed, by when, and to what decision will it contribute?
2. **Evidence map:** what is already known; data freshness, ground resolution, temporal coverage, calibration, uncertainty, rights, sensitivity and gaps.
3. **Candidate generation:** compare fixed sensors, remote imagery, manual surveys, and permitted aerial/surface/subsurface platforms.
4. **Feasibility filter:** reject options that violate the site's environmental, legal, airspace/waterway, weather, energy, navigation, privacy, or recovery constraints. Incomplete safety-critical inputs mean “abstain / obtain evidence,” not “probably feasible.”
5. **Portfolio/route planning:** maximize scientifically useful information under energy, time, cost, maintenance and risk constraints. Show a Pareto set rather than hiding ecological impact, data quality, cost and risk in one weighted score.
6. **Independent mission gate:** recheck current geofence/no-go areas, environment, platform configuration, sensor status, reserves, recovery site, and operator approval immediately before dispatch. Symthaea advice must not mint flight, navigation, launch, actuator, or regulatory authority.
7. **Domain-specific execution:** use qualified interfaces to existing flight-controller and marine stacks. Cognition can propose a task; the low-level controller, safety interlocks and responsible operator retain their separate roles.
8. **Observation and evidence capture:** bind each record to time, coordinate reference/datum, sensor identity/configuration, calibration state, raw/derived distinction, uncertainty, environment, vehicle/configuration identity and data rights.
9. **Quality gate:** compare against independent check points/reference instruments when the mission requires it. Missing, stale, contradictory or out-of-envelope observations are marked invalid/uncertain rather than silently filled.
10. **Update and replan:** update the information-gap map, explain why the next mission changed, and preserve the prior plan, failures and negative results.

A regional edge node should keep safe local behavior possible during network loss. Remote cloud availability, Mycelix, or the cognitive layer must not be dependencies for an emergency stop or a vehicle’s immediate domain-specific failsafe.

## 5. Paradigm shifts worth testing

### A. Optimize information gain per scarce resource—not raw flight time or square kilometers

A wide survey with unusable resolution or uncalibrated measurements has little scientific value. Rank candidate missions by expected reduction in decision-relevant uncertainty, weighted by declared ecological/public benefit and constrained by risk, energy, cost, latency and data rights. Preserve the component values and uncertainties; do not present an opaque “mission intelligence score.”

### B. Treat the outpost as a robotic utility

Replace bespoke one-off deployments with modular services: surveyed site profile, energy budget, compatible docking interfaces, calibration, fleet health, local data custody, spares and retrieval. Compare a fixed sensor station plus occasional mission against a continuously cycling fleet. The stationary baseline may win for many measurements.

### C. One evidence fabric, multiple physical models

Common contracts can carry mission requirements, site restrictions, configuration identity, calibrated observations, provenance and lifecycle records. Vehicle models, control authority and physical validation stay specific to the mode. Mycelix should record who supplied and validated evidence, rights and changes; it must not turn federation consensus into proof that a reading or design is physically correct.

### D. Robust planning beats nominal optimality

Plan over weather, currents, navigation errors, sensor outages, energy uncertainty, temporary site closure and link partitions. Report expected performance **and** worst cases / uncertainty bounds. Include “wait,” “return,” “retry later,” “switch to fixed sensors,” and “no mission” as first-class options.

### E. Co-design vehicle, payload and support infrastructure

Choose sensor mass/power, sampling cadence, airframe/hull, energy capacity, charging interface, docking geometry and maintenance together. A slightly less capable vehicle may win when its payload is interchangeable, its parts are locally repairable, and recovery is reliable.

### F. Failure containment as a service property

The objective is not a robot that never fails; it is a survey service that detects failures, limits their scope, preserves evidence, recovers safely and returns to a known state. Redundant sensors are not independent when they share power, calibration, software, location, clock, or a common environment. Test correlated failures and graceful degradation.

### G. Open hardware as a reproducible engineering process

Publish editable source, reference configurations, bills of materials, interface drawings, calibration procedures, manufacturing tolerances, test fixtures and known limitations where safety, rights and sensitive site constraints permit. An open CAD file is not a validated vehicle; each manufactured configuration needs identity, inspection and tests.

## 6. Research and implementation sequence

### Phase 0 — owner and requirements audit

Map every required equation, crate, interface, benchmark, evidence owner and external permission to existing Symthaea/Mycelix work. In particular, compose #6244/#6245, #6266, #7343, #7650, #1291 and #1259. Do not extract or rename a shared abstraction just because multiple crates contain modules with similar names.

**Exit:** dependency/owner map; no duplicate core; mission archetypes and source-rights registry; unresolved gaps explicitly listed.

### Phase 1 — deterministic, effects-disabled reference cases

Create a compact, standard-library-friendly Rust reference model (or extend the canonical owner) for site/platform/outpost feasibility and resource budgets. Input data must include explicit SI units, coordinate frames/datum, time windows, sensor/configuration identity and provenance. Reject non-finite or physically incoherent values; do not use hidden defaults, pretend precision, implicit unit conversion or a scalar score that hides hard constraints.

Inputs should include environmental envelope, platform/payload profile, mission task, path/coverage requirements, measured or declared energy profile, outpost interfaces, service duty cycle, evidence freshness and constraints. Outputs must separate:

- feasible / infeasible / insufficient evidence;
- expected service capacity and resource margins;
- predicted information/coverage quality and uncertainty;
- lifecycle and operator effort;
- unresolved safety/legal/site prerequisites;
- a replay receipt with exact input, corpus and evaluator digests.

This is arithmetic and planning support, not a CFD/FEA model, autopilot, airworthiness tool, marine approval or electrical safety assessment.

### Phase 2 — frozen scenario corpus and independent oracle

Start with **48 synthetic cases** across six archetypes (watershed/lake, coast/estuary, agriculture/catchment, disaster aftermath, biodiversity/habitat, remote infrastructure). Vary operating envelope, visibility/weather/current, energy budget, communication availability, reference data quality and duty cycle. Keep the truth table independent of the production selector; use a separately implemented validator/checker.

Required hostile cases include:

- stale or missing calibration treated as current;
- a sensor timeout turned into a valid zero measurement;
- estimated data promoted to measured ground truth;
- inaccurate GNSS or datum/frame identity ignored;
- an airspace/waterway/no-go or weather limit ignored;
- current/wave/depth or wind uncertainty omitted;
- predicted range based on nominal energy without required reserve/temperature/age derating;
- a route exists but safe recovery/docking does not;
- the dock is unavailable or the charging interface is incompatible;
- repeated failed docking is hidden from service statistics;
- shared power/backhaul/clock failure mistaken for independent redundancy;
- a communications partition blocks emergency-local behavior;
- a remote command bypasses local stop or domain-specific failsafe;
- a sensitive habitat/site is over-sampled or data released without proper rights;
- simulated output presented as a field result;
- “fixed sensor,” “manual survey” or “no new mission” is the correct baseline.

**Initial qualification targets:**
- 100% exact agreement with the independent oracle on the declared synthetic hard-feasibility and rejection labels;
- zero false accepts for declared safety/legal/configuration hard constraints in the frozen corpus;
- zero acceptance of missing required calibration, stale required evidence or unresolvable coordinate-frame ambiguity;
- deterministic output for identical versioned inputs; all corpus/configuration digests recorded;
- every refusal / abstention identifies which evidence or constraint blocked the decision.

A single hard-constraint false accept blocks the associated claim. These are software/fixture targets, not a claim of safe real-world flight or marine operation.

### Phase 3 — comparative mission-quality study

Freeze simple and strong baselines before tuning Symthaea: fixed lawnmower/grid sweep, greedy next-best-view, static schedule, fixed-sensor-only, and manual-survey estimate where reproducible. Compare:

- validated observations per Wh and per operator-hour;
- useful spatial/temporal coverage at stated resolution;
- information-gain or uncertainty reduction against independent reference data;
- false-positive/false-negative rates for the declared survey task;
- position/measurement residuals against independent check points and their uncertainty;
- on-time completion, safe abort/return rate, recovery success and missed task rate;
- outpost energy / charge contention / downtime / mean repair time;
- life-cycle cost per valid observation, including maintenance, travel, data processing, failed missions and consumables;
- environmental disturbance and privacy/data-policy violations.

Report metrics separately by environment and platform. Publish a Pareto frontier and confidence intervals instead of one aggregate winner. Treat a hypothesis such as “at least 20% more validated information gain per Wh than a fixed-sweep baseline in three scenario families” as a preregistered research hypothesis—not a result or an acceptance claim—until independent tests support or reject it.

### Phase 4 — simulation, hardware-in-the-loop, then limited pilots

Order of evidence:

1. Hand-calculated reference examples, property tests and mutation tests.
2. Deterministic simulation with seed, version and scenario manifest bound to every result.
3. Hardware-in-the-loop, including partitions, stale readings, failed docks and safe-stop paths. Begin station and power-interface tests with propulsion disabled or physically isolated.
4. Bench tests for sensing, charging, cleaning/drying, connector durability and environmental resistance by qualified people.
5. Small, benign physical trials in an approved test area with trained operators and explicit recovery plans.
6. Site-specific field pilots only after the applicable local legal, airspace, waterway, environmental, electrical and operator approvals.

Each phase has its own evidence ceiling. Passing simulation is not passing HIL; HIL is not a field demonstration; a field demonstration is not regulatory certification. Hosted CI may queue; continue review, analytic checks, fixture work and evidence audits, but report exact status honestly.

## 7. Safety, stewardship and privacy invariants

- Civilian scientific, conservation, mapping, infrastructure-inspection and disaster-response purposes only for this workstream.
- No covert people tracking, intrusive surveillance or repurposing environmental sensing into targeting.
- Explicit site permissions and prohibited areas; protect sensitive ecological locations and personal data.
- No dispatch if a required environment/safety input is stale, missing, contradictory or outside the declared envelope.
- Mission planner, cognition, fleet manager, ground station and physical controller remain separate authority layers.
- Local, network-independent stop/disable path and domain-specific failsafe required before physical testing.
- Emergency reserves include return/recovery and uncertainty margin; do not plan to consume the minimum theoretical battery or fuel.
- Never retry a failed docking/landing indefinitely; bound retry count and provide alternate recovery/abort behavior.
- Keep raw measurements, derived products, estimates, failures, calibration events, manual interventions and aborted runs distinguishable.
- A fleet may be unavailable; it must not falsely report completed/valid survey coverage because an upload, mission or sensor was scheduled.
- No automatic deployment, procurement, charging energization or actuator authority is granted by a benchmark result.

## 8. Evidence and benchmark taxonomy

Every reported result must declare one class, exact configuration and claim ceiling:

- **SyntheticReferenceCase**
- **ClosedLoopSimulation**
- **DatasetOfflineEvaluation**
- **HardwareInTheLoop**
- **BenchPhysicalTest**
- **SupervisedFieldTrial**
- **IndependentExternalEvaluation**
- **RegulatoryOrStandardsEvidence**

The last category should usually be a source reference to an external authority or qualified assessment, not a status generated by Symthaea. Each observation/test records source and data rights, software/model/configuration versions, raw artifact digest where appropriate, instrument/calibration identity, environment, units, datum/frame, uncertainty and reviewer disposition. Use open geospatial conventions rather than inventing a parallel geospatial API.

## 9. External research and interoperability anchors

These sources motivate the architecture; none validates a Symthaea implementation:

- NOAA, [Uncrewed Systems Strategy](https://sciencecouncil.noaa.gov/noaa-science-technology-focus-areas/noaa-uncrewed-systems-strategy/): multi-domain uncrewed platforms support high-accuracy and time-sensitive science, with a focus on safety, cost-effectiveness and operational transition.
- NOAA, [Uncrewed Systems Operations Center expands environmental observation footprint](https://omaostg.omao.noaa.gov/omao/news-media/article/noaa%E2%80%99s-uncrewed-systems-operations-center-expands-environmental-observation-footprint), 31 December 2025: examples across distinct science missions reinforce matching platform to environment and scientific objective.
- Open Geospatial Consortium, [SensorThings API](https://www.ogc.org/standards/sensorthings/): an established interoperable, geospatial-enabled way to connect heterogeneous IoT sensors and observations; [Tasking Core](https://docs.ogc.org/is/17-079r1/17-079r1.html) covers tasking devices including unmanned systems. Prefer compatibility with these conventions over inventing a new exchange standard.
- USGS, [Guidelines for Calibration of Uncrewed Aircraft Systems Imagery](https://pubs.usgs.gov/publication/ofr20231033): emphasizes radiometric/geometric calibration and independent check points; the reported accuracy is bounded by reference-point measurement quality.
- PX4, [Safety configuration](https://docs.px4.io/main/en/config/safety): documents battery, data-link and geofence failsafes. Use the relevant, version-pinned flight-controller documentation and verify actual configuration; the existence of a feature is not proof it is enabled or adequate.
- Liu et al., [A review of underwater docking and charging technology for autonomous vehicles](https://doi.org/10.1016/j.oceaneng.2024.117154), *Ocean Engineering* 297 (2024), article 117154: surveys docking geometry, acoustic/optical navigation, current-resistant stabilization, charging and data exchange. This supports treating underwater dock and recovery as a separate hard subsystem rather than assuming it is solved.
- [Inductive Wireless Power Transfer for Autonomous Underwater Vehicles: A Comprehensive Review of Technological Advances and Challenges](https://www.mdpi.com/2077-1312/13/10/1855), *Journal of Marine Science and Engineering* (2025): reviews couplers, power losses and seawater effects relevant to any future subsurface charge interface.

## 10. Exit criteria and decision points

Do not claim implementation readiness until:

- existing owners and adjacent issues are mapped and duplication is ruled out;
- a typed, versioned environment/mission/platform/outpost contract is frozen;
- the synthetic corpus and independent oracle are committed with a reproducible invocation;
- every hostile case above has expected disposition and mutation coverage;
- resource calculations have hand-checked examples and uncertainty is explicit;
- the mission-quality study uses frozen baselines and independent reference measurements;
- safety, legal, ecological, privacy, data-rights and recovery prerequisites are explicit per site;
- each prototype increment has an evidence ceiling and a stop condition.

**Stop or narrow scope** if the model cannot distinguish feasible from under-specified missions, if a hard-boundary mutation is accepted, if the outpost cannot provide safe recovery, if results depend on hidden defaults, or if benchmark performance improves only by degrading measurement validity, safety margin, privacy or ecological stewardship.

The intended outcome is not “a drone that does everything.” It is an open, repairable and auditable network of task-appropriate survey tools whose claims stay bounded by evidence—and whose safe fallback may be to wait, return, use a stationary instrument, request a human survey, or do nothing.
