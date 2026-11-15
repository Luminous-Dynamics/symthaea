# Mobile Logistics Carrier: EV Truck + Dockable Drone Fleet + Residential Delivery

**Status:** Research proposal / design contract v0.1  
**Date:** 2026-11-15  
**Repository snapshot:** Symthaea `main` at `42446eccd8205c1acf5bff8c48a96a43d8fd2776`  
**Scope:** Documentation and research contract only. No flight-control, vehicle-control, dispatch-authority, or hardware-readiness claim is made.

## 1. Thesis

Build an electric carrier vehicle that acts as a mobile logistics and energy hub for interoperable land, air, and residential robotic systems. The truck carries heavy and bulky cargo, dockable drones, modular battery and charging infrastructure, package-handling equipment, and optional ground robots. Drones handle eligible aerial legs; ground robots handle the last metres and difficult access; the residential receiver accepts, verifies, and stores deliveries.

The system-level objective is not to make one drone carry every item. It is to allocate each package to the safest, lowest-lifecycle-cost, most reliable feasible mode across road, air, robotic ground transfer, and fixed receiving infrastructure.

**Working programme name:** AeroWeft Mobile Logistics  
**Residential receiver:** NestPort  
**Reference demonstrator:** SkyGarden / SkyCommons logistics interface

## 2. Paradigm shift

The proposed system differs from a conventional delivery van plus one drone in six ways:

1. **Mobile hub, not merely a launch vehicle.** The EV carries inventory, charging, maintenance, package custody, and compatible drone docks. It can reposition the service boundary without building a permanent depot at every location.
2. **Heterogeneous robot team.** Truck, multirotor, manipulator, mobile ground robot, and (later) quadruped cooperate under explicit capability and safety contracts.
3. **Flexible recovery and handover.** A drone may be planned to return to its originating carrier, a compatible parked carrier, a fixed community hub, or an approved receiver. Cross-carrier recovery is a later-stage capability, not an assumption.
4. **Payload-aware mode selection.** Heavy or bulky goods remain on the road vehicle or a ground handling platform. A drone is assigned only if the package, aircraft, route, weather, reserve, delivery site, and approvals fit its declared envelope.
5. **Home as logistics infrastructure.** The destination has an authenticated, weather-resistant, condition-aware receiving and storage interface rather than an improvised drop point.
6. **Evidence-carrying operations.** Every mission records the relevant plan, authority, configuration, limits, observations, handoffs, and delivery result. Unknown state must not be silently interpreted as success.

The opportunity is a cooperative logistics fabric, not a claim that truck-drone routing is a new field. Truck–UAV collaborative routing is already well researched; differentiation must come from validated end-to-end integration, robust operations, residential receiving, heterogeneous robot handling, and lifecycle performance.

## 3. Product architecture

### 3.1 EV carrier

The carrier should be a modular commercial EV platform selected from a payload and duty-cycle study rather than prematurely fixed to one truck class.

Core modules:

- Road vehicle chassis with load-aware mass properties, axle-load monitoring, braking and stability constraints.
- Segregated cargo bay with modular shelves, restraint points, parcel bins, and heavy-item handling.
- Mechanically retained drone docks with positive latch sensing, environmental sealing, inspection access, and safe charging interlocks.
- Separate aviation battery storage and charging zone with independent electrical isolation, battery monitoring, thermal sensing, fault containment, and emergency procedures.
- DC power conversion and energy management connecting traction battery, auxiliary systems, stationary storage, and drone charging only within qualified electrical limits.
- Weather station and local site assessment sensors.
- Optional loading arm/manipulator, conveyor, lift gate, pallet handling, or powered cart interface.
- Secure communications and auditable mission gateway.

Initial operational rule: **the carrier must be stationary, immobilized, and in an approved operating configuration before drone launch or recovery.** In-motion launch/recovery is explicitly out of scope for v0.1. It can be studied later as a separate, high-risk research problem with dedicated test infrastructure and regulatory engagement.

### 3.2 Dockable drones

Use multiple aircraft classes rather than one universal design:

| Class | Intended role | v0.1 boundary |
|---|---|---|
| Light parcel multirotor | Small parcels, urgent components, remote access | Payload and range set by a measured aircraft-specific envelope |
| Medium cargo multirotor / VTOL | Larger eligible packages or specialist remote access | Separate mass, propulsion, thermal, and contingency qualification |
| Fixed-wing / eSTOL | Longer rural legs where suitable landing infrastructure exists | Not assumed to use the truck roof dock |
| Heavy-lift aircraft | Exceptional cargo missions | Research only; not the default residential delivery method |

Drone dock contract:

- Mechanical compatibility and verified retention.
- Unique vehicle and dock identity.
- Latch state, contact state, charge state, temperature, moisture, and fault status.
- Aircraft configuration and payload manifest binding.
- Charging only with verified docked state and electrical interlocks.
- No undocked charging assumption; no blind hot-swap assumption.
- Preflight check of aircraft health, reserve, weather, geofence, route, receiver readiness, and contingency.
- Postflight inspection, fault quarantine, and battery health update.

A dock is not just a landing pad. It is a controlled mechanical, electrical, data, and custody interface.

### 3.3 Heavy-item handling

The EV truck should be the primary carrier for heavy, bulky, awkward, or high-volume goods. The truck can use a powered lift gate, pallet jack, winch, cargo manipulator, or mobile base to deliver the item to the property boundary or accessible service area.

Drones should be reserved for packages whose mass, dimensions, centre of gravity, drag, attachment, energy cost, and failure consequences fit the aircraft's tested envelope. Heavy-lift drone capability may eventually support selected loads, but it must not be assumed to be cheaper, safer, or more efficient than road transport plus ground handling.

A useful allocation hierarchy is:

1. Truck-only for heavy/bulky/oversized cargo.
2. Truck + wheeled ground robot for final-site delivery where ground access is safe.
3. Truck + multirotor for eligible light parcels and time-sensitive access.
4. Truck + drone + ground robot for a handoff where the last section is inaccessible to the truck.
5. Special heavy-lift aircraft only after separate aircraft-specific qualification and approval.

### 3.4 Ground robotics

- **Symthaea manipulator:** loading, unloading, package transfer, tote handling, and secure-receiver interaction, subject to its workspace, force, torque, and final-authority safety limits.
- **Symthaea vehicle:** ground vehicle autonomy and navigation research; the current bicycle-model plant is not a validated loaded-truck dynamics model.
- **Wheeled mobile robot:** preferred for routine flat, accessible final metres because it is mechanically simpler than a humanoid or quadruped.
- **Symthaea quadruped:** later option for rough terrain, steps, or off-road property access, after carrying capacity, stability, fall risk, and human/animal interaction are measured.
- **Symthaea humanoid:** not a first-release logistics dependency. Use only where human-scale interfaces make its added complexity worthwhile.
- **Symthaea agribot / subterranean platforms:** potential adjacent capabilities for rural/estate logistics and site operations, not initial delivery-critical components.

## 4. Symthaea / Symtropy / Mycelix integration

### 4.1 Existing public substrate inspected

This proposal is grounded in the public repository snapshot above.

**Symthaea**
- `crates/domains/symthaea-multirotor`: HDC-LTC + FEP quadrotor controller, with a documented multi-rate design (500 Hz motor reflex and 25 Hz cognitive tick), a physics simulator, and optional MuJoCo integration.
- `crates/domains/symthaea-multirotor/src/formation.rs` and `examples/quadrotor_formation.rs`: multi-agent formation types and a three-agent example.
- `examples/mujoco_swarm_training.rs`: a 128-agent training example with randomized microdrone masses of 20–35 g and wind magnitudes up to 0.15 m/s. This is a research/training example, not evidence of cargo-drone performance.
- `crates/domains/symthaea-vehicle`: vehicle controller and bicycle-model simulator. Its own simulator documentation describes a reduced-order single-track model, so a loaded multi-axle delivery truck plant is still needed.
- `crates/domains/symthaea-manipulator/src/safety_supervisor.rs`: final-authority manipulator supervisor with state validation, force/workspace checks, torque limits, and fail-safe interventions.
- `crates/domains/symthaea-helicopter`: a substantial flight-assurance research codebase. Its `docs/FLIGHT_TRUTH_ACCEPTANCE_GATES.md` explicitly says the simulator is not airworthy and lists missing validated aerodynamics, structural/fatigue loads, redundant certified hardware, and physical qualification. It is a source of assurance patterns, not a substitute for a multirotor cargo aircraft qualification.
- `docs/EFE_DISPATCH_GATE_2026-07-31.md`: an earlier EFE-based fleet-dispatch study was correctly halted before simulator construction. The documented action-dependence and purity preconditions failed, and the estimated controller headroom was small. Do not revive that experiment by merely wiring EFE into this logistics dispatcher.

**Symtropy**
- The robotics bridge declares platform categories including quadrotor, vehicle, manipulator, quadruped, and other embodiments.
- Use Symtropy for coupled scenario orchestration, replay, and the physical digital twin where its models are appropriate. It is not a substitute for validated FEA, CFD, battery, thermal, or flight-dynamics solvers.

**Mycelix**
- The public robotics-dispatch zome supports robotic asset registration, mission dispatch, telemetry, completion, and recall workflows.
- Treat Mycelix as the asset identity, authorization, custody, governance, and evidence coordination layer. It must not become the real-time motor or flight-control loop.
- The public housing and maintenance domains provide a foundation for shared receiving infrastructure, community facilities, and service records; their current maturity must be verified before operational reliance.

These files establish reusable interfaces and research scaffolding only. They do not prove that a truck, drone, manipulator, dock, or delivery operation is ready for deployment.

### 4.2 Separation of authority

Keep four planes distinct:

1. **Cognitive / planning plane:** Symthaea proposes assignments, routes, schedules, and adaptations.
2. **Optimization / feasibility plane:** deterministic optimization and explicit physical constraints validate plans.
3. **Real-time control and safety plane:** independently bounded platform controllers and safety supervisors own actuator command authority.
4. **Civic / evidence plane:** Mycelix records authorized asset state, mission intent, receipts, custody, maintenance, and disputes.

No LLM, FEP score, governance vote, remote message, or ledger record can directly bypass the platform's local safety supervisor.

Symthaea EFE-based mission selection remains experimental until action-conditioned epistemic value is demonstrated, scoring is pure with respect to candidate enumeration, and the pre-registered baseline comparison is valid. The deterministic baseline must remain available and independently testable.

## 5. Mission graph and interfaces

Represent the logistics environment as a time-dependent graph:

- Nodes: depot, carrier, parked carrier rendezvous, drone dock, community hub, home receiver, storage compartment, and ground-robot transfer point.
- Edges: road legs, flight corridors, ground-robot paths, transfer operations, charging actions, and approved recovery routes.
- State: time, payload, dimensions, mass properties, battery state of charge and health, weather, wind, airspace constraints, receiver availability, dock capacity, traffic, and uncertainty.
- Hard constraints: aircraft envelope, payload, reserve, no-fly zones, permissions, geofences, weather, C2/navigation health, receiver compatibility, dock state, and contingency reachability.
- Costs: energy, time, degradation, handling, road kilometres, failed handoffs, service quality, noise exposure, emissions, and lifecycle infrastructure burden.

Proposed versioned Rust contracts (names illustrative until a code-level design review):

- `CarrierState`: pose, speed, mass, axle loads, traction/auxiliary energy, cargo manifest, available docks, and operating mode.
- `AircraftState`: identity, configuration, mass, payload, battery SOC/SOH, navigation health, C2 health, weather estimate, faults, and flight mode.
- `DockState`: dock identity, aircraft compatibility, latch/retention state, charging state, thermal state, contamination, and fault state.
- `PackageManifest`: mass, dimensions, centre-of-gravity range, handling class, permitted modes, temperature range, security class, delivery window, and return/collection requirement.
- `ReceiverState`: authorization, compartment availability, environmental state, capacity, tamper state, and service health.
- `MissionPlan`: ordered road/air/ground legs, rendezvous, payload custody, reserve budgets, validity interval, alternate destinations, and contingencies.
- `MissionEvidence`: plan digest, configuration digests, preflight evidence, dispatch authority, telemetry references, handoff receipts, custody changes, and delivery result.

All physical quantities must carry units. Inputs must be schema-versioned, range-checked, and reject non-finite values. Unknown or stale safety-critical state means no launch or no continuation under the affected authority.

## 6. Operating concept and staged autonomy

### Stage 0 — offline and simulation

- No real aircraft command path.
- Replay published truck–drone datasets.
- Exercise load plans, dock state machines, battery and weather uncertainty, receiver capacity, and route scheduling.
- Build deterministic, explainable baselines before introducing learning.

### Stage 1 — parked carrier, remotely piloted / supervised tests

- Truck is stationary and immobilized before every launch and recovery.
- Flight test area is controlled and approved.
- Drones return to the same carrier or a fixed approved station.
- Ground package handling may be demonstrated separately.
- Human operator remains responsible within the approved operating concept.

### Stage 2 — multi-drone, multi-carrier coordination

- Multiple parked carriers and fixed hubs.
- Shared fleet assignment, charging queues, spare batteries, inventory, and alternate return sites.
- Cross-carrier recovery only when both carrier/dock combinations and the route are explicitly qualified.
- No assumption of in-motion docking or autonomous BVLOS authority.

### Stage 3 — residential and community pilots

- Integrate NestPort receiving units with ambient, refrigerated, and frozen storage compartments.
- Introduce ground robots for selected properties.
- Verify identity, authorization, tamper evidence, temperature history, resident consent, and fallback collection.
- Operate only under regulator-approved aircraft and service conditions.

### Stage 4 — advanced research

- Moving-carrier launch or recovery, automated heavy-lift, distributed charging, and more autonomous routing are separate research programmes.
- Each requires a distinct hazard analysis, test envelope, independent controls, and regulatory determination. No Stage 4 capability is implied by Stage 0–3 evidence.

## 7. Dock and receiving architecture

### 7.1 Carrier drone dock

A dock must provide:
- Physical retention verified independently of software intent.
- Interlock preventing aircraft control authority until latch and platform state are valid.
- Safe charging authorization only after verified dock engagement.
- Thermal and electrical fault isolation.
- Water, dust, and foreign-object monitoring appropriate to the environment.
- Positive release/retention state and emergency access.
- Aircraft-specific mechanical interface and compatibility record.
- Maintenance and cycle-count history.

### 7.2 NestPort home receiver

Provide separate, independently monitored compartments:
- Ambient parcels.
- Refrigerated food or other approved goods.
- Frozen goods.
- Secure high-value deliveries.
- Returns and reusable packaging.

The receiver should authenticate the operator and package manifest, confirm available space and environmental range, accept only compatible loads, secure the compartment, and issue a receipt. A failed identity check, wrong compartment, broken seal, out-of-range temperature, or power failure must result in a defined rejection/hold/escalation path.

Do not store unnecessary personal delivery contents in a public DHT. Use minimal metadata and encrypted or selectively disclosed records.

## 8. Dispatch and control strategy

### 8.1 Baselines first

Compare at least:
1. Truck-only vehicle routing.
2. Truck + one drone with fixed launch/return.
3. Truck + multiple identical drones.
4. Heterogeneous drones with fixed stations.
5. Shared fleet allocation across trucks.
6. Flexible recovery to a different parked carrier or fixed hub.
7. Integrated truck + drone + ground robot + receiving station.

Use exact MILP/CP-SAT or another independently checked optimizer for small instances and lower bounds. Use ALNS or another competitive metaheuristic for larger instances. Learning-based or active-inference policies are evaluated only against the same frozen action set, constraints, information, and compute budget.

### 8.2 Why the EFE gate matters

The existing `docs/EFE_DISPATCH_GATE_2026-07-31.md` identifies a prior failure in which the epistemic and novelty terms did not vary meaningfully across candidate actions and scoring was enumeration-order dependent. This logistics programme must not repeat that failure. Before Symthaea EFE can influence even advisory ranking:
- demonstrate action-dependent unweighted epistemic and novelty terms across candidates;
- prove candidate scoring is pure and order invariant;
- use negative controls and ablations;
- pre-register the comparison and estimator;
- retain a deterministic baseline fallback.

### 8.3 Dispatch decision sequence

1. Classify package and custody requirements.
2. Identify eligible road, drone, and ground-robot modes.
3. Reject any option that violates a hard constraint.
4. Estimate energy with measured payload-, wind-, temperature-, and battery-health-dependent models.
5. Schedule truck, aircraft, dock, receiver, and ground robot jointly.
6. Check synchronization slack and alternate recovery options.
7. Generate a mission plan with digest, expiry, authority, and contingency.
8. Require the platform's local safety supervisor and the authorized operator/operating system to approve launch.
9. Record telemetry and handoffs; reconcile unknown outcomes before any new mutation or package reassignment.

## 9. Benchmark programme

### 9.1 Reproducible comparison set

Start with:
- Amazon Last Mile Routing Dataset (2021), used in the 2026 empirical truck–drone MILP study.
- The public 2026 Zenodo experimental data for truck–drone routing (4,347 records across truck-only, comparison, ablation, warm-start, and large-instance groups).
- Standard truck-routing and vehicle-routing instances for truck-only baselines.
- A new openly specified Luminous suite with urban, suburban, rural, steep-terrain, remote-estate, and emergency/humanitarian profiles.

The 2026 systematic review stresses that spatial structure, payload eligibility, wind, airspace, launch/recovery handling, battery turnaround, synchronization, and facility costs can materially change reported gains. Those variables must be explicit, not hidden in synthetic assumptions.

### 9.2 Required metrics

**Mission and service**
- Delivered parcels per operating hour.
- Makespan and route duration.
- P50 / P95 / P99 delivery latency.
- On-time delivery rate and deadline violations.
- Failed, rejected, and returned deliveries.
- Customer handoff and receiving-station success.

**Energy and lifecycle**
- Total kWh per successfully delivered package.
- Truck kWh/km and drone Wh/kg-km.
- Energy at matched payload and route conditions.
- Battery throughput, degradation, and replacement assumptions.
- Lifecycle CO2e including vehicle, aircraft, battery, charging, and infrastructure.
- Charging dwell and charger utilization.

**Coordination**
- Truck waiting for drone and drone waiting for truck.
- Launch/recovery synchronization failures.
- Dock occupancy and turnaround.
- Repositioning overhead.
- Cross-carrier recovery success.
- Replanning compute time and infeasible-plan rejection rate.

**Safety and robustness**
- Accepted infeasible mission plans: target zero in the tested suite.
- Reserve margin violations: target zero.
- Geofence, no-fly, wind, payload, or receiver constraint violations: target zero.
- Fault detection and safe-state transition latency.
- C2/navigation loss, stale sensor, GNSS degradation, weather shift, truck delay, road closure, and battery-state uncertainty response.
- Near misses and human/animal exposure must be reported, never hidden in an average.
- Simulator/HIL/field evidence must be clearly separated.

**Storage and custody**
- Temperature excursions against the payload's validated limits.
- Tamper and unauthorized-access events.
- Wrong-compartment or wrong-recipient events.
- Chain-of-custody completeness.
- Receiver uptime and backup-power endurance.

### 9.3 Proposed research targets (not achieved results)

- Reproduce the 2026 study's truck-only comparison before claiming a gain.
- Target at least 15% lower makespan and 10% lower operational energy than truck-only on a pre-registered subset where aerial service is physically eligible; publish all scenarios where it loses.
- Target at least 5% improvement in a pre-registered combined objective over the strongest fair ALNS / exact-bound baseline on held-out instances, without weakening any feasibility constraint.
- Target zero accepted hard-constraint violations in exhaustive small cases and adversarial property tests.
- Target at least 99.5% successful automated dock cycles inside a declared HIL test envelope before any field-use consideration. This is a development target, not a safety certification level.
- Do not set accident-rate or airworthiness claims from simulation. Those require a regulator-aligned safety case, aircraft-specific evidence, and independent review.

Report paired results, confidence intervals, effect sizes, compute budgets, seeds, dataset splits, and all failed or incomplete runs. A result must never be called SOTA without a matched public baseline and independently reproducible evidence.

## 10. Current industry comparison

- **Amazon Prime Air MK30:** public operations information describes Part 135 certification, approved BVLOS operations, planned routes, trained flight monitors, weather go/no-go, ADS-B ingestion, detect-and-avoid, contingency landing, and UTM coordination. Our design should meet the same seriousness of operational-system integration; we cannot claim superiority by virtue of using Symthaea.
- **Zipline Platform 2:** an FAA environmental assessment describes a proposed operation with 63 lb MTOW and up to 8 lb payload, and up to 1,000 deliveries per day per Charger as a proposed maximum. Those are proposal figures, not verified daily throughput. It illustrates the maturity of dedicated drone infrastructure.
- **2026 truck–drone research:** a published empirical MILP study used three electric trucks and a shared fleet of ten drones on Amazon last-mile routes, with drone energy/time regression calibrated from flight tests. It reports up to 24% makespan reduction and 16% energy savings versus truck-only delivery under its modeled conditions. This is a research result, not a universal or field guarantee.
- **2026 review:** identifies synchronization fragility, battery/energy uncertainty, payload and wind sensitivity, regulation, scaling, and operational realism as persistent bottlenecks.

Our potential differentiation: a mobile carrier with interoperable docks; heterogeneous robots; configurable home/community receivers; shared energy and inventory; flexible recovery; verified custody; and full lifecycle accounting. This is a proposed product position, not an established competitive advantage.

## 11. Safety, regulatory, and environmental boundary

- The South African SACAA UAS guidance currently describes UAS approval, registration, operating-certificate, and operational requirements, while the published regulations have amendments affecting Part 101. The SACAA page and the 2026 amendments must be reconciled with the regulator before any operating concept is treated as approved.
- The 2026 South African amendments state that autonomous unmanned aircraft are excluded from Part 101's scope; this must not be interpreted as proof that autonomous delivery is unrestricted or unregulated. Obtain a written SACAA determination of the applicable certification/operating pathway for the exact aircraft and operation.
- FAA Part 135 package delivery rules apply in the United States and are not a South African approval pathway.
- EASA U-space and SORA 2.5 are useful reference frameworks for structured airspace and risk assessment, not a substitute for local approvals.
- No commercial delivery, BVLOS, operation over people, hazardous-goods carriage, or autonomous flight is assumed permitted by this document.
- Environmental assessment must count noise, wildlife disturbance, battery production/replacement, electricity mix, vehicle utilization, and supporting infrastructure. Electric propulsion alone does not establish lifecycle sustainability.

## 12. Immediate implementation sequence

1. Add this design contract and freeze the v0.1 scope.
2. Create a small Rust logistics domain crate with typed manifests, carrier/dock/aircraft/receiver state, mission graph, and plan-validation functions. No actuator interface.
3. Implement a deterministic truck-only baseline and a single-truck/single-drone stationary-launch simulation using explicit conservative envelopes.
4. Reproduce the published Amazon last-mile comparison and report the exact assumptions.
5. Add heterogeneous fleet allocation and ALNS / exact small-instance oracle.
6. Extend vehicle physics with loaded-truck mass/axle/braking/rollover-aware models; retain the existing bicycle model only as a baseline abstraction.
7. Integrate manipulator transfer and NestPort state machines.
8. Add Symtropy digital-twin replay and Mycelix asset/mission/receipt records after the data contracts stabilize.
9. Add HIL dock-cycle tests, charging interlocks, battery fault injection, and recovery drills.
10. Seek independent aviation, vehicle, battery, structural, privacy, and regulatory review before field trials.

## 13. Decision

Proceed with the mobile carrier and delivery-receiver architecture as a research and simulation programme. Start with parked-carrier multirotor missions, ground-based heavy cargo handling, deterministic optimization, and auditable delivery/storage receipts. Keep moving-carrier operations, heavy-lift flight, and autonomous BVLOS as separately gated research topics.

The success criterion is not a visually impressive prototype. It is a system that can state what it knows, prove what it checked, reject what it cannot safely do, recover cleanly from faults, and demonstrate measurable service and lifecycle benefits against reproducible baselines.

## References

- Symthaea repository, multirotor domain: https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-multirotor
- Symthaea fleet-dispatch precondition record: https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/EFE_DISPATCH_GATE_2026-07-31.md
- Symthaea manipulator safety supervisor: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-manipulator/src/safety_supervisor.rs
- Symthaea helicopter flight truth acceptance gates: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-helicopter/docs/FLIGHT_TRUTH_ACCEPTANCE_GATES.md
- Symtropy robotics platform bridge: https://github.com/Luminous-Dynamics/symtropy/tree/main/crates/bridges/symtropy-robotics-bridge-core
- Mycelix robotics dispatch: https://github.com/Luminous-Dynamics/mycelix/tree/main/mycelix-civic/zomes/robotics-dispatch
- Liu, Fei & Ma (2026), *Advances in truck–drone cooperative delivery routing and coordination: a review*: https://doi.org/10.1186/s12544-026-00808-2
- Cicek, Simsek & Kantarci (2026), *Data-Driven Optimization of Truck–Drone Collaborative Delivery with Shared Fleet Allocation*: https://doi.org/10.3390/drones10070476
- Fu (2026), *Experiment Data for Truck-Drone Cooperative Delivery Routing Optimization*: https://zenodo.org/records/21403496
- Liang, Chen & Du (2026), *Multi-visit truck-drone pickup and delivery problem with release dates and drone resupply*: https://doi.org/10.1016/j.ejor.2026.10.020
- FAA, Package Delivery by Drone (Part 135): https://www.faa.gov/uas/advanced_operations/package_delivery_drone
- FAA, Prime Air environmental review material: https://www.faa.gov/uas/advanced_operations/nepa_and_drones
- Amazon Prime Air operations: https://www.primeair.amazon/operations
- SACAA UAS guidance: https://www.caa.co.za/industry-information/uas/
- EASA, Easy Access Rules for UAS (June 2026): https://www.easa.europa.eu/en/document-library/easy-access-rules/easy-access-rules-unmanned-aircraft-systems
- EASA, U-space: https://www.easa.europa.eu/en/domains/air-traffic-management/u-space
