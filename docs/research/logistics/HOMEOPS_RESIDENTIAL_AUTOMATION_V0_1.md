# NestPort / HomeOps: Automated Residential Receiving, Sorting, Moving and Containerization

**Status:** Research and systems contract v0.1  
**Date:** 2026-11-15  
**Parent programme:** AeroWeft Mobile Logistics / SkyGarden / SkyCommons

## 1. Design thesis

A home should not merely accept a parcel. It should be able to receive, authenticate, move, sort, containerize, store, inventory and return compatible goods—while protecting residents, privacy, accessibility, food safety and household sovereignty.

Working names:
- **NestPort:** secure exterior receiving, custody and storage interface.
- **HomeOps:** internal home logistics automation.
- **Hearthline:** the physical service spine linking receiver, transfer equipment, storage and resident handoff.

The home becomes a local, user-controlled logistics node. It is not a public warehouse and is not required to expose its inventory to Mycelix.

## 2. The paradigm shift: design the home around material flow

Conventional houses treat receiving, pantry storage, refrigeration, parcel storage, recycling and movement as unrelated chores. HomeOps designs a service zone that handles these flows coherently.

The ideal architecture is a dedicated service vestibule and internal service spine:
- an external receiving boundary separated from living areas;
- an enclosed transfer shelf, drawer or tote elevator;
- a standard tote buffer;
- a protected sorting cell;
- fixed storage modules and/or a vertical lift;
- ambient, chilled and frozen compartments with separate controls;
- an accessible resident pickup interface;
- optional wheeled mobile manipulator for tasks outside the fixed station.

Do not default to unrestricted robots roaming the whole home. Fixed mechanisms are safer and more reliable for repetitive paths; mobile robots are used only where the environment and task justify them.

## 3. End-to-end operating model

### Phase 1 — exterior receipt

The delivery system deposits a sealed package in a secure exterior receiving vestibule. It may be delivered by courier, ground robot, drone winch or a compatible landed drone.

Requirements:
- conventional courier access remains available;
- aerial delivery is optional and never required for home operation;
- the drone does not fly inside the home;
- weatherproofing, access control, drainage and tamper detection;
- resident notification and manual acceptance;
- separate fire and emergency egress review.

### Phase 2 — authenticate and inspect

The receiver:
- checks expected opaque delivery reference;
- scans barcode, QR, RFID or other permitted identifier;
- measures mass and outer dimensions;
- checks tamper/seal state;
- captures only minimum evidence needed;
- confirms target household and compartment;
- verifies receiver capacity and environmental readiness.

Use manifest-first logic; computer vision is a fallback, not an unbounded authority. Unrecognized or conflicting packages go to an accessible quarantine drawer or human review.

### Phase 3 — sort

Initial classes:
- ambient parcels;
- groceries for chilled storage;
- frozen items;
- pantry and household consumables;
- tools and repair parts;
- reusable containers;
- returns;
- recycling;
- human-review or restricted items.

Classification is based first on package manifest and sender declarations, then on local sensing. A label or visual prediction alone must not be treated as proof of safe contents.

### Phase 4 — move

Preferred mechanisms:
- short, enclosed low-speed conveyor or transfer shelf;
- tote elevator between service-room levels;
- fixed gantry or guarded manipulator for predictable pick/place;
- vertical lift module or carousel for dense storage;
- low-speed wheeled AMR for flexible movement on accessible floors.

The home robot must yield to people and animals, stop on uncertain occupancy, avoid blocking escape routes, and expose a physical emergency stop. A door or lift failure must leave a safe manual path.

### Phase 5 — containerize

In v0.1, containerization means placing the sealed parcel or approved group of parcels into a standardized reusable tote with a unique ID.

The tote is the common logistics unit across carrier, community hub, delivery robot, NestPort, internal lift/conveyor and storage rack.

The system records tote ID, contents references, mass, dimensions, seal state, source, destination and storage location.

Automated package opening, depackaging, sharp-tool use, food unpacking and individual-item sorting are explicitly deferred. They require a guarded cell, known package class, tool supervision, hygiene controls and a validated exception path. No autonomous blade cutting in the first prototype.

### Phase 6 — store and inventory

The destination confirms:
- correct tote/package ID;
- valid compartment;
- compartment capacity;
- door/lock state;
- current temperature and humidity, where applicable;
- package/tote seal;
- storage event and evidence digest.

An item that cannot be placed safely is held in receiving quarantine. Do not mark it stored merely because a robot reached the shelf.

### Phase 7 — resident retrieval and returns

The resident can request a tote or item, authorize another household member or helper, receive a package at an accessible-height drawer, return packaging and reusable containers, schedule a courier/robot return pickup, or disable automated movement and aerial handoffs.

## 4. Product tiers

### Tier 0 — Smart receiving wall
Secure parcel vestibule, scan/weight/seal, notifications, manual collection. No robot required.

### Tier 1 — Assisted storage
NestPort plus standardized totes, powered drawer, tote lift and fixed destination compartments.

### Tier 2 — Automated home logistics
Guarded sorter/manipulator, conveyor, vertical lift storage, refrigerator/freezer interfaces and inventory reconciliation.

### Tier 3 — Mobile home assistant
Wheeled robot or mobile manipulator moves totes, supports resident retrieval and handles accessible multi-room tasks.

### Tier 4 — Advanced household automation
Individual-item unpacking, pantry replenishment, returns preparation, laundry/workshop supply flows and more generalized mobile manipulation. Each capability requires its own safety case.

This tiering makes the platform accessible: a family can install a simple receiver first and add automation without replacing the whole house.

## 5. Hardware architecture

| Module | Main responsibility |
|---|---|
| Exterior receiver | Secure parcel intake and access control |
| Sensor bay | Barcode/RFID, scale, dimensional sensing, seal/tamper and local vision |
| Transfer interface | Drawer, conveyor or tote lift |
| Standard tote | Reusable intermodal container with physical ID |
| Fixed handling cell | Guarded manipulator, gripper and package transfer |
| Storage lift | Vertical movement and dense storage |
| Ambient racks | Dry parcels, household goods and tools |
| Cold modules | Separate refrigeration and freezer with calibrated sensors |
| Mobile base | Flexible floor transport where useful |
| Resident interface | Accessible collection, approvals and exception resolution |
| Local controller | Safety interlocks, state machine, logs and emergency stop |

No particular vendor is assumed. Design the mechanical and software interfaces before selecting hardware.

## 6. Rust-first control and data architecture

Keep low-level controls local. Proposed crate/contracts, subject to a separate code review:
- homeops-types: typed package, tote, compartment and task models;
- homeops-state: append-only local task and custody state machine;
- homeops-receiver: sensor and door/lock interfaces;
- homeops-material-flow: validated transfer and storage graph;
- homeops-robot-bridge: Symthaea manipulator, AMR and device bridges;
- homeops-safety: independent interlocks and safe-stop arbitration;
- homeops-evidence: deterministic receipts and replay;
- homeops-commons-bridge: minimal Mycelix service metadata and custody references.

Use Rust for the core. Use external robot middleware or vendor bridge adapters only where required by physical platforms; do not allow a web UI or Holochain zome to be the real-time safety controller.

A Holochain record authorizes or records a service; it does not issue motor torque or unlock a door without the local controller's independent decision.

## 7. State machine and fail-closed rules

Required states:
- Empty
- DeliveryExpected
- Receiving
- IdentityPending
- Accepted
- Quarantined
- Sorting
- Moving
- Containerizing
- StoragePending
- Stored
- ResidentCollectionReady
- ReturnPending
- Returned
- Faulted
- ManualRecovery

Examples of hard rules:
- Unknown identity → quarantine, never silent acceptance.
- Package/manifest mismatch → human review.
- Full compartment → no placement.
- Temperature sensor stale or out of range → reject cold-chain placement.
- Door/lock disagreement → stop transfer and retain custody.
- Occupancy uncertain → robot yields.
- E-stop or fire signal → safe stop and do not actuate an interlock that could trap an occupant.
- Lost network → local safety and manual access remain available.
- Unknown handoff result → no new mutation until custody is reconciled.

## 8. Benchmark programme

### External comparison points

- **BEHAVIOR-1K:** 1,000 human-grounded household activities across 50 interactive scenes and more than 10,000 objects; the 2026 challenge is active. Useful for long-horizon household activities and household context.
- **HomeRobot OVMM:** open-vocabulary mobile pick-and-place in unseen home environments, with simulation and a real-world Stretch robot stack. The original paper reports 20% real-world success for its baselines—an important reminder that simulation transfer is difficult.
- **RoboCasa365:** 365 everyday tasks across 2,500 kitchen environments. Its public leaderboard, updated 13 November 2026, reports an 85.6 overall score for its top listed model on the 50-task multi-task benchmark. This is a benchmark-specific result, not an end-to-end home delivery success rate.
- **VDA 5050 v3.0:** released 20 April 2026; an open interoperability interface that expands mixed-fleet mobile robot control to freely navigating robots. Use as a gateway/reference where applicable rather than inventing a closed fleet protocol.

These benchmarks are not interchangeable. Publish results on each exact evaluation split and add a home logistics suite designed around receiving, tote transport, storage and exception handling.

### Luminous HomeOps tasks

1. Receive correctly labelled sealed parcel.
2. Detect tamper or manifest mismatch and quarantine.
3. Sort ambient/chilled/frozen from a validated manifest.
4. Transfer tote through a narrow service spine.
5. Place a tote into an assigned vertical storage slot.
6. Retrieve the correct tote for a resident.
7. Resolve a blocked aisle without entering an occupied zone.
8. Recover from conveyor jam or lift fault.
9. Keep food-separated items in correct compartments.
10. Reconcile an interrupted custody update.
11. Operate during network interruption and local power reserve.
12. Support a resident with limited reach or mobility through accessible drawers.

### Proposed research targets (not achieved)

- Target at least 90% routine end-to-end receive-to-store success on the frozen closed-world task suite.
- Target at least 85% on held-out homes and unseen approved packaging classes.
- Target 99.5% correct storage-class assignment in the HIL qualification envelope.
- Target 100% custody completeness for accepted items.
- Zero accepted hard-safety violations in the frozen test suite.
- Fewer than five routine human interventions per 100 accepted deliveries after stabilization.
- Target at least five percentage points over the best reproducible public baseline on a genuinely matched evaluation split; no cross-benchmark score substitution.
- Report P50/P95/P99 task time, energy/item, damage, failed grasps, mis-sorts, false accepts, quarantines, human minutes and system availability.

### Safety and quality metrics

Missed human/animal detection and stopping distance; emergency stop response and braking; pinch/crush and reachable-force envelope; lift overload and dropped-load rate; door/egress obstruction duration; food separation and environmental excursions; package damage and tamper detection; local/manual access during power/network loss; privacy leakage and unnecessary image retention.

A zero-violation target applies to the defined tests, not as a claim that real-world risk is zero.

## 9. Design for dignity and architecture

The system must be available in multiple forms:
- simple external parcel wall for small homes;
- utility-room tote lift for compact urban apartments;
- service spine for new-build houses;
- detached delivery pavilion for estates;
- shared community receiving and sorting hub for cooperatives.

Make the mechanism repairable and visually quiet. Access panels, manual drawers, standard fasteners, replaceable modules, acoustic isolation and cleanability matter. A beautiful home should not be forced to look like a warehouse.

Design for people who cannot carry heavy packages or reach high shelves. Accessibility is a core requirement, not a premium option.

## 10. Integration

- **Symthaea:** identifies objects, proposes sorting and movement plans, learns bounded skills, explains uncertainty.
- **Symtropy:** replays physical transfer, contact, obstacles, load and fault scenarios.
- **Mycelix Commons/Transit:** coordinates external delivery service, shared assets, authorization and custody events; household inventory remains private.
- **Sol-Atlas:** supplies site/environment context and home/receiver constraints; not indoor perception.
- **AeroWeft:** provides external aircraft and charging constraints; no indoor flight requirement.
- **NestPort:** is the boundary device and interface to the residence.

## 11. First demonstrator

Build the simplest meaningful chain:
1. One stationary external receiving locker.
2. One barcode/RFID/scale-based local intake.
3. Two standardized tote sizes.
4. One enclosed transfer drawer or short conveyor.
5. One fixed storage rack with three destination classes (ambient, chilled simulation interface, quarantine).
6. One small manipulator or powered drawer to transfer totes.
7. One local state machine with signed receipts, manual access and fail-safe stop.
8. One Symtropy replay scene and one Mycelix Commons receipt integration.

Keep packages sealed. Do not use drones inside the building. Do not automate cutting or food unpacking. Prove reliable receiving, custody, sorting, containerization and storage first.

## 12. Non-claims

No claim of a working home robot, certified storage system, food-safety qualification, medicine handling, fire approval, building approval, or benchmark result is made by this contract. Simulation, software tests, HIL, physical pilot and external qualification must be reported separately.
