# Symthaea Transport: Repairability-First Personal and Family Mobility

**Status:** Research proposal; requirements and targets are provisional, not validated vehicle specifications.  
**Date:** 2026-10-17  
**Scope:** A reproducible engineering program for affordable personal and family transport, beginning with a benchmark study before committing to a clean-sheet vehicle.

## Thesis

The objective is not the cheapest vehicle to manufacture in isolation. It is the lowest *verified lifetime cost of safe, useful mobility* for the people and roads the vehicle is designed for.

A successful design should be affordable to acquire, efficient for its real duty cycle, understandable to independent repairers, supportable with locally available parts, and safe throughout its service life. Repairability is a system property: service access, fault isolation, repair information, parts supply, diagnostics, durability, and the economics of keeping the vehicle all matter.

This document opens a research workstream. It does not assert that Symthaea can certify a design, that a proposed architecture is already optimal, or that a simulation result establishes roadworthiness.

## First paradigm shift: optimize service life, not sticker price

Vehicle architecture should be evaluated on a lifecycle model that includes:

- Purchase and financing cost, depreciation, taxes and registration.
- Real-world fuel or electricity consumption on representative routes.
- Scheduled maintenance, common repairs, tyres, consumables and replacement parts.
- Insurance, likely downtime, towing and the availability of repair facilities.
- Parts lead times, end-of-life reuse, remanufacturing and recycling.
- Safety and practical utility: passenger space, child-seat fit, luggage, weather protection, payload and road capability.

Report assumptions and uncertainty. Do not hide a poor result by selecting a favourable energy price, annual mileage, occupancy rate, resale value or repair-labour assumption.

Compare internal-combustion, hybrid and battery-electric candidates against actual use profiles instead of declaring one powertrain universally superior in advance. Charging access, route lengths, stop-start operation, climate, local repair capability, purchase price and energy prices can change the ranking.

## Initial vehicle portfolio hypotheses

Keep vehicle classes distinct; a low-speed urban runabout must not be presented as a substitute for a road-speed family car.

1. **Primary target — compact family vehicle.** Four to five belted occupants, practical child-seat installation, useful luggage space, climate control, good visibility, predictable handling and enough ground clearance for the chosen operating environment. Prefer simple, proven, widely serviceable mechanical systems where lifecycle evidence supports them.
2. **Shared-component utility derivative.** A cargo/work variant may reuse proven service parts and selected assemblies, but only where commonality saves real money without compromising occupant protection, payload or durability.
3. **Optional urban runabout.** Investigate separately if user research shows a need. Define its speed and road-use class explicitly, and do not blur its capabilities with those of the family vehicle.

These are hypotheses for comparison, not a decision to build three vehicles. Begin with one well-defined mission and the smallest credible demonstrator.

### Powertrain and platform strategy

For the first study, compare:

- A proven small-displacement combustion drivetrain with broad local repair support.
- A hybrid option if the duty cycle and purchase-cost model justify its added complexity.
- A battery-electric option with a serviceable, protected battery system and a realistic charging plan.

Do not assume that a clean-sheet engine, proprietary electronics, a bespoke transmission or a fully novel battery pack is necessary. Reuse established components when that produces a better safety, cost, spares and support outcome. A donor-platform prototype may be a safer and cheaper first experiment than a new road-going vehicle design.

## Second paradigm shift: make serviceability measurable

Treat the maintenance plan as an input to architecture, not a manual written after design freeze.

### Design hypotheses to test

- Give routine service items direct, clearly labelled access paths.
- Use a small, documented set of common fasteners and tool sizes where practical.
- Make wear items and frequently damaged non-structural parts economical to replace independently.
- Provide stable diagnostic interfaces, readable fault codes, wiring diagrams, torque specifications, parts catalogues and repair instructions that remain usable offline.
- Separate replaceable trim, lamps and sacrificial exterior pieces from the engineered occupant cell and safety-critical load paths.
- Design for diagnosis as well as replacement: a technician should be able to distinguish the likely failed component from symptoms without indiscriminate parts swapping.
- Keep essential maintenance and repair possible without a cloud account, recurring software subscription or an exclusive dealer tool, subject to lawful security and safety controls.
- Use modularity selectively. Extra connectors, interfaces, brackets and module boundaries can add cost, mass, failure modes and assembly effort; each proposed module needs a measured benefit.

**Safety boundary:** “Repairable” does not mean that owners should casually open high-voltage battery packs or modify steering, braking, restraints or structural load paths. Those systems require engineered procedures, suitable equipment and qualified verification.

## Third paradigm shift: a serviceable vehicle platform plus an open repair ecosystem

A vehicle is only maintainable if its parts and knowledge remain obtainable.

The program should explore a public, versioned service-data package containing a bill of materials, part identifiers and substitutions, maintenance schedules, wiring diagrams, diagnostic definitions, repair sequences, compatible tool requirements and relevant safety notices. Parts availability and price should be tracked by target market rather than assumed from an international catalogue.

Avoid vague “standard parts” claims. For every proposed common component, record its identifier, dimensions or interface, actual sources, price, availability date, and any safety or quality qualification required before substitution. A cheap component that is intermittently unavailable may be more expensive over the vehicle's lifetime than a slightly more expensive, reliably sourced part.

## How Symthaea should contribute

Symthaea should initially be an **engineering copilot and design-space exploration system**, not an unverified authority over vehicle hardware.

1. **Requirements graph:** Connect user needs and use profiles to measurable requirements, design choices, constraints, hazards, analyses and physical tests. Flag missing evidence and contradictory assumptions.
2. **Architecture trade studies:** Generate and rank candidate layouts and powertrains using lifecycle cost, mass, energy use, serviceability, parts availability, durability and safety constraints. Show Pareto trade-offs rather than disguising all objectives in one opaque score.
3. **Repairability graph:** Model each service task as diagnosis, access, isolation, removal, replacement, reassembly and verification. Estimate time only when assumptions are explicit; replace estimates with timed workshop observations as soon as possible.
4. **Robust optimization:** Sweep uncertainty in energy prices, annual distance, passenger load, repair labour, part lead time, climate and component failure rates. Prefer architectures that remain acceptable across plausible scenarios, not just one ideal case.
5. **Failure analysis and maintenance experiments:** Organize FMEA-style failure modes, observed symptoms, diagnostic coverage, repair history and fleet feedback. Evaluate predictive-maintenance ideas against false alarms, missed failures and actual downtime savings.
6. **Reproducible evidence:** Version inputs, parameter files, solver versions, test data and outputs. Distinguish measured observations, published data, simulation, estimates and unresolved questions in every report.

Use Rust for reproducible orchestration, data validation and optimization components where appropriate, and connect to established CAD/CAE and dynamics tools for the physics. Do not recreate validated solvers merely to claim a Rust-only stack. Vehicle dynamics, structural analysis, braking, thermal behaviour, battery safety and crash performance need suitable models with known limits and validation against physical evidence.

Any Symthaea-generated result must remain advisory until independently checked. If future work reaches safety-related control, independent conventional safety mechanisms, fault containment, deterministic limits and qualified review must remain outside the authority of the cognitive model.

## Benchmark plan

Establish the baseline before setting hard numeric claims. Select three current, actually available vehicles in the intended market and class, document their variants and condition, and measure them with the same procedures. Published specifications alone are not a repairability or lifetime-cost benchmark.

### A. Lifetime economics

Model 10 years / 150,000 km as an initial comparison horizon, then test lower-mileage, high-mileage and shorter/longer ownership cases.

Record:
- Total cost of ownership and cost per usable passenger-kilometre.
- Scheduled and unscheduled service cost, including labour and parts.
- Energy consumption on defined city, mixed and intercity cycles, with load and climate recorded.
- Parts cost and lead time for the most common maintenance and repair tasks.
- Assumptions, ranges and sensitivity results for resale value, energy prices, repair rates and labour cost.

**Provisional ambition:** Target at least a 15% reduction in modelled 10-year ownership cost versus a matched, available baseline while meeting the same safety and utility requirements. Treat this as a design hypothesis to test; revise it if baseline measurement shows it is not credible.

### B. Repair and maintainability

For a pre-registered set of at least ten common jobs, measure hands-on time and elapsed time separately. Start with routine inspections, filters/fluids where relevant, 12 V battery, lamps, wipers, wheel/tyre access, brake service access, diagnostic fault isolation and replacement of common non-structural damage parts.

Record for each job:
- Median and 90th-percentile completion time, including diagnosis and post-repair checks.
- Tool list, specialist-tool dependencies, fastener count and components that must be removed for access.
- Parts required, repair information available, and whether the task can be completed by a competent independent workshop.
- Errors, damaged clips/connectors, rework and unsafe service states encountered.

**Provisional ambition:** Improve median and 90th-percentile time on high-frequency service tasks by at least 25% relative to the baseline, without increasing safety risk or moving costs elsewhere. Confirm with timed, repeatable workshop trials rather than CAD access alone.

### C. Parts and repair independence

For the top 20 service and repair parts identified by expected cost, frequency and downtime impact, publish local prices, suppliers, lead times, interchange restrictions and the documentation needed to fit them safely.

Measure the share available through independent channels, time to diagnose with a standard workshop tool, and the number of jobs requiring proprietary software or account access. Set market-specific availability targets only after supplier research.

### D. Durability in the target environment

Define representative use profiles before testing. For a Southern African launch scenario, investigate heat, dust, rough surfaces/potholes, long intercity trips, payload and repair-network distances with local drivers and technicians instead of relying on assumed conditions.

Combine instrumented road tests, component rigs and accelerated testing. Correlate accelerated-test predictions with observed field failures. Track unscheduled repairs per distance, time out of service, wear rates, water/dust ingress, suspension and tyre damage, electrical faults and degradation with age. A simulated lifetime is not a demonstrated lifetime.

### E. Safety and legal readiness

Safety is a hard constraint, not a weighted objective that can be traded away for lower cost or weight.

- Identify the actual regulatory route before choosing a road-going architecture or selling a kit.
- Require independent engineering review and the applicable vehicle/equipment approvals before public-road use.
- Include occupant protection, restraints, practical child-seat installation, braking, handling, lighting, tyres, visibility and pedestrian safety in the requirements.
- Use validated simulation to guide design and reduce wasted prototypes, then verify with physical tests and the applicable approval process.
- Keep the test vehicle, its configuration, software and component revisions tied to the reported results.

Euro NCAP's published methodology describes destructive crash testing as central to its assessment and uses validated simulation to extend analysis. A good simulation result—or a consumer safety rating used as a proxy—is not by itself legal type approval.

### F. Repairability and durability are not enough without utility

Every result should be paired with mission requirements: usable seats, secure child restraints, luggage volume, payload, weather protection, range, route capability and driver comfort. Do not “win” the maintenance benchmark by reducing the vehicle to a low-speed or exposed design that cannot safely meet the family transport mission.

## Proposed sequence of work

**Phase 0 — Define the mission.** Choose market, users, routes, passenger/cargo requirements, budget ceiling and powertrain constraints. Separate urban-only needs from family road travel.

**Phase 1 — Build the evidence baseline.** Select three locally available comparator vehicles; collect specifications, actual parts quotes, service procedures and technician timings. Mark unavailable evidence rather than filling gaps with guesses.

**Phase 2 — Build the design-space model.** Create a traceable requirements set and comparable architecture options. Estimate lifecycle cost with explicit uncertainty, then identify which measurements most affect the ranking.

**Phase 3 — Serviceability prototype.** Use an existing donor vehicle or subsystem to test access, replaceable non-structural parts, diagnostic procedures and open service documentation before touching the occupant cell or safety-critical structure.

**Phase 4 — Validated demonstrator.** Only after a credible architecture emerges, work with qualified automotive engineers and test facilities on CAD/CAE, structural integrity, braking, stability, crashworthiness, durability and regulatory approval.

**Phase 5 — Repair and manufacturing ecosystem.** Validate supply chains, independent workshop procedures, repair manuals, replacement parts and locally serviceable production economics.

The first deliverable should be the **market-specific benchmark protocol and requirements matrix**, not a speculative render or an unsupported claim that a new vehicle is ready to build.

## Cross-project architecture: reuse the existing transport owners

A source review on 2026-10-17 found that we should specialize existing work instead of creating another generic transport framework.

- Symthaea [TRANSPORT-ENG-001 #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244) is the shared transport engineering architecture.
- Symthaea [TRANSPORT-ROAD-001 #6247](https://github.com/Luminous-Dynamics/symthaea/issues/6247) already owns tire/contact, braking, handling, stability, payload/CG and vehicle qualification semantics. The family-car program should add a bounded passenger-vehicle profile and benchmark there, not duplicate a road-dynamics solver.
- Symthaea [TRANSPORT-CLOSURE-001 #6253](https://github.com/Luminous-Dynamics/symthaea/issues/6253) owns productive-closure/resilience concerns; [TRANSPORT-PILOT-001 #6254](https://github.com/Luminous-Dynamics/symthaea/issues/6254) covers the low-consequence physical qualification ladder.
- Symtropy [TRANSPORT-OPT-001 #1601](https://github.com/Luminous-Dynamics/symtropy/issues/1601) already owns whole-life social-cost transport comparison, and [TRANSPORT-SAFETY-001 #1636](https://github.com/Luminous-Dynamics/symtropy/issues/1636) addresses physical-load qualification.
- Mycelix has transport under Commons, but [DORMANT_TRANSPORT.md](https://github.com/Luminous-Dynamics/mycelix/blob/main/mycelix-commons/zomes/DORMANT_TRANSPORT.md) records serious correctness, deployment and multi-agent test gaps for route/sharing/impact zomes. Treat existing transport capabilities as dormant until their exact current head, WASM builds, HApp packaging and real multi-agent flows are independently verified.

### Evidence path across all three projects

```text
Symthaea requirements + candidate design
        -> Symtropy versioned simulation + Pareto/TCO scenarios
        -> independent physical testing / engineering review / applicable authority
        -> Mycelix owner-consented asset dossier + repair/part/inspection claims
        -> minimized, consented field observations feed future design
```

Use immutable artifact references and typed claim classes across repositories, not synchronous dependencies. The service-procedure/repairability input is Symtropy's [Repairability Task Graph v0.1](https://github.com/Luminous-Dynamics/symtropy/blob/research/family-vehicle-simulation-v0-1/schemas/transport/repair_task_graph_v0_1.schema.json), which captures tasks, dependencies, tools, safety preconditions, parts revisions, timing basis and post-repair verification. That graph describes a procedure or test plan; Mycelix's maintenance event describes what a named actor asserts was actually done. An estimated duration must not become measured performance, and an observed-shaped synthetic fixture must not be counted as a physical workshop result. The common machine-readable contract is [Mobility Engineering Evidence Envelope v0.1](https://github.com/Luminous-Dynamics/symtropy/blob/research/family-vehicle-simulation-v0-1/schemas/transport/engineering_evidence_envelope_v0_1.schema.json), maintained beside Symtropy's model substrate so design, simulation, measurement, independent review and regulatory references share one versioned vocabulary. It binds schema version, artifact identity, exact subject/configuration, producer repository/revision/tool, input/output digests, evidence class, scope/conditions, uncertainty, source/rights references, privacy class, limitations and claim ceiling. Its fixture set explicitly rejects promoting a simulation to a physical test or an independent review to a regulatory receipt. The schema validator still needs exact-head CI execution; source presence is not qualification. Owner identity, financial agreements and raw trip/location data are excluded from the public engineering envelope by default. Mycelix's vehicle-event signature/consent envelope remains separate: the common engineering reference does not replace event authorization, and a signed event does not upgrade the evidence class.


For an initial South African comparison, Symtropy maintains a dated [baseline screening shortlist](https://github.com/Luminous-Dynamics/symtropy/blob/research/family-vehicle-simulation-v0-1/fixtures/personal_vehicle/south_africa_baseline_shortlist_v0_1.json) covering Toyota Vitz, Suzuki Swift, Toyota Starlet, Starlet Cross and Rumion profiles. It records official advertised-from pricing and sources/unknowns, not dealer quotations, total ownership costs, safety rankings or donor-platform decisions. The exact grade/year, written quotes, safety/utility fit, parts baskets and physical service-task observations still need to be captured before the values can drive an optimization.

Keep authority separate: Mycelix records and validates signed claims but does not establish legal title, workmanship or safety by signature alone; Symtropy simulation is not physical qualification; Symthaea does not certify its own designs. No consciousness or Phi measure may grant safety-critical actuator authority.

### Additional paradigm shifts and testable targets

- **Repair cost is designed in:** optimize the tasks/components that contribute most to lifetime service and collision-repair cost, not modularity as an aesthetic. SAE J1555 links repairability, serviceability and diagnostics and recommends focusing on the largest repair-cost contributors ([source](https://saemobilus.sae.org/standards/j1555_201910-recommended-practice-optimizing-automobile-damageability-repairability)).
- **Optimize under uncertainty:** compare actual local powertrain/platform options across mileage, energy prices, repair labour, downtime, supplier availability and climate; report Pareto alternatives, not an opaque score.
- **Use provisional numeric targets as hypotheses:** at least 15% lower modelled 10-year/150,000 km TCO than a matched local baseline with equivalent safety and utility; at least 25% lower median and 90th-percentile time for 10+ preregistered common service jobs, physically measured; and dated local price/lead-time/supplier data for the 20 highest-impact repair parts.
- **Simulation must prove usefulness before it steers decisions:** on independently held-out measured profiles, initially target about 10% error for explicitly defined energy and braking observables, while reporting absolute error, confidence/measurement uncertainty, exclusions and out-of-domain cases. This is a design-screening ambition—not certification.
- **Autonomy is a later, separately qualified program:** use an explicit operational design domain, frozen nominal/edge/adversarial/negative scenarios, sensor/actuator/software fault injection, fallback metrics and a safety case. Zero collisions within a finite simulation corpus is not proof of road safety. NHTSA's [published ADS reports](https://www.nhtsa.gov/automated-vehicles-safety/published-reports-and-documents) distinguish modeling, simulation, track and open-road testing; [ISO 21448](https://www.iso.org/standard/77490.html) addresses SOTIF but is not satisfied merely by referencing it.

## Decision gates

Do not advance an architecture merely because an optimizer ranks it first.

- **Gate 1:** Mission and constraints are explicit; evidence sources and assumptions are traceable.
- **Gate 2:** Baseline measurements reproduce across repeated trials.
- **Gate 3:** The candidate shows a defensible cost/serviceability advantage without relaxing safety or mission requirements.
- **Gate 4:** Key physical predictions are validated against measured data.
- **Gate 5:** A qualified review confirms a credible route to legal approval, production quality and field servicing.

A failed gate should trigger redesign or a narrower scope, not a weaker benchmark.

## Initial references

- Euro NCAP, [How cars are tested for safety](https://www.euroncap.com/how-are-cars-tested-for-safety/).
- SAE International, [J1555: Recommended Practice for Optimizing Automobile Damageability and Repairability](https://saemobilus.sae.org/standards/j1555_201910-recommended-practice-optimizing-automobile-damageability-repairability).
- South African Government, [Letter of authority for an imported or rebuilt motor vehicle](https://www.gov.za/services/register-motor-vehicle/letter-authority-respect-motor-vehicle). The approval route should be checked against the exact vehicle and current requirements before any South African road use.
- South African Government, [Roadworthy certificate](https://www.gov.za/services/services-residents/driving/roadworthiness/roadworthy-certificate).

**Open questions to resolve before vehicle selection:** initial market; family size and child-seat needs; urban versus intercity versus rough-road share; new-build versus donor-platform preference; purchase budget; access to home/work charging; and whether the priority is an affordable road car, a runabout, or a broader family of transport products.
