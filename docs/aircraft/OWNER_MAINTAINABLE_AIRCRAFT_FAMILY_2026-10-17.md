# Owner-Maintainable General Aviation Family: Product Architecture and Benchmarks

**Date:** 2026-10-17  
**Status:** concept/product-line study; not an aircraft design, approved maintenance manual, certification plan accepted by an authority, or performance prediction.  
**Tracking:** [AIRCRAFT-LINE-001 #7761](https://github.com/Luminous-Dynamics/symthaea/issues/7761)  
**Related:** [aircraft-energy screening #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647), [Roadster/air-road benchmarks #7692](https://github.com/Luminous-Dynamics/symthaea/pull/7692), [aircraft ownership-cost model #7763](https://github.com/Luminous-Dynamics/symthaea/pull/7763), [S4 next-generation personal aircraft concept](S4_NEXT_GENERATION_PERSONAL_AIRCRAFT_2026-10-17.md), [maintenance task-card contract](MAINTENANCE_TASK_CARD_CONTRACT_2026-10-17.md), [candidate-specific mission model contract](CANDIDATE_SPECIFIC_MISSION_MODEL_CONTRACT_2026-10-17.md), [Skyhawk reference data card](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md).

## 1. North star

Build an aircraft family people enjoy owning because it is understandable, comfortable, predictable to maintain, supportable for decades, and honest about what it can carry and where it can go.

The innovation is **maintenance-first product engineering**, not the assertion that a novel wing shape or powertrain automatically makes an aircraft cheaper. Design access, task time, training, parts logistics, reliability evidence, operating-cost accounting, and owner comprehension into the first requirements baseline.

The product promise should be something like:

- **Nothing routine should require unnecessary disassembly.** Inspection and service access are defined before packaging is frozen.
- **The owner understands the aircraft.** Useful health state and maintenance records can be inspected locally and offline, with clear fault meaning and task boundaries.
- **Repairs are planned, not mysterious.** Approved parts, tool lists, task cards, skill level, verification steps, and return-to-service sign-off are accessible.
- **The experience is delightful.** Easy entry/exit, cabin visibility, useful storage, low vibration/noise, good thermal comfort, legible controls, and a cabin that ages gracefully matter as much as cruise speed.
- **Cost claims include ownership reality.** Fuel/electricity, inspection, shop labor, engine/propeller overhaul reserves, battery replacement, insurance, storage, parts shipping and downtime are all visible.
- **Safety boundaries stay intact.** Personally maintainable does not mean owners are encouraged to perform work outside the tasks their certificate, aircraft category, approved instructions and jurisdiction permit.

A well-documented, conventional fixed-wing aircraft that is easy to inspect and repair is the primary control group. Any new technology has to earn its complexity.

## 2. Market and regulatory evidence (reviewed 2026-10-17)

### Existing Cessna product classes are useful mission baselines

Textron's public specifications give the family at least three substantially different reference missions:

- The **Skyhawk** class is a four-occupant, fixed-gear training/personal airplane. The current manufacturer page lists a 180 hp Lycoming IO-360-L2A, 1,157 kg MTOW, 398 kg useful load, 124 ktas maximum cruise speed and 640 nm maximum range. That range is a manufacturer headline under a defined ferry profile, not a universal point-to-point promise; the separate reference card preserves the source conditions. Source: https://cessna.txtav.com/en/piston/cessna-skyhawk
- The **Skylane** class is a four-occupant, 230 hp touring aircraft with a different range/payload-speed trade. The manufacturer page lists 1,406 kg MTOW, 503 kg useful load, 145 ktas maximum cruise speed and 915 nm maximum range. Treat these as published reference claims whose conditions must accompany any comparison. Source: https://cessna.txtav.com/piston/cessna-skylane
- The **Turbo Stationair HD** is a utility mission, not simply a larger Skyhawk. The manufacturer lists six occupants, 310 hp, 1,719 kg MTOW, 654 kg useful load, 161 ktas maximum cruise speed and dual aft cargo doors. Its rough-strip/cargo/water operations imply different structural, access, payload and operating requirements. Source: https://cessna.txtav.com/en/piston/cessna-turbo-stationair-hd

These are competitor baselines, not designs we have validated or numbers we should paste into a new airplane's requirements. The product families should share a service philosophy and evidence tooling—not be forced onto one airframe or one powerplant where the mission physics differ.

Textron also advertises global service support and says it ships 99.6% of *in-stock* parts the same day. This is a manufacturer-reported service metric, not proof of end-to-end parts availability or the result any new entrant can immediately match. It nevertheless identifies a critical benchmark: owning a maintainable aircraft is not enough if the required part is unavailable. Source: https://customer.cessna.com/en/service

### Certification options changed during 2026

The FAA says the MOSAIC final rule was published on 2025-07-24 and makes certification of light-sport-category aircraft under the new framework effective 2026-07-24. The FAA's accepted means of compliance for the new framework include ASTM integration standards for airplanes, gliders, powered-lift and gyroplanes. This creates a current route to evaluate for **some** light two-seat/trainer concepts; it does not automatically make a four-seat touring aircraft or every intended operation a light-sport design. Select the category only after freezing intended use, configuration and operating envelope, and confirm the exact standards/means of compliance with the FAA and certification specialists.

Sources:
- FAA MOSAIC: https://www.faa.gov/aircraft/MOSAIC
- FAA Light-Sport Category Aircraft and accepted means of compliance: https://www.faa.gov/aircraft/gen_av/light_sport
- FAA, “How it Works: Aircraft Certification” (2026-04-23): https://www.faa.gov/newsroom/how-it-works-aircraft-certification

A US owner-maintenance concept must also be separated from an aircraft's serviceability concept. Current 14 CFR 43.3(g) permits qualifying pilot-certificate holders to perform defined preventive maintenance on aircraft they own or operate, subject to exclusions, and the scope is bounded by the applicable rule/Appendix A and records requirements. That is not general permission to perform any repair or inspection. Source: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/section-43.3 and https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/appendix-Appendix%20A%20to%20Part%2043

South Africa requires a separate SACAA path; do not transplant FAA owner-maintenance assumptions. SACAA's public FAQ recommends contacting its Engineering Office for a proposed owner-built aircraft and preparing aircraft type, engine, propeller, material concept, proposed schedule, and a three-view drawing or photograph where possible. This is a prompt for early engagement, not a finding that a proposed design qualifies under a particular South African category. Source: https://www.caa.co.za/information-for-the-public/frequently-asked-questions/

### Electric training aircraft is a distinct operating model

Pipistrel describes the Velis Electro as optimized for local training missions, with endurance appropriate to a 50-minute sortie plus reserve rather than a generic A-to-B range claim. Its public page lists a 2,000-hour motor TBO and an initial 500-hour battery limitation, with a possibility of adjustment as testing/experience accumulates. These manufacturer statements highlight how battery limits, charging dwell, temperature handling, calendar life and replacement cost must be included in electric-aircraft economics. They do not imply the same values apply to another battery or aircraft. Source: https://www.pipistrel.eu/products/general-aviation/velis-electro/

## 3. Proposed family architecture

These are working product roles, not final names or dimensions.

| Working family | Mission and ownership promise | Candidate architecture to study | What must not be assumed |
|---|---|---|---|
| **F2 — Trainer / Light Utility** | Two seats, instruction and local personal flights; high annual utilization; straightforward daily/100-hour service access | Conservative high-wing, fixed-gear, single-engine configuration; compare light-sport-category and normal-category paths where applicable | Do not assume the light-sport route fits all payload, weather, IFR, or commercial-school missions |
| **S4 — Personal / Skyhawk-class successor** | Four seats, training crossover and weekend/family travel; transparent useful load with realistic fuel/reserves; low cognitive and maintenance burden | High-wing, fixed gear, naturally aspirated piston baseline as control; compare approved alternative fuels/propulsion only as separate studies | Do not treat four installed seats as four adults plus full fuel and baggage; payload/mission closure comes first |
| **T4 — Touring / Skylane-class** | Four-seat longer-trip ownership: comfort, real reserves, reliable dispatch, good cruise economy and luggage capacity | Separate wing/structure/propulsion trade, with fixed-gear conventional baseline before any retractable or boosted alternatives | Do not use speed/range headlines to conceal payload, high-altitude, maintenance or ownership trade-offs |
| **U6 — Rugged Utility / Stationair-class** | Four to six seats depending on a rigorously defined configuration; cargo access, field servicing, utility payload and harsh-environment durability | Dedicated fuselage/cargo door, landing gear, structure and powerplant trade; utility aircraft must get its own mission/load spectrum | Do not derive this from S4 by stretching the cabin or using a common structure without re-analysis |
| **E2 — Electric Trainer / Local Flight** | Quiet, repeatable short training sorties with known sortie cadence and charging access | Separate two-seat electric demonstrator; designed around charging, thermal boundaries, safe isolation and battery replacement | No claim that present electric energy storage supports the range/payload of S4/T4/U6; do not compare without equivalent missions and end-of-life battery accounting |

### Product-line sequencing

Do not attempt to certify five new aircraft simultaneously.

1. **Begin with the S4 as the anchor mission definition** because the user's “next-generation Cessna” idea needs a precise Skyhawk-class comparator: seats, mission, payload-with-fuel, comfort, useful load, service tasks and cost boundaries.
2. **Build a serviceability demonstrator before a flying prototype.** Mock up engine/cowl access, electrical/harness routing, battery/alternator access, landing-gear inspection zones, avionics replacement, drain/inspection points, and cabin service. Time actual tasks with representative maintainers, and record reassembly/verification time.
3. **Study F2 as the first flying product candidate** if the intended mission fits the relevant category and a credible certification/production path is found. Its architecture may be simpler and it can validate maintenance documentation, parts operations and service conventions. This still requires a proper business/certification case; it is not an automatic fast path.
4. **Keep E2 as a parallel technology/operations demonstrator** for high-utilization local flights, where its battery and charging lifecycle can be measured honestly. Do not make it the assumed base architecture for the touring family.
5. **Advance T4 and U6 only when S4 service and manufacturing evidence demonstrates what really generalizes.** Reuse task systems, diagnostics, connector/labeling conventions and supply-chain learnings—but keep the structural and flight systems mission-specific.

## 4. Maintenance-first design rules

### Access is a requirement, not a styling afterthought

For every scheduled or likely unscheduled task, record:
- access path, line of sight, lighting, tool envelope, working posture and allowed support equipment;
- tools, calibrated instruments, fasteners and consumables;
- required skill/certificate category and whether an owner is permitted to do the task in each jurisdiction;
- isolation/depressurization/grounding steps where applicable, hazard warnings and stop conditions;
- removal/reinstall dependencies: number of unrelated assemblies removed to reach the target;
- repair, inspection, adjustment and required post-maintenance functional check;
- acceptance criteria, log/record requirements and who may return the aircraft to service;
- measured median and 90th-percentile labor/time, rework, parts waiting and any deviation.

Design for inspection and error prevention: clear routing, readable labels, visible witness points where appropriate, trapped hardware controls, corrosion prevention and drainage, good access to filters/fluids/inspection points, and minimized hidden chafe/fretting. Any physical implementation must use approved aerospace hardware, torque practices, repair limits and materials; “common fasteners” is a standardization goal, not permission to substitute hardware.

### Standardize service interfaces, not everything

Shared across the family where justified:
- maintenance-task and parts-identification system;
- data model for serial/lot, as-built configuration, discrepancies, maintenance actions and return-to-service evidence;
- electrical drawing, harness identification, connector/labeling conventions and diagnostic reporting principles;
- standard tool kits, documented consumables and task-card format;
- local-first service tools, offline manuals, searchable fault guidance and auditable revision history;
- common cabin UX principles and human-factors evaluation;
- procurement, supplier-qualification, parts supersession and service bulletin processes.

Keep mission-specific:
- wings, spar structures and load paths;
- landing gear and cargo-door structure;
- propeller/powerplant size and installation;
- fuel/battery thermal management and fire protection;
- flight controls, envelope, fatigue substantiation and maintenance limits.

A “line-replaceable unit” is a good idea only if the replacement boundary can be safely inspected, reinstalled, functionally checked, and kept within the approved design. Do not use modularity to make a safety-critical connection harder to verify.

### Design the owner experience

- Plain-language explanation of what a fault means, its urgency, confidence and evidence.
- Clear stop-flying/no-go states; never hide a safety-critical discrepancy behind a single reassuring health score.
- Owner-facing checks and tasks shown separately from tasks requiring an appropriately authorized technician or inspector.
- Local, exportable maintenance records; no internet connection required to understand an urgent fault or access the approved manual.
- Repair instructions with expected time, tools, skill, parts, consumables and required follow-up checks.
- Cabin designed around real people: entry/egress, seating, visibility, ventilation/heat, noise, vibration, storage and cleaning.
- No false promise of “DIY repair everything.” A great ownership experience also makes qualified help easy to find when the task requires it.

## 5. Benchmarks: measure ownership rather than slogans

First collect a baseline from reference aircraft and maintainers. Then set quantitative goals. The following are proposed target-setting directions, not present results.

| Dimension | Measure | Benchmark method |
|---|---|---|
| Planned maintenance | Labor hours per 100 flight hours; annual burden at 50, 100 and 200 flight hours | Separate owner checks, legally permitted owner preventive maintenance, licensed labor, required inspection, and engine/propeller/battery reserves |
| Task accessibility | Median and 90th-percentile time for each selected service task, including return-to-service checks | Video or structured observation; identical task boundaries and safety steps across compared products |
| Disassembly burden | Fraction of selected tasks requiring removal of unrelated assemblies; count of parts disturbed | Task-level dependency graph before packaging freeze and after physical mock-up |
| First-time fix | Repairs completed without repeat discrepancy within a defined observation window | Service records with configuration, repeat visit and corrective action linked |
| Parts support | Fill rate, median/p90 order-to-dispatch, order-to-owner, and AOG days | Track in-stock and out-of-stock items separately; include the parts that actually ground the fleet |
| Lifecycle cost | Fully loaded cost per flight hour and per completed passenger-km | Model 50/100/200 hours per year; include acquisition/finance/depreciation, fuel/energy, inspections, labor, parts, overhauls, battery replacement, insurance, storage, downtime and resale sensitivity |
| Mission utility | Payload after realistic fuel/energy, reserve and luggage; completed route with planned reserve | Frozen mission card and candidate-specific mass/energy profiles; disclose aborts and exclusions |
| Dispatch | Completed / eligible attempts with failure and cancellation reasons | Publish denominators, configuration, weather and uncertainty; do not remove inconvenient aborts |
| Comfort | Cabin dBA under defined phases, vibration, thermal comfort, visibility, ingress/egress and workload | Instrumented runs plus structured user studies; do not turn subjective preference into a single unexplained score |
| Reliability | Unscheduled events per flight-hour, repeat discrepancies and AOG duration | Configuration-controlled fleet/service log; stratify by age, environment and utilization |
| Manufacturing | First-pass yield, rework labor/unit, lead time, scrap, supplier escape and configuration traceability | Pilot line with serial units and timestamped records |
| Documentation | Correct task completion and error rate among appropriately trained participants | Supervised, non-flight task trials with stop criteria; do not ask unqualified owners to perform prohibited or safety-critical maintenance |

### Lifecycle-cost rule

Publish both annual owner cost and utilization-normalized cost. Low annual hours can make fixed costs dominate; high utilization can make fuel, maintenance labor, battery life, engine reserves and dispatch matter more. A comparison that omits downtime, overhaul/replacement reserves or parts availability is not a credible “cost-effective” claim. AOPA gives a useful public overview of separating direct, fixed and reserve costs: https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/operating-costs-and-financing.

### Proposed numeric goals to consider after baseline capture

- **Routine task goal:** 80% of eligible non-structural service tasks completed within 60 minutes, including prescribed verification. Publish the exact task denominator; do not count tasks that require deeper inspection as simple swaps.
- **Service information:** 100% of serial-numbered aircraft have a complete as-built configuration, applicable manual/service-document revision, component/lot identity where required, and maintenance history access.
- **Parts process:** 95% or better fill rate for the predefined high-turn parts set at the declared service hub, measured separately from all catalog part numbers and all regions. State geography/stock policy; do not claim global fill rate from a single hub.
- **Repair quality:** establish first-time-fix and repeat-discrepancy baselines, then set improvement targets only after enough events exist.
- **Cost:** aim for 20% lower fully loaded lifecycle cost per completed mission than an equivalent frozen baseline as a design hypothesis; do not claim this until independent model review and real service/manufacturing data support it.
- **Manufacturing:** target 10% lower rework labor/unit against a measured pilot-line baseline once the production process is stable.
- **Comfort:** specify measurable task-specific targets only after baseline instrumented comparisons, then pair them with user ratings and clear protocols.

The numbers are targets to test, not promises. If measured results miss a goal, expose the failure and change the design or goal with a documented reason.

## 6. How Symthaea, the factory scheduler and Mycelix can support the line

- **Symthaea aircraft-energy work:** screen candidate mission profiles and battery assumptions. It must keep fixed-profile screening distinct from a vehicle model that updates loads when mass and geometry change.
- **Candidate mission contract:** preserve configuration, route, mass, environment, model version, phase demand, uncertainty and evidence digests. The current validator checks input consistency; it does not validate aerodynamics or physical truth.
- **Ownership cost model:** the new [Rust ownership-economics PR #7763](https://github.com/Luminous-Dynamics/symthaea/pull/7763) separately accounts for fixed ownership costs, owner/professional labor, hourly operating costs, reserves, failed dispatches and completed passenger-kilometres. Its example inputs are synthetic, and real affordability claims still need sourceable costs and observed maintenance/dispatch data.
- **Factory operations research:** schedule operations with skills, tools, resources and dependencies; use measured time/rework values only after observing a real process. The current scheduler is a heuristic, not proof of globally optimal line planning.
- **Mycelix evidence layer:** preserve traceability and governed sharing for as-built parts, repairs, revisions, inspections and training. Provenance is not proof that a measurement is correct; bad evidence must be correctable and auditable.
- **Maintenance task cards:** start as versioned, reviewed schema/documentation, then evaluate a suitable implementation location after architecture ownership review. Do not create a second overlapping aircraft platform or a parallel crate just to hold a product matrix.
- **Praxis/technician training:** assess task competency and documentation usability only after confirming the actual system/API and who is authorized to perform the task.

## 7. Stage gates

**G0 — requirements and user needs:** define personas (school, owner-pilot, touring family, utility operator), actual missions, terrain/weather and ownership economics. Interview maintainers and aircraft owners; don't infer usability solely from engineering preferences.

**G1 — reference dataset:** preserve current Skyhawk/Skylane/Stationair manufacturer data with source conditions; add maintenance task and parts/service baselines. The public brochure alone is not a validated mission model.

**G2 — maintenance demonstrator:** timed physical access/mock-up tasks and service documentation review. Include blind tasks by trained maintainers; report errors, median/p90, rework and inaccessible steps.

**G3 — vehicle trade study:** candidate geometry/weight/performance model with traceable inputs, mass closure, uncertainty and independent review. Reject options that only win under undocumented optimistic assumptions.

**G4 — certification and production plan:** decide intended jurisdiction/category with competent authorities and qualified certification professionals; define compliance, quality, configuration control, continued operational safety, maintenance instructions and production evidence.

**G5 — prototype and test program:** only after qualified engineering review; use an approved/appropriate test program, instrumentation, hazard controls, and independently reviewed acceptance criteria.

**G6 — fleet learning:** service records, failure analysis, parts, time-on-task, comfort and cost data feed controlled revisions; maintain compatibility and continuing-airworthiness obligations.

A passed software test is not physical validation. A passing conceptual model is not an airworthiness approval. No human-carrying or public-flight authorization is inferred by this roadmap.

## 8. Immediate work plan

1. Freeze an S4 reference mission and an owner-maintenance persona set.
2. Build the maintenance task taxonomy and timed-task templates before detailing geometry.
3. Collect baseline requirements from Cessna's Skyhawk, Skylane and Stationair published data and from appropriate owner/maintenance sources; mark unknown data explicitly.
4. Construct a lifecycle-cost model that keeps acquisition, energy/fuel, utilization, labor, overhauls/replacements and downtime visible.
5. Run the same maintenance-task protocol on a mock-up before any aircraft-level performance optimization.
6. Feed evidence requirements into the existing Symthaea energy, factory and provenance work.
7. Contact the relevant authority/certification experts early; keep FAA/MOSAIC and SACAA pathways separate.

## 9. Research sources

- Textron Aviation, [Skyhawk specifications](https://cessna.txtav.com/en/piston/cessna-skyhawk).
- Textron Aviation, [Skylane specifications](https://cessna.txtav.com/piston/cessna-skylane).
- Textron Aviation, [Turbo Stationair HD specifications](https://cessna.txtav.com/en/piston/cessna-turbo-stationair-hd).
- Textron Aviation, [service and parts support](https://customer.cessna.com/en/service).
- FAA, [MOSAIC](https://www.faa.gov/aircraft/MOSAIC).
- FAA, [Light-Sport Category Aircraft and accepted consensus standards](https://www.faa.gov/aircraft/gen_av/light_sport).
- FAA, [Aircraft certification process](https://www.faa.gov/newsroom/how-it-works-aircraft-certification).
- eCFR, [14 CFR 43.3](https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/section-43.3) and [Appendix A to Part 43](https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/appendix-Appendix%20A%20to%20Part%2043).
- South African Civil Aviation Authority, [public FAQ / owner-built aircraft guidance](https://www.caa.co.za/information-for-the-public/frequently-asked-questions/).
- Pipistrel, [Velis Electro performance and battery-life limits](https://www.pipistrel.eu/products/general-aviation/velis-electro/).
- AOPA, [aircraft operating costs and financing](https://www.aopa.org/go-fly/aircraft-and-ownership/buying-an-aircraft/tips-on-buying-used-aircraft/operating-costs-and-financing).

## Nonclaims

This document is not an aircraft design, a performance prediction, an approved maintenance program, legal advice, or an accepted certification plan. It does not claim that any proposed aircraft is feasible, airworthy, cheaper, or ready to build. All maintenance permissions are jurisdiction-, category-, aircraft- and task-specific. Qualified engineering and regulatory review is required before physical implementation.
