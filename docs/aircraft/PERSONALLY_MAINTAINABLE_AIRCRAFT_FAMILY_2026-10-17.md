# Personally Maintainable Aircraft Family: Product Architecture and Benchmarks

**Date:** 2026-10-17  
**Status:** concept architecture and test plan; not an aircraft design approval, build specification, performance prediction, or authorization to maintain/fly an aircraft.  
**Tracking:** [AIRFAMILY-001 #7745](https://github.com/Luminous-Dynamics/symthaea/issues/7745)  
**Research context:** [Roadster / air-road platform benchmark #7692](https://github.com/Luminous-Dynamics/symthaea/pull/7692), [aircraft-energy screening #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647), [candidate-specific mission model contract](CANDIDATE_SPECIFIC_MISSION_MODEL_CONTRACT_2026-10-17.md), [Skyhawk reference card](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md).

## 1. Product ambition

Build the aircraft family people *want to keep*: understandable, comfortable, inexpensive to keep available, thoughtfully documented, repairable with long-lived parts, and enjoyable for years. The competitive goal is not the highest headline speed or a futuristic exterior. It is the best honest mix of mission utility, purchase and operating cost, safety, maintainability, parts support, cabin experience and owner confidence.

The family should learn from the best of the Cessna design philosophy—conventional, useful, familiar aircraft that support training and everyday personal flying—without assuming that copying a Skyhawk layout or published performance numbers is the same as improving it. The Cessna Skyhawk is a baseline comparator, not a design to copy or an endorsement.

**Core principle: every maintenance task is part of the product.** Access, visibility, tools, skills, parts, manuals, approval boundaries, and return-to-service checks must influence the layout while it is still cheap to change. NASA's maintainability guidance explicitly treats maintainer access, tools, skill level, support equipment and task frequency as system-design concerns, not late service manual work.

## 2. Recommended family: three core products, one gated technology branch

These names are working labels only. They do not imply finalized specifications or brands.

### A. Wren 2 — two-seat trainer and local flyer

**First product candidate.** The job is training, proficiency, local recreation, short practical trips, and owner-pilot confidence.

Begin by evaluating a conventional fixed-gear, single-engine layout with excellent visibility, robust ground handling, accessible routine service points, readable instrumentation, and a simple cockpit workflow. Keep the powerplant, airframe material and certification category as trade-study choices, not assumptions. A training-focused airframe should be designed for frequent cycles, student handling, and fast return to service rather than minimum empty weight alone.

**Why first:** it creates serviceability and training evidence that can inform the whole family while limiting the number of seats, mission variants and cabin systems.

### B. Wayfarer 4 — four-seat personal touring, a Skyhawk-class successor

The job is useful local and cross-country travel for owner-pilots, with practical bags, comfortable ingress/egress, good visibility, cabin ventilation/heating and maintainable systems.

The primary requirements are real payload with usable fuel and the declared reserve, comfortable cabin geometry, accessible baggage, a calm ride, a supportable powerplant, and transparent total cost per flight hour. Separate marketing range from a fully specified mission profile. Every range/payload result must disclose people/baggage, environmental assumptions, cruise policy, reserves, and exclusions.

Treat retractable gear, pressurization, turbocharging, complex hybrid power, and novel flight-control architecture as options only if a frozen use case demonstrates a compelling benefit that offsets added maintenance, training, mass and certification burden. The default comparison is a conventional, fixed-gear single-engine design—not because it is automatically best, but because complexity must earn its place.

### C. Ranger 4 — utility / rough-field derivative

The job is practical load-carrying, farm and regional access, unpaved strips where lawful and approved, and mission flexibility for owners who need a tougher tool.

Start with a distinct requirement set for payload, ground clearance, surface condition, corrosion and debris exposure, inspection access, tie-down/handling, braking and tire service. Do not merely lift the Wayfarer and add oversized tires. The utility version may warrant a different structure, wing, landing gear and certification basis. Shared parts are a benefit only when load ratings, inspection intervals, interchangeability and approved configuration support it.

### D. Wren-E — electrified trainer demonstrator (gated branch, not a promised product)

The job is very short, repeatable training/local missions at a site with dependable charging and a utilization pattern that could justify battery replacement and ground infrastructure.

Do not make this the first aircraft or assume electric propulsion makes ownership cheaper. Require a mission closure covering installed battery mass, end-of-life capacity, temperature derating, reserve, charging losses, turnaround, battery replacement, thermal management, dispatch effects and local site/grid requirements. Begin with ground powertrain and charging tests, then an appropriately approved uncrewed/subscale test programme if the case survives. Passenger operation requires the separate applicable certification and testing path.

## 3. Family architecture: commonality without a common-airframe trap

Use a **common service architecture**, not compulsory identical parts.

### Standardize where evidence shows a win

- Owner-facing cockpit logic, labels, alerts, checklist style, and technical-document navigation.
- Maintenance task-card format, part/revision identity, serial-number configuration, and service-record export.
- Common service toolkits, common diagnostic connector and offline diagnostic/export workflow, where appropriate to the certified system.
- Electrical connector families and documented pinouts for non-interchangeable functions; coded/keyed interfaces to reduce misconnection. Any safety-critical interface requires the qualification dictated by the applicable design.
- Cabin touchpoints (handles, seat adjustment approach, baggage restraint concepts, lighting controls) where geometry and loads truly align.
- Routine consumables, filters, tires/brakes, lamps and service hardware only where ratings, environment, approved parts data and life limits permit commonality.
- Common maintenance training, parts search, illustrated parts catalogue conventions and service-bulletin communication.
- Shared non-flight-critical modules only where integration, failure analysis, electromagnetic/environmental suitability and approvals support the reuse.

### Do not force commonality where it harms the aircraft

Do not impose one wing, airframe, landing gear, powerplant, propeller, battery, or certification route across products just to advertise parts reuse. Different missions can justify different structures and power ratings. The shared platform should be interfaces, evidence, service tools, repair doctrine, owner training and support—not arbitrary percentage commonality.

### Material and powerplant trade studies

Compare aluminum alloys, composites, steel/tubular and other plausible structures using the whole lifecycle: mass, purchase/manufacturing cost, fatigue/corrosion, inspection burden, repair access, field-repair capability, special materials/processes, repair qualification, supplier resilience and recyclable/end-of-life paths. The lightest airframe is not necessarily the most maintainable or economical.

For propulsion, compare only properly scoped candidates with applicable source data: acquisition, fuel/energy, overhaul/replacement reserves, availability, service skill requirements, parts lead time, noise/vibration, environmental exposure and certification/integration effort. Use conventional certified piston propulsion as an honest baseline; alternative power should win against it in a stated mission, not in a brochure slogan.

## 4. Personal maintainability means legal, safe, bounded tasks

Design for the owner to understand the aircraft and safely perform tasks that the aircraft's governing rules and approved documentation actually permit. It does **not** mean that every owner may perform structural repair, alter flight controls, open an engine, change a safety-critical system, sign a maintenance release, or approve the aircraft for return to service.

Use four product-design task classes. They are workflow categories, **not a replacement for a jurisdiction-specific legal determination**.

| Class | Purpose | Design obligation |
|---|---|---|
| P0 — owner observation and normal servicing | Preflight/postflight observation, cleaning, documenting symptoms, and normal checks described in the approved operating documentation | Clear access, unambiguous labels and limits, no unsafe implication that a check substitutes for required inspection |
| P1 — owner-performed work when permitted | Tasks a suitably eligible owner may legally perform on this aircraft and under this certification basis | Each task explicitly states eligibility, source of authority, prerequisites, tools, parts, inspection steps and required record/approval; otherwise the task is marked "refer to authorized maintainer" |
| P2 — trained repairman/mechanic work | Scheduled inspections, repairs, adjustments, component replacement, and return-to-service actions requiring specific training or authorization | Reachability, safe isolation, calibrated/test equipment access, standard work package, inspection/verification and records |
| P3 — approved specialist/manufacturer work | Major structural repairs, engine internal work, high-energy battery pack service, safety-critical software/configuration changes or other specialist-restricted tasks | Minimize avoidable disassembly and grounding; provide specialist fixtures, approved repair paths, traceable exchange/repair parts, and clear escalation support |

For every task, list what tools are required, expected skill level, estimated time, access panels/assemblies removed, fastener/connector count, safety hazards, energy isolation steps, required measurements, acceptable result, independent inspection or sign-off requirement, return-to-service authority, and maintenance-record fields.

**Never label a task "owner-serviceable" until the applicable type/certification and local legal review have established what that means.** In the US, FAA MOSAIC certification/maintenance changes and light-sport repairman privileges have category- and training-dependent limits. Standard-category aircraft have different maintenance privileges from light-sport and experimental aircraft. A product team must confirm the actual certification basis, not infer permissions from the aircraft's marketing label. In South Africa, confirm the applicable SACAA certification category, approved maintenance programme, Part 43/related requirements and Part 21/24 pathway before release. Europe requires a separate EASA/competent-authority analysis.

## 5. Proposed serviceability and ownership benchmarks

These are project targets for prototype evaluation, not existing performance, industry requirements or certification thresholds. Freeze a comparable baseline before claiming improvement.

| Metric | Proposed target | How to test and report it |
|---|---|---|
| Routine task access | At least 90% of planned routine service tasks reachable without removing unrelated assemblies | Task-by-task access audit against CAD/service mock-up; count tasks, disassembly steps and blocked access |
| Owner-authorized routine tasks | At least 80% of eligible tasks completable within 30 minutes at the 90th percentile for trained representative owners, where the rules permit the task | Use a frozen task list; measure from setup through required checks and log entry; do not omit preparation or clean-up |
| Scheduled maintenance labor | At least 20% lower technician labor hours against a frozen matched baseline after equivalent task coverage | Compare same aircraft class, utilization and inspection scope; publish exclusions and uncertainty |
| Unscheduled return to service | At least 30% lower median maintenance-controlled grounding time for comparable repair events | Include diagnosis, parts wait, access, repair, required verification and log/approval time; report 90th percentile too |
| First-time fix | At least 90% of eligible service events fixed without repeat visit or rework | Track serial numbers, reported defects, repair action and recurrence window; never merge duplicate events silently |
| Routine tool burden | At least 90% of routine tasks use a documented standard owner toolkit or common mechanic toolkit; special tools explicitly listed for specialist tasks | Tool/task matrix, including calibration and access requirements; do not ban special tools where safety needs them |
| Parts availability | At least 95% fill rate for designated dispatch-critical spare parts within two business days in the initial support region | Publish denominator, supplier geography, backorders, AOG events and substitute/repair availability |
| Repair over replacement | Every eligible repairable assembly has a documented repair/overhaul path or an explicit safety/economic reason for replacement | Repairability review, vendor data, test/recertification requirements, price and lead-time comparison |
| Service information | 100% of service tasks tied to correct serial/configuration/revision and an offline-accessible controlled manual | Adversarial tests for stale manual, wrong part revision, missing torque/inspection values, and mismatched serials |
| Long-term support | Proposed 20-year parts/service strategy from last delivery for safety-critical and long-lead items, with approved supersession/repair paths | Supplier agreements and annual obsolescence review; label uncontracted ambitions as targets, not promises |
| Lifecycle cost | Target at least 15% lower cost per flight hour than a frozen reference over low-, normal- and high-utilization scenarios | Fully loaded, scenario-based model, independently reviewed; replace estimates with real fleet data when available |
| Owner delight | Median score at least 4.5/5 from structured owner tests for cabin usability, access, documentation clarity, service confidence and day-to-day operation | Predefined questions and cohort; record task errors and negative feedback as well as ratings |
| Documentation safety | Zero unresolved critical ambiguity findings in task instructions before release | Independent technician/owner usability review, negative-control instructions and documented correction loop |

### Cost-per-flight-hour model boundary

Include the purchase/financing/depreciation assumptions, fuel or electrical energy and charging losses, scheduled maintenance, unscheduled maintenance, engine/propeller/battery overhaul or replacement reserves, tyres/brakes/consumables, subscriptions where unavoidable, training, insurance, storage/hangar, inspection and compliance expense, maintenance-controlled downtime, ferry/transport to a maintainer, and resale/value assumptions where relevant. Also report cash outlay and owner labor separately; don't make owner labor "free" solely to claim a low operating cost.

Show results across several utilization scenarios, not one favorable annual-hours number. Include sensitivities to fuel prices, labor rates, battery/engine life, parts lead time, insurance, and aircraft age. Compare completed missions and payload, not only hours flown.

## 6. Cabin delight is an engineering requirement

Make aircraft ownership pleasant in small, repeatable ways:
- Natural ingress/egress with no avoidable gymnastics, a reachable baggage area, and restraint points that are obvious and accessible.
- Cabin dimensions and useful load stated honestly; don't sell four seats if four typical occupants plus bags and required reserves are not a credible use case.
- Good sight lines, ventilation/heating, lighting, sun management, noise/vibration control, and controls with consistent logic.
- A cockpit that supports learning and normal operation without turning every function into a screen menu. Critical information must stay readable and unambiguous.
- Inspection panels with captive or standardized fasteners where approved, clear witness marks, inspection lighting and safe body/hand access.
- No "smart" features that require a cloud connection to start, inspect, log or maintain the aircraft. The baseline owner workflow should work offline.
- Plain-language manuals with a clear legal/skill boundary, illustrations, expected task time, the correct revision and a parts order path.
- Respectful owner support: a real escalation path for a grounded aircraft, transparent labor/parts pricing, and no punitive proprietary lock-in for ordinary maintenance records.

Validate these with representative pilots, maintainers and accessibility/ergonomics experts. Do not infer owner joy from a designer's opinion.

## 7. Symthaea, Mycelix, factory and training integration

- **Symthaea:** bounded advisory analysis, requirements/maintenance-task completeness checks, diagnostics interpretation and reproducible lifecycle/mission scenario analysis. It does not control the aircraft, determine airworthiness, sign logs, or override the POH/AFM, approved maintenance data, limitations or authorized personnel.
- **Mycelix:** potentially provide controlled provenance linking serial number, as-built configuration, service tasks, part/source revisions, evidence and approvals. A record must be correctable, access-controlled and attributable; immutable storage does not make an assertion true. No completed aircraft integration is claimed.
- **Factory scheduler:** schedule build, inspection, rework and service-part supply based on real task-time observations when available. Synthetic duration fixtures are useful for software tests only—not throughput or cost claims.
- **Praxis/training:** task-based training and competence assessment for both operators and maintainers, with observed task performance and clear limits. Confirm actual repository/API ownership before connector work.
- **Aircraft-energy screen:** assess powertrain alternatives against explicit mission profiles. Until a coupled vehicle model exists, it cannot make a mass-sensitive complete-aircraft performance claim.

Design a shared evidence format only after task semantics and approved workflows are understood. Keep manufacturer engineering authority and licensed/authorized maintenance decisions outside an AI assistant's control.

## 8. Certification strategy is a product decision

Do not design first and assume one global certificate later.

### Candidate pathways to study

| Product candidate | Initial certification question | Why |
|---|---|---|
| Wren 2 trainer | Does the mission and operating goal fit the FAA Part 22 / MOSAIC light-sport category and accepted standards, or is a different type/certification path required? | Potentially shorter/focused entry path, but eligibility and operating/maintenance privileges must be proved against the exact design and market |
| Wayfarer 4 touring | Evaluate FAA Part 23 normal-category certification alongside any other authority-accepted pathway that truly fits the intended four-seat mission | Cross-country utility, cabin/payload and equipment goals may exceed a chosen light-sport pathway or simply fit it differently; authority engagement is needed |
| Ranger 4 utility | Establish intended operations, field conditions, payload, damage tolerance and mission boundaries before choosing a Part 23/Part 22/other path | Structural and landing-gear demands could change the design and compliance plan |
| Wren-E electric demonstrator | Define aircraft category, energy-storage/propulsion compliance basis, battery/environmental test programme and continuing-airworthiness path before selecting the demonstrator architecture | Electric energy storage, charging, thermal management and maintenance require evidence beyond nominal energy arithmetic |

As checked on **17 October 2026**, the FAA's MOSAIC page says its light-sport certification changes became effective on **24 July 2026**, and the FAA's accepted-standard page lists Part 22 light-sport standards including F3815-26a for airplanes. The FAA also describes light-sport repairman privileges and limitations by training/rating and aircraft category. EASA lists a CS-23 Amendment 6 / AMC & GM Issue 5 update dated **15 April 2026**. SACAA publishes distinct Part 21, Part 24, Part 43 and maintenance/production-approval material. This research does not establish the actual eligible path for any proposed aircraft; seek a certification-basis meeting with the relevant authorities and specialist counsel/DER/DAR as applicable.

Sources:
- FAA [MOSAIC](https://www.faa.gov/aircraft/MOSAIC), including effective dates and scope.
- FAA [Light-Sport Category Aircraft and accepted standards](https://www.faa.gov/aircraft/gen_av/light_sport).
- FAA [Repairman certificates and privilege limitations](https://www.faa.gov/repairmen).
- FAA [Small airplanes / Part 23 regulations, guides and consensus standards](https://www.faa.gov/aircraft/air_cert/design_approvals/small_airplanes/small_airplanes_regs).
- EASA [CS-23 Normal, Utility, Aerobatic and Commuter Aeroplanes](https://www.easa.europa.eu/en/document-library/certification-specifications/group/cs-23-normal-utility-aerobatic-and-commuter-aeroplanes), updated 15 April 2026.
- SACAA [Airworthiness entry point and approval forms](https://www.caa.co.za/industry-information/airworthiness/).

## 9. Competitor/reference comparison

Start with the Cessna Skyhawk as the best-known four-seat conventional baseline, then include at least one modern LSA trainer, one four-seat touring airplane, and a utility aircraft relevant to the intended region. Keep all published specs in dated source cards with their definitions and performance conditions.

The official Skyhawk page lists 4 occupants, 398 kg useful load, 395 kg maximum payload, 254 kg payload with full fuel, 124 KTAS maximum cruise speed, 640 nmi maximum range, and 497 m takeoff distance. These are different operating points, not one simultaneously achievable mission. The separate [reference data card](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md) records the source's performance assumptions and calculated ratios.

For each competitor, collect:
- purchase price with date, market, currency, included equipment and taxes;
- cabin dimensions, loading constraints, payload with a defined usable-energy/fuel reserve, baggage access and seating assumptions;
- exact fuel/electric energy burn and mission profile for a comparable route where evidence exists;
- scheduled task card list, labor estimates, intervals and scheduled/unplanned downtime data;
- engine/propeller/battery overhaul or replacement intervals/costs and parts support;
- support network and fleet-age constraints; and
- certification/maintenance basis and owner/repairman privileges.

If competitor maintenance data or comparable cost data are unavailable, label the field unknown. Never substitute a forum estimate or aircraft listing as though it were audited fleet data.

## 10. Immediate execution plan: prototype serviceability before aircraft geometry freezes

### Phase A — requirements and owner research

1. Interview owner-pilots, instructors, independent mechanics, repair stations and aircraft builders across at least three intended use cases.
2. Rank the top 20 downtime causes, top maintenance costs and tasks owners most want to understand or perform legally.
3. Freeze one reference mission for the first product and one baseline aircraft.
4. Create a source/data manifest for performance and lifecycle assumptions.

### Phase B — physical serviceability mock-up

Build a non-flying, non-airworthy full-scale or representative mock-up of the engine-bay/accessory region, cabin access, baggage path, inspection points, and typical service panels. Do not build flight-critical structure for this test.

Conduct task-based usability tests with representative owners and licensed/authorized maintainers. Measure tool changes, fasteners moved, unrelated parts removed, access conflicts, task time, errors, instruction lookups, body posture and task confidence. Include deliberately wrong part/manual revision negative controls. Have qualified reviewers identify unsafe or legally misclassified tasks.

### Phase C — cost and family-architecture decisions

Run the whole-life cost model across utilization and maintenance scenarios. Compare the three architectures: independent aircraft with shared service ecosystem; partial common structure/systems; and highly common airframe. Do not assume the most common architecture has the lowest lifecycle cost.

Freeze only interfaces with a demonstrated benefit. Record rejected ideas and why they lost.

### Phase D — design and compliance gates

Only after Phase A–C:
- write top-level aircraft requirements and derived verification plans;
- complete initial hazard assessment and certification-basis discussions;
- choose aircraft materials/powerplant and control architecture with qualified aerospace engineers;
- produce preliminary sizing/mass properties and performance models with sourceable coefficients and uncertainty;
- perform independent design reviews and the required approved testing programme.

No conceptual spreadsheet, digital twin or Symthaea result authorizes flight or return-to-service.

## 11. Go/no-go rules

Advance a product only when:
1. one frozen mission is well supported by owner research and an honest baseline;
2. mass, payload, useful fuel/energy, reserve and performance are analyzed together;
3. service tasks are reachable and the maintenance-authority boundary is explicit;
4. ownership economics include downtime, overhaul/replacement and parts;
5. the support/maintenance evidence can be traced to the actual serial configuration and revision;
6. the certification/continued-airworthiness path is credible and accepted by the relevant authority; and
7. qualified independent reviewers agree the current evidence meets the next gate.

If a common part increases weight, complexity, inspection time or risk without a demonstrated lifecycle benefit, do not standardize it. If electric propulsion fails the target mission or lifecycle model, keep researching it without forcing it into the line. If the family requires unaffordable certification or support to enter the market, say so before spending on a full prototype.

## Nonclaims

This document does not specify a flight-ready aircraft, finalize detailed dimensions or control laws, assert any performance/cost improvement, grant maintenance privileges, or establish safety, airworthiness or certification. Product names and all numeric targets are provisional research goals. Physical design and operation require qualified aerospace engineering, verification, approved testing and the relevant regulator's authorization.
