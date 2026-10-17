# Roadster 2026: Air–Road Platform Benchmark and Evidence Gates

**Date:** 2026-10-16  
**Status:** Research proposal; no vehicle design, feasibility, certification, cost advantage, or measured performance is claimed.  
**Owner:** Symthaea / Luminous Dynamics  
**Related:** [AIRROAD-002](https://github.com/Luminous-Dynamics/symthaea/issues/7692), [aircraft energy screening](https://github.com/Luminous-Dynamics/symthaea/pull/7647), [personal aircraft maintainability study](https://github.com/Luminous-Dynamics/symthaea/pull/7649), [modular flying-vehicle study](https://github.com/Luminous-Dynamics/symthaea/pull/7676), [factory scheduler](https://github.com/Luminous-Dynamics/symthaea/pull/7678).

## 1. Executive conclusion

The next-generation Tesla Roadster unveiled on 2026-10-15 is a high-performance electric sports car with a demonstrated thruster-assisted aerial stunt. Reporting describes a cold-gas thruster system and a fan-based suction system, with a proposed 2028 production start; Tesla did not give a final price at the event. Those details should be treated as reported reveal claims, not as proof of a repeatable passenger-carrying aircraft mission.

Our response should not be to copy a stunt, or to claim that a jump or flip makes a road car a practical aircraft. We should build a benchmarkable mobility *system* that is demonstrably better for specific use cases:

1. **Road vehicle:** performance, everyday usability, repairability, low ownership cost, and long service life.
2. **Flight vehicle:** genuine controlled flight for a defined payload and mission, safe operating limits, supportable maintenance, and an independently evidenced certification path.
3. **Ground carrier and service ecosystem:** affordable transport and charging, safe handling, parts availability, technician training, repeatable production, and fast return to service.
4. **Evidence layer:** immutable configuration identity and traceable evidence for design, manufacturing, maintenance, software, battery health, and operator qualification.

These should be designed as linked but separately qualified systems. Do not make the road vehicle, flight module, and their regulators inherit one another's unresolved risks unnecessarily.

## 2. What the public evidence says—and does not say

### Tesla Roadster reveal, 15 October 2026

Reuters reported that the revealed vehicle drove upside down on a ramp using a suction system, turned in the air, landed using built-in thrusters, and later jumped over vehicles. The report states Tesla advertised over 230 mph top speed, more than 300 miles of range, and sub-one-second 0–60 mph acceleration; it also reports the 2028 production target and that final pricing was not announced. The figures are manufacturer statements reported during a product reveal, not independently audited production-vehicle results.

TechCrunch's event reporting distinguishes three configurations: Performance (road-ready), Thrust (two seats and rear cold-gas thrusters), and Rocket (single-seat, with ten thrusters). It reports a 600-liter compressed-gas tank for the aerial manoeuvre system. Other coverage is inconsistent about the exact gas description and details of the operating cycle, so those specifics should remain provisional until Tesla releases a controlled technical specification. The demonstration itself was reported as autonomous for safety.

The Associated Press reports cold-gas thrusters and a multi-mode suction system, notes regulatory hurdles, and reports production expectation as 2028. These details sharpen the comparison: the most dramatic Rocket configuration sacrifices passenger capacity and appears to target short, controlled stunts rather than a published, repeatable point-to-point flight mission. This is an engineering distinction, not a claim that the demonstration was fake or that the hardware could never be developed further. It does **not** by itself establish:
- sustained, repeatable, controlled flight across a useful route;
- payload and reserve capability under a defined mission profile;
- takeoff/landing-site requirements and operating-weather envelope;
- a certified configuration or approval for public-road and public-air use;
- lifecycle cost, scheduled maintenance burden, service availability, or repair time;
- production repeatability, long-duration reliability, or field support.

Sources:
- Reuters, 2026-10-15: https://www.reuters.com/business/autos-transportation/tesla-set-unveil-next-generation-roadster-sports-ev-after-years-delay-2026-10-15/
- Associated Press, 2026-10-16: https://apnews.com/article/tesla-musk-roadster-rocket-7d5f19665435b7408025bfb7a2f2f0b7
- TechCrunch, 2026-10-15: https://techcrunch.com/2026/10/15/tesla-reveals-spacex-powered-roadster-that-flies-and-falls-with-style/

### Modular air–road competition

XPENG AEROHT’s Land Aircraft Carrier is a direct reminder that a road carrier with a separate flight module is not a unique concept. Guangzhou’s government reported that AEROHT rolled aircraft from a batch trial-production line and conducted multi-aircraft test flights at its Huangpu factory on 2026-03-05.

Source: https://eng.gdd.gov.cn/20260309/e76aa18d-d141-45d5-ae10-637a187fc1fc.html

**Implication:** modularity is an architectural choice, not the competitive moat. Our differentiator would need measured advantages in ease of repair, whole-system lifecycle economics, manufacturing yield, evidence quality, and real mission utility.

### Maintenance and regulation are design inputs

NASA’s maintainability guidance treats access, support equipment, tools, skill level, preventive/corrective maintenance, modular replacement, and technician time as system-design considerations—not work to postpone until after the hardware is frozen.

Source: https://www.nasa.gov/reference/9-0-hardware-and-equipment-vol-2/

For the United States, the FAA has a powered-lift operating/pilot framework and guidance for certification of powered-lift aircraft. This is evidence that aircraft-category, production, airworthiness, pilot, and operational issues are distinct obligations—not a claim that the proposed concept would qualify under a particular category. Any South African path must be determined with SACAA and relevant road-vehicle authorities; US rules do not authorize operation in South Africa.

Sources:
- FAA Advanced Air Mobility: https://www.faa.gov/air-taxis
- FAA AC 21.17-4, Type Certification—Powered-Lift: https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1044836

## 3. Paradigm shifts

### Shift A — from a stunt to a useful mission

A stunt answers “can the vehicle do this once under this demonstration setup?” A useful-mission benchmark answers “can a defined configuration repeatedly transport a defined payload across a defined route, under stated conditions, while preserving reserves and passing the relevant inspections?”

Every flight claim should carry a mission card:
- payload and occupant assumptions;
- start/end location and route distance;
- weather and operating limits;
- takeoff/landing method and site requirements;
- energy used, reserve remaining, and uncertainty;
- configuration and software identity;
- test count, aborts, failures, maintenance events, and exclusions;
- independent witness/calibration records where appropriate.

A short jump is recorded as a jump. It is not relabelled as useful range, cruise capability, or aircraft certification.

### Shift B — from one vehicle to a portfolio of qualified modules

Evaluate at least these alternatives before committing to a layout:

| Architecture | Primary benefit | Main trade-off | Required comparator |
|---|---|---|---|
| High-performance road EV with no flight claim | Lowest aviation complexity; direct Roadster comparison | Does not offer air travel | Production road EV in the same cost/performance class |
| Integrated roadable aircraft | One physical vehicle | Coupled mass, packaging, and two-regime design constraints | Existing roadable-aircraft concepts and conventional aircraft plus car |
| Road carrier plus separate flight module | Independent servicing and potential mission flexibility | Two vehicles, transfer interfaces, storage/charging, added capital | XPENG-type modular system; separately owned car plus aircraft |
| Conventional road vehicle plus conventional aircraft | Strong baselines, existing operational patterns | Not a single integrated product | Real owner/operator costs and mission completion |
| Ground logistics vehicle plus remotely piloted uncrewed demonstrator | Useful early validation without passenger exposure | Does not prove passenger operation | Research-phase test target only |

Do not choose the modular option because it sounds elegant. Freeze the selection only after a weighted trade study using measured mass, energy, cost, supportability, performance, and hazard analyses. If the modular platform cannot beat a separate car-plus-aircraft baseline on the selected use case, say so.

### Shift C — from “cheap to buy” to lifecycle cost and uptime

Purchase price alone hides downtime, scheduled inspections, battery replacement, spares, specialist labor, insurance, hangar/storage, training, software support, and failed dispatches. Compare complete ownership or operator cost per completed passenger-kilometre and report availability separately. A low-cost vehicle that is grounded for weeks is not an affordable mobility system.

### Shift D — from repair after launch to repair-driven design

Create a service task list and maintainability mock-ups before packaging is frozen. For each replaceable unit record access steps, tool list, skill level, lifting/support needs, connector inspection, isolation procedure, calibration, verification, and post-maintenance acceptance. Reduce fastener and connector proliferation where safe, but do not optimize modularity at the expense of structural integrity, fire containment, electrical isolation, or safety-critical interfaces.

### Shift E — from marketing claims to an evidence-carrying product

Each design revision, manufactured serial number, safety-critical part, firmware/software build, inspection, repair, battery record, and training assessment should map to explicit requirements and evidence. Symthaea may provide analysis and decision support; it must not silently become a flight-authority substitute. Mycelix could later provide provenance and consent-controlled evidence sharing, but no such aircraft integration is claimed here.

## 4. Proposed benchmark scorecard

All figures below are **proposed project targets**, not measured results, industry-standard thresholds, or commitments. They are hypotheses to refine after vehicle mass, propulsion architecture, mission, applicable rules, and baseline datasets are known. No target should override the applicable safety requirements.

| Area | Proposed target or gate | Measurement protocol | Evidence required |
|---|---|---|---|
| Useful mission | Carry two occupants plus baggage, with a defined total payload target of at least 200 kg, on a 100 km planned point-to-point flight mission while landing with the applicable required reserve | Define route, wind, temperature, payload, takeoff/landing, diversion and reserve assumptions before testing; use measured energy and calibrated payload | Reviewed mission definition, mass properties, energy logs, test report, and applicable regulatory acceptance |
| Flight claim integrity | Zero marketing use of “range” for a jump, hover demo, or simulated mission | Keep stunt, hover, transition, and point-to-point results in separate metric families | Claim-to-evidence ledger; negative-control review |
| Serviceability | At least 80% of eligible non-structural service tasks completed within 60 minutes using documented common tools | Timed task trials by representative trained maintainers; report median, 90th percentile, rework and safety steps separately | Video/logged trials, task definitions, tool list, reviewer sign-off |
| Replaceable module | A defined propulsion or avionics line-replaceable unit can be swapped and returned to test-ready state in 60 minutes or less, *including required post-installation verification* | Repeated timed swaps with production-intent access, connectors, tools and checks; no skipped acceptance steps | Raw task times, configuration IDs, checklist and independent verification |
| Reliability / dispatch | Establish a baseline first; subsequent target is at least 99% completion of eligible, pre-defined dispatch attempts on an agreed test profile | Pre-register inclusion rules; publish completed, aborted, maintenance-cancelled and excluded attempts; stratify by weather and configuration | At least 1,000 representative attempts before claiming the 99% target; uncertainty intervals and full failure/abort ledger |
| Whole-life economics | At least 25% lower lifecycle cost per completed passenger-kilometre than a frozen baseline for the same mission and utilization scenario | Include acquisition/depreciation, energy, charging losses, inspections, labor, spares, battery/major-component replacement, storage, training, downtime and insurance when supported by data | Reproducible model, source data, sensitivity analysis, independent review |
| Maintenance burden | At least 30% lower scheduled-maintenance labor per flight-hour than a frozen, mission-appropriate reference | Compare like-for-like mission, utilization, age and task boundaries; report unscheduled labor separately | Time-and-task records for both systems; no extrapolation from a single prototype |
| Energy / payload | Beat the frozen reference on energy per completed passenger-kilometre without reducing payload, route, reserve, or weather envelope | Meter from a defined energy boundary; disclose charging conversion losses, reserve, payload and mission assumptions | Calibrated energy meters, mass sheet, raw logs and independent rerun |
| Manufacturing | Reduce measured lead time and rework by at least 10% from an observed pilot-line baseline before claiming factory superiority | Track serial units, station queues, rework hours, scrap, first-pass yield and supplier escapes; compare the same product revision and takt assumptions | Timestamped production ledger, nonconformance records, unit traceability |
| Traceability | 100% of flight-critical installed items and software/configuration identifiers traceable to their approved source/revision and maintenance history | Attempt deliberate missing, duplicate, stale, and mismatched identity cases | Machine-checkable trace matrix; fail-closed negative-control suite |
| Safety case | No unresolved unacceptable catastrophic hazards at each project gate; all safety-critical claims reviewed by independent qualified specialists | Hazard analysis, fault-tree/common-cause work as appropriate, verification records and controlled configuration | Signed gate record and closure evidence; not a substitute for authority approval |
| Repair economics | Publish repair cost, parts waiting time, and grounded hours per event; target a 50% reduction versus baseline only after a verified baseline exists | Track from fault report to restored dispatch, including parts logistics and required retest | Service logs, parts invoices, time stamps, audit sample |

**Interpretation rules**
- The 200 kg payload / 100 km mission is a demanding candidate requirement, not a prediction that a compact electric aircraft can meet it. Run mass, drag, propulsion, thermal, reserve and certification trades before treating it as a feasible baseline.
- The 99% dispatch figure is a later operational target, not an early flight-test pass criterion. Never continue an unsafe test to improve a metric.
- The 25% lifecycle-cost and 30% labor improvements are only meaningful against frozen reference definitions. Pick an honest baseline before tuning the comparison.
- Separate statistical confidence from a point estimate. Report denominators and failed attempts, not just successful demonstrations.

## 5. Evidence levels and fail-closed gates

These are internal evidence levels for project governance, not regulatory classifications.

- **E0 — hypothesis:** written claims, architecture options, and uncertainty register. No feasibility claim.
- **E1 — analytical screening:** reproducible mass/energy/cost calculations with versioned assumptions and sensitivity sweeps. Passes may reject concepts; they do not demonstrate flight capability.
- **E2 — component bench:** calibrated tests for components, thermal/electrical boundaries, replacement access and interfaces, with raw data.
- **E3 — subscale/uncrewed:** controlled test range, predeclared abort criteria, telemetry, independent observation, and complete test accounting. This does not demonstrate passenger safety.
- **E4 — full-scale uncrewed:** configuration-controlled prototype tests under an appropriately approved test plan, with validated instrumentation and closed test hazards.
- **E5 — crewed experimental testing:** only under the relevant authority, permit, qualified personnel, test plans, and operating limitations. Not public passenger service.
- **E6 — approved production/operation:** applicable design/type, production, operator, pilot, maintenance and operating authorizations obtained from the competent authorities, with continuing airworthiness and service systems.

A gate fails closed when evidence is missing, stale, from a different configuration, or not independently reproducible. A queued, skipped, cancelled, or absent software workflow is not a pass. A passed software test does not prove physical safety.

## 6. Execution plan that does not wait on CI

### Work now in software/research

1. Maintain a source-linked claim ledger distinguishing Tesla statements, press-reported observations, unknowns, and independently measured facts.
2. PR #7647 now has a bounded six-axis battery sensitivity sweep across pack specific energy, installed-system/pack mass ratio, temperature capacity derating, end-of-life state of health, reserve, and power derating. It caps work at 50,000 combinations, canonicalizes axis ordering, preserves invalid configurations, and computes separate Pareto frontiers per reserve policy so different reserve assumptions are not ranked as if equivalent. The branch also has candidate-specific mission-profile comparison and an evidence-carrying mission manifest validator for configuration/system-envelope digest linkage, model/data source references, mass closure, takeoff-mass limits, CG frame, environment, and phase-demand integrity. These are input-contract and arithmetic layers—not aerodynamic validation. The fixed-profile sweep does not propagate changed system mass back into lift/drag/propulsive demand, and candidate-specific profiles still come from a caller. A vehicle model must couple and validate those calculations before claiming a physically fair mass-sensitive architecture comparison. Payload effects, aerodynamics, thermal behavior, transition, weather, charging loss, and uncertainty remain future model work. See [candidate mission model contract](https://github.com/Luminous-Dynamics/symthaea/blob/research/aircraft-energy-architecture-2026-10-16/docs/aircraft/CANDIDATE_SPECIFIC_MISSION_MODEL_CONTRACT_2026-10-17.md) and issue [#7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729).
3. Turn the factory scheduler into a deterministic, versioned factory experiment: task durations and skills remain synthetic until time-and-motion measurements exist. Add baseline/optimized scenarios, throughput/WIP/rework accounting, and an independent output checker.
4. Define a maintenance task ontology and task-card schema: operation, part/revision, required skill/tool, isolation steps, safety-criticality, measured labor, verification, and evidence digest. Reuse established project structures; do not add a parallel robotics/aircraft abstraction without an ownership review.
5. Draft the Mycelix evidence contract separately: serial/lot provenance, configuration, acceptance, service events, software signatures, and revocation/correction semantics. Do not place unverified safety conclusions on a ledger or treat immutable data as inherently true.
6. Build one scenario comparison report using frozen assumptions and a complete manifest. Add adverse cases and negative controls before any aggregate score.

### Work that requires a physical test program

- Weigh representative assemblies and determine actual center-of-gravity/inertia data.
- Test component performance, derating, thermal rejection, noise, vibration and fatigue in appropriate facilities.
- Conduct maintainability trials with representative technicians and production-intent access.
- Validate prototype energy and mission profiles with calibrated instruments.
- Qualify a controlled factory process using observed takt, work-in-process, rework and supplier-quality data.
- Obtain specialist engineering review and engage the relevant aviation and road authorities before any flight, road, or passenger-use claims.

## 7. How Symthaea, Mycelix, factory operations and training connect

- **Symthaea:** analysis, scenario exploration, uncertainty accounting, bounded advisory tools, and reproducible evaluation. It must never override the aircraft's approved control/safety system.
- **Aircraft-energy screening:** early rejection of configurations that miss mass, power, reserve or mission constraints before expensive hardware work.
- **Factory scheduler:** deterministic capacity/skill/precedence planning. The existing scheduler is a heuristic; it is not proof of globally optimal production.
- **Praxis/training system:** map tasks to competencies and assess observed performance. Confirm the actual repository and API before implementing a connector; do not invent an integration.
- **Mycelix:** evidence provenance, revision relationships, attestations, corrections, and carefully governed access. Provenance strengthens auditability; it does not make a bad measurement true.
- **Sol-Atlas/Symtropy:** potentially useful for mission-site context and factory/process simulation after stable interface ownership is agreed. Keep these as later interfaces, not hidden dependencies for the first benchmark.

The first deliverable is a repeatable evidence package and a set of rejected/qualified design candidates—not a rendered concept car.

## 8. Decision rule

Advance a candidate only if it:
1. satisfies the current gate without bypassing a critical control;
2. beats the frozen baseline on the use case actually selected;
3. shows the uncertainty and costs of its worst credible failure modes;
4. can be inspected, repaired, and returned to service with documented tools and competency;
5. preserves complete configuration and test provenance; and
6. has a credible route to the required approvals and continuing maintenance.

If no architecture meets these conditions, report the failure, revise the requirement, or choose the conventional baseline. The ambition is an extraordinary product; the proof standard must be more demanding than the marketing.
