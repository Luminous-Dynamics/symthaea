# Marine mission service-continuity and accepted-restoration benchmark — V0.1

**Status:** benchmark contract and synthetic event fixtures; no physical/fleet reliability or availability result is claimed.  
**Parent:** [MARINE-SERVICE-CONTINUITY-001 #11115](https://github.com/Luminous-Dynamics/symthaea/issues/11115)  
**Mission basis:** [MARINE-LIVEABOARD-001 #10926](https://github.com/Luminous-Dynamics/symthaea/issues/10926)

## 1. Research question

Can an exact vessel configuration sustain every required service above its owner-defined floor, transition into only explicitly permitted degraded modes, and restore failed services through approved repair, testing, configuration update and authorized release under the same mission, resource, crew, spare and event constraints?

The key change is to evaluate **delivered service over time**, rather than static inventory, component health labels, nameplate capacity or a single autonomy-days value.

## 2. Existing owners and non-duplication

- Mission and configuration: #10926.
- Static resource endurance vector: #10956.
- Exact mission-evidence join: #10988.
- Serviceability task/return-to-service profile: #11049.
- Slow service/load scheduling: #11008.
- Fast power/protection/transient evidence: #11028.
- Qualified watermaker operating boundary: #10994.
- Instrument/calibration/clock/raw campaign evidence: #11064.
- Mass/loading configuration: #11074.
- Generic service task/phase/return-to-service semantics: Symtropy #4056.
- Component identity/as-built/inspection lineage: Symtropy #3331.
- Physical demonstrator campaign: #11093.

This benchmark consumes those authorities. It does not create another battery, water, thermal, habitat, failure-rate, maintenance or scheduling model.

## 3. Core semantics

A service is **available** only when its delivered quantity and quality satisfy the exact service contract and all hard safety/permission conditions are met.

States:
- **Nominal:** delivered service meets nominal declared requirements.
- **Permitted degraded:** delivered service is below nominal but remains inside a specifically approved degraded envelope and time limit.
- **Unavailable:** required service is not delivered at the minimum floor.
- **Unknown:** required evidence or state is missing, stale, contradictory or unsupported.
- **Unsafe / not authorized:** a safety, procedure, permission or competent-authority gate is not satisfied. It is never counted as available.

The allowed floor, degraded envelope, duration limit and criticality must come from the responsible engineering/operator/authority owner. This benchmark does not invent requirements for water, electrical power, propulsion, thermal comfort, refrigeration, sanitation, fire, flood, steering, navigation or emergency systems.

A repaired component is not restored service until:
1. the exact task is authorized and completed;
2. the required post-maintenance tests pass;
3. the exact configuration/as-built record is updated;
4. the required independent release/return-to-service authority is present;
5. the delivered service again satisfies its contract.

## 4. Metrics

Report a vector per required service and exact operating case:
- demanded and delivered service quantity over time;
- nominal, permitted-degraded, unavailable, unknown and unsafe intervals;
- below-floor interval count, duration and integrated unmet service;
- time to detect, diagnose, isolate, begin repair, complete repair, pass tests, update configuration and receive release;
- Time to Accepted Service Restoration (TASR), from the declared fault-observation trigger to restored service meeting its contract and release gates;
- hands-on repair time, spare/logistics wait, crew/tool wait, testing time and authorization wait separately;
- spare stockout and incompatible-part contribution;
- crew/skill availability, workspace/tool contention and concurrent-task conflicts;
- redundancy/interface burden and common-cause exposure;
- mission continue/degrade/divert/return/abort result, without turning this evaluator into an autonomous navigation or command authority.

Do not publish one aggregate availability score that allows surplus power to compensate for missing potable water, failed refrigeration or an unavailable safety system. If a summary index is later useful for a clearly scoped economic study, show the full per-service vector and hard gates beside it.

## 5. Baseline policies and fair comparison

Use the same configuration, initial state, mission, event tape, spare stock, crew/skills, environment, observations and compute/time budget.

Compare:
- no recovery / nominal-only baseline;
- source-supported fixed preventive maintenance;
- critical-service-first repair priority;
- spare-aware repair planning;
- qualified redundancy/degraded-mode design;
- existing resource-aware rolling-horizon schedule where its owner model supports the necessary constraints;
- Symthaea advisory policy only with the same information timeline and budget;
- perfect-foresight solution as an unattainable labelled upper bound only.

The evaluator reconstructs the time history from event receipts. It does not trust a scheduler-supplied availability or return-to-service boolean. Future events remain hidden until their declared observation time in online comparisons.

## 6. Synthetic corpus

The accompanying corpus has six authored cases:
1. potable-water buffer preserves the declared service floor until an authorized repair and requalification complete;
2. potable-water buffer exhausts before a compatible spare is available, causing a service-floor breach;
3. two nominally redundant pumps share a controller/power dependency and fail together;
4. a mechanical repair completes but a required post-maintenance test fails, so no service restoration is accepted;
5. an onboard spare has the wrong revision/effectivity and is rejected;
6. resource inventories remain sufficient but a non-compensable safety gate fails, so mission service is not declared safe/available.

All quantities, identities, event times and service floors in the corpus are synthetic contract fixtures. They are not LOS-18 requirements, real component characteristics, operating procedures or physical observations.

## 7. Adversarial cases to add after the first oracle

- compatible spare arrives late;
- authorized technician unavailable or already committed to another task;
- tool/calibration certificate stale;
- repair requires a maintenance window unavailable in current sea/weather conditions;
- backup unit has a shared cooling/intake/control dependency;
- component change invalidates the mission evidence join until requalification;
- watermaker output is present but quality measurement is absent or off-spec;
- power service is present but protection/fault-monitoring evidence is stale;
- delayed/duplicated/out-of-order observations;
- event-tape tampering or future-event leakage;
- service scheduler postpones a critical inspection;
- permitted degraded mode expires before a spare or repair arrives.

## 8. Standards and external grounding

- IEC 60300-3-10:2025 is a current lifecycle guide to maintainability and maintenance and their interfaces with reliability, availability and supportability. Use it for requirements/data structure, not invented acceptance thresholds: https://webstore.iec.ch/en/publication/65334
- ISO 14224:2016 is a useful equipment/failure/maintenance vocabulary reference, but its stated sector scope is petroleum, petrochemical and natural-gas operations. Do not present it as the governing small-craft standard: https://www.iso.org/standard/64076.html
- ISO 16315:2026 is the current second edition for electrical propulsion and the electrical part of hybrid propulsion on applicable small craft up to 24 m hull length. Its published revision includes clarification of overcurrent protection and revised fault-monitoring requirements for isolated DC systems: https://committee.iso.org/standard/84203.html
- The 2026 shipyard production-planning review distinguishes simulation, historical replay, partial shop-floor integration and sustained live KPI evidence. Apply that evidence hierarchy to our own claims: https://doi.org/10.3390/jmse14040396

Actual standard applicability, electrical architecture, service floors and maintenance permissions require the complete applicable documents and qualified professional review.

## 9. Proposed acceptance gates — targets, not results

- 100% of required services are represented in the frozen case roster.
- 100% of service-state intervals bind to exact configuration, owner, unit, floor and evidence class.
- Zero unsafe/not-authorized intervals are counted as available.
- Zero restorations without passing post-maintenance tests, current configuration and authorized release.
- Zero future-event leakage in non-anticipative tests.
- 100% timeline coverage: every interval is nominal, permitted-degraded, unavailable, unknown, unsafe/not-authorized or explicitly not applicable.
- 100% agreement with a separately authored oracle on supported tiny fixtures.
- All failures, blocked tasks, incomplete cases and unknown intervals remain visible.

Numeric MTTR, availability or mission-success improvements are not set until exact physical or fleet baselines and sufficient exposure data are admitted.

## 10. Claim ceiling

Synthetic passes validate only the event/evaluator semantics on the frozen corpus. They do not establish real component failure rates, fleet availability, safe degraded operation, emergency survivability, vessel seaworthiness, permission to repair underway, class/flag compliance or authority to sail. Those require exact vessel engineering, approved procedures, qualified operators, measured physical/fleet exposure and competent independent authorities.
