# LOS-18 staged machinery-space demonstrator — V0.1

**Status:** proposed physical test programme; no hardware purchase, assembly, operation or test PASS is claimed.  
**Parent:** [LOS18-DEMONSTRATOR-001 #11093](https://github.com/Luminous-Dynamics/symthaea/issues/11093)  
**Product basis:** [MARINE-LIVEABOARD-001 #10926](https://github.com/Luminous-Dynamics/symthaea/issues/10926)

## 1. Decision this demonstrator should resolve

Can we identify a machinery-space architecture that is measurably easier to inspect, isolate, service and restore, while preserving delivered water/electrical services, configuration correctness and safety boundaries?

Build a staged bench demonstrator before committing to the full LOS hull. The demonstrator is a physical test article, not a miniature certified vessel.

## 2. Composition and authority boundaries

| Concern | Canonical owner | Demonstrator role |
|---|---|---|
| Product mission / exact case | Symthaea #10926 | Supplies requirements and operating cases |
| Mechanical serviceability | Symthaea #11049 | Supplies task/access contracts and inert mock-up plan |
| Watermaker performance | Symthaea #10994 | Supplies equipment envelope and qualified water-quality boundary |
| Electrical transients/protection | Symthaea #11028 | Supplies fast-event definitions and electrical acceptance ownership |
| Flexible service scheduling | Symthaea #11008 | Supplies slow resource-service scheduling semantics |
| Measurement/campaign evidence | Symthaea #11064 | Owns instruments, calibration, timebase, raw data and campaign binding |
| Mass/loading state | Symthaea #11074 | Consumes measured item mass/CG and configuration changes |
| Generic repair/service phases | Symtropy #4056 | Supplies generic maintainability and return-to-service semantics |
| Shipyard DFM/assembly | Symtropy #2048 | Supplies build sequence, inspection/repair access and process/resource context |
| Component provenance/as-built | Symtropy #3331 | Supplies component identity, manufacturing/inspection and as-built lineage |

This plan does not create another telemetry stack, observation ontology, maintenance planner, watermaker solver, electrical solver, geometry authority, or release authority. Mycelix records authorized custody/provenance and approval receipts where appropriate; a signed receipt is not proof that a physical measurement is true or an approval is competent.

## 3. Staged physical architecture

### Stage 0 — freeze the test basis

Before procurement or fabrication, define:
- test article boundaries, intended decisions and claim limitations;
- provisional machinery-space envelope and interface-control register;
- exact equipment model/options and current manufacturer documentation;
- electrical, hydraulic, thermal, mechanical, environmental and stored-energy hazard boundaries;
- isolation, containment, emergency stop and abort responsibilities;
- measurement quantities, instrument specifications, calibration needs, uncertainty and timebase requirements;
- acceptance criteria and analysis plan before observing comparative outcomes.

No unmeasured clearance, voltage, current, pressure, flow, seawater chemistry, mass, lifting rating, sampling rate or safe working procedure may be invented. Unresolved values remain Unknown and block dependent work.

### Stage 1 — inert mechanical access mock-up

Use only an inert, non-powered, non-pressurized dummy module. Compare:
1. fixed installation with declared access;
2. hinged/removable service access while the module remains mounted;
3. slide-out module/carrier.

Use the same module mass/envelope, task definition, tools, completion criterion and spare identity across layouts. Randomize/counterbalance layout order and retain familiarization trials separately.

Measure phase time, tool changes, fastener operations, minimum clearance, contacts/collisions, handling/assistance, part loss, identification/revision errors and configuration-record completion. Include wrong-spare and obstructed-path negative controls. Do not infer full-population ergonomics from a small pilot.

### Stage 2 — separately guarded watermaker skid

Use a selected real watermaker only in a professionally reviewed test arrangement, operated within its exact manufacturer-rated envelope and approved instructions. This is not part of the inert access mock-up.

Instrument the declared feed and product boundary. Where applicable, record:
- feed salinity/conductivity, temperature, pressure and flow;
- prefilter differential pressure and consumable identity/state;
- product flow, product quality/conductivity and off-spec diversion;
- energy input, instantaneous power and accumulated energy;
- brine/reject flow, recovery, flush/preservation cycles, alarm/trip events and run time.

Report product volume, litres/hour and kWh/m³ only with the exact test conditions, calibration status, uncertainty and equipment revision. Acceptance limits come from the selected equipment documentation and applicable requirements, not this research plan.

### Stage 3 — instrumented electrical rig

Begin with a professionally designed, isolated and current-limited low-energy configuration. Any expansion in voltage, current, stored energy or fault energy requires a separate design/hazard review and test-readiness approval.

Record declared bus quantities, load steps, power/energy, temperature, protection trips, transfer events and recovery. Derive bandwidth and sampling from the fastest phenomenon being claimed; retain anti-aliasing, trigger, clock and synchronization evidence. A slow average is never evidence of a fast transient.

The mock-up experiment excludes live batteries, high-voltage maintenance, inverters, propulsion, pressure systems and any real hazardous service procedure.

### Stage 4 — controlled subsystem integration

Integrate the watermaker and electrical bench only after their independent stage gates pass. Exercise bounded load profiles and measure:
- delivered water service and quality under changing available power;
- scheduling of production/flush/preservation against reserves;
- electrical load-step response and protection behavior;
- equipment derating/faults and explicit unmet-service outcomes;
- safe test isolation, reset and recovery;
- exact configuration and evidence re-binding after a changed module.

V0 remains advisory: no Symthaea, Symtropy or digital-twin policy directly actuates critical physical equipment.

### Stage 5 — approved maintenance and restoration trials

Only equipment-specific tasks covered by current approved procedures and qualified personnel may be performed. Separate:
- diagnosis and planning;
- authorization and isolation;
- hands-on access/removal/repair/replacement;
- spare and logistics wait;
- installation/reconnection;
- post-maintenance test/recommission;
- record/configuration update;
- authorized return-to-service.

A completed mechanical swap is not restored service. Missing tests, stale configuration or absent release authority prevent a return-to-service disposition.

### Stage 6 — independent evidence review

Bind every run to the exact physical article/as-built digest, instrument identities, calibration/currentness, units, uncertainty, timebase, procedure revision, environmental conditions, operator/authority roles, raw artifact digests and deviations. Retain failed, aborted, blocked and indeterminate runs.

Only after independent review should the programme decide whether to scale to a larger machinery-space mock-up or a professionally reviewed on-vessel trial.

## 4. Comparison matrix

| Architecture | Hypothesis to test | Penalties that must be counted |
|---|---|---|
| Fixed installation | Simplicity and fewer interfaces may outperform modular designs when access is genuinely good | access envelope, deep disassembly, service downtime |
| Hinged/removable access | Better access without moving the equipment may shorten the service sequence | hinge/latch mass, sealing, inspection, interference |
| Slide-out carrier | Carrier extraction may reduce awkward handling and unrelated disassembly | rails, stops, locking, alignment, corrosion, added envelope/mass |
| Line-replaceable assembly | Standardized boundaries may simplify replacement and stock | connectors, seals, fasteners, leakage/contact faults, test burden |
| Redundant/degraded mode | Service may be deferred without losing critical delivered services | duplicate mass/volume, cost, common-mode failures, inspection and energy |

No architecture wins by label. Compare matched tasks, conditions and service requirements. Keep a Pareto vector rather than a single weighted score.

## 5. Shared metrics

**Serviceability:** phase duration, hands-on technician-hours, total downtime, p50/p90/p95 when sample size supports it, crew count, tools, handling aids, access/extraction margin, disassembly depth, unrelated critical systems disturbed, assistance, rework, part/identity errors.

**Watermaker:** qualified product volume/flow and quality, kWh/m³, recovery/reject volume, operating envelope, alarms/trips, consumable change burden, off-spec diversion.

**Electrical:** voltage/current/power traces, energy, thermal state, load-step response, protection behavior, event-capture completeness, uncertainty, clock alignment, fault recovery.

**Integrated service:** requested versus delivered water/power/thermal services, reserve violations, deferred/failed service, maintenance-window collision, degraded-mode duration and restoration time.

**Lifecycle burden:** module/interface mass and volume, connector/seal/fastener count, inspection burden, common parts, onboard stock, lead time, substitute approval, acquisition and service cost where source-complete.

**Evidence quality:** valid test channels, current calibration, raw-data completeness, exact subject/revision, procedure applicability, uncertainty, missingness and independent acceptance.

Do not calculate MTTR from synthetic fixtures or only from successful tasks. Report blocked/failed/indeterminate cases and the worst context, not just the average.

## 6. Evidence maturity

| Level | Evidence | Permitted claim |
|---|---|---|
| E0 | authored requirement / synthetic contract | contract semantics only |
| E1 | synthetic simulation or fixture | modeled behavior on the stated fixture |
| E2 | measured bench run on an identified physical article | measured result within that article and test envelope |
| E3 | repeated matched campaigns with uncertainty and independent review | repeatability within tested conditions |
| E4 | professionally reviewed on-vessel validation | exact-vessel result within declared operating envelope |
| E5 | longitudinal operational evidence | fleet/task reliability and availability only with sufficient exposure |

This project ladder complements, but does not replace, the shipyard-production literature's simulation → historical replay → shop-floor integration → sustained live KPI maturity hierarchy.

## 7. Hard negative controls

- wrong part, serial, revision or effectivity;
- stale/missing as-built geometry or access path;
- obstructed extraction or unplanned removal of an unrelated critical system;
- unknown mass/handling or absent lifting basis;
- missing isolation, approved procedure or task permission;
- missing product-quality observation or stale calibration;
- feed condition outside the declared equipment envelope;
- inadequate bandwidth, aliasing or unsynchronized fast-event data;
- protection trip not captured or required protective test not completed;
- changed article/configuration without a new exact subject digest;
- required post-maintenance test absent or failed;
- no authorized return-to-service decision.

Unknown or unsupported inputs are never coerced into feasible, safe, available or complete.

## 8. Initial gates — goals, not results

- 100% of test runs bind to one exact article/configuration and controlled procedure revision.
- 100% of reported channels expose instrument, calibration, units, uncertainty and timebase.
- 100% of planned tests have predeclared acceptance and abort rules.
- zero false-complete synthetic return-to-service dispositions on hard-negative controls.
- zero transient claims unsupported by instrument bandwidth and time synchronization.
- all failed, aborted, blocked and indeterminate runs remain visible in the report.
- any comparative improvement is based on matched conditions, a frozen baseline and a preregistered analysis.

## 9. Regulatory and standards applicability

IEC 60300-3-10:2025 is the current general dependability application guide for maintainability and maintenance; use it to structure lifecycle trade-offs, not as a source of invented pass thresholds: https://webstore.iec.ch/en/publication/65334

ISO 12215-7:2020 was confirmed current in 2026 and applies within its stated multihull/small-craft scope; service openings must be coordinated with qualified structural design: https://www.iso.org/standard/73457.html

ISO 14224:2016 provides a useful reliability/maintenance data structure, but its scope is petroleum, petrochemical and natural-gas operations; it is a semantic reference here, not the default governing vessel standard: https://www.iso.org/standard/64076.html

Actual legal, flag, class, machinery, electrical, fire, stability and environmental requirements must be determined by vessel length, use, tonnage, flag, operating area, passenger status and selected class. Qualified naval architects, marine electrical/machinery engineers, surveyors and competent authorities retain their approval roles.

## 10. Claim ceiling

This is a proposed test plan. No hardware has been built or tested by this document. Passing a schema, mock-up, bench test, integrated campaign or digital model does not prove vessel seaworthiness, offshore range, safe at-sea repair, fleet MTTR, class/flag approval or autonomous operation. Those are separate claims requiring exact physical configurations, applicable professional engineering, approved procedures, independent inspection and competent authority decisions.
