# Luminous Aqua — Shower Service Contract v0.1

**Status:** draft requirements contract; not a product specification or safety approval  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Scope:** one shower service from user request through post-use readiness

## 1. Service definition

A shower service is a bounded task that begins when a user requests a declared shower profile and ends only when:
1. the requested task has been completed or explicitly abandoned;
2. all water-path, thermal, electrical and room states have reached an approved post-use state;
3. required post-session cleaning and purge have completed or a maintenance lockout is active;
4. all resources, outputs, failures and unknowns have been recorded.

A timer expiry, commanded valve position, controller success message or nominal flow rate is not proof of completed service.

## 2. Initial service profiles

Each profile is separately qualified. Profiles are user-facing descriptions, not medical categories.

| Profile ID | User-visible task | Candidate mode | Qualification dimensions |
|---|---|---|---|
| QUICK_RINSE | Short rinse | Direct-flow first | Delivered flow, temperature stability, coverage, task success |
| STANDARD_SHOWER | General 5/10/15-minute shower | Direct-flow or qualified recirculation | Full resource boundary, comfort, hygiene, spray performance |
| HAIR_WASH | Wet/rinse hair | Direct-flow initially | Spray distribution, pressure, rinse performance, duration |
| SEATED_SHOWER | Seated or supported shower | Direct-flow initially | Reach, controls, seat/support compatibility, drainage, egress |
| ASSISTED_SHOWER | Shower with an assistant | Direct-flow initially | Dual-user reach, communication, manual control, safe egress |
| LOW_MOBILITY_SHOWER | Reduced standing / transfer ability | Direct-flow initially | Step-free access, reach envelope, support placement, slip/egress |
| RECIRCULATION_RESEARCH | Controlled recirculation test profile | Unoccupied bench only until separately cleared | Exposure characterization, treatment validation, cleanability, full water/energy ledger |

Do not assume one shower geometry or user profile serves every person. The design should permit interchangeable room configurations and independently validated supports, controls and seating.

## 3. Modes

### DIRECT_FLOW

A conventional once-through bathing path. Incoming water is conditioned and delivered to the user; used water goes to the approved drain. Optional drain-water heat recovery uses an indirect heat exchanger only.

### WITHIN_SESSION_RECIRCULATION

A research mode in which a bounded quantity of water is treated and recirculated during one active session. It is not household greywater storage, pool circulation or reuse by a later user.

Required rules:
- No cross-user storage or reuse.
- The active session has an explicit identity and a hard end.
- Startup, initial purge, fresh make-up, treatment flow, recirculation flow, water age/residence time, cleaning, purge and reject are separately metered or measured.
- Recirculation may run only inside a specifically validated process envelope.
- Missing, stale, invalid or conflicting critical observations prevent entry or cause safe termination.
- A filter, membrane, UV source, ozone system, ORP, turbidity, conductivity or appearance is not independently sufficient evidence of pathogen control.
- Microbiological and aerosol-exposure assessment must be designed by competent independent specialists before human exposure.
- Every wetted surface and cavity must have a validated drain, clean, inspect and service path.
- If safety or quality evidence is lost, isolate the recirculation path and transition to a qualified direct-flow mode only if the physical architecture makes that transition demonstrably safe; otherwise stop water delivery and hold/divert.

### MAINTENANCE / CLEANOUT

No user bathing. Isolate appropriate energy and water sources; drain, clean, inspect, replace components and requalify required barriers before return to service.

## 4. State machine

The following states are normative names for the service contract. Exact physical implementations and interlock circuits require qualified engineering design.

| State | Entry condition | Permitted behaviour | Exit condition |
|---|---|---|---|
| IDLE_READY | Previous service complete; drain/clean state admitted | Await request; local controls available | Valid request and preconditions -> PRECHECK |
| PRECHECK | Request received | Check selected profile, water source, equipment state, maintenance, meter/sensor validity, safe room conditions | All required preconditions met -> WARMUP; otherwise HOLD |
| WARMUP | PRECHECK admitted | Condition supply water within independently set mechanical safety limits; count startup water and energy | Stable measured output within qualified envelope -> DIRECT_ACTIVE or RECIRC_PREP |
| DIRECT_ACTIVE | Direct path admitted | Deliver once-through shower; measure inlet and outlet | User ends, fault or task ends -> POST_USE |
| RECIRC_PREP | Recirculation mode requested and preconditions met | Prepare bounded same-session circuit; record fill, treatment, flush, residence-time origin and clean-state identity | Independent process gates current and physical interlocks satisfied -> RECIRC_ACTIVE; else HOLD |
| RECIRC_ACTIVE | Recirculation admitted | Recirculate only within validated envelope; maintain fresh make-up; monitor critical conditions and all flows | User ends, qualified limit or any fault -> RECIRC_TERMINATE |
| RECIRC_TERMINATE | Session ends or recirculation invalidated | Stop recirculation; isolate circuit; route water to approved disposition; no cross-user carryover | Physical isolation confirmed -> POST_USE or FAULT_LOCKOUT |
| POST_USE | Bathing is over | Run required drain/clean/purge; measure water, energy, chemicals, time and rejects | Cleaning evidence admitted -> IDLE_READY; failed/unknown -> MAINTENANCE_LOCKOUT |
| HOLD | Preconditions unknown or unqualified | No recirculation release; provide safe user-visible explanation; qualified fallback only if independent safe path exists | Evidence reconciled -> PRECHECK; otherwise FAULT_LOCKOUT |
| FAULT_LOCKOUT | Critical fault, ambiguous valve state, failed barrier or physical interlock | Physically isolate relevant water/energy; safe drain/bypass only as designed; require human maintenance | Inspection and requalification receipt -> PRECHECK |
| MAINTENANCE_LOCKOUT | Maintenance or overdue service | Prevent bathing mode that requires affected subsystem | Maintenance, replacement and requalification admitted -> PRECHECK |

### State-machine invariants

1. RECIRC_ACTIVE can never be entered from an unknown, expired, stale or failed critical evidence state.
2. IDLE_READY cannot be declared until required post-use cleaning / drain / purge has completed and been admitted.
3. FAULT_LOCKOUT cannot be cleared solely by Symthaea or a remote/cloud command.
4. A controller's commanded state is not proof of valve position or water-path isolation.
5. A user preference or learned routine cannot change a hard safety limit.
6. The system must remain manually operable for safe shutoff and basic direct-flow use where that fallback is physically qualified.
7. When water balance or quality evidence cannot be reconciled, do not grant reuse credit or silently impute missing flow as zero.

## 5. Service request and result records

The following fields are conceptual interface requirements, not a finalized JSON schema.

### Service request
- service_id, service_contract_version, profile_id
- requested task and permitted alternatives, user-selected mode
- local start time and clock/timebase identity
- declared environmental and supply conditions, including explicit unknowns
- equipment, fixture, firmware and configuration identities
- required safety, accessibility and evidence profiles
- consent/rights and data-minimization scope for any personal or user-reported data

### Service result
- service_id, configuration and test-run identity
- outcome: VERIFIED, PARTIALLY_VERIFIED, ABSTAINED, FAILED, INTERRUPTED
- task outcome and assessor/method, distinct from controller assertion
- measured fresh water, recirculated water, discharge, purge, cleaning, reject and storage change
- measured thermal input/output, heat recovery, electric/fuel input and parasitic loads
- pressure, flow, temperature, spray coverage/force, noise and room conditions
- sensor calibration, placement, uncertainty, timebase and quality binding
- cleaning/disinfection state, water-quality assessment reference and its named use/exposure scope
- fault, retry, interruption, fallback and manual-intervention trace
- consumables, maintenance time, service parts, downtime and unresolved work
- provenance, source revision, raw data digest and assessment revision

Unknown or unmeasured fields remain explicitly unknown. Do not copy setpoints into measured-outcome fields.

## 6. Measurement and claim semantics

- **Fresh water** is water crossing the potable/approved-source boundary into the shower system.
- **Recirculated water** is water moved within the active session loop; do not count it as fresh water or greywater reuse.
- **Discharged water** includes drain, purge, backwash, reject and flush streams, each separately identified.
- **Cleaning water** includes every scheduled or required post-session clean and an allocated share of periodic deep cleaning.
- **Useful heat recovered** is measured heat actually transferred to the intended supply or load; exchanger-side theoretical availability is not useful recovery.
- **Comfort** is independently assessed against a frozen protocol and comparator, not inferred from flow or temperature alone.
- **Hygiene-qualified** is an externally assessed result for a named exposure pathway, method, sampling population and operating envelope; it is not a generic property of a device.

Primary fresh-water intensity:

W_service = W_startup + W_active_fresh + W_clean + W_purge + W_allocated_periodic + W_failed_retry

Keep recirculated volume, total delivered shower volume and discharge mass balance in separate fields.

Primary energy intensity:

E_service = E_heater + E_pump + E_control + E_treatment + E_cleaning + E_allocated_room

Declare the boundary and allocate shared room/plant energy once, not to multiple services.

## 7. Luminous system responsibilities

- Symthaea: analyze read-only observations; interpret user-selected tasks; recommend profiles within qualified options; explain abstention. No safety interlock, water-path or primary-disinfection authority.
- Symtropy: own physical simulation results through existing validated hydraulic, thermal, moisture and energy owners. A simulation result remains a model result.
- Mycelix: preserve instrument identity, sample provenance, calibration, uncertainty, quality assessment and maintenance evidence. Raw observation is not assessment, certification or authorization.
- Sol-Atlas: supply site/material/landscape context and consume its existing pool / thermal / microclimate qualification work.
- Luminous Platform: compose versioned records and local interfaces; not a substitute for specialist process or building control.

## 8. Acceptance criteria

- Each service run has a complete start-to-ready boundary.
- Resource streams are separately metered and uncertainty-bounded.
- Unknown / stale / conflicting critical evidence blocks recirculation admission.
- Any loss of required barrier or uncertain valve state leads to physical isolation/hold, not an app notification alone.
- Cleaning and periodic maintenance burdens are included in per-service accounting.
- Human exposure remains blocked until independent water-quality, microbiological, aerosol and safety review admits the exact process and operating envelope.
- Direct-flow, recirculating and pool/greywater systems remain separate evidence and physical domains.
- No modelled or synthetic result is described as a physical hygiene, comfort or product-performance result.
