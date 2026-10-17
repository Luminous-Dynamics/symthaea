# Aircraft Maintenance Task Card Contract

**Date:** 2026-10-17  
**Status:** proposed records contract for research and usability testing; not approved maintenance data or a legal authorization.  
**Parent specification:** [Personally Maintainable Aircraft Family](PERSONALLY_MAINTAINABLE_AIRCRAFT_FAMILY_2026-10-17.md)  
**Tracking:** [AIRFAMILY-001 #7745](https://github.com/Luminous-Dynamics/symthaea/issues/7745)

## 1. Why a task card is an aircraft-family interface

A personally maintainable aircraft needs a consistent way to tell an owner and an authorized maintainer what a task is, which exact aircraft/configuration it applies to, what qualifications and tools are needed, what data controls the procedure, what must be measured, and who can accept or release the result.

A friendly app cannot grant maintenance privileges. This record is a structured wrapper around the current approved/accepted maintenance information for one aircraft and one jurisdiction; it must never invent a procedure, torque value, limit, interval, or authorization.

## 2. Required task-card fields

The canonical contract should include the following groups. A first Rust implementation may split these into typed structs, but must not drop any field simply because one UI surface does not display it.

| Group | Required fields |
|---|---|
| Identity | Stable task ID, task revision, aircraft family/model/serial effectivity, configuration revision, applicable date/version range |
| Authority/source | Jurisdiction, airworthiness/certification basis, approved/accepted manual identifier and revision, document digest, task source/paragraph, date verified |
| Task category | P0/P1/P2/P3 product-design class, explicitly stating that this is not itself the legal permission |
| Eligibility | Required person/license/repairman rating, ownership or employment limitation where applicable, course/type/category/class eligibility, required supervision, sign-off restrictions |
| Preconditions | Aircraft state, safe environment, access configuration, weight/energy isolation state, inspection/maintenance status, required prior tasks |
| Tools and parts | Tool IDs, calibration requirements, part number/revision/effectivity, approved consumables, allowed substitutions and their authority |
| Procedure | Numbered, controlled steps linked to source data, warnings/cautions, inspection points, required values/tolerances and approved disposition for each possible outcome |
| Completion and return-to-service | Independent inspection or dual-control requirements, evidence fields, authorized signatory role, required record, next scheduled action and any operational restriction |
| Measurement | Planned task time, elapsed time, setup time, work time, independent inspection time, clean-up/log time, parts-wait and scheduling-wait reported separately |
| Evidence | Aircraft serial/config digest, maintainer identity/privilege reference, source/manual revision, tool calibration identity, measurement/test results, photos/files where required, timestamps, signature/attestation |
| Human factors | Required skill level, PPE/ergonomics, access constraints, step ambiguity/near miss reporting, maximum reasonable task duration and escalation path |
| Lifecycle/support | Repair versus replacement path, availability/lead time, price quote/date/currency, warranty/repair vendor, obsolescence/supersession state |

For safety-critical or regulated task procedures, link to the source document section rather than copying a potentially stale procedure into an application. If a simplified checklist is necessary, it must be revision-bound and reviewed against the controlling maintenance data.

## 3. Permission and safety state machine

Use the following internal states, which must be bound to the governing jurisdiction and aircraft certification basis:

1. **Draft** — not usable for performing maintenance.
2. **Source verified** — controlling source/revision and serial/configuration effectivity checked.
3. **Eligibility checked** — task-specific permissions and prerequisites checked for the intended performer.
4. **Ready to perform** — all required parts, tools, conditions and source data are present and in date.
5. **Work recorded** — actual procedure steps and results recorded; this does not by itself approve return to service.
6. **Inspection/review pending** — additional inspection or licensed/authorized approval is required.
7. **Accepted / released** — only an authorized person has completed the required sign-off under applicable rules.
8. **Rejected / escalated** — an out-of-limit, uncertain, damaged, missing, mismatched or failed condition requires stopping and referring to the approved escalation path.
9. **Superseded** — the task/manual/configuration has changed; prior records are retained but cannot be used for new work without revalidation.

A model must not transition a card from work-recorded or inspection-pending to accepted/released. An AI or automated checker may flag missing fields or compare recorded values with explicitly supplied limits, but the authorized human and approved procedure remain responsible for the required maintenance decision.

## 4. Fail-closed validation cases

The software contract must reject or hold the task as indeterminate when any of these occur:

- Missing or stale controlling manual revision, unresolvable source digest, or revision mismatch.
- Task applies to a different model, serial range, installed option, part number or configuration.
- Missing jurisdiction/certification basis or ambiguous permission class.
- Performer eligibility, required training, rating, supervision or sign-off authority is missing or not verified.
- Torque/clearance/tolerance/inspection criterion is missing from the controlling source or the measurement is outside its stated limit.
- Required tool calibration is overdue, unavailable, or not traceable when calibration is required.
- Isolation, support equipment, environmental conditions or prerequisites cannot be confirmed.
- A required step is skipped, a measurement is absent, or an adverse result has no approved disposition.
- Part number/revision, supplier status, release/effectivity or substitution authority is absent.
- Required second inspection/sign-off is missing.
- Timestamp ordering, source revision, aircraft identity, or evidence digest binding is inconsistent.
- Offline work is later synchronized against a different manual/configuration revision; the system must preserve the original record and escalate rather than silently rewrite it.

A missing value must not be replaced by a “reasonable” default. A failed or indeterminate task is not a claim that an aircraft is unsafe in every case; it means the software lacks evidence to record the task as completed under this contract.

## 5. Benchmark instrumentation

For each service task, record these time components separately:

- **Preparation:** workspace, safety setup, tools and documentation.
- **Access:** opening access panels and removing only the specified parts.
- **Execution:** inspection or maintenance steps.
- **Verification:** required measurements/tests and independent inspections.
- **Close-up:** reassembly, cleanup, restoring the aircraft state and checks.
- **Record/release:** maintenance log and required sign-off.
- **Wait:** scheduling or parts lead time; never blend this into technician hands-on time.

The serviceability prototype should compare the baseline aircraft with the proposed family using the same defined task boundaries and trained participant profiles. Report median, 90th percentile, task failures, instruction lookup count, tool changes, unrelated assemblies removed, rework, and every exception. A single fast demonstration is not a reliable benchmark.

## 6. Minimal structured example (synthetic only)

This is a schema-shape illustration, **not a maintenance procedure** and not a valid operational card. It intentionally lacks actual controlling source documents and authorization evidence, so a validator must leave it in Draft/indeterminate.

```yaml
task_id: EXAMPLE-NOT-A-REAL-TASK
task_revision: 0
status: draft
aircraft:
  family: Wren-2-proposal
  model: not-yet-defined
  serial_effectivity: []
  configuration_digest: null
authority:
  jurisdiction: unresolved
  certification_basis: unresolved
  controlling_manual_id: null
  controlling_manual_revision: null
  controlling_manual_sha256: null
  source_section: null
classification:
  product_design_class: P2
  legal_permission_verified: false
eligibility:
  required_authorization: null
  required_training: null
  supervision: unresolved
prerequisites: []
tools: []
parts: []
procedure_steps: []
acceptance_criteria: []
independent_inspection: required-unresolved
return_to_service:
  authorized_role: null
  signoff_required: true
measurements:
  prep_seconds: null
  access_seconds: null
  execution_seconds: null
  verification_seconds: null
  closeout_seconds: null
  record_seconds: null
  wait_seconds: null
evidence: []
limitations:
  - "Synthetic schema example only"
  - "Not approved maintenance data"
  - "Not authorization to perform or release work"
```

## 7. Acceptance criteria for a Rust validator

- Stable, canonical task identity and revision.
- Exact applicability check for aircraft model, serial/configuration and controlling data revision.
- Typed task class distinct from verified legal permission.
- Required fields and permissible value ranges fail closed.
- Result/measurement values link to their source criteria and units.
- Every state transition is attributable to an authorized actor or an explicitly non-authoritative software checker.
- No path by which AI can sign a maintenance release, bypass a required inspection, clear a safety limitation, or rewrite a prior immutable record.
- Deterministic offline-capable serialization, with conflict detection when records synchronize after source revisions changed.
- Unit tests for all negative controls in Section 4; no test pass claimed until executed on an exact code revision.

## References

- FAA [MOSAIC](https://www.faa.gov/aircraft/MOSAIC): category-specific changes to light-sport design, manufacture, maintenance and operation.
- FAA [Repairman certificates](https://www.faa.gov/repairmen): privilege, training and limitation summaries.
- FAA [Small-airplane regulations and Part 23 guidance](https://www.faa.gov/aircraft/air_cert/design_approvals/small_airplanes/small_airplanes_regs).
- EASA [CS-23](https://www.easa.europa.eu/en/document-library/certification-specifications/group/cs-23-normal-utility-aerobatic-and-commuter-aeroplanes).
- SACAA [Airworthiness and approval resources](https://www.caa.co.za/industry-information/airworthiness/).
- NASA [Maintainability design guidance](https://www.nasa.gov/reference/9-0-hardware-and-equipment-vol-2/).

## Nonclaims

This schema is a requirements artifact only. It grants no maintenance privilege, gives no aircraft-specific maintenance instructions, substitutes for no approved technical data, and makes no claim that a proposed aircraft or maintenance task is safe, compliant or airworthy.
