# Aircraft Maintenance Task Card Contract

**Date:** 2026-10-17  
**Status:** data/UX and evidence contract; not an aircraft maintenance manual and contains no executable maintenance procedures.  
**Parent:** [Owner-maintainable aircraft family](OWNER_MAINTAINABLE_AIRCRAFT_FAMILY_2026-10-17.md)  
**Flagship concept:** [S4 personal aircraft](S4_NEXT_GENERATION_PERSONAL_AIRCRAFT_2026-10-17.md)  
**Related issue:** [AIRCRAFT-LINE-001 #7761](https://github.com/Luminous-Dynamics/symthaea/issues/7761)

## 1. Purpose

Make every approved maintenance task legible, appropriately authorized, repeatable, auditable, and designed into aircraft packaging before the design freezes. This contract describes the record and governance that a future tool would use; it does not authorize work, replace approved instructions, or invent task steps.

The system must be as clear about who may not perform a task as it is about how authorized personnel find the approved task data.

## 2. Task card structure

Each task card should carry the following versioned fields.

| Field | Required meaning |
|---|---|
| Task ID / revision | Stable identity and controlled revision; revisions must never silently overwrite the version used on a past task |
| Task title / purpose | Plain-language description of the work and intended result |
| Aircraft applicability | Approved model, serial-number range, configuration, installed option and affected assembly/part revisions |
| Trigger | Approved interval, flight/cycle/calendar threshold, defect, service instruction or authorized inspection finding; source is explicit |
| Authority state | Proposed, technical-review, compliance-review, approved-for-use, superseded, withdrawn; only approved-for-use cards are actionable |
| Approved-data reference | Exact manual/section/revision, approved service instruction, AD/mandatory directive or other applicable controlled source with digest/locator |
| Legal/certificate boundary | Jurisdiction and aircraft category; distinct owner-permitted, trained-person, licensed-maintainer and inspector/sign-off requirements as applicable |
| Competence preconditions | Required qualification/authorization, training, task experience and supervision rules |
| Tools / equipment | Standard tools, calibrated instruments, support fixtures and their calibration/inspection status |
| Parts / materials | Approved part numbers, revision/serial/lot where needed, consumables, shelf-life and substitution rules from approved data |
| Hazard and isolation prerequisites | Applicable safety hazards and approved isolation/grounding/pressure/energy procedures by reference; no generated substitute steps |
| Access envelope | Zones opened, assemblies removed, working clearance, lighting, visibility, allowed posture/support equipment and expected task dependencies |
| Procedure locator | Pointer to the controlled approved procedure; this schema does not embed novel steps or AI-generated repair instructions |
| Acceptance criteria | Verifiable criteria from approved data; do not derive limits from a language model or generic threshold |
| Required inspection | Required independent inspection, duplicate inspection, functional/operational check and authorized return-to-service sign-off, as applicable |
| Records produced | Work description, discrepancy/correction, parts used, measured values, calibrated equipment, deferred/rectified status and record format required |
| Evidence attachments | Controlled photos, measurements, test results, supplier certificates and provenance; record retention and privacy rules apply |
| Expected labor baseline | Median/p90 measured task time and defined time boundary; estimated times must be labeled estimates |
| Tool/part delay | Actual tool setup and parts waiting time kept separate from touch labor |
| Failure/deviation handling | Stop criteria, out-of-limit escalation, rejected part, uncertainty and referral to authorized personnel |
| Sign-off / return to service | The authorized person and required approval/signature fields; no automation or AI system may impersonate the authorizer |
| Record provenance | Source digests, timestamps, actor identity/role, equipment calibration, task revision, as-built aircraft configuration and correction history |

Every field must be populated, explicitly not applicable with a reason, or marked missing/blocked. A missing safety-critical field fails closed.

## 3. Authorization model: separate access from permission

An aircraft designed for personal maintenance should make safe, legally permitted tasks easier—not reclassify tasks because they are physically easy.

The permission decision must be bound to at least:
- jurisdiction and effective regulation revision;
- aircraft certification category, model/serial/configuration and relevant limitations;
- person's current certificate, authorization, training and role;
- exact task and approved-data revision;
- commercial/operational context where it affects eligibility;
- whether an inspection or independent sign-off is required.

Return one of these states:

- **Permitted for this person and aircraft:** exact task and context passed applicable policy checks.
- **Requires authorized maintainer:** the task requires qualified/authorized maintenance personnel.
- **Requires inspector or return-to-service sign-off:** additional inspection/approval is required even if work steps were performed.
- **Not permitted:** task or context is prohibited under the known rules.
- **Indeterminate:** jurisdiction, category, task, credentials, approved data or applicability is missing or stale.

Only the first state allows presentation of the applicable owner task guidance, and even then only the approved content may be shown. The final logbook/return-to-service rules still apply. The UI must not turn this policy check into legal advice.

References for regulatory discovery:
- US 14 CFR §43.3: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/section-43.3
- US Appendix A to Part 43: https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/appendix-Appendix%20A%20to%20Part%2043
- FAA MOSAIC and LSA maintenance links: https://www.faa.gov/aircraft/MOSAIC
- South African SACAA public FAQ / owner-built aircraft guidance: https://www.caa.co.za/information-for-the-public/frequently-asked-questions/

This is a system-design contract. It is not an interpretation of every regulation for every aircraft/task; the applicable rules and authority guidance must be confirmed for the selected design.

## 4. Task lifecycle and change control

1. **Proposed:** engineering or service team drafts metadata; never actionable.
2. **Engineering review:** verifies applicability, parts, tools, access, hazard analysis and serviceability.
3. **Authority/compliance review:** resolves the applicable approved-data and certification path; retains evidence of required approval.
4. **Approved for use:** released only under the proper design/maintenance authority with a controlled content revision.
5. **Superseded or withdrawn:** blocks new starts; past task records remain linked to the version actually used.
6. **Corrected:** correction creates an auditable new record/version; no deletion or silent mutation of previous evidence.

A service bulletin, airworthiness directive, approved alteration, part supersession, configuration change or newly identified hazard must trigger applicability review. The system must identify affected aircraft/tasks, but the competent organization/person remains responsible for making and executing the applicable airworthiness decision.

## 5. Serviceability design feedback

Task cards should be used upstream during architecture and packaging, not just after the aircraft exists.

For each candidate task, record:
- number of access panels opened;
- number and type of fasteners removed/reinstalled;
- unrelated assemblies disturbed;
- tool changes, calibration and setup time;
- measured line-of-sight and working envelope;
- connectors/hoses/controls disconnected;
- risk of incorrect reconnection or trapped hardware;
- inspection/functional checks needed after reassembly;
- corrosion, chafe, leakage, contamination and damage exposure;
- labor time, rework, errors and parts/tool waiting time.

Design the aircraft based on this information before geometry freezes. Compare packaging alternatives before selecting one. Optimize total service task and verification burden, not only time to remove a component.

Do not use a low average as the whole story. Keep median, 90th percentile, operator skill, failed attempts, rework and safety-step omissions visible. A task-time reduction achieved by omitting an inspection is a failure, not an improvement.

## 6. Evidence integrity for Symthaea and Mycelix

A future maintenance application may use Symthaea to:
- locate relevant approved data;
- compare the observed aircraft configuration to declared applicability;
- surface known discrepancy history with confidence and source evidence;
- detect missing signatures, stale task revisions, mismatched part serials or overdue records;
- summarize measured task time and recurring defects;
- propose serviceability improvements based on fleet evidence.

It must not:
- invent or rewrite approved task steps;
- declare a task legally permitted from natural-language similarity alone;
- suppress a warning due to one aggregate health score;
- clear a mandatory directive or defect without authorized evidence;
- make an airworthiness or return-to-service decision;
- count a prediction, queued test or missing evidence as a pass.

Mycelix can preserve attestations, source relationships, correction history and access-governed evidence. The ledger or provenance graph cannot make untrue measurements true. Allow correction/revocation records, preserve originals and their status, and scope sensitive owner/location/aircraft data. No actual aircraft integration is claimed.

## 7. Benchmarks

| Metric | Method |
|---|---|
| Task-card completeness | 100% of required fields populated or explicit, reasoned not-applicable; zero silent defaults |
| Applicability correctness | Negative tests for wrong model, serial range, optional configuration, superseded manual and jurisdiction |
| Permission gating | Every role/task/certification combination tested; unknowns return indeterminate and block actionable content |
| Approved-source binding | All actionable cards resolve to the controlled source revision/digest; mismatched digest blocks release |
| Task execution | Trained participants perform supervised non-flight tasks; record correct completion, errors, omitted steps and escalation |
| Serviceability | Median/p90 elapsed and touch labor, parts/tool delay, unrelated disassembly and verification included separately |
| Evidence traceability | 100% of recorded task instances bind aircraft serial/configuration, task revision, approved-data revision, actor authorization and required sign-off |
| Offline usability | Required approved documents and current aircraft maintenance records remain available locally without network access |
| Revision response | Time to identify affected cards/configurations after a controlled task or part revision; no past record silently changes |
| Correction behavior | Missing, stale, invalid and revoked attestations trigger fail-closed states; original record remains inspectable |

The percentages are internal design targets, not measured outcomes or legal thresholds. The authorized organization must define acceptance criteria for a real product.

## 8. Initial implementation gates

- **T0 — schema:** versioned Rust/data types and validation for identity, applicability, authority status, approved-source references, roles, records and evidence.
- **T1 — negative controls:** wrong serial, wrong option, wrong jurisdiction, stale approved data, revoked source, missing sign-off, task role mismatch and unapproved procedure all fail closed.
- **T2 — mock-up trials:** approved non-flight representative tasks used for timed accessibility trials with trained maintainers.
- **T3 — controlled implementation:** selected maintenance documents and records are tied to a configuration-controlled design and compliance process.
- **T4 — fleet operation:** measure repeat discrepancies, task duration, parts supply, AOG time, owner usability and corrections.

Do not implement aircraft maintenance instructions or task-authority policy as a side effect of the current energy-screening crate. This contract should be reviewed with architecture owners before an implementation location and data schema are selected.

## Nonclaims

This schema is not a maintenance manual, does not grant maintenance privileges, and does not establish the airworthiness of any aircraft. It provides no repair procedure or certification shortcut. Actual task content and release must originate from the applicable approved maintenance data and competent organization/authority.
