# CP-03 Manufacturing Digital Thread Contract v1

Status: architecture contract / representation-only qualification target

## Claim ceiling

This contract establishes only deterministic software semantics for binding manufacturing intent, process definition, capability profile, execution context, as-built identity, metrology references, and engineering disposition. It establishes no physical manufacturing capability, machine performance, process qualification, product conformity, safety certification, production authorization, or operational authority.

## 1. Boundary and ownership

CP-03 composes existing manufacturing and engineering owners. It does not create a replacement manufacturing ontology, machine registry, metrology system, commissioning authority, or operational ledger.

Canonical lineage reused by reference includes:
- MFG-PROC-002 / issue #5690 for process-state transformation boundaries.
- MFG-PROC-003 / issue #5692 for evidence-bounded manufacturing capability.
- CP-01 Metrology for measurement, uncertainty, calibration/currentness, and traceability semantics.
- ENG-DESIGN / SE-VV for requirements and verification semantics.
- Existing physical/as-built owners for article and configuration identity.
- Mycelix / CIV-Service only for custody, coordination, service, and operational consequences where explicitly assigned.

A manufacturing record MUST identify its canonical owner and exact generation. A human-readable “latest” alias is never an evidence identity.

## 2. Core distinctions

The following are distinct semantic subjects:

`engineering requirement != process definition != process plan != capability profile != execution != as-built state != observation != engineering disposition != operational authority`

Also:

`capability != availability != capacity != execution authorization`

And:

`declared transformation != observed physical state`

A process contract may state what transformation is intended or expected. Only an observation-bearing evidence chain can establish an observed state, and CP-01 evidence remains context-bound to its physical subject and measurement generation.

## 3. Canonical manufacturing thread

The closed thread is:

`engineering requirement -> process definition -> capability/profile match -> process plan -> execution context -> physical/as-built generation -> CP-01 measurement -> observation/uncertainty/traceability -> engineering disposition`

Each edge is explicit. Missing edges produce an unresolved disposition rather than an inferred PASS.

## 4. Process identity and generation

A `ProcessStateRefV1` identifies:
- process_id
- process_generation
- process_definition_ref
- process_profile_ref
- equipment_ref
- equipment_configuration_ref
- tooling_fixture_ref
- material_feedstock_ref
- parameter_set_ref
- environment_profile_ref
- execution_ref
- as_built_ref
- measurement_refs
- dependency_refs
- currentness_ref
- authority_disposition
- claim_ceiling

A new semantic generation is required when an evidence-relevant dependency changes, including:
- process definition or recipe;
- equipment identity/configuration;
- tool, fixture, datum, or mounting context;
- material/feedstock lot or relevant material state;
- parameter set;
- environment profile when applicability depends on it;
- execution firmware/controller/parser where it can affect process semantics;
- as-built identity/configuration;
- acceptance or measurement profile.

Historical executions and observations remain attached to their original generations. Requalification creates derived dispositions; it does not rewrite history.

## 5. Design/planning/execution/as-built separation

These states MUST remain separate:

1. as-designed
2. as-planned
3. as-programmed
4. as-executed
5. as-built
6. as-tested
7. as-commissioned
8. as-served

A successful process plan is not evidence of execution. An execution record is not evidence of conformance. An as-built record is not evidence of a property unless the relevant measurement exists.

Rework, repair, remanufacture, or material substitution MUST create an explicit new generation or transformation event. It MUST NOT mutate the original as-built state.

## 6. Capability semantics

A `ProcessCapabilityProfileV1` is profile-relative and generation-bound. It may declare:
- admissible input class;
- required equipment/configuration;
- process envelope;
- declared throughput/capacity;
- required controls;
- required metrology coverage;
- known limitations;
- qualification evidence references;
- currentness/applicability.

It MUST NOT be interpreted as:
- current machine availability;
- permission to execute;
- evidence that a particular execution succeeded;
- evidence that a population conforms.

Capacity and availability are operational states and remain distinct from engineering capability.

## 7. Metrology seam

CP-03 references CP-01 records by exact immutable identity.

Manufacturing MUST NOT copy authoritative observation values into a new manufacturing evidence store. A manufacturing record may cache derived projections, but the canonical observation, uncertainty, calibration/currentness, datum/frame, and traceability remain owned by CP-01 or the existing canonical owner.

The acceptance chain is:

`as-built_ref -> measurement_subject_ref -> observation_ref -> uncertainty_ref -> traceability_ref -> engineering_disposition`

A missing, stale, mismatched, or insufficient measurement narrows the manufacturing claim.

## 8. Negative evidence and non-conformance

Negative evidence is first-class:
- process envelope exceeded;
- required parameter not recorded;
- equipment configuration mismatch;
- tooling/fixture state unknown;
- material lineage incomplete;
- measurement capability insufficient;
- calibration stale;
- observation contradicts expected state;
- acceptance requirement not satisfied;
- rework performed without a new generation.

Negative records are immutable inputs to later recomputation. Recomputed dispositions may change; historical failures cannot disappear because a later evaluator or ranking changed.

## 9. Independence and common-mode

Multiple manufacturing measurements are not independent merely because they have different labels.

Common dependencies that can create common-mode evidence include:
- same calibration root;
- same sensor/reference chain;
- same fixture datum;
- same acquisition/parser;
- same correction model;
- same process parameter source;
- same machine controller.

`numeric agreement != independent evidence`.

## 10. Lifecycle boundary

Manufacturing evidence establishes a historical manufacturing state. Later service, maintenance, degradation, repair, or operational events may create new lifecycle state but MUST NOT retroactively alter the original manufacturing record.

A repair can establish a new service/as-repaired generation. It cannot turn the original as-built state into an as-repaired state.

## 11. Authority boundary

Software qualification can establish representation semantics and deterministic replay over synthetic/reference records.

It cannot:
- start a machine;
- command a process;
- certify a physical article;
- approve production;
- authorize a human operator;
- grant regulatory approval;
- convert a CIV service event into engineering evidence.

Physical execution remains behind existing equipment/HAL/operator/commissioning authority.

## 12. Mycelix and CIV seam

Mycelix may preserve identity, custody, provenance, coordination, consent, or operational events where those systems own them. Such provenance does not manufacture engineering truth.

CIV-Service may project a qualified capability into service availability/continuity state. A service event cannot create or upgrade manufacturing evidence.

## 13. Replay contract

A qualification replay is over immutable identity-bearing inputs:
- exact requirement generation;
- process definition/profile generations;
- execution context;
- as-built identity;
- referenced CP-01 measurement identities;
- dependency/currentness identities;
- ordered case manifest.

Derived dispositions are excluded from replay identity. Replaying the same immutable inputs MUST yield the same dispositions.

## 14. Synthetic qualification corpus

The companion corpus is `cp-03-manufacturing-corpus-v1`. It is representation-only and has zero physical execution authority.

Minimum adversarial coverage:
1. complete closed thread;
2. missing process generation;
3. changed equipment configuration;
4. changed tooling/fixture;
5. changed material/feedstock generation;
6. plan without execution;
7. execution without as-built identity;
8. as-built without CP-01 measurement;
9. stale/mismatched measurement;
10. one specimen presented as population evidence;
11. capability confused with availability;
12. capacity confused with capability;
13. rework without new generation;
14. negative evidence lost on recomputation;
15. common-mode measurements presented as independent;
16. operational event presented as engineering qualification;
17. synthetic PASS presented as physical qualification;
18. changed acceptance profile requiring requalification.

## 15. Production adapter gate

A production adapter is admissible only when it:
- reuses canonical owners;
- records exact generations;
- distinguishes plan/execution/as-built/test/commissioning;
- references CP-01 evidence without copying authority;
- preserves negative evidence;
- exposes unresolved states;
- binds a reproducible receipt to an exact source snapshot;
- has independent qualification evidence;
- keeps physical execution behind existing authority boundaries.

The synthetic qualifier MUST NEVER be represented as production process qualification.
