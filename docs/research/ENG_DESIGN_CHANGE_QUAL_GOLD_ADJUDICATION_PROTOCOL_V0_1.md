# ENG-DESIGN-CHANGE-QUAL-001 — Independent gold-label adjudication protocol v0.1

**Status:** DESIGN_ONLY / evaluator protocol specification  
**Parent benchmark:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Public blueprints:** [corpus PR #10546](https://github.com/Luminous-Dynamics/symthaea/pull/10546)  
**Machine-readable blueprint registry:** [manifest PR #10547](https://github.com/Luminous-Dynamics/symthaea/pull/10547)  
**Execution:** no candidate system, CAD/CAE tool, ETK action, external MBSE/PLM product, production record or physical process is authorized by this protocol.

## 1. Purpose and non-goal

This document defines how independent reviewers should author and seal gold labels for the 24 public change-impact blueprints before any scored candidate evaluation.

The core evaluation question is not merely "did the candidate list the changed nodes?" It is:

> For each claim and prior evidence item, what is the strongest justified applicability disposition for the candidate configuration, given the exact source and target generations, semantic delta, dependency contract, and evidence lineage?

The evaluator must score both:
- **under-impact:** a truly claim-relevant change is missed, allowing stale or inapplicable evidence to appear current;
- **over-impact:** an irrelevant or equivalence-preserving change triggers unnecessary review/requalification.

A candidate is not rewarded for reopening everything, nor for preserving everything. Critical false-current errors are hard failures; excess requalification is measured separately and may not offset them.

This is an evaluator protocol, not a new dependency graph, semantic delta engine, evidence-currentness system, requirement authority, qualification kernel, review workflow, or release authority.

## 2. Existing authorities and ownership boundaries

The adjudication process consumes exact references to existing owners:

- [ENG-DESIGN-005 #6117](https://github.com/Luminous-Dynamics/symthaea/issues/6117): authored transformation/delta semantics.
- [Systems Engineering #3677 / SE-001](https://github.com/Luminous-Dynamics/symthaea/issues/3677): requirements, interfaces, assumptions, hazards, verification intent and semantic dependency projections.
- [ETK #1738 / #1730 / #1760](https://github.com/Luminous-Dynamics/symthaea/issues/1738): accepted requirements, evidence applicability/currentness, assurance and discharge authority.
- [SYM-FV #5794 / #6065 / #6073](https://github.com/Luminous-Dynamics/symthaea/issues/5794): generic evidence closure and non-amplification.
- [ENG-TOL-001 #4883](https://github.com/Luminous-Dynamics/symthaea/issues/4883): dimensional requirement, metrology and as-built semantics.
- [ROB-CATALOG #4877](https://github.com/Luminous-Dynamics/symthaea/issues/4877) and [ROB-REALIZE #4859](https://github.com/Luminous-Dynamics/symthaea/issues/4859): exact article/variant lineage and as-built identity.
- [CAD-HISTORY-001 #10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510) and [HARDWARE-TOPOLOGY-001 #10530](https://github.com/Luminous-Dynamics/symthaea/issues/10530): geometry/edit, assembly, ECAD and manufacturing task semantics.

The gold set may reference these subjects, relations and evidence classes. It must not recreate them as a competing canonical schema. The oracle's expected disposition is an evaluation label only: it cannot make an ETK decision or issue an engineering approval.

## 3. Unit of adjudication

The primary scoring unit is a **claim–evidence applicability pair**, not an entire design package and not an entire prior receipt.

For each pair, the oracle records:
- exact claim/requirement/hazard/verification proposition;
- exact prior evidence subject and evidence class;
- exact source configuration and target configuration;
- dependency path or qualified independence witness;
- transformation intent and observed semantic delta;
- applicability disposition;
- mandatory follow-up obligation, if any;
- criticality class and whether a false-current result is a hard failure.

A single change may affect several claims differently. For example, changing a load-bearing cross-section can affect a stress analysis, a dimensional inspection plan, a paint specification, and a user-interface validation in different ways. Do not attach one global impact status to the whole project.

## 4. Gold-label disposition vocabulary

Each prior evidence item must receive exactly one primary disposition for the target generation. Supplemental obligations may be attached, but must not replace the primary state.

| Disposition | Meaning | Required evidence in the gold record |
|---|---|---|
| `PreservedByIdentity` | The evidence subject and every claim-relevant dependency are exactly unchanged under the owner-defined identity contract. | Exact identity/ref equality and dependency closure. |
| `PreservedByQualifiedEquivalence` | Source and target subjects differ, but a named, qualified equivalence witness covers this exact claim and transformation. | Exact witness ID/profile, covered fields/claims and scope. |
| `PreservedByExactRecalculation` | The old evidence is not reused as current; the claim is re-established by a new deterministic calculation on the target configuration. | New calculation input/output identity, tool/profile, oracle and provenance. |
| `RequiresApplicabilityReview` | Impact cannot yet be resolved; an authorized owner must review scope/applicability. | Exact unknown dependency or missing equivalence/authority fact. |
| `RequiresVerification` | A verification activity is needed against the target configuration. | Requirement, verification method, input configuration and acceptance contract. |
| `RequiresPhysicalEvidence` | The claim depends on physical inspection, test, metrology or operational evidence not supplied by the existing record. | Exact measurement/test subject, instrument/process and required evidence class. |
| `RequiresHumanOrExternalApproval` | A licensed professional, customer, authority, manufacturer or other named external actor must decide. | Named approval authority class and the exact decision scope. |
| `NotPreserved` | The source evidence does not establish the target claim; the change is known to break or materially alter the relevant condition. | Counterexample, changed dependent property, violated invariant or failed proof. |
| `UnknownImpact` | The available dependency/equivalence information is insufficient to decide whether the claim is affected. | Missing relation, incomplete model, conflicting input or non-identifiable dependency. |
| `Unsupported` | The required semantics or evidence class is outside the benchmark's declared supported profile. | Exact unsupported construct/profile and safe disposition. |

### Prohibited collapses

These are distinct:
- evidence was once valid;
- evidence is valid for a prior generation;
- evidence remains applicable to a candidate generation;
- evidence was recalculated for a candidate generation;
- evidence has been admitted/current under ETK authority;
- a requirement was accepted or discharged;
- a design was approved or released.

A gold label must never imply the latter states merely from a benchmark answer.

## 4.1 Machine-readable evaluator contract

The strict JSON Schema 2020-12 counterpart is maintained separately at:

`benchmarks/engineering-change-impact/gold-record.schema.json`

It is evaluator-only, not part of the public 24-blueprint manifest and must not be mounted into a candidate workspace. It constrains the disposition vocabulary, source/target generation digests, dependency paths, equivalence-witness scope, follow-up obligations, criticality, assessor records, adjudication state, seal state and claim ceiling. The schema does not create gold labels or seal any record; the current public corpus remains unadjudicated.

## 5. Gold record format

The evaluator-only gold record for each claim–evidence pair should contain:

```text
ChangeImpactGoldRecordV1
├── case_id
├── pair_id
├── claim_ref
├── evidence_ref
├── source_generation_ref + digest
├── target_generation_ref + digest
├── transformation_ref + digest
├── source_evidence_class
├── dependency_contract_ref + revision
├── dependency_paths[]
│   ├── relation_ref
│   ├── relation_authority
│   └── semantic_role
├── equivalence_witness_ref? + scope?
├── expected_disposition
├── required_follow_up_obligations[]
├── criticality / hard_gate
├── rationale_codes[]
├── assessor_records[]
├── adjudication_record
└── provenance / digest / sealing state
```

Use the exact owner IDs where available. If a scenario is purely synthetic and has no upstream owner-issued identity, bind a fixture-local ID and clearly mark it as synthetic. Never invent an ETK evidence ID or claim an upstream authority decision.

Suggested rationale codes include:
- `IDENTITY_UNCHANGED`
- `DEPENDENCY_OUTSIDE_PROVEN_NONIMPACT_CONE`
- `QUALIFIED_EQUIVALENCE_COVERS_CLAIM`
- `CLAIM_DEPENDENCY_CHANGED`
- `REQUIREMENT_REVISION_CHANGED`
- `ASSUMPTION_DRIFT`
- `INTERFACE_FRAME_CHANGED`
- `MATERIAL_PROPERTY_CHANGED`
- `LOAD_OR_DUTY_CHANGED`
- `PROCESS_CAPABILITY_CHANGED`
- `MEASUREMENT_PROFILE_CHANGED`
- `CONFIGURATION_GENERATION_CHANGED`
- `DEPENDENCY_MODEL_INCOMPLETE`
- `CONFLICTING_REFERENCE`
- `UNSUPPORTED_ANALYSIS_CLASS`

Rationale codes are descriptive. They do not replace an exact dependency path, witness or counterexample.

## 6. Adjudication decision procedure

Reviewers must apply the following sequence independently for each claim–evidence pair.

1. **Resolve exact subjects.** Confirm source and target generation IDs/digests, claim identity, evidence identity and evidence class. Reject ambiguous or duplicated subject resolution.
2. **Resolve the governing requirement.** Identify the exact requirement/constraint revision, applicable profile and intended-use context. Do not infer requirement equivalence from similar wording.
3. **Resolve the transformation.** Read the authored intent and the actual semantic delta. A caller's `DocumentationOnly` or `NoImpact` assertion is not proof.
4. **Trace the dependency contract.** Identify direct and transitive dependencies, including assumptions, interfaces, load cases, material properties, component variants/lots, process routes, measurement profiles and software/calibration dependencies where relevant.
5. **Test non-impact claims.** Preserve prior applicability only when the owner-defined dependency contract proves the changed facts are outside the claim's dependency cone, or a qualified equivalence witness explicitly covers the transformation and claim.
6. **Test changed dependencies.** Where a claim-relevant dependency changed, determine whether exact recalculation, verification, physical evidence, applicability review or external approval is required.
7. **Preserve uncertainty.** If the graph is incomplete, evidence is contradictory, the equivalence witness does not cover the affected property, or the input cannot identify the dependency, label `UnknownImpact` or `RequiresApplicabilityReview`. Do not guess the missing edge.
8. **Check historical integrity.** Preserve the old evidence as a historical record even when it is no longer applicable to the target generation.
9. **Assign criticality.** Mark whether a false-current answer is a non-compensatory critical error. Never assign criticality based on the candidate's own output.
10. **Record rationale and disagreements.** Store each assessor's independent label and supporting path/witness before adjudication.

No scalar risk or readiness score may override the pair-level disposition.

## 7. Worked synthetic examples

These examples illustrate how a future gold set could be authored; they are not labels for any current public blueprint and must not be copied into the public candidate-visible corpus.

### Example A — exact documentation-only edit

**Source generation:** `G-ALPHA-01`.

**Target generation:** `G-ALPHA-02`, where only a non-identity-bearing explanatory paragraph changes.

**Evidence pair:** a solver-independent calculation receipt bound to the exact requirement revision, exact geometry subject and exact load profile. The semantic graph projection, geometry, interface, assumption, material, load and requirement revisions are identical.

**Gold disposition:** `PreservedByIdentity` only if the upstream identity contract explicitly says that this text field is non-identity-bearing and the claim/evidence binding is exactly unchanged. The source byte digest will differ; the gold record must not pretend otherwise.

**Adversarial near miss:** alter one numeric load value in the same paragraph while keeping the change classified as `DocumentationOnly`. The actual semantic delta must dominate the caller's label. If load is a dependency of the calculation, the old evidence is not preserved for the new generation.

### Example B — changed operating load with unchanged CAD geometry

**Source generation:** `G-BETA-01`, geometry (G), material (M), load profile (L_1), structural calculation evidence (E_1).

**Target generation:** `G-BETA-02`, exact same geometry and material, but load profile (L_2) differs in magnitude or direction.

**Gold disposition for the old structural calculation (E_1):** `RequiresVerification` or `RequiresExactRecalculation` (represented using the primary vocabulary as `RequiresVerification` unless a new deterministic calculation is already included in the task). It is not preserved just because the CAD geometry hash is unchanged.

**Gold disposition for an unrelated enclosure-colour claim:** may be `PreservedByIdentity` only if the owner-defined claim dependencies prove independence from the load profile.

**Adversarial near miss:** change a preference-only property (e.g. visual finish preference) while leaving (G, M, L_1) and the requirement revision unchanged. Do not create a structural requalification obligation without a real dependency path.

### Example C — material substitution with visually identical geometry

**Source generation:** exact supplier/material variant (M_1), geometry unchanged, evidence (E_1) depends on modulus, temperature range and corrosion profile.

**Target generation:** same nominal geometry, new material variant (M_2). The new material has different physical properties.

**Gold disposition:** `RequiresApplicabilityReview` or `RequiresVerification` for claims whose dependency set includes the changed material properties. Claims tied only to geometry may remain applicable if their exact owner contract proves no material dependency. A visually identical mesh is not an equivalence witness.

**Adversarial near miss:** manufacturer-approved equivalence witness covers a named dimensional envelope only. It cannot preserve structural, thermal, electrical or environmental claims outside that scope.

## 8. Criticality and scoring rules

Primary metrics:
- affected-path recall and precision, by relation/claim class;
- exact evidence applicability disposition accuracy;
- stale-evidence false-current count;
- missed mandatory verification/physical evidence/external approval obligations;
- unnecessary review/requalification burden;
- correct unknown/unsupported/clarification disposition;
- preservation-witness validity;
- full case/claim/evidence denominator coverage.

Non-compensatory hard gates:
- any seeded critical false-current disposition fails the critical gate;
- any omitted mandatory physical evidence/approval obligation fails the critical gate;
- any candidate claim that silently deletes or overwrites historical evidence fails the integrity gate;
- any invented dependency or equivalence witness fails the provenance/semantic gate;
- any attempt to mint ETK acceptance, professional approval or release from a benchmark output fails the authority-boundary gate.

Over-impact is measured as the count/cost of avoidable review and requalification obligations only after under-impact hard gates are satisfied. Do not trade safety-critical false negatives for efficiency.

Use paired per-task results, cluster uncertainty by design family/change lineage, and report each family separately. A 24-blueprint pilot is not a statistically powered confirmatory study.

## 9. Assessor procedure and independence

For each gold record:
- Two qualified assessors independently label the claim–evidence pair without seeing candidate outputs or each other's label.
- Each assessor must cite the exact dependency contract, source/target refs, path/witness/counterexample and rationale code.
- If both labels agree and the evidence is complete, seal the gold record.
- If labels disagree, a third adjudicator reviews the two records and resolves the disagreement with a written rationale; preserve the original assessor records.
- If the dependency contract is insufficient to decide, the correct gold label is `UnknownImpact` or `Unsupported`, not a forced binary label.
- For critical cases, domain qualification must match the relevant claim class (structural, electrical, material, metrology, process, software/control, or external authority).
- Assessors must declare conflicts of interest and whether they authored the fixture or upstream owner contract.

Record agreement statistics by disposition/family (for example, confusion matrices and Cohen's kappa where its assumptions are appropriate). Do not hide disagreement by reporting only the adjudicated label.

## 10. Leakage resistance and split design

- Public blueprints remain authoring specifications.
- Candidate-visible tasks must not contain gold labels, hidden dependency paths, evaluator mutation names or assessor rationales.
- Split by base design family and change lineage, not by random row, so closely related transformations do not leak across development and holdout.
- One base design with many mutations is one correlated family for statistical purposes.
- Keep source generation, candidate generation, gold evaluator, scorer, and task-custody digests separately bound.
- A future hidden holdout needs independent design-family authoring, independent assessor review and access control. The current 24 blueprint bank is **not** a hidden holdout.
- Do not include real production, supplier, customer, safety-sensitive or personal data without the relevant authorization and custody policy.

## 11. Mutation tests for the adjudicator and scorer

Before scoring a candidate, mutate the evaluator/record and require fail-closed rejection for at least:
1. remove one critical affected evidence pair;
2. delete a mandatory physical verification obligation;
3. change the source or target generation digest;
4. substitute a witness from another claim or transformation;
5. truncate a dependency path while leaving its terminal claim;
6. duplicate a pair ID or evidence ID;
7. relabel a stale evidence item as current;
8. omit an unknown/unsupported case from the denominator;
9. alter a criticality flag after candidate scoring;
10. copy a public blueprint into a claimed hidden holdout;
11. modify the assessor decision after seal without a superseding adjudication;
12. promote a benchmark disposition into ETK acceptance/release.

A passing evaluator mutation suite establishes only integrity of the named evaluator implementation and frozen test set, not engineering correctness.

## 12. Runner-independent next steps

1. Review the 24 blueprints in #10546 against this adjudication record format.
2. Map each blueprint to exact existing dependency/claim/evidence owner references without creating a parallel graph.
3. Select a small pilot across at least four change families; do not mark the remaining labels as adjudicated by extrapolation.
4. Author source/target synthetic package facts and claim–evidence pair tables.
5. Obtain independent domain assessor labels, record disagreement, and seal only fully supported cases.
6. Add evaluator mutation tests and denominator checks before any candidate agent is connected.
7. Create a separate held-out design-family split only after the pilot schema and adjudication protocol are stable.
8. Only then, and only after custody/isolation prerequisites qualify, run matched candidate baselines.

## 13. Claim ceiling

This protocol defines an independent gold-label authoring and scoring method. It does not establish that any change-impact engine, Symthaea feature, ETK service, MBSE/PLM product or human team achieves a particular score. It does not establish safety, manufacturability, compliance, professional approval, release or physical qualification. All 24 current public blueprints remain unadjudicated and unscored until independent assessors finish the specified process.
