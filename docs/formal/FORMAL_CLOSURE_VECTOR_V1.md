# Formal Closure Vector V1

Tracking: SYM-FV-020 / #5730, SYM-FV-020A / #6012, implementation #6013.

## Purpose

A formal proof should make both its achievement and its remaining assurance gaps visible.

The closure vector records those gaps without inventing a universal verification score.

```text
proved theorem
!= source refinement
!= compiler theorem
!= exact build
!= released artifact
!= running process
```

## Core dimensions

Every record carries the same horizontal dimensions:

- `TheoremStatementIdentity`
- `MathematicalSpecification`
- `CheckerIndependence`
- `SourceRefinement`
- `OptimizedImplementationRelation`
- `CompilerSemantics`
- `DependencyClosure`
- `BuildIdentity`
- `ArtifactIdentity`
- `ReleaseIdentity`
- `RuntimeIdentity`
- `EnvironmentRealization`
- `DistributedAssumptions`
- `CrossRepositorySubjectClosure`

Domains may add `extensions`, but extension values use the same state contract.

## State vocabulary

`Closed`
: Requires exact supporting receipt/capsule identity. It means only that the named dimension is closed under the cited claim ceiling.

`Open`
: Requires an explicit unresolved obligation. Open is not failure.

`NotApplicable`
: Requires a reason. It may not be used merely to hide an inconvenient gap.

`ImportedAssurance`
: Requires provider, exact profile/version identity, claim ceiling, **and exact provider evidence/support identity**. A provider name, product name, paper, certification label, or version string by itself is not imported evidence. Imported assurance is not local proof.

`BoundedOnly`
: Requires an exact bound/profile and exact supporting evidence. Finite-state or bounded evidence must remain visibly bounded.

`Unknown`
: Requires a reason. Unknown is neither false nor open-by-default.

## No aggregate score

Fields such as `coverage_percent`, `verified_percent`, `verification_score`, or `overall_score` are forbidden.

Semantic criticality matters more than line count or the number of green cells.

## Anti-amplification

A single receipt may not be copied across heterogeneous closure dimensions to create an all-green assurance story.

For example:

```text
AbstractFormalTheorem receipt
!= SourceRefinement closure
!= CompilerSemantics closure
!= RuntimeIdentity closure
```

Each transition needs compatible exact evidence.

Likewise:

```text
named external provider/profile
!= ImportedAssurance
```

until an exact provider evidence/support identity and its claim ceiling are bound.

## Extensions

Extensions exist for domain-local assurance seams without changing the horizontal vocabulary.

Example ZKP extensions include:

```text
GuestImplementationRefinement
ImageMethodIdBinding
ProofSystemSoundness
ReplayCurrentness
AuthorizationAdmission
```

They do not become global core dimensions merely because one domain needs them.

## Seeded subjects

`formal_closure_vector_v1.examples.json` contains current review fixtures for:

1. repaired BinaryHV abstract XOR algebra;
2. evidence-composition non-amplification theorem;
3. abstract ZKP balance statement relation.

The examples deliberately preserve queued/source-authored states as `Open`/`Unknown`; they do not manufacture `Closed` from unexecuted workflows.

The ZKP fixture deliberately leaves `ProofSystemSoundness` as `Open` until an exact RISC Zero assurance profile/receipt is imported. Merely knowing that the external proof system has its own assurance work is not enough to mark the dimension `ImportedAssurance`.

## Boundary

```text
closure manifest != proof
many Closed dimensions != universal correctness
Open != failure
Unknown != false
ImportedAssurance != local proof
BoundedOnly != unbounded theorem
closure vector != quality score
```

The first implementation validates structure and hostile semantic mutations. Future children may bind closure states to exact proof-capsule/evidence-graph receipts, but must not create a second dependency DAG.