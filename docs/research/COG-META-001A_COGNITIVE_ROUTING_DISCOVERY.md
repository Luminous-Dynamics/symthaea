# COG-META-001A-R2 — Cognitive Routing / Scheduling Discovery Contract

Status: measurement-only product specification  
Parent: #5468  
Issue: #5469  
Supersedes unexecuted draft product #5472 and draft qualifier #5473.

## Frozen source subject

This R2 subject remains a direct semantic child of:

```text
main = adb69f11fa8068b019cc5bb598d0c7726a197fc9
tree = 35a6c5fdba319556af9bb487838734f67c8ac0d6
```

R2 exists because static review of R1 demonstrated two measurement defects before
qualification:

1. R1 enumerated filesystem `*.rs` paths with `rglob` rather than binding the
   census to Git-tracked Rust source.
2. R1 did not freeze the demonstrated cross-domain legacy control chain:

```text
CognitiveDepth
  -> CycleUrgency
  -> subsystem cadence

CycleUrgency
  -> MeshUrgency
  -> physical transport class
```

nor the existing `CognitiveDepth` consumers for memory breadth,
neuromodulation, and budget scaling.

No R1 workflow executed and no PASS was claimed, so R2 discards no admitted
evidence.

## Measurement claim

The script performs a deterministic, conservative, file-level lexical census
over **Git-tracked production Rust files**.

```text
file appears in census
!= runtime reachable

token appears in file
!= token controls behavior on every path

same name
!= same semantics
```

The script may emit `result=PASS_DISCOVERY` when its own frozen measurement
contract is satisfied. That is not a repository qualification result.

## Subject binding

The report emits:

```text
subject_commit
subject_tree
production_rust_file_count
production_rust_path_set_sha256
```

Every category member emits:

```text
repository-relative path
byte length
Git blob SHA-1
SHA-256
lexical match count
```

Category and definition sets receive domain-separated SHA-256 digests.

## Required categories

R2 freezes conservative file-level surfaces for:

```text
cadence_scheduling
cognitive_regime_routing
content_competition
metacognitive_control
specialist_selection
deliberation_stopping
resource_gating
planning_search_simulation
memory_operation_selection
external_effect_proposal
mandatory_or_protective
transport_qos_projection
```

## Required cross-domain witnesses

R2 additionally requires exact source-level lexical witnesses for:

```text
CognitiveDepth -> CycleUrgency
CycleUrgency -> MeshUrgency
CognitiveDepth -> memory recall breadth
CognitiveDepth -> neuromodulation
CognitiveDepth -> resource/budget scale
meta-reasoning confidence -> learning modulation
workspace winner-take-all
metacognitive recommendation consumption
```

These witnesses establish only that the coupling exists in the frozen source
subject.

They do **not** establish that the coupling is wrong or that the proposed
replacement is better.

## Same-name semantic collision census

Definition sets are frozen for:

```text
CycleUrgency
CognitiveDepth
CognitiveSubsystem
AttentionBid
GlobalWorkspace
MetacognitiveRecommendation
MetaCognitiveReasoner
MetacognitiveReasoner
MeshUrgency
```

Multiple definitions emit:

```text
SEMANTIC_COLLISION_CANDIDATE
```

This marker means only that the same lexical symbol is defined in more than one
production source location under the scanner's exact regex.

It does not authorize deduplication or unification.

## Important boundaries

```text
surprise / prediction error
!= compute value

CognitiveDepth
!= reliability class

CycleUrgency
!= task deadline

CycleUrgency
!= transport QoS by semantic identity

AttentionBid
!= CognitiveBid

metacognitive recommendation
!= calibrated value-of-computation

operation selected
!= operation useful

cognitive allocation
!= external action authority
```

## Qualification contract

A future never-merge qualifier should:

1. prove exact `main -> R2 product -> qualifier` ancestry;
2. prove R2 product delta is exactly this document + the R2 discovery script;
3. detach the exact R2 product;
4. syntax-check the script;
5. execute it twice;
6. require byte-identical reports;
7. require exact schema/scope/reference semantics;
8. require subject commit/tree to equal the frozen R2 product;
9. require all explicit cross-domain control witnesses;
10. preserve the full report plus SHA-256 as an artifact.

Only that execution may establish a `PASS_INVENTORY`-style qualification under
its exact claim ceiling.

## No behavior change

R2 changes no Rust production code, cadence, routing, workspace, memory,
network transport, learning, action, or authority behavior.

```text
inventory PASS
!= metareasoner implemented
!= legacy CognitiveDepth invalid
!= separated scheduler better
!= transport policy better
!= System 1/System 2 theory validated
```
