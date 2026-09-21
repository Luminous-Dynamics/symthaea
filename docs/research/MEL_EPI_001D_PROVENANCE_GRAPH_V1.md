# MEL-EPI-001D — Typed Provenance Graph V1

Tracker: #5286
Parent architecture: #5275
Source parent: MEL-EPI-001G draft #5365 / `62d021bfc055fafcdd067c697df7d39a05c50bd5`

Status: source-staged only. This tranche defines generic provenance shape and validation. It establishes no executable qualification, source-native admission, or claim propagation.

## Purpose

MEL-EPI needs exact lineage and context without turning graph structure into authority.

The governing distinction is:

```text
provenance recorded
!= provenance true
!= source artifact valid
!= producer authenticated
!= claim transferred
!= causal effect established
```

001D therefore models provenance as a **typed graph with an acyclic derivation subgraph**, not as one universal DAG.

## Exact node identity

A critical V1 rule is that provenance does **not** identify every node only by `EvidenceSemanticIdV1`.

001B already establishes:

```text
source-native commitment
!= MEL-EPI semantic identity
```

Two different source-native artifacts may legitimately project to the same semantic identity while retaining different native commitments or preservation ceilings.

If provenance collapsed both to the semantic ID, exact source lineage would be lost.

Therefore V1 node endpoints use:

```text
ProvenanceNodeReferenceV1 {
  SemanticId(EvidenceSemanticIdV1)
  SourceRef(EvidenceSourceRefV1)
}
```

For `SourceRef`, the complete source reference—including semantic ID, source-native commitment theorem/digest, and preservation ceiling—is part of provenance endpoint identity.

Core rule:

```text
same semantic identity
+ different source-native commitment
-> distinct provenance nodes
```

A regression test freezes this behavior.

## Node classes

V1 freezes the following narrow classes:

```text
EvidenceArtifact
Activity
ProfileOrPolicy
DecisionOrReceipt
AgentOrAuthorityRef
```

There are no display-label node identities.

Semantic-only activity/policy/agent nodes may use `SemanticId`. Source-backed artifacts/receipts may retain their complete `SourceRef`.

## Typestate-aware construction

A source-backed node may be built directly from:

```text
ValidatedEvidenceSourceRefV1
```

through `ProvenanceNodeV1::from_validated_source(...)`.

This helps preserve the stronger in-process structural-validation boundary from 001G.

However the serialized graph stores the ordinary source-ref DTO because a serialized graph is not itself source admission.

```text
node constructed from validated source ref
!= serialized graph becomes source-admitted
```

## Derivation relations

V1 derivation relations are:

```text
DerivedFrom
ProjectedFrom
AnalyzedFrom
ReportedFrom
AggregatedFrom
```

Stored direction is:

```text
child/output --relation--> parent/input
```

Only derivation edges participate in the acyclic partial order.

The validated graph exposes deterministic parent/input-before-child/output ordering induced solely by these edges.

## Association/context relations

V1 separately models:

```text
GeneratedBy
ObservedDuring
AssignedUnder
AdmittedUnder
ExcludedUnder
AttestedBy
ExecutedBy
GovernedBy
```

These are directional and type-checked, but they do **not** participate in derivation-cycle detection or claim inheritance.

Therefore:

```text
association path exists
!= derivation path exists
```

## Endpoint compatibility

V1 freezes a conservative endpoint type table.

Examples:

```text
artifact/receipt --GeneratedBy--> Activity
EvidenceArtifact --ObservedDuring--> Activity
EvidenceArtifact --AssignedUnder--> ProfileOrPolicy
artifact/receipt --AdmittedUnder--> ProfileOrPolicy
EvidenceArtifact --ExcludedUnder--> ProfileOrPolicy
artifact/receipt --AttestedBy--> AgentOrAuthorityRef
Activity --ExecutedBy--> AgentOrAuthorityRef
Activity --GovernedBy--> ProfileOrPolicy
```

All derivation edges require artifact-like endpoints:

```text
EvidenceArtifact | DecisionOrReceipt
```

Unsupported relation/class pairings fail closed.

## Canonical node ordering

Nodes must already be strictly increasing and unique by the canonical bytes of the **complete node reference**.

Reference canonical bytes bind:

```text
reference kind
+ complete semantic identity
```

or, for source-backed nodes:

```text
reference kind
+ complete canonical EvidenceSourceRefV1 payload
```

This avoids relying on display strings, array indices, or semantic-ID-only aliases.

The implementation rejects duplicate or noncanonical reference ordering rather than silently sorting caller data.

## Canonical edge ordering

Edges carry exact `ProvenanceNodeReferenceV1` values at both endpoints.

The frozen V1 sort key is:

```text
relation-domain-rank
relation-rank
canonical from-reference bytes
canonical to-reference bytes
```

Relation ranks are explicit V1 protocol values rather than incidental enum/JSON ordering.

The graph rejects:

- endpoint references absent from the exact node set;
- duplicate/noncanonical nodes;
- duplicate/noncanonical edges;
- self-edges;
- invalid relation/class combinations;
- derivation cycles.

## Deterministic semantic payload

`ProvenanceGraphV1::semantic_payload()` binds the complete validated graph through the existing 001A typed transcript:

```text
graph version
ordered node payloads
ordered edge payloads
```

Node payloads bind:

```text
node class
complete canonical node-reference bytes
```

Edge payloads bind:

```text
relation domain
relation code
complete from-reference bytes
complete to-reference bytes
```

No arbitrary Serde JSON is hashed as authority identity.

## Validated graph wrapper

001D adds:

```text
ValidatedProvenanceGraphV1
```

Construction:

```text
ProvenanceGraphV1
  ↓ TryFrom / full graph validation
ValidatedProvenanceGraphV1
```

The validated wrapper:

- owns raw graph state privately;
- caches canonical graph bytes;
- caches deterministic derivation topological order;
- exposes read-only graph access;
- requires explicit `into_raw()` demotion before mutation;
- does not implement unconstrained `Deserialize`;
- carries no source-admission or claim-propagation semantics.

The cached topological order contains exact node references, not semantic IDs alone.

## No claim propagation

001D intentionally contains **no claim sets and no ancestor-claim API**.

There must never be a generic rule like:

```text
claims(node) = union(claims(all ancestors))
```

That would launder authority across transformations.

Claim propagation remains the separate fail-closed boundary tracked by #5355 / MEL-EPI-001F:

```text
no registered claim-derivation theorem
-> no derived positive claim
```

## Topological-order semantics

The graph stores derivation as:

```text
child -> parent
```

For execution/reproduction planning, the validated wrapper derives the reverse dependency order:

```text
parent/input before child/output
```

Association/context edges do not alter this order.

Ties are resolved using the already-canonical node-vector order, which is based on complete node-reference bytes.

This ordering proves only graph structure under V1. It does not establish chronological execution, trusted time, protocol validity, or causal ordering.

## Source tests written

The implementation includes source tests for:

1. canonical graph validation and payload freezing;
2. construction of source-backed nodes from `ValidatedEvidenceSourceRefV1`;
3. preservation of distinct source-native nodes sharing one semantic identity;
4. noncanonical/duplicate full-reference node ordering rejection;
5. missing exact endpoint rejection;
6. invalid relation/node-class rejection;
7. derivation-cycle rejection;
8. deterministic parent-before-child derivation ordering;
9. semantic payload change when a derivation relation changes;
10. JSON round-trip returning raw graph followed by explicit validation.

## Interoperability boundary

The internal V1 representation remains narrow Rust types.

Later exporters may project this graph to W3C PROV / RO-Crate, but external graph formats do not automatically round-trip back into stronger native authority.

An exporter must preserve the distinction between semantic-only and source-backed node identity or explicitly record that it lost it.

```text
PROV/RO-Crate import
!= source-native validation
!= LosslessUnderProfile
!= claim admission
```

Any lossy external mapping must carry an explicit preservation profile.

## Relationship to MEL-EPI-MUSE-001

The first Muse source adapter (#5351) can eventually use provenance to connect exact objects such as:

```text
confirmatory protocol
cohort registry
final monitor snapshot
participant evidence
collection-close receipt
```

while its claim-scope profile separately distinguishes:

- rederived facts;
- policy-admitted assertions;
- signoff presence;
- authenticated authority once independently established.

The provenance graph itself does not collapse those categories.

## Explicit nonclaims

001D does not establish:

- source-native artifact validity;
- producer/signer identity;
- cryptographic authentication;
- signer authorization;
- trusted chronology;
- preregistration authenticity;
- consent/export authority;
- protocol adherence;
- randomization quality;
- statistical significance;
- causal validity;
- population generalization;
- listener preference;
- artistic or musical quality;
- product authority.

It establishes only a typed, deterministic, fail-closed representation of generic provenance graph structure once exact executable qualification eventually confirms the implementation subject.
