# Melothaea Epistemic Kernel Roadmap

Tracker: #5275

This roadmap intentionally leaves domain-native evidence as the source of authority. The shared kernel normalizes only cross-cutting semantic identity, source references, preservation, claim scope, and provenance concepts.

## 001A — Semantic evidence identity

Freeze a stable domain-separated SHA-256 preimage over a small typed canonical transcript. Do not hash arbitrary JSON. Do not let generic transcript construction mint source authority.

V1 field kinds are algorithm/semantic-specific. In particular, field kind `0x08` means **SHA-256 digest reference**, not an arbitrary 32-byte digest. The protocol also freezes one complete cross-implementation byte vector.

## 001B — Source reference and semantic preservation

Bind an independently recomputed native source commitment to one exact MEL-EPI semantic identity without duplicating namespace/schema/profile/record fields beside that ID.

Preferred conceptual shape:

```text
SourceNativeCommitmentV1 {
  scheme_id,
  sha256_hex,
}

EvidenceSourceRefV1 {
  semantic_id,
  native_commitment,
  preservation,
}
```

Preservation vocabulary:

- `LosslessUnderProfile`
- `ProjectedWithLoss { loss_codes }`

`LosslessUnderProfile` means no known semantic loss under the frozen projection profile; it does **not** mean byte-identical serialization or universal semantic completeness.

## 001C — Positive closed-world claim scope

Every admitted source reference may grant only explicitly enumerated canonical namespaced claims under one exact scope profile.

```text
claim in establishes -> potentially usable under that admitted source/profile
claim absent          -> NOT ESTABLISHED
```

Explicit nonclaims remain useful for auditability and regression protection, but they are not an allowlist inverse. Legacy E/N/M ordinal labels cannot mint claims automatically.

## 001D — Typed provenance graph

Represent provenance with typed nodes and relations, while separating:

- **derivation relations**, whose induced subgraph must be acyclic;
- **association/context relations**, which are directional and type-checked but need not participate in the derivation DAG.

Examples of derivation relations include `DerivedFrom`, `ProjectedFrom`, `AnalyzedFrom`, `ReportedFrom`, and `AggregatedFrom`.

Examples of association/context relations include `ObservedDuring`, `AssignedUnder`, `AdmittedUnder`, `ExcludedUnder`, `AttestedBy`, `ExecutedBy`, and `GovernedBy`.

Reject missing references, invalid relation/node-class combinations, identity substitution, and derivation cycles. A shape-valid provenance graph still does not validate its native source artifacts.

## 001E — Qualification receipt projection

Project existing exact-head qualification receipts into the shared kernel without changing their native workflow or subject authority. Source-specific adapters must run the native closed-world verifier first and independently recompute both native and MEL-EPI commitments.

The first target remains the Melothaea tonal-chain qualification receipt frozen by #5229.

## Native Muse registration/lifecycle authority

Do **not** build a second preregistration or analysis lifecycle beside Muse.

Muse already owns source-native authority for methodology freezing, external preregistration receipt binding, confirmatory amendment/refreeze control, irreversible collection close, controlled unblinding, analysis execution, replication protocol, and hash-chained study orchestration.

MEL-REG-001 (#5309) is therefore an adapter/admission program that projects those native authorities into MEL-EPI source references and bounded claim scopes, not a replacement registry/state machine.

MEL-UNBLIND-001 (#5310) separately hardens the remaining authentication/chronology boundary for unblinding approvals using Xenia/formal-safety attestation where appropriate.

## Attestation

MEL-ATT-001 (#5288) reuses the canonical formal-safety/Xenia receipt-attestation architecture rather than inventing a Melothaea-specific signature format.

```text
attestation present
!= signature valid
!= signer enrolled
!= signer authorized
!= evidence admitted
!= claim qualified
```

## Interoperability

Later adapters may project typed provenance to W3C PROV and RO-Crate. External interchange formats are one-way interoperability views unless an explicit preservation profile proves what survives projection. Lossy external representations cannot round-trip back into stronger native authority by shape alone.

## Non-goals

The shared kernel must not become:

- a universal `valid` flag;
- a universal evidence enum;
- a replacement for source-specific validators;
- a second preregistration/analysis state machine;
- a generic cryptographic signer;
- a scientific truth oracle;
- an automatic bridge from statistical strength to cryptographic/reproducibility claims.
