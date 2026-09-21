# Melothaea Epistemic Kernel Roadmap

Tracker: #5275

This roadmap intentionally leaves domain-native evidence as the source of authority. The shared kernel normalizes only cross-cutting identity, provenance, preservation, and claim-scope concepts.

## 001A — Semantic evidence identity

Define a stable domain-separated digest over a small typed canonical transcript. Do not hash arbitrary JSON. Do not let generic transcript construction mint source authority.

## 001B — Source reference and semantic preservation

Introduce exact source namespace/schema/profile/record/digest references plus:

- `Exact`
- `Projected { losses }`

Projection losses remain namespaced, sorted, unique, and source-defined.

## 001C — Positive closed-world claim scope

Every projection/receipt may grant only explicitly enumerated namespaced claims.

```text
claim in establishes -> potentially usable
claim absent          -> not established
```

Explicit nonclaims remain required where useful for auditability and regression protection, but they are not an allowlist inverse.

## 001D — Typed provenance DAG

Bind exact semantic IDs through typed relations such as:

- `DerivedFrom`
- `GeneratedBy`
- `ObservedDuring`
- `AssignedUnder`
- `AdmittedUnder`
- `ExcludedUnder`
- `AnalyzedBy`
- `ReportedFrom`

Reject cycles, unknown references, contradictory duplicate edges, and relation/type mismatches.

## 001E — Qualification receipt projection

Project existing exact-head qualification receipts into the shared kernel without changing their native workflow or subject authority. This is an interoperability view only.

## Successors

- MEL-ATT-001: detached evidence attestation
- MEL-REG-001 / #5276: externally anchored preregistration receipts
- MEL-ANA-001 / #5277: immutable analysis-input manifests and unblinding order
- MEL-INTEROP-001: W3C PROV and RO-Crate export adapters

## Non-goals

The shared kernel must not become a universal `valid` flag, universal evidence enum, generic scientific truth oracle, or replacement for source-specific validators.
