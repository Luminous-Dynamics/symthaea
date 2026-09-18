# EKM-053 — Sealed Quarantine Authority Facade

## Purpose

EKM-051 and EKM-052 correctly keep restart state read-only, but their raw implementation objects retain a cloned `EpistemicLedger` and the complete decoded restart snapshot for self-verification. If those raw objects remain public, callers can clone descriptive internals even though no activation authority exists.

EKM-053 closes the **external-byte restart admission** capability surface without changing the older trusted in-memory capsule APIs.

The central invariant is:

**externally supplied restart bytes may be inspected after admission, but public callers must not receive the raw reconstructed ledger, decoded wire snapshot, inner quarantine object, or joint trusted-context object.**

## Visibility closure

On the EKM-053 branch:

- `epistemic_restart_quarantine_admission` becomes `pub(crate)`;
- `epistemic_restart_trusted_context` becomes `pub(crate)`;
- the new `epistemic_restart_quarantine_facade` remains public.

This keeps the rich EKM-051/052 types available for internal composition and verification while removing them from the external API surface.

## Public trust-context handle

`RestartTrustContextHandleV1` wraps the internal EKM-052 joint context and exposes only:

- deployment ID;
- trust-domain ID;
- context commit cycle;
- trusted anchor sequence/capture cycle;
- trusted verifier trust-snapshot sequence;
- stable context digest;
- read-only integrity verification.

It does not expose the internal anchor tracker, verifier checkpoint, or joint context.

## Public quarantine handle

`ReadOnlyEpistemicRestartQuarantineV2` deliberately does **not** implement `Clone` and exposes only:

- capture cycle;
- claim/evidence/provenance counts;
- immutable record-level lookup by stable ID;
- source outer checksum;
- source V2 digest;
- trusted-context digest;
- quarantine digest;
- read-only self-verification.

It does not expose:

- `&EpistemicLedger`;
- `EpistemicLedger` ownership;
- the decoded `EpistemicRestartWireSnapshotV2`;
- the inner EKM-051 quarantine;
- the inner EKM-052 contextualized quarantine;
- the inner joint trusted context;
- mutable support/history state;
- an activation or hydration capability.

Individual `KnowledgeClaim`, `EvidenceRecord`, and `ProvenanceRecord` values remain descriptive data and may be cloned by callers. Cloning a record does not create a mutable ledger or a restore capability.

## Error boundary

The public facade exposes stable high-level error categories with diagnostic strings rather than placing crate-private EKM-051/052 error types in public signatures.

## Scope boundary

EKM-053 seals the **external-byte admission path** introduced by EKM-051/052.

It does not change the older `EpistemicRestartCapsuleV1/V2 -> QuarantinedEpistemicRestartV1/V2` in-memory APIs. Those APIs begin from already-typed locally captured capsule objects and remain a separate compatibility surface. They still expose immutable ledger views and remain non-activating.

A future major-version cleanup may choose to place all quarantine representations behind one facade, but EKM-053 does not make that unrelated breaking API change inside the unqualified restart stack.

## Authority boundary

The facade hard-codes:

- `writable_hydration_authorized = false`
- `activation_authorized = false`

and adds no API for:

- writable support-store hydration;
- writable revision-history reconstruction;
- live-state swap;
- anchor/verifier/context checkpoint advancement;
- evidence/belief/causal/world-model/action mutation;
- file/network I/O;
- signing or key custody.

## Remaining trust boundary

`RestartTrustContextHandleV1::capture` still begins from caller-supplied in-memory trusted anchor/verifier state. EKM-053 does not prove that those inputs came from rollback-resistant or independently attested storage.

Binding/protecting the joint context digest through protected storage, hardware attestation, signed/witnessed checkpoints, or transparency infrastructure remains a separate deployment/qualification tranche.

## Qualification status

This tranche is stacked on EKM-052. GitHub Actions remains the executable authority; queued or cancelled jobs are not format, compile, Clippy, test, or runtime evidence.
