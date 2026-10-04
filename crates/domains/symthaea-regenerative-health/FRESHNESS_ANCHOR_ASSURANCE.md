# Freshness Anchor Assurance Contract

## Purpose

This crate separates receiver freshness recovery from the security properties required to make recovered state authoritative after local rollback, snapshot restore, or replica fork.

> A valid hash, signature, or monotonic field inside rollbackable storage does not by itself establish anti-rollback.

An authoritative freshness anchor therefore requires a security property that lives outside the rollbackable recovery record.

## Property model

| Property | Meaning | Sufficient by itself? |
|---|---|---|
| Integrity protection | Detect mutation of the protected anchor value | No |
| Authentication | Identify or authenticate the producer of the value | No |
| Monotonicity | Accepted anchor generations cannot decrease | No |
| Rollback resistance | Restoration of an older valid anchor is detectable/prevented | Required |
| Atomic update | Competing updates cannot silently split the accepted head | Required |
| Crash persistence | Accepted state survives the crash boundary | Required |

The implementation additionally requires integrity protection, authentication, and monotonicity because an anti-rollback claim without those surrounding properties is not sufficient to safely identify the authoritative state.

## Assurance levels

FreshnessAnchorAssurance deliberately distinguishes:

- Untrusted: no meaningful integrity/authentication basis;
- IntegrityOnly: corruption/tamper detection without authority;
- Authenticated: origin authentication without monotonic anti-rollback;
- MonotonicAuthenticated: authenticated monotonic state that may still be restorable;
- RollbackResistant: all authoritative properties satisfied.

Only RollbackResistant may authorize the authoritative recovery commit path.

Backing labels such as HardwareProtected, RemoteAuthority, and ReplicatedQuorum are provenance categories, not security guarantees. The capability set must still be verified.

## Evidence-to-capability boundary

A FreshnessAnchorProfile is an exact commitment to:

- schema version;
- backing category;
- provenance;
- all six security capabilities.

A FreshnessAnchorVerificationReceipt additionally binds:

- the exact profile fingerprint;
- receiver identity;
- recovery generation;
- recovery-state fingerprint;
- verifier identity;
- evidence reference; and
- evidence digest.

VerifiedFreshnessAnchor is deliberately non-serializable and can only be minted after a verifier accepts that exact profile/receipt pair.

The authoritative recovery path then compares the verified profile and verified subject against the runtime store and recovery record.

This prevents capability-flag substitution, detached evidence reuse, cross-receiver proof reuse, cross-generation proof reuse, and cross-state proof reuse.

## RATS correspondence

The design follows the RATS separation between Evidence, Verifier appraisal, and Relying Party authorization.

The deployment-specific verifier is responsible for converting hardware, remote, or quorum evidence into an accepted verification receipt. This crate does not pretend that a boolean or a caller-supplied string is equivalent to real attestation evidence.

Epoch freshness remains a separate receiver state machine. An unavailable or stale epoch Handle requires resynchronization rather than local timestamp inference.

## TPM-oriented deployment

A concrete TPM backend may use protected NV state such as an NV Counter as the rollback-resistant primitive.

The TCG TPM 2.0 architecture specifies that an NV Counter is modified through increment semantics, cannot move backward when read, and cannot be rolled back by deleting and recreating the counter at a lower lifetime value.

A real adapter must still independently verify:

- the exact NV index identity/name;
- counter type and relevant NV attributes;
- authorization policy;
- TPM identity/attestation binding;
- the observed counter value;
- persistence semantics required by the deployment; and
- the relationship between the counter value and the freshness recovery generation.

Merely reporting TPM present or NV storage available is insufficient.

## Remote and quorum deployment

A remote authority may provide the external monotonic state; a replicated quorum may provide it through a protocol whose accepted head cannot be rewritten by a single receiver.

For either model, the verifier must establish that the authority is outside the receiver's rollback domain and that the accepted head transition has the required atomic or consensus semantics.

A local cache of remote state is still rollbackable and therefore cannot substitute for the authority.

## Crash ordering

The recovery implementation uses the following ordering:

1. persist the rollbackable recovery record;
2. atomically advance the external anchor;
3. only then treat the new generation as authoritatively committed.

If the record exists but the anchor did not advance, retry or quarantine.

If the anchor advanced but the record is missing, quarantine.

The receiver must never synthesize missing state from an anchor alone.

## Conflict preservation

An ordinary recovery operation may restore a conflict but may not clear it.

Conflict replacement belongs to the explicit authenticated resynchronization boundary implemented by the adjacent freshness resynchronization module.

This keeps contradictory evidence visible across reboot and replica recovery.

## Claim ceiling

Even a verified RollbackResistant anchor does not establish:

- trusted wall-clock time;
- continuous runtime integrity;
- firmware, kernel, or TPM correctness beyond the verifier's evidence policy;
- readiness;
- physical authority;
- absence of compromise between attestation events.

Those are separate assurance claims and require separate evidence.

## Current implementation boundary

Implemented:

- exact capability model;
- fail-closed software-only default;
- domain-separated deterministic profile commitment;
- evidence-bound verification receipt;
- opaque verified-anchor capability;
- exact receiver, generation, and state binding;
- authoritative commit gate;
- deterministic regression coverage for the above.

Not yet implemented:

- concrete TPM/NV adapter;
- concrete HSM adapter;
- remote-authority protocol;
- quorum/consensus adapter;
- production evidence parser/attestation verifier.

This separation is intentional: platform-specific evidence verification must remain an independently reviewable trust boundary.