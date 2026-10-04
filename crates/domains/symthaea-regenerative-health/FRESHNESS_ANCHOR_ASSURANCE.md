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
- recovery policy fingerprint;
- recovery authority reference;
- recovery authority statement digest;
- recovery authentication binding;
- verifier identity;
- verifier appraisal-policy digest;
- reference-values digest;
- evidence reference;
- evidence digest; and
- a typed evidence mechanism.

The typed evidence mechanism records one of:

- a hardware monotonic counter with backend and counter-namespace identities;
- a remote monotonic sequence with authority and namespace identities; or
- a quorum monotonic sequence with quorum policy, member-set, threshold, sequence, and certificate identities.

The mechanism type must match the declared backing category before the evidence verifier is invoked. This prevents a proof envelope from silently changing mechanism class.

The mechanism's observed monotonic value must also equal the recovery generation being asserted. The verifier cannot therefore attest generation N while presenting counter/sequence N-1 as the anti-rollback evidence.

The Handle, verifier policy, reference values, and verifier trust-anchor set are also explicit inputs to the evidence appraisal boundary. They are not treated as decorative metadata.

The receipt exposes a domain-separated `binding_digest()` over every security-relevant receipt field, including the typed evidence mechanism. A concrete verifier can bind its signed Attestation Result to this exact digest instead of signing an implicitly reconstructed subset of fields.

A concrete Ed25519 attestation envelope is provided by the adjacent `freshness_anchor_attestation` module. It signs the exact receipt binding digest with a domain-separated message and refuses verification when any signed receipt field changes.

This cryptographic layer authenticates the verifier's statement, not the verifier's authority by itself.

The attestation module therefore also exposes an explicit verifier trust policy. The stronger verification helper requires exact agreement with externally expected verifier identity, verifier-key fingerprint, and trust-anchor-set digest before it mints the opaque VerifiedFreshnessAnchor capability.

Deployment policy remains responsible for establishing those expected trust values, and the verifier implementation remains responsible for hardware/remote/quorum evidence appraisal.

The authoritative recovery commit additionally requires the verified receipt to match the recovery record's policy fingerprint, authority reference, authority statement digest, and authentication binding. A receipt for one recovery authorization context cannot be spliced onto another record that happens to share the same receiver, generation, and state fingerprint.

VerifiedFreshnessAnchor is deliberately non-serializable and can only be minted after a verifier accepts that exact profile/receipt pair.

The authoritative recovery path then compares the verified profile and verified subject against the runtime store and recovery record.

This prevents capability-flag substitution, detached evidence reuse, cross-receiver proof reuse, cross-generation proof reuse, and cross-state proof reuse.

## RATS correspondence

The design follows the RATS separation between Evidence, Verifier appraisal, and Relying Party authorization.

The deployment-specific verifier is responsible for converting hardware, remote, or quorum evidence into an accepted verification receipt. This crate does not pretend that a boolean or a caller-supplied string is equivalent to real attestation evidence.

Epoch freshness remains a separate receiver state machine. An unavailable or stale epoch Handle requires resynchronization rather than local timestamp inference.

## TPM-oriented deployment

A concrete TPM backend may use protected NV state such as an NV Counter as the rollback-resistant primitive.

The adjacent `freshness_anchor_tpm` module provides a deployment-neutral contract for TPM NV-counter evidence. It explicitly binds TPM identity, NV Index Name/public-area, authorization policy, attestation key, quote Handle, quote, PCR binding, and counter value. The structural gate requires the counter to equal the recovery generation, the typed TPM identity/NV-name fields to equal the generic evidence identity fields, the quote Handle to equal the receipt Handle, and `receipt.evidence_digest` to equal the canonical digest of the exact TPM evidence envelope; cryptographic quote verification remains platform-specific. This digest binding is checked before the external verifier, so a valid signature over a different evidence envelope cannot be spliced into the receipt. For a TPM NV counter, the contract also requires a distinct NV certification attestation digest: a PCR Quote alone authenticates selected PCR state, whereas NV certification is what can bind the attested NV Index Name and its contents to the TPM-generated attestation. The platform verifier must verify that NV certification against the same freshness challenge and expected counter object.

The adjacent TPM contract also provides a typed 256-bit quote challenge with canonical digesting and OS randomness. The challenge gate runs before the platform verifier, so an outdated or mis-bound quote cannot reach cryptographic appraisal under the wrong freshness context.

TCG TPM 2.0 defines `TPM_NT_COUNTER` as an 8-octet counter whose value is modified with `TPM2_NV_Increment()`. The deployment must separately establish the required persistence and lifecycle properties of the selected NV index; those properties are not inferred merely from the fact that the object is an NV counter. The TCG structures define `TPMS_NV_DIGEST_CERTIFY_INFO` as carrying the NV Index Name and a hash of the certified NV contents.

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
- domain-separated deterministic profile commitment with explicit field framing;
- evidence-bound verification receipt;
- deterministic, domain-separated verification-receipt statement commitment;
- concrete Ed25519 attestation-result protection over that commitment;
- deployment-neutral TPM NV-counter evidence contract;
- typed, randomized TPM quote challenge and pre-verifier Handle gate;
- explicit verifier key/trust-anchor policy binding;
- explicit recovery-policy and authority binding;
- typed hardware/remote/quorum evidence envelope;
- verifier-policy and reference-value binding;
- opaque verified-anchor capability;
- exact receiver, generation, and state binding;
- authoritative commit gate;
- deterministic regression coverage for the above.

Not yet implemented:

- concrete TPM/NV adapter;
- concrete HSM adapter;
- remote-authority protocol;
- quorum/consensus adapter;
- production platform evidence parser/attestation verifier;
- production TPM quote/NV-public-area verification adapter;
- deployment-specific verifier-key/trust-anchor resolution.

This separation is intentional: platform-specific evidence verification must remain an independently reviewable trust boundary.