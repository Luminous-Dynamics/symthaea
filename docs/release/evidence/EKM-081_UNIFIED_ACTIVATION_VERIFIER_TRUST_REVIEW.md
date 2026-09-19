# EKM-081 — Unified Activation Verifier Trust Review

## Purpose

EKM-080 proves that one stable verifier profile accepted all nine EKM-072 activation-phase evidence claims, but deliberately does not establish that the verifier is allowed by local deployment policy or that its trust snapshot has not rolled back.

EKM-081 adds a separate, non-authorizing trust-review layer.

The central invariant is:

**phase evidence acceptance, local verifier policy, monotonic verifier trust continuity, independent verifier trust, and global current-head proof remain separate claims.**

## Canonical verifier comparison profile

`CanonicalVerifierTrustProfileV1` normalizes both:

- EKM-049 `RestartVerifierProfileV1`
- EKM-080 `ActivationEvidenceVerifierProfileV1`

into the same comparison fields:

- verifier ID
- implementation ID
- implementation version
- trust-snapshot digest
- trust-snapshot sequence
- configuration digest
- validity window

This normalization does not alter either prior schema.

The common view is intentionally stricter than historical EKM-049: zero trust-snapshot and configuration digests are rejected during normalization.

This creates one comparison vocabulary without retroactively changing the already-authored restart-verifier contract.

## Local trust policy

`ActivationVerifierTrustPolicyV1` binds:

- exact allowed verifier ID
- exact allowed implementation ID
- exact allowed implementation version
- exact allowed configuration digest
- minimum acceptable trust-snapshot sequence
- maximum permitted profile validity span

The policy is canonicalized under its own domain-separated digest.

A cryptographically valid or runtime-accepted verifier profile therefore cannot bypass local deployment policy.

## Caller-held trust checkpoint

`CallerTrustedActivationVerifierCheckpointV1` records:

- one canonical verifier profile
- the cycle at which the caller accepted that profile as trusted
- a non-zero external trust-anchor identity
- a canonical checkpoint digest

This is explicitly a **caller trust assertion**. EKM-081 does not claim that the external trust-anchor digest is independently verified merely because it is present.

The checkpoint profile must have been fresh at its trusted cycle.

## Review semantics

`review_activation_evidence_verifier_trust(...)`:

1. re-runs EKM-080 against the exact EKM-079 issuance record and verifier
2. requires the EKM-081 review cycle not to predate EKM-080 validation
3. requires the review cycle not to predate the caller-held checkpoint
4. normalizes the EKM-080 verifier profile
5. requires the candidate profile to remain fresh at the EKM-081 review cycle
6. applies the exact local verifier trust policy
7. verifies the caller checkpoint's canonical digest
8. requires verifier identity, implementation identity/version, and configuration to remain unchanged
9. rejects trust-snapshot sequence rollback
10. rejects same-sequence trust-snapshot substitution
11. distinguishes stable trust reuse from monotonic trust-snapshot advance
12. emits a replay-verifiable trust-review receipt

## Receipt claims

A successful `ActivationVerifierTrustReviewReceiptV1` may state:

- local policy accepted = true
- continuity relative to caller checkpoint proven = true
- verifier profile fresh at review cycle = true
- disposition = stable reuse or trust-snapshot advance

It deliberately continues to state:

- provider trust independently established = false
- global current head independently proven = false
- epoch issuance chain verified = false
- epoch issuance authorized = false
- mutation authority = false
- activation authorized = false

## Why global currentness remains false

A monotonic advance relative to a caller-held checkpoint proves only:

**candidate trust state >= the caller's trusted reference state**

It does not prove that no newer trust state exists elsewhere.

Likewise, a non-zero external trust-anchor digest identifies the caller's trust source but is not itself proof that the trust source is genuine, uncompromised, or globally current.

A future tranche must bind this review to independently verifiable current-head evidence before the verifier can participate in a verified epoch-issuance chain.

## Complexity reduction

EKM-081 is also a consolidation step.

Rather than create a third incompatible verifier identity vocabulary, it provides a strict canonical comparison profile shared across EKM-049 and EKM-080 semantics. This lets future trust/currentness logic converge on one identity model without rewriting the historical restart stack.

## Authority boundary

EKM-081 does **not**:

- change EKM-049 semantics
- change EKM-080 phase verification
- establish that a verifier is honest
- prove a global verifier trust head
- mutate a trusted verifier checkpoint
- issue an authority epoch
- construct operational V2 receipts
- append operational revision history
- integrate with `BeliefMutationAuthority`
- export mutation authority
- authorize activation

## Qualification status

This tranche is stacked on EKM-080 / PR #4135.

EKM-080 exact-head CI #7611 remained queued when EKM-081 was prepared.

GitHub Actions remains the executable qualification authority. Static review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, trust-root, hardware-attestation, or deployment evidence.

## Next boundary

The next safe step is an **independent verifier current-head attestation** over the exact EKM-081 canonical verifier profile and caller checkpoint. Only after that should EKM-080/081 evidence be joined to the matching EKM-078 receipt segment and EKM-079 epoch record.

Even after that join, epoch issuance verification must remain separate from operational mutation-facade epoch enforcement.
