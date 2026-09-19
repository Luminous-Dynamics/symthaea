# EKM-082 — Activation Verifier Current-Head Attestation

## Purpose

EKM-081 proves that the EKM-080 activation-evidence verifier satisfies local policy and has not rolled back relative to a caller-held verifier checkpoint. That is not the same as proving the candidate trust snapshot is the current/latest verifier trust head.

EKM-082 adds that question as a separate time-bounded provider boundary.

The central invariant is:

**checkpoint-relative monotonicity does not imply global currentness, and currentness does not imply verifier correctness or mutation authority.**

## Evidence semantics

`ActivationVerifierCurrentnessEvidenceKindV1` deliberately permits only evidence whose provider contract means current monotonic head state:

- monotonic protected state
- hardware monotonic counter
- transparency-log head
- current-head witness quorum

A generic detached signature is intentionally absent. A valid signature can establish authenticity of an old state; it cannot by itself establish that the signed state is still the latest state.

## Statement binding

`ActivationVerifierCurrentnessStatementV1` binds the exact EKM-081 trust review and candidate verifier state:

- EKM-081 review digest
- EKM-080 activation-evidence receipt digest
- EKM-079 issuance-record digest
- canonical verifier-profile digest
- EKM-081 trust-policy digest
- EKM-081 reference-checkpoint digest
- verifier / implementation identity and version
- trust-snapshot digest and sequence
- configuration digest
- verifier-profile expiry cycle
- EKM-081 review cycle
- attestation cycle
- currentness expiry cycle
- currentness authority ID
- evidence kind

The currentness validity window must fit entirely within the verifier profile's own validity window.

## Provider contract

`ActivationVerifierCurrentnessVerifierV1` receives:

- the selected current-head evidence kind
- authority ID
- canonical statement digest
- bounded proof bytes

Returning `true` means only that the provider attests the exact named verifier state is the current monotonic/head state at the statement's attestation cycle under that evidence kind.

## Verified result

`VerifiedActivationVerifierCurrentnessV1` binds:

- canonical statement digest
- proof digest
- verification cycle
- a canonical receipt digest

A valid result may state:

- `global_current_head_independently_proven = true`

It deliberately continues to state:

- `verifier_correctness_independently_proven = false`
- `trusted_state_mutated = false`
- `epoch_issuance_chain_verified = false`
- `epoch_issuance_authorized = false`
- `mutation_authority = false`
- `activation_authorized = false`

Currentness answers **which trust snapshot is latest**, not whether the verifier implementation or trust root is intrinsically correct.

## Time bounds

The currentness attestation:

- may not predate EKM-081 review
- may not occur after the verifier profile has expired
- must expire after attestation
- may not outlive the verifier profile
- must still be unexpired when verification is consumed

This prevents one once-current attestation from becoming permanent currentness authority.

## Authority boundary

EKM-082 does **not**:

- alter EKM-081 trust policy or caller checkpoint
- mutate verifier trust state
- establish verifier correctness
- validate an epoch segment
- verify the epoch-issuance chain
- issue an authority epoch
- create operational V2 receipts
- integrate with `BeliefMutationAuthority`
- export mutation authority
- authorize activation

## Qualification status

This tranche is stacked on EKM-081 / PR #4139.

EKM-081 exact-head CI #7615 remained queued when EKM-082 was prepared.

GitHub Actions remains the executable qualification authority. Static review and authored tests are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, hardware-attestation, transparency-log, witness-quorum, or deployment evidence.

## Next boundary

Once EKM-080 phase evidence, EKM-081 policy/continuity review, and EKM-082 current-head evidence all agree on the exact same verifier profile, the next safe tranche is a **read-only epoch-issuance chain validator**.

That validator should bind the exact EKM-079 epoch record to the corresponding EKM-078 receipt segment and prove predecessor/sequence/cursor continuity. Even then, operational active-epoch installation and mutation-facade enforcement remain separate later boundaries.
