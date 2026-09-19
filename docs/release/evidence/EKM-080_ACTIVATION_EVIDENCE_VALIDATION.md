# EKM-080 — Independent Activation Evidence Validation

## Purpose

EKM-079 binds one non-zero evidence digest to each of the nine canonical EKM-072 activation phases, but deliberately does not prove that those digests correspond to genuine runtime or external evidence. EKM-080 introduces a phase-specific verification-provider boundary over that passive record.

The central invariant is:

**every EKM-072 phase must be accepted against the complete EKM-079 activation context by one stable verifier profile, while provider acceptance remains separate from provider trust, epoch issuance, and mutation authority.**

## Phase-specific provider contract

`ActivationExecutionEvidenceVerifierV1` exposes nine named checks rather than one generic `verify(hash)` entry point:

1. exclusive live-epoch guard acquired
2. activation review reverified under guard
3. expected live generation compared
4. rollback bundle captured
5. complete candidate bundle staged
6. atomic live-state swap executed
7. installed state verified under guard
8. trusted checkpoints committed
9. live-epoch guard released

Each method receives the complete immutable `ActivationEvidenceContextV1` plus the exact evidence digest claimed by the EKM-079 record.

The context binds:

- deployment ID
- trust-domain ID
- epoch sequence and predecessor epoch identity
- activated restart V2 digest
- EKM-071 activation-review receipt digest
- canonical EKM-072 transaction-contract digest
- expected and committed live generations
- rollback bundle digest
- candidate bundle digest
- installed-state digest
- activation cycle
- activation-commit receipt digest
- authority-epoch digest
- EKM-079 issuance-record digest

This prevents the validator from accepting a phase result detached from the activation record it is meant to support.

## Verifier provenance

The verifier supplies `ActivationEvidenceVerifierProfileV1`, binding:

- verifier ID
- implementation ID
- implementation version
- trust-snapshot digest
- monotonic trust-snapshot sequence
- configuration digest
- validity window

The profile is sampled before and after all nine phase checks and must remain exactly identical. Validation fails if the verifier profile changes during the call.

The profile must also be valid at the EKM-080 verification cycle.

## Verified receipt

A successful validation emits `VerifiedActivationEvidenceReceiptV1` containing:

- exact EKM-079 record digest
- activation-commit receipt digest
- authority-epoch digest
- verifier profile and canonical profile digest
- one phase-acceptance record for every canonical EKM-072 phase
- one domain-separated acceptance digest binding:
  - issuance-record digest
  - verifier-profile digest
  - phase identity
  - claimed phase-evidence digest
  - verification cycle
- a canonical overall receipt digest

The receipt is replay-verifiable through `verify_against(...)`, which re-runs all nine provider checks and requires exact receipt equality.

## Claim boundary

A successful EKM-080 receipt may state:

- `all_phase_evidence_provider_accepted = true`
- `verifier_profile_stable_during_validation = true`

It deliberately continues to state:

- `provider_trust_independently_established = false`
- `epoch_issuance_chain_verified = false`
- `epoch_issuance_authorized = false`
- `mutation_authority = false`
- `activation_authorized = false`

The reason is important: EKM-080 verifies that the configured provider accepted the exact phase evidence. It does not itself establish that the provider is honest, correctly implemented, independently trusted, or backed by genuine runtime/hardware/transparency evidence.

## Failure semantics

EKM-080 fails closed when:

- EKM-079 passive record verification fails
- verification predates the claimed activation cycle
- verifier profile is malformed
- trust snapshot or configuration digest is zero
- verifier profile is outside its validity window
- the provider returns an error
- any one of the nine phase checks rejects
- the verifier profile changes during validation
- replay verification does not reproduce the same receipt

## Authority boundary

EKM-080 does **not**:

- implement an activation executor
- acquire a live-state guard
- perform compare-and-swap
- capture rollback state
- swap live state
- commit trusted checkpoints
- establish provider trust
- issue an authority epoch
- construct operational V2 receipts
- append to operational revision history
- integrate with `BeliefMutationAuthority`
- export mutation authority
- authorize activation

## Qualification status

This tranche is stacked on EKM-079 / PR #4119. EKM-079 exact-head CI #7580 remained queued when EKM-080 was prepared.

GitHub Actions remains the executable qualification authority. Static/API review and authored unit tests are not rustfmt, compilation, Clippy, integration-test, runtime, hardware-attestation, or deployment evidence.

## Next boundary

The next safe tranche should establish an explicit trust-policy/currentness layer for activation-evidence verifier profiles, then bind EKM-080-verified issuance records to the matching EKM-078 epoch segments. Only after verifier trust and exact segment/issuance correspondence are both proven should `epoch_issuance_chain_verified` be eligible to become true. Mutation-facade authority remains a later boundary.
