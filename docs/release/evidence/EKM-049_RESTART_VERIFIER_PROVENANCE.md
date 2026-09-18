# EKM-049 — Restart Verifier Provenance

## Purpose

EKM-048 binds a validated restart candidate to trust policy, anchor proof, continuity state, and a staged admission-review receipt. It still treats the external verifier as an abstract boolean provider.

EKM-049 makes that verifier state auditable without granting any restore authority.

The central invariant is:

**"proof accepted" must be bound to the verifier profile and trust snapshot that were stable across the verification call.**

## Added contract

`RestartVerifierProfileV1` binds:

- canonical verifier profile ID
- implementation ID
- implementation version
- trust-snapshot digest
- positive trust-snapshot sequence
- trust-snapshot validity window
- verifier configuration digest

`ProfiledRestartAnchorEvidenceVerifierV1` extends the EKM-046 proof-verifier boundary with a read-only profile snapshot.

`RestartVerifierProvenanceReceiptV1::validate_and_review`:

1. captures and validates the verifier profile before review;
2. rejects a trust snapshot that is not fresh at the observed cycle;
3. executes the complete EKM-048 staged admission review;
4. captures and validates the verifier profile again;
5. fails closed if any profile field changed across verification;
6. binds the exact EKM-048 review digest to the canonical verifier-profile digest;
7. emits a separate provenance digest;
8. hard-codes quarantine and activation authority to false.

## Why profile-before/profile-after matters

A verifier may rotate software, configuration, or trust state. Without a stability check, an audit artifact could record one profile while the actual proof decision was produced under another.

EKM-049 requires exact equality across the verification call. This catches observable profile substitution during the call.

It does **not** prove that a malicious verifier truthfully reports its profile or that its trust-snapshot digest corresponds to a legitimate external trust database. Those remain deployment/provider responsibilities.

## Digest boundary

The verifier profile uses explicit canonical field hashing rather than `Debug` output or Rust enum discriminants.

The provenance receipt binds:

- exact EKM-048 admission-review digest
- exact verifier-profile digest
- observed cycle
- profile-stability result
- explicit non-authority flags

Because the EKM-048 review digest already binds the candidate validation receipt, local trust policy, anchor statement, proof digest, prior trust anchor and continuity outcome, the provenance receipt transitively binds verifier identity to the exact candidate review rather than to a free-floating proof verdict.

## Authority boundary

EKM-049 does **not**:

- construct quarantine state
- hydrate writable epistemic state
- activate a restart
- advance the real trusted-anchor tracker
- persist trust state
- manage keys or signature algorithms
- perform file/network I/O
- mutate evidence, belief, causal, world-model or action state

The receipt reports:

`quarantine_construction_authorized = false`

`activation_authorized = false`

## Remaining trust-continuity gap

The profile records a trust-snapshot sequence, but EKM-049 does not yet compare that sequence/digest against the previously accepted verifier trust state. A future continuity tranche should reject verifier trust-snapshot rollback, same-sequence substitution, and unauthorized verifier-profile replacement before quarantine construction is considered.

## Qualification status

This tranche is stacked on EKM-048. The forced exact-head EKM-046 Format Check and the EKM-048 full CI run remain queued at authoring time. Queued or cancelled CI is not compile, format, Clippy, test or runtime evidence.
