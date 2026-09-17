# EKM-046 — Externally Verified Restart Anchor Evidence

## Purpose

EKM-045 distinguishes valid forward progress from rollback and same-cycle equivocation, but its trust guarantee depends on a caller-supplied checkpoint. EKM-046 makes the provenance of that checkpoint explicit without choosing one deployment-specific security mechanism.

The central invariant is:

**a continuity anchor may enter the epistemic restart path only after an external verification provider accepts proof bound to the exact EKM-044 validation receipt, and the accepted anchors must form a strict monotonic predecessor chain.**

## Added contract

- `RestartAnchorStatementV1`
  - strict positive sequence
  - exact EKM-044 capture cycle and receipt digest
  - exact predecessor-anchor digest
  - issue/expiry cycle
  - bounded canonical authority identifier
  - typed evidence-kind descriptor
- `RestartAnchorEvidenceV1`
  - bounded non-empty opaque proof bytes
- `RestartAnchorEvidenceVerifierV1`
  - deployment/provider integration boundary
  - receives evidence kind, authority ID, canonical statement digest and proof bytes
- `VerifiedRestartAnchorEvidenceV1`
  - constructible only through provider acceptance
  - records statement digest, proof digest and verification cycle
  - converts to the narrow EKM-045 continuity anchor
  - hard-codes quarantine and activation authority to false
- `RestartAnchorTrackerV1`
  - requires genesis sequence 1
  - requires contiguous sequence progression
  - binds every successor to the exact predecessor digest
  - rejects rollback, same-sequence substitution, replay, sequence gaps and predecessor mismatch
  - re-checks the anchor validity window when tracker admission occurs

## Evidence-kind boundary

The following kinds describe the upstream mechanism but are **not** implemented cryptographically in this module:

- protected checkpoint
- signed checkpoint
- witnessed checkpoint
- hardware-attested checkpoint
- transparency checkpoint

The verifier provider remains responsible for the actual signature, TPM/secure-element, transparency-log, witness-quorum or protected-storage checks. This mirrors the existing fabrication checkpoint architecture, where cryptographic providers and trust-policy state are separate from domain logic.

## Trust boundary

EKM-046 does not make an arbitrary verifier trustworthy. A malicious or misconfigured provider can still accept invalid proof.

Likewise, `RestartAnchorTrackerV1` is in-memory state. If its persisted state can itself be rolled back, the deployment can still lose monotonic history.

Production trust can later bind this interface to already-established Symthaea mechanisms such as signed transparency checkpoints, independent witnesses, protected monotonic state, or Xenia/operator authority. That integration remains a separate qualified tranche.

## Authority boundary

EKM-046 does **not**:

- construct an EKM restart quarantine
- hydrate writable belief/support state
- activate restored state
- authorize evidence or belief mutation
- mutate causal/world-model/action state
- implement cryptographic signing or key custody
- persist anchor-tracker state
- perform file or network I/O

All verified-anchor and continuity paths still report:

`quarantine_construction_authorized = false`

`activation_authorized = false`

## Qualification status

This tranche is stacked on EKM-045. GitHub Actions remains the executable authority; queued jobs are not compile, format, Clippy, test or runtime evidence.
