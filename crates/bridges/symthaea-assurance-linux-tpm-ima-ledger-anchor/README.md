# Linux TPM2 + IMA ledger-relative anchor

This bridge composes an opaque `VerifiedTpmImaPcrBinding` with the existing fresh-attestation possession policy and acceptance ledger.

It validates the supplied ledger's internal append-only structure before mutation, binds the exact whole-policy commitment from `symthaea-assurance-tpm2-possession-policy-binding`, proves that the supplied quote/challenge/AK objects are the exact evidence already committed by the PCR-binding capability, and only then performs one-shot acceptance.

The output is intentionally named `LedgerRelativeLinuxTpmImaAnchor`: replay resistance is established relative to the supplied, structurally valid ledger state. This crate does not prove that the caller supplied the latest authoritative ledger head. A later persistent-currentness theorem must provide that upgrade.

The capability grants no physical authority and does not replace runtime hash-link continuity.
