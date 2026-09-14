# symthaea-assurance-tpm2-possession-policy-binding

Canonical whole-policy commitment for the inherited TPM2 attestation-possession policy.

The upstream `AttestationPossessionPolicy` already validates the semantic inputs to fresh quote possession, but its `policy_id` is only a name. This crate adds a domain-separated BLAKE3 commitment over every current policy field so a stable ID cannot conceal changes to platform qualification, AK binding, PCR selection, verifier identity, verification-tool identity, timing bounds, fixed/restricted AK requirements, or evidence references.

Evidence references are sorted before hashing because their order is not treated as semantic. Their multiplicity remains committed.

`canonical_possession_policy_digest` returns `None` for an upstream-invalid policy. It grants no authority and does not replace quote verification, challenge freshness, acceptance-ledger replay protection, TPM trust, IMA replay, runtime continuity, or policy provenance/currentness.

This small crate is intended to be the single policy-commitment authority reused by the Linux TPM+IMA provider composition instead of re-encoding the possession policy there.
