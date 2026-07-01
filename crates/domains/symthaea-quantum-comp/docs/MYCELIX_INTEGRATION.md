# Mycelix Integration Notes

Alpha.6 introduces `ResearchArtifactReceipt` as a local receipt shape for future Mycelix integration.

It is not a cryptographic signature. It is not a source-chain entry. It is not a security commitment.

It records:

- artifact name
- crate version
- claim boundary
- manifest fingerprint
- report fingerprint
- environment fingerprint
- combined receipt fingerprint
- optional operator label
- caveat string

## Intended future mapping

A later Mycelix-backed version should replace or wrap the local receipt with:

- cryptographic artifact digests
- agent/source-chain signature
- toolchain receipt
- review/witness signature
- claim-boundary attestation
- immutable experiment manifest
- reproducible run bundle hash

## Design rule

The research crate may produce local receipts. Mycelix should eventually decide what counts as signed civic or scientific provenance.
