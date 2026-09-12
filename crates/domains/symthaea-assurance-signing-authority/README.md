# symthaea-assurance-signing-authority

Independent trust-root governance for who may sign Symthaea assurance-policy manifests.

The assurance manifest does **not** define its own signing authority. That would be circular: a compromised replacement key could simply sign a manifest declaring itself trusted.

Instead this crate models an externally reviewed signing-authority policy and explicit signer/key transitions. It validates a signed assurance-policy lineage against that policy and rejects:

- unreviewed signer replacement,
- unreviewed key rotation,
- transitions from the wrong previous signer/key,
- duplicate or unused transitions,
- unsupported signature algorithms,
- signer/self-verification identity reuse,
- and any already-invalid assurance-policy lineage.

The governance policy and transition set have a deterministic digest so downstream readiness can bind to the exact trust-root rules used for assessment.

Cryptographic signature verification remains represented by the existing external signature-verification receipts; this crate governs authorization/provenance of the signer and key identities in those receipts.

No report from this crate grants physical authority.
