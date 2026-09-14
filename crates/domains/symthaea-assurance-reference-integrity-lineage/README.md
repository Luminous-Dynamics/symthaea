# symthaea-assurance-reference-integrity-lineage

Rollback-resistant lineage qualification for signed reference-integrity manifests.

A valid lineage starts at revision 1, advances contiguously, binds every revision to the exact digest of its predecessor, preserves manifest/issuer/subject identity, verifies every source RIM signature and normalization mapping, and admits signer/key changes only through an explicit reviewed signer-authority policy.

The policy also pins the exact expected current tip revision and digest, so a truncated but internally valid older lineage cannot become current by omission.

Release identifiers may change across revisions; that is the purpose of a reference history. Signer changes may also occur, but only when an applicable reviewed signer rule permits the exact signer and key digest for that revision.

This crate evaluates assurance provenance only. It never grants physical authority.
