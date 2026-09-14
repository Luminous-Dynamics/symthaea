# symthaea-assurance-policy-lineage-anchor

Externally provisioned monotonic checkpoints for assurance-policy lineage rollback resistance.

An internally valid signed lineage is not enough to prove that no newer signed revisions were omitted. A truncated history can still look contiguous from revision 1 to its apparent tip.

This crate therefore models a trust-store checkpoint that binds the latest accepted policy-lineage tip revision and manifest digest. Anchors are themselves versioned and hash-chain to their predecessor anchor.

The anchor-chain assessor rejects:

- missing anchor history,
- duplicate or skipped anchor revisions,
- predecessor-anchor digest mismatch,
- manifest/deployment identity changes,
- decreasing anchored policy revision,
- same policy revision with a different tip digest,
- and trust-time regression in anchor recording.

Downstream readiness should additionally require the current anchor's deterministic digest to equal a digest provisioned by the deployment trust root. That external expected digest is what prevents an attacker from substituting an older but internally valid anchor chain.

No anchor or report grants physical authority.
