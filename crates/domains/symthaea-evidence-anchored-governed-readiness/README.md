# symthaea-evidence-anchored-governed-readiness

Top-level rollback-resistant assurance composition for Symthaea policy readiness.

The gate recomputes signer-governed policy readiness and also validates an external monotonic policy-lineage anchor chain. The current anchor must:

- be the valid tip of the supplied anchor chain,
- have a deterministic digest equal to the externally provisioned expected anchor digest,
- identify the same manifest/deployment lineage,
- bind exactly the current signed policy-lineage tip revision and manifest digest,
- and have been recorded no later than the earliest point in the trusted clock-uncertainty interval.

If the signed lineage is behind the anchor, readiness is blocked as rollback. If the signed lineage is ahead of the anchor, readiness is blocked until the new policy revision is externally checkpointed. Same-revision/different-digest disagreement is invalid.

This prevents a truncated but internally valid signed lineage from making an older policy revision appear current.

The report never grants physical authority.
