# Evidence-retention head v1

This crate upgrades one opaque interval-safe `ClockGovernedEvidenceRetentionPolicyV1` into same-view currentness.

The capability requires the exact `QuorumObservedWitnessRegistryHeadV1` already named by the exact `ContainmentCurrentWitnessRegistryHeadV1`; the same trust snapshot; the same highest opaque containment state and compromise tracker; the exact retention threshold ceremony; explicit trusted-clock ancestry from retention authorization to the composite observation basis; and the exact transparency log already authenticated by the composite governance view.

It proves only that the candidate is the highest strictly-monotonic retention-policy sequence published in that exact authenticated log view. It does not claim that no newer transparency checkpoint exists outside the view.

The portable `EvidenceRetentionHeadPublicationV1` is not live authority. The opaque `CurrentEvidenceRetentionHeadV1` can only be minted by exact rebinding to the live retention authority and composite governance view.
