# Upgrade operational head authority

This crate proves a bounded currentness theorem for `FabricationUpgradeOperationalState`.

It proves that one exact operational state is the latest handoff-scoped publication inside one exact append-only transparency-log view covered by one exact interval-valid signed checkpoint and witness quorum. Witness organization/failure-domain metadata must match one exact threshold-authorized `WitnessAuthorityRegistryV1`, and independent verifier providers must verify the exact raw checkpoint and witness signature bytes.

It does not prove that no later checkpoint exists outside the supplied view, that the witness registry is globally latest, or that the supplied trust snapshot is globally latest.

A downstream no-rollback capability may rely on this observed head only within those same bounded currentness semantics.
