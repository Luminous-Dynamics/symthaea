# Archived observation-attestation source

This directory preserves the unfinished `symthaea-observation-attestation` implementation exactly as it existed on the Broca audit stack when an unrelated workspace-quality failure exposed its stale dependency boundary.

The archived source imports `symthaea_core::observation_fabric`, while the current `symthaea-core` exports the narrower `observation` contract instead. The implementation is therefore not a buildable workspace crate in this public topology.

The source is retained rather than deleted so its work and provenance remain available. Reintroduction belongs with the observation-fabric contract that defines the referenced receipt/envelope types, a complete dependency closure, lockfile validation, and exact-head CI evidence.

This archive is intentionally non-buildable and outside the Cargo workspace member globs.

Related workspace-hygiene precedent: PR #6742. The observation-fabric foundation remains separately tracked in PR #6579.