# Archived observation-attestation source

This directory preserves the unfinished `symthaea-observation-attestation` implementation exactly as it existed on the repository default branch when it was discovered as an unintended workspace member.

The archived source imports `symthaea_core::observation_fabric`, while the current `symthaea-core` exports the narrower `observation` contract instead. It is therefore not a buildable workspace crate at this point.

The implementation is retained here rather than deleted so its work and provenance remain available. Reintroduction should happen only with the matching observation-fabric contract, a complete crate manifest, dependency closure, lockfile validation, and exact-head CI evidence.

This archive is intentionally non-buildable and outside the Cargo workspace member globs.
