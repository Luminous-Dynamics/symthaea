# Observation Attestation (archived source)

This directory contains an in-progress detached Ed25519 attestation adapter.

It is intentionally **not a workspace crate** at present: the source targets an
observation-fabric API that is not present in the current `symthaea-core`
public surface. Keeping the source here preserves the work without allowing an
incomplete adapter to block the workspace build.

Restore it only together with:
1. a matching `Cargo.toml`;
2. the required `symthaea-core::observation_fabric` provider API; and
3. exact-head workspace/lockfile validation.

This is an archive/quarantine boundary, not a deletion of the implementation.
