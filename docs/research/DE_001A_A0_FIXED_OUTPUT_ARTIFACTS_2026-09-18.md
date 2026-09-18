# DE-001A0 — Nix fixed-output scientific inputs

**Date:** 2026-09-18  
**Authority:** input realization and byte identity only  
**Scientific claim:** NONE

## Purpose

The eight A0 inputs are now represented as Nix fixed-output derivations inside the isolated cosmology subflake.

This creates a two-layer integrity chain:

1. Nix accepts an upstream retrieval only when its bytes match the preregistered SHA-256.
2. The Rust `de001a-a0-verify` binary independently verifies regular-file status, byte count, and SHA-256 and emits an A0 receipt.

Neither layer performs statistical inference.

## Why fixed-output derivations

The upstream files should not be copied into the Symthaea Git history merely to make them stable. Their scientific identity is their exact content, not their location.

Nix fixed-output fetches let the repository specify:

`URL + cryptographic digest -> immutable store object`

while still failing closed if a server changes bytes in place.

The URL is therefore a retrieval hint; the digest remains the authority.

## Artifact package

The subflake exposes:

`packages.x86_64-linux.a0-artifacts`

which copies all eight fixed-output sources into regular files named by their A0 roles. This is important because the Rust verifier intentionally rejects symlinked inputs.

The flake also exposes:

`checks.x86_64-linux.a0-artifacts`

which independently checks all eight byte counts and SHA-256 values with coreutils.

The existing exact-head Nix workflow runs `nix flake check`, so this check will be included whenever this child subject reaches a runner.

## Cross-representation drift guard

`de001a_a0_nix_sources_v1.json` records the URL, hexadecimal SHA-256, Nix SRI digest, and byte count used by the subflake.

A Rust integration test requires its `(role, size, SHA-256)` map to be exactly equal to `de001a_a0_artifacts_v1.json`.

This prevents the Nix fetch layer and the runtime verifier from silently diverging.

## Boundary

Successful realization establishes only that the retrieved scientific/reference bytes match the preregistered identities.

It does not establish that the numerical environment is correct, that the likelihood can be reproduced, that LambdaCDM fits the universe, or that dark energy is or is not dynamical.
