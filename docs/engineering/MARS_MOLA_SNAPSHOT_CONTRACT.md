# MOLA Verified Snapshot Contract

Status: design contract for issue #6653. This document does not claim the snapshot API is implemented.

## Problem

The MOLA adapter currently hashes the detached label and raster during `open()`, but cell reads later reopen the raster pathname. A digest therefore authenticates the bytes observed at verification time; it does not guarantee that a later pathname read observes the same bytes.

An open file descriptor is not treated as an immutability guarantee either. Reproducible experiments need a stronger byte-lifetime boundary.

## Contract

The reproducible path is:

`source product -> verified snapshot -> sampler -> terrain provenance`

A **verified snapshot** must:

1. identify one exact raster byte sequence with a SHA-256 digest;
2. be created under caller-controlled storage intended to be immutable/read-only after verification;
3. retain enough identity to reopen the same artifact without silently falling back to the original mutable pathname;
4. expose the digest as part of its public identity;
5. remain separate from the detached-label identity;
6. fail closed if the snapshot identity cannot be verified.

The sampler should accept the verified snapshot identity for reproducible runs rather than accepting an arbitrary pathname and calling the result reproducible.

## Storage boundary

The design deliberately does **not** require loading a global MEGDR raster into memory.

A caller may stage the raster in content-addressed or otherwise immutable/read-only storage. The snapshot abstraction owns the verified artifact identity; the storage policy is responsible for preventing replacement after verification.

Copying bytes into a snapshot is not itself considered proof of immutability. The post-copy storage boundary must provide that guarantee.

## API shape

The eventual API should make the evidence distinction visible in types:

- path-based product opening remains available as a convenience for exploratory/non-reproducible use;
- reproducible sampling consumes a verified snapshot;
- a verified snapshot cannot silently degrade into path-based sampling;
- provenance records the snapshot artifact identity alongside the detached-label identity.

Exact type names are intentionally deferred until compiler-backed implementation work begins.

## Companion products

Topography and observation-count rasters remain separate artifacts. Their individual content identities must survive provenance composition.

The snapshot contract therefore must not rely on a single flattened pathname or digest for the pair. A reproducible sample should be able to establish which topography snapshot and which count snapshot supplied the observation.

## Failure semantics

The reproducible API should reject:

- a snapshot whose current bytes no longer match its recorded digest;
- a missing snapshot;
- a mutable/non-verified path presented as a reproducible snapshot;
- a provenance record that omits the snapshot identity;
- a topography/count pair whose registration or identity cannot be established.

It should never silently re-hash and continue against a replacement artifact, because that would change the experiment input while preserving the appearance of a pinned run.

## Storage integration direction

A useful deployment backend for Luminous Dynamics is a Nix store path or an equivalent content-addressed artifact store. Nix store paths are designed as opaque identities for exactly one store object, which aligns naturally with the snapshot contract's requirement that an experiment refer to a stable artifact identity rather than a mutable source pathname.

The physics crate should **not** depend directly on Nix semantics. Instead, the snapshot abstraction should represent the generic contract and allow a caller/integration layer to supply an artifact from a content-addressed immutable store. A Nix-backed implementation can then map a verified store artifact into that generic abstraction without making the scientific model Nix-specific.

This distinction matters because content addressing and filesystem immutability are related but separate properties: the artifact identity says which bytes are intended, while the storage boundary must prevent or detect post-verification replacement. Nix documentation likewise treats store paths as unique references to store objects and documents read-only filesystem requirements for stronger read-only guarantees.

## Verification plan

Implementation should add regression coverage for:

1. verified snapshot creation and digest identity;
2. successful cell sampling through the snapshot;
3. replacement/mutation of the original source path after snapshot creation;
4. replacement of the snapshot artifact itself;
5. provenance continuity through topography + count sampling;
6. explicit rejection of unverified path-based input on reproducible APIs;
7. preservation of large-raster streaming/random-access behavior rather than whole-file RAM loading.

Compiler and CI results remain the authoritative implementation evidence once the API is introduced.
