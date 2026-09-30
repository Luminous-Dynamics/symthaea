# MOLA Verified Snapshot Contract

Status: partially implemented on PR #6588. The MOLA adapter now exposes a verified open-handle snapshot and snapshot-backed sampler; the caller-attested immutability guarantee remains external to the physics crate.

## Problem

The MOLA adapter currently hashes the detached label and raster during `open()`, but cell reads later reopen the raster pathname. A digest therefore authenticates the bytes observed at verification time; it does not guarantee that a later pathname read observes the same bytes.

An open file descriptor is not treated as an immutability guarantee either. Reproducible experiments need a stronger byte-lifetime boundary.

## Contract

The reproducible path is:

`source product -> verified artifact identity -> immutable snapshot -> sampler -> terrain provenance`

A **verified snapshot** must:

1. identify one exact raster byte sequence with a SHA-256 digest;
2. retain the source/product identity independently of the byte digest;
3. be created under caller-controlled storage intended to be immutable/read-only after verification;
4. retain enough identity to reopen the same artifact without silently falling back to the original mutable pathname;
5. expose the digest as part of its public identity;
6. remain separate from the detached-label identity;
7. fail closed if the snapshot identity cannot be verified.

The sampler should accept the verified snapshot identity for reproducible runs rather than accepting an arbitrary pathname and calling the result reproducible.

## PDS identity versus byte identity

The snapshot contract deliberately distinguishes two different questions:

- **Which archived product/version did we intend to consume?**
- **Which exact bytes did we actually authenticate?**

For PDS4-backed sources, a product's **LIDVID** is a logical product/version identity, while the cryptographic digest is byte identity. PDS explicitly recommends using LIDVIDs to identify the precise PDS4 product/version used in research, and PDS products carry both LID and version identifiers. The snapshot should therefore be able to retain a source/product identifier such as LIDVID when available, without treating it as a substitute for the cryptographic digest.

This also prevents a future API from conflating an archive-level revision with a content-addressed storage identity. They are complementary evidence fields.

## Storage boundary

The design deliberately does **not** require loading a global MEGDR raster into memory.

A caller may stage the raster in content-addressed or otherwise immutable/read-only storage. The snapshot abstraction owns the verified artifact identity; the storage policy is responsible for preventing replacement after verification.

Copying bytes into a snapshot is not itself considered proof of immutability. The post-copy storage boundary must provide that guarantee.

## API shape

The eventual API should make the evidence distinction visible in types:

- path-based product opening remains available as a convenience for exploratory/non-reproducible use;
- reproducible sampling consumes a verified snapshot;
- a verified snapshot cannot silently degrade into path-based sampling;
- provenance records both the archive/product identity, when available, and the exact snapshot byte identity;
- artifact-local metadata remains attached to the artifact rather than being flattened into one shared terrain envelope.

Implemented types currently include `MolaRasterSnapshot`, `MolaSnapshotStorage`, `MolaTerrainObservation`, `TerrainArtifactIdentity`, and `TerrainArtifactComposition`. The snapshot sampler returns the existing `TerrainSample` plus the independent artifact composition and a canonical `TerrainObservationIdentity` SHA-256. This identity uses domain-separated, length-prefixed fields and IEEE-754 bit encodings; it is an observation-record identity, not a processing-history or full experiment identity.

## Companion products

Topography and observation-count rasters remain separate artifacts. Their individual content identities must survive provenance composition.

The snapshot contract therefore must not rely on a single flattened pathname or digest for the pair. A reproducible sample should be able to establish which topography artifact/snapshot and which count artifact/snapshot supplied the observation.

## Artifact composition

The multi-product observation boundary should preserve three distinct layers:

1. **Artifact identity** — source/product identity, revision/version, byte digest, and relevant coordinate/data semantics.
2. **Composition identity** — the exact set of artifacts registered together for one observation operation.
3. **Observation identity** — the selected grid cell, sampling method, values, quality state, and experiment provenance.

Two artifacts with the same logical name but conflicting source/version/digest identity must fail closed. Identical identities may deduplicate deterministically. Composition must not silently overwrite artifact-local metadata from the right-hand input.

Issue #6660 tracks this contract before the snapshot API becomes final.

## Failure semantics

The reproducible API should reject:

- a snapshot whose current bytes no longer match its recorded digest;
- a missing snapshot;
- a mutable/non-verified path presented as a reproducible snapshot;
- a provenance record that omits the snapshot identity;
- a topography/count pair whose registration or identity cannot be established;
- contradictory artifact metadata even when the digest fields happen to be independently valid.

It should never silently re-hash and continue against a replacement artifact, because that would change the experiment input while preserving the appearance of a pinned run.

## Storage integration direction

A useful deployment backend for Luminous Dynamics is a Nix store path or an equivalent content-addressed artifact store. Nix store paths are designed as opaque identities for store objects, which aligns naturally with the snapshot contract's requirement that an experiment refer to a stable artifact identity rather than a mutable source pathname.

The physics crate should **not** depend directly on Nix semantics. Instead, the snapshot abstraction should represent the generic contract and allow a caller/integration layer to supply an artifact from a content-addressed immutable store. A Nix-backed implementation can then map a verified store artifact into that generic abstraction without making the scientific model Nix-specific.

This distinction matters because content addressing and filesystem immutability are related but separate properties: the artifact identity says which bytes are intended, while the storage boundary must prevent or detect post-verification replacement.

## Verification plan

Implementation should add regression coverage for:

1. verified snapshot creation and digest identity;
2. successful cell sampling through the snapshot;
3. replacement/mutation of the original source path after snapshot creation;
4. replacement of the snapshot artifact itself;
5. provenance continuity through topography + count sampling;
6. explicit rejection of unverified path-based input on reproducible APIs;
7. preservation of large-raster streaming/random-access behavior rather than whole-file RAM loading;
8. preservation of PDS product/version identity alongside the cryptographic snapshot identity;
9. rejection of conflicting artifact metadata during multi-product composition.

Compiler and CI results remain the authoritative implementation evidence. Source changes and regression fixtures alone are not evidence that the code compiles or that tests pass.
