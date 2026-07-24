# Series 24 authoring checklist

## Boundary inventory

- Enumerate every public byte decoder, archive path, canonicalizer, lineage walk, external verifier adapter, and report renderer.
- Require a caller-owned limit profile at untrusted entrypoints.
- Check raw bytes and cheap counts before expensive work wherever semantics permit.

## Required abuse classes

- oversized bytes and declared lengths;
- deep nesting and huge strings;
- duplicate-key and duplicate-signer amplification;
- long or cyclic lineage;
- signature-verifier call explosion;
- subprocess timeout and output flood;
- archive expansion, traversal, link, special-file, and manifest confusion;
- cancellation and partial-state persistence;
- cache reuse under changed policy, limits, verifier, or lineage context.

## Acceptance

- Existing within-limit Series 16–23 fixtures retain exact results.
- Rust and independent verifier agree on stable resource failure stages/codes.
- Worst-case-valid reference bundles fit the documented default offline profile.
- Series 23 cumulative replay and reproducibility remain green.
