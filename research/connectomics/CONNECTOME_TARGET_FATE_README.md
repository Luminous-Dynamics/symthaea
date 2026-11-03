# Connectome target inventory fate protocol

This directory contains the V0.1 contract for exhaustive target-side object accounting across versioned connectome worlds.

See:
- [Protocol](../../research/connectomics/CONNECTOME_TARGET_FATE_PROTOCOL_V0.1.md)
- [Research issue #10328](https://github.com/Luminous-Dynamics/symthaea/issues/10328)

The schema validates record shape. The Rust semantic verifier under `tools/connectome-target-fate-verifier/` cross-checks exact target-manifest and correspondence-receipt bytes, mapping membership and alternative-set grouping, one fate per target object, and all counters.

The fixture is synthetic. It demonstrates a mapped segment, unresolved alternative neuron candidates, a target-only unresolved synapse and an explicitly out-of-scope artifact. It does not demonstrate biological identity or new biological formation.

## Semantic verifier

A Rust CLI verifier and nineteen authored tests are present in `tools/connectome-target-fate-verifier/`. It recomputes exact manifest/receipt hashes, target inventory coverage, object-type correspondence, mapping and alternative membership, status counters, and per-type counts. It never authenticates evidence artifacts or promotes biological identity. Tests are authored but not compiled or executed yet.
