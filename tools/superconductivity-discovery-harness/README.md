# Superconductivity discovery registry harness

This is a standalone Rust tool for validating the versioned superconductivity discovery registry.

## Run

From the repository root:

```bash
cargo test --manifest-path tools/superconductivity-discovery-harness/Cargo.toml
cargo run --manifest-path tools/superconductivity-discovery-harness/Cargo.toml -- \
  --schema data/schemas/superconductivity_discovery_case_v0_1.schema.json \
  --instance data/research/sc_superconductor_discovery_cases_v0_1.json \
  --out /tmp/sc-discovery-registry-receipt.json
```

The audit:

1. Parses and meta-validates the declared JSON Schema Draft 2020-12 contract.
2. Refuses `$ref`, `$dynamicRef` and `$recursiveRef` so the schema cannot cause external or indirect schema resolution.
3. Validates the registry instance against the committed schema.
4. Enforces unique candidate/source identifiers and closed source references.
5. Prevents computational / ML-predicted Tc values from being recorded as observed transitions.
6. Requires actinide-bearing candidates to carry an explicit computational-only or exclusion gate.
7. Emits deterministic SHA-256 digests and record counts in a JSON receipt.

The `jsonschema` dependency is pinned and built with default features disabled. The validator does not retrieve HTTP or filesystem references. Cargo may still need to download dependencies when the package is first built; validation itself is offline and reads only the explicit input files.

## Claim ceiling

A passing registry audit verifies data shape and selected evidence semantics. It does not verify source truth, reproduce the source calculations, establish phase stability, or qualify a material experimentally or for engineering use.
