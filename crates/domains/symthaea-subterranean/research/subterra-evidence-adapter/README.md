# Subterra evidence adapter v0.1

A standalone Rust adapter for the exact Subterra v0.1 site-evidence and cave-network JSON shapes authored in Symtropy GeoBio + Subterra research PR #5486.

## Source contracts

- Symtropy site schema: `research/geobio-foundry/SUBTERRA_SITE_EVIDENCE_SCHEMA_v0.1.json`, blob SHA `970b48703a8147cd3047cce05c5fc0631d1f03d9`.
- Symtropy network schema: `research/geobio-foundry/SUBTERRA_NETWORK_SCHEMA_v0.1.json`, blob SHA `bfe797f741d930ffcde2ce9d33a21c75396f46c0`.
- Site fixture: blob SHA `31813b808d596f9ddb78c4de49a15f1773026fc9`.
- Network fixture: blob SHA `95fa93de34773569aa3abcb9c3f808f2f9986e97`.
- Upstream research PR: https://github.com/Luminous-Dynamics/symtropy/pull/5486

Fixtures in this package are copies of the cited synthetic examples for deterministic offline tests. They are not empirical planetary observations.

## Design invariants

1. Morphology, geological origin, connectivity, accessibility, surveyed extent, engineering suitability and preservation remain separate claims.
2. A claim retains its original state and evidence references. The adapter does not calculate posterior probabilities or turn a state into a stronger state.
3. A directly observed connection does not imply a robot has traversed it. An interior survey does not imply every network edge is traversable.
4. Physical connectivity components use only directly observed edges; demonstrated routes additionally require demonstrated traversability and obey direction.
5. Candidate and geophysically supported edges are preserved as hypotheses and are not promoted into observed topology.
6. Evidence references are opaque identifiers. The adapter does not authenticate, resolve, or dereference them.
7. Planetary-protection and activity permissions are represented as declarations only. They are never converted into permission for a measurement, sample, biological activity or actuator.
8. No Earth coordinate assumptions, geodetic transforms, SLAM, geomechanics or engineering qualification are implemented here.

## Run locally

```bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml --all-targets -- -D warnings
```

The package has its own Cargo workspace and does not join the production Symthaea workspace. `serde` and `serde_json` are confined to this research adapter.

## Validation boundary

- Rust source and unit-test fixtures: AUTHORED.
- Compilation, formatting, tests and Clippy: NOT EXECUTED in the current authoring environment (no Rust toolchain available).
- JSON Schema meta-validation: not performed by this crate; upstream schemas have a separate validation record.
- Evidence authenticity, planetary coordinate correctness, network physics, traversability, geology, safety, planetary protection and physical qualification: NOT ESTABLISHED.
