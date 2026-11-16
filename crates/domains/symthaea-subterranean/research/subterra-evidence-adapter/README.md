# Subterra evidence adapter v0.1

A standalone Rust adapter for the exact Subterra v0.1 site-evidence and cave-network JSON shapes authored in Symtropy GeoBio + Subterra research PR #5486.

## Source contracts

Pinned provenance manifest: `upstream-schema-manifest-v0.1.json`. It binds the upstream schema IDs, versions, Git blob object IDs and copied synthetic fixture object IDs. Git blob IDs are repository provenance identifiers, not signatures or independent authenticity proofs.

- Symtropy site schema: `research/geobio-foundry/SUBTERRA_SITE_EVIDENCE_SCHEMA_v0.1.json`, blob SHA `970b48703a8147cd3047cce05c5fc0631d1f03d9`.
- Symtropy network schema: `research/geobio-foundry/SUBTERRA_NETWORK_SCHEMA_v0.1.json`, blob SHA `bfe797f741d930ffcde2ce9d33a21c75396f46c0`.
- Symtropy profile schema: `research/geobio-foundry/SUBTERRA_PROFILE_SCHEMA_v0.1.json`, blob SHA `87d77b208f20227797ff59fd1f6d4c01bbb1c72c`.
- Site fixture: blob SHA `31813b808d596f9ddb78c4de49a15f1773026fc9`.
- Network fixture: blob SHA `95fa93de34773569aa3abcb9c3f808f2f9986e97`.
- Profile fixture: blob SHA `a8305c9881f3483a762e3bc6717b86009d583039`; synthetic non-uniform profile with an explicit unobserved span from 16 m to 24 m.
- Upstream research PR: https://github.com/Luminous-Dynamics/symtropy/pull/5486

Fixtures in this package are copies of the cited synthetic examples for deterministic offline tests. They are not empirical planetary observations.

## Supported records

- Site evidence: typed DTO, semantic consistency checks and non-probabilistic `SiteBeliefInput` conversion; source schema version is retained.
- Cave network: typed DTO, observed-topology components, evidence-state summary and directional demonstrated-route queries.
- Non-uniform profile: typed DTO, explicit unknown intervals, measurement uncertainty preservation, coverage summary and observed-only length-weighted dimensions.
- Science-return lifecycle: monotonic evidence events from proposal through admission, with failure/cancellation and store-and-forward branches; delivery and scientific admission are distinct.

All three inputs retain schema-specific states and evidence references. The profile summary is a lightweight adapter view, not a replacement for Symtropy's morphometry reference kernel.

## Design invariants

1. Morphology, geological origin, connectivity, accessibility, surveyed extent, engineering suitability and preservation remain separate claims.
2. A claim retains its original state and evidence references. The adapter does not calculate posterior probabilities or turn a state into a stronger state.
3. A directly observed connection does not imply a robot has traversed it. An interior survey does not imply every network edge is traversable.
4. Physical connectivity components use only directly observed edges; demonstrated routes additionally require demonstrated traversability and obey direction.
5. Candidate and geophysically supported edges are preserved as hypotheses and are not promoted into observed topology.
6. Evidence references are opaque identifiers. The adapter does not authenticate, resolve, or dereference them.
7. Profile means are calculated only over observed intervals. An unobserved interval is never treated as zero width/height or interpolated.
8. Science-return events distinguish proposed, executed, acquired, integrity-verified, persisted, store-and-forwarded, delivered and scientifically admitted states. Store-and-forward is not delivery.
9. The v0.1 observation `captured_at` string is retained but not independently checked as RFC 3339; JSON Schema format validation remains a separate gate.
10. Planetary-protection and activity permissions are represented as declarations only. They are never converted into permission for a measurement, sample, biological activity or actuator.
11. No Earth coordinate assumptions, geodetic transforms, SLAM, geomechanics or engineering qualification are implemented here.

## Run locally

```bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-evidence-adapter/Cargo.toml --all-targets -- -D warnings
```

The package has its own Cargo workspace and does not join the production Symthaea workspace. `serde` and `serde_json` are confined to this research adapter.

## Validation boundary

- Site, cave-network and non-uniform-profile adapters, science-return lifecycle, pinned upstream-schema manifest and unit-test fixtures: AUTHORED.
- Authored tests: 21 site/network + 6 profile + 19 science-return = 46 adapter tests.
- The profile adapter validates ordered contiguous coverage and computes observed-only geometry summaries; it does not yet propagate uncertainty through the summary statistics.
- The science-return record validates stage sequence, canonical UTC timestamp syntax, clock identity, declared uncertainty and definite same-clock regressions. It does not prove clock synchronization, cryptographic authenticity, immutable storage, custody authority or transport delivery.
- The upstream manifest pins source Git objects but does not fetch or authenticate upstream content at runtime.
- The adapter rejects malformed required fields and unknown fields, but does not replace full Draft 2020-12 schema validation. Science-return uses a deliberately narrow UTC timestamp subset, not a general RFC 3339 parser.
- Rust source and unit-test fixtures: AUTHORED.
- Compilation, formatting, tests and Clippy: NOT EXECUTED in the current authoring environment (no Rust toolchain available).
- JSON Schema meta-validation: not performed by this crate; upstream schemas have a separate validation record.
- Evidence authenticity, planetary coordinate correctness, network physics, traversability, geology, safety, planetary protection and physical qualification: NOT ESTABLISHED.
