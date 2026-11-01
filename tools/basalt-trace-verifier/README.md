# BASALT trace verifier

**Status:** reference-model verifier prototype. It does not validate a live database backend.

## Run

From the repository root:

```sh
cargo fmt --manifest-path tools/basalt-trace-verifier/Cargo.toml -- --check
cargo test --locked --manifest-path tools/basalt-trace-verifier/Cargo.toml
cargo clippy --locked --manifest-path tools/basalt-trace-verifier/Cargo.toml --all-targets -- -D warnings
cargo run --locked --manifest-path tools/basalt-trace-verifier/Cargo.toml -- \
  --trace benchmarks/basalt/storage-conformance-trace.v1.json \
  --schema benchmarks/basalt/schema/storage-conformance-trace.v1.schema.json \
  --out basalt-reference-report.json
```

The standalone package is intentionally not a root workspace member. It uses a private lockfile so the repository-wide workspace glob/manifest problems do not contaminate this focused check.

## What it verifies

- Strict typed deserialization with unknown fields rejected.
- Unique operation IDs, exact reference linearization membership, dependencies, per-client sequence and same-runner real-time precedence.
- No comparison of monotonic clock values across different runner IDs.
- Fault-to-operation identity, runner, interval, class and expected outcome.
- Explicit fixture digest preimages against SHA-256.
- Deterministic replay of source generation/predecessor, log checkpoint, policy compare, exact receipt identity, current authority revision and receipt acceptance revision.
- Lost-response commit as a server-side state transition with an indeterminate client result, followed by exact-receipt recovery.
- Full transaction retry grouping and identical request replay after injected serialization failure.
- Witness sequence, authority/incarnation, predecessor digest chain, strictly advancing authority revisions, final state binding and the exact witness sequence consulted by restore verification.
- Final source cursors, log cursors, policy head and immutable accepted receipt set.
- Six deliberately seeded faulty reference traces; every one must be rejected.

## What it does not verify

- It does not execute SQLite, PostgreSQL, Turso or etcd.
- It does not ingest the separate observed backend event-log schema yet.
- It does not find legal linearizations for arbitrary concurrent observed histories; the fixture supplies one deterministic legal reference order.
- It does not verify witness signatures, key rotation, quorum independence or trust-root authorization. The witness digests are symbolic fixture records.
- It does not test process crashes, filesystem ordering, power loss, hardware flush behavior, replication or physical restore.
- It does not qualify performance, availability or any storage engine.

The optional `--schema` argument binds the schema file's SHA-256 into the report. The current CLI uses strict Rust types for the reference fixture; it does not claim to be a full JSON Schema Draft 2020-12 implementation.

## Result interpretation

- `pass`: the reference fixture is internally consistent with this reducer and every seeded negative control is detected.
- `fail`: the well-formed reference trace contradicts the reducer or a seeded negative control was not detected.
- `invalid_evidence`: strict deserialization, required-field, dependency, ordering, digest, fault-schedule or witness-chain validation failed.

A `pass` is not a backend result or a cryptographic/storage durability claim. Live adapter observations must be written separately and checked by the next verifier increment.
