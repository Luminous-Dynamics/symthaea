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

The standalone package is intentionally not a root workspace member. It uses a private lockfile so the repository-wide workspace glob/manifest problems do not contaminate this focused check. The source is normalized with the repository-pinned Rust 1.96 formatter; the dedicated verifier workflow checks formatting again on every exact PR head.

## What it verifies

- Strict typed deserialization with unknown fields rejected.
- Unique operation IDs, exact reference linearization membership, dependencies, per-client sequence and same-runner real-time precedence.
- Bounded legal-linearization search over fully completed trace-shaped histories (up to 32 operations), independent of the fixture's declared tie-break.
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
- The new search API does not yet ingest raw backend observation artifacts; callers must provide a strict Trace-shaped object populated with actual observed request/response values.
- The v1 search is limited to fully completed operations and at most 32 operations. Pending/no-response operations return unsupported/inconclusive until their commit/no-commit possibilities are modelled.
- Search-budget exhaustion returns `inconclusive`; it never becomes a pass or a definitive fail.
- It does not verify witness signatures, key rotation, quorum independence or trust-root authorization. The witness digests are symbolic fixture records.
- It does not test process crashes, filesystem ordering, power loss, hardware flush behavior, replication or physical restore.
- It does not qualify performance, availability or any storage engine.

The optional `--schema` argument binds the schema file's SHA-256 into the report. The current CLI runs the deterministic reference fixture. The history-search API ignores the fixture's declared reference order; `reference_linearization_order` defaults empty when deserializing an observed history. The library additionally exposes `verify_observed_history(&Trace, &HistorySearchConfig)`, which searches for a legal order subject to causal constraints and a hard state-expansion budget. It currently uses the same strict Rust Trace types; it does not claim to ingest the backend-observation JSON schema or implement full JSON Schema Draft 2020-12 validation.

## Result interpretation

- `pass`: a legal sequential interpretation was found, final state matches, and all relevant operation outcomes agree. For reference CLI mode, every seeded negative control is also detected.
- `fail`: a fully explored, well-formed history has no legal sequential interpretation, or the reference fixture contradicts the reducer.
- `invalid_evidence`: strict deserialization, required-field, dependency, ordering, digest, fault-schedule or witness-chain validation failed.
- `inconclusive`: the search bound was reached before a legal order was found or all candidate orders were eliminated.

A `pass` is not a backend result or a cryptographic/storage durability claim. Live adapter observations must be written separately and checked by the next verifier increment.
