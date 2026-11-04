# BASALT trace and observation verifier

**Status:** design/reference contract only in this checkout. `tools/basalt-trace-verifier/Cargo.toml` and its Rust source files are absent; the commands below are intended interfaces only and are not currently runnable. This README describes target semantics, not an implemented or executed verifier. No passing result is claimed.

A future verifier result would mean only that submitted evidence is internally well-formed and admits a bounded legal sequential interpretation under the BASALT model. It would not by itself prove physical durability or end-to-end anti-rollback.

## Run the standalone Rust package

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

The standalone package is not a root workspace member. It has a private lockfile. The exact-tree CI workflow regenerates the lockfile when dependencies change, then runs formatting, locked tests, strict Clippy and the reference CLI before committing a tested source tree.

## Reference reducer

The deterministic reference reducer checks strict typed trace deserialization, operation identities, dependency order, per-client sequence, same-runner real-time precedence, fault schedules, fixture digest preimages, source generations/predecessors, log checkpoints, policy compare, immutable receipts, current authority revision, receipt acceptance revision, retry grouping, witness sequence/incarnation/predecessor links and final source/log/policy/receipt state.

Six seeded faulty reference traces are required to be rejected. The exact-head reference CLI report from the checked-in 12-operation fixture recorded 11 invariant expectations and all 6 mutants detected, with no counterexamples. That report's claim ceiling is reference-trace semantics only: no backend run, cryptographic witness signature, physical durability or engine qualification.

## Bounded legal-linearization search

The library exposes:

```rust
verify_observed_history(&Trace, &HistorySearchConfig)
```

It searches for a sequential order satisfying per-client program order, explicit `happens_after` edges and same-runner real-time precedence. Monotonic clocks from different runners are never compared. The search branches only on currently enabled operations, memoizes by remaining-operation bitset plus deterministic model-state digest, and has a hard expansion budget.

The v1 search is limited to 32 fully completed operations. Its statuses are:
- `pass`: a legal ordering was found and the modeled final revision/cursors/policy/receipt state agrees;
- `fail`: exhaustive search found no legal ordering;
- `invalid_evidence`: the operation set or its causal constraints are malformed;
- `inconclusive`: the operation or search bound prevents a verdict.

Budget exhaustion never becomes pass. The final opaque state digest is not yet recomputed from a full canonical state manifest.

The unoptimized exhaustive enumerator and memoized search use the same `apply_operation` transition function. Their agreement checks that the optimized search finds the same legal-order set for small histories; it is **not** an independent second implementation of BASALT business semantics. An independent model implementation and differential backend testing remain future work.

## Backend observation ingestion

The library exposes:

```rust
verify_backend_observation(
    observation_bytes,
    reference_trace_bytes,
    expected_reference_trace_sha256,
    &HistorySearchConfig,
)
```

The caller supplies the expected reference-trace digest from an external test manifest. The observation cannot nominate its own expected reference. The v1 envelope requires:

- backend ID, engine/version, adapter commit SHA, executable/configuration digests and durability-profile label;
- a runner-local monotonic clock window for every participating runner;
- raw events with unique IDs and unique per-runner sequence numbers;
- observed operations, request intent, retry metadata, explicit source/log scope for `advance`, applied fault records, witness observations and a final state observation;
- a raw event-log digest recomputed using RFC 8785 JCS after sorting events by `(runner_id, runner_sequence, event_id)`.

The validator checks runner-local clock monotonicity, event/operation/runner binding, operation boundaries, raw event digest, transaction begin/read/decision and commit/abort evidence, atomic source/log/receipt writes for committed advances, and applied/unapplied faults against explicit `fault_id` events.

### Whole-transaction serialization retries

A captured SQLSTATE 40001 retry group must show, for each attempt, a transaction begin, state read and decision. Each attempt must use a different transaction ID. The read-snapshot and decision-input fingerprints must match their raw events, while request intent and source/log scope remain unchanged. A retry that repeats only the final write without beginning a new transaction, rereading and recomputing is rejected.

### Pending operations

An operation with `completed_at_ns = null` and `outcome = null` is listed as pending and forces `inconclusive`. Even when a raw commit event exists but the response/capture is incomplete, the verifier does not synthesize success or abort. A declared retry group with only one captured attempt is also `inconclusive`.

### Validation boundary

The Rust parser uses strict Serde structs with unknown-field rejection and explicit structural/semantic checks. It does **not** yet implement a complete JSON Schema Draft 2020-12 validator or prove schema/adapter parity across an independently generated positive/negative corpus. The CLI currently runs the deterministic reference fixture; backend observation ingestion is a library API in this increment.

## What this verifier does not prove

- It does not execute SQLite, PostgreSQL, Turso or etcd; the observation adapter validates supplied events and does not connect to a backend.
- The raw event artifact is adapter-supplied evidence. A matching hash proves that the submitted artifact is self-consistent, not that the adapter faithfully instrumented the engine.
- It does not prove that an fsync reached stable hardware, that a power-loss event was physically induced, or that a restored snapshot is fresh.
- It does not verify DSSE witness signatures, offline trust-root rotation, a complete checkpoint predecessor chain to pinned genesis, quorum-member operational independence or persist-before-sign behavior.
- It does not resolve pending commit/no-commit branches or verify histories larger than 32 operations.
- It does not qualify throughput, tail latency, availability or any storage engine.

These are deliberate claim boundaries. BASALT backend qualification must still run the same scenarios and observation contract against pinned SQLite, Turso, PostgreSQL and etcd builds with independently controlled process-crash, power-loss, I/O-failure, network-partition, stale-snapshot and restore interventions.

## Result interpretation

Observation reports use `pass`, `fail`, `invalid_evidence` and `inconclusive`. A missing acknowledgement or missing capture is never interpreted as a definite abort. A passing observation report does not supersede the separate storage durability and witness freshness gates.
