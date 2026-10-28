# Broca evidence payload ABI for LongMemEval-V2

**State:** Rust evidence-contract prototype plus deterministic snapshot/restore, a V2 generation-bound manifest, a fail-closed Unix disk-generation store, and a first query-serving API pinned to one committed generation. The typed production commit binds the actual HdcStore generation, logical checksum and active live-ID set. `query_committed_context` reopens the exact committed HDC generation, rechecks its identity, and resolves only eligible context from the matching evidence snapshot. Tests are authored but not compiled in this environment; this is not a completed LongMemEval-V2 integration or persistence qualification.

## The concrete interface mismatch

The official LongMemEval-V2 backend API calls `insert(trajectory)` and requires `query(query, query_image)` to return a list of `{ "type": "text" | "image", "value": ... }` context items. Text must be non-empty; image paths must exist. The reader receives these items as memory context. See the [official backend contract](https://github.com/xiaowu0162/LongMemEval-V2#implementing-your-method).

The current `crates/domains/symthaea-broca/src/memory_bridge.rs` queries `HdcStore::scan_similar()` and returns blended `ContinuousHV` vectors. `crates/domains/symthaea-hdc-store/src/store.rs` describes vector entries containing metadata plus a `BinaryHV`. Neither API alone resolves a hit to answer-bearing text or an image path. Therefore a vector hit cannot be passed off as LongMemEval evidence.

## The query encoder is also an explicit gap

The `crates/domains/symthaea-broca/src/encoder.rs` module exposes `ThoughtLanguageEncoder::encode(&ThoughtChannels)`. `ThoughtChannels` is a structured scalar-channel representation, not arbitrary query or trajectory text. It must not be mistaken for a text embedding model.

The next worker must select and pin a real retrieval representation before a full run:
- a deterministic lexical retrieval baseline (e.g. BM25 or token/phrase postings) with no embedding-model dependency;
- a pinned text embedding model, if allowed by the hardware/cost plan; or
- a separately trained and validated text-to-HDC encoder with its own retrieval benchmark.

Report these as distinct retrieval arms. Do not label a lexical feature hash as semantic HDC, and do not use answer/question labels or benchmark metadata to build the query representation.

## What this increment provides

`memory_evidence.rs` defines a typed payload contract, ID resolver, versioned snapshot codec, and generation-bound commit manifest:

- each evidence record binds an HDC numeric ID to a stable evidence ID, environment/scope, trajectory and step provenance, source digest, payload and payload digest; `EvidenceIndex::insert` verifies the source digest against the exact source bytes supplied by the ingestion adapter;
- active evidence is eligible only within the matching scope and its optional validity interval;
- revoked, superseded and deleted records are not returned;
- deletion keeps a tombstone in the resolver; snapshot/restore preserves deleted, revoked and superseded records and rejects a duplicate-ID reinsert after restart;
- deterministic snapshot bytes require an explicit nonzero generation, are HDC-ID ordered and schema-versioned, and bind schema + generation + records in the snapshot digest; restore rebuilds ID maps from records and rejects malformed digests, duplicate identities, corrupt text content, and dangling/cross-scope supersession links;
- `EvidenceCommitManifest` binds exact HDC-index bytes and evidence-snapshot bytes to one nonzero generation, rejects a manifest/snapshot generation mismatch, and detects later artifact substitution; the coordinator must still ensure the index bytes were produced for that generation;
- text payload digests are verified before emission; image paths are canonicalized under a trusted payload root and their file-byte digests are checked;
- duplicate HDC/evidence IDs, malformed digests, invalid lifecycle shapes, path traversal, and corrupt payloads fail closed;
- resolved context items follow the official `{type,value}` envelope, and ranked HDC-hit order is retained;
- `EvidenceIndex::resolve_with_provenance` preserves the HDC ID, stable evidence ID, trajectory ID, and step coordinates beside each emitted item. `resolve` is a simple ABI projection that strips this audit envelope only when the caller explicitly needs the official `type/value` shape.

## Disk-generation persistence prototype

`crates/domains/symthaea-broca/src/memory_evidence_persistence.rs` now exposes a single-writer store:

```rust
let store = EvidenceDiskStore::open("/var/lib/broca/memory-generations")?;
let generation = store.commit_hdc_store(&mut hdc_store, &evidence_index)?;
let recovered = store.recover()?.ok_or("no committed generation")?;
assert_eq!(recovered.generation, generation);
```

The query-serving path accepts an already encoded query vector and returns candidate scores plus resolved source attribution:

```rust
let result = store.query_committed_context(
    &query_hv,
    32,
    "web",
    now_ms,
    payload_root,
    8,
)?;
let official_context = result
    .map(|result| result.context_items())
    .unwrap_or_default();
```

Integrity errors propagate through `?`; only an uninitialized store maps to no context. The resolved list retains each emitted item's HDC ID, evidence ID and source trajectory/step. The API uses exact search and is a correctness reference, not a measured latency claim.

The exact API may evolve during review; the example communicates intended use, not a compiled integration result.

- Data is staged in an immutable generation directory. HDC-index and evidence-snapshot bytes are synced; the manifest is written and verified; then `CURRENT` is atomically replaced last.
- Recovery verifies the pointer, manifest digest, artifact digests and snapshot invariants. For typed commits it opens the committed bytes with HdcStoreReader and verifies internal HDC generation, logical checksum, live count, and exact equality between HDC live IDs and Active evidence IDs. It never falls back to an older generation on corruption.
- A delete tombstone survives disk restore, and a previous ID cannot be reinserted in the restored evidence index.
- An exclusive writer lock prevents concurrent committers. A process crash may leave the lock behind; operators must verify that the prior writer is gone before removing it.
- The implementation currently requires Unix directory fsync and same-filesystem rename semantics. It does not claim portable power-loss guarantees or malicious-writer authenticity.
- The production API is typed: commit_hdc_store syncs the real HdcStore and rejects any mismatch between live vector IDs and Active evidence IDs. The opaque-byte commit helper is available only to module tests. The caller still must apply vector deletion/update and lifecycle mutation as one application-level operation; the store can detect a mismatch, not infer operator intent.
- `prune_history(retain_generations)` removes visible old generation directories outside an explicit generation window, drops unpublished future-generation artifacts and orphan staging directories, and refuses to run without a verified CURRENT commit. It is protected by the writer lock and reports which generation directories were removed.
- `query_committed_context(&BinaryHV, candidate_k, scope_id, now_ms, payload_root, max_items)` reads `CURRENT` once, verifies the named commit, opens the HDC file from that immutable generation, rechecks HDC generation/checksum/live-ID parity, performs exact nearest-neighbor search, and resolves context against the recovered evidence snapshot. It returns the selected generation and ranked candidate IDs/scores for audit; opaque-byte test fixtures cannot serve context. It is a binary-query storage API, not a raw-text query encoder or complete benchmark adapter.
- See [BENCHMARK_PLAN.md](./BENCHMARK_PLAN.md) for matched-comparison rules and explicit targets against the current public LongMemEval-V2 frontier.

Authored persistence tests include successful generation rollover, delete/restart non-resurrection, interrupted commits, fail-closed corrupt artifacts, lock refusal, dangling CURRENT symlink rejection, empty-index rejection, corrupt snapshot rejection, real-HdcStore generation/checksum/live-ID binding, mismatch rejection, delete-before-tombstone ordering, substituted-index rejection, retention cleanup/fail-closed behavior, and the three committed-query checks (typed generation + scope filtering, opaque fixture refusal, uninitialized-versus-corrupt distinction). They have **not** been compiled or executed in this environment. In-process failpoints are not equivalent to actual process-kill or power-cut tests.

## Intended end-to-end path

1. Ingest each official trajectory. Store its raw evidence payload and provenance in an evidence store, and assign the same stable numeric ID to its HDC representation.
2. Encode query text (and, where supported, the question image) using the chosen Broca encoder; retrieve ranked candidate IDs from HDC.
3. Resolve candidates through this contract, applying scope, temporal validity, revocation/supersession/deletion and digest checks before context is returned.
4. Pass only resolved evidence items to the official harness. Do not give the memory backend benchmark question IDs, question types, gold answers, or evaluator configuration.
5. Pin encoder, index, payload manifest, query budget, and official evaluator in the run receipt.

## Hard limitation

This increment is deliberately **not** the real LongMemEval-V2 worker. It does not ingest trajectory objects, encode raw query text into `ContinuousHV`, or wire into `MemoryBridge`. The disk coordinator and typed HDC binding are a prototype, not a qualified storage engine: tests are not yet compiled, the supported filesystem assumptions need crash/power-loss qualification, and the app-level mutation path must coordinate vector and evidence updates. The unkeyed digests detect accidental corruption and mismatched artifacts, not a malicious writer; hostile storage requires an external digest pin or signature. Restoring a source digest does not re-verify source provenance without the original canonical source bytes. Image bytes are verified at resolution time. No benchmark score or performance claim is made.

## Local validation

Once a Rust toolchain is available, run:

```sh
cargo test -p symthaea-broca --lib memory_evidence
```

Unit tests cover ranked resolution, scope/time exclusion, revocation/deletion, duplicate-ID refusal, supersession, malformed provenance/lifecycle, text digest tampering, image digest verification and path traversal, snapshot round-trip with tombstone preservation, snapshot tampering and duplicate-ID attacks, and HDC/evidence generation pairing. This environment has no `cargo`/`rustc`, so those tests are authored but not claimed as executed.
