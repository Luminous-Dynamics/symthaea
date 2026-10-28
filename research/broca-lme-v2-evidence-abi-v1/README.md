# Broca evidence payload ABI for LongMemEval-V2

**State:** Rust evidence-contract prototype plus deterministic snapshot/restore and paired-artifact commit-manifest primitives. Unit tests are authored but not compiled in this environment; this is not a completed benchmark integration.

## The concrete interface mismatch

The official LongMemEval-V2 backend API calls `insert(trajectory)` and requires `query(query, query_image)` to return a list of `{ "type": "text" | "image", "value": ... }` context items. Text must be non-empty; image paths must exist. The reader receives these items as memory context. See the [official backend contract](https://github.com/xiaowu0162/LongMemEval-V2#implementing-your-method).

The current `crates/domains/symthaea-broca/src/memory_bridge.rs` queries `HdcStore::scan_similar()` and returns blended `ContinuousHV` vectors. `crates/domains/symthaea-hdc-store/src/store.rs` describes vector entries containing metadata plus a `BinaryHV`. Neither API alone resolves a hit to answer-bearing text or an image path. Therefore a vector hit cannot be passed off as LongMemEval evidence.

## What this increment provides

`memory_evidence.rs` defines a typed payload contract, ID resolver, versioned snapshot codec, and generation-bound commit manifest:

- each evidence record binds an HDC numeric ID to a stable evidence ID, environment/scope, trajectory and step provenance, source digest, payload and payload digest; `EvidenceIndex::insert` verifies the source digest against the exact source bytes supplied by the ingestion adapter;
- active evidence is eligible only within the matching scope and its optional validity interval;
- revoked, superseded and deleted records are not returned;
- deletion keeps a tombstone in the resolver; snapshot/restore preserves deleted, revoked and superseded records and rejects a duplicate-ID reinsert after restart;
- deterministic snapshot bytes are HDC-ID ordered, schema-versioned and checksummed; restore rebuilds ID maps from records and rejects malformed digests, duplicate identities, corrupt text content, and dangling/cross-scope supersession links;
- `EvidenceCommitManifest` binds exact HDC-index bytes and evidence-snapshot bytes to one nonzero generation, detecting stale index/snapshot pairings before a query is admitted;
- text payload digests are verified before emission; image paths are canonicalized under a trusted payload root and their file-byte digests are checked;
- duplicate HDC/evidence IDs, malformed digests, invalid lifecycle shapes, path traversal, and corrupt payloads fail closed;
- resolved context items follow the official `{type,value}` envelope, and ranked HDC-hit order is retained.

## Intended end-to-end path

1. Ingest each official trajectory. Store its raw evidence payload and provenance in an evidence store, and assign the same stable numeric ID to its HDC representation.
2. Encode query text (and, where supported, the question image) using the chosen Broca encoder; retrieve ranked candidate IDs from HDC.
3. Resolve candidates through this contract, applying scope, temporal validity, revocation/supersession/deletion and digest checks before context is returned.
4. Pass only resolved evidence items to the official harness. Do not give the memory backend benchmark question IDs, question types, gold answers, or evaluator configuration.
5. Pin encoder, index, payload manifest, query budget, and official evaluator in the run receipt.

## Hard limitation

This increment is deliberately **not** the real LongMemEval-V2 worker. It does not ingest trajectory objects, encode raw query text into `ContinuousHV`, or wire into `MemoryBridge`; no durable storage engine or disk transaction is implemented. Snapshot/restore gives the adapter a deterministic format and tests in-memory restart reconstruction, but callers must still atomically publish the HDC index and evidence snapshot under a commit manifest and fail closed on mismatches. The unkeyed digests detect accidental corruption and mismatched artifacts, not a malicious writer; hostile storage requires an external digest pin or signature. Restoring a source digest does not re-verify source provenance without the original canonical source bytes. Image bytes are verified at resolution time. No benchmark score or performance claim is made.

## Local validation

Once a Rust toolchain is available, run:

```sh
cargo test -p symthaea-broca --lib memory_evidence
```

Unit tests cover ranked resolution, scope/time exclusion, revocation/deletion, duplicate-ID refusal, supersession, malformed provenance/lifecycle, text digest tampering, image digest verification and path traversal, snapshot round-trip with tombstone preservation, snapshot tampering and duplicate-ID attacks, and HDC/evidence generation pairing. This environment has no `cargo`/`rustc`, so those tests are authored but not claimed as executed.
