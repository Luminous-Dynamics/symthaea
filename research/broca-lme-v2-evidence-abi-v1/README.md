# Broca evidence payload ABI for LongMemEval-V2

**State:** Rust evidence-contract prototype with unit tests authored; not yet compiled in this environment and not a completed benchmark integration.

## The concrete interface mismatch

The official LongMemEval-V2 backend API calls `insert(trajectory)` and requires `query(query, query_image)` to return a list of `{ "type": "text" | "image", "value": ... }` context items. Text must be non-empty; image paths must exist. The reader receives these items as memory context. See the [official backend contract](https://github.com/xiaowu0162/LongMemEval-V2#implementing-your-method).

The current `crates/domains/symthaea-broca/src/memory_bridge.rs` queries `HdcStore::scan_similar()` and returns blended `ContinuousHV` vectors. `crates/domains/symthaea-hdc-store/src/store.rs` describes vector entries containing metadata plus a `BinaryHV`. Neither API alone resolves a hit to answer-bearing text or an image path. Therefore a vector hit cannot be passed off as LongMemEval evidence.

## Deterministic lexical baseline added in a stacked research change

The follow-on branch adds `EvidenceIndex::search_bm25`, a model-free lexical retrieval baseline directly over eligible text evidence. It uses deterministic Unicode alphanumeric tokenization, corpus-relative BM25 term weighting, stable tie-breaking by HDC ID, and filters scope/validity/lifecycle before calculating document frequencies. Text digests are checked before a document affects ranking. The result includes a score and provenance metadata for the experiment receipt, while the adapter should send only each hit's `context` to the reader.

This baseline is intentionally not called semantic retrieval or HDC retrieval. It makes a reproducible comparison possible before adding a text embedding model or a separately trained text-to-HDC encoder. It also provides a lower-complexity control for evaluating whether more elaborate memory architecture provides measurable value.

Configuration defaults are `k1=1.2`, `b=0.75`, `max_items=5`. These are conventional baseline starting points, not tuned LongMemEval settings. Tune only on development data, freeze before held-out runs, and report all configs in the run receipt.

The companion tests cover relevance ranking, exact stable ties, scope/lifecycle filtering before scoring, zero-hit query behavior, invalid configuration and corrupted evidence rejection. They are authored but uncompiled in this environment.

## The query encoder is also an explicit gap

The `crates/domains/symthaea-broca/src/encoder.rs` module exposes `ThoughtLanguageEncoder::encode(&ThoughtChannels)`. `ThoughtChannels` is a structured scalar-channel representation, not arbitrary query or trajectory text. It must not be mistaken for a text embedding model.

The next worker must select and pin a real retrieval representation before a full run:
- a deterministic lexical retrieval baseline (e.g. BM25 or token/phrase postings) with no embedding-model dependency;
- a pinned text embedding model, if allowed by the hardware/cost plan; or
- a separately trained and validated text-to-HDC encoder with its own retrieval benchmark.

Report these as distinct retrieval arms. Do not label a lexical feature hash as semantic HDC, and do not use answer/question labels or benchmark metadata to build the query representation.

## What this increment provides

`memory_evidence.rs` defines a typed payload contract and a process-local ID resolver:

- each evidence record binds an HDC numeric ID to a stable evidence ID, environment/scope, trajectory and step provenance, source digest, payload and payload digest; `EvidenceIndex::insert` verifies the source digest against the exact source bytes supplied by the ingestion adapter;
- active evidence is eligible only within the matching scope and its optional validity interval;
- revoked, superseded and deleted records are not returned;
- deletion keeps a tombstone in the resolver so an old candidate ID cannot silently resurrect its payload;
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

This increment is deliberately **not** the real LongMemEval-V2 worker. It does not ingest trajectory objects, encode raw query text into `ContinuousHV`, persist the payload map/tombstones, or wire into `MemoryBridge`; the resolver is process-local. The source digest must be computed over the exact canonical source bytes supplied by the adapter; the contract does not silently reserialize JSON. A production implementation must atomically persist the vector-to-payload mapping and lifecycle changes, and tests must demonstrate that deletion/revocation survive process restart and index rebuild. No benchmark score or performance claim is made.

## Local validation

Once a Rust toolchain is available, run:

```sh
cargo test -p symthaea-broca --lib memory_evidence
```

Unit tests cover ranked resolution, scope/time exclusion, revocation/deletion, no resurrection on duplicate IDs, supersession, malformed provenance/lifecycle, text digest tampering, image digest verification, and path traversal. This environment has no `cargo`/`rustc`, so those tests are authored but not claimed as executed.
