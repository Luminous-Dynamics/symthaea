# MACE raw-session BM25 reference adapter v0.1

**Status:** executable source implementation and authored unit tests; uncompiled/unexecuted in this environment. This is a local lexical reference, not a hosted AML Add/Search service, dense retriever, production privacy boundary, or benchmark score.

## Purpose

This adapter takes one raw conversational session (one JSON object per line), preserves verbatim user/assistant messages and turn metadata, deterministically segments it using the provider-neutral Textual chunk contract, and builds a user-scoped BM25 reference index.

The source JSONL contract is in `data/schemas/mace_raw_session_v0_1.schema.json`.

## Invariants

- Every input event must use schema version 1 and the exact allowlisted fields.
- A parsed batch contains one user and one session only; turn IDs must be unique.
- Exact source text, role, timestamp, turn ID, raw-line digest and whole-session digest are retained.
- Chunking uses the existing injected frozen word counter and the AML Textual limits (20 messages / 2,000 words); it never rewrites source messages.
- Chunk identity binds user, session, ordinal, frozen word-counter version, raw-session SHA-256, Add-payload digest, turn IDs and their individual source-line digests.
- BM25 normalization is used only for the derived index. Original text remains unchanged in the returned evidence chunk.
- The index is user-scoped. Cross-user ingestion and search are rejected.
- Neighbor expansion is session-local and returned separately from scored hits. Neighbor chunks are not assigned fabricated relevance scores.
- Reciprocal-rank fusion accepts externally ranked lists and validates uniqueness/weights. It is ready to fuse a pinned dense backend ranking with BM25, but this crate does not implement or claim a dense embedding model.
- The BM25 index emits a resource receipt with measured index-build latency, measured per-query latency, source-text bytes, derived index-text bytes and explicit lexical/index metadata payload lower bounds. The byte estimate excludes allocator, tree-node and container overhead, process RSS, filesystem/database overhead, replicas, embeddings and external verification work; it must not be presented as all-in memory cost.

## Usage

```rust
use broca_aml_adapter::raw_session::{
    chunk_raw_session, Bm25ProfileV1, RawSessionBm25IndexV1, RawSessionV1,
};

let session = RawSessionV1::parse_jsonl(raw_session_jsonl)?;
let chunks = chunk_raw_session(&session, "session-add-v1", &frozen_word_counter)?;
let index = RawSessionBm25IndexV1::build(
    session.user_id.clone(),
    chunks,
    Bm25ProfileV1::default(),
)?;
let result = index.search(&session.user_id, "the user's query", 5, 1)?;
let resource_receipt = index.resource_receipt();
let query_latency_ns = result.search_latency_ns;
```

The concrete word counter must be pinned to the exact AML adapter implementation. The included unit-test counter is intentionally only a fixture.

## Competitive protocol

This enables the raw-session side of the MACE baseline matrix:

1. BM25 over verbatim topic/turn chunks.
2. Hybrid BM25 + dense via `reciprocal_rank_fusion` once a pinned embedder supplies a dense ranking.
3. Neighbor-expanded context, with neighbors separate from scored hits.
4. Raw evidence plus extracted facts as a combined arm, once a pinned fact extractor is added.
5. Fixed reader/judge/backbone and all-in cost/latency measurements.

It does **not** yet reproduce Redis Remis + Instruct, Agent Zero Memory, AML Cycle 2, LongMemEval or LoCoMo. Those require their exact chunking, embeddings, prompts, extraction, judge, datasets and resource accounting. The current module is a fair, transparent lexical baseline component—not an SOTA result. It is a reusable library; it is not yet wired into the canonical #13310 decision-sufficiency frontier runner.

## Tests

Fourteen deterministic tests are authored for raw text fidelity, strict schema/unknown-field handling, duplicate IDs, session/user isolation, stable chunking, source mutation identity, chunk-metadata tamper detection, BM25 matching, neighbor isolation, lower-bound resource receipts, measured latency fields, unsupported tokenizer-version rejection, invalid input and RRF. They have not been executed here.


## LongMemEval-V2 trajectory adapter

The sibling `longmemeval_v2` module parses the official V2 trajectory JSONL record shape and builds a text-only state-window BM25 reference index. It preserves the full original trajectory JSON, state JSON, source state indices/steps, actions, thoughts, accessibility-tree text and screenshot references. Each candidate chunk is a centered state window; neighbor expansion remains within the same trajectory.

The dataset attaches actions to destination states. The adapter keeps the action on its original destination state and describes it as the transition from the prior state; it does not shift the action backward. Web and enterprise trajectories cannot share one index.

Screenshots remain references rather than embedded pixels. This increment adds a fail-closed resolver that canonicalizes each dataset-relative screenshot path, rejects absolute paths and traversal, blocks symlinks that escape the dataset root, and requires the target to exist as a regular file. The resolved context projection returns absolute paths in the official text/image context-item shape and refuses the whole projection when any selected image is unavailable.

This is still not a complete official Python `Memory.insert/query` plugin. The BM25 component does not rank by screenshot pixels or consume `query_image`; a multimodal retriever and the thin official-harness bridge remain separate work. The official harness must continue to enforce its own reader/context-token budget, and the backend must not receive evaluator labels or question metadata.

Resource receipts report local measured index-build and query latency, source/state/accessibility/evidence-text bytes, screenshot-reference bytes, and lexical payload lower bounds. They exclude image pixels, allocator/container overhead, embeddings, LLM/reader cost, verification and persistent storage infrastructure. Sixteen deterministic LongMemEval-V2 adapter tests are authored across source preservation, state/action semantics, BM25 retrieval, path resolution, missing images, traversal, symlink escape, context projection and tamper detection. The tests have not been executed.

No V2 run or score is claimed.
