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

The `longmemeval_v2` module parses the official V2 trajectory JSONL record shape and builds the text state-window BM25 reference index. The companion `longmemeval_v2_entity` module builds a distinct exact structured-literal index over trajectory IDs, URLs, numeric state/step values, goals and structured action fields. Ordinary prose is intentionally excluded from the literal channel, so it complements rather than duplicates BM25. The sidecar now exposes three identity-bound benchmark arms: BM25-only, entity-literal-only, and hybrid RRF (k=60, equal weights). Single-channel arms build only their selected index; the hybrid arm builds both and fuses chunk ranks by immutable chunk ID. Candidate records and trace receipts are bound to the selected backend/channel identity. Both indexes use the same centered state-window chunk IDs and slice radius. The parser preserves the full original trajectory JSON, state JSON, source state indices/steps, actions, thoughts, accessibility-tree text and screenshot references. BM25 neighbor expansion remains within the same trajectory and is not assigned inherited relevance scores by RRF.

The dataset attaches actions to destination states. The adapter keeps the action on its original destination state and describes it as the transition from the prior state; it does not shift the action backward. Web and enterprise trajectories cannot share one index.

Screenshots remain references rather than embedded pixels. This increment adds a fail-closed resolver that canonicalizes each dataset-relative screenshot path, rejects absolute paths and traversal, blocks symlinks that escape the dataset root, and requires the target to exist as a regular file. The resolved context projection returns absolute paths in the official text/image context-item shape and refuses the whole projection when any selected image is unavailable.

This is still not a complete official Python `Memory.insert/query` plugin. The BM25 component does not rank by screenshot pixels or consume `query_image`; a multimodal retriever and the thin official-harness bridge remain separate work. The official harness must continue to enforce its own reader/context-token budget, and the backend must not receive evaluator labels or question metadata.

Resource receipts report local measured index-build and query latency, source/state/accessibility/evidence-text bytes, screenshot-reference bytes, and lexical payload lower bounds. They exclude image pixels, allocator/container overhead, embeddings, LLM/reader cost, verification and persistent storage infrastructure. Sixteen deterministic LongMemEval-V2 adapter tests are authored across source preservation, state/action semantics, BM25 retrieval, path resolution, missing images, traversal, symlink escape, context projection and tamper detection. The tests have not been executed.

The sidecar's retrieval arms are fixed benchmark ablations, not learned strategy selection. The selected mode, active channel list, backend identity, entity-literal field weights (when used), extraction rule, backend binary digest and RRF configuration (when hybrid) are included in the Python bridge's backend profile commitment. Each executed action is independently recorded even when it returns no candidates.

No V2 run or score is claimed.


## DSSE-backed independent qualification exchange

The `independent_qualification_exchange` module packages the exact canonical candidate JSONL and consumer-owned use binding for an injected issuer transport. The issuer returns a DSSE v1.0.2 JSON envelope containing an in-toto Statement v1 with the MACE qualification predicate. The consumer verifies the exact PAE, signed candidate-set subject, complete DecisionNeed/use binding and pinned Ed25519 signer, then returns only a dependency-closed qualified projection or explicit abstention. The original unqualified batch is never a fallback.

The exchange samples protected consumer time after the issuer response; the request has no caller-controlled timestamp. The result retains the exact signed envelope and can reconstruct a fresh verifier from the current pinned trust profile before each subsequent use.

This is still a consumer-side library contract, not an independent qualifier or production transport. The production issuer must run under separate credentials and trust roots, independently resolve payload/source material, recompute provenance, and check rights, purpose, scope, correction/revocation and erasure state. The public test-signing key is deterministic and must never be used in production.

Protocol: [MACE independent qualification exchange v1](../../../docs/research/MACE_INDEPENDENT_QUALIFICATION_EXCHANGE_V1.md). Request schema: [mace_evidence_qualification_request_v1.schema.json](../../../data/schemas/mace_evidence_qualification_request_v1.schema.json).


## Consumer DecisionNeed sufficiency verifier

The `decision_sufficiency` module verifies a separately DSSE-signed DecisionSufficiency receipt against a qualified candidate projection and a consumer-owned `DecisionNeedV1` manifest. Load the DecisionNeed from an authorized policy registry; never accept it from candidate memory or the qualifier. The manifest digest pins the decision domain/scope/purpose, allowed action set, utility profile, risk and confidence thresholds, regret tolerance, calibration artifacts, evidence-age policy, exact requirement/evidence classes, independent-root minimums, and resource budgets.

A sufficient status is rejected unless every requirement is marked met with its exact DecisionNeed evidence-class digest, a supporting basis receipt, and at least the required number of distinct root-family commitments paired with provenance-receipt digests. The follow-on `source_lineage_receipt` module verifies the referenced DSSE/in-toto lineage receipts under a third pinned resolver profile, checks each exact envelope-byte SHA-256, binds each receipt to the same qualified evidence set, DecisionNeed, requirement, evidence class and root family, and revalidates qualification, sufficiency and lineage signatures at one use-time clock. This verifies the resolver's signed lineage statement. The `source_lineage_graph` module additionally replays the exact graph-manifest digest, checks candidate parent bindings, rejects cycles/dangling edges, and recomputes root-family and closure digests against the signed commitments. This proves only the consistency of the disclosed graph: it does not fetch/authenticate raw source bytes or prove that every real-world parent/common cause was disclosed. The independent resolver must derive lineage from source-origin evidence, collapse shared roots/common causes, and treat unknown ancestry as non-independent. The risk upper bound and regret upper bound must also be within the manifest limits, and signed telemetry must stay within cost, latency, context-token, tool-call and evidence-age budgets. Unmet/unknown requirements may truthfully carry empty basis/root lists. The evaluator role/profile/key must be distinct from the qualification authority. The DSSE in-toto Statement subject is the exact qualified-evidence-set digest, and its predicate binds the original qualification receipt plus the DecisionNeed.

At every subsequent use, callers must revalidate the original qualification using a fresh qualifier verifier and the sufficiency receipt using a fresh evaluator verifier at the same trusted current time. A sufficient receipt cannot revive expired or revoked qualification. The result is never an action permit.

This is a consumer-side verification library. It does not implement the independent sufficiency evaluator, calibration pipeline, independent telemetry measurement, held-out regret calculation, live action authorization, consume-once execution, or outcome receipt. The verifier confirms what the separate evaluator signed; it does not prove the evaluator's underlying risk model or measurements.


## Independent source-origin verification

The `source_origin_receipt` module adds a fourth, separately pinned DSSE authority after lineage-graph replay. It requires exact source bytes, evidence-payload bytes and derivation-receipt bytes for **every** replayed graph node, including ancestors. The consumer recomputes each SHA-256 and verifies one origin attestation per node, bound to the exact graph digest, node identity, parent list, derivation receipt, common-cause groups and measured byte lengths. It rejects missing/extra/duplicate nodes, changed bytes, stale signatures and reuse of the qualifier, evaluator or lineage authority profile/key.

This is a consumer-side verifier, not a fetcher or publisher-trust oracle. The origin service must independently acquire source records, validate publisher/transform metadata and attest to the exact bytes. The client verifies byte integrity and the pinned service's assertion; content hashes alone do not prove authorship, accuracy, freshness or complete disclosure of all upstream dependencies. At use time, re-run the three earlier authority checks and graph replay, re-acquire the exact bytes, then call `VerifiedSourceOriginSetV1::validate_at_use` at the same trusted time. This result is still not an action permit.

Protocol: [MACE Source-Origin Attestation V1](../../../docs/research/MACE_SOURCE_ORIGIN_ATTESTATION_V1.md). Schema: [mace_source_origin_receipt_v1.schema.json](../../../data/schemas/mace_source_origin_receipt_v1.schema.json). Assurance evaluation: [MACE Source-Origin and Provenance Assurance Benchmark V1](../../../docs/research/MACE_SOURCE_ORIGIN_ASSURANCE_BENCHMARK_V1.md). This increment is authored but not compiled or executed here; no production service or benchmark score is claimed.


## Independent source-origin witness quorum

The `source_origin_witness` module adds a consumer-owned per-node corroboration policy over the four-authority source-origin result. A 2-of-3 profile can require two distinct pinned operator trust profiles **and** two distinct consumer-declared acquisition failure domains for every graph node. Each witness binds its observation to the exact replayed graph and the exact SHA-256 of the fourth-authority origin receipt.

The verifier also requires normalized publisher identity and source revision to agree across witnesses, and requires acquisition-record digests to be distinct across operators. A source/ancestry/common-cause disagreement is a typed failure; the verifier never drops a dissenting witness and takes a majority. If a witness is unavailable, a per-node quorum may still pass only when all configured minimums remain satisfied. Otherwise the decision is insufficient.

Policy: [mace_source_origin_witness_policy_v1.schema.json](../../../data/schemas/mace_source_origin_witness_policy_v1.schema.json). Receipt: [mace_source_origin_witness_receipt_v1.schema.json](../../../data/schemas/mace_source_origin_witness_receipt_v1.schema.json). Protocol: [MACE Source-Origin Witness Quorum V1](../../../docs/research/MACE_SOURCE_ORIGIN_WITNESS_QUORUM_V1.md).

This proves consistency with independently signed observations under the consumer's trust assumptions. It does not prove that operator identities or declared failure domains are truly independent, or that any observer discovered every upstream dependency. The operator registry, independent fetch adapters, publisher metadata validation and transparency-service integration remain deployment work. Tests are authored but have not been executed.


## Provenance interoperability

The companion [MACE W3C PROV and SCITT Interoperability V1](../../../docs/research/MACE_PROV_SCITT_INTEROP_V1.md) document defines a JSON-LD context for mapping MACE source/evidence nodes and acquisition/transformation activities to W3C PROV entities, activities and agents. It also specifies a future per-issuer SCITT transparency integration. The JSON-LD context is published at [mace_prov_jsonld_context_v1.jsonld](../../../data/schemas/mace_prov_jsonld_context_v1.jsonld). An executable exporter and SCITT client are not yet implemented.


## Streaming temporal session ledger

The `streaming_session` module is an in-memory, event-sourced reference state machine for a single user/session. It separates valid/effective time from trusted observed/transaction time, applies contiguous consumer sequence numbers, supports exact mutation replay, and emits before/after state-hash receipts. Mutations include append, correction/supersession, quarantine, revalidation, revocation and a rights-erasure request.

The snapshot exposes only active messages, ordered by valid timestamp and then original ingestion sequence. Out-of-order valid-time corrections do not rewrite the monotonic observed-time history. Event-ID reuse with different mutation content, sequence gaps, duplicate turn IDs, malformed targets and consumer clock rollback fail closed. Failed transitions are staged before commit.

This is not yet a hosted AML Add/Search service, durable append-only database, distributed CAS writer, or complete rights-erasure implementation. The erasure request removes payload from this in-memory projection, but message-derived mutation commitments remain; production must separately erase derived indexes, caches, replicas and backups and verify deletion receipts. Raw JSONL bytes and publisher/source authenticity remain the responsibility of the upstream ingestion/qualification boundaries.

Protocol and benchmark: [MACE Streaming Temporal Session Ledger V1](../../../docs/research/MACE_STREAMING_TEMPORAL_SESSION_LEDGER_V1.md). Fourteen deterministic test functions are authored in the latest streaming-ledger increment; they have not been executed.


## Dependency-aware derived-memory invalidation

The `temporal_projection` module tracks summaries, embeddings, preference models, plans, fact indexes and answer caches as versioned derived projections with exact source-event and parent-projection dependencies. It consumes the streaming ledger's hash-committed invalidation receipts and propagates correction, quarantine, revocation and rights-erasure states through all descendants. A revalidated source does **not** automatically reactivate old summaries or plans; a fresh projection must be built with a new ID, current dependencies and a new derivation receipt.

The index rejects registration against stale source events or non-current parent projections, applies transitions in contiguous sequence order, and produces deterministic invalidation receipts. It is an in-memory reference, not a durable registry or a distributed invalidation service. The erasure acknowledgement clears the in-memory projection commitments only; external index/cache/replica/backup deletion must be separately executed and verified.

Protocol: [MACE Derived-Memory Invalidation V1](../../../docs/research/MACE_DERIVED_MEMORY_INVALIDATION_V1.md). Tests are authored but not executed.


## Derived projection invalidation and rebuilds

The `temporal_projection` module consumes the streaming ledger's scoped transition receipts and invalidates all derived projections that depended on affected source events. Invalidation cascades through parent-projection edges so a source correction can stale a summary and every plan/cache downstream of that summary. Quarantine, revocation and rights-erasure requests use progressively stronger non-current states; a later source revalidation does not reactivate an old derived artifact.

New projections must be built under a new ID against current source events and current parent projections. The old projection is linked to its replacement using a separate derivation/rebuild receipt commitment. An erasure acknowledgement clears the content and dependency commitments in this in-memory index only; the external deletion receipt must be independently verified by the caller.

Protocol: [MACE Derived-Memory Invalidation V1](../../../docs/research/MACE_DERIVED_MEMORY_INVALIDATION_V1.md). This is an in-memory reference index, not a durable dependency database or distributed invalidation service.


## Transitive derived-memory invalidation

The `temporal_projection` module is exported alongside the streaming ledger. It registers derived summaries, embeddings, preference models, plans, fact indexes and answer caches with exact source-event and parent-projection dependencies. A transition receipt carries both the events newly removed from the active view and the source IDs targeted by correction/quarantine/revocation/erasure, so later lifecycle escalation still reaches dependent projections even when the source had already been hidden.

The index propagates invalidation through all descendants, emits a self-committing projection-invalidation receipt, refuses new projections built from non-current parents, and supports rebuilds under new IDs. Source revalidation does not revive old projections. Rights-erasure acknowledgements clear the in-memory projection content/dependency commitments but do not prove deletion from external stores.

Protocol: [MACE Derived-Memory Invalidation V1](../../../docs/research/MACE_DERIVED_MEMORY_INVALIDATION_V1.md). Twelve deterministic projection-index tests are authored; none have been executed.
