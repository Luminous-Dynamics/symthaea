# Broca closed-world event enumeration

**Protocol:** BROCA-CLOSED-WORLD-EVENTS-V1  
**Status:** source inventory, extraction coverage verifier and event enumerator authored; Rust tests authored, not compiled here.  
**Motivation:** counting, multi-hop, speaker-attributed and cross-session questions must not be answered from a plausible top-k subset.

## The failure mode

Top-k retrieval answers the question “what did this ranker return?” It does not answer “did we examine every source that could contain a matching event?” This distinction matters for counts, “all of them” questions, chronology, user-versus-assistant attribution and cross-session aggregation. A ranker may surface five plausible events while omitting a sixth. The retrieved packet can be internally sound and still be incomplete relative to the source corpus.

Broca therefore separates **retrieval relevance** from **inventory completeness**. The closed-world path first commits to the entire eligible source inventory at one exact EvidenceIndex state, then requires one extraction disposition for every inventoried source, then evaluates a typed query over the complete extracted event set.

## Stage 1: exact evidence inventory

`EvidenceIndex::enumerate_active_at_snapshot(scope, valid_at, known_at, payload_root, max_items)` scans all active records in scope and valid time. It does not use BM25 or semantic ranking. The receipt binds:

- exact index-state digest and knowledge-time coordinate;
- scope and valid-time coordinate;
- full eligible inventory digest, ordered by HDC ID;
- total eligible record count, returned record count and configured response cap;
- a complete flag.

If the eligible count exceeds the cap, the result is explicitly incomplete. `EvidenceInventorySnapshot::verify_internal_consistency()` rejects truncated inventories. Increasing the cap and rerunning the same immutable snapshot is required; the prefix must never be treated as a closed world.

The inventory materializes text and image context items only after the underlying EvidenceRecord has passed its normal digest and path checks. The inventory digest commits the original EvidenceRecord, not a generated summary.

## Stage 2: one extraction disposition for every source

`EventExtractionBatch` is bound to the exact inventory digest, extractor ID/digest and extraction-schema digest. It must contain exactly one row for every evidence ID. Each row binds the source payload digest and source bytes digest. An empty event list with status `Complete` means the extractor examined the record and found no event under the pinned extraction contract. `Incomplete`, `UnsupportedSource` and `ParseFailure` are not equivalent to an empty result.

The validator rejects missing rows, extra rows, duplicate rows, duplicate event IDs, payload/source digest mismatches and admissions from another EvidenceIndex snapshot. The source admission must match the immutable trajectory, step, pointer, scope and validity fields. It must be authorized for the requesting principal and current at the inventory's known_at coordinate.

Session identity is derived from the trusted admission's source trajectory, after requiring that trajectory to match the exact indexed EvidenceSource. In the AML conversation adapter, trajectory_id is populated from Add.session_id. Source-family identity comes from the trusted admission envelope; it remains separate from session identity.

## Stage 3: typed event query

`EventEnumerationQuery` includes an explicit event type, canonical entity ID, optional actor, optional speaker ID/role, a half-open occurrence-time interval, and minimum source-family/session coverage. Event selection uses exact typed equality and a deterministic time filter; it is not a second semantic retrieval stage.

For speaker-conditioned queries, an event with unknown speaker identity/role that otherwise matches the event/entity/actor/time predicate is counted as potentially relevant but unattributed. That makes `speaker_attribution_complete = false` and disables answer permission. A known assistant/tool event does not count as user-authored activity. This prevents assistant advice from being silently reinterpreted as a user action.

## Receipt and answer gate

The receipt binds the inventory digest, extraction batch digest, canonical query digest and sorted matching-event-ID digest. It reports source rows examined, extraction completeness, matching event count, unattributed relevant event count, distinct sessions and distinct source families. `answer_is_permitted` is true only when:

1. the evidence inventory is complete and internally consistent;
2. exactly one extraction row exists for each inventory source;
3. every extraction row has status `Complete`;
4. required speaker attribution is complete;
5. minimum distinct-session and independent-source-family constraints are met.

Before the event count is used, call `EventEnumerationResult::verify_against` with the original inventory, batch, admissions and query. This reruns the derivation. SHA-256 digests are integrity commitments, not signatures or proof of semantic truth.

## What this does and does not prove

**It proves source-processing coverage:** every evidence record in the exact inventory has one explicit extraction disposition; event matches were selected deterministically from that full batch; scope, bitemporal coordinates, source provenance, authorization and speaker attribution are bound into the computation.

**It does not prove semantic extraction recall:** an extractor could omit a real event from an otherwise `Complete` row. That is an empirical extraction false-negative. The benchmark must compare extracted event sets against annotated gold events and report event-level precision/recall, missed-event rate by source type, speaker attribution precision/recall and temporal normalization error.

A stronger future release can add independent second-pass extraction, span-level gold evidence and contradiction-driven adjudication. Do not call a source-inventory receipt a proof that an LLM understood every sentence.

## Qualification tests

The synthetic Rust tests cover:

- full inventory includes low lexical-relevance documents and excludes wrong scope, revoked and out-of-valid-time records;
- a response cap is explicitly incomplete and cannot be verified;
- inventory digest changes with scope/time or record mutation;
- all source records require exactly one extraction row;
- extractor failures disable answer permission;
- user-authored event counts exclude assistant suggestions;
- unknown speaker attribution forces abstention;
- changed snapshot, bitemporal metadata or receipt contents are rejected.

These are conformance tests only. No BEAM, GroupMemBench, Mem2ActBench or user-memory benchmark has been executed by this branch.

## Integration targets

1. BEAM multi-session counting/compositional questions and cross-session event ordering.
2. GroupMemBench speaker/asker-grounded belief attribution and term ambiguity.
3. Memora/FAMA stale-value and supersession cases.
4. RHELM cross-source conversation/document/email aggregation.
5. Mem2ActBench / MemoryArena tool-use tasks where an omitted past event changes the chosen action.

