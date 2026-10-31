# Broca event enumeration benchmark protocol

**Protocol:** BROCA-EVENT-ENUMERATION-EVAL-V1  
**Scope:** exhaustive source processing, event extraction, speaker attribution, bitemporal selection and count/action correctness.  
**Status:** evaluation contract; the runner and gold event annotations are future work.

## The two completeness boundaries

A strong claim of completeness requires two separate boundaries:

1. **External source-universe coverage:** did the connector or ingestion adapter enumerate every source object that belongs to the requested account/project/scope at the declared knowledge time? This requires a provider-specific source manifest, pagination/cursor-exhaustion receipt, source object revisions and durable-ingestion receipts.
2. **Internal evidence/extraction coverage:** did Broca enumerate every eligible record in the exact EvidenceIndex snapshot, and give every record one extraction disposition?

The new `EvidenceInventorySnapshot` proves only boundary 2. It commits all active eligible records in one exact index state and refuses truncated responses. It does not prove the connector included every external source object. Until the source-universe manifest layer is integrated, the accurate statement is **complete over this EvidenceIndex snapshot**, not complete over the user's entire external history or the real world.

Future source manifest contract:

| Field | Meaning |
|---|---|
| `connector_id` / `scope_id` | Trusted source adapter and namespace for the source universe |
| `manifest_id` / `manifest_sha256` | Immutable identifier and digest of the provider's enumerated source set |
| `known_at` | Exact provider snapshot / sync knowledge-time coordinate |
| `source_object_id` / `revision_id` / `source_sha256` | Exact source identity and content revision |
| `cursor_exhausted` | Adapter attests that no further pages remain for this query/scope |
| `authoritative_count` | Provider total count is available and reconciles to enumerated IDs, or explicitly unavailable |
| `durable_ingestion_receipt` | Confirms each manifest source is durably committed and searchable or has an explicit non-indexed disposition |

A source-manifest digest is not by itself proof of connector honesty. The connector adapter is a trust boundary; its pagination, authorization and reconciliation logic require independent conformance tests. Missing pages, unavailable authoritative counts, unsupported source types or non-durable ingestion must be reported as incomplete rather than hidden.

## Implemented internal coverage boundary

`EvidenceIndex::enumerate_active_at_snapshot` returns every active record in one scope and valid-time coordinate, independent of BM25 or semantic ranking. Its receipt binds the index-state digest, scope, valid_at, known_at, complete eligible record count, returned count, response cap and inventory digest. A cap-truncated result cannot be internally verified.

`EventExtractionBatch` then requires one row per inventory record. `Complete` with zero events means the source was processed under the extractor contract and no event was emitted. `Incomplete`, `UnsupportedSource` and `ParseFailure` are explicit failed dispositions and disable answer permission. The batch binds the exact source inventory digest and the pinned extractor and schema digests.

`EventEnumerationResult::verify_against` reruns the deterministic selection over the original inventory, extraction batch, admissions, trusted speaker attributions and query. A receipt hash alone is not an independent verifier; the original evidence objects are required.

## Trusted speaker attribution

Speaker identity and role must come from a trusted source envelope, not from the event extractor's interpretation of prose. Each `TrustedSpeakerAttribution` binds:

- evidence ID and immutable source revision;
- exact source SHA-256 and source pointer;
- trajectory/session ID;
- speaker ID and role;
- attribution receipt digest.

The event extractor's speaker fields must exactly match the trusted envelope. A user-filtered event query must require the speaker-attribution gate. If a potentially relevant event has unknown attribution, the result cannot authorize a count or answer. Assistant suggestions, tool output and external messages must never be silently treated as user-authored actions.

The receipt digest is an integrity commitment, not a digital signature. The trust anchor is the source adapter that reads the original message envelope and emits the attribution. This adapter must have its own source-format conformance tests and must preserve original role metadata.

Current limitation: one evidence record has one trusted speaker attribution. This matches the current message/step-oriented ingestion contract. A multi-speaker external document should be split into source-anchored message spans or receive a future span-level attribution model; assigning one role to an entire multi-speaker transcript is not acceptable.

## Exact event selection

Queries are typed, not natural-language substring filters. They bind event type, canonical entity, optional actor, optional speaker ID/role, half-open occurrence-time interval, scope, valid_at, known_at, and minimum session/source-family diversity.

Selection is performed over all extracted events in the complete batch. It reports matching event IDs, source records, source families, sessions and a receipt digest. Session diversity and independent source-family diversity remain separate constraints.

## Metric suite

| Metric | Definition | What failure it exposes |
|---|---|---|
| External source coverage | manifest sources durably ingested / authoritative source manifest count | Connector pagination, account-scope or ingest omissions |
| Internal inventory coverage | eligible indexed records returned / eligible record count | Top-k/cap truncation or scope/lifecycle leakage |
| Extraction disposition coverage | inventory records with exactly one extraction row / inventory record count | Silent skipped or duplicate source extraction |
| Event extraction precision / recall / F1 | predicted normalized events vs annotated gold events, matched by type/entity/actor/time/source | Semantic extraction false positives and false negatives |
| Speaker attribution precision / recall / F1 | exact speaker ID/role against source-envelope gold | User/assistant/tool/external misattribution |
| Temporal normalization error | absolute timestamp error and interval-boundary error | Wrong chronology and invalid_at errors |
| Exact count accuracy | exact predicted count vs gold count for answerable queries | Duplicate, missed or overcounted events |
| Count interval accuracy | predicted event set/count over half-open requested period | Boundary errors and date interpretation errors |
| Abstention precision / recall | abstain when inventory/extraction/speaker/proof prerequisites fail; answer when all pass | Over-abstention or unsafe closed-world claim |
| Proof/source diversity | distinct sessions and source families per required claim | One-session/one-source overcounting |
| Action correctness | selected ANSWER / WARN_STALE / EXPLAIN_CONFLICT / RETRIEVE_MORE / ABSTAIN against gold policy | Memory that retrieves facts but triggers the wrong action |
| Cost per correct count | all ingestion, indexing, extraction, retrieval, verification and reader cost / correct supported count | Hidden model/context cost and deferred compute |

Event-level recall must be reported independently from `all_rows_complete`. The latter measures source processing coverage; it does not establish that the extractor understood or found every event.

## Benchmark composition

Build a gold set with stratified tasks:

1. **Counting and aggregation:** “How many X did the user do?” across multiple sessions; positive, zero, duplicate and near-duplicate events.
2. **Speaker attribution:** user statement vs assistant suggestion vs tool result vs external content; quote/forward/role-play cases.
3. **Temporal windows:** before/after/between dates, half-open interval boundaries, timezone conversion, corrections and retroactive updates.
4. **Cross-session and source diversity:** same event repeated across sessions, distinct independent sources, two sources in one session, multiple sessions from one source family.
5. **Heterogeneous evidence:** conversation, email, documents, structured events and multimodal records. Require source-specific extraction and report unsupported-source abstention.
6. **Lifecycle and forgetting:** superseded, retracted, corrected, stale-only and conflicting facts.
7. **Long-horizon scale:** reproduce official BEAM 100K/500K/1M/10M splits and report its ten abilities; add 20M/50M only as separately named stress tests.
8. **Action transfer:** tool choices and parameters must change appropriately when earlier experiences are remembered; include MemoryArena and Mem2ActBench-style tasks.

Relevant external benchmarks:

- BEAM: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html
- GroupMemBench: https://www.microsoft.com/en-us/research/publication/groupmembench-benchmarking-llm-agent-memory-in-multi-party-conversations/
- RHELM: https://arxiv.org/abs/2605.31086
- Memora / FAMA: https://aclanthology.org/2026.findings-acl.1337/
- MemoryArena: https://proceedings.mlr.press/v306/he26am.html
- Mem2ActBench: https://aclanthology.org/2026.acl-long.370/
- Mem-Gallery: https://aclanthology.org/2026.acl-long.1892/
- EvalMem: https://aclanthology.org/2026.findings-emnlp.1134/

## Experimental arms

Compare paired queries against:

- BM25 top-k retrieval with the same index snapshot;
- vector/dense retrieval top-k;
- hybrid retrieval/reranking;
- full-context input up to the identical token budget;
- extractor + top-k without exhaustive inventory;
- exhaustive inventory + one disposition/source + speaker gate;
- exhaustive inventory + multi-pass extraction/independent event adjudication;
- an oracle event set as an upper bound for the answerer and action policy.

Use the same answerer, prompts, event schema, tokenizer, context budget, tool environment and judge wherever a head-to-head is claimed. Compare models and systems separately where an identical shared configuration is impossible.

## Prospective release targets

Targets below are engineering ambitions, not measured Broca results:

| Gate | Initial target | Hard safety rule |
|---|---:|---|
| External source-manifest reconciliation | 100% for an answerable closed-world query | no answer when cursor exhaustion or authoritative source-set status is unknown |
| Internal inventory coverage | 100% | any truncation makes count answer impermissible |
| Extraction row disposition | 100% | every inventory record has exactly one row; unsupported/parse failure blocks answer |
| Event extraction quality | ≥0.95 recall and ≥0.98 precision on adjudicated gold | report by source type and event type; no aggregate-only pass |
| Speaker attribution | ≥0.99 precision and ≥0.98 recall | zero assistant/tool-to-user misattribution in deterministic conformance tests |
| Exact count accuracy | ≥0.90 on answerable multi-session count questions | zero answer when any completeness or attribution gate fails |
| Abstention | ≥0.99 recall on deliberately incomplete/untrusted cases | don't silently convert missing coverage into zero events |
| Temporal selection | ≥0.99 exact interval-boundary accuracy | no future-known/retracted/out-of-valid-time evidence used |
| Proof soundness | zero false certificates in adversarial deterministic tests | independent verifier must recompute over immutable inputs |

Use paired question deltas and bootstrap at the independent source/conversation/user group level. Publish 95% confidence intervals, answerable/unanswerable strata and each failure type.

## Cost and scaling targets

Count all costs, not only final reader tokens: source scanning, indexing, extraction, speaker attribution, event-store maintenance, retrieval, packet rendering, proof verification, answer generation, tool calls and retries.

Initial efficiency objective: a 2× reduction in all-in context tokens versus full-history prompting at non-inferior exact count/action correctness. Report write and maintenance costs amortized across the actual query frequency; do not move extraction or connector work off-budget.

Scale curves must include wall time and storage by indexed source count, extraction event count, number of sessions, unique source families and query complexity. Measure p50/p95/p99. An O(N) inventory scan may be valid for strict count queries, but it must be recognized as O(N) and compared to maintained event indexes with snapshot consistency.

