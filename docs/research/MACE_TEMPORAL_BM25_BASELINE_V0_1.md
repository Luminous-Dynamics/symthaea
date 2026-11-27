# MACE temporal BM25 and rank-fusion reference baseline v0.1

**Status:** standalone Rust reference and synthetic candidate-visible fixture authored. Exact-head Rust 1.96 workflow is the execution gate. This is not a production Add/Search service, an external leaderboard reproduction, or a claim of superiority over current memory systems.

## Why this exists

The MACE paired decision evaluator accepts candidate-retrieved evidence IDs, but those IDs do not by themselves constitute a competitive retrieval baseline. In particular, the existing typed Experience fixture is not a general natural-language trajectory corpus. Scoring that fixture with a token counter would not reproduce BM25, dense retrieval, or a hybrid Add/Search competitor.

This increment introduces a separate raw-text candidate-visible surface so lexical baselines can be measured honestly without changing the existing paired decision evaluator or its hidden evaluator-only truth.

## Reference contract

The standalone implementation is `research/mace-decision-sufficient-memory-bench-v0-1/baselines/bm25_temporal_reference.rs`.

A source document carries:
- stable candidate-visible `document_id`;
- `scope_id` for tenant/run/purpose filtering;
- `observed_at_ms` as knowledge/transaction time;
- `valid_from_ms` and exclusive `valid_until_ms` as valid time;
- verbatim source text.

A query carries `scope_id`, `as_of_ms`, free-text query and `top_k`. A record is eligible only when its scope matches, it was observed no later than the query cut, and its valid-time interval contains that cut.

Crucially, BM25 corpus statistics (eligible document count, document frequency and average document length) are recomputed over the eligible scope/time cohort. Future documents and other scopes therefore do not change the score ordering through corpus statistics. The work receipt reports eligible cohort and scored-posting counts only; it does not pretend to report wall-clock or measured hardware cost.

## BM25 definition

The reference uses:
- (k_1=1.2) and (b=0.75) defaults;
- positive Robertson-style IDF (\log(1+(N-df+0.5)/(df+0.5)));
- standard BM25 term-frequency and length normalization;
- deterministic Unicode-aware lowercase tokenization over alphanumeric, underscore and hyphen characters;
- stable score-descending, document-ID-ascending tie-breaking.

No stemming, learned vocabulary, stopword list, query expansion, embeddings or reranking are included. Those are explicit future arms, not hidden assumptions.

## Rank fusion

`weighted_rrf_v1` composes externally produced ranked channels using (w_c/(k+r)), with one-based ranks. It validates:
- unique channel IDs;
- positive finite weights and a nonzero RRF constant;
- no duplicate document within a channel;
- all returned IDs resolve to the canonical candidate-visible index;
- every fused ID is still eligible at the same scope and as-of cut.

It can compose lexical, entity, date or graph result lists, but this reference does not implement those channels. Dense retrieval and cross-encoder reranking remain separate benchmark arms.

## Negative controls

The Rust suite covers:
- multi-term BM25 ranking and partial matches;
- knowledge-time and valid-time cutoffs;
- exclusive valid-until semantics;
- scope isolation, including invariance of in-scope scores when another scope gains documents;
- empty queries and zero top-k;
- deterministic tie order independent of ingest order;
- non-ASCII tokenization examples;
- duplicate IDs, invalid time intervals and invalid BM25 parameters;
- complementary-channel RRF behavior;
- duplicate channel, duplicate result, invalid weight, unknown ID, wrong-scope and expired-document rejection.

The JSON example is synthetic protocol data only. It is not a real-world conversational history or a qualified benchmark corpus.

## Competitive baseline plan

Compare matched task sets, reader, token/context budget, hardware and latency protocol:

1. no memory and full-history diagnostic ceiling;
2. BM25 lexical retrieval (this reference);
3. sparse/dense hybrid RRF;
4. entity/date/literal and neighboring-context channels;
5. causal-graph retrieval;
6. state-first temporal retrieval;
7. typed-HDC and HDC-LTC variants;
8. evidence-qualified retrieval and minimal sufficient context;
9. evaluator-only oracle evidence selector.

For every arm report retrieval recall/precision, decision-support precision/recall, paired decision gain/harm, memory/query context bytes, index/retained bytes, write and maintenance costs, p50/p95 query latency, and stale/revoked/cross-scope violations. Never collapse hard privacy or authority failures into a weighted accuracy score.

External anchors must remain track-specific. AML Cycle 2 publishes separate textual, coding and multimodal boards; its open-method textual leader is Resurface (61.22), while its multimodal leader is Khipu (43.26). LongMemEval-V2 measures context-gathering accuracy and query latency with a fixed reader and a log-latency accuracy frontier. TimlMemBench studies governed activation and lifecycle use; MemSecBench follows poisoned memory through persistence, execution and repair. These benchmarks complement rather than replace MACE's paired decision-utility evaluation.

## Claim boundary and next work

This increment establishes a transparent lexical comparator and validated time/scope filters. It does not reproduce Resurface/Khipu/InsightMem, does not establish an AML score, and does not show SOTA superiority. The next integration step is an adapter from the same canonical source-event projection used by the paired evaluator; until then, the BM25 run and paired decision run must be joined by exact document/source IDs and frozen source digests, not inferred from text similarity.
