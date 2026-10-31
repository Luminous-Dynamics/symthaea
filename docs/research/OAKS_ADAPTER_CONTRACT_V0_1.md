# OAKS Continual-Knowledge Adapter Contract (v0.1)

**Status:** adapter protocol only; no OAKS data downloaded, transformed, or evaluated  
**Related benchmark:** NEURO-SILICON-007 / #9394 / draft PR #9395  
**Structured counterpart:** T10 bitemporal knowledge stream in `research/temporal-bench/src/knowledge_stream.rs`  
**Audit date:** 2026-10-31

## 1. Purpose

OAKS (Online Adaptation to Continual Knowledge Streams) is a natural-language benchmark for tracking facts as a story/context stream evolves. Unlike static long-context QA, every question has a time-indexed answer trajectory. The same question is evaluated after successive context chunks, allowing measurement of how quickly a system adapts when facts change.

T10 in the Symthaea temporal suite isolates the underlying two-clock semantics with exact structured events:
- valid time: when a fact applies in the modeled world;
- transaction/knowledge time: when the memory system received the fact or correction.

OAKS tests the language-level expression of continual adaptation; T10 tests exact `(valid_at, known_at)` state semantics. Their scores must remain separate. A T10 pass does not imply OAKS QA accuracy, and an OAKS aggregate score does not prove transaction-time correctness.

## 2. Pinned external artifacts

### Evaluation/code repository

- Repository: [kaistAI/OAKS](https://github.com/kaistAI/OAKS)
- Inspected latest commit: `c1a76e66b5b2276973ee8ee3de7063721e22a88f`
- README blob: `3f72afe9cb612c0a6d71f98101dcd94ad539ed6a`
- ACL Anthology paper: [2026.acl-long.1956](https://aclanthology.org/2026.acl-long.1956/)
- DOI: `10.18653/v1/2026.acl-long.1956`
- The public README describes a chunk-by-chunk evaluator using the same question set at every interval, base full-history accumulation with truncation at context limits, a rolling recent-chunk variant, and a RAG retrieval variant.
- A `LICENSE` file was not found at the inspected repository path; do not assume the code is open-source or that a missing license grants reuse rights. Obtain/confirm the code license before redistribution or dependency use.

### Dataset repository

- Repository: [adobe-research/OAKS](https://github.com/adobe-research/OAKS)
- Inspected latest commit: `0bbc4b3877eae5f9335f07c5e485aabf1d9136e9`
- README blob: `3132750f6b0f52aade75709e99274370f96b7d1a`
- License blob: `b0def00fcdb416f7542ca07504b74687677f3104`
- The Adobe Research License grants use for noncommercial research and teaching and excludes commercial product development/distribution and other activity resulting in commercial gain. Do not include the dataset in commercial product work or commercial training/evaluation without a separate written license. Keep dataset bytes out of Symthaea public repositories and CI artifacts.

### Manifest discrepancy to resolve first

The inspected `kaistAI/OAKS` code README reports 435 OAKS-Novel multiple-choice questions; the `adobe-research/OAKS` dataset README reports 870. Do not silently choose one count. Before reproducing any paper result:
1. enumerate the exact checked-in JSON files at the pinned data commit;
2. count documents, unique `question_id` values, per-chunk gold rows, and multiple-choice options separately;
3. explain whether 435 counts underlying questions and 870 counts repeated/evaluated items, or whether the snapshots differ;
4. pin the reconciled manifest digest and the official evaluator version.

The OAKS-BABI README describes 1,200 questions, about 65 context chunks per document, about 4.7 answer changes per question, and about 128k tokens of average context. OAKS-Novel is described as using 19 public-domain novels and about 71 chunks per document, with the README count discrepancy above. Treat these as source-reported metadata pending manifest verification.

## 3. Data contract and mapping

Each document/story contains:
- a stable document/book ID and a chunk count;
- ordered context chunks indexed by chunk number;
- a question set;
- for each question, `chunk_to_answer[t]`, the correct answer after consuming context through chunk (t);
- for OAKS-Novel, multiple-choice options and per-option supporting evidence spans/chunks.

The adapter must preserve the original document and question IDs. It must not flatten away the chunk axis.

### Required output schema

For each `document_id × question_id × chunk_index`, write exactly one prediction row:

- document ID;
- question ID;
- chunk index;
- model/memory configuration digest;
- prediction text or selected option label;
- explicit abstain/unknown state if supported by the official evaluator;
- cited evidence chunk IDs/spans if the model produces them;
- input bytes/tokens supplied at this step;
- persistent memory bytes before/after update;
- update compute time, query compute time, and end-to-end latency;
- prediction artifact digest.

The evaluator joins these rows with the official `chunk_to_answer[t]` labels. No gold answer, future chunk, future evidence span, or future question-specific annotation may enter the candidate process.

## 4. Required baseline arms

Reproduce compatible official baselines at the pinned commit/configuration, keeping model and hardware constraints explicit:

1. Full-history accumulation with the official truncation rule.
2. Rolling recent-(N)-chunk context using the official rolling setting.
3. RAG retrieval over prior chunks with the official retriever/index procedure.
4. Symthaea temporal memory: incremental update, selective retention, and explicit correction/retraction state.
5. HDC-only associative retrieval without a bitemporal version ledger.
6. Current-state key/value memory with overwrite-on-update.
7. Recency/decay-only memory.
8. Append-only bitemporal ledger with exact `(valid_at, known_at)` queries.

The last five are Symthaea research baselines, not official OAKS submissions. Label them accordingly. Keep parameter/model family, prompt, tokenizer, context budget, retrieval top-k, update frequency, answer-generation budget, and hardware boundary pinned.

The RAG pipeline must include index construction and update costs. Do not benchmark only the query step while excluding indexing, refresh, embedding, and retrieval preparation.

## 5. Metrics

### Official task accuracy

Use the official per-chunk evaluator and report:
- accuracy by chunk index;
- aggregate accuracy using the official aggregation rule;
- per-question answer trajectory;
- OAKS-BABI question-type strata (tracking, counting, bridge, comparison);
- OAKS-Novel option accuracy and evidence-support accuracy where officially defined.

### Temporal adaptation

For each known fact change:
- adaptation lag in chunks until the first correct post-change answer;
- stale-answer duration after the change;
- update detection precision/recall;
- missed update rate;
- false update rate when facts did not change;
- answer oscillation rate after the correct update;
- performance on unchanged facts after nearby distractor updates.

### Proposed Temporal Error Debt extension (not an official OAKS metric)

Tracked separately in NEURO-SILICON-008 (#9492). For a given document/question, hold each answer state until the next declared chunk/checkpoint. Integrate the duration for which the emitted state is wrong after a fact change:

- wrong-belief exposure: time with an explicit but incorrect answer/no-fact state;
- abstention exposure: time with an explicit abstention;
- missing exposure: time without a prediction by the checkpoint deadline.

Use source timestamps where available. If only chunk order is available, use chunk-index units and do not convert these to seconds. Report the three exposure components independently, alongside time-to-first-correct-answer. A late answer, an incorrect answer, abstention, and missing output must remain distinguishable. This is a proposed extension to the adapter analysis, not part of the official OAKS score or a claim about OAKS authors' metric definitions.

The required invariant is that equal aggregate per-chunk accuracy can still produce different wrong-belief exposure when one model updates faster. Never combine these into an opaque weighted score.

### Evidence/provenance

Where the official evidence labels permit:
- supporting-evidence recall/precision;
- unsupported answer rate;
- stale-source use after correction;
- future-chunk leakage negative control;
- consistency between selected answer and cited evidence.

### Resource accounting

- input tokens/bytes per chunk;
- total context tokens over the stream;
- persistent memory bytes and peak working set;
- memory read/write bytes and index-update cost;
- update compute and query compute separately;
- p50/p95/p99 per-chunk latency;
- full-stream wall time;
- calibrated energy per full document/question stream on named hardware;
- energy per correct post-change answer.

Do not report energy from simulator estimates as hardware measurements. Do not conflate model-only inference cost with full stream ingestion/update/query cost.

## 6. Leakage and split discipline

- Treat the document/book as the grouping unit. No chunks from one document may appear in different data splits.
- Freeze development, calibration, and confirmatory document/question sets before tuning.
- A candidate receives only chunks through the current index (t).
- The official per-chunk gold trajectory stays evaluator-only.
- Do not use later chunk content, future answer labels, evidence annotations, or test-set metadata for memory initialization, prompt selection, hyperparameter tuning, or retrieval index construction.
- Preserve all failed, malformed, timed-out, abstained, and incomplete prediction rows.
- Run an explicit future-chunk canary: add a distinctive answer-changing fact only to a later chunk and prove that earlier predictions are byte-identical.
- Report both the exact official evaluator score and the separate temporal adaptation metrics; do not create a private aggregate replacing the official score.

## 7. Relationship to the structured T10 benchmark

T10 is the exact-oracle unit test for two-clock state:
- late correction: same valid time, different `known_at`, different answer;
- delayed retraction: pre-retraction historical knowledge remains queryable; post-retraction eligible state abstains;
- earlier valid interval: a later correction does not change facts outside its valid-from boundary.

OAKS adds natural-language complexity, long context, distractors, bridge/counting/comparison reasoning, and repeated per-chunk answer evaluation. Use it only after T10's exact controls are stable and the external license permits the intended research use.

## 8. Execution status and claim boundary

- Public repository metadata and README/license snapshots were inspected and pinned.
- No OAKS dataset was downloaded, opened, copied, transformed, or evaluated in this work.
- No model was run on OAKS.
- The OAKS-Novel question-count discrepancy is unresolved.
- The code repository license was not found at the inspected path and must be confirmed before code reuse.
- No OAKS/Symthaea comparative score is claimed.
- Temporal Error Debt is proposed in #9492; no time-integrated stale-belief exposure scorer is implemented in this adapter contract yet.

A future result can establish performance only for the exact OAKS release, licensed use, split, evaluator, model/runtime, prompt, budget, and hardware boundary. It does not establish general temporal reasoning or transaction-time correctness outside the measured benchmark.
