# Broca memory benchmark comparison and decision gates

**Version:** 2026-10-28  
**Status:** preregistered comparison plan; no Broca benchmark score is claimed.

## 1. Research conclusion from the current field

The strongest paradigm shift is to treat memory as a causal state-transition and intervention system, not merely a vector database.

- LongMemEval-V2 asks a memory backend to ingest long multimodal agent trajectories and return compact evidence for a fixed reader; its five abilities include static-state recall, dynamic-state tracking, workflow knowledge, environment gotchas, and premise awareness. Its published page reports 451 questions and histories reaching 25M–115M tokens.
- Mem2ActBench tests whether memory is applied to tool selection and argument grounding, not just recalled as an answer.
- Memora/FAMA explicitly penalizes reuse of obsolete or invalidated memories.
- MemoryArena tests interdependent multi-session tasks in which prior actions/feedback must be distilled and later used in a multi-session agent–environment loop.
- AMA-Bench emphasizes causal/objective information and reports that causal-graph-plus-tool retrieval outperforms its strongest baseline by 11.16 percentage points in its own setup.

These results support three different evaluations—evidence retrieval, mutation/governance, and action transfer. They do not establish that a single score or model comparison is valid across benchmarks.

References:
- LongMemEval-V2: https://xiaowu0162.github.io/longmemeval-v2/
- Mem2ActBench: https://aclanthology.org/2026.acl-long.370/
- Memora / FAMA: https://arxiv.org/abs/2604.20006
- MemoryArena: https://proceedings.mlr.press/v306/he26am.html
- AMA-Bench: https://proceedings.mlr.press/v306/zhao26bs.html

## 2. Primary benchmark: LongMemEval-V2

### Official released reference points

The official benchmark page currently lists these reference results:

| Method | Small accuracy | Small query latency | Medium accuracy | Medium query latency |
|---|---:|---:|---:|---:|
| No retrieval | 1.3% | 0 s | 1.3% | 0 s |
| RAG: query to slice | 42.8% | 0.1 s | 38.1% | 0.1 s |
| RAG: query to slice + notes | 51.0% | 0.2 s | 45.9% | 0.3 s |
| AgentRunbook-R | 58.6% | 26.9 s | 57.0% | 25.8 s |
| Codex | 69.9% | 177.2 s | 68.7% | 185.8 s |
| AgentRunbook-C | 74.9% | 108.3 s | 70.1% | 139.9 s |

Source: https://xiaowu0162.github.io/longmemeval-v2/ (accessed 2026-10-28). Pin the benchmark source commit and run configuration in every receipt; these published values are contextual reference points, not matched measurements of Broca.

The released LAFS reference frontier uses RAG slice+notes, Codex, AgentRunbook-R and AgentRunbook-C. Keep this frontier unchanged. The current upstream source also contains AgentRunbook-C V2; measure it as a separate supplementary arm rather than silently substituting it for the historical C point.

### Required baseline arms

Run all arms on identical official question sets, reader model/revision, temperature, evaluator, context-token cap and output constraints:

1. no_retrieval;
2. official rag_query_to_slice;
3. official rag_query_to_slice_notes;
4. broca_bm25 at max_items = 1, 3, 5, 10 (BM25 k1=1.2, b=0.75 fixed before held-out evaluation);
5. AgentRunbook-C V2 as a separately named supplementary result.

Use the official evaluator and LAFS calculator. Report web and enterprise separately, then the official aggregation. Do not tune thresholds or BM25 parameters on the test split.

### Primary acceptance gates

- **Integrity gate:** exact official evaluator/tooling and dataset revision pinned; all question receipts accounted for; no unresolved or filtered failures silently discarded.
- **Frontier gate:** strictly positive official ΔLAFS against the unchanged released reference frontier.
- **Paired quality gate:** at least +3 percentage points in task success at a matched budget, with grouped paired-bootstrap 95% CI excluding zero; this is an internal project criterion, not an external standard.
- **Alternative efficiency gate:** at least 25% lower total lifecycle cost while quality is within a preregistered 1-point non-inferiority margin.
- **Governance gate:** zero observed critical cross-scope exposure, deleted-memory resurrection, or unauthorized stale-memory reuse in the qualifying run. State the number of adversarial trials; zero observed failures is not proof of zero true risk.
- **Cost gate:** publish write/index time, bytes on disk including journal and metadata, query p50/p95/p99, returned context tokens, reader cost, and background work. The official LAFS number itself uses query latency; do not imply it includes other lifecycle costs.

## 3. Secondary benchmark lanes

Keep scores in separate tables; do not produce a cross-benchmark average.

| Lane | Use | Primary metrics |
|---|---|---|
| Mem2ActBench | Transfer from recalled state to real tool parameters | exact tool, argument correctness, task success |
| Memora / FAMA | Memory updates and obsolete-memory reuse | FAMA, obsolete-memory errors, update/delete effectiveness |
| MemoryArena | Interdependent multi-session agent–environment tasks | cumulative task success, cross-session transfer, write/read cost |
| AMA-Bench | Very long trajectories and causal/action information | official accuracy, horizon scaling, latency and context cost |
| LoCoMo / LongMemEval-V1 | Conversational and temporal memory | each official score, separately reported |

Start with LongMemEval-V2 small on both domains. After the adapter is valid, choose one action benchmark and one mutation benchmark. Expand only after receipts demonstrate that the system is correctly wired; the point is not to accumulate benchmark names but to falsify clear hypotheses.

## 4. Paradigm-shift hypotheses to test next

### A. Causal episode memory

Store event triplets or small typed graphs: pre-state -> action -> post-state, plus provenance, scope, freshness, and lifecycle. Retrieval should return the chain that explains how the outcome arose, not just the nearest matching state text. Test this against BM25 and official RAG with an ablation that removes causal links.

### B. Risk-limited sequential retrieval

Let the policy choose among answer-with-evidence, retrieve more, verify/contradiction-check, and abstain. Compare at equal query/reader budgets and plot quality, harmful-memory rate, cost and coverage. Keep any oracle strictly evaluator-only.

### C. Governance as an operational invariant

Model updates as append-only events and test revoke/delete/supersede after memory save/load, worker restart, index rebuild and compaction. This branch's temporary journal is only replayable after the harness saves it; it is **not** a crash-safe WAL, a signed audit system or physical secure erasure. Those remain distinct engineering milestones.

### D. Retrieval substrate should earn its complexity

BM25 is the transparent control. Add dense retrieval only with a pinned embedding model. Add HDC/hybrid retrieval only after a real text-to-HDC representation exists and is separately measured. The existing ThoughtLanguageEncoder consumes structured ThoughtChannels, not arbitrary question text, and must not be presented as a semantic encoder.

## 5. What would justify a SOTA claim?

Only an apples-to-apples score from an official benchmark revision with a pinned reader, prompts, evaluation code and data split can support a benchmark-specific claim. Report (1) exact accuracy, (2) official LAFS gain, (3) latency/cost frontier, and (4) governance failure counts, with per-question artifacts. A positive synthetic result or a faster retrieval microbenchmark cannot establish end-to-end SOTA.

A valid negative result is useful: if BM25 does not improve LAFS, keep it as a baseline and test whether causal event structure, better text embeddings, or sequential intervention adds measured value. Do not weaken a gate merely to produce a win.
