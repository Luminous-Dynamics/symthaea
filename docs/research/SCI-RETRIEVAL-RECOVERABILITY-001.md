# SCI-RETRIEVAL-RECOVERABILITY-001 — recoverability and agency credit

**Issue:** Symthaea #10690  
**Parent:** SCI-RETRIEVAL-CAUSAL-001 / #8907  
**Implementation:** PR #10695

## Research problem

A single end-to-end score conflates at least six capabilities:

1. whether the required evidence was reachable in the benchmark's frozen search surface;
2. whether the retriever found it;
3. whether the reader extracted what it said;
4. whether the system used it correctly in the answer;
5. whether the agent's actions added value beyond a fixed iterative policy; and
6. whether the result is valid under the exact evidence, rights, time and contamination controls.

A benchmark with unreachable evidence can punish a good reader. A reader that fails to use a gold-bearing document should not be diagnosed as a retrieval failure. An agent that only performs more search rounds should not be credited with open-ended agency.

## Research anchors

- **REALM (ACL 2026):** proposes an LLM-free recoverability diagnostic and a one-shot / choice-free fixed-loop / full-agent credit ladder. In its study, headline agent gain varied from -0.21 to +0.30 across corpora in their native metrics; the fixed loop recovered 65–96% of gains otherwise attributed to agency on reformulation-recoverable corpora. These figures are paper-specific, not Symthaea measurements. https://aclanthology.org/2026.realm-1.47/
- **Retrieval Myopia (EMNLP Findings 2026):** reports up to 46.2% wrong answers when the gold-bearing document is already present in retrieved documents; clean-format re-presentation of the same evidence recovered up to 48.9% of those cases in its study. https://aclanthology.org/2026.findings-emnlp.1713/
- **Agentic-R (ACL Findings 2026):** learns retrievers for multi-turn agentic search using both local passage utility and global answer correctness, and reports gains over baselines on seven single-/multi-hop QA benchmarks. We should include it as a retriever baseline, not treat its results as a directly comparable end-to-end score. https://aclanthology.org/2026.findings-acl.785/
- **SPARKLE (ACL 2026):** proposes a structured plug-and-play agentic retrieval policy and reports mean improvements of 9.17% in-domain and 2.85% out-of-domain over the adaptive-RAG baselines it evaluated. https://aclanthology.org/2026.acl-long.1793/
- **RAMP (EMNLP 2026):** probes retriever behavior and adapts a single agent to multiple retrievers; its abstract reports 96–98% of retriever-specific specialist performance and 91.8% of a retrained upper bound on unseen retrievers. Include it for backend-transfer testing. https://aclanthology.org/2026.emnlp-main.1977/
- **Microsoft AgenticRAG (May 2026):** a production-oriented enterprise search/find/open/summarize harness; the paper reports 49.6% Recall@1 on BRIGHT and 0.96 factuality on WixQA in its evaluation. These are paper-reported results, not independently verified universal industry baselines. https://www.microsoft.com/en-us/research/publication/agenticrag-agentic-retrieval-for-enterprise-knowledge-bases/
- **ACL Industry 2026 comparison of RAG variants:** compares enhanced and agentic RAG across scenarios and cost/quality dimensions, supporting matched-cost evaluation rather than assuming agentic orchestration is always better. https://aclanthology.org/2026.acl-industry.5/

## Paradigm shift

Move from "the agent got the right answer" to **causal, stage-resolved evidence acquisition**.

The central evaluation unit is not just a question. It is a required evidence unit plus a frozen set of permitted routes, a known evidence frontier, a budget and a downstream inference obligation.

### Recoverability classes

A finite directed graph represents the exact frozen search surface and allowed route operations:

- DirectlyReachable: initial retrieval reaches the pinned evidence.
- ReformulationRecoverable: a first retrieval occurs, then a declared query reformulation, then a post-reformulation retrieval reaches the evidence; no structural navigation is required.
- NavigationRecoverable: a first retrieval occurs, followed by a structural action such as open, follow link/TOC, paginate, or in-document find.
- UnreachableWithinBudget: no path fits the route policy and budget, but only when the finite surface is declared a complete frozen inventory and is bound to evaluator-side completeness-evidence references.
- NotEvaluable: unknown surface/artifact, or no route found in a partial surface.

Positive/negative evidence is asymmetric:
- a concrete pinned route is positive evidence of reachability even on a partial observed surface;
- absence of a route is informative only on a complete frozen inventory;
- an unknown surface never supports an absence conclusion.

The evaluator itself is model-free. The agent cannot choose its own recoverability class.

### Agency-credit ablation

Use equal model, prompt, corpus/index revision, context, tool permissions, stop rules and budget wherever the intervention allows:

- A0: one-shot RAG / first-pass retrieval.
- A1: choice-free fixed-iteration retrieval with a frozen reformulation schedule.
- A2: full open-ended agent.
- A3: exact gold-evidence injection (isolates reader/extractor capability).
- A4: clean-format re-presentation of the same evidence bytes and source locators (reader-format intervention).
- A5: evidence withheld (abstention and parametric-leakage control).

Attribute A2−A0 as total agent gain, A1−A0 as fixed-iteration gain, and A2−A1 as incremental agency. A4−A3 estimates presentation-format effect. Do not label fixed iteration or clean re-reading as agency.

These arms compose with #8929 (pre-run intervention plan) and #8960 (post-run outcome receipts). They must be added as versioned condition/analysis extensions to those owners, not as a competing plan or scorecard.

## Implementation status

PR #10695 contains a standalone Rust validator and CLI, strict JSON Schemas, a pinned five-unit synthetic graph, and an independent Python reference. It is isolated from the larger science-benchmark package so its formatting, compile, test and Clippy checks can run without broad unrelated rustfmt differences.

The current synthetic graph contains exactly one example of each class. It pins:
- the corpus snapshot, search-surface revision and route policy;
- the first-pass state and typed route edges;
- the maximum route cost and unit;
- each answer-critical artifact digest and source-family reference;
- an explicit completeness-evidence reference for the complete frozen inventory.

Metamorphic controls cover route deletion/addition, lower budget, partial/unknown surface, missing completeness evidence, reformulation before post-retrieval, first-pass retrieval before reformulation/navigation, deterministic shortest-route witnesses, source-family reference counting, unknown model labels and manifest tampering.

A passing synthetic contract does **not** prove that the referenced completeness evidence is true; the validator binds opaque references but does not independently verify them. The completeness claim and manifest commitment need independent pre-run anchoring.

## Benchmark and comparison strategy

Compare at matched compute, tokens, calls, wall time, context size, corpus, model and search surface:

**Retriever baselines:** BM25; multilingual dense retrieval; sparse+dense hybrid; standard cross-encoder reranker; multi-query RAG; Agentic-R; and RAMP-style retriever adaptation.

**Policy baselines:** one-shot retrieval; deterministic fixed-iteration retrieval; an enhanced-RAG controller; SPARKLE-like structured adaptive policy; Microsoft AgenticRAG-style search/find/open navigation; and the full Symthaea policy.

**Evidence/reader controls:** gold-evidence injected, withheld, decoy, updated/contradicted, and clean-format re-presentation.

Use public datasets only after revision, license, source-family overlap and split provenance are pinned. Candidates include TEMPO-compatible temporal retrieval, AfrIFact-compatible African-language evidence tasks, and RDQA document/video/audio tasks where source terms and formats permit. Never imply that any one benchmark covers all South African languages or domains.

## Metrics — do not collapse

- **Reachability:** class distribution, path-cost distribution, shortest-route witness validity, reachable evidence by budget.
- **Retrieval:** Recall@k, MRR, nDCG@k, source-family recall, per-required-period temporal coverage, cross-language recall and navigation success.
- **Reading:** gold-conditional extraction, span/locator precision and recall, relation correctness, clean-format recovery.
- **Evidence use:** answer change under add/remove/update/contradict intervention; evidence-support accuracy; calibrated abstention; parametric leakage rate.
- **Agency:** paired A2−A1 incremental benefit by recoverability class and budget.
- **Systems:** p50/p95 latency, evidence tokens, total tokens, calls/actions, memory and cost per qualified answer.
- **Validity:** contamination and rights failures, failed/blocked/not-started denominator cells, source-family leakage and evaluator uncertainty.

## Qualification targets (proposed, not results)

- 100% Rust/Python agreement on every class and shortest route witness in the synthetic corpus.
- 100% enforcement of route ordering and complete-inventory evidence gates.
- Zero negative labels from partial/unknown surfaces.
- Zero model-dependent reachability labels.
- Zero policy-visible gold route witnesses or evaluator answers.
- A2−A1 is credited only when positive on preregistered navigation/recoverability strata at matched budget, with paired uncertainty and family-wise correction.
- Any claimed improvement is reported on a cost/quality Pareto frontier, not only as a single average score.

## Claim ceiling

The graph validator measures benchmark-relative reachability for a specific committed finite search surface, route policy and budget. It does not establish open-web completeness, actual model route-following, source authenticity, factual truth, rights compliance, language parity, or SOTA. Empirical superiority requires real paired runs, licensed/pinned data and independent adjudication.
