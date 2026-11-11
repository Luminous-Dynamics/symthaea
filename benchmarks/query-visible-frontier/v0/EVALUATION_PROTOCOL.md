# QVF-R and Temporal Retrieval — Evaluation Protocol v0.1

## Status and claim ceiling

This document preregisters the proposed evaluation design and target thresholds. It contains no measured Symthaea retrieval result. All numeric targets below are **goals**, not achieved scores.

QVF-R itself is a synthetic qualification lane for historical retrieval eligibility. The larger retrieval programme must combine benchmark-standard ranking metrics with independent correctness, provenance, temporal and operational controls. Do not collapse these into one groundedness or intelligence score.

## Research questions

1. Does retrieval return the correct proposition version for the requested valid-time, even when obsolete and current versions are semantically overlapping?
2. Was each returned representation materialized and visible to the exact serving replica by the query's known-at cutoff?
3. Are derived bridge facts, embeddings, summaries and caches invalidated or marked stale when any parent is corrected, withdrawn, rights-revoked or transformed?
4. Does temporal/proposition-aware retrieval improve useful evidence coverage without degrading static retrieval?
5. Does index-time compilation reduce online calls/latency while preserving lineage and lifecycle correctness?
6. Do query-planning or agentic retrieval gains survive fixed-budget, recoverability and same-frontier controls?
7. Are filter constraints actually applied, and what eligible candidates are lost under the provider's filter execution mode?

## Benchmark tracks and baselines

Use each benchmark only under its exact version, license, split, scoring protocol and contamination restrictions.

| Track | Public comparison | Required comparison |
|---|---|---|
| Temporal multi-hop retrieval | TEMPO (EMNLP 2026) | nDCG@10, Temporal Coverage@10, Temporal Precision@10; per required time period and retrieval step |
| Overlapping amendments | TimelyRAG / TimelyQABench (Findings EMNLP 2026) | Correct clause-version selection; current vs historical intent; overlapping amendment cases |
| Evolving semantic conflict | FRESCO (Findings EMNLP 2026), Re³ / Re² Bench (ACL 2026) | Obsolete-version suppression and new-version recall, with non-evolving controls |
| Multi-hop bridge retrieval | IndexRAG (Findings IJCNLP-AACL 2026), HotpotQA, 2WikiMultiHopQA, MuSiQue | F1, answer-bearing bridge recall, online inference calls and index-time cost |
| Proposition relationships | ProStruct-RAG / Asteria (EMNLP 2026) | Update, contradiction and support relation retrieval; version-consistent synthesis |
| Multilingual retrieval | SHIFT, AfrIFact, LAURA and relevant language-specific tracks | Same-language and cross-language retrieval, language/culture preservation and unsupported-language abstention |
| Query integrity | Evidence-anchor-preserving retrieval tests | Harmful rewrite rate, anchor retention, exact named entity/date/numeric preservation |
| Source-grounded reading / evidence use | RDQA (EMNLP 2026), Retrieval Myopia (Findings EMNLP 2026) | Parametric Leakage Rate, gold-present reading failure, evidence-use attribution and clean-format recovery |
| Query-visible frontier | QVF-R synthetic + lawful real-world telemetry cohorts | False eligibility, post-cutoff leakage, replica visibility, as-of version correctness |
| Enterprise provider adapters | Azure AI Search, Amazon Bedrock Knowledge Bases, Google Vertex AI RAG Engine | Same query/corpus/compute budgets; exact exposed execution and index facts; provider-opaque fields remain Unknown |

Published results are paper-specific. For orientation, TEMPO reports a best evaluated DiVeR result of 32.0 nDCG@10 and 71.4% Temporal Coverage@10; TimelyRAG reports gains up to +28.6% nDCG@10 in its tested settings; FRESCO reports up to 27% improvement on its evolving-knowledge tasks while maintaining competitive non-evolving performance; IndexRAG reports +4.6 F1 points over Naive RAG on average across its three multi-hop datasets. Reproduction is required before treating any figure as a baseline for Symthaea.

## Ablation matrix

Hold the corpus, candidate budget, answer model and evaluation split constant within each paired block.

| ID | Retrieval configuration | Purpose |
|---|---|---|
| A0 | BM25 / lexical only | Exact term, identifier and clause baseline |
| A1 | Dense semantic only | Semantic retrieval baseline |
| A2 | Hybrid BM25 + dense with RRF | Industry-standard hybrid baseline |
| A3 | A2 + temporal embedding/reranking | TEMPS / TimelyRAG-style temporal effect |
| A4 | A3 + evolving-conflict reranking | FRESCO / Re³-style version arbitration |
| A5 | Proposition/version-relational retrieval | ProStruct-RAG-style relational evidence |
| A6 | Index-time bridge materialization | IndexRAG-style offline reasoning |
| A7 | A6 + source-family/rights/temporal lineage checks | Measure bridge lift without provenance laundering |
| A8 | Multilingual index-side harmonization | SHIFT-style language correction; only with lawful, qualified parallel data |
| A9 | A8 + anchor-preserving query plan | Measure cross-language query reformulation harm |
| A10 | Full stack + QVF-R eligibility gate | End-to-end query-visible and lifecycle correctness |
| A11 | Force-injected gold evidence; compare native agent format with clean QA presentation of the exact same evidence | Isolate reading/interpretation failures from retrieval failures without changing evidence availability |

Run one-shot, choice-free fixed-iteration and adaptive-agent variants under the same evidence frontier, model, token/call budget and candidate ceiling. Symthaea #10719 owns causal attribution; do not use different evidence availability or format as a hidden advantage for the adaptive arm.

For index-time systems, report ingestion/materialization compute, index size, update/invalidation lag and rebuild cost separately from query latency. A bridge representation is a derived retrieval unit, not an independent source family.

## Metric definitions

### Retrieval quality

- **nDCG@k / MRR / Recall@k:** use the benchmark's official relevance judgments and evaluator version.
- **Temporal Coverage@k:** follow TEMPO's official definition and scorer. Report by required time segment as well as overall.
- **Temporal Precision@k:** follow the official benchmark definition. A system must not improve coverage by flooding the result set with irrelevant historical material.
- **Answer F1 / exact match:** report separately from retrieval. Correct answers without required evidence may be parametric recall or leakage.

### Proposition and version correctness

- **Version Selection Accuracy:** number of query-level proposition-version decisions matching the externally resolved canonical version divided by all version-disambiguation decisions.
- **Obsolete-Clause Substitution Rate:** queries for which an obsolete/superseded clause is selected as the governing version when the requested valid-time resolves to a successor, divided by queries with an available overlapping predecessor/successor pair.
- **Historical Version Preservation:** queries explicitly asking about the earlier state where the earlier canonical version is retrieved and not replaced by the current one.
- **Exact Source-Span Precision / Recall:** exact or benchmark-defined span match to the canonical source locator; report exact-match and token-overlap variants separately.
- **Amendment Propagation Accuracy:** percentage of versioned representations whose update/supersession relationships agree with adjudicated source chronology.

### Query-visible and lifecycle correctness

Let E(q,K) be the independently established set of representations eligible for query q at known-at cutoff K, and R(q,k) be the top-k representations actually returned.

- **Eligibility Precision@k** = |R ∩ E| / |R|. Ineligible material in an answer context is a correctness failure, not a relevance penalty.
- **Eligible Candidate Recall@k** = |R ∩ E| / |E|, only where E is exactly enumerable under a frozen corpus/frontier.
- **False Eligibility Rate** = representations admitted as eligible despite failing at least one required valid-time, known-at, rights, materialization or serving-visibility gate, divided by all admitted candidates.
- **Post-cutoff Visibility Leakage** = representations not visible on the queried replica by K but exposed to the answer context, divided by all answer-context representations.
- **Unknown Visibility Promotion Rate** = Unknown visibility cases treated as visible/eligible, divided by Unknown cases. The qualification target is zero.
- **Derived Artifact Leakage** = derived representations returned after any required parent is invalidated, withdrawn, rights-denied or otherwise ineligible.
- **Visibility Freshness Lag** = observed replica-visible time minus exact materialization/index-revision time. Preserve clock/uncertainty semantics; do not compute a precise lag where either event time is untrusted or unknown.
- **Filter False-Negative Rate** = eligible exact-corpus candidates omitted due to provider filter execution, divided by all eligible candidates, stratified by selectivity, k, shards/partitions and provider filter mode.

Do not call a provider's request filter commitment proof that it applied the filter. Record prefilter, postfilter, strict postfilter, client-side validation, provider-opaque or Unknown as separate execution observations. Client-side validation can reject an ineligible returned item but cannot prove exhaustive provider recall.

### Provenance and evidence use

- **Source-Family Recall:** count canonical provenance families, not URLs, chunks, mirror copies or derived bridges.
- **Independent-Family Precision:** independently adjudicated provenance families among families treated as independent.
- **Bridge Lineage Completeness:** fraction of returned derived propositions whose full parent statement-version, source-span, transform and rights dependencies are present.
- **Revocation Propagation Delay:** time from effective parent revocation/correction to descendant materialization invalidation or verified stale marking. Keep rights state distinct from index freshness.
- **Evidence Use / Attribution:** evaluate whether the answer actually uses the required retrieved span, separately from whether that span was retrieved. Use a process-aware source-attribution protocol and human adjudication on a stratified sample.
- **Parametric Leakage Rate (PLR):** use RDQA's official definition/scorer to quantify correct answers produced without the required source-access path. This prevents parametric recall, guessing or contamination from being counted as source-grounded success.
- **Gold-Present Reading Failure:** force-inject the gold-bearing evidence into the reader's input and measure failure to extract/use it. This isolates reading/interpretation from retrieval, as in Retrieval Myopia.
- **Clean-Format Recovery:** with identical retrieved evidence, compare the agent's native interaction format against a clean QA presentation. Report recovery as a reader/presentation effect, not a retrieval improvement or model-learning gain.

A digest or provenance chain proves binding/lineage properties under its contract; it does not prove factual truth, semantic entailment, source independence or a correct rights decision.

### Temporal Error Debt and abstention

Feed the resulting belief/answer trajectory into Symthaea #9492's existing Temporal Error Debt programme. QVF-R supplies availability events; it does not redefine TED's units or formula.

Report separately:
- correct answers;
- unsupported/incorrect answers;
- false abstentions;
- justified abstentions under Unknown or insufficient evidence;
- time spent holding stale/wrong beliefs after a correction became query-visible.

### Efficiency and operations

Report p50/p95 query latency, index-time compute, update/rebuild latency, index size, token use, model calls, reranker calls and provider cost. Also report the quality/efficiency Pareto frontier. Do not hide correctness failures inside a weighted cost score.

## Proposed qualification targets

These are aspirational gates to test, not current Symthaea results.

| Area | Initial target | Qualification condition |
|---|---|---|
| Synthetic QVF-R eligibility | 0 false eligibility | All negative controls rejected; Unknown never promoted |
| Proposition-version selection | 100% on frozen deterministic fixtures | Exact canonical version and source-span refs |
| Derived lineage / rights | 0 stale or denied descendant payloads | Mutation tests cover every parent and transform |
| TEMPO temporal coverage | ≥85% Temporal Coverage@10 | Same split, scorer, candidate budget; report nDCG alongside it |
| TEMPO relevance | ≥36.0 nDCG@10 initial target | Same official metric/scorer; paired confidence intervals |
| TimelyQABench | ≥5% relative nDCG@10 improvement over strongest faithfully reproduced baseline | Static/non-evolving nDCG degradation no worse than 1% relative |
| FRESCO-style conflict | ≥50% relative reduction in obsolete-version selections vs strong reranker baseline | No more than 1% absolute nDCG loss on non-evolving control |
| IndexRAG bridge baseline | Reproduce or exceed the reported +4.6 F1-point average over Naive RAG | Include index compute, parent lineage and invalidation tests |
| Exact finite-corpus filtering | 0 false negatives in fully enumerable controls | Provider filter mode and scope fixed/verified |
| Serving-frontier correctness | 0 post-cutoff leakage in deterministic replay | Exact replica and index revision bound |
| Source-required evidence use | PLR <=5% on source-required held-out tasks | Official RDQA scorer and full process trace |
| Gold-present reading | <=5% failure when exact gold evidence is force-injected | Same evidence bytes and answer budget across reader-format arms |
| Efficiency | ≤20% p95 overhead for the QVF eligibility checks in the first integrated prototype | Same hardware, workload and output quality; revise after measuring baseline |

These targets are not interchangeable. In particular, the TEMPO ranking target must not be purchased by dropping historical evidence needed for coverage, and temporal coverage must not be purchased by admitting the wrong proposition version.

## Statistical design and anti-contamination controls

1. Freeze benchmark manifests, splits, corpus digests, source snapshots, index-building code, embedding/reranker/model versions, prompts, token budgets and evaluator/scorer versions before confirmatory runs.
2. Keep train/dev/test separated by source family and document lineage, not merely by chunk or URL. Prevent mirrors, amendments and derived bridges from crossing splits.
3. Randomize system order within paired query blocks. Use fresh process/cache namespaces for arms that require isolation; record warm/cold conditions separately.
4. Treat a complex query and its decomposed steps as a grouped unit for uncertainty estimates. Do not pretend dependent retrieval steps are independent samples.
5. Report paired bootstrap confidence intervals or an appropriate paired test over query groups. Publish per-domain, per-language and per-temporal-case strata; do not let aggregate means hide a safety-critical subgroup.
6. Use at least two independent human annotators for proposition identity, source-span boundaries, update/contradiction relationships and answer evidence-use on the adjudicated subset. Record agreement and adjudication outcomes. Do not use an LLM judge as the sole oracle.
7. Keep benchmark answers, gold spans and scoring files out of online agent access. Audit inference-time web, local files, mounted corpora, caches, prompts and external tools for leakage.
8. Audit dataset and source licenses before download, training, redistribution or commercial evaluation. A paper's benchmark release does not imply that all underlying source material is commercially reusable.
9. Preserve these currently verified license distinctions in the experiment register:
   - FRESCO's official repository states that most of the project/data are CC-BY-NC; the instruction-optimization code has separate MIT terms, and third-party source content is governed by its own licenses. Do not assume the benchmark data is cleared for commercial use: https://github.com/facebookresearch/fresco
   - IndexRAG's official code repository identifies Apache-2.0 for the code, but the benchmark datasets retain their original licenses: https://github.com/Continuum-AI-Corp/IndexRAG
   - TimelyRAG's official README lists four "public" overlapping-evolving sets (law, university, company policies, terms of service), each with 30,000 documents and 3,000 queries. No root LICENSE file was found in the checked repository paths; treat code/data permissions as unresolved and obtain author/source clearance before use: https://github.com/kaist-dmlab/TimelyRAG
   - ProStruct-RAG/Asteria is described as fictional/contamination-resistant, but its dataset reuse terms still require direct repository/license review before redistribution or commercial use: https://aclanthology.org/2026.emnlp-main.294/
10. Publish all failed, abstained, truncated, Unknown and invalidated cases. A run with unavailable evidence is not silently dropped from the denominator.
11. Preserve exact-head source, artifact digests and validator outputs. A rebase or fixture mutation invalidates earlier qualification until rerun.
12. For evidence-use/reader-format experiments, inject byte-identical gold evidence, randomize arm order, isolate session/cache state and bind the exact reader input. A clean-format recovery cannot be counted as a retrieval recall gain.

## Required result packet

Every evaluation packet should include:
- exact source commit and clean diff identity;
- benchmark and source-pack manifests plus license ledger;
- exact model, embedding, reranker and index revision identities;
- plan, execution receipt, temporal-resolution, materialization and replica-visibility refs;
- all metric definitions/scorer hashes and adjudication protocol;
- per-query results including abstentions, Unknowns, false negatives and failures;
- latency/cost/index-maintenance measurements;
- paired confidence intervals and subgroup results;
- explicit nonclaims and remaining unobservable provider state.

## References

- TEMPO: https://aclanthology.org/2026.emnlp-main.291/
- TimelyRAG / TimelyQABench: https://aclanthology.org/2026.findings-emnlp.398/
- FRESCO: https://aclanthology.org/2026.findings-emnlp.54/
- Re³ / Re² Bench: https://aclanthology.org/2026.acl-long.1180/
- IndexRAG: https://aclanthology.org/2026.findings-aacl.175/
- ProStruct-RAG / Asteria: https://aclanthology.org/2026.emnlp-main.294/
- RDQA: https://aclanthology.org/2026.emnlp-main.1855/
- Retrieval Myopia: https://aclanthology.org/2026.findings-emnlp.1713/
- Azure vector filtering: https://learn.microsoft.com/en-us/azure/search/vector-search-filters
- AWS Bedrock Knowledge Bases: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
- Google RAG Engine Vector Search: https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/use-rag-managed-vertex-ai-vector-search
