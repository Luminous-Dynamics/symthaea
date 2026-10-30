# Broca industry baseline ledger

**As of:** 30 October 2026  
**Status:** research registry; no Broca performance score is claimed.

This ledger prevents a recurring benchmarking error: treating numbers from different answer models, judges, prompts, corpus splits, context limits and benchmark revisions as if they were directly comparable.

The machine-readable registry is \`industry-baseline-ledger-v1.json\`.

## What the current evidence says

| System / study | Benchmark | Reported result | Evidence class | Important qualification |
|---|---|---:|---|---|
| QEMem | LoCoMo | 91.43% | Primary paper | Claude Opus 4.5; paper's shared-memory-store and LLM-judge protocol |
| Zep Cloud | LoCoMo | 94.7% | Vendor self-report | GPT-5.4; vendor's own corpus, reader/judge and context configuration |
| Zep Cloud | LongMemEval | 90.2% | Vendor self-report | 451/500; GPT-5.4; p50/p95 retrieval latency 104/162 ms; median 4,408 tokens |
| Mem0 Platform | LoCoMo | 92.5% | Vendor result / third-party method audit | 1,425/1,540; reported GPT-5 reader and judge |
| Mem0 Platform | LongMemEval | 94.4% | Vendor result / third-party method audit | 472/500; original LongMemEval, not LongMemEval-V2 |
| AgentRunbook-C | LongMemEval-V2 | 74.9% Small / 70.1% Medium | Benchmark reference | Separate V2 protocol, latency 108.3s / 139.9s |
| MemDerive | Derived-state tasks | 30.41–48.30% evaluated memory systems | Primary paper | 10 scenarios, 4,692 events, 1,503 queries; BM25 58.28%, oracle support 96.01% |
| SciValidityBench | Decision-use validity | +9.2 to +18.8 pp vs generic event graph | Primary paper | 64 TaskSpecs, 5 validity modules, 6 scientific domains |
| Vera-Life | Vera-Life-Eval | 62.3% LifeEM (1,500 subset); 61.1% full | Peer-reviewed workshop paper | Qwen3-8B; action-conditioned lifecycle decisions; 8,500 questions / 41,203 evidence pieces |
| AgentMemory BM25 | LongMemEval-S | 84.7% strict recall-all@5 (398/470) | Project component report with per-question receipts | Retrieval only; no answer model, no abstention score; multi-session strict recall 71.1% |
| HALT | Three multi-hop QA benchmarks | Redundant search reduced; exact match largely preserved | Peer-reviewed Findings paper | Stops on expected claim-evidence coverage; no percentage pooled here |
| Active RAG utility evaluation | Multi-hop QA / HotpotQA | Nominal 50% retrieval budget overshot in 40–100% of splits for common triggers | Primary research preprint | Measures marginal correctness change, threshold transfer, realized budget, retrieval harm and all-in cost |
| Cost-aware additional-retrieval router | QASPER | 78.70% evidence Hit@10; 24.5% routed to hybrid retrieval | 27 Oct 2026 preprint | 188 questions / 60 held-out articles; paired bootstrap does not establish reliable utility advantage; answer-level review provisional |
| CONFRAG | Production RAG | 5–39% fewer retrieval calls; up to 800 ms P90 reduction | Peer-reviewed EMNLP Industry paper | Matched answer quality to always-on RAG in its reported system; confidence-based gate, not lifecycle proof |
| AgentAbstain | 263 paired agent tasks | 59.5% best paired accuracy across 17 models | Peer-reviewed NeurIPS benchmark | 42 environments, 8 abstention scenarios; broader agent action benchmark, not memory QA |
| Evidence Sufficiency Benchmark | Five-level L1–L5 evidence quality | 65–91% over-answer rate at L5 conflict | Peer-reviewed benchmark | Seven LLMs / five families; 200-sample human validation; evidence abstention, not lifecycle replay |

Zep's and Mem0's published results are strong commercial reference points. They are **not direct head-to-head comparisons** with QEMem or Broca. Their models and evaluation setups differ.

Two independent shared-harness snapshots are also informative but should not be promoted to definitive leaderboard results:

- An 18 October 2026 pilot using Gemini Flash-Lite reported Mem0 at 64.3% on 42 LongMemEval questions, Letta at 70.0% on 40, and Graphiti at 63.6% on only 11 completed questions. The sample was small and Graphiti's run incomplete.
- A 23 September 2026 run on 419 turns and 120 probes reported recall and adversarial-refusal rates across GoodMem, Letta, Mem0, LangMem and Graphiti. It is a different workload and metric, useful for restraint diagnostics—not an answer-accuracy comparison with LoCoMo or LongMemEval.

## Broca's comparison strategy

Broca should aim to win on a *multi-dimensional frontier*, not one headline number:

1. **Answer quality:** replicate QEMem/LoCoMo with its exact reader, judge, prompts, shared memory store and token budget.
2. **Long-horizon efficiency:** LongMemEval-V2 Small and Medium, reporting latency-adjusted frontier performance, p50/p95/p99 latency and full context cost.
3. **State maintenance:** MemDerive-style exact state and derivation-DAG correctness under corrections, retractions, alternative proofs and implicit dependency changes.
4. **Decision validity:** SciValidityBench-style counterfactual pairs that test whether evidence is admissible to use, not merely retrievable.
5. **Governance:** zero cross-scope emissions, zero use of evidence retracted by known_at, zero invalid proof certificates, zero false optimality claims and zero contradictory supported packets in controlled conformance tests.
6. **Economics:** storage/event, ingestion cost, tokens/correct answer, model calls/query, replay and invalidation cost, and p95/p99 latency.

## Required matching protocol

A result may be called a direct comparison only when it matches:

- benchmark revision, split and source corpus;
- answer model and exact version;
- judge model, prompt and rubric;
- memory formation, query prompt and top-k;
- maximum tool/model calls;
- full context-token accounting, including packet/rendering overhead;
- sample size and scoring method.

Store paired per-question results, confidence intervals and raw receipts. Do not tune on held-out splits. Do not conflate evidence recall with answer accuracy, nor combine LoCoMo, LongMemEval and LongMemEval-V2.


## Lifecycle faithfulness: a first-class benchmark axis

Vera-Life-Eval is the most relevant new comparator for Broca's bitemporal/proof line. It tests whether the system takes the right action when evidence changes over time, not merely whether the answer text resembles a reference answer.

| Vera-Life-Eval dimension | Reported result | What it tests |
|---|---:|---|
| LifeEM, 1,500-question stratified subset | 62.3% | Correct answer or lifecycle-aware action under the paper's scoring rules |
| LifeEM, full 8,500 questions | 61.1% | Full benchmark performance |
| Best prompt-only baseline, same Qwen3-8B generator | 37.4% | Prompting without the full lifecycle controller |
| Timestamp-only metadata regime | 58.4% | Loss of richer lifecycle relations |
| Text-only metadata regime | 50.4% | Loss of structured time/lifecycle metadata |
| Lifecycle-graph ablation | 31.7% | Dependence on explicit update/supersede/conflict/expiry relations |
| Stale rejection rate, full metadata | 100% | Avoiding stale-only evidence as if it were current |
| Over-abstention rate, full metadata | 0% | Avoiding unnecessary refusal when evidence is sufficient |

**Do not compare LifeEM numerically with LoCoMo or LongMemEval answer accuracy.** LifeEM is action-conditioned: appropriate `WARN_STALE`, `EXPLAIN_CONFLICT`, `RETRIEVE_MORE` or `ABSTAIN` behavior can be credited when a direct answer is not justified. The paper uses Qwen3-8B and deterministic gold constraints rather than an LLM judge. Its 62.3% result is the 1,500-question stratified subset; the full-set result is 61.1%.

This directly motivates a Broca evaluation contract with five explicit actions: `ANSWER`, `WARN_STALE`, `EXPLAIN_CONFLICT`, `RETRIEVE_MORE`, `ABSTAIN`. Each action should be accompanied by a claim-level lifecycle ledger tying the decision to exact source revisions, valid-time intervals, transaction-time cutoff, supersession/retraction state, conflict edges and authorization evidence.

### Broca lifecycle scorecard

Broca should reproduce the Vera-Life protocol under the same generator, metadata regimes and scoring rules before claiming superiority. Proposed gates:

- **Safety conformance:** zero `ANSWER` actions supported only by stale, retracted, unauthorized, cross-scope or contested evidence in the deterministic fixture suite.
- **Stale handling:** match the paper's 100% stale rejection under the full-metadata regime; report warning/abstention calibration separately from false stale warnings.
- **Over-abstention:** target at or below the published 0% full-metadata result on the exact same test split; always report answerable abstentions separately.
- **Action quality:** LifeEM at least 62.3% on the stratified subset and 61.1% on the full set as replication gates, then seek statistically significant improvement with paired per-item outputs and fixed Qwen3-8B generator/judge protocol.
- **Ablations:** removing the lifecycle graph, bitemporal metadata, proof verifier or action policy must be measured independently; never rely on a single aggregate score.
- **Transfer:** use the paper's source-disjoint REAL-mini-style domains as external validation; report each domain and action rather than only a pooled mean.

These are prospective targets, not Broca results. Vera-Life's own paper reports that lifecycle metadata and graph structure drive much of the gain; Broca must demonstrate this under the exact protocol, not infer it from the architecture.

## Retrieval is not the same as proof sufficiency

The AgentMemory 0.9.31 BM25 component report gives a useful retrieval-only comparator on LongMemEval-S: 398/470 answerable questions with *all* required source sessions in the top five (84.7% strict recall-all@5), fractional recall@5 of 91.7%, any-hit@5 of 96.2%, and only 71.1% strict recall on the 121 multi-session questions. Its report includes 30 unanswerable questions but does not score abstention and does not run an answer model.

Broca should therefore publish three independent stages:

1. **Candidate recall:** required source/session recall@k (including strict recall-all@k), MRR/nDCG, multi-hop coverage, stale-source retrieval rate and cross-scope candidate rate.
2. **Admissible evidence:** fraction of retrieved candidates accepted after bitemporal, lifecycle, authorization and contradiction checks; proof sufficiency rate; false proof-certificate rate; support-family diversity; exact packet cost.
3. **Reader outcome:** answer/action accuracy, lifecycle LifeEM, abstention calibration, unsupported assertion rate, citation/proof completeness and task success.

Do not collapse these into one number. A retriever can have high recall and still select stale evidence; a verifier can be sound but starved by low recall; an answerer can be accurate while using an invalid proof. Report per-stage latency and tokens so the Pareto trade-off is visible.

## Retrieval stopping and epistemic restraint

Recent work makes clear that deciding whether to retrieve more is its own benchmark problem. HALT uses expected claim–evidence coverage as a stopping condition and reports lower redundant search while largely preserving exact match. Active RAG utility work goes further: it measures marginal correctness change with versus without retrieval, and requires realized budget usage, threshold transfer, retrieval harm and total trigger/retrieval cost. CONFRAG provides an industry operating point with 5–39% fewer external retrievals and up to 800 ms P90 latency reduction while matching its own always-on RAG quality.

A further caution is provided by Zhao's 27 October 2026 cost-aware retrieval study: its policy sent 46/188 QASPER questions (24.5%) to hybrid retrieval and obtained 78.70% evidence Hit@10 versus 76.92% for always-keep, but paper-cluster paired-bootstrap intervals did not establish a reliable utility advantage. Answer-level correctness/faithfulness assessment was provisional, model-assisted and not independently human validated. Broca must therefore evaluate downstream action/answer utility, not merely evidence Hit@k or fewer calls.

AgentAbstain and the Evidence Sufficiency Benchmark expose the complementary stop/abstain failure: task completion does not imply that acting was justified. AgentAbstain's 263 paired tasks across 42 environments report 59.5% best paired accuracy; the Evidence Sufficiency Benchmark reports 65–91% over-answering under conflicting evidence for the evaluated models.

### What Broca should measure

1. **Claim coverage:** fraction of required claims with at least one admissible proof; uncovered-claim count at stop; multi-hop completeness.
2. **Marginal utility:** paired correctness/action-loss delta with retrieval versus no retrieval; report how often retrieval helps, is neutral, or harms.
3. **Calibration:** Brier/ECE, confidence intervals on utility gain, and lower-bound coverage; calibrate on one split and report realized usage on held-out splits.
4. **Budget transfer:** target versus realized call/token/cost usage across domains and seeds; no nominal-budget claims without observed usage.
5. **Retrieval harm:** stale/conflicting distractor rate, unsupported-claim increase, and fraction of retrieval calls that worsen the downstream decision.
6. **All-in cost:** estimator compute, probe calls, retrieval calls, prompt tokens, p50/p95/p99 latency and cost per correct supported answer.

Broca's conservative VOI gate combines a lower bound on correctness gain and lifecycle-risk reduction with an upper bound on retrieval harm and measured cost. It stops when the lower-bound net benefit does not exceed a policy margin, even when calls remain. This is a prospective architecture/benchmark target, not a measured advantage.

### Direct baselines

- Always-retrieve fixed top-k.
- BM25-score threshold.
- Confidence-only trigger (CONFRAG-like).
- Claim–evidence coverage stop (HALT-like).
- Broca lifecycle proof coverage plus calibrated VOI.

The comparison must hold generator, reader prompt, memory corpus, proof checker, answer judge and total budget accounting constant. Report full utility/cost frontiers and paired per-question receipts; do not compare a retrieval-call reduction against an unmatched accuracy score.


## External sources

- Vera-Life, *Lifecycle Faithfulness for Grounded Retrieval-Augmented Generation* (GroundLM 2026, 29 October 2026): https://aclanthology.org/2026.groundlm-1.18/
- HALT, *Verification-Aware Stopping for Retrieval-Augmented Search Agents*, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.343/
- Active RAG, *When Should Active RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost* (27 July 2026): https://arxiv.org/abs/2607.24010
- Zhao, *When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG* (27 October 2026): https://arxiv.org/abs/2610.37358
- CONFRAG, *Confidence-Guided Retrieval-Augmented Generation*, EMNLP 2026 Industry Track: https://aclanthology.org/2026.emnlp-industry.171/
- AgentAbstain, NeurIPS 2026 Evaluations & Datasets Track: https://agentabstain.github.io/
- Evidence Sufficiency Benchmark, *Do LLMs Know When Evidence is Insufficient?* (13 August 2026): https://doi.org/10.32604/cmc.2026.086343
- AgentMemory 0.9.31 BM25 LongMemEval-S component report (28 October 2026): https://www.agent-memory.dev/benchmarks

- QEMem, *From Relevant Memories to Sufficient Evidence* (15 October 2026): https://arxiv.org/abs/2610.22668
- Zep research and vendor methodology: https://www.getzep.com/research/
- Third-party Mem0/Zep/Letta matched-harness review (18 October 2026): https://agentsystems.blog/agent-memory-benchmark
- LongMemEval-V2 baseline protocol: https://xiaowu0162.github.io/longmemeval-v2/
- MemDerive (18 October 2026): https://arxiv.org/abs/2610.25373
- SciValidityBench (26 October 2026): https://arxiv.org/abs/2610.34767
- Agent Memory Bench matched-harness run: https://github.com/hamza-dev-tech/agent-memory-bench

All scores above remain attached to the source protocol. The ledger does not assert that a vendor result has been independently reproduced by Broca.
