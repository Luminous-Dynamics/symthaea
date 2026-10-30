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

## External sources

- QEMem, *From Relevant Memories to Sufficient Evidence* (15 October 2026): https://arxiv.org/abs/2610.22668
- Zep research and vendor methodology: https://www.getzep.com/research/
- Third-party Mem0/Zep/Letta matched-harness review (18 October 2026): https://agentsystems.blog/agent-memory-benchmark
- LongMemEval-V2 baseline protocol: https://xiaowu0162.github.io/longmemeval-v2/
- MemDerive (18 October 2026): https://arxiv.org/abs/2610.25373
- SciValidityBench (26 October 2026): https://arxiv.org/abs/2610.34767
- Agent Memory Bench matched-harness run: https://github.com/hamza-dev-tech/agent-memory-bench

All scores above remain attached to the source protocol. The ledger does not assert that a vendor result has been independently reproduced by Broca.
