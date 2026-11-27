# MACE matched benchmark contract and competitive protocol v0.1

**Status:** run-manifest and matched-comparison validator authored; dedicated Rust validation pending. This is an evaluation contract, not a score or a claim that MACE has reproduced an external method.

## 1. Result identity

Every result is one benchmark × track × protocol × dataset revision × split. A published external result, local reproduction and local MACE experiment are different evidence classes. AML textual, coding and multimodal are distinct tracks; the contract refuses to pool them.

The matcher allows memory method, implementation and adapter to differ. It freezes benchmark/protocol, dataset/split, seed set, reader model, prompt, judge, tool contract, evaluator, metric definitions, evaluator isolation and the full declared resource budget. The resulting receipt records comparability; it is not a significance test. Paired effects and cluster uncertainty still require task-level outcomes.

## 2. Current external reference points (published results, not MACE results)

| Benchmark / track | Public reference | Matched comparison requirement |
|---|---:|---|
| AML Cycle 2 — Textual, open-source | Resurface 61.22 | Official Add/Search API, dataset, scoring, division; capability scores and streaming separate |
| AML Cycle 2 — Coding, open-source | InsightMem 63.33% resolved | Relevant and Noisy conditions and denominators separate |
| AML Cycle 2 — Multimodal, open-source | Khipu 43.26 | Same multimodal track; never compare with textual/coding scores |
| AML Cycle 2 — Commercial stretch | MemoraX 69.26 textual, 68.00% coding, 53.68 multimodal | Separately report commercial division; not a matched MACE result |
| LongMemEval-V2 | AgentRunbook-C v2 Online: 75.6% / 130.5 s / LAFS +2.823; Perpetual Recall V3: 53.2% / 1.96 s / +1.096; OPS-Agentic: 81.4% / 153.6 s / +0.322 | Reproduce accuracy-latency frontier with exact reader, tier, split and latency definition |
| AMA-Bench | AMA-Agent 57.22% average, +11.16 points over strongest baseline | Qwen3-32B and paper's open-ended QA/judge setup; report causal, state-update and abstraction strata |
| COLLABMEM | TMS-GRAPH++ 69.6% non-Oracle average | Exact matched trace, target role, answerer, judge and gold event |
| TimlMemBench | 208 samples, 1,566 sessions, 1,715 questions, 1,821 governance/activation cases | Canonical-pool control and raw-native ingestion separate; nine axes remain distinct |
| MemSecBench | 310 cases, 48 contexts; 84.2% poisoning persistence, 50.3% full Write–Execute success, 56.1% selective repair among successful poison cases | Reproduce all seven lifecycle checkpoints and separate persistence, consequence and repair |

These are externally reported references. Pin each paper/leaderboard version and protocol before running; do not infer local MACE performance from them.

## 3. MACE baseline ladder

Run on identical raw sessions/trajectories and benchmark-native splits: no memory; full-history information reference; raw-session BM25 (#13346); dense; hybrid BM25+dense/RRF; structured facts or versioned key-value; episodic summary/cue-indexed hierarchy; causal/relational graph; typed-content/event-ID HDC; Liquid Broca recurrent-only; HDC + decision-sufficiency frontier/EVSI; and full evidence-gated MACE.

Do not compare memory substrates under different answer models and claim a memory-method improvement. Backbone comparisons are a separate lane. Pin each dense embedder, index, reranker, extractor, adapter and reader revision.

## 4. Required metrics

Task: official benchmark metrics, factual/temporal/cause/state-update correctness, tool/action/argument correctness, appropriate abstention, decision regret.

Governance: revoked/stale/superseded evidence use, unauthorized action, cross-user/role disclosure, unsupported provenance, contradiction, erasure resurrection, correction propagation and appropriate non-use.

Resources: p50/p95/p99 end-to-end latency, input/output/context tokens, model calls, CPU/GPU time, energy if measured, money, persistent bytes including indexes/sidecars/tombstones/journals/caches, write/consolidation/replay cost.

A retrieval hit is not a correct answer; a correct answer is not a correct action; a logical deletion test is not proof of physical erasure.

## 5. Proposed targets (not achievements)

- AML: +5 absolute points over the open-source leader in each separate track under official evaluation, with capability-level non-regression; report commercial references separately.
- LongMemEval-V2: LAFS gain above +2.823 and a better accuracy/latency Pareto frontier under the exact official reader and budget; separately seek an operating point below 30 s without material accuracy loss.
- AMA-Bench: at least 60% average under the paper's exact Qwen3-32B setup, with causal/state-update strata reported separately.
- COLLABMEM: 74.6% non-Oracle average (+5 points over 69.6%) with no regression in role access, visibility, provenance, authority or retention.
- TimlMemBench: +5 absolute points on preregistered raw-native governance dimensions versus strongest reproduced baseline, without sacrificing restraint.
- MemSecBench: zero unauthorized side effects and zero post-erasure resurrection on deterministic fixtures; report repair performance separately.
- Efficiency: within a preregistered 1-point quality non-inferiority margin, at least 25% lower all-in lifecycle cost; or a paired +3-point quality gain at matched total budget.

These are stretch targets. Beating an aggregate score while violating a hard governance gate is failure.

## 6. Statistical protocol

- Split by independent source/world/task clusters, never by correlated message or intervention rows.
- Keep calibration, tuning and test clusters disjoint. Freeze threshold, loss, policy, prompt, model, judge, tools, split and metric definitions before held-out evaluation.
- Use paired seeds and cluster-level uncertainty for ablations. Report coverage-risk and quality-cost curves, not one scalar.
- Keep benchmark-native official scores apart from MACE sidecars; sidecars must not alter official scoring.
- Use evaluator-only oracle as an upper bound, never a deployable baseline.
- Candidate execution must not see labels, future observations, answer keys or evaluator utilities.
- Treat published external results as references until exact code, reader, dataset and evaluator are reproduced.

## 7. Implemented validator

The standalone Rust crate validates one run record and sets of runs claimed to be matched. It rejects cross-track mixing, changed reader/prompt/judge/tool/evaluator/budget, evaluator-label exposure, duplicate arms, metric-definition mismatches and invalid latency quantiles. It emits a deterministic comparison receipt. The receipt is an integrity commitment, not an external signature; the validator does not compute a leaderboard or statistical significance.

## 8. Related work

- #8427: cross-benchmark cost-quality frontier for Liquid Broca.
- #13310: decision-sufficiency frontier and EVSI planner.
- #13346: raw-session BM25 baseline adapter.
- #13335 / #13332 / #13999: selective evidence sufficiency contract and cluster-aware calibrator.
- #13825: governed-memory external benchmark bridge.
- #13967 / #13973 / #13985: use-time revalidation, exact context exposure lineage and paired evidence-span ablation.