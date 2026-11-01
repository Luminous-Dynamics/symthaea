# POD-Meta External Benchmark Matrix v1

**Last checked:** 2026-10-31  
**Purpose:** pin external comparisons to their own datasets, evaluators, release versions and metric meanings. This document is a research plan, not a report of Symthaea benchmark results.

## Evaluation thesis

POAR/POD-Meta is not intended to replace existing public benchmarks. It should supply a controlled causal experiment for one additional question:

> At a matched model, tool set and total budget, does proof-obligation-aware operation selection produce verified conclusions or state transitions at lower all-in cost, without increasing false support, unsafe actions or incorrect abstention?

Use the official evaluator for every external suite. Keep every suite's native metric separate. A result on a public benchmark does not establish correctness outside its task distribution.

## Benchmark registry

| Benchmark / paper | What it measures | Published scale or reference result | Symthaea integration |
|---|---|---|---|
| [Think-at-Hard (ICML 2026)](https://proceedings.mlr.press/v306/fu26t.html) | Selective latent iteration; tests whether additional recurrent compute helps or harms individual token predictions | With identical parameter counts, 3.8–4.4 percentage-point gain over always-iterate baselines while skipping 93% of token iterations; an oracle iteration policy improves by up to 7.3% | Methodological comparator for selective computation. Do not equate token iterations with Symthaea operations or compare those percentages directly. |
| [TEMPO (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.291/) | Multi-domain temporal reasoning-intensive retrieval; step-wise gold evidence and temporal completeness | 1,730 queries, 13 domains, 3,976 decomposed steps; DiVeR's reported NDCG@10 is 32.0 and Temporal Coverage@10 is 71.4% | Evaluate temporal retrieval before proof construction; report Temporal Coverage@k and Precision@k separately and use official split/evaluator. |
| [TEMPS (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.2657/) | Temporal sentence embeddings / retrieval ranking | On TS-Retriever, reported R@1 rises from 19.92 to 25.39; improvements are reported across three temporal benchmarks | Include the paper's checkpoint/ranking method as a retrieval arm where licensing and compute permit. Do not conflate TEMPS R@1 with TEMPO coverage. |
| [Vera-Life-Eval (GroundLM 2026)](https://aclanthology.org/2026.groundlm-1.18/) | Claim-level lifecycle faithfulness: target-time validity, supersession, conflict and provenance | 8,500 questions; 41,203 versioned evidence pieces; six lifecycle settings. On a shared Qwen3-8B generator LifeEM rises from .374 for the best prompt-only baseline to .623 for Vera-Life | Direct external test of bitemporal/lifecycle evidence. Compare current retrieval, temporal filter, lifecycle graph, proof-aware packet and combined controller. Preserve LifeEM plus subcase metrics. |
| [LitTraceQA (GroundLM 2026)](https://aclanthology.org/2026.groundlm-1.81/) | Multi-stage paper retrieval, typed evidence location and answer jointness | 4,978 questions grounded in 4,859 papers; 64.85% require multiple papers; reports paper retrieval, grounding, answer correctness and strict joint success | Test minimum sufficient evidence packets with exact page/figure/table/equation/text/citation locators. A correct answer without required evidence is not full success. |
| [GRACE (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1582/) | Step-level reasoning faithfulness and separation of grounding from inference failures | Use the paper's official annotated splits and metrics; do not substitute POAR labels | Compare proof/trace steps with evidence actually available at that step. Keep native faithfulness scores separate. |
| [FaithCoT-Bench (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html) | Faithfulness of generated reasoning traces | Use the official evaluator and pinned paper/checkpoint configuration | Evaluate whether exposed reasoning/trace claims are causally grounded. It does not by itself prove a formal derivation or a correct external state transition. |
| [AgentRx (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.598/) | Critical failure step localization and grounded failure categories | 170 annotated trajectories across 11 task settings; paper reports 75% average step-localization improvement over prior systems | Evaluate earliest invalid transition, cause and evidence. Preserve its native judge-assisted result and separately report deterministic POAR-RX state predicates. |
| [ParaRecover (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.975/) | Error localization and recovery in multi-turn parallel tool-use | 10,626 instances, 14 error types, two difficulty levels; SDE covers structural integrity, diagnostic reasoning and evolutionary strategy | Primary process-level recovery comparator. POAR-RX adds authority, independent state read-back, compensation safety and truthful residual-state reporting. |
| [OSWorld 2.0 / 2.1](https://osworld-v2.xlang.ai/) | Long-horizon computer-use workflow execution | Official OSWorld 2.0 page: 108 tasks; median human execution around 1.6 hours; average 318 tool calls for Claude Opus 4.7 maximum-thinking run; best stated 500-step binary completion 20.6% with 54.8% partial for Claude Opus 4.8. Third-party records list 81.8% partial on v2.1 for Claude Opus 5.5, explicitly as provider self-report | Use version-pinned task/evaluator releases. Separate binary completion from partial reward. Treat provider-reported v2.1 rows as non-independent evidence until independently reproduced. |

## Important OSWorld comparability rule

OSWorld 2.0 and 2.1 scores must not be collapsed. The official OSWorld 2.0 page describes its own task/evaluator version and reports 20.6% binary completion / 54.8% partial score for Claude Opus 4.8 under its stated setup. A third-party v2.1 score registry reports Claude Opus 5.5 at 81.8% partial, but says its rows are provider self-reports, not independent reruns, and settings differ. See:
- Official project: https://osworld-v2.xlang.ai/
- Provider announcement: https://www.anthropic.com/claude-opus-5-5
- Provider-score registry caveat: https://benchlm.ai/benchmarks/osworld-2-1

These figures are not a like-for-like improvement estimate. Report exact release, model, effort, step cap, task subset, binary/partial metric, and whether the score was independently run.

## Causal experiment protocol

For each external benchmark where architecture integration is allowed:

1. Freeze the benchmark version, official evaluator commit, dataset split and task IDs.
2. Match underlying model/checkpoint, prompt, context, tools, permissions, decoding settings, task deadlines and total budget.
3. Compare at least: single-pass/reactive; fixed-order compute; generic VOC; budget-matched random; proof-obligation-guided selection; proof-guided + recovery; and synthetic oracle upper bound.
4. Pre-register the target metric and hard safety invariants before confirmatory runs.
5. Report per-task and per-family results, not only averages. Cluster confidence intervals by task template/source family/tool family where applicable.
6. Report actual cost: tokens, tool calls, p50/p95 latency, CPU/GPU time, memory, energy method, money and human intervention.
7. Preserve native benchmark metrics. Publish a separate Symthaea Pareto frontier for verified correctness, safety and all-in cost.

## New POAR/POD-Meta metrics

These complement rather than replace public benchmarks:

- Verified obligations discharged per abstract work unit and, later, measured compute unit.
- Cost to first sufficient proof and cost regret against evaluator-only oracle selection.
- False support / false closure / missed mandatory verification.
- Correct abstention and lifecycle-aware evidence rejection.
- Rejected evidence acquisition cost versus genuinely redundant/irrelevant operation cost.
- Critical-operation omission rate.
- Verified state-transition rate, unauthorized effects, duplicate effects and residual-state correctness.
- First invalid transition localization and time/operations to recovery.

The oracle is a diagnostic lower bound only. Never allow it to participate in planner execution or public leaderboard entries.

## Promotion gates

- Qualify the exact-head Rust source and matrix before publishing any numerical result.
- Replace public fixed fixtures with generator/operator-family held-out splits.
- Move oracle data and confirmatory seeds to a separately permissioned evaluator domain.
- Add a strong independently implemented saga/retry controller before claiming recovery superiority.
- Measure actual compute and monetary cost; abstract work units are only a deterministic harness unit.
- Do not optimize call count by omitting mandatory verification or by misclassifying rejected evidence as waste.
- Publish all failure families and confidence intervals. Hard safety regressions block promotion regardless of average task score.
