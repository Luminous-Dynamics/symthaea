# Financial Research Benchmark Crosswalk — 9 November 2026

This document extends the first benchmark map in [SYM-FIN-EVAL-002 #11233](https://github.com/Luminous-Dynamics/symthaea/issues/11233). It is a benchmark-discovery and ownership crosswalk, not a leaderboard result.

## 1. Benchmarks by construct

| Benchmark | Primary construct | Size / design reported by source | What it adds here |
|---|---|---|---|
| [Stanford/industry Finance Agent v1.1](https://www.vals.ai/benchmarks/finance_agent) | 537 entry-level financial-analyst questions; 2026 AI Index reports a 63.33% leading score and 50.62% at the low end of its displayed cohort | End-to-end entry-level analyst research, retrieval, market research and financial projections | Exact benchmark version, harness, model snapshot and date must match before numeric comparison; a published leader is not a universal competence or human-parity claim |
| [FORCE-Bench](https://arxiv.org/abs/2607.19409) | 251 expert-annotated enterprise-finance queries across three workflows | Financial obligations/ERP research, financial-entity performance research and business briefs; eight rubric dimensions | Enterprise workflow and operational constraints are valuable; ERP data semantics differ from public-company research and synthetic scenarios |
| [AgenticFinBench](https://arxiv.org/abs/2611.04813) | Real-world agentic financial research requests; claim-level evidence audit | 230 tasks; 8 categories; 2 languages; 4 complexity levels | Realistic finance research, source grounding, independent claim audit and harness sensitivity |
| [FinTrace](https://arxiv.org/abs/2604.10015) | Financial tool-use trajectories | 800 expert-annotated trajectories across 34 task categories; 9 metrics; 8,196 training preference trajectories | Tool/action correctness, process quality, execution efficiency and final-output quality |
| [ICBCBench](https://arxiv.org/abs/2606.17458) | Financial deep research and long-form analyst reports | Industry collaboration reported as 50+ experts across 40+ organizations; objective and subjective tracks | Industry-facing report tasks, expert alignment, citation consistency and source quality |
| [FinDeepResearch](https://arxiv.org/abs/2510.13936) | Multilingual, multi-market company analysis | 64 listed companies, 8 markets, 4 languages, 15,808 rubric items | Cross-market and multilingual corporate research |
| [BigFinanceBench](https://arxiv.org/abs/2606.03829) | Open-ended professional-finance tasks with checkable derivation rubrics | 928 expert-authored tasks; 36,241 rubric points | Step-wise derivation and expert rubric scoring, not just end-answer judging |
| [ProFinR / ProFinAgent](https://proceedings.mlr.press/v306/huang26ap.html) | Professional financial analysis with tool use | 528 tasks; 53 financial tools in 13 categories | DAG/parallel tool orchestration and case-based memory ablations |
| [FinancialAuditBench](https://arxiv.org/abs/2609.32835) | Synthetic multi-file financial audit engagements | 90 tasks across 6 engagements, averaging 179 files each | Privacy-aware audit-workpaper retrieval and multi-file evidence analysis |
| [FinanceBench](https://arxiv.org/abs/2311.11944) | Public-company filing question answering | Public benchmark; source-specific task split | Filing-grounded QA baseline |
| [FinQA](https://arxiv.org/abs/2109.00122), [ConvFinQA](https://arxiv.org/abs/2210.03849), [TAT-QA](https://aclanthology.org/2021.acl-long.254/) | Numerical, conversational and table/text financial reasoning | Distinct datasets and test splits | Targeted regression tasks; scores are not mutually interchangeable |
| [FINBALANCE](https://aclanthology.org/2026.findings-emnlp.675/) | Multi-document source-to-journal-to-statement reconstruction | 710-record evaluation split; 23 inconsistency labels in the paper | Accounting reconciliation and contradiction detection |

Published counts and scores are statements from the referenced papers/projects. They are not independent reproductions and are not results achieved by Symthaea.

## 2. Keep benchmark ownership separate

- **General finance research correctness and grounding:** SYM-FIN-EVAL-002 (#11233).
- **Whole financial-agent trajectory and authority safety:** SYM-FIN-EVAL-001 (#5399).
- **Tool-use trajectory/action metrics:** FinTrace adapter, composed into #5399/#11233.
- **Long-form industry research reports:** ICBCBench adapter, where rights and evaluator access permit.
- **Company and cross-market reports:** FinDeepResearch adapter, where terms and split semantics permit.
- **Document/numerical subtasks:** FinanceBench, FinQA, ConvFinQA, TAT-QA and BigFinanceBench.
- **Accounting ledger / journal / financial statements:** Mycelix [FIN-ACC-BENCH-001 #5782](https://github.com/Luminous-Dynamics/mycelix/issues/5782); do not duplicate its ledger or independent accounting oracle.
- **Audit workpapers / engagement analysis:** FinancialAuditBench; keep it separate from general company research and journal correctness.
- **Strategy performance / portfolio outcomes / market frictions:** SYM-QUANT-EVAL-001 (#11234), not #11233.

## 2a. Evaluator reliability is a competitive dimension

[Better Deck or Different Judge? (30 September 2026)](https://arxiv.org/abs/2609.39958) studied an agentic corporate/investment-banking presentation harness and five judges. In its reported comparison, judges scored the complete system 20.4–33.6 points above the same model prompted directly, while the margin against direct Opus generation ranged from -4.7 to +0.8; repeat grading also changed scores on unchanged decks. This is a study-specific result, not proof that agentic systems do not help. It does show that judge calibration, repeatability and ranking stability are necessary before interpreting score gains as system gains.

The Stanford [AI Index Report 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report) reports 537 questions in Finance Agent v1.1, with its displayed cohort ranging from 50.62% to 63.33%. Treat these as version-specific task scores, not financial-professional competence. Our external baseline table should pin the exact version and use it as a reference point, not mix it with ProFinR, FORCE-Bench or other scales.

## 3. Industry comparators

- **Bloomberg ASKB / Terminal:** compare capability families such as market coverage, firm knowledge, news, research discovery, workflow integration and source attribution. No scored superiority claim unless exact product versions, lawful data rights, task inputs, user workflows and budgets can be matched.
- **OpenBB:** compare open/extensible data access, workspace integration and deployment. Prefer interoperability over duplicating connectors and charting without a distinct advantage.
- **Robinhood Cortex / agent practices:** compare trajectory-level tool use, evidence handling and safety on suitable tasks. Mycelix remains canonical for financial state and effect authority.
- **LEAN / FinRL:** useful quant/backtest and execution research comparators for #11234, not substitutes for finance research-agent QA.

A directory/aggregator such as [FinanceBenchmark.ai](https://financebenchmark.ai/benchmarks) can provide a discovery map, but its composite score and leaderboard methodology are publisher-defined. Treat aggregator ranks as leads for further investigation; use primary papers, exact benchmark versions and locally reproduced runs for claims.

## 3.1 Current proprietary analyst-agent comparator — Vals Finance Agent v2.1

The official [Vals Finance Agent v2.1 leaderboard](https://www.vals-ai.com/benchmarks/fabv2-1), updated 5 November 2026, reports partial-credit accuracy and cost per test as follows:

| Rank | System | Partial-credit accuracy | Reported cost/test | Reported duration/test |
|---:|---|---:|---:|---:|
| 1 | Claude Opus 5.5 | 67.85% | $8.91 | 33m00s |
| 2 | Gemini 4 Argon | 67.66% | $4.32 | 13m13s |
| 3 | Wonder Full | 67.58% | $2.68 | 14m10s |
| 4 | Inkling 2 Small | 67.54% | $0.18 | 4m56s |
| 5 | Claude Sonnet 5.5 | 67.28% | $6.95 | 34m26s |
| 6 | Gemini 4 Carbon | 67.16% | $7.29 | 29m40s |

Interpretation:
- The top three are separated by only 0.27 percentage points. Without confidence intervals and task-level paired results, this is not evidence of a meaningful capability ordering.
- Inkling 2 Small is 0.31 points below the leader while reporting about 1/50 the per-question cost. This is a better reminder to evaluate the quality-cost frontier than to optimize rank alone.
- Vals v1.1 and v2.1 use different question sets; do not compare their scores as a time series.
- These are publisher-reported results on a proprietary task set. They are not a Symthaea result, independent reproduction, human-expert score or proof of broad investment/research competence.
- A numerical match against this benchmark is only meaningful if the same release, task access, scoring rules, tool availability, run budget, refusals/fallback handling and model versions can lawfully be reproduced. Otherwise treat it as a capability/cost reference, not a scored head-to-head.

The broader [Vals Finance Index](https://www.vals-ai.com/benchmarks/finance_index) covers additional finance work (research/modeling/tax) and applies publisher-defined GDP-share weights. Keep it separate from the narrower Finance Agent v2.1 task score.

## 4. Matched evaluation rules

For every external dataset or product:
- pin release, version, split, task definition, license/terms, evaluator and retrieval permissions;
- preserve the exact prompt, model/provider, tools, budget, retry policy, source frontier and all costs;
- report the complete task denominator, timeouts, refusals, abstentions, exclusions and not-comparable cases;
- separate expert rubric quality, numerical correctness, evidence support, unsupported claims, temporal leakage, tool correctness, calibration, latency and cost;
- use hidden or evaluator-held confirmation sets only when their custody and separation are genuine;
- do not merge scores across task constructs or benchmark families.

## 5. Research targets

The strategic objective is not to win a benchmark by maximizing answer fluency. Aim to improve:
1. material-claim support precision and recall;
2. temporal-frontier correctness and resistance to future-information leakage;
3. contradiction and source-family dependence detection;
4. task completion at matched tool/time/cost budget;
5. calibrated abstention and selective risk;
6. reproducibility and independent replay;
7. robustness across market, language, jurisdiction and source-family holdouts.

These are target dimensions. Numeric thresholds must be pre-registered after the development cohort and baseline dispersion are characterized. No alpha, investment performance, legal compliance or universal professional competence claim follows from a research-agent benchmark pass.

## 6. Rights and data boundaries

An openly published paper or repository does not automatically permit dataset redistribution, model endpoint access, proprietary source reuse or benchmark answer-key mirroring. Record each task set's license and permitted-use conditions. A benchmark adapter that requires user-supplied files is preferable to an unauthorized dataset copy.
