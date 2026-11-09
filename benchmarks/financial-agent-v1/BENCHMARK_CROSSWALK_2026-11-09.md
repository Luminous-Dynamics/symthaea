# Financial Research Benchmark Crosswalk — 9 November 2026

This document extends the first benchmark map in [SYM-FIN-EVAL-002 #11233](https://github.com/Luminous-Dynamics/symthaea/issues/11233). It is a benchmark-discovery and ownership crosswalk, not a leaderboard result.

## 1. Benchmarks by construct

| Benchmark | Primary construct | Size / design reported by source | What it adds here |
|---|---|---|---|
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

## 3. Industry comparators

- **Bloomberg ASKB / Terminal:** compare capability families such as market coverage, firm knowledge, news, research discovery, workflow integration and source attribution. No scored superiority claim unless exact product versions, lawful data rights, task inputs, user workflows and budgets can be matched.
- **OpenBB:** compare open/extensible data access, workspace integration and deployment. Prefer interoperability over duplicating connectors and charting without a distinct advantage.
- **Robinhood Cortex / agent practices:** compare trajectory-level tool use, evidence handling and safety on suitable tasks. Mycelix remains canonical for financial state and effect authority.
- **LEAN / FinRL:** useful quant/backtest and execution research comparators for #11234, not substitutes for finance research-agent QA.

A directory/aggregator such as [FinanceBenchmark.ai](https://financebenchmark.ai/benchmarks) can provide a discovery map, but its composite score and leaderboard methodology are publisher-defined. Treat aggregator ranks as leads for further investigation; use primary papers, exact benchmark versions and locally reproduced runs for claims.

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
