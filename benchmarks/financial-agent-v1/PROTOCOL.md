# Financial Research Agent Benchmark v1

**Status:** research protocol / development fixture foundation  
**Parent:** [SYM-FIN-EVAL-002 #11233](https://github.com/Luminous-Dynamics/symthaea/issues/11233)  
**System owner:** Symthaea (research and evaluation); Mycelix (evidence, identity, financial state and authority); Sol-Atlas (research cockpit); Symtropy (physical/economic simulation where independently qualified).

## 1. Evaluation question

Can a financial-research workflow produce more decision-useful, correctly grounded, reproducible analysis than strong model-only, retrieval, tool-using, structured-agent and human workflows when all receive the same permitted evidence, source frontier, time and cost budget?

The benchmark evaluates **research quality**, not investment return, broker execution, or policy desirability. A fluent answer is not a correct answer; a correct final number is not an auditable derivation.

## 2. Why existing SOTA is not one score

Use these as complementary, non-interchangeable cohorts:

| Benchmark | Scope | What it adds | Limitation for our use |
|---|---|---|---|
| [BigFinanceBench](https://arxiv.org/abs/2606.03829) | 928 expert-authored open-ended tasks; 36,241 rubric points | Point-weighted, independently checkable derivation steps; paper reports best evaluated system at 58.8% rubric score | Its task/rubric design and evaluated cohort define the result; not a universal professional-competence score |
| [ProFinR / ProFinAgent](https://proceedings.mlr.press/v306/huang26ap.html) | 528 expert-designed professional-finance problems; 53 tools in 13 categories | Structured DAG tool execution and case-based memory; paper reports 49.81% gain and 47.1% lower inference latency against its sequential baselines | Reproduction requires exact benchmark/tool/model conditions; those gains do not establish alpha |
| [AgenticFinBench](https://arxiv.org/abs/2611.04813) | 230 real-world requests; 8 categories, 2 languages and 4 complexity levels | Two-stage claim audit; paper reports a 7.58-point reduction to the initial baseline leader after hallucination audit | Recent arXiv preprint; small task set and reference-free evaluation require independent replication |
| [FinancialAuditBench](https://arxiv.org/abs/2609.32835) | 90 tasks across 6 synthetic audit engagements, averaging 179 files each | Privacy-aware synthetic engagements using differentially private aggregate priors and extensive expert review | Audit-workpaper specialization, not general market research |
| [FinanceBench](https://arxiv.org/abs/2311.11944), [FinQA](https://arxiv.org/abs/2109.00122), [ConvFinQA](https://arxiv.org/abs/2210.03849), [TAT-QA](https://aclanthology.org/2021.acl-long.254/) | Filing QA, numerical reasoning, dialogue and table/text tasks | Mature task-level regression suites | Isolated subtasks do not measure an end-to-end analyst workflow |

Published scores are not directly comparable unless the benchmark, split, evaluator, tools, model version, data frontier and budget match. Do not concatenate scores into one leaderboard. FinanceBenchmark's own published methodology attributes scores to underlying evaluations and uses tier weights; treat any composite as that publisher's method, not an objective universal measure ([methodology](https://financebenchmark.ai/methodology)).

## 3. Competitive position

| System | Documented strength to benchmark | Our response |
|---|---|---|
| Bloomberg Terminal + ASKB | Interconnected market data, news, research, analytics, firm knowledge and multi-agent conversational workflows ([official ASKB](https://professional.bloomberg.com/products/bloomberg-terminal/ai/askb/)) | Do not claim parity without licensed feeds and matched user tasks. Differentiate through open/federated deployment, explicit evidence frontiers, machine-replayable provenance and independently scored claim support |
| OpenBB | Open financial workspace / extensible data-provider and analysis ecosystem | Integrate or interoperate where useful; avoid rebuilding connectors, charts or data APIs without a clear advantage |
| Robinhood Cortex / agent tooling | Account-context assistant and published agent trajectory/evaluation practices | Match whole-trajectory evaluations while keeping Mycelix as financial state/authority owner and Symthaea read-only/proposal-only in qualified V1 |
| Finance research agents | ProFinAgent DAG + case memory; AgenticFinBench claim audits; BigFinanceBench derivation rubrics | Test DAG, memory and specialist teams as ablations, not assumed wins; require claim-level evidence closure and leakage-resistant temporal replay |

**Data boundary:** open source code does not grant exchange, newswire or proprietary research entitlements. Public data may still have attribution, rate-limit, database-rights or redistribution restrictions. Use licensed adapters for paid feeds; store license/permission metadata with every evidence artifact. Never scrape around or redistribute restrictions.

## 4. Architectural division of responsibility

- **Mycelix:** source/provider identity, artifact and observation identity, provenance, rights/consent, evidence currentness, canonical financial subjects and instruments, ledger/accounting/settlement, authority, federation and immutable historical projections.
- **Symthaea:** research planning, evidence selection, entity-resolution candidates, competing hypotheses, calculations, forecasts, counterfactuals, uncertainty, value-of-information and proposal candidates. No canonical financial truth, custody or self-authorized financial effects.
- **Sol-Atlas:** human-facing evidence cockpit, temporal/event maps, entity and supply-chain graphs, spatial-economic overlays, watchlists and provenance traversal. Visualization cannot upgrade evidence or authority.
- **Symtropy:** explicitly bounded physical/operational and economic scenario models; model output must not become measured fact.

No duplicate ledger, broker adapter, portfolio ontology, general market data platform or underwriting engine is introduced here.

## 5. Paradigm shifts under test

1. **Claim graph before prose.** Every material statement is typed as observed, derived, hypothetical, forecast, interpretation or normative. It points to evidence, derivations, assumptions and uncertainty—or is labelled unsupported.
2. **Dual-time / point-in-time replay.** Keep valid/event time separate from known/recorded time. A historical analyst sees only what was knowable at its decision timestamp, including then-current revisions and source latency.
3. **Source-family independence.** Ten aggregators copying one filing are one provenance family, not ten independent confirmations.
4. **Evidence conflicts are retained.** Reconciliation produces a conflict object and disposition, not silent winner selection or false consensus.
5. **Decision value over information volume.** Rank candidate evidence requests by expected decision improvement net of acquisition cost, delay and risk; generic entropy reduction is insufficient.
6. **Multi-agent structure as a hypothesis.** Compare sequential, DAG, specialist and case-memory approaches under identical tool, time and token budgets. Partition memory by permission and cohort; holdout answers never enter production memory.
7. **Calibrated abstention.** Unknown, stale, conflicted and not-comparable are valid outputs. Measure selective risk versus coverage, not only forced-answer accuracy.
8. **Whole-economy context.** Link companies and instruments to suppliers, customers, energy, transport, assets, geography, labor, public infrastructure and ecological dependencies where evidence supports the relation.
9. **Federated data sovereignty.** Support multiple providers, local deployments and jurisdiction profiles without making one data vendor or monetary regime canonical.
10. **No single welfare or investment score.** Preserve Pareto trade-offs and the distinct epistemic, financial, distributional, physical and ecological coordinates.

## 6. Cohorts and leakage controls

Keep the following cohorts separate in reports:

- **D — Public development:** legally usable public tasks for implementation and regression.
- **S — Sealed synthetic confirmation:** precommitted generator, hidden seeds/parameters and independently held oracle.
- **H — Historical point-in-time:** rights-cleared entities/events reconstructed from exact historical known-at frontiers.
- **P — Prospective shadow:** timestamped forecasts and hypotheses before outcomes; read-only, no capital allocation.
- **A — Analyst comparison:** blinded human, human+tools and agent workflows on matched evidence and time budgets.

Split by issuer/entity, time period and provenance family, not random rows. Never tune on sealed answers. Record contamination exposure and benchmark access. Keep failures, timeouts, abstentions, unsupported cases and not-comparable dispositions in the denominator.

## 7. Matched baseline tournament

At minimum:

1. Deterministic arithmetic / rules / retrieval baseline where applicable.
2. Model-only (no tools).
3. Strong model + standard retrieval/search.
4. Sequential tool-using workflow.
5. DAG/parallel orchestration.
6. Specialist-agent decomposition.
7. Case-based memory, with cohort-isolation ablation.
8. ProFinAgent reproduction where its code, dependencies, endpoints and rights permit.
9. Human analyst and human+tool workflows.

Pin model/provider/version, prompts, tools, retrieval index, exact evidence frontier, token/tool/time budgets, retry policy, total cost, software versions and evaluator version. Commercial products such as Bloomberg ASKB should be called capability comparators unless they can legally be run on the same inputs and workload.

## 8. Score vector

Publish each axis separately, with numerator, denominator, exclusions, confidence interval and cohort:

- task rubric correctness and expert-adjudicated decision usefulness;
- derivation-step correctness / numerical reproducibility;
- material-claim support precision and recall;
- unsupported and contradicted material-claim rates;
- source, entity and instrument resolution accuracy;
- point-in-time / known-at correctness and future-information leakage;
- contradiction detection and preservation;
- evidence-family diversity and independence;
- retrieval recall and tool-selection correctness;
- evidence-gap precision/recall and decision value;
- probability calibration (Brier/log loss); distributional calibration (CRPS, pinball loss or WIS, as appropriate);
- selective risk versus coverage / abstention quality;
- assumption, scenario and model-lineage completeness;
- explanation-to-evidence coverage;
- privacy, permission and authority hard-gate violations;
- p50/p95/p99 latency, token/tool counts, and fully loaded cost per correct, grounded task;
- robustness by language, jurisdiction, asset class, source type, complexity and provider.

Use paired comparisons, bootstrap confidence intervals or other pre-registered uncertainty estimates. Correct for multiple comparisons where appropriate. Preserve raw task-level results.

## 9. Non-compensable hard gates

Any of the following blocks a qualified label, whatever the average score:

- future-known evidence in a historical information frontier;
- missing/forged/incorrect artifact identity or digest;
- unsupported material claim labelled verified;
- a name/ticker-only join silently treated as canonical identity;
- source-family duplicates represented as independent corroboration;
- conflict silently discarded;
- scenario or forecast overwrites observed state;
- incomplete connected accounts or entity coverage represented as complete;
- read-only mode has a reachable provider/broker write capability;
- recommendation/proposal creates authority or executes an order;
- secret, raw account identifier or protected data leak;
- prohibited/unlicensed benchmark or source data copied into artifacts;
- failed, unknown or timed-out required verification treated as pass.

## 10. First public evidence pack

Use source-specific adapters and preserve release, query, retrieval time, coverage, terms and digests:

- SEC EDGAR submissions and XBRL APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- GLEIF LEI data/API: https://www.gleif.org/en/lei-data/gleif-api/
- FRED/ALFRED (including vintage/real-time-period semantics): https://fred.stlouisfed.org/docs/api/fred/overview.html
- BIS SDMX API: https://stats.bis.org/api-doc/v2/
- OECD income and wealth distribution: https://www.oecd.org/en/data/datasets/income-and-wealth-distribution-database.html
- South African Reserve Bank Web API: https://custom.resbank.co.za/SarbWebApi/swagger/index.html
- UN System of Environmental-Economic Accounting: https://seea.un.org/en/methodology

Begin with a single narrow workflow: **explain a company risk event caused by an evidenced supplier/energy/shipping disruption**. Every causal link remains a hypothesis until supported by data and appropriate identification.

## 11. Runner-free implementation sequence

1. Freeze this protocol and strict task/evidence manifest schemas.
2. Add a synthetic positive/negative/indeterminate development fixture with a public development oracle clearly labelled non-sealed.
3. Implement a dependency-light Rust verifier in an isolated package: schema checks, IDs, digests, temporal frontier, claim/evidence closure, units, disposition and score-vector output.
4. Add negative fixtures and independently calculate expected outcomes.
5. Run locally when a Rust toolchain is available; CI is additive, not a substitute for local evidence. This file/schema commit alone is not a qualification.
6. Add one lawful external benchmark adapter at a time; record license status before importing data.
7. Only then run model comparisons and prospective shadow tests.

## 12. Stretch hypotheses (not achieved results)

After a baseline is frozen, test:
- ≥15% relative reduction in unsupported material claims against the strongest matched retrieval/tool baseline;
- ≥10% relative improvement in expert rubric quality under equal total cost/time;
- zero false acceptances on the finite public identity/temporal/authority mutation panel;
- better calibration and abstention on sealed and prospective cohorts;
- lower cost per correct, fully grounded task, not merely lower tokens.

A positive synthetic result is a synthetic result. No benchmark PASS establishes alpha, investment returns, real-world causality, suitability, legal compliance, macroeconomic superiority, or universal Bloomberg parity.
