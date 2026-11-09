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
| [Herculean](https://arxiv.org/abs/2605.14355) | v4 reports 424 offline and 246 live instances across Trading, Hedging, Market Insights and Auditing | MCP-based end-to-end workflows with task-specific tools and operational constraints | Trading/hedging/auditing workflows are not the same construct as evidence-grounded company research; match only comparable tasks |
| [FinTrace](https://arxiv.org/abs/2604.10015) | 800 expert-annotated tool-use trajectories across 34 financial task categories; companion preference set of 8,196 trajectories | Nine metrics across action correctness, execution efficiency, process quality and output quality | Excellent trajectory comparator, but tool/action correctness does not alone establish claim truth or financial decision quality |
| [ICBCBench](https://arxiv.org/abs/2606.17458) | Industry-consortium deep research; 50+ experts across 40+ organizations, with objective and subjective tracks | Retrieval/reasoning plus expert-facing long-form report quality, citation consistency and source quality | Access/terms and evaluator protocol must be checked; a consortium label is not a universal human gold standard |
| [FinDeepResearch](https://arxiv.org/abs/2510.13936) | 64 listed companies across 8 markets, 4 languages and 15,808 rubric items | Hierarchical company-analysis reports spanning data recognition, calculation, abstraction and interpretation | Public benchmark and its rubric define its own scope; preserve market/language splits and avoid equating LLM rubric scores with human truth |
| [FinBalance](https://arxiv.org/abs/2606.15949) | 710-record evaluation split; eight industries, three period types and 23 inconsistency labels | Multi-document source-to-journal-to-statement reconciliation; ledger-replayed balance sheets | Accounting-specific; compose with Mycelix FIN-ACC-BENCH rather than duplicate its ledger/oracle |

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

## 8b. Source timestamp precision

The v0.1 shared temporal kernel (#11241 / PR #11253) accepts only exact, offset-aware availability instants. Date-only filing dates and macro vintage dates must not be converted to midnight and treated as exact intraday availability. Preserve source precision; use an explicit interval or return INDETERMINATE when the possible availability interval straddles the decision frontier. The interval-censored extension is tracked in [SYM-FIN-TIME-002 #11256](https://github.com/Luminous-Dynamics/symthaea/issues/11256). A date can identify a vintage without proving its release hour. The shared temporal programme also tracks [unresolved revision forks #11300](https://github.com/Luminous-Dynamics/symthaea/issues/11300): two admissible successors of the same source version remain contested until an explicit resolution, not “latest timestamp wins.”

## 8a. Evaluator, rubric and reference metrology

A frozen rubric, a committed reference digest or agreement among multiple LLM judges does not establish that the evaluation is valid. The scorer is itself a measurement instrument and must be qualified before it is used to claim a financial-agent improvement.

This composes, rather than duplicates:
- [EVAL-JUDGE-METROLOGY-001 #9022](https://github.com/Luminous-Dynamics/symthaea/issues/9022) — judge accuracy, repeatability, prompt/order sensitivity and calibrated uncertainty.
- [SCI-RUBRIC-EXPLOITABILITY-001 #9123](https://github.com/Luminous-Dynamics/symthaea/issues/9123) — rubric adequacy, score gaming and Goodhart effects.
- [SCI-REFERENCE-ASSURANCE-001 #9081](https://github.com/Luminous-Dynamics/symthaea/issues/9081) — evidence-backed reference labels and adjudicator provenance.
- [FIN-EVAL-METROLOGY-001 #11254](https://github.com/Luminous-Dynamics/symthaea/issues/11254) — finance-specific integration.
- [SYM-FIN-TIME-001 #11241](https://github.com/Luminous-Dynamics/symthaea/issues/11241) — shared source-version and decision-frontier admissibility.

Keep four objects distinct:

1. **Claim correctness:** whether an atomic claim is accurate within its entity, instrument, unit, currency, period, jurisdiction and temporal scope.
2. **Trace support:** whether the permitted trace actually contains admissible evidence and a correct derivation supporting the claim.
3. **Reference confidence:** how strongly the independent reference process supports a disposition.
4. **Judge reliability:** how accurately and robustly an evaluator reproduces those reference decisions.

A true statement absent from the permitted trace can fail trace support without being factually false. A statement missing from the trace is not automatically contradicted. A valid URL does not prove that the cited span supports the claim.

Reference dispositions must preserve at least: supported, contradicted, mixed/contested, insufficient evidence, scope/unit mismatch, temporally inadmissible/stale and not evaluable. Do not force unresolved cases into binary gold labels.

For each scoring-eligible material claim, retain the exact claim, type, subject, temporal target/frontier, source snapshot digests, evidence spans, support/refute/qualify/contradict relations, source-family ancestry, calculation lineage and separate blinded adjudicator votes. Preserve abstentions and disagreement; adjudication must not erase initial votes.

Compare deterministic calculation/reference oracles where exact ground truth exists, a single LLM judge, a calibrated multi-judge jury, a two-stage claim-extraction/evidence-audit pipeline, blinded human evaluators and human adjudication. Pin model/provider/version, prompt/rubric digests, decoding, retrieval access, candidate ordering, repetitions and cost.

**Rater identity rule:** an AI adjudicator (including this assistant when used under a masked candidate ID) is an AI judge, not a human rater. Report AI votes separately from human votes and never count an AI score as a blinded-human observation. An AI-assisted human reviewer remains a human vote only when the human independently inspects the packet, makes and records their own decision, and retains responsibility for it. Preserve whether the reviewer saw other votes or prior scores.

Required evaluator metrics include class-specific false-accept/false-reject rates, balanced accuracy, macro-F1, claim-extraction and trace-support precision/recall, evidence-span and temporal-admissibility accuracy, amount/unit/currency/period consistency, calibration, abstention/selective risk, prompt/order/repeat flip rates, judge-to-human and human-human disagreement, rubric exploit success, latency and full cost. Cluster uncertainty by task and source family; several claims from one filing are not independent observations.

Adversarial controls must cover citation laundering, stale or superseded filings, wrong entity/instrument, mirrored sources presented as independent, causal overreach, future-known information, unit/currency/period mismatches, incomplete coverage, scenario-as-fact, unsupported forecast precision, irrelevant citation volume, verbosity/keyword stuffing, candidate-order swaps and consistently wrong but repeatable judges.

A consistently wrong judge must fail. A missing or uncertain reference must become contested/not-evaluable, not an invented gold answer. Surface-cue attacks must not improve scores without corresponding reference-grounded improvement.

AgenticFinBench's November 2026 preprint is particularly relevant: in its tested configurations, evaluator-harness differences explained more variation in evidence grounding than model differences (28.6% versus 8.8%). Treat this as a study-specific result, not a universal law; it is a strong reason to qualify the harness and reference process rather than attributing every improvement to the model. [AgenticFinBench](https://arxiv.org/abs/2611.04813)

FinanceBenchmark's methodology also keeps attempts, confidence intervals, held-out items, task-family scores and benchmark coverage visible. We should adopt those transparency practices without treating its weighted composite as a universal financial competence score. [FinanceBenchmark methodology](https://financebenchmark.ai/methodology)

No SOTA or Symthaea-superiority claim is permitted until blind reference-label audit, judge qualification and matched-budget holdout evaluation are complete.

## 8d. Second masked AI rubric audit

A second AI-only rubric consistency/traceability review is recorded in [BLINDED_AI_ADJUDICATION_REVIEW_2026-11-09.md](BLINDED_AI_ADJUDICATION_REVIEW_2026-11-09.md). It identifies a missing alias-to-canonical-evidence-ID crosswalk in the first pilot draft and one-point ambiguity in the competing-hypotheses score anchor. The alias crosswalk is now explicit in the pilot artifact.

The second reviewer saw the first score table, so this is **not an independent blind replicate**. It is a rubric audit only. The AI vote is not a human vote and must not enter human agreement or human-vs-agent comparisons.

## 8c. Masked AI adjudication calibration pilot

A deliberately synthetic calibration artifact is recorded in [BLINDED_AI_ADJUDICATION_PILOT_2026-11-09.md](BLINDED_AI_ADJUDICATION_PILOT_2026-11-09.md). One GPT-based AI adjudicator scored four masked, rubric-probe responses against the public supplier/energy fixture. The best response scored 20/20, a terse but safe response 14/20, and two responses containing temporal/unit or fabricated-financial claims failed hard gates.

**Interpretation ceiling:** the candidate responses were authored as synthetic rubric probes, not collected from independent model executions. This is a test that the scoring rubric can distinguish known-good and known-bad examples. It is not a model tournament, human-subject evaluation, inter-rater reliability result, generalized accuracy estimate or SOTA claim. The adjudicator is an AI system, not a human. Genuine blind human adjudication remains required for human-comparison results.

## 8e. Sequential and adversarial AI adjudication pilot

The second calibration artifact [BLINDED_AI_ADJUDICATION_PILOT_V2_2026-11-09.md](BLINDED_AI_ADJUDICATION_PILOT_V2_2026-11-09.md) adds synthetic cases for filing restatement without hindsight rewrite, same-name legal-entity collision, source-family duplication with a material contradiction, and annual-versus-half-year financial comparison. One GPT-6 AI adjudicator scored masked, authored probe responses. It is not a human vote, model-output comparison or benchmark result.

The next benchmark construct is [SYM-FIN-SEQ-001 #11290](https://github.com/Luminous-Dynamics/symthaea/issues/11290): prequential financial evidence revision and forecast calibration. It tests immutable snapshots at every frontier, justified updates as evidence arrives, conflict preservation, irrelevant-update stability and properly timed probabilistic forecasts. Keep this sequential construct separate from the static task/report score vector and from the quantitative portfolio tournament.

## 8f. Sequential supplier-correction development fixture

The first six-frontier public synthetic stream is in `sequential-v0/`:
- `supplier-correction.stream.json` — six versions/events and F0–F5 decision frontiers;
- `supplier-correction.oracle.json` — expected admissible evidence, claims, unknowns and hard gates;
- `README.md` — candidate delivery/oracle isolation contract;
- `BLINDED_AI_SEQUENTIAL_PILOT_2026-11-09.md` — one AI-only adjudication of authored masked trajectories.

The fixture tests an original supplier report, later source correction, buyer component purchase-share evidence, a copied mirror, inventory buffer, and a later shipment receipt. It explicitly makes F5 unavailable to F0–F4, keeps component-specific 35% exposure scoped, and expires the inventory snapshot before F5.

**Claim ceiling:** this is a public synthetic development fixture, not a sealed cohort. The AI pilot uses authored candidate probes, not live systems or human raters. Its 20/20 vs 1/20 diagnostic is not a benchmark result, inter-rater statistic or SOTA claim. A real tournament must stream one frontier at a time and keep the event tape and oracle outside candidate access.

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

## 10a. Digest and manifest closure

- The task's `evidence_manifest_sha256` is SHA-256 over the exact UTF-8 bytes of the referenced evidence-manifest file, including its line endings; do not normalize before hashing.
- Each structured JSON payload's `content_sha256` is SHA-256 over RFC 8785 JSON Canonicalization Scheme bytes. Text/binary payloads use exact source bytes. The canonicalization profile/version must be pinned by the implementation.
- Inline synthetic fixture payloads are canonicalized from the `metadata.synthetic_record` object. The `payload_ref` JSON Pointer must resolve to that object.
- The task's `evidence_ids` must equal the unique artifact-ID set in the evidence manifest. Duplicate artifact IDs or dangling task references reject the manifest.

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
