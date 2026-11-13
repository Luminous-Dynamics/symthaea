# Engineering-Change Benchmark Frontier: SOTA, Industry and Decision-Time Assurance — 2026-11-13

**Status:** research synthesis and comparison protocol; no benchmark candidate runs or vendor trials were performed.  
**Parent owner:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Core experiment:** [Engineering-change preregistration V0.1](ENG_CHANGE_EXPERIMENT_PREREGISTRATION_V0_1_2026-11-13.md)  
**Decision-time overlay:** [TDA overlay V0.1](ENGINEERING_CHANGE_DECISION_TIME_ADMISSIBILITY_OVERLAY_V0_1_2026-11-13.md)  
**Gold-contract successor:** [PR #11964](https://github.com/Luminous-Dynamics/symthaea/pull/11964)  
**Lifecycle reasoning owner:** [KNOW-LIFE-001 #10600](https://github.com/Luminous-Dynamics/symthaea/issues/10600)

## 1. The wrong competition is "which model writes the best impact report?"

The existing industry comparison establishes that AI-assisted BOM/change workflows, linked-object impact analysis, requirements traceability and human-approved writes already exist in commercial products. A generated affected-item list, five specialist agents, or a review button is not a defensible category claim.

A more meaningful comparison separates five questions that most simple leaderboards collapse:

1. **Lifecycle truth:** is the cited evidence semantically relevant, valid at target time T and not superseded?
2. **Decision-time admissibility:** was that exact evidence version available to the declared principal by frontier K?
3. **Actual system availability:** had the relevant ingestion, index/materialization and run-context path made the exact version available to this candidate run?
4. **Engineering applicability:** does the evidence still support this exact claim for the new design/configuration, or is review, recalculation, retest, physical evidence or approval required?
5. **Authority and effect:** was the consequential action permitted for this actor, resource, profile and current policy, and what effect actually occurred?

These are separate evaluation axes. A system may answer a temporal question correctly while lacking proof that it retrieved the source. A system may identify the correct affected item without determining that its prior qualification evidence is stale. A valid recommendation does not authorize a write. A recorded write does not prove the engineering decision was correct.

## 2. What current research contributes—and what it does not establish

| Comparator | Published contribution | Useful diagnostic | Not an apples-to-apples engineering-change result |
|---|---|---|---|
| **Vera-Life-Eval / Vera-Life** (GroundLM 2026) | 8,500 questions, 41,203 versioned evidence pieces and six lifecycle settings (current, past, stale-only, update, conflict, multi-hop). The paper reports LifeEM .374 for its best prompt-only baseline and .623 with Vera-Life on a shared Qwen3-8B generator. | Claim-level semantic support, target-time validity, supersession, conflict and warn/retrieve/abstain behavior. | The score belongs to the paper's data, generator and protocol. It does not by itself measure PLM claim–evidence applicability, engineering requalification obligations, or proof of actual agent run-context inclusion. |
| **TRACE** (Findings of EMNLP 2026) | More than 16,000 model-specific cases across Modify and Insert, a priori exact-match probing and a seven-dimensional temporal/chronological metric matrix. | Timeline preservation, adjacent-fact interference, forward/backward temporal inference and leakage-resistant task construction. | It evaluates model knowledge editing, not source-vintage custody, engineering evidence applicability, vendor workflows or authorized physical effects. |
| **TKE-Bench / ChronoExpert** (Findings of EMNLP 2026) | Tests temporal accessibility across chronologically structured knowledge edits; the paper reports temporal interference in its evaluated editing baselines and proposes time-slice experts. | Whether a model can preserve earlier and future facts after updates rather than merely return the latest value. | A model-side editing architecture is not a replacement for versioned external evidence history, rights, source receipts or engineering qualification. |
| **Siemens Teamcenter 2606 / AI BOM agent** | BOM-aware impact analysis and change execution within governed workflows, with human oversight. | Strong PLM-native baseline for product structure and workflow execution. | Vendor feature descriptions do not provide our frozen claim–evidence confusion matrix or matched benchmark measurements. |
| **Aras InnovatorEdge AI** | Impact Analysis AI Agent v1.0.1; wider platform 2.4 documents configurable MCP profiles and tool invocation. | Direct AI impact-analysis comparison plus a separate profile/tool/authority security test. | Platform 2.4 is not the Impact Analysis agent's version. Neither product documentation nor a mock test is a vendor runtime result. |

Sources:
- Vera-Life-Eval: https://aclanthology.org/2026.groundlm-1.18/
- TRACE: https://aclanthology.org/2026.findings-emnlp.1045/
- TKE-Bench / ChronoExpert: https://aclanthology.org/2026.findings-emnlp.1823/
- Siemens Teamcenter AI BOM agent: https://blogs.sw.siemens.com/teamcenter/ai-bom-agent-plm-engineering-change/
- Aras Edge AI Release Notes: https://docs.aras.com/aras-innovatoredge-ai-book/edge-ai-release-notes
- Aras Administrator Guide: https://docs.aras.com/aras-innovatoredge-ai-book/innovatoredge-ai-administrator-guide

**Comparison rule:** use these papers to select failure modes and diagnostic baselines; never transfer their headline scores into the engineering-change scoreboard. Use vendor documentation to define eligible product/configuration arms; never turn a feature description into a measured result.

## 3. The Luminous research wedge

The proposed differentiator is not an AI that knows more PLM terminology. It is an assessment that can be independently reconstructed from:

- exact source and target design/configuration generations;
- the exact claim–evidence pair and dependency path;
- the evidence version and its source/byte identity;
- validity at T and admissibility at K;
- source publication/issuer availability and canonical receipt;
- acquisition/ingestion, derived materialization and actual run-context evidence;
- rights at decision time and rights at replay/query time;
- the existing engineering disposition and required follow-up obligations;
- current actor, principal, resource, tool/profile and effect-entry authority;
- an independent observation of the mock or real permitted effect.

This is a composition over existing owners (#10543/#10581/#10600, Mycelix #1112/#3022/#3046/#4778/#5395/#5451/#5860, XSTACK #10771, #10064 and #10369). It does not create a competing impact graph, temporal ledger, standard vocabulary, authority engine or benchmark receipt.

## 4. Matched temporal ablations

Keep the preregistered engineering arms B0–B5 and conditional vendor arms V1–V4. Add the following as **diagnostic temporal ablations within the existing experiment**, not as a new task universe or a second leaderboard:

| Diagnostic arm | Method | Failure it exposes |
|---|---|---|
| D0 — latest-state retrieval | Current-value RAG / latest source view without a frozen decision frontier | Historical answer drift and hindsight |
| D1 — timestamp-only retrieval | Filter on one selected timestamp | Confusion between event, publication, receipt, ingestion and actual visibility |
| D2 — lifecycle-aware retrieval | Claim/source/time reasoning comparable in intent to Vera-Life | Staleness, correction, conflict and temporal validity |
| D3 — repository as-of only | Source/record frontier without downstream materialization or context receipts | Confusion between canonical receipt and model-visible evidence |
| D4 — full decision-time composition | T/K, exact source versions, rights, ingestion/materialization/context and claim–evidence applicability | Whether each added evidence boundary changes a decision correctly |

All arms receive the same frozen source versions, task set, rights profile, decision frontier, target T and resource budget. When a candidate cannot expose run-context telemetry or a vendor hides a required observation, mark that submetric **unobservable/incomparable**. Do not infer it passed or failed.

## 5. Benchmark target hierarchy

Do not optimize one composite score. Use non-compensatory gates, then quality, then economics.

### Tier 1 — non-negotiable integrity / safety gates

- zero critical engineering-impact misses in the frozen hard-gate suite;
- zero stale-evidence false-current decisions;
- zero future-knowledge leakage at K;
- zero retroactive rewrite of an earlier as-of state;
- zero invalid preservation witnesses;
- zero unauthorized mock effects or protected-field egress;
- zero silent semantic loss or unsupported claim promotion;
- zero evaluator, denominator or sealed-gold mutation acceptance;
- zero promotion of benchmark output into ETK authority, professional approval, manufacturing release or actuation.

A single failure blocks an efficiency or superiority claim.

### Tier 2 — decision quality

Provisional pilot targets, to be reviewed after blinded adjudication:

| Metric | Proposed target |
|---|---:|
| Supported impact precision | ≥95% |
| Correct preservation classification | ≥90% |
| Critical impact recall in frozen hard-gate suite | 100% |
| Stale-evidence false-current | 0 |
| Unknown/unsupported dependency forcing | 0 |
| Historical evidence identity preservation | 100% |
| Supported-profile semantic round-trip preservation | 100% |

These are proposed targets, not measured values or industry certification thresholds.

### Tier 3 — operational and economic outcomes

- ≥30% fewer unnecessary requalification obligations than full requalification, without relaxing any Tier 1 gate;
- ≥10% lower fully burdened cost per independently accepted assessment, if human, integration, provider, verification and rework costs are actually measured;
- report human minutes, time-to-accepted decision, p50/p95 latency where sample sizes support it, false refusals, escalation, rework, integration and material unknown costs separately.

The pilot may discover that some proposed efficiency targets are not achievable. Do not alter them after observing the holdout.

## 6. Statistical discipline: a zero-miss pilot is not a low-risk guarantee

For zero misses in n independent Bernoulli critical cases, the exact one-sided 95% upper bound is:

[
p_{upper}=1-0.05^{1/n}
]

The existing planning values remain:

| Independent critical cases | Zero-miss 95% upper bound |
|---:|---:|
| 24 | 11.735% |
| 100 | 2.951% |
| 299 | 0.997% |
| 600 | 0.498% |

299 is the minimum under the independent-binomial assumptions for an upper bound below 1%. It is **not** a claim that 299 near-duplicate mutations are sufficient. Design-family, base-configuration and change-lineage clusters must be reported; the confirmatory design needs cluster-aware power and uncertainty. The existing 24 public blueprints remain a feasibility / rubric / variance / cost pilot, not confirmatory evidence.

## 7. What would justify a "better than SOTA" claim?

A defensible claim needs a preregistered, same-task contrast where:

1. the exact same source/candidate generations and evidence manifest are provided;
2. every arm has the same task eligibility, budget and frozen decision frontier;
3. the candidate is scored against independently adjudicated claim–evidence pairs;
4. vendor products are pinned by edition, release, configuration, integrations and licensed environment;
5. hard-gate misses remain non-compensatory;
6. source version, rights, actual availability and run-context evidence are reconstructed independently;
7. a second verifier reproduces the assessment from retained evidence;
8. fully burdened cost and human time are measured, not estimated away;
9. confidence intervals and family-cluster sensitivity support the declared non-inferiority/superiority claim.

Until those conditions hold, the accurate statement is that Luminous has a **stronger proposed assurance contract and benchmark hypothesis**, not that it has beaten Siemens, Aras, PTC, IBM or temporal-reasoning SOTA.

## 8. Paradigm shift: abstention should produce a decision-relevant evidence plan

A system that returns `UnknownImpact` instead of guessing is safer than one that fabricates a preservation witness. But the next capability is to make uncertainty **actionable**: identify the smallest authorized observation that could change the engineering decision, estimate its decision value and cost, and stop when further evidence cannot improve the permitted choice.

For an unresolved claim–evidence pair, candidate acquisition actions may include:
- retrieve the exact source vintage or artifact bytes and capture its receipt/digest;
- request the missing supplier certificate or inspection evidence;
- refresh and attest the permitted retrieval/index materialization state;
- request an independent domain assessor;
- run a bounded numerical recalculation or physical measurement through the appropriate existing owner;
- ask a discriminating clarification about configuration, operating envelope, lot or requirement scope.

This is not a permission to fetch restricted material or to initiate a physical test. Every acquisition action must be authorized under the existing rights, identity and effect owners; a missing permission yields abstain/escalate, not a workaround.

Compose the research with existing value-of-information owners rather than adding a new planner:
- Symthaea [SE-OPT-BENCH #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803) — decision-impact experiment selection;
- Symthaea [EV-DIAGNOSTIC-VOI #11277](https://github.com/Luminous-Dynamics/symthaea/issues/11277) — battery diagnostic tests by expected decision value;
- Symthaea [REA-RUST-014 #10109](https://github.com/Luminous-Dynamics/symthaea/issues/10109) — robust investigation beyond information gain;
- Symthaea [CEB-CAUSE-001 #10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366) — hypothesis-driven causal debugging and experiment selection.

### Prospective acquisition-policy comparison

| Policy | What it tests |
|---|---|
| A0 — always abstain / escalate | Safe but potentially costly and slow |
| A1 — cheapest-first | Whether naive cost minimization acquires low-value evidence |
| A2 — entropy / information gain | Whether uncertainty reduction alone selects the right observation |
| A3 — decision-aware EVSI | Whether expected reduction in decision loss justifies acquisition cost |
| A4 — robust / misspecified-world EVSI | Whether the selected observation remains useful under uncertain model structure and costs |
| A5 — qualified human workflow | Expertise, disagreement, latency and review-cost reference |

Measure expected decision-regret reduction, realized decision changes, acquisition spend/time, human minutes, rights/authorization failures, unresolved fraction, unnecessary acquisition, and the value of preserving reversible options. Report calibrated prediction versus realized value; do not treat estimated EVSI as observed benefit.

**Non-compensatory gates:** zero unauthorized acquisition, zero unapproved physical or production action, zero use of future evidence, and zero conversion of an observation plan into approval or release. A planner can recommend an action; the owner of that action still authorizes and verifies it.

This is a high-value follow-on, not a result in the existing 24-case pilot. First qualify the V1.1 evidence record and independent gold process; then add a preregistered decision-aware acquisition comparison only where the frozen cases contain a genuine unresolved decision and an admissible observation path.

## 9. Immediate execution sequence

1. Complete the current-main V1.1 gold-schema review and add a qualifying Draft 2020-12 validator/evaluator once the owner accepts the contract.
2. Obtain two qualified, independent engineering assessors; the synthetic schema fixture is not a human gold label.
3. Map the nine TDA overlays onto existing claim–evidence pairs; mark absent source/receipt/context evidence unknown.
4. Capture GS1 raw artifact bytes and metadata under Mycelix #5395; run official JSON Schema, JSON-LD expansion and SHACL as separate receipts.
5. Provision an Aras environment and pin both Impact Analysis agent v1.0.1 and platform 2.4; include MCP profile tests only where the actual configuration is available.
6. Run the 24-case pilot only after independent gold adjudication; size the confirmatory holdout from observed paired discordance and clustering.
7. Qualify untrusted-code isolation (#10064) and independent runner trust (#10369) before stronger execution or signed-regrade claims.

Nothing in this note represents a candidate result, vendor trial, standards conformance, gold adjudication or production qualification.
