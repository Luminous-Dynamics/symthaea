# Engineering Decision Correctness Benchmark v0.1

**Status:** research protocol; proposed targets only  
**Date:** 2026-11-14  
**Owner:** Symthaea engineering-change benchmark  
**Issue:** [ENG-DECISION-CORRECTNESS-001 #12039](https://github.com/Luminous-Dynamics/symthaea/issues/12039)  
**Parent:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Companion:** [ENG-DESIGN-CHANGE-TEMPORAL-001 #12017](https://github.com/Luminous-Dynamics/symthaea/issues/12017), [decision-replay PR #11996](https://github.com/Luminous-Dynamics/symthaea/pull/11996)

This document defines an engineering-specific benchmark extension. It is not an implementation result, product claim, safety case, certification, or vendor comparison.

## 1. Thesis: evaluate the decision process and the decision outcome separately

A system can return the correct answer for the wrong reason. It can retrieve an authentic, current record that is not eligible to support the requested engineering claim. It can use evidence that was legitimately available at the time and still experience a poor downstream outcome because the world was uncertain. It can also express a calibrated probability but choose an action that violates the declared loss policy.

The benchmark therefore distinguishes five evaluation axes:

1. **Process admissibility:** the exact evidence available to the decision process by knowledge frontier K, with applicability at target-world time T, source completeness, rights and trust maturity qualified.
2. **Evidence-use validity:** whether the evidence is licensed to support this particular claim, configuration, intended use and decision.
3. **Justification and causal support:** whether the evidence supports the proposed impact or disposition, rather than merely correlating with or preceding it.
4. **Policy and action correctness:** whether the selected disposition satisfies hard constraints and is rational under a frozen action set and declared loss model.
5. **Realized outcome:** what happened after the decision at a declared observation horizon, using independently qualified outcome evidence.

These are separate estimands. The benchmark reports a vector of results and non-compensatory gates; it does not create one weighted decision-correctness score.

A successful outcome by coincidence cannot repair invalid evidence use. A bad outcome does not, by itself, prove that the decision was unjustified at K. A decision that was justified at K does not establish physical safety.

## 2. Research comparators and scope

### 2.1 Closest conceptual comparator: SciValidityBench

Pan and Chen's SciValidityBench preprint (arXiv:2610.34767, 26 October 2026) shifts scientific-agent evaluation from retrieving a correct fact to checking whether that fact is eligible to influence a decision under the state governing its interpretation. The authors report 64 TaskSpecs across five validity modules and six scientific domains, deterministic generated worlds and decision oracles, paired counterfactual interventions, corruption/repair controls, and a 9.2–18.8 percentage-point improvement from compiled state over a generic event graph on their own evaluation.

This is the closest conceptual comparator, not an engineering benchmark or a directly comparable result. Treat the paper as a preprint. Inspect the public release and license before any reuse. Do not copy its task data or code into Symthaea. Reproduce the relevant comparison methods on an independently authored engineering corpus and cite the source.

- Paper: https://arxiv.org/abs/2610.34767
- Implementation: https://github.com/SciValidityBench/SciValidityBench

### 2.2 Complementary research

- TEMPO (EMNLP 2026) evaluates temporal reasoning-intensive retrieval over 1,730 complex queries, 13 domains and 3,976 decomposed steps. The best reported system achieved 71.4% Temporal Coverage@10. It is a retrieval diagnostic, not an engineering decision score: https://aclanthology.org/2026.emnlp-main.291/
- HEDRA (ICLR 2026) studies heterogeneous event-level causal disentanglement in temporal knowledge graphs. It supports keeping causal support separate from event correlation and temporal ordering: https://proceedings.iclr.cc/paper_files/paper/2026/hash/cafcbf5d00a961a05bd5875c17ef0fe7-Abstract-Conference.html
- RiskEval (EMNLP 2026) evaluates whether an LLM changes abstention behaviour as error penalties change. Its reported failures motivate measuring the chosen action against the declared loss matrix rather than trusting verbal confidence: https://aclanthology.org/2026.emnlp-main.759/
- CALIBER (Findings EMNLP 2026) distinguishes prompt-level policy-success confidence from post-answer correctness confidence. The two estimates require different targets: https://aclanthology.org/2026.findings-emnlp.2420/
- NIST TEVV-Athlon / AI 200-2 is an initial public draft for evaluating AI systems, including agentic systems. It is not a certification or endorsement: https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems
- NIST IR 8536 is a final manufacturing supply-chain traceability meta-framework for interoperable, verifiable traceability information. It is not an engineering decision benchmark: https://csrc.nist.gov/pubs/ir/8536/final

## 3. Existing owner composition

The benchmark must not duplicate canonical source, retrieval, time, trust, authority, confidence or replay infrastructure.

| Existing owner | Responsibility retained | Required composition |
|---|---|---|
| Symthaea #10543 | Engineering change tasks, applicability labels, impact/obligation scoring and cost accounting | Frozen engineering task and independent gold labels |
| Symthaea #11996 / #12017 | V1.1 T/K contract, exact evidence identity, temporal frontier, causal cut, trust maturity and semantic evaluator | Per-evidence admissibility result and reason codes |
| Symthaea #9112 | Decision-governing evidence-use validity | Reusable types only after source/type/consumer audit |
| Symthaea #11453 | Cost-sensitive decision policy, abstention, evidence acquisition, escalation and regret oracle | Frozen loss/action semantics and policy comparison |
| Symthaea #11367 | Probability calibration and selective-risk analysis | Stage- and target-specific confidence metrics |
| Symthaea #10237 / #10318 | Counterfactual action regret and reference-disagreement envelope | Qualified action/reference comparisons where applicable |
| Symthaea #10816 | Retrieval planning and execution, authorized acquisition, materialization and actual query-visible frontier | Evidence actually exposed to the candidate |
| Mycelix #2631 / #5843 / #5923 | Evidence frontier, multi-conductor partial order and dependency/source completeness | Owner-native typed receipts |
| Symtropy and domain systems | Deterministic replay, simulation and measured/physical outcomes | Replay/outcome evidence within qualified context of use |

Before implementation, inspect exact current paths, revisions, types, schema versions, tests, identifier/digest semantics, producers, consumers, unknown states and claim ceilings. An issue description or a historical PR test is not implementation evidence. If an owner output is absent or unqualified, preserve UNKNOWN, UNOBSERVABLE or NOT_COMPARABLE. Do not create a replacement service.

## 4. Task and information-boundary contract

Each task has three isolated data domains.

### 4.1 Candidate-visible domain

Contains only what the candidate process is entitled to observe:
- exact accepted baseline and candidate design/configuration references;
- proposed change and declared intent;
- evidence records and exact versions exposed to the candidate;
- permitted dependency/claim projections;
- tool results, retrieval outputs and run-context manifest;
- declared task objective and permitted action vocabulary.

Candidate-facing input must not include the gold decision, hidden causal mechanism, evaluator-only source inventory, hidden temporal-bin denominator, future correction, outcome label or expected counterfactual relation.

### 4.2 Evaluator-only domain

Contains:
- independently authored affected and unaffected claims;
- required review, verification, revalidation and physical-evidence obligations;
- complete source-role and temporal-bin denominators;
- admissibility and evidence-use rules;
- causal mechanism and qualified reference envelope;
- legal action set, hard constraints and loss matrix;
- expected counterfactual relation;
- outcome observation and horizon labels;
- corruption, repair and hidden mutation controls.

Gold labels must not be generated by the candidate or accepted from its self-reported affected-item list.

### 4.3 Provenance and replay domain

Binds exact task, baseline, candidate, delta, evidence, source inventory, T/K, run context, tool/model/build, policy/loss, action set, evaluator and output digests. Retain every attempted case and its terminal disposition. No successful-only denominator.

The candidate-visible, evaluator-only and replay domains must have separate access paths and explicit leakage tests.

## 5. Counterfactual decision tests

The key extension is to measure whether a decision depends on the right information.

Every paired task declares one expected relation:

- **FLIP:** a decisive governing fact, admissibility condition or qualified causal mechanism changes; the candidate decision must change to the specified alternative.
- **INVARIANT:** only irrelevant information, formatting, ordering, or an unrelated source/claim changes; the candidate decision must remain unchanged.
- **NOT_APPLICABLE:** no registered pair exists.

Use evaluator-authored pairs, not candidate-selected interventions.

### 5.1 Evidence dependence without simplistic deletion rules

A single-source deletion must not automatically require a decision flip. Engineering evidence can be redundant. For each task, the evaluator must freeze the sufficient evidence sets or the specific intervention that removes decisive support. Replacing one sufficient source with an independently admissible equivalent may preserve the decision.

Measure:
- whether the candidate actually used the exact evidence exposed to it, using #10816 execution/run-context receipts rather than narrative self-report;
- whether removing or invalidating decisive evidence changes the decision as prescribed;
- whether irrelevant changes leave the decision invariant;
- whether target repair restores the expected decision;
- whether operation-matched irrelevant repair is correctly recognized as irrelevant;
- whether evidence attribution remains valid under redundant sufficient sets.

Use the SciValidityBench distinction between paired choice changes and invariance as a design reference. Adapt it to engineering claim applicability, source rights, temporal admissibility, lifecycle obligations and risk-sensitive dispositions.

### 5.2 Required counterfactual families

At minimum:
1. valid at T but unavailable at K;
2. available at K but invalid for T or the requested configuration;
3. late correction after K, preserving the historical decision snapshot;
4. decisive source invalidated versus equivalent admissible substitute;
5. unrelated source added, removed or reordered;
6. source role or required time period missing;
7. apparent causal edge removed by a qualified counterfactual or mechanism test;
8. correlation-only edge retained without causal proof;
9. loss matrix changed while facts remain fixed;
10. one legal action removed or added before the evaluator freezes the action set;
11. evidence acquisition option introduced with a known or bounded value of information;
12. a post-decision outcome changes while the original decision-time evidence remains frozen.

## 6. Frozen action and loss contract

Each task declares before candidate evaluation:
- legal disposition/action set (preserve, review, reverify, revalidate, block, request evidence, abstain, escalate or defer);
- non-negotiable constraints and authority requirements;
- asymmetric costs for false-safe, false-block, delay, evidence acquisition, human review, avoidable requalification and irreversible effects;
- action-set completeness evidence;
- qualified model/reference applicability and uncertainty envelope, if used;
- outcome observation horizon and missing/censored-state semantics.

Hard constraints are gates, not terms that can be offset by a favourable utility score.

If calibrated probabilities are not available, do not manufacture expected utility from verbal confidence. Use deterministic symbolic oracles for synthetic worlds, or bounded/minimax regret only where the reference set and assumptions support it.

For a scalar loss L(a, w) over a frozen feasible action set A(w), simple regret is:

    regret(a, w) = L(a, w) - min over a' in A(w) of L(a', w)

Where multiple qualified references r apply, report per-reference regret. A worst-reference minimax regret may be reported only when the set of qualified references and applicability domains are explicit:

    worst_regret(a) = max over applicable r of
                      [L_r(a, w) - min over a' in A(w) L_r(a', w)]

Do not use a scalar regret when objectives are incomparable, hard constraints are unresolved, or the reference set is incomplete. In those cases report the objective-wise partial order and an explicit unresolved/incomplete result.

## 7. Scoring: hard gates first, independent dimensions second

Report these metrics independently. Do not sum or average them into a single leaderboard score.

| Dimension | Metric | Rule |
|---|---|---|
| Safety | Critical false-safe decisions | Zero allowed on frozen hard-gate controls |
| Temporal | Future-knowledge leakage | Zero allowed on hard-gate controls |
| Provenance | Unsupported evidence-admissibility claims | Zero allowed on hard-gate controls |
| Authority | Unauthorized action/effect acceptance | Zero allowed |
| Causality | Unsupported cross-source causal attribution | Zero allowed on causal hard negatives |
| Completeness | Critical missed-impact recall | 100% on adjudicated hard-gate cases |
| Evidence use | Use-validity accuracy | Report by claim type, validity mechanism and criticality |
| Counterfactual | FLIP and INVARIANT relation accuracy | Proposed target ≥95% on blinded held-out pairs; report each separately |
| Applicability | Supported impact precision | Proposed target ≥95% |
| Temporal retrieval | Temporal Coverage@k | Proposed target ≥95% against evaluator-frozen required bins |
| Source coverage | Required source-role coverage | Proposed target ≥99% when a complete inventory is declared |
| Policy | Regret / robust dominance | Report per loss matrix, objective, task family and qualified reference |
| Selective decision | Risk-coverage and utility-coverage curves | Publish complete curves; always-abstain must fail usefulness qualification |
| Outcome | Outcome quality at horizon | Separate from process correctness; preserve censoring and missingness |
| Economics | Reviewer effort, requalification burden and fully burdened cost per accepted decision | Paired distributions and real-pilot measurement; retain ≥10% cost reduction as a hypothesis, not a result |

For every metric, publish numerator, denominator, exclusions, unknown count and applicability count. If the denominator is candidate-selected or source completeness is unverified, a completeness claim is blocked.

### 7.1 Why 41 controls are not a reliability certificate

The existing 41 controls are valuable falsification controls. They are not 41 independent population samples. Even zero failures on all 41 does not prove a low real-world failure rate.

As a rough binomial reference, zero failures in about 299 independent Bernoulli trials yields a one-sided 95% upper bound near 1%. The independence assumption must be justified. Engineering cases share design families, organizations, suppliers, assessors and change lineages; therefore use cluster-aware sampling and report exact finite-sample intervals at the cluster/task-family level. Do not use row counts to imply independence.

## 8. Benchmark arms and matched comparisons

Compare under identical frozen tasks, action sets, loss/constraint declarations, time limits and evidence-access budgets:

1. Human-only engineering review.
2. Native PLM/ALM workflows and vendor AI features, where trials/access permit.
3. BM25 or standard lexical retrieval.
4. Temporal retrieval/RAG.
5. Model-only.
6. Model plus pinned evidence.
7. Model plus compiled decision-governing state.
8. Symthaea composed with qualified Mycelix and existing owner receipts.

Industry products are functional baselines, not automatically benchmark competitors. Teamcenter 2606 documents AI BOM/change workflows; Aras InnovatorEdge AI 2.4 documents multiple MCP profiles with separate endpoints, Access Points and tool sets; IBM DOORS Next documents traceability and impact analysis. These product descriptions do not provide independent accuracy, safety or cost scores. Compare only in an instrumented, matched trial and report unavailable vendor data as unavailable.

Blind domain assessors to candidate identity. Use two independent assessors and a documented adjudication path; retain disagreement, uncertainty and abstention labels. Record reviewer time, rework and clarification burden.

## 9. Ablations

The benchmark should determine which part of the composition creates value:

- raw event history vs retrieval vs generic event graph vs bitemporal state vs typed governing state;
- with and without T/K, source-frontier completeness, evidence-use validity, causal support and action-time authority;
- candidate-selected vs evaluator-frozen source inventory and action set;
- full evidence vs admissible evidence only vs decision-minimal sufficient evidence;
- decisive evidence mutation vs irrelevant mutation;
- exact target repair vs operation-matched irrelevant repair;
- single qualified reference vs qualified reference envelope;
- verbal confidence vs stage-matched calibrated uncertainty vs direct loss-aware action policy;
- decision-time justification vs post-decision outcome evaluation.

Each ablation must preserve all other task inputs, budgets, candidate versions and evaluator versions.

## 10. Statistical and leakage design

- Split by task template, engineering family, change lineage and organization—not random rows alone.
- Maintain a hidden confirmatory set and a public development set.
- Freeze task and evaluator digests before candidate runs.
- Keep future corrections and outcome observations evaluator-only until the candidate decision and evidence manifest are committed.
- Ensure source, time-bin, action-set and cost denominators are evaluator-owned.
- Use paired comparisons and cluster-aware uncertainty estimates.
- Report per-family and worst-family results, not only pooled averages.
- Repeat runs under fixed and varied seeds where stochastic candidate behaviour is part of the claim.
- Publish negative controls that expose source substitution, future leakage, missingness laundering, false causal order, confidence-policy mismatch and action-set pruning.

## 11. Implementation sequence

1. **WP0 — owner/type audit:** verify exact reusable contracts and tests for #10543, #11996/#12017, #9112, #11453, #11367, #10237/#10318 and #10816.
2. **WP1 — gold task pairs:** author decision-changing and decision-invariant pairs with hidden evaluator labels and redundancy-aware sufficient evidence sets.
3. **WP2 — frozen action/loss oracle:** establish action sets, constraints, loss matrices, reference envelopes and censoring semantics before candidate runs.
4. **WP3 — Rust consumer:** implement only the narrow deterministic adapter required to compose owner outputs and emit an exact-byte-bound decision receipt.
5. **WP4 — offline qualification:** test positive, negative, FLIP, INVARIANT, repair and replay controls; use a separately implemented reference checker for selected controls.
6. **WP5 — matched benchmark:** only after exact-head qualification and independent gold adjudication, run baseline and candidate arms.

No CI runner is needed to perform the first two steps. Local validation may provide evidence, but it must not be described as hosted CI or end-to-end qualification.

## 12. Acceptance gates

1. Exact owner/type/consumer matrix reviewed; no duplicate authority or evidence services.
2. Process correctness, policy quality and realized outcome estimands are separately specified.
3. Candidate-visible and evaluator-only data boundaries pass leakage controls.
4. Every positive, negative, FLIP and INVARIANT control matches its registered expected disposition.
5. Zero critical false-safe, future-knowledge leakage, unsupported causal attribution, fabricated completeness and unauthorized-action acceptance on the frozen hard-gate suite.
6. Candidate decisions bind the complete evaluator-frozen legal action set.
7. Deterministic receipts replay byte-identically for identical inputs and evaluator build.
8. Independent oracle/checker agrees on declared controls; assessor disagreement remains explicit.
9. No benchmark output creates engineering approval, requirement acceptance, release, manufacture, operation or external effect authority.
10. Comparisons disclose exact task/evaluator hashes, source revisions, prompts, tools/models, hardware, budgets, exclusions, raw per-family results and statistical units.

## 13. Claim ceiling

A pass is scoped to exact task families, task bytes, accepted baseline/configuration, evidence frontier, T/K, source inventory, action set, loss model, reference applicability, evaluator build and outcome horizon.

This benchmark does not by itself establish physical safety, real-world design correctness, global source completeness, legal admissibility, causality from correlation, global optimality outside the frozen action set, low population risk from a small synthetic suite, vendor superiority or general state-of-the-art performance.

## 14. Prospective decision commitment and decision-minimal evidence

### 14.1 Seal before outcome disclosure

The benchmark must use a prequential protocol. Before the evaluator releases any post-decision outcome, later correction, realized failure, or outcome-dependent label, the candidate must commit a decision packet containing:
- selected disposition/action and any abstention, evidence request or escalation;
- exact candidate-visible evidence manifest, source versions and digests;
- target time T and knowledge frontier K;
- source/temporal denominators and their evaluator commitments;
- complete legal action set and its completeness receipt;
- hard constraints, governing policy version and loss matrix;
- causal assumptions/reference applicability and uncertainty state;
- confidence/calibration profile identity if used;
- evaluator-independent task/run IDs, tool/model/build identities and packet digest.

The evaluator seals this pre-outcome packet. Only after the commitment is immutable may the evaluator disclose later corrections and outcome observations. Candidate-side post-seal mutation, future evidence access before commitment, or an outcome-derived field in the decision packet is a leakage failure.

A later correction or new evidence creates a **new decision episode with a new K** and a reference to the earlier decision. It must never rewrite what the earlier process knew or selected. Historical justification and eventual outcome are scored separately.

### 14.2 Decision-minimal sufficient proof

Evaluate whether the system can establish a sufficient proof packet without indiscriminately retrieving or requalifying everything. Reuse qualified lifecycle proof/optimality work and engineering obligation labels; do not create another proof solver.

Report:
- mandatory-claim proof coverage and unsupported-claim count;
- selected evidence cost/size and the exact minimum only when the feasible proof space is complete and the exact search finishes;
- proof search state (VerifiedOptimal, SearchLimitReached, BudgetInsufficient, NotRun, Rejected);
- unnecessary evidence burden and provenance-family duplication;
- missed-impact and omitted-obligation rates;
- avoidable requalification and reviewer burden.

A heuristic low-cost packet is not an optimal packet. When the candidate set is incomplete, source inventory unknown, cost unqualified, or exact search truncated, report INDETERMINATE / NOT_COMPARABLE rather than a minimality claim.

### 14.3 Value of information and cost of delay

When the legal action set contains RequestEvidence, measure whether the requested observation could discriminate between the leading dispositions and whether its acquisition cost/delay is justified under the task's frozen loss model.

Separate:
- predicted expected value of information (only if its outcome probabilities are empirically or symbolically qualified);
- observed information gain / disposition change after the measurement;
- realized reduction in declared decision loss;
- acquisition, waiting, reviewer and requalification cost.

Synthetic known-answer probabilities qualify arithmetic only. They do not qualify empirical acquisition success, sensor reliability, reviewer quality or real-world VOI. If those distributions are unknown, preserve uncertainty and compare deterministic/bounded cases instead.

### 14.4 Additional hard-gate metrics

| Metric | Proposed rule |
|---|---|
| Pre-outcome commitment completeness | 100% of required fields bound before outcome disclosure |
| Post-outcome information leakage | 0 on hard-gate controls |
| Historical decision preservation | 100% of earlier packets unchanged after later corrections |
| Mandatory proof coverage | 100% on adjudicated critical obligations |
| Proof optimality | Claim only when exact search completes over a frozen complete candidate set |
| Avoidable requalification | Report against adjudicated obligation gold labels; retain ≥30% reduction as a pilot hypothesis |
| Value-of-information quality | Score only under a qualified probability/reference model; otherwise mark unmeasured |

### 14.5 Required adversarial pairs

Add controls where:
1. a late correction is revealed only after the original decision packet is sealed;
2. the candidate attempts to alter its original action/evidence manifest after outcome disclosure;
3. two different actions have the same realized outcome by chance but different decision-time justification;
4. a small sufficient evidence set and a large redundant packet support the same decision;
5. one required premise is missing and a cheap discriminating evidence request is available;
6. evidence acquisition is expensive or delayed enough that a different safe action is optimal;
7. exact proof search reaches its declared limit and must not claim optimality;
8. the next decision epoch legitimately changes after new information arrives while the previous epoch remains byte-identical.

These controls extend the benchmark from retrospective explanation toward auditable, prospective decision accountability.

## 15. Agentic abstention and defeater-first assurance

### 15.1 AgentAbstain is the closest action-timing comparator

Liu et al., **AgentAbstain: Do LLM Agents Know When Not to Act?**, arXiv:2607.10059; NeurIPS 2026 Evaluations & Datasets Oral, reports:
- 263 paired tasks in 42 executable sandbox environments;
- eight abstention scenarios covering pre-execution and runtime-discovered triggers;
- a should-act task paired with a should-abstain twin under one controlled perturbation;
- 17 frontier models across four agent harnesses;
- best reported paired accuracy 59.5% (Gemini 3.1 Pro), with post-hoc abstention identified as a particularly hazardous failure.

This is a direct methodological comparator for our action/hold tests, but not a directly comparable engineering benchmark. The tasks, tools, stakes and action semantics differ. We should use the paired design and runtime-trigger categories, not import its leaderboard as our score.

- Paper: https://arxiv.org/abs/2607.10059
- Project and benchmark: https://agentabstain.github.io/
- Dataset: https://huggingface.co/datasets/antiquality/agentabstain

Engineering-specific pairs should cover:
1. should preserve/release to the next governed *human* workflow stage versus should block/request evidence;
2. a critical missing design parameter;
3. ambiguity about the affected configuration or requirement;
4. mutually inconsistent engineering constraints;
5. insufficient authority or an absent qualified verifier;
6. runtime retrieval, proof, simulator or measurement failure;
7. conflicting but individually authentic evidence;
8. emergent risk discovered after initial plan execution.

The benchmark must distinguish safe, reversible preparation (read, inspect, draft, simulate, stage) from irreversible or externally consequential effects (approve, publish, release, purchase, program hardware, actuate, deploy). The benchmark itself has no such authority.

### 15.2 Add action-timing and useful-abstention metrics

| Metric | Proposed target / rule |
|---|---|
| Paired ACT/HOLD accuracy | ≥85% on blinded engineering pairs; not comparable to AgentAbstain without matched tasks/harnesses |
| Critical HOLD correctness | 100% on frozen hard-gate cases |
| Critical irreversible action before trigger resolution | 0 |
| Post-hoc abstention after a prohibited/irreversible action | 0 |
| Eligible ACT-side success | ≥95% on independently adjudicated should-act controls |
| Unnecessary abstention | ≤5% on the eligible should-act set; publish per family |
| Runtime-trigger recognition | Report recall and latency from first observable trigger, by trigger family |
| Useful coverage | Report paired accuracy and risk–coverage together; always-abstain fails |

A paired score is essential: a system that always acts or always refuses must not pass. Keep ACT-side and HOLD-side rates visible so the paired average cannot hide a failure mode.

### 15.3 Defeater-first rather than support-only assurance

An assurance case is not just a set of supporting citations. It is a structured claim/argument/evidence relationship with explicit context and assumptions. ISO/IEC/IEEE 15026-2:2022 standardizes assurance-case structure and terminology; it explicitly does not guarantee the quality of the argument's contents. ISO/PAS 8800:2024 addresses safety-related AI in series-production road vehicles, not general engineering-change software, but is a useful domain-specific reference for linking AI safety properties to assurance claims.

We should borrow the structure, not claim standards conformity. Add a **defeater ledger** for each consequential claim:
- missing premise or hidden assumption;
- alternative causal mechanism;
- common-cause or correlated-source dependence;
- configuration/scope mismatch;
- stale, late or future information;
- source authenticity or rights gap;
- measurement uncertainty or censored result;
- incomplete action set or loss model;
- operational/environmental shift.

The evaluator should ask not only “what supports this claim?” but “what plausible evidence would defeat it, and has the system attempted to find it?” A critical unresolved defeater blocks a supported-at-K result. An accepted residual defeater must have an explicit rationale and remain visible; it is not silently converted to PASS.

Proposed goals:
- 100% recall of evaluator-authored critical defeaters on the frozen hard-negative set;
- ≥90% precision for major/critical defeaters on blinded adjudicated cases;
- 0 critical unresolved defeaters in assessments marked supported-at-K;
- 100% of accepted residual defeaters retain scope, rationale and evidence references.

The literature on structured assurance cases warns that support-only arguments can reinforce confirmation bias. Eliminative argumentation and explicit defeaters give us a stronger falsification-oriented benchmark than measuring citation quantity alone.

References:
- ISO/IEC/IEEE 15026-2:2022: https://www.iso.org/standard/80625.html
- ISO/PAS 8800:2024: https://www.iso.org/standard/83303.html
- Argument-based assurance cases: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/explaining-decisions-made-with-artificial-intelligence/annexe-5-argument-based-assurance-cases/

### 15.4 Traceability interoperability update

NIST IR 8536 was finalized on 9 September 2026. It defines an industry-neutral manufacturing traceability meta-framework for linking and querying traceability data across diverse ecosystems, with cryptographic linkage and selective disclosure. It is a traceability framework, not a decision-correctness benchmark or proof of source truth.

This reinforces our architecture choice: keep PLM, supplier, MES, test, quality and lifecycle systems as systems of record; compose their versioned provenance receipts at the decision boundary rather than forcing a centralized duplicate data lake.

- Final report: https://csrc.nist.gov/pubs/ir/8536/final
- NIST project: https://www.nccoe.nist.gov/projects/supply-chain-traceability-principles-manufacturing-meta-framework

## 16. Machine-readable assessment envelope v0.1

Three synthetic/public-development artifacts establish the first consumer contract:

- `schemas/engineering-decision-assessment-v0.1.schema.json`
- `examples/engineering-change/decision-assessment-v0.1.synthetic.json`
- `benchmarks/engineering-change-impact/decision-assessment-mutations-v0.1.json`

The envelope is evaluator-side and non-authoritative. It references owner-native receipt IDs and digests; it does not copy their internal facts or replace their verifiers. The axes remain process admissibility, evidence-use validity, engineering applicability, causal justification and action policy. Later outcome is a separate object and may be not_observed.

The contract requires a pre-outcome candidate commitment over the decision packet, evidence manifest, complete action set and loss model. It carries a defeater-search result and explicit counterfactual relation. The output's effect authority is statically `none`.

The 21 registered controls are preregistration, not results. They include schema/receipt omissions, unknown denominators, false causal order, action-set pruning, unresolved defeaters, missing run-context proof, post-outcome rewriting/leakage, search-limit/false optimality, ACT/HOLD failures, redundant-source invariance and outcome-state laundering.

**Structural schema validation and semantic evaluator validation have not yet been run.** JSON parsing and read-back establish only that the artifacts were written and syntactically readable; they do not qualify Draft 2020-12 conformance, cross-field consistency or decision semantics. The next offline step is to run the exact Draft 2020-12 engine, then add an independent semantic Rust consumer after owner compatibility is reconciled.
