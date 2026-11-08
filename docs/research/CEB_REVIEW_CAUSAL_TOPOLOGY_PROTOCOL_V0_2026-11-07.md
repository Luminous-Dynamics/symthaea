# CEB Review Causal-Topology Protocol V0 — 2026-11-07

**Status:** Public, source-only research design. It creates no admitted tasks, sealed holdout, model score, causal finding, new metric ID, task schema, receipt or execution authority.  
**Owner:** Existing CEB-REVIEW / CEB-CAUSAL research; #10367, #10419, #10723 and draft #10987.  
**Composes with:** the P/I/F/N/M/IRR causal-diagnosis preregistration, counterfactual-control qualification V0, gold-completeness/open-set protocol, evidence-action crosswalk and dual-annotator calibration pack.  
**Primary measures:** unchanged — \`defect_recall_at_5\`, \`actionable_precision\`, \`reviewer_burden\`.  
**Execution/regrade boundaries:** #10064 and #10369. All examples in this pack are inert, synthetic and source-only.

## 1. Research question

Can an engineering reviewer reconstruct a **minimal, evidence-bounded causal topology** for a defect, identify all materially distinct in-scope paths to the failure, distinguish conjunctive causes from alternative sufficient paths, and correctly predict how the topology changes under P/I/F/N/M/IRR controls?

The existing five-link diagnosis contract is a useful minimum:
\`trigger → path/state → operation → invariant → consequence\`.

It can nevertheless be inadequate when a defect is not a single chain. A vulnerable operation may be reachable through multiple callers; a defect may require two interacting conditions; a temporal race may depend on a partial order; and a repair may remove one route while leaving another reachable.

The proposed research extension is therefore not “draw a bigger graph.” It is: **make the diagnosis identify the smallest causally sufficient structure that explains the outcome and survives attempts to falsify it.**

## 2. Paradigm shift: from linear explanation to falsifiable topology

| Existing review habit | Causal-topology requirement |
|---|---|
| Point to a file, function or line | Identify the source-grounded state transitions and operations that make the outcome reachable |
| Narrate one plausible chain | Enumerate distinct in-scope paths and state which are alternatives versus jointly necessary |
| Treat each condition as independent | Express AND-gated interactions and OR-gated sufficient routes |
| Flatten concurrent behavior into a sequence | Preserve a partial order and explicitly state which events may overlap |
| Call a fix complete when the reported line changes | Search for residual paths to the same protected outcome |
| Count changed lines or matched nodes | Assess semantic path and intervention equivalence |
| Treat a large graph as a stronger explanation | Prefer the smallest graph that is sufficient for the scoped causal claim |

The unit of explanation is a *mechanism topology*, not a source-code dependency diagram. A call graph, import graph, data-flow graph or architecture map can provide candidate edges, but none of those edges is automatically a causal edge.

## 3. Typed representation contract

The following is a research annotation vocabulary, not a new Rust API, task schema or receipt. Reuse existing source anchors and evidence-authority classes.

### 3.1 Node kinds

| Node kind | Meaning | Example |
|---|---|---|
| \`condition\` | Input, configuration, feature flag or precondition | \`legacy_mode = true\` |
| \`state\` | Relevant system or authority state | \`permission revoked\` |
| \`event\` | A temporally located operation | \`ledger commit\`, \`queue redelivery\` |
| \`operation\` | Computation, allocation, authorization or mutation | \`reserve(declared_len)\` |
| \`gate\` | Explicit AND / OR combination of incoming requirements | \`AND(revoked, stale_snapshot_used)\` |
| \`invariant\` | Frozen safety, security or correctness requirement | \`current tenant authority required at write\` |
| \`outcome\` | Failure or protected consequence | \`cross-tenant write\` |
| \`assumption\` | Environment or applicability boundary | \`worker may execute after revocation\` |
| \`intervention\` | Proposed or qualified change to a causal parent or path | \`re-authorize at effect time\` |

A node must describe a semantically meaningful fact, not merely repeat a file or identifier. A source location belongs in its evidence anchors.

### 3.2 Edge kinds

Use a typed, directed multigraph with explicit gates. Every material edge needs a source/evidence anchor and the scope in which it is asserted.

- \`control_flow\`: one operation can lead to another under declared conditions.
- \`data_flow\`: a value is passed, transformed, cached or consumed.
- \`authority_flow\`: identity, permission, capability or trust is transferred or reused.
- \`happens_before\`: event ordering justified by the source contract or an admissible observation.
- \`enables\`: one fact makes a transition or outcome possible.
- \`inhibits\`: a guard or control blocks a path under its declared conditions.
- \`violates\`: an operation or state contradicts a named invariant.
- \`produces\`: a transition produces a state or outcome.
- \`requires\`: a gate requires the listed condition(s).

Do not label an edge \`causes\` merely because two nodes are adjacent in a call graph or share a file. The causal claim must be justified through the mechanism and its counterfactual predictions.

### 3.3 AND, OR and alternative paths

- An **AND gate** is active only when every required incoming condition holds in the declared context. It represents an interaction or a jointly necessary precondition set.
- An **OR gate** is active when any one of its incoming alternatives holds. It represents alternative sufficient routes to the same outcome.
- A topology can contain both. For example, each of two independent ingress routes may have its own AND gate before converging on a vulnerable operation.
- Do not convert an AND into an OR because one condition is easier to inspect. Do not convert alternative sufficient routes into one conjunctive chain.
- If the source package does not enumerate the full caller/configuration space, graph completeness is **unknown outside the declared scope**. Never claim that no other path exists without a justified boundary.

### 3.4 Temporal behavior and cycles

Software execution may be concurrent, recurrent or event-driven. Do not force every system into a single total order or assume a static directed acyclic graph is adequate.

- Represent time-sensitive transitions as event/state instances with explicit \`happens_before\`, overlap or ordering constraints.
- Use bounded event/time expansion for retry loops, repeated messages and periodic jobs.
- Preserve concurrency when the available evidence does not order two events.
- A source-level temporal prediction is not a runtime observation. Runtime order requires the existing qualified evidence path and exact-run binding.
- When a feedback loop is bounded/unrolled, record the horizon and what behavior lies outside it.

## 4. Causal topology claim contract

For each target finding, the reviewer supplies:

1. **Outcome and invariant:** exact protected property, scope and conditions.
2. **Causal nodes:** source-grounded conditions, states, events and operations.
3. **Typed edges:** control/data/authority/temporal and causal relations, with anchors.
4. **Gate semantics:** explicit AND/OR membership and the conditions that activate each gate.
5. **Sufficient paths:** each materially distinct path in scope that can produce the target outcome.
6. **Interaction set:** conditions that must jointly hold for an AND-gated mechanism.
7. **Alternative explanations:** nearest neighboring mechanism(s), common causes and unresolved confounders.
8. **Interventions:** smallest sufficient intervention set(s), plus an irrelevant intervention that should preserve the target.
9. **Residual reachability:** which target paths remain after the proposed repair, including paths through alternate entry points or fallback behavior.
10. **Applicability boundary:** repository/source revision, configuration, feature flags, target/platform, actor/tenant, timing and environment assumptions.

A reviewer may provide more than one minimal sufficient path or more than one minimal intervention. Do not collapse a multi-path defect into one root-cause line.

## 5. Minimal sufficient paths and intervention sets

Let \(\mathcal{P}_T\) denote the set of minimal, in-scope sufficient path/condition sets that can reach target outcome \(T\) under the declared context. A path may contain AND gates; distinct alternatives are OR branches.

An intervention set \(J\) is sufficient for the target claim only when it blocks or neutralizes every qualified target path in \(\mathcal{P}_T\), while preserving the declared unrelated behavior. A repair that disables one path but leaves another target path reachable is partial, not complete.

Use the following terms carefully:

- **Necessary condition:** removing it blocks the specified mechanism under the declared assumptions.
- **Sufficient set:** the full set of conditions is enough to produce the outcome within the declared model.
- **Minimal sufficient set:** no proper subset of that set remains sufficient.
- **Minimal intervention set:** no proper subset of the proposed intervention set blocks all relevant target paths while satisfying the repair contract.
- **Residual path:** a materially distinct in-scope route that remains after an intervention.
- **Open-world remainder:** possible routes outside the evidence boundary; these remain unknown, not absent.

The graph may support bounded structural claims. It does not by itself establish that an intervention was executed, that a test passed, or that a production failure occurred.

## 6. Counterfactual integration

Use the existing six source states. Do not create a second counterfactual vocabulary.

| State | Topology expectation |
|---|---|
| P — Parent | Target path(s) not yet introduced |
| I — Introducing | The actual target topology becomes reachable |
| F — Fixed | Qualified repair blocks the target path set; inspect all in-scope residual paths |
| N — Neighbor | Similar location/symptom but a distinct topology or mechanism |
| M — Minimal intervention | Change only the necessary causal parent/path; predict the target topology no longer reaches the outcome |
| IRR — Irrelevant intervention | Change nearby but non-causal context; the target topology remains reachable |

All historical and constructed controls remain subject to the six Q1–Q6 counterfactual-control qualification dimensions. A historical fix is not automatically a minimal intervention. A nearby line edit is not automatically irrelevant. If a repair cannot be disentangled or residual paths cannot be inspected, mark the causal claim limited or unresolved.

## 7. Graph equivalence, adjudication and anti-gaming

An exact node/edge match is too brittle: two reviewers may use different but semantically equivalent intermediate nodes, or compress a sequence differently. Conversely, a large supergraph can contain the correct path while adding unsupported edges.

Adjudicate topology in this order:

1. **Outcome identity:** same frozen invariant and target consequence.
2. **Path identity:** same minimal sufficient path or an independently justified equivalent.
3. **Gate correctness:** AND/OR logic and temporal ordering are correct.
4. **Intervention prediction:** qualified P/I/F/N/M/IRR outcomes are predicted correctly.
5. **Residual-path accounting:** no known in-scope target path is incorrectly declared eliminated.
6. **Evidence lineage:** material nodes and edges are bound to the exact source and authority class.
7. **Minimality:** unnecessary or unsupported nodes/edges are not used to disguise uncertainty.

The answer key defines acceptable semantic equivalence classes, not a single canonical drawing. Human adjudicators should accept a different graph when it preserves the correct causal semantics and intervention predictions. A location overlap or graph edit-distance score alone is insufficient.

### Diagnostic annotations only

The following may be reported as descriptive diagnostics under the existing CEB adjudication process; they are **not new primary metrics or metric IDs**:

- minimal-path coverage and unsupported-path count;
- AND/OR gate correctness;
- interaction-condition identification;
- temporal-order correctness;
- residual-path detection after repair;
- minimal-intervention prediction consistency;
- causal-topology uncertainty / open-world boundary;
- graph over-generation and source-anchor coverage.

Do not create a topology composite score. Continue to evaluate the review with the three existing primary CEB-REVIEW measures.

## 8. Relationship to external research and benchmarks

The topology extension is motivated by distinct research strands, none of which should be misrepresented as the same task:

- **SecPRBench (2026):** real security PR review; reports that many misses locate the root-cause file but explain a different mechanism, and lower detection as root-cause scope spans more files. This directly motivates alternative-path and multi-file mechanism diagnosis. [Paper](https://arxiv.org/abs/2610.33509).
- **AACR-Bench (2026):** 200 PRs from 50 active projects across 10 languages with full repository context and expert-verified annotation. It motivates cross-file and multi-language context tests, but does not itself establish counterfactual graph diagnosis. [Paper](https://arxiv.org/abs/2601.19494) · [artifacts](https://github.com/alibaba/aacr-bench).
- **SDRCA (Journal of Systems and Software, 2026):** a runtime root-cause framework using causal graphs, in-distribution interventions and Shapley-style multi-cause attribution, evaluated on PetShop, RE1 and RE3 with Recall@1/Recall@3. It is a close causal-method comparator, but its monitored runtime anomaly-localization setting is not pre-merge, source-only PR review. Shapley allocation is a contribution decomposition under a defined game; it is not by itself proof that a proposed software mechanism is causally identified. [Paper](https://doi.org/10.1016/j.jss.2026.112931).
- **UniVal (ICSE 2021 Distinguished Paper):** combines predicate outcomes and variable values in causal statistical fault localization, evaluated on 835 Defects4J versions in the published artifact description. It motivates confounding-aware program fault localization, but it uses execution profiles and is not an agentic review benchmark. [Paper](https://arxiv.org/abs/2102.06292) · [ICSE record](https://2021.icse-conferences.org/details/icse-2021-papers/72/Improving-Fault-Localization-by-Integrating-Valueand-Predicate-Based-Causal-Inference-Techniques).
- **ReviewBench (live high-severity grounded leaderboard):** useful for matched real-PR review comparisons, not graph-equivalence or causal-path completeness. Its ranking is a dated snapshot and must be re-captured before any public comparative claim. [Leaderboard](https://review-bench.ai/).

CEB should compare with these systems by matching task population, evidence, scope, budget, evaluator lineage and outcome definition. Do not compare runtime RCA Recall@1 with CEB Recall@5 or treat a reported 285% annotation coverage increase as a model-performance improvement.

## 9. Source-only experiment proposal

Use the six-case public challenge in [CEB Review Causal-Topology Development Challenge V0](CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md). The separate public key is [here](CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_KEY_V0_2026-11-07.md).

A development-only contrast can compare:
- L0: the existing five-link linear causal diagnosis;
- L1: five-link diagnosis plus typed topology, AND/OR gates, temporal ordering and residual-path predictions.

Keep the same source evidence, model/harness, context, output cap and total inference budget. Randomize and isolate each source-state variant. Do not let a model see sibling variants from one anchor in the same context. Have two independent human reviewers apply the existing dual-annotator pack; a third expert resolves disputes without overwriting first-pass labels.

This is an authoring and rubric-development study, not an additional confirmatory Phase A arm or a new metric lane. Before any confirmatory evaluation, preregister the topology contrasts, sample size, repositories, fault families, independent clusters, analysis, missingness and stopping rule.

## 10. Development acceptance and stop rules

The public challenge is ready for human annotation only when:

- all six cases provide a stable requirement, evidence boundary and source-state matrix;
- each P/I/F/N/M/IRR state has a separately reviewable semantic delta;
- every expected AND/OR gate and temporal edge is documented in the separate key;
- the key lists at least one nearest alternative mechanism or explains why none is in scope;
- every repair/intervention has a residual-path check;
- the dual-reviewer form can capture graph structure without forcing a single canonical node decomposition;
- no candidate-controlled execution or hidden evaluator access is required.

Stop or mark unresolved when source context is incomplete, a path cannot be bounded, the counterfactual intervention is not qualified, temporal order is unknown, or multiple topologies fit the evidence. Do not infer absence from a missing source file or a green-but-unrelated check.

## 11. No-runner work sequence

1. Issue the six public topology anchors to independent reviewers using the dual-annotator protocol.
2. Collect first-pass causal graphs and six-state predictions without exposing the key.
3. Produce path/gate/temporal disagreement tables and adjudicate each difference.
4. Revise the graph vocabulary only where repeated disagreements show a real ambiguity; version all changes.
5. Test L0 versus L1 as a matched development ablation.
6. Extend the public scenario authoring set only after the first six reveal which topology patterns are insufficiently represented.
7. Reconcile any future runtime/intervention claims with #10064 and independent pristine regrade with #10369.

## 12. Claim boundary

This protocol and its six cases are synthetic research design. They are not a causal graph engine, production capability, benchmark result, model evaluation or SOTA claim. No candidate code, build, test, PoC, grader or attack payload is executed. The representation does not upgrade source inference into runtime/intervention evidence. The CEB-REVIEW primary metric set remains unchanged.
