# CEB Review Causal-Identifiability Protocol V0 — 2026-11-07

**Status:** Public research protocol / source-only development. It is not an admitted task schema, executable causal engine, benchmark result or new metric lane.  
**Companion:** [Six-case identifiability development challenge](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md) and [separate public answer key](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_KEY_V0_2026-11-07.md).  
**Composes with:** CEB causal-diagnosis preregistration, causal-topology protocol, counterfactual-control qualification, sequential evidence acquisition, gold-completeness/open-set discovery and dual-annotator calibration.  
**Primary CEB-REVIEW measures:** unchanged — \`defect_recall_at_5\`, \`actionable_precision\`, \`reviewer_burden\`.  
**Execution boundary:** candidate-controlled execution remains blocked by #10064; independent pristine regrade remains blocked by #10369.

## 1. Problem statement

A causal graph can be internally consistent and still not be identified by the evidence. Multiple causal models may explain the same source excerpt, test outcome, log sequence or repaired revision while predicting different outcomes under the intervention being considered.

CEB must distinguish:

- a mechanism that is supported and identified within a declared scope;
- a claim for which only a bounded set of outcomes is supported;
- multiple observationally compatible explanations that disagree about causation;
- a claim contradicted by admissible evidence;
- a case in which the evidence packet is incomplete or its authority is insufficient.

This is not a confidence-score problem. High subjective confidence does not resolve non-identifiability. An agent may be highly confident that one compatible model is plausible while another model, equally compatible with the supplied evidence, predicts a different effect.

## 2. Paradigm shift: reason over the compatible-model set

Let \(E\) be the supplied evidence and \(A\) the explicit assumptions/scope. Let \(\mathcal{M}(E,A)\) be the set of causal models consistent with both.

For a claim or intervention query \(q\), each \(M\in\mathcal{M}(E,A)\) implies a predicted outcome set \(\mathcal{Y}_M(q)\).

A claim is **identified for q within scope** only when every admissible model agrees on the relevant claim/outcome under the declared semantics. If the admissible models imply different outcomes, the claim is not point-identified by this evidence.

The annotation should not require enumerating every imaginable model. It should identify the smallest *material alternative set* that is sufficient to show whether the claim is identifiable. Each alternative must be grounded in a plausible completion of the missing context, not invented merely to force abstention.

### 2.1 Five claim dispositions

| Disposition | Rule | Permitted language |
|---|---|---|
| \`identified\` | All admissible models agree on the claimed mechanism/outcome for the declared query and scope | “The supplied evidence establishes…” |
| \`partially_identified\` | The evidence constrains the possible mechanisms/outcomes but does not select a unique one | “The evidence bounds the possibilities to…” |
| \`non_identifiable\` | At least two evidence-compatible models imply materially different answers to the same claim/query | “The supplied evidence cannot distinguish…” |
| \`contradicted\` | Admissible evidence rules out the candidate claim within scope | “The claim is inconsistent with…” |
| \`unresolved\` | The evidence, assumptions, identity, authority or scope boundary is insufficient even to construct a defensible compatibility set | “There is insufficient admissible evidence to determine…” |

Do not collapse \`partially_identified\`, \`non_identifiable\` and \`unresolved\`. A known finite set of alternative mechanisms is stronger than an unknown evidence gap; neither establishes the candidate mechanism.

### 2.2 Claim-specific, not graph-global

A system can identify one statement while a related statement remains non-identifiable. For example:

- The source may establish that a queue worker uses a serialized authorization result.
- It may not establish that a particular production job executed after revocation.
- It may establish that a code path permits a stale-authority outcome under a declared event order.
- It may not establish that this order occurred in the observed incident.

Never assign one global “identified graph” badge. Record dispositions per causal claim and intervention prediction.


### 2.3 Do not collapse six different causal claims

These statements are related, but they are not interchangeable evidence levels. A result supporting one does not automatically establish the next.

| Claim kind | Exact question | Minimum evidence needed | What it does not establish |
|---|---|---|---|
| **Source structural possibility** | Does this source/configuration permit a path under stated assumptions? | Exact source revision, relevant caller/configuration scope, and source-grounded control/data/authority edges | That the path was compiled, selected, executed or caused an incident |
| **Build-world selection** | Which source/feature/target implementation is present in this artifact? | Frozen build invocation, resolved features/cfg/target, artifact digest and provenance | That the selected path executed for a given request |
| **Execution realization** | Did this path/state transition occur in this concrete run? | Qualified trace/test/instrumentation bound to exact run, source and environment | That the event was an actual cause rather than correlated, redundant or pre-empted |
| **Actual-cause attribution** | In this context and under the declared causal model, did this realized factor make a counterfactual difference to the outcome? | A contextual causal model, actual values/events, justified structural equations/causal semantics, and a valid contrast/intervention or other accepted actual-cause criterion | That the same factor is a general causal effect across other models, builds or populations |
| **Interventional-effect identifiability** | Do all evidence-compatible models agree on the effect of a specified intervention? | A declared model class/assumptions and compatible-model set, with intervention semantics and query-specific predictions | That the intervention occurred, or that a specific historical incident was caused by the factor |
| **Incident attribution** | Is the causal claim tied to this incident, asset, actor, revision, artifact, environment and time window? | Identity/lineage binding plus admissible event and causal evidence for that incident | A generalizable causal law beyond the attributed scope |

The table is a claim-disambiguation aid, not a compulsory linear ladder. For example, interventional-effect identifiability can be studied from a qualified model and assumptions without claiming that a particular historical incident occurred. Conversely, a trace may establish execution realization while the actual cause remains ambiguous.

Use the following wording discipline:

- “The source permits…” for source structural possibility.
- “The artifact selects…” only for qualified build-world selection.
- “The trace records…” for execution realization.
- “This factor is an actual cause under model/semantics X and context C…” only when the actual-cause criterion is justified.
- “The effect of intervention do(X=x) is identified within model class A…” only when all admissible models agree on the query.
- “This incident is attributed to…” only when event identity, source/artifact lineage and causal evidence are all bound.

**Relationship to the literature:** actual-causality fault localization such as SoftReX addresses responsibility/actual-cause questions over an execution- and mutation-grounded evaluation; dynamic slicing such as CausalRepair identifies executed dependencies relevant to a failing test; partially observed SCM theory studies whether interventional quantities are identified across compatible causal models. These are complementary constructs, not alternative names for one task. CEB must preserve each paper's target claim and evidence assumptions.



### 2.4 The burden of proof is asymmetric

A single valid pair of evidence-compatible models that predicts different outcomes for the same query is sufficient to demonstrate **non-identifiability** within the declared scope.

The converse does not hold: failing to discover a competing model does not establish **identifiability**. A positive identified disposition requires a defensible completeness argument for the relevant model class and scope. The reviewer must state:

- which source, caller, dependency, feature, target, environment and event-state space is included;
- why relevant alternatives outside that boundary cannot affect the claim/query, or why they are explicitly excluded;
- which structural, policy or causal assumptions close the model class;
- how uninspected callers, hidden state, latent context, measurement/selection processes and alternate paths were audited;
- what open-world remainder remains unknown.

If the model class is closed only by an assumption (for example, “these are the only in-scope callers”), label the result **identified conditional on that assumption**. The assumption is part of the claim and must be visible. If the boundary/completeness argument is not defensible, use \`partially_identified\` or \`unresolved\` even if the reviewer did not find a counterexample.

Do not reward breadth of imagination or unlimited alternative generation. A non-identifiability witness must be materially compatible with the evidence, and a completeness argument must be auditable against a concrete scope boundary. This is a rigor requirement, not a confidence score.


## 3. Evidence-compatible alternative models

For each material claim, the reviewer records:

1. **Claim/query:** exactly what relationship or intervention effect is being evaluated.
2. **Observed evidence:** source anchors, tests, traces, configuration, contracts and authority class.
3. **Assumptions:** build features, target, environment, caller population, event ordering, state durability, data provenance and boundary conditions.
4. **Candidate model:** the proposed topology and mechanism.
5. **Material alternatives:** the smallest set of other plausible models that fit the same evidence.
6. **Disagreement witness:** one query/intervention on which the models predict different outcomes.
7. **Current disposition:** identified / partially identified / non-identifiable / contradicted / unresolved.
8. **Minimum additional evidence:** the smallest observation or safe intervention that could remove the material disagreement.
9. **Claim ceiling:** what can be asserted before that evidence is acquired.

### 3.1 Unknown is not false; absent is not unobserved

Maintain separate edge/node statuses:

- \`supported\`: positively supported by an admissible source/evidence anchor;
- \`contradicted\`: an admissible anchor rules out the relationship in scope;
- \`unknown\`: the relevant source/context was not provided or inspected;
- \`not_applicable\`: the relationship is outside the declared model/scope;
- \`latent_context\`: a plausible unobserved common context is required to represent the current ambiguity;
- \`unresolved_authority\`: a datum exists but its provenance, identity or authority is insufficient.

Do not silently turn a missing caller into a nonexistent caller, a missing configuration into a default configuration, absent telemetry into a negative observation, or a missing edge into a proven absent edge.

### 3.2 Latent context and confounding

When a hidden factor could influence both a suspected cause and the observed outcome, represent it as a latent/context node or an explicit confounder note. Do not insert a fictitious source node or pretend its value is known.

A correlation, temporal adjacency or co-occurrence does not distinguish:
- \(X \rightarrow Y\);
- \(Z \rightarrow X\) and \(Z \rightarrow Y\);
- \(X \leftarrow Z \rightarrow Y\) with no \(X \rightarrow Y\);
- a selection/measurement process that changes which observations are visible.

If the evidence does not distinguish these models, preserve the ambiguity. Do not use a Bayesian prior or language-model preference as if it were identification.

## 4. Minimal discriminating evidence

Evidence acquisition should be driven by the *disagreement witness*, not by generic “run more tests” advice.

A proposed evidence request must state:

| Field | Required content |
|---|---|
| Claim/query | Which disputed causal claim or intervention effect is being decided |
| Compatible alternatives | Which models remain possible before the request |
| Predicted outcomes | What each alternative predicts for this exact observation |
| Evidence type | Source definition, caller inventory, manifest/configuration, static query, test, trace, intervention or independent oracle |
| Authority | Existing CEB evidence-authority class; do not upgrade a source inference into runtime evidence |
| Scope | Exact source revision, build/feature/target, actor, resource, time window and environment |
| Cost | Human time, compute, tool, execution and interruption cost |
| Risk/disturbance | Security, privacy, production disturbance, state mutation and contamination risk |
| Stop rule | What result resolves the disagreement; what result leaves it unresolved |

Prefer a safe, low-cost observation when it separates the models. Use a controlled intervention only if an observation cannot discriminate and the action is authorized under the existing execution gates.

A query that every alternative predicts identically is not discriminating evidence. A broad test suite is not automatically informative. A minimal probe that can separate two competing models may be more useful than hundreds of un-targeted checks.

### 4.1 Selection rule

For the development protocol, rank candidate evidence requests lexicographically:

1. admissible and within authority/safety policy;
2. discriminates at least one material pair of compatible models;
3. targets a claim with high severity or high downstream decision impact;
4. lowest total cost and disturbance among equivalently discriminating requests;
5. preserves a reproducible, exact-revision evidence record.

Do not convert this ordering into an uncalibrated numeric “value of information” score. If a probabilistic policy is later introduced, its forecasts and realized information gain must be separated and calibrated under the existing prospective VoC contract.

## 5. Confounder and non-identifiability taxonomy

The following are development annotation categories, not new CEB primary metrics or task schema:

- **missing-context ambiguity:** caller, helper implementation, dependency, manifest or configuration absent;
- **structure–mechanism confounding:** a latent context changes both which edge/path exists and how the operation behaves;
- **common-cause confounding:** a hidden workload, tenant, state or environment drives both the suspected condition and failure;
- **overdetermination:** two or more mechanisms are independently sufficient for the same observed consequence;
- **pre-emption:** one mechanism produces the observed outcome before another candidate mechanism can do so;
- **masking/compensation:** an intermediate operation hides, repairs or neutralizes an earlier fault before output;
- **temporal ambiguity:** source code permits multiple event orders but the supplied evidence does not establish which occurred;
- **configuration ambiguity:** feature, target, build, policy or environment selects between different mechanisms;
- **selection/censoring ambiguity:** only failures, surviving requests, accepted suggestions or observed logs are included;
- **repair-entanglement ambiguity:** a historical repair changes several causal parents or paths at once;
- **oracle ambiguity:** the observed test or expected outcome does not distinguish the competing mechanisms.

More than one category may apply. Keep the causal alternative explicit; a label alone does not explain the ambiguity.

## 6. Relationship to causal-topology protocol

The Causal-Topology protocol describes the mechanism graph. This protocol qualifies whether that graph is identified by the evidence.

| Causal topology | Identifiability extension |
|---|---|
| AND/OR gates | Are the gate conditions and alternative routes actually supported, or merely possible? |
| Temporal partial order | Does the source establish the order, or is an event trace/synchronization contract missing? |
| Minimal sufficient paths | Are these the only paths in the declared scope, or only known paths? |
| Intervention set | Do all compatible models predict that it blocks the target? |
| Residual reachability | Has every in-scope path been inspected, or is the remainder open-world? |
| Latent context | Which hidden context could change graph structure or mechanism? |
| Repair topology | Does the evidence identify which repair component removed the target path? |

A graph with unsupported but plausible edges is not a stronger explanation. A graph that leaves a material edge unknown can be more scientifically accurate.

## 7. Six public development cases

Use [CEB Review Causal-Identifiability Challenge V0](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md) and the separate [public answer key](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_KEY_V0_2026-11-07.md).

The cases cover omitted helper/caller source, configuration-dependent paths, common-cause confounding, overdetermined faults, temporal ordering and bundled repair attribution. The six anchors are deliberately public synthetic development examples, not independent statistical samples or sealed holdout tasks.

Reviewers must identify the material alternative model(s), claim ceiling and minimum discriminating evidence before seeing the key. Use the existing dual-annotator first-pass lock; preserve both model sets and evidence requests.

## 8. Experimental design and measurements

Compare three development conditions on identical evidence packets and matched budgets:

- **I0 — unstructured diagnosis:** existing reviewer instruction and evidence.
- **I1 — causal topology:** typed graph, AND/OR, temporal edges and residual paths.
- **I2 — topology + identifiability:** I1 plus compatible-model alternatives, claim dispositions and minimum discriminating evidence.

I2 is supported only if it reduces unsupported causal assertions and improves the usefulness of evidence requests without unacceptable loss of target detection, actionability or reviewer time. The comparison is an exploratory development ablation, not a new confirmatory Phase A hypothesis.

### 8.1 Diagnostic annotations only

Report, separately and descriptively under existing CEB adjudication:

- exact disposition agreement with blinded adjudication;
- unjustified identified/causal claims on deliberately non-identifiable packets;
- recognition of materially distinct compatible models;
- whether a requested observation actually separates the disputed models;
- evidence-request authority and safety errors;
- cases where the system abstains despite the claim being identifiable;
- additional human minutes and total evidence-acquisition cost.

These are proposed diagnostic annotations, **not new primary metric IDs**. Do not create a combined identifiability score or silently alter Recall@5, actionable precision or reviewer burden.

### 8.2 Prospective development targets

Use the six-case pack to debug the rubric only. After it is stable, author a larger repository-/fault-family-disjoint development set, pilot it, and preregister a power-derived holdout before viewing holdout outputs.

Provisional aspirations for a future confirmatory study:

- at least **95%** correct non-identifiability recognition on an explicitly constructed non-identifiable slice;
- **zero critical-severity unjustified identified claims** in the evaluated critical slice;
- at least **90%** of evidence requests judged genuinely discriminating and within authority;
- no material reduction in the existing primary CEB-REVIEW measures;
- total human + model + tool cost reported and no hidden second-reviewer time omitted.

These values are future research targets, not measured performance, safety certification or a guarantee. Confidence intervals and independent adjudication are mandatory; a small synthetic pack cannot substantiate them.

## 9. Industry and research comparison

The new protocol deliberately composes three different kinds of external work:

- **Causality-Based Fault Localization / SoftReX (arXiv:2611.01292):** actual causality, responsibility, mutation interventions, overdetermination, pre-emption and masked faults; evaluated on Defects4J against eight MBFL techniques. It localizes program elements under failing-test execution, not source-only PR claims under missing context. Its result motivates intervention-aware debugging, not direct CEB metric equivalence.
- **CausalRepair (PACMSE/ISSTA 2026):** minimal causal context assembled from test-semantic slicing and dynamic execution slicing. It demonstrates the value of narrowing context to dependencies that explain a concrete failure. It assumes an executable test/trace context, unlike CEB's source-only development cases.
- **Partially Observed Structural Causal Models (UAI 2026):** formalizes latent context that co-determines interaction structure and downstream mechanisms, with node- and edge-level intervention hierarchies and identifiability results. It motivates explicit compatible-model sets and claim-specific intervention identifiability; it is not a software code-review benchmark.

CEB's proposed research contribution is to bridge these ideas at the review boundary: source/repository evidence, exact revision and configuration, human-adjudicated claims, causal graph alternatives, intervention prediction and an explicit evidence request. This contribution remains a hypothesis until a matched evaluation demonstrates it.

## 10. Integration and stop/go boundaries

- Compose with the existing CEB sequential evidence acquisition and prospective VoC contracts; do not create a second scheduler.
- Reuse existing evidence-authority classes, event records and run receipts; do not create new receipt formats.
- Keep graph and model-set annotations in the existing human calibration process; do not create a parallel annotation authority.
- P/I/F/N/M/IRR controls remain governed by the counterfactual-control qualification protocol.
- No candidate code, test, PoC, hidden grader or production intervention is executed by this protocol.
- Do not claim that a source-only model set proves a production incident's cause.
- If the admissible model set cannot be bounded, use \`unresolved\`; do not invent completeness.
- If all models disagree, do not average them into one consensus causal graph.
- If evidence acquisition is unsafe or unauthorized, stop and report the missing authority.

## 11. No-runner implementation sequence

1. Have two independent reviewers annotate the six public packets without seeing the key.
2. Compare material-model sets, claim dispositions, disagreement witnesses and evidence requests.
3. Adjudicate whether each request truly separates the expected alternatives.
4. Revise only the ambiguous wording or faulty case design; issue a new version and retain the original first-pass annotations.
5. Run the I0/I1/I2 *development-only* comparison once task and rubric ambiguity is reduced.
6. Expand to real repository evidence only with independent task admission, privacy/licensing review, oracle review and the existing execution/regrade gates.

## 12. Claim boundary

This is a research protocol and synthetic development design. It reports no model evaluation, no human reliability estimate, no causal identification result and no SOTA achievement. The aim is to make “I do not know from these observations” a precise, useful engineering conclusion rather than an unhelpful abstention.
