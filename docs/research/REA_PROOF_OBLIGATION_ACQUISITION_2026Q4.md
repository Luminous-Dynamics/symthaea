# REA — claim-obligation-directed evidence acquisition benchmark v0 (2026 Q4)

**Status:** research contract and original synthetic development fixtures. Labels are design expectations, not human-adjudicated ground truth. No evaluator, planner superiority, or benchmark result is claimed.

**Owner:** [#11108 — REA-RUST-050](https://github.com/Luminous-Dynamics/symthaea/issues/11108), composed with [#9911](https://github.com/Luminous-Dynamics/symthaea/issues/9911).

## Executive thesis

Information gain and claim readiness are different objectives.

An experiment can sharply distinguish hypotheses while leaving a required proof obligation unresolved. A cheap, low-information observation can instead close the only missing obligation and enable a safe, correctly scoped claim.

The benchmark measures whether the system acquires sufficient admissible evidence for a declared claim—not merely whether it reduces entropy or produces a high-confidence hypothesis.

A true proposition is not automatically assertable:

```
world-state truth != evidence sufficiency != authorized decision
```

## Research foundation

- Usmanova et al., *How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation*, EMNLP 2026: replacing retrieved evidence with gold annotations improved veracity accuracy by 14–22 percentage points in their evaluated settings. Results also varied by domain and metric. This motivates separate retrieval, evidence selection and warrant tracks; it is not a REA result. https://aclanthology.org/2026.emnlp-main.1915/
- Akhtar et al., *Ev2R: Evaluating Evidence Retrieval in Automated Fact-Checking*, TACL 2026: combines reference alignment with verdict-support evaluation and compares with human ratings and adversarial tests. This motivates evaluating both evidence alignment and whether the required verdict is warranted. https://aclanthology.org/2026.tacl-1.25/
- Sahitaj et al., *From Articles to Premises*, LREC 2026: reports improved retrievability from decontextualized evidence premises and higher verdict prediction scores in its evaluated corpora. REA should preserve the original source span and bytes when extracting standalone premises; a rewritten premise is a derived object, not a replacement for its source. https://aclanthology.org/2026.lrec-1.613/
- Harikumar et al., *Robust Bayesian Decision Making under Adversarial Uncertainty*, UAI 2026: shows why nominal information gain can yield fragile downstream decisions under hidden or weakly modelled effects. REA must report in-ensemble and out-of-ensemble decision performance separately. https://proceedings.mlr.press/v337/harikumar26a.html
- ForceBench's evidence-force contrast motivates fixing the evidence and varying claim force. Its reported results are external measurements, not REA outcomes. https://aclanthology.org/2026.findings-emnlp.1577/

## 1. Paradigm shift

### From hypothesis ranking to minimum sufficient evidence

Keep three structures distinct:

1. **World-state hypotheses:** possible truths about the target or event.
2. **Claim proof obligations:** requirements that must be satisfied before a particular wording or action is justified.
3. **Candidate observations:** permitted, bounded experiments that can satisfy, contradict or leave obligations unknown.

A proposition may be true while the available evidence remains insufficient. A model's posterior, confidence, EIG, citation count or planner utility must never directly mark an obligation satisfied.

### Typed obligation states

The initial vocabulary is:

- `satisfied`: admissible evidence establishes the obligation within its declared scope.
- `unsatisfied`: available evidence does not establish it.
- `conflicted`: admissible evidence conflicts and the conflict is unresolved.
- `unknown`: the required observation or authority is unavailable.
- `incomplete`: a defined collection/process is known to have gaps.
- `contradicted`: evidence supports an incompatible obligation value.

Do not merge unknown, not-run, missing, zero, failed, unavailable and contradicted.

### Claim readiness

A claim is ready only when every mandatory obligation is satisfied and no mandatory conflict remains. Optional obligations do not become mandatory by model preference. The exact readiness policy is owned by the claim rule and adjudicator, not the experiment planner.

For v0, this benchmark uses conjunctions of mandatory obligations. Later versions may define typed AND/OR proof graphs, but must preserve explicit alternatives and their admissible evidence.

## 2. Existing planner composition

No new EIG, robust-EIG, EVSI or minimax implementation is justified here. Compare the existing policies:

| Arm | Existing owner | Role |
|---|---|---|
| Random eligible observation | benchmark baseline | Lower bound |
| Cheapest eligible observation | benchmark baseline | Cost-only control |
| Point EIG | #10074 | Nominal information gain |
| Robust EIG | #10105 | Scenario expected/worst/CVaR information gain |
| EVSI / minimax regret | #10109 / #10174 | Downstream decision quality and regret |
| Obligation-first greedy | benchmark baseline | Direct missing-obligation closure |
| Combined gated policy | future composition | Decision value subject to mandatory proof gates |

The planner only proposes observations. Execution produces new Evidence. A separate verifier/replay step determines whether the observed result is admissible for the obligation.

Do not let a candidate experiment's predicted outcome satisfy an obligation before it is executed and independently assessed.

### Planner-runnability boundary

The current 18-case v0 corpus is a claim-readiness and fixture-integrity dataset. It supplies candidate outcomes, costs and grader outcome-to-obligation effects, but it does not yet include the complete planner API inputs: grounded hypothesis priors/Evidence IDs, per-hypothesis outcome likelihood rows, robust scenario weights, or terminal decision utilities. Therefore the listed EIG/EVSI/minimax policies are proposed matched arms, not currently runnable v0 comparisons.

Follow-on [#11125 — REA-RUST-051](https://github.com/Luminous-Dynamics/symthaea/issues/11125) owns the planner-runnable extension. It will reuse #10074/#10105/#10109/#10174 and #10314 rather than copying planner or receipt implementations. No comparative planner result or policy superiority claim is valid until that extension is qualified.

## 3. Fixture v0

The fixture contains 18 original synthetic cases spanning:

- high-EIG but non-decisive observation versus low-EIG proof closure;
- shared-source mirrors and same-session tool outputs;
- provider-reported digest versus local artifact hashing;
- missing pagination pages;
- registered transparency view versus issuer history;
- filtered query versus whole-archive inventory;
- stale advisory versus current release;
- irrelevant correlated logs versus controlled intervention;
- unresolved contradictions and configuration differences;
- numeric intent and canonical quantity-kind;
- bounded observation versus universal negative;
- authorization and irreversibility controls;
- processing failure after successful enumeration and acquisition;
- hidden issuer omission;
- multi-step build identity, device binding and behavior reproduction.

The cases are synthetic. Their expected outcomes are not model labels or human adjudications. The `world_state_oracle` field is grader-only and must never be shown to an evaluated system.

### Admissibility rule

Every evidence effect names an evidence class and source family. A candidate cannot satisfy an obligation unless its effect uses one of that obligation's declared admissible evidence classes. Source-family IDs support deduplication and common-mode controls; they do not prove independence by themselves.

A self-report can satisfy an obligation to reproduce what a provider reported. It cannot satisfy an obligation to independently verify artifact bytes. A complete filtered query cannot satisfy whole-artifact completeness. A transparency checkpoint cannot prove issuer participation without a separate issuer-history obligation.


## 3.0 From EIG maximization to proof-complete acquisition

The planner objective should be studied as a **constrained sequential evidence-acquisition problem**, not as a renamed entropy score.

For claim c, define a state:

```
s_t = (belief_state_t, obligation_state_t, evidence_lineage_t, target_scope_t)
```

where:
- `belief_state` represents uncertainty over world-state hypotheses;
- `obligation_state` represents the separately adjudicated state of each required proof obligation;
- `evidence_lineage` identifies exact observations, source families, derivations, freshness and known shared roots;
- `target_scope` binds the claim to the exact artifact, operation, environment and time interval.

An experiment proposal is an action. Its predicted outcomes may forecast state transitions but cannot satisfy an obligation. Only after execution does a new observation become Evidence; the independent verifier then determines admissibility and the obligation transition.

Let `G_c` be the set of states in which every mandatory obligation for c is satisfied and no mandatory conflict/defeater remains. A candidate policy pi is evaluated by:

```
minimize   E[cumulative acquisition cost under pi]
subject to P(false-ready under pi) <= epsilon
           every proposed action passes authorization/impact gates
           useful-claim coverage >= declared minimum
```

The probability/risk statement is meaningful only under a declared and evaluated data-generating distribution; it is not a universal guarantee. Report the empirical false-ready rate and its confidence interval, plus distribution-shift tests, rather than claiming the constraint is proved by a model's own likelihoods.

### Why this is a paradigm shift

- EIG rewards information about hypotheses; it may not close a mandatory obligation.
- EVSI rewards expected downstream decision improvement; it may still accept a decision whose source/provenance obligation is unresolved unless that constraint is encoded explicitly.
- Greedy obligation coverage targets missing obligations, but ignores uncertainty, conflicting outcomes, source dependence and downstream decision utility.
- A proof-complete acquisition policy makes evidence admissibility and mandatory obligation closure hard constraints, then optimizes decision utility, risk and cost inside that admissible region.

### Minimum useful baseline

Add a deterministic obligation-coverage baseline:
- consider only authorized, reversible, non-destructive candidates under the same target-impact/resource budget;
- estimate expected *admissible obligation closure* per unit cost from the planner-visible prediction only;
- never count predicted closure as actual satisfaction;
- use stable deterministic tie-breaking;
- abstain when no eligible action has positive predicted admissible value.

This is a baseline, not a new production planner. It should be compared with random eligible, cheapest-first, EIG, robust-EIG, EVSI and minimax-regret policies.

### Exact oracle for small cases

For small synthetic cases, enumerate every policy/action sequence under a hard cap and compute the true minimum-cost path to G_c using hidden grader outcomes. This gives an exact small-instance oracle against which heuristics can be measured. The exact oracle is a diagnostic optimum for the finite synthetic case only; it is never provided to the evaluated system.

For larger cases, study constrained stochastic shortest-path / POMDP approximations, receding-horizon planning and proof-obligation coverage heuristics. Do not claim set-cover or submodular approximation guarantees unless the required monotonicity/submodularity assumptions are formally proved. Contradictions, expiry, source common-cause and population gaps can make evidence updates non-monotone.

### Evidence can retract readiness

Evidence is not always a monotone accumulation process. A newly verified contradiction, expiry, revocation, artifact mismatch or discovery of shared provenance may move an obligation from satisfied to conflicted/unknown and revoke a previous readiness result. Benchmark sequences must include such invalidation events; the planner/verifier must not keep a stale success certificate.

## 3.0.1 Executable readiness reference

The standalone fixture checker defines a small, deterministic readiness rule:

- Ready iff every mandatory obligation's current, independently assessed status is exactly `satisfied`.
- `unknown`, `unsatisfied`, `conflicted`, `incomplete` and `contradicted` all block readiness.
- Optional obligations do not block readiness.
- Candidate predictions do not change the current obligation state.
- A fixture-only simulator can apply the grader's effect for an explicitly supplied *observed* candidate outcome; it checks that the candidate is authorized, reversible, non-destructive, that the outcome was declared, and that the evidence class is admissible.

Tests cover a high-information static probe that leaves readiness unchanged, an exact manifest observation that closes only the binding obligation, a page reconciliation that closes enumeration/acquisition, conflict persistence under score averaging, configuration-bound reproduction that resolves a conflict, and bounded no-event evidence that does not prove universal absence.

This executable reference is intentionally narrow. It does not authenticate actual Evidence records, verify producer signatures, execute target experiments, choose actions, optimize acquisition cost, or establish that the synthetic oracle matches the real world. The grader data is public development-only and is not a hidden benchmark.

## 3.1 Agent-visible and grader-only views

The combined fixture JSON is a development authoring source only. It must never be used directly as the agent prompt or planner input.

The split development views are:

- `rea-proof-obligation-acquisition-agent-view-v0.json`: claim, visible obligations and admissible evidence classes, candidate descriptions/outcomes, authorization, reversibility, impact and cost. It excludes world-state labels, expected policy labels, and outcome-to-obligation transition annotations.
- `rea-proof-obligation-acquisition-grader-v0.json`: synthetic world-state oracle, expected-policy annotations and candidate effects. It explicitly declares `never_mount_in_agent_workspace`.
- Their paired Draft 2020-12 schemas reject undeclared keys.

A Rust fixture checker validates both views, checks exact case/candidate correspondence, requires each grader effect to refer to an agent-visible outcome and obligation, and applies the admissibility no-upgrade rule. Mutation tests inject oracle leakage and effect leakage into the agent view and require rejection.

Important limitation: the grader-only development file remains public in this repository. It is not a hidden test set and cannot support contamination-resistant evaluation. Hidden evaluation cases and gold labels must be held in a separately access-controlled service or private repository, with no oracle material available to the agent runtime. A filename, metadata flag, or branch boundary is not access control.

## 3.2 Decompose retrieval, extraction, warrant and provenance

Recent evidence-based QA and fact-checking studies reinforce that "retrieval failure" is not one failure mode:

- Usmanova et al. (EMNLP 2026) found that replacing retrieved evidence with gold annotations raised veracity accuracy by 14–22 percentage points in their evaluated cross-domain settings; rankings changed across datasets and metrics. This is an oracle gap, not a REA result. https://aclanthology.org/2026.emnlp-main.1915/
- Mo et al. (EMNLP 2026), *Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering*, separate retrieval failures from extraction failures. They report extraction failures account for nearly half of per-hop deficiencies in their evaluation, and that tested retrieval interventions did not resolve those extraction failures. https://aclanthology.org/2026.emnlp-main.834/
- Akhtar et al. (TACL 2026), Ev2R, combine reference alignment with verdict-support evaluation and use adversarial perturbations such as evidence deletion, irrelevant insertion, redundancy, argument reordering and meaning-preserving number/word transformations. https://aclanthology.org/2026.tacl-1.25/
- Qian et al. (Findings EMNLP 2026), ForceBench, hold cited evidence fixed while altering claim strength; their reported generic prompting MVR is 47.2%, reduced to 24.5% by explicit warrant prompts, but not eliminated. These are external measurements. https://aclanthology.org/2026.findings-emnlp.1577/

REA should measure four serial but independently fallible stages:

1. **Retrieval:** Was the necessary source/span made available to the system?
2. **Extraction:** Did the system recover the correct atomic proposition, value, relation, time and scope from the source?
3. **Warrant:** Does that proposition actually establish the required claim obligation at the requested force?
4. **Provenance/admissibility:** Is the observation bound to the right artifact, time, source lineage and population, and is its evidence class admissible?

A source can be retrieved but misread. A fact can be extracted exactly but not entail the claim. A passage can entail a claim but be stale or derived from a non-independent source. A supported local observation can still be insufficient for a whole-population claim.

### Matched oracle interventions

Where a case has human-adjudicated source spans and propositions, evaluate matched arms under the same task, target, compute budget and grader:

- **R0 — system retrieval:** end-to-end retrieval and claim evaluation.
- **R1 — gold document/span availability:** replace only the retrieved source set with the adjudicated relevant spans; keep extraction and warrant unchanged.
- **R2 — gold proposition extraction:** provide adjudicated atomic propositions extracted from the same source spans; keep warrant unchanged.
- **R3 — gold admissibility/warrant map:** provide human-adjudicated obligation-to-evidence relations; measure the residual readiness/decision implementation error.
- **R4 — adversarial evidence controls:** remove a premise, insert irrelevant evidence, duplicate a source, reorder dependent premises, alter time/scope/number wording, or substitute a stale/conflicted source.

Derived diagnostic gaps:
```
retrieval_gap   = score(R1) - score(R0)
extraction_gap = score(R2) - score(R1)
warrant_gap    = score(R3) - score(R2)
```

These differences are diagnostics, not additive capability scores. They may be negative, and each must be reported with paired confidence intervals. R3 is not a model capability leaderboard result: it is an upper-bound diagnostic with oracle adjudication.

### Extraction representation

An extracted proposition is a derived object, never a replacement for the raw source. It must retain:
- exact source/evidence ID and raw digest;
- exact byte/line/JSON-pointer span or other source-native locator;
- extractor identity/version and extraction operation digest;
- proposition predicate, subject, object/value, relation, modality, time interval and scope;
- ambiguity/alternatives and unresolved coreference;
- whether the proposition was directly stated, normalized, inferred or model-generated.

Do not let paraphrase or premise normalization silently upgrade evidence force. Every extracted atom must be replayable to its exact source span; unsupported components remain explicit.

### Context-preserving evidence atoms and failure-localized repair

Three further 2026 findings inform the implementation direction:

- PrimeFacts (LREC 2026) builds 49,718 source anchors across 13,106 PolitiFact articles, then rewrites anchored sentences into standalone premises. In its evaluated corpora, this improved cross-article evidence retrieval by up to 30% relative MRR and verdict Macro-F1 by 10–20 points over its baseline. These gains are specific to its datasets and methods. https://aclanthology.org/2026.lrec-1.613/
- Evidence-Based Generation (Findings EMNLP 2026) adds line-level citations and joint answer-plus-evidence scoring. Its authors report that similarity retrieval may retrieve a relevant chunk more reliably than all required chunks, and that chunking can destroy document-level identity information needed for cross-passage reasoning. https://aclanthology.org/2026.findings-emnlp.2222/
- Dual Entity Recovery RAG (Findings EMNLP 2026) identifies entity-loss failures at both retrieval time and generation time: a passage can be retrieved correctly yet become uninterpretable after losing its antecedent/context. https://aclanthology.org/2026.findings-emnlp.1068/
- LoCoRAG (EMNLP 2026) studies global failure localization and matched repair, reporting a 10.8-point average LLM-Acc gain over its strongest baseline on 3-hop-or-deeper settings. This is an external multi-hop QA result, not a REA result. https://aclanthology.org/2026.emnlp-main.2183/

REA should adopt the *representation lesson*, not copy a particular RAG architecture.

Keep two distinct layers:
- **Immutable source layer:** original bytes, artifact identity, exact source-native locator, acquisition context, time, provider and lineage.
- **Derived proposition layer:** normalized subject/relation/object/value, modality, time interval, scope, entity anchors, coreference dependencies and extraction provenance.

A proposition may be decontextualized for retrieval, but its source span and all entity/context dependencies remain attached. It is a derived object, not a substitute for the source. A model-generated proposition is not independent Evidence.

For multi-hop claims, the agent must retrieve the complete dependency set, not merely one topically relevant passage. The evaluator should include:
- entity alias and coreference perturbations;
- same-name/different-entity controls;
- missing antecedent and missing document-metadata controls;
- passage-only vs full-document context;
- source-order and argument-order changes;
- multi-fact tasks requiring two or more independently sourced premises.

### Failure-localized repair loop

Do not respond to every failed claim by retrieving more text. Diagnose the earliest failed stage and choose a matched repair:

| Earliest failure | Appropriate repair |
|---|---|
| Query/plan decomposition | Revise the obligation decomposition, preserving the original claim and policy constraints |
| Retrieval | Change source discovery, query expansion or candidate ranking |
| Extraction | Re-read the exact source span, restore entity/context anchors, or use a different extractor |
| Warrant | Re-evaluate the entailment/non-implication relation with claim force held fixed |
| Provenance/admissibility | Reacquire, resolve dependency lineage, freshness, artifact identity or population evidence |
| Replay/identity | Reconstruct from exact bytes and pinned rule/schema inputs; do not ask a language model to explain away a digest mismatch |

A repair must be localized and its downstream derived objects recomputed. If an upstream proposition is withdrawn or its source is superseded, dependent inferences must be marked stale/revoked and replayed. This composes with evidence-continuity and causal-diagnosis owners; it is not a new memory graph.

### Metrics added

- retrieval recall@k for mandatory admissible premises, plus precision and source-family diversity;
- extraction exactness by subject/relation/value/modality/time/scope fields;
- source-span traceability and extraction replay;
- warrant correctness conditional on identical fixed evidence;
- evidence-admissibility error conditional on the same extracted propositions;
- oracle intervention gap by stage and domain;
- adversarial degradation and semantics-preserving invariance;
- false-ready rate and useful coverage after each intervention.

Never collapse these into one score. If R1 closes most of the gap, invest in retrieval. If R2 closes it, improve extraction. If only R3 closes it, the bottleneck is warrant/admissibility. If none closes it, inspect claim semantics, action policy, oracle quality or hidden-state mismatch.

## 4. Metrics

Report independent scorecards, not one composite:

- **False-ready rate:** fraction of ready claims with at least one unsatisfied mandatory obligation.
- **Verified readiness:** fraction of eligible cases where every mandatory obligation is satisfied and the final wording stays within the claim ceiling.
- **Useful coverage:** eligible cases receiving a useful, correctly qualified conclusion; always-abstain is not a winner.
- **Obligation closure:** mandatory obligations satisfied per executed observation and resource unit.
- **Cost-to-sufficient-evidence:** wall time, observation count, bytes, compute, energy and monetary cost when known.
- **Unnecessary acquisition:** observations that do not change an obligation or downstream decision.
- **Admissibility error:** evidence classes accepted where they are not permitted.
- **Lineage inflation:** shared-root or derived observations counted as independent.
- **Conflict handling:** unresolved conflicts improperly averaged, voted away or omitted.
- **Safety:** unauthorized, irreversible, destructive, or over-budget proposals selected.
- **Reproducibility:** exact manifests, inputs, evidence, policy, model and grader digests.

The benchmark's strongest release requirement is zero false-ready accepts in deterministic and hidden negative controls. This must be paired with useful coverage and cost to prevent universal abstention from appearing successful.

## 5. Experimental protocol

### Separate policy from execution

Each trial has distinct phases:

1. Freeze the claim, mandatory obligations, source-family map, authorization and budget.
2. Provide the planner with only current-time, agent-visible information.
3. Record its proposal, ranking, rejections, predicted outcomes and claim ceiling.
4. Execute only the approved observation under the case harness.
5. Create new observation Evidence with exact artifact/session/source binding.
6. Verify its identity and lineage and replay any applicable claim derivation.
7. Update obligation states through the independent adjudicator.
8. Determine readiness and compare the selected policy with the hidden oracle.

The future observations and hidden `world_state_oracle` must not leak into the planner context.

### Matched comparisons

All arms receive the same eligible task set, claim, current evidence, candidate observations, authorization, resource budgets, model/tool versions and hidden oracle. Report official planner metrics separately from derived claim-readiness metrics.

For learned likelihoods, hold calibration data out from the benchmark. For scenario planners, reserve hidden data-generating processes outside the declared scenario set. Do not describe a finite-scenario worst case as distribution-free robustness.

### Human adjudication

Before scoring:
- Two independent reviewers label every mandatory obligation.
- Domain reviewers resolve disagreements or mark the case not adjudicable.
- Reviewers see original source bytes/spans and lineage, not model explanations alone.
- A multi-axis audit ensures a supposed single-obligation case does not secretly change multiple obligations.
- Freeze development and hidden source-family splits before model evaluation.

Automated judges are measurement instruments, not gold truth.

## 6. Proposed qualification targets

Targets, not achieved results:

| Measure | Proposed release gate |
|---|---|
| Deterministic false-ready | 0 accepts |
| Hidden false-ready | 0 accepts on preregistered negative controls |
| Shared-source independence uplift | 0 |
| Filtered-to-whole-population completeness upgrade | 0 |
| Provider-report-to-local-verification upgrade | 0 |
| Unauthorized/irreversible selection | 0 |
| Useful coverage | Strictly above always-abstain baseline |
| Cost to sufficient evidence | Publish median/p90 and paired comparisons |
| Decision regret | Report p50/p90 and out-of-ensemble penalties |
| Reproduction | Independent exact-input replay for every scored result |

A numeric percentage gain in cost or coverage must be set only after a pilot establishes human agreement, task difficulty, baseline variance and statistical power. Do not pick an arbitrary improvement threshold and treat it as a scientific result.

## 6.1 External benchmark baselines and denominator discipline

Use external benchmarks as separate reference tracks; never pool their scores.

### AgentRE-Bench v3.1 (November 2026)

The current published Windows PE run reports 20 artifacts, five attempts per artifact, six models and 600 episodes. Muse Wonder reports 87.37/100 (100/100 valid answers); GPT-6 Astra reports 87.04/100 (100/100 valid answers). Their published 95% intervals overlap substantially (Muse 85.18–89.56; Astra 85.80–88.29), so the reported ordering is not evidence of a clear winner. The benchmark reports a 90% Main / 10% Bonus score and says all 600 scores were independently reproduced. https://www.agentre-bench.ai/

REA alignment:
- retain the official AgentRE score unchanged;
- add claim-level source-grounded accuracy and obligation completion as separate supplemental measures;
- distinguish valid structured answer, evidence coverage and conditional answer quality;
- replay the exact tool/evidence trace and independently reproduce the artifact-to-claim mapping;
- keep all attempts, failures and budget exits visible.

### SRE-Bench (updated October 2026)

SRE-Bench contains 19 private programs, 262 binary instances, 1,572 deterministically graded tasks, five RE domains and 44 anti-analysis mechanisms. Its published public-facing GPT-6-Astra row has 181 graded instances, of which 149 are fully solved: 82.3% conditional on graded instances. Graded coverage is 181/262 = 69.1%. The derived full-solve yield over all eligible instances is 149/262 = 56.9% if ungraded instances are treated as not yet solved; this is not the benchmark owner's official metric and must be labelled as a coverage-adjusted derived measure. https://daplab.cs.columbia.edu/benchmarks/sre-bench/

REA alignment:
- publish eligible, attempted, graded, completed and refused/unavailable counts separately;
- show official conditional scores beside coverage-adjusted yields;
- never treat a refusal or ungraded task as absent from the denominator;
- separate pass@1 from pass@k and don't count repeated attempts as independent benchmark cases;
- use artifact/program/source-family clusters—not individual attempts—as the resampling unit.

### REA's competitive scorecard

Do not claim superiority from one mean score. Report:
1. Official task score under each external benchmark's original formula.
2. Eligible coverage and valid structured-answer coverage.
3. Correct claim detail conditional on attempted/graded tasks.
4. Source-grounded detail and exact evidence-span traceability.
5. Unsupported-claim / false-ready rate.
6. Independent replay and artifact binding.
7. Abstention correctness and useful coverage.
8. Cost, time, tool calls, memory and energy under matched budgets.

For pairwise comparisons, freeze the task/artifact cohort and system configurations, then use paired cluster bootstrap intervals over artifacts/programs or independent source families. Model attempts nested inside one artifact are repeated measurements, not independent samples. If a cohort has a high refusal/ungraded rate, report it as a separate capability/coverage outcome rather than silently comparing only its answered subset.

## 6.2 Reverse-engineering tool ecosystem: compose, do not clone

The incumbent tools already provide deep specialist analysis:

| System | Native strength | REA/Symthaea relationship |
|---|---|---|
| Ghidra | Multi-platform SRE framework: disassembly, assembly, decompilation, graphing, scripting, extension points and interactive/headless use | Import typed observations and exact source locations; do not recreate its processor/lifter/decompiler stack |
| IDA Pro | Broad processor support, mature disassemblers/decompilers, debugging, plugins, IDAPython/C++ SDK, headless/library automation | Treat decompiler, signatures, debugger and plugins as distinct operation/provider evidence with exact version and configuration |
| Binary Ninja | Interactive disassembler/decompiler/debugger, BNIL family, automation and APIs including Rust bindings | Use its IL and analysis outputs as alternate derived observations; retain engine/version and transformation lineage |
| angr | Scriptable binary analysis with CFG recovery, IR lifting, symbolic/concolic execution, data-dependency analysis and path constraints | Use symbolic reachability and counterexample traces as experiment proposals or observations, not as proof without exact path/target binding |

Official capability references:
- Ghidra: https://github.com/NationalSecurityAgency/ghidra
- IDA Pro: https://hex-rays.com/ida-pro
- Binary Ninja: https://binary.ninja/
- angr: https://angr.io/

### What REA should add around these tools

REA is not another decompiler. It should provide a common evidence and claim layer:
1. **Artifact binding:** exact subject digest, container/format, architecture, build and target identity.
2. **Operation binding:** provider/tool version, executable or container digest where available, operation, configuration/profile digest, environment and limits.
3. **Observation binding:** raw output bytes/digest, source-native location (file offset, VA/RVA, function/block, instruction range, log/session span), extraction/normalization lineage and limitations.
4. **Claim binding:** typed predicate, scope, evidence IDs, required obligations, falsification conditions and explicit non-implications.
5. **Independent replay:** recompute supported claim derivations from request and exact Evidence without calling the producer evaluator.
6. **Dependency-aware corroboration:** track shared disassembler/lifter, parser, signature database, emulator, runtime, acquisition session or model roots. Different tool names do not prove independent observations.

### Competitive implementation strategy

Use an adapter portfolio rather than build a proprietary analysis engine:

- Static lane: Ghidra, IDA Pro, Binary Ninja and format-specific readers.
- Program-analysis lane: angr and other symbolic/dataflow tools.
- Dynamic lane: authorized debuggers, emulators, instrumented runtime and device observations.
- Evidence lane: REA admission, artifact/operation scope, lineage, numeric intent, population completeness, typed claims and replay.
- Decision lane: Symthaea point-EIG, robust-EIG and decision-aware EVSI/minimax selection, constrained by proof obligations and authorization.

An adapter's output is not automatically an independent corroboration source. For example, two decompilers using shared lift/decoder dependencies may share failure modes; two reports from one mounted acquisition session remain one acquisition root. Unknown dependency ancestry stays unknown.

### Fair comparison arms

On the same held-out binaries, budgets and oracles, compare:

A. One specialist tool + expert/operator.
B. Multi-tool union without lineage or independent replay.
C. Multi-tool + artifact/operation binding and claim evaluation.
D. Full REA + Symthaea: source lineage, independent replay, proof-obligation gap detection and decision-aware next-observation selection.

Report tool-native output quality and the added REA layer separately. The goal is to outperform the best tool-only and agent-only baselines on source-grounded, reproducible conclusions and investigation efficiency—not to claim that REA itself disassembles better than Ghidra or IDA.

## 6.3 Additional benchmark tracks: behavior reconstruction and trajectory diagnosis

### Behavior2Code — black-box functional equivalence

Turing Frontier Research Lab's Behavior2Code (September 23, 2026) evaluates reconstructing command-line programs without source. Its strongest reported model passed 19 of 60 programs across three attempts (31.7% pass@3); 35 programs remained unsolved by all six tested models. Its protocol validates that the original executable passes every generated test, that requirements can be satisfied together, and that tested behavior is discoverable from documentation or execution. It highlights exit-status and error-boundary mismatches even when most normal outputs are correct. https://www.turing.com/blog/new-behavior2code-benchmark

REA alignment:
- Use functional-equivalence tasks as a separate behavioral-oracle track, not as REA claim entailment score.
- Bind every observed input/output/exit status/file effect to the exact original executable digest, invocation, environment and test vector.
- Reward only reproduction of declared behavior; partial test coverage remains a separate diagnostic.
- Include invalid input, boundary cases, error outputs, exit codes, filesystem side effects, determinism and dependency-isolation controls.
- The original executable is a behavioral oracle for its observed interface, not proof of source intent, safety or absence of untested behavior.

### AgentRx — first unrecoverable failure localization

Microsoft Research's AgentRx (March 12, 2026) reports a benchmark of 115 manually annotated failed agent trajectories across τ-bench, Flash and Magentic-One. The framework normalizes heterogeneous traces, synthesizes guarded executable constraints from tool schemas/domain policies, logs evidence-backed violations, then uses a judge to identify the first critical failure. The paper/blog reports +23.6% failure localization and +22.9% root-cause attribution over prompting baselines. https://www.microsoft.com/en-us/research/blog/systematic-debugging-for-ai-agents-introducing-the-agentrx-framework/

REA alignment:
- Add first-unrecoverable-step accuracy as a separate trace-quality metric.
- Require each failure annotation to point to the exact tool call, request/response, scope, and violated obligation/constraint.
- Separate deterministic executable constraint violations from judge-classified root causes; report both.
- Measure detection latency, missed earlier failures, false critical failures, root-cause category and evidence traceability.
- A later successful action must not erase an earlier irreversible/unauthorized/unsupported state transition.

### TRACE — dependency-aware evidence decomposition

TRACE (EMNLP 2026) treats multi-table claims as dependency-aware semantic trees: decompose claim nodes, retrieve local evidence for each node, and compose evidence bottom-up. It reports improvements over strong baselines across TabFact, OTT-FV and Hybrid-FV under gold- and open-evidence settings. https://aclanthology.org/2026.emnlp-main.968/

REA implication:
- Keep v0 as flat conjunctions for executable simplicity and adjudication.
- Design a future claim-local proof graph with typed AND/OR dependencies, explicit alternative admissible paths, shared-premise reuse, and defeater nodes.
- A proof graph is claim-local semantics, not a duplicate general evidence graph. Every leaf remains tied to immutable Evidence IDs and exact source spans.
- Score correctness of the decomposition separately from retrieval, extraction, warrant and source admissibility.

## 7. Industry boundary

REA composes rather than replaces:
- SRE-Bench and AgentRE-Bench for task-level binary reverse-engineering performance;
- SLSA, Sigstore, SCITT and TUF for their respective provenance, transparency, signing and update-security contracts;
- ForceBench, Ev2R and automated fact-checking work for evidence-force and retrieval evaluation;
- existing #6283 lineage/measurement principles and #10410 provider dependency graph.

The REA differentiator is not another source count or generic confidence score. It is the independently testable relation between a specific claim, its mandatory proof obligations, admissible observations, and the next authorized acquisition that could close the gap.

## 8. Integration sequence

1. Review schema and 18 synthetic cases.
2. Independently meta-validate Draft 2020-12 and validate fixture invariants.
3. Adjudicate a pilot and version the labels.
4. Implement a small scorer adapter using the existing benchmark receipt contract; no competing receipt format.
5. Add deterministic obligation-state transition tests.
6. Run all planner arms under equal budgets.
7. Add hidden source-family and omitted-obligation shifts.
8. Integrate into REA + Symthaea A/B/C evaluation only after exact-head qualification.

## 9. Additional research alignment — set sufficiency and evidence continuity

### Set-level sufficiency, not passage voting

Qiu et al., *SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation* (May 2026), frame support, refutation and insufficiency as three different outcomes. Their central point is that sufficiency is a property of the retrieved set: a missing required hop or unresolved conflict cannot be detected reliably by independently scoring each passage and taking a maximum or average. Their controlled HotpotQA-RAG v3 evaluation reports 0.9075 Macro-F1 for calibrated SURE-RAG and a reduction in unsafe-answer risk from 0.2588 to 0.1642 at 30% answer coverage. These are results on that paper's task and protocol—not REA measurements or a direct binary-RE comparison.

Reference: https://arxiv.org/abs/2605.03534

REA extension: obligations must be evaluated jointly. A claim with four required premises and three valid premises remains not ready, even when all three passages are individually relevant and supportive. Preserve the missing premise and contradiction state in the result rather than averaging local scores.

### Evidence-based generation and line-level grounding

Banerjee et al., *The Bottleneck Is Retrieval, Not Reading* (Findings of EMNLP 2026), introduce evidence-based generation with line-level citations and an Ans+Evid metric. They report that similarity retrieval finds at least one relevant chunk more reliably than all required chunks for multi-fact queries, and that chunking can destroy document-level context needed for identity-bearing questions.

Reference: https://aclanthology.org/2026.findings-emnlp.2222/

REA extension: every evidence item should retain its original byte/span and document identity. Extracted premises are derived representations that link back to source spans; they must not silently replace the source. Report answer correctness and evidence completeness separately, then combine only under a declared claim-readiness rule.

### Evidence continuity across changing targets

Deng et al., *SWE-Milestone: Evaluating AI Agents on Continuous Software Evolution* (ICML 2026), evaluate agents across sequential milestone DAGs rather than isolated tasks. Their paper reports performance above 80% on isolated tasks dropping to at most 38% in continuous settings across the evaluated models and agent frameworks. That is an external software-maintenance result, not a reverse-engineering or REA score.

Reference: https://proceedings.mlr.press/v306/deng26d.html

REA must extend the claim-obligation benchmark across target revisions and time. An earlier claim can become stale when the binary, deployment, dependency, policy, toolchain, advisory, or measurement authority changes. The benchmark should test selective invalidation:
- retain claims whose dependencies and scope remain unchanged;
- invalidate claims whose source or target dependencies changed;
- mark claims unresolved when the change impact is unknown;
- request only the revalidation needed to restore a claim's proof obligations;
- never treat a once-passing test or replay receipt as permanent evidence for a later target revision.

Compose with [#10630 — evidence continuity and selective revalidation](https://github.com/Luminous-Dynamics/symthaea/issues/10630), [#10760 — time-scoped and dependency-aware claim validity](https://github.com/Luminous-Dynamics/symthaea/issues/10760), and the existing benchmark run receipt owner. Do not create a parallel repository graph, temporal database, or receipt format.

### Resulting benchmark implication

The complete evaluation should therefore have three nested but separately reported levels:

1. **Evidence-set sufficiency:** Are all required premises present, admissible, unconflicted and correctly scoped?
2. **Acquisition policy quality:** Did the chosen experiment efficiently resolve the missing obligations under uncertainty and authorization constraints?
3. **Temporal continuity:** After the artifact or operating context changes, does the system correctly retain, invalidate or revalidate the old conclusion?

No single average should hide a false-ready event or stale claim. Publish false-ready rate, useful coverage, obligation closure cost, and stale-claim retention as separate release gates.

## Claim ceiling

This work defines a research protocol and synthetic fixtures. It does not implement an acquisition policy, authenticate source lineage, calibrate likelihoods, prove world-state truth, validate utility elicitation, or establish superiority over existing experiment-selection or evidence-retrieval systems.
