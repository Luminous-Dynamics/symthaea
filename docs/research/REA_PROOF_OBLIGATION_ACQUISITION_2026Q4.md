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
