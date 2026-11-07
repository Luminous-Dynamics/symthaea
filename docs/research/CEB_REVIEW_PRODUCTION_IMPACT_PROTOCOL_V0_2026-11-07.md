# CEB Review Production-Impact Protocol V0 — 2026-11-07

**Status:** Research protocol and preregistration template. No production experiment, user-level randomization, human outcome study or CEB benchmark run is claimed.  
**Owners:** Existing CEB-REVIEW / CEB-HUMAN / CEB-CONTINUITY work; [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723), [#10637](https://github.com/Luminous-Dynamics/symthaea/issues/10637), [#10630](https://github.com/Luminous-Dynamics/symthaea/issues/10630), [#10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419).  
**Composition:** [CEB Review Benchmark Triangulation V0](CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md), [Finding Adjudication Card V0](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md), [Causal-Diagnosis Preregistration V0](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md), existing CEB-HUMAN and CEB-CONTINUITY contracts.  
**Metric boundary:** Keep the CEB measurement matrix at 14 lanes / 56 primary metrics. Do not add an uptake score, deployment-impact lane, opaque composite or new receipt schema. No uptake score is added; preserve the existing three CEB-REVIEW primary measures.

## 1. Research question: does a correct review improve engineering outcomes?

A code-review agent can produce a factually correct finding that is not acted on. A developer can accept an incorrect finding. A correct suggestion can be rejected because it is out of scope, too late, redundant, unaffordable or already fixed independently. A later code change can coincide with a review comment without being caused by it.

Therefore four states must remain distinct:

1. **Truth:** independently adjudicated factual/causal correctness on the exact reviewed revision.
2. **Utility:** whether the finding is specific, relevant, actionable and justified under the frozen scope.
3. **Uptake:** what the developer did with it (accept, adapt, reject, defer, discuss, independently reproduce, or no observable response).
4. **Outcome:** what happened to the code/product after the interaction (target mechanism removed, regression introduced, issue recurred, protected invariant changed, rework performed).

The production impact question is not “what percentage of comments became code?” It is:

> For findings whose truth and actionability are independently known, what is the causal effect of a review presentation/intervention on correct developer action, verified remediation, downstream regressions and human cost?

The treatment under study may be presentation, prioritization, evidence packaging or escalation policy. It is not permission to change the truth label after uptake is observed.

## 2. Why this matters — current external evidence

### 2.1 ReviewBench: offline quality can be tested against production behavior

GitHub's ReviewBench methodology explicitly checks whether offline score movement anticipates online experiments. The company reports an online A/B test in which addressed rate rose 8.0%, recall rose 13.6%, comment volume rose 61% and cost per review fell 8.0% relative to production control. It also reports an online increase in critical comments. This is useful evidence that offline review evaluation can be production-relevant, but it is GitHub's own product experiment; it does not prove that every addressed comment is correct or that all teams will see the same effects. [ReviewBench methodology](https://github.blog/ai-and-ml/reviewbench-an-open-benchmark-for-ai-code-review/).

### 2.2 Kodus: implementation and unresolved flags are not correctness

Kodus's State of AI Code Review 2026 analyzes its platform's production pipeline across a window beginning 2025-09-01 and ending at a 2026-06-08 snapshot. It reports 180,739 suggestions, with 33% “implemented/adapted” overall; its time series rises from 25% to 48% by PR-created month, with recent months right-censored. Bugs have a reported 34.5% implementation/adaptation rate; security findings 31.2%. Its AI-vs-human comparison reports differing PR size, finding volume and implementation rates. It further reports that 71.8% of 13,609 flagged merged PRs had at least one flag still open; the report explicitly says this folds in false positives and accepted risk. These are descriptive platform analytics, not randomized treatment effects or independent truth audits. [Kodus State of AI Code Review 2026](https://kodus.io/data) (window/methodology described in the report).

The report itself notes that balanced-panel comparisons change the interpretation of platform-wide PR-size and merge-time trends. This is a useful warning against compositional confounding: adoption growth and aggregate time trends can reflect which teams entered the sample, not a behavior change within the same teams.

### 2.3 Martian: follow-up timelines measure workflow behavior

Martian's production review benchmark analyzes bot suggestions alongside developer responses, follow-up code changes and resolved conversations. Its November 7 dashboard view lists 17,768 scored PRs and reports per-system samples with different denominators. This makes it useful for measuring production adoption patterns, but an LLM-derived match between a comment and a later diff is not proof that the comment caused the change or that the change was correct. [Martian Code Review Benchmark](https://www.codereviewbenchmark.com/) · [alternate dashboard](https://codereview.withmartian.com/).

### 2.4 The measurement gap

ReviewBench supplies a fixed-gold quality framework and a production experiment; Kodus provides longitudinal implementation/follow-up analytics; Martian supplies review timelines and adoption signals. None alone identifies the causal effect of a review finding on verified downstream quality across all teams. CEB should connect these dimensions without collapsing them.

## 3. Primary estimand: intention-to-treat effect of a safe review presentation

The recommended first production study is a **cluster-randomized presentation experiment** for eligible low-risk findings only.

- **Control:** existing standard review presentation.
- **Treatment:** same finding and same severity, but with a structured evidence/actionability presentation: concise causal explanation, exact source location/revision, confidence/uncertainty, applicable contract, suggested minimal next action and explicit unresolved evidence.
- **Optional second treatment:** a bundle/digest presentation for non-urgent findings, compared with standard inline presentation. This is a different treatment and requires a separate preregistered contrast.

Do not randomize away or delay critical/security/safety warnings, legal/compliance blockers, confirmed high-severity vulnerabilities, active incidents, or findings subject to a mandatory team policy. These remain on the existing immediate policy path in every arm. Do not randomize access to the underlying source evidence or required approval.

### 3.1 Eligibility and unit of assignment

- Eligible units: low/medium-risk, non-urgent review findings where both presentation options satisfy the team's existing policy and response SLA.
- Assignment unit: preferably PR or repository-team-time block, not individual comment, when several findings in a PR can influence each other.
- Stratify randomization by repository/team, severity, defect category, PR size, language and AI-authored versus human-authored code where the sample supports it.
- Randomize within a predeclared time block or use a switchback only if carryover can be bounded and logged.
- Freeze agent, model, harness, findings, truth labels, scorecard and tool budget before treatment assignment.
- Keep all findings in the denominator, including no-response, timeout, unavailable, rejected and deferred states.

The PR/team cluster avoids treating comments from the same review as independent and reduces spillover from one comment changing how the developer treats another.

### 3.2 Blinding and separation of truth from treatment

Before exposure to the assigned presentation:
1. Two independent reviewers adjudicate each finding's factual/causal truth, actionability, severity, exact-revision scope and uncertainty using the existing finding card.
2. The adjudicators do not see treatment assignment, developer response or downstream implementation.
3. A third expert resolves disagreement under the existing policy; unresolved stays unresolved.
4. Freeze the truth/actionability label and the source revision. Do not rewrite it because the developer accepted or rejected the finding.

The engineering team and agent may know the presentation, but primary adjudicators must remain blind to uptake. Outcome adjudicators must be blind to assignment where feasible.

## 4. Event taxonomy: preserve the complete response lifecycle

Record each finding's lifecycle as separate observed events, not one “accepted” boolean:

- surfaced / delivered;
- viewed or opened, only when the product exposes consented, policy-compliant telemetry;
- explicitly accepted;
- accepted with modification;
- rejected with reason;
- deferred with reason or requested review window;
- discussed / clarified;
- independently reproduced by developer;
- linked to an implementation commit;
- linked by the developer as the reason for the change;
- closed as duplicate / already fixed / out of scope;
- no observable response by the end of the observation window.

No-response is not rejection. A resolved GitHub thread is not necessarily a correct finding. A code change after a comment is not necessarily a response to that comment. A statement that a comment caused a change is useful provenance but remains self-report, not ground truth.

For every linked code change, retain exact PR/commit/tree and change interval references under the existing repository/evidence contracts. Do not expose private source content beyond the user's authorization.

## 5. Outcome hierarchy and adjudication

### 5.1 Uptake outcome (process)

Describe the observed event distribution by truth and utility stratum:
- accepted/adapted;
- explicitly rejected;
- deferred;
- discussed/clarified;
- independently reproduced;
- linked code change;
- no observable response.

Report all eligible findings and PRs, not only comments that received a reply. Report both cluster-level and finding-level counts, but inference must cluster by assignment unit.

### 5.2 Verified remediation (engineering correctness)

A code change counts as a verified remediation only when an independent process establishes:
- it addresses the same target mechanism;
- the target mechanism is absent in the exact resulting revision;
- the fix preserves declared protected invariants and valid neighboring behavior;
- required regression checks pass under the applicable qualified environment;
- the result survives the existing independent regrade requirements.

Before #10064 qualifies candidate-controlled execution, this section may only use source-only/manual evidence and historical, already-qualified records. Do not run candidate-controlled tests, builds or production code as part of this design.

### 5.3 Downstream outcome

Where available and privacy/consent permit, observe:
- target recurrence or re-opened incident;
- regressions or new vulnerabilities introduced by the response;
- rollback/revert/hotfix;
- subsequent human rework and review burden;
- protected invariant changes;
- evidence continuity across later revisions;
- time to verified remediation, not merely time to merge.

These are downstream contextual outcomes mapped to existing CEB-CONTINUITY and CEB-HUMAN ownership. Do not create a new primary metric lane.

## 6. Estimation and analysis

### 6.1 Primary causal contrast

For the safe eligible stratum, estimate the intention-to-treat difference between treatment and control at the PR/team assignment unit. ITT preserves the randomized assignment and avoids comparing only users who chose to open or accept a comment.

Predeclare one primary process endpoint only if the study genuinely needs one (for example, correct resolution of independently adjudicated actionable low/medium findings within the agreed SLA). Keep verified remediation, regressions and reviewer burden as separate key outcomes. Do not make the study's primary result “acceptance rate” without truth stratification.

### 6.2 Secondary and diagnostic analysis

Report separately:
- uptake by blinded truth/actionability stratum;
- implementation/adaptation conditional on truth, utility and severity;
- verified remediation among all assigned PRs and among truth-positive eligible findings;
- incorrect acceptance of false findings;
- correct findings explicitly rejected/deferred;
- unresolved/no-response rate;
- time to verified remediation;
- downstream recurrence/regression/rework;
- human minutes and model/tool/judge costs.

An IV/complier effect may be estimated from assignment only if the instrument's exclusion restriction, monotonicity and interference assumptions are defensible and preregistered. Do not use assignment as an instrument simply because exposure is noncompliant. ITT remains the primary randomized estimate.

### 6.3 Statistical safeguards

- Analyze at the assignment cluster (PR/team/time block), with clustered confidence intervals or a predeclared hierarchical model.
- Report exact denominators, assignment balance, attrition, missingness and censoring.
- Use fixed follow-up horizons; distinguish right-censored events from true no-response.
- Preregister strata, exclusion rules, minimum sample-size/power method, stopping rules and multiple-comparison correction.
- Preserve all randomized units; do not drop difficult or unresolved findings after assignment.
- Report both effect size and uncertainty. A null or harmful treatment effect is a valid result.
- Measure team composition over time; do not infer within-team behavioral improvements from platform-wide compositional changes.
- Audit spillovers: a new review style may affect developer behavior on control PRs in the same team.

## 7. Safety, ethics and product boundaries

- Never hide, delay or downgrade critical, security, privacy, safety, compliance or active-incident findings for experimental purposes.
- No covert employee scoring, individual productivity ranking, sentiment inference or preference profiling.
- Use opt-in or organization-approved research controls, minimum necessary data and retention limits.
- Respect existing user/team notification preferences; no-response is not permission to adapt preferences.
- Preserve correction, appeal and explanation paths for findings and labels.
- The experiment changes presentation of eligible low-risk findings, not whether a real defect is disclosed.
- If lawful consent, organizational approval, adequate power or a defensible safety boundary is absent, use retrospective descriptive analysis only and explicitly make no causal claim.

## 8. Metric mapping — reuse the CEB portfolio

| Evidence | Existing CEB owner | Interpretation |
|---|---|---|
| Correct target findings among top five; actionable precision | CEB-REVIEW | Truth and utility are adjudicated before uptake |
| Human adjudication time / review effort | CEB-HUMAN / reviewer_burden | Include rework and second-review effort |
| Correct handoff, preference and process events | CEB-COLLAB / CEB-HUMAN | Uptake is a process event, not truth |
| Repair survival / stale evidence / sequence validity | CEB-CONTINUITY | Downstream outcome over subsequent revisions |
| Causal forecast and intervention evidence | CEB-CAUSE | Distinguish observed outcome from attribution hypothesis |
| Claims and experiment validity | CEB-EVAL / #10619/#10624 | Preserve assignment, denominator and uncertainty |

No uptake score is added to the primary metric matrix. If a new contextual observation is valuable, register it under the appropriate existing lane with an explicit unit and denominator, not as a new metric.

## 9. Prospective success criteria

The existing CEB-REVIEW targets remain unchanged: Recall@5 ≥0.85, actionable precision ≥0.80, and reviewer burden ≥20% lower at non-inferior recall.

For this production-impact research, do not set a universal “good adoption rate.” A treatment is promising only if:
- it improves a predeclared, truth-stratified correct-action/remediation outcome at acceptable uncertainty;
- it does not increase false acceptance or high-severity misses;
- verified remediation and later survival do not regress;
- human burden and all-in cost are measured;
- effects are not explained by team composition or selective missingness.

A higher acceptance rate with no improvement in verified remediation—or with more incorrect acceptance—is not a success. A lower uptake rate can be beneficial if fewer false findings consume attention and true defects still receive timely action.

## 10. Runner-independent work sequence

1. Map the lifecycle fields to the existing #10637 collaboration and #10630 continuity records; do not create a new event or receipt schema.
2. Add synthetic examples for correct-but-rejected, false-but-accepted, accepted-but-not-implemented, independently fixed, and implemented-but-regressed.
3. Specify the blinded pre-uptake adjudication and separate downstream outcome adjudication.
4. Draft the low-risk eligibility policy and immutable critical-finding exclusion.
5. Complete retrospective descriptive analysis only after independent label quality and consent/privacy review; do not claim causal effect.
6. Seek a randomized presentation trial only with organizational authorization, safe eligible treatment options, preregistered power and independent oversight.

## 11. Claim boundary

This protocol does not claim Kodus or Martian adoption statistics are causal, does not claim accepted suggestions are correct, and does not claim post-review diffs were caused by a bot comment. It defines a possible CEB production experiment. No human/user data was collected, no treatment was randomized, no model was evaluated, and no new metric, task schema or receipt was created. Candidate-controlled execution and independent regrade remain subject to #10064 and #10369.
