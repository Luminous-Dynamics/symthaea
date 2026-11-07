# CEB Review Finding Adjudication Card V0 — 2026-11-07

**Purpose:** Human annotation aid for existing CEB-REVIEW tasks and the benchmark-triangulation research.  
**Status:** Public design-only template. It is not a new task schema, receipt, evaluator, label authority or primary metric.  
**Use with:** [CEB Review Benchmark Triangulation V0](CEB_REVIEW_BENCHMARK_TRIANGULATION_V0_2026-11-07.md), [Review Evidence-Grounding Challenge V0](CEB_REVIEW_EVIDENCE_GROUNDING_CHALLENGE_V0_2026-11-06.md), [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723).

---

## A. Finding identity and scope

- Evaluation/task reference:
- Candidate/run reference:
- Finding ordinal in submitted review:
- Exact submitted finding text:
- File/path and line span cited:
- Reviewed source revision/tree digest:
- Task specification/requirement revision:
- Review scope (PR, changed files, dependency/configuration world):
- Was the finding within the top five findings? **yes / no / not applicable**
- Is this a known target defect in the frozen task? **yes / no / unresolved**
- If not the target, does it warrant independent new-defect adjudication? **yes / no / unresolved**

Do not silently substitute the current branch head for the reviewed source revision. Do not infer a task/target identity from a finding's wording.

## B. Stage 1 — Discovery

**Question:** Does the finding identify a concrete behavior, requirement violation or risk worth adjudicating?

- Discovery result: **candidate issue / no issue / duplicate / out of scope / unresolved**
- Observable claim:
- Alleged consequence/impact:
- Minimum requirement or invariant needed to judge it:
- Is the claim falsifiable from the supplied evidence? **yes / no / partially**

A warning that says only “this looks risky” is a lead, not a confirmed defect. A useful design suggestion may be valid even when it is not a defect.

## C. Stage 2 — Localization

**Question:** Does the cited region participate in the relevant execution/configuration/data/control path?

- Localization result: **direct / contributing / nearby-but-not-causal / unrelated / unresolved**
- Relevant symbol(s)/path(s):
- Caller/callee or dependency edges:
- Configuration/feature/platform conditions:
- Source evidence anchors:

**Rule:** Correct file or line overlap is not sufficient for a target-defect hit. SecPRBench reports that many misses locate the root-cause file but describe the wrong mechanism.

## D. Stage 3 — Causal mechanism

**Question:** Does the finding explain the actual mechanism rather than a neighboring, generic or merely plausible failure?

Complete each link, citing source/evidence or explicitly marking it unresolved:

1. **Trigger/precondition:** What input, event, state or configuration activates the behavior?
2. **Path/state transition:** Which caller, data/control path or state transition carries the trigger?
3. **Operation:** What exact operation becomes unsafe, incorrect or non-compliant?
4. **Violated condition:** Which frozen requirement, invariant, security boundary or contract is violated?
5. **Consequence:** What observable failure or risk follows under the stated conditions?

- Mechanism result: **correct target mechanism / correct non-target mechanism / neighboring mechanism / contradicted / incomplete / unresolved**
- What alternative mechanism was considered?
- What counterexample or changed precondition would make this diagnosis false?
- What evidence would distinguish the proposed mechanism from the nearest alternative?

**Target-hit rule:** credit a known target only when the explanation matches the target's causal failure and that failure is live in the exact reviewed revision. Location overlap, vulnerability-family labels and semantic similarity alone do not qualify.

## E. Stage 4 — Evidence authority and freshness

Use the existing six CEB evidence classes; do not create a new authority system.

| Class | Mark only when the evidence actually supports it |
|---|---|
| source_declared | Literal source/configuration/requirement content |
| syntax_derived | Parser-derived structure without semantic resolution |
| compiler_resolved | Pinned compiler/language-server fact for the exact configuration |
| runtime_observed | Measured execution with command, target, environment and output |
| intervention_observed | Controlled intervention with declared assignment and control strategy |
| inferred / unresolved | Candidate reasoning or insufficient/conflicting evidence |

- Strongest evidence class actually established:
- Exact source/configuration digest:
- Run/receipt reference, if any:
- Command and target, if relevant:
- Environment/toolchain, if relevant:
- Freshness: **current / stale / superseded / not bound / unresolved**
- Missing, skipped, timeout or infrastructure-error state:
- Does the evidence actually discriminate the proposed cause from an alternative? **yes / no / partially**

**No authority upgrade:** source comments do not prove runtime behavior; a passing compile does not prove tests passed; stale CI does not validate a new head; a nonzero exit caused by missing infrastructure is not automatically a candidate defect. Candidate-controlled execution stays blocked until #10064 is qualified.

## F. Stage 5 — Decision utility and disposition

- Finding utility: **actionable defect / valid non-target suggestion / useful context / duplicate / unsupported / noise / out of scope / unresolved**
- Severity under frozen rubric:
- Is the finding specific enough for a maintainer to act?
- Does it identify a minimal corrective direction without prescribing an unsupported rewrite?
- Is the severity justified by the demonstrated impact?
- Would a reviewer need additional evidence or clarification?
- Estimated human adjudication minutes:
- Does the finding repeat an already reported issue?
- Did the finding cause a code change, follow-up investigation or decision? (Record only when observed; do not infer intent from silence.)

A useful non-target finding may contribute to comment usefulness/actionable precision after independent review, but it does not count as a hit on the task's labeled target defect. An unverified candidate finding is not gold truth.

## G. Final independent adjudication

- **Known-target disposition:** hit / miss / unresolved / not a target task
- **Finding disposition:** actionable / non-actionable / unresolved
- **Causal diagnosis:** correct / wrong / incomplete / unresolved
- **Evidence claim:** supported / overclaimed / stale / missing / unresolved
- **Duplicate relation:** unique / duplicates finding ___ / unresolved
- Reviewer ID/role:
- Reviewer model/tool disclosure (if applicable):
- Blind to other adjudicators? **yes / no**
- Rationale and evidence references:
- Dissent/uncertainty:

A second reviewer must adjudicate independently before viewing prior labels. Resolve disagreements by evidence and frozen criteria, not by majority vote alone. Preserve unresolved findings rather than forcing a binary match.

## H. Ground-truth and evaluator audit

Complete only for the evaluator/benchmark maintainer, not the candidate reviewer:

- Original target provenance:
- Fixing change and source-revision binding:
- Target label validation: **mechanical / human / mixed / incomplete**
- Finding-to-defect matcher version:
- Judge/model/prompt version:
- Known model-family or source-lineage dependence:
- Human-audit sample stratum: **accepted / rejected / unresolved / severity / model-family**
- Human-audit outcome:
- Matcher false match / missed match / unresolved:
- Does this evidence change the frozen target label? **no / propose review / re-adjudicated**

Do not use the model being evaluated as the only truth authority. Report judge versions and per-judge disagreements. A fixed companion only checks whether the specific target mechanism remains live; it does not establish that the whole PR is defect-free.

---

## I. Mapping to existing CEB primary measures

This card does not create metrics. It supplies richer adjudication evidence for:

- **defect_recall_at_5:** only a correct target-mechanism hit among the top five findings counts.
- **actionable_precision:** findings adjudicated as actionable, including independently verified non-target defects when within scope; keep the original benchmark's target-hit denominator separate.
- **reviewer_burden:** capture the human effort to read, reconstruct, investigate, dispute and disposition findings, including second reviewers and adjudication.

Report additional diagnostic annotations by category and severity, but do not combine them into an opaque score. The registered CEB metric matrix remains authoritative.

## J. Quick adjudicator consistency check

Before finalizing a label, confirm:

- [ ] Exact reviewed revision and requirements are known.
- [ ] A correct line citation has not been mistaken for a correct causal mechanism.
- [ ] The target defect has not been conflated with a useful non-target finding.
- [ ] Comments/PR narrative are not being treated as stronger evidence than source/contract facts.
- [ ] Stale, skipped, timeout and infrastructure-error states remain distinct.
- [ ] A fixed companion is interpreted only as a target-specific control.
- [ ] Candidate-controlled execution was not performed without #10064.
- [ ] The judge/matcher is not treated as independent truth without audit.
- [ ] Reviewer time and total model/tool cost are retained.
- [ ] Unresolved cases remain unresolved when evidence is insufficient.
