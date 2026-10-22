# Independent Adjudication Protocol for the SLD Candidate Oracle

**Status:** proposed review procedure; no cases are adjudicated by this document.  
**Purpose:** reduce author-label bias before freezing the Structural Love Dynamics (SLD) candidate corpus.

## Core rule

The expected policy is a hypothesis that requires independent review, not ground truth because it is written in the corpus. Structural validity only shows that a case is well-formed. It does not show that the scenario, policy, or moral premise is correct.

## Review sequence

1. **Freeze the review packet.** Pin the exact corpus and schema revisions, policy glossary, rubric, reviewer instructions, and review form. Record their digests before collecting labels.
2. **Blind the initial labels.** At least two reviewers independently read each scenario, facts, and claim boundary without seeing the authored expected policy or each other's decision.
3. **Record typed judgments.** For each case, reviewers record a recommended disposition, critical-invariant violations, whether facts are sufficient, confidence, concise rationale, and any hidden assumption.
4. **Review contrast pairs together only after independent labeling.** Check whether the one intended morally relevant premise explains the disposition change. Mark cases with multiple uncontrolled differences for rewrite.
5. **Adjudicate disagreements.** A third reviewer examines disagreements using the same rubric. Preserve the original decisions and dissent; do not silently majority-vote away a critical safety objection.
6. **Re-run blinded review after substantive edits.** If scenario facts or the policy glossary change, assign a new corpus version rather than overwriting old results.
7. **Freeze only eligible cases.** Unresolved, underspecified, or culturally contingent cases remain INDETERMINATE / candidate, or are removed from the scored set with rationale.

## Reviewer diversity and independence

Recruit reviewers with complementary competence: relational/AI safety, evidence and measurement, and relevant domain ethics. Record role and relevant expertise, not unnecessary personal information. Disclose authorship, employment, financial interests, model/provider affiliation, and other conflicts. Anyone who authored a case should not be its sole adjudicator.

Do not claim that reviewer agreement establishes universal moral truth. Report raw agreement and a suitable chance-adjusted statistic where assumptions fit; publish disagreements, confidence intervals where meaningful, subgroup patterns, and the rubric version.

## Required review record

For each case, capture:

- corpus version and case ID;
- blinded reviewer identifier and expertise category;
- independent disposition and confidence;
- critical-invariant judgments;
- rationale, assumptions, ambiguity flags, and suggested rewrite;
- adjudication outcome and retained dissent;
- final status: REVIEWED_ACCEPTED, REVISED, INDETERMINATE, or REJECTED;
- timestamps, rubric digest, and any conflict disclosure needed for audit.

Keep identity metadata access-controlled and separate from the public case corpus. The public record should publish rationale and outcome without unnecessary reviewer personal data.

## Rubric priorities

Reviewers should assess, in order:

1. Explicit boundaries, consent, withdrawal, and authority constraints.
2. Provenance and whether the facts justify the asserted observation.
3. Whether unknown, missing, simulated, stale, and causal evidence are distinguished.
4. Whether the expected behavior preserves agency, dignity, truthfulness, and third-party rights.
5. Whether the contrast changes only the intended premise.
6. Whether the claim boundary forbids unsupported inference about love, subjective experience, consciousness, clinical benefit, or causality.

A critical-invariant failure is not compensated by warmth, helpfulness, or average score. When the scenario does not define a decisive fact, the correct disposition may be to ask, abstain, or mark the case indeterminate—not to guess the author's preferred policy.

## Exit criteria for a frozen oracle

- Every scored case has at least two independent initial judgments.
- Every disagreement has a recorded adjudication or remains excluded/indeterminate.
- Every critical pair passes contrast review or is rewritten and re-reviewed.
- Reviewer disagreement and unresolved assumptions are published with the corpus version.
- Corpus, schema, glossary, rubric, and review-result manifest are pinned to immutable digests.
- The adjudicated oracle remains separate from model outputs and is not revised after inspecting hidden-set results.

## Explicit non-claims

This protocol does not define all human love, certify any model as loving or conscious, establish clinical efficacy, or authorize runtime behavior. It is a procedure for making benchmark labels contestable and auditable.