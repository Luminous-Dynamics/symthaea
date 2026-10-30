# Prospective proofreading workflow trial protocol V1

**Status:** preregistration template; no trial has been enrolled or run.
**Tracking:** [SCI-CONNECTOMICS-008 / #9242](https://github.com/Luminous-Dynamics/symthaea/issues/9242).
**Generator workstream:** [SCI-CONNECTOMICS-007 / #9241](https://github.com/Luminous-Dynamics/symthaea/issues/9241).

## 1. Causal question

Does the frozen morphology-conditioned candidate-generation and ranking workflow reduce independently adjudicated scientific-query error more efficiently than the current workflow, measured per actual expert person-minute, without increasing unsafe corrections or missed high-impact errors?

The retrospective six-policy replay in #9134 is not causal. The candidate-pool probability audit in #9187 and the independent search-frame audit in #9204 estimate different finite-frame risks; neither identifies the causal effect of assigning a reviewer to a workflow.

## 2. Initial two-arm design

- **Control:** frozen current generator plus frozen current ranking policy.
- **Treatment:** frozen morphology-conditioned generator plus the selected ranking policy.
- Keep user interface, evidence presentation, reviewer permissions, and correction safety controls identical across arms.
- Do not bundle an interface redesign into this first comparison. A later factorial design may separate generator and ranker effects if power and operational capacity support four arms.

## 3. Randomization and blocking

The randomization unit should be a non-overlapping spatial work block, or a neuron cluster where spatial partitioning is scientifically invalid. A neuron, source operation, or overlapping EM region must never be assigned to both arms.

Block on reviewer, source volume / region, species, and pretrial workload difficulty where feasible. Use concealed permuted-block assignment. Freeze eligibility and both workflow versions before randomization. Commit the assignment-seed hash and protocol digest to a trusted preregistration record before assignments or outcomes are visible.

Prefer parallel reviewer blocks with balanced workloads over an unstructured crossover. If a reviewer must cross arms, use disjoint neurons / regions, counterbalance order, model period effects, and retain initial assignment for intention-to-treat analysis. Reviewers may be blinded to arm labels and model confidence where operationally possible, but cannot be blinded to evidence differences that are inherent to the treatment.

## 4. Endpoints

**Primary endpoint:** independently adjudicated signed reduction in error for a preregistered bundle of scientific queries per actual expert person-minute, analyzed by randomized block under intention-to-treat.

Key safety and coverage endpoints:
- false accepted-correction rate and applied-correction precision;
- missed high-impact errors from an independent probability sample of the declared search frame (#9204);
- candidate-generation recall and impact-weighted recall by preregistered error family;
- candidate-pool residual risk from an independent sample of selected and unselected proposals (#9187);
- stale/conflicting proposal rate, reversals, rework, abstention, and escalation;
- actual reviewer time for navigation, evidence inspection, deliberation, and correction.

Secondary endpoints include verified error yield per minute, time to a fixed query-error target, false proposals per neuron, context switches / cognitive load measured with a preregistered instrument, compute, feature extraction, storage, data egress, and independent adjudication costs.

## 5. Independent outcome ascertainment

Adjudicators are blind to assignment, policy ranking, and model score. Preserve reviewer accept / reject / abstain separately from adjudication correct / incorrect / indeterminate.

Maintain two independent audit streams:
- #9187 samples candidates inside the frozen candidate pool, including candidates not selected by the assigned queue;
- #9204 samples outside the candidate pool from the declared spatial/neuron frame, including PSUs with zero proposals.

All sample seeds, sampled locations, adjudication labels, error identities, and impact weights remain evaluator-only until lockbox closure. They must never feed either workflow arm. Report independent audit search time and person-minutes separately rather than concealing them in reviewer-minute denominators.

## 6. Statistical analysis

Before enrollment, register the estimand, query bundle and weights, minimum scientifically relevant effect, exclusions, nonresponse / missingness rules, analysis code, multiplicity handling, and stopping rules.

Use intention-to-treat as the primary analysis. Confirm with block-level randomization inference and report cluster-robust estimates at the true randomization unit. Do not treat candidates, crops, or edges within a neuron/spatial block as independent observations. Use paired contrasts only where the randomization actually creates matched blocks.

Estimate sample size from pilot block-level variance and the smallest scientifically relevant effect; do not choose it from a desired headline threshold after observing outcomes. Do not stop early for efficacy through repeated unblinded peeking. Permit only preregistered safety stops for harmful corrections, data-integrity failures, or unacceptable reviewer burden.

Report absolute and relative effects, design-aware uncertainty intervals, deviations, allocation violations, missing data, protocol changes, and all-arm all-in costs.

## 7. Safety and promotion

Every suggestion remains review-only. A human accepts each correction; an independent validator checks it; application occurs only in a reversible sandbox or versioned annotation branch. Candidate generation and ranking never receive production write credentials.

A failed integrity, coverage, critical-family, or false-correction gate blocks promotion regardless of pooled efficiency. Promote only when the treatment improves the primary endpoint or reduces expert time at matched independently measured quality and coverage, with no breach of safety limits. A point estimate alone is insufficient.

Do not describe retrospective replay as causal evidence. Do not describe candidate-pool recall as whole-connectome recall. No trial result is claimed by this protocol.
