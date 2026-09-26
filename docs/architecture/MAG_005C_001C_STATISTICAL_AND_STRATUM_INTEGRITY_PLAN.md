# MAG-005C-001C — Exact finite-sample inference and stratum-integrity plan

Status: planning-only follow-on; do not implement from unqualified MAG-005C-001B

Date: 2026-09-26

Related:

- MAG-005 #4485
- MAG-005C-000A PR #6063
- MAG-005C-001A PR #6064, repaired head `f01c8e2112061734bf693f6b96cbe391869dabcb`
- MAG-005C-001B PR #6066, head `9d0d0fe14ed64178a990b5d47310c3db529d7739`

## Purpose

Freeze the next scorecard-extension requirements without mutating or stacking executable work on the still-unqualified MAG-005C-001B reference subject.

Two gaps remain after the first 24-case corpus:

1. a positive synthetic case where a predeclared finite-sample control-difference test actually resolves under its own protocol;
2. explicit primary-stratum integrity so candidates cannot be double-counted across top-K and control arms.

This document is planning evidence only. It creates no statistical or scientific PASS.

## Statistical authority separation

Preserve:

```text
observed point-estimate enrichment
!= finite-sample evidence against a frozen null
!= causal effect
!= universal strategy superiority
!= scientific discovery
```

`control_difference_established` must therefore mean only:

> the exact preregistered statistical procedure, applied to the exact frozen terminal census and metric profile, crossed its preregistered decision threshold.

It must not mean that the acquisition strategy is universally superior or causally responsible for the difference.

## First exact reference profile

For the first simple binary hit/miss top-K-vs-random comparison, use a predeclared exact 2x2 Fisher test profile with:

- exact two-sided alternative by default;
- exact integer cell counts;
- fixed alpha declared before outcomes;
- no post-outcome choice of one-sided/two-sided alternative;
- no post-outcome threshold selection;
- exact test-method/version identity;
- raw contingency table retained;
- point estimates and effect sizes retained separately from the test decision.

Reference implementation should be possible using Python stdlib integer combinatorics only; production code need not depend on SciPy.

A positive synthetic known-answer candidate is:

```text
top-K:   16 hit / 4 non-hit
random:   4 hit / 16 non-hit
```

Under the conventional two-sided Fisher definition that sums fixed-margin tables with probability less than or equal to the observed-table probability, the expected p-value is approximately:

`0.0003599673667865379`

This fixture is synthetic only and exists to prove that the `control_difference_established` vector can become true under a preregistered exact profile.

## Finite-sample result object

The future scorecard should keep at least:

- exact 2x2 cell counts;
- primary metric identity;
- test-profile identity;
- alternative;
- alpha;
- exact/derived p-value;
- top-K precision;
- control precision;
- absolute risk difference;
- risk ratio when defined;
- odds ratio when defined;
- interval method/profile when an interval is reported;
- multiplicity family/profile;
- terminal-census identity;
- decision state.

Suggested decision states:

```text
ControlDifferenceEstablishedUnderProfile
ControlDifferenceNotEstablishedUnderProfile
ControlDifferenceTestInapplicable
ControlDifferenceEvidenceInvalid
```

Do not expose only a boolean without retaining the test profile and counts.

## Multiplicity boundary

A campaign may preregister many:

- hit thresholds;
- properties;
- control arms;
- subgroup slices;
- time points;
- secondary endpoints.

Therefore:

```text
one nominal p < alpha somewhere
!= preregistered primary control difference
```

The commitment must identify primary tests before outcomes. If more than one inferential test belongs to the primary family, the multiplicity policy must also be preregistered. A later implementation may support a simple auditable correction such as Holm-Bonferroni, but MAG-005C must not invent or select a correction after observing results.

Exploratory secondary analyses remain reportable as exploratory.

## Primary scoring-stratum integrity

Every committed evaluation candidate should have exactly one primary scoring stratum for denominator accounting.

Examples:

```text
TopK
RandomControl
DiversityControl
OODControl
ConventionalBaseline
```

Secondary descriptive tags may overlap:

```text
OOD
chemistry-family-X
prototype-Y
high-uncertainty
diversity-frontier
```

but secondary tags do not create additional scoring slots.

Core theorem:

```text
one committed candidate identity
-> exactly one primary scoring slot
+ zero or more diagnostic tags
```

not:

```text
one candidate
-> counted once in TopK
+ counted again in OOD control
+ counted again in diversity control
```

unless a future protocol explicitly defines paired/repeated evaluation and uses a different estimand.

## Allocation integrity requirements

Before outcomes exist, validate:

- every primary-stratum member is inside the committed candidate universe;
- no candidate appears more than once within a primary stratum;
- primary strata are pairwise disjoint;
- exact expected stratum sizes match actual committed memberships;
- control-allocation seed/algorithm reproduces the committed controls;
- canonical candidate identity, not presentation label, is used for collision checks;
- aliases known before commitment are resolved before allocation;
- aliases discovered after commitment receive terminal historical treatment rather than silent reassignment;
- no candidate can be replaced after reveal while preserving the original denominator claim.

## Required next hostile fixtures

The next synthetic generation should add at minimum:

1. statistically resolving 16/20 vs 4/20 positive control-difference case;
2. same counts but alpha changed after reveal -> new lineage, no rewritten decision;
3. same counts but one-sided alternative selected after reveal -> new lineage;
4. multiple primary endpoints with no preregistered multiplicity profile -> inferential decision invalid;
5. candidate appears in both TopK and RandomControl -> score-input refusal;
6. candidate appears twice inside TopK -> score-input refusal;
7. two labels resolve to the same canonical candidate before commitment -> allocation refusal until deduplicated;
8. alias discovered only after commitment -> preserve historical slot/disposition, no silent denominator repair;
9. OOD diagnostic tag overlaps TopK -> allowed because tag is not a primary stratum;
10. expected stratum size 20 but only 19 committed unique candidates -> allocation invalid;
11. random-control seed reproduces a different set than the stored membership -> allocation invalid;
12. p-value stored in fixture disagrees with exact recomputation -> reject stored summary;
13. significant exact test with contaminated Phase-A answer access -> no prospective credit despite the p-value;
14. significant exact test with incomplete terminal census -> evidence invalid;
15. significant scorecard result -> no synthesis, novelty, physical validation, manufacturing, or economic promotion.

## Suggested implementation sequence

Only after MAG-005C-001B exact workflow qualification succeeds:

```text
MAG-005C-001C
  synthetic corpus v2 extension
  + exact positive inference fixture
  + stratum-integrity hostile cases

MAG-005C-001D
  independently extend stdlib reference scorer
  + exact Fisher calculation
  + allocation collision checks
  + multiplicity-profile refusal checks
```

Do not rewrite the already frozen v1 corpus after qualification. Extend by a new generation/lineage.

## External motivation boundary

Current materials-discovery benchmarking emphasizes prospective evaluation rather than relying only on retrospective splits, and reliability-focused materials ML work emphasizes held-out calibration, contingency-style evaluation, fixed budgets, reproducible splits/seeds, and explicit negative results. These motivate the architecture but do not become local Symthaea qualification evidence.

## Claim ceiling

This planning subject may define how a future synthetic/reference scorecard should represent exact finite-sample evidence and allocation integrity. It does not establish a statistical difference in any real materials campaign, validate any MLIP or DFT result, establish scientific discovery, or authorize physical experimentation.
