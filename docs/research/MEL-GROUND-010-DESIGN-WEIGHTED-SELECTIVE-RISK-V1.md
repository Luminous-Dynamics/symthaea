# MEL-GROUND-010 — Design-weighted selective temporal-grounding risk/coverage v1

**Status:** draft implementation in PR #13695  
**Date:** 25 November 2026  
**Sampling authority:** #13624 / #13626  
**Selective decision authority:** #13644 / #13648  
**Two-stage estimator:** #13692  
**Implementation:** temporal_selective_design.rs

## Objective

Estimate risk and coverage at a frozen confidence threshold over the finite population of eligible music query items, respecting the two-stage source-work-first probability-audit sample. The active-refinement stream is adaptive and is not part of the confirmatory denominator.

The adapter consumes selective decisions already produced by #13648 and the exact selection/inclusion records from #13626. It reuses #13692's HT pairwise variance primitive. It does not create another sampler, selective classifier, temporal matcher or truth adjudicator.

## Input contract

A strict versioned input contains:
- the complete candidate frame and frozen two-stage acquisition configuration;
- optional expected selection-receipt digest (null only for deterministic demonstration);
- immutable model/runtime/decoding/hardware/evaluator/scoring-plan identities;
- a threshold plan with unique ascending confidence thresholds, a fixed greater_or_equal tie rule and canonical digest;
- exactly one completed selective outcome for every selected ProbabilityAudit item.

Each outcome binds the item/query/source-work, selective report, adjudication profile, execution identities, evidence stratum, identifiability basis, decision, confidence, decision correctness, unconditional-candidate correctness and false-absence disposition.

The planner is replayed from the exact frame, configuration and seed. Outcomes that are missing, duplicated, extra, unselected, active-refinement, bound to another source work/query, associated with another model/runtime/plan, or carrying duplicate report digests are rejected. The threshold-plan digest is recomputed before scoring.

## Threshold semantics

At threshold tau, a case is covered only when:
- its decision is Answer or ClaimAbsent; and
- decision confidence is greater than or equal to tau.

Both abstention reasons remain uncovered. The selective-error indicator is one only for a covered incorrect decision. False absence claims are reported as a separate covered-error subtype, not merged into ordinary localization errors.

Thresholds must be chosen on source-lineage-disjoint calibration/development data and frozen before confirmatory outcomes are opened. The adapter refuses unsorted/duplicate thresholds or a threshold-plan digest mismatch.

## Two population estimands

For each selected query item i, let C_i(tau) be coverage and E_i(tau) be covered error. Let N be the known number of eligible query items and pi_i the two-stage first-order inclusion probability.

### Query-item-weighted

\[
\widehat{T}_C=\sum_{i\in s}\frac{C_i}{\pi_i},
\qquad
\widehat{T}_E=\sum_{i\in s}\frac{E_i}{\pi_i}.
\]

\[
\widehat{Coverage}_{item}=\frac{\widehat{T}_C}{N},
\qquad
\widehat{Risk}_{item}=\frac{\widehat{T}_E}{\widehat{T}_C}.
\]

### Equal-source-work contribution

For each source work w with N_w eligible query items, use C_wi/N_w and E_wi/N_w as the item contributions. Their HT totals are divided by the known number of eligible source works for coverage, and their error/covered total ratio gives the source-work-standardized selective risk.

Both curves appear in the output. They are distinct targets: the first gives equal weight to query items, the second standardizes each source work's contribution before pooling.

Risk is undefined if the estimated covered total is zero. It is never replaced with zero. Coverage HT estimates are not artificially clipped to one.

## Ratio uncertainty

At a frozen threshold, define R_hat = T_hat_E / T_hat_C and the linearized item outcome:

\[
Z_i=E_i-\widehat{R}C_i.
\]

Then use the general two-stage HT pairwise-inclusion variance estimator:

\[
\widehat{Var}(\widehat{R})\approx
\frac{\widehat{Var}_{HT}(\sum_i Z_i/\pi_i)}
{\widehat{T}_C^2}.
\]

The source-work risk uses the same formula on Z_wi/N_w. Coverage uncertainty uses the HT variance of the covered total divided by the known population denominator squared. False-absence rate among covered items uses its own linearized ratio with F_i (covered false-absence claim) in place of E_i.

This adapter directly reuses the pairwise inclusion-probability routine already qualified by #13692, including unequal source-work sizes and within-work sampling fractions. It does not assume independent excerpts from the same work.

The reported 95% intervals are normal-approximation, clipped to the proportion domain. Small-sample intervals may not achieve nominal coverage; the report retains variance-unavailable reasons and does not claim statistical authentication.

## AURC limitation

V1 reports the full threshold-specific risk/coverage points and intervals. It **does not** report an AURC confidence interval. The threshold estimates share the same sampled items, so their errors are correlated; treating thresholds as independent would understate or misstate AURC uncertainty. A future AURC interval must propagate cross-threshold covariance or derive a per-item linearized influence for the integrated curve.

## Baselines

At every threshold, the report supplies the always-answer/unconditional-candidate error rate as the random-retention expected risk at the same expected coverage, under retention independent of error. It is calculated from the same probability-audit items and the same design weights.

The report also keeps false-absence error separate. A selective policy cannot improve its risk by relabelling unobservable or ambiguous evidence as absent.

## Synthetic control

The CLI fixture is a complete census of one synthetic source work with two eligible query items:
- query 1: Answer, confidence 0.9, correct;
- query 2: ClaimAbsent, confidence 0.6, incorrect false-absence claim.

At threshold 0: coverage 1.0, risk 0.5, false-absence rate 0.5. At threshold 0.7 and 0.9: coverage 0.5, risk 0, false-absence rate 0. The always-answer and random-retention risk are 0.5. Since this is a complete census of a one-work synthetic population, design variance is zero.

The Rust tests additionally cover:
- zero-coverage thresholds and undefined risk;
- false absence in both risk numerator and its separate subtype;
- selection/source/model/threshold-plan mismatches;
- active or unselected outcome rejection;
- item- versus source-work-weighted differences under unequal work sizes;
- withholding variance when one work is selected from a larger work population.

These controls validate estimator plumbing, not model performance or musician outcomes.

## Claim ceiling

The report sets:
- selection_receipt_match_verified=false when no expected digest is supplied;
- selective_outcomes_authenticated=false;
- population_confidence_claim_permitted=false;
- aurc_confidence_interval_computed=false.

A valid HT design variance cannot repair a biased or incomplete frame, unauthenticated decisions, missing rights/consent, or an invalid ground-truth process. The next empirical requirement is a rights-cleared source-work frame with an independently verified selective-execution/scoring receipt, frozen calibration thresholds, enough source-work clusters and a preregistered power analysis.

## Implementation

- Issue #13694 — design-weighted selective risk/coverage integration.
- PR #13695 — draft implementation.
- Acquisition planner: #13626.
- Selective evaluator: #13648.
- Two-stage HT estimator: #13692.
