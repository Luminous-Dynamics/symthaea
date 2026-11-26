# MEL-GROUND-010 — Design-weighted selective risk and coverage v1

**Status:** evaluator implementation; synthetic controls only. No Melothaea model result or population claim.

## Purpose

This adapter composes three existing owners rather than replacing them:

- #13626 supplies the source-work-first acquisition plan, frozen probability-audit sample and item inclusion probabilities.
- #13648 supplies the selective decision (Answer, ClaimAbsent, AbstainAmbiguous, AbstainNotEstimable), decision confidence, decision correctness and the paired unconditional-answer counterfactual.
- #13692 supplies the two-stage Horvitz–Thompson estimator and the shared first/second-order inclusion-probability variance primitive.

Only the frozen ProbabilityAudit stream is eligible. ActiveRefinement is adaptive and must never enter a confirmatory finite-population estimate.

## Input and identity contract

The case input binds the acquisition dataset, synthetic/production mode, expected selection-receipt digest, model/runtime/decoding/hardware identity, selective evaluator version, scoring-plan digest, a frozen threshold plan and one selective outcome for each selected probability-audit item.

The join is exact on:
- item ID and query ID;
- source-work digest and **recording digest**;
- model, runtime, decoding, hardware, evaluator and scoring-plan identities;
- selective report digest and adjudication-profile digest;
- selection-receipt replay and item inclusion probability.

The recording digest is checked against the exact candidate in the acquisition manifest, not inferred from source-work identity. Duplicate selective-report receipts, duplicate items, missing outcomes, extra/unselected outcomes and non-audit stream rows fail closed.

synthetic_demonstration=true is required exactly when the expected selection receipt is omitted. A reviewed/audited production input must supply the expected selection receipt. An expected digest only checks deterministic replay; it is not an authenticated authorization or sampling receipt.

The false_absence_claim flag must equal (decision == ClaimAbsent && !decision_correct). This prevents both omitted false-absence flags and false-positive false-absence labels.

## Frozen threshold plan

The threshold plan is hashed in canonical form. V1 supports:
- integer confidence in parts per million, 0–1,000,000;
- a strictly increasing, unique list of at most 101 thresholds;
- one tie rule: confidence greater than or equal to threshold.

A row is covered only when its selective decision is Answer or ClaimAbsent and confidence meets the threshold. Both abstention types remain uncovered. The threshold list must be chosen on source-lineage-disjoint calibration data before opening the confirmatory audit.

## Estimands

For threshold \(\tau\), define:
- \(C_i(\tau)=1\) if the decision is covered;
- \(E_i(\tau)=1\) if it is covered and incorrect;
- \(F_i(\tau)=1\) if it is a covered false absence claim.

The item-weighted totals use the first-/second-order inclusion probabilities from #13626:

\[
\widehat T_C=\sum_{i\in s} C_i/\pi_i,\qquad
\widehat T_E=\sum_{i\in s} E_i/\pi_i.
\]

With known eligible item count \(N\):

\[
\widehat{Coverage}_{item}=\widehat T_C/N,\qquad
\widehat{Risk}_{item}=\widehat T_E/\widehat T_C.
\]

The equal-source-work estimand applies the same estimator to the per-item contribution \(y_{wi}/N_w\), then divides by the eligible source-work count. It gives each source work equal total weight regardless of query count.

The following are emitted as distinct values; none is replaced by another:
- item-weighted coverage;
- item-weighted selective risk;
- item-weighted false-absence rate;
- equal-source-work coverage;
- equal-source-work selective risk;
- equal-source-work false-absence rate.

Risk and false-absence rate are undefined when the estimated covered total is zero.

## Variance and intervals

At a fixed threshold, risk uses ratio linearization. For \(\widehat R=\widehat T_E/\widehat T_C\), the linearized value is \(Z_i=E_i-\widehat R C_i\). The adapter feeds this value into #13692's existing two-stage HT variance routine and divides the estimated total variance by \(\widehat T_C^2\). Coverage variance uses the HT variance of the covered total divided by \(N^2\). The same calculation is performed separately for the equal-source-work estimand.

V1 reports clipped normal-approximation 95% intervals where the design supports variance. These intervals are **design diagnostics**, not guaranteed nominal-coverage intervals for small samples. Missing pairwise inclusion support and other unsupported variance cases remain explicitly unavailable; they are never replaced with zero.

V1 does not report an AURC confidence interval because covariance across thresholds is not implemented. It reports only threshold-specific estimates/intervals.

Random-retention-at-the-same-coverage is compared to the unconditional candidate error rate under the stated independence assumption. It is not treated as a separate observed model policy. Always-answer risk is computed from the actual unconditional candidate correctness supplied by #13648.

## Synthetic fixture

mel-selective-design-risk.synthetic.json is a complete census of one synthetic source work with two query items:
- threshold 0: item coverage 1.0, selective risk 0.5, false-absence rate 0.5;
- threshold 0.7: item coverage 0.5, selective risk 0.0, false-absence rate 0.0;
- threshold 0.9: item coverage 0.5, selective risk 0.0;
- unconditional/always-answer risk: 0.5.

Because the one-work finite population is fully observed, its design variance is zero. This fixture validates joins, threshold semantics and the CLI path only. The two-of-three work sampling and unequal-work tests in Rust exercise nontrivial variance behavior.

## Claim ceiling

The report hard-codes:
- active_refinement_stream_used=false;
- selective_outcomes_authenticated=false;
- population_confidence_claim_permitted=false;
- aurc_confidence_interval_computed=false.

It also distinguishes synthetic demonstration mode from a run with a supplied selection receipt. The adapter does not authenticate source rights, consent, model execution, selective-report records, the acquisition sample receipt or labels. It does not establish that the decision confidence is calibrated or that a musician benefits from abstention.

A population claim requires independently verified sampling/authorization and outcome receipts, a validated measurement process, a frozen confirmatory threshold policy and a sample-size/power design appropriate to the target strata.

## References and related owners

- #13624/#13626 — source-balanced annotation acquisition and frozen probability audit.
- #13644/#13648 — selective decisions and reason-specific abstention.
- #13627/#13692 — finite-population Horvitz–Thompson estimator and design variance.
- #13694 — parent integration issue.
- TAG-Bench and MusTBench remain separate temporal grounding benchmarks; no selective-risk estimate should be compared to their mIoU leaderboards as if it were the same metric.
