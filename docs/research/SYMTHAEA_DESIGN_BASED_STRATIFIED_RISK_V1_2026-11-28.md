# Symthaea design-based stratified audit risk v1

Status: research implementation; exact-head Rust qualification pending.

## Composition

This module composes:
- the typed prediction and selective-policy record from selective_prediction.rs;
- the custodian-randomized stratified SRSWOR plan from stratified_audit.rs;
- independent correctness labels for selected random-audit units only.

The input must contain every record in the frozen prediction cohort. Candidate predictions, confidence and abstention status must be frozen before the audit sample is selected. The selected random-audit units must receive independent adjudication. Nonselected units remain pending in this audit lane; selected units with missing labels fail closed.

The plan's randomization manifest and the adjudicator's independence must be externally audited. The candidate never receives random priorities, selected membership, hidden labels, future outcomes or evaluator scores.

## Estimand

For a declared acceptance set A over the full frozen cohort, let N_A be the known number of accepted predictions. For each population unit i, define:

x_i(A) = 1 if the candidate answers and is wrong, and 0 otherwise.

The population error total is T_A = sum_i x_i(A), and selective risk is R_A = T_A/N_A when N_A > 0.

For each pre-label stratum h, the audit uses SRSWOR with population size N_h and audit quota n_h. The per-unit inclusion probability is pi_i = n_h/N_h. The Horvitz-Thompson estimate of the error total in stratum h is:

T_hat_h(A) = (N_h/n_h) * sum_{i in sample_h} x_i(A).

The overall error estimate is the sum over strata. Risk divides that error estimate by the exact accepted count N_A in the frozen prediction cohort. The point estimate is not clipped; in a finite audit sample the estimated risk can exceed one.

## Design-based variance

Within stratum h, let s_h^2(A) be the ordinary sample variance of the sampled binary losses x_i(A). Under SRSWOR, the standard unbiased estimator for the variance of the stratum HT total is:

V_hat(T_hat_h) = N_h^2 * (1 - n_h/N_h) * s_h^2 / n_h.

Assuming independent randomization across strata, the total variance estimate is the sum of the stratum estimates. For a fixed acceptance count N_A > 0, the estimated risk variance is:

V_hat(R_hat_A) = V_hat(T_hat_A) / N_A^2.

A census stratum contributes zero variance. A non-census stratum needs at least two sampled units to estimate its within-stratum sample variance. A stratum with no accepted population units has identically zero loss and contributes zero variance even with a one-unit audit. Other one-unit non-census strata make the overall design variance unavailable, though the point estimate remains visible.

The implementation reports pointwise normal-approximation intervals only when every non-census stratum has estimable variance and the accepted audit-label count meets the configured minimum. The intervals are asymptotic approximations, not finite-sample guarantees. They are not simultaneous across coverage thresholds or strata, and no anytime-valid risk-control guarantee is claimed.

The AURC point estimate is computed over confidence prefixes with equal-confidence ties treated atomically and item-wise risk averaged over within-tie permutations. Its covariance across nested prefixes is not yet implemented, so no AURC uncertainty interval is reported.

## Efficiency and complexity

The risk–coverage curve is updated incrementally in confidence order. Adding a confidence tie group updates per-stratum accepted-population and accepted-audit error counts. Each curve point is then evaluated from those counts in O(H), where H is the number of strata, rather than rescanning all N records for every unique confidence value. Sorting costs O(N log N); curve accumulation costs O(N + U H), where U is the number of distinct confidence values.

## Validation boundaries

- The evaluator validates cohort identity, sample membership, quotas, first-order inclusion probabilities and pair counts before estimating risk.
- It does not authenticate random priorities, prove uniform randomness or verify that the custodian remained independent.
- It assumes every selected random-audit unit was adjudicated; nonresponse is a separate selection mechanism and is not silently treated as random.
- The pointwise normal interval is not a multiple-comparison band or a formal risk certificate.
- No real adjudication campaign, independent label set, external benchmark score or SOTA result is claimed.

## Tests authored

Eight production-source tests cover the hand-calculated HT risk and SRSWOR variance, census variance, one-unit non-census variance withholding, selected/nonselected adjudication completeness, tie-invariance, cohort ID/count integrity, operating-point configuration and policy/threshold reporting. The code is imported by path into the standalone Rust 1.96 adjudication harness. Tests are authored, not claimed as run until an exact-head Actions result is observed.
