# Symthaea design-based stratified audit risk v1

Status: research implementation; exact-head Rust qualification pending.

## Composition

This module composes the typed prediction record and the custodian-randomized stratified SRSWOR plan. It evaluates the complete frozen prediction cohort while only selected random-audit units carry independent correctness labels.

Candidate predictions, confidence and abstention status must be frozen before audit selection. Selected units must receive independent adjudication. Nonselected units remain pending in this audit lane. The candidate never receives random priorities, selected membership, hidden labels, future outcomes or evaluator scores.

## Estimand

For a declared acceptance set A over the full frozen cohort, let N_A be the known number of accepted predictions. For each unit i define x_i(A)=1 when the candidate answers and is wrong, and 0 otherwise.

The population error total is T_A=sum_i x_i(A), and selective risk is R_A=T_A/N_A when N_A>0.

For each stratum h, the audit uses SRSWOR with population N_h and quota n_h. The per-unit inclusion probability is pi_i=n_h/N_h. The Horvitz-Thompson estimate of stratum error total is:

T_hat_h(A) = (N_h/n_h) * sum_{i in sample_h} x_i(A).

Risk divides the sum of the stratum HT totals by the exact full-cohort accepted count N_A. The point estimate is not clipped; finite-sample estimates can exceed one.

## Design-based variance

Let s_h^2(A) be the ordinary sample variance of the sampled binary losses x_i(A). Under SRSWOR, the unbiased stratum total variance estimate is:

V_hat(T_hat_h) = N_h^2 * (1 - n_h/N_h) * s_h^2 / n_h.

Assuming independent randomization across strata, these variance estimates add. Risk variance divides the total variance by N_A^2.

A census stratum contributes zero variance. A non-census stratum needs at least two sampled units to estimate its within-stratum sample variance. A stratum with no accepted population units has identically zero loss and contributes zero variance even with one audit unit. Other non-census one-unit strata make the overall design variance unavailable, while retaining the point estimate.

The implementation reports pointwise asymptotic normal intervals only when every non-census stratum has estimable variance and the accepted audit-label count meets the configured minimum. No finite-sample, anytime-valid, or simultaneous guarantee is claimed. AURC is reported as a tie-aware point estimate; its covariance across nested coverage prefixes is not yet estimated.

## Efficiency

The risk-coverage curve is updated incrementally in confidence order. Each newly accepted tie group updates per-stratum counts; each prefix is then scored from those counts. Sorting costs O(N log N), followed by O(N + U*H), where U is the number of unique confidence levels and H the number of strata.

## Validation boundaries

- The evaluator validates the complete cohort mapping, sample membership, quotas and inclusion-probability formulas.
- It does not authenticate random priorities, prove uniform randomization or verify custodian independence.
- Every selected unit must be adjudicated; selected-label nonresponse is a separate selection mechanism.
- Normal intervals are pointwise asymptotic approximations, not simultaneous or formal risk certificates.
- Decision-cost inputs are rejected until a design-based cost estimator exists.
- No live audit, benchmark result, SOTA claim or scientific-performance improvement is claimed.

## Tests authored

Production-source tests cover hand-computed HT risk and SRSWOR variance, census, singleton-stratum withholding, independent-stratum variance addition, inclusion-probability tampering, selected/nonselected adjudication boundaries, tie handling, cohort integrity, configuration and baselines. They are authored but not claimed as executed until exact-head Rust 1.96 qualification is observed.
