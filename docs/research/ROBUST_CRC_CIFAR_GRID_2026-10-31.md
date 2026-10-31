# Dense-grid distributionally robust CRC: 171 finite population mixtures

**Date:** 2026-10-31  
**Status:** independent reference LP replay across 171 finite-population mixtures; all 1,378 augmented-state CRC constraints included. Ten tests are authored in the branch but have not been run against exact committed GitHub bytes. This is not a continuous distribution-robustness theorem.

## Ambiguity set

To check whether a five-scenario result was merely a convenient hand-picked set, this benchmark uses every integer three-type population composition on a 500-example lattice with each type count at least 500:

[
N_0+N_1+N_2=10{,}000,quad N_ige500,quad N_iequiv500pmod{500}.
]

There are (inom{19}{2}=171) such population mixtures. The nominal published composition (7,300,969,1,731) is not itself on this lattice; the lattice is a separate, deliberately broad stress set.

For each scenario, calibration-state probabilities are multivariate hypergeometric, and the next-test type distribution is the residual population after calibration. One common randomized policy (P(amid D)) is optimized over the whole scenario set.

## Optimization

The robust LP maximizes a scalar t subject to:

[
tle E_s[	ext{correct specificity}]
]

for every one of the 171 scenarios, together with the original 1,378 augmented-multiset finite-exchangeability constraints:

[
sum_z S_z E[ell_z(A(S-e_z))]le(n+1)alpha.
]

The risk constraints are universal over the three-type augmented-state space and are independent of the objective scenario weights. The 171 scenarios only define the finite maximin utility objective.

## Result

At n=50, alpha=.10:

| Policy | Worst-case specificity | Worst scenario |
|---|---:|---|
| Robust maximin LP | 1.44542794% | (500,500,9,000) |
| Equal-share, 500 rounds | 1.44542828% | (500,500,9,000) |
| Nominal oracle LP | approximately 0% | (500,9,000,500) |

The robust LP preserves a worst-case specificity floor around 1.44543%. Equal-share achieves essentially the same floor; the difference is below numerical solver tolerance (about (3.4	imes10^{-7}) percentage points). For this common-depth family and this ambiguity grid, equal-share appears already maximin-optimal.

The nominal oracle policy is a much more revealing failure mode: under the extreme (500,9,000,500) mixture it falls back to the root-only prediction for typical calibration states, yielding almost zero specificity. It remains risk-valid, but its utility is brittle when the reference population changes substantially.

Across the 171 scenarios, maximum expected risk was:
- robust policy: .0999999763;
- nominal policy: .0999998858;
- equal-share: .1000000000 (only (1.6	imes10^{-14}) over due floating-point arithmetic).

Formal validity is not inferred from these expected risks: the LP enforces all 1,378 augmented-state constraints. The separately recomputed 500-round equal-share controller had maximum augmented-budget excess (2.66	imes10^{-15}).

## LP verification

The robust LP has 3,979 variables, 1,549 inequalities (1,378 risk constraints plus 171 scenario-utility constraints) and 1,326 state-normalization equalities.

HiGHS solve time was 1.60 seconds. KKT residuals:
- primal inequality violation: (5.36	imes10^{-11});
- primal equality residual: (2.22	imes10^{-16});
- dual sign violation: 0;
- stationarity residual: (9.74	imes10^{-10});
- complementarity: (1.78	imes10^{-15});
- primal-dual gap: 0.

The broader comparison pipeline, including matrix construction and policy evaluation, took about 16.3 seconds in the independent scratch replay.

## Interpretation

This is a negative result for a stronger optimizer, but a useful result for the architecture:
- robustification protects against a brittle nominal objective;
- the equal-share policy is already essentially maximin for this 171-point grid;
- a new algorithm should not be advertised as better than equal-share without beating this robust benchmark;
- the larger remaining opportunity is likely confidence-adaptive per-image breakpoints, not further common-depth optimization.

This is a finite scenario grid, not a continuous simplex guarantee. No protection is claimed for distributions outside these 171 mixtures. The expected-risk CRC validity contract remains finite exchangeability, not arbitrary covariate or label shift.

## Reproduction

    PYTHONPATH=validation python validation/robust_crc_cifar.py --mode grid --output /tmp/robust-crc-grid.json
    PYTHONPATH=validation python -m unittest -v test_robust_crc_cifar.py

The independent numerical replay used a scratch implementation of the same LP equations. The branch has 10 authored tests; exact committed source and tests have not yet been run in a clean checkout. No CI pass is claimed.
