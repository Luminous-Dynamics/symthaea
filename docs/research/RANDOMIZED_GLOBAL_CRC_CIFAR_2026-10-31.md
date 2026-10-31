# Randomized global CRC for common-depth CIFAR: finite-population LP

**Date:** 2026-10-31  
**Status:** local finite-population reference execution; all 1,378 augmented-state constraints pass at each alpha. The committed module is not yet run in a clean checkout and no CI pass is claimed. This is an oracle objective benchmark, not a deployable policy.

## Idea

The common-depth CIFAR controller has piecewise-linear loss with a kink at depth a=1:

    ell_0(a)=0
    ell_1(a)=0.5*(a-1)_+
    ell_2(a)=0.5*a

Its hierarchy-specific utility (correct specificity) is also piecewise linear, with breakpoints at 0, 1 and 2. We can therefore represent the convex hull of randomized continuous-depth decisions by a categorical action distribution supported at {0,1,2}.

For each calibration count vector D, optimize a distribution P(a|D). For every augmented count vector S of total n+1, enforce

    sum_z S_z * E[ell_z(A(S-e_z))] <= (n+1)*alpha.

These are the exact finite-exchangeability constraints; all 1,378 augmented states are included. The action distribution is sampled once per calibration sample and the resulting common depth is reused for its test batch. The expected-risk guarantee is over calibration/test sampling and this action randomization.

## Exact finite-population utility objective

The published pool counts are N=(7,300,969,1,731). For calibration count vector D of size 50:
- the probability of D is multivariate hypergeometric, proportional to product_z binomial(N_z,D_z);
- conditional on D, the next test image has type probabilities (N-D)/(10,000-50).

The LP maximizes exact expected correct specificity under this finite-pool split law. This objective uses published full-pool type counts, so it is an **oracle benchmark**: it must be frozen using independent data if deployed. The risk constraints themselves do not use the objective weights.

## Results

| alpha | Envelope specificity | Equal-share 500 | Global randomized LP | Gain vs 500 |
|---:|---:|---:|---:|---:|
| .02 | 12.41758% | 12.41802% | 12.41802% | ~0.000 pp |
| .05 | 30.66029% | 30.69199% | 30.69833% | +0.00634 pp |
| .10 | 52.76078% | 53.10382% | 53.11684% | +0.01302 pp |
| .15 | 62.02611% | 62.46973% | 62.47048% | +0.00074 pp |

Expected risk and mean node size:

| alpha | Method | Expected risk | Mean node size |
|---:|---|---:|---:|
| .02 | Equal-share 500 | .0199963 | 78.06764 |
| .02 | Global randomized | .0199963 | 78.06764 |
| .05 | Equal-share 500 | .0499334 | 45.94038 |
| .05 | Global randomized | .0499781 | 45.93906 |
| .10 | Equal-share 500 | .0997941 | 10.32362 |
| .10 | Global randomized | .0998855 | 10.32091 |
| .15 | Equal-share 500 | .1480430 | 3.31987 |
| .15 | Global randomized | .1480482 | 3.31972 |

The largest additional specificity is only 0.013 percentage points at alpha .10. This means equal-share is already close to the global utility optimum for the common-depth family and this specificity objective. The LP is useful as an upper-bound-style optimality benchmark, not a dramatic replacement.

The optimized policy randomizes across two or three depth actions for roughly 45-52% of calibration states. LP solve time is about 25-37 ms per alpha in the local reference run. Maximum augmented-state budget residual is below 3.4e-14.



## Proof-carrying certificate

The module now emits a `symthaea-risk-certificate-v1` manifest binding:
- the randomized policy table, action grid and canonical calibration-state order;
- the loss family, risk target and complete augmented-state domain;
- the objective-reference pool and conditional test-type distribution;
- the finite-exchangeability assumption contract and expected-risk scope;
- the exact verifier receipt and full LP dual witness.

The independent verifier recomputes the policy/constraint/objective digests, rechecks all augmented-state inequalities, validates the scope and risk target, and recomputes dual feasibility, stationarity, complementarity and the primal-dual gap. Regression tests are authored for tampering with policy rows, alpha, state-domain counts, assumptions, verifier receipts and dual marginals. Those exact committed module tests have not yet been run in a clean checkout; the certificate API is not claimed as validated yet. Hashes bind content but do not authenticate an issuer; signing/attestation remains a separate layer.

The module now contains 16 authored tests. A corrected independent Python replay (separate scratch implementation of the committed equations) re-ran all four LPs and all four 500-round equal-share controllers successfully. It confirms the metrics and residuals reported above; full matrix-build plus solve took 4.00 seconds across the four levels. It does not import or execute the exact GitHub module. The branch currently has 22 authored tests, but the exact committed module/tests have not been run in a clean checkout. No remote CI pass is claimed. arithmetic checks cover all four alpha values and all primal/dual residual families.
## Interpretation

This is a small improvement, but an important scientific result:
- the global LP confirms the remaining utility gap after 500 equal-share rounds is tiny;
- any proposed algorithmic improvement to this common-depth family should be measured against this global optimum, not only against the CRC baseline;
- the confidence-adaptive family remains the likely source of larger gains because it uses per-image confidence breakpoints rather than one common depth.

## Limits

The reference utility distribution is derived from the whole published pool, including data which would be held out in actual deployment. This is acceptable as an oracle benchmark for the finite pool but not as a deployable calibration procedure. Objective weights must be estimated from independent data and frozen before calibration to claim deployment validity for the chosen utility. Universal risk constraints are checked exactly, but utility optimality is only for this reference distribution and common-depth action family.

No high-probability risk bound, no conditional risk guarantee, and no distribution-shift robustness is claimed. This does not reproduce the confidence-adaptive family or the paper's original per-image split results.

## Reproduction

    PYTHONPATH=validation python validation/randomized_global_crc_cifar.py --output /tmp/randomized-global-crc.json
    PYTHONPATH=validation python -m unittest -v test_randomized_global_crc_cifar.py

Reference: Bai & Ramdas (2026), https://arxiv.org/abs/2610.35821
