# Randomized global CRC for common-depth CIFAR: finite-population LP

**Date:** 2026-10-31  
**Status:** independent finite-population LP and 500-round equal-share replays completed; all 1,378 augmented-state constraints pass for all four targets. The exact committed module and its 22 authored tests have not yet been run in a clean checkout. No remote CI pass is claimed. This is an oracle objective benchmark, not a deployable policy.

## Idea

The common-depth CIFAR controller has piecewise-linear loss with a kink at depth a=1:

    ell_0(a)=0
    ell_1(a)=0.5*(a-1)_+
    ell_2(a)=0.5*a

The hierarchy-specific correct-specificity utility is also piecewise linear, with breakpoints at 0, 1 and 2. Therefore, the convex hull of randomized continuous-depth policies is represented exactly by categorical action distributions over {0,1,2}.

For each calibration count vector D, the LP selects a distribution P(a|D). For every augmented count vector S of total n+1, it enforces:

    sum_z S_z * E[ell_z(A(S-e_z))] <= (n+1)*alpha.

The 1,378 augmented states are all included. The policy samples one common depth per calibration sample and reuses it for the test batch. The expected-risk calculation includes this action randomization.

## Exact finite-population utility objective

The published pool counts are N=(7,300,969,1,731). For calibration count vector D of size 50:
- the probability of D is multivariate hypergeometric, proportional to product_z binomial(N_z,D_z);
- conditional on D, the next test image has type probabilities (N-D)/(10,000-50).

The LP maximizes expected correct specificity under this finite-pool sampling design. Since the objective uses the published full-pool counts, this is an **oracle benchmark**. A deployable method must estimate/freeze objective weights independently before calibration. Universal risk constraints do not depend on the weights.

## Independent numerical replay

A separate scratch implementation of the committed equations re-solved the global LP at all four levels, then recomputed the 500-round equal-share controller from the common-depth CRC equations. Full matrix-build plus solve took about 4.00 seconds total across the four levels in that replay.

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

The incremental specificity gain over equal-share is only 0.000 pp, 0.00634 pp, 0.01302 pp and 0.00074 pp across the four targets. Equal-share is already close to the global utility optimum for this common-depth family and objective. This LP is a useful optimality ceiling rather than a dramatic replacement.

The optimized policy randomizes over two or three depth actions in roughly 45-52% of calibration states.

## Primal-dual optimality evidence

The LP solver's dual marginals were independently checked for primal inequality/equality/bound residuals, dual sign feasibility, stationarity, complementary slackness and primal-dual objective gap. Across the four risk targets, the maximum residuals were:

- primal inequality: 3.33e-14;
- primal equality: 2.93e-14;
- dual sign: 1.42e-15;
- stationarity: 1.74e-18;
- complementarity: 1.32e-18;
- primal-dual gap: 8.89e-16;
- randomized action-row normalization: 2.93e-14.

The equal-share replay had maximum augmented-budget excess (3.56	imes10^{-15}). These numerical receipts corroborate the encoded finite LP and recertification equations; they are not by themselves a theorem about larger or different action spaces.

## Proof-carrying certificate

The module emits a `symthaea-risk-certificate-v1` manifest binding:
- randomized policy probabilities, action grid and canonical state ordering;
- loss family, risk target and complete augmented-state domain;
- objective-reference pool counts and residual test distribution;
- finite-exchangeability assumption contract and expected-risk scope;
- exact validity receipt and LP dual witness.

The independent verifier recomputes the policy/constraint/objective digests, checks the risk target, proof lane, assumptions, scope and verifier receipt, enumerates all augmented states, and recomputes KKT residuals. The 22 authored tests include tampering controls for the policy, digest, alpha, state domain, assumptions, proof lane, verifier receipt, KKT summary and malformed dual witnesses.

**Validation boundary:** the latest exact GitHub blobs are source `60784684550cd0727e4a638190ae6fbd6844ab15` and tests `b146697ba316b2d3f6c69f2ad8acda3facdda3d4`. These exact committed bytes have not yet been executed in a clean checkout; the numerical replay used a separate Python implementation of the equations. No remote CI pass is claimed. The hashes bind certificate content but do not authenticate an issuer; signing/attestation remains separate.

## Interpretation and limits

The global randomized LP adds only a tiny amount of common-depth specificity over equal-share recertification. The larger performance opportunity remains the confidence-adaptive family using per-image confidence breakpoints, for which capping and grouping have been implemented but the CIFAR per-image ResNet-18 predictions are still missing.

The reference utility distribution uses the published full-pool type counts. This is appropriate for an oracle benchmark on that finite population but not as a directly deployable utility estimator. The expected-risk guarantee is under finite exchangeability, including action randomization. No high-probability, conditional-risk or distribution-shift guarantee is claimed. This is not a reproduction of the confidence-adaptive result or the paper's original per-image split IDs.

## Reproduction

    PYTHONPATH=validation python validation/randomized_global_crc_cifar.py --output /tmp/randomized-global-crc.json
    PYTHONPATH=validation python -m unittest -v test_randomized_global_crc_cifar.py

Reference: Bai & Ramdas (2026), https://arxiv.org/abs/2610.35821
