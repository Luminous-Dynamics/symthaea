# Exact-rational verifier for common-depth CRC

**Date:** 2026-10-31  
**Status:** 7/7 local tests passed; all four alpha targets were checked for 500 recertification rounds with exact rational arithmetic. No remote CI pass is claimed.

## Motivation

The floating-point common-depth recursion already checked all 1,378 augmented count vectors. The next assurance step is to remove tolerances from the verifier: for decimal risk targets such as .02, .05, .10 and .15, use the exact rational levels 1/50, 1/20, 1/10 and 3/20. Every loss and budget operation is then performed using arbitrary-precision fractions.

For calibration count vector h=(h0,h1,h2) and augmented vector q of size n+1, the exact validity inequality is:

    sum_{c=0}^2 q[c] * ell_c(A(q-e_c)) <= (n+1)*alpha

where ell0(a)=0, ell1(a)=max(a-1,0)/2, and ell2(a)=a/2.

The code independently solves the piecewise-linear CRC and envelope-CRC baseline on [0,1] and [1,2], and computes every equal-share recertification update using exact slacks. Negative slack is a hard error; no clipping, epsilon or Monte Carlo result can convert a failing table into a pass.

## Exact state universe

At n=50:
- 1,326 calibration count vectors;
- 1,378 augmented count vectors.

Every recertification round checks the input table and updated table against all 1,378 augmented constraints. The report's check count accounts for each table validation, including snapshot validations.

## Executed results

All four risk targets were run through 500 rounds.

| alpha | exact budget | worst augmented vector | worst total | exact excess | final denominator digits |
|---:|---:|---|---:|---:|---:|
| .02 | 51/50 | (0,0,51) | 51/50 | 0 | 858 |
| .05 | 51/20 | (0,0,51) | 51/20 | 0 | 863 |
| .10 | 51/10 | (0,0,51) | 51/10 | 0 | 865 |
| .15 | 153/20 | (0,0,51) | 153/20 | 0 | 870 |

Every saved snapshot at rounds 0, 1, 10, 20, 50, 100, 200 and 500 had exact maximum budget excess 0. The same all-type-2 augmented vector was worst and exactly binding at every saved round. Both standard CRC and envelope CRC baselines also passed exact checks at all four levels.

Across 500 rounds, 1,009 full-table verification passes are recorded per target, or 1,390,402 exact augmented inequalities per alpha and 5,561,608 across four targets. A full 500-round alpha=.10 run using the committed-code-equivalent reference completed in 43.41 seconds. The independent four-target prototype runs took 19.23, 23.83, 33.24 and 32.99 seconds respectively.

The largest final rational denominators range from 858 to 870 decimal digits. This shows both the strength and the cost of arbitrary-precision rational checking.

## Negative controls and tests

7/7 tests passed on a local executable copy matching the committed logic:
- count-vector cardinalities;
- exact piecewise loss values;
- exact inversions on both action intervals;
- standard and envelope CRC exact validity at all four paper alpha levels;
- 50 rounds with every table validation exact;
- an over-lossy depth table fails closed and returns the exact witness (0,0,3);
- decimal alpha parsing maps .10 exactly to 1/10.

The local test run did not use a clean independent checkout, and remote CI was not run. Committed source/test blobs:
- source: `4246b82635d20ff7b8aa4f83024d11b3671fc37b`
- tests: `61b7a92efe0cf1f514a1daab78d3620e3b4e5326`

## Engineering implication

The exact rational implementation should remain a *reference oracle*, not the production engine. A production Rust verifier should target certified interval arithmetic or a proven error-bound scheme, and should be differentially tested against this exact oracle on all finite fixtures. The certificate should preserve the exact rational target, controller digest, loss definition, augmented-state universe, worst-state witness and verifier identity.

This closes a meaningful gap between “the float calculation stayed within tolerance” and “the finite exchangeability constraints were exactly satisfied” for the common-depth family.

## Scope

This verifies the common-depth finite-exchangeability family only. It does not reproduce the confidence-adaptive per-image controller, prove robustness under distribution shift, or prove a new theorem.

Reference: Bai & Ramdas (2026), Appendix B.7, https://arxiv.org/abs/2610.35821
