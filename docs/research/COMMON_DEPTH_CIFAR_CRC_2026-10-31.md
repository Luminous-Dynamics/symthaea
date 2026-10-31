# Common-depth CIFAR-100 CRC: independent finite-population reproduction

**Date:** 2026-10-31  
**Status:** 300 paired new splits; n=50; 16 test images; all four risk targets; 500 recertification rounds; 1,378 augmented constraints verified after every update. This is an independent common-depth type-count reproduction, not the paper's original split-manifest reproduction and not the confidence-adaptive family.

## Why the common-depth experiment is reproducible without per-image scores

Appendix B.7 defines three error types and states that, for the common-depth family, the controller depends on the calibration sample only through the three type counts. Its loss also depends on observations only through type, and the expected prediction metrics are analytic in depth and type.

The published pool contains 7,300 fine-correct, 969 superclass-correct, and 1,731 superclass-wrong examples. Therefore, for the common-depth controller, sampling uniformly without replacement from this finite population can be simulated exactly at the type-count level. The original image IDs and exact split manifest are not required to reproduce the distribution of common-depth results, although they are required to match the exact 300 published splits.

This shortcut does **not** apply to the confidence-adaptive family, which needs every image's fine-class and superclass confidence breakpoints (u_i,v_i).

## Method

For depth a in [0,2], prediction randomization selects root/superclass/leaf with probabilities

\[
\pi(a)=((1-a)_+,\,1-|a-1|,\,(a-1)_+).
\]

The three loss functions are

\[
\ell_0(a)=0,\qquad
\ell_1(a)=\frac12(a-1)_+,\qquad
\ell_2(a)=\frac12a.
\]

For calibration type counts h=(h0,h1,h2), the CRC/envelope controller solves the largest a with

\[
h_1(a-1)_+ + \frac{h_2a}{2}+U(a)\le(n+1)\alpha,
\]

where U(a)=1 for CRC and U(a)=a/2 for envelope CRC. The objective is piecewise-linear with a kink at a=1, so the implementation solves it exactly on [0,1] and [1,2].

Equal-share recertification uses the Appendix B.7 update

\[
A^{k+1}(D,x)=
\min\left\{
2,\,
a+2s_{k,2}(D,x),\,
1+(a-1)_+ +2s_{k,1}(D,x)
\right\},
\]

where a=A^k(D,x) and each s_{k,c} is the per-hypothetical-label unused budget computed from the full augmented multiset. In the common-depth family, the table has \(\binom{52}{2}=1,326\) entries for n=50; all \(\binom{53}{2}=1,378\) augmented states are checked after each update.

The common-depth table is built once per alpha, and the exact same 300 split draws are used for CRC, envelope CRC, 1/50/500 rounds, and all target levels.

## Protocol

- Finite population error-type counts: (7,300, 969, 1,731).
- Calibration size: 50.
- Test size: 16.
- Paired splits: 300.
- Sampling: uniform without replacement within each split, using seed 20261031.
- Risk levels: 0.02, 0.05, 0.10, 0.15.
- Methods: CRC, envelope CRC, one round, 1/3/10/20/50/100/200/500 rounds.
- Prediction randomization is integrated analytically, following the paper.
- All augmented count-vector constraints are verified for every controller table.

The local reference driver completed the four 500-round tables in 7.72 seconds total, about 1.85–1.91 seconds per target.

## Results against the paper's common-depth table

The table below compares our 300 new type-count splits with the paper's reported common-depth envelope and 500-round values. Differences are expected because we use a new split seed and not the paper's original image IDs.

| alpha | Method | Empirical risk | Correct specificity | Mean node size |
|---:|---|---:|---:|---:|
| .02 | Envelope CRC | .02042 | 12.417% | 77.99 |
| .02 | Recertified 500 | .02042 | 12.417% | 77.99 |
| .05 | CRC | .03552 | 21.611% | 61.70 |
| .05 | Envelope CRC | .05089 | 30.745% | 45.56 |
| .05 | Recertified 500 | .05112 | 30.780% | 45.55 |
| .10 | CRC | .09039 | 49.215% | 15.23 |
| .10 | Envelope CRC | .09945 | 52.648% | 10.16 |
| .10 | Recertified 500 | .10192 | 52.996% | 10.08 |
| .15 | CRC | .14243 | 60.945% | 3.72 |
| .15 | Envelope CRC | .14786 | 61.790% | 3.36 |
| .15 | Recertified 500 | .15105 | 62.229% | 3.26 |

Paired improvement over envelope CRC:

| alpha | One round | 50 rounds | 500 rounds | Paper's 500-round gain |
|---:|---:|---:|---:|---:|
| .02 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp |
| .05 | +0.005 pp | +0.035 pp | +0.035 pp | +0.036 pp |
| .10 | +0.077 pp | +0.348 pp | +0.348 pp | +0.337 pp |
| .15 | +0.113 pp | +0.439 pp | +0.439 pp | +0.443 pp |

Paired 95% intervals for the 500-round improvement:
- alpha=.05: +0.03465 pp [0.01188, 0.05742].
- alpha=.10: +0.34839 pp [0.30167, 0.39510].
- alpha=.15: +0.43856 pp [0.40433, 0.47280].

At alpha=.02, there is no gain, as in the paper. At the other levels, the gains closely match the published common-depth results. Fifty rounds already reach essentially the 500-round fixed point in all three nonzero-gain settings.

The empirical mean test risk at some levels is slightly above alpha, but the Monte Carlo standard errors across 300 splits are approximately 0.0018, 0.0033 and 0.0044 at levels .05, .10 and .15, respectively. The deviations are within the split-sampling uncertainty. More importantly, the finite-exchangeability validity check is not inferred from these empirical means: every augmented multiset constraint is checked directly and all controllers remain valid.

## Published reference comparison

The paper's Table 2 reports for the common-depth family:

| alpha | Envelope CS / size | Recertified 500 CS / size | Specificity gain |
|---:|---:|---:|---:|
| .02 | 12.538% / 77.96 | 12.538% / 77.96 | 0.000 pp |
| .05 | 30.989% / 45.62 | 31.025% / 45.61 | 0.036 pp |
| .10 | 52.883% / 10.65 | 53.220% / 10.58 | 0.337 pp |
| .15 | 62.279% / 3.37 | 62.722% / 3.28 | 0.443 pp |

Our reproduced gains are very close, which is a good independent check of the common-depth formula and recertification recursion.

## What this does not reproduce

The stronger confidence-adaptive result (e.g. 67.572% to 68.235% correct specificity at alpha=.10) is a different method. It requires:
- per-image superclass and fine-class confidence breakpoints u_i=1-p_g(X_i), v_i=1-p_f(X_i);
- recursion with each calibration point and the test image held out in turn;
- hypothetical type assignments for every active point;
- exact breakpoint capping and grouping to reduce the r*2^(r-1) table.

The three pool-wide type counts do not contain those per-image breakpoints. We have not reproduced that adaptive family, and should not imply otherwise.

## Next gates

1. Execute the committed Python module/test suite against exact fetched branch bytes; the current numerical run used an independent local reference driver implementing the same formulas.
2. Differential-test one/two/500 rounds against exact common-depth rows from the paper's Table 2.
3. Obtain or regenerate cached ResNet-18 confidence breakpoints and reproduce the confidence-adaptive family with exact capping/grouping.
4. Compare the feature-adaptive implementation against MAPIE's ConditionalExpectedRiskController only after aligning risk definitions, data splits and guarantee assumptions.
5. Benchmark controller-build time, per-image latency, table size and grouping compression.
6. Treat covariate/label shift only as an out-of-assumption stress test; finite exchangeability does not guarantee distribution-shift validity.

References:
- Bai & Ramdas (2026), Section 3.6/Table 2/Appendix B.7: https://arxiv.org/abs/2610.35821
- Full text: https://arxiv.org/pdf/2610.35821
