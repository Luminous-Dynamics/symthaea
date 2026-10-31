# Corrected exact-rational verifier for common-depth CRC

**Date:** 2026-10-31  
**Status:** Corrected baseline inversion; all four targets checked through 500 rounds with exact rational arithmetic. The earlier draft's exact feasibility checks were valid, but its starting envelope baseline was too conservative because it omitted a factor of one-half on type-1 loss.

## Correction

The hierarchical loss for a type-1 error is
[
\ell_1(a)=\frac12(a-1)_+,
]
not ((a-1)_+). Above depth one, the envelope CRC inequality is therefore

[
\frac{h_1}{2}(a-1)+\frac{h_2+1}{2}a\le(n+1)\alpha.
]

For standard CRC, replace the envelope term (a/2) with 1:

[
1+\frac{h_1}{2}(a-1)+\frac{h_2}{2}a\le(n+1)\alpha.
]

The earlier rational implementation had the correct loss function and correct augmented-state verifier, so all of its returned tables still satisfied the exact risk inequalities. However, its closed-form baseline inversion used (h_1(a-1)) above depth one, making the starting controller unnecessarily conservative. Its baseline/recertification results are superseded by this corrected version.

A new exact regression uses (h_1=4,h_2=3,B=51/10):
- envelope depth: (71/40);
- standard CRC depth: (61/35).

These values satisfy their respective piecewise inequalities with equality.

## Exact state universe

For n=50:
- 1,326 calibration count vectors;
- 1,378 augmented count vectors.

Every equal-share round checks its input and output table against every augmented count vector using Fraction arithmetic. The full 500-round protocol records 1,009 complete table validations per alpha (baseline, pre/post-update checks and saved snapshots), or 1,390,402 exact inequalities per alpha.

## Corrected results

| alpha | Exact budget | Worst augmented state | Worst total | Excess | Max denominator digits |
|---:|---:|---|---:|---:|---:|
| .02 | 51/50 | (0,0,51) | 51/50 | 0/1 | 857 |
| .05 | 51/20 | (0,0,51) | 51/20 | 0/1 | 857 |
| .10 | 51/10 | (0,0,51) | 51/10 | 0/1 | 857 |
| .15 | 153/20 | (0,0,51) | 153/20 | 0/1 | 857 |

At rounds 0, 1, 10, 20, 50, 100, 200 and 500, the maximum exact excess is zero for each target. The same all-type-2 augmented state is the worst and exactly binding. Both corrected standard CRC and envelope CRC baselines passed exact checks at every target.

Across four risk levels: 5,561,608 exact augmented-state inequalities were evaluated. Final controller depths use rational denominators of at most 857 decimal digits.

## Local validation and provenance

Nine tests are now authored, including the two new factor-of-one-half regressions:
- count-vector cardinalities;
- exact piecewise loss values;
- both sides of the depth-one kink;
- exact type-1 slope and direct budget equality;
- standard and envelope baseline validity at all paper targets;
- 50-round exact validity;
- invalid controller with a concrete witness;
- exact decimal alpha parsing.

Committed blobs:
- source: `c0c57ec2d78b8060dfd3e09bd61f3939f645bb87`;
- test: `3a27bb8d1fd9f1790467896bffbaaef4548f0cb7`.

The numerical run used a local exact-rational reference matching the corrected formulas. It was not a clean detached checkout and no remote CI pass is claimed.

## Engineering implication

Keep arbitrary-precision rational arithmetic as a trusted reference oracle. A production verifier should use directed interval arithmetic or a separately proved fixed-point error budget, with differential vectors against this oracle. This work validates finite exchangeability only and does not imply robustness under arbitrary distribution shift.

Reference: Bai & Ramdas (2026), Appendix B.7, https://arxiv.org/abs/2610.35821
