# RCAF participant-level sign-test power planning V0

**Status:** Exact binomial design calculations; not an observed biological result.  
**Date:** 2026-11-21

## 1. Purpose

RCAF compares participant-paired within-state and between-state similarity. Its exact sign test asks whether positive paired gaps are more common than negative paired gaps, excluding exact-zero ties.

This note quantifies one narrow design question: how many **non-tied independent participants** are needed for useful power under a specified probability of a positive paired gap?

It does not power the median-gap bootstrap, estimate a clinically meaningful difference, account for effect magnitude, or replace a design-specific simulation.

## 2. Exact test

Let (g_i = S_{within,i} - S_{between,i}). Define:

- (n_{eff}): number of participants with (g_i 
e 0);
- (K): number of positive gaps among these non-tied participants;
- (q = P(g_i > 0 mid g_i 
e 0)).

Under the null, (K sim mathrm{Binomial}(n_{eff}, 0.5)). For a two-sided exact sign test:

[
p = minleft(1,;2sum_{j=0}^{min(K,n_{eff}-K)}
inom{n_{eff}}{j}2^{-n_{eff}}ight).
]

Power at alternative (q) is (P_q(p le 0.05)). The table below enumerates the exact binomial distribution; there is no normal approximation.

## 3. Exact power grid

<small>Two-sided alpha = 0.05; all participants independent; no participant-level ties among the (n_{eff}) used in the test.</small>

| Non-tied participants | q=0.60 | q=0.65 | q=0.70 | q=0.75 | q=0.80 |
|---:|---:|---:|---:|---:|---:|
| 23 | 0.125 | 0.254 | 0.440 | 0.654 | 0.840 |
| 30 | 0.177 | 0.358 | 0.589 | 0.803 | 0.939 |
| 50 | 0.237 | 0.506 | 0.782 | 0.945 | 0.994 |
| 80 | 0.369 | 0.724 | 0.941 | 0.995 | 1.000 |
| 100 | 0.462 | 0.828 | 0.979 | 0.999 | 1.000 |
| 150 | 0.663 | 0.955 | 0.999 | 1.000 | 1.000 |
| 200 | 0.788 | 0.988 | 1.000 | 1.000 | 1.000 |

Approximate minimum non-tied sample sizes for at least 80% power under this test:

| Assumed q | Approximate n_eff |
|---:|---:|
| 0.60 | 199 |
| 0.65 | 90 |
| 0.70 | 49 |
| 0.75 | 30 |
| 0.80 | 20 |

The exact-test power is discrete and is not strictly monotonic at every adjacent sample size because the rejection boundary changes in integer steps. The figures are planning aids, not a substitute for a locked calculation script.

## 4. What this says about small cohorts

For (n_{eff}=23), the sign test is weak for moderate directional consistency:
- if (q=0.60), power is approximately 12.5%;
- if (q=0.70), power is approximately 44%;
- if (q=0.75), power is approximately 65%;
- power exceeds 80% only around (q=0.80).

Thus, a 23-person reanalysis can still provide useful participant-level effect distributions and method-reproduction evidence, but a non-significant sign test should not be interpreted as evidence of no state difference. Conversely, an extremely small exact p-value at the all-positive boundary does not establish a large effect, causal explanation or consciousness interpretation.

## 5. Ties and missingness

The sign test excludes exact-zero participant gaps. Its effective sample is therefore not necessarily the recruited sample. Plan for this explicitly:

[
N_{recruited} approx rac{n_{eff}}{1-r_{tie}-r_{missing}}
]

only when tie and missingness rates are plausible and independent of the outcome. This approximation is not valid when missingness or ties depend on condition, data quality, participant or effect magnitude. In those cases, simulate the full sampling and QC process.

Do not turn near-zero gaps into ties after inspecting the data. Any practical equivalence margin must be fixed before analysis and analysed as a separate estimand.

## 6. Required design practice

1. Predeclare the minimum meaningful paired gap and the expected distribution of participant gaps.
2. Plan both the sign-test consistency target and the median-gap confidence-interval precision.
3. Use participant-level, not segment-level, sample size.
4. Include expected non-tie and missingness rates in the recruitment target.
5. Simulate hierarchical participant/session/run/segment effects when those levels are sampled.
6. Preserve site/device holdouts; do not use all participants for training and then call cross-validation a new external replication.
7. Report achieved precision even when the formal test is underpowered.

The sign test is one complementary diagnostic. It is not a power calculation for every RCAF outcome, does not encode effect magnitude, and does not prove causality.

## 7. Reproducibility

The machine-readable table is in `RCAF_SIGN_TEST_POWER_V0.json`. The calculations use the exact binomial probability under the stated alternative; no neural data were used.
