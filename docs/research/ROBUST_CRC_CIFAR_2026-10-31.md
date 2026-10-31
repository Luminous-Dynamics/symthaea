# Distributionally robust utility optimization for CRC: five finite-population scenarios

**Date:** 2026-10-31  
**Status:** independent Python reference execution of the five-scenario maximin LP; all 1,378 augmented-state risk constraints remain feasible. The committed module has eight authored tests which have not yet been run against exact GitHub bytes. This is not a distribution-shift guarantee.

## Research shift

The nominal global LP optimizes expected specificity under one finite-pool composition. That is an oracle benchmark, not necessarily the right operational objective if the class/type proportions change.

The new formulation chooses one randomized common-depth policy P(a|D) that maximizes the *minimum* expected specificity across a predeclared set of finite population scenarios:

    maximize t
    subject to t <= E_s[specificity(policy)] for every scenario s

The finite-exchangeability risk constraints remain unchanged:

    sum_z S_z E[ell_z(A(S-e_z))] <= (n+1)*alpha

for every augmented multiset S. Therefore, utility robustness changes only the objective; it does not weaken the risk certificate.

## Frozen scenario set

All scenarios have population size 10,000 and three common-depth error types:

| Scenario | Fine correct | Superclass correct | Superclass wrong |
|---|---:|---:|---:|
| Nominal | 7,300 | 969 | 1,731 |
| Fewer superclass-correct | 7,300 | 500 | 2,200 |
| More fine-class errors | 6,500 | 1,200 | 2,300 |
| Higher accuracy | 8,000 | 800 | 1,200 |
| Mixed shift | 7,000 | 1,500 | 1,500 |

For each scenario, calibration count-vector weights are multivariate hypergeometric and the conditional next-test distribution is the residual population after the calibration count vector. The same randomized policy is optimized across the five scenarios.

This is a frozen, illustrative objective ambiguity set. It must be selected independently of test outcomes and does not cover arbitrary or unseen distributions.

## Results

At n=50, alpha=.10, action support {0,1,2}:

| Metric | Robust maximin LP | Nominal oracle LP | Equal-share 500 |
|---|---:|---:|---:|
| Worst scenario specificity | 41.59103% | 41.59103% | 41.58356% |
| Worst scenario risk | .09999518 | .09999518 | .09990744 |
| Worst scenario mean node size | 21.88393 | 21.88393 | 21.88593 |

The robust LP improves worst-case specificity over equal-share by about **0.00747 percentage points**. The nominal oracle LP is essentially tied with the maximin solution on this scenario set (about (8.1	imes10^{-9}) absolute specificity difference). Thus the nominal global policy is already unexpectedly robust over these five finite mixtures.

Scenario-by-scenario specificity:

| Scenario | Robust | Nominal oracle | Equal-share |
|---|---:|---:|---:|
| Nominal | 53.11683% | 53.11684% | 53.10382% |
| Fewer superclass-correct | 43.99505% | 43.99506% | 43.99151% |
| More fine-class errors | 41.59103% | 41.59103% | 41.58356% |
| Higher accuracy | 65.52343% | 65.52453% | 65.51763% |
| Mixed shift | 56.22941% | 56.22948% | 56.22924% |

All five robust-policy expected risks are at or below alpha=.10. The actual risk guarantee is stronger than this scenario table: the LP enforces every one of the 1,378 augmented-multiset constraints. The five scenarios appear only in the maximin utility objective.

## Optimization receipt

The HiGHS robust LP completed in 36 ms. The independent KKT receipt reports:
- primal inequality violation: (8.9	imes10^{-16});
- primal equality residual: 0;
- dual sign violation: 0;
- stationarity residual: (9.99	imes10^{-10});
- complementary-slackness residual: (4.21	imes10^{-9});
- primal-dual gap: (2.22	imes10^{-16}).

The separately recomputed equal-share controller passed every augmented state with maximum budget excess (2.66	imes10^{-15}).

## What this means

This is a modest but useful robustness improvement. It gives us a third comparison target:
1. equal-share recertification (distribution-free validity, not utility-optimized);
2. nominal global LP (optimal for one reference finite population);
3. scenario-robust maximin LP (optimal for a declared finite ambiguity set).

The nominal and robust LPs are almost tied in this pilot, which suggests the nominal solution already protects these five scenarios. That is useful evidence, not proof about all population shifts.

## Guarantees and limits

- The risk constraints are universally checked over the finite three-type augmented-state space and retain finite-exchangeability expected-risk validity.
- The utility objective is robust only over these five named finite populations. There is no distribution-shift guarantee outside this scenario set.
- The scenarios must be frozen independently of test outcomes.
- This is the common-depth family only; confidence-adaptive per-image thresholds are separate.
- No high-probability or conditional-risk guarantee is claimed.
- The local numerical replay used an independent scratch implementation of the LP equations. Eight tests are authored in the branch but have not been run against exact committed source bytes; no remote CI pass is claimed.

## Files

- `validation/robust_crc_cifar.py`
- `validation/test_robust_crc_cifar.py`
- `validation/ROBUST_CRC_CIFAR_2026-10-31_RESULTS.json`
- `docs/research/ROBUST_CRC_CIFAR_2026-10-31.md`
