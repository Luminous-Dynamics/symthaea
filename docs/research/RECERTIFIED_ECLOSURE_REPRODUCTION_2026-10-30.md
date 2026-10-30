# Independent reproduction: recertified randomized e-closure

**Date:** 2026-10-30  
**Status:** 11/11 local tests passed; three 10,000-repetition confirmatory cells completed; not the full 50,000-repetition paper replication and not a SOTA claim.

## 1. Research result

Bai & Ramdas, *Recertifying decisions via e-closure: uniform improvements of conformal risk control and randomized e-BH*, arXiv:2610.35821 (submitted 2026-10-26), develop a general decision-theoretic principle: represent a valid procedure by a family of e-value certificates, identify unused conditional risk budget, and enlarge the certified action menu without changing its risk guarantee.

For multiple testing, the paper derives two randomized recertification methods:

- RU-eBH: recertified U-eBH, which improves shared-uniform U-eBH using the original e-BH e-collection.
- RC-eBH: recertified closed e-BH, which uses the stronger mean e-collection from closed e-BH.

The algorithms use the same shared uniform U. Pointwise in the observed e-values and U, the rejection counts are nested:

\[
\mathrm{eBH}\subseteq\mathrm{U\mbox{-}eBH}
\subseteq\mathrm{RU\mbox{-}eBH}
\subseteq\mathrm{RC\mbox{-}eBH}.
\]

RU-eBH maximizes the rejection-count tails among monotone randomized procedures certified by the e-BH collection. RC-eBH improves that menu using the larger closed e-BH collection. Its expected number of rejections is at least that of deterministic closed e-BH. Neither method is an online or SupFDR procedure.

Paper: https://arxiv.org/abs/2610.35821

## 2. Independent algorithm implementation

The research module implements all eight methods in the paper's comparison:

1. eBH
2. minimally adaptive eBH (MA-eBH)
3. closed eBH
4. anchored eBH
5. U-eBH
6. J-eBH
7. RU-eBH
8. RC-eBH

All methods use the same marginal e-values. The simulation uses the paper's likelihood-ratio model:

\[
E_i=\exp(2Z_i-2),\quad
Z_i\sim N(0,1)\text{ under a null},\quad
Z_i\sim N(\mu,1)\text{ under an alternative}.
\]

This gives E[E_i]=1 under each null. Cross-hypothesis dependence is either independent or Gaussian equicorrelation. The theoretical methods allow arbitrary dependence among valid marginal e-values; the simulations only explore a subset of dependence structures.

### RU-eBH

For ordered e-values e[1] >= ... >= e[K], the conditional rejection-count tails are

\[
q_r^*=
\min\left\{
1,\frac{\alpha}{K}
\left(r e_{[r]}+\sum_{j>r}e_{[j]}\right)
\right\}.
\]

Draw one U independent of the e-values and reject the largest r for which U <= q_r*. Computing suffix sums after sorting takes O(K log K) time.

### RC-eBH

Let

\[
m_s=\frac{1}{K-s+1}\sum_{j\geq s}e_{[j]},\quad
\beta_1=\min(1,\alpha m_1),\quad
\beta_s=\alpha m_s\;(s\geq2),\quad
\beta_{K+1}=0.
\]

Let S-sharp be the greatest convex minorant of the sequence beta_s. Its rejection-count tails are

\[
q_r^\sharp=rS_r^\sharp-(r-1)S_{r+1}^\sharp.
\]

The lower convex hull is computed in O(K) operations after sorting. Using the same U as RU-eBH gives the pointwise nesting above.

### Deterministic closed e-BH and anchoring

Closed e-BH is implemented independently from the mean-consistency equations. For every candidate top-r rejection set, the verifier checks all possible intersections via a convex one-dimensional minimization in the number of non-rejected hypotheses. The implementation is exact to numerical tolerance and is intended as a research oracle, not yet a production-speed implementation.

Anchored e-BH uses the intersection certificate from Ramdas (2026). The existing eclosure GPL-3 Python/R project was inspected for API and mathematical comparison only. No external source code was copied into this branch.

## 3. Correctness and independent checks

**Local result: 11/11 tests passed.**

The suite tests:

- Greatest convex minorant is below the input and convex.
- Closed e-BH agrees with exhaustive mean-closure checking for 50 random families at each K=1,...,9.
- RC-eBH's charge constraints pass exhaustive checks for 30 random families at each K=2,...,10.
- RU-eBH's per-rank charges satisfy the e-BH certificate constraints.
- eBH, U-eBH, RU-eBH and RC-eBH rejection counts are nested for the same shared U.
- Anchored and closed e-BH contain the eBH baseline.
- A strict RU witness: for K=2, alpha=.05, e=(39.5,1), U=.99, U-eBH rejects zero while RU-eBH rejects one and RC-eBH rejects at least one.
- Small-simulation reproducibility and the complete eight-method result schema.

The code is an independent implementation from the published formulas. Small-family exhaustive checks validate algorithmic reductions; they do not prove the paper's statistical theorems.

## 4. Benchmark protocol and scale

The paper's main comparison uses K=100, alpha=.1, 50,000 datasets per cell, 10/50/90 false nulls, correlations 0/.5/.9, and signal means from 1 to 4.

This local work is deliberately staged:

| Cohort | Cells | Repetitions per cell | Purpose |
|---|---:|---:|---|
| Development grid | 18 | 1,000 | q=10/50/90, rho=0/.5/.9, signal mean 2/2.5 |
| Held-out checks | 8 | 3,000 | Separate seeds, repeated representative settings |
| Confirmatory | 3 | 10,000 | Independent-statistic representative cells |
| Global-null stress | 3 | 3,000 | q=0, rho=0/.5/.9 |

The full 50,000-run, all-correlation, mean-1-to-4 replication has not been completed. The three confirmatory cases use rho=0; correlated cases are lower-repetition exploratory checks. The machine-readable JSON records exact settings, seeds, FDR/power intervals, paired differences and nesting checks.

## 5. Confirmatory results

Each row uses 10,000 paired repetitions, K=100, alpha=.1, independent statistics (rho=0).

| False nulls | Signal mean | eBH | Closed eBH | U-eBH | RU-eBH | RC-eBH | RC FDR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.5 | 4.60% | 6.35% | 21.40% | 31.61% | 32.17% | 8.87% |
| 50 | 2.5 | 25.91% | 58.45% | 54.71% | 62.99% | 69.14% | 6.30% |
| 90 | 2.0 | 10.54% | 55.05% | 41.08% | 51.74% | 63.97% | 1.93% |

Paired RC-minus-RU power improvements, with 95% Monte Carlo intervals:

- 10 false nulls: +0.563 percentage points [0.515, 0.611].
- 50 false nulls: +6.147 percentage points [6.066, 6.228].
- 90 false nulls: +12.230 percentage points [12.089, 12.371].

Paired RC-minus-closed-eBH improvements:

- 10 false nulls: +25.823 percentage points [25.249, 26.397].
- 50 false nulls: +10.687 percentage points [10.331, 11.042].
- 90 false nulls: +8.920 percentage points [8.468, 9.372].

The 90-false-null, signal-2 result reproduces the paper's headline reference closely: RC-eBH 63.97% versus RU-eBH 51.74% and closed eBH 55.05%. The sparse 10-false-null result likewise matches the reported RU-eBH power near 32% and U-eBH near 22%.

The mechanism is regime-dependent. With few false nulls, the e-BH certificate's alpha |T|/K budget is already relatively tight and RU-eBH and RC-eBH are nearly equal. With many false nulls, the larger closed e-BH e-collection has more available budget; RC-eBH can spend it through a randomized action. Its empirical FDR can exceed alpha |T|/K, while remaining below the overall alpha level as guaranteed by the method.

## 6. All-null stress and guarantee boundaries

At alpha=.1, K=100, and 3,000 all-null repetitions, empirical FDR:

| Dependence | eBH | U-eBH | J-eBH | RU-eBH | RC-eBH |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.10% | 3.97% | 9.83% | 9.57% | 9.57% |
| 0.5 | 0.07% | 3.33% | 9.07% | 8.93% | 8.93% |
| 0.9 | 0.40% | 3.43% | 6.37% | 7.17% | 7.17% |

All point estimates are below the nominal 10%; Monte Carlo intervals are stored in the JSON. This is a finite-simulation screen, not a proof of validity. Formal guarantees rely on each null input being a valid e-value and on the paper's randomized-action/closure arguments.

**Important:** randomized U/RU/RC methods are offline fixed-family procedures. Recomputing them at successive time points does not establish SupFDR. Their shared-U nesting is a pointwise property for one fixed observed e-value vector, not an anytime-valid process.

## 7. Comparison to the state of the art

This is an independent reproduction of a very recent method, not a novel SOTA algorithm. The research result itself already says RC-eBH maximizes expected rejections over the monotone randomized rules certified by the closed e-BH e-collection. Our measured result is consistent with the published examples.

A credible claim beyond SOTA would require a new certificate or decision objective, not merely retuning the shared uniform or the convex minorant. The most promising follow-on is utility-aware recertification:

- Keep all closed-eBH certificate constraints.
- Replace the objective "maximize expected number of rejections" with predeclared, nonnegative utility for each ranked rejection.
- Solve the resulting linear optimization over the convex cumulative-charge sequence.
- Compare on held-out settings against RC-eBH, because RC-eBH is optimal for unweighted expected rejection count, not every possible utility.

This is a research proposal, not an implemented or validated improvement. It may be valuable for settings where false discoveries have unequal costs or discoveries differ in scientific value. Any state-dependent or data-adaptive utility must be declared and certified correctly.

## 8. Reproduction

    PYTHONPATH=validation python -m unittest -v test_recertified_eclosure.py
    PYTHONPATH=validation python validation/recertified_eclosure.py --output /tmp/recertified-eclosure-results.json --repetitions 3000

Committed artifacts:
- validation/recertified_eclosure.py
- validation/test_recertified_eclosure.py
- validation/RECERTIFIED_ECLOSURE_2026-10-30_RESULTS.json

Local SHA-256:
- source: edb1e657136c09290be490ed4c94491851562d02aea0b78279f54bb4d9636fe0
- tests: e5c07516e36a4e7f7e689acb3ef2d5b1e234eb7d2e5de50338d7fc64c56d4706

## 9. Next gates

1. Run the full 50,000-repetition, 18-cell paper design, including all signal means 1-4 and all three dependence settings.
2. Differentially compare exact closed-eBH decisions against the official eclosure implementation in a separate GPL-3 environment; keep that implementation external and do not copy its code.
3. Stress arbitrary dependence with common-factor, block, antithetic and extremal constructions, not only Gaussian equicorrelation.
4. Measure per-cell FDR, power, paired gain, expected rejection count, decision latency and memory.
5. Build utility-aware randomized recertification as a separate method; freeze utility and optimization choices before held-out evaluation.
6. Keep offline FDR and sequential SupFDR in separate benchmark tracks. Do not turn this branch into a production or online claim until new proofs exist.
