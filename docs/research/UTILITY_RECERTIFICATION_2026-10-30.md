# Utility-aware recertification under the closed e-BH certificate

**Date:** 2026-10-30  
**Status:** independent cutting-plane prototype; 9/9 local tests passed. Exploratory capacity experiments only; not a SOTA claim.

## Research question

RC-eBH optimizes expected total rejection count over the monotone randomized procedures certified by the closed e-BH e-collection. That is not necessarily the operational objective.

A clinical panel, security operations team, laboratory or review board may have a fixed follow-up capacity \(B\). Its utility is the number of actionable candidates:

\[
U_B(\kappa)=\min(\kappa,B),\qquad
\mathbb E[U_B\mid e]=\sum_{r=1}^{B}q_r.
\]

This branch optimizes that objective over the **same complete closed-eBH certified menu**. It does not claim to improve RC-eBH for its original objective; RC-eBH is optimal for expected total rejection count within that menu.

## LP formulation

For ordered e-values \(e_{[1]}\ge\cdots\ge e_{[K]}\), let \(q_r=\Pr(\kappa\ge r\mid e)\), per-rank FDP charges \(c_r=\mathbb E[\mathbf 1\{\kappa\ge r\}/\kappa\mid e]\), and cumulative charges \(S_r=\sum_{j=r}^K c_j\), with \(S_{K+1}=0\).

\[
c_r=S_r-S_{r+1},\qquad q_r=rS_r-(r-1)S_{r+1}.
\]

The monotone rejection-count distribution requires \(S\) nonnegative, nonincreasing and convex, with \(S_1\le1\). For every nonempty rank subset \(T\), the closed-eBH certificate requires

\[
\sum_{r\in T}(S_r-S_{r+1})
\le \frac{\alpha}{|T|}\sum_{r\in T}e_{[r]}.
\]

For nonnegative utility \(u_r\), maximize \(\sum_r u_r q_r\). Uniform utility recovers RC-eBH; capacity \(B\) uses \(u_r=1\) for \(r\le B\) and zero afterwards.

## Critical correction: suffix constraints alone are insufficient

An earlier draft used only suffix subset constraints. That can be wrong for arbitrary utility: a decreasing convex candidate \(S\) can satisfy every suffix bound yet violate a non-suffix intersection.

The committed optimizer therefore uses a separation oracle. For a candidate \(S\), let \(c_r=S_r-S_{r+1}\). For each subset size \(t\), the most violated subset consists of the \(t\) largest values of

\[
c_r-\frac{\alpha e_{[r]}}{t}.
\]

If their sum is positive, add that exact subset inequality to the LP and resolve. Continue until no cardinality has a violated subset. Suffix constraints are only an efficient warm start, not treated as the complete menu.

## Local correctness evidence

**9/9 local tests passed.**

- Uniform utility recovers the independent RC-eBH implementation over 190 random families, \(K=2,\ldots,20\); maximum absolute tail difference \(4.44\times10^{-16}\).
- Capacity \(B=K\) also recovers RC-eBH.
- All-subset constraints checked exhaustively for 360 random utility families, \(K=2,\ldots,10\); maximum observed violation \(2.22\times10^{-16}\).
- Cutting-plane optimum matches an independent LP with every nonempty subset constraint inserted upfront for 56 small random families.
- A non-suffix counterexample forces an additional cut beyond the initial suffix constraints.
- Rejection-count probabilities, convex-minorant helper and shared-uniform sampler pass boundary tests.

These finite checks verify the algorithmic reduction, not the statistical theorem.

## Exploratory capacity results

Simulation uses \(K=100\), \(\alpha=.1\), \(E_i=\exp(2Z_i-2)\), null \(Z_i\sim N(0,1)\), alternative \(Z_i\sim N(\mu,1)\), and Gaussian equicorrelation \(\rho\). RC-eBH and the capacity LP use a shared uniform.

| Scenario | Method | FDR | Power | Mean total rejections | Actionable slots | True actionable |
|---|---|---:|---:|---:|---:|---:|
| 10 alternatives, \(\mu=2.5,\rho=0\), n=200 | RC-eBH | 9.99% | 33.25% | 6.00 | 3.61 | 3.12 |
| same | Capacity LP | 7.81% | 32.95% | 4.21 | 3.79 | 3.22 |
| 50 alternatives, \(\mu=2.5,\rho=.5\), n=200, B=10 | RC-eBH | 6.66% | 68.39% | 39.51 | 8.61 | 8.61 |
| same | Capacity LP | 0.00% | 22.04% | 11.02 | 8.70 | 8.70 |
| 50 alternatives, \(\mu=2.5,\rho=0\), n=200 | RC-eBH | 6.79% | 69.00% | 38.65 | 10.00 | 9.98 |
| same | Capacity LP | 0.38% | 23.96% | 12.03 | 10.00 | 9.98 |
| 90 alternatives, \(\mu=2,\rho=0\), n=200 | RC-eBH | 2.25% | 66.93% | 62.24 | 10.00 | 10.00 |
| same | Capacity LP | 0.04% | 13.33% | 12.00 | 10.00 | 10.00 |
| 50 alternatives, \(\mu=2.5,\rho=.5\), n=200, B=20 | RC-eBH | 6.94% | 65.48% | 38.01 | 15.67 | 15.67 |
| same | Capacity LP | 0.00% | 35.68% | 17.84 | 16.05 | 16.05 |

For the sparse 10-alternative setting, the LP adds 0.175 actionable slots per replicate (paired 95% interval 0.088–0.262) and 0.095 true actionable discoveries (0.042–0.148), with fewer total rejections. For \(q=50,\rho=.5,B=20\), it adds 0.38 true actionable discoveries (0.165–0.595).

Where RC-eBH already fills the capacity, the LP produces no actionable-count gain. It can sharply reduce total rejections and empirical FDR, but at a large cost in conventional power. This optimizes **actionable output**, not the number of all true alternatives discovered. Each alternative cell has only 200 repetitions; these are exploratory, not confirmatory.

### All-null stress

For \(q=0,\rho=0,B=10,n=500\), both methods produced empirical FDR 10.6%, with a replicate-FDP interval 7.90–13.30%. Both had zero true actionable discoveries. The LP increased actionable slots from 0.556 to 0.646, all false in this all-null setting.

The interval crosses nominal 10%. This is inconclusive at 500 trials, not evidence of a guarantee violation. It illustrates why slot count does not itself encode truth or downstream value.

## Computational cost

At \(K=100\), the LP used about 100–435 active subset cuts. Mean solve times were roughly 7.6–76 ms per e-value vector; measured p95 values were 10–107 ms. These are Python/HiGHS research measurements, not production comparisons.

A production implementation would need a Rust separation oracle and warm-startable incremental LP or specialized solver. A batch solve per family is not an online decision algorithm.

## Next gates

1. Increase to at least 5,000 paired repetitions on frozen capacity regimes and add sharp all-null boundary cases.
2. Evaluate \(B\in\{1,5,10,20,50,K\}\), including where RC already saturates capacity.
3. Add predeclared hypothesis-specific utilities and measure true actionable utility under priority misspecification.
4. Compare against randomized top-B truncation of RC-eBH and other certified capacity policies.
5. Compare cutting-plane LP with the full all-subsets LP for \(K\le10\), tracking cuts/runtime as \(K\) grows.
6. Keep fixed-family FDR separate from sequential SupFDR. No repeated-snapshot or SupFDR claim.
7. Port to Rust only after theorem review and numerical oracle tests.

## Reproduction

    PYTHONPATH=validation python -m unittest -v test_utility_recertification.py
    PYTHONPATH=validation python validation/utility_recertification.py --output /tmp/utility-recertification.json

Local SHA-256:
- source: 51f9f03a9215094a819e4455b4468499f69201d72ace5724975bfbb98e6c9799
- tests: 87d3d49d991060204c733c7b3f2d01eb519bb49e1e2c5a6e498469b3803c48df
- pilot JSON: aa997715d743084b56b63ad9d7538ea9b1a91480ec7b8eea2478e686e53a7f2e
