# SDCB noisy Bernoulli sample-complexity and induced anytime-test fixture v1

**Status:** exact-rational Rust reference authored; not compiled, run or CI-qualified. Synthetic thresholded observation channel only.

## Why this slice matters

The structural pole-swap work establishes that some experiments are exactly non-identifying and that a true sensor-inlet calibration breaks the symmetry in the idealized model. That is not yet a finite-sample statement.

This fixture adds a noisy measurement channel whose two worlds are statistically distinguishable but have overlapping outcome support. It asks how many observations are required to certify a decision at a declared type-I level, how contamination changes that requirement, and whether a fixed-horizon terminal test can be induced into an anytime-valid process without changing its terminal power.

The channel probabilities are synthetic known-answer values. They are **not** inferred from the #14003 physical response or a real instrument. They provide an exact finite-outcome reference for later Gaussian/CDF enclosure work.

## Frozen experiment channel

Let X_i in {0,1} be a thresholded sensor response.

- Null / World B: X_i ~ Bernoulli(p0), nominal p0=1/4.
- Alternative / World A: X_i ~ Bernoulli(p1), nominal p1=3/4.
- Conditional observations are iid within this base M0 fixture.
- alpha=1/20, target worst-case power 4/5.

The likelihood ratio is monotone in the success count S_n=sum_i X_i. The exact Neyman–Pearson test fills the largest success counts first and randomizes at no more than one boundary count.

## Exact sample-size / TV-radius frontier

For Bernoulli distributions, TV(Bernoulli(p),Bernoulli(q))=|p-q|. With symmetric radius epsilon, the composite families are:

p0 in [1/4-epsilon, 1/4+epsilon],
p1 in [3/4-epsilon, 3/4+epsilon].

For a nondecreasing count test, the least-favourable null is the upper null endpoint and the least-favourable alternative is the lower alternative endpoint. Exact rational results:

| epsilon | Worst null | Worst alternative | Minimum n for power >=0.8 | Power at n_min |
|---:|---:|---:|---:|---:|
| 0 | 1/4 | 3/4 | 6 | 1079/1280 = 0.84297 |
| 1/20 | 3/10 | 7/10 | 10 | 21309567/25000000 = 0.85238 |
| 1/12 | 1/3 | 2/3 | 14 | 726367/885735 = 0.82007 |

The predecessor sample sizes do not meet the target:

| epsilon | n_min-1 | Exact predecessor power |
|---:|---:|---:|
| 0 | 5 | 471/640 |
| 1/20 | 9 | 11809567/15000000 |
| 1/12 | 13 | 1398367/1771470 |

This tiny exact family illustrates why sample complexity should be reported as a function of the uncertainty radius. It is not a general sample-complexity theorem. The broader literature has established that robust-testing sample complexity can be highly sensitive to contamination-radius perturbations; see Vallinayagam, Pensia & Jog (2026), https://arxiv.org/abs/2605.24741.

## Identical-law negative control

For passive or pole-symmetric upstream controls, set both worlds to the identical law Bernoulli(1/2). The constant test phi(x)=alpha achieves power alpha, and the common-law witness upper-bounds every size-alpha test by alpha. More samples cannot create robust distinction from identical observation families.

## Induced anytime-valid process

For each frozen terminal test phi_N, define

g_n(s)=E_{p0*}[phi_N(S_N) | S_n=s],

where p0* is the least-favourable upper null endpoint. This is the conditional-expectation induction from Koning & van Meer (2026), not a new SDCB theorem: https://doi.org/10.1093/jrsssb/qkag050.

The exact verifier checks:

1. g_0=alpha;
2. g_N(s)=phi_N(s) for every terminal count;
3. (1-p0*)g_(n+1)(s)+p0*g_(n+1)(s+1)=g_n(s) at every public history;
4. g_(n+1)(s+1)>=g_(n+1)(s) at every history.

Because the conditional row is increasing in the next Bernoulli observation, for every null p<=p0*,

E_p[g_(n+1)(S_(n+1)) | S_n=s] <= g_n(s).

Thus the induced process is a supermartingale throughout the declared composite null. Optional stopping at any bounded public stopping time preserves the alpha bound, and the full-horizon terminal test has exactly the same power as the original fixed-horizon NP test.

This reproduces a published construction and supplies an SDCB-specific robust-radius / sample-size regression. It does **not** assert that induction reduces the sample requirement. Any acquisition policy and expected-cost comparison must be reported separately.

## Industry and SOTA comparison

This is the statistical-method layer, not a replacement for an industry sensor protocol. The external tournament must retain:

- EPA base field testing and enhanced laboratory testing, including reference collocation and multiple identical units;
- CEN/TS 17660-2:2024 device-level PM sensor classification;
- GUM/JCGM uncertainty propagation and explicit reference ancestry;
- held-out device, site and season transfer slices.

Compare the fixed-horizon NP test, induced test, SPRT/e-process baselines, TV-radius robust endpoints, #14218 finite-family authority and the later #13904 continuous-time sensor model. Keep physical validation and SOTA superiority as separate gates.

Relevant comparators:
- Gül (2026), finite-sample minimax robust tests under TV/band uncertainty: https://arxiv.org/abs/2602.19803
- Vallinayagam, Pensia & Jog (2026), robust binary sample complexity and contamination sensitivity: https://arxiv.org/abs/2605.24741
- Koning & van Meer (2026), induced anytime-valid tests: https://doi.org/10.1093/jrsssb/qkag050

## Validation boundary

The Rust file contains seven authored tests for exact power, minimal n, predecessor failure, TV-radius endpoints, terminal-test match, martingale identities, monotonicity and composite-null supermartingale inequalities. The test source is authored but has not been compiled or executed in this environment. The exact fractions were separately replayed with Python Fraction arithmetic.

This is an exact Bernoulli finite-outcome benchmark. It is not a physical sensor calibration, continuous Gaussian likelihood, metrological traceability claim, external industry result or SOTA result.
