# Running-maximum group aggregators and a calibrated power mixture

**Date:** 2026-10-30  
**Status:** Independent implementation and local synthetic pilot; 11/11 local tests passed. Not a theorem proof, not a full reproduction of the authors' study, and not a SOTA claim.

## Research shift

The 2026 *Carefree multiple testing with e-processes* paper argues that simultaneous sequential testing should use each process' running maximum, because this preserves the strongest evidence for every stream even when data collection is asynchronous. Direct e-BH on those maxima is not valid in general; the paper uses marginal adjusters to restore FDR-sup control under arbitrary dependence.

A separate SAVI 2026 poster by Tavyrikov, Goeman and de Heide presents a general tail bound for sums of running maxima of arbitrarily dependent e-processes. The author research page lists the related article as work in preparation. This branch independently implements the poster's formulas for research comparison; the result should not be described as peer-reviewed until the full paper is available.

References:
- Poster: https://riannedeheide.github.io/PosterSAVI2026Yury.pdf
- Published 2026 paper: https://doi.org/10.1214/26-EJS2546
- Author research page / publication status: https://riannedeheide.github.io/research

Let (V_k=\sup_t E_t^k), with each (E^k) an e-process on a common filtration and (\mathbb E[E_\tau^k]\leq a_k). The poster's main result is

\[
\Pr\left(\sum_{k=1}^K V_k>\lambda\right)
\leq \frac{2H_K}{\lambda}\sum_{k=1}^K a_k,
\qquad H_K=\sum_{r=1}^K r^{-1}.
\]

More generally, for a nondecreasing (f) with (f(0)=0), define (\mu) through
\[
\sum_{r=1}^K f(\mu/r)=\lambda,
\]
and
\[
\Lambda_K(\lambda)=\lambda\mu-
\sum_{r=1}^K r\int_0^{\mu/r} f(u)\,du.
\]
The poster gives
\[
\Pr\left(\sum_k f(V_k)>\lambda\right)
\leq \frac{\lambda}{\Lambda_K(\lambda)}\sum_k a_k.
\]

For the power family (f(x)=x^s), (H_K^{(s)}=\sum_{r=1}^K r^{-s}), this yields
\[
Z_s=
\frac{(\sum_k V_k^s)^{1/s}}
{\frac{s+1}{s}K(H_K^{(s)})^{1/s}},
\qquad
\Pr(Z_s\geq t)\leq 1/t
\]
when all (a_k=1). The harmonic factor grows like \(\log K\) for (s=1), but is bounded for (s>1\); for (s=2), (H_K^{(2)}\leq \pi^2/6\). The total normalization still scales linearly in K.

## One calibrated mixed-power statistic

A naive data-driven maximum over several (s) values needs an error-budget split. The general-(f) theorem allows a different construction: fix
\[
f(x)=\sum_j w_j x^{s_j},\quad w_j>0,
\]
then compute (F=\sum_k f(V_k)), solve
\[
F=\sum_j w_j H_K^{(s_j)}\mu^{s_j},
\]
and calculate
\[
Z_f=\frac{\Lambda_K(F)}{KF}
=\frac{\mu}{K}
\sum_j q_j\frac{s_j}{s_j+1},
\quad
q_j=\frac{w_jH_K^{(s_j)}\mu^{s_j}}{F}.
\]
This uses one theorem calibration rather than Bonferroni across the powers. The implementation solves for (\log\mu) by stable bisection and checks the pure-power limits in unit tests.

The default pilot uses (s\in\{1,2,4\}), weights proportional to (10^{-s}). These are fixed scale-tuned weights; they are not data-adaptive, not proven optimal, and not selected on an independent holdout.

## Executed local pilot

The generator simulates independent-over-time equicorrelated Gaussian innovations, with marginal (N(0,1)), common-filtration e-processes
\[
E_t^k=\exp\left(\sum_{u=1}^t(\mu X_{ku}-\mu^2/2)\right),
\quad V_k=\max(1,\sup_t E_t^k).
\]
The group is pre-registered. Under the all-null intersection, each coordinate is conditionally (N(0,1)) given the prior joint filtration, so each likelihood-ratio process is an e-process. This is one structured dependence test, not proof of arbitrary-dependence performance.

### Empirical all-null Type-I screen

(K=200), 500 timepoints, 3,000 repetitions per dependence setting, alpha=0.05:

| Stream dependence | Method | False rejections | Wilson 95% interval |
|---|---|---:|---:|
| Independent | All eight methods | 0/3,000 | 0–0.128% |
| Equicorrelation 0.3 | Bonferroni max | 1/3,000 | 0.006–0.189% |
| Equicorrelation 0.3 | A1, A2, direct s=1/2/4, portfolio, mixed-f | 0/3,000 each | 0–0.128% |

The zero counts do not demonstrate nominal calibration at 5%; they indicate that these particular finite-horizon tests are very conservative in this regime. The theoretical guarantee, if its assumptions hold, is the validity argument; simulation is a regression screen only.

### Group detection power

All rows use alpha=0.05, (\mu=0.1), 1,000 observations, equicorrelation 0.3, 500 repetitions. This is the probability of rejecting one pre-registered group's *global intersection null*, not per-hypothesis power, FDR or FDR-sup.

| Group size K | Signals in group | Bonferroni max | A1 mean | A2 mean | Direct s=1 | Direct s=2 | Direct s=4 / mixed-f |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 44.2% | 5.6% | 1.6% | 22.0% | 35.6% | 39.6% |
| 20 | 20 | 99.0% | 67.4% | 72.4% | 96.8% | 98.0% | 98.2% |
| 100 | 20 | 92.0% | 26.6% | 12.2% | 70.2% | 88.0% | 90.4% |
| 100 | 100 | 99.0% | 74.8% | 76.4% | 96.2% | 98.6% | 99.0% |
| 200 | 20 | 83.0% | 14.4% | 4.4% | 54.6% | 75.2% | 80.6% |
| 200 | 100 | 97.4% | 51.6% | 44.2% | 90.6% | 96.4% | 96.4% |

Wilson intervals are retained for each result in the JSON artifact. For example, K=200 with 20 signals gives direct s=4 power 80.6% (76.9–83.8%) versus A1 14.4% (11.6–17.7%), A2 4.4% (2.9–6.6%), and Bonferroni max 83.0% (79.5–86.0%).

Interpretation:
- Direct tail-bound aggregation substantially beats the simple mean of the two adjusted maxima for these group-level detection settings.
- It does not beat the simple Bonferroni max in the tested cases. Its strongest s=4 variant comes close, but is 2.4 percentage points lower for K=200 and 20 signals.
- The equal-alpha portfolio across s=1/2/4 pays a visible power cost compared with s=4 alone.
- The fixed mixed-f statistic tied s=4 in these selected settings; it has not yet demonstrated meaningful adaptation across sparsity regimes.
- These group-level results cannot be used to claim improved individual FDR or FDR-sup.

## Comparison boundary

This branch does not replace PR #9135's FDR-sup simulation or the published adjuster guarantee. It investigates a different task: testing a pre-registered group intersection null with each stream's own running maximum.

Relevant baselines:
1. Bonferroni max: (\max_k V_k/K), a strong and simple arbitrary-dependence familywise baseline.
2. Mean A1/A2-adjusted maxima: marginal e-value group baselines.
3. Direct power aggregators, s=1/2/4.
4. Fixed alpha-split portfolio over those powers.
5. Single calibrated mixed-f aggregator.

No SOTA claim is made. A fair comparison must match the error target, stopping-time rules, dependence model, and hypothesis-arrival assumptions.

## Next research gates

1. Independently verify the full general-(f) theorem algebra and assumptions; derive a tail-bound proof implementation that another reviewer can check.
2. Stress-test sharp/adversarial dependence, including asynchronous harmonic-ladder constructions, rather than only Gaussian equicorrelation.
3. Implement hierarchical/structured closure so group intersection certificates can produce individual FWER or true-discovery-proportion guarantees without naively enumerating (2^K) subsets.
4. Sweep group sizes, signal fractions, effect sizes, time horizons, heavy tails, misspecification and drift. Tune f only on a development split; freeze it before evaluation.
5. Compare against author code and available e-closure software under matched hypotheses and error metrics.
6. Only after those tests, port the core scoring and certificates to Rust and use exact-head tests in Symthaea.

## Reproduction

```sh
PYTHONPATH=validation python -m unittest -v test_multistream_group_aggregators.py
PYTHONPATH=validation python validation/multistream_group_aggregators.py \
  --output /tmp/multistream-group-aggregators.json
```

SHA-256:
- source: `425f4ac336cbd65c7e2bab5be00e7c92533797c0b89197024239a8a9a5c9cdc6`
- tests: `6a38149574212eea7672c87fb4a1b11688b19d11fdfbcf0504792cbdeda7c636`
- pilot JSON: `ca3909476d7aa2549e539cc425a6352058e39d50e2f7246639a0a004421ee54f`
