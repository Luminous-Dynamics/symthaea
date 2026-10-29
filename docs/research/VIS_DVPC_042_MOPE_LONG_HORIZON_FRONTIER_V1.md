# VIS-DVPC-042 — MOPE long-horizon power and decision frontier

**Status:** implementation requires exact-head Rust qualification; simulation estimates are not coverage proofs.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8869

## Why this follows the exact-path audit

VIS-DVPC-041 tests exact path probability and endpoint chronology at horizon 8. A preliminary independent reference port yielded no endpoint crossings on those short controls, so the interval stayed [0,1]. Such a result is calibrated but not decision-useful. This increment studies the coverage–width–power–latency frontier at longer horizons instead of relying on a vacuous high-coverage number.

## Reproducible design

Use 2,000 sequences per DGP, horizon 512, with reported snapshots at 32, 128 and 512. The standard-library-only SplitMix64 PRNG has a fixed base seed and scenario/replicate mixing formula. No external datasets, dependencies or asynchronous workers are required. A deterministic short replay compares all statistical output while excluding wall-clock timing rows, which naturally vary between runs. Per-method elapsed time is measured separately from the shared simulator; the MOPE figure deliberately includes the expensive direct-wealth certificates used in this audit and therefore must not be presented as optimized deployment throughput.

| DGP | Logger P(A=1) | Reward means (A=0,A=1) | True value | W |
|---|---:|---:|---:|---:|
| healthy null | 0.50 | (0.20, 0.80) | 0.50 | 1 |
| healthy alternative | 0.50 | (0.50, 0.80) | 0.65 | 1 |
| weak-overlap null | 0.99 | (0.20, 0.80) | 0.50 | 50 |
| weak-overlap alternative | 0.99 | (0.50, 0.80) | 0.65 | 50 |

The target policy is P_pi(A=1)=0.5 for all scenarios. Each sample uses known-logger importance ratio w=pi(a)/h(a) and observed IPS reward X=wR. Every method sees the same generated sequence within each replicate.

## Comparators

- **MOPE analytic root:** two one-sided streaming confidence sequences, alpha_side=0.025. Endpoint advances use an analytic root of the quadratic log-wealth lower bound, gated by direct polynomial and product-wealth certificates.
- **MOPE upstream-style grid:** evaluates the actual log-wealth process on 1,000 evenly spaced values in [0,1] and picks the first grid point whose half-initialized log wealth is below log(1/alpha_side). The value-grid calculation follows the endpoint lookup structure of the original public `n17s/mope/opebet.py` implementation; to keep the comparison focused, it uses this branch's same streaming optimizer rather than the original Python/Numba Cholesky optimizer. It is therefore a **grid-endpoint comparator**, not an exact port of the full published implementation.
- **Reused Wald:** a conventional pointwise 95% normal interval re-evaluated at every look. It is explicitly an invalid anytime-valid comparator.
- **Hoeffding alpha spending:** per-side delta_t = 0.05 / [2 t(t+1)], radius W sqrt(log(1/delta_t)/(2t)); each tail spends alpha/2 over all looks.
- **Empirical-Bernstein alpha spending:** a two-sided interval with delta_t = 0.05 / [t(t+1)] and the SCOPE-RL documented finite-sample empirical-variance radius using W and sample variance; the two-sided per-look allocations sum to alpha.

For both time-uniform baselines, raw intervals are combined with a running intersection at every observation: lower endpoints can only increase and upper endpoints can only decrease. This matches the sequential set semantics used for both MOPE variants and avoids making the comparison artificially unfavorable by measuring a fresh non-nested bound at each look. The Wald comparator intentionally remains a reused pointwise interval. All intervals are clipped to the known target-value range [0,1]. Clipping alone does not make Wald anytime-valid.

The upstream repository is at https://github.com/n17s/mope. Its `wealth_lb_2d` uses a 1,000-value grid for actual wealth while choosing bets from the quadratic lower bound and an optimizer. This grid comparator isolates the value-grid-versus-analytic-root question while sharing our Rust optimizer across the two MOPE variants. Its output must not be described as a line-for-line reproduction.

## Metrics

At each horizon and for every method:
- pointwise coverage at that look and sequence-wide coverage through that look;
- Wilson 95% interval and Monte Carlo standard error for each reported probability;
- mean interval width and standard error, mean lower/upper endpoint;
- empty/reversed interval probability;
- false promotion probability under each null DGP, where the lower endpoint ever exceeds tau=0.55;
- promotion power under the true-value-0.65 alternative at the same threshold;
- conditional median and 90th-percentile promotion time when promotion occurs;
- MOPE analytic-root direct-certificate checks/failures and endpoint collisions; grid endpoint advances; per-method elapsed seconds and observations/second at each horizon; overall scenario wall time. The analytic-root timing includes direct root/wealth-certificate recomputation by the research harness, so it is conservative audit-path cost—not a pure production-kernel latency claim. The grid method performs 1,000 value updates per observation, which makes the expected work contrast explicit.

A 2,000-sequence simulation cannot prove coverage. Its purpose is to quantify approximate operating characteristics with uncertainty bars, while the confidence-sequence validity argument and exact-path controls remain separate.

## Hard controls

- deterministic PRNG output for a repeated seed;
- frozen DGP and seed schedule included in the receipt;
- all generated weights respect W;
- no MOPE endpoint advances unless the polynomial root and direct log wealth certify it;
- zero root-certificate failures;
- correct metric counts, finite probability values, source/result hashes and exact commit SHA.

The repeated Wald baseline is expected to expose optional-stopping degradation; that is diagnostic data, not a failure of the test harness. Do not weaken the MOPE certificate or alpha-spending budgets to improve width.

## Current research / industry context

SCOPE-RL includes estimators such as DM, trajectory and per-decision importance sampling, doubly robust methods, and high-confidence bounds. Its evaluation suite reports MSE, regret, rank correlation, Type-I/II errors and safety violations. This benchmark borrows those decision-quality axes and adds sequence-wide coverage, interval width at fixed horizons, and time-to-correct-promotion.

Ryu, Kwon, Koppe & Jun (COLT 2025), Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing, is a relevant next competitor for the off-policy selection layer. It is not implemented in this PR, so no claim is made against that result.

## References

- Karampatziakis, Mineiro & Ramdas, Off-Policy Confidence Sequences, ICML 2021: https://proceedings.mlr.press/v139/karampatziakis21a.html
- SCOPE-RL supported OPE and HCOPE implementation: https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html
- Ryu, Kwon, Koppe & Jun, Betting and Freezing, COLT 2025: https://proceedings.mlr.press/v291/ryu25a.html

The success criterion is a trustworthy operating frontier, not unconditional superiority. If the MOPE bound stays too wide under weak overlap, the correct outcome is to expose that limitation and explore valid variance adaptation or explicit partial identification—not manufacture a tighter interval.
