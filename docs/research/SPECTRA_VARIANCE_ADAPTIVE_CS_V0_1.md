# SPECTRA Variance-Adaptive Confidence Sequences — Research Note v0.1

**Status:** experimental Rust primitive; production receipt integration and Rust validation are pending.
**Implementation:** src/variance_adaptive_cs.rs
**Method ID:** spectra-subgaussian-variance-adaptive-lil-cs-ramdas-martinez-taboada-2026-v1

## 1. Why this is worth doing

The current SPECTRA score-bound implementation uses bounded Hoeffding intervals with an anytime alpha-spending schedule. This is simple, interpretable and robust to unknown variance under the declared range, but it uses the worst-case range even when observed within-stream variance is much smaller.

Ramdas and Martínez-Taboada (2026), arXiv:2610.42058, derive an empirical-variance time-uniform confidence sequence for conditionally sub-Gaussian streams with a fixed conditional mean. Their Theorem 3.1 / equation (3.9) uses observed centered scatter while preserving finite-sample time-uniform validity. Its sharp LIL width conclusion additionally needs convergence to positive average conditional variance.

Primary source: https://arxiv.org/abs/2610.42058

The paper reports 28–36% lower mean widths at n=1,000 and 47–52% at n=10,000 in its tested Gaussian / symmetric two-point designs with true standard deviation half the supplied sub-Gaussian proxy, compared to its normal-mixture CS. With a tight proxy, it reports comparable widths at n=1,000 and 5–15% lower at n=10,000. At n=100 its method is generally wider. These are the authors' reported experiments, not SPECTRA results.

## 2. Estimand and assumptions

For each frozen action × scenario × metric stream, the proposed method assumes:

- observations are adapted to a filtration;
- E[X_i | F_(i-1)] = μ for one fixed μ over the stream;
- conditional sub-Gaussian proxy s is known;
- all score values lie in the frozen range [a,b], so Hoeffding's lemma permits s=(b-a)/2.

The theorem's finite-sample coverage does not require independent observations or variance convergence, but the fixed conditional mean and conditional MGF bound remain mandatory. Shared CRN vectors or adaptive allocation are only admissible when each individual stream still satisfies its conditional model under the complete filtration used by the proof.

This is not the Chugg–Ramdas estimand. Their December 2025 bounded-stream CS targets the average conditional mean and can track drift; its closed-form approximation begins at a data-dependent t0 and may be wider early. Keep a separate method and receipt identity:

- Ramdas–Martínez-Taboada 2026: constant conditional mean; sub-Gaussian proxy or known p-th moment; time-uniform LIL sequence.
- Chugg–Ramdas 2025: bounded observations; average conditional mean; closed-form empirical-Bernstein CS and exact mixture reference; use for drift-aware estimands.

Neither method should silently replace the other. No data-driven choose-the-narrowest operation across multiple methods is allowed unless alpha is allocated / a joint validity proof is supplied.

Additional references:
- Chugg & Ramdas (2025), arXiv:2512.21300: https://arxiv.org/abs/2512.21300
- Waudby-Smith & Ramdas (2024), JRSS B: https://doi.org/10.1093/jrsssb/qkad009
- Howard et al. (2021), Annals of Statistics: https://doi.org/10.1214/20-AOS1991

## 3. Implemented primitive

The new module is deliberately independent of ScoreBoundsReceipt at this stage.

It maintains:
- sample count;
- mean and centered sum of squares via Welford's update;
- harmonic sum H_n = sum from i=1 to n of 1/i;
- frozen score interval and the induced proxy s=(b-a)/2;
- frozen two-sided error budget.

At look n it calculates the calibrated bet grid from equations (3.5)–(3.8), then the minimum of:
- the selected fine-grid radius;
- four coarse-bet radii.

The radius uses:

B_n(lambda,L) = L/(n*lambda) + lambda*(Q_n+s^2*H_n)/(2n) + (n-H_n)/(n*lambda)*log(1+s^3*lambda^3),

where Q_n = sum from i=1 to n of (X_i - Xbar_n)^2.

All intervals are intersected with the frozen score range. A zero-width range is deterministic. Missing observations, changed simulator versions, target drift, or invalid range declarations must not be handled as if they were ordinary samples.

### Alpha composition with the existing SPECTRA six-claim contract

The current score-bound receipt allocates delta = familywise_alpha/(6R) to each directional claim over R = actions × scenarios cells.

Equation (3.9) is a two-sided CS. The primitive's new_for_one_sided_claim(..., delta) constructor therefore sets the CS error budget to 2*delta. The paper's symmetric positive/negative bet allocation bounds each directional miss probability by delta. Thus:
- target-information lower, discrimination lower, representativeness lower and participant-burden upper each use one directional failure budget delta;
- target-utility lower and upper together use the two-sided budget 2*delta;
- a union bound over the original six directional claims still spends at most 6R*delta = familywise_alpha.

This must be independently reviewed against the current receipt's actual claim definitions before production integration. Do not count one two-sided CS as only one claim if both endpoints are used.

## 4. Python mirror experiment — 2026-11-06

A separate Python implementation of equation (3.9) was exercised against the current SPECTRA anytime Hoeffding radius using the same six-claim alpha allocation:

- R=4 action × scenario cells;
- familywise alpha 0.05;
- directional claim alpha delta=0.05/(6×4);
- two-sided sequence alpha 2*delta;
- bounded Bernoulli scores with known range [0,1], hence s=0.5;
- 10,000 independent paths per mean, followed continuously from n=1 through n=4,096;
- means 0.01, 0.1, 0.5, 0.9, 0.99.

### Empirical anytime coverage over 4,096 looks

| Bernoulli mean | Missed streams / 10,000 | Observed whole-stream coverage |
|---:|---:|---:|
| 0.01 | 0 | 100.000% |
| 0.10 | 0 | 100.000% |
| 0.50 | 1 | 99.990% |
| 0.90 | 0 | 100.000% |
| 0.99 | 0 | 100.000% |

For the one observed miss at mean 0.5, the two-sided 95% Wilson interval for coverage is approximately 99.943%–99.998%. With zero misses, the corresponding Wilson lower bound is approximately 99.962%. These finite simulations are diagnostic only; they do not establish the theorem or substitute for 10,000+ outer campaigns in the locked benchmark.

### Mean clipped interval-width ratio to current anytime Hoeffding

Both intervals are clipped to the same [0,1] support and use the same directional alpha allocation.

| Mean | n=32 | n=64 | n=128 | n=256 | n=512 | n=1,024 | n=2,048 | n=4,096 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.39 | 0.98 | 0.71 | 0.56 | 0.48 | 0.47 | 0.43 | 0.41 |
| 0.10 | 1.35 | 1.02 | 0.83 | 0.75 | 0.65 | 0.54 | 0.46 | 0.45 |
| 0.50 | 1.14 | 1.11 | 0.89 | 0.80 | 0.79 | 0.69 | 0.68 | 0.62 |
| 0.90 | 1.35 | 1.02 | 0.83 | 0.75 | 0.65 | 0.54 | 0.46 | 0.45 |
| 0.99 | 1.39 | 0.98 | 0.71 | 0.56 | 0.48 | 0.47 | 0.43 | 0.41 |

Interpretation: the prototype is wider during early looks, then materially narrower in these Bernoulli examples. At n=4,096 the simulated width reduction is about 38% at mean 0.5 and about 55–59% near the tails. These figures compare to SPECTRA's alpha-spending Hoeffding schedule, not to every modern normal-mixture or betting CS baseline. They are one bounded simulation family, not a general performance claim.

## 5. Rust test declarations

The experimental module includes tests for:
- exact formula parity against a separately calculated two-pass equation (3.9) vector;
- one-sided alpha to two-sided alpha composition;
- zero-width deterministic range;
- invalid range, alpha, non-finite and out-of-range samples;
- Welford scatter against a two-pass calculation;
- deterministic stream replay;
- positive affine scale equivariance;
- alpha ordering;
- 4,096 deterministic sample prefixes remaining finite and inside the declared range.

The tests have not been executed: this environment has no cargo, rustc, or rustfmt. No compile, test, formatting, clippy, CI, inferential coverage, or wall-time pass is claimed.

## 6. Locked next validation

Compare the new sequence against:
1. SPECTRA's current anytime Hoeffding alpha-spending bound;
2. Howard et al. HRMS empirical-Bernstein CS;
3. Waudby-Smith–Ramdas betting / empirical-Bernstein CS;
4. Chugg–Ramdas exact mixture C_mix and closed-form C_apx, on the average-conditional-mean track;
5. oracle variance-aware LIL width, for diagnostic lower-bound comparison only.

DGPs:
- Bernoulli means near 0, central, near 1;
- Beta and bounded skewed mixtures with matched means but different variances;
- near-constant, high-variance, endpoint-heavy, and degenerate streams;
- independent replicates and joint CRN vector generation;
- adaptive but predictable sampling schedules;
- fixed conditional mean and separately labelled time-varying conditional means;
- score range changes, missingness, simulator version shifts, resets, stale calibration and hidden target-shift controls.

Metrics:
- pointwise and whole-stream coverage, with binomial confidence bounds;
- familywise coverage across the complete 6R directional claims;
- interval width by look and variance regime;
- fully charged cost to a valid SPECTRA certificate;
- time to first valid certificate and unresolved-at-budget rate;
- numerical failure, CPU time, memory and p50/p95/p99 verifier cost.

Run at least 10,000 independent outer campaigns per primary regime. Preserve exact alpha allocations, root seeds, input streams, method IDs, and replay receipts.

## 7. Promotion gates

Keep Hoeffding as the default until the new method:
- passes independent Rust reference-vector and property tests;
- is integrated into a separately identified receipt method and independently replayed from raw observations;
- demonstrates the constant conditional-mean and sub-Gaussian scope is satisfied by the ingestion contract;
- preserves at least 95% familywise coverage with Monte Carlo confidence bounds over all predeclared supported regimes;
- has zero false passes on model, source, target, simulator-version and support negative controls;
- reduces median fully charged cost-to-valid-certificate by at least 15% in predeclared low-variance regimes without worsening coverage;
- has no unacceptable regression in high-variance, boundary-heavy or early-look regimes.

Abstention, fallback to the existing Hoeffding CS, or a wider certificate is acceptable when assumptions cannot be qualified. A data-driven switch to whichever bound is narrowest is prohibited unless the method selection itself has a valid joint alpha allocation.
