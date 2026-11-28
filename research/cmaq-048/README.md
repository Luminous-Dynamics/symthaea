# CMAQ-048 — Fixed Fourier-feature kernel mean approachability monitor

## Motivation: move beyond first moments without abandoning bounded evidence

CMAQ-046 tests whether the conditional mean of a bounded vector remains inside a frozen convex target. That guarantee is exact and useful, but a raw-mean monitor can be blind to scale, tail or shape changes that preserve the raw conditional mean.

CMAQ-048 applies a fixed finite Fourier feature map before CMAQ-046. The resulting feature mean approximates a kernel mean embedding, so changes in variance or distribution shape can become mean shifts in feature space.

This is not a new null-free detector. The conditional-mean target in feature space must be fixed and independently justified. The guarantee is exact for the registered finite feature map and target contract.

## Feature map

For a fixed list of M frequencies omega_j in R^d, define

    phi(x) = 1/sqrt(M) *
      [cos(omega_1' x), sin(omega_1' x), ..., cos(omega_M' x), sin(omega_M' x)].

For every finite input,

    ||phi(x)||_2^2 = (1/M) * sum_j (cos^2 + sin^2) = 1.

Thus the monitor's feature-observation norm bound is exactly one (up to floating-point roundoff). The map's inner product is

    <phi(x),phi(y)> = (1/M) * sum_j cos(omega_j' (x-y)).

When each omega_j is independently sampled from the Gaussian spectral measure N(0, ell^-2 I), this approximates the Gaussian RBF kernel with lengthscale ell.

This crate deliberately accepts the fixed frequency matrix rather than hiding random-number generation or frequency selection inside the runtime. The exact matrix must be frozen and digested. The code checks digest shape and immutability, but does not authenticate signatures or recalculate SHA-256 itself.

## Null and evidence process

Let F be the frozen feature map and let S_phi be a compact convex target box defined from an independent null reference and its pre-registered uncertainty envelope. The null is

    H0: E[phi(X_t) | F_(t-1)] ∈ S_phi   for every t.

The transformed observations have norm one. CMAQ-046 then chooses a predictable OCO normal w_t and forms

    q_t = <w_t, phi(X_t)> - h_(S_phi)(w_t).

Under the null, E[q_t | F_(t-1)] <= 0. With B = 1 + max_(s∈S_phi)||s||_2 and fixed lambda=1/2,

    K_t = K_(t-1) * (1 + lambda*q_t/B)

is a nonnegative test supermartingale. The Ville threshold is 1/alpha. The pathwise distance/residual/signed-regret certificate is inherited from CMAQ-046.

The target is an envelope for the *conditional mean embedding*, not merely the empirical mean of one calibration sample. In an i.i.d. stationary null, this mean is the fixed kernel embedding of the reference distribution. Under serial dependence, the target must cover every reachable conditional embedding. If that is not true, the validity claim does not apply.

## Implementation

The dependency-free Rust crate depends only on CMAQ-046 by local path and provides:
- fixed finite frequency matrix, map digest and deterministic feature ordering;
- 2M-dimensional paired sine/cosine features;
- mathematical unit-norm feature identity;
- fixed feature-target BoxTarget and null-reference receipt binding;
- input dimension, finiteness and raw input norm validation;
- log-domain CMAQ-046 e-process, predictable OCO witness and signed regret certificate;
- atomic failure behavior: invalid raw input or transform failure does not update evidence;
- no post-change target or frequency refitting.

The Fourier feature matrix is supplied externally. The map digest, target digest and reference receipt are structurally checked only. Independently authenticate them before deployment.

## Tests

The deterministic tests cover:
- feature dimension and unit norm;
- paired sine/cosine parity and known symmetric-distribution means;
- a pure scale shift with raw mean exactly zero but feature-mean distance greater than 0.5;
- finite input, frequency bank, digest, dimension and input norm checks;
- atomic invalid-input behavior;
- frozen map/target/reference bindings;
- evidence growth on a persistent scale alternative;
- approximate RBF inner-product identity.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-048/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-048/Cargo.toml --locked --all-targets -- -D warnings

## Paired scale-only probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-048/Cargo.toml --locked --bin fourier_scale_probe -- 1000 2000 20261128 1000 > cmaq-048-scale-results.csv

The probe compares:
- raw scalar-mean CMAQ-046 with target [-0.05, 0.05];
- CMAQ-048 with four fixed sine/cosine features generated from frequencies {1,3} and a frozen target around the symmetric +/-0.2 reference embedding.

Both arms receive identical symmetric samples. The alternative changes magnitude from +/-0.2 to +/-0.8 while preserving raw mean zero. This is a deliberately narrow test of the higher-moment blind spot. It does not establish general arbitrary-distribution detection or SOTA.

## Required research benchmarks

Use a paired design with frozen frequencies, bandwidth, target, reference receipt, alpha, horizon and split. Sweep:
- feature counts M in {8, 16, 32, 64, 128, 256, 512};
- Gaussian RBF lengthscales over a pre-registered logarithmic grid;
- raw dimensions d in {1, 2, 8, 32, 128};
- effective calibration blocks in {19, 49, 99, 199, 499, 999, 1,999};
- alpha in {0.01, 0.05, 0.10};
- pre-change delays t0 in {0, 125, 500, 2,000, 8,000};
- horizons H in {500, 2,000, 10,000}.

Alternatives should include:
- symmetric variance/scale shifts with unchanged mean;
- tail-weight and kurtosis changes;
- skewness changes and symmetric skew-cancelling mixtures;
- multimodal mixtures with equal first moment;
- conditional shape shifts by covariate/context;
- abrupt, gradual and transient shifts;
- changes in temporal dependence with unchanged one-step feature mean (negative control for this monitor);
- shifts that lie in the finite feature bank's null space (explicit approximation failure cases).

Nulls must preserve E[phi(X_t)|F_(t-1)] inside S_phi for the full horizon. A null generated only to have a matching unconditional mean is insufficient. Report full-horizon ever-alarm probability with confidence intervals, power, H+1-penalized misses/delay, worst morphology, feature count, bandwidth, target dimension, update p50/p95/p99, memory and frequency-matrix storage. Use at least 10,000 independent null paths and 5,000 paths per primary alternative cell. Use CMAQ-044 for paired effective-block inference.

## External comparison

Compare against, but do not conflate assumptions with:
- PF-CCTM / fixed-reference conditional conformal test martingales (ICML 2026);
- the NeurIPS 2025 online random-Fourier-feature change detector, which has a different online/no-pretrained-reference design and reported minimax delay guarantees;
- SCPT (Zheng & Zhang, 17 Nov 2026), a sequential conditional paired-sample kernel-discrepancy test with time-uniform Type-I control;
- sequential energy-statistic change detection;
- CMAQ-046 raw conditional means and CMAQ-038/045 rank-based monitors.

For each external method, reproduce its stated assumptions, input pairing, reference data, stopping rule, and risk criterion. Do not claim a head-to-head win from the small smoke probe.

## Claim boundary

A finite Fourier bank is not a universally characteristic kernel for every realization. Some distinct distributions can share the registered finite embedding. It may fail to detect changes outside the chosen frequencies or bandwidth. The e-process validity is exact only for the fixed bounded feature map and independently valid conditional-mean target envelope. Reference estimation, map digest authentication, target justification, data provenance, common-cause closure and physical sensor truth remain external obligations.
