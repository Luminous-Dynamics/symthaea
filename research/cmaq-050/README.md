# CMAQ-050 — Continuous-covariate product Fourier paired monitor

## Why move beyond discrete contexts?

CMAQ-049 uses one-hot context blocks to prevent conditional discrepancies from cancelling in a pooled marginal comparison. That is auditable, but it requires a frozen finite partition. A smooth effect can be missed inside a bin, and arbitrary bin edges can create discontinuities.

CMAQ-050 replaces the one-hot context with a frozen Fourier feature map of the shared covariate. It tensors this map with the paired response-feature difference. The result is a smooth finite-feature interaction between covariates and response discrepancies.

## Null and construction

For paired responses X_t,Y_t sharing covariate Z_t, the null is

    H0: Law(X_t | Z_t,F_(t-1)) = Law(Y_t | Z_t,F_(t-1)).

Let phi_X be a fixed unit-norm Fourier response map and psi_Z a fixed unit-norm Fourier covariate map. Define

    r_t = 0.5 * psi_Z(Z_t) ⊗ [phi_X(Y_t)-phi_X(X_t)].

Since ||phi_X||=||psi_Z||=1,

    ||r_t||_2 <= 0.5 * ||psi_Z|| * (||phi_X(Y_t)||+||phi_X(X_t)||) <= 1.

Under the conditional distribution null, E[phi_X(Y_t)-phi_X(X_t) | Z_t,F_(t-1)] = 0. Therefore, by iterated expectation,

    E[r_t | F_(t-1)] = 0.

CMAQ-046 applies its predictable OCO witness and fixed-bet e-process to this bounded product vector. A change in the marginal distribution of Z alone does not create evidence when the conditional response distributions remain equal.

The guarantee is exact for the frozen finite product-feature map and the zero/tolerance conditional-mean target. Finite Fourier banks do not identify every conditional distribution; alternatives can lie in their feature null space.

## Relation to CMAQ-049 and SCPT

- **CMAQ-049**: one-hot context blocks × finite response Fourier features. It gives explicit per-context separation and simple context-coverage accounting, but requires frozen discrete strata.
- **CMAQ-050**: continuous covariate Fourier features × response Fourier features. It avoids hard bin edges and captures smooth covariate interactions, but uses a finite product-feature approximation and may require more dimensions.
- **SCPT** (Zheng & Zhang, 17 Nov 2026): uses a bounded characteristic kernel, sequential nearest-neighbor estimates of conditional RKHS mean embeddings based on past pairs, and a covariate-indexed predictable betting function. Its principal construction states a conditional-independence structure given the covariate sequence; broader dependence structures are handled in separate extensions. CMAQ-050 is not an SCPT reproduction and does not inherit SCPT's distribution-identification guarantees.

The product map has dimension 4 times M_response times M_covariate, capped at 1,024 features. This makes memory and compute predictable and permits a direct latency comparison with SCPT's sequential KNN estimator.

## Implementation

The Rust crate depends on CMAQ-046, CMAQ-048 and CMAQ-049 by local path:
- two independent immutable Fourier maps, one for responses and one for covariates;
- product feature vector 0.5 times psi(z) tensor (phi(y)-phi(x));
- explicit checked product dimension and maximum 1,024 features;
- fixed target, response/covariate reference digests and target digest;
- monotonic one-based sequence and unique pair identity;
- fail-closed duplicate pair, malformed digest and sequence handling;
- atomic pair-ID consumption only after successful evidence update;
- no KNN index, no refitting and no adaptive frequency selection.

Digests are structurally checked only. Authenticate the feature matrices and reference receipts externally. Pair-ID replay protection is in-process; a durable ledger is required across restarts.

## Tests

Deterministic tests cover:
- product feature norm bound;
- identical response pairs return zero regardless of covariate;
- swapping X and Y negates the vector;
- conditional discrepancies with equal marginal mixtures produce evidence;
- invalid sequences and duplicate pair IDs are rejected without mutation;
- feature/target digest and dimension checks;
- maximum product feature dimension.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-050/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-050/Cargo.toml --locked --all-targets -- -D warnings

## Paired conditional-effect probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-050/Cargo.toml --locked --bin product_conditional_probe -- 1000 2000 20261128 > cmaq-050-product-results.csv

The probe compares product Fourier monitoring with CMAQ-049 configured as one pooled context:
- conditional-pair null: Z is random; X and Y are independent draws from the same symmetric conditional distribution;
- conditional alternative: Z is random with equal positive/negative probability. When Z is positive, X=0.2 and Y=0.8; when Z is negative, X=0.8 and Y=0.2. Both marginal X and Y mixtures are identical, while their conditional distributions differ.

The pooled monitor sees an unpredictable zero-mean paired feature difference; the product map retains the interaction between the covariate and response difference. This is a narrow diagnostic smoke test, not a calibrated Type-I or SOTA result.

## Full benchmark

Sweep response/covariate frequency counts and bandwidths, raw response/covariate dimensions, effective reference blocks, alpha and horizon. Compare:
- SCPT's native sequential KNN conditional-MMD method;
- CMAQ-049 fixed one-hot context blocks;
- CMAQ-050 continuous product Fourier interactions;
- pooled CMAQ-048/CMAQ-049 monitors without covariate interaction;
- raw paired mean-difference betting.

Alternatives: smooth conditional regression changes, localized covariate regions, oscillating conditional effects, equal marginal mixtures, conditional scale/tail/skew changes. Negative controls: covariate-only shift with unchanged conditional response law, equal conditional response laws under serial/heteroskedastic covariates, and finite product-feature null-space alternatives.

Use at least 10,000 independent full-horizon null paths and 5,000 alternatives per primary cell. Report full-horizon false alarms with confidence intervals, unconditional power/misses, H+1-restricted delay, worst covariate region, feature count, p99 update time, memory and frequency-matrix storage. Use CMAQ-044 paired effective-block inference.

## Claim boundary

CMAQ-050 is a bounded finite product-feature conditional-mean test. It does not use a characteristic infinite-dimensional kernel, sequential KNN conditional estimator or SCPT's native martingale. The statistical guarantee depends on the conditional equality null and fixed map/target, not on marginal X/Y equality alone. Pair matching, covariate truth, reference validity, cryptographic authentication and physical co-location are external evidence obligations.
