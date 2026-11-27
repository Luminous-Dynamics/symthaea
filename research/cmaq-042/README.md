# CMAQ-042 — Finite-horizon reference-null threshold calibration

## Purpose

The primary CMAQ PRM monitor remains an unbounded-horizon anytime-valid e-process and retains the Ville threshold 1/alpha. CMAQ-042 adds an optional, separate threshold for a predeclared finite mission window.

The idea comes from Ding, Wei, Zhu & Dai (arXiv:2609.32678, 26 September 2026): calibrate the maximum of a sequential evidence path against independent trajectories from an exactly specified reference null. If the deployed path and B reference paths are exchangeable under the null, the test path's rank among the B+1 maxima controls the finite-horizon false-positive probability.

This does not make an arbitrary e-process threshold smaller while retaining its original unbounded-horizon guarantee. The calibrated threshold applies only to the frozen horizon T and the frozen monitor/score/reference bindings.

## Exact finite-sample threshold

Let B be the number of independent reference-null paths and let alpha=a/b. For each path, record

    L_i = max_{0 <= t <= T} log E_i(t),

including initial wealth E_i(0)=1, so L_i >= 0.

Compute the one-based order rank

    k = ceil((1-alpha) * (B+1)).

Sort the reference maxima and set q=L_(k), provided k <= B. For the deployed path, reject only if

    L_test > q.

The comparison is strict. A tie with the threshold is not rejected, making the procedure conservative in the presence of atoms. If k=B+1, there is no finite rejecting threshold and CMAQ-042 returns Disabled rather than inventing one. At alpha=0.05, B must be at least 19 for a finite threshold; larger B is required for useful quantile resolution.

The rank is computed with exact integer arithmetic. All path maxima are stored as log e-values to avoid exponentiating large wealth.

## Evidence contract

The reference bank binds:
- bank ID, horizon T and calibration-reference size;
- alpha numerator/denominator;
- exact monitor-build, score-definition and rank-reference digests;
- null-simulator digest;
- each reference path's unique trajectory ID, unique trace SHA-256 and log maximum. Duplicate IDs and duplicate trace digests are rejected; the deployed test path cannot reuse either a reference ID or trace digest.

The deployed path must match the frozen horizon, calibration size, monitor build, score definition and rank reference; it must have its own trajectory ID and trace digest. It cannot be one of the reference trajectories. The gate object is one-shot in memory and rejects a second test path. A cross-process consumption ledger still has to be enforced by the external evidence authority.

All digests are checked for canonical lowercase SHA-256 shape only. This crate does not authenticate signatures, prove that the null simulator is correct, or establish exchangeability. Those remain external, independently qualified inputs.

## Separation from the always-on gate

- **Always-on path:** keep the primary Ville threshold 1/alpha for arbitrary stopping time under the e-process assumptions.
- **Mission-window path:** use this calibrated threshold only up to its predeclared T, with a separate one-use reference bank and receipt.
- **After T:** the calibrated threshold expires. Continue using the Ville threshold or obtain a new, independently qualified calibration for a new mission window with a separately budgeted error allocation.
- **Do not reuse:** reusing the same reference bank for multiple test deployments is not covered by this one-test exchangeability argument. A real deployment needs a bank-consumption ledger and explicit multiplicity policy.

This is a secondary decision layer. It does not change CMAQ-040's evidence path, validate hazard semantics, or guarantee timely recovery.

## Tests

The 16 deterministic Rust tests cover exact order-rank arithmetic, alpha=0.05 bank sizes 18 and 19, strict ties, strict exceedance, disabled thresholds, horizon/calibration/build/score/reference mismatch, duplicate IDs and trace digests, malformed digests, non-finite/negative log maxima, test/reference overlap, one-shot consumption, and exhaustive exchangeable rank-position checks showing the exact finite-sample rejection fraction is no greater than alpha.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-042/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-042/Cargo.toml --locked --all-targets -- -D warnings

## Generic null-path smoke probe

The workflow also runs:

    cargo +1.100.0 run --release --manifest-path research/cmaq-042/Cargo.toml --locked --bin reference_null_probe -- 64 256 20261127 > cmaq-042-reference-null-smoke.csv

This is deliberately a *generic simple-null e-process* check, not a PRM result. It uses independent deterministic pseudo-random fair-coin paths and the nonnegative martingale

    M_t = product_{i<=t} (1 + 0.5 X_i),  X_i in {-1,+1} with equal probability.

For each deployment trial, it builds a fresh bank of 99 reference paths, generates one separate test path, applies the finite-horizon order-statistic boundary, and compares against the Ville threshold 1/alpha on that same test path. The CSV contains per-trial test maximum, calibrated threshold and both decisions. The workflow stores the artifact.

The CI sample size is only a wiring / deterministic-run smoke check. It is far too small to estimate a five-percent false-alarm probability tightly, and the SplitMix64 streams are a reproducible simulation device, not a cryptographic random source. The formal validity statement is the exchangeable-rank argument, not the observed CI count.

## Required experiment before claiming utility

Use the exact Pólya-urn predictive-rank null for each calibration size n and the exact deployed CMAQ monitor build. For each pre-registered T:
1. Generate B independent reference-null paths and one independent test-null path using identical frozen bindings.
2. Calibrate q once and apply the one-shot gate.
3. Measure false rejection with a binomial confidence interval on independent test paths, rebuilding the bank per test path (or use a formally derived shared-bank procedure).
4. Repeat paired alternatives and report power, median/p90 delay, miss fraction and worst morphology.
5. Compare with Ville at 1/alpha on the same path. Report the finite horizon and the bank size with every result.

Do not use the scorecard's finite-T result as an anytime guarantee, or compare it to Ville over an unbounded horizon as though they were the same estimand.
