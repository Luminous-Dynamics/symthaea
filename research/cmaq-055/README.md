# CMAQ-055 — Restartable conditional kernel-family portfolio

## Research target

CMAQ-054 supplies a fixed, evidence-adaptive mixture over conditional paired Fourier experts. CMAQ-040 supplies restart-time mixtures for unknown change times. CMAQ-055 composes the two dimensions:

    E_t = sum_k w_k sum_(s>=1) nu_s M_(k,s,t)

where w_k is a frozen representation prior, nu_s is a frozen start-time prior, and M_(k,s,t) is a child e-process initialized at unit wealth at start s. Each child is one before its start time and updates only from predictable post-start observations. Under the common conditional paired null, the product prior and the child e-process assumptions make the full sum a nonnegative supermartingale. Ville's threshold 1/alpha_test is spent once.

This is not a novelty claim for method-of-mixtures or unknown-time restarting. Direct frontiers include Sreedhar & Li (ICML 2026 Workshop) on anytime-valid kernel-family mixtures, Liu, Prinster & Orabona (Oct 2026) on parameter-free conditional conformal restart portfolios, SCPT (Nov 2026) on sequential conditional paired tests, and NeurIPS 2025 Online RFF-MMD.

CMAQ-054's child family currently stops when its own threshold is crossed. Therefore the component for start time s must use the stricter component threshold alpha_s = alpha_test * nu_s, with per-kernel threshold alpha_(s,k) = alpha_test * nu_s * w_k. A stopped component then contributes at least 1/alpha_test after multiplying by nu_s, so it cannot silently stop below the outer mixture boundary. Alternatively, the implementation may keep child e-processes unthresholded and let only the outer mixture decide. The shared reference-envelope failure event is paid once; every restart uses the same frozen reference and target.

## Start-time prior

Partition positive start times into dyadic epochs:

    I_j = {2^(j-1), ..., 2^j - 1}
    p_j = 1/[j(j+1)]
    nu_s = 1/[2^(j-1) j (j+1)] for s in I_j.

The epoch mass telescopes:

    sum_(s in I_j) nu_s = 1/[j(j+1)]
    sum_(j>=1) 1/[j(j+1)] = 1.

For s in epoch j, the complexity penalty is

    -log(nu_s) = (j-1)log(2) + log(j) + log(j+1)
              = log(s) + O(log log(s)).

This is the right order for the unknown-onset price, but a prior alone is not an algorithm. The exact per-start portfolio must still update every active child correctly.

The Rust crate now implements the bounded-horizon all-start correctness oracle. At each observation t it creates the new component s=t, updates every active component s<=t, and includes the exact residual mass for all s>t at wealth one. It computes start-time and joint kernel/start posterior weights in log space.

Each child family is constructed with effective threshold alpha_s = alpha_test*nu_s; each kernel member therefore receives alpha_(s,k)=alpha_test*nu_s*w_k. The reference envelope and calibration event are shared across all restarts, so delta_reference is not multiplied by the number of starts. The parent owns the only pair-ID ledger; children use CMAQ-054's externally guarded replay path and retain local sequence checks without duplicating pair IDs.

The direct oracle performs O(K*T^2) expert-update work through horizon T. Numeric monitor state is O(K*T), plus O(T) shared replay IDs. The configured horizon cap is 1,024 observations. It is a correctness oracle for small horizons, not a deployable optimized implementation. Updates are staged across all active children and committed only after the full mixture and posterior normalization succeed.

## Stage 2 — efficient restart aggregation

Only after the all-start oracle is validated should we investigate an O(K log T) or O(K log^2 T) update. Candidate approaches include a restartable universal portfolio, FFT divide-and-conquer, and carefully proven dyadic aggregation. Any truncated start grid must account for omitted prior mass; any approximate wealth calculation must have a conservative lower bound or a separate validity proof.

## Stage 2 — efficient restart aggregation

Only after the oracle is validated should we investigate an O(K log T) or O(K log^2 T) update. Candidate approaches include a restartable universal portfolio, FFT divide-and-conquer, and carefully proven dyadic aggregation. Any truncated start grid must account for omitted prior mass; any approximate wealth calculation must have a conservative lower bound or a separate validity proof.

## Tests in this increment

- Exact weight formula over dyadic epoch boundaries and telescoping epoch mass.
- Exact finite-start mass plus future-start residual equals one.
- Correct future mass at epoch boundaries and inside epochs; u64 edge handling and overflow checks.
- Unit-wealth preservation under equal paired observations, including the unstarted residual.
- Exact parent-prior scaling of each child family threshold.
- Parent-level pair replay rejection and no partial mutation after a later child fails.
- Normalized start-time and joint kernel/start posterior weights.
- Fail-closed horizon cap and permanent stop semantics.

Run:

    cargo +1.100.0 fmt --manifest-path research/cmaq-055/Cargo.toml --all -- --check
    cargo +1.100.0 test --manifest-path research/cmaq-055/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-055/Cargo.toml --locked --all-targets -- -D warnings
    cargo +1.100.0 run --release --manifest-path research/cmaq-055/Cargo.toml --locked --bin restart_portfolio_probe

The probe contains an equal-pair null wiring arm and an equal-marginal conditional-pairing alternative. It emits mixture wealth, future-start posterior and the leading joint start/kernel posterior. These deterministic fixtures establish end-to-end wiring only. Neither the tests nor the probe empirically establish Type-I calibration or detection power.

## Benchmark obligations

Use identical conditional pair streams, common covariates, feature maps, reference manifest, alpha_test and horizon for CMAQ-050/051/054/055. Compare faithful SCPT and ICML kernel-family methods on this conditional-paired task. Compare PF-CCTM only on its fixed-reference scalar-score protocol, and Online RFF-MMD on its eligible multivariate two-sample change-detection protocol. Those are related but not interchangeable statistical problems.

Use at least 10,000 full-horizon null paths per primary cell and 5,000 alternatives. Misses count H+1. Report envelope coverage separately from conditional betting errors; ever-false-alarm confidence intervals; unconditional power; restricted mean delay and p50/p90/p95; worst scenario family; p99 latency; memory; and expert/start posterior weights. Paired comparisons resample effective common-cause blocks.

## Claim boundary

The product-mixture theorem requires every child to be a valid e-process for the same conditional null and all maps, targets, priors and reference evidence to be frozen. It does not make the finite expert family universal, verify source independence or guarantee fast detection of alternatives outside the registered family. No comparison win or SOTA claim is made in this increment.