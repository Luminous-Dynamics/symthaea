# CMAQ-051 — Pre-registered representation e-portfolio

## Motivation

CMAQ-049 and CMAQ-050 test the same conditional paired-distribution null, but use different finite representations:

- CMAQ-049: one-hot pre-response contexts times paired response Fourier differences.
- CMAQ-050: continuous covariate Fourier features times paired response Fourier differences.

The first is explicit and easy to audit for known low-cardinality operating modes. The second avoids hard bins and captures smooth covariate interactions. Neither finite representation is universally complete.

Selecting whichever representation looks strongest after inspecting the stream would invalidate the ordinary fixed-process guarantee. CMAQ-051 instead uses a fixed prior mixture of their e-processes.

## Validity

Let E_ctx,t and E_prod,t be nonnegative test supermartingales under the same conditional paired-distribution null

    Law(X_t | Z_t,F_(t-1)) = Law(Y_t | Z_t,F_(t-1)).

For fixed prior weights 1/2 and 1/2,

    E_port,t = 0.5 * E_ctx,t + 0.5 * E_prod,t

is also a nonnegative test supermartingale. No independence between the representation processes is required. The portfolio rejects when E_port,t >= 1/alpha.

The implementation computes the mixture in log space:

    log E_port,t = logaddexp(log E_ctx,t, log E_prod,t) - log(2).

Weights are fixed at construction. There is no member addition, removal, reweighting, maximum-over-members, or post-hoc selection after observing results.

Each member is also allowed to stop when its own e-value crosses its own 1/alpha threshold. The member is then held at its stopped value while the other continues. This is optional stopping of a nonnegative test supermartingale; its frozen wealth remains in the portfolio. The portfolio still applies its own mixture threshold.

## Transaction and execution behavior

Every pair creates both:
1. a context-blocked response feature difference;
2. a continuous product Fourier covariate-response difference.

The portfolio owns one replay inventory and one monotonically increasing sequence. Both compact CMAQ-046 monitor states are staged before commit. Invalid input, unknown context, duplicate pair ID, sequence mismatch or arithmetic failure commits neither member and consumes no pair ID.

The pair-ID set is not cloned on each update. Only the bounded CMAQ-046 monitor states are staged, so transactional cost is proportional to feature-state size rather than deployment history. Cross-process durable replay protection remains an external evidence-ledger obligation.

## Implementation

The dependency-free Rust crate depends on CMAQ-046 and CMAQ-048 by local path and provides:
- frozen response and covariate Fourier maps;
- frozen context count and inventory digest;
- independent contextual and product BoxTargets and target digests;
- one common alpha for both members;
- fixed 50/50 prior weights;
- log-sum-exp mixture wealth and Ville threshold;
- child stop/freeze semantics;
- transactional staged updates and one replay inventory;
- fixed feature/context dimension limits.

The feature maps, targets, context mapping and weights must be frozen before paired responses are observed. Context IDs must depend only on pre-response covariates.

## Tests

Deterministic tests cover:
- initial mixture wealth and exact equal-weight arithmetic identity;
- both members receiving the same pair exactly once;
- equal-marginal conditional mismatch accumulating mixture evidence;
- one child stopping while the other continues;
- atomic duplicate/sequence/context/input failure;
- immutable digests and dimensions;
- permanent mixture quarantine.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-051/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-051/Cargo.toml --locked --all-targets -- -D warnings

## Paired portfolio probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-051/Cargo.toml --locked --bin representation_portfolio_probe -- 1000 2000 20261128 > cmaq-051-portfolio-results.csv

The probe uses randomized covariate signs on every round. Under the conditional-pair null, X and Y are independent draws from the same symmetric response distribution. Under the alternative, positive Z pairs 0.2 with 0.8 and negative Z pairs 0.8 with 0.2. Marginal X and Y mixtures are identical.

It reports null alarms, conditional-shift detections/misses, median alarm time, each member's stopped/current log wealth, and mixture log wealth. The CI uses a small number of replications as a wiring smoke test, not a calibrated false-alarm qualification.

## Benchmark and negative controls

Compare:
- fixed 50/50 context/product mixture;
- CMAQ-049 context member alone;
- CMAQ-050 product member alone;
- pre-registered unequal prior weights;
- an invalid post-hoc maximum over member e-values as a negative control only (it must not be presented as a valid anytime test);
- faithful SCPT on identical paired covariate sequences.

Include alternatives visible primarily to the discrete context representation, primarily to the continuous product representation, visible to both, and hidden from both finite maps. Report per-member contribution, full-horizon null alarms with exact/Wilson intervals, unconditional power/misses, H+1-restricted delay, p99 update latency, memory and member dimensions. Use at least 10,000 independent null paths and 5,000 alternatives per primary cell with CMAQ-044 paired effective-block inference.

## Claim boundary

The fixed mixture is valid only when every member is a nonnegative test supermartingale under the same conditional paired-distribution null and the member inventory/weights are pre-registered. Mixture validity does not fix a false context assignment, invalid feature target, incorrect pair match, map null-space blind spot or unauthenticated sensor. It is not a reproduction of SCPT and does not claim SOTA.
