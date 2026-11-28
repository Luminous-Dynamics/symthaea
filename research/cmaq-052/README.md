# CMAQ-052 — Uncertainty-aware null-reference envelope and alpha split

## The target-validity gap

CMAQ-046's betting guarantee requires the true conditional null mean to remain inside a frozen convex target. A digest can bind a target file, but cannot prove the target contains the true mean.

CMAQ-052 makes one important special case statistically explicit: an iid stationary null with a separate independent calibration/reference split. It builds a simultaneous confidence box for the true mean, reserves a reference-coverage failure budget delta, and gives the remaining risk to the sequential e-process.

## Statistical construction

Let r_1,...,r_m be independent reference blocks from a common null distribution P, with

    ||r_i||_2 <= R.

Every coordinate lies in [-R,R]. Let the sample mean be mu_hat in R^d. For a reference coverage failure budget delta, define

    epsilon = R * sqrt(2 * log(2d/delta) / m)
    S_hat = product_j [mu_hat_j - epsilon, mu_hat_j + epsilon].

Hoeffding's inequality and a union bound imply

    P_P(mu not in S_hat) <= delta,

where mu = E_P[r].

Now let the test stream be independent of the reference and iid from the same stationary P. Conditional on the reference sample and on the event mu in S_hat, the test observations have conditional mean mu in the frozen CMAQ-046 target. The e-process run at alpha_test therefore satisfies

    P(false alarm ever | valid reference envelope) <= alpha_test.

Unconditionally,

    P(false alarm ever) <= alpha_test + delta <= alpha_total.

The alpha allocation is provided as exact positive rational fractions and the budget inequality is checked with integer arithmetic. The CMAQ-046 betting threshold is 1/alpha_test, not 1/alpha_total. The outer delta budget covers the possibility that the independently estimated reference envelope fails to contain the true mean.

## Assumptions that must not be silently widened

This construction is for:
- mutually independent reference blocks from one stationary null distribution;
- reference and test samples independent of one another;
- an iid test stream from that same distribution under the null;
- valid per-block L2 bound R;
- a frozen feature/residual representation and dimension d.

It is **not** a confidence sequence for arbitrary dependent data. It does not cover arbitrary serial dependence, changing conditional means, covariate-dependent means, or common-cause duplicates. For these settings a separate dependence-aware reference confidence method is required; merely grouping repeated records under one ID does not create independence.

Each block must represent one independently justified calibration unit. Unique IDs/digests are structural checks only; they do not prove that blocks are statistically independent or drawn from the same distribution.

## Implementation

The dependency-free Rust crate depends on CMAQ-046 by local path. It provides:
- typed bounded ReferenceBlock vectors with stable block IDs and trace digests;
- duplicate ID and duplicate trace rejection;
- finite value, dimension and L2 norm checks;
- simultaneous coordinatewise Hoeffding radius;
- frozen BoxTarget around the reference sample mean;
- exact rational alpha_test + delta <= alpha_total budget check;
- ReferenceEnvelopeReceipt containing dimension, reference block count, R, radius, risk split and digest references;
- CMAQ-046 monitor initialized with alpha_test;
- test-block ID exactly-once protection in-process;
- atomic invalid-test behavior and permanent CMAQ-046 quarantine.

SHA-256 strings are checked for 64-character lowercase hexadecimal shape only. Canonical serialization, digest recomputation, source authentication, reference independence and split provenance remain external evidence responsibilities.

## Tests

Deterministic tests cover:
- exact risk budget equality and over-budget rejection;
- invalid risk fraction rejection;
- Hoeffding radius identity;
- target containment of the sample mean and a known symmetric null mean;
- duplicate reference IDs and trace digests;
- malformed digest, non-finite reference value and norm violations;
- reference dimension checks;
- invalid test observation atomicity;
- duplicate test block rejection;
- persistent mean shift beyond the calibrated envelope accumulating evidence.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-052/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-052/Cargo.toml --locked --all-targets -- -D warnings

## Required research benchmark

Sweep:
- dimensions d in {1, 4, 16, 64, 256, 1024};
- independent reference blocks m in {32, 64, 128, 256, 512, 1024, 4096};
- observation norm bound R and feature scales;
- delta / alpha_test splits at fixed alpha_total;
- mean departures relative to the calibrated epsilon;
- test horizons 500, 2,000 and 10,000.

Compare:
- oracle known-mean target (upper bound);
- CMAQ-052 confidence-envelope target;
- naive plug-in point target with no coverage budget (invalid negative control);
- a predeclared conservative hand-set target.

Include iid exact null, wrong-reference-distribution negative control, serially dependent null negative control, and duplicated/common-cause reference block attacks. Report reference coverage, target widths, full-horizon null alarm probability with CIs, detection/misses/delay, and p99 test-update latency. Use at least 10,000 independent calibration/test replications per primary cell and CMAQ-044 paired effective-block inference.

## Claim boundary

The unconditional bound alpha_test+delta follows only under the iid stationary assumptions above. It is not a general confidence sequence for an arbitrary sequential process. The confidence envelope does not authenticate reference data or prove independence, and no physical safety conclusion follows without an independent sensor/provenance and ODD validation chain.
