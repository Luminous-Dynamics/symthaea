# CMAQ-052 — Common-conditional-mean martingale reference envelope and alpha split

## Why this is needed

CMAQ-046's e-process requires the true conditional null mean to remain inside a frozen convex target. A digest can bind a target file, but does not establish that the target contains the true mean.

CMAQ-052 constructs a simultaneous confidence box from a bounded reference phase, reserves a target-coverage failure budget delta, and allocates the remaining risk to CMAQ-046's sequential e-process.

## Statistical contract: a common conditional mean

Let r_1,...,r_m be reference increments adapted to a filtration G_i. Assume there is one fixed vector mu such that

    E[r_i | G_(i-1)] = mu
    ||r_i||_2 <= R

for every reference increment i. Thus every coordinate is in [-R,R]. Let mu_hat be the reference sample mean in dimension d. For a reference coverage failure budget delta, set

    epsilon = R * sqrt(2 * log(2d/delta) / m)
    S_hat = product_j [mu_hat_j - epsilon, mu_hat_j + epsilon].

Each coordinate's centered increments are a martingale difference with conditional range width at most 2R. Azuma-Hoeffding plus a union bound gives

    P(mu not in S_hat) <= delta.

Now continue with a test phase under an enlarged filtration F_t that contains all reference history and previous test observations. Require the *same fixed conditional mean*:

    E[r_t | F_(t-1)] = mu
    ||r_t||_2 <= R.

On the event mu in S_hat, the conditional mean of each test observation remains in the frozen CMAQ-046 target. The inner e-process, run at alpha_test, therefore controls the probability of ever crossing 1/alpha_test. Unconditionally,

    P(false alarm ever) <= alpha_test + delta <= alpha_total.

The split alpha_test + delta <= alpha_total is checked with exact rational arithmetic. The runtime betting threshold is 1/alpha_test, not 1/alpha_total; delta accounts for failure of the reference envelope.

## What dependence is allowed—and what is not

The common-conditional-mean martingale contract permits:
- serial dependence through the filtration;
- predictable heteroskedasticity, where variance changes with past observations;
- test data whose variance depends on the reference history;
- non-identically distributed increments, provided their conditional mean remains the same fixed mu and the L2 bound R holds.

It does not permit:
- a conditional mean that drifts over time;
- conditional means that depend on unmodeled covariates or regimes;
- reference/test phases with different conditional means;
- common-cause duplicate blocks that violate the conditional-mean construction;
- unbounded reference or test vectors.

This is not a general confidence sequence for arbitrary changing-mean processes. Those require a separate conditional-mean confidence-sequence construction.

## Implementation

The dependency-free Rust crate depends on CMAQ-046 by local path. It provides:
- typed bounded reference increments with stable block IDs and trace digest references;
- duplicate reference ID and duplicate trace rejection;
- finite value, dimension and L2 norm checks;
- simultaneous coordinatewise Azuma-Hoeffding confidence box;
- frozen BoxTarget around the reference sample mean;
- exact rational alpha_test + delta <= alpha_total;
- ReferenceEnvelopeReceipt containing dimension, reference count, R, radius, risk split and digest references;
- CMAQ-046 monitor initialized at alpha_test;
- test-block ID exactly-once protection in-process;
- atomic invalid-test behavior.

SHA-256 values are checked for shape only. They do not authenticate source identity, prove reference provenance or establish the common-conditional-mean assumption. Cross-process replay prevention still requires a durable external ledger.

## Tests

Deterministic tests cover:
- exact risk budget equality and over-budget rejection;
- invalid risk fractions;
- Hoeffding/Azuma radius identity;
- reference mean containment in the constructed target;
- duplicate reference IDs and trace digests;
- malformed digest, nonfinite reference value and norm violations;
- dimension checks;
- atomic invalid test observations;
- duplicate test block rejection;
- persistent mean shift outside the target accumulating evidence;
- a deterministic predictable-heteroskedastic martingale wiring fixture that carries the final reference sign into the test phase, varies conditional variance from past sign only, and checks the frozen reference target and subsequent test updates.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-052/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-052/Cargo.toml --locked --all-targets -- -D warnings

## Paired reference-envelope probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-052/Cargo.toml --locked --bin reference_envelope_probe -- 1000 2000 20261128 1000 > cmaq-052-envelope-results.csv

Each replication generates an adapted reference phase and then an adapted test phase. Three scenarios are included:
- iid symmetric null;
- predictable heteroskedastic martingale null, where variance depends on the previous sign but the conditional mean remains zero;
- mean shift after a fixed pre-change prefix.

Each run compares:
- calibrated Hoeffding/Azuma target, alpha_test=0.025 and delta=0.025;
- oracle known-mean point target at alpha_test=0.025 (upper-bound comparator);
- naive reference sample-mean point target without coverage budget (invalid negative control).

The output includes whether the calibrated target contains the known null mean, false alarms, pre-change alarms, detections, misses and detection delay. The CI run is a small wiring smoke test, not a calibrated confidence claim.

## Required benchmark

Sweep:
- dimensions d in {1, 4, 16, 64, 256, 1024};
- reference increments m in {32, 64, 128, 256, 512, 1024, 4096};
- observation norm bound R;
- delta / alpha_test splits at fixed alpha_total;
- target width and mean shift size relative to epsilon;
- test horizons 500, 2,000 and 10,000.

Compare:
- oracle known-mean target (upper bound only);
- CMAQ-052 simultaneous confidence envelope;
- naive plug-in point target without a coverage budget (invalid negative control);
- conservative fixed hand-specified target.

Include iid null, predictable heteroskedastic martingale null with constant conditional mean, changing-conditional-mean negative control, reference/test common-mean mismatch, and duplicated/common-cause reference block attacks.

Report empirical reference-envelope coverage, target radius/width, full-horizon false alarm probability with confidence intervals, misses, H+1-restricted delay, p50/p90/p95 detection delay, and p99 test-update latency. Use at least 10,000 independent full-horizon paths per primary cell and CMAQ-044 paired effective-block inference.

## Claim boundary

The unconditional bound alpha_test+delta applies under the common-conditional-mean martingale assumptions above. It is not valid for processes whose conditional mean changes over time or depends on unmodeled covariates. The confidence envelope does not authenticate reference data, prove source independence/provenance, or establish physical sensor truth.
