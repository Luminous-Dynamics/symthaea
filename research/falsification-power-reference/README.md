# Falsification power reference v0

A dependency-free Rust reference crate for the statistical power ceiling created by a fixed experimental design.

This crate is a narrow work item for Symthaea issue #12375. It does not add a model generator, experiment selector, posterior engine, model IR, evidence ledger, or production model-adequacy decision.

## What it computes

For each evaluator-declared alternative, provide its expected residual at every observation in the fixed design:

- residual = alternative-model prediction minus working-model prediction;
- sigma = known standard deviation of independent Gaussian observation noise;
- alpha = predeclared one-sided per-alternative false-alarm level.

The crate computes the alternative-specific Gaussian matched-filter noncentrality and best-case power. It then reports every alternative separately, the least-detectable alternative, and whether all alternatives meet the requested power target.

The power envelope refuses alternatives with different vector lengths, preventing comparisons with different observation counts. The v0 API does not receive observation IDs, timestamps, sensor identity or cost weights: the calling evaluator must bind identical design IDs/order/units and account for acquisition and execution costs separately.

## Assumptions and claim ceiling

This is an **evaluator-side oracle diagnostic**, not an operational adequacy test:

- independent observations;
- common, known Gaussian noise standard deviation;
- a fixed design and fixed stopping rule;
- known discrepancy direction for each declared alternative;
- one-sided matched-filter test aligned to that known alternative;
- no nuisance-parameter estimation, sequential peeking, multiplicity correction, correlated noise, censoring, missingness or sensor uncertainty.

Those assumptions make the calculation optimistic. When the model parameters or noise process must be estimated, the actual test can have lower power. Adaptive policies require an appropriate sequentially valid test or predeclared alpha-spending rule. Correlated and heteroskedastic noise require a covariance-aware extension and independent validation.

A failed target is reported as **underpowered**, not as evidence for or against the working model. Meeting the target only refers to the finite, declared alternative envelope. It does not prove that the envelope contains every scientifically material alternative.

## API

- **known_direction_power** — best-case power for one declared discrepancy.
- **evaluate_power_envelope** — retain alternative-specific rows and compute worst-case power.
- **minimum_replicates_for_power** — solve the constant-effect known-variance sample-size reference.

## Run

From the repository root:

    cargo fmt --manifest-path research/falsification-power-reference/Cargo.toml -- --check
    cargo test --manifest-path research/falsification-power-reference/Cargo.toml
    cargo clippy --manifest-path research/falsification-power-reference/Cargo.toml --all-targets -- -D warnings

No external Rust crates are required.

## Known-answer controls

The 11-test suite includes:
- null power equals nominal alpha;
- the 5% one-sided normal critical value;
- an external normal-tail known answer (three observations, delta=0.24, sigma=0.15, alpha=0.05);
- sign symmetry of two-sided power;
- monotonicity with additional informative observations;
- the minimum-replicate boundary for delta=0.24, sigma=0.15, alpha=0.05 and 80% target (three replicates);
- alternative-by-alternative reporting and worst-case identification;
- rejection of mismatched budgets, duplicate identifiers and malformed inputs;
- no finite sample-size claim for a zero discrepancy.

These are authored Rust tests. They must be compiled and executed on the exact candidate head before any test-pass claim is made.


The alpha level is per declared alternative. A campaign-level claim that searches across alternatives must separately control family-wise error or use a qualified joint/sequential test; this reference does not do that.


## Sequential falsification reference

The sequential module adds predictable Gaussian likelihood-ratio e-processes and a deterministic adaptive-null simulation smoke test (20,000 paths, 100-look maximum horizon). The current exact-head Rust workflow passed 24/24 tests, rustfmt and strict Clippy; 13 tests are sequential-specific. The simulation is a smoke bound, not a formal type-I calibration claim:

- PredictableGaussianEProcess: one predeclared alternative, with a commit-prediction then observe protocol.
- PredictableGaussianMixture: a fixed, normalized mixture over predeclared alternatives, also using commit then observe.
- SequentialUpdate / MixtureSequentialUpdate: current and running-maximum log-evidence, an anytime p-value, and the alpha-boundary disposition.

The one-step log likelihood ratio is:

    d_t * y_t / sigma_t^2 - d_t^2 / (2 * sigma_t^2)

The per-step sigma_t is committed before the observation. The convenience constructor supplies a default sigma; commit_prediction_with_sigma and commit_predictions_with_sigma support known action-dependent measurement precision.

The reported anytime p-value uses the running maximum e-value: min(1, exp(-max_log_e_value)). It does not reset when later evidence moves against an alternative. Updates retain alpha, the committed discrepancy, sigma_t, observation/decision identities, component weights and the running-max evidence needed for independent replay.

This is valid under a simple conditional Gaussian null with known, positive sigma_t committed predictably with the discrepancy before each observation. It therefore supports known action-dependent measurement variance. It is not automatically valid for unknown or estimated variance, composite nulls, serial dependence not captured by the conditional model, post-hoc alternative selection or physical trajectories. The commit/observe API enforces local call order but cannot prove absence of out-of-band leakage.

See docs/research/SEQUENTIAL_FALSIFICATION_EPROCESS_V0_2026-11-15.md for assumptions, equations, benchmark controls and the distinction from SLICE and terminal-test induction.


## Sequential calibration report

The standalone binary sequential_null_calibration runs a deterministic synthetic operating-characteristic study:

    cargo run --locked --manifest-path research/falsification-power-reference/Cargo.toml --bin sequential_null_calibration

It reports, separately for a fixed-known-variance null, a predictable adaptive-variance null, and a material alternative:
- path count, horizon, fixed root seed and PRNG version;
- e-process boundary-crossing rate and 95% Wilson interval;
- naive repeated fixed-time 5% test crossing rate and 95% Wilson interval;
- mean look of e-process detection among paths that cross;
- undetected path count.

The focused GitHub workflow runs this binary and uploads the JSON, source commit identity and SHA-256 receipt. The fixed synthetic campaign uses 20,000 paths and 100 looks per scenario. Its checks are smoke gates for this exact generator: null e-process rates <=6%, and the adaptive-null naive repeated-look rate >10%. These are not universal statistical guarantees, a 5% calibration proof, confidence intervals across generator families, or physical validation. The naive repeated fixed-time test is deliberately invalid under optional stopping and is included only as a negative control.
