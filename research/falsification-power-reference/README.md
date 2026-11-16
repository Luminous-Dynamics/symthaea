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


## Fixed-design Gaussian terminal-test induction

The gaussian_terminal_induction module derives an anytime-valid randomized test process from a frozen simple-vs-simple Gaussian terminal Neyman–Pearson test.

For a deterministic design with H0 residuals Normal(0, sigma_t^2) and H1 residuals Normal(d_t, sigma_t^2), the induced prefix value is the null-conditional probability that the fixed terminal NP test rejects. It is a [0,1]-valued test value, not a p-value. At any bounded stopping time, an independent uniform randomizer U and the rule U < phi_tau preserve the terminal test's level; at the planned horizon the process equals the original terminal test exactly.

This v0 requires all future d_t and sigma_t to be frozen before observations. It is not the adaptive Gaussian e-process in the sequential module, does not cover unknown variance or composite nulls, and does not automatically transfer to physical trajectories. See docs/research/GAUSSIAN_TERMINAL_TEST_INDUCTION_V0_2026-11-16.md.


## Paired Gaussian terminal-test method comparison

The standalone binary gaussian_terminal_induction_calibration runs the fixed-design comparator over a preregistered induced-test stopping-threshold grid of 0.75, 0.90 and 0.99:

    cargo run --locked --manifest-path research/falsification-power-reference/Cargo.toml --bin gaussian_terminal_induction_calibration

For the same generated outcome path it compares:
- the fixed-horizon Neyman–Pearson terminal test;
- the induced conditional-rejection test with an early-stop threshold and independent randomization at the stopping time;
- the Gaussian e-process Ville-boundary crossing rule;
- naive repeated fixed-time z tests as an invalid optional-stopping control.

It reports null/alternative rejection rates with Wilson intervals for every threshold, early-stop frequency, mean stopping look, e-process crossing delay, the analytically computed terminal NP power and a terminal-equivalence mismatch count. The same observation paths are reused across all four methods; each threshold has a separate randomizer stream. The binary uses separate deterministic observation and randomizer streams; its PRNG is for simulation only.

This campaign is restricted to one frozen simple Gaussian design. It is not a general adaptive-discovery benchmark or physical validation.


### Exact-head paired calibration receipt

Run #41 on current PR head e9770b48170016e51b270b0bcfdd67a95c5604c2 passed the focused workflow: rustfmt, all-target tests, sequential null calibration, paired Gaussian calibration, artifact upload and strict Clippy. Workflow: https://github.com/Luminous-Dynamics/symthaea/actions/runs/43822231600

The campaign used 20,000 paired paths per scenario, an 80-look fixed design, alpha 0.05, discrepancy 0.22 and sigma 0.80. The predeclared induced-test thresholds were 0.75, 0.90 and 0.99. Every method shared the same observations; each threshold used a separately seeded simulation randomizer stream. The PRNG is for synthetic replay only.

| Method | Null rejection | Alternative rejection | Mean observations used under alternative |
|---|---:|---:|---:|
| Fixed-horizon terminal NP | 5.435% | 79.970% | 80.00 |
| Induced test, threshold 0.75 | 5.320% | 66.055% | 60.69 |
| Induced test, threshold 0.90 | 5.415% | 75.150% | 65.95 |
| Induced test, threshold 0.99 | 5.410% | 79.550% | 71.43 |
| Gaussian e-process | 2.815% | 62.420% | 58.53 |
| Naive repeated fixed-time test | 29.795% | 91.055% | 26.21 |

The induced-method null Wilson 95% intervals are [5.017%,5.640%], [5.110%,5.737%] and [5.105%,5.732%], respectively. Terminal NP has [5.129%,5.758%]. The e-process null Wilson interval is [2.595%,3.053%]. The naive repeated-test null interval is [29.165%,30.433%].

The analytic terminal NP power is 79.241%; the observed terminal estimate is 79.970% (Wilson 95% [79.410%,80.519%]), a 0.73 percentage-point Monte Carlo deviation. The terminal induced decision matched the fixed NP decision on all 20,000 paths. All result rates are conditional on this one declared generator/design and do not establish general calibration.

Interpretation:
- Threshold 0.99 is the near-terminal point: 79.55% power with about 10.7% fewer observations than the 80-look test.
- Threshold 0.90 gives about 17.6% fewer observations, with 75.15% power.
- Threshold 0.75 gives about 24.1% fewer observations, with 66.06% power.
- The Gaussian e-process crosses less often and is more conservative here; its average observation cost counts non-crossers as running to the horizon.
- The naive repeated test appears cheapest and most powerful, but its ~29.8% null rejection makes it invalid under repeated looks.

The terminal NP method is an evaluator oracle because it knows the correct discrepancy direction and frozen design. This is not a Symthaea agent result or an unknown-mechanism discovery score. Final artifact archive SHA-256: a45b26b33efb0c1caa1140c06da4411705534e187a867775bf783e01d5f982ef; report JSON SHA-256: 3ce1b598bf5ff280ace6d4c8901b2df36ce4ea08a02e1d2d70f6d757d2b0b67f.


## Multi-seed terminal-test calibration

The standalone binary `gaussian_terminal_multiseed_calibration` repeats the frozen 80-look simple Gaussian null/alternative campaign across five preregistered root seeds, with 20,000 paths per seed and scenario:

    cargo run --locked --manifest-path research/falsification-power-reference/Cargo.toml --bin gaussian_terminal_multiseed_calibration

It preserves every seed, emits per-seed and pooled terminal NP rates/Wilson intervals, all three induced stopping thresholds (0.75, 0.90, 0.99), randomized rejection rates, early-stop rates, mean stopping looks and zero-mismatch counts. The same observation path is used within each method comparison; each threshold uses a separate randomizer stream.

The workflow checks 100,000 pooled paths per scenario, the five-seed inventory, Wilson interval consistency, null operating characteristics, alternative analytic-power proximity and the monotone early-stop-cost relationship. This is a numerical stability diagnostic for one frozen synthetic Gaussian design only. See `docs/research/GAUSSIAN_TERMINAL_MULTISEED_CALIBRATION_V0_2026-11-16.md`.


### Exact-head multi-seed stability result (2026-11-16)

Exact head `eadfdca561b50fda49ba5f61591e08071aaf9b04` passed format, all-target tests, both existing calibration campaigns, the five-seed campaign, all artifact uploads and strict Clippy. The new campaign produced:
- pooled null terminal NP rejection: 5.026% (Wilson95 4.892–5.163%);
- pooled alternative terminal NP rejection: 79.476% (Wilson95 79.225–79.725%) against analytic power 79.241%;
- 0.75 / 0.90 / 0.99 stopped-test alternative rejection: 65.477% / 74.723% / 79.104%;
- corresponding early-stop rates: 79.246% / 75.714% / 70.680%;
- zero terminal-induced mismatches across 200,000 paths.

The five-seed output retains every per-seed record and is source/report-digest bound. This remains a fixed Gaussian simulation calibration, not a general sequential-test or physical-validation claim. Details and digests are in `docs/research/GAUSSIAN_TERMINAL_MULTISEED_CALIBRATION_V0_2026-11-16.md` and workflow https://github.com/Luminous-Dynamics/symthaea/actions/runs/43849094332.


## Gaussian assumption-stress campaign

Run the fixed terminal-test assumption map:

    cargo run --locked --manifest-path research/falsification-power-reference/Cargo.toml --bin gaussian_assumption_stress

The binary holds the fixed iid-Gaussian test unchanged and measures its terminal rejection rate under matched iid Gaussian, stationary AR(1), variance-standardized Student-t(5), smooth heteroscedasticity, abrupt variance-regime shift and sparse gross contamination. Only the iid Gaussian positive control is labelled IN_CONTRACT. Every other stress row is labelled OUT_OF_CONTRACT regardless of its empirical rate.

Five frozen root seeds × 10,000 paths per seed × two paired scenarios are reported per regime, with Wilson intervals, residual moments and lag-one autocorrelation. This is an assumption-failure map for a fixed terminal test, not a robust replacement method or an extension of the Gaussian theorem.

See docs/research/GAUSSIAN_ASSUMPTION_STRESS_V0_2026-11-16.md and Symthaea issue #12540.


### Known-rho conditional AR(1) comparator

The v1 stress report adds a separately labelled conditional-innovation terminal-test comparator for the AR(1) regime only. It uses the frozen evaluator-known rho=0.65 and transforms raw residuals into independent innovations with per-look conditional means and standard deviations. The comparator has its own analytic power and null/alternative rates; it is not the same test as the original IID Gaussian terminal statistic and is not a blind rho estimator.

Only the AR(1) row contains `conditional_known_rho_ar1`; all other rows emit null. This comparator is valid only under the explicitly known, correctly specified Gaussian AR(1) conditional model. See #12442 for the conditional-innovation scope and the stress note for the exact transformation.


The stress report v1 also includes a **known-rho conditional AR(1) comparator** using the same 50k+50k AR(1) paths. On the exact run, the naive IID test's null rejection was 21.820%; the known-rho innovation test returned 5.026% null rejection (95% Wilson 4.838–5.221%) and 31.112% alternative power versus analytic 31.348%. This restores null size at the cost of lower power under the correctly specified AR(1) model. It is an evaluator-known-rho fixed-terminal comparator, not a blind noise estimator and not #12442's sequential e-process.
