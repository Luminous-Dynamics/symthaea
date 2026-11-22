# CETB RCAF research kernel

A dependency-free Rust research crate for repeatability-calibrated comparison of already-computed correlation matrices. RCAF is not a consciousness detector, clinical classifier, neuroimaging preprocessor or correlation estimator.

## Implemented components

- Weighted Jaccard for non-negative edge-weight vectors.
- Absolute, shared-positive, shared-negative and signed-shift correlation-matrix comparisons.
- Labeled ROI comparison that rejects mismatched order, duplicate labels and invalid dimensions.
- Participant-keyed within/between similarity summaries.
- Participant-level bootstrap confidence intervals and exact two-sided sign tests.
- Direct edge-polarity transition counts, including neutral-band transitions.
- Synthetic Gaussian-factor null/injection calibration plus a shared observation-only common-mode nuisance challenge in `src/simulation.rs`.
- Separate graph-Laplacian spatial nuisance generators in `src/spatial_nuisance.rs`: graph-smooth low-rank fields and localized heat-kernel fields, with deterministic per-mode RNG substreams and optional target-process coupling.
- Event-triggered broad and graph-localized exponential transient nuisance in `src/motion_transient.rs`, with explicit event onsets, signed amplitudes and decay constants.
- Matched signal-plus-nuisance dose response, with the same deterministic innovation stream as its nuisance-only counterfactual and separate direction-aware recovery outputs.
- Draft 2020-12 schemas and semantic validation for RCAF result receipts.

## Run the kernel tests

```sh
cargo fmt --manifest-path research/rcaf/Cargo.toml -- --check
cargo test --manifest-path research/rcaf/Cargo.toml
cargo clippy --manifest-path research/rcaf/Cargo.toml --all-targets -- -D warnings
```

The crate is deliberately isolated from the root Cargo workspace and has no third-party Rust dependencies.

## Run synthetic injection-recovery calibration

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_injection_recovery
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_injection_recovery_sweep
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_estimator_tournament
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_estimator_dose_response_tournament
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_common_mode_nuisance_sweep
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_spatial_nuisance
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_spatial_variance_calibration
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_factorial_counterfactual
```

The first example outputs a JSON report conforming to `injection-recovery.schema.json`. The sweep example evaluates loading increments `[0.0, 0.10, 0.20, 0.30, 0.40]` with common random numbers and outputs `injection-recovery-sweep.schema.json`. It estimates:
- null false-positive frequency under exchangeable same-state samples;
- two-sided difference detection, expected-direction recovery and wrong-direction rejection under a declared module-loading perturbation;
- 95% Wilson Monte-Carlo confidence intervals;
- median participant-paired gaps and positive-gap fractions.

At nuisance dose zero, the signal-plus-nuisance outputs must exactly match the clean injection outputs. At positive doses, the combined signal+nuisance cell is matched to the nuisance-only cell through a shared deterministic innovation stream; only the planted module loading differs. This measures whether nuisance reduces or reverses directional recovery.

The generator creates time-series observations from participant-specific global/module factors and region residuals, then estimates correlation matrices. Latent factors and region residuals use stationary AR(1) dependence (default rho=0.30). A separate common-mode nuisance factor can be injected into one observed repeat while holding latent global/module loadings fixed. The null compares two independent samples from the same underlying participant network. The alternative changes one predefined module's factor loading. Full protocol, assumptions and the next sensitivity grid are in `docs/neuroscience/RCAF_INJECTION_RECOVERY_PROTOCOL_V0.md`.

This is a software calibration world, not a model of biological consciousness. A passed injection test cannot establish that an empirical network shift is real, causal or clinically meaningful. A single Gaussian factor model does not characterize the full range of acquisition artifacts, autocorrelation, nonstationarity or preprocessing choices.

## Statistical unit

Each observation passed to participant-level inference must already be reduced within participant. Repeated windows, segment placements and trials are nested measurements; they must not be counted as independent people.

For participant p:

```text
paired_gap[p] = within_similarity[p] - between_similarity[p]
```

Positive gap means the between-condition network estimate is less similar than the within-condition repeat. Exact-zero gaps are ties in the sign test.

The descriptive normalized excess-distance fraction is not a variance decomposition, causal quantity or percentage of brain changed.

## Report integrity

`tools/validate_report.py` checks Draft 2020-12 shape and semantic invariants. Empirical signal analysis requires a verified source-data manifest. Independent replay additionally requires immutable code, configuration, environment and result digests.

Run the Python research checks:

```sh
python -m pip install -r research/rcaf/requirements-validation.txt
python -m unittest discover -s research/rcaf/tools -p 'test_*.py' -v
python -m unittest discover -s research/cetb/theory_design -p 'test_*.py' -v
```

## Evidence boundaries

The current kernel and synthetic harness establish only that the code behaves as specified on declared fixtures and simulated data. They do not establish:
- correctness of a neuroimaging preprocessing pipeline;
- calibration against EEG, MEG or fMRI signals;
- biological validity, causal attribution or consciousness inference;
- independent cohort reproducibility;
- clinical utility or superiority over established estimators.

A workflow result applies only to the exact source commit checked. Skipped, queued, absent or earlier-head jobs are not passes.


## Matched-input estimator tournament

`run_estimator_tournament` and the `rcaf_estimator_tournament` example replay the same simulated observations with three declared similarity definitions:
- absolute Weighted Jaccard: magnitude overlap, ignores polarity;
- signed-shift Weighted Jaccard: transforms each correlation from [-1, 1] to [0, 2], penalizing sign disagreement;
- normalized mean absolute edge error: one minus mean upper-triangle absolute error divided by two.

Each estimator receives the same seed, participant draws, and injection configuration. The tournament reports null false-positive frequency, two-sided difference detection, expected-direction recovery, wrong-direction rejection, Monte-Carlo Wilson intervals, and participant-level gap summaries. It does not automatically select a universal winner: different estimators have different construct semantics, and estimator performance must be read together with null calibration and effect sensitivity.

The output contract is `estimator-tournament.schema.json`.


## Reproducible CI evidence artifacts

The focused validation workflow runs all three simulation examples and creates:
- single-point null/injection report;
- five-point common-random-number dose-response curve;
- three-estimator matched-input tournament;
- `rcaf.execution-receipt/0.1.0` manifest hashing the RCAF source tree, exact commit, Rust/Cargo/Python environment and report outputs.

These are uploaded as a workflow artifact named `rcaf-simulation-evidence-<commit SHA>`. The receipt proves provenance of that CI run, not independent replay or biological validity.


## Estimator-by-effect-size surface

`run_estimator_dose_response_tournament` evaluates absolute WJ, signed-shift WJ and normalized mean absolute edge error across the same five injection strengths. Each metric receives the same seed and random-draw order at every dose. This is more discriminating than comparing three metrics only at a strong effect that saturates all methods.

The example output is governed by `estimator-dose-response-tournament.schema.json`. It records a separate full dose-response curve, including null rejection, detection power, Wilson intervals and paired-gap distributions, for each metric. It must not be used to select a universal winner: the similarity definitions have different construct semantics.


## Validate the proposed adversarial calibration plan

The plan is currently a proposed protocol; validating it does not execute simulations or qualify biological data.

```sh
python research/rcaf/tools/validate_adversarial_plan.py research/rcaf/adversarial-calibration-plan.example.json
python -m unittest discover -s research/rcaf/tools -p 'test_validate_adversarial_plan.py' -v
```

The validator checks the Draft 2020-12 schema plus semantic controls for separated seeds, exact nulls, positive-dose effects, matrix-level and pre-estimation signal-level injections, nuisance-only nulls, and required evidence outputs. It uses the existing `jsonschema` dependency in `requirements-validation.txt`.


## Direction-aware and nuisance-only calibration

Injection-recovery reports use schema version `rcaf.injection-recovery/0.3.0`. The separate `nuisance_only_rejection_rate` measures rejection under a common-mode observation artifact with unchanged latent loadings; it is not the exact same-observation Type-I error. The legacy `injection_detection_power` remains the two-sided significant-difference rate. It is decomposed into:
- `injection_directional_detection_power`: significant paired gaps in the preregistered positive direction of the module-loading injection;
- `injection_wrong_direction_rejection_rate`: significant paired gaps in the opposite direction.

The two directional rates partition the two-sided detections under the exact sign test. A wrong-direction rejection is never counted as successful recovery. The same fields are carried through the sweep and both estimator tournament reports. The report also records `common_mode_nuisance_loading` and `median_nuisance_only_paired_gap`. At zero nuisance strength, the nuisance control reuses the exact null sample and must match the null rejection rate and median gap exactly. A nonzero nuisance-only rejection rate is evidence of artifact sensitivity, not proof that the latent network changed.




The `rcaf_common_mode_nuisance_sweep` example evaluates nuisance strengths `[0.00, 0.10, 0.20, 0.30]`. It requires an exact zero-dose control. The nuisance RNG is derived from stable participant/replicate-specific streams separate from the latent-signal RNG, so the null and injected outcomes must remain identical across nuisance doses. Only the nuisance loading changes. The result contract is `common-mode-nuisance-sweep.schema.json`. This is still a synthetic common-mode artifact challenge, not a model of all motion, scanner, physiology or preprocessing artifacts.

### Shared observation-only nuisance challenge

The current generator adds an independent AR(1) common-mode factor to one observed repeat while leaving the participant's latent global/module loadings unchanged. Its rejection rate and median paired gap are reported separately from the exact same-observation null and the known latent module injection. This is the first nuisance family only; it does not model all physiology, motion, scanner drift, censoring or missingness. The output field `nuisance_only_rejection_rate` measures artifact-only rejection relative to latent truth, not conventional Type-I error under identical observed-data distributions.


## Spatial nuisance field generators

The separate `spatial_nuisance` module operates on an explicit weighted, undirected ROI graph. It does not infer spatial relationships from ROI ordering and does not depend on an anatomical atlas.

The first implementation provides:
- graph-smooth low-rank nuisance from non-constant low-frequency Laplacian modes;
- localized nuisance from heat-kernel diffusion seeded at declared ROI indices;
- an independent deterministic random substream per spatial mode;
- stationary AR(1) independent temporal factors and optional mixing with supplied target processes;
- exact reuse of one generated field across nuisance-only and signal-plus-nuisance counterfactuals;
- rejection of invalid graph weights/labels, invalid time parameters, degenerate bases and rank cuts through repeated Laplacian eigenvalues.

Run the synthetic generator smoke example:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_spatial_nuisance
```

The output contract is `spatial-nuisance-field.schema.json`. It records the exact ROI labels, weighted edges, per-family rank/diffusion parameters, target coupling, deterministic RNG algorithm/seed, realised field RMS and realised temporal autocorrelation so the field can be reconstructed from the archived report. Its path-graph fixture is a toy topology, not an anatomical atlas. The current `dose` parameter is an additive loading, not a calibrated nuisance variance fraction. The module generates nuisance fields only; it does not perform preprocessing, estimate connectivity, determine causality, or qualify biological claims. The spatial simulator must still be crossed with the four matched counterfactual cells and calibrated against empirical acquisition data before any biological robustness claim is supported.


## Spatial nuisance variance calibration

Raw additive loadings are not comparable across graph ranks, diffusion scales, seed sets, event amplitudes, temporal patterns or ROI geometries. The `spatial_calibration` module therefore exposes a family-agnostic matrix calibrator, `calibrate_nuisance_matrix_to_component_fraction`, alongside a compatibility wrapper that preserves spatial field factor metadata.

For any clean observation matrix X and already-generated nuisance matrix A, it scales A to a requested component-variance fraction:

```
Var(A) / (Var(X) + Var(A))
```

Variance and covariance are computed after centering each ROI's temporal series, then pooled equally across ROIs. The report separately records Cov(X,A), the covariance contribution to the total, and Var(X+A). The same calibrator can be applied to common-mode, graph-smooth, heat-kernel, transient, physiological or future empirical nuisance matrices. It calibrates magnitude only: it does not establish that a synthetic nuisance distribution represents a real scanner or biological artifact.

The spatial calibration smoke example remains:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_spatial_variance_calibration
```

The report contract is `spatial-variance-calibration.schema.json`. The factorial suite now uses the matrix calibrator to match all four nuisance families at a requested 10% component-variance fraction and reports the realized fraction and covariance separately. The family-specific event/spatial generators and all nuisance fields remain unchanged apart from a scalar multiplier.


## Four-cell matched counterfactual experiment engine

The `factorial_counterfactual` module accepts one clean matrix X, one predeclared target contribution S and one already-generated nuisance matrix A. It constructs all four cells:

| Cell | Observation |
|---|---|
| `clean_null` | X |
| `signal_only` | X + S |
| `nuisance_only` | X + A |
| `signal_plus_nuisance` | X + S + A |

The same immutable A matrix is used in both nuisance-containing cells; the engine never calls the nuisance generator or redraws its innovations. It reports matched-nuisance and matched-signal residuals plus the additive interaction residual. These are numerical construction invariants, not claims that real measurement artifacts are additive.

The report also records each cell's RMS, maximum absolute value and FNV-1a-64 diagnostic fingerprint. The fingerprints are convenient stable diagnostics, not cryptographic hashes; the workflow receipt provides the content-addressed evidence chain. Clean–nuisance variance and covariance are pooled within ROI and reported separately. A nominal component-variance fraction must never conceal the covariance contribution to the combined observation.

Run the multi-family smoke fixture:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_factorial_counterfactual
```

The fixture applies the same target signal to four synthetic nuisance families: common-mode, graph-smooth low-rank, localized heat-kernel and event-triggered motion-like transient. All four nuisance matrices are calibrated against the same clean observations to a requested 10% component-variance fraction using `calibrate_nuisance_matrix_to_component_fraction`; requested and realized fractions and clean–nuisance covariance remain separately visible. This example is a software integration fixture, not a physical or biological simulation. The output contract is `factorial-counterfactual.schema.json`. The complete design, benchmark targets and industry comparison protocol are specified in `docs/neuroscience/RCAF_FACTORIAL_COUNTERFACTUAL_ENGINE_V1.md`.


## Matched factorial signal × nuisance dose-response surface

The `factorial_dose_response` module varies target-signal amplitude and nuisance component-variance fraction while holding the underlying target and nuisance innovations fixed. For each point, it runs the four-cell engine `X`, `X+S`, `X+A`, `X+S+A`. It uses common random numbers across the grid: changing dose scales the frozen matrices instead of regenerating them. This makes neighbouring points directly comparable without stochastic redraw noise.

Run the 4 × 4 S3 transient smoke surface:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_factorial_dose_response
```

The example crosses signal scales `[0, 0.5, 1, 2]` with nuisance component-variance fractions `[0, 0.05, 0.10, 0.20]`. The report stores raw innovation fingerprints, every four-cell report, requested/realized fractions, covariance, combined variance and matched-cell invariants. The focused workflow validates exact zero controls and stable nuisance-only fingerprints across signal strengths, then hashes and archives the report. This is a construction/calibration surface only; estimator recovery, null false-positive rates and confidence coverage are the next qualification layer. The output contract is `factorial-dose-response.schema.json`, with the design specified in `docs/neuroscience/RCAF_FACTORIAL_DOSE_RESPONSE_SURFACE_V1.md`.


## Estimator response across factorial dose-response points

The factorial dose-response runner now converts each of the four raw time × ROI cells to a Pearson ROI-connectivity matrix, then evaluates the three existing RCAF similarity estimators:
- absolute weighted Jaccard;
- signed-shift weighted Jaccard;
- normalized mean absolute edge error.

Each dose point reports target signal distance, nuisance-only displacement, residual displacement on the signal background, normalized signal-retention fraction and nuisance-to-signal distance ratio. Zero-signal ratios are explicitly null rather than fabricated. It also reports the direction and projection gain of the signal-induced connectivity change before and after nuisance.

Run the same smoke fixture:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_factorial_dose_response
```

The JSON report adds estimator summaries and connectivity-space direction metrics for every one of the 16 points. This closes the gap between merely constructing four matched cells and evaluating how the existing connectivity estimators respond to them. These are synthetic per-matrix scores, not participant-level power, false-positive rates or confidence coverage. See `docs/neuroscience/RCAF_ESTIMATOR_RESPONSE_SURFACE_V1.md` for the metric definitions and next confirmatory benchmark requirements.


## Held-out null calibration and direction-aware power

The `held_out_null_calibration` module separates null threshold fitting from held-out evaluation. It uses disjoint, uniquely identified calibration and evaluation replicate sets, a conservative finite-sample upper-tail rank threshold, and two-sided 95% Wilson intervals for held-out null false-positive rate, signal detection, directionally correct detection and wrong-direction acceptance.

The estimator-response adapters make the statistic explicit:
- Calibration and held-out null statistic: nuisance-only connectivity displacement.
- Held-out signal statistic: clean-to-signal-plus-nuisance connectivity displacement.
- Direction: the connectivity-space signal directional cosine.

Run the deterministic 99-calibration / 100-evaluation smoke fixture:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_held_out_null_calibration
```

The report contract is `held-out-null-calibration.schema.json`. The implementation refuses overlapping calibration/evaluation replicate identifiers, duplicate IDs, invalid statistics and undefined signal directions. If the requested alpha is below the finite calibration resolution, it returns no critical value and makes no rejections. See `docs/neuroscience/RCAF_HELD_OUT_NULL_CALIBRATION_V1.md`.

These Wilson intervals describe Monte Carlo binomial uncertainty only. The current smoke fixture does not establish empirical FPR, power, participant/site independence or biological transportability. Calibration/evaluation splits in real data must be separated by independent participant/session/site clusters where dependence exists.


## Stratum-conditional calibration and worst-group shift guard

The `groupwise_null_calibration` module addresses a limitation of pooled null thresholds: an acceptable average false-positive rate can conceal a badly miscalibrated site, scanner, echo regime or motion-burden subgroup.

It supports two policies:
- a **stratum-specific threshold**, calibrated only on that predeclared stratum;
- a **common worst-group guard**, set to the maximum of the thresholds across all expected strata.

The common guard is unavailable if any expected stratum lacks enough calibration replicates to resolve alpha. An unseen evaluation stratum is explicitly assigned `abstain_unseen_stratum`; it never inherits another group's threshold. Missing evaluation groups are marked incomplete rather than recorded as zero false positives.

Run the synthetic three-stratum smoke fixture:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_groupwise_null_calibration
```

The report contract is `groupwise-null-calibration.schema.json`. It reports local and common-guard FPR, signal detection, directionally correct detections, wrong-direction acceptance, Wilson intervals, calibration resolution and worst observed group results. The specification is `docs/neuroscience/RCAF_GROUPWISE_SHIFT_CALIBRATION_V1.md`.

The finite-sample interpretation remains conditional on exchangeability within each predeclared stratum. This does not solve arbitrary continuous covariate shift, participant/site dependence or unseen-domain generalization. Per-estimator alpha is reported separately; post-hoc selection among the three estimators requires multiplicity control.


## Weighted continuous covariate-shift calibration

The `weighted_covariate_shift` module extends held-out calibration to continuous source/target covariate shift using supplied importance weights. Each evaluation point receives its own weighted upper-tail threshold, with its own test-point weight placed at (+infty). If that mass consumes more than the alpha tail budget, no finite threshold exists and the point is explicitly abstained.

Run the synthetic continuous-weight smoke fixture:

```sh
cargo run --manifest-path research/rcaf/Cargo.toml --example rcaf_weighted_covariate_shift
```

The report contract is `weighted-covariate-shift.schema.json`. It includes calibration ESS, an empirical chi-square divergence proxy, maximum normalized calibration weight, maximum/mean concentration and test-point mass shares. A known-ratio fixture can be marked exact-guarantee-applicable only with explicit source-support and conditional-score-invariance attestations. Estimated ratios always remain diagnostic-only, even when cross-fitted and trained on a disjoint ratio-estimation split.

The module does not estimate density ratios or claim arbitrary shift validity. Zero target weights are rejected as outside-support points; low ESS, unverified assumptions and excessive test-point mass produce explicit abstentions. See `docs/neuroscience/RCAF_WEIGHTED_CONTINUOUS_SHIFT_V1.md` for the weighted conformal assumptions and research frontier on sketched calibration and score-conditional leakage.


The weighted calibration module also exposes adapters from `FactorialEstimatorResponseReport`, pinning calibration/null scores to nuisance-only displacement and signal scores to clean-to-combined displacement, with the direction cosine carried through. Reordered/missing estimators and undefined signal directions fail closed.
