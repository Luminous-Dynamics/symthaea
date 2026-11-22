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
