# FDM adaptive spatial experiment design: synthetic benchmark v1

## Purpose

Compare ways to choose the next spatial calibration point under the same observation budget. The paradigm tested is **measurement planning as an optimization problem**: spend measurements where they are informative, while retaining explicit evidence gaps and critical-region sensitivity.

This benchmark is synthetic. It is not a validated printer model, a calibration recommendation, a certification method, or machine-control authority.

## Research foundation

ISO/ASTM 52902:2023 describes artifact geometries and measurements for additive-manufacturing geometric capability assessment and calibration. It addresses the features to measure and measurement uncertainty without imposing one universal measurement technique or machine-setting recipe: https://www.iso.org/standard/79683.html.

The ASTM listing emphasizes that observed performance belongs to a particular AM system setup, process parameters, and feedstock material; re-use across changed conditions requires evidence: https://store.astm.org/f3345-23.html.

NIST's AM test artifact documentation similarly calls for recording process parameters and machine settings and comparing measured deviations from nominal geometry: https://www.nist.gov/el/intelligent-systems-division-73500/production-systems-group/nist-additive-manufacturing-test.

These sources support measuring across features/positions and preserving context. They do **not** establish that the adaptive selectors in this code improve a physical process.

## Frozen protocol

- Evidence label: `synthetic` for every report.
- Common start: five observations, at four inner build-volume corners and the center.
- Candidate pool: a fixed 9 × 9 lattice from normalized coordinates 0.1 through 0.9.
- Additional measurement budget: eight observations per strategy.
- Total budget: 13 observations per arm, including the five shared starting measurements.
- Candidate selectors receive the available candidate locations, measurements acquired so far, seed and selection step. They are not passed the latent surface function or the evaluation grid.
- Evaluation occurs after each complete trace is selected, on a fixed 21 × 21 grid. Evaluation targets do not select further measurements.
- Seeded measurement noise is deterministic for replay. The heteroskedastic case has larger measurement noise on one side; the null-effect case has zero latent spatial error with small measurement noise.
- A domain-separated SHA-256 binds schema, surface, strategy, seed, ordered observations and recomputed metrics. It binds bytes/identity; it is not a signature or proof of trusted origin.

## Strategy arms

1. `fixed_stratified`: frozen inner-grid positions.
2. `farthest_point`: choose the candidate with greatest distance from any previously measured position.
3. `model_disagreement`: choose the candidate where nearest-neighbor and inverse-distance-weighted predictions disagree most.
4. `seeded_random`: deterministic random candidate baseline.

The selector logic receives only observed values and the candidate pool. No strategy is presumed best. The disagreement policy is a simple baseline, not a claim of optimal Bayesian experimental design.

## Adversarial surface panel

- Smooth quadratic variation.
- Anisotropic/directional response.
- Edge roll-off.
- Narrow localized hotspot.
- Piecewise discontinuous boundary.
- Smooth latent response with heteroskedastic measurement noise.
- Null latent spatial effect with measurement noise, to detect spurious confidence in apparent patterns.

Surface selection, parameters and noise are part of the frozen schema/version; new cases should be versioned rather than silently replacing prior fixtures. Build-to-build drift, orientation strata, feature classes, repeated physical artifacts and real measurement uncertainty are future extensions rather than silently approximated here.

## Metrics

The report preserves multiple dimensions:

- Observation count and incremental budget.
- Mean and maximum distance from the evaluation grid to the nearest measurement.
- MAE and maximum absolute error for a simple nearest-measured-value predictor.
- Normalized path length through the measurement sequence as a proxy only; it is not actual probe time.
- Number of critical evaluation points at a preregistered latent-error threshold.
- Critical-region recall: fraction of those grid points within the declared detection radius of a sample whose measured error crosses the threshold. This is a synthetic spatial-detection proxy, not physical defect detection.
- False-alarm observations where measured magnitude crosses the threshold but the latent synthetic value does not.

The critical threshold (0.10 mm), detection radius (0.08 normalized units), evaluation grid and budget are explicit fixture constants. They are **benchmark parameters**, not universal dimensional tolerances or acceptance recommendations.

## Negative controls and target gates

- Deterministic replay: identical seed, code, policy and surface produce equal traces and reports.
- Every strategy has exactly 13 distinct positions, with the same five initial points and eight additional measurements.
- Duplicate positions, invalid/non-finite measurements, altered report lineage, and accidental split overlap must be rejected as extensions add importable corpora.
- Null effect: latent critical-region count is zero and recall is undefined (not zero or 100%).
- Localized hotspot and piecewise boundary must remain present as adversarial cases.
- Compare Pareto trade-offs; do not declare an adaptive winner based on one surface or one seed.
- A stronger claim requires repeated surface/noise seeds, alternative grid resolutions, budget curves, equal realistic measurement costs, and independent evaluation of uncertainty and critical-feature strata.
- Physical-pilot claims require an identified machine/profile, material lot, feature and orientation, instrument/calibration, repeated builds, uncertainty and failure/missingness lineage. No synthetic result counts as physical confirmation.

## How to run once an executable checkout is available

From the repository root, run focused commands on the **exact PR head SHA**:

```sh
cargo fmt --check -- crates/domains/symthaea-fabrication-kernel/src/fdm_spatial_experiment_design.rs
cargo test -p symthaea-fabrication-kernel fdm_spatial_experiment_design
cargo clippy -p symthaea-fabrication-kernel --all-targets -- -D warnings
```

The formatting command may need workspace-supported syntax depending on the pinned Rust toolchain; if so, run workspace `cargo fmt --check` and record the exact command/output. Do not represent skipped or queued CI as a pass.
