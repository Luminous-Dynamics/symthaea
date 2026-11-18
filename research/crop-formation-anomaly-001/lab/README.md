# Crop Formation Lab — dependency-free Rust reference

This is the first executable component of CROP-INTEL-001. It is deliberately small and independent of Symthaea's root Cargo workspace.

## Current scope

- 2D point-set geometry descriptors: centroid, bounds, radius statistics, radial/angular histograms and approximate rotational symmetry residuals.
- Circle-layout replay: strict CSV ingestion, circle validation, tangent-gap contact graph, quantile-initialized radius/radial classes and centred neighbourhood motif counts.
- Frozen circle-grammar family evaluator: reruns centre, classing, radial bands, contacts and motifs for every declared class-count/contact-threshold configuration.
- Conditional radius-permutation null ensemble: holds centres and the radius multiset fixed, permutes radius-to-location assignments, and reruns every frozen family member for each replicate.
- Event-grouped benchmark splitting: deterministic approximately balanced grouped K-folds with train/validation complement and event/site/source leakage checks.
- Finite-hypothesis expected information gain in bits for candidate measurements, with strict prior/conditional distribution validation and known-answer tests.
- Candidate experiment ranking by raw and cost-normalized information gain. Actions failing caller-supplied hard gates remain visible but unscored with a required reason; predictive probabilities are caller-supplied and the selector never grants permission.
- Seeded measurement-sensitivity audit: explicit uniform centre/radius pixel-error bounds, full-pipeline reruns, sample spread, radius-class agreement and exact-signature stability; stress-test assumptions only, not inferred measurement errors.
- Crop-flow orientation-field descriptors: circular mean/resultant, direction entropy, bend-magnitude spread and nearest-neighbour alignment.
- 3D stem-pose orientation statistics: base-to-tip vector concentration, inclination from vertical, horizontal azimuth, 3D nearest-neighbour alignment and optional centre-relative radial/tangential alignment.
- First-order SI-unit mechanics: hollow elliptical section properties, section moduli, three-point and cantilever beam-test estimates, local curvature and outer-fibre stress.
- Piecewise-EI Euler–Bernoulli cantilever profiles under tip force or uniform transverse load, with separate linear translational and rotational root springs. Includes exact segment-wise integrals and ten new analytical tests. The event schema also records root-anchorage force/translation and moment/rotation curves to support independent root-spring calibration.
- Binary-grid representation with explicit D4 transforms (four rotations/reflections) and row-major/column-major traversal.
- Immutable-in-use decoder manifests with unique IDs; a standard 32-way D4 × traversal × ASCII-7 bit-order family; persist a canonical manifest digest/timestamp externally before challenge analysis.
- Predeclared 7-bit ASCII decoder specifications with complete attempt logging.
- Family-wise max-statistic matrix evaluator: reruns/receives the complete same-width test family per null replicate and rejects incomplete matrices rather than silently dropping tests; an integrated test connects observed family → null ensemble → max-statistic correction.
- Deterministic 2x2 grid switches preserving row/column margins for a declared null family (not perimeter or connected components).
- Binary entropy and Wilson intervals for simple benchmark reporting.

## Explicit non-goals

- No automatic semantic interpretation of decoded bytes.
- No claim that low rotational residuals, entropy, or printable text establish intentional encoding.
- No image segmentation, photogrammetry, field registration, validated living-plant constitutive model or physical mechanism inference.
- No production integration with Symthaea, Symtropy or Mycelix yet.

## Run locally

From this directory:

```sh
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

The crate has no third-party dependencies. It includes a local `[workspace]` declaration so that it is not silently pulled into the root Symthaea workspace. The current source contains 104 authored unit tests; their execution remains pending because this editing environment has no Rust toolchain.

## Statistical interpretation

The decoder runner records every configured attempt and does not rank candidates by linguistic plausibility. The empirical maximum-statistic p-value is valid only relative to the supplied exchangeable null replicates and the exact same complete search family. If rotations, reflections, thresholds, dimensions, bit orders or codebooks are searched, they must be included in each replicate's maximum. Post-hoc changes invalidate the original calibration.

This is a reference implementation, not a crop-circle decoder or a scientific result. Tests must be executed on the exact commit before any test-pass claim is made.
