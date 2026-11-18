# Crop Formation Lab — dependency-free Rust reference

This is the first executable component of CROP-INTEL-001. It is deliberately small and independent of Symthaea's root Cargo workspace.

## Current scope

- 2D point-set geometry descriptors: centroid, bounds, radius statistics, radial/angular histograms and approximate rotational symmetry residuals.
- Crop-flow orientation-field descriptors: circular mean/resultant, direction entropy, bend-magnitude spread and nearest-neighbour alignment.
- Binary-grid representation with explicit D4 transforms (four rotations/reflections) and row-major/column-major traversal.
- Immutable-in-use decoder manifests with unique IDs; a standard 32-way D4 × traversal × ASCII-7 bit-order family; persist a canonical manifest digest/timestamp externally before challenge analysis.
- Predeclared 7-bit ASCII decoder specifications with complete attempt logging.
- A family-wise maximum-statistic empirical p-value helper. The caller must run the **entire** decoder/transform family on each null replicate and provide that replicate's maximum score.
- Deterministic 2x2 grid switches preserving row/column margins for a declared null family (not perimeter or connected components).
- Binary entropy and Wilson intervals for simple benchmark reporting.

## Explicit non-goals

- No automatic semantic interpretation of decoded bytes.
- No claim that low rotational residuals, entropy, or printable text establish intentional encoding.
- No image segmentation, photogrammetry, field registration, plant biology or physical mechanism inference yet.
- No production integration with Symthaea, Symtropy or Mycelix yet.

## Run locally

From this directory:

```sh
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

The crate has no third-party dependencies. It includes a local `[workspace]` declaration so that it is not silently pulled into the root Symthaea workspace. The current source contains 36 authored unit tests; their execution remains pending because this editing environment has no Rust toolchain.

## Statistical interpretation

The decoder runner records every configured attempt and does not rank candidates by linguistic plausibility. The empirical maximum-statistic p-value is valid only relative to the supplied exchangeable null replicates and the exact same complete search family. If rotations, reflections, thresholds, dimensions, bit orders or codebooks are searched, they must be included in each replicate's maximum. Post-hoc changes invalidate the original calibration.

This is a reference implementation, not a crop-circle decoder or a scientific result. Tests must be executed on the exact commit before any test-pass claim is made.
