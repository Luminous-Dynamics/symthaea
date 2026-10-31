# Hidden co-linear merge benchmark V1 — executable synthetic baselines

**Status:** executable 1D change-point baseline and adversarial synthetic fixture committed; no EM/mesh/skeleton dataset has been evaluated.  
**Tracking:** [SCI-CONNECTOMICS-016 / #9265](https://github.com/Luminous-Dynamics/symthaea/issues/9265).

## Why this is a distinct target

NEURD's published discussion identifies false merges that remain within a single non-branching, co-linear skeleton segment: graph-node and branch heuristics do not expose the interior boundary. It separately identifies incomplete neurite extension as a gap. The latter is tracked in [#9249](https://github.com/Luminous-Dynamics/symthaea/issues/9249). This benchmark addresses only the former.

## V1 prototype

The binary `connectome-colinear-merge-audit` consumes ordered 1D centreline feature profiles:
- radius;
- local tangent angle in a consistently unwrapped coordinate convention;
- membrane-integrity score;
- local synapse density.

It compares:
1. `no_proposal`, which estimates the miss rate when no candidate is generated;
2. `radius_only_piecewise_linear`;
3. `multifeature_piecewise_linear`.

At each possible split location, it compares one linear fit over a symmetric local window with two independent linear fits on either side. The score is the square root of summed, noise-normalized residual reduction, less a two-parameter overfit penalty per feature. A threshold and spatial non-maximum suppression produce boundary proposals.

This is deliberately a baseline, not a learned detector. It can be fooled by abrupt within-neuron compartment transitions, boutons, or acquisition artifacts. Such cases must remain hard negative controls; the output is only a review hypothesis.

## Optional V2 physical-arclength mode

The detector configuration schema now supports two optional fields:
- `resample_interval_nm`: fixed uniform physical arclength spacing for a deterministic resampled profile;
- `half_window_nm`: physical half-window used by the same local fit after resampling.

Both fields must be present together. Rust validation requires finite positive values, at least three grid samples per half-window, and an integer ratio `half_window_nm / resample_interval_nm`. Source samples may be irregularly spaced but must have strictly increasing physical coordinates. All four input feature channels are linearly interpolated onto the same declared arclength grid. Grid generation starts at the first source coordinate and includes only points at or before the last source coordinate; the full source interval is retained separately in the report's cable-length denominator.

When configured, the tool preserves the V1 sample-index radius-only and multifeature results and appends two explicitly named physical-mode results:
- `physical_arclength_radius_only_piecewise_linear`;
- `physical_arclength_multifeature_piecewise_linear`.

The report records the interval, physical half-window, effective resampled half-window sample count, interpolation policy and per-track original/resampled sample counts. This allows a reviewer to confirm that all compared profiles used the same physical receptive field. Keep the resampling interval, half-window, feature transforms, thresholds and minimum separation identical across a sampling-density metamorphic comparison.

The physical mode fixes the sample-index receptive-field problem; it does **not** solve statistical calibration. The score threshold remains a heuristic based on supplied scalar noise scales, features are still treated as independently scaled SSE gains, and an abrupt morphology transition is not proof of neuronal identity change. The physical variant must be calibrated on a disjoint same-neuron null set and lockbox before biological use. V1 remains the transparent control.

## Reproducible run

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-colinear-merge-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_COLINEAR_PROFILE_INPUT.example.json \
  --truth docs/neuroscience/examples/CONNECTOME_COLINEAR_BOUNDARY_TRUTH.example.json \
  --tolerance-nm 500 \
  --output colinear-merge-report.json

cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml connectome_colinear_merge_audit
```

The detector input and evaluator-only labels have separate schemas/files. The truth file binds the exact input SHA-256 and must include every track, including negative controls. Code executes both proposal generators before parsing truth. Do not expose truth labels or the input's truth digest to training or detector selection; for formal lockbox use, keep the truth file in an evaluator-only environment.

## Synthetic fixture and controls

The checked-in fixture contains:
- an abrupt radius-conditioned hidden boundary;
- a hidden boundary with constant radius but tangent, membrane, and synapse-density changes;
- a normal smooth linear taper;
- a bouton-like local swelling / synapse-density pulse;
- a temporary membrane-integrity acquisition artifact;
- an abrupt synapse-density compartment transition.

Only the first two tracks are labelled as hidden merge boundaries. The remaining four are explicit negative controls. The feature arrays are synthetic signals, not extracted EM or morphology data. With the checked-in threshold and feature-noise scales, the expected synthetic fixture report is: no-proposal 0/2 true positives; radius-only 1/2 true positives, 1 false proposal; multifeature 2/2 true positives, 7 false proposals. These are deterministic synthetic test expectations, not measured biological metrics. The high multifeature false-positive burden is an intentional warning about confusing boutons, imaging artifacts, and genuine compartment transitions with cell-identity boundaries.

## Metrics

The evaluator performs one-to-one nearest-error matching within a preregistered tolerance and reports:
- point recall and precision;
- F1;
- median and P90 absolute boundary localization error (nm);
- false proposals per cable mm;
- candidate counts and complete matched/unmatched counts.

For biological evaluation, also report results by species, region, neurite class, voxel anisotropy, radius regime, synapse density, and error subtype. Use cluster-level bootstrap at the source-volume/neuron level, never sample individual profile windows as independent observations.

## Required next comparisons

Before promotion, benchmark against:
- no-proposal and fixed radius-discontinuity baselines (the no-proposal comparator is included in V1);
- skeleton branch/node rules;
- NEURD morphology rules where the same data and adjudication are applicable;
- mesh-only, skeleton-only, EM-only and fused feature variants;
- calibrated change-point / Bayesian online change-point / multiscale detectors;
- expert reviewers blinded to detector identity.

Evaluate candidate recall at matched false proposals per cable mm and matched review-minute budgets. Include boundary-location error, precision, risk-coverage, calibration, latency, and compute/annotation cost.

## Research targets (not achieved)

- At least 95% point recall, with lower 95% confidence bound at least 90%, on a preregistered eligible boundary sample.
- At least 90% of localized boundaries within 500 nm, median absolute error no greater than 200 nm where voxel resolution and expert annotation uncertainty make this meaningful.
- At least 20% fewer expert minutes per independently verified hidden merge at matched precision and matched error-family coverage.
- False-positive burden no greater than the strongest specialist baseline at matched recall; report the actual false proposals/km instead of hiding the trade-off in F1.

## Promotion and safety gate

A proposal is never an automatic edit. Require independent human adjudication, source-image/mesh/skeleton evidence, exact source revision digests, conflict checks, and a query-scoped downstream impact assessment. The synthetic prototype does not establish a hidden merge, biological reconstruction correctness, or performance over NEURD, RoboEM, or ConnectomeBench2.
