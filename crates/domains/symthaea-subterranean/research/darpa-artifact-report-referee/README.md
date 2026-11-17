# DARPA Artifact Report Referee v0.1

**Status:** standalone reference implementation and synthetic tests authored. No external dataset bytes are vendored.

## Purpose

Evaluate candidate artifact reports against a separately held ground-truth catalogue, using exact source type labels and an explicit 3-D localization tolerance. The evaluator is intentionally separate from the perception/planning path.

The initial target is a compact, rights-aware adapter for the historical DARPA SubT Systems Finals ground-truth CSVs. Source manifest: `DARPA_GROUND_TRUTH_SOURCE_MANIFEST_V0_1.json`.

## Contract

Inputs:
- ground-truth artifact ID, exact coordinate-frame ID, original source type string and x/y/z coordinates;
- candidate report ID, exact coordinate-frame ID, original source type string, x/y/z estimate and caller-provided priority;
- an expected coordinate-frame ID, explicit localization tolerance and optional maximum report count.

All coordinates must already be expressed in the exact expected frame and in metres. Frame mismatches reject; the evaluator never guesses CRS, axis order, units or transforms.

Evaluation:
- selects at most the configured report budget, highest priority first, stable lexical ID tie-break;
- matches reports one-to-one to ground truth;
- requires exact type-string equality unless a separately versioned external alias table is supplied in a future extension;
- computes Euclidean 3-D distance and admits an edge only when distance is within the explicit tolerance;
- uses deterministic maximum-cardinality bipartite matching, so duplicate reports cannot claim the same truth item twice;
- reports true positives, false positives, false negatives, precision, recall, per-class counts, matched distances, unselected reports and report-budget exclusions.

No source type normalization is performed. For example, `cellphone` and `cell_phone` remain distinct strings; any cross-system alias policy must be explicit and independently versioned.

## Historical DARPA scoring caveat

The published challenge used a five-metre location tolerance and a finite report allowance in its artifact-report scoring. This crate's generic configurable evaluator is **not yet an exact reproduction of every round's official scoring rules**. A 5 m setting is a local historical proxy unless the exact round/task rule, report allowance and any artifact-type scoring nuances are pinned against authoritative rules.

## Data and licensing

The supplementary repository is:
https://github.com/subtchallenge/systems_finals_ground_truth

Its source README describes the DARPA coordinate frame, fiducials, round CSVs, point cloud and maps. The repository's API metadata did not expose a clear SPDX licence for dataset redistribution. The manifest pins source commits/blob IDs and row counts but deliberately does not include the dataset files. Confirm data reuse rights before committing data bytes or derived maps.

The candidate-report generator must never receive ground-truth coordinates, artifact labels or point-cloud truth. Ground truth is evaluator-only.

## Run

```sh
cargo fmt --all -- --check
cargo test --offline --all-targets
cargo clippy --offline --all-targets -- -D warnings
```

No external crate dependencies are required.

## Claim ceiling

Passing the synthetic tests establishes only finite evaluator conformance for the defined class/3-D distance/report-budget contract. It does not validate perception, localization, mapping, operator workflow, communication, mission survival, structural/geotechnical safety, mine access or planetary transfer.
