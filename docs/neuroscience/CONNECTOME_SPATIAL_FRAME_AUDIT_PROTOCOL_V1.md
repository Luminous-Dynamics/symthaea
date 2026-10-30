# Spatial PSU lattice frame audit protocol V1

**Status:** executable frame-integrity contract and synthetic fixture; no biological volume frame has been audited.
**Tracking:** [SCI-CONNECTOMICS-013 / #9260](https://github.com/Luminous-Dynamics/symthaea/issues/9260).
**Downstream evaluator:** [SCI-CONNECTOMICS-006 / #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204).

## 1. Why frame integrity is a separate gate

A probability sample is only meaningful for the finite population actually represented by its frame. The current candidate-generation coverage evaluator receives PSU IDs and strata; those values do not establish that spatial units tile their declared domain without gaps or overlaps. A numerically correct Horvitz–Thompson estimator over a malformed frame estimates the wrong target population.

V1 therefore makes a narrow, testable claim: every integer voxel in an explicitly declared axis-aligned bounding box belongs to exactly one spatial PSU tile, and every tile is either eligible for sampling with one declared stratum or explicitly excluded with a reason.

## 2. Domain and tiling semantics

The frame binds:
- source volume ID, revision ID and exact content digest;
- segmentation ID, revision ID and exact content digest;
- coordinate frame ID, transform digest, and voxel size in nanometres;
- signed integer origin voxel and positive integer extent in voxels;
- positive integer tile shape in voxels;
- fixed `clip_last_tile` boundary rule;
- eligibility-mask artifact digest and selection-rule digest;
- exact frame protocol digest and frame-generator tool/version/binary digest.

Grid dimensions are derived per axis using overflow-safe ceiling division:

`grid_dim = 1 + (extent_voxels - 1) / tile_shape_voxels`.

Each tile is identified by a zero-based integer `grid_index = [x,y,z]` and canonical PSU ID `x:y:z`. Its start is `origin + grid_index * tile_shape`; its extent is the tile shape clipped to the remaining declared box at the boundary. Integer voxel coordinates—not floating physical coordinates—define membership. The coordinate transform and voxel size are provenance bindings.

The manifest must contain exactly one record for every grid coordinate. Each record has exactly one disposition:
- `eligible`: a non-empty `stratum_id` and null `exclusion_reason`;
- `excluded`: null `stratum_id` and a non-empty `exclusion_reason`.

The eligible `population_frame` exported by the auditor is sorted by canonical PSU ID and consists only of `{psu_id, stratum_id}` pairs accepted by #9204.

## 3. What the verifier proves—and does not prove

Given the supplied manifest, it proves:
- all grid coordinates inside the declared voxel box are present exactly once;
- no declared tile lies outside the box;
- no lattice cell is duplicated or assigned multiple dispositions;
- partial boundary tiles have deterministic clipped extents;
- eligible and excluded tiles partition the declared voxel count exactly;
- canonical frame and eligible population digests are deterministic under input-row shuffling.

It does **not** prove:
- the declared bounding box covers all available tissue or all biologically eligible tissue;
- the eligibility-mask artifact corresponds to the stated source bytes or selection rule;
- source, segmentation, transform, and mask SHA-256 claims are true (the verifier does not fetch external artifacts);
- tile-search units are biologically independent or that expert search is exhaustive;
- a lattice frame is appropriate for neuron-based or irregular PSUs.

Those claims require separate source-artifact inspection, domain review, and a different protocol for non-lattice sampling units. A passing frame audit only validates internal geometry within the declared box.

## 4. CLI and downstream integration

Run the synthetic fixture:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-spatial-frame-audit -- \
  --frame docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME.example.json \
  --output spatial-frame-report.json
```

Before running #9204, export the report's `population_frame` array as the candidate-coverage audit's `population_frame`. Do not independently maintain a second PSU list. Bind #9204's frame digest to the `canonical_frame_sha256` from this report and retain both the eligible population digest and complete frame digest.

## 5. Adversarial controls

The Rust suite covers valid partial boundary tiles, exact voxel partition totals, missing cells, duplicate coordinates, canonical ID mismatch, out-of-bounds coordinates, eligibility/stratum errors, exclusion-reason errors, all-excluded frames, zero extents/tiles, malformed source/transform digests, invalid voxel sizes, row-order-independent hashes, changed stratum identity, u128 overflow, and negative origin handling.

A biological intake must additionally verify that source volume bytes, segmentation revision, transform, and eligibility mask are exactly those named in the manifest. A synthetic pass tests arithmetic and manifest semantics only.

## 6. Promotion gate

No candidate-generation recall estimate is promoted unless its spatial PSU frame first passes this audit, the source/mask artifacts are independently verified, and the exact exported eligible population is used by #9204. The claimed population is always limited to the declared frame and eligibility rule.

## Downstream binding

The coverage auditor accepts this tool's exact report through `--spatial-frame-report`. Its report binds the complete canonical spatial-frame digest and an eligible population digest computed with the same length-prefixed SHA-256 encoding as #9204. The population frame in the coverage audit must be byte-semantically identical after sorting by PSU ID; no second hand-maintained frame is accepted.
