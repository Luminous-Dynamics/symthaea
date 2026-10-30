# Spatial PSU lattice frame and eligibility-mask audit protocol V1

**Status:** executable geometry, byte-integrity, bit-order, and PSU-disposition verifier; synthetic fixture checked in, no biological volume has been audited.  
**Tracking:** [SCI-CONNECTOMICS-013 / #9260](https://github.com/Luminous-Dynamics/symthaea/issues/9260).  
**Downstream evaluator:** [SCI-CONNECTOMICS-006 / #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204).

## 1. Why this gate exists

A probability sample is only meaningful for the finite population represented by its frame. PSU IDs and strata do not establish that spatial units tile their declared domain without gaps or overlaps. Nor does an eligibility-mask digest alone show that each PSU's eligible/excluded label agrees with the mask bytes.

V1 now requires both an exact geometric lattice audit and a byte-level eligibility-mask audit before the report can carry `status: PASS`.

## 2. Domain and tiling semantics

The frame binds:
- source volume ID, revision ID, and content digest;
- segmentation ID, revision ID, and content digest;
- coordinate frame ID, transform digest, and voxel size in nanometres;
- signed integer voxel origin and positive integer extent;
- positive integer tile shape and fixed `clip_last_tile` boundary rule;
- eligibility-mask artifact ID, SHA-256, selection-rule SHA-256, and encoding;
- exact frame protocol digest and frame-generator tool/version/binary digest.

Grid dimensions use overflow-safe ceiling division:

`grid_dim = 1 + (extent_voxels - 1) / tile_shape_voxels`.

Each tile is indexed by zero-based `grid_index = [x,y,z]` and canonical PSU ID `x:y:z`. The last tile is clipped to the declared box. The full integer lattice is x-fastest for mask linearization: `linear = (z * extent_y + y) * extent_x + x`.

## 3. Eligibility-mask artifact contract

V1 accepts two representations of the same compact bitset:
- `packed_bitset_lsb0_x_fastest_v1`: raw bytes, 1 bit per voxel;
- `hex_bitset_lsb0_x_fastest_v1`: ASCII lowercase/uppercase hexadecimal encoding of those raw bytes, with no whitespace or newline.

Within each byte, bit 0 is the first x-fastest voxel, then bit 1, and so on. A set bit means the voxel belongs to the preregistered eligibility mask. The expected decoded byte count is `ceil(extent_x * extent_y * extent_z / 8)`. The auditor streams the file, verifies its exact artifact-byte SHA-256 and byte length, rejects malformed hex, short or trailing bytes, and rejects non-zero padding bits beyond the last declared voxel.

A PSU marked `eligible` must contain at least one set mask bit; a PSU marked `excluded` must contain zero. The report includes foreground voxel count/fraction and per-PSU foreground counts in addition to geometric tile volumes. This prevents a frame from declaring a tile eligible when the exact bound mask contains no eligible voxels, or excluding a tile that does.

The checked-in fixture's hex file is only a tiny synthetic demonstration. For real volumes, prefer the raw packed bitset to avoid hex expansion. The source mask file is read in chunks; the implementation does not load the full packed mask into memory.

## 4. What the verifier proves—and does not prove

Given the supplied files, a PASS proves:
- every lattice coordinate in the declared integer box appears exactly once;
- each clipped tile is in bounds and the eligible/excluded geometric partition sums to the box volume;
- the exact mask artifact bytes match the manifest digest and encoding contract;
- the decoded bit count matches the declared voxel extent with no non-zero padding or trailing bytes;
- every eligible PSU contains at least one foreground mask voxel and every excluded PSU contains none;
- the canonical frame and eligible population digests are deterministic under cell-row shuffling.

It does **not** prove:
- the declared bounding box contains all tissue or all biologically eligible tissue;
- the mask selection rule is scientifically justified or correctly generated from the named source volume;
- the source volume, segmentation, or transform digests correspond to real upstream artifacts (this tool does not fetch them);
- experts can exhaustively search a PSU or that spatial PSUs are independent;
- a regular lattice is appropriate for neuron-based or irregular PSUs.

Those claims require independent source-artifact intake, domain review, biological mask validation, and potentially a separate sampling design.

## 5. CLI and downstream integration

Run the synthetic fixture:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-spatial-frame-audit -- \
  --frame docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME.example.json \
  --eligibility-mask docs/neuroscience/examples/CONNECTOME_SPATIAL_FRAME_ELIGIBILITY_MASK.hex \
  --output spatial-frame-report.json
```

Then use the report's `population_frame` verbatim as #9204's population frame. Do not hand-maintain a second PSU list. #9204 binds the exact report bytes, raw frame manifest bytes, canonical full-lattice digest, and eligible population digest.

A geometry-only internal stage is labelled `FRAME_GEOMETRY_PASS_MASK_UNVERIFIED`; it is not certifying. Only the CLI path that verifies the separately supplied mask artifact can emit `PASS`.

## 6. Adversarial controls

The Rust tests cover valid partial boundary tiles, exact voxel partition totals, missing/duplicate/out-of-bounds cells, canonical PSU IDs, stratum/exclusion semantics, malformed provenance digests, invalid voxel sizes, row-order-independent hashes, arithmetic overflow, negative origins, LSB-first voxel order, mask hash and byte-length mismatch, non-zero padding bits, trailing bytes, and eligible/excluded PSU-mask disagreement.

## 7. Promotion gate

No candidate-generation recall estimate is promoted unless:
1. the spatial frame emits PASS with an exact verified mask file;
2. source volume, segmentation, transform and selection-rule artifacts are independently inspected;
3. the complete declared search frame and mask domain are justified;
4. #9204 consumes the exact eligible population exported by this report;
5. the sampling protocol and seed commitment were preregistered independently.

The population claim remains limited to the exact declared frame and eligibility rule.
