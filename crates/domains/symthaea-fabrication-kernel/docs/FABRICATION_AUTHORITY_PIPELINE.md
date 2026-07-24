# Fabrication Authority Pipeline

Version 0.8 extends the capability boundaries between descriptive data and
artifacts that may progress toward machine execution.

## Stages

1. **Raw geometry** — `TriangleMesh` may contain malformed, open, duplicated,
   non-manifold, or non-finite data.
2. **Closed-solid authority** — `FabricationReadyMesh::try_new` records a full
   `ValidationReport` and grants a wrapper only after topology, finite-data,
   orientation, positive-volume, and bounded self-intersection gates pass.
3. **Process preparation** — `ProcessPreparedMesh` checks build-plate placement,
   connected-component policy, local overhangs, support permission, and support
   planner completion.
4. **Strict slicing** — `slice_fabrication_ready` and the `try_slice_*` APIs
   reject invalid dimensions, tolerances, infill settings, heights, vertices,
   and indices rather than clamping them silently.
5. **Strict toolpath generation** — `try_generate_gcode` rejects empty jobs,
   invalid material-flow parameters, and non-finite layer coordinates.
6. **Machine authority** — `ValidatedGCode::try_new` binds a program to a named
   `MachineProfile` after checking build bounds, homing order, feed rates,
   temperatures, finite values, maximum retraction, and the presence of positive
   extrusion motion.
7. **Submission** — `submit_validated_gcode` accepts only `ValidatedGCode`.
   Live network transports remain unavailable and fail closed; the operational
   backend is still `MockPrinter`.

## Unit contract

Mesh, slicing, and machine coordinates are millimetres. Analytical mechanics is
SI: metres, newtons, and pascals. The `units` module makes conversions explicit,
and generative fitness converts mesh extents from millimetres before invoking
the analytical backend.

## What this does not prove

These stages do not establish exact-arithmetic geometric predicates, minimum
wall or hole size, process-specific thermal suitability, optimized or removable
support structures, collision-free machine kinematics, firmware dialect
compatibility, material certification, or safe operation of real hardware.
Those remain separate evidence campaigns.

## Version 0.9 qualification boundary

`ManufacturingReadyMesh` now sits after `ProcessPreparedMesh` and before trusted slicing. A complete job may be bound into a `FabricationManifest` and packaged with its 3MF model. These fingerprints are deterministic but non-cryptographic and do not replace live-operation authorization.

## v0.10 secure release extension

After machine validation, trusted release may continue through a SHA-256-bound
manifest attestation, authenticated capability negotiation, and a single-use
session-bound submission authority. See `FABRICATION_SECURE_RELEASE.md`.
