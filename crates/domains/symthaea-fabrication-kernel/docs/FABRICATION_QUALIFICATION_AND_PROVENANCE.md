# Fabrication Qualification and Provenance

Version 0.9 adds two trust boundaries after geometry preparation and before any
machine submission.

## Conservative repair is not geometric healing

`repair_mesh` performs a bounded sanitation pass over untrusted triangle data:

- rejects work above an explicit triangle budget;
- removes triangles with invalid indices, non-finite coordinates, or negligible area;
- welds coincident vertices under an explicit millimetre tolerance;
- removes duplicate geometric triangles;
- restores positive global winding; and
- rebuilds vertex normals from repaired faces.

It does not fill holes, reconstruct missing surfaces, resolve self-intersections,
or produce a fabrication capability. The repaired mesh must cross
`FabricationReadyMesh` and every later gate normally.

## Minimum-feature qualification

`analyze_minimum_features` casts bounded inward rays from triangle centroids and
records the nearest opposing surface. `ManufacturingReadyMesh` is granted only
when:

1. closed-solid validation passes;
2. process placement and support planning pass;
3. the minimum opposing-surface policy passes;
4. every source ray resolves; and
5. the configured narrow-phase budget is not exhausted.

This is a conservative screening oracle. It is not an exact medial-axis,
minimum-wall, nozzle-flow, or material certification proof.

## Deterministic provenance

`FabricationManifest` binds stable, domain-separated fingerprints for:

- canonical oriented mesh geometry;
- process policy and support-plan evidence;
- minimum-feature policy and measured evidence;
- slicer configuration and exact ordered slice layers;
- toolpath configuration;
- the exact machine profile retained by `ValidatedGCode`;
- final machine-validated G-code; and
- aggregate layer, command, and extrusion counts.

Mesh identity is independent of vertex-table order, triangle order, and cyclic
triangle rotation. Winding remains part of identity. Fingerprints are
non-cryptographic reproducibility identifiers: they detect accidental mismatch
but do not prove authorship or resist malicious substitution.

Manifest construction rejects a supplied machine profile unless it exactly
matches the profile retained when machine authority was granted, even when two
profiles share the same name.

`export_3mf_package_with_manifest` packages the manifest as
`/Metadata/fabrication-manifest.json` and binds it from the root OPC
relationships document.

## Remaining authority boundary

Version 0.9 still does not authorize physical operation. It does not establish
material/process certification, exact thin-wall proof, support removability,
thermal distortion, firmware dialect semantics, collision-free kinematics,
machine calibration, sensor health, or operator authorization. Live transports
remain fail-closed.
