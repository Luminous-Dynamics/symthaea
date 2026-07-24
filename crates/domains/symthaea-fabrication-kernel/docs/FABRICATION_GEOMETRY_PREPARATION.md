# Fabrication Geometry Preparation

Version 0.8 adds bounded geometric and process evidence before slicing.

## Geometry resolution

`resolve_to_mesh_with_policy` accepts a `TessellationPolicy` expressed in
millimetres. Curved primitives derive facet counts from world-space scale and a
maximum chord-error target. Both minimum and maximum segment counts are
mandatory bounds. The sphere topology uses unique poles and does not rely on
degenerate pole triangles.

The default `resolve_to_mesh` path uses `TessellationPolicy::default`.
Applications requiring evidence reproducibility should persist the complete
policy beside the generated mesh.

## Closed-solid authority

`FabricationReadyMesh` now requires a complete bounded self-intersection scan in
addition to finite data, valid indices, non-degenerate triangles, manifold edge
incidence, consistent normals, watertightness, and positive volume.

The broad phase is sweep-and-prune over triangle AABBs. The narrow phase handles
non-coplanar crossings and coplanar interior overlap. Full shared-edge neighbors
and contacts confined to a shared vertex are not reported. Exhausting the pair
budget fails closed rather than granting geometry authority.

This is stronger evidence, not a proof-grade exact-arithmetic CAD predicate.
Near-coincident surfaces and adversarial floating-point conditioning still need
independent qualification for safety-critical workflows.

## Process authority

`ProcessPreparedMesh` narrows `FabricationReadyMesh` by checking:

- no vertex lies below the configured build plate beyond tolerance;
- connected-component policy;
- local downward-face overhang analysis;
- whether sacrificial supports are permitted; and
- whether the bounded support plan completed without truncation.

The baseline planner emits vertical sacrificial columns under quantized local
overhang cells. It reports overhang triangle IDs and unsupported surface area.
It does not yet optimize tree supports, interface layers, removal access,
thermal distortion, trapped support, or material-specific process behavior.

## Import and export limits

STEP and STL parsing expose explicit resource-limit profiles. Default entry
points use conservative defaults, while `*_with_limits` functions allow callers
to apply tenant- or service-specific ceilings before allocation.

`export_3mf_package` produces a complete core stored ZIP/OPC package containing
`[Content_Types].xml`, `_rels/.rels`, and `3D/3dmodel.model`. It does not yet
include materials, production extensions, thumbnails, signatures, or slicer
metadata.

## Qualification handoff

A `ProcessPreparedMesh` can be promoted to `ManufacturingReadyMesh` only after bounded minimum opposing-surface screening completes without thin or unresolved rays. The retained process and feature policies become part of downstream provenance.
