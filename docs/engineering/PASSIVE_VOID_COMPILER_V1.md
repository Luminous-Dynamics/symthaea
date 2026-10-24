# Passive Void Compiler v1

`symthaea-passive-void-compiler` converts an embedded functional void graph into
candidate CSG representing void geometry.

## Required embedding

The semantic graph has IDs and relationships, but not spatial coordinates. Compilation
therefore requires an explicit `GeometryEmbedding` containing:

- region centers and radii in millimetres;
- port centers and radii in millimetres.

Missing anchors are hard compilation errors.

## Compilation

For the first flow-oriented compiler:

- each non-barrier region becomes a spherical void volume;
- each `FlowPath` connection becomes a cylindrical channel between port anchors;
- those volumes are unioned into one candidate void CSG;
- the caller can subtract the resulting void CSG from a material body.

Non-flow relations are intentionally not converted into channels by this compiler.

## Directionality boundary

`bidirectional` and edge direction remain functional intent metadata. They do not change
the generated static geometry. A static hole is not a one-way valve.

Directional behavior must emerge from asymmetric geometry, boundary conditions, fields,
gravity, phase change, or other physical mechanisms and must be demonstrated by a
downstream solver or measurement.

## Truth boundary

The compiler proves only that a valid semantic graph and a valid spatial embedding were
translated into a candidate CSG tree. It does not prove geometric connectivity, wall
thickness, manufacturability, solver convergence, flow performance, or physical function.

## Why this matters

Recent physics-guided inverse-design research is explicitly moving physics guidance into
the generation process instead of treating simulation as a final classifier. Other 2026
work on fixed-geometry fluidic diodes demonstrates multi-objective topology optimization
for no-moving-part rectification. citeturn283135search0turn283135search1

The Symthaea architecture adapts the insight while retaining a symbolic/HDC-compatible
representation:

`functional intent -> embedded topology -> candidate geometry -> physics -> Pareto archive`.