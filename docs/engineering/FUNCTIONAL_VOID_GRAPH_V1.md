# Functional Void Graph v1

`symthaea-passive-void-graph` moves the design representation one level above raw CSG.

## Model

A functional void graph contains:

- regions with explicit semantic roles (`Inlet`, `Outlet`, `Junction`, `Cavity`, `Reservoir`, `Barrier`, `Sink`, `Unknown`);
- ports attached to regions;
- typed relations such as `FlowPath`, `PressureCoupling`, `ThermalPath`, `AcousticCoupling`, and `ElectromagneticCoupling`;
- directional or bidirectional connection intent;
- a deterministic graph digest.

## Intent-only rule

`declares_path()` answers only whether the submitted *graph* contains a path.
It does not infer that the corresponding CSG geometry is connected, open, sealed, accessible,
or physically capable of carrying the stated quantity.

Physical connectivity must be demonstrated later by geometry analysis, meshing, solver
results, or measurement.

## Why this abstraction

Research on metamaterials increasingly treats network topology and geometry together as
design variables. Recent work describes network theory enriched with geometry and physics
as a natural framework for irregular metamaterial design, while graph-space generative
systems use topology-aware representations to target physical response. citeturn489996search7turn489996search1

That maps cleanly onto Symthaea:

`CSG = constructive geometry`
`VoidGraph = functional structural intent`
`Physics = truth boundary`
`Mycelix = durable provenance`

## Search usage

A future candidate generator can mutate the void graph first—add a port, split a region,
insert a junction, reverse a path, or change a relation—then compile that intent into CSG
and ask geometry/physics whether the proposed function survives.

This is materially more expressive than mutating arbitrary triangles because the search
operators correspond to engineering concepts.