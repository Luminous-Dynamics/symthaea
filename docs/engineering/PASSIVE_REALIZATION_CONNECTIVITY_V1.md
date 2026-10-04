# Passive Realization Connectivity v1

`symthaea-passive-connectivity` is the first bridge from semantic void intent to
realized mesh-topology evidence.

## Evidence classes

`InvalidGraph` means the functional intent graph itself is malformed.

`InvalidMesh` means the candidate mesh failed baseline validation or watertightness.

`UndeclaredPath` means the requested port pair was never asserted as a flow path in the
functional graph.

`AnchorNotRepresented` means a declared port could not be associated with the realized
mesh.

`AmbiguousAnchor` means the nearest realized topology maps to multiple components and the
system refuses to choose one arbitrarily.

`Disconnected` means the two declared ports resolve onto different connected components.

`Connected` means both ports resolve onto the same mesh-topology component.

## Critical epistemic rule

Even `Connected` does not mean that fluid, heat, sound, light, or electromagnetic energy
can successfully traverse the candidate.

The field `physical_transport_unproven` remains true for every observation from this
layer.

Actual transport belongs to a solver or measurement layer.

## Why this is useful

This creates a cheap pre-physics rejection stage:

`semantic graph -> spatial embedding -> candidate CSG -> mesh validation -> port topology`

Candidates that fail obviously at this stage never need expensive CFD/FEA.

That matches the broader direction of current inverse-design research, where topology and
physics are increasingly coupled while expensive physical evaluation remains a distinct
stage. Graph-based metamaterial work also shows that connectivity itself can be a useful
design variable and explanatory signal. citeturn134657search0turn134657search2turn134657search6

## Research boundary

This is not a fluid solver, an accessibility solver, or a manufacturability certificate.
It is deliberately a low-cost geometric/topological evidence gate before those stages.