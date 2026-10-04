# Passive Port Boundary Policy v1

The baseline mesh validator is a closed-solid gate: watertightness is required for
fabrication readiness. Ported fluid domains are different because an inlet or outlet
can be an intentional boundary.

symthaea-passive-connectivity therefore provides an explicit PortBoundaryPolicy
rather than weakening the global closed-solid rule.

## Policies

closed() accepts only a mesh with zero boundary edges.

with_allowed_open_port(port) permits boundary edges only when every open edge can be
associated with that port anchor, using the anchor radius plus an explicit tolerance.

The evaluator reports:

- Closed
- ExpectedOpeningsOnly
- UnexpectedOpenings
- MissingPortAnchor
- AmbiguousOpening
- InvalidMesh

Only Closed and ExpectedOpeningsOnly are admissible.

## Epistemic boundary

This proves only that observed open boundaries are geometrically consistent with an
explicit opening policy.

It does not prove:

- that the opening reaches a fluid volume,
- that the inlet/outlet has the intended normal direction,
- that a solver boundary condition is correctly applied,
- that the aperture has acceptable hydraulic resistance,
- or that the fabricated surface matches the digital candidate.

physical_transport_unproven remains true.

## Why this is a separate layer

Silently changing a global watertightness rule would make ordinary invalid/open CAD
more permissive. Keeping boundary intent explicit lets the system distinguish:

closed solid from ported device from accidentally open geometry.

This is consistent with current topology-optimization work that treats connectivity and
boundary conditions as first-class constraints during design rather than relying on
post-processing repair.