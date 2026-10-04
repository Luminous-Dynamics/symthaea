# Passive Port Interface v1

A `PortInterface` is the typed geometric boundary identity used by the passive
device realization pipeline.

It binds, as one immutable value:

- port identifier
- port position
- aperture geometry
- outward normal
- interface plane
- solver-boundary identity
- deterministic interface digest

The important invariant is that these fields cannot silently diverge between
material-device compilation and boundary validation.

## Geometry contract

The current aperture is `Circular { radius_mm }`.

The interface validates that:

- position coordinates are finite
- aperture radius is finite and positive
- outward and plane normals are nonzero and normalized
- outward and plane normals agree in orientation
- the port position lies on the declared interface plane

Boundary realization additionally requires observed open-boundary edges to:

1. lie on the declared interface plane within tolerance
2. have endpoints near the declared aperture circumference
3. belong to the declared port interface without ambiguity

This closes the earlier failure mode where an arbitrarily small opening could
be accepted merely because its boundary edges fell inside a large port radius.

## Solver identity

`SolverBoundaryIdentity` is deliberately solver-neutral:

`(domain, id)`

The identity can be mapped by a concrete solver adapter to an inlet pressure,
velocity, temperature, acoustic, electromagnetic, or other boundary condition.

The geometry layer does not claim that the solver consumed that identity.
`physical_transport_unproven` remains true until solver evidence exists.

## Pipeline

The intended progression is:

`functional graph`
→ `geometry embedding`
→ `typed port interface`
→ `void realization`
→ `material device`
→ `boundary realization`
→ `solver binding`
→ `physics evidence`
→ `measurement`

The interface artifact is therefore a binding contract between geometry and the
future solver adapter, not a transport certificate.

## Compatibility

Legacy `PortAnchor` remains available for conservative graph and geometry
workflows. Legacy boundary matching is retained for compatibility.

New ported-device workflows should use `GeometryEmbedding::with_port_interface`
and `ExternalPortSpec::new(PortInterface)` so aperture, plane, normal, and solver
identity remain coupled.
