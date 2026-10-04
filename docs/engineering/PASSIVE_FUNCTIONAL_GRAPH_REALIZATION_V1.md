# Passive Functional-Graph Realization v1

The graph-realization report aggregates port-level topology evidence across every
declared FlowPath in a functional void graph.

## What it establishes

For each declared flow connection, the report records the existing
port-path observation:

- Connected: both port anchors resolve to the same realized mesh component.
- Disconnected: both anchors resolve, but to different components.
- AnchorNotRepresented: at least one declared anchor is not represented by the
  candidate mesh.
- AmbiguousAnchor: an anchor maps to multiple topology components and the system
  refuses to guess.
- InvalidMesh / InvalidGraph: the candidate or intent graph is not admissible
  to this observation layer.

The aggregate status is:

- AllDeclaredPathsConnected
- PartialRealization
- NoDeclaredPathsConnected
- NoFlowPathsDeclared
- InvalidGraph

## Epistemic boundary

The aggregate is a geometric realization report, not a transport certificate.

Even when every declared connection is connected in the mesh, this does not establish:

- pressure-driven flow,
- low hydraulic resistance,
- correct flow direction,
- absence of stagnation or recirculation,
- heat transport,
- acoustic transmission,
- electromagnetic coupling,
- or fabricated-device performance.

physical_transport_unproven therefore remains true.

## Why aggregate the graph?

A single port-pair result can miss systemic topology failure.

The aggregate turns a candidate from a collection of local observations into a
machine-readable realization record:

functional graph -> candidate geometry -> mesh -> per-edge realization -> aggregate status

That gives the search layer a deterministic, cheap gate before expensive solver or
measurement stages.

A partial realization is especially valuable evidence. For example, an intended
three-edge flow network can become a two-edge realized network even though both
individual endpoints exist. The report preserves that failure rather than collapsing
it into a generic "geometry invalid" result.

## Current scope

The current connectivity profile expects a valid watertight mesh. That is appropriate
for closed-volume topology checks but is intentionally conservative for real devices
whose fluid domains have external openings.

Open inlet/outlet semantics should be handled by an explicit future boundary-policy
layer rather than weakening watertight validation globally. This prevents accidental
acceptance of arbitrary open meshes while still leaving a principled path for genuine
ported flow devices.

## Relationship to inverse design

Current inverse-design research increasingly uses graph/topology as an explicit design
representation while retaining physics as a separate validation layer. This report
implements the corresponding realization side: intent topology is compared with
realized geometry before expensive physical evaluation.
