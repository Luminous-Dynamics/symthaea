# Passive Topology v1

`symthaea-passive-topology` introduces a structural descriptor for positive and negative
space in the CSG representation.

## Negative-space semantics

A `Subtract` node is recorded as explicit negative-space intent. The descriptor reports:

- total CSG node count;
- primitive count;
- explicit subtraction count;
- nodes contained in subtraction tool subtrees;
- intersection count;
- transform count;
- deepest subtraction depth;
- deterministic whole-tree digest.

The digest normalizes commutative union/intersection operand ordering but preserves
subtraction operand ordering.

## Epistemic boundary

The descriptor does **not** prove:

- that a cavity is enclosed;
- that a channel is connected;
- that fluid can traverse it;
- that a tool can reach it;
- that wall thickness is acceptable;
- that the result is printable;
- or that the removed volume exists after numerical geometry resolution.

Those claims belong to downstream geometry validation, manufacturability analysis, and
physics solvers.

## Why it matters for passive design

Negative space can itself carry function: channels, resonant cavities, pressure paths,
thermal pathways, acoustic volumes, and electromagnetic exclusions. Current topology
optimization literature repeatedly treats internal channels and void distributions as
first-class design variables, while also showing that manufacturing constraints must be
considered during optimization. citeturn932315search2turn932315search5

The descriptor therefore gives the search system a topology identity without pretending
that topology alone establishes function.