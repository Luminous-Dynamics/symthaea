# Passive Device Realization v1

The passive void compiler has two distinct products:

1. functional void geometry, used to reason about the intended internal topology;
2. material device geometry, produced by subtracting that void from an explicit body envelope.

The material realization layer also creates explicit external port tunnels. A tunnel
starts at the declared port anchor, follows a validated outward direction, intersects
the body envelope, and extends beyond it. This gives the CAD candidate a physical opening
without changing the semantic void graph.

## Invariants

- the body envelope must be finite and non-degenerate;
- every external port must refer to a graph port with a valid anchor;
- external anchors must start strictly inside the body;
- external directions must be finite and nonzero;
- duplicate external-port specifications are rejected;
- the result is still candidate geometry, not a fabrication certificate;
- transport remains unproven.

## Why the split matters

Connectivity of the material solid is not fluid connectivity. The functional void and
the fabricated material therefore receive separate identities and should be validated
against different criteria.

This also provides a clean route to solver setup: later CFD/FEA adapters can consume
the void domain plus explicit external port interfaces, while manufacturing adapters
consume the material candidate.