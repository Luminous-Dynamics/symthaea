# Functional Void Graph Search Operators — Draft

Candidate mutations should operate on semantic structure before geometry:

`AddPort(region)` — expose a new interface.

`AddJunction(region_a, region_b)` — propose a new coupling point.

`SplitRegion(region)` — increase internal topology complexity.

`ReverseConnection(edge)` — explore directional behavior.

`ToggleBidirectionality(edge)` — distinguish one-way and reciprocal hypotheses.

`ChangeRelation(edge, relation)` — move between flow, thermal, acoustic, pressure, and EM hypotheses.

`ReplaceRegionRole(region, role)` — explore whether a region acts as inlet, cavity, reservoir, etc.

Every mutation should be scored only after downstream compilation/evaluation.

## Hard rule

A semantic mutation is not allowed to claim physical validity merely because the graph
remains internally valid. Geometry, manufacturing, and physics remain separate evidence
stages.

## Search architecture

`intent HV + void graph + topology fingerprint` should be treated as the candidate identity.

This makes negative results especially useful: a failed graph/geometry pair can be
remembered as a particular structural hypothesis rather than as a generic 'bad design'.

Current physics-guided inverse-design research supports this general architecture: graph
or morphology priors provide structure, while differentiable or exact physics guides
candidate generation and selection. citeturn489996search0turn489996search7