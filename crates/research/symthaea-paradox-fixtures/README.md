# symthaea-paradox-fixtures

PARADOX-002A deterministic fixture and manipulation-oracle qualification layer.

This crate generates matched C0-C6 structured-inconsistency fixtures under the frozen PARADOX-002 contract and independently verifies their manipulation properties. It is stacked on exact qualified PARADOX-001 subject `3f8e53e4d6b89895dc5097b2dccfe4f3868d53f9`.

The fixture generator does not execute Symthaea cognition, metacognition, action selection, learning, workspace logic, Phi logic, or self-model mutation. The PARADOX-001 observatory is consumed only by the independent qualification oracle after a fixture already exists.

A later PARADOX-002B cognitive adapter is permitted to consume only `AgentView`, which intentionally excludes the condition label, expected response, and hidden oracle truth.

Confirmatory fixture qualification uses the frozen 16 seeds and 64 trials per condition from issue #3168. Development seeds are disjoint.

A PARADOX-002A PASS establishes only that the experimental manipulations are deterministic, typed, resource-auditable, and satisfy their frozen condition contract. It is not a consciousness, metacognition, intelligence, or ontology-repair result.

See issues #3168 and #3247.
