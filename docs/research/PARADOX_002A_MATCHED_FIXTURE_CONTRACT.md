# PARADOX-002A — Matched Fixture Qualification Contract

Status: preregistered implementation companion to issues #3168 and #3247.

Authority: **MeasurementOnly / fixture qualification**.

## Qualified parent

PARADOX-002A is stacked on exact qualified PARADOX-001 subject:

`3f8e53e4d6b89895dc5097b2dccfe4f3868d53f9`

The PARADOX-001 crate must remain byte-identical to that qualified subject throughout 002A qualification.

## Scope

002A qualifies deterministic synthetic fixtures and an independent manipulation oracle for the seven frozen condition families:

- C0 coherent control
- C1 surprise only
- C2 transient conflict
- C3 persistent resolvable conflict
- C4 persistent irreducible conflict
- C5 self-referential conflict
- C6 ontology failure

It does not execute or score production cognition.

## Non-circularity rule

Fixture construction and fixture qualification are separate operations.

The generator may not consume:

- PARADOX-001 observatory output;
- the later cognitive adapter's response;
- metacognitive telemetry;
- workspace/Phi/self-model state;
- wall-clock or network state.

The independent oracle may inspect a completed fixture and may then call the qualified PARADOX-001 observatory for declared measurement fields.

## Agent-view firewall

A later 002B adapter may consume only `AgentView`.

`AgentView` contains the prior expectation, four observation events, query cue, and frozen resource budget. It does not expose:

- `Condition`;
- `ExpectedResponse`;
- private `OracleTruth`;
- oracle-derived manipulation labels;
- hidden target context.

This boundary prevents experimental metadata from becoming a shortcut solution.

## Frozen resources

Every C0-C6 fixture carries the same declared envelope:

- 4 observation slots;
- 3 proposition-bearing claim events;
- 1 auxiliary/context slot;
- 2 candidate hypotheses;
- 4 update steps;
- nominal search budget 8;
- 4 timing slots;
- 2 source identities.

Condition differences must arise from preregistered relational structure, not hidden extra evidence or extra compute.

## Evidence provenance

Every event retains source identity and fault-domain identity separately. Opposing evidence counts as independently conflicting only when at least one proposition-supporting event and one negation-supporting event come from different fault domains.

Duplicated provenance must not masquerade as independent disagreement.

## Confirmatory constants

Confirmatory seeds:

`11, 29, 47, 71, 101, 149, 197, 257, 331, 419, 521, 631, 751, 887, 1021, 1171`

Stateless fixture qualification: 64 trials per condition per seed.

Development seeds are disjoint from the confirmatory set. There is no adaptive stopping.

## Determinism

For fixed `(condition, seed, trial_index)`, both fixture canonical bytes and oracle-report canonical bytes must be exactly reproducible.

Canonicalization uses explicit field ordering and fixed-width little-endian encodings. No hash-map ordering, wall-clock time, OS randomness, or network data may enter fixture generation.

## Claim boundary

A successful 002A qualification establishes only that the synthetic experimental manipulations are deterministic, resource-auditable, provenance-aware, and satisfy the frozen C0-C6 contract.

It does not establish that paradox is generative, that metacognition is recruited, that Symthaea repairs ontologies, or that Symthaea is conscious. Those require later separately qualified tranches.
