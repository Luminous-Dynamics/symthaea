# NEUROARCH-001 — Experimental topology contract

Status: design freeze for issue #3317  
Parent program: #3316

## Purpose

NEUROARCH introduces a mesoscale description of Symthaea's HDC/LTC cognitive topology without changing the production evolution path. The topology layer answers **what circuits and declared routes exist**. It does not, by itself, claim what causal influence those routes have.

The current `HdcLtcUnifiedNetwork` remains the incumbent. NEUROARCH-001 must be additive and observational around it.

## Separation of concerns

The research program keeps four concepts distinct:

1. **Connectome** — declared structural circuits and routes.
2. **Effectome** — causal influence measured by controlled perturbation.
3. **Epistemome** — confidence/provenance for causal and structural claims.
4. **Runtime state** — changing neural/HDC state and activity metrics.

Only (1) is introduced by NEUROARCH-001. Runtime state must not affect topology identity.

## Canonical identifiers

Implementations should expose stable identifiers equivalent to:

```rust
pub struct CircuitId(pub u32);
pub struct EdgeId(pub u64);
```

Identifiers must be deterministic for a fixed topology description. Hash-map iteration order, thread scheduling, construction order, and runtime activity must not alter canonical identity.

## Circuit descriptor

A circuit descriptor must contain only static architectural metadata. Minimum semantics:

```text
CircuitDescriptor
  id
  role              // neutral symbolic label; no biological equivalence claim
  timescale_class   // e.g. Fast, Medium, Slow, Custom
  state_dimension
  implementation    // incumbent HDC-LTC, CfC, future island engine, etc.
  modulation_profile (optional declarative capabilities only)
```

The descriptor must not contain task answers, learned runtime state, held-out labels, or mutable activity counters.

## Edge descriptor

Minimum semantics:

```text
EdgeDescriptor
  id
  source
  target
  channel
  direction
  recurrence
  budget_class
  merge_policy_hint (optional)
```

A structural edge means only that influence is permitted by the architecture. It is not evidence that the source actually causes a downstream effect.

## Validation rules

Topology construction must fail closed on:

- dangling source or target circuit IDs;
- duplicate canonical circuit IDs;
- duplicate canonical edge IDs;
- duplicate structural edges where the topology kind does not explicitly permit parallel channels;
- invalid state dimensions;
- malformed timescale metadata;
- forbidden self-edges unless the topology explicitly declares recurrent self-routing;
- non-canonical or ambiguous identifiers.

Parallel edges are allowed only when they are distinguishable by a declared semantic channel and the topology kind permits them.

## Canonicalization

A topology commitment must be independent of construction order.

Canonical form should sort circuits by stable circuit identity and edges by a stable tuple such as:

```text
(source, target, channel, edge_id)
```

Canonical bytes must bind all behaviorally relevant static fields. Cosmetic descriptions may be excluded only if documented.

Changing any of the following must change the topology commitment:

- circuit count or identity;
- circuit role/timescale/state dimension;
- edge source/target/channel;
- recurrence semantics;
- budget class or other execution-relevant route semantics.

Changing runtime state, counters, timestamps, or observations must **not** change it.

## Incumbent adapter

The current layered `HdcLtcUnifiedNetwork` is the required baseline adapter.

The adapter is descriptive: it maps the incumbent's existing layer/neuron structure and declared inter-layer connectivity into the topology interface while leaving current stepping untouched.

It must preserve existing behavior including:

- fixed-interval stepping;
- irregular timestamp stepping and timing validation;
- current layer-binding semantics;
- current skip-connection semantics;
- bundled layer outputs;
- reset behavior;
- existing public constructors and call sites.

The adapter must not require a reimplementation of `evolve_closed_form`, layer aggregation, or timestamp handling.

## Replay parity gate

NEUROARCH-001 is not complete until an incumbent network can be observed through the topology adapter while producing the same runtime output as the unadapted incumbent under the same seed/config/input/timestamp stream within the existing numerical tolerance.

The topology inspection path must not mutate network state.

## Initial test matrix

Required tests:

1. deterministic canonical commitment for identical topology built in different insertion orders;
2. mutation tests showing behaviorally relevant static changes alter the commitment;
3. dangling-edge rejection;
4. duplicate-ID rejection;
5. forbidden self-edge rejection;
6. explicit parallel-channel acceptance/rejection according to topology policy;
7. topology inspection leaves live network state unchanged;
8. incumbent fixed-step replay parity;
9. incumbent irregular-time replay parity;
10. runtime activity changes do not alter static topology commitment.

## Evidence boundary

Passing NEUROARCH-001 establishes only that Symthaea can describe and commit to an experimental cognitive topology reproducibly.

It does **not** establish that:

- a topology is biologically realistic;
- a declared edge is causally important;
- modular/rich-club organization is beneficial;
- a topology improves cognition;
- any architecture has consciousness-related significance.

Those questions belong to later preregistered tranches.

## Follow-on dependency

NEUROARCH-002 (#3318) may build deterministic benchmark receipts on the canonical topology commitment. Candidate architecture code should not bypass this contract.
