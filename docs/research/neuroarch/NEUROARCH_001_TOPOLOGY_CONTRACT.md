# NEUROARCH-001 — Experimental topology contract

Status: executable draft; qualification pending for issue #3317  
Parent program: #3316

## Purpose

NEUROARCH-001 introduces a mesoscale description of Symthaea's HDC/LTC cognitive topology without changing the production evolution path. The topology layer answers **what circuits and declared routes exist and which static execution semantics they carry**. It does not, by itself, establish causal influence, biological realism, or architectural benefit.

The current `HdcLtcUnifiedNetwork` remains the incumbent. The NEUROARCH-001 adapter is additive, observational, and descriptive around it.

## Separation of concerns

Four identities remain distinct:

1. **Topology / connectome** — declared structural circuits, routes, and static execution semantics.
2. **Subject/model identity** — seed, realized parameters, complete model/config identity.
3. **Effectome** — causal influence measured by controlled perturbation.
4. **Epistemome/runtime evidence** — confidence, provenance, observations, mutable activity and state.

A topology commitment is not a substitute for the other identities.

## Canonical identifiers

V1 exposes stable structural identifiers equivalent to:

```rust
pub struct CircuitId(pub u32);
pub struct EdgeId(pub u64);
```

Construction order, map iteration order, thread scheduling, runtime activity, and random initialization seed must not change structural identity.

## Circuit descriptor

Minimum static semantics:

```text
CircuitDescriptor
  id
  role                 // descriptive symbolic label only
  timescale_class
  state_dimension
  unit_count
  implementation
  input_merge_policy   // None | Single | BundleAll
  modulation_profile?  // declarative capability/profile only
```

`role` is not verified function. Runtime systems must not treat a label such as `causal`, `self`, or `planning` as evidence that a circuit actually performs that function.

## Edge descriptor

Minimum static semantics:

```text
EdgeDescriptor
  id
  source
  target
  channel
  direction
  recurrence
  budget_class
  transform            // route-local Direct | Bind in V1
```

A structural edge means only that influence is permitted by the architecture. It is not evidence that the source actually causes a downstream effect.

### Route transform vs target merge

V1 deliberately separates route-local transformation from target-level multi-input merging.

For the incumbent:

- inter-layer routes are `Direct` or `Bind`;
- skip routes are explicit `Direct` routes from external input to the deeper layer;
- a deeper target with skip connectivity declares `input_merge_policy = BundleAll`.

This prevents skip semantics from being double encoded as both an edge transform and an explicit skip route, and gives later resource accounting one unambiguous graph.

## Symbolic-token grammar

Symbolic fields participating in V1 topology identity are case-sensitive bounded ASCII tokens.

Allowed characters:

```text
A-Z a-z 0-9 _ - . : /
```

Maximum length: 128 UTF-8 bytes.

Whitespace, Unicode lookalikes, empty names, and over-length symbolic fields fail closed. V1 does not silently trim, case-fold, or Unicode-normalize identity-bearing labels. A future grammar or normalization change requires an explicit topology schema/version change.

## Validation rules

Topology construction fails closed on at least:

- unsupported topology schema version;
- no computational circuits;
- duplicate circuit IDs;
- duplicate edge IDs;
- dangling source or target IDs;
- invalid state dimensions or zero implementation-unit counts;
- invalid/malformed timescale declarations;
- forbidden non-recurrent self-edges;
- ambiguous duplicate structural routes;
- duplicate source/target/channel routes;
- malformed symbolic identifiers.

Parallel channels are allowed only when the topology explicitly permits them and the semantic channels remain distinct.

## Bidirectional canonicalization

A bidirectional route has no meaningful source/target orientation for topology identity. Therefore V1 canonicalizes its endpoints before duplicate detection, sorting, and encoding:

```text
A <-> B
```

and

```text
B <-> A
```

produce identical canonical bytes when all other static identity fields, including `EdgeId`, are identical.

Directed routes remain orientation-sensitive.

## Canonical commitment

Canonical form sorts circuits by stable circuit identity and edges by canonical direction/endpoints/channel/edge identity. Canonical bytes bind all declared behaviorally relevant static topology fields and contain no runtime state.

V1 uses a domain-separated BLAKE3 commitment over those canonical bytes.

Changing structural properties such as the following must change the commitment:

- circuit count or identity;
- layer size represented through unit/state dimensions;
- role/timescale/implementation/merge-policy declarations;
- route source/target/channel/direction/recurrence;
- route budget class;
- route-local transform;
- skip-connectivity structure.

Changing runtime state, current observations, counters, timestamps, cached outputs, learned weights, realized random binding vectors, or initialization seed must **not** change the structural topology commitment.

## Topology identity vs subject/model identity

The topology commitment answers **which static architecture is being described**. It is intentionally narrower than the complete experimental subject.

For the incumbent, topology identity binds structural properties such as:

- layer count and layer sizes;
- represented HDC state dimensions;
- declared inter-layer and skip routes;
- binding enabled/disabled route semantics;
- target merge semantics;
- route budget semantics;
- declared timescale metadata.

It deliberately does **not** bind:

- random initialization seed;
- realized random weight/binding-vector bytes;
- learned/adapted parameter values;
- activation/optimizer or other non-topological model parameters unless separately elevated into a future structural schema;
- current neuron/HDC state;
- cached outputs, timestamps, counters, or observations.

Those values belong to benchmark subject/model/config/runtime receipts. Two subjects may therefore share one topology commitment while having different exact subject identities. Comparative evidence must bind both.

## Incumbent adapter

The layered `HdcLtcUnifiedNetwork` is the required baseline adapter.

The adapter must leave untouched:

- fixed-interval stepping;
- irregular timestamp stepping and timing validation;
- current layer-binding computation;
- current skip-connection computation;
- bundled layer outputs;
- reset behavior;
- existing public constructors and call sites;
- neuron `evolve_closed_form` dynamics.

Topology inspection is observational and may not mutate live network state.

## Replay parity gate

NEUROARCH-001 is not qualified until an incumbent network observed through the topology interface produces the same runtime output as an otherwise identical unobserved incumbent under the same seed, configuration, input stream, and timestamp stream within the existing numerical tolerance.

Both fixed-step and irregular-time paths are protected.

## Test matrix

Required evidence includes:

1. insertion-order-independent canonical bytes/commitment;
2. behaviorally relevant static mutation changes commitment;
3. dangling-edge rejection;
4. duplicate-ID rejection;
5. forbidden self-edge rejection;
6. explicit parallel-channel policy;
7. bidirectional orientation canonicalization and reversed-duplicate rejection;
8. bounded canonical symbolic-token enforcement;
9. topology inspection leaves runtime behavior/state unchanged;
10. incumbent fixed-step replay parity;
11. incumbent irregular-time replay parity;
12. runtime activity does not change topology commitment;
13. changing incumbent seed alone does not change topology commitment;
14. changing layer size, binding semantics, skip connectivity, or target merge semantics changes topology commitment;
15. skip connectivity is represented once: route-local transforms plus explicit skip route plus target merge policy.

## Evidence boundary

Passing NEUROARCH-001 would establish only that Symthaea can describe, validate, and commit to an experimental cognitive topology reproducibly while preserving incumbent behavior.

It would **not** establish that:

- a topology is biologically realistic;
- a declared edge is causally important;
- modular/rich-club organization is beneficial;
- a topology improves cognition or efficiency;
- the architecture has consciousness-related significance.

Those questions belong to later preregistered experiments.

## Follow-on dependency

NEUROARCH-002 (#3318) may build deterministic benchmark receipts on the canonical topology commitment. Later candidates must bind both topology identity and exact subject/config identity and may not bypass this contract.
