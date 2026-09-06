# Symthaea RCA Canonical-Lineage-Bound Disposition Preflight

RCA-003b.3c closes the remaining evidence-lineage generation join **after** the exact cross-artifact preflight and **before** any shadow disposition engine.

## Why a wrapper exists

The RCA-003b.3b raw preflight preserves the structural case's existing `lineage_graph_id` as a case-local/audit reference. #578 subsequently established a stronger generic governance identity:

```text
legacy/local graph reference
        !=
canonical complete lineage generation
```

Rather than silently redefining the already-reviewed raw preflight identity, this tranche wraps it.

```text
ShadowDispositionPreflightV1
        +
exact bound case
        +
exact same evidence-witness slots
        ↓
LineageBoundShadowDispositionPreflightV1
```

A future disposition engine must accept the lineage-bound wrapper, not raw preflight.

## Exact raw-preflight re-binding

Before doing any lineage work, the wrapper requires the raw preflight to match the supplied bound case on:

- case id;
- proposition id;
- case-scope digest;
- the historical/local case lineage reference.

It also requires every supplied support/opposition/defeater evidence witness ID to equal the exact optional witness ID already stored in the raw preflight.

Therefore this layer cannot take a valid preflight and substitute another witness while asking only for a generation check.

## Canonical runtime case lineage

RCA V1 runtime evidence has explicit lineage semantics:

```text
one frozen observation event
        = RootObservation

one selected field candidate
        = Transformation child of that observation root
```

The wrapper reconstructs exactly that graph from the bound case's immutable item facts:

```text
candidate_id
observation_root_id
```

Shared observation roots are emitted once. Every candidate receives exactly one transformation edge from its observation root.

The existing local `structural_case.lineage_graph_id()` is used only as the required legacy wire label when validating the reconstructed graph. #578 canonical identity explicitly excludes that field.

The validated graph then receives:

```text
canonical_evidence_lineage_graph_id_v1(...)
```

which commits the complete validated graph semantics.

## Evidence-witness generation join

Each supplied `IndependentEvidenceSetWitnessV1` now carries its exact canonical complete lineage generation.

The wrapper requires:

```text
witness.lineage_graph_id()
        ==
canonical case lineage generation
```

for every supplied support/opposition/defeater evidence witness.

Therefore all of these fail closed:

- witness issued from a subset lineage;
- witness issued from a superset lineage;
- witness issued from another generation with the same selected local roots;
- raw preflight paired with a different evidence witness;
- raw preflight paired with another case.

Changing only the legacy/local graph label does **not** change canonical generation.

## Issued capability

`LineageBoundShadowDispositionPreflightV1` retains:

- the exact raw `ShadowDispositionPreflightV1`;
- the canonical complete evidence-lineage generation id;
- its own profile/schema;
- a domain-separated BLAKE3 `binding_id` over raw preflight identity + canonical lineage generation.

It has private fields and derives `Serialize` only. Archived bytes cannot restore current evaluation eligibility.

## Non-scope

This layer performs no:

- witness cardinality threshold comparison;
- relation-strength arithmetic;
- independent-set or clique discovery;
- shadow disposition;
- canonical belief admission;
- workspace/GWT mutation;
- external action authorization;
- self-improvement promotion.

```text
LineageBoundShadowDispositionPreflightV1
        !=
ShadowDispositionV1
        !=
canonical epistemic state
        !=
action authority
```

## Engine boundary

The intended next dependency is one-way:

```text
future pure shadow disposition engine
        ↓ accepts only
LineageBoundShadowDispositionPreflightV1
```

The engine should never accept raw `ShadowDispositionPreflightV1` directly.
