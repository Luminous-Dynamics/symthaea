# RSK Resource Runtime Dimension-ID Binding v0.1

**Status:** Normative semantic-identity supplement; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

A resource-accounting scheme is not fully identified if canonical semantic dimension names are digested but the runtime numeric dimension IDs used by the constitutional arithmetic engine are assigned separately.

The governing invariant is:

```text
ResourceAccountingSchemeId
    commits semantic dimension identity
    AND runtime numeric dimension identity
```

This document contains no physical resource recipe, physical replication mechanism, manufacturing process, or measurement procedure.

## Current reference-profile limitation

The current v0.1 golden corpus includes canonical semantic dimension identifiers such as:

```text
budget.compute
budget.energy
```

The Rust reference tests separately use numeric IDs `0` and `1` for those dimensions.

That numeric mapping is useful reference scaffolding but is not currently encoded in the canonical v0.1 resource-schema JSON that produces the frozen test digest.

Therefore:

> the current v0.1 golden resource digest MUST NOT be interpreted as production evidence that it commits the numeric runtime mapping.

## Production canonical form

A production-target resource schema revision must bind an unambiguous runtime numeric ID for every dimension in the canonical bytes from which `ResourceAccountingSchemeId` is derived.

Conceptually:

```text
ResourceDimensionDefinition {
    semantic_id,
    runtime_numeric_id,
    canonical_unit,
    scale,
    minimum,
    maximum,
    required,
    rounding,
    aggregation,
    ...
}
```

The exact encoding may differ, but `runtime_numeric_id` cannot live only in mutable registry/display metadata.

## One-to-one mapping

Within one immutable scheme:

```text
semantic_id <-> runtime_numeric_id
```

is one-to-one.

The canonicalizer rejects:

- duplicate semantic IDs;
- duplicate runtime numeric IDs;
- ambiguous aliases;
- out-of-range numeric IDs;
- runtime IDs represented in more than one canonical numeric form.

## Identity sensitivity

Changing only the runtime numeric ID of a dimension changes the canonical bytes and therefore changes `ResourceAccountingSchemeId`.

For example, a mapping change from:

```text
budget.compute -> 0
budget.energy  -> 1
```

to:

```text
budget.compute -> 1
budget.energy  -> 0
```

must produce a different accounting-scheme identity even if every other field is unchanged.

## No local remapping

Runtime code MUST NOT maintain an independent mutable table that remaps canonical semantic IDs into different `ResourceDimensionId` values while retaining the same scheme ID.

A parser/registry may build an efficient lookup table, but that table is derived exactly from the canonical schema bytes and is checked against the committed scheme identity.

## Version transition

Adding the numeric mapping to the canonical resource schema is a semantic encoding transition.

Existing v0.1 test IDs are not silently upgraded. A revised canonical schema produces a new `ResourceAccountingSchemeId` and is treated as a cross-scheme transition.

Existing grants, limits, counters, or descendant state do not automatically carry over.

## Conservative migration

Any authority-bearing migration from an older resource scheme to a new numeric-ID-bound scheme requires the ordinary conservative resource translation proof:

```text
target_remaining <= ConservativeImage(source_remaining)
```

Changing a numeric ID is not treated as a cosmetic operation unless a separately verified translation proves exact semantic preservation without widening authority.

## Registry rule

A production `VerifiedSchemaRegistrySnapshot` rejects a resource schema profile that does not commit the runtime numeric mapping required by its admitted runtime profile.

The registry cannot repair an under-specified schema by attaching an unsigned or separately mutable map.

## Golden-vector requirements

A revised golden corpus must prove at least:

1. canonical semantic IDs plus numeric IDs hash deterministically;
2. changing only one numeric ID changes the scheme digest;
3. duplicate numeric IDs are rejected;
4. duplicate semantic IDs are rejected;
5. Python/reference and Rust implementations agree on exact bytes/digest;
6. Rust resolution derives the expected numeric `ResourceDimensionId` values from the committed schema rather than hard-coding them externally.

## Non-claims

This contract does not define what physical resources mean or how they are measured. It only ensures that the runtime dimension key used by RSK arithmetic is part of the same immutable semantic identity as the rest of the accounting scheme.
