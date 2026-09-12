# ADR-014: RSK Executable Semantic-Schema Reference Tooling

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

ADR-013 freezes capability/resource semantic identity by canonical schema digest and requires golden vectors, exact schema binding, and conservative migration rules.

A prose-only schema contract still leaves room for implementations to disagree on canonical encoding, ordering, unknown bits/dimensions, or digest calculation.

This ADR introduces a small Rust-independent executable reference profile for those structural semantics.

It contains no physical replication mechanism or physical resource recipe.

## Decision

Add:

```text
scripts/rsk_semantic_schema.py
scripts/test_rsk_semantic_schema.py
docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json
```

as Class A-governed reference tooling/evidence.

## Scope

The reference toolkit may:

- validate the frozen v0.1 canonical JSON profile;
- compute SHA-256 semantic-schema IDs;
- validate capability-schema shape and ordering;
- reject unknown/reserved/retired capability bits;
- validate resource-scheme shape and required dimensions;
- reject schema digest mismatch;
- compute checked remaining resource vectors;
- verify that a proposed target remaining vector does not exceed a supplied conservative envelope.

## Explicit non-authority boundary

The toolkit does **not** prove that a cross-schema translation is semantically safe.

In particular:

```text
verify_conservative_remaining_transition(..., conservative_image)
```

only checks that the proposed target remainder is bounded by an already supplied conservative image.

Establishing that the image is a valid semantic mapping between two accounting schemes remains a separately verified translation-policy/proof responsibility.

Likewise, the toolkit does not infer capability-semantic attenuation from names or descriptions.

## Canonical JSON profile

The v0.1 profile intentionally permits only:

- objects with string keys;
- arrays;
- strings;
- booleans;
- integers.

Floats and null are excluded from normative schema content.

Serialization uses:

```text
UTF-8
sort_keys = true
separators = (",", ":")
ensure_ascii = false
```

SHA-256 over those bytes is the schema ID.

This profile is intentionally narrow. A future change is a governed schema-profile change, not an invisible serializer upgrade.

## Capability validation

The toolkit requires:

- exact v1 top-level fields;
- bounded bit width;
- entries sorted by bit;
- unique bits;
- unique capability IDs;
- explicit `assignable` or `retired` status;
- sorted/unique reserved bits;
- no overlap between entries and reserved bits.

A bound capability set is valid only when its supplied schema digest exactly matches the schema and every active bit is assignable.

## Resource validation

The toolkit requires:

- exact v1 top-level fields;
- nonempty dimensions sorted by canonical ID;
- unique dimension IDs;
- explicit unit/scale/bounds/required/rounding/aggregation fields;
- nonnegative bounded amounts;
- every required dimension present;
- no unknown active dimensions.

Remaining budget is computed only after exact-scheme validation and checked `consumed <= limit` per dimension.

## Golden vectors

The committed corpus uses abstract test semantics only. It freezes:

- capability-schema canonical digest;
- resource-accounting-schema canonical digest;
- representative valid/invalid capability sets;
- representative valid/invalid resource vectors;
- a checked remaining-budget vector.

The fixed v0.1 digest values are:

```text
capability schema:
da004c77da0df512ef772aa167fd386b61d6a581a3be40ded93d056e36dbc856

resource schema:
d055f5e3937473800e216cfa442da2e61dd98624357ea6abed3d2e7a90c8e03a
```

These vectors define encoding mechanics only; they are not a production capability vocabulary or resource policy.

## Verification evidence

Before this ADR was committed, the candidate reference implementation was exercised locally with Python against 13 cases:

1. stable capability digest;
2. semantic-description change changes digest;
3. same bits/different schema rejected;
4. reserved/retired bits rejected;
5. resource unit semantic change changes digest;
6. missing required dimension rejected;
7. resource scheme mismatch rejected;
8. checked per-dimension remaining budget;
9. conservative transition envelope cannot be exceeded;
10. floats/null rejected by canonical profile;
11. noncanonical capability entry ordering rejected;
12. noncanonical resource dimension ordering rejected;
13. golden corpus digests/vectors verified.

The local run completed 13/13 successfully.

This is local candidate evidence only; exact-head GitHub workflow evidence remains independently required.

## Governance

Because the toolkit defines the canonical identity used by future authority objects, the following are Class A surfaces:

- `scripts/rsk_semantic_schema.py`;
- `scripts/test_rsk_semantic_schema.py`;
- `docs/architecture/replicator-safety/golden/`;
- the existing normative RSK architecture directory;
- focused RSK workflow and Class A detector.

The focused governance job should execute the semantic-schema self-test on every relevant RSK PR.

## Alternatives considered

### Wait for the Rust implementation to define canonical bytes

Rejected. The Rust CI stack is still queued, and a separate executable reference profile provides useful implementation diversity and golden vectors now.

### Use serializer-default JSON

Rejected. Defaults can drift and may differ across languages/tool versions.

### Use floating-point resource values

Rejected for the v0.1 authority schema profile because cross-language numeric/rounding ambiguity is unnecessary at this layer.

### Let the reference tool automatically translate schemas

Rejected. Generic structural code cannot infer semantic attenuation merely from identifiers or units.

## Consequences

### Positive

- canonical schema identity is executable rather than prose-only;
- golden vectors are fixed and independently reproducible;
- same-bits/different-schema failures have a concrete reference test;
- required resource dimensions and remaining-budget checks are executable;
- future Rust implementations can be cross-checked against a small independent implementation.

### Costs

- one additional Class A Python reference implementation must be maintained;
- canonical profile changes require explicit migration/versioning;
- duplicate logic between Rust and Python is intentional and must be reconciled through golden vectors rather than hidden shared code.

### Residual risk

- the reference implementation itself may contain bugs;
- SHA-256/canonical structure proves identity, not that a schema is substantively safe;
- cross-schema semantic translation remains a separate high-trust proof boundary;
- local Python tests do not replace Rust, formal, fuzz, or production evidence.

## Production status

This tooling does not create production authority.

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```
