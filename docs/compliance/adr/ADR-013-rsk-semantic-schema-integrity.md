# ADR-013: RSK Semantic Schema Integrity for Capabilities and Resource Accounting

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The reference RSK currently represents:

- capabilities as a `u64` bitset;
- resource accounting as a scalar `u64`.

The numeric subset/intersection and budget arithmetic are useful reference semantics, but production authority also depends on the *meaning* of each bit and resource unit remaining stable.

A byte-identical grant can become more permissive if software reinterprets one bit. A numerically identical budget can become incomparable if a later generation changes units, scale, dimensions, or rounding semantics.

This ADR contains no physical replication mechanism or physical resource recipe.

## Decision

RSK production authority will treat capability and resource meaning as schema-bound state.

### Capability identity

```text
CapabilitySet = (CapabilitySchemaId, Bits)
```

### Resource identity

```text
ResourceAmount = (ResourceAccountingSchemeId, DimensionVector)
```

The schema/scheme ID is a canonical cryptographic digest of the immutable semantic definition. Human-readable versions are not authority identity.

## Same bytes do not imply same authority

RSK will explicitly reject the assumption:

```text
same bits/numbers -> same authority
```

unless the semantic schema identity is also equal.

A schema mismatch fails closed by default.

## Capability schema rules

Within one capability schema:

- bit/discriminant meanings are immutable;
- bit width is frozen;
- reserved/retired bits are explicit;
- retired bits are not silently reused;
- unknown/reserved bits cannot become active positive authority;
- all capability-bearing objects bind the schema ID.

Cross-schema subset/intersection is rejected unless a separately verified conservative translation exists.

## Resource accounting rules

Resource accounting uses a typed bounded vector with explicit dimensions, unit identifiers/scales, numeric bounds, rounding policy, and aggregation semantics.

Production profiles decide which abstract dimensions are required.

Direct arithmetic is valid only under one exact accounting scheme.

Missing required dimensions, unknown active dimensions, overflow/underflow, or scheme mismatch deny new positive authority.

## Ancestral anti-laundering

A descendant or newer grant generation may not escape ancestor restrictions by changing semantics rather than values.

Forbidden implicit paths include:

- capability bit reinterpretation;
- resource scheme swap;
- denomination/scale reset;
- dropping required dimensions;
- resetting consumed counters;
- permissive rounding;
- splitting/merging dimensions in a way that duplicates remaining authority;
- fresh grant generation under an incompatible accounting meaning.

## Translation is exceptional

Cross-schema transitions require explicit verified translation evidence.

No translation evidence means no authority carryover.

### Capability translation

Target semantic authority must be a subset of the source's conservatively understood authority:

```text
Meaning(target) ⊆ ConservativeMeaning(source)
```

### Resource translation

Translation is evaluated over **remaining authority**, not only nominal totals:

```text
old_remaining = old_limit - old_consumed
new_remaining <= ConservativeImage(old_remaining)
```

This avoids rounding or separate limit/consumption conversion increasing remaining budget.

## Rounding policy

Authority-sensitive conversions must round conservatively.

A resource transition must not:

- round remaining allowance upward;
- round consumed amount downward;
- saturate overflow into a more permissive value.

Ambiguous rounding fails closed.

## Required bindings

The exact capability/resource schema tuple must be bound through:

- lineage hard policy;
- parent/ancestor constraints;
- grant;
- requested action;
- evaluated authorization;
- committed descendant event;
- durable journal/checkpoint;
- safety case;
- admitted release/runtime identity;
- recovery/epoch transitions.

An authorization cannot be evaluated under one semantic tuple and committed under another.

## Upgrade rule

An authority-relevant schema change is a governance event, not an implementation detail.

Safe options are:

1. preserve the exact old schema/scheme;
2. introduce a new schema plus verified attenuating translation/requalification;
3. enter a fresh governed epoch with no implicit positive-authority carryover.

Unknown or unprovable transitions fail closed.

## Recovery rule

Recovery does not gain a special semantic shortcut.

Same-schema state may carry forward only with replay/continuity evidence. Cross-schema state requires verified conservative migration. Otherwise recovery establishes a fresh epoch and positive authority is re-established explicitly.

## Durable evidence

Durable records always carry exact schema IDs. A current parser's ability to read old bytes is not authority.

Replay must reject:

- missing/unknown schema;
- schema substitution;
- incompatible grant/action/commit tuple;
- generation laundering;
- stale migration evidence;
- checkpoint normalization that changes semantic meaning.

## Type-state direction

Production implementation should distinguish raw semantic data from verified migration capabilities, for example:

```text
CapabilitySchemaId
BoundCapabilitySet
VerifiedCapabilityTranslation

ResourceAccountingSchemeId
ResourceVector
VerifiedResourceTranslation
```

Verified translation types should not be directly deserializable as trusted process capabilities.

## Golden vectors

Each schema family requires canonical golden vectors for definition encoding, digest identity, representative authority values, invalid/unknown cases, and supported transition examples.

Changing a golden vector either indicates an implementation defect or a real governed schema change.

## Verification plan

`RSK_SEMANTIC_SCHEMA_TRANSITIONS_TEST_PLAN_V0_1.md` defines CS-*, RA-*, and ST-* families covering:

- same bits/different schema;
- unknown/reserved bits;
- capability remap widening;
- schema rollback;
- resource dimension/scale drift;
- overflow and missing dimensions;
- generation laundering;
- conservative translation and rounding;
- dimension drop/split/merge;
- uncertainty;
- checkpoint equivalence;
- combined capability/resource migrations;
- exact action binding;
- recovery/fresh epoch;
- property/formal/API/fuzz targets.

## Relationship to build/runtime admission

ADR-012 binds admitted releases and runtime instances to exact capability/resource schema digests.

This ADR defines what those schema digests mean and how they may transition.

A release/runtime identity match is insufficient if the loaded schema definition does not match the admitted canonical schema content.

## Alternatives considered

### Keep raw bitset/scalar and document meanings informally

Rejected. Documentation drift can change authority while preserving serialized values.

### Use only human semantic-version strings

Rejected. Reusing a version string with changed content or parser differences can silently alter authority. Canonical content digest is required.

### Automatically translate schemas by name/unit

Rejected. Name equality does not prove semantic attenuation, and automatic unit conversion can widen remaining authority through rounding/dimension loss.

### Reset resource counters on new grant generation

Rejected. It enables generation laundering around ancestor budgets.

### Permit unknown future bits/dimensions for forward compatibility

Rejected for positive authority. Unknown critical semantics may be retained for audit but cannot become active authority.

## Consequences

### Positive

- old grants cannot silently change semantic meaning after upgrade;
- descendants cannot launder authority through schema changes;
- resource accounting becomes dimensionally and semantically comparable;
- migration/recovery semantics are explicit and attenuating;
- release/runtime admission can bind exact semantic vocabularies;
- property/formal tests gain a precise transition relation.

### Costs

- schema registries and canonical encoders must be maintained;
- upgrades require governance/migration evidence;
- resource vectors are more complex than one scalar;
- some migrations may require a fresh epoch rather than automatic conversion.

### Residual risk

- a flawed schema definition may be consistently enforced while still being unsafe;
- a compromised translation authority can be dangerous if its proof/trust boundary is weak;
- measurement quality for externally observed resource dimensions remains a separate trust problem;
- semantic integrity does not replace the other constitutional, containment, time, provenance, or runtime-identity gates.

## Production status

No capability/resource schema is production-admitted by this ADR.

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```

#1678 and #1679 remain open until implementation and exact-head executed evidence satisfy their acceptance criteria.