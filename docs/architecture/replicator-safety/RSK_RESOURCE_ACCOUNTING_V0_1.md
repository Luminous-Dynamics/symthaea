# Replicator Safety Kernel — Resource Accounting v0.1

Status: **normative accounting-integrity contract; not a production admission record**

This document defines how RSK binds every resource-bearing amount to one exact accounting scheme so numeric limits cannot be preserved while their denomination or meaning silently changes.

It contains no physical replication mechanism, physical material recipe, manufacturing process, or operational recipe.

---

## 1. Problem

The reference RSK uses an abstract `u64` for resource ceilings, requested amounts, and consumed amounts.

That arithmetic is useful only while every participant agrees on what one unit means.

Unsafe semantic drift can occur even when every numeric value remains unchanged:

```text
scheme A: 100 units = accounting meaning X
scheme B: 100 units = accounting meaning Y
```

A new grant or descendant could otherwise escape an ancestor's resource restriction by changing the accounting definition rather than the number.

The governing rule is:

```text
ResourceAmount = (ResourceAccountingSchemeId, DimensionVector)
```

---

## 2. Accounting scheme identity

A resource accounting scheme is a canonical immutable definition.

Conceptually:

```text
ResourceAccountingScheme {
    family,
    version,
    dimensions,
    numeric_representation,
    rounding_policy,
    aggregation_policy,
    canonical_encoding_version,
}

ResourceAccountingSchemeId = Digest(canonical(ResourceAccountingScheme))
```

The digest is the authority identity. Human-readable names/versions are metadata.

---

## 3. Generic dimensional model

Production accounting SHOULD support a bounded vector rather than assuming one scalar is universally sufficient.

A dimension definition may include:

- stable dimension identifier;
- canonical unit identifier;
- scale / fixed-point exponent if applicable;
- minimum/maximum representable value;
- whether the dimension is mandatory for a policy profile;
- aggregation semantics;
- rounding direction;
- saturation/overflow policy;
- measurement/evidence class identifier where applicable.

Abstract dimension classes can include resource, energy, compute, elapsed/operational, or other policy-defined budgets without encoding physical recipes.

---

## 4. Typed resource amount

Conceptually:

```text
ResourceVector {
    scheme_id: ResourceAccountingSchemeId,
    dimensions: Map<DimensionId, BoundedAmount>,
}
```

Every dimension required by the scheme/profile must be present.

Missing does **not** mean zero unless the scheme explicitly defines that exact semantics.

Unknown dimensions in positive-authority inputs fail closed unless the schema defines forward-compatible audit-only handling.

---

## 5. Exact bindings

The resource accounting scheme ID MUST be bound to:

- lineage hard policy;
- ancestor/subtree budget scope;
- subject authority state;
- replication grant;
- requested action;
- evaluated authorization;
- committed descendant event;
- durable state/checkpoint digest;
- safety-case snapshot;
- admitted release/runtime configuration;
- recovery/epoch transition evidence.

A numeric resource amount detached from its scheme ID is incomplete authority evidence.

---

## 6. Arithmetic only within one scheme

Direct comparison, addition, subtraction, and remaining-budget calculation are valid only when all operands use the same accounting scheme and compatible dimension set.

Default mismatch behavior:

```text
lhs.scheme_id != rhs.scheme_id -> DENY
```

No automatic unit conversion occurs inside the constitutional evaluator.

---

## 7. Checked arithmetic

All per-dimension arithmetic uses checked operations.

Required behavior:

- overflow -> deny;
- underflow -> deny;
- negative remaining amount -> deny;
- invalid scale -> deny;
- non-canonical numeric representation -> deny;
- lossy rounding that could widen authority -> deny.

Production code MUST NOT saturate to a more permissive value after overflow.

---

## 8. Required dimensions

Policy determines the required accounting dimensions for a risk/deployment profile.

A grant/action cannot drop an ancestor-required dimension.

For every required dimension `d`:

```text
Consumed_d <= Limit_d
Remaining_d = Limit_d - Consumed_d
```

and descendant activity consumes against every applicable ancestor scope according to the scheme's aggregation semantics.

---

## 9. Ancestral anti-laundering

A descendant or new grant generation MUST NOT escape ancestral constraints by:

- switching accounting scheme;
- renaming a dimension;
- changing scale/denomination;
- dropping a dimension;
- resetting consumed counters;
- reclassifying consumption into an unbounded dimension;
- applying a permissive translation;
- entering a newer grant generation with incompatible accounting semantics.

The core property is:

```text
semantic remaining authority after transition
    <= semantic remaining authority before transition
```

for every applicable ancestor constraint.

---

## 10. Grant generation does not reset accounting

A new grant generation may establish new positive authority only under its governing policy.

It MUST NOT implicitly reset ancestral consumed resource state.

Where a new generation changes accounting semantics, it requires the schema-transition process described below.

---

## 11. Cross-scheme translation

Cross-scheme translation is exceptional.

Conceptually:

```text
VerifiedResourceTranslation {
    source_scheme_id,
    target_scheme_id,
    mapping_policy_id,
    proof/evidence_root,
    rounding_policy,
    validity_scope,
}
```

Default:

```text
No VerifiedResourceTranslation -> No CrossSchemeAuthority
```

---

## 12. Conservative translation

A valid translation must conservatively map **remaining** authority, not merely convert nominal totals.

For old state:

```text
old_remaining = old_limit - old_consumed
```

A translated target state is admissible only if policy proves:

```text
target_remaining <= ConservativeImage(old_remaining)
```

for every target dimension and every applicable ancestor scope.

The translation MUST NOT independently convert the limit and consumed values in ways that produce a larger remainder through rounding.

---

## 13. Rounding

Rounding direction is authority-sensitive.

When converting **remaining allowance**, rounding must be conservative against widening authority.

When converting **consumed amount**, rounding must not understate consumption.

A scheme/translation must define which direction is conservative for each representation.

Undefined or ambiguous rounding -> deny.

---

## 14. Dimension drop

Dropping a source dimension is not automatically safe.

A target scheme that lacks a source required dimension is admissible only if one of the following is proven:

1. the remaining authority in that dimension is zero and policy allows retirement; or
2. the constraint is conservatively embedded in another target dimension; or
3. a fresh epoch explicitly establishes new authority without implicit carryover.

Otherwise, deny translation.

---

## 15. Dimension split/merge

### Split

One source dimension mapping into multiple targets may widen authority if each target independently receives the full remaining allowance.

Default: reject unless the translation proves a conservative partition.

### Merge

Multiple source dimensions mapping into one target may lose the strictest source constraint.

Default: reject unless the translation proof demonstrates no widening.

---

## 16. Measurement provenance

Where production accounting depends on measurements rather than purely internal counters, the accounting scheme/profile must state what evidence establishes the measurement.

Measurement evidence may include:

- trusted meter/sensor identity;
- freshness;
- calibration/qualification state;
- uncertainty bound;
- failure domain;
- attestation/provenance.

A cryptographically authenticated measurement can still be too stale, uncertain, or unqualified for authority.

---

## 17. Uncertainty

If measurement uncertainty affects whether a limit is exhausted, positive authority uses the conservative side of the uncertainty interval.

Conceptually:

```text
possible_consumed = [min, max]
```

Budget eligibility must use `max` when determining whether consumption could already have reached the limit.

Uncertainty moves authority toward denial, not extension.

---

## 18. Durable evidence and replay

Every resource-bearing durable record binds the accounting scheme ID and canonical dimension vector.

Replay rejects:

- missing scheme;
- unknown scheme;
- inconsistent dimension set;
- scale/denomination drift;
- arithmetic overflow;
- consumption decrement within the same lineage/epoch where monotonicity is required;
- grant-generation reset;
- incompatible scheme transition;
- translation without verified evidence;
- checkpoint whose reconstructed budgets differ from replay-from-genesis.

---

## 19. Checkpoint equivalence

A checkpoint is valid only if replay proves it preserves the same resource state as the authoritative history.

At minimum compare:

- scheme ID;
- dimension definitions/profile;
- ancestor scopes;
- limits;
- consumed values;
- grant generations;
- epoch;
- head/event commitment.

A checkpoint cannot normalize an old scheme into a new meaning without explicit translation evidence.

---

## 20. Recovery and fresh epoch

Recovery cannot silently reinterpret prior resource counters.

Safe recovery choices:

### Same scheme

Carry state forward only if replay/continuity proves exact accounting-state preservation.

### Verified conservative migration

Carry state through a separately verified translation.

### Fresh epoch

When preservation cannot be proven, establish a fresh epoch with no implicit positive-authority carryover. New resource authority must be explicitly re-established.

---

## 21. API/type boundary

Conceptually distinct production types:

```text
ResourceAccountingSchemeId
DimensionId
BoundedAmount
ResourceVector
VerifiedResourceTranslation
```

The API should not allow arbitrary numeric maps to become trusted resource authority without scheme/dimension validation.

Verified translations should be opaque process capabilities, reconstructed from durable evidence after replay rather than trusted by deserialization.

---

## 22. Golden vectors

Golden vectors freeze:

- canonical scheme encoding;
- scheme digest;
- dimension identifiers/order;
- scales;
- numeric bounds;
- required dimensions;
- rounding/aggregation semantics;
- representative vectors;
- overflow/missing/unknown cases;
- translation examples where supported.

A changed golden vector implies either implementation drift or a real scheme change requiring governance.

---

## 23. Failure semantics

New positive authority is denied when:

- accounting scheme missing/unknown;
- scheme mismatch;
- required dimension missing;
- unknown active dimension;
- scale mismatch;
- numeric overflow/underflow;
- consumed > limit;
- uncertainty could place consumption at/over limit;
- ancestor scheme differs without verified translation;
- translation would increase remaining authority;
- grant generation attempts to reset counters;
- durable replay/checkpoint disagrees;
- runtime accounting scheme differs from admitted release.

Denial/freeze is not destructive action.

---

## 24. Formal invariants

Future formal/property models should target:

```text
SchemeMismatchCannotAuthorize
RequiredDimensionCannotDisappear
ConsumptionNeverDecreasesWithinScope
GrantGenerationCannotResetAncestorConsumption
TranslationNeverIncreasesRemainingAuthority
OverflowCannotIncreaseAuthority
UncertaintyCannotIncreaseAuthority
RecoveryCannotReinterpretAccountingImplicitly
RuntimeSchemeMustMatchAdmission
```

---

## 25. Production status

This document does not define physical resource recipes or amounts and does not productionize the reference scalar `u64` accounting.

It specifies the semantic integrity required before resource arithmetic can become production authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.