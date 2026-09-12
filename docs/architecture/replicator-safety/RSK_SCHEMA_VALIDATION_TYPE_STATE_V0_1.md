# RSK Schema Validation Type State v0.1

**Status:** Reference contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

This contract separates two facts that must never be conflated:

```text
schema-bound != schema-validated
```

A value may carry the exact digest/identity of a capability or resource schema and still be invalid under that schema. Examples include a reserved capability bit, a retired capability bit, an unknown resource dimension, or omission of a required resource dimension.

This contract contains no physical replication mechanism, manufacturing recipe, biological/molecular design, or autonomous fabrication path.

## Type-state boundary

The reference boundary is:

```text
BoundCapabilitySet
    + ResolvedCapabilitySchema
        -> validate()
            -> ValidatedCapabilitySet

ResourceVector
    + ResolvedResourceAccountingScheme
        -> validate()
            -> ValidatedResourceVector
```

`Validated*` values are opaque construction capabilities. Their internal value is private and there is no public unchecked constructor.

A future production authority API should accept validated values rather than caller-constructed bound values.

## Resolved is not verified

`ResolvedCapabilitySchema` and `ResolvedResourceAccountingScheme` prove only structural consistency of one supplied schema table and exact identity binding.

They do **not** prove:

- registry provenance;
- cryptographic signatures;
- trust-snapshot freshness;
- signer role/lifecycle;
- policy approval;
- supersession/revocation state;
- semantic equivalence across schema versions.

That future boundary should produce separately opaque verified registry/schema capabilities. Structural validation must remain usable as a lower-level pure function underneath that verifier.

## Capability validation

A resolved capability schema binds:

- exact `CapabilitySchemaId`;
- bit width in `1..=64`;
- a canonical strictly increasing bit-rule table;
- each known bit to one of `Assignable`, `Reserved`, or `Retired`.

Validation requires:

1. the bound value's schema ID exactly equals the resolved schema ID;
2. every active bit is inside the declared width;
3. every active bit has a known rule;
4. every active bit is `Assignable`;
5. reserved, retired, unknown, and out-of-width bits fail closed.

Unclassified in-width bits are unknown, not implicitly assignable.

## Resource validation

A resolved resource-accounting scheme binds:

- exact `ResourceAccountingSchemeId`;
- a non-empty canonical strictly increasing dimension-rule table;
- required/optional status per dimension;
- exact numeric minimum and maximum per dimension.

Validation requires:

1. the vector's scheme ID exactly equals the resolved scheme ID;
2. every present dimension is known;
3. every required dimension is present exactly once;
4. the underlying vector remains canonically ordered and duplicate-free;
5. every amount is within the exact dimension range.

Unknown dimensions and missing required dimensions fail closed.

## Arithmetic after validation

Capability intersection between two validated sets under the same schema remains validated because intersection can only remove active bits.

Resource arithmetic is different. Addition or subtraction may produce a value outside the resolved scheme's allowed range. Therefore the resolved resource scheme must revalidate arithmetic output before returning another `ValidatedResourceVector`.

For a conservative migration envelope:

```text
target_remaining <= supplied_conservative_envelope
```

may be verified structurally, but this layer cannot manufacture the envelope or infer semantic equivalence between schemas.

## Authority integration rule

Until a later Class A integration tranche lands, existing RSK grants/evaluator/ledger continue to use their legacy reference fields.

When schema-aware authority integration begins, positive-authority paths must not accept merely caller-constructed schema-bound values. The intended production-target sequence is:

```text
canonical schema bytes
 -> exact schema identity
 -> verified registry/trust resolution
 -> structural validation
 -> opaque Validated* value
 -> verified policy/grant evaluation
 -> bounded authorization
 -> atomic commit
```

The exact ordering of verified resolution and structural validation may be composed internally, but both predicates must hold before positive authority is created.

## Required negative tests

At minimum:

- malformed capability bit width;
- duplicate/out-of-order capability rules;
- rule outside bit width;
- schema-ID substitution;
- reserved capability bit;
- retired capability bit;
- unknown capability bit;
- active bit outside width;
- empty resource schema;
- duplicate/out-of-order resource rules;
- invalid numeric range;
- resource scheme-ID substitution;
- unknown resource dimension;
- missing required resource dimension;
- below-minimum/above-maximum amount;
- arithmetic output that violates a range;
- conservative-transition widening.

## Non-claims

This contract does not establish:

- a production capability vocabulary;
- physical resource measurement semantics;
- trusted schema registry state;
- cryptographic verification;
- safe cross-schema translation;
- production admission.

It closes one narrower class of semantic confusion: a caller cannot turn arbitrary schema-bound numbers into an opaque validated value without satisfying the resolved schema's structural rules.
