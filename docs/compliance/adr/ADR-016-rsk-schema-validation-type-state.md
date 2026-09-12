# ADR-016: RSK schema validation type state

**Status**: Accepted for reference implementation

**Date**: 2026-09-12

**Change Class**: A

## Context

ADR-015 and draft PR #1936 introduce schema-bound Rust values for RSK capability/resource semantics. Static review identified a remaining authority hazard:

```text
schema-bound != schema-validated
```

Binding a numeric value to an exact schema digest prevents cross-schema substitution, but it does not by itself prove that the value is legal under that schema. A caller could otherwise bind a reserved/retired capability bit or omit a required resource dimension while still presenting the correct schema identity.

Issue #1938 tracks this production-admission boundary.

## Decision

Extend the minimal `symthaea-replicator-semantics` TCB with a pure structural validation module.

The module introduces:

- `ResolvedCapabilitySchema`;
- `CapabilityBitRule` and `CapabilityBitClass`;
- opaque `ValidatedCapabilitySet`;
- `ResolvedResourceAccountingScheme`;
- `ResourceDimensionRule`;
- opaque `ValidatedResourceVector`;
- `SchemaValidationError`.

The validated types have private fields and no public unchecked constructors.

## Capability rule

A capability value is validated only when:

1. its bound schema ID exactly matches the resolved schema ID;
2. all active bits fall within the declared width;
3. all active bits are known;
4. all active bits are explicitly `Assignable`.

Reserved, retired, unknown and out-of-width bits deny validation.

## Resource rule

A resource vector is validated only when:

1. its bound accounting scheme ID exactly matches the resolved scheme;
2. all dimensions are known;
3. all required dimensions are present;
4. all amounts satisfy the exact per-dimension numeric range.

The existing `ResourceVector` constructor continues to enforce canonical strictly increasing dimensions and uniqueness.

## Arithmetic rule

Validated capability intersection remains validated because it can only attenuate active bits.

Resource arithmetic is revalidated by the resolved resource scheme before another validated value is returned. This prevents a mathematically valid operation from silently producing a value outside the schema's declared numeric range.

## Trust boundary

`Resolved*Schema` deliberately does **not** mean `Verified*Schema`.

This tranche does not establish registry provenance, signatures, trust-snapshot freshness, signer lifecycle, policy approval, or cross-version semantic equivalence. Those remain separate positive-evidence/registry gates under #1668/#1678/#1679.

Future production authority APIs should accept opaque validated values produced under separately verified schema resolution rather than caller-constructed bound values.

## Validation

The new adversarial test surface covers:

- malformed schema tables;
- schema-ID substitution;
- reserved/retired/unknown/out-of-width capability bits;
- missing/unknown resource dimensions;
- invalid resource ranges;
- arithmetic output requiring revalidation;
- conservative-transition widening;
- the exact schema IDs already frozen by the cross-language golden corpus.

Because repository Actions remain queued and `Cargo.lock` is stale for the RSK packages (#1926), these tests are authored evidence only until the diagnostic/exact-head lanes execute.

## Non-goals

This ADR does not:

- modify existing grant/evaluator/ledger fields;
- approve a production capability vocabulary;
- define physical resource units;
- verify schema-registry provenance;
- approve cross-schema migration;
- establish production admission.

## Consequence

The RSK semantic stack now distinguishes three separate states:

```text
numeric value
 -> schema-bound value
 -> schema-validated value
```

A later verified registry layer adds a fourth trust predicate rather than overloading any of these types with provenance they do not prove.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
