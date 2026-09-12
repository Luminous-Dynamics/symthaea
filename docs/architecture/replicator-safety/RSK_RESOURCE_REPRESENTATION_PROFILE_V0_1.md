# Replicator Safety Kernel — Resource Representation / Arithmetic Profile v0.1

Status: **normative runtime-representation contract; not production admission**

This document binds the current Rust RSK resource-accounting implementation to an explicit representation/arithmetic profile so a structurally valid resource schema cannot enter authority paths when the admitted runtime cannot execute its semantics exactly.

It contains no physical resource model, physical replication mechanism, manufacturing recipe, or material specification.

---

## 1. Governing distinction

The generic resource-accounting schema is intentionally more expressive than the current Rust implementation.

The current Rust semantic TCB stores:

```text
ResourceQuantity.amount: u64
```

and implements checked integer addition, subtraction, remaining-budget calculation and componentwise comparison.

Therefore:

```text
structurally valid resource schema
    !=
executable by current Rust resource arithmetic profile
```

---

## 2. Current profile identity

The current Rust resource profile is:

```text
symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

Current eligibility requires:

- runtime-ID-bound resource schema v0.2;
- all minimum/maximum amounts exactly representable as `u64`;
- `aggregation = sum` for every dimension;
- `rounding = exact` for every dimension;
- exact scheme identity;
- checked arithmetic with no saturating widening.

---

## 3. Generic schema compatibility

Generic/reference schemas may remain valid while using semantics outside this profile, including:

- wider integer ranges;
- `aggregation = max`;
- non-exact rounding policies.

Such schemas are not current-Rust eligible.

Conceptually:

```text
validate_resource_schema(schema)            -> generic validity
require_current_rust_resource_schema(schema)-> runtime eligibility
```

The second check is intentionally stricter.

---

## 4. Numeric width

For every dimension:

```text
0 <= minimum <= maximum <= u64::MAX
```

A schema with `maximum = u64::MAX + 1` may remain generic schema evidence, but current-Rust validator derivation fails closed.

Forbidden behavior includes:

- truncation;
- saturation;
- wrapping;
- narrowing casts;
- clamping to `u64::MAX`;
- treating the excess range as unreachable.

---

## 5. Aggregation semantics

The current profile admits:

```text
aggregation = sum
```

because the current resource state machine uses checked additive consumption/accounting.

A dimension declaring:

```text
aggregation = max
```

cannot be interpreted with the same runtime arithmetic. It is denied for this profile until a separately implemented and qualified profile exists.

No adapter may silently convert `max` to `sum` or vice versa.

---

## 6. Rounding semantics

The current profile admits:

```text
rounding = exact
```

Non-exact rounding policies remain part of generic schema semantics but are not implemented by the current Rust arithmetic path.

They therefore fail current-profile eligibility.

This prevents a verifier from committing one rounding rule while runtime arithmetic effectively uses another.

---

## 7. Scale and unit identity

Canonical `unit` and `scale` remain committed by the resource scheme identity.

For this profile, arithmetic operates on the already represented integer amount under one exact scheme. No cross-scheme unit conversion is performed by the constitutional evaluator.

A later profile that performs conversion or fixed-point rescaling requires separate explicit qualification.

---

## 8. Adapter binding

The deterministic resource adapter emits:

```text
{
    schema_id,
    representation_profile,
    rules
}
```

where `representation_profile` is:

```text
symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

and the rules are derived only after current-profile eligibility succeeds.

For the abstract v0.2 resource golden schema, the current resource adapter-output SHA-256 is:

```text
baf0022df5f556743032e0ab1c1105838d120461649167b34a433679cd810b0a
```

This supersedes the previous adapter-output digest that did not yet bind the arithmetic profile.

---

## 9. Boundary tests

The reference adapter must cover at least:

```text
maximum = u64::MAX     -> eligible
maximum = u64::MAX + 1 -> current profile DENY
aggregation = sum      -> eligible
aggregation = max      -> current profile DENY
rounding = exact       -> eligible
non-exact rounding     -> current profile DENY
```

Generic schema validity and current-runtime eligibility must remain visibly separate in those tests.

---

## 10. Future arithmetic profiles

A future implementation may support:

- wider integers;
- `max` aggregation;
- other aggregation operators;
- fixed-point conversion;
- schema-directed rounding.

Each such change requires a new explicit representation/arithmetic profile and qualification evidence.

It is not a transparent implementation detail.

Existing authority must not silently move between profiles.

---

## 11. Build/runtime identity

The admitted resource representation/arithmetic profile must eventually be bound through:

- build/release provenance;
- runtime configuration identity;
- schema-derived validator identity;
- lineage/resource policy;
- grants and requests;
- evaluated authorization;
- ledger accounting;
- checkpoints/replay;
- recovery/epoch transition evidence.

---

## 12. Authority non-amplification

Profile eligibility does not:

- mint a grant;
- reset resource consumption;
- widen a budget;
- clear quarantine/revocation;
- extend expiry;
- satisfy quorum;
- authorize cross-schema conversion;
- establish production admission.

It proves only that the current Rust TCB can represent and execute the declared accounting semantics without reinterpretation.

---

## 13. Production status

Until this profile is integrated into the production Rust semantic path and bound through durable/runtime evidence, production admission remains:

**DENIED / NOT YET ELIGIBLE**.
