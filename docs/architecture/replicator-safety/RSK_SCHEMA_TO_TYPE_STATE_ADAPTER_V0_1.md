# Replicator Safety Kernel — Verified Schema to Type-State Adapter v0.1

Status: **normative semantic-provenance contract; not production admission**

This document defines the deterministic bridge between exact canonical RSK schema bytes and the structural validation rule tables consumed by the minimal Rust semantic TCB.

It contains no physical replication mechanism, manufacturing recipe, biological/molecular design, or physical resource model.

---

## 1. Governing distinction

The structural validator introduced by the reference Rust semantics crate accepts a schema identity plus a rule table.

That is useful for isolated testing, but it is not a production trust boundary.

The production rule is:

```text
correct schema ID + caller-supplied rules != verified schema-derived validator
```

Only rules deterministically derived from the exact verified canonical schema bytes are eligible for production positive-authority validation.

---

## 2. Required production chain

Conceptually:

```text
SignedSchemaRegistrySnapshot
    -> verify trust/policy/freshness/lifecycle
    -> VerifiedSchemaRegistrySnapshot
    -> resolve exact canonical schema bytes
    -> recompute exact schema ID
    -> deterministic schema-to-rule adapter
    -> structurally validate bound value
    -> validated semantic value
```

No production shortcut may replace the deterministic adapter with a caller-supplied rule table.

---

## 3. Capability adapter

For the v0.1 capability schema, the adapter derives:

```text
{
    schema_id,
    bit_width,
    rules: [
        { bit, class },
        ...
    ]
}
```

where `class` is one of:

```text
assignable
reserved
retired
```

Rules are sorted by numeric bit index.

The adapter derives:

- assignable/retired rules from canonical schema entries;
- reserved rules from canonical `reserved_bits`;
- exact `bit_width` from canonical schema bytes;
- exact `schema_id` by validating and hashing those same bytes.

No caller-supplied bit classification may override the result.

Bits not represented by assignable/reserved/retired rules remain unknown and fail closed in structural validation.

---

## 4. Resource adapter

For `symthaea.rsk.resource-accounting-schema.v2`, the adapter derives:

```text
{
    schema_id,
    rules: [
        {
            numeric_id,
            required,
            minimum,
            maximum
        },
        ...
    ]
}
```

Rules are sorted by committed runtime `numeric_id`.

Each rule is derived only from the exact canonical v0.2 dimension bytes.

The adapter MUST NOT accept an external semantic-name -> numeric-ID side table.

---

## 5. v0.1 resource schemas

Historical resource-schema v0.1 evidence does not bind runtime numeric IDs.

Therefore:

```text
resource_rule_table(v1) -> DENY
```

The adapter requires the v0.2 runtime-ID-bound profile or a later explicitly admitted equivalent.

This prevents a side assumption from upgrading historical evidence.

---

## 6. Determinism

Given identical canonical schema bytes, independent conforming implementations must produce identical adapter output.

Changing any field consumed by structural validation must either:

1. change the schema ID, or
2. fail schema validation.

In particular, changing only a v0.2 `numeric_id` changes both the scheme ID and the derived rule table.

---

## 7. No side-loaded rules in production

Reference constructors such as:

```text
ResolvedResourceAccountingScheme::new(id, rules)
ResolvedCapabilitySchema::new(id, rules)
```

remain useful for isolated semantic tests.

They MUST NOT become production authority constructors merely because `id` is a known digest.

A production integration must distinguish schema-derived/verified type state from caller-constructed reference state.

---

## 8. Provenance retention

A future production `VerifiedResolved*Schema` should retain enough evidence identity to prove which verified registry snapshot and canonical schema produced it.

At minimum bind:

- exact schema ID;
- verified registry snapshot identity;
- registry epoch/sequence;
- applicable policy identity;
- trust snapshot identity;
- verification/evaluation time evidence as required by the trusted-time contract.

The structural rule table itself is not sufficient provenance.

---

## 9. Restart behavior

Trusted validators are not durable `verified=true` objects.

After restart:

1. raw/signed registry evidence is reverified;
2. exact canonical schema bytes are resolved again;
3. schema ID is recomputed;
4. deterministic rule tables are rebuilt;
5. bound values are structurally revalidated as needed.

Cached rule tables may be retained only as non-authoritative acceleration data and must be checked against the reverified schema identity before use.

---

## 10. Authority non-amplification

Producing a deterministic rule table does not:

- mint a replication grant;
- satisfy replication quorum;
- clear quarantine/revocation;
- extend expiry;
- widen capability/resource ceilings;
- prove a cross-schema translation;
- establish trusted time;
- establish runtime build identity.

It proves only that one structural validator configuration is the deterministic image of one exact canonical schema.

---

## 11. Golden-vector profile

The Python reference adapter is:

```text
scripts/rsk_schema_adapter.py
```

and its self-tests are:

```text
scripts/test_rsk_schema_adapter.py
```

The adapter uses the committed semantic golden corpora and must demonstrate at least:

- exact capability rule derivation;
- exact v0.2 resource rule derivation;
- v0.1 resource rejection for runtime-rule construction;
- numeric-ID remap changes both scheme identity and rule output;
- output ordering matches Rust structural-validator expectations.

Rust must converge on these outputs before production integration.

---

## 12. Production gate

Production positive-authority APIs must consume values validated through a verified schema-derived path, not values minted from arbitrary reference constructors.

Until that invariant is implemented and executed under exact-head qualification, production admission remains:

**DENIED / NOT YET ELIGIBLE**.
