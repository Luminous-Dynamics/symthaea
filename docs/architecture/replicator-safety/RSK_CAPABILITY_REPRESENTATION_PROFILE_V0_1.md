# Replicator Safety Kernel — Capability Representation Profile v0.1

Status: **normative runtime-representation contract; not production admission**

This document binds the current Rust RSK capability representation to an explicit profile so a structurally valid capability schema cannot enter authority paths if the admitted runtime cannot represent it exactly.

It contains no physical replication mechanism.

---

## 1. Governing distinction

The generic capability schema contract uses a conceptual `BitVector` and may describe schemas wider than today's implementation.

The current Rust semantic TCB uses:

```text
BoundCapabilitySet.bits: u64
```

Therefore:

```text
structurally valid capability schema
    !=
representable by current Rust capability profile
```

---

## 2. Current profile identity

The current Rust representation profile is:

```text
symthaea.rsk.capability-representation.rust-u64.v1
```

Its exact width requirement is:

```text
1 <= bit_width <= 64
```

A schema wider than 64 bits may remain valid generic/reference evidence but is not eligible for the current Rust authority path.

---

## 3. Fail-closed behavior

For the current profile:

```text
bit_width > 64 -> DENY current-Rust validator derivation
```

Forbidden compatibility behavior includes:

- truncating high bits;
- masking to the low 64 bits;
- modulo mapping;
- silently dropping rules above bit 63;
- accepting a wider schema merely because one particular value currently uses only low bits;
- treating `u64` storage as an implementation detail outside the evidence identity.

---

## 4. Adapter binding

The deterministic schema adapter emits the representation profile as part of the capability validator-table output:

```text
{
    schema_id,
    representation_profile,
    bit_width,
    rules
}
```

Thus validator-table identity binds both:

- semantic schema identity; and
- the runtime representation used to interpret that schema.

For the abstract golden capability schema, the current adapter-output SHA-256 is:

```text
d118f3777c4d78ac05293e9228706afa31a57ba69c7f1bd65dde9aca96c8c460
```

This supersedes the earlier capability adapter-output digest from ADR-020 because that earlier table did not yet bind the representation profile.

---

## 5. Boundary tests

The reference adapter must prove at least:

```text
width 64 -> current Rust u64 profile eligible
width 65 -> generic schema may remain structurally valid
width 65 -> current Rust u64 profile DENIED
```

The denial occurs before a Rust validator rule table is emitted.

---

## 6. Runtime/build identity

The admitted representation profile must ultimately be bound to:

- release/build evidence;
- runtime policy/config identity;
- schema-derived validator identity;
- capability-bearing grants/requests/authorizations;
- durable checkpoint/replay semantics;
- recovery/epoch transition evidence.

A runtime using a different representation profile is not automatically equivalent even if the same schema digest is understood.

---

## 7. Future wider representations

A future implementation may introduce a wider canonical capability vector.

That requires a new explicit representation profile and qualification evidence.

Examples might include a fixed wider bit-vector or another canonical representation, but no such profile is admitted by this document.

A representation-profile transition must not silently reinterpret existing authority.

Existing grants and durable evidence remain bound to the representation/profile under which they were evaluated unless a separately verified migration establishes safe equivalence/attenuation.

---

## 8. Registry composition

Verified schema provenance and representation eligibility are separate checks:

```text
VerifiedSchemaRegistrySnapshot
    -> exact capability schema
    -> structural schema validity
    -> admitted runtime representation profile
    -> deterministic validator derivation
```

A valid registry signature cannot make an unrepresentable schema usable by the current runtime.

---

## 9. Authority non-amplification

Representation-profile validation does not:

- mint a grant;
- satisfy quorum;
- clear quarantine/revocation;
- extend expiry;
- translate cross-schema capability meaning;
- establish trusted time;
- establish production admission.

It only proves lossless representability by one admitted semantic TCB profile.

---

## 10. Production gate

Until the representation profile is bound through the production Rust validator, grants, ledger/replay and exact build/runtime admission, production status remains:

**DENIED / NOT YET ELIGIBLE**.
