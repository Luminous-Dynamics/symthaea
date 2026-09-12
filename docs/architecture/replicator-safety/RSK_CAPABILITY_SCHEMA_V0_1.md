# Replicator Safety Kernel — Capability Schema v0.1

Status: **normative semantic-integrity contract; not a production admission record**

This document defines how RSK binds capability-bearing values to a frozen semantic vocabulary so an unchanged bitset cannot silently change meaning across software versions.

It contains no physical replication mechanism.

---

## 1. Problem

The reference RSK correctly treats capabilities as a subset/intersection problem over a `u64` bitset.

That is numerically sound only while every participant agrees on what every bit means.

Unsafe semantic drift can occur even when all bytes remain unchanged:

```text
schema A: bit 3 -> capability_X
schema B: bit 3 -> capability_Y
```

An old grant could then pass every numeric subset check while authorizing a different semantic operation.

The governing rule is therefore:

```text
CapabilitySet = (CapabilitySchemaId, Bits)
```

and not merely:

```text
CapabilitySet = Bits
```

---

## 2. Capability schema identity

A capability schema has both a human-readable identity and a canonical cryptographic identity.

Conceptually:

```text
CapabilitySchema {
    schema_family,
    schema_version,
    bit_width,
    entries,
    reserved_bits,
    canonical_encoding_version,
}

CapabilitySchemaId = Digest(canonical(CapabilitySchema))
```

The digest is the authority identity. Human-readable version strings are metadata and MUST NOT substitute for the digest.

---

## 3. Frozen entries

Each schema entry binds at least:

- bit/discriminant index;
- canonical capability identifier;
- normative semantic description;
- authority class/category if policy uses one;
- deprecation status;
- whether the bit is assignable, reserved, or permanently retired.

Within one schema ID, those meanings are immutable.

Changing any authority-relevant meaning creates a different schema ID.

---

## 4. Bit reuse prohibition

A retired bit MUST NOT be silently reused for a new meaning within the same schema family/version lineage.

Safer options are:

1. leave the bit permanently retired/reserved; or
2. create a new schema with an explicit translation/requalification rule.

This protects old durable grants/evidence from semantic reinterpretation.

---

## 5. Reserved and unknown bits

A capability set is invalid if it sets:

- a bit outside the declared schema width;
- a bit reserved by the schema;
- an unknown/undefined bit;
- a permanently retired bit unless the schema explicitly allows historical decoding without authority.

For positive authority, unknown/reserved bits fail closed.

Historical evidence MAY preserve unknown future bits as opaque bytes for audit, but those bytes MUST NOT become active authority under an older verifier.

---

## 6. Typed capability set

The production semantic type is conceptually:

```text
BoundCapabilitySet {
    schema_id: CapabilitySchemaId,
    bits: BitVector,
}
```

Subset/intersection operations are valid only if:

```text
lhs.schema_id == rhs.schema_id
```

unless a separately verified monotonic translation is supplied.

Default cross-schema behavior is denial.

---

## 7. Exact bindings

The capability schema ID MUST be bound to every capability-bearing authority object, including:

- lineage hard policy;
- subject capability ceiling;
- replication grant;
- action request;
- evaluated authorization;
- descendant commit event;
- durable checkpoint/state digest;
- safety-case snapshot;
- admitted release capsule;
- runtime policy/config identity;
- recovery/epoch transition evidence.

A capability object detached from its schema ID is incomplete authority evidence.

---

## 8. Parent/child attenuation

The existing constitutional rule remains:

```text
EffectiveChildCapabilities
    ⊆ ParentCapabilityCeiling
    ∩ ExternalGrant
    ∩ CurrentPolicy
    ∩ CurrentContainment
```

This relation is evaluated only under one exact semantic schema or through a verified conservative translation.

A child MUST NOT gain semantic authority because an upgrade reinterprets bits.

---

## 9. Cross-schema translation

Cross-schema translation is exceptional, explicit, and separately verified.

Conceptually:

```text
VerifiedCapabilityTranslation {
    source_schema_id,
    target_schema_id,
    mapping_policy_id,
    proof/evidence_root,
    validity_scope,
}
```

The translation must prove that target authority is no more permissive than the source authority being carried forward.

Default rule:

```text
No VerifiedCapabilityTranslation -> No CrossSchemaAuthority
```

---

## 10. Conservative translation requirement

A translation MUST NOT create capability meaning that the source set did not already conservatively imply.

For each source set `S`, translated target set `T` must satisfy the policy's semantic attenuation relation:

```text
Meaning(T) ⊆ ConservativeMeaning(S)
```

If that relation cannot be mechanically or externally established, translation is denied.

A human statement such as “these are basically equivalent” is insufficient production evidence.

---

## 11. One-to-many and many-to-one mappings

One-to-many or many-to-one translations require special care.

### One-to-many

A source capability must not expand into multiple target capabilities unless the conjunction is demonstrably no more permissive than the source meaning.

Default: reject.

### Many-to-one

Combining several source capabilities into one target capability may accidentally widen authority if the target capability is broader than their intersection.

Default: reject unless the mapping proof demonstrates attenuation.

---

## 12. Schema upgrades

A software release that changes capability semantics is a governance event.

Safe upgrade paths:

### 12.1 Exact preservation

The new software retains the exact existing schema ID and semantics.

### 12.2 Explicit translated migration

A new schema ID is introduced and old authority is migrated only through a verified conservative translation plus any required requalification.

### 12.3 Fresh epoch

If semantic preservation cannot be proven, enter a governed fresh epoch with no implicit positive-authority carryover.

---

## 13. Downgrade and rollback

An older capability schema may remain historically valid while being ineligible for current authority.

Runtime/admission policy must distinguish:

- readable historical schema;
- currently accepted authority schema;
- deprecated schema;
- revoked/forbidden schema.

Rollback to an older binary MUST NOT silently reactivate an older schema and reinterpret current durable state.

---

## 14. Durable evidence

Every durable capability-bearing record includes the schema ID.

Replay verification must reject:

- capability record without schema;
- schema substitution;
- unsupported schema;
- active unknown bit;
- inconsistent schema within one grant/action/commit chain;
- cross-schema descendant history without verified migration evidence.

A durable byte sequence cannot be promoted to authority merely because the current software can parse it.

---

## 15. Golden vectors

The schema family requires golden vectors that freeze:

- canonical schema encoding;
- schema digest;
- bit width;
- bit/discriminant assignments;
- reserved/retired bits;
- representative capability-set encodings;
- malformed/unknown cases.

A change that alters a golden vector either:

- is a bug in the implementation and must be fixed; or
- is a real schema change requiring a new schema identity and governance record.

---

## 16. API/type boundary

Production code SHOULD make category errors difficult.

Conceptually distinct types:

```text
CapabilitySchemaId
RawCapabilityBits
BoundCapabilitySet
VerifiedCapabilityTranslation
```

The API should not expose a trusted constructor that turns arbitrary `(schema, bits)` into positive authority without validation.

Verified translation capabilities should be opaque/non-deserializable process capabilities; durable storage retains raw migration evidence and re-verifies on replay.

---

## 17. Capability schema registry

A production deployment may maintain a registry of recognized schema definitions.

The registry itself is authority-relevant policy evidence and therefore must be:

- versioned/digested;
- authenticated;
- lifecycle/revocation aware;
- bound to admitted release/runtime identity;
- governed under Class A change control.

Adding a schema to a registry is not equivalent to approving translation from every existing schema.

---

## 18. Failure semantics

New positive authority is denied when:

- schema ID missing;
- schema unknown;
- schema digest mismatch;
- bit width mismatch;
- reserved/unknown bit set;
- grant/request/parent schema mismatch;
- authorization/commit schema mismatch;
- stale/deprecated schema disallowed by policy;
- translation missing;
- translation unverified;
- translation would widen semantics;
- runtime schema differs from admitted schema.

Denial/freeze is not itself destructive action.

---

## 19. Formal invariants

Future formal models should target at least:

```text
SameBitsDifferentSchemaNotEquivalent
CrossSchemaAuthorityRequiresVerifiedTranslation
TranslationNeverWidensSemanticAuthority
DescendantSemanticAuthorityNeverExceedsAncestor
UnknownSchemaCannotAuthorize
ReservedBitCannotAuthorize
RecoveryCannotReinterpretOldCapabilitiesImplicitly
RuntimeSchemaMustMatchAdmission
```

---

## 20. Production status

This document does not change the current reference `u64` implementation by itself.

It specifies the production semantic contract required before that representation can become production authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.