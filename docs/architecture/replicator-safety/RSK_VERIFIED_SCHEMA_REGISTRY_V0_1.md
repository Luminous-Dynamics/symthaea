# RSK Verified Semantic Schema Registry v0.1

**Status:** Reference verification contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## 1. Purpose

RSK now distinguishes numeric values, schema-bound values, and structurally schema-validated values. A further provenance predicate is required before those meanings may participate in production authority:

```text
resolved schema != verified schema provenance
```

This contract defines the evidence boundary for deciding that exact canonical schema bytes are the currently trusted meaning for one RSK schema family/version.

It contains no physical replication mechanism, manufacturing recipe, biological/molecular design, autonomous fabrication path, or physical resource model.

## 2. Separation of concerns

The registry has one job:

> authenticate and freshness-bind semantic definitions.

It does **not**:

- grant replication authority;
- satisfy replication quorum by itself;
- clear quarantine or revocation;
- extend an authorization lifetime;
- widen capability/resource ceilings;
- approve a physical process;
- prove cross-schema semantic equivalence.

Registry verification is necessary semantic provenance, never sufficient positive replication authority.

## 3. Evidence layers

Conceptually:

```text
canonical schema bytes
      -> schema identity
      -> signed registry snapshot
      + verified trust snapshot
      + registry verification policy
      + trusted time/continuity evidence
      -> verify
      -> VerifiedSchemaRegistrySnapshot
          -> resolve exact current schema
          -> structural schema validation
          -> opaque Validated* semantic value
```

`VerifiedSchemaRegistrySnapshot` is a process capability, not a durable wire type. Persistent storage retains raw/signed evidence. Restart requires verification again.

## 4. Canonical registry snapshot

A v0.1 registry snapshot conceptually binds:

```text
SchemaRegistrySnapshot {
    registry_schema_version,
    registry_id,
    sequence,
    issued_at,
    expires_at,
    previous_snapshot_digest,
    registry_policy_id,
    entries,
}
```

Each field is authority-relevant evidence and is included in canonical snapshot bytes.

The snapshot has a domain-separated digest over exact canonical bytes. Canonicalization rules must be versioned and deterministic.

Limits on total snapshot bytes, entry count, identifier lengths, schema bytes, and signature-envelope sizes are part of the verification policy and fail closed when exceeded.

## 5. Registry entries

Each entry conceptually binds:

```text
SchemaRegistryEntry {
    schema_kind,
    family,
    version,
    canonical_encoding_version,
    canonical_schema_bytes,
    schema_id,
    lifecycle_state,
    supersedes,
}
```

`schema_kind` distinguishes capability vocabularies from resource-accounting schemes and any future schema classes.

The verifier independently recomputes `schema_id` from the exact canonical schema bytes according to that schema class's frozen identity profile. A claimed ID never overrides recomputation.

Duplicate `(schema_kind, family, version)` entries fail closed. Ambiguous resolution fails closed.

## 6. Resource numeric dimension identity

For a resource-accounting scheme, the exact runtime `ResourceDimensionId` mapping is authority semantics.

Production-admissible resource schema bytes MUST commit the one-to-one mapping between canonical semantic dimension identity and runtime numeric ID.

Registry metadata outside the canonical resource-schema bytes cannot repair this requirement. A registry may index or display the mapping, but `ResourceAccountingSchemeId` itself must change when a runtime numeric ID changes.

The current v0.1 golden corpus is reference/test material and must not be reinterpreted as binding a numeric mapping that its bytes did not encode. See the dedicated dimension-ID binding contract and #1973.

## 7. Signed envelope

The raw durable form is conceptually:

```text
SignedSchemaRegistrySnapshot {
    snapshot,
    snapshot_digest,
    signatures,
}
```

Signatures cover a domain-separated message binding the registry protocol/version and exact snapshot digest/canonical bytes.

The envelope itself is not authority. Cryptographic validity is only one input to registry verification.

## 8. Trust and signer lifecycle

Verification consumes a separately verified/current trust snapshot.

For every counted signer, verification establishes at least:

- signer identity;
- key identity;
- signature algorithm/profile;
- key validity window;
- active lifecycle state;
- required schema-registry usage/role;
- revocation state;
- trusted failure-domain metadata.

One signer identity with multiple keys remains one signer identity. Multiple keys or algorithms do not manufacture independence.

## 9. Failure-domain independence

When registry policy requires independent approval, independence is derived from trusted metadata, not signer self-assertion.

The verifier evaluates policy over distinct required failure domains. Examples may include separately administered trust roots or organizational/operator domains, but the exact domain vocabulary is deployment policy.

Missing required failure-domain metadata fails closed.

## 10. Registry policy

Registry verification policy is distinct from evidence.

Policy determines at least:

- permitted signature algorithms/profiles;
- minimum verified signer identities;
- required failure-domain diversity;
- permitted signer roles/usages;
- maximum snapshot/schema/signature sizes;
- freshness requirements;
- allowed registry ID;
- allowed canonical encoding versions;
- schema-family/version monotonicity rules;
- whether supersession is required for a version transition.

The snapshot cannot choose a weaker policy for itself.

## 11. Freshness and time

A registry snapshot is current only when trusted time/continuity evidence proves the entire required interval lies inside its validity window.

Authentication of a timestamp source is not sufficient time assurance. RSK's trusted-time contract remains controlling.

Ambiguous, unavailable, stale, or regressed time removes eligibility; it never extends schema validity.

## 12. Sequence and anti-rollback tracker

A verifier maintains durable anti-rollback state for each registry ID.

At minimum it tracks:

- highest accepted sequence;
- accepted digest at that sequence;
- latest accepted issuance time;
- previous-snapshot digest / chain continuity where configured;
- current accepted registry policy identity.

Required failure behavior:

```text
proposed.sequence < latest.sequence -> DENY
same sequence + different digest -> FORK/COLLISION -> NON-OPERATIONAL
issued_at regression -> DENY
previous-digest mismatch -> FORK -> NON-OPERATIONAL
stale snapshot -> DENY
```

Replaying the exact already accepted snapshot may be idempotent for evidence reconstruction but does not refresh its expiry.

## 13. Freeze resistance

An attacker who withholds newer registry state must not gain indefinite authority from the last known snapshot.

Expiry/freshness is therefore mandatory for production profiles. When freshness cannot be proven, new positive authority that depends on registry semantics freezes/denies.

Existing negative safety state remains effective.

## 14. Schema lifecycle

A schema entry has an explicit lifecycle such as:

- `Active`;
- `Superseded`;
- `Revoked`;
- `Tombstoned` / permanently non-eligible where policy requires.

Only policy-eligible active entries may support new positive authority.

Historical evidence may continue to reference an old/superseded schema for audit/replay interpretation, but historical validity does not make it current authority.

Revocation or tombstoning dominates prior positive semantic provenance.

## 15. Version transition

A new schema version always has a distinct exact schema identity when canonical bytes change.

A newer registry snapshot may establish that a new schema is current, but it does not translate old capability/resource authority into the new schema.

Cross-schema authority requires a separately verified conservative translation object governed by the capability/resource transition contracts.

No translation evidence -> no cross-schema positive authority.

## 16. Opaque verified capability

Conceptually:

```text
VerifiedSchemaRegistrySnapshot {
    verified snapshot,
    snapshot_digest,
    verified signer identities,
    verified failure domains,
    trust_snapshot_digest,
    registry_policy_digest,
    evaluation_time/interval,
    anti_rollback_tracker_state,
}
```

Fields that would let callers forge this capability are private.

The verified type is not deserialized from persistent storage. Raw evidence is reverified after restart.

A verification report may be serialized for audit, but `report.trusted == true` is not itself an authority capability.

## 17. Exact resolution

Resolution from a verified snapshot requires exact lookup by the governing schema key and expected identity.

Conceptually:

```text
resolve_capability(family, version, expected_schema_id)
resolve_resource_scheme(family, version, expected_scheme_id)
```

A mismatch between lookup metadata, recomputed bytes digest, expected identity, lifecycle state, or active registry state denies.

Human-readable names never substitute for exact IDs.

## 18. Composition with structural validation

Verified registry resolution and structural value validation are separate predicates.

A production path conceptually requires both:

```text
VerifiedSchemaDefinition
    + BoundCapabilitySet
        -> structural validate
        -> ValidatedCapabilitySet
```

and equivalently for resource vectors.

Whether implementation composes resolution and validation into one API is an internal design choice; neither predicate may be skipped.

## 19. Negative-state precedence

Registry verification cannot override:

- subject/lineage quarantine;
- subject/lineage/grant revocation;
- containment failure;
- stale monitoring;
- budget exhaustion;
- expired authorization;
- forked ledger/recovery state;
- runtime identity mismatch.

A newly verified schema is semantic evidence only.

## 20. Recovery

Registry fork/collision recovery follows the separate governed recovery ceremony.

Ordinary schema signers cannot resolve a registry fork merely by publishing another snapshot unless policy explicitly defines and verifies a higher-authority recovery transition.

Fresh-epoch recovery does not carry active replication grants or authorization tokens across the boundary.

## 21. Required evidence before implementation promotion

Production implementation requires at least:

- frozen canonical registry encoding;
- golden vectors;
- independent canonicalization/digest implementation;
- adversarial signature/lifecycle tests;
- rollback/collision/freeze tests;
- failure-domain collapse tests;
- restart/reverification tests;
- schema lifecycle/supersession tests;
- resource numeric-ID commitment tests;
- property/fuzz tests over malformed envelopes and duplicate/ambiguous entries;
- exact build/runtime identity evidence;
- Class A review.

## 22. Non-claims

This contract does not prove a schema scientifically correct, physically safe, or sufficient for replication authority. It authenticates which exact semantic definition is current under a declared trust/policy process.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
