# RSK Schema Registry Reference State Verifier v0.1

**Status:** normative reference-verifier contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

This contract defines the executable reference state machine implementing the non-cryptographic verification semantics from `RSK_VERIFIED_SCHEMA_REGISTRY_V0_1`.

It contains no physical replication mechanism, manufacturing recipe, biological/molecular design, autonomous fabrication path, or physical resource model.

---

## 1. Purpose

The verified-schema-registry contract separates two questions:

```text
Did an authorized cryptographic/trust boundary authenticate this evidence?
```

and:

```text
Does this authenticated evidence satisfy RSK registry semantics now?
```

This reference verifier implements the second question only.

Its inputs include signer metadata that a separate cryptographic/trust verifier has already authenticated. The reference code does not parse signatures, verify public-key cryptography, establish trust roots, or make a `signature_verified=true` boolean authoritative.

The governing boundary is therefore:

```text
external authenticated signer/trust evidence
    + canonical registry snapshot
    + external registry policy
    + trusted interval
    + durable anti-rollback state
    -> RSK semantic registry evaluation
```

---

## 2. Reference implementation

The executable reference is:

```text
scripts/rsk_schema_registry.py
```

Its adversarial self-test is:

```text
scripts/test_rsk_schema_registry.py
```

The committed cross-language/reference corpus is:

```text
docs/architecture/replicator-safety/golden/
  RSK_SCHEMA_REGISTRY_GOLDEN_V0_1.json
```

The reference module is Class A-protected and executed by the focused RSK governance workflow.

Python types in this module are **not** production opaque authority capabilities. They model type-state semantics only.

---

## 3. Canonical snapshot profile

The v0.1 reference snapshot has exact fields:

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

The schema tag is:

```text
symthaea.rsk.schema-registry-snapshot.v1
```

Snapshot identity is domain-separated:

```text
SHA256(
  "symthaea.rsk.schema-registry-snapshot.v1\0"
  || canonical_json(snapshot)
)
```

A claimed digest never overrides recomputation.

---

## 4. Embedded canonical schema bytes

Each registry entry embeds exact canonical UTF-8 JSON bytes through `canonical_schema_json`.

The verifier:

1. parses the embedded JSON;
2. independently canonicalizes the parsed object;
3. requires the embedded bytes to equal that canonical form exactly;
4. validates the schema class;
5. recomputes its schema ID;
6. requires the recomputed ID to equal the registry entry claim;
7. requires family and version metadata to agree with the canonical schema.

This prevents registry metadata from redefining schema identity.

For an active resource entry under this current positive-provenance profile, the schema must be resource v0.2 or later according to the current reference profile and therefore commit runtime numeric dimension IDs. Historical v0.1 resource schemas may remain historical evidence, but they do not resolve as current positive semantic provenance.

---

## 5. Registry entry key and ordering

The exact key is:

```text
(schema_kind, family, version)
```

Entries are canonical only when strictly sorted by that key.

Duplicate keys fail closed.

Schema kinds currently recognized by the reference profile are:

- `capability`;
- `resource`.

The full snapshot retains all previously known schema keys when the governing policy enables `require_known_entries_retained`.

Silent deletion is not lifecycle transition evidence.

---

## 6. Lifecycle monotonicity

The reference lifecycle order is:

```text
active < superseded < revoked < tombstoned
```

A known schema key may move only toward an equally or more restrictive state.

It cannot move from:

- superseded back to active;
- revoked back to superseded/active;
- tombstoned back to any positive state.

The exact canonical schema identity bound to an existing `(kind, family, version)` key is immutable. A semantic change requires a new version/key and new schema identity.

Only `active` entries resolve through positive exact resolution.

---

## 7. Version transition and supersession

When one family advances to a higher active version under a policy requiring explicit supersession:

```text
new_active.supersedes
    contains
prior_active.schema_id
```

Active-version rollback is denied.

The current reference policy may require at most one active version per `(schema_kind, family)`.

A new active registry version does not translate grants, resource budgets, capability sets, or other authority from the old schema. Cross-schema authority remains a separate conservative-transition problem.

---

## 8. Authenticated signer evidence boundary

The reference signer record contains:

```text
AuthenticatedSignerEvidence {
    signer_id,
    key_id,
    role,
    failure_domain,
    signature_profile,
    valid_from,
    valid_until,
    lifecycle,
    bound_snapshot_digest,
}
```

There is intentionally no raw signature and no trusted boolean saying a signature is valid.

A production implementation must receive an opaque authenticated result from the real cryptographic/trust subsystem. The reference dataclass cannot provide that security property in Python.

The semantic verifier then requires each supplied counted record to:

- bind the exact recomputed snapshot digest;
- use a policy-permitted role;
- use a policy-permitted signature profile;
- be active;
- be valid for the full trusted evaluation interval;
- include trusted signer identity and failure-domain metadata.

Multiple keys for one signer identity count once at most.

Distinct signer identities are not sufficient when policy also requires independent trusted failure domains.

Key-to-signer ownership consistency remains a responsibility of the external authenticated trust snapshot in this reference tranche; production cryptographic integration must not permit one key to authenticate multiple distinct signer identities.

---

## 9. Policy is external to evidence

The snapshot carries an exact `registry_policy_id`, but it does not choose its verification policy.

The caller supplies the externally governed policy object. The verifier:

- validates policy shape;
- canonically hashes it;
- requires the snapshot policy ID to equal that hash;
- applies the external policy limits and quorum rules.

Therefore a snapshot cannot lower:

- signer identity quorum;
- failure-domain quorum;
- allowed roles/profiles;
- byte/count limits;
- canonical-encoding allowlists;
- freshness bounds;
- lifecycle/version-transition rules.

Policy changes are not silently accepted inside one anti-rollback tracker epoch. A different policy digest requires an explicit higher-level epoch/recovery transition not implemented by this reference tranche.

---

## 10. Trusted-time/freshness semantics

The reference verifier consumes a trusted interval:

```text
TrustedInterval { start, end }
```

It requires the entire interval to fit within:

```text
snapshot.issued_at <= start <= end <= snapshot.expires_at
```

Signer/key validity must also cover the entire trusted interval.

An expired last-known snapshot is denied even when it is an exact replay of previously accepted evidence.

Exact replay never refreshes expiry.

This module does not establish trusted time. Authentication, monotonicity and uncertainty of the trusted interval belong to the RSK trusted-time boundary.

---

## 11. Anti-rollback state

The reference durable state models:

```text
AntiRollbackState {
    registry_id,
    highest_sequence,
    accepted_digest,
    latest_issued_at,
    policy_digest,
    forked,
    known_entries,
}
```

For genesis:

```text
sequence == 1
previous_snapshot_digest == 0^64
```

when predecessor chaining is enabled.

For later snapshots:

```text
sequence < highest_sequence
    -> DENY

sequence == highest_sequence && digest == accepted_digest
    -> exact REPLAY (freshness still required)

sequence == highest_sequence && digest != accepted_digest
    -> FORKED / NON-OPERATIONAL

previous_snapshot_digest != accepted_digest
    -> FORKED / NON-OPERATIONAL

issued_at < latest_issued_at
    -> DENY
```

Fork state is sticky in ordinary evaluation. There is no ordinary un-fork API.

---

## 12. Full-snapshot retention semantics

This reference profile treats an accepted snapshot as a complete registry view when `require_known_entries_retained` is enabled.

A previously known exact schema key cannot simply disappear.

Removal requires explicit retained lifecycle evidence such as revocation/tombstoning rather than omission.

This property prevents deletion from laundering historical or negative semantic state.

---

## 13. Exact replay

Exact replay is useful for evidence reconstruction after process restart, but does not create freshness.

A replay must independently satisfy the current:

- policy;
- trusted interval;
- authenticated signer evidence;
- exact canonical snapshot digest;
- schema integrity predicates.

The returned process-local reference type-state is reconstructed only after those predicates pass.

---

## 14. Process-local verified reference type

Successful evaluation can produce:

```text
VerifiedSchemaRegistrySnapshotReference
```

This type provides exact active-schema resolution.

It is explicitly a **reference type-state object**, not production cryptographic opacity. Its constructor has a module-local marker only to prevent accidental ordinary construction in reference code.

Production must replace this with a genuinely opaque type rooted in verified cryptographic/trust evidence.

Serialization of `verified=true` or this reference object is not restart authority.

---

## 15. Exact resolution

Resolution requires exact:

```text
schema_kind
family
version
expected_schema_id
```

and exactly one matching active entry.

Missing, ambiguous, superseded, revoked, tombstoned, or identity-mismatched entries deny resolution.

Resolution returns canonical schema content only. It does not mint replication grants, reset budgets, satisfy replication quorum, clear quarantine, extend authorization, or translate authority.

---

## 16. Golden reference vector

The committed genesis corpus uses:

```text
registry_id = rsk.test.registry
sequence = 1
issued_at = 1000
expires_at = 2000
```

with two abstract signer identities in two abstract failure domains.

The policy identity is:

```text
433f3749986653fd2c18d8626f2a758d22ee77e35d50629dea394a9459acd968
```

The domain-separated genesis snapshot digest is:

```text
0186c4f99015dc5bf449dd57c392c306a77ebe539b06aa0ce3f6698fa8f7d653
```

These are test-only identities. They are not production registry, signer, resource, or authority definitions.

---

## 17. SRV coverage in this tranche

### Directly implemented/reference-tested

The v0.1 reference state verifier is designed to exercise the non-cryptographic semantics of:

- SRV-001–005 canonical identity/key/resolution;
- SRV-010–014 resource numeric-ID/schema commitment through the semantic-schema layer;
- SRV-027 duplicate signer identity;
- SRV-028 failure-domain collapse;
- SRV-029 required failure-domain metadata presence;
- SRV-030–034 external policy identity/limits/encoding checks;
- SRV-040–046 rollback/collision/freeze/trusted-interval semantics;
- SRV-050–054 schema lifecycle/version rollback;
- SRV-060–062 restart/type-state/anti-rollback semantics at the reference level;
- SRV-070–074 authority non-amplification by interface separation;
- SRV-080–083 composition with separately validated semantic types.

Not every identifier above has a one-test-per-ID implementation yet; the adversarial suite groups some related predicates. Executed evidence is still required.

### Deliberately delegated to cryptographic/trust integration

This reference tranche does **not** implement or claim:

- SRV-020 raw invalid-signature verification;
- SRV-021 unknown-signer trust-root lookup;
- SRV-022–025 cryptographic key lifecycle/revocation proof acquisition;
- SRV-026 cryptographic key-usage proof acquisition.

It consumes already-authenticated evidence containing the resulting trusted metadata and then applies policy to it.

### Still unresolved beyond this reference state machine

SRV-063 durable anti-rollback-storage rollback detection requires a monotonic/external anchor. A caller can always hand pure reference code an older internally valid `AntiRollbackState`; this module cannot distinguish that from legitimate state without trusted durable continuity evidence.

Production admission therefore still requires Xenia/OS/hardware/durable-store evidence proving the tracker itself was not rolled back.

---

## 18. Adversarial properties

The reference suite targets at least:

1. exact committed genesis acceptance;
2. exact active resolution;
3. duplicate signer identity cannot inflate quorum;
4. one failure domain cannot satisfy independent-domain quorum;
5. stale snapshots cannot gain time through replay;
6. claimed snapshot digest cannot override recomputation;
7. schema ID cannot override exact embedded bytes;
8. non-canonical embedded schema bytes are denied;
9. duplicate schema keys are denied;
10. active historical resource v0.1 is denied for current positive provenance;
11. exact replay is idempotent;
12. sequence rollback is denied;
13. same-sequence collision creates sticky fork state;
14. predecessor mismatch creates sticky fork state;
15. issuance regression is denied;
16. bytes cannot mutate under an existing family/version key;
17. explicit higher-version supersession is required and accepted;
18. lifecycle cannot move more permissively;
19. previously known keys cannot silently disappear.

Future property/fuzz work should generate arbitrary valid/malformed histories rather than only example snapshots.

---

## 19. Non-amplification

A successful registry decision proves only semantic provenance under the reference policy/state model.

It cannot:

- create `ReplicationGrant`;
- create `AuthorizedReplication`;
- satisfy replication approval quorum;
- alter lineage/resource counters;
- clear quarantine/revocation/fork state;
- extend authorization expiry;
- reset zero/exhausted resource authority;
- authorize cross-schema migration;
- prove physical safety.

The registry has no API for those effects.

---

## 20. Promotion blockers

Before production promotion, at minimum:

- real signature verification and verified trust-snapshot integration;
- signer/key identity ownership enforcement under the trusted snapshot;
- trusted-time integration;
- monotonic durable anti-rollback anchoring;
- Rust/independent implementation parity;
- malformed-input property/fuzz campaigns;
- bounded parser/allocation review;
- exact-head CI/qualification evidence;
- #1926 controlled Cargo.lock qualification;
- #1682 exact admitted artifact/runtime identity;
- broader RSK production-admission gates.

---

## 21. Production admission status

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```
