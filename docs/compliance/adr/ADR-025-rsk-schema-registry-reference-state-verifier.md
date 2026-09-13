# ADR-025: RSK schema registry reference state verifier

**Status**: Proposed

**Change Class**: A

**Date**: 2026-09-13

## Context

ADR-017 / #1980 froze the verified semantic-schema registry contract but intentionally stopped before implementing a verifier. The RSK semantic stack now has stronger schema identity, deterministic validator derivation, runtime representation/arithmetic profiles, and a compact semantic execution profile. The remaining provenance layer still needs executable state semantics before production cryptography is integrated.

Implementing ad-hoc signatures in Python would be the wrong next step. Cryptographic signature verification, trust-root verification, signer/key ownership and trusted-time provenance belong at the Xenia/trust boundary and must eventually yield opaque authenticated evidence.

However, many registry safety properties are independent of the cryptographic primitive itself:

- exact canonical schema-ID recomputation;
- registry policy identity;
- signer identity and failure-domain counting;
- freshness;
- sequence rollback;
- same-sequence collision;
- predecessor-chain fork detection;
- issuance regression;
- schema lifecycle monotonicity;
- exact active resolution;
- full-snapshot retention;
- explicit version supersession;
- restart reconstruction semantics;
- authority non-amplification.

This ADR contains no physical replication mechanism, fabrication recipe, biological/molecular design, physical resource model, or autonomous manufacturing path.

## Decision

Add a Class A Python reference state verifier:

```text
scripts/rsk_schema_registry.py
```

with adversarial tests:

```text
scripts/test_rsk_schema_registry.py
```

and a shared golden corpus:

```text
docs/architecture/replicator-safety/golden/
  RSK_SCHEMA_REGISTRY_GOLDEN_V0_1.json
```

The reference verifier deliberately does **not** verify raw digital signatures or trust roots.

It consumes `AuthenticatedSignerEvidence` records representing metadata already authenticated by a separate cryptographic/trust boundary and then evaluates RSK registry semantics.

No `signature_verified=true` boolean is accepted as cryptographic authority.

## Canonical snapshot identity

The reference snapshot schema is:

```text
symthaea.rsk.schema-registry-snapshot.v1
```

Its identity is:

```text
SHA256(
  "symthaea.rsk.schema-registry-snapshot.v1\0"
  || canonical_json(snapshot)
)
```

The snapshot embeds each canonical schema as exact canonical UTF-8 JSON. The verifier reparses and independently canonicalizes those bytes, recomputes the schema ID using the existing semantic-schema profile, and requires exact family/version agreement.

## Resource positive-provenance rule

Active resource entries under this current reference production-provenance profile must use the runtime-ID-bound resource schema profile from #2035. Historical v0.1 resource schemas may remain historical evidence but cannot resolve as active positive provenance.

## Signer/quorum semantics

The reference evaluator requires counted signer evidence to bind the exact recomputed snapshot digest and satisfy externally supplied policy constraints for:

- signer identity;
- key identity metadata;
- role;
- failure domain;
- signature profile;
- active lifecycle;
- validity interval.

Multiple keys for one signer identity count once at most.

Independent-domain policy is evaluated over trusted failure-domain metadata, not signer self-assertion.

The external trust verifier remains responsible for proving actual signature validity, trust membership, revocation evidence and key-to-signer ownership consistency.

## Policy separation

Registry evidence cannot choose a weaker policy for itself.

The evaluator receives policy separately, canonically hashes it, and requires the snapshot's exact `registry_policy_id` to match.

Policy changes are denied inside one ordinary anti-rollback epoch. A policy transition requires a separately governed epoch/recovery mechanism rather than implicit acceptance.

## Freshness

The evaluator receives a trusted interval rather than calling the wall clock.

The entire interval must lie inside the snapshot validity window and inside the counted signer/key validity windows.

Exact replay still requires current freshness and does not refresh expiration.

Trusted-time provenance is not implemented here.

## Anti-rollback and fork state

The reference `AntiRollbackState` binds:

```text
registry_id
highest_sequence
accepted_digest
latest_issued_at
policy_digest
forked
known_entries
```

The evaluator enforces:

```text
lower sequence -> DENY
same sequence + same digest -> REPLAY
same sequence + different digest -> FORKED
previous digest mismatch -> FORKED
issued_at regression -> DENY
```

Forked state is sticky. There is no ordinary un-fork API.

Genesis requires sequence 1 and, when chaining is enabled, the all-zero predecessor digest.

## Full-snapshot and lifecycle semantics

Under the current policy profile, previously known schema keys must remain represented in later full snapshots.

The exact canonical schema identity attached to one `(kind, family, version)` key is immutable.

Lifecycle may only become more restrictive:

```text
active < superseded < revoked < tombstoned
```

A higher active version must explicitly identify the prior active schema as superseded when policy requires that transition proof.

Active-version rollback is denied.

## Process-local verified reference result

Successful evaluation can produce `VerifiedSchemaRegistrySnapshotReference`, which supports exact active resolution.

This is reference type-state only. Python cannot provide the cryptographic opacity required for production. Serialization of this object or a `verified=true` report is not restart authority.

Production must replace it with an opaque capability rooted in actual verified cryptographic/trust evidence.

## Golden vector

The abstract reference policy digest is:

```text
433f3749986653fd2c18d8626f2a758d22ee77e35d50629dea394a9459acd968
```

The domain-separated abstract genesis snapshot digest is:

```text
0186c4f99015dc5bf449dd57c392c306a77ebe539b06aa0ce3f6698fa8f7d653
```

The corpus uses two abstract signer identities in two abstract failure domains. It contains no signatures or private keys.

## Adversarial evidence

The authored self-test covers or composes tests for:

- committed golden acceptance and exact resolution;
- duplicate signer identity not inflating quorum;
- failure-domain collapse;
- snapshot-binding/lifecycle/role signer failures;
- stale replay;
- claimed snapshot digest mismatch;
- embedded schema digest mismatch;
- non-canonical embedded schema bytes;
- duplicate schema key;
- active resource-v1 denial;
- exact replay;
- sequence rollback;
- same-sequence collision/fork;
- sticky fork state;
- predecessor mismatch;
- issuance regression;
- schema mutation under existing family/version key;
- valid explicit higher-version supersession;
- non-monotonic lifecycle reactivation denial;
- silent disappearance of a known schema key.

Authored tests are design evidence only until exact-head execution.

## Deliberate non-coverage

This tranche does not claim to implement SRV cryptographic verification families for raw invalid signatures, trust-root membership, revocation proof acquisition or cryptographic key usage.

It also cannot prove that a caller has not rolled back the durable `AntiRollbackState` itself. Detecting rollback of a locally self-consistent tracker requires trusted monotonic/external anchoring and remains a production blocker.

## Governance

The new registry verifier and self-test are added to:

- the generic Class A detector;
- focused RSK workflow path triggers;
- the focused RSK blocking ADR surface;
- the focused RSK governance self-test lane.

Future Python-only modifications therefore cannot bypass RSK safety governance.

## Consequences

Positive:

- the #1980 contract now has executable non-cryptographic state semantics;
- anti-rollback/fork behavior becomes testable before Rust/crypto integration;
- canonical schema provenance composition is explicit;
- registry authority non-amplification is reinforced by interface separation;
- cryptographic and semantic verification responsibilities are kept distinct;
- future Rust/Xenia implementations receive a concrete adversarial oracle.

Costs:

- Python reference code is larger and must remain clearly non-production;
- an external signature/trust verifier is still required;
- durable anti-rollback anchoring is still required;
- independent Rust implementation/parity is still required;
- queued CI means this tranche is not yet executed evidence.

## Related work

- #1668 — verified positive evidence
- #1669 — trusted time
- #1678 — capability semantic integrity
- #1679 — resource accounting integrity
- #1682 — exact build/runtime identity
- #1926 — controlled Cargo.lock qualification blocker
- #1969 / #1980 — verified semantic schema registry
- #2035 — resource runtime numeric-ID binding
- #2057/#2064 — deterministic schema-derived validator / golden corpus
- #2143 — semantic execution profile
- #2164 — zero-representable resource exhaustion

## Production admission

```text
DENIED / NOT YET ELIGIBLE
```
