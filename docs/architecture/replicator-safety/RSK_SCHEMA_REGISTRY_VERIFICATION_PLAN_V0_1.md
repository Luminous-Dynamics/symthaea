# RSK Verified Schema Registry — Adversarial Verification Plan v0.1

**Status:** Test-plan contract; not executed evidence  
**Change Class:** A

This plan defines negative and convergence tests for the future verified semantic-schema registry. It contains no physical replication mechanism or physical resource recipe.

## Test-family identifiers

### Canonical identity

- **SRV-001 — canonical stability:** equivalent canonical snapshot construction produces identical bytes/digest.
- **SRV-002 — snapshot tamper:** any authority-relevant byte change changes the registry snapshot digest.
- **SRV-003 — schema digest mismatch:** claimed schema ID differing from recomputed canonical schema bytes is denied.
- **SRV-004 — duplicate schema key:** duplicate `(kind, family, version)` entries are denied.
- **SRV-005 — ambiguous lookup:** multiple current candidates for one exact schema key deny resolution.

### Resource runtime-ID commitment

- **SRV-010 — numeric-ID sensitivity:** changing only one runtime dimension numeric ID changes `ResourceAccountingSchemeId`.
- **SRV-011 — numeric-ID collision:** duplicate runtime numeric IDs are denied.
- **SRV-012 — semantic-ID collision:** duplicate canonical semantic dimension IDs are denied.
- **SRV-013 — detached map substitution:** a registry-side numeric mapping not committed by the resource schema ID cannot support production resolution.
- **SRV-014 — v0.1 reference isolation:** the existing test-only golden digest is never promoted as proof of numeric-ID commitment.

### Signatures and trust lifecycle

- **SRV-020 — invalid signature:** invalid cryptographic signature is denied.
- **SRV-021 — unknown signer:** signer absent from the verified trust snapshot is denied.
- **SRV-022 — not-yet-valid key:** denied.
- **SRV-023 — expired key:** denied.
- **SRV-024 — retired key:** denied.
- **SRV-025 — revoked key:** denied.
- **SRV-026 — wrong key usage/role:** denied.
- **SRV-027 — duplicate signer identity:** multiple signatures/keys from one signer identity count once at most.
- **SRV-028 — failure-domain collapse:** multiple signers in one required failure domain do not satisfy an independent-domain quorum.
- **SRV-029 — missing failure-domain metadata:** denied when policy requires that domain.

### Registry policy

- **SRV-030 — requester-chosen policy:** registry evidence cannot lower its own verification policy.
- **SRV-031 — wrong registry ID:** denied.
- **SRV-032 — unsupported canonical encoding:** denied.
- **SRV-033 — oversized snapshot/schema/signature:** bounded before expensive processing where feasible.
- **SRV-034 — policy digest mismatch:** denied.

### Rollback, fork, and freeze

- **SRV-040 — sequence rollback:** lower sequence than durable accepted state is denied.
- **SRV-041 — same-sequence collision:** same sequence with different digest becomes fork/collision and non-operational.
- **SRV-042 — issuance regression:** regressed `issued_at` is denied.
- **SRV-043 — chain mismatch:** configured previous-snapshot digest mismatch becomes fork/non-operational.
- **SRV-044 — exact replay:** exact accepted snapshot may reconstruct evidence but cannot refresh expiry.
- **SRV-045 — freeze attack:** expired/stale last-known snapshot cannot authorize new positive decisions.
- **SRV-046 — trusted-time ambiguity:** an uncertainty interval that cannot prove freshness denies.

### Schema lifecycle

- **SRV-050 — active schema:** exact active entry resolves when all verification predicates hold.
- **SRV-051 — superseded schema:** may remain auditable but cannot support new positive authority when policy disallows it.
- **SRV-052 — revoked schema:** denied regardless of prior validity.
- **SRV-053 — tombstoned schema:** permanently non-eligible for new positive authority under the governing policy.
- **SRV-054 — version rollback:** policy-forbidden family/version rollback is denied even if signatures are otherwise valid.

### Restart and durable evidence

- **SRV-060 — no serialized Verified bypass:** serialized raw evidence cannot deserialize directly into `VerifiedSchemaRegistrySnapshot`.
- **SRV-061 — restart reverification:** restart reconstructs verified capability only by re-running cryptographic/lifecycle/policy/freshness checks.
- **SRV-062 — durable anti-rollback state:** accepted sequence/digest continuity survives restart.
- **SRV-063 — rollback of durable tracker:** detected storage rollback/fork becomes non-operational rather than selecting a favorable older state.

### Authority non-amplification

- **SRV-070 — registry cannot mint grant:** verified schema provenance alone cannot produce a replication grant.
- **SRV-071 — registry cannot satisfy replication quorum:** schema signers are not automatically replication grant approvers.
- **SRV-072 — registry cannot clear negative state:** quarantine/revocation remains dominant.
- **SRV-073 — registry cannot extend authorization:** schema freshness cannot extend an already expired grant/token.
- **SRV-074 — registry cannot translate authority:** schema-version transition requires separately verified conservative translation evidence.

### Structural-validation composition

- **SRV-080 — verified schema + illegal capability bits:** reserved/retired/unknown bits still fail structural validation.
- **SRV-081 — verified scheme + malformed resource vector:** unknown/missing/range-invalid dimensions still fail structural validation.
- **SRV-082 — structurally valid value + unverified registry:** cannot enter production positive-authority path.
- **SRV-083 — exact provenance + exact structural validation:** only the conjunction may produce the semantic capability consumed by a future grant evaluator.

## Property/fuzz targets

Fuzz/property campaigns should cover:

- canonical ordering permutations;
- duplicate keys/entries/signatures;
- boundary lengths/counts;
- sequence and timestamp boundaries;
- schema-family/version collisions;
- random bit flips in canonical bytes/digests;
- random signer lifecycle permutations;
- domain-metadata omission/collision;
- dimension-name/numeric-ID remappings;
- malformed resource dimension sets;
- replay/fork histories.

## Evidence classification

Authored tests are design evidence only.

Promotable evidence requires exact-head execution under the pinned build environment, retained logs/receipts, exact source/lock/toolchain identity, and the broader RSK production-admission requirements.
