# Replicator Safety Kernel — Trust Snapshot and Failure-Domain Contract v0.1

Status: **normative design contract; NOT an implemented production trust store**

This document defines the trust/lifecycle/failure-domain semantics used by the RSK verified-positive-evidence boundary.

It contains no physical replication mechanism.

---

## 1. Purpose

A valid signature proves only that some key produced a signature accepted by a cryptographic verifier.

RSK positive authority additionally needs to know:

- which identity/key is trusted;
- for which role/usage;
- during which validity interval;
- under which trust epoch/snapshot;
- whether the key/identity is active, retired, revoked, or otherwise non-authorizing;
- whether the signer is controlled by the requesting subject;
- which practical failure domains the signer belongs to;
- whether the trust state is current and not rolled back.

The trust snapshot is the bounded, digest-addressed answer to those questions.

---

## 2. Snapshot identity and monotonicity

Conceptual identity:

```text
TrustSnapshot {
    schema_id,
    trust_epoch_id,
    sequence,
    issued_at,
    expires_at,
    signer_records,
    registry_versions,
    policy_root_digest,
    previous_snapshot_digest?,
}
```

Normative rules:

- `sequence > 0`;
- `issued_at < expires_at`;
- signer records are canonical sorted/unique;
- unknown critical schema/registry versions reject;
- all fields are bounded before allocation;
- snapshot digest is deterministic/domain-separated;
- a verified capability retains the snapshot digest/epoch/sequence it used.

### 2.1 Tracker rules

For one `trust_epoch_id`, accepting a new snapshot requires:

```text
new.sequence > accepted.sequence
new.issued_at >= accepted.issued_at
```

unless a separately governed recovery/epoch rule explicitly defines another transition.

Reject/freeze on:

- sequence rollback;
- same sequence with different digest;
- predecessor mismatch where predecessor chaining is enabled;
- issued-at regression prohibited by policy;
- expired snapshot;
- unknown registry/schema;
- unverifiable rotation/supersession.

A process restart must not reset the accepted sequence/head without durable anti-rollback evidence.

---

## 3. Signer identity versus key identity

RSK distinguishes:

```text
SignerIdentityId != KeyId
```

One signer identity may rotate among multiple keys over time. Multiple keys owned by one identity do not count as multiple independent signers.

A signer record binds:

```text
SignerIdentityId
KeyId
SignatureProfileId
not_before
not_after?
lifecycle_state
roles/usages
failure_domain_refs
subject_control_refs
metadata/profile digest as needed
```

The canonical key identity is `(SignatureProfileId, KeyId)` or another frozen equivalent. The signer identity is a separate stable identifier governed by trust policy.

---

## 4. Lifecycle states

Minimum lifecycle:

- `Active` — eligible subject to time/role/policy checks;
- `Retired` — not eligible to create new positive authority;
- `Revoked` — not eligible; prior capabilities may also be invalidated according to revocation policy.

A deployment may define additional states such as:

- `Suspended`;
- `Compromised`;
- `PendingActivation`.

Unknown lifecycle values never map to `Active` by default.

### 4.1 Historical signatures

Historical cryptographic validity and current authority are separate facts.

A signature may remain historically authentic while current policy says its key is revoked and therefore cannot support new positive authority.

Durable audit may preserve the historical signature; current authorization still observes present revocation/supersession rules.

---

## 5. Role/usage separation

Example role registry:

| Role | Positive/negative nature | Notes |
|---|---|---|
| ReplicationGrantIssuer | positive | may issue bounded grants only |
| ReplicationApprover | positive | participates in independent quorum |
| RuntimeMonitor | primarily negative | health/containment/freshness evidence; not a general grant mint |
| RiskPolicyAuthority | policy | signs/authorizes policy snapshots |
| QuarantineAuthority | negative | may add quarantine according to scope |
| RevocationAuthority | negative | may revoke according to scope |
| RecoveryAuthority | recovery | distinct high-trust role |
| TrustRotationAuthority | trust | updates trust/key state |
| TransparencyWitness | audit | receipt/witness evidence, non-authoritative for `Allow` |
| BuildAdmissionAuthority | admission | source/artifact/runtime promotion evidence |

Role separation is enforced from the trusted snapshot/policy. A key's own signed payload cannot grant itself a new role.

---

## 6. Failure-domain representation

### 6.1 Trusted references

Use canonical references such as:

```text
FailureDomainRef {
    kind_id: FailureDomainKindId,
    domain_id: Digest32,
}
```

The domain ID may be a privacy-preserving digest of an internal deployment identifier. Equality must be stable for the policy lifetime.

### 6.2 Domain kinds

The registry can define kinds such as:

- KeyCustodyRoot
- AdministrativeControl
- WritableTrustState
- ProcessHost
- ImplementationLineage
- OrganizationControl
- SubjectControl
- SensorSource
- NetworkPowerInfrastructure

Custom deployment kinds require a frozen registry mapping/version. Unknown required domain kinds fail closed.

### 6.3 Canonicalization

For each signer:

- failure-domain refs are sorted by `(kind_id, domain_id)`;
- duplicates reject or canonicalize once according to frozen semantics;
- maximum refs per signer is bounded;
- maximum domain-kind registry size is bounded.

---

## 7. Subject-control relationship

External independence requires more than organizational naming.

Trust metadata should be able to answer whether a signer/domain is controlled by:

- the requesting subject;
- an ancestor/descendant under policy-relevant control;
- the same administrative domain as the subject;
- another entity forbidden by the deployment profile.

A signer inside a forbidden subject-control domain cannot satisfy an `ExternalIndependent` requirement even with a valid signature.

The controlled subject cannot self-declare itself external.

---

## 8. Quorum independence evaluation

A verified quorum policy can require:

```text
minimum_unique_identities >= N
minimum_distinct_domains(kind=K1) >= N1
minimum_distinct_domains(kind=K2) >= N2
forbid_subject_controlled_signers = true
allowed_roles = {...}
```

Evaluation is over verified signer records, not raw approval declarations.

### 8.1 Example semantics

If three signatures come from:

- identity A / admin domain X;
- identity B / admin domain X;
- identity C / admin domain Y;

then:

- unique identity count = 3;
- distinct admin domains = 2.

A policy requiring 3 unique identities and 3 distinct admin domains fails.

### 8.2 Missing metadata

If policy requires distinct `AdministrativeControl` domains and a signer lacks verified metadata for that kind, that signer cannot satisfy the independence requirement.

Unknown does not count as distinct.

---

## 9. Trust snapshot verification

A production snapshot verifier establishes:

1. strict bounded canonical decode;
2. schema/registry support;
3. canonical digest;
4. authorized trust-rotation/issuance signature/quorum;
5. signer/key record validity;
6. no duplicate canonical keys;
7. no duplicate/conflicting identity mapping forbidden by policy;
8. valid lifecycle/time intervals;
9. valid role/usage sets;
10. bounded/canonical failure-domain refs;
11. monotonic sequence/epoch transition;
12. predecessor/supersession binding where required;
13. policy-root compatibility;
14. freshness at verification time.

Only then may the verifier produce a `VerifiedTrustSnapshot` handle.

---

## 10. Trust rotation and recovery

Ordinary rotation and catastrophic recovery are distinct.

### 10.1 Normal rotation

Normal rotation may:

- add a new active key;
- retire an old key;
- revoke a compromised key;
- update roles/failure-domain metadata under authorized governance;
- advance snapshot sequence.

It must not:

- roll sequence backward;
- silently reactivate revoked key without explicit governed semantics;
- weaken policy/trust roots outside the authorized transition;
- change identity mappings to manufacture quorum independence without evidence/governance.

### 10.2 Fresh-epoch recovery

If trust roots/recovery policy require a fresh epoch:

- old terminal trust state is bound into the recovery evidence;
- new `trust_epoch_id` is explicit;
- active grants/quorums do not automatically cross the epoch;
- new trusted capabilities are reverified/reissued;
- incident linkage is durable.

Compromise of the configured recovery threshold remains an explicit catastrophic trust-root assumption.

---

## 11. Revocation propagation

Revocation may target:

- one key;
- one signer identity;
- one role usage;
- one trust epoch/root;
- one grant ID/generation through grant-specific revocation;
- one policy snapshot/version;
- another bounded scope defined by policy.

Current positive-authority use must check the revocation sources required by the production profile.

A previously created `Verified*` process capability may need to become unusable after later revocation. Implementations may achieve this through short validity, revocation generation binding, current-state rechecks, or another verified mechanism; they must not assume verification creates eternal validity.

---

## 12. Denial-of-service posture

Trust uncertainty causes loss of new positive authority, not automatic trust relaxation.

Examples:

- trust snapshot unavailable -> use already accepted snapshot only until its existing expiry/policy permits;
- rotation source unavailable -> no grace extension;
- revocation source freshness uncertain -> freeze according to policy;
- failure-domain metadata unavailable -> cannot satisfy required independence;
- signer verification provider unavailable -> no positive capability from that evidence.

---

## 13. Threat mapping

Primary threats:

- T01 local positive-grant forgery;
- T03 duplicate signer/Sybil quorum;
- T04 common-mode quorum compromise;
- T11 revoked/expired signer replay;
- T15 monitor common-mode failure;
- T17 policy downgrade/self-policy;
- T18 trust-state rollback as a durable-state special case;
- T25/T30 recovery-role/root assumptions.

Primary verification IDs:

- TV01, TV03, TV04, TV11, TV15, TV17, TV25, TV30;
- VF-QUORUM, VF-MON, VF-GOV.

---

## 14. Golden/negative vector requirements

Before implementation promotion, retain vectors for:

- canonical empty/minimal valid snapshot;
- multiple signer records sorted canonically;
- duplicate key rejection;
- duplicate signer identity behavior according to policy;
- unsupported signature profile;
- invalid key window;
- active/retired/revoked eligibility;
- wrong role/usage;
- snapshot expired/not-yet-valid;
- sequence rollback;
- same-sequence different-digest collision;
- predecessor mismatch;
- missing required failure-domain metadata;
- domain collision causing quorum failure;
- subject-controlled signer rejected for external role;
- legitimate key rotation;
- explicit revocation;
- fresh-epoch recovery boundary.

---

## 15. Current status

The Fabrication Kernel provides useful proven-in-repository patterns for lifecycle snapshots and verified threshold ceremonies, but this RSK contract is not yet implemented.

A future shared trust crate may be appropriate only after semantic review demonstrates it is truly domain-generic.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
