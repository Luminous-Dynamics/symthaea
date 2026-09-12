# Replicator Safety Kernel — Verified Positive-Evidence Boundary v0.1

Status: **normative design contract; NOT an implemented production trust boundary**

This specification defines the raw/signed/verified evidence boundary that a production RSK implementation must satisfy before caller-provided positive claims can influence replication authority.

It contains no physical replication mechanism.

---

## 1. Foundational rule

> **Data is not authority merely because it is well-formed, signed, or deserialized.**

Positive RSK authority requires a verifier-produced capability proving the relevant cryptographic, lifecycle, role, scope, freshness, policy, and failure-domain predicates.

The production authority path is conceptually:

```text
raw bytes
  -> bounded canonical decode
  -> structural validation
  -> evidence digest verification
  -> cryptographic signature verification
  -> trust snapshot / lifecycle / role verification
  -> scope and policy verification
  -> failure-domain / quorum verification
  -> opaque Verified* capability
  -> current constitutional + negative-state evaluation
  -> bounded authorization
```

Any failure returns diagnostics and **no capability**.

---

## 2. Type-state boundary

### 2.1 Raw types

Raw types are untrusted data and may be parsed from network/disk/API input.

Examples:

```rust
SignedReplicationGrant
SignedApproval
SignedRuntimeObservation
SignedRiskPolicySnapshot
SignedTrustSnapshotUpdate
```

Raw types may implement canonical decode/encode after strict bounds and validation.

They are not accepted directly by the production authority evaluator.

### 2.2 Verification reports

Verification reports are diagnostic evidence.

Conceptually:

```rust
VerificationReport {
    evidence_digest,
    signer_results,
    trust_snapshot_digest,
    policy_digest,
    violations,
}
```

A report with no violations may accompany a verified capability, but the report itself is not authority.

### 2.3 Verified capability types

Examples:

```rust
VerifiedReplicationGrant
VerifiedIndependentQuorum
VerifiedRuntimeSafetyWitness
VerifiedRiskPolicySnapshot
VerifiedTrustSnapshot
```

Requirements:

- fields are private outside verifier modules;
- no public trusted constructor;
- no `Deserialize`/generic decode directly into verified state;
- read-only accessors expose only required facts;
- retain verification provenance/digests needed for audit/revalidation;
- cannot silently outlive their validity/trust/policy epoch;
- cannot be widened by cloning, projection, or conversion.

`Clone`/`Copy` is a security design decision, not a convenience default. A production implementation should prefer explicit cheap handles or reference-counted immutable capabilities if ownership semantics matter.

---

## 3. Canonical evidence envelope

Each signed evidence family has a frozen schema identifier and bounded canonical representation.

Conceptual envelope:

```text
schema_id
payload_bytes / payload_digest
signature_profile_id
key_id
signature_bytes
```

The signature message is domain separated:

```text
H(
  "symthaea.rsk.signature-message.v1\0" ||
  evidence_family_id ||
  schema_id ||
  payload_digest
)
```

Exact binary encoding is deferred to the canonical-evidence implementation, but the following are normative:

- deterministic field ordering;
- deterministic integer endianness/width;
- explicit lengths;
- strict maximum sizes/counts before allocation;
- no unbounded nested containers;
- no implicit/default omitted authority fields;
- unknown critical discriminants reject;
- non-canonical alternate encodings reject;
- every authority-relevant field changes the evidence digest.

Do not sign/debug-hash:

- Rust `Debug` text;
- process memory representation;
- unordered map iteration;
- serializer defaults whose canonical format is not frozen.

---

## 4. Cryptographic agility

Constitutional semantics do not depend on a specific signature algorithm.

Use a policy-controlled registry identifier:

```text
SignatureProfileId(u16 or equivalently frozen registry value)
```

A signature profile can bind:

- algorithm/family;
- parameter set;
- key encoding;
- signature encoding;
- verification provider profile;
- deprecation/validity policy.

Examples may include Ed25519, ML-DSA parameter sets, or future schemes, but these are deployment/trust policy choices, not constitutional invariants.

Unknown/disallowed profiles fail closed.

Algorithm migration requires explicit policy/trust evidence and must not reinterpret old signature bytes under a new profile.

---

## 5. Trust snapshot contract

### 5.1 Trust snapshot identity

A production trust snapshot binds:

```text
TrustEpochId
sequence
issued_at
expires_at
schema_registry_version
signature_profile_registry_version
signer trust records
policy-root metadata
previous snapshot/head binding where applicable
```

The canonical snapshot digest is retained by every verified capability created from it.

### 5.2 Signer trust record

A signer trust record binds at least:

```text
SignerIdentityId
KeyId
SignatureProfileId
not_before
not_after?
lifecycle_state
allowed_roles/usages
failure_domain_refs
subject_control_domains
```

Lifecycle states minimally distinguish:

- Active
- Retired
- Revoked

A policy may add additional non-authorizing states such as Suspended/Compromised, but unknown states never become Active implicitly.

### 5.3 Key role/usage

Production roles should be explicit, for example:

- ReplicationGrantIssuer
- ReplicationApprover
- RuntimeMonitor
- RiskPolicyAuthority
- QuarantineAuthority
- RevocationAuthority
- RecoveryAuthority
- TrustRotationAuthority
- TransparencyWitness
- BuildAdmissionAuthority

A key valid for one role is not valid for another unless the trusted snapshot explicitly grants both.

### 5.4 Snapshot freshness and rollback

Verification requires:

```text
issued_at <= verification_time < expires_at
accepted_sequence >= previous_accepted_sequence
```

and durable anti-rollback semantics appropriate to the deployment.

Reject/freeze on:

- sequence rollback;
- same-sequence different digest collision;
- issued-at regression where forbidden;
- expired snapshot;
- unsupported schema/registry;
- invalid predecessor/rotation evidence;
- unavailable continuity where policy requires it.

Trust snapshot freshness is not supplied by the requesting subject.

---

## 6. Failure-domain contract

### 6.1 Representation

Failure-domain facts are trusted metadata, not signer self-assertions.

A compact canonical representation may use:

```text
FailureDomainRef {
    kind_id,
    domain_id_digest,
}
```

where `kind_id` comes from a policy/registry and `domain_id_digest` identifies a deployment-specific domain without requiring public disclosure of sensitive topology.

Relevant kinds can represent:

- key-root/HSM custody;
- admin/IAM control;
- writable state store;
- process/host/hypervisor;
- implementation/code lineage;
- organization/control domain;
- subject-control relationship;
- sensor/observation source;
- network/power/common infrastructure.

### 6.2 Quorum policy

A verified policy may require rules like:

```text
minimum_unique_signer_identities >= N
minimum_distinct_domains(kind=A) >= X
minimum_distinct_domains(kind=B) >= Y
no signer in requesting_subject_control_domain
```

The exact deployment rules are policy, but the evidence type must represent enough verified metadata to enforce them.

### 6.3 Duplicate and collision behavior

Reject or de-count according to policy when:

- identical signer identity appears twice;
- two keys map to one identity;
- two identities collapse into a domain required to be distinct;
- signer control is attributable to the requesting subject when external independence is required;
- failure-domain metadata is missing for a required dimension.

Missing required independence evidence is not permission.

---

## 7. VerifiedReplicationGrant

### 7.1 Raw signed grant payload

The canonical payload binds at least:

- schema/version;
- grant ID;
- subject ID;
- lineage ID;
- grant generation;
- capability schema ID and allowed capability set;
- resource-accounting schema ID;
- direct-child limit;
- total-descendant limit;
- lineage-depth limit;
- resource limit;
- not-before/expiry;
- safety-case digest;
- containment-envelope digest;
- risk/policy snapshot digest or policy epoch;
- trust epoch/snapshot compatibility requirements;
- optional exact action/purpose scope where the grant is narrower than general bounded authority.

### 7.2 Verification

A grant verifier checks:

1. bounded canonical decode;
2. canonical payload digest;
3. allowed signature profile;
4. signature validity;
5. signer exists in accepted trust snapshot;
6. signer lifecycle active at verification time;
7. grant-issuer role/usage allowed;
8. subject-control/independence policy;
9. grant validity window;
10. policy/schema compatibility;
11. subject/lineage/generation validity;
12. revocation/supersession state;
13. all numeric ceilings/schema values valid.

Only then may it return `VerifiedReplicationGrant`.

### 7.3 Retained provenance

The verified capability retains read-only facts including:

- grant payload/digest;
- signer identity/key/profile;
- trust snapshot digest/sequence/epoch;
- verified policy digest/epoch;
- verification time/continuity reference;
- validity bounds;
- relevant signer failure-domain references.

---

## 8. VerifiedIndependentQuorum

### 8.1 Approval payload

Every approval signs the same exact commitment, such as:

```text
purpose
subject
lineage
grant_id/generation or action commitment
policy_digest
safety_case_digest
containment_digest
validity window
```

Approvals over different payload digests do not combine.

### 8.2 Verification

The verifier:

1. bounds approval count and signature sizes;
2. verifies every canonical payload/digest/signature;
3. verifies trust snapshot freshness;
4. verifies signer lifecycle/role;
5. rejects duplicate identity inflation;
6. resolves trusted failure-domain metadata;
7. applies verified quorum/failure-domain policy;
8. rejects stale/expired approvals;
9. rejects requester-controlled signers where forbidden;
10. returns an opaque quorum capability only if every threshold/domain rule passes.

### 8.3 Retained audit facts

The capability retains:

- exact payload commitment;
- ordered canonical signer identities;
- key/profile IDs;
- trust snapshot digest;
- policy digest;
- verified failure-domain summaries sufficient to audit threshold satisfaction;
- earliest expiry / validity intersection.

The production evaluator never consumes `independent_approvals: u16` as trusted evidence.

---

## 9. VerifiedRuntimeSafetyWitness

### 9.1 Raw observation

A runtime-monitor observation binds:

- monitor identity/profile;
- subject/scope;
- monotonic observation sequence/epoch;
- observation time/freshness bound;
- monitoring health state;
- containment state/digest;
- safety-case digest;
- policy/profile digest;
- any explicit negative facts;
- optional sensor/evidence digest roots.

### 9.2 Verification

The verifier establishes:

- signed/attested source provenance;
- active monitor role/lifecycle;
- trusted time/continuity/freshness;
- expected monotonic sequence;
- exact subject/scope;
- expected containment/safety-case/policy binding;
- required monitor failure-domain properties;
- no revocation/supersession.

A verified healthy witness does not create a grant. A verified negative witness can veto/freeze according to policy.

### 9.3 Multi-monitor composition

For profiles requiring more than one monitor path, compose verified monitor capabilities under explicit failure-domain policy. Two monitors sharing a required-distinct domain do not satisfy that requirement.

---

## 10. VerifiedRiskPolicySnapshot

A verified policy snapshot is separate from the request and binds at least:

- policy ID/version/digest/epoch;
- policy authority identity and trust snapshot;
- risk-class rules;
- quorum signer/failure-domain requirements;
- monitor requirements;
- maximum evidence ages/time uncertainty;
- accepted signature profiles;
- capability schema IDs;
- resource-accounting schema IDs;
- grant/recovery role requirements;
- supersession/rollback rules;
- validity interval where applicable.

Production rule:

```text
request self-restriction may raise requirement
verified policy may raise requirement
kernel constitutional floor cannot be lowered
```

Policy downgrade or stale policy never becomes valid because the requester prefers it.

---

## 11. Verification-time versus commit-time facts

Verified positive capabilities are not a substitute for current commit-time checks.

At commit, RSK still checks:

- exact ledger cursor/head;
- current quarantine/revocation/fork state;
- current grant revocation/supersession;
- authority/action exact binding;
- current time/continuity and expiry;
- current runtime witness/freshness;
- inherited capability/budget ceilings;
- exact admitted runtime/build identity as required.

A capability that was valid at verification time can become unusable later.

---

## 12. Durable replay rule

Never persist a trusted capability as the authoritative source of truth.

Persist:

- canonical raw signed evidence;
- evidence digests;
- trust/policy/checkpoint references;
- verification report/audit metadata where useful.

After restart/replay:

```text
raw durable evidence
  -> canonical validation
  -> verify current/required historical trust context
  -> reconstruct verified capability
  -> re-evaluate current negative/revocation/fork/time facts
```

This prevents `Deserialize<VerifiedReplicationGrant>` from becoming an authority bypass.

Historical replay may need the exact historical trust/policy snapshot plus proof that it was accepted at the relevant epoch, while present-time use still observes later revocations/supersession according to policy.

---

## 13. Parser/resource bounds

Every external evidence family defines strict maxima for:

- message bytes;
- signature bytes;
- key ID bytes;
- signer/approval count;
- failure-domain references per signer;
- trust keys per snapshot;
- policy rules/entries;
- nesting depth;
- string/identifier lengths.

Bounds are checked before large allocation. Oversized/unknown/noncanonical input fails closed and cannot force authority parser resource exhaustion.

---

## 14. Verification errors

Errors should be typed enough to support audit/fuzz/property assertions, including classes such as:

- unsupported schema/profile;
- noncanonical encoding;
- digest mismatch;
- invalid signature;
- unknown signer;
- signer not yet valid/expired/retired/revoked;
- role/usage not allowed;
- stale/rolled-back trust snapshot;
- wrong subject/lineage/generation;
- policy mismatch/stale policy;
- duplicate signer identity;
- failure-domain collision;
- insufficient independent quorum;
- monitor stale/sequence rollback;
- evidence too large;
- revocation/supersession;
- continuity unavailable.

Errors are denial evidence, not alternate authority paths.

---

## 15. API layering

Recommended crate/module direction after reference CI is healthy:

```text
symthaea-replicator-safety
    constitutional pure semantics

symthaea-replicator-ledger
    lineage/budget/reference commit semantics

small generic trust/verification layer OR RSK-local verifier modules
    canonical signed envelopes
    trust snapshot/lifecycle
    signature verifier traits
    failure-domain policy
    opaque Verified* capabilities

RSK production adapter
    consumes Verified* + current ledger/negative/time state
```

Do not make the production RSK authority crate depend directly on the Fabrication Kernel merely for its trust types.

A future generic trust crate is only justified if both domains can depend on it without importing each other's authority semantics.

---

## 16. Threat/test mapping

Primary threat IDs:

- T01, T02, T03, T04, T05;
- T11;
- T14, T15;
- T17;
- T27 in conjunction with build admission.

Primary verification plan IDs:

- TV01–TV05;
- TV11;
- TV14–TV17;
- VF-QUORUM, VF-MON, VF-BUILD.

Additional property:

> **Fully verified positive evidence + any current hard negative fact still results in deny/freeze/non-operational authority.**

---

## 17. Acceptance criteria before implementation promotion

- [ ] canonical envelope/wire discriminants frozen and golden-vector tested;
- [ ] trust snapshot canonical bytes/digest/sequence semantics frozen;
- [ ] verified types have no public trusted constructor/deserializer;
- [ ] unknown/retired/revoked/wrong-role signer rejects;
- [ ] stale/rollback/colliding trust snapshot rejects;
- [ ] wrong subject/lineage/generation/policy rejects;
- [ ] duplicate identity cannot inflate quorum;
- [ ] required failure-domain collision prevents independence satisfaction;
- [ ] raw approval count cannot enter production evaluator as trusted quorum;
- [ ] raw runtime booleans cannot enter production evaluator as trusted witness;
- [ ] verified positive evidence cannot override current local negative state;
- [ ] durable replay reconstructs verified capabilities through verification, not deserialization;
- [ ] malformed/oversized input bounded/fuzz tested;
- [ ] exact-head Class A CI/evidence retained;
- [ ] exact build/runtime identity bound by #1682.

---

## 18. Current status

This document freezes the target boundary only.

Current RSK Rust crates remain reference semantics and `publish = false`. No verified positive evidence implementation is production-admitted.

**Production admission status: DENIED / NOT YET ELIGIBLE.**
