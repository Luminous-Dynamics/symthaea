# ADR-007: RSK Verified Positive-Evidence Boundary

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The RSK reference semantics intentionally use ordinary Rust values for grants, quorum counts, runtime witness facts, and semantic quorum policy. That is useful for modeling constitutional logic, but those public values are not a sufficient production trust boundary.

A controlled caller must never be able to manufacture positive authority by:

- constructing a `ReplicationGrant` and selecting an issuer enum;
- incrementing an approval count;
- claiming signer/failure-domain independence;
- constructing healthy/fresh runtime-monitor booleans;
- selecting a policy value favorable to its own request;
- deserializing an object that looks like a previously verified capability.

This ADR contains no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path.

## Decision

Adopt a strict raw/signed/verified type boundary for all production positive-authority evidence.

Conceptually:

```text
Raw/Signed Evidence
    -> structural/canonical validation
    -> cryptographic verification
    -> trust-snapshot freshness/lifecycle/role verification
    -> scope/policy/failure-domain verification
    -> private opaque Verified* capability
    -> constitutional authority evaluator
```

The production evaluator may consume opaque verified positive types, but must never accept caller-constructed raw/reference positive claims as sufficient authority.

## Required verified capabilities

Exact names are non-normative, but production needs distinct capability-bearing types equivalent to:

- `VerifiedReplicationGrant`
- `VerifiedIndependentQuorum`
- `VerifiedRuntimeSafetyWitness`
- `VerifiedRiskPolicySnapshot`
- `VerifiedTrustSnapshot` or an equivalent monotonic trusted snapshot handle

Verified types:

- have private fields/private construction outside verifier modules;
- are not directly deserializable into trusted state;
- retain the exact evidence/trust/policy digests and verification time/continuity references needed for audit and revalidation;
- expose only read-only authority-relevant facts;
- are invalid outside their bound subject/lineage/generation/policy/trust epoch and validity interval.

## Verification reports are not authority

A verification report may explain:

- valid/invalid signatures;
- signer lifecycle results;
- duplicate identities;
- failure-domain collisions;
- policy violations;
- scope mismatches;
- stale trust snapshots;
- revocation or expiry.

The report is diagnostic evidence only. It does not become positive authority unless the verifier returns the opaque verified capability.

## Trust snapshot semantics

Production verification requires an explicit, versioned, bounded trust snapshot containing or binding:

- schema/version;
- monotonic sequence/epoch;
- issued-at/expiry/continuity state;
- accepted signature algorithms/profiles;
- signer/key identities;
- key lifecycle (`active`, `retired`, `revoked` or equivalent);
- allowed key usages/roles;
- signer-control relationship to the requesting subject where relevant;
- trusted failure-domain metadata;
- policy/trust-root identifier or digest;
- supersession/rotation state.

A stale/rolled-back/colliding trust snapshot cannot create positive authority.

## Failure-domain metadata rule

Failure-domain independence is derived from trusted metadata/policy, not from a signer's self-asserted claim.

A signed approval may name its signer identity, but the verifier obtains authoritative failure-domain facts from the trusted snapshot or another separately verified trust source.

For a policy requiring distinct domains, the verifier must reject or de-count approvals that collide on required dimensions.

## Canonical signed representation

Every signed RSK evidence type must have:

- a frozen schema identifier;
- bounded field sizes/counts before allocation;
- deterministic canonical bytes;
- a domain-separated evidence digest;
- a domain-separated signature message;
- exact subject/lineage/generation/policy/trust-epoch binding as applicable;
- explicit validity interval/freshness fields where applicable.

Do not sign Rust `Debug`, default serializer output whose canonical semantics are not frozen, process memory representation, or unordered map iteration.

Cryptographic algorithm selection remains policy-controlled and agile. The evidence contract must not make one present-day algorithm a constitutional semantic.

## Verified grant requirements

A verified grant must prove at least:

- canonical payload/digest match;
- cryptographic signature validity;
- accepted algorithm/profile;
- signer identity/key ID;
- active lifecycle and allowed grant usage/role;
- fresh accepted trust snapshot;
- issuer independence requirements defined by policy;
- exact subject;
- exact lineage;
- exact grant generation;
- allowed capability schema/set;
- direct/descendant/depth/resource ceilings and resource-accounting schema;
- not-before/expiry;
- safety-case digest;
- containment-envelope digest;
- risk/policy version/epoch binding;
- no revocation/supersession invalidating the grant.

## Verified quorum requirements

A verified quorum must prove at least:

- every approval signs the same canonical purpose/payload commitment;
- every signer is unique under trusted identity semantics;
- signer keys are active, role-eligible, and unrevoked;
- signer approvals are current;
- signer identities satisfy policy-required failure-domain separation;
- duplicate/same-domain approvals cannot inflate the independent quorum;
- constitutional floors remain non-weakenable;
- the quorum binds the exact subject/action/policy/grant context it is intended to authorize.

A raw integer approval count is not production evidence.

## Verified runtime witness requirements

A verified runtime witness must prove or bind:

- monitor/source identity;
- allowed monitor role/profile;
- cryptographic or attested provenance as applicable;
- monotonic observation sequence/continuity;
- observation/freshness interval;
- exact subject/scope;
- safety-case digest;
- containment-envelope digest;
- policy/profile version;
- relevant failure-domain metadata;
- monitor lifecycle/revocation state.

The monitor remains primarily negative authority. A verified healthy witness is one required predicate, not a grant mint.

## Verified policy requirements

A verified risk/quorum policy must bind:

- exact policy schema/version/digest;
- authority that approved the policy;
- trust snapshot/epoch used to verify that authority;
- risk-class floors;
- quorum/failure-domain rules;
- monitor requirements;
- freshness/time uncertainty requirements;
- accepted capability/resource schemas;
- accepted cryptographic profiles;
- supersession state;
- validity interval where applicable.

The requester may self-restrict beyond the policy. It may never lower the verified policy or kernel floor.

## Negative facts remain separate and dominant

Even fully verified positive evidence cannot cancel:

- local quarantine/revocation;
- stale cursor/head;
- grant revocation learned after verification;
- containment drift;
- stale/failed runtime monitoring;
- fork/non-operational state;
- expired evidence;
- unadmitted runtime/build identity.

The constitutional evaluator must still combine verified positive capabilities with current local negative facts and inherited ceilings at the point of decision/commit.

## No serialization shortcut

A `Verified*` object is a process capability, not a durable wire format.

Durable storage records raw signed evidence plus verification/replay metadata. On restart/replay, trusted capabilities are reconstructed only by re-verifying against the accepted trust/policy/time/revocation state required by the durable-evidence contract.

Do not serialize a `VerifiedReplicationGrant` and later deserialize it directly as trusted.

## Relationship to Fabrication Kernel

The Fabrication Kernel already demonstrates useful patterns:

- canonical payload/digest binding;
- signer/verifier traits;
- explicit trust snapshots;
- lifecycle/usage checks;
- duplicate signer rejection;
- opaque `VerifiedAttestation` / `VerifiedThresholdCeremony` outputs.

RSK should reuse those architectural patterns, not depend directly on `symthaea-fabrication-kernel`.

If common trust primitives are extracted later, that extraction must be a separately reviewed small crate with frozen generic semantics and no fabrication- or replication-specific authority meaning.

## Threat coverage

This ADR primarily addresses:

- T01 local grant forgery;
- T02 approval-count forgery;
- T03 duplicate signer/Sybil quorum;
- T04 common-mode quorum compromise;
- T05 grant scope substitution;
- T11 revoked/expired signer replay;
- T14 runtime witness forgery;
- T15 monitor common-mode failure;
- T17 policy downgrade/self-policy;
- T27 runtime/build evidence substitution in combination with #1682.

## Evidence discipline

This ADR freezes semantics only. It does not claim #1668 is implemented.

Before implementation promotion:

- wire/canonical bytes and trust-snapshot schema must have golden vectors;
- verified types must have no public trusted constructor/deserializer;
- crypto/trust/failure-domain negative vectors must execute;
- trust snapshot rollback/collision/freshness tests must execute;
- exact-head RSK CI must pass;
- production evaluator must consume verified positive capabilities only;
- local negative-state dominance must remain proven after integration.

## Consequences

### Positive

- callers cannot create trust by selecting enum variants or constructing structs;
- signer count is replaced by identity/failure-domain-aware evidence;
- trust lifecycle and rollback become first-class;
- durable replay cannot bypass verification by deserializing a cached capability;
- algorithm agility remains a policy concern rather than constitutional semantics;
- the future implementation can be small and independently reviewed.

### Cost

- more explicit trust metadata and key lifecycle operations;
- more verification work at admission/replay boundaries;
- operational key/role/failure-domain governance becomes unavoidable;
- production admission remains blocked until actual verifier code and executed evidence exist.

## Related work

- #1335 production-admission umbrella
- #1668 verified positive evidence
- #1669 trusted time
- #1673 verified risk policy
- #1676 threat model
- #1678 capability schema
- #1679 resource accounting schema
- #1682 exact build/runtime identity
- #1728 semantic quorum-policy separation
- #1762 threat/compromise model candidate

**Production admission remains DENIED.**