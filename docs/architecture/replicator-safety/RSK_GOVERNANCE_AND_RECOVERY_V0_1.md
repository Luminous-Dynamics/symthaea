# Replicator Safety Kernel — Governance Commands and Recovery v0.1

Status: **normative design contract; NOT an implemented production governance layer**

This document specifies authenticated governance command semantics and fresh-epoch recovery for RSK.

It contains no physical replication mechanism.

---

## 1. Governing principle

> **Authority to replicate, authority to add negative state, authority to change policy, and authority to recover are distinct capabilities.**

No role inherits another role merely because it is trusted for one operation.

---

## 2. Governance command families

Production command kinds should be explicit, frozen discriminants rather than strings interpreted ad hoc.

Example semantic families:

```text
RegisterNarrowerLineagePolicy
FreezeSubjectAuthority
QuarantineSubject
QuarantineLineage
RevokeGrantGeneration
RevokeSubject
RevokeLineage
RevokePolicyOrKeyScope
DeclareForkIncident
AcceptCheckpointAnchor
```

Recovery is intentionally **not** an ordinary command family. It is a separate ceremony/transition.

Unknown command kinds fail closed.

---

## 3. Canonical command envelope

A governance command binds:

```text
schema_id
command_id
command_kind
target_scope
epoch_id
expected_sequence
expected_head_digest
policy_digest
trust_snapshot_digest
issued_at
not_before?
expires_at
reason_code
parameters_digest
issuer_identity
issuer_role
signature_profile
signature
```

Normative properties:

- bounded canonical encoding;
- deterministic digest;
- domain-separated signature message;
- strict target/kind/scope binding;
- expected ledger cursor/head binding;
- validity/freshness interval;
- verified role/lifecycle/trust state;
- one command ID cannot produce two distinct successful mutations;
- command valid in one epoch is not valid in another unless explicitly defined by recovery semantics.

---

## 4. Target scopes

Scopes are typed, not ambiguous byte strings.

Examples:

```text
Subject(SubjectId)
Lineage(LineageId)
GrantGeneration(GrantId, Generation)
Policy(PolicyId, Version)
TrustKey(KeyId)
Epoch(EpochId)
LedgerHead(EpochId, Sequence, Digest)
```

A `QuarantineSubject` command cannot be reinterpreted as `QuarantineLineage` merely because the identifier bytes happen to match.

---

## 5. Exact action binding

The governance verifier returns an opaque capability binding:

- exact command digest;
- exact kind;
- exact target;
- exact epoch;
- exact expected sequence/head;
- exact command-specific parameters;
- verified issuer identity/role;
- trust/policy snapshot digests;
- validity interval.

Mutation applies only if the current authoritative state exactly matches those preconditions.

Mismatch returns without mutation.

---

## 6. Role capability matrix

| Role | May add negative state | May revoke | May narrow policy | May widen policy/ceilings | May clear negative state | May recover/new epoch | May mint replication grant |
|---|---:|---:|---:|---:|---:|---:|---:|
| Runtime Safety Monitor | profile-scoped | no by default | no | no | no | no | no |
| Quarantine Authority | yes, scoped | no unless separately authorized | no | no | no | no | no |
| Revocation Authority | optionally quarantine | yes, scoped | no | no | no | no | no |
| Policy Authority | no by role alone | no by role alone | yes according to governance | never below constitutional/inherited restrictions | no | no | no |
| Ordinary Operator | only explicitly delegated commands | only explicitly delegated | no by default | no | no | no | no |
| Recovery Quorum | recovery-defined | recovery-defined | recovery-defined | only via explicit new-epoch policy/ceilings and never above constitutional limits | yes only through recovery semantics | yes | no implicit grant mint |
| Grant Issuer | no | no | no | no | no | no | yes, bounded by policy |

Combined roles are possible only if the verified trust/policy snapshot explicitly grants them. Co-location is a common failure domain and must be counted as such.

---

## 7. Negative-state operations

### 7.1 Freeze

Freeze is the default response to uncertainty/unavailability.

Properties:

- denies new positive replication authority;
- may be transient;
- does not assert confirmed compromise;
- does not imply destructive action;
- recovery may be simple re-evaluation when the underlying uncertainty clears, if no monotonic negative fact was committed.

### 7.2 Quarantine

Quarantine is a committed bounded negative fact.

Properties:

- scope-specific;
- inherited through applicable ancestry according to constitutional rules;
- ordinary authority cannot clear it;
- incident/evidence reference retained;
- externally governed recovery required to clear/replace it where policy requires.

### 7.3 Revocation

Revocation is explicit invalidation of a bounded target.

Properties:

- monotonic within the epoch unless recovery policy explicitly defines a new-epoch replacement;
- exact target/generation scope;
- does not automatically create replacement authority;
- revocation evidence remains durable after recovery.

---

## 8. Policy-governance constraints

Policy authority cannot use an ordinary policy update to:

- lower kernel constitutional floors;
- widen immutable ancestor ceilings;
- clear quarantine/revocation;
- choose a fork winner;
- reactivate an expired/revoked grant;
- transfer recovery role to itself without separately authorized trust/governance transition.

Policy changes bind exact predecessor policy version/digest and obey monotonic supersession/rollback rules.

---

## 9. Governance replay and stale-head rules

Before mutation, verify:

```text
command.epoch == current.epoch
command.expected_sequence == current.sequence
command.expected_head_digest == current.head_digest
command not expired
command not already consumed
issuer/trust/policy still valid
```

A stale governance command cannot be "best effort" applied to a newer head.

The caller must obtain a fresh command or follow an explicitly defined rebasing/governance procedure.

---

## 10. Fork declaration

Fork detection itself may arise from the durable verifier/witness system rather than a human command.

Once two independently valid conflicting successors are established for the same predecessor:

- affected scope enters `Forked/NonOperational`;
- both branches are retained;
- new positive authority denied;
- ordinary policy/operator commands cannot choose the winner;
- recovery ceremony is required.

A governance command may record/acknowledge the fork incident, but cannot erase one branch to make the conflict disappear.

---

## 11. Recovery ceremony inputs

A recovery proposal binds at least:

```text
recovery_schema
recovery_id
old_epoch_id
old_terminal_head_digest
old_terminal_sequence
fork_branch_heads[] if applicable
incident/evidence root digest
recovery_policy_digest
trust_snapshot_digest
preservation_plan_digest
new_hard_ceiling_commitment
new_policy_root_digest
new_trust_root_digest
new_epoch_id
new_genesis_commitment
issued_at / expiry
```

All arrays/counts are strictly bounded and canonical.

---

## 12. Recovery quorum

Recovery approvals sign the exact same recovery proposal digest.

Verification requires:

- approved recovery role;
- active/unrevoked signer lifecycle;
- fresh accepted trust snapshot;
- configured minimum unique signer identities;
- configured distinct failure-domain requirements;
- no duplicate/same-domain inflation;
- exact old state and proposal digest;
- approval validity intersection;
- recovery policy version/digest.

Recovery signer requirements should normally be stronger than ordinary replication grant requirements.

---

## 13. Recovery transition

A successful recovery transition:

1. verifies old terminal state/fork evidence;
2. verifies recovery proposal/quorum;
3. creates a **new epoch ID**;
4. creates a new genesis event binding old terminal evidence and recovery evidence;
5. declares any preserved subject/lineage mapping explicitly;
6. declares new immutable/hard ceilings explicitly;
7. establishes new accepted trust/policy roots as specified;
8. starts with **no implicit active replication grants or authorization tokens**;
9. requires positive authority to be reissued/reverified under the new epoch;
10. preserves durable incident linkage to old epoch/branches.

Old history remains immutable evidence.

---

## 14. No implicit carry-over

The following do not cross the epoch automatically:

- active `ReplicationGrant` / `VerifiedReplicationGrant`;
- bounded authorization token;
- verified quorum capability;
- runtime monitor witness;
- old ledger cursor/head capability;
- old trust snapshot as current trust;
- old policy as current policy;
- old checkpoint as new genesis authority.

Historical objects remain evidence only.

---

## 15. Recovery preservation plan

A preservation plan may identify subjects/lineages/state to retain across recovery, but retention does not equal positive authority.

For every preserved item, the new epoch records:

- old identifier and old-state digest;
- new identifier or explicit identity continuity rule;
- preserved immutable ancestry where applicable;
- new hard ceilings that are no wider than allowed by recovery/constitutional policy;
- whether negative incident flags remain attached;
- reason/evidence for preservation.

Unknown/ambiguous preservation fails closed.

---

## 16. Recovery from compromised key/trust state

If recovery is motivated by signer/trust compromise:

- compromised key/root is explicitly excluded/revoked in new trust state;
- new trust epoch/root is explicitly bound;
- old signatures remain historical evidence but do not regain positive authority;
- key identity continuity/rotation evidence is preserved where useful;
- recovery quorum cannot rely solely on the compromised domain unless that is explicitly the catastrophic outside-guarantee case.

---

## 17. Recovery from storage fork/rollback

For a fork:

- proposal lists all known conflicting branch heads;
- neither branch wins implicitly;
- preservation decision is explicit and evidence-backed;
- new genesis binds the entire incident evidence root;
- old branch histories remain retained/auditable.

For rollback ambiguity without a confirmed fork:

- new positive authority remains frozen until a trusted continuity/recovery decision establishes an accepted terminal state.

---

## 18. Transparency/Mycelix role

Mycelix/transparency may publish:

- governance command digests;
- quarantine/revocation statements;
- fork incident claims;
- recovery proposal/quorum/checkpoint digests;
- new epoch genesis claim;
- support/challenge/supersession relations.

Receipts strengthen auditability and equivocation detection.

They do not independently authorize a governance command or recovery.

---

## 19. Threat mapping

Primary threats:

- T16 negative-state suppression;
- T20 same-predecessor fork/equivocation;
- T24 governance command substitution;
- T25 recovery-role confusion;
- T26 recovery grant carry-over;
- T30 catastrophic recovery-root compromise.

Primary attack trees:

- AT-03 stale/negative authority revival;
- AT-05 durable-history corruption;
- AT-06 illegitimate recovery/fork winner selection;
- AT-08 availability pressure.

---

## 20. Production API shape

Recommended layering:

```text
SignedGovernanceCommand
  -> GovernanceCommandVerifier
  -> VerifiedGovernanceCommand
  -> exact ledger transition

SignedRecoveryProposal + SignedRecoveryApprovals
  -> RecoveryVerifier
  -> VerifiedRecoveryPlan
  -> fresh-epoch transition
```

`VerifiedGovernanceCommand` cannot call the recovery transition unless it is specifically a verified recovery capability.

The reference ledger's direct mutation methods remain useful for semantic testing but should sit behind this production adapter.

---

## 21. Current status

This is a design contract only. No production governance verifier/recovery store transition is implemented by this document.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
