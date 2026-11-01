# BASALT: Scoped etcd CAS and Native Transaction Revisions

**Research specification:** v0.1  
**Date:** 2026-11-01  
**Status:** draft; no etcd adapter implemented  
**Tracking:** [#9724 — scoped CAS](https://github.com/Luminous-Dynamics/symthaea/issues/9724)  
**Parent contract:** [#9702 — remote authoritative state anchor](https://github.com/Luminous-Dynamics/symthaea/issues/9702)

## 1. Purpose

The remote anchor must atomically advance the exact source cursor, log checkpoint cursor, receipt record, and relevant policy bindings. It must also preserve an auditable monotonically increasing revision and exact-retry behavior.

The design must avoid two different mistakes:

1. Treating etcd's Raft ordering as if transactions can commit in parallel without a global order.
2. Adding an application-level global counter key to every transaction when the etcd store already maintains a global logical revision.

This design compares two implementations:

- **Global-counter arm:** every acceptance reads, compares and updates a mutable /authority/revision key.
- **Scoped-CAS arm:** each acceptance compares only its source, log, policy and idempotency scopes. The successful transaction header revision is the global audit watermark; the immutable receipt key's create_revision is the original acceptance revision.

The second arm is a testable hypothesis about reducing application-level compare conflicts, not a promise of parallel Raft commits or unlimited throughput.

## 2. Relevant etcd semantics

The v3.7 API describes a transaction as an atomic If/Then/Else operation guarded by comparisons. A transaction can apply several key updates under one store revision. The top-level transaction response header carries the key-value store revision when the request was applied.

Each KeyValue also carries:
- create_revision: revision of the last creation of the key;
- mod_revision: revision of its last modification;
- version: key version, reset to zero on deletion.

Range is linearizable by default. Setting serializable uses member-local reads that can be stale with respect to the cluster quorum. Use linearizable reads/transactions for freshness decisions. A watch is an event notification/replay mechanism, not a replacement for a current linearizable read.

Sources:
- https://etcd.io/docs/v3.7/learning/api/
- https://etcd.io/docs/v3.7/dev-guide/api_reference_v3/

## 3. Key layout and invariants

Use a versioned key namespace with deterministic, collision-resistant key components. Do not interpolate untrusted IDs without canonical encoding or hashing.

| Key | Mutation semantics | Purpose |
|---|---|---|
| /basalt/v1/source/<lineage-digest> | Mutable by accepted transition | Current source generation, predecessor, seal, profile/boot and last receipt |
| /basalt/v1/log/<origin-digest> | Mutable by accepted transition | Current tree size, root, checkpoint identity |
| /basalt/v1/receipt/<envelope-digest> | Immutable; create once | Accepted receipt plus original acceptance context |
| /basalt/v1/policy/<scope-digest> | Mutable only by policy administration | Active policy bundle/version pointer |
| /basalt/v1/witness/<scope-digest> | Mutable only by witness-policy administration | Witness roster and policy pointer |

A receipt record binds immutable identity fields:
- envelope/payload digest and type;
- stable host/source/partition lineage;
- source generation, predecessor seal and new seal;
- observer profile and boot incarnation;
- continuity-evidence references;
- original log origin/checkpoint identity;
- original signer/witness policy bundle identities and versions;
- acceptance-record schema/profile version.

It may also reference the exact proof material used by the original admission decision.

Proof refresh is separate from immutable receipt identity. A retry with new checkpoint material, newer signer status or newer witness policy must not rewrite the historical acceptance record or pretend the new material was part of the original decision. It may trigger a separate re-verification/currentness result.

### 3.1 Required invariants

1. Receipt, source cursor and log cursor advance in one transaction or none do.
2. One source generation has at most one immutable receipt digest.
3. The receipt key is immutable and is not attached to an expiring lease.
4. Exact receipt retry returns the original receipt record and original acceptance revision.
5. A conflicting immutable identity at an existing envelope digest is equivocation/collision.
6. Source-key and log-key comparisons bind the exact prior values or mod revisions.
7. Policy and witness pointers used by the proof decision are compared in the commit transaction.
8. No global application counter key is required for every transition.
9. The etcd store revision is retained as an audit watermark, not misrepresented as a per-source or proof-validity revision.
10. Authority snapshot restore remains outside etcd's ordinary crash-fault claim and requires an independent monitor/witness or an unknown verdict.

## 4. Snapshot, verify, and compare-and-advance

### 4.1 Snapshot

Use one linearizable transaction containing the relevant Range operations, or a range request that returns the complete relevant prefix at a single snapshot revision. Do not issue unrelated sequential reads and assume they represent one coherent snapshot.

Read:
- source cursor;
- log cursor;
- exact receipt key;
- active signer/trust-root policy pointers;
- active witness-policy pointer(s);
- any provisioning or source-lineage key required by the profile.

Record the top-level snapshot revision and every key's exact value, mod_revision, create_revision, and version.

### 4.2 Verify outside the write transaction

Verify DSSE/in-toto envelope and payload binding, signer authorization, checkpoint signature/origin, inclusion and consistency proof, witness quorum, source predecessor/generation, and claim applicability against the snapshot.

No cryptographic or remote proof operation should be assumed atomic with etcd. The compare phase ensures the relevant state has not changed since verification.

### 4.3 One compare-and-advance transaction

Illustrative compare set:

    VERSION(receipt_key) == 0
    MOD(source_key) == snapshot.source.mod_revision
    MOD(log_key) == snapshot.log.mod_revision
    MOD(signer_policy_key) == snapshot.signer_policy.mod_revision
    MOD(witness_policy_key) == snapshot.witness_policy.mod_revision

For a key that does not exist (for example, first source/log bootstrap), compare VERSION(key) == 0 and ensure the candidate includes the independently pinned bootstrap proof required by the profile.

Success operations in the same transaction:

    PUT(receipt_key, immutable_accepted_record)
    PUT(source_key, proposed_source_cursor)
    PUT(log_key, proposed_log_cursor)

The exact production compare semantics must also bind absent/present keys and compare full prior values where mod_revision alone is insufficient under the selected restore threat model. Policy key values are immutable digests/version pointers, and policy mutation must advance the scoped pointer revision.

The successful transaction response's top-level header.revision is the global audit watermark for that commit. All writes in the transaction share that store revision. The newly created receipt key's create_revision is the receipt's original acceptance revision. Do not write a separate global revision key merely to duplicate this etcd-provided logical clock.

### 4.4 Compare failure

A failed comparison means the snapshot is no longer the state against which the candidate was verified. Return a typed conflict, fetch a fresh linearizable snapshot, and rerun all checks whose inputs may have changed.

Do not reuse old proof outcomes where the changed key is a dependency of the decision. A changed unrelated key does not require restarting verification if the exact relevant scopes are still proven current.

### 4.5 Lost response and exact retry

After a timeout or lost transaction response:

1. Perform a linearizable read of the exact receipt key.
2. If absent, state is not yet proven committed; retry only after taking a new snapshot and re-verifying.
3. If present, verify all immutable receipt identity bindings.
4. Return the original record's create_revision and stored acceptance context.
5. If the same digest is present with conflicting immutable identity, return collision/equivocation.
6. Do not report the current store header revision as the original acceptance revision.

Do not attach a lease to immutable receipt records. Do not delete/recreate receipt keys: deletion resets key version and a later recreation changes create_revision. If archival or retention policy requires removing active keys, define and qualify an immutable archive that preserves the original acceptance revision and proof material before any deletion is allowed.

## 5. Contention model

Expected conflict domains:

- **same source lineage:** serialize on the source cursor. This is required to prevent two accepted generations from sharing one predecessor.
- **same transparency-log origin:** serialize on its log cursor. This is required to preserve checkpoint progression.
- **distinct source and distinct log origin:** no application-level conflict solely because an unrelated source advanced.
- **shared policy pointer:** policy is read/compared, not rewritten by ordinary evidence acceptance. A real policy change conflicts with in-flight transactions that verified against the old policy.
- **global store revision:** advances for successful transactions as the etcd global logical clock; it is not itself a mutable compare key in each write.

The scoped design cannot eliminate the shared Raft log, resource limits, network latency, quorum availability, or real hot-key contention. It aims only to remove unnecessary conflicts introduced by an application-level global counter key.

## 6. Restore and witness boundary

The etcd cluster is a crash-fault state service, not a Byzantine witness quorum. A whole-cluster restore may restore an internally consistent older key space and lower logical revision. A separate monitor/witness domain must retain a more recent authenticated checkpoint of the authority state, or report freshness unknown.

For log split-view detection, the Sigstore Rekor architecture specifies independent witnesses that persist checkpoints and distribute/gossip them; one witness alone is insufficient to rule out split views. BASALT should keep the state-authority monitor and transparency-log witness roles logically separable and declare their administrative failure domains.

No witness quorum should be called BFT without a stated node/adversary model, quorum intersection argument, durable non-equivocation state, and restore/reprovisioning analysis.

## 7. Adversarial conformance plan

| ID | Scenario | Required invariant |
|---|---|---|
| SCAS-001 | Distinct sources and logs start from same snapshot | Both commit; unrelated key changes do not cause artificial conflict |
| SCAS-002 | Distinct sources share one log cursor | One advances; loser rereads and re-verifies |
| SCAS-003 | Same source/generation, different receipts | Exactly one can commit; other is equivocation |
| SCAS-004 | Retry receipt A after unrelated transactions advanced store revision | Return A's original create_revision |
| SCAS-005 | Existing receipt digest has different immutable source/generation/seal | Reject collision |
| SCAS-006 | Signer-policy pointer advances after snapshot, before CAS | Compare fails; old policy cannot commit |
| SCAS-007 | Witness roster/policy pointer advances during verification | Compare fails or result unknown |
| SCAS-008 | CAS commits but client response is dropped | Exact lookup recovers same receipt and original create_revision |
| SCAS-009 | Receipt missing after ambiguous result | Indeterminate; no inference from unrelated cursor |
| SCAS-010 | Whole authority cluster restored to older snapshot | Independent witness detects rollback or result is unknown |
| SCAS-011 | Serializable stale read supplied as currentness evidence | Unknown; no freshness promotion |
| SCAS-012 | Unrelated global key advances | Relevant capsule scopes remain reusable after linearizable currentness check |
| SCAS-013 | Leader change/network partition during transaction | No partial source/log/receipt update |
| SCAS-014 | Receipt archival/compaction/deletion | Original create_revision and evidence remain recoverable |
| SCAS-015 | Source or log key deleted then recreated | Detect/reset prohibited; lineage continuity cannot silently restart |

## 8. Benchmark design

Compare:
- **Arm A:** global mutable authority counter key read/compared/updated by every acceptance.
- **Arm B:** scoped compare set; etcd store revision and receipt create_revision provide audit/acceptance watermarks.

Keep etcd version, cluster size, hardware, network, durability, payload size, operation count and failure schedule identical.

Workloads:
- 1/2/8/16/32/64/128 clients;
- 1/10/100/1,000 source lineages;
- 1/10/100 log origins;
- one hot shared log versus distributed log origins;
- duplicate/conflicting retry rates from 0% to 50%;
- 1/10/50/90% read/write mixes;
- leader changes, response loss, network partition and quorum loss.

Safety gates before efficiency:
- zero partial source/log/receipt writes;
- zero source-generation equivocation;
- zero policy/witness downgrade;
- exact retry returns original acceptance revision;
- zero stale/serializable read promotion;
- 100% explicit unknown where authority or witness freshness is unavailable;
- authority restore never promoted without independent monitor evidence.

Efficiency metrics:
- accepted transactions/sec;
- p50/p95/p99 commit and exact-retry latency;
- conflict rate by key scope;
- retries per successful commit;
- per-log hot-key wait time;
- CPU, disk write bytes, fsync/raft commit cost, and network bytes;
- cache rebuild time and bytes;
- detection delay after authority restore.

Do not claim throughput superiority from a reduction in compare conflicts alone. Publish both the app-level conflict rate and total cluster-level throughput/latency. Report local SQLite latency separately from remote etcd.

## 9. Implementation sequence

1. Freeze key encoding and immutable receipt schema.
2. Build a deterministic in-memory scoped-CAS model with exact receipt accepted_revision.
3. Add reference tests for source/log/policy compare scope and conflict semantics.
4. Implement a read-only multi-key snapshot using etcd v3 transaction/range operations.
5. Implement compare-and-advance using exact key version/mod_revision and receipt absence.
6. Recover lost responses through a linearizable receipt-key lookup and original create_revision.
7. Add a local cache materializer that only reads authoritative records.
8. Add fault-injection and cluster restore/witness simulation.
9. Run global-counter and scoped-CAS benchmark arms on identical cluster/hardware.
10. Integrate with #9702 and #9722 only after conformance parity.

## 10. Claim boundary

This document is a design and benchmark specification. It does not implement an etcd client, establish real linearizability, prove malicious snapshot resistance, qualify witnesses, or measure any performance result. The scoped design still relies on etcd's cluster-level consensus order and the independent witness assumptions.

## 11. References

- etcd v3.7 API and transaction semantics: https://etcd.io/docs/v3.7/learning/api/
- etcd v3.7 API reference, response revision and linearizable Range: https://etcd.io/docs/v3.7/dev-guide/api_reference_v3/
- etcd KeyValue create_revision/mod_revision: https://github.com/etcd-io/etcd/blob/main/api/mvccpb/kv.proto
- Rekor witness and split-view requirements: https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md
- BASALT remote anchor: https://github.com/Luminous-Dynamics/symthaea/issues/9702
- BASALT scoped-CAS follow-up: https://github.com/Luminous-Dynamics/symthaea/issues/9724
