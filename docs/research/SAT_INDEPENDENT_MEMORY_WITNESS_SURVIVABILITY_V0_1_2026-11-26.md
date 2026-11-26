# SAT Independent Memory Witness Survivability and Recovery v0.1

Date: 2026-11-26  
Status: research protocol, result contract and synthetic fault-injection fixtures. No remote witness implementation or security qualification is claimed.

## 1. Why the witness must be tested

SAT's T9/T10 lab demonstrates that an anchor and erasure tombstone can survive restoration of the data-store bundle, while separate controls show that restoring the store and those local trust-domain files together defeats the illustrative design. A directory boundary is not an administrative trust boundary.

The next question is not simply whether a store has a hash chain. It is whether the independent state witness remains current, consistent, authenticated and available when the store, host, backup or local receipt cache is rolled back or corrupted.

This is a new SAT assurance lane. It does not modify the native IETF T1-T10 score, and it does not claim that the IETF at-rest method itself defines remote quorum witnesses.

## 2. What existing standards and systems provide

SAT should compose existing primitives rather than invent another transparency-log format:

- **RFC 9162 Certificate Transparency v2.0** defines Merkle inclusion and consistency proofs for append-only logs. Consistency proves that an earlier tree is a prefix of a later tree; it does not by itself prove that a client has the latest checkpoint. A malicious log can also present inconsistent views, so monitors and cross-client gossip remain important. https://www.rfc-editor.org/rfc/rfc9162.html
- **Sigstore Rekor** provides an API-driven transparency log with signed checkpoints and inclusion/integrity verification. It is useful for durable public evidence receipts, but is not itself a per-memory-state atomic compare-and-append coordinator or a guarantee of currentness. https://docs.sigstore.dev/logging/overview/
- **The Update Framework (TUF)** explicitly addresses rollback and freeze attacks using trusted metadata versions and expiry. SAT can borrow the distinction between an older valid-looking state and a fresh state, while documenting the time/freshness assumptions used by a witness client. https://theupdateframework.github.io/specification/latest/

A Merkle inclusion proof answers “was this commitment included in this checkpoint?” A consistency proof answers “is this checkpoint an append-only extension of that prior checkpoint?” A signed checkpoint answers “did this key sign this tree head?” None alone answers “is this the newest authorized state for this memory partition, and has the witness shown the same state to every verifier?”

## 3. Threat model and trust domains

The candidate memory store may be restored, replaced or replayed by an attacker. The attacker may also control the host, local receipt cache, local journal, local high-water marker and every backup/restore path declared inside the SUT boundary.

For the reference 3-of-4 profile, the witness layer has four independently administered operators. Each witness maintains durable per-state compare-and-append semantics and will not sign two different successor states for the same (state_id, generation, predecessor_digest). A valid transition requires three matching, authenticated witness receipts.

The stated safety assumption is **at most one Byzantine witness**. Three-of-four quorums intersect in at least two witnesses; with at most one Byzantine witness, every pair of conflicting quorums has at least one honest intersection member. This only helps if honest witness implementations durably lock their vote and the operators are genuinely independent. It is not a substitute for a reviewed consensus protocol, authenticated keys or operational audit.

The independence manifest must disclose, for every witness:
- legal operator and accountable organization;
- cloud/account/subscription and administrative principals;
- identity provider and key custody;
- storage and backup/restore domain;
- deployment region and control plane;
- incident response and recovery authority;
- software build/version and policy digest.

Four endpoints, four key IDs or four VM instances under one administrator do not establish four independent witnesses. Publish common-cause cut sets and the number of independent administrative domains.

## 4. State transition protocol under test

### 4.1 Canonical state commitment

For each declared memory partition, the client maintains:
- random, non-PII state_id;
- monotonic generation;
- previous_transition_digest;
- a domain-separated snapshot_root committing to the *complete declared persisted state* (primary store, indexes, summaries, sidecars and any other component inside the rollback boundary);
- transition type: write, correction, revocation, erasure, restore or key/policy rotation;
- idempotency operation_id;
- profile and policy digests;
- opaque erasure/revocation commitments where needed.

Do not put raw memory content or personal identifiers in a public transparency log. A plain hash of a predictable name or fact may be dictionary-attacked. Use random opaque record identifiers or keyed commitments whose key is held outside the witness log; document key rotation and recovery.

### 4.2 Prepare, witness, commit

The proposed write/erase flow is:

1. Compute a candidate complete-state snapshot and durably stage it locally without serving it.
2. Construct a canonical transition that binds the prior generation/digest, candidate snapshot root, operation id, transition kind and policy/profile digests.
3. Submit the same transition to the four independently operated witnesses. Each witness performs compare-and-append against the exact expected predecessor and persists its vote before returning it.
4. Commit only after three matching authenticated receipts are obtained. Preserve the complete signed receipts and transparency inclusion/consistency evidence.
5. Atomically promote the staged local snapshot to the committed state.
6. Before serving memory after restart, compare the local state with a fresh, authenticated quorum view of the current witness state. A receipt cached only beside the data store is not authoritative.

This order cannot make a distributed transaction magically atomic. It makes ambiguity explicit and recoverable.

### 4.3 Unknown outcomes and idempotency

A timeout after submitting a transition is not evidence that it failed. Query the same operation id and exact transition digest. If the witness quorum committed it, finish local promotion. If it did not commit, keep the candidate unservable and reconcile before issuing another mutation. Reusing an operation id with different content must be rejected as an idempotency conflict.

Unknown outcome means no new mutation until reconciliation. Never retry with a new operation id merely because the first response was lost.

### 4.4 Read, restore and recovery

On startup or before a consequential memory read:
- obtain a current witness quorum view;
- verify each witness identity and authorization;
- verify matching state id, generation, predecessor and snapshot root;
- verify the quorum signatures;
- where a transparency log is used, verify inclusion and append-only consistency against a previously trusted checkpoint and compare independent monitors' views;
- compare the complete local snapshot root with the witnessed root.

If the witness is newer than the local store, reconcile from a retained committed journal/snapshot or apply witnessed revocation/erasure transitions, then verify exact root equality. If complete reconciliation is impossible, hold/deny the affected read and mutation. Never serve a store whose currentness cannot be established under the declared policy.

A deployment may define a bounded-staleness mode for low-risk reads, but that is a downgraded assurance profile. It cannot be counted as a T9/T10 read-time pass or as current erasure protection.

### 4.5 Erasure and T10

A successful erasure transition must bind:
- the SUT's own erasure operation and acknowledgement;
- exact before/after state roots;
- the opaque erased-record commitment and dependency closure, as required by the declared memory boundary;
- a new witness generation and predecessor;
- the committed post-erasure snapshot root.

When a pre-erasure backup is restored, the current witness still names the post-erasure state. The SUT must refuse to serve the resurrected record or reconcile it out, then reproduce the witnessed current root. If the old store and old local receipts are restored together but the independent witness is not, T10 must still detect the stale state.

This tests only the declared store/sidecar boundary. It does not prove complete deletion from every external cache, replica, export or archival backup; that remains the separate Data Deletion and Right to Be Forgotten metric.

## 5. Fault-injection campaign

The development case set covers:
- store-only and store-plus-local-cache rollback;
- old receipt replay and same-operation-id payload substitution;
- witness equivocation / split view;
- stale/frozen checkpoints;
- one Byzantine witness and one unavailable witness;
- insufficient quorum;
- crash before quorum, and crash after quorum acknowledgement but before local promotion;
- ambiguous timeout and idempotent retry;
- T10 stale-backup restore after a witnessed erasure;
- common-cause operator, identity, backup and control-plane dependencies.

The benchmark reports safety, liveness, recoverability, independence and overhead separately. A witness that refuses every operation may protect safety but fail availability and benign controls.

## 6. Required measurements

Publish a non-compensatory vector:

1. **Rollback safety:** whether any state older than the current witnessed generation was served.
2. **Equivocation resistance:** whether conflicting successors were ever committed for one predecessor.
3. **Freshness:** checkpoint age and maximum accepted staleness; specify secure time or monotonic-client-state assumptions.
4. **Quorum safety:** whether all accepted commits have the required matching quorum.
5. **Liveness:** percentage of valid writes/erasures completed under the declared witness fault budget.
6. **Recovery correctness:** whether crash/timeout cases converge to exactly one committed outcome without duplicate effects.
7. **Erasure resurrection:** whether T10 restored state is blocked/reconciled after a real SUT deletion.
8. **Independence:** number of genuinely independent operators and common-cause cut sets.
9. **Benign operation:** successful no-op reload, normal post-restart write, legitimate erasure and authorized read.
10. **Cost:** p50/p95 write/read latency, bytes written, witness calls, storage growth and operational recovery effort.

Do not average safety and availability into one score. A stale memory served once is a critical safety failure even if the system has excellent uptime. A witness outage that correctly holds writes is a liveness cost, not a safety pass to hide.

## 7. Proposed research gates

These are SAT research targets, not established thresholds:

- zero stale or erased memory served in the declared holdout;
- zero conflicting same-generation successors committed with at most one Byzantine witness;
- every accepted transition has three matching authenticated receipts under the 3-of-4 profile;
- 100% of ambiguous operation outcomes reconcile by the same operation id without duplicate state transitions;
- a single unavailable or Byzantine witness does not permit stale state and does not prevent valid commits when three honest witnesses are available;
- an insufficient quorum never commits a transition and causes the declared safe hold;
- zero T10 resurrection after restoring the complete store and local receipt cache while witnesses remain outside that rollback domain;
- all four witness independence declarations are independently reviewed; common-cause dependencies are explicit;
- benign reload and valid-write controls pass at the locked policy.

A zero observed failure result must state the independent scenarios, fault combinations and confidence assumptions. Repeated retries of one fixture are not independent scenarios.

## 8. Relationship to existing Luminous work

This protocol is a SAT benchmark and reporting layer, not a second witness implementation:
- MACE already defines MonotonicWitnessStore, checkpoint chaining, erasure/revocation tombstones and a file-backed witness reference: [#13327](https://github.com/Luminous-Dynamics/symthaea/pull/13327) and [#13373](https://github.com/Luminous-Dynamics/symthaea/pull/13373).
- MACE's T9/T10 qualification validates the evaluator evidence but explicitly notes that it does not authenticate a live SUT observer or provide independent remote witness custody: [#13657](https://github.com/Luminous-Dynamics/symthaea/pull/13657).
- SAT's synthetic at-rest lab deliberately demonstrates complete local-domain rollback limits: [#13803](https://github.com/Luminous-Dynamics/symthaea/pull/13803).
- SAT's three-surface federation and write-path/isolation adapters are [#13821](https://github.com/Luminous-Dynamics/symthaea/pull/13821) and [#13841](https://github.com/Luminous-Dynamics/symthaea/pull/13841).

Next implementation should extend the existing MACE witness trait with a genuinely independent remote adapter and a fault-injectable 3-of-4 test fixture; SAT should own the benchmark receipts and qualification rules, not duplicate MACE's storage kernel.

## References

- [RFC 9162: Certificate Transparency Version 2.0](https://www.rfc-editor.org/rfc/rfc9162.html) — Merkle inclusion and consistency proofs; work on split-view detection remains necessary.
- [Sigstore Rekor documentation](https://docs.sigstore.dev/logging/overview/) — signed transparency records and inclusion/integrity verification.
- [The Update Framework Specification](https://theupdateframework.github.io/specification/latest/) — rollback and freeze attack defenses through versioned/expiring metadata.
- [IETF agent-security terminology draft](https://datatracker.ietf.org/doc/html/draft-hcf-bpm-agent-security-benchmark-terminology-00), §4.6.8 and related memory metrics — work in progress.
- [IETF agent-memory-at-rest method](https://datatracker.ietf.org/doc/html/draft-khandelwal-bpm-agent-memory-integrity-00), T9/T10 — work in progress.
