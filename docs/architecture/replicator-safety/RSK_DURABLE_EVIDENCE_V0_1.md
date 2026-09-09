# Replicator Safety Kernel — Durable Evidence Contract v0.1

Status: **design contract; implementation intentionally deferred until the authority/ledger crates compile cleanly**

This document specifies how the in-memory `symthaea-replicator-ledger` state machine should become durable, tamper-evident, replay-verifiable evidence without introducing a second interpretation of replication authority.

It contains **no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path**.

---

## 1. Governing principle

> **Durability must preserve the exact safety semantics of the reference ledger; persistence is not allowed to mint, widen, recover, or reinterpret authority.**

The durable layer is evidence and transaction machinery. It is not another policy engine.

A reconstructed ledger must either reproduce the reference state exactly or fail closed.

---

## 2. Relationship to the existing Fabrication Kernel evidence pattern

The Fabrication Kernel already establishes the architectural pattern RSK should reuse:

- schema-versioned records;
- domain-separated cryptographic hashing;
- explicit sequence numbers;
- predecessor-hash binding;
- full-journal verification;
- evidence anchors/checkpoints separated from ordinary events.

RSK should follow this shape while using an RSK-specific canonical encoding and domain strings. It should not reuse fabrication audit event schemas directly because descendant lineage and replication-budget transitions have different invariants.

---

## 3. Prerequisite: stable canonical identity bytes

The current RSK identifier types are intentionally opaque:

- `SubjectId`
- `LineageId`
- `GrantId`
- `EvidenceDigest`

Durable evidence **must not** derive protocol bytes from:

- Rust `Debug` output;
- `Hash` implementation behavior;
- memory layout;
- architecture endianness;
- serializer defaults that are not frozen by the protocol;
- map iteration order.

Before code implements this contract, the authority crate should expose a narrow read-only canonical representation such as:

```text
as_bytes(&self) -> &[u8; 32]
```

or an equivalent fixed-width copy operation.

This exposes no authority and does not permit mutation. It only makes the already-opaque identity stable enough for a durable protocol.

---

## 4. Cryptographic primitive and domain separation

RSK v0.1 should use the repository's existing SHA-256 digest implementation rather than adding a new hash primitive.

Every digest purpose receives a unique ASCII domain prefix terminated by `0x00`.

Minimum domains:

```text
symthaea.rsk.event.v1\0
symthaea.rsk.journal.v1\0
symthaea.rsk.checkpoint.v1\0
symthaea.rsk.state.v1\0
symthaea.rsk.safety-case-snapshot.v1\0
symthaea.rsk.dkg-binding.v1\0
```

A digest from one domain must never be accepted in another domain merely because the raw 32 bytes match.

---

## 5. Canonical wire encoding

The evidence protocol must define a deterministic byte representation independent of Rust implementation details.

### 5.1 Integer encoding

- `u8`: one byte.
- `u16`, `u32`, `u64`: unsigned big-endian fixed width.
- No variable-length integer representation in v0.1.

### 5.2 Boolean encoding

- `false = 0x00`
- `true = 0x01`
- all other values invalid.

### 5.3 Fixed digests and identifiers

All RSK identifiers/digests are exactly 32 bytes with no length prefix.

### 5.4 Optional values

- absent: `0x00`
- present: `0x01 || canonical(value)`

### 5.5 Capability sets

`CapabilitySet` is encoded as its `u64` bit representation in unsigned big-endian order.

### 5.6 Enumerations

Every enum receives an explicitly assigned numeric discriminant frozen by this document/implementation. Reordering Rust enum variants must not change the protocol.

Unknown discriminants are rejected, not mapped to an `Other` case unless that case is explicitly designed into a future schema.

### 5.7 Collections

For any bounded vector:

```text
u32_count || item_0 || item_1 || ...
```

Maps/sets are not serialized in native iteration order. If a future record contains one, entries must be sorted by canonical key bytes first.

### 5.8 Strings

The core RSK event protocol should avoid free-form strings. If future records require text:

- UTF-8 only;
- explicit maximum byte length;
- `u32_len || bytes`;
- no Unicode normalization performed implicitly by the serializer.

---

## 6. Durable event envelope

Every persisted mutation becomes exactly one `RskEvidenceEvent` envelope.

Conceptual fields:

```text
schema_version
ledger_epoch
sequence
mutation_id
previous_event_hash
kind
kind_payload
record_hash
```

### Required invariants

1. `schema_version == symthaea.rsk.event.v1`.
2. `ledger_epoch` is constant for one journal.
3. genesis has sequence `0`.
4. every successor sequence is exactly predecessor sequence + 1.
5. every non-genesis event carries the exact predecessor `record_hash`.
6. every non-genesis mutation has a unique `MutationId` within the epoch.
7. `record_hash` is recomputed from all fields except itself.
8. an event whose payload cannot be applied by the reference state machine is invalid even if its hash is correct.

The hash chain proves integrity/order; it does **not** prove that an event was constitutionally valid. Replay validation supplies that second property.

---

## 7. Canonical event kinds

The durable schema must represent the semantic ledger events one-to-one:

1. `RootRegistered`
2. `LineageBranchRegistered`
3. `DescendantCommitted`
4. `SubjectQuarantined`
5. `SubjectRevoked`
6. `LineageQuarantined`
7. `LineageRevoked`
8. `GrantRevoked`

No durable event should exist for a denied authority request. Denials belong in an audit/observability stream, not the authoritative mutation journal, because they do not change ledger state.

### 7.1 `DescendantCommitted` evidence

At minimum preserve:

- mutation ID;
- parent subject ID;
- child subject ID;
- output lineage ID;
- child capability ceiling;
- grant ID and generation used for the commit;
- consumed resource units;
- safety-case digest;
- containment-envelope digest.

The durable implementation should additionally bind the **pre-commit authority cursor** so investigators can show exactly which evaluated snapshot was consumed.

---

## 8. Replay-from-genesis is the source of truth

Loading a durable journal is not deserialization of trusted state.

The required recovery algorithm is:

1. validate schema and bounded sizes;
2. verify canonical decoding;
3. verify sequence continuity;
4. verify predecessor hashes;
5. recompute every record hash;
6. reject duplicate mutation IDs;
7. instantiate a fresh reference ledger from genesis;
8. replay every event through replay-specific state-transition validation;
9. recompute the final state digest;
10. compare to any supplied checkpoint/state digest;
11. only then expose the reconstructed ledger for authority decisions.

Any discrepancy yields a non-operational state. There is no "best effort" recovery that skips an invalid authority event.

---

## 9. Replay-specific validation

Replay must not call public mutation methods in a way that requires re-presenting external authority artifacts that are not part of the journal. Instead, implement a dedicated verifier that proves each recorded mutation was a valid successor of the previous durable state using the evidence captured at commit time.

For `DescendantCommitted`, replay must at least verify:

- parent existed before child;
- child did not already exist;
- subject ancestry was acyclic;
- lineage ancestry was acyclic;
- output lineage was same-lineage or an immediate registered child lineage;
- no ancestor subject/lineage was quarantined or revoked at that sequence;
- referenced grant generation was not already revoked;
- direct-child count remained within its hard ceiling;
- every ancestor subject's descendant/resource/depth ceilings remained satisfied;
- every ancestor lineage's descendant/resource/depth ceilings remained satisfied;
- child capability ceiling was exactly the effective ceiling recorded for the commit and was a subset of the parent subject and lineage ceilings;
- arithmetic did not overflow.

If replay cannot prove one of these properties from durable evidence, the event format is incomplete and must be extended rather than assuming the missing fact.

---

## 10. State digest

A canonical `RskStateDigest` should commit to the complete authority-relevant reconstructed state.

It must include, sorted canonically:

### Ledger metadata

- schema version;
- epoch;
- current sequence;
- current event head hash.

### Every subject

- subject ID;
- parent ID or none;
- lineage ID;
- capability ceiling;
- absolute depth;
- direct-child count;
- subtree-descendant count;
- subtree-resource count;
- negative state;
- authority-scope fields if present.

### Every lineage

- lineage ID;
- parent-lineage ID or none;
- full immutable lineage policy;
- subtree-descendant count;
- subtree-resource count;
- negative state.

### Revoked grants

Every `(grant_id, generation)` pair in canonical sorted order.

A checkpoint omitting an authority-relevant field is not a full RSK state checkpoint.

---

## 11. Checkpoints and compaction

Checkpoints are acceleration/evidence objects; they do not replace the genesis lineage.

A checkpoint contains conceptually:

```text
checkpoint_schema
ledger_epoch
sequence
journal_head_hash
state_digest
previous_checkpoint_hash?
safety_case_snapshot_digest
signer/trust evidence (integration layer)
checkpoint_hash
```

Rules:

- checkpoints are append-only;
- accepting a checkpoint never permits events before it to be silently reinterpreted;
- compaction may archive old events only when their original bytes remain retrievable under retention policy or an independently verifiable archival commitment exists;
- a checkpoint is trusted only after its signer/trust policy is independently verified;
- RSK state reconstructed from a checkpoint plus suffix must equal state reconstructed from genesis plus the full journal.

The last equivalence should become a property test.

---

## 12. Transaction/CAS storage contract

The durable store must expose semantics equivalent to:

```text
compare_and_append(
    expected_epoch,
    expected_sequence,
    expected_head_hash,
    event_bytes
) -> new_cursor | conflict
```

An implementation is safe only if the comparison and append are atomic.

### Prohibited implementation

```text
read cursor
if cursor == expected:
    write event
```

when another writer can interleave between read and write.

### Required guarantees

- no two different events can claim the same `(epoch, sequence)`;
- event bytes are durable before a success response is returned;
- retrying the same `MutationId` is idempotently recognized;
- retrying a different mutation against an old cursor returns conflict;
- partial writes are detectable;
- a failed append does not mutate reconstructed authority state.

---

## 13. Crash-consistency state machine

The implementation and tests must model these boundaries explicitly:

### Crash A — before durable append

Result after restart: event absent, mutation not consumed.

### Crash B — append fully durable, before caller receives success

Result after restart: event present and mutation consumed. Caller retry with same `MutationId` resolves to the already-committed outcome rather than creating a second child.

### Crash C — torn/corrupted append

Result after restart: journal fails canonical/hash verification and remains non-operational until repaired through an explicit recovery protocol. It must not skip the malformed tail and continue granting authority.

### Crash D — checkpoint written before corresponding journal prefix is durable

Prohibited by transaction ordering. Checkpoint must never outrun the authoritative event journal.

---

## 14. Fork handling

Two valid hash chains with the same epoch and predecessor but different successor records represent a ledger fork.

The RSK store must not resolve this by "latest timestamp" or arbitrary last-write-wins.

Minimum safe behavior:

1. detect the conflicting `(epoch, sequence, predecessor)`;
2. mark the authority state **forked / non-operational**;
3. deny new replication authority;
4. preserve both branches as incident evidence;
5. require an explicit external recovery/governance ceremony to establish a new epoch;
6. never delete the losing branch from incident evidence merely because a recovery epoch is chosen.

A fork therefore becomes a containment event, not a consensus heuristic.

---

## 15. Epoch transitions

A new ledger epoch is a constitutional discontinuity and must not be an ordinary local operation.

The future transition evidence should bind:

- old epoch;
- old final journal head;
- old final state digest;
- transition reason;
- recovery/governance evidence digest;
- new epoch ID;
- new genesis policy;
- any explicitly preserved subjects/lineages and their ceilings.

No authority crosses an epoch boundary implicitly.

---

## 16. Safety-case snapshot

The DKG should not ingest the mutable Rust ledger directly. RSK should export a compact, immutable, digest-bound safety-case snapshot.

Suggested fields:

```text
schema
ledger_epoch
ledger_sequence
journal_head_hash
state_digest
subject_or_lineage_scope
risk_class
capability_ceiling_digest
population_budget_summary
resource_budget_summary
containment_envelope_digest
monitoring_evidence_digest
policy_digest
supporting_evidence_root
challenge_evidence_root
valid_from
valid_until
snapshot_digest
```

The snapshot reports the state/evidence boundary. It does not itself grant replication authority.

---

## 17. Mycelix epistemic-DKG binding

Mycelix can represent the safety case as immutable claims/evidence relations without becoming the runtime hard dependency.

Conceptual claim classes:

- `RskStateCheckpointClaim`
- `ReplicationSafetyCaseClaim`
- `ContainmentEnvelopeClaim`
- `MonitoringHealthClaim`
- `PolicyVersionClaim`
- `ReplicationIncidentClaim`
- `ReplicationRevocationClaim`

Conceptual edges:

- `supports`
- `challenges`
- `supersedes`
- `replicates`
- `derived_from`
- `bound_to_checkpoint`

Runtime rule:

> The hard RSK kernel consumes a previously verified compact snapshot/digest. It must not require live DKG availability to enforce quarantine, budgets, monitor failure, lineage validity, or revocation.

This preserves a small trusted computing base while still gaining Mycelix provenance, contradiction tracking, independent replication, and evolving consensus.

---

## 18. Evidence freshness

Every safety-case snapshot must have explicit validity bounds.

Unknown/stale evidence cannot be converted into positive authority by availability failure.

If DKG connectivity is unavailable:

- an already-verified snapshot may remain usable only until its existing expiry;
- no implicit expiry extension occurs;
- negative local facts discovered after snapshot creation still dominate;
- once the snapshot expires, new replication authority fails closed until fresh evidence is verified.

---

## 19. Limits and resource-exhaustion defense

Durable evidence parsing is part of the trusted safety boundary. Every decoder must have explicit maxima for:

- journal event count per load/replay operation;
- encoded event size;
- checkpoint size;
- branch count held during fork investigation;
- DKG proof/snapshot attachment size;
- lineage depth;
- subject count;
- lineage count.

Inputs exceeding limits are rejected before unbounded allocation.

---

## 20. Required fault-injection matrix

Before the durable implementation can graduate from draft, tests should cover at least:

| Fault | Required result |
|---|---|
| duplicate delivery, same mutation | idempotent already-committed outcome |
| duplicate delivery, different mutation | conflict/deny |
| stale cursor | conflict/deny |
| stale head hash | conflict/deny |
| wrong epoch | conflict/deny |
| event byte mutation | hash verification failure |
| predecessor substitution | chain verification failure |
| event deletion | sequence/predecessor failure |
| event reordering | sequence/predecessor failure |
| unknown enum discriminant | canonical decode failure |
| non-canonical encoding | canonical decode failure |
| counter overflow | replay failure |
| lineage cycle | replay failure |
| subject cycle | replay failure |
| branch policy widening | replay failure |
| revoked ancestor grant | replay failure |
| revoked/quarantined ancestor | replay failure |
| capability re-expansion | replay failure |
| crash before append | mutation absent after recovery |
| crash after append before acknowledgement | exactly one committed mutation |
| corrupted tail | non-operational, no silent truncation |
| fork at same predecessor | forked/non-operational |
| checkpoint/state mismatch | checkpoint rejected |
| checkpoint+suffix != full replay | checkpoint rejected |
| expired safety-case snapshot | deny |
| DKG unavailable before snapshot expiry | existing snapshot only |
| DKG unavailable after snapshot expiry | deny |

---

## 21. Implementation sequence

Do not implement the durable layer as one large PR.

### PR A — canonical primitives

- read-only canonical bytes for RSK IDs/digests;
- frozen enum discriminants;
- standalone canonical encoder/decoder tests;
- malformed/non-canonical input tests.

### PR B — hash-chained evidence journal

- `RskEvidenceEvent`;
- domain-separated record hashing;
- sequence/predecessor verification;
- strict limits;
- mutation replay detection.

### PR C — replay verifier

- reconstruct ledger from genesis;
- compare reconstructed snapshots to live semantic ledger fixtures;
- corruption/fork tests.

### PR D — transactional store

- abstract CAS trait;
- file/SQLite or other durable adapter only after the trait semantics are tested;
- crash-consistency harness.

### PR E — checkpoint/export

- canonical full state digest;
- signed/checkpoint-ready envelope;
- full replay == checkpoint+suffix property.

### PR F — Mycelix epistemic bridge

- compact safety-case snapshot;
- DKG claim/edge mapping;
- offline/expiry behavior;
- DKG remains outside the hard runtime dependency.

---

## 22. Promotion gates

The durable RSK evidence layer should remain experimental until all of these are demonstrated:

1. authority and ledger crates compile/test cleanly;
2. canonical format has versioned golden vectors;
3. cross-process encode/decode reproduces identical bytes;
4. corrupted or non-canonical events never replay;
5. replay from genesis reproduces live reference state;
6. crash tests demonstrate exactly-once mutation semantics;
7. fork tests fail closed;
8. capability attenuation survives replay/checkpoint recovery;
9. ancestral population/resource/depth constraints survive replay/checkpoint recovery;
10. revocation/quarantine survives replay/checkpoint recovery;
11. expired/stale safety evidence cannot become authority;
12. checkpoint verification is independent of Symthaea's higher-level reasoning;
13. Mycelix DKG loss cannot disable the hard local safety invariants.

Only after these gates should RSK durable evidence be considered suitable as a foundation for higher-consequence autonomous fabrication governance.
