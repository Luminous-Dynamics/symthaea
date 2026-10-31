# BASALT trusted-state atomicity and anti-rollback — 2026-11-01

## Purpose

This specification isolates a missing trust boundary between the integrity-only epoch seal in PR #9558 and the authenticated receipt/transparency protocol in issue #9560.

The central distinction is:

- **Atomicity / crash consistency:** a state transition is all-or-nothing across process failure and the declared storage failure model.
- **Anti-rollback:** an attacker cannot restore an older but internally consistent state and make it appear current.
- **Freshness:** the verifier has evidence that its checkpoint view is sufficiently current under an explicit policy.
- **Non-equivocation:** two different receipts cannot both be accepted for the same source generation or log position.

A local database transaction can provide the first property without providing the other three. BASALT must report them independently.

This document is a protocol and benchmark design, not an implementation or runtime qualification.

## Threat model and claim boundaries

Assume an attacker may:
- replay an older valid seal, DSSE envelope, checkpoint, cache or complete local filesystem/VM snapshot;
- race multiple verifier processes or submit conflicting receipts for one source generation;
- kill the verifier at any instruction boundary;
- cause process/kernel crashes, storage-full conditions, failed synchronization, log outages or witness outages;
- present split log views or a checkpoint that is validly signed but stale;
- rotate or revoke signer keys under a policy transition;
- cause a host reboot, monotonic-clock reset or boot-identity mismatch.

The local storage backend may be crash-faulty, but should not be treated as maliciously rollback-resistant merely because it uses ACID transactions or fsync.

The trust anchor is outside the local receipt directory and outside any attacker-restorable local snapshot. Candidate anchors are:
1. a remote, linearizable compare-and-advance service with independently controlled durable state and witness/replication policy;
2. a hardware-backed monotonic anchor with a separately proven recovery protocol for the gap between counter advancement and data durability;
3. a federated witness quorum whose consistency and anti-equivocation assumptions are stated explicitly.

A local file, SQLite database, TPM quote without a monotonic state contract, signed checkpoint, or predecessor hash alone is not sufficient to assert current anti-rollback state.

## Independent verdict dimensions

Every verification result must carry separate fields. Suggested v1 vocabulary:

| Dimension | Values | Meaning |
|---|---|---|
| payload_binding | valid / invalid | Exact seal and ledger subject digests match |
| envelope_signature | valid / invalid / unknown | DSSE PAE signature verifies under a candidate key |
| signer_authorization | authorized / unauthorized / unknown | Verified identity is permitted for this host/source/profile |
| log_inclusion | valid / invalid / unknown | Exact receipt is included in the identified log |
| checkpoint_authenticity | valid / invalid / unknown | Signature and log origin are trusted |
| checkpoint_consistency | valid / invalid / unknown | New tree extends the previously trusted tree |
| witness_agreement | quorum / insufficient / equivocation / unknown | Required independent witnesses agree |
| freshness | current_under_policy / stale / unknown | Latest-view policy can be established |
| local_atomicity | committed / not_committed / indeterminate | State transaction is all-or-nothing and recoverable |
| source_monotonicity | advanced / duplicate_same_receipt / gap / equivocation / invalid | Source generation transition is valid |
| anti_rollback | protected / detected / unknown | An independent anchor detects restoration of stale state |
| qualification | always false in this work item | No transaction or crypto result qualifies the observer or close coordinator |

Do not collapse these into one boolean. A locally committed source cursor with an unanchored local database may have local_atomicity=committed while anti_rollback=unknown.

## State model

### Stable source key

Use a stable provisioned host/device identity plus:
- source ID;
- observer-profile digest;
- partition ID.

The boot incarnation is required per epoch and binds kernel/build identity, boot ID, and a boot-attestation or provisioning reference where available. Do not use a mutable boot ID to reset durable source generation. A reboot changes the monotonic-clock origin, so monotonic timestamps are comparable only within the same boot incarnation. A boot transition requires a policy-authorized transition or remains unknown.

Key rotation, observer restart, kernel upgrade, host reboot and partition remapping do not silently create a fresh trust lineage. If policy authorizes a changed observer profile or boot, the transition must bind the old and new identities and preserve the durable generation history.

### Source cursor

For each stable source key, persist:
- last accepted generation;
- last accepted seal digest;
- last accepted DSSE envelope/receipt digest;
- observer-profile digest and current boot-incarnation reference;
- state revision.

A candidate is a successor only when:
- generation equals previous_generation + 1 without overflow;
- previous_seal_digest exactly equals the persisted last seal digest;
- the stable source key matches;
- any profile or boot transition is explicitly authorized;
- the candidate receipt digest is not already bound to a different source identity.

At the already accepted generation:
- exact same receipt digest and same seal digest => idempotent duplicate;
- any different receipt or seal digest => equivocation; reject and preserve evidence.

A skipped generation is not repaired by guessing. It is a gap requiring reconciliation or a separately authorized recovery event. Generation counters use checked arithmetic; overflow is a terminal invalid state, never wrapping or saturating.

### Log cursor

For each configured log origin, persist:
- immutable log ID/origin identity;
- accepted tree size and root hash;
- exact accepted checkpoint digest and signature identity;
- witness-policy version and accepted witness set/quorum result;
- state revision.

Rules:
- smaller tree size => stale/rollback rejection;
- equal size and same root/checkpoint identity => idempotent observation;
- equal size and different root => fork/equivocation;
- larger size => require a valid consistency proof from the persisted trusted checkpoint;
- changed log origin or trust-root set => explicit policy transition; never silently replace the log identity;
- a valid signed checkpoint does not establish that it is the globally latest checkpoint without a trusted prior state and/or independent monitoring/witness policy.

The source cursor and log cursor are separate logical dimensions but are committed together for one accepted receipt.

## Transition protocol

### Phase A — prepare and verify external evidence

1. Read a versioned state snapshot containing the expected global/log revision, log cursor, source cursor and policy versions.
2. Verify the exact DSSE payload bytes and payload type using DSSE pre-authentication encoding. The envelope key ID is a lookup hint only.
3. Verify signer identity and authorization for the precise stable source key and observer profile.
4. Verify the in-toto Statement subject digests and BASALT predicate type/schema.
5. Verify the exact receipt's transparency-log inclusion proof, checkpoint signature/origin, witness policy and consistency proof from the snapshot's trusted checkpoint.
6. Validate the source-generation successor, predecessor digest, boot transition and policy-version constraints.
7. Derive the proposed source cursor, log cursor and accepted-receipt record. Do not mutate trusted state during this phase.

### Phase B — atomic compare-and-advance

The backend must expose one linearizable operation conceptually equivalent to:

    compare_and_advance(
        expected_state_revision,
        expected_log_cursor,
        expected_source_cursor,
        proposed_log_cursor,
        proposed_source_cursor,
        accepted_receipt_record
    )

The operation must atomically:
- compare the current revision and both cursors to the expected values;
- insert the immutable accepted-receipt/idempotency record;
- advance the source cursor;
- advance the log cursor;
- increment the durable state revision.

All updates commit or none do. A storage backend that updates the source cursor and log cursor separately does not conform.

Possible outcomes:
- committed;
- already_committed_exact_receipt;
- compare_conflict(current_snapshot);
- rejected_equivocation;
- rejected_stale_or_forked_checkpoint;
- rejected_invalid_transition;
- unavailable_before_commit;
- commit_outcome_indeterminate.

On compare conflict, reread the entire state and rerun the cryptographic and policy checks against that state. Never blindly retry a write computed against a stale checkpoint.

### Phase C — crash and retry recovery

A transparency-log append is an external effect and cannot generally participate in the local database transaction. Do not claim distributed exactly-once semantics unless the log adapter provides an idempotent append key or exact-entry lookup.

The receipt identity is the digest of the exact immutable DSSE envelope bytes. Retain the envelope unchanged across retries.

If append succeeds but local state has not advanced:
1. restart from the same envelope digest;
2. query the log for the exact existing entry using the same idempotency key or exact digest;
3. verify its inclusion and checkpoint evidence again against the current trusted state;
4. perform the local compare-and-advance transaction;
5. if exact-entry lookup/deduplication is unavailable, return external_append_outcome_unknown or disclose duplicate log entries; do not silently claim exactly once.

If the transaction response is lost:
1. reopen/recover the durable store;
2. query the receipt idempotency record and both cursors;
3. return already_committed_exact_receipt only if the exact receipt, source generation, seal digest and checkpoint advancement are all present;
4. if none are present, retry the full verification and CAS;
5. if the database or external anchor disagrees, report indeterminate and fail closed.

### Phase D — anti-rollback anchor

The local CAS transaction proves only local atomicity under that backend's declared crash model.

For production anti-rollback, an independently protected anchor must bind at least:
- stable host/device identity;
- monotonic state revision;
- latest accepted source generation/seal digest;
- latest accepted log origin/tree size/root;
- accepted receipt digest or an immutable transaction digest.

The anchor must itself provide linearizable compare-and-advance or witness-backed non-equivocation. Its authoritative state record must durably bind the new source cursor, new log cursor, exact receipt digest, policy versions and monotonically increasing revision as one commit unit. It must retain enough immutable state (or a content-addressed, independently durable state record) to recover the accepted cursors after a client crash.

For a remote-anchor deployment, the remote CAS is the authoritative acceptance/commit point. The local database is a materialized cache, not a second authority: after the remote CAS commits, the client updates its cache; if it crashes before doing so, it reconstructs the cache from the exact committed remote state record. If the remote CAS response is lost, the client queries by the exact receipt digest/idempotency key and returns already_committed only when the complete committed state record matches. If the remote service cannot be queried, the result is indeterminate/unknown, not a guessed success or failure. A local cache that is ahead of the remote record is never promoted.

Do not try to make a remote quorum and a local SQLite transaction appear atomic through ordering alone. The remote commit record must contain both logical cursors; the local database is derived afterward. The external transparency-log append remains a prior, independently retryable effect and still requires exact-entry lookup or idempotency support.

If the local database is restored to an older snapshot:
- anchor newer than local state => rollback detected;
- local state newer than anchor => inconsistent/uncommitted state, unknown or rejected;
- anchor unavailable or no anchor provisioned => anti_rollback=unknown;
- never infer protection from a local backup, journal, file lock, hash chain or signed checkpoint alone.

A TPM counter and a filesystem/database commit are two separate durability domains. Do not assume they form one atomic transaction. Hardware-backed deployment requires explicit recovery for counter-first and data-first interruption; absent a proven protocol, the hardware mode remains unqualified.

## API and backend conformance

A backend adapter should expose:
- read_consistent_snapshot();
- compare_and_advance(expected, proposed);
- lookup_receipt(exact_envelope_digest);
- verify_external_anchor(local_state);
- recover_after_indeterminate_commit().

The snapshot must be internally consistent across revision, log cursor and all source cursors relevant to the transaction. The CAS must be linearizable for competing writers. The backend must document process-crash, OS-crash, power-loss, filesystem, controller-cache, network-partition and restore assumptions separately.

An in-memory model is useful for state-machine/property tests but cannot qualify durability. A local durable database can qualify crash consistency for its pinned storage configuration, but not malicious snapshot rollback. In a remote-anchor deployment, test the remote linearizable CAS as the single authoritative commit and treat local state as a recoverable cache. A hardware anchor requires a separate device-specific transaction/recovery proof; this benchmark does not assume TPM counter + filesystem atomicity.

## Failure taxonomy

Use specific outcomes rather than a generic false:
- invalid_payload;
- invalid_signature;
- signer_unauthorized;
- invalid_inclusion;
- invalid_checkpoint_signature;
- wrong_log_origin;
- stale_checkpoint;
- checkpoint_fork;
- invalid_consistency_proof;
- witness_quorum_insufficient;
- witness_equivocation;
- source_generation_gap;
- source_generation_equivocation;
- predecessor_mismatch;
- unauthorized_boot_transition;
- compare_conflict;
- storage_unavailable;
- durability_failed;
- commit_outcome_indeterminate;
- external_append_outcome_unknown;
- rollback_detected;
- anti_rollback_unknown;
- freshness_unknown.

Errors must retain the candidate receipt digest, observed trusted revision and relevant proof/checkpoint digests without leaking signing secrets.

## Benchmark design

The v1 benchmark contains clean, idempotent, conflict, fork, crash, restore, offline and policy-transition scenarios. It must test crash points at every durable write boundary, not merely kill the process at arbitrary times. A deterministic backend fault-injection interface should expose hooks before write, after each write, before commit, during commit, after durable commit and before acknowledgement.

Mandatory safety gates:
- zero partial state advances across the source cursor, log cursor and receipt record;
- zero acceptance of two different digests for one source generation;
- zero stale checkpoint or same-size fork acceptance;
- 100% exact-receipt retry idempotence after append/local-commit interruption when the log adapter supports exact lookup/idempotency;
- zero anti-rollback promotion from local storage alone;
- explicit unknown when external freshness or rollback state cannot be established;
- zero generation reset on key rotation, reboot or observer restart;
- zero close-coordinator or observer qualification promotion.

Count host/device provisioning lineage × source/profile × boot transition × backend/configuration as the independent unit. Events and repeated retries within one scenario are not independent samples. Preregister mutations and reserve separate device/provisioning lineages for confirmatory testing.

Efficiency is scored only after every safety gate passes. Initial research targets (not measured):
- local durable CAS p95 <= 5 ms on a pinned reference NVMe configuration with one writer;
- local durable CAS p95 <= 25 ms with 16 competing verifiers;
- exact-retry recovery p95 <= 100 ms when the local store is healthy.
Measure external log/witness latency separately. Record p50/p95/p99, throughput, conflict rate, write amplification, bytes per accepted epoch, recovery time and resource use. Do not optimize by weakening fsync, quorum, consistency or anti-rollback checks.

## Industry positioning

- DSSE supplies the envelope and PAE signature input; signer authorization remains independent.
- in-toto Statement v1 binds immutable subjects to a typed predicate.
- Rekor-style transparency provides verifiable inclusion/checkpoints/consistency; monitors and witnesses are needed to expose split views and stale views.
- TUF provides a useful rollback/freeze and trusted-metadata model.
- SQLite illustrates how ACID/atomic commit and WAL durability levels are distinct from malicious rollback resistance; the configured synchronous mode and filesystem matter.

BASALT's differentiated contribution is the application-level transition that atomically couples source generation, log checkpoint and exact receipt identity while preserving explicit unknown/fork/rollback outcomes. These industry components do not prove producer quiescence, consumer drain or kernel event completeness.

## Current status and non-claims

This work item adds a protocol specification and benchmark definition only. It does not implement DSSE, a transparency client, a state backend, a trusted anchor, crash injection, or a live BPF observer. No runtime, performance, anti-rollback, production or SOTA result is claimed.

## References

- BASALT receipt/transparency protocol: https://github.com/Luminous-Dynamics/symthaea/issues/9560
- DSSE protocol / envelope: https://github.com/secure-systems-lab/dsse
- in-toto Statement v1: https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- Sigstore/Rekor overview: https://docs.sigstore.dev/logging/overview/
- Rekor checkpoint and consistency model: https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md
- TUF specification: https://theupdateframework.github.io/specification/latest/
- SQLite atomic commit: https://www.sqlite.org/atomiccommit.html
- SQLite WAL: https://www.sqlite.org/wal.html
- SQLite durability modes: https://www.sqlite.org/pragma.html
