# Agency witness sequence CAS V0.1

## Purpose

This profile closes the durable monotonicity gap left deliberately open by the qualification-witness service.

A witness attestation already signs `witness_sequence`, but a caller-supplied number is not monotonic state. V0.1 therefore moves sequence ownership behind a file-backed SQLite reservation store and makes the production-shaped order:

```text
stable attempt id
    + exact witness / policy / verifier / release bindings
        ↓
BEGIN IMMEDIATE
        ↓
reserve sequence N + advance immutable reservation frontier
        ↓ COMMIT (WAL + synchronous=FULL)
run exact evidence verifier
        ↓
sign exact acceptance with reserved N
        ↓
BEGIN IMMEDIATE
        ↓
persist exact attestation commitment + bytes
        ↓ COMMIT
release attestation to caller
```

The central invariant is:

> A reserved sequence is never released or reused. A crash may consume availability; it must not create duplicate notarization identity.

## Attempt idempotency

Each logical notarization attempt carries a nonzero 128-bit `attempt_id`.

The durable binding commits:

- attempt ID;
- witness ID;
- witness-policy epoch;
- independently supplied archive SHA-256;
- independently supplied Git HEAD;
- independently supplied Git tree;
- composite verifier-runtime digest from the witness service;
- exact witness-policy digest.

If the same attempt ID is retried with the same bindings, the store returns the same sequence and reservation commitment.

If the same attempt ID is reused with different bindings, it fails closed.

This means retries after process death do not allocate a second sequence for the same logical attempt.

## Reservation frontier

For each witness identity, V0.1 maintains:

```text
high_watermark
reservation_head
```

Every new reservation commits the previous reservation head plus the exact attempt binding and sequence under the domain:

`symthaea.qualification-witness.sequence-reservation.v1\0`

The initialized database schema enforces:

- unique `(witness_id, attempt_id)`;
- unique `(witness_id, sequence)`;
- positive sequence and epoch;
- exact fixed-length security commitments.

`audit_witness` recomputes the full reservation chain in sequence order and rejects gaps, changed bindings, malformed attempt states, or disagreement with the recorded frontier.

### Reservation order is not publication order

`witness_sequence` records durable reservation order.

A rejected or ambiguous attempt can remain permanently `Reserved` at sequence `N` while a later independent attempt becomes `Signed` at `N+1`. V0.1 intentionally permits this because reclaiming an ambiguous sequence would weaken uniqueness.

Therefore:

- gaps in *published signed attestations* are valid;
- gaps in the durable reservation chain are not valid;
- a sequence number must never be interpreted as proof that every lower sequence was successfully published.

## SQLite transaction and identity profile

V0.1 uses one file-backed SQLite database and opens a fresh connection per operation.

The connection profile requires:

- `SQLITE_OPEN_READ_WRITE`;
- `SQLITE_OPEN_CREATE`;
- `SQLITE_OPEN_NO_MUTEX` because each connection is owned by one thread;
- `SQLITE_OPEN_NOFOLLOW` so the final database path cannot be a symlink;
- extended result codes;
- `PRAGMA journal_mode=WAL`;
- `PRAGMA synchronous=FULL`;
- `PRAGMA foreign_keys=ON`;
- `PRAGMA trusted_schema=OFF`;
- bounded busy timeout.

The service reads the effective PRAGMA values back and rejects the connection unless `synchronous`, foreign-key enforcement, and trusted-schema state match the required profile.

The configured database parent directory is canonicalized before use, and an already existing target must be a regular file. SQLite additionally receives `SQLITE_OPEN_NOFOLLOW` for the final path component.

The database claims an explicit V1 identity with:

- SQLite `application_id = 0x53595731` (`SYW1`);
- `user_version = 1`.

A truly empty, unclaimed database can be initialized. A non-empty unclaimed database is never adopted. Once claimed, the database must retain the exact V1 user-schema *object surface*: the two expected tables and no extra user tables, views, indexes, or triggers. A missing table or injected trigger therefore fails closed rather than being silently normalized into fresh state.

This schema-object check is corruption/wrong-database defense. It is not a claim that SQLite metadata is protected from a privileged attacker capable of coherently rewriting the entire database.

Sequence allocation and signed-state persistence both use `BEGIN IMMEDIATE` transactions. Concurrent writers therefore serialize before reading/updating the frontier.

## Crash cuts

### Crash before reservation commit

No reservation exists. A retry may allocate the next sequence.

### Crash after reservation commit but before verification

The sequence remains `Reserved`. A retry of the same attempt gets the same sequence.

### Verification rejects after reservation

The sequence remains reserved. V0.1 intentionally does not release it.

This can waste sequence numbers under repeated invalid requests, but sequence space is 63-bit in the SQLite profile and uniqueness is preferred over reclamation ambiguity. Request admission/rate limiting belongs outside this evidence primitive.

### Crash after signature generation but before signed-state commit

The reservation remains. Ed25519 signing is deterministic for the same key/transcript, so retrying the same attempt regenerates the same attestation for the same reserved sequence and then persists it.

No signature is returned by `verify_reserve_sign_persist_v1` until signed-state persistence succeeds.

### Crash after signed-state commit but before caller receives result

A retry reuses the same reservation, reruns verification, regenerates the exact attestation, and requires the persisted attestation bytes/commitment to agree before returning.

## Persisted attestation state

A signed attempt stores:

- exact acceptance digest;
- domain-separated digest of the serialized witness attestation;
- exact serialized witness attestation bytes.

Raw qualification archive contents, TPM material, application payloads, secrets, and signing private keys are not stored in this database.

The persistence mutation is private to the crate and consumes a `VerifiedThenSignedQualificationV1` produced by the #445 witness-service boundary. External callers cannot mark an arbitrary reservation as signed through the public API.

The local chain audit checks the structure and stored attestation commitment. It does not replace cryptographic witness verification: consumers of an attestation must still validate it under #439's witness policy/signature semantics.

## Pre-reservation checks

Before burning a sequence, the orchestration verifies cheaply available facts:

- nonzero attempt and release bindings;
- archive path is a regular file rather than a symlink;
- witness policy parses/validates;
- composite verifier digest is explicitly allowed by witness policy;
- witness ID is enrolled;
- signing key matches the enrolled public key.

The expensive evidence verifier runs only after the durable reservation. If it then rejects, the reservation remains consumed by design.

## Canonical external frontier statement

A bare head value is too easy for an integration to associate with the wrong witness. V0.1 therefore exposes `WitnessSequenceFrontierStatementV1`, which commits under:

`symthaea.qualification-witness.sequence-frontier.v1\0`

and contains:

```text
schema version
witness_id
high_watermark
reservation_head
```

The statement's domain-separated digest is the intended value for a later Xenia/TPM/independent-witness/transparency anchor.

## External anti-rollback boundary

SQLite/WAL is a durable serialization mechanism, not a root-resistant monotonic counter.

A privileged attacker that restores an internally consistent older copy of the entire database can roll back both `high_watermark` and `reservation_head` without violating the local chain.

Therefore a stronger layer should retain an independently trusted `WitnessSequenceFrontierStatementV1` commitment, for example in:

- Xenia signed ledger state;
- TPM NV monotonic state;
- an independently administered witness service;
- transparency/SCITT publication.

A later recovery classifier should treat the relationship asymmetrically:

```text
local < trusted external frontier
    = rollback / fail closed

local == trusted external frontier
    = anchored current state

local > trusted external frontier
    = locally durable but not yet externally anchored
```

Local-ahead state must never be rolled back merely to match an older anchor. Instead, publication/release should remain contained until the newer local frontier is safely anchored.

Until that anchor exists, the correct claim is:

> V0.1 prevents sequence reuse across ordinary process crashes and concurrent writers sharing one non-rolled-back SQLite database.

It does not claim Byzantine/root-resistant rollback protection.

## Concurrency and hostile tests

The source suite covers:

- retry of one attempt returns sequence 1 twice without advancing the frontier;
- same attempt ID with changed security binding is rejected;
- different attempts receive distinct monotonically increasing sequences;
- two independent SQLite connections racing different attempts serialize to sequences 1 and 2;
- crash/reopen after reservation does not release sequence 1;
- database symlink target is rejected;
- claimed database missing a required table is rejected;
- extra trigger in a claimed database is rejected;
- unrelated non-empty database is not adopted;
- reservation-row mutation is detected by chain audit;
- the external frontier statement binds witness identity, high watermark, and reservation head.

The integration fixture additionally exercises the public composition:

```text
fixture verifier runtime
    ↓
#445 verify-then-sign service
    ↓
sequence reservation
    ↓
Ed25519 witness attestation
    ↓
durable signed state
    ↓
retry same attempt
```

and requires the retry to return the same sequence, reservation digest, attestation digest, and exact signature before a distinct attempt may advance to sequence 2.

The fixture intentionally has a different verifier-runtime digest from production because `require_nix_store_paths=false`; it cannot satisfy a production witness policy accidentally.

## Non-authority statement

Witness sequence state is evidence chronology only.

Neither a reservation, a persisted signature, an audited frontier, nor a future transparency anchor creates or amplifies Symthaea execution authority.
