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

The database also enforces:

- unique `(witness_id, attempt_id)`;
- unique `(witness_id, sequence)`;
- positive sequence and epoch;
- exact fixed-length security commitments.

`audit_witness` recomputes the full reservation chain in sequence order and rejects gaps, changed bindings, malformed attempt states, or disagreement with the recorded frontier.

## SQLite transaction profile

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

The configured database parent directory is canonicalized before use, and an already existing target must be a regular file.

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

## Pre-reservation checks

Before burning a sequence, the orchestration verifies cheaply available facts:

- nonzero attempt and release bindings;
- archive path is a regular file rather than a symlink;
- witness policy parses/validates;
- composite verifier digest is explicitly allowed by witness policy;
- witness ID is enrolled;
- signing key matches the enrolled public key.

The expensive evidence verifier runs only after the durable reservation. If it then rejects, the reservation remains consumed by design.

## External anti-rollback boundary

SQLite/WAL is a durable serialization mechanism, not a root-resistant monotonic counter.

A privileged attacker that restores an internally consistent older copy of the entire database can roll back both `high_watermark` and `reservation_head` without violating the local chain.

Therefore V0.1 explicitly exposes `WitnessSequenceFrontierV1` so a stronger layer can retain an independently trusted frontier, for example:

- Xenia signed ledger state;
- TPM NV monotonic state;
- an independently administered witness service;
- transparency/SCITT publication.

At restore/admission time, a local frontier older than such an external trusted head must fail closed.

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
- reservation-row mutation is detected by chain audit.

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
