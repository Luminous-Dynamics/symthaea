# Agency witness frontier SQLite guard V0.1

## Purpose

This profile supplies the concrete local-history boundary required by the ancestry-aware recovery classifier in #452 without weakening #449's ownership of witness-sequence integrity.

The key problem is a point-in-time race. A detached audit snapshot can become stale immediately:

```text
audit local frontier = H2
        ↓
external anchor also = H2
        ↓
concurrent process reserves H3
        ↓
old snapshot says PublishAllowed
```

That is not acceptable for a chronology gate whose policy says local-ahead state must be re-anchored before publication.

V0.1 therefore uses a SQLite writer barrier:

```text
open exact #449 DB
        ↓
BEGIN IMMEDIATE
        ↓
#449 full-chain audit
        ↓
#449 current frontier statement
        ↓
read minimal historical sequence -> reservation-head index
        ↓
#452 ancestry classification
        ↓
opaque guarded permit
        ↓
anchor / publish while guard remains alive
        ↓
ROLLBACK read-only guard + release writer barrier
```

## Audit ownership

The adapter does **not** reproduce #449's reservation commitment algorithm or Signed/Reserved state audit.

After the writer reservation is held, it invokes:

- `SqliteWitnessSequenceStore::audit_witness`;
- `SqliteWitnessSequenceStore::frontier_statement`.

Those remain authoritative for the complete local chain and attempt-state validation.

The adapter performs only the storage-specific reads needed to serve #452's historical-prefix contract:

- current `(high_watermark, reservation_head)`;
- ordered `(sequence, reservation_digest)` rows.

It requires contiguous sequences, nonzero heads, exact count/high-watermark agreement, and exact final-head agreement with the #449 audit.

## Why the writer barrier precedes audit

An earlier draft audited first and acquired the writer reservation second. That was rejected.

A malicious or concurrent writer could change persisted Signed attempt state between those operations without necessarily changing the reservation frontier. The final design acquires `BEGIN IMMEDIATE` first. In WAL mode, #449's separate read connections can still audit the committed database while the guard prevents another writer from advancing or modifying local witness state.

## Schema identity

The adapter is intentionally SQLite/#449-specific. Under the guard it verifies the claimed #449 database identity:

- SQLite `application_id = 0x53595731` (`SYW1`);
- `user_version = 1`;
- exact user schema surface of only:
  - `witness_sequence_attempts`;
  - `witness_sequence_frontier`.

Unexpected tables, indexes, triggers, or other user schema objects fail closed.

This check is not a substitute for #449's full schema/state validation; it prevents the concrete adapter from reading a different database shape after the writer barrier is acquired.

## Historical ancestry

The guarded view implements #452's `LocalWitnessFrontierHistory` contract from one immutable in-memory head vector built under the barrier.

For an external trusted frontier at sequence `N`, #452 therefore receives the exact local reservation head stored at sequence `N` while no local writer can change it.

The two important cases remain:

```text
trusted A -> B
local   A -> B -> C
```

=> `LocalAheadVerifiedDescendant` => `AnchorRequired`

versus:

```text
trusted A -> B
local   A -> X -> C
```

=> `DivergentTrustedPrefix` => `Contained`.

## Opaque permits

Classification exposes two future-integration permit types.

### `GuardedPublicationPermitV1`

Available only for `PublishAllowed`, which in #452 currently means exact `AnchoredCurrent` equality.

The permit borrows the SQLite guard and carries the exact guarded current frontier. A reviewed publication adapter can require this type instead of accepting a copied boolean/disposition.

### `GuardedAnchorPermitV1`

Available only when #452 returns `AnchorRequired` **and a concrete local frontier exists**. In V0.1 that means:

- `InitialAnchorRequired`;
- `LocalAheadVerifiedDescendant`.

The permit borrows the guard and carries the exact nonempty local frontier plus recovery relation. Divergence/rollback states cannot obtain it.

`EmptyUnanchored` deliberately does **not** obtain an anchor permit: there is no sequence/head statement to write yet and #452's external V1 claim format begins at a nonzero high watermark. The first durable witness reservation creates the initial anchorable frontier.

These types do not make it impossible for arbitrary unreviewed code to ignore the protocol; they provide a structural boundary for the reviewed Xenia/transparency adapter that comes next.

## Writer-barrier availability tradeoff

Holding `BEGIN IMMEDIATE` prevents #449 sequence writers from reserving or persisting new witness attempts in the same database.

That is intentional for the bounded anchor/publication critical section, but external calls must therefore have hard deadlines.

A future anchor writer must not hold the guard indefinitely while waiting on an unavailable network service. It should:

1. use a bounded external operation;
2. classify timeout/ambiguous append as `OutcomeUnknown`;
3. release the local writer guard;
4. reconcile the external source by idempotency key/source sequence before retrying;
5. never blindly append again after an ambiguous result.

Availability loss is preferable to duplicate/forked chronology.

## Tests authored

The source suite covers:

- exact current external anchor produces only a guarded publication permit;
- a trusted sequence-2 prefix of local sequence 3 is proven as `LocalAheadVerifiedDescendant` and produces only an anchor permit;
- an empty unanchored witness domain produces neither publication nor anchor permit;
- a second #449 writer cannot reserve sequence 2 while the guard is held and succeeds after guard release.

The parent #452 suite separately covers rollback, same-height divergence, wrong-prefix divergence, missing local history, and no-anchor bootstrap semantics.

## Deliberate non-claims

V0.1 does not:

- authenticate or establish freshness of an external anchor;
- perform a Xenia/TPM/SCITT append;
- make a local SQLite database root-resistant against coherent rollback;
- guarantee availability while an external operation holds the writer barrier;
- create any execution authority.

The next concrete layer should consume `GuardedAnchorPermitV1` and implement a bounded, idempotent, reconciliation-safe Xenia anchor operation. Only after re-verifying the resulting fresh Xenia frontier should publication obtain `GuardedPublicationPermitV1`.
