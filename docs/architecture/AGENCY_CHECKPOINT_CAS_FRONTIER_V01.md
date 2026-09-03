# Agency Checkpoint CAS Frontier v0.1

Status: **draft / unqualified**

## Purpose

Close two remaining ordering problems in the bounded systemd witness:

1. establish an anti-rollback checkpoint **before** external authority is requested;
2. prevent two stale processes from both publishing successors from the same trusted checkpoint.

## Generation-zero ordering

Before requesting Xenia or other consequential authority:

```text
CapabilityGrant
    -> empty GrantAccount
    -> GrantAccountCheckpoint(sequence=0)
    -> atomic external CAS: None -> checkpoint-0
    -> trusted CheckpointHead(0)
    -> request external authority bound to head-0
```

Generation zero contains no committed use, no reservation, and no delegation escrow. It is an authority-lineage anchor, not evidence that an effect happened.

## CAS contract

`CheckpointCasStore::compare_and_swap(expected_previous, checkpoint)` must be linearizable.

Success means:

- the durable current frontier exactly equalled `expected_previous`;
- the supplied checkpoint became the new durable frontier atomically;
- the returned head identifies that exact checkpoint.

A storage implementation that performs an unlocked read followed by an independent write does **not** satisfy this contract.

## Broker adapter

`CasCheckpointStoreAdapter` implements the existing #305 `CheckpointStore` trait while retaining the expected trusted head internally.

Before calling the underlying CAS store it checks:

- generation zero has no predecessor;
- successor sequence is exactly previous sequence + 1;
- successor predecessor digest equals the exact expected head.

Store errors, wrong acknowledgements, or malformed successor lineage latch the adapter into containment.

The #305 broker retains its own independent persistence-containment latch, giving two fail-closed layers.

## Concurrency property

If two processes start from the same head H and each attempts to publish a successor:

```text
writer A: CAS(H -> A1) = success
writer B: CAS(H -> B1) = conflict
```

Both cannot obtain a successful durable transition from H when the concrete store honors the CAS contract.

This is the persistence property needed before an affine Xenia proof token can be treated as meaningfully one-use across multiple processes.

## Trust boundary

This crate specifies and tests the state-machine contract. It does not provide a production persistence backend.

Candidate concrete implementations include:

- transactional SQLite/PostgreSQL row version/CAS;
- Xenia-owned append/compare frontier service;
- TPM/supervisor monotonic state;
- a single-writer daemon with fsync + authenticated external head;
- Holochain/Mycelix consensus where the chosen consistency semantics satisfy the required single-frontier property.

Durability and linearizability must be qualified for the selected implementation; interface conformance alone is not evidence of either.

## Relationship to Xenia authority

The intended order is:

1. establish generation-zero checkpoint through CAS;
2. externally retain/authenticate its head;
3. Xenia signs the exact `CapabilityGrant` + executor workload + that checkpoint head;
4. Symthaea verifies Xenia authority and freshness;
5. effect entry uses the same CAS adapter for every reservation/outcome successor.

This removes the previous mismatch where the first local checkpoint did not exist until after a restart use had already been reserved.

## Non-claims

V0.1 does not establish:

- a production durable CAS backend;
- cross-datacenter consensus;
- Byzantine store resistance;
- Xenia attestation persistence ordering;
- atomicity spanning Xenia's ledger and Symthaea's checkpoint store;
- physical-host production readiness.
