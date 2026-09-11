# symthaea-replicator-ledger

> **Status: REFERENCE SAFETY SEMANTICS — NOT PRODUCTION-ADMITTED**

This crate is the process-local reference state machine for descendant lineage, inherited capability ceilings, replication/resource accounting, negative-state propagation, and stale-cursor/replay resistance in the Symthaea Replicator Safety Kernel (RSK).

It contains **no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path**.

## What the ledger establishes

The reference ledger models several invariants that must survive any future durable/distributed implementation:

- creation records an unprivileged descendant; it does not grant authority;
- descendant capability ceilings attenuate and cannot widen through fresh grants;
- descendant creation consumes applicable subject-subtree and lineage-tree budgets;
- ancestor population, depth, and resource scopes remain binding on later descendants;
- lineage branch policy can only become equally or more restrictive;
- subject and lineage quarantine/revocation propagate through ancestry;
- grant revocation remains effective against descendant operations that depend on the revoked ancestral scope;
- a successful mutation consumes one ledger cursor;
- competing allows evaluated from one cursor cannot both commit;
- duplicate mutation identifiers and stale cursors fail closed;
- runtime safety facts are rechecked before a descendant mutation is committed.

The current implementation is intentionally in-memory and semantic. It is useful for exercising the authority model and specifying what a durable store must preserve.

## Known Class A production blockers

The current API is **not yet a production authorization boundary**. In particular:

### Exact action binding

`BoundedReplicationAuthorization` currently binds the evaluated subject, lineage, grant generation, risk class, capability set, evidence digests, expiry, and grant ceilings, but it does not yet bind the exact `resource_units` value passed separately to `commit_descendant`.

The hard ceilings still bound total resource consumption, but this means the committed action need not be byte-for-byte identical to the resource amount evaluated by the authority decision. Production admission therefore requires binding the evaluated requested resource amount—and any other authority-relevant operation commitment—into the opaque authorization token and rejecting mismatches at commit.

### Trusted time

`commit_descendant` currently receives `now_unix_secs` from its caller. Production use requires independently verified monotonic/clock-continuity evidence so a caller cannot extend authority through clock rollback or fabricated time.

### Authenticated runtime evidence

`RuntimeSafetyWitness` is currently a reference snapshot. Production use requires an authenticated/verified witness type whose provenance, freshness, and independence have been established outside the controlled autonomy.

### Authenticated grants and quorum

The authority crate currently models grants and quorum evidence as plain values. Production use requires verified signer/trust-root/failure-domain evidence before the ledger receives a bounded authorization.

### Durable evidence and real atomic storage

The in-memory cursor provides reference compare-and-swap semantics inside one process. It does not provide:

- durable transactional storage;
- multi-process or distributed compare-and-append;
- canonical cryptographic event encoding;
- predecessor-hash binding;
- replay-from-genesis verification;
- crash/torn-write recovery;
- signed checkpoints;
- fork/equivocation detection across replicas.

These are specified separately in the durable-evidence contract and must be implemented before production admission.

### Authorized governance mutations

The reference methods for quarantine, revocation, grant revocation, and lineage registration assume trusted access to the ledger object. A production service must authenticate and authorize mutation actors according to distinct roles; recovery authority must remain separated from ordinary replication authority.

## Required production shape

A production implementation should preserve the reference ledger as the semantic core while wrapping it with verified evidence types and a durable transaction boundary:

```text
verified external evidence
        |
        v
constitutional authority evaluation
        |
        v
opaque exact-action authorization
        |
        v
trusted runtime assurance recheck
        |
        v
atomic durable compare-and-append
        |
        v
replay-verifiable unprivileged descendant record
```

No stage may infer positive authority from storage success, transparency receipts, DKG availability, physical existence, or a descendant's technical capabilities.

## Intended use today

Appropriate uses:

- reference state-machine semantics;
- adversarial and property tests;
- comparison against formal models;
- simulation of lineage/budget/revocation behavior;
- designing durable adapters and replay verification.

Not appropriate today:

- production physical actuation authority;
- treating in-process values as authenticated evidence;
- distributed operation without transactional/fork handling;
- describing the implementation as certified or formally proven.

The crate is intentionally marked `publish = false` until the production-admission gates are satisfied.
