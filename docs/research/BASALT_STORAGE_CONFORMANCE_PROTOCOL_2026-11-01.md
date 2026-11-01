# BASALT Storage Conformance Protocol v1

**Status:** protocol and reference fixture only. No backend adapter or verifier is implemented by this change.

## Purpose

BASALT's storage problem is not simply "which database is fastest?" The trusted operation compares a source generation and predecessor, a log-origin checkpoint, a policy revision, and an exact immutable receipt, then advances the accepted state atomically. A separate witness must detect rollback to an internally consistent but stale authority snapshot.

The storage qualification suite already lists candidate engines, operational configurations, fault scenarios and measurements. This document makes the next step explicit: every backend must execute the same versioned trace and produce a replayable history that a backend-independent verifier can check.

The benchmark is therefore a **claim-carrying execution history**, not only a throughput report.

## Industry reference points

The protocol borrows and composes three mature testing ideas:

- **Jepsen:** record operation invocation/completion histories, inject faults with a nemesis, then check whether the history conforms to a declared consistency model. BASALT needs the same distinction between observed history and the claim made about that history. See [Jepsen's consistency models](https://jepsen.io/consistency/models) and [Jepsen's test architecture](https://github.com/jepsen-io/jepsen).
- **FoundationDB:** deterministic simulation of a cluster, network, machine and storage failures, with reproducible seeds and aggressive failure schedules. BASALT should adopt deterministic replay early, while retaining a separate physical-failure qualification tier. See [FoundationDB Simulation and Testing](https://apple.github.io/foundationdb/testing.html).
- **SQLite:** VFS-based crash simulation, I/O failure injection, recovery checks, compound failures, regression tests and mutation testing. SQLite explicitly distinguishes process/application crash from simulated filesystem/power-loss behavior. See [How SQLite Is Tested](https://www.sqlite.org/testing.html).

These are reference practices, not claims that BASALT currently matches their test volume, test coverage or fault fidelity.

## Protocol objects

The first artifact pair is:

- `benchmarks/basalt/schema/storage-conformance-trace.v1.schema.json`
- `benchmarks/basalt/storage-conformance-trace.v1.json`

The fixture is intentionally small and deterministic. It is a reference input, not an observed backend run. Its result fields describe the expected semantic outcome of the reference trace; they must not be presented as engine measurements.

### Trace structure

A trace binds:

1. **Suite and generator identity:** schema version, trace ID, deterministic seed, monotonic-clock basis and clock-ordering rule.
2. **Authority identity:** stable authority ID and incarnation ID.
3. **Initial state:** authority revision, canonical state digest, active policy digest, source cursors and log-origin cursors.
4. **Operations:** client ID and program order; source/log scope; invocation and completion times; complete compare conditions; receipt identity; typed outcome; immutable receipt acceptance revision; current authority revision.
5. **Fault schedule:** typed fault, target operation, deterministic injection point and action.
6. **Witness observations:** witness identity and sequence, authority/incarnation, state digest, predecessor witness digest and signed-record digest.
7. **Final observation:** complete cursor state, accepted receipt set, policy digest, current authority revision and canonical state digest.
8. **Expected invariants and negative controls:** assertions the verifier must establish, plus seeded faulty implementations it must reject.

All digests in the fixture are deterministic SHA-256 values for the named fixture inputs. A real run must compute digests from a documented canonical encoding of actual records; it must not reuse fixture constants.

### Time and ordering

Monotonic timestamps establish intervals within one runner. They are not wall-clock timestamps and cannot order events across hosts. The verifier must honor:

- per-client sequence order;
- explicit `happens_after` dependencies;
- real-time precedence when operation A completed before operation B was invoked;
- overlap as concurrency, not as an assumed order.

A later implementation should use an event log with explicit invocation, completion, fault and observation records. The v1 fixture compresses each operation to one interval record for readability; this does not relax the ordering semantics.

## Reference transition rules

An `advance` is one atomic semantic operation. The backend may use different tables, keys, transactions or consensus primitives, but the externally observed outcome must refine the same pure transition model:

- Compare authority incarnation, source generation and source predecessor.
- Compare log-origin identity and expected checkpoint.
- Compare the exact policy digest at commit time, not only when a proof was first checked.
- Enforce immutable receipt identity and digest.
- Advance the source cursor, log checkpoint, accepted receipt and policy binding together, or advance none.
- Enforce at most one accepted semantic winner for a given predecessor.
- Permit independent scopes to progress when no shared invariant conflicts.
- Preserve exact retry semantics after a lost response and later unrelated commits.

A duplicate-receipt response carries two independent facts:

- `receipt_acceptance_revision`: the historical revision at which that immutable receipt was first accepted;
- `current_authority_revision`: the revision of the authority state observed by the current lookup/retry.

Neither field is an alias for the other. The exact retry after receipt B advanced the store is specifically designed to catch implementations that overwrite the receipt's original acceptance revision with the current revision.

A client that loses a successful commit response must treat the outcome as indeterminate and resolve by exact receipt identity. It must not blindly replay a new semantic transition.

## Fault semantics

Never conflate these fault classes:

- **Process crash:** kills the process but leaves the OS page cache and storage stack alive.
- **Power-loss/storage-image model:** discards writes that were not durably synchronized and may reorder or damage unsynchronized writes according to the declared model.
- **I/O failure:** short write, disk full, fsync error, read error or storage-device failure.
- **Response loss:** server may have committed; client cannot infer commit status from the missing response.
- **Network partition/delay:** changes reachability and timing; it is not itself evidence that an acknowledged durable write was lost.
- **Serialization failure:** PostgreSQL `40001` aborts the transaction. The retry boundary is the entire transaction, including all reads and decisions derived from those reads.
- **Snapshot restore:** may restore a coherent but stale state. Coherence is not freshness.
- **Witness unavailable:** freshness is unknown; the backend cannot substitute its own restored database for the independent witness.

Every scenario declares its fault model. A SIGKILL test cannot be reported as a physical power-loss test. A simulated storage image cannot be reported as a hardware campaign. A normal PostgreSQL primary commit cannot be reported as synchronous-standby durability unless the standby policy is explicitly configured and measured.

## Witness and rollback rules

The witness must be independent of the database being qualified. Its authenticated record binds at least:

- authority ID and incarnation ID;
- monotonically increasing witness sequence;
- canonical accepted-state digest;
- predecessor witness-record digest;
- policy digest / relevant authority metadata;
- witness identity and signature-verification context.

The verifier checks the witness chain under a separately configured trust root. Merely storing a witness pointer inside the candidate database is insufficient: snapshot rollback would roll back both.

If a restored authority snapshot cannot prove continuity with the current independently witnessed head, the result is `rejected_anti_rollback` or `anti_rollback_unknown`. It is never fresh merely because its rows are internally consistent or its local integer revision looks large.

## Verifier design

Implement the verifier as a pure Rust reference reducer with no DB driver dependency. The runner adapters should only:

1. instantiate the declared backend and schema;
2. apply operations in the trace under the declared schedule;
3. emit raw operation/fault observations and a final canonical state image;
4. record the backend/client/configuration build identity;
5. retain raw artifacts and their digests.

The verifier should:

- reject schema-invalid or incomplete traces before evaluating behavior;
- reconstruct the operation DAG and legal real-time constraints;
- replay the semantic state machine independently of backend code;
- check all four-record atomicity and per-scope concurrency outcomes;
- distinguish definite commit, definite abort, conflict and indeterminate response;
- validate retry groups and ensure a serialization retry begins a new whole transaction;
- compare final records and receipt history with the reference state;
- verify witness-chain linkage and restore freshness;
- emit a machine-readable report and human-readable counterexample on failure.

Do not let the backend adapter supply its own expected result or grade itself. The expected semantic result belongs to the trace/reference model; observed facts belong to the adapter's event log; the independent verifier decides whether they agree.

## Negative controls

A verifier that always passes a correct fixture is not yet credible. It must also fail seeded mutants, including:

1. omitting the log cursor from the atomic transaction;
2. accepting both concurrent receipts from one predecessor;
3. replacing an old receipt acceptance revision with the current store revision;
4. accepting a stale policy after policy publication;
5. retrying only the last SQL statement after `40001`;
6. treating an old restored snapshot as fresh despite a newer witness head.

Report `mutants_detected / mutants_seeded`, with the exact mutant IDs and artifacts. A mutation score is evidence about the verifier's sensitivity; it is not a proof of correctness and cannot substitute for independent review.

## Result envelope required for a real run

A future run report should bind:

- trace SHA-256 and schema SHA-256;
- backend adapter commit SHA and executable/build digest;
- engine version and image/package digest;
- client-library version;
- database configuration and isolation level;
- topology, region, network profile and replica policy;
- OS, kernel, filesystem, mount options and storage-device model;
- durability settings, sync/flush evidence and fault-model implementation;
- deterministic seed, exact fault schedule and raw event-log digest;
- final durable image or reproducible image fixture;
- witness identity, signed checkpoint and verifier build digest;
- verifier status and all counterexamples.

Missing provenance, unavailable witness, incomplete raw history, or a mismatch between artifact and declared digest yields `inconclusive` or `invalid_evidence`, never `pass`.

## Qualification gates and comparison

### Safety gates (non-compensable)

- zero acknowledged durable commits missing under the declared durability model;
- zero partial four-record transitions;
- zero double winners for one predecessor;
- zero stale-policy commits;
- zero duplicate-retry semantic advancement;
- zero false freshness after restore;
- all seeded negative controls detected;
- 100% of traces independently replayable from retained artifacts.

A single violation disqualifies that configuration for the claimed safety level. Throughput cannot offset it.

### Efficiency and operations (measured only after safety)

Report raw distributions, not a scalar score:

- commit/lookup latency p50, p95 and p99;
- throughput by transaction shape, scope contention and client count;
- conflict and whole-transaction retry rates;
- sync/flush count and latency;
- bytes written per accepted transition;
- CPU, memory, disk I/O and network bytes;
- recovery, reopen, witness-reconciliation and cache-rebuild duration;
- deployment, backup, restore and operator complexity.

Keep local embedded, same-host server, remote primary and synchronous-replica arms separate. Compare etcd and PostgreSQL only at equivalent authority/consistency boundaries; do not let local SQLite latency masquerade as remote-coordination superiority.

### Initial scale targets

Start with deterministic traces at 1, 2, 8, 16, 32, 64 and 128 clients; 1k, 10k and 100k accepted receipts; uniform and hot-scope distributions; and the declared fault set. Add 1M+ receipt soak runs only after the same trace/replay and evidence-retention path is stable.

The first target is not a specific p99. It is zero safety violations plus reproducible, fully attributed measurements. Performance targets should be set from measured production-shaped workloads after the candidate backends pass the same safety gates.

## Implementation status and non-claims

This PR adds a versioned trace schema, one deterministic reference fixture and this protocol. It does **not** add the Rust trace runner, pure reference reducer, SQLx/SQLite/Turso/etcd adapters, Jepsen integration, simulation engine, physical failure tests, or verifier. It does not qualify any storage engine. Those remain follow-on implementation work tracked under #9764 and #9792.
