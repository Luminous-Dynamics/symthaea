# MACE Observer Independence and Evidence-Origin Profile

**Status:** proposed MACE research profile, not an external standard or certification scheme.

## Why an independence level is not a Boolean

An observer can be outside the agent's process and still share the agent's host kernel, storage, credentials, administrator and failure domain. A database read-only connection can independently verify stored transaction facts without independently proving that a remote service performed the business action represented by those rows.

MACE therefore records observer independence as a **trust-domain vector**, with a shorthand E0–E4 class for discussion. The shorthand is a proposed local taxonomy, not a score, not a guarantee, and not a substitute for naming the threat actors and failure modes that matter in a given deployment.

## Proposed observer classes

| Class | Observation source | What it can establish | Shared trust / key limitation |
|---|---|---|---|
| **E0 — self-report** | Agent-authored result, tool-call transcript, or agent-generated receipt | What the agent claims it requested or observed | The agent can omit, fabricate or misdescribe its own effects |
| **E1 — co-resident instrumentation** | Runtime hooks, middleware, logs or an observer in the same process/service | Events visible to that instrumentation point | Shares process memory, code path, privileges and often the same operator |
| **E2 — separate local observer** | Separate process and read-only connection to the destination's durable state | Stored transaction consistency, receipt-to-row linkage, chain integrity and a stable read snapshot | Shares host kernel, storage device, filesystem/VFS, database and often administrator; cannot prove a remote side effect or physical settlement |
| **E3 — service-native evidence** | Destination-owned commit log, gateway audit stream, or provider-native immutable event stream accessed under separately scoped credentials and retention | The service's own authoritative account of accepted/committed events | Still shares the provider and potentially its administrators, implementation, control plane and evidence-generation path |
| **E4 — external counterpart / witness** | Independent settlement counterparty, receiving system, external auditor, or physical sensor in a distinct administrative and failure domain | Corroboration that the effect crossed a boundary visible to the external witness | Limited to the witness's own coverage, calibration, identity binding and observation window; never universal completeness |

Classes are not additive. Two E2 observers on one host do not become E3. Two providers may share a cloud control plane, identity provider, network or operator. A higher class does not automatically imply lower measurement error.

For every observer, publish the full trust-domain vector:

- source owner and evidence producer;
- process and host boundary;
- kernel, hypervisor and storage/VFS boundary;
- credentials, privileges and administrative domain;
- network path and transport termination;
- clock and sequencing authority;
- parser, verifier and code provenance;
- retention, deletion and rollback controls;
- declared event coverage and known blind spots.

State the adversary/failure domain against which independence is claimed. “Independent” without that statement is underspecified.

## Current SQLite companion: E2, not E3/E4

The local MACE companion contains a separate sqliteaudit process. It opens the database read-only, pins one SQLite read snapshot, runs database and foreign-key integrity checks, validates the receipt hash chain and intent digests, and reconciles effect receipts against business rows. It does not import the writer's effectbridge package.

That is useful **E2** evidence about what is durably represented in one local database snapshot. It is not an independent host or remote service witness: observer and writer share the database file, host kernel, storage/VFS and administrative domain. It cannot prove that an external payment, API action or physical process occurred.

The observer also deliberately reports the writer's synchronous setting as not observable from the read-only connection. SQLite connection pragmas are connection-scoped. The writer verifies its own WAL/FULL profile at startup; a separate observer must not claim it has independently measured the writer's PRAGMA synchronous setting.

SQLite documents that WAL processes must share the same host (WAL is not supported over a network filesystem), and that synchronous=FULL in WAL mode adds a WAL sync after each commit to improve power-loss durability. These are SQLite VFS/filesystem/storage-contract claims, not evidence that a particular device correctly implements flushes. Process-kill tests do not qualify power-cut recovery. Sources: [SQLite WAL](https://www.sqlite.org/wal.html) and [SQLite PRAGMA synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous).

## Observer qualification contract

For a fixed, independently constructed ground-truth corpus, report at least:

1. **Completeness / recall:** fraction of true in-scope effects represented by the observer.
2. **Soundness / precision:** fraction of observer-claimed effects that match the ground truth.
3. **Causal fidelity:** exact source sequence and parent-receipt agreement; wall-clock order is diagnostic, not a causal proof.
4. **Attribution fidelity:** principal, target, payload digest and idempotency key match exactly.
5. **Snapshot coherence:** all reported rows and chain heads come from one declared consistent snapshot.
6. **Freshness / lag:** source commit to observer availability, with p50/p95/p99 and timeout semantics.
7. **Indeterminate handling:** observer gaps, stale cursors, unavailable sources, sequence gaps or integrity errors must not be counted as no effect.
8. **Common-cause exposure:** which actors can suppress, rewrite or fabricate both the effect and the observer evidence.

The ground-truth producer must not derive truth from the observer under test. Maintain separate source code, ownership, credentials and storage where feasible. If both streams share a producer, report that common cause.

## Required adversarial and metamorphic corpus

Every positive event must have a known-good baseline. Every negative event must include an equivalent condition-removed control where practicable. Include:

- drop, delay, duplicate, reorder and replay;
- fabricated receipt, forged principal, substituted payload or target;
- incorrect or missing causal parent and source sequence;
- observer restart, source restart, stale cursor and retention expiry;
- transaction rollback, partial commit and commit acknowledgement loss;
- read-snapshot skew and concurrent writes;
- event deletion, log truncation, hash-chain splice and business-row substitution;
- unavailable observer, inaccessible key, permission downgrade and clock discontinuity;
- effect that occurs at a downstream counterparty but is absent from the local service log.

A missing observation is unknown or indeterminate, never an inferred non-effect. A correct local ledger with no independent downstream witness is not proof of settlement.

## Benchmark experiment

Use the same task set, agent, model, scaffold, tool schemas, budgets, seeds and failure injections across matched configurations:

- **A — E1 only:** existing runtime instrumentation.
- **B — E2:** add the separate read-only local observer.
- **C — E3:** consume the destination's service-native audit/commit stream under separate credentials.
- **D — E4:** add an independent downstream counterpart or physical witness where available.

Measure utility, effects and benchmark integrity independently. For each configuration, publish precision, recall, causal/attribution errors, unknown rate, observer lag, cost, latency and failure behavior. Also publish a common-cause matrix. Do not collapse this into one weighted assurance score.

### Proposed gates (not achieved results)

- 100% detection of the registered deterministic dropped/fabricated/substituted/replayed controls.
- Zero false claims of non-execution when the observer is unavailable, stale or incomplete.
- 100% exact receipt-to-effect identity and causal-parent agreement in the registered corpus.
- Every observer integrity failure produces indeterminate / not qualified, never pass.
- No generalized error-rate claim from hand-authored fixtures; infer population rates only from a pre-registered representative sampling frame with dependence handled explicitly.

## Independent conformance tools

MACE should reuse independent verification tools rather than create a proprietary linearizability checker as its only oracle. [Porcupine](https://github.com/anishathalye/porcupine) is an open-source Go linearizability checker that accepts an executable sequential model and concurrent operation histories. It is a candidate independent cross-check for exported MACE histories. The current MACE in-repo checker is bounded test code and is not a replacement for that independent implementation.

TLA+/TLC is the complementary route for checking a finite protocol specification before comparing it with implementation histories. A model-checker pass establishes properties only for the declared model, constants and invariants; it does not prove that the implementation refines the model without an explicit refinement/evidence argument.

## Next engineering gates

1. Keep E2 SQLite audit as a read-only, full-snapshot reference implementation; do not let it certify its own writer configuration.
2. Export an explicit versioned operation history format and check identical histories with both MACE's reference checker and Porcupine.
3. Build an E3 adapter for a real service-native transaction/audit source; separate credentials and retention, and record source sequence/commit identity.
4. Add an E4 counterpart fixture only where a genuine external party or physical sensor exists; do not synthesize an independent witness.
5. Run common-cause fault injection and report which failures affect both the effect producer and observer.

**Claim ceiling:** an observer can only establish claims within its declared source, coverage, trust and freshness boundaries. No E0–E4 class proves complete observation of every real-world consequence.

## Hash-chain consistency is not history authenticity

A recomputed hash chain proves that the rows currently visible form a self-consistent sequence. It does not authenticate that sequence against an administrator or other actor who can rewrite every receipt and matching business row and recompute the hashes. An unanchored audit may correctly accept such a rewritten database.

To detect a coordinated rewrite, compare the observed journal head with a previously trusted digest obtained **outside** the database, under a separately declared trust domain. The local companion adds an `AuditWithExpectedHead` call for this purpose and a test that rewrites a receipt plus its business row, recomputes the intent/record hashes, confirms that the unanchored audit accepts the self-consistent rewrite, and confirms that the original out-of-band head rejects it.

This is a caller-supplied checkpoint comparison, not remote signing or key custody. The local prototype does not issue, sign, retain or protect the external checkpoint. A digest stored beside the database is not independent. A production E3/E4 design needs an independently administered checkpoint source (for example, a signed remote receipt or transparency log), key-management separation and rollback/fork detection.

