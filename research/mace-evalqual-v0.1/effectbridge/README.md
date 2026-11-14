# MACE Effect Bridge v0.1 — effect-or-fence experiment

**Status:** local stateful-service reference and fault-injection harness. Not a production API, broker, distributed-consensus protocol, or exactly-once guarantee for arbitrary remote systems.

This package addresses one narrow MACE boundary: what can be concluded after a dispatch acknowledgement is lost, and when is it safe to conclude that a delayed operation will never execute?

## Why a normal not-found is not enough

A client times out after sending a state-changing request. A reconciliation read returns no record. The original request may still be delayed in the network and arrive after that read. Therefore:

    not found now != proved not executed

The client keeps this result indeterminate. It may not promote absence to permission to issue a new non-idempotent action.

## Effect-or-fence protocol

The reference service has one append-only journal owner and two terminal record types:

- effect_committed: the exact intent was committed as the service-side effect.
- fence_committed: the destination durably reserved the idempotency key for that exact intent without executing it. Every later arrival with the same key is rejected.

The operations are linearized under the same journal lock:

- POST /v1/effects: commit the exact intent, or return the same prior effect receipt for an exact duplicate. A changed intent under the same key is rejected. A key already fenced is rejected.
- GET /v1/reconcile/{key}: return a committed effect, committed fence, or not_found_unfenced. The last state is deliberately indeterminate to the client.
- PUT /v1/fence/{key}: atomically return an already committed effect or append/fsync a fence tombstone. If an old request later arrives, the POST path sees the tombstone and rejects it.

For a concurrent effect and fence, exactly one terminal journal record wins: either the effect is committed and the fence request returns it, or the fence is committed and the late effect is rejected. There is no valid state containing both terminal records for one key.

Repeating a fence is idempotent. If its acknowledgement is lost, the client repeats the same exact fence intent; it receives the same terminal state. A lost effect acknowledgement is handled similarly by a read/reconcile call. The client never automatically retries a request after a transport error.

## Journal contract

- One service process owns the journal. Linux flock(LOCK_EX|LOCK_NB) rejects a second process owner.
- Records are versioned JSONL frames with contiguous sequence numbers, previous-record SHA-256, exact intent digest, record digest and explicit durability mode (fsync_per_record).
- Each appended effect or fence frame is written and fsynced before the service acknowledges it. Opening a journal also syncs its parent directory entry.
- Startup replays the complete chain. A truncated final frame, invalid digest, sequence gap, duplicate key, changed intent or semantically invalid record rejects startup; no silent tail truncation or skipping is attempted.
- A write or fsync error poisons the running journal. It returns an unknown/indeterminate outcome and refuses further mutations until restart and recovery.
- Fenced tombstones are retained in the append-only journal; compaction must preserve equivalent anti-replay state.

The event log is also the service fixture's state representation: a committed effect_committed record is the modeled effect. This avoids a second non-atomic business-state store in the demonstrator. A real adapter must place the business mutation and idempotency/fence record inside the same destination transaction or use a destination-supported equivalent. If it cannot provide an atomic fence, an absent lookup remains indeterminate.

## Run

Requires Go 1.23+ and uses only the standard library.

    go run ./cmd/mace-effectsvc -listen 127.0.0.1:8087 -journal ./effects.jsonl

The server defaults to loopback. It has no production authentication or TLS and is intended for a disposable local experiment only. Deterministic acknowledgement-loss injection is disabled unless the explicit -allow-test-fault-injection flag is set; never enable it in a deployed service.

## Local qualification tests

The public test file currently contains 12 qualification scenarios plus three child-process helper entrypoints; the expanded local source set used for the larger verification run had 24 named effectbridge test functions. This distinction is recorded to avoid implying every expanded local test is already in this public commit.

The 24 effectbridge test functions cover strict request/journal validation, hash-chain tampering and torn-tail rejection, exact duplicate idempotency, same-key intent conflict, 64 concurrent goroutine requests, one-writer process locking, eight independent client processes, remote success with lost acknowledgement across service process kill/restart, ordinary absence remaining indeterminate, fence/dispatch races, delayed-request rejection after fence, lost fence acknowledgement and idempotent fence recovery, and 10,000 seeded effect/fence operations against a separate finite-state reference model.

The service runs as a real local child process over loopback TCP. The separate journal reader validates the service-authored journal after the service is closed. This is not an independent physical observation of a third-party effect boundary.

## Composition and limitations

This is a destination-side adapter/conformance fixture, not the MACE broker and not an authority issuer. It does not mint permits, authenticate principals, authorize actions, provide multi-host coordination, or prove storage-device power-loss durability. Advisory flock assumes clients cannot bypass the service and directly write the journal; production deployments need OS permission separation and an isolated service identity. The self-hashed receipt provides byte integrity only; it is not a signature or proof of issuer identity.

Compose with:
- Symthaea #11647 for the crash-recoverable one-shot broker, durable grant consumption, broker IPC and budget accounting;
- #9490 for general effect authority and pre-dispatch revalidation;
- MACE #11920 / PR #11965 for independent benchmark/effect/observer qualification.

Do not infer end-to-end exactly-once behavior from this test service. The target is a narrower property: within the declared service transaction boundary, an exact idempotency key reaches one terminal state, and uncertain outcomes cannot be converted into a blind replay.


## Concurrent history qualification

The JSONL reference now has a history-based linearizability test. It records invocation and return order for concurrent `Append`, `Fence` and `Lookup` operations, then backtracks over candidate sequential histories while preserving real-time precedence. The observed outcomes must match the reference state machine, including changed-intent conflicts and late-dispatch rejection. Each generated history is followed by close/reopen reconciliation.

This is bounded executable conformance evidence for the local single-writer journal, not a proof of every schedule or of a distributed destination. It complements, rather than replaces, process-crash, observer-independence and native-database transaction qualification.


## Bounded protocol-state exploration

`modelcheck_test.go` exhaustively explores a finite two-intent abstraction of send, delayed delivery, destination serialization, receipt/business-row staging, atomic commit, rollback/crash and reconciliation. Its current state space is 68 reachable states / 200 transitions. The companion history checker validates concurrent `Append`, `Fence` and `Lookup` results against real-time precedence.

These checks are deliberately bounded and independent of the storage implementation. They complement the concrete journal/SQLite tests; they do not prove the implementation, distributed liveness, multi-host safety or physical power-loss behavior.
