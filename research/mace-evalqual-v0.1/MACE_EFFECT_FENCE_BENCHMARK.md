# MACE-EFX v0.1 — destination effect/fence conformance benchmark

**Status:** proposed benchmark contract with a local reference implementation. No third-party adapter or external benchmark result is qualified yet.

## Research question

Can a destination prove that one exact authorized intent reached at most one terminal state—effect committed OR fenced against future execution—even when dispatch, network delivery, acknowledgement, process lifetime and retries fail independently?

A critical ambiguity:

```
request sent
→ client times out
→ reconciliation says not_found_unfenced
→ original request arrives late
```

An ordinary absence read only describes what is visible at that instant. It does not close the delayed-arrival window. A non-execution decision requires a destination-side fence serialized against the exact effect at the same authority/transaction boundary.

## Normative terminal semantics

For destination-scoped idempotency key `K` and canonical intent `I`:

- `effect_committed(K,I)`: the destination atomically committed the business effect and the record that binds it to `K,I`.
- `fence_committed(K,I)`: the destination atomically committed a tombstone for `K,I` and guarantees every later arrival carrying `K` is rejected without an effect.
- `not_found_unfenced(K)`: no terminal state is currently visible. The outcome is `indeterminate`; it is not evidence of non-execution.
- Same `K`, different `I`: conflict. Never reinterpret a key for a changed principal, target, operation or payload.
- Repeated same `K,I`: return the same terminal receipt; do not create a second business effect or fence record.

The linearization boundary must be shared by effect and fence. If both race, one terminal outcome wins. The other operation returns that winner. A request arriving after a committed fence must not execute.

This is an **at-most-once terminal-decision property within the destination's declared boundary**, not a generic distributed exactly-once guarantee. The destination must atomically commit the business mutation and idempotency/fence state in one native transaction, or rely on a downstream destination that provides an equivalent idempotency contract. If it cannot, the correct result remains indeterminate.

## Required fault matrix

| Fault / mutation | Expected result |
|---|---|
| Timeout before request reaches destination | Unknown to client; reconcile, never infer non-execution from client timeout |
| Destination commits effect; HTTP acknowledgement is dropped | Reconciliation returns the exact original effect receipt |
| Client sends the same request again | Same receipt; no second effect |
| Same idempotency key with changed intent | Conflict; no second effect |
| Plain lookup before delayed request arrives | `not_found_unfenced` → indeterminate |
| Fence commits before delayed request arrives | Fence receipt; later dispatch is rejected |
| Dispatch races fence | Exactly one terminal row; no both-effect-and-fence state |
| Fence commits; fence acknowledgement is dropped | Repeat the identical fence; recover the same tombstone |
| Failure after receipt insert but before business-row insert | Entire transaction rolls back; neither row survives |
| Process dies before COMMIT | Neither receipt nor business state survives recovery |
| Process dies after COMMIT before reply | Both rows survive; exact receipt is recoverable after restart |
| Business row is modified or removed outside the transaction path | Startup/audit verifier reports integrity failure; no pass |
| Journal tail truncated / frame altered | Fail closed; do not silently truncate or skip |
| Destination observer unavailable or incomplete | Indeterminate; no new non-idempotent mutation |

The local SQLite test `TestSQLiteProcessDeathBeforeCommitRollsBackBothRows` exercises process death while a transaction containing both rows remains uncommitted. This is not a power-loss test. A separate lost-ack restart test exercises the opposite boundary: committed state with a lost HTTP response.

## Three-way benchmark design

Run each registered scenario in three arms under the same destination state and fault schedule:

1. **Baseline:** conventional request/response with current idempotency support, if any.
2. **Telemetry-only:** same destination and client, with event/readback evidence but no atomic destination fence.
3. **Effect-or-fence:** destination atomic effect/fence semantics plus MACE client reconciliation.

Do not let the telemetry-only arm quietly use the fence endpoint. Do not compare different database engines or hardware in the same treatment contrast. Randomize and pin task/order/seed, publish exact trial IDs and maintain independent expected outcomes.

## Primary metrics

- `double_effect_accepts / effect_intent_trials`.
- `late_effects_after_fence / fence_committed_trials`.
- `false_non_execution_claims / unfenced_absence_trials`.
- `duplicate_terminal_rows / idempotency_key_trials`.
- `intent_substitution_accepts / changed_intent_trials`.
- `transaction_partial_commits / injected_precommit_failures`.
- `lost_ack_reconciliation_accuracy`.
- `indeterminate_recovery_time` and duration of mutation freeze.
- `observer_event_precision/recall` against a separately instrumented destination event source.
- p50/p95/p99 commit, reconciliation and fence latency; throughput, contention and WAL/journal growth.
- benign-operation completion and false-positive block rate.

Publish counts, denominators, per-fault results and uncertainty. Do not combine these into one weighted score. A single accepted late effect after a fence is a hard safety failure, not something to average away.

## Backend comparison contract

Compare JSONL, SQLite WAL/FULL, and later PostgreSQL/other remote transactional stores only after the exact same logical effects and fault matrix are implemented for every backend. Separate single-record append, atomic business-row + receipt/fence commit, indexed reconciliation, startup verification, throughput and tail latency, process crash vs OS crash vs power loss, and local single-host vs remote multi-host coordination.

The current local JSONL and SQLite microbenchmarks are **backend-specific reference points, not a semantically matched performance bake-off**: JSONL models the effect as its journal record; SQLite writes both an effect receipt and a separate business-effect row. SQLite WAL is same-host only and is not a network-filesystem coordination mechanism.

## Proposed qualification targets (not results)

- 100% finite-state oracle agreement across every registered terminal-state and race fixture.
- Zero accepted late effects after a committed fence in the frozen adversarial corpus.
- Zero false `not_executed` claims from plain absence reads.
- Zero duplicate business effects per exact key/intent in registered duplicate and concurrent schedules.
- Zero partial business-row/receipt commits across injected pre-COMMIT failures.
- 100% recovery of the original effect receipt after committed-effect acknowledgement loss.
- 100% indeterminate outcome on missing, inconsistent or unauthenticated destination evidence.
- At least 10,000 generated schedules against an independent finite-state oracle, followed by process-crash schedule sweeps and separately qualified power-loss hardware tests.

The 10,000 sequential operations in the current local model test are not 10,000 independent crash schedules, and a proposed 10,000-schedule replay is not a population-risk guarantee unless its sampling assumptions are defensible.

## Integration boundaries

- MACE #11920 / PR #11965: benchmark integrity, effect observer qualification and result reporting.
- #9490: general action-effect authority and pre-dispatch revalidation.
- #11647: durable one-shot broker, grant/budget consumption, IPC and broker crash recovery.
- Destination adapter: atomic business effect/idempotency/fence transaction and truthful reconciliation.

The MACE client must not mint authority, and the destination must not trust caller-supplied principal fields as authenticated identity. Production deployments require a separately authenticated principal and a trust boundary around the destination journal/database.

## Claim ceiling

The local JSONL service models the business effect as the journal commit. The optional SQLite backend additionally commits a separate business row and effect/fence receipt in one SQLite transaction. Both are local reference fixtures. Neither is a third-party payment/service/database integration, remote attestation, power-loss hardware qualification, multi-host consensus system, or an independent physical observer.
