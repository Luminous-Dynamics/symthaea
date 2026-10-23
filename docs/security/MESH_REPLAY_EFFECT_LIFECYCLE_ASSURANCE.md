# Mesh Replay / Application-Effect Crash Consistency

**Status:** research model; not a production implementation or proof of exactly-once delivery  
**Model:** `docs/security/models/MESH_REPLAY_EFFECT_LIFECYCLE.tla`  
**Scope:** the crash boundary between authenticated replay admission and a durable application effect

## Why this is a separate property

A replay window prevents duplicate packet admission only while its state is preserved. Even with a durable replay record, a stateful receiver can mishandle a crash between replay bookkeeping and an application side effect.

Two straightforward orderings fail in opposite directions:

1. **Effect, then persist replay state.** The application effect succeeds; a crash happens before the sequence is durable. After restart, the same still-fresh authenticated message is admitted again and applies its effect a second time.
2. **Persist replay state, then effect.** The sequence becomes durably consumed; a crash happens before the application effect. After restart, replay protection correctly rejects the message, but its intended effect is now lost.

A sequence bitmap alone does not solve both outcomes. The requirement is about the transaction boundary between authenticated message identity and the side effect, not just the packet parser.

## Model variants

The finite-state model holds one authenticated packet sequence constant and tracks durable replay state, application-effect count, volatile in-flight flags, crash/restart, and whether recovery has occurred.

| Configuration | Required result | Purpose |
|---|---|---|
| `replay_effect_before_commit_legacy.cfg` | Counterexample to `NoDuplicateEffect` | Demonstrates duplicate application effect after effect-first crash window |
| `replay_commit_before_effect_legacy.cfg` | Counterexample to `NoLostCommittedEffect` after restart | Demonstrates lost effect after commit-first crash window |
| `replay_effect_atomic_strict.cfg` | Both invariants hold in the bounded model | Reference abstraction of one atomic/idempotent transaction boundary |

The workflow parses the exact model, verifies the pinned TLA+ tools digest, requires the strict configuration to report successful model completion, and requires each mutant run to report the **intended named invariant** violation. A generic nonzero exit is not counted as a successful counterexample. Model/config hashes and raw logs are uploaded as run artifacts.

## Implementation design choices

A real implementation must choose and review one of these approaches; the abstract atomic action does not choose it for the project:

- **Transactional inbox + effect:** commit message identity and the business effect in one storage transaction.
- **Durable idempotency key:** use a stable authenticated message identity as the application's unique effect key, and make the effect store reject repeated application.
- **Transactional inbox/outbox:** persist admission and work intent atomically, then process the effect through a retriable idempotent worker.
- **Domain-level idempotency:** where the effect is naturally idempotent, specify and test that property rather than claim packet-level exactly-once behavior.

External effects such as radio transmission, e-mail delivery, an actuator, or a remote service may not share a transaction with the receiver's local database. For such effects, the implementation needs a receiver/worker protocol with idempotent operation identifiers, acknowledgements, and reconciliation; the packet replay window alone cannot guarantee exactly-once physical effects.

## Required future tests

- Crash after successful authentication but before durable replay reservation.
- Crash after durable replay reservation but before application effect.
- Crash after application effect but before acknowledgement.
- Duplicate delivery after process restart, snapshot rollback, and restore from backup.
- Full storage / write failure / power loss during commit.
- Concurrent duplicate delivery on two workers.
- Recovery after the replay table has reached capacity.
- Positive controls: fresh packet applies once; an authorized epoch transition permits new traffic; a duplicate does not repeat the side effect.

## Claim boundary

This model is intentionally finite and represents one message identity. The `effectCount` variable models a durable application-side effect count, while `volatileEffectDone` is lost on restart. The `atomic` mode abstracts a real transaction or idempotency boundary; TLC cannot establish that a real database, filesystem, remote API, or physical effect actually implements that abstraction.

The workflow is currently designed to run on the exact branch head, but **no TLC pass or counterexample is claimed until the pinned model-checking job completes and the uploaded logs are reviewed**. This is an assurance-model proposal, not proof of protocol security or a production change.

The model should become one of the required oracles under the mesh benchmark contract PR #8507 once it has been independently checked.
