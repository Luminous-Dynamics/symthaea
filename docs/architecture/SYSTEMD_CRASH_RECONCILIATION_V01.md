# Systemd Crash Reconciliation v0.1

Status: **draft / unqualified**  
Branch: `agency/systemd-crash-recovery-v0.1`  
Base: `#305`

## Problem

#305 persists a `Reserved` action-runtime checkpoint before dispatch. That ordering prevents an effect from being launched without a durably acknowledged authority reservation.

After a process or host crash, however, the latest trusted checkpoint may still say `Reserved` even though the restart might have been dispatched after that checkpoint and before the crash.

The state is therefore crash-ambiguous:

```text
checkpoint persisted: Reserved
             |
             +-- crash before dispatch  -> no effect
             |
             +-- dispatch -> crash      -> effect may exist
```

Treating the restored reservation as ordinary pre-dispatch work would be too optimistic. Releasing or retrying it would risk duplicate effects.

## Conservative restore rule

On trusted crash restoration:

```text
Reserved -> OutcomeUnknown
```

for every in-flight execution reservation, followed by persistence/acknowledgement of a successor checkpoint.

Only the newly acknowledged successor is returned as normalized recovery state.

If persistence fails or acknowledges a different head, normalization fails closed. The old trusted checkpoint still charges the reservation, so no authority is silently recovered.

If a checkpoint contains no `Reserved` executions, normalization is idempotent: no successor is emitted and the trusted head is unchanged.

## Read-only reconciliation

Recovery uses a dedicated `ServiceObserver` interface:

```text
observe(host, unit) -> ServiceObservation
```

There is no restart/mutation method in the reconciliation algorithm's trait bound.

A #305 `ServiceBackend` can be viewed through this read-only interface, but the recovery code itself cannot invoke its mutation method.

## Applied reconciliation

For an `OutcomeUnknown` restart, the reconciler requires:

- exact externally trusted checkpoint head;
- checkpoint payload valid against the exact capability grant;
- exact plan actor/audience/task/resource/operation bindings;
- exact supplied before-observation matching the plan/world commitment;
- exact reservation ID in `OutcomeUnknown` state.

It then observes the same host/unit.

If both before and after have non-empty systemd `InvocationID`s and the ID changed:

```text
OutcomeUnknown -> Committed
```

The accounting transition is persisted as a successor checkpoint before the reconciled state is returned.

## Still-unknown rule

If observation succeeds but does not prove a changed invocation:

```text
OutcomeUnknown -> OutcomeUnknown
```

No checkpoint rewrite is needed and the reservation remains charged.

If observation is unavailable, the same rule applies.

Ordinary service state cannot prove that *no* restart effect occurred, so v0.1 has no automatic `OutcomeUnknown -> Released` path.

## Revocation and expiry

Reconciliation is not a new mutation and does not consume another use. It records what an already-initiated effect did.

Therefore a later grant revocation or expiry must not block truthful reconciliation of existing `OutcomeUnknown` accounting. Those conditions prevent future actuation; they do not rewrite history.

The reconciliation path still verifies that the checkpoint/grant/plan/resource bindings are exact so it cannot use one unknown execution to account for a different host/unit/action.

## Evidence

`ReconciliationReceipt` contains:

- reservation ID;
- grant digest;
- plan digest;
- before-world digest;
- optional after-world digest;
- previous trusted checkpoint head;
- current checkpoint head;
- reconciliation outcome.

It retains no journal text, raw command output, stderr, or secrets.

## Tests authored

The deterministic suite covers:

- restored `Reserved` becomes `OutcomeUnknown` and remains charged;
- already-normalized state is idempotent and emits no extra checkpoint;
- changed InvocationID reconciles to committed use;
- unchanged InvocationID remains unknown without a checkpoint rewrite;
- observation failure remains unknown without a checkpoint rewrite;
- wrong externally trusted head rejects before normalization;
- mismatched before-observation/target rejects before observation;
- wrong checkpoint acknowledgement fails closed;
- the reconciliation API is typed around `ServiceObserver`, not a mutation interface.

## Non-claims

v0.1 does not claim:

- `InvocationID` proves user intent;
- unchanged InvocationID proves non-application;
- current health alone proves the uncertain restart caused recovery;
- checkpoint durability/authenticity beyond the supplied `CheckpointStore`/trusted-head boundary;
- host/kernel/root compromise resistance;
- automatic release of unknown effects;
- a right to perform another restart.

## Exit gate

Before calling #305 crash-recoverable on a real host:

1. #305 and this crate must pass exact-head compiler/Clippy/tests;
2. the checkpoint head must be retained independently of the checkpoint bytes;
3. a real crash-injection lane must terminate the broker at every boundary between reservation persistence, pre-dispatch observation, dispatch, post-dispatch persistence, and verification;
4. recovery must never issue a second restart while the first effect is unresolved;
5. where a new invocation occurred, recovery must converge to one committed use without double charging.
