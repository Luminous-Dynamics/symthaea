# Coordinator v1 state machine

```text
BoundEligibility(A -> B)
    |
    | prepare (consume eligibility)
    v
PendingAnchoredKnownGoodExecution
    |
    | expose exact durable intent only
    v
Persist + reconstruct journal
    |
    | exact attempt present / eligibility spent / no receipt
    v
Qualify direct successor journal anchor
    |
    | exact predecessor + profile/root + epoch + digest + count
    v
ReadyKnownGoodExecutionAttempt
    |
    | physical adapter executes and consumes token
    v
Backend receipt (audit claim only)
```

Any restart after durable intent formation loses the non-Serde executable path and therefore enters reconciliation. Durable intent is never retry permission.
