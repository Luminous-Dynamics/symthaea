# Known-good execution coordinator v1

The coordinator turns the crash-boundary rule into a type-level protocol.

```text
ActiveLkg(A)
  + TrustedEligibility(B)
  + PreviousQualifiedJournalAnchor
      -> PendingAnchoredKnownGoodExecution
          exposes: durable A -> B intent only
          withholds: executable prepared attempt

Persist exact intent
  -> reconstruct journal
  -> eligibility is spent
  -> attempt is AwaitingReconciliation
  -> qualify fresh rollback-resistant journal anchor
  -> fresh anchor directly extends captured predecessor
      -> ReadyKnownGoodExecutionAttempt
          -> physical backend may execute
```

## Required release conditions

The pending state releases execution only when all of the following are exact:

- the durable journal contains the exact attempt id;
- the journal maps the exact trusted eligibility to that attempt;
- the eligibility is spent;
- no receipt already exists for the attempt;
- the disposition is `AwaitingReconciliation`;
- the qualified anchor covers the exact reconstructed journal digest and entry count;
- the anchor subject and trusted epoch match the execution session;
- anchor profile/root lineage matches the captured predecessor;
- anchor sequence is exactly predecessor + 1;
- the anchor explicitly names the captured predecessor anchor id;
- the journal entry count advanced;
- anchor time matches the exact commit epoch and does not move backwards;
- the privately held prepared attempt still matches the durable A -> B intent.

## Security properties

`PendingAnchoredKnownGoodExecutionV1` is non-Serde and non-Clone. It has no accessor for the prepared physical attempt.

`ReadyKnownGoodExecutionAttemptV1` is non-Serde and non-Clone and owns both the prepared attempt and the executor session. The backend can inspect exact execution identifiers and then consumes the ready token to produce one backend receipt.

A backend receipt remains audit material only. Independent post-execution observation, local health, and distributed health remain required before checkpointing or promotion.

## Crash behavior

If the process crashes after the durable intent is persisted but before the ready token is released, the prepared attempt is lost while the eligibility remains spent in the reconstructed journal. Restart therefore enters reconciliation rather than retry.

If the process crashes after the anchor is qualified but before physical mutation, restart still sees the exact spent intent and must reconcile actual state. The anchor does not grant retry authority.

## Non-claims

The coordinator does not authenticate journal-anchor platform evidence by itself, perform physical I/O, infer success from a receipt, promote Last Known Good, or permit recovery. Those remain distinct boundaries.
