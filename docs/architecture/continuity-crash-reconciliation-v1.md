# Crash reconciliation v1

An exact A -> B execution intent can survive a process or host crash even when no trustworthy backend result survives. The durable intent remains spent anti-replay state and therefore creates a reconciliation obligation, never retry permission.

```text
anchored durable intent A -> B
  + no receipt
  + independently qualified physical observation
      -> classify current world
```

The classification is one of:

```text
exact source A observed  -> source-health proof required
exact target B observed  -> target-health proof required
other exact C observed   -> explicit intervention required
unreachable              -> fresh physical observation required
unknown                  -> fresh physical observation required
```

There is deliberately no `Retry` classification.

## Exact reconciliation boundary

`QualifiedCrashReconciliationV1::qualify()` requires:

- exact known-good-bound durable A -> B intent;
- exact reconstructed journal containing that attempt;
- exact trusted eligibility marked spent by that attempt;
- no result receipt for the attempt;
- disposition exactly `AwaitingReconciliation`;
- a qualified rollback-resistant anchor covering that exact journal/subject world;
- an independently qualified post-execution observation for the exact attempt, subject, B target, and distributed context;
- observation qualification strictly after the A -> B commit epoch and not before the intent journal anchor.

The existing post-execution observer is reused. No second observer vocabulary is introduced.

## Interpretation of the observation

`ExpectedTargetObserved` means exact B is currently observable and maps to `TargetObserved`.

`DifferentKnownRealization` is compared against exact source A:

- if observed realization == A: `SourceKnownGoodObserved`;
- otherwise: `OtherKnownRealizationObserved`.

`Unreachable` and `Unknown` remain unresolved.

Known physical states must carry a non-zero physical-state digest. Unknown/unreachable states must not fabricate an identity or state digest.

## Persistence and rebinding

`CrashReconciliationRecordV1` is serializable audit evidence. It commits the exact A -> B intent lineage, journal digest, qualified journal-anchor id, independent observation id, resulting classification, exact observed realization/state digest where known, and reconciliation time.

A persisted record cannot recreate the proof. `QualifiedCrashReconciliationV1::rebind()` recomputes the classification against the exact live intent, journal, anchor, and observation and requires record equality.

## Security theorem

```text
IntentWithoutReceipt != RetryPermission
ObservedA != HealthyA
ObservedB != HealthyB
ObservedA != RecoveredA
ObservedB != PromotedB
```

Reconciliation answers only: **what exact world can independent evidence establish now?**

## Next boundary

The current target-health type intentionally applies only when the exact expected target B was observed. Therefore `SourceKnownGoodObserved` cannot be silently fed into B-target health logic.

The clean next abstraction is an exact local-health snapshot whose identity basis is explicit. It can later accept:

- expected post-execution target B;
- crash-reconciled source A;
- initial baseline A for first Spore adoption.

That shared abstraction can close both crash-source health and issue #1429 bootstrap without fabricating execution lineage.
