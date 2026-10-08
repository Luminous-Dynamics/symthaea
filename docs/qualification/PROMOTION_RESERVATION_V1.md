# Promotion Reservation v1

## Purpose

This tranche makes the promotion effect boundary explicit without pretending the local ledger and GitHub are one atomic transaction.

The composition is:

```text
HistoricalQualificationReceipt
        ↓
QualificationClaimDispositionV1
        ↓
PromotionEligibilityLeaseV1
        ↓
PromotionReservationV1
        ↓
PromotionDispatchIntentV1
        ↓
external provider
        ↓
provider/effect observation
        ↓
PromotionCompleted | PromotionRejected
                 \→ PromotionOutcomeUnknown
                         ↓
                 PromotionReconciliationRequired
```

PromotionEligibilityLeaseV1 remains the narrow decision-time projection. PromotionReservationV1 consumes that lease exactly once. PromotionDispatchIntentV1 is the crash-recovery fence created before the external call.

The reservation is therefore not an authorization that can be reused indefinitely, and a provider handle is not the local operation identity.

## Reservation contract

A valid reservation binds, at minimum:

- schema/version;
- domain identity;
- unique reservation_id;
- unique local promotion_operation_id;
- exact ledger predecessor/head;
- exact qualification subject;
- exact lease identity;
- exact disposition identity and generation;
- exact trust-root identity/generation;
- exact governance snapshot;
- repository identity;
- exact PR/base/head identity;
- exact expected PR-head SHA;
- provider capability profile;
- decision-time validity/deadline.

The reservation is admitted only when:

```text
observed_ledger_head == current_ledger_head
AND
active_lease == reservation.lease_id
AND
no reservation already consumes that lease
AND
reservation.predecessor == observed_ledger_head
```

The successful ledger successor consumes the lease. Any competing candidate from the old predecessor is stale/non-authoritative even if its Git object still exists.

## Why the dispatch-intent record is separate

The external boundary has an unavoidable crash window:

```text
persist reservation
persist dispatch intent
        ↓
provider accepts request
        ↓
process crashes before local response persistence
```

Without the intent, recovery cannot distinguish "never attempted" from "may already have happened".

PromotionDispatchIntentV1 therefore records before the provider call:

- the same promotion_operation_id;
- reservation identity;
- reservation predecessor/head;
- exact expected PR-head SHA;
- provider name/profile;
- requested merge action/method;
- dispatch deadline;
- a local attempt sequence;
- the fact that dispatch is now permitted.

The provider may still accept or reject the operation independently. The local record does not claim that the effect occurred.

## One-shot rules

For one lease identity:

```text
0 active reservation
    -> exactly 1 reservation
    -> 0 or 1 dispatch intents
```

A second coordinator may not create a second reservation. A second dispatch intent for the same reservation is invalid.

A retry after an unknown outcome may only reuse the same local operation identity and must first reconcile the external effect. A different operation identity is never an automatic recovery path.

## Provider boundary: GitHub

The current GitHub REST API documents that:

- the normal ref-update endpoint defaults to force=false, preserving fast-forward-only updates, and returns 409 Conflict for conflicts;
- asynchronous pull-request merge uses an exact PR-head SHA (sha);
- a new async request can return 202 with a provider UUID;
- a duplicate pending async request returns 409 with the existing UUID;
- an already merged or already queued PR can return 200;
- enqueued does not mean the PR has merged;
- async result records are retained for 24 hours and then their UUID lookup can return 404.

These capabilities are useful, but they are four separate properties:

```text
exact-subject CAS
provider operation handle
duplicate-request reconciliation
durable effect reconciliation
```

Do not collapse them into one "idempotent transaction" capability.

## Terminology correction for the ledger CAS

The REST Git-ref lab establishes a strong and useful theorem for this ledger shape:

```text
single-parent append-only successor
+ force=false
+ all trusted writers share one ref
→ stale sibling successor cannot become current
```

That is stronger than last-write-wins, but it should not be described as a generic compare-and-set primitive.

GitHub's GraphQL updateRefs mutation provides the more explicit primitive: beforeOid requires the ref to point to the expected OID, and multiple ref updates are applied atomically. That can be named explicit ref CAS where available.

The current ledger design does not require GraphQL; the REST fast-forward fence is sufficient for its append-only single-parent theorem. The distinction matters when extending the protocol to non-fast-forward or multi-ref transactions.

## Failure semantics

```text
Reservation failure
    -> no provider call

Dispatch intent persisted, provider call not known to have happened
    -> reconcile before any retry

Provider accepted, response lost
    -> PromotionOutcomeUnknown
    -> PromotionReconciliationRequired
    -> same operation identity only

Provider returns duplicate UUID
    -> recover the existing provider handle
    -> do not create a second operation

Provider returns enqueued
    -> not PromotionCompleted
    -> continue/reconcile against PR merged state

Provider UUID expires
    -> provider handle is no longer available
    -> use durable PR/effect state
    -> never infer success from expiry

Exact subject SHA mismatch
    -> PromotionRejected
    -> no substitution to the current head

Trust root changes after dispatch
    -> do not rewrite historical dispatch
    -> fence follow-on promotion
    -> reconcile/adjudicate the effect under the newer root
```

## Reconciliation rule

Terminal completion requires independent observation of the actual effect surface.

For GitHub merge, a provider async result of merged is useful evidence, but the durable PR merged state is the final reconciliation surface. An enqueued result is not terminal completion.

When the provider result UUID is unavailable or expired, recovery must inspect the PR state. If the PR is merged, record the observed merge commit. If it is not merged but there is still evidence of a queued/pending operation, remain reconciliation-required. A new dispatch is permitted only when the external system's durable state establishes that the previous effect did not happen and no conflicting operation remains.

## Deterministic acceptance corpus

The independent model exercises:

1. all 20 legal read/construct/CAS interleavings for two writers;
2. duplicate reservation;
3. dispatch-intent persistence before provider call;
4. crash before provider call;
5. timeout after provider acceptance;
6. duplicate async request;
7. enqueued not equal to completion;
8. already-merged response;
9. expired provider UUID with merged PR;
10. expired provider UUID without merged effect;
11. exact subject-head mismatch;
12. trust-root change before dispatch;
13. trust-root change after dispatch;
14. stale coordinator retry from an old ledger predecessor.

The harness imports no Symthaea production code and has no promotion credentials or merge authority.

## Claim ceiling

A passing model establishes only:

```text
the local single-use reservation/dispatch-intent state machine is
internally consistent under the enumerated synthetic failure cases.
```

It does not establish:

- production implementation correctness;
- GitHub provider truthfulness;
- atomicity between the ledger and GitHub;
- governance legitimacy;
- scientific correctness;
- successful external promotion without independently observed effect evidence.

Related: #7067, #7068, #7080, #7085.
