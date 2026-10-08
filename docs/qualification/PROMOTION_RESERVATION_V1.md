# Promotion Reservation v1

This tranche turns the lease-to-external-effect boundary into an explicit single-use transaction model.

HistoricalQualificationReceipt -> QualificationClaimDispositionV1 -> PromotionEligibilityLeaseV1 -> PromotionReservationV1 -> PromotionDispatchIntentV1 -> external provider -> observed outcome.

## Single-use reservation

A reservation binds the exact ledger predecessor/head, lease identity, qualification disposition generation, trust-root identity/generation, governance snapshot, repository and PR identity, exact expected PR-head SHA, a unique local promotion_operation_id, and the provider capability profile.

The successful reservation is itself a ledger successor and consumes the active lease. A competing coordinator cannot reserve the same lease from the old predecessor.

## Dispatch-intent recovery fence

A provider call has an unavoidable crash window:

persist reservation -> persist dispatch intent -> provider may accept -> process dies before response persistence.

PromotionDispatchIntentV1 is therefore durable before the external call. It records the reservation identity, local operation identity, exact expected subject SHA, provider/profile, action, deadline, and attempt sequence.

After dispatch intent exists, uncertainty is never resolved by blindly creating a new operation identity. Recovery must first reconcile the external effect.

## Two-writer publication model

The independent oracle executes all 20 legal interleavings of evaluator and invalidator read/construct/commit steps.

18 schedules have concurrent commits from the same predecessor: exactly one successor wins.

2 schedules are genuinely sequential: the second writer reads the already-advanced head and legitimately creates the next successor.

This is the stronger claim: the model executes each schedule against a state machine rather than merely counting permutations.

## Failure semantics

    reservation failure
      -> no provider call

    provider acceptance with lost response
      -> PromotionOutcomeUnknown
      -> PromotionReconciliationRequired
      -> no blind second operation

    duplicate async request
      -> recover existing provider UUID
      -> do not create another operation

    enqueued
      -> not PromotionCompleted

    provider result UUID expires
      -> query durable effect state
      -> never infer success from expiry

    exact subject SHA mismatch
      -> PromotionRejected

    trust-root change before dispatch
      -> reservation superseded
      -> no external dispatch

    trust-root change after dispatch
      -> preserve historical dispatch
      -> fence follow-on promotion
      -> reconcile/adjudicate effect

## Provider capability boundary

Keep these properties separate:

exact-subject CAS
provider operation handle
duplicate-request reconciliation
durable effect reconciliation

The REST Git-ref lab in #7067 establishes a narrower publication theorem: same-parent append-only successors with force=false serialize at the ref publication point. It is not a generic CAS primitive.

GitHub GraphQL updateRefs is a stronger expected-OID primitive for cases that require explicit beforeOid checks or atomic multi-ref updates.

The async pull-request merge API supports exact expected PR-head SHA binding, asynchronous provider UUIDs, duplicate-pending UUID recovery, and distinct enqueued versus merged states. Provider result retention is bounded, so durable PR merged state remains a reconciliation surface.

## Claim ceiling

A passing lab establishes only internal consistency of the synthetic local reservation/reconciliation model under the enumerated failure cases.

It does not establish production implementation correctness, atomicity between the ledger and GitHub, provider truthfulness, governance legitimacy, scientific correctness, or successful external promotion without independently observed effect evidence.

Related: #7067, #7068, #7080, #7085.
