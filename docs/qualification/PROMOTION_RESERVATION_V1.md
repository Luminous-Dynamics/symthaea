# Promotion Reservation v1

This tranche turns the lease-to-external-effect boundary into an explicit single-use transaction model.

HistoricalQualificationReceipt -> QualificationClaimDispositionV1 -> PromotionEligibilityLeaseV1 -> PromotionReservationV1 -> PromotionDispatchIntentV1 -> external provider -> observed outcome.

## Single-use reservation

A reservation binds the exact ledger predecessor/head, lease identity, qualification disposition generation, trust-root identity/generation, governance snapshot, repository and PR identity, exact expected PR-head SHA, a unique local promotion_operation_id, and the provider capability profile.

The successful reservation is itself a ledger successor and consumes the active lease. Reservation admission also requires the supplied trust-root generation to equal the current ledger trust-root generation. A competing coordinator cannot reserve the same lease from the old predecessor or from a stale trust root.

## Dispatch-time re-fencing

A reservation is not perpetual authority. Before dispatch, the coordinator must re-read the shared ledger fence and require:

    current ledger head == reservation_head
    current fencing token == reservation.fencing_token
    current trust-root generation == reservation.trust_root_generation
    reservation state == Reserved

A monotonic fencing token moves the stale-holder check to the protected resource boundary: a suspended coordinator with an older token must be rejected rather than trusting its historical lease. The token is a local ledger fence; GitHub does not enforce it.

This closes a second-order race where an unrelated ledger transition advances the shared current head without explicitly mutating the old reservation record.

## Dispatch-intent recovery fence

A provider call has an unavoidable crash window:

persist reservation -> persist dispatch intent -> provider may accept -> process dies before response persistence.

PromotionDispatchIntentV1 is therefore durable before the external call. It records the reservation identity, local operation identity, exact expected subject SHA, provider/profile, action, deadline, and attempt sequence.

After dispatch intent exists, uncertainty is never resolved by blindly creating a new operation identity. Recovery must first reconcile the external effect. Completion additionally requires a receipt bound to the same local promotion-operation identity and exact expected PR-head SHA; a local "completed" transition without effect evidence is invalid.

## Two-writer publication model

The independent oracle executes all 20 legal interleavings of evaluator and invalidator read/construct/commit steps, then separately checks that an unrelated later ledger transition invalidates the old dispatch fence.

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
      -> compare returned provider UUID and stored merge options
      -> exact parameter match: recover existing provider UUID
      -> parameter mismatch: explicit reconciliation/parameter-mismatch state; do not silently reuse

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

## Provider effect observation versus causal attribution

External effect state and operation-level causality are separate propositions.

A provider result may establish a narrow causal edge only when the durable dispatch intent already binds the exact local promotion operation identity to the provider operation identity, and the provider result itself reports the corresponding successful effect.

The following do not establish that causal edge by themselves:

    expected PR head matches
    merge commit is non-empty
    local receipt carries a promotion operation ID
    a provider UUID exists or once existed
    durable PR state later reports merged

In particular, an asynchronous result of enqueued is final for the merge-queue request, while the eventual merged PR state is a separate durable observation. An expired provider UUID followed by an observed merged PR is therefore effect evidence without automatic causal attribution. An already-merged retry likewise observes the effect state but must not backdate the retry as the historical causal operation.

The dedicated provider-effect attribution qualifier records two explicit classes:

    DirectProviderResult + exact local/provider operation binding
        -> narrow causal attribution may be established

    DurableSubjectObservation
        -> effect observed; causal attribution remains unestablished

A locally authored reconciliation record is not independent provider evidence.

## Independent oracle integrity

The Python oracle is itself subject to the same execution-evidence discipline as the Rust reference.

The repaired revision requires the dispatch-fence arguments used by its tests, implements the
reconciliation completion transition its tests invoke, and registers every defined adversarial test
in the executable `TESTS` list. A source-level test definition that is not actually reachable from
the runner is not execution evidence.

The current receipt object remains a local reconciliation-model record only. It does not provide
independent provider provenance or prove that a later observed effect was caused by the specific
provider operation. That stronger proposition is tracked separately in #7101.

The dedicated qualification workflow remains deliberately provider-dispatch-free, so neither the
reference transaction nor the oracle may be described as hosted-executed until an actual trusted
run records their execution.

## Claim ceiling

A passing lab establishes only internal consistency of the synthetic local reservation/reconciliation model under the enumerated failure cases.

It does not establish production implementation correctness, atomicity between the ledger and GitHub, provider truthfulness, governance legitimacy, scientific correctness, or successful external promotion without independently observed effect evidence.

Related: #7067, #7068, #7080, #7085, #7101, #7113.
