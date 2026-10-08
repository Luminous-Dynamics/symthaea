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

## Effect observation versus causal attribution

A terminal effect observation is not automatically a causal attribution to the local promotion operation.

The reference provider model distinguishes:

    provider-operation result reports merged
      -> effect observed
      -> causal attribution established

from:

    enqueued final queue result
      -> later durable PR state reports merged
      -> effect observed
      -> causal attribution unestablished

and:

    async result expired
      -> durable PR state reports merged
      -> effect observed
      -> causal attribution unestablished

A local receipt carrying `promotion_operation_id`, expected PR-head SHA, and merge commit cannot
mint the stronger causal proposition by itself. This is why #7101 is a separate semantic boundary.

The distinction is especially important for stacked merges: GitHub documents that a stack merge can
merge or queue every open PR in the stack up to the requested PR, so a later observed group effect must
not be compressed into one single-PR operation identity. citeturn965026search2turn965026search6

The provider API currently documents `enqueued` as final for the merge-queue request, with eventual
merge state exposed separately through pull-request state; asynchronous result records expire after
24 hours. citeturn965026search0

## Claim ceiling

A passing lab establishes only internal consistency of the synthetic local reservation/reconciliation model under the enumerated failure cases.

It does not establish production implementation correctness, atomicity between the ledger and GitHub, provider truthfulness, governance legitimacy, scientific correctness, or successful external promotion without independently observed effect evidence.

Related: #7067, #7068, #7080, #7085.
