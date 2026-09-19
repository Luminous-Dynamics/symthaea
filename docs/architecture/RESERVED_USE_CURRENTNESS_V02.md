# Reserved-Use Currentness v0.2 — V2D-A

## Exact composition lineage

This tranche is the first explicit composition of the two authority-v0.2 descendants that were intentionally kept separate until their respective theorems were strong enough:

- durable action runtime/checkpoint/frontier: `agency/action-frontier-v0.2-d-convergence@f85c1688138f17008070088e6d22484619852eb0` (#4332);
- verifier-owned current authority state: `agency/verified-authority-state-v0.2-r2@3650bbac8b22900a5c9dca804fc85d8d761532ba` (#4264).

The source convergence commit is a real two-parent Git merge:

`e8dfbaf27f4c7903dab7510c687688efc32e7091`

No qualification result transfers merely because those source histories are composed.

## The question V2D-A answers

A new use and an already-allocated use are not the same authority question.

For a one-use grant, after an exact reservation is durably persisted, ordinary new-use evaluation correctly sees:

```text
committed + reserved == max_uses
    -> UseBudgetExhausted
```

That predicate must block a second allocation. It must not make the already-owned exact reservation impossible to progress.

V2D-A therefore asks only:

> Is this exact already-persisted reservation still current under every authority predicate that can invalidate the grant, without allocating another use?

It does not subtract counters, fabricate a smaller `GrantUseState`, or refund authority.

## Reserved-use theorem

```text
exact CapabilityGrant v2
+ affine PersistedReservationV2 for that exact grant
+ adapter-local exact reservation/head identity
+ fresh VerifiedAuthorityStateV2 for that exact grant
+ VerifiedAuthorityTime bound to the same grant/time policy
+ exact current authority epoch
+ exact current AuthorityContextRef
+ expiry still valid under conservative trusted time
+ no applicable verified negative-authority fact
    -> ReservedUseCurrentnessV2
```

The non-use predicate preserves the authority-core semantics for:

- structural grant validity;
- root-only live execution (`DelegationChainRequired` remains closed);
- exact authority epoch;
- exact authority context;
- expiry;
- exact grant revocation;
- context revocation;
- issuer/subject/audience tombstone;
- resource freeze;
- minimum resource epoch.

The only intentionally absent predicate is `UseBudgetExhausted`, because the affine reservation proves that this use was already allocated.

A differential test compares this predicate with `evaluate_authority` whenever a new use remains available. A separate one-use regression proves the semantic distinction explicitly.

## Proof-object retention

`ReservedUseCurrentnessV2` owns both:

- the affine `PersistedReservationV2`;
- the opaque `VerifiedAuthorityStateV2`.

Neither proof can be moved back out through the public API. This prevents later code from detaching the reservation from the verified state that was checked for it.

The type is non-Clone and non-Serde.

## Local-frontier non-claim

Constructing `ReservedUseCurrentnessV2` checks that the affine reservation still names the frontier adapter's expected head and exact expected `Reserved` record.

That is deliberately **not** described as a fresh durable-store observation.

The stronger operation consumes reserved-use currentness through #4332's existing frontier CAS:

```text
ReservedUseCurrentnessV2
+ re-check trusted-time/state freshness
+ re-check all non-use authority predicates
+ exact Reserved -> OutcomeUnknown checkpoint successor
+ compare_and_swap(expected_head, successor)
+ exact acknowledgement
+ fresh current_head() == successor head
    -> CurrentnessBoundArmedReservationV2
```

If another writer advances the local frontier after currentness was constructed, the stronger arming transition fails closed. A stale currentness object therefore cannot become the stronger armed object.

## Authority-state race remains explicit

`CurrentnessBoundArmedReservationV2` retains the exact verified authority-state proof used immediately before durable arming, but it is still **not** an effect-entry capability.

A source authority can change after a short-lived verified state was obtained. V2D-B must close that final point-of-effect race with a guarded/admission design appropriate to the authoritative source before defining an affine `DispatchPermit`.

V2D-A intentionally stops before that boundary.

## Failure semantics

Fail closed on at least:

- exact grant mismatch among grant/frontier/reservation/verified state;
- contained frontier;
- persisted head mismatch;
- reservation record or effect-binding mismatch;
- no active verified authority context;
- stale trusted time or verified state;
- epoch/context mismatch;
- expiry;
- any applicable verified negative-authority fact;
- exact-head CAS conflict;
- wrong CAS acknowledgement;
- fresh post-CAS durable frontier mismatch.

No failure returns or refunds authority.

## Non-claims

This tranche does not:

- allocate a new grant use;
- verify delegation ancestry;
- authenticate a restart anchor;
- release/refund a reservation;
- prove effect success;
- produce a dispatch permit;
- execute an effect;
- turn simulation, cognition, Phi, confidence, scientific evidence, or utility into permission.

## Qualification boundary

The implementation and tests are authored evidence only while the PR is draft. The dedicated exact-head workflow is draft-suppressed and must execute pinned Rustfmt/check/test/strict-Clippy plus clean-lock and clean-tree gates on the exact candidate head before any qualification claim is made.

Queued, skipped, cancelled, partial, or predecessor results do not become V2D-A PASS evidence.
