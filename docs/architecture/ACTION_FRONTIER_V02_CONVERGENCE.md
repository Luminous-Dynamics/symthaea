# Action Frontier v0.2 — Convergence Contract

## Purpose

This tranche converges the strongest properties from the parallel authority-v0.2 action-runtime line and the stricter authority/accounting/frontier audit line without transferring qualification from either.

The common parent remains authority-core #4258. The source lineage below this document is the `ACTION-RUNTIME-V2-001` implementation family (#4277, #4301, `agency/action-frontier-v0.2-c`).

The convergence corrections are informed by the separate #4289/#4329 audit, but this tranche does not merge those branches or claim their results.

## Preserved V2A/V2B/V2C theorem

Keep:

```text
EffectIntentId
    != AttemptId
    != derived ReservationId
    != EffectBindingDigest

serialized snapshot
    != live GrantAccountV2

checkpoint payload validity
    != legal checkpoint transition
    != durable CAS persistence

PersistedReservationV2
    != DurablyArmedReservationV2
    != DispatchPermit
    != effect
```

The canonical checkpoint identity and strict predecessor/state-transition theorem from V2B remain authoritative for this line.

`PersistedReservationV2` and `DurablyArmedReservationV2` remain affine, process-local evidence of exact persisted transitions. They are neither Clone nor Serde and cannot be recreated from checkpoint bytes alone.

## Convergence correction 1 — no unauthenticated writable restart

The predecessor V2C exposed:

```text
resume_from_expected_current(
    store,
    grant,
    checkpoint,
    caller_supplied_expected_head,
)
```

That proves local payload/head equality but does not prove why the supplied head should be accepted as current after process loss. It could also recreate a writable frontier after an in-memory containment latch disappeared on restart.

This tranche removes that production constructor.

```text
serialized checkpoint/head
    != authenticated restart anchor
    != writable frontier
```

Generation-zero establishment is the only production constructor for a writable frontier until a future verifier-owned authenticated-anchor layer exists.

## Convergence correction 2 — no containment escape hatch

The predecessor exposed `into_inner()`, allowing the backing store to be extracted after the frontier object had entered containment.

This tranche removes that API.

Containment is intentionally monotonic within the frontier abstraction:

```text
Contained
    -> no frontier progression
    -> no typed persistence token minting
    -> no store extraction through this API
```

A future recovery path must establish new externally authenticated authority evidence rather than clearing this object's latch.

## Convergence correction 3 — fresh durable observation

`CheckpointCasStoreV2` now requires a linearizable:

```text
current_head()
```

against the same durable frontier used by CAS.

After a successful CAS acknowledgement, the frontier re-reads the store before it updates local expected state or returns a positive typed persistence fact.

```text
CAS(H -> H+1) acknowledged
+ fresh current_head() == H+1
    -> local expected frontier may advance

fresh current_head() != H+1
    -> StoreFrontierChanged
    -> containment
    -> no positive persistence token returned
```

This prevents a writer from knowingly minting a `PersistedReservationV2` or `DurablyArmedReservationV2` against a frontier that another writer already advanced again before the call returned.

The resulting proof is still point-in-time, not indefinite currentness. V2D must perform its own point-of-use current-authority composition immediately before dispatch permit minting.

## Convergence correction 4 — exact grant ownership

`CasFrontierV2` now owns the exact `CapabilityGrant` whose checkpoint lineage it progresses.

Callers no longer supply a grant on every persistence method. This prevents accidental grant substitution across calls and reduces the authority-bearing parameter surface.

The frontier's public local state is named:

```text
expected_checkpoint()
expected_head()
```

rather than `current_*`, because cached in-process state is not itself a fresh durable observation.

## Generation zero

Establishment requires:

```text
store.current_head() == None
+ GrantAccountCheckpointV2::first(exact grant, exact root account)
+ CAS(None -> generation zero)
+ acknowledgement == exact generation-zero head
+ fresh store.current_head() == exact generation-zero head
```

If a durable frontier already exists, this tranche returns `StoreNotEmpty`; it does not reinterpret that state as safe restart evidence.

## Reservation / arming order

The strong V2C order remains:

```text
exact account PRE-state
    -> reserve exact use/risk
    -> strict successor with one new Reserved record
    -> CAS + fresh store observation
    -> PersistedReservationV2

fresh point-of-use current-authority check (future V2D)
    -> strict exact Reserved -> OutcomeUnknown successor
    -> CAS + fresh store observation
    -> DurablyArmedReservationV2

DurablyArmedReservationV2
+ same-object fresh authority/admission proof (future V2D)
    -> DispatchPermit
    -> effect entry
```

Durable `OutcomeUnknown` must exist before a dispatch permit can exist.

## Authority refund policy

The V2A/V2B model retains `Released` as a semantic state because future verified reconciliation may need it.

This convergence contract narrows production authority:

```text
ordinary caller assertion
pre-dispatch convenience API
revocation
failure/timeout
OutcomeUnknown
    != authority refund
```

Returning capacity must require a future verifier-owned exact-grant/exact-attempt `ProvenNotApplied` reconciliation receipt and a strict persisted transition authorized by that receipt.

Until that layer exists, normal production admission should not expose an authority-return path.

This combines the V2A identity/state model with the stricter fail-closed accounting insight from the parallel audit.

## Restart target

The next restart layer should follow the evidence architecture already demonstrated by the witness-frontier line (#452/#465/#469/#473):

```text
source-specific authenticated/fresh external frontier evidence
        +
transport-neutral verified external frontier projection
        +
live guarded local durable frontier
        ->
restart/currentness decision that retains provenance
```

Do not replace this with a caller-implementable `Authenticator` trait that can mint an opaque restart token from arbitrary bytes.

A Xenia-specific verifier may produce opaque authenticated checkpoint-anchor evidence; a small adapter may project it into a generic external-anchor proof while retaining the source-specific object. Writable restart should require that opaque proof plus a fresh local durable-store observation of the exact same checkpoint.

## V2D admission target

The eventual point-of-use theorem should consume, by ownership where appropriate:

```text
exact CapabilityGrant v2
+ VerifiedAuthorityTime
+ VerifiedAuthorityStateV2
+ exact durable frontier/accounting for the same grant
+ PersistedReservationV2
+ unchanged exact effect binding
    -> reserved-use currentness
    -> strict durable OutcomeUnknown transition
    -> DurablyArmedReservationV2
    -> same-object affine DispatchPermit
```

The second currentness check must not count the owned reservation as a request for another use.

Revocation/context shutdown after reservation blocks permit minting but does not refund the reservation.

## Scientific/cognitive non-authority

Nothing in this frontier accepts or derives authority from Phi, confidence, posterior probability, scientific support, reputation, utility, urgency, anomaly score, or simulation success.

Those signals may constrain later policy or safety layers; they cannot create a missing persistence/admission token.

## Qualification boundary

All changes remain authored source until exact-head Rust 1.96 rustfmt/check/test/strict-Clippy plus lock freshness execute successfully.

No result transfers from historical authority-v0.1, #4289/#4329, or predecessor draft branches merely because algorithms are reused.
