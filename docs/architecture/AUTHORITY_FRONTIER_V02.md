# Authority Frontier v0.2 — r2

## Purpose

Port the historical compare-and-swap authority-frontier theorem onto the v0.2 grant/accounting lineage without overclaiming bootstrap-head authenticity, restart authority, or execution authority.

This tranche sits above #4289's grant-bound accounting and below any future Xenia/TPM/append-only-log authenticator.

## Narrow theorem

```text
exact CapabilityGrant
+ empty linearizable durable store
+ exact generation-zero checkpoint
+ every successor payload valid under the exact grant
+ every successor installed by full-head compare-and-swap
+ fresh store read still equals adapter expected head
+ accounting bound to the same exact head/grant
    -> FrontierBoundGrantAccounting
```

`FrontierBoundGrantAccounting` is non-Serde and non-Clone.

## No writable restart path in v0.2

This tranche intentionally exposes **no production reopen constructor**.

A plain serialized `CheckpointHead` is not enough to recreate a writable frontier after process loss. That prevents a caller from clearing an in-memory containment latch simply by replaying the same local head after restart.

Therefore:

```text
persisted head
    != authenticated restart anchor
    != writable frontier
```

Restart/recovery activation is deferred to the next external-authentication tranche, which must prove the anchor's source/currentness through Xenia, TPM custody, an append-only log, or an equivalent authority.

## What this does not prove

`FrontierBoundGrantAccounting` is a point-in-time local durability fact. It does not prove:

```text
external anchor authenticity
global currentness
indefinite freshness
execution admission
```

Another authorized writer may advance the durable frontier immediately after the object is created. Any future admission theorem must consume it together with a fresh or atomic frontier transition rather than treating it as a capability.

## Exact grant ownership

The writable frontier owns the exact `CapabilityGrant` whose accounting it governs.

Generation-zero establishment requires:

```text
grant.validate() == OK
store.current_head() == None
```

All later checkpoint validation is performed against that exact owned grant.

## Full checkpoint validation before CAS

Outer successor shape is not enough. Before a successor may reach the durable store, the frontier executes:

```text
checkpoint.verify_payload(&exact_grant)
```

This revalidates the checkpoint schema and the complete crash-conservative accounting snapshot under the exact grant ceiling. A malformed snapshot with a correct sequence, predecessor, and grant digest cannot advance the frontier.

Only after payload verification does the frontier require:

```text
checkpoint.grant_digest == exact_grant.digest()
checkpoint.sequence == expected_head.sequence + 1
checkpoint.previous_checkpoint_digest == expected_head.digest
```

and attempt the full-head CAS.

## Full-head identity

`CheckpointHead` v0.2 includes:

```text
grant_digest
sequence
checkpoint_digest
```

The CAS contract compares the complete head. Successors may not change grants, skip sequence, or name a predecessor digest other than the adapter's exact expected head.

## Fresh observations around progression

`expected_head()` is intentionally named as cached adapter state, not current durable state.

After a successful CAS acknowledgement, `persist_successor()` performs another linearizable `current_head()` read before returning. If another writer already moved the store again, the frontier latches containment instead of returning a known-stale expected head.

Likewise, `bind_current_accounting()` performs a fresh store read before producing a frontier-bound object.

This closes races such as:

```text
frontier A expects H
another writer advances durable store to H+1
frontier A tries to bind accounting for H
    -> StoreFrontierChanged
    -> frontier A contained
```

## Containment

The frontier latches containment after:

- checkpoint payload validation failure during progression;
- CAS/store failure;
- a store head change observed outside the frontier;
- invalid successor sequence/digest/grant;
- store acknowledgement of an unexpected successor head.

Contained frontiers cannot continue authority progression or bind accounting.

The production API exposes no `clear_containment()`, no `into_inner()` escape hatch, and no unauthenticated restart constructor.

## Generation zero

`establish_grant_frontier()` requires a valid grant and empty store, creates an exact grant-bound generation-zero checkpoint, installs it with `compare_and_swap(None, checkpoint)`, and re-reads the store before returning.

If the store has already advanced before the final observation, establishment fails rather than returning a stale writable frontier.

## Store contract

`CheckpointCasStore` requires:

- `current_head()` to be a linearizable read of the same durable frontier used by CAS;
- `compare_and_swap()` to compare the full current head and durably install the complete checkpoint atomically;
- successful CAS acknowledgement to equal the exact installed checkpoint head.

A backend that implements only ordinary read-then-write does not satisfy this contract.

## Next composition step

The next external-authentication tranche should establish an authenticated restart/current-head fact, then introduce the only allowed production activation path for a persisted frontier.

Eventually, admission should compose:

```text
fresh authenticated checkpoint-head evidence
+ exact CheckpointHead identity
+ FrontierBoundGrantAccounting
+ VerifiedAuthorityStateV2
+ VerifiedAuthorityTime
    -> admission-eligible current authority facts
```

Even that composition must remain separate from durable one-use reservation and effect execution.

## Non-claims

This tranche does not authenticate Xenia, TPM, or institutional custody; does not implement restart recovery; does not establish global consensus on a checkpoint; does not construct `AuthorityEvaluationInput`; does not mint live authority; and does not execute an effect.
