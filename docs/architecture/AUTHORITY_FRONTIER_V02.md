# Authority Frontier v0.2 — r2

## Purpose

Port the historical compare-and-swap authority-frontier theorem onto the v0.2 grant/accounting
lineage without overclaiming bootstrap-head authenticity.

This tranche sits above #4289's head-bound accounting and below any future Xenia/TPM/append-only-log
authenticator.

## Narrow theorem

```text
supplied bootstrap anchor
+ durable store currently equals that anchor
+ every successor installed by full-head compare-and-swap
+ fresh store read still equals adapter expected head
+ accounting bound to the same exact head
    -> FrontierBoundGrantAccounting
```

`FrontierBoundGrantAccounting` is non-Serde and non-Clone.

## What this does not prove

The supplied bootstrap anchor is not authenticated by this crate. A caller can possess or replay a
`CheckpointHead`. Therefore:

```text
FrontierBoundGrantAccounting
    != externally authenticated accounting
    != globally current accounting
    != execution admission
```

A later Xenia/TPM/append-only-log verifier must authenticate the bootstrap/current anchor before a
stronger currentness claim is possible.

## Full-head identity

`CheckpointHead` v0.2 includes:

```text
grant_digest
sequence
checkpoint_digest
```

The CAS contract compares the complete head. Successors may not change grants, skip sequence, or
name a predecessor digest other than the adapter's exact expected head.

## Fresh bind check

The historical in-memory expected-head pattern is strengthened here. `bind_current_accounting()`
performs a fresh durable-store read before producing a frontier-bound object.

This closes the race:

```text
adapter A reopens at H
adapter B advances store to H+1
adapter A tries to bind H
    -> StoreFrontierChanged
    -> adapter A contained
```

The adapter getter is intentionally called `expected_head()`, not `current_head()`, because its cached
value is not itself a fresh store observation.

## Containment

The frontier latches containment after:

- CAS/store failure;
- a store head change observed outside the adapter;
- invalid successor sequence/digest/grant;
- store acknowledgement of an unexpected successor head.

Contained frontiers cannot continue authority progression or bind accounting.

## Generation zero

`establish_grant_frontier()` requires an empty store, creates an exact grant-bound generation-zero
checkpoint, and installs it with `compare_and_swap(None, checkpoint)`.

The returned head is current relative to that store at establishment, but it still requires external
authentication before it can be used as a trusted restart anchor.

## Restart semantics

`reopen_from_anchor_claim()` succeeds only when the durable store currently reports exactly the
supplied full head. The function name deliberately says *claim*: equality with local storage does not
prove the anchor itself is externally trustworthy.

## Next composition step

The next external-authentication tranche should compose:

```text
fresh Xenia/other authenticated head evidence
+ exact CheckpointHead identity
+ FrontierBoundGrantAccounting
+ VerifiedAuthorityStateV2
+ VerifiedAuthorityTime
    -> admission-eligible current authority inputs
```

Even that composition must remain separate from effect execution and one-use reservation.

## Non-claims

This tranche does not authenticate Xenia, TPM, or institutional custody; does not establish global
consensus on a checkpoint; does not construct `AuthorityEvaluationInput`; does not mint live
authority; and does not execute an effect.
