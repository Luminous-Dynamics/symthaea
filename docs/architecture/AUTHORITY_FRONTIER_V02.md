# Authority Frontier v0.2 — r2

## Purpose

Port the historical compare-and-swap authority-frontier theorem onto the v0.2 grant/accounting lineage without overclaiming bootstrap-head authenticity or execution authority.

This tranche sits above #4289's grant-bound accounting and below any future Xenia/TPM/append-only-log authenticator.

## Narrow theorem

```text
exact CapabilityGrant
+ supplied bootstrap anchor bound to that grant
+ durable store currently equals that anchor
+ every successor payload valid under the exact grant
+ every successor installed by full-head compare-and-swap
+ fresh store read still equals adapter expected head
+ accounting bound to the same exact head/grant
    -> FrontierBoundGrantAccounting
```

`FrontierBoundGrantAccounting` is non-Serde and non-Clone.

## What this does not prove

The supplied bootstrap anchor is not authenticated by this crate. A caller can possess or replay a `CheckpointHead`. Therefore:

```text
FrontierBoundGrantAccounting
    != externally authenticated accounting
    != globally current accounting
    != execution admission
```

The resulting object is also a point-in-time observation. Another authorized writer may advance the durable frontier immediately afterward. Any future admission theorem must therefore consume it together with a fresh/atomic frontier transition rather than treating it as an indefinitely current capability.

## Exact grant ownership

The frontier owns the exact `CapabilityGrant` whose accounting it governs. Reopen requires:

```text
grant.validate() == OK
bootstrap_head.grant_digest == grant.digest()
store.current_head() == bootstrap_head
```

A head from another grant cannot reopen the frontier even if its sequence/digest shape is otherwise valid.

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

After a successful CAS acknowledgement, `persist_successor()` performs another linearizable `current_head()` read before returning. If another writer already moved the store again, the adapter latches containment rather than returning a stale expected frontier.

Likewise, `bind_current_accounting()` performs a fresh store read before producing a frontier-bound object.

This closes races such as:

```text
adapter A reopens at H
adapter B advances store to H+1
adapter A tries to bind accounting for H
    -> StoreFrontierChanged
    -> adapter A contained
```

## Containment

The frontier latches containment after:

- checkpoint payload validation failure during progression;
- CAS/store failure;
- a store head change observed outside the adapter;
- invalid successor sequence/digest/grant;
- store acknowledgement of an unexpected successor head.

Contained frontiers cannot continue authority progression or bind accounting.

## Generation zero

`establish_grant_frontier()` requires a valid grant and an empty store, creates an exact grant-bound generation-zero checkpoint, installs it with `compare_and_swap(None, checkpoint)`, and re-reads the store before returning.

The returned head is current relative to that store at the final observation made by the function, but it still requires external authentication before it can be used as a trusted restart anchor.

## Restart semantics

`reopen_from_anchor_claim()` succeeds only when the supplied head names the exact grant and the durable store currently reports exactly that full head. The function name deliberately says *claim*: equality with local storage does not prove the anchor itself is externally trustworthy.

## Store contract

`CheckpointCasStore` requires:

- `current_head()` to be a linearizable read of the same durable frontier used by CAS;
- `compare_and_swap()` to compare the full current head and durably install the complete checkpoint atomically;
- successful CAS acknowledgement to equal the exact installed checkpoint head.

A backend that implements only an ordinary read-then-write sequence does not satisfy this contract.

## Next composition step

A future external-authentication/admission tranche should compose:

```text
fresh Xenia/other authenticated checkpoint-head evidence
+ exact CheckpointHead identity
+ FrontierBoundGrantAccounting
+ VerifiedAuthorityStateV2
+ VerifiedAuthorityTime
    -> admission-eligible current authority facts
```

Even that composition must remain separate from durable one-use reservation and effect execution.

## Non-claims

This tranche does not authenticate Xenia, TPM, or institutional custody; does not establish global consensus on a checkpoint; does not construct `AuthorityEvaluationInput`; does not mint live authority; and does not execute an effect.
