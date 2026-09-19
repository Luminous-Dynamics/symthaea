# Grant-Bound Accounting v0.2 — r2

## Lineage

This tranche is stacked on `agency/verified-authority-state-v0.2-r2` and ports the strongest
ideas from the historical `symthaea-action-runtime` / `symthaea-action-checkpoint` lineage
without transferring historical qualification.

It exists because bare `GrantUseState { committed, reserved }` counters are not authority
evidence. Current-state verification and durable accounting must remain separately proven and
then be joined only by exact grant identity.

## Central invariant

```text
committed uses/risk
+ in-flight execution reservations
+ open/uncertain delegation escrow
<= exact grant ceiling
```

Every account is bound to `grant.digest()` plus the grant's exact use/risk ceilings.
Rehydration against another grant fails closed.

## Exact execution identity

Each execution reservation commits an exact nonzero `effect_digest` before dispatch. The accounting
layer does not interpret that digest, but capacity reserved for one immutable effect/admission intent
cannot silently become an unbound generic reservation in persisted state.

```text
reservation
    -> exact grant digest (through account)
    -> exact execution id
    -> exact effect digest
    -> exact risk charge
```

Execution IDs are unique within an account. A second reservation may not reuse the same execution
identity under another reservation ID.

## Exact delegation escrow identity

Historical delegation escrow stored only numeric use/risk ceilings. v0.2 binds escrow to the exact
attenuated `child_grant_digest` and derives the allocation from the child record itself.

```text
parent GrantAccount
+ child.validate_attenuation(parent)
    -> reserve exactly child.max_uses
    -> reserve exactly child.risk_budget
    -> bind exact child.digest()
```

A different child with identical numeric ceilings cannot close or consume that escrow, and the same
exact child grant cannot be escrowed twice in one parent account.

## Monotonic authority accounting

This first accounting schema contains no capacity-return transition.

Allowed reservation progression:

```text
Reserved -> OutcomeUnknown -> Committed
Reserved --------------------> Committed
```

Allowed escrow progression:

```text
Open -> OutcomeUnknown -> ClosedFullyCharged
Open --------------------> ClosedFullyCharged
```

There is deliberately no:

```text
Reserved -> Released
OutcomeUnknown -> Released
partial child escrow refund
```

Even an abandoned pre-dispatch reservation remains charged in this tranche. Returning capacity is an
authority-increasing transition and belongs in a future exact-grant-bound verified reconciliation
protocol. This is intentionally conservative.

A future execution-admission layer must make the durable transition to `OutcomeUnknown` occur before
or atomically with dispatch; this accounting crate itself has no effect-dispatch API.

## Transactional mutation

Commit transitions validate the existing account and compute all checked successor counters before
writing the successor state. Failed arithmetic/invariant checks do not partially advance accounting.

## Anti-rollback checkpoint

`GrantAccountCheckpoint` schema v2 binds:

- exact grant digest;
- exact runtime snapshot;
- checkpoint sequence;
- exact predecessor digest;
- execution effect identities;
- child escrow identities;
- all accounting records and counters.

The checkpoint head also carries `grant_digest`, preventing cross-grant head substitution.

## Head-bound accounting is not verified-current accounting

Chain reconstruction alone does not establish currentness. `verify_chain()` is an audit/recovery
operation only.

This crate can deterministically bind one checkpoint to one supplied `CheckpointHead`:

```text
checkpoint.head() == supplied_head
    -> HeadBoundGrantAccounting
```

`HeadBoundGrantAccounting` is non-Serde and non-Clone and binds the exact grant digest, exact head,
crash-conservative use state, and charged risk.

However, `CheckpointHead` is only serializable identity data here. A caller can construct one. This
crate does not authenticate Xenia, TPM, append-only-log, supervisor, or institutional custody of the
head. Therefore the output is deliberately **not** named `VerifiedGrantAccounting` and must not be
treated as verified currentness.

A later external-head verifier must authenticate the head's source/currentness before promoting the
head-bound accounting state into verified-current accounting evidence.

## Composition target

This tranche still does not construct `AuthorityEvaluationInput` or execution admission. A later
composition boundary must ultimately prove:

```text
grant.digest()
    == VerifiedAuthorityStateV2.grant_digest()
    == externally-verified accounting grant digest
```

and separately require fresh verified time/state plus authenticated current accounting before pure
evaluation.

## Non-claims

This tranche does not prove an effect occurred, authenticate a checkpoint head, prove currentness of
a reconstructed chain, verify delegation ancestry beyond one static attenuation edge, return unused
authority after reservation/child execution, mint a live capability, dispatch hardware/software
effects, or establish scientific authority.
