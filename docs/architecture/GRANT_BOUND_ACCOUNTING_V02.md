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
    -> exact effect digest
    -> exact risk charge
```

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

A different child with identical numeric ceilings cannot close or consume that escrow.

## Monotonic uncertainty rule

Authority may be charged conservatively without proof; authority may not be returned after an
uncertain external outcome without proof.

This tranche therefore allows:

```text
Reserved -> OutcomeUnknown
OutcomeUnknown -> Committed
OpenEscrow -> OutcomeUnknown
OutcomeUnknownEscrow -> ClosedFullyCharged
```

It deliberately does **not** expose historical-style:

```text
OutcomeUnknown -> Released
partial child escrow refund
```

Those transitions increase remaining authority. They require a future exact-grant-bound verified
reconciliation receipt.

Cancellation before dispatch remains available because the reservation has not entered the
uncertain-dispatch state.

## Anti-rollback checkpoint

`GrantAccountCheckpoint` schema v2 binds:

- exact grant digest;
- exact runtime snapshot;
- checkpoint sequence;
- exact predecessor digest;
- execution effect identities;
- child escrow identities;
- all accounting records and counters.

The checkpoint head now also carries `grant_digest`, preventing cross-grant head substitution.

## VerifiedGrantAccounting

Chain reconstruction alone does not establish currentness. `verify_chain()` is an audit/recovery
operation only.

An opaque `VerifiedGrantAccounting` is produced only when one checkpoint exactly matches an
externally retained/authenticated `CheckpointHead`:

```text
checkpoint.head() == trusted_external_head
```

The external head remains an explicit trust boundary. This crate does not claim to authenticate
Xenia, TPM, append-only-log, or supervisor custody of that head.

The proof binds:

```text
VerifiedGrantAccounting {
    exact grant digest,
    exact trusted checkpoint head,
    crash-conservative use state,
    charged risk,
}
```

It is non-Serde and non-Clone.

## Composition target

This tranche still does not construct `AuthorityEvaluationInput` or execution admission. The next
composition boundary must prove:

```text
grant.digest()
    == VerifiedAuthorityStateV2.grant_digest()
    == VerifiedGrantAccounting.grant_digest()
```

and separately require fresh verified time/state before pure evaluation.

## Non-claims

This tranche does not prove an effect occurred, authenticate the externally trusted checkpoint head,
verify delegation ancestry beyond one static attenuation edge, return unused authority after uncertain
child execution, mint a live capability, dispatch hardware/software effects, or establish scientific
authority.
