# Verified Authority State v0.2 — r2

## Lineage

This tranche is stacked on `agency/authority-core-v0.2-r2` and preserves the historical
`agency/verified-authority-state-v0.1@9da533c672520b6292b28263d318a509b32639cc`
design without treating that historical branch as current qualification evidence.

`symthaea-authority-time` is imported byte-for-byte from that historical lineage because
its challenge-bound time theorem is grant-schema neutral. `symthaea-authority-state`
advances to protocol/schema v2 because the v0.2 grant introduces `AuthorityContextRef`.

## Core theorem

```text
caller-selected epoch/context/negative facts
    != verified current authority state

fresh grant-bound challenge
+ verified authority time
+ threshold witness agreement
+ organization/service diversity
+ exact source frontier
+ exact state sequence
+ current authority epoch
+ current authority-context state
+ complete relevant negative-authority facts
    -> VerifiedAuthorityStateV2
```

`VerifiedAuthorityStateV2` is non-Serde, non-Clone, and grants no effect.

## Current-context state

Witnesses report `Option<AuthorityContextRef>` from their authoritative source:

- `Some(context)` means that exact context is currently active;
- `None` means no active context currently exists for the challenged authority domain.

`None` is committed directly into the signed snapshot. It is not represented by an all-zero digest,
a magic namespace, or another synthetic placeholder.

A threshold-verified `None` is auditable evidence, but it cannot be turned into positive grant
eligibility. Production code receives only the verified state facts; it does not receive a helper
that manufactures evaluator input.

## Context rotation

The challenge does not tell witnesses which context to sign. A current context different from the
grant is a valid verified observation. A later admission/composition layer may compare that verified
context with the grant; an old grant bound to a different context must be denied.

## Accounting boundary

A bare `GrantUseState { committed, reserved }` is not grant-bound evidence. The two counters do not
prove which grant/account they came from, whether they came from the latest durable checkpoint, or
whether crash-conservative reservations and delegation escrow were included.

Therefore this production crate deliberately does **not** expose:

```text
VerifiedAuthorityStateV2 + caller-supplied GrantUseState -> AuthorityEvaluationInput
```

The small composition helper used by unit tests exists only under `#[cfg(test)]` so the pure evaluator
can be exercised without creating a production authority path.

The next accounting/admission tranche must instead establish an opaque exact-grant-bound accounting
proof before composition. The intended identity theorem is:

```text
grant.digest()
    == verified_state.grant_digest()
    == verified_accounting.grant_digest()
```

Only a separate admission/composition boundary may then construct evaluator input, and doing so still
does not itself execute an effect.

## Protocol non-equivalence

Protocol v2 assigns new canonical domains and negative-fact tags, including `RevokeContext`.
The snapshot also domain-separates present versus absent current context with an explicit tag.
V1 signatures, snapshot hashes, and negative-fact digests are not silently accepted as v2.

## Repository workflow governance

Current main requires runner-capable PR workflows to remain runnerless while draft and to include
`ready_for_review`. The focused qualification lane follows that contract: the Rust job is skipped
while draft and becomes eligible only after an explicit ready-for-review transition.

## Non-claims

This tranche does not verify delegation ancestry, verify durable use accounting, reserve uses,
authenticate a Xenia ledger, mint a live capability, authorize motor execution, alter embodiment
behavior, or turn cognitive/scientific confidence into authority.
