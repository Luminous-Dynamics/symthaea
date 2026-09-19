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

A threshold-verified `None` is auditable evidence, but `evaluation_input()` fails with
`NoCurrentAuthorityContext`. It cannot be converted into positive grant eligibility.

## Context rotation

The challenge does not tell witnesses which context to sign. A current context different from the
grant is a valid verified observation. Downstream pure evaluation then returns `ContextMismatch`,
so context rotation invalidates old grants without verifier-side rewriting.

## Evaluation bridge

`VerifiedAuthorityStateV2::evaluation_input()` combines:

- verified authority time for the exact grant and time policy;
- verified current epoch;
- a verified active authority context;
- separately supplied `GrantUseState`.

The result is still only `AuthorityEvaluationInput`. It is not execution admission. If the verified
state contains no active context, the bridge refuses to construct evaluator input.

## Protocol non-equivalence

Protocol v2 assigns new canonical domains and negative-fact tags, including `RevokeContext`.
The snapshot also domain-separates present versus absent current context with an explicit tag.
V1 signatures, snapshot hashes, and negative-fact digests are not silently accepted as v2.

## Repository workflow governance

Current main requires runner-capable PR workflows to remain runnerless while draft and to include
`ready_for_review`. The focused qualification lane follows that contract: the Rust job is skipped
while draft and becomes eligible only after an explicit ready-for-review transition.

## Non-claims

This tranche does not verify delegation ancestry, reserve uses, authenticate a Xenia ledger, mint a
live capability, authorize motor execution, alter embodiment behavior, or turn cognitive/scientific
confidence into authority.
