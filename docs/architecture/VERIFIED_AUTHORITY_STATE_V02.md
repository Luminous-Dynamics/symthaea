# Verified Authority State v0.2

## Lineage

This tranche is stacked on `agency/authority-core-v0.2` and preserves the historical
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
+ current AuthorityContextRef
+ complete relevant negative-authority facts
    -> VerifiedAuthorityStateV2
```

`VerifiedAuthorityStateV2` is not serializable, cloneable authority and grants no effect.

## Context rotation

Witnesses report the current authority context from their authoritative source. The
challenge does not tell them which context to sign. A current context different from the
grant is a valid verified observation. Downstream pure evaluation then returns
`ContextMismatch`, so context rotation invalidates old grants without verifier-side
rewriting.

## Evaluation bridge

`VerifiedAuthorityStateV2::evaluation_input()` combines:

- verified authority time for the exact grant and time policy;
- verified current epoch;
- verified current authority context;
- separately supplied `GrantUseState`.

The result is still only `AuthorityEvaluationInput`. It is not execution admission.

## Negative-state protocol

Protocol v2 assigns new canonical tags and domain separators, including
`RevokeContext`. V1 signatures, snapshot hashes, and negative-fact digests are not
silently accepted as v2.

## Non-claims

This tranche does not verify delegation ancestry, reserve uses, authenticate a Xenia
ledger, mint a live capability, authorize motor execution, alter embodiment behavior, or
turn cognitive/scientific confidence into authority.
