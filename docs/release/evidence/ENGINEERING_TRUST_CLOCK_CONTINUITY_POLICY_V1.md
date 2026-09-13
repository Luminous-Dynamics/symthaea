# Engineering Trust Clock Continuity Policy V1

## Purpose

Freeze the semantic identity of clock-continuity policy before any successor accepted-clock authority is implemented.

The existing fabrication-compatible `VerifiedClockContinuity` V1 digest proves the transition evidence fields, but its digest does **not** commit the `ClockContinuityPolicy` that was used to decide whether the transition was acceptable.

## Core theorem

```text
VerifiedClockContinuity V1
!= proof of which continuity policy authorized it

caller-selected ClockContinuityPolicy
!= trusted continuity policy
```

## Frozen policy identity

`ClockContinuityPolicyRevisionV1` content-addresses:

```text
maximum_epoch_step
maximum_forward_gap_ms
maximum_consensus_jump_ms
minimum_shared_sources
require_shared_algorithm
```

Frozen default-policy ID:

```text
904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06
```

A weaker policy with `maximum_forward_gap_ms = 20_000` has a distinct identity:

```text
6bf922e4e7806fc9928a001928f4bfbf5ad3cdc370a8395e21c6a8c8706c70a6
```

## Legacy evidence ambiguity

For the frozen epoch-42 -> epoch-43 fixture, both the strict policy and the weaker policy accept the transition and produce the same existing V1 continuity digest:

```text
ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15
```

That is intentional evidence of the gap: the legacy continuity object contains transition evidence but no policy identity.

The oracle also checks a 15,000 ms forward gap: the strict policy rejects it while the weaker 20,000 ms policy accepts it. Therefore policy choice is authority-relevant even when some transitions happen to produce identical evidence under both policies.

## Exact-byte evidence

The exact checked-in source bytes were executed locally with Python 3.13.5:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         667d505021814fbe4328444c986bd5ee20041086f326f40beb8110845d9a9b70
locally computed Git blob  45e9b3efaaa6e562abcecf7cbc7ebbd8aa5fef4f
GitHub stored Git blob      45e9b3efaaa6e562abcecf7cbc7ebbd8aa5fef4f
```

Repository CI remains a separate qualification layer.

## Production consequence

No successor `AcceptedClockBasis` should be mintable from a caller-supplied `ClockContinuityPolicy`.

Before successor authority exists, the trust kernel needs:

1. a typed/content-addressed `ClockContinuityPolicyRevisionV1`;
2. an explicit authority path that binds the selected continuity-policy ID;
3. successor continuity evidence bound to that exact policy ID in the higher authority object;
4. explicit migration authority for any policy change;
5. explicit trust-snapshot rotation authority for any snapshot change.

## Authority-model decision

The safest default is to authenticate the continuity-policy ID in the same initial clock authority policy that authenticates the quorum/evaluation policy. That gives every accepted bootstrap basis an exact successor-policy lineage from inception. Changing either policy later then requires a verified migration capability rather than a caller parameter.

This intentionally re-keys the still-draft bootstrap authority chain before it is treated as protocol-stable.

## Deliberate nonclaims

This reference does not authenticate a policy owner, bootstrap provider, continuity transition, trust-snapshot rotation, policy migration, real clock source, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2283, #2316, #2410, #2444, #2450, #2458.