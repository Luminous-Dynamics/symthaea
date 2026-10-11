# Engineering Trust Clock Policy V4

## Status

Independent reference theorem. No production authority is granted by this document or oracle.

## Purpose

Bind the complete normal-operation clock policy under one externally authenticated bootstrap lineage:

```text
ClockQuorumPolicyRevisionV1
+ ClockContinuityPolicyRevisionV1
        ↓
ClockEvaluationPolicyV4
        ↓
externally authenticated ClockBootstrapClaimV2
        ↓
VerifiedClockBootstrapAuthorityV2
        ↓
ClockEvaluationPermitV4
```

This prevents continuity rules from appearing later as a caller-selected runtime argument after bootstrap authority has already been granted.

## Frozen identities

```text
ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockContinuityPolicyRevisionV1
904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06

ClockEvaluationPolicyV4
4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234

ClockBootstrapClaimV2
c913a1829fe0bde63bbd2f0a4fda9344ed4de09fb9dece23ebf3189cb39e9fbd

ClockBootstrapAuthorityEvidenceV2
42d1b984349521b7d9dfc0d7ef149094a15231bd557c698acc7e276efb8a901d

VerifiedClockBootstrapAuthorityV2
aecff7b0ad2a40e2ab4a0dd7fc6dca4684046cceda5df0b36071e6bf623f8582

ClockBootstrapAnchorV2
c9fc46d4bbdccef92c2bc86c3682f2319d6837f753cf954fff30dcd451b5b588

ClockEvaluationPermitV4
1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f
```

## Authority theorem

A V4 evaluation policy commits both:

- the exact candidate-window quorum/uncertainty/consensus policy; and
- the exact successor-continuity policy.

Changing either policy changes `ClockEvaluationPolicyV4`, which changes the bootstrap claim, verified bootstrap authority lineage, bootstrap anchor, and permit.

A permit retains both validated policy records. Normal successor continuity must inherit the exact retained continuity policy. A policy change requires an explicit verified migration capability rather than an ordinary function argument.

## Negative controls

The oracle requires fail-closed behavior for:

- substituting a weaker continuity policy under an already authenticated evaluation policy;
- constructing a new evaluation policy around that weaker continuity policy and attempting to reuse the old bootstrap claim;
- substituting a weaker quorum policy;
- reauthorizing a changed continuity policy only through a fully re-keyed bootstrap lineage.

## Exact-byte evidence

Exact checked-in oracle subject:

```text
scripts/etk-clock-policy-v4-oracle.py
```

Executed evidence:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         e2a133ed232f74346eba92e70627aa3877c2fa07e60ddb314be549edc822be8d
GitHub stored Git blob      78b1149d519fd078e9f9fc0ab2b4906edc8723a1
locally executed Git blob   78b1149d519fd078e9f9fc0ab2b4906edc8723a1
```

Repository CI is a separate qualification layer.

## Protocol migration

This intentionally supersedes the normal-operation policy semantics of `ClockEvaluationPolicyV3` / `ClockEvaluationPermitV3` for any lineage that intends to authorize successor continuity.

The existing V3 work remains useful design history and an intermediate bootstrap theorem, but V4 is the cleaner final policy boundary because both quorum and continuity rules are authenticated before candidate clock evidence is accepted.

## Deliberate nonclaims

This reference does not establish a real bootstrap provider, key, trust snapshot, clock source, policy-migration authority, trust-snapshot-rotation authority, successor accepted basis, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.
