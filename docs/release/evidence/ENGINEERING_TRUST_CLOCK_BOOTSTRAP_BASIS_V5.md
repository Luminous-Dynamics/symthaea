# Engineering Trust Clock Bootstrap Basis V5

## Status

Independent reference theorem. No production authority is granted by this document or oracle.

## Purpose

Bind an accepted bootstrap clock basis directly to the complete unified clock-policy lineage:

```text
ClockEvaluationPermitV4
        ↓ retains quorum + continuity authority
original signed clock observations
        ↓ exact retained quorum policy
VerifiedClockWindow V1
+ ClockWindowEvaluationWitnessV1
        ↓ permit compatibility
AcceptedClockBasisV5
```

V5 is intentionally a new authority schema because the accepted basis now commits the exact evaluation, quorum, and continuity policy IDs and retains the originating permit for successor authority.

## Frozen identities

```text
ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockContinuityPolicyRevisionV1
904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06

ClockEvaluationPolicyV4
4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234

ClockEvaluationPermitV4
1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f

VerifiedClockWindow V1
5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229

ClockWindowEvaluationWitnessV1
e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2

AcceptedClockBasisV5
241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8
```

## V5 authority preimage

The basis identity commits:

- exact permit ID;
- exact `ClockEvaluationPolicyV4` ID;
- exact `ClockQuorumPolicyRevisionV1` ID;
- exact `ClockContinuityPolicyRevisionV1` ID;
- exact trust-snapshot digest;
- exact verified-window evidence digest;
- exact evaluation-witness digest;
- epoch and accepted lower/upper/consensus times.

The reference result also retains the complete originating permit object. A successor transition therefore does not need callers to reconstruct or resupply policy authority.

## Fail-closed corpus

The oracle rejects:

- one-source input under the authorized two-source quorum policy;
- mutation of retained continuity-policy semantics under the old policy identity;
- observation uncertainty above the authorized maximum;
- a signer absent from the permit's whole-envelope eligible-key set;
- an individual observation interval outside the permit even when the final window intersection remains inside.

## Exact-byte evidence

The exact checked-in oracle was executed locally before check-in:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         09b43f63a877e1198e5f4aba86db4010e2af93c2f3a09eb9900ba1cd65d59a18
locally computed Git blob  16608e1f2177e4280c6138b2087a40c4a0632eaa
GitHub stored Git blob      16608e1f2177e4280c6138b2087a40c4a0632eaa
```

Repository CI remains a separate qualification layer.

## Successor rule

The production capability should retain the private originating `ClockEvaluationPermitV4` in addition to the exact verified window/witness.

Normal successor authority must inherit:

- the same evaluation-policy ID;
- the same quorum-policy ID;
- the same continuity-policy ID;
- the same trust-snapshot digest.

A policy change is not an ordinary successor argument and requires a verified policy-migration capability. A trust-snapshot change requires a separate verified trust-rotation capability.

## Supersession

This V5 authority theorem supersedes #2450/#2458's V4 accepted-basis semantics, which were built on the provisional V3 permit that did not yet bind continuity policy.

## Deliberate nonclaims

This reference establishes no real clock source/verifier, policy owner, policy-migration authority, trust-rotation authority, successor accepted basis, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.
