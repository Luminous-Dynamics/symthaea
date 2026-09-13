# Engineering Trust Clock Bootstrap Basis V4

## Purpose

Freeze the corrected accepted-clock theorem after `ClockEvaluationPolicyV3` / `ClockEvaluationPermitV3` bind the exact candidate-window quorum policy.

This is an independent stdlib Python reference. It grants no production authority.

## Core theorem

```text
private ClockEvaluationPermitV3
        ↓ retains exact ClockQuorumPolicyRevisionV1
original signed observations
        ↓ exact authority-bound quorum semantics
VerifiedClockWindow V1
+ exact ClockWindowEvaluationWitnessV1
        ↓ permit compatibility
AcceptedClockBasisV4
```

The important correction is that two thresholds are now distinct:

```text
minimum whole-envelope eligible ClockAuthority keys
!= minimum candidate clock sources/signers
```

The first qualifies the permit's available key pool. The second is part of the exact authority-bound quorum policy used by candidate-window verification.

## Frozen V4 chain

```text
ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockEvaluationPolicyV3
7a782b8adbf4e66a352c4a26ae614a458971d78c33c136cb8eeee6c6c28433b4

ClockEvaluationPermitV3
716c65376a721574bb53d2415e6b58f83d9c7483666b3799ca8fc4fe214f2cce

VerifiedClockWindow V1
5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229

ClockWindowEvaluationWitnessV1
e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2

AcceptedClockBasisV4
77728f18e60bb095bcc744657bd1c2faa15f86e85853d363ebb4145d8a4ee3de
```

## Acceptance rules

The reference:

1. validates the permit's exact V3 identity;
2. validates the retained quorum-policy record against the permit's quorum-policy ID;
3. reconstructs the candidate window from observations under that exact quorum policy;
4. reconstructs the exact clock-window witness;
5. requires the window trust snapshot and whole interval to match the permit;
6. requires every accepted signer to belong to the permit's whole-envelope eligible-key set;
7. requires every observation's full uncertainty interval to lie inside the permit;
8. mints V4 identity only after all checks succeed.

## Fail-closed corpus

The oracle rejects:

- one-source input under an authorized two-source quorum policy;
- mutation of the quorum-policy record under an unchanged quorum-policy ID;
- uncertainty above the authorized maximum;
- an unpermitted signer;
- an observation whose individual uncertainty interval extends outside the permit even when the final intersected window remains inside.

## Why V4

The earlier V3 accepted-basis reference was built on provisional permit V2 semantics and reused the permit's eligible-key-pool minimum as a candidate signer threshold. Policy V3 separates those concepts and binds the candidate quorum theorem explicitly, so the accepted basis receives a new schema rather than silently changing V3 semantics.

## Exact-byte evidence

The exact checked-in oracle bytes were executed locally with Python 3.13.5:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         e46344b9fb31d5443dabd92b902334d8fc27214aec7fb0da361da08c2a2203b7
locally computed Git blob  75ed33f4ffacfd2c9f18dbb9fd506bcea8b2e8c1
GitHub stored Git blob      75ed33f4ffacfd2c9f18dbb9fd506bcea8b2e8c1
```

Repository CI remains a separate qualification layer.

## Production consequence

Production accepted-clock admission should take no caller-supplied `ClockQuorumPolicy`. The private `ClockEvaluationPermitV3` must retain the validated `ClockQuorumPolicyRevisionV1` and reconstruct the runtime verifier policy internally.

The API should accept original signed observations, exact trust snapshot, and cryptographic verifier; internally call the single-pass witness-producing verifier; then apply permit compatibility and mint a private/non-deserializable `AcceptedClockBasisV4`.

## Deliberate nonclaims

This reference does not establish real clock-source trustworthiness, real verifier correctness, trust-snapshot authenticity, bootstrap-provider trustworthiness, continuity, policy migration, trust-snapshot rotation, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2316, #2367, #2386, #2410, #2444. The earlier #2405/#2407/#2408 lineage is superseded by the quorum-authority correction.