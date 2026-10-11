# Engineering Trust Clock Evaluation Basis V2

## Status

Independent reference theorem only. This document records the frozen semantics and exact-byte execution lineage for `scripts/etk-clock-evaluation-basis-oracle.py`.

It does **not** grant runtime, engineering, fabrication, deployment, or actuation authority.

## Governing theorem

```text
signed clock sample
!= authority to choose the time used to validate its own signer

well-formed continuity digest
!= continuity between this exact prior window and this exact successor window
```

A candidate clock window is considered only after a `ClockEvaluationPermitV2` has been derived from an already trusted basis, an exact transition-policy revision, and an exact trust snapshot.

The candidate itself is not an input to permit construction.

## V2 composition

```text
ExternalBootstrap authority record
        +
trusted prior time interval
        +
exact TrustSnapshot digest
        +
ClockEvaluationPolicyV2
        ↓
prove snapshot + eligible ClockAuthority keys cover
THE WHOLE conservative evaluation envelope
        ↓
ClockEvaluationPermitV2
        ↓
reconstruct candidate legacy clock-window witness
        ↓
prove every candidate signer is permit-eligible
prove minimum signer count / algorithm diversity
prove every observation uncertainty interval is inside permit
        ↓
AcceptedClockBasisV2
        ↓
prior accepted basis + next pre-candidate permit
        ↓
reconstruct successor window
        +
recompute continuity over exact prior/successor windows
        ↓
AcceptedClockBasisV2 successor
```

## Exact legacy compatibility inputs

The reference deliberately reproduces the existing fabrication V1 identities rather than redefining them:

- trust snapshot: `609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633`
- verified clock window epoch 42: `5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229`
- verified clock window epoch 43: `d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b`
- continuity 42 -> 43: `ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15`

Legacy fabrication JSON identity remains serialization-order compatible. New V2 root-of-time authority objects use sorted-key compact JSON. Extraction compatibility and new-protocol canonicalization are intentionally separate concerns.

## Frozen V2 vectors

```text
bootstrap anchor
34a24d22b986744a2f823eead0b4a0566806763df7a19fdc9af97467dc5f5e68

transition policy
7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318

bootstrap evaluation permit
746a0d77056d1e35a4ec2f32c2cc56a97e5778544f9653cec170e65b062371a8

accepted basis epoch 42
c84a78db6f6396593b63f2207f604d94be6022788c9647ec981934d2095a7b9f

continuity evaluation permit
d76e0dc9214eec044e3fef0c939e7002870bef47e7732f3e2662f8f0e2127e62

accepted basis epoch 43
719d387aa5d1e7f74cd21e12c8acbd671491bd6c57b0b2945ea717126c5a0888
```

## Fail-closed cases frozen by the oracle

The reference denies:

- candidate signer not included in the precomputed eligible-key set;
- insufficient candidate signer count;
- candidate loss of required algorithm diversity;
- any accepted observation uncertainty interval extending outside the pre-authorized envelope;
- internally inconsistent clock-window witness metadata;
- stale trust snapshot across the evaluation envelope;
- trust-snapshot rotation without separate authority;
- tampered policy or bootstrap-anchor content under an unchanged ID;
- continuity whose prior window is not the exact accepted prior window;
- continuity whose successor is not the exact candidate successor;
- continuity epoch mismatch;
- continuity digest mismatch after independent recomputation.

## Exact-byte execution lineage

The final checked-in candidate bytes were executed locally before publication:

```text
Python                     3.13.5
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         4cf8281530ff93039c89969ae85d8c70d07550c64fe5252315a250f1aecdd2a3
locally computed Git blob  c11fbeb370c200b78c44a13ef2d26a5f923e2607
GitHub stored Git blob     c11fbeb370c200b78c44a13ef2d26a5f923e2607
```

The Git object identity therefore establishes that the checked-in source is the exact source that produced the local self-test result.

Repository CI remains a separate qualification layer and must not be inferred from this local reference execution.

## Architectural consequence for the generic trust kernel

The existing `VerifiedClockWindow` retains source IDs and algorithms but does not retain the exact signer key IDs / accepted-observation witness needed to prove compatibility with a pre-candidate evaluation permit.

Production integration should therefore add a companion evidence type rather than mutate the frozen legacy `VerifiedClockWindow` protocol in place, for example:

```text
VerifiedClockWindowV1
        +
ClockWindowEvaluationWitnessV1
    - exact window evidence digest
    - exact trust-snapshot digest
    - exact sorted accepted observation digests
    - exact sorted (algorithm, key_id) signers
    - exact observation intervals
        ↓
ClockEvaluationPermitV2 compatibility check
```

The witness must be produced as part of successful clock verification, not reconstructed from caller assertions later.

## Deliberate nonclaims

This reference does not establish:

- authenticity or organizational legitimacy of the bootstrap authority record;
- authenticity or authorization of the transition-policy record;
- private-key security or cryptographic-provider qualification;
- trust-snapshot rotation authority;
- TPM/secure-element/RTC-backed root of time;
- network time correctness;
- ETK evidence currentness by itself;
- requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Those remain separate authority layers.