# Engineering Trust Clock Operational Recursive V1

## Purpose

Freeze an independent recursive normal-operation clock theorem before replacing the one-shot V5→V6 successor API.

The key correction is structural: the normal-operation output type must be usable as the next transition's input type. A design that proves only V5→V6 but requires a new V7/V8 type for every later epoch is not a closed operational authority model.

## Authority theorem

```text
AcceptedClockBasisV5
        ↓ bind exact bootstrap authority/witness context
OperationalClockBasisV1(epoch 42)
        ↓
ClockSuccessorEvaluationPermitV2
        ↓ original signed observations
OperationalClockBasisV1(epoch 43)
        ↓
ClockSuccessorEvaluationPermitV2
        ↓ original signed observations
OperationalClockBasisV1(epoch 44)
        ↓ ... recursively
```

Normal succession does not accept a caller-selected policy or trust-snapshot identity. Policy/snapshot records are witness material only and must reproduce identities already retained by the opaque operational basis.

## Frozen vectors

```text
TrustSnapshot
609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633

ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockContinuityPolicyRevisionV1
904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06

ClockEvaluationPolicyV4
4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234

AcceptedClockBasisV5
241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8

OperationalClockBasisV1 epoch 42
f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b

ClockSuccessorEvaluationPermitV2 42→43
750c044e48a5395ee04a457ba7b89cb2397c2bf2cea76e96e71e50dcd338b73b

OperationalClockBasisV1 epoch 43
0e36e238e300ae1d626846220bbba84f5e87a50be7ffa2ac3b8e7822f4d47810

ClockSuccessorEvaluationPermitV2 43→44
68b6dd92d42f3b42093f33d310d69eb8ac1bb259b02bd698256f67ea872b8a3d

OperationalClockBasisV1 epoch 44
9373fac49b1b488859de56cd6cd588b8e5868cf9fb455c8b3546b46974d8ffce
```

The legacy epoch-43 and epoch-44 window/witness/continuity identities are independently recomputed inside the oracle rather than supplied as trusted constants.

## Fail-closed corpus

The exact oracle rejects:

- continuity-policy substitution;
- trust-snapshot substitution;
- a successor permit bound to the wrong prior operational basis;
- one-source quorum input;
- a successor window outside the pre-candidate envelope;
- a trust snapshot that was sufficient for an earlier epoch but expires before the next envelope.

It also proves the epoch-43 operational basis can itself mint the epoch-44 permit without a new basis type.

## Exact-byte evidence

```text
Python                     3.13.5
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         17dcc20af178ee5e8a08b405068140a933f4cf32cf39fac285526eb206092ccb
locally computed Git blob  c366d520d29fd58fc31738792dace24803a98320
GitHub stored Git blob     c366d520d29fd58fc31738792dace24803a98320
```

Repository CI is separate qualification evidence.

## Production consequence

The one-shot `AcceptedClockBasisV6` path is useful evidence but should not be the final normal-operation API. Production should move to one recursive opaque operational-basis capability whose private witness context contains enough material to:

- validate the exact evaluation/quorum/continuity policy lineage;
- validate the exact trust snapshot;
- retain the exact current verified window and witness;
- derive the next permit before seeing candidate observations;
- verify continuity from current to successor window;
- return the same operational-basis capability type.

Policy migration and trust-snapshot rotation remain separate authority transitions and must not be ordinary successor arguments.

## Deliberate nonclaims

This reference establishes no Rust compilation, provider qualification, policy owner, policy migration, trust rotation, ETK currentness, engineering requirement satisfaction, deployment approval, or physical actuation authority.
