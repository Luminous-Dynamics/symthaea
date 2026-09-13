# Engineering Trust Clock Successor Production V6

## Status

Draft production authority boundary stacked on `AcceptedClockBasisV5` / `ClockEvaluationPermitV4`.

This document records source-level intent only. Exact-head Rust qualification remains dependent on repository CI.

## Governing theorem

```text
prior opaque AcceptedClockBasisV5
        +
matching evaluation-policy witness
        +
matching trust-snapshot witness
        ↓
ClockSuccessorAuthorityContextV1
        ↓ no candidate clock input
ClockSuccessorEvaluationPermitV1
        ↓
original signed successor observations
        ↓ retained quorum policy
VerifiedClockWindow V1 + exact witness
        ↓ retained continuity policy
recomputed VerifiedClockContinuity V1
        ↓
AcceptedClockBasisV6
```

Caller-supplied policy/snapshot records are witness material only. They must reproduce IDs/digests already committed by the prior opaque basis before a successor context can exist.

## Frozen independent vectors

Reference: #2523.

```text
prior AcceptedClockBasisV5
241a627e04efb18d15bfc738a26846d0dfbbcef18ac57370c1a4f3340cab12b8

ClockSuccessorEvaluationPermitV1
eec874620319419ac4ef0663997ee2a6700db268fad6367ee28b684816c5d5d8

successor VerifiedClockWindow V1
d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b

successor ClockWindowEvaluationWitnessV1
6d330cbf15d9dde7bf11aada0fb5b53feed204573308c6bcafd5569f07b4e3a3

VerifiedClockContinuity V1
ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15

AcceptedClockBasisV6
6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e
```

## Fail-closed boundary

Production tests cover:

- policy witness substitution denied;
- trust-snapshot witness substitution denied;
- prior V5 window/witness revalidated before binding;
- candidate observations absent from permit derivation;
- one-source successor denied by retained quorum policy;
- successor window outside pre-candidate envelope denied;
- epoch skip denied by retained continuity policy;
- snapshot valid for bootstrap but expired before successor envelope denied;
- signature rejection cannot mint V6;
- normal derive/accept APIs expose no caller policy, snapshot, lifecycle time, precomputed window, or precomputed witness authority.

## Deliberate nonclaims

This tranche establishes one normal successor transition only. It does not yet provide a recursively reusable accepted-basis abstraction for arbitrary future epochs, policy migration authority, trust-snapshot rotation authority, ETK currentness, requirement satisfaction, design qualification, deployment approval, or actuation authority.
