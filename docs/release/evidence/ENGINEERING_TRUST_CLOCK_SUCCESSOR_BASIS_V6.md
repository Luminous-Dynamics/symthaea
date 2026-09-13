# Engineering Trust Clock Successor Basis V6

## Status

Independent reference theorem only. This document records the successor-clock authority semantics frozen by `scripts/etk-clock-successor-basis-v6-oracle.py`.

It grants no runtime, engineering, deployment, actuation, policy-migration, or trust-rotation authority.

## Governing theorem

```text
AcceptedClockBasisV5
        ↓ retains originating authority/witness context
no caller policy choice
no caller trust-snapshot choice
        ↓
ClockSuccessorEvaluationPermitV1
        ↓
original signed epoch-(N+1) observations
        ↓ exact retained quorum policy
verified window + exact evaluation witness
        ↓ exact retained continuity policy
recomputed continuity from exact prior window
        ↓
AcceptedClockBasisV6
```

The successor candidate cannot choose the policy, snapshot, lifecycle instant, or previous clock endpoint used to authorize itself.

## Frozen lineage

```text
TrustSnapshot
609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633

ClockQuorumPolicyRevisionV1
4cbcce7db5128b87d060ad5cfe97b111c95799ec07742e2b3d9cb63341cdd304

ClockContinuityPolicyRevisionV1
904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06

ClockEvaluationPolicyV4
4644ae46ecbebb83094d40036f0bad278dc63f32cec8ffa01e9bc6a10d8ff234

ClockEvaluationPermitV4
1b837997dd8a4c34d8e4ad24c9b85e3718f834dd79e2f5f25c650f7d2279245f

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

successor AcceptedClockBasisV6
6b386d75ac26802b5879e4ff820da5ceb6285ab9cc3d6d96729ffb665c422a6e
```

The successor permit envelope for the fixture is:

```text
[1_499_920 ms, 1_502_100 ms]
```

It is derived from the prior accepted window plus the exact retained `max_transition_ms = 2_000`; no candidate observation participates in permit construction.

## Authority retention requirement

The independent model deliberately retains, behind the prior accepted capability:

- exact `ClockEvaluationPolicyV4` record;
- exact `ClockQuorumPolicyRevisionV1` record;
- exact `ClockContinuityPolicyRevisionV1` record;
- exact `TrustSnapshot` record;
- exact prior verified window.

This is witness retention, not a protocol identity change. The already-frozen V4 permit and V5 bootstrap-basis IDs do not move merely because their private capability representation retains enough immutable witness material to derive the next envelope.

Production must not invent `max_transition_ms` from an ID, infer future key validity from signer names, or treat a previously accepted snapshot as valid forever.

## Fail-closed corpus

The exact oracle rejects:

- retained continuity-policy mutation under the old policy ID;
- retained trust-snapshot mutation under the old snapshot ID;
- one-source successor input under the retained two-source quorum policy;
- a successor window outside the pre-candidate permit envelope;
- an epoch jump that exceeds the retained continuity policy;
- a trust snapshot that was valid for bootstrap but expires before the full successor evaluation envelope.

It also independently recomputes the exact legacy continuity digest between the frozen epoch-42 and epoch-43 windows.

## Exact-byte execution lineage

The exact candidate bytes were executed before publication:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         697c3da1417b7493a4c74dc12f3e8c6f71b7feaed1add90f42004bc8d1a9e7be
locally computed Git blob  def6561e04de10417b3506a94015e89bcc7e63dc
GitHub stored Git blob     def6561e04de10417b3506a94015e89bcc7e63dc
```

Repository CI remains a separate qualification layer.

## Production consequence

Before a normal successor Rust API is introduced, the private production authority chain must retain enough exact witness material to derive the successor envelope without caller policy or snapshot authority.

The intended production split is:

```text
prior AcceptedClockBasisV5
        ↓
derive ClockSuccessorEvaluationPermitV1
        ↓
accept successor signed observations
        ↓
verify exact continuity using permit-retained policy
        ↓
private AcceptedClockBasisV6
```

A policy change must require separate verified policy-migration authority. A trust-snapshot change must require separate verified trust-rotation authority.

## Deliberate nonclaims

This reference does not establish a real clock source, cryptographic-provider qualification, policy owner, policy migration, trust-snapshot rotation, TPM/RTC trust root, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.
