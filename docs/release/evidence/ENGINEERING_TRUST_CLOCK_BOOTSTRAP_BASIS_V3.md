# Engineering Trust — bootstrap accepted clock basis V3

## Scope

This note records an independent reference theorem for the first accepted clock basis after a policy-bound bootstrap permit exists.

It changes no production authority code.

The governing distinction is:

```text
ClockEvaluationPermitV2
+ verified V1 clock window
+ exact ClockWindowEvaluationWitnessV1
        ↓
AcceptedClockBasisV3

window evidence alone
!= permit-compatible signer proof

witness content identity alone
!= proof a trusted verifier executed
```

V3 intentionally binds the exact verifier-emitted witness digest into accepted-basis identity. This is a protocol strengthening relative to the older #2315 `AcceptedClockBasisV2` preimage and therefore uses a new schema rather than silently re-keying V2.

## Frozen lineage

```text
ClockEvaluationPermitV2
12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9

VerifiedClockWindow V1 evidence
5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229

ClockWindowEvaluationWitnessV1
e7c4af861e34a7fb8a841e35d49681e24798507c802ec9b8bcdf3c066e9819a2

AcceptedClockBasisV3
1442964fea0e569d5ef59e2df593b5399eb2a7cc568417e1a4c34c796a20d72e
```

The accepted basis commits:

- schema `symthaea.trust.accepted-clock-basis.v3`;
- acceptance kind `Bootstrap`;
- exact private permit identity;
- exact trust-snapshot digest;
- exact frozen V1 window evidence digest;
- exact window-evaluation witness digest;
- exact epoch;
- exact lower/upper interval;
- exact consensus time.

## Permit-compatibility theorem

Acceptance requires the witness/window pair to reconstruct consistently and then proves:

- window trust snapshot equals permit trust snapshot;
- whole window lies inside the permit envelope;
- every accepted signer is in the permit's exact eligible-key set;
- accepted signer count satisfies the permit minimum;
- algorithm diversity satisfies the permit when required;
- every accepted observation's full uncertainty interval lies inside the permit envelope.

The final condition is deliberately stronger than checking the intersected window alone: a broad out-of-envelope observation cannot be hidden by another source's narrower interval.

## Fail-closed corpus

The independent oracle rejects:

- witness signer tampering that no longer reconstructs the window witness;
- a self-consistent window/witness using an unpermitted signer;
- a self-consistent single-signer window below the permit minimum;
- a window whose final intersection is acceptable but one source's full uncertainty interval exceeds the permit;
- a permit whose content identity is tampered.

## Exact-byte reference evidence

The final source bytes were executed locally with Python 3.13.5:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         e6259214cfcf279bc2bb86d9f93f86740ea3759f5fccb21c58b5f2fa907eff68
locally computed Git blob  27a86241e87cfe948054ba93e19fde17d58d8025
GitHub stored Git blob      27a86241e87cfe948054ba93e19fde17d58d8025
```

The checked-in oracle bytes are therefore the exact locally executed bytes. Repository CI remains a separate qualification layer.

## Production consequence

A production bootstrap-basis admission API should consume the private `ClockEvaluationPermitV2` capability, the original signed clock observations, the exact trust snapshot, the existing clock-quorum policy/verifier, and the #2316 witness-producing verifier path.

It should mint a private, non-serializable `AcceptedClockBasisV3` only after:

1. cryptographic clock verification succeeds;
2. the emitted witness reconstructs the exact window;
3. the permit compatibility theorem above succeeds.

A caller-supplied serializable window or witness must not directly mint the basis capability.

## Deliberate nonclaims

This reference does not establish the trustworthiness of any real clock source, private key, verifier implementation, trust snapshot, bootstrap provider, clock policy, persistence system, continuity transition, policy migration, ETK currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2283, #2286, #2315, #2316, #2367, #2386, #2398, #2405.
