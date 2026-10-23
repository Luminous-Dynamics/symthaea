# Engineering Trust Kernel — Clock Bootstrap External Authority V2

Status: independent reference theorem only. This document and oracle change no production authority.

## Governing law

```text
bootstrap interval claim
!= external authority evidence
!= verified bootstrap authority capability

verified bootstrap interval
!= authority to choose an arbitrary transition policy

candidate clock time
must not authenticate the authority that bootstraps that clock
```

V2 strengthens the bootstrap claim so the same external authority that authenticates the initial trusted interval also authenticates the exact initial `ClockEvaluationPolicyV2` identity. This prevents a valid bootstrap capability from later being paired with a caller-chosen transition horizon/quorum policy.

## Reference objects

`ClockBootstrapClaimV2` binds:

- schema `symthaea.trust.clock-bootstrap-claim.v2`;
- purpose `ClockBootstrap`;
- exact trust-snapshot digest;
- exact initial `ClockEvaluationPolicyV2` ID;
- exact trusted lower/upper Unix-millisecond interval.

`ClockBootstrapAuthorityEvidenceV2` binds:

- exact bootstrap-claim ID;
- exact external provider ID;
- exact authority-policy digest;
- exact external-evidence digest.

`VerifiedClockBootstrapAuthorityV2` binds the exact claim, authority-evidence record, provider, and authority-policy revision after the external provider accepts that evidence.

The reference intentionally models the provider verdict as an external input. It does not claim to implement TPM/HSM/operator cryptography.

## Frozen vectors

```text
ClockEvaluationPolicyV2 (from #2315)
7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318

bootstrap claim V2
15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877

external authority evidence V2
62991edeba3a575b28801bcceaca99dceaec476a0bab699792df6fbe2ec0cee7

verified bootstrap authority V2
b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3
```

## Fail-closed corpus

The oracle rejects:

- bootstrap-claim content edited under an unchanged claim ID;
- an otherwise valid bootstrap claim with a different evaluation-policy ID when paired with old evidence;
- evidence bound to a different bootstrap claim;
- external-evidence content edited under an unchanged evidence ID;
- provider mismatch;
- authority-policy revision mismatch;
- external authority rejection.

Critically, the verifier function has no candidate clock-window or candidate-time argument.

## Exact-byte evidence

The final V2 source bytes were executed locally:

```text
Python                     3.13.5
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         0dca93f91b46c4edd5416b8c76a995de7c69559c5af5b6de31f983237ea245b4
locally computed Git blob  a04f8669aa681f791dd6af9728e85fe2d96644dd
GitHub stored Git blob      a04f8669aa681f791dd6af9728e85fe2d96644dd
```

Repository CI remains a separate qualification layer and is not inferred from local execution.

## Why V1 is superseded before production freeze

V1 authenticated the trust snapshot and initial trusted interval but did not commit the transition-policy identity. That left a privilege-amplification path:

```text
valid bootstrap capability
+ caller-chosen ClockEvaluationPolicyV2
-> widened/different evaluation authority
```

V2 closes the path by putting the initial evaluation-policy ID inside the externally authenticated bootstrap claim itself.

For successor clock bases, policy continuity should be the default. A change of `ClockEvaluationPolicyV2` must require a separate policy-migration capability rather than accepting a new caller-supplied policy record.

## Production shape

A production adapter should expose the external verification boundary without offering a built-in permissive implementation. Successful verification should mint an opaque/private capability, conceptually:

```text
verify_clock_bootstrap_authority_v2(
    claim_v2,
    authority_evidence_v2,
    configured_provider,
    configured_authority_policy,
    external_verifier,
) -> VerifiedClockBootstrapAuthorityV2
```

`VerifiedClockBootstrapAuthorityV2` should have private fields and no deserialization path. A serializable authority-evidence record is audit evidence only; its content hash is not proof that a trusted provider verified it.

## Relationship to trusted time

```text
external bootstrap authority
        ↓
VerifiedClockBootstrapAuthorityV2
(exact initial evaluation policy bound)
        ↓
ClockEvaluationPermitV2 construction
        ↓
original signed clock observations
+ permit-aware quorum verification
        ↓
opaque AcceptedClockBasisV2
        ↓
same policy by default
or explicit policy-migration capability
        ↓
trusted-time ETK currentness
```

This theorem complements #2315 and #2316. #2315 freezes non-circular permit/basis semantics. #2316 preserves exact accepted clock signer/observation witnesses. This reference freezes how the first trusted basis and its initial transition policy enter the system without being self-authorized by the clock.

## Deliberate nonclaims

This does not establish the authenticity of any particular hardware root, operator, HSM/TPM, secure RTC, platform attestation service, private key, provider policy, trust-snapshot rotation authority, clock policy migration, clock quorum, engineering currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2283, #2286, #2315, #2316.
