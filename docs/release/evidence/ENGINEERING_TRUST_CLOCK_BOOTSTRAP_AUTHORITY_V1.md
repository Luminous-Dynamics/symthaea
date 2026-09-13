# Engineering Trust Kernel — Clock Bootstrap External Authority V1

Status: independent reference theorem only. This document and oracle change no production authority.

## Governing law

```text
bootstrap claim
!= external authority evidence
!= verified bootstrap authority capability

candidate clock time
must not authenticate the authority that bootstraps that clock
```

The initial trusted interval is descriptive until an external authority adapter authenticates evidence bound to the exact claim. The trust kernel does not interpret a candidate clock window or candidate time while performing this bootstrap binding.

## Reference objects

`ClockBootstrapClaimV1` binds:

- schema `symthaea.trust.clock-bootstrap-claim.v1`;
- purpose `ClockBootstrap`;
- exact trust-snapshot digest;
- exact trusted lower/upper Unix-millisecond interval.

`ClockBootstrapAuthorityEvidenceV1` binds:

- exact bootstrap-claim ID;
- exact external provider ID;
- exact authority-policy digest;
- exact external-evidence digest.

`VerifiedClockBootstrapAuthorityV1` binds the exact claim, authority-evidence record, provider, and authority-policy revision after the external provider accepts that evidence.

The reference intentionally models the provider verdict as an external input. It does not claim to implement TPM/HSM/operator cryptography.

## Frozen vectors

```text
bootstrap claim
5bebb59ba5b95bd487977ffd296bf398e815a80eebca36813f80344e3b68439d

external authority evidence
1bf502c3fe07d5dc7151b35eb93e6c12e49bb1f465ae55b74a60ca3aabc09d7d

verified bootstrap authority
0298a2d298c7e1f1cfebee21fb91aac61c08f749f11de368910f4d3118e48760
```

## Fail-closed corpus

The oracle rejects:

- bootstrap-claim content edited under an unchanged claim ID;
- evidence bound to a different bootstrap claim;
- external-evidence content edited under an unchanged evidence ID;
- provider mismatch;
- authority-policy revision mismatch;
- external authority rejection.

Critically, the verifier function has no candidate clock-window or candidate-time argument.

## Exact-byte evidence

The final source bytes were executed locally:

```text
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         76cefb55cc8fc767a7b9d4e6d4d33dba96497f23ff77f4b3db3d7e1bf3cf9096
locally computed Git blob  39d68923b93fea707bac938409729bf8802a28d3
GitHub stored Git blob      39d68923b93fea707bac938409729bf8802a28d3
```

Repository CI is a separate qualification layer and is not inferred from this local execution.

## Production shape

A production adapter should expose the external verification boundary without offering a built-in permissive implementation. Successful verification should mint an opaque/private capability, conceptually:

```text
verify_clock_bootstrap_authority(
    claim,
    authority_evidence,
    configured_provider,
    configured_authority_policy,
    external_verifier,
) -> VerifiedClockBootstrapAuthorityV1
```

`VerifiedClockBootstrapAuthorityV1` should have private fields and no deserialization path. A serializable `ClockBootstrapAuthorityEvidenceV1` is audit evidence only; its content hash is not proof that a trusted provider verified it.

The trust-kernel layer should not define a default `AcceptAll`/boolean bootstrap verifier. Deployments must supply the actual authority adapter (for example a hardware-root, operator ceremony, secure platform attestation, or another separately qualified root).

## Relationship to trusted time

```text
external bootstrap authority
        ↓
verified bootstrap capability
        ↓
ClockEvaluationPermitV2 construction
        ↓
original signed clock observations
+ permit-aware quorum verification
        ↓
opaque AcceptedClockBasisV2
        ↓
continuity permits / successor bases
        ↓
trusted-time ETK currentness
```

This theorem complements #2315 and #2316. #2315 freezes the non-circular permit/basis semantics. #2316 preserves exact accepted clock signer/observation witnesses. This reference freezes how the first trusted basis must enter the system without being self-authorized by the clock.

## Deliberate nonclaims

This does not establish the authenticity of any particular hardware root, operator, HSM/TPM, secure RTC, platform attestation service, private key, provider policy, trust-snapshot rotation, clock quorum, engineering currentness, requirement satisfaction, design qualification, deployment approval, or physical actuation authority.

Related: #1738, #2226, #2283, #2286, #2315, #2316.
