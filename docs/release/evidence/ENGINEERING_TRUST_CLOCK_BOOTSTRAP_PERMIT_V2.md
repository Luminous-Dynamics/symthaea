# Engineering Trust Kernel — Verified Bootstrap → Clock Evaluation Permit V2

Status: independent composition theorem only. No production authority changes.

## Governing law

```text
VerifiedClockBootstrapAuthorityV2
+ matching bootstrap claim
+ matching ClockEvaluationPolicyV2
+ matching trust snapshot
!= automatically valid clock evidence

but may derive one bounded pre-candidate ClockEvaluationPermitV2
```

The permit is constructed **before** any candidate clock window is considered. It proves the exact initial policy and trust snapshot authenticated by bootstrap authority and enumerates the ClockAuthority keys that remain eligible across the entire conservative evaluation envelope.

## Frozen lineage

```text
ClockEvaluationPolicyV2
7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318

ClockBootstrapClaimV2
15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877

VerifiedClockBootstrapAuthorityV2
b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3

bootstrap anchor V2
fdfb5cfe737fc326792ed57dfe6aa93b1a2005fef021de6ddcb60e6a56144282

ClockEvaluationPermitV2
12ed76d13425ad5345fb99dd12436890c7cc33a32a919ec22c71fe9f21bd07d9
```

The `ClockEvaluationPolicyV2` identity remains the #2315 policy identity. The anchor/permit re-key because the anchor now contains a real verified bootstrap-authority capability ID rather than an opaque placeholder authority digest.

## Permit envelope

Fixture bootstrap interval:

```text
[1,499,000 ms, 1,499,500 ms]
```

Policy maximum transition:

```text
2,000 ms
```

Therefore permit evaluation envelope:

```text
[1,499,000 ms, 1,501,500 ms]
```

Both `clock-a / Ed25519` and `clock-b / MlDsa65` must remain active, `ClockAuthority`-eligible, and temporally valid across the entire envelope. The trust snapshot itself must cover the entire envelope.

## Fail-closed corpus

The independent oracle rejects:

- substitution of a different but valid `ClockEvaluationPolicyV2`;
- substitution of a different trust-snapshot digest;
- a trust snapshot expiring inside the evaluation envelope;
- loss of enough eligible ClockAuthority keys;
- loss of required algorithm diversity.

No candidate observation/window is an input to permit construction.

## Exact-byte evidence

The exact checked-in oracle bytes were executed locally:

```text
Python                     3.13.5
--self-test                PASS
python -m py_compile       PASS
raw source SHA-256         46291dfa1401af32a0cfd53a9c32395ac7a42aeb0517c80796f0dddc8bba9270
locally computed Git blob  e145937283888a9baf2d206569dd247aa7a80bbc
GitHub stored Git blob      e145937283888a9baf2d206569dd247aa7a80bbc
```

Repository CI remains a separate qualification layer.

## Production consequence

Production permit construction should consume the private `VerifiedClockBootstrapAuthorityV2` capability from #2386, plus the exact matching claim/policy/snapshot records. It must not accept a raw authority-evidence record as a substitute.

The permit itself should be a private, non-deserializable capability. Any serializable permit audit record is evidence/reconstruction material only.

## Successor-policy rule

This theorem covers the initial permit. For a successor accepted clock basis:

```text
same policy ID -> continuity path may proceed
changed policy ID -> explicit verified policy-migration capability required
```

A caller-supplied replacement policy must never silently widen authority.

## Deliberate nonclaims

This does not establish external bootstrap-provider authenticity, cryptographic provider qualification, clock-observation validity, accepted clock basis, trusted clock continuity, policy-migration authority, ETK currentness, requirement satisfaction, design qualification, deployment approval, or actuation authority.

Related: #1738, #2226, #2315, #2316, #2367, #2386.
