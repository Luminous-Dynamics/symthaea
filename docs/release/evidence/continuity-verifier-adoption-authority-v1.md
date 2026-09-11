# Continuity Verifier Adoption Authority V1 — Frozen Subject

## Stack base

Stacked directly on draft PR #1503 exact head:

`e442b9f70ef80cc5d229f0249a7963a4f37ff397`

## Exact code subject

`81acb9b4143e1511fd92b8d53239163f5bcf87b1`

## Core separation

```text
VerifierProfileV1
!= VerifierProfileAdoptionTransitionV1
!= QualifiedVerifierProfileAdoptionV1
!= CurrentAuthorizedVerifierProfileV1
```

Existing profile-adoption subject/transition identities remain unchanged.

## Admission theorem

An exact adoption transition can become `QualifiedVerifierProfileAdoptionV1` only when:

- the transition and subject validate intrinsically;
- the exact verifier profile matches the adoption subject;
- the adoption-authority subject/root ID/root digest match an already provisioned non-Serde `TrustedVerifierAdoptionAuthorityRootV1`;
- external authentication evidence binds the exact domain-separated transition payload;
- admission occurs inside the exact adoption validity interval.

The trusted adoption root has no public constructor in V1. Test construction is crate-owned; production provisioning remains an explicit future integration boundary.

## Current-head theorem

`VerifierAdoptionCurrentnessClaimV1` binds the exact admitted transition/profile to:

- rollback-resistant platform/root profile + epoch;
- exact currentness sequence and predecessor-currentness ID;
- exact adoption transition digest + subject ID + profile ID + generation;
- exact predecessor transition digest;
- adoption-authority root digest and validity interval;
- fresh challenge;
- boot identity/counter + monotonic counter;
- exact anchor time and raw anchor evidence.

`CurrentAuthorizedVerifierProfileV1` is non-Serde and can only be qualified against an exact admitted adoption and authenticated currentness claim.

## Progression rules

Initial currentness is accepted only for generation 1 / Bootstrap with sequence 1 and no predecessor currentness.

Subsequent attestations allow only:

1. fresh re-attestation of the exact same adoption head; or
2. exact generation+1 advancement whose adoption transition names the previous transition digest and preserves the adoption authority + verifier role lineage.

Rollback, skipped generations, same-generation drift, platform-root substitution, boot-counter rollback, boot-identity drift without counter advance, non-increasing same-boot monotonic counters, and anchor-time rollback fail closed.

## Validity is not currentness

A valid time interval is necessary but not sufficient. V1 does not use TTL as a substitute for supersession/current-head authority.

## Non-claims

This frozen subject does not yet bind effect-coverage proofs to the current-authorized verifier head and grants no physical execution, recovery, promotion, bootstrap, or verifier self-adoption authority.

CI/compiler/test qualification remains outstanding until exact-head executable evidence exists.