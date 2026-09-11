# Continuity Effect-Scope Restart Rebind V1 — Frozen Subject

Status: **DESIGN/IMPLEMENTATION SUBJECT ONLY — NOT QUALIFIED**

Exact code subject:

`7dd819948d01a19bb8d789558217e75a905b5aa3`

Direct parent stack head (#1490):

`011fd099014a633ea1d2a1dfba1f567dccfa9ccf`

## Core theorem

```text
persisted EffectScopedExecutionEnvelopeV1
+ exact durable A -> B intent
+ exact pre-authorized ExternalEffectPlanV1
+ exact reconstructed QualifiedExternalEffectAuthorizationV1
+ freshly rederived attempt-specific ExternalEffectContractV1
    -> ReboundEffectScopedExecutionV1
```

The serialized envelope cannot self-authorize after restart.

## Fail-closed bindings

Rebind rejects drift in:

- known-good intent / attempt;
- subject / source A / target B / distributed context;
- plan identity;
- qualified effect authorization identity;
- plan commit time and authorization time;
- coverage-manifest digest;
- freshly rederived external-effect contract identity.

## Non-claims

This proof grants no execution, retry, compensation, LKG promotion, bootstrap authority, or claim that all possible external effects were declared. It only reconstructs the exact protected effect-scope semantics required by downstream recovery proofs.
