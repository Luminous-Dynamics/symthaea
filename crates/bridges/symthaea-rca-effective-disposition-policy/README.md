# Symthaea RCA Effective Shadow Disposition Policy

RCA-003b.3a freezes the final preregistered profile boundary before any case evaluation exists.

## Why this wrapper exists

`RegisteredShadowDispositionPolicyV1` was frozen before the independent interpretation-root-set witness contract existed. Rewriting that reviewed policy body would blur provenance. This layer instead wraps the registered policy and adds the missing exact witness-profile binding.

```text
RegisteredShadowDispositionPolicyV1
        +
current independent interpretation-root-set witness profile
        ↓
RegisteredEffectiveShadowDispositionPolicyV1
```

The effective policy ID binds:

- exact base policy ID;
- exact base policy profile-contract digest;
- exact current interpretation-set-witness profile digest;
- effective-policy profile/schema.

A change to any of those inputs requires a new effective policy identity.

## Persistence

The wrapper is persistable because deserialization:

1. revalidates the wrapped registered base policy;
2. recomputes the current interpretation-set-witness profile digest;
3. recomputes the effective-policy profile digest;
4. recomputes the effective policy ID;
5. rejects any mismatch.

## Authority separation

This crate has no case, evidence-witness, interpretation-lineage, or interpretation-set-witness **instance** input. It consumes profile identity only.

```text
RegisteredEffectiveShadowDispositionPolicyV1
        !=
case evaluation
        !=
shadow disposition
        !=
canonical belief
        !=
workspace/GWT authority
        !=
action authority
        !=
self-improvement promotion
```

The next layer may accept this artifact as one input to a cross-artifact preflight check. Only a later qualified pure shadow engine may consume an issued preflight result.
