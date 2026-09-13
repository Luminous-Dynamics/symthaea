# symthaea-assurance-core

ASSURE-000 provides domain-neutral claim/evidence qualification semantics for Symthaea.

Its core rule is intentionally conservative:

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

The crate keeps exact subject identity, claim identity, evidence provenance, positive support tiers, negative/inconclusive findings, claim ceilings, and invalidation conditions explicit. It deliberately does not produce a scalar safety or trust score.

ASSURE-000 is not a certification engine, red-team runner, deployment authorizer, compliance mapper, logging proxy, or runtime sandbox. Later assurance tranches may consume external evidence and execute qualification campaigns on top of this kernel.
