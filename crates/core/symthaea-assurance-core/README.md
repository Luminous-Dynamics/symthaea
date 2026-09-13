# symthaea-assurance-core

ASSURE-000 provides domain-neutral claim/evidence qualification semantics for Symthaea.

Its governing rule is conservative:

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

The crate content-addresses exact subjects, claims, qualification plans, and results; keeps evidence provenance explicit; preserves negative and inconclusive findings as first-class outcomes; enforces preregistered claim ceilings; and records explicit invalidation conditions. It deliberately exposes no scalar safety or trust score.

Positive support is an evidence ladder, not a numeric grade:

`Structural -> Observed -> CausallySupported -> FunctionallySupported -> IndependentlyReproduced -> DeploymentQualified`

Higher tiers require explicit evidence classes. Independent reproduction additionally requires a verifier distinct from producer and executor, and deployment qualification requires an explicit deployment envelope. Negative outcomes (`NotDemonstrated`, `Contradicted`, `Inconclusive`, `Expired`, `Invalidated`) remain orthogonal to that ladder.

ASSURE-000 is not a certification engine, red-team runner, deployment authorizer, compliance mapper, logging proxy, or runtime sandbox. Later assurance tranches may consume external evidence and execute qualification campaigns on top of this kernel.
