# symthaea-assurance-core

ASSURE-000 provides domain-neutral claim/evidence qualification semantics for Symthaea.

Its governing rule is conservative:

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

The crate content-addresses exact subjects, claims, qualification plans, evidence commitments, and results; keeps evidence provenance explicit; preserves negative and inconclusive findings as first-class outcomes; enforces preregistered claim ceilings; and records explicit invalidation conditions. It deliberately exposes no scalar safety or trust score.

Positive support is an evidence ladder, not a numeric grade:

`Structural -> Observed -> CausallySupported -> FunctionallySupported -> Reproduced`

Higher tiers require explicit evidence classes. `Reproduced` requires reproduction evidence with a verifier identity distinct from producer and executor. That establishes identity separation only; it does **not** establish common-cause independence across organization, review process, verification toolchain, or evidence source. Symthaea's stronger verifier-diversity work models those dimensions separately.

Deployment is intentionally not another support tier. Runtime evidence may be admitted, but evidence strength does not itself establish deployment eligibility or authority. A richer subject/deployment envelope belongs to later assurance tranches.

Negative outcomes (`NotDemonstrated`, `Contradicted`, `Inconclusive`, `Expired`, `Invalidated`) remain orthogonal to the positive ladder.

ASSURE-000 is not a certification engine, red-team runner, deployment authorizer, compliance mapper, logging proxy, runtime sandbox, or independent-audit authority. Later assurance tranches may consume external evidence and execute qualification campaigns on top of this kernel.
