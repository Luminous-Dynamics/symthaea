# symthaea-assurance-core

ASSURE-000 provides domain-neutral claim/evidence qualification semantics for Symthaea.

Its governing rule is conservative:

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

The crate content-addresses exact subjects, claims, qualification plans, evidence commitments, and results; keeps evidence provenance explicit; preserves negative and inconclusive findings as first-class outcomes; enforces plan-bound claim ceilings; and records explicit invalidation conditions. It deliberately exposes no scalar safety or trust score.

Positive **support strength** is an evidence ladder, not a numeric grade:

`Structural -> Observed -> CausallySupported -> FunctionallySupported`

Higher support tiers require explicit evidence classes. Reproduction evidence is deliberately **not** another support tier. `ReproductionStatus::EvidenceFromDistinctVerifier` is an orthogonal result dimension and requires reproduction evidence with a verifier identity distinct from producer and executor.

That status proves only the admitted evidence/provenance predicate. It does **not** establish that reproduction succeeded, semantically matched the original result, or was common-cause independent across organization, review process, verification toolchain, or evidence source. Successful/failed replication belongs to ASSURE-003 evidence resolution; common-cause verifier independence belongs to the stronger verifier-diversity layer.

Deployment is likewise not another support tier. Runtime evidence may be admitted, but evidence strength does not itself establish deployment eligibility or authority. A richer subject/deployment envelope belongs to later assurance tranches.

`QualificationResult::validate_and_bind` validates **caller-supplied** result dimensions against exact subject/claim/plan/evidence bindings and minimum predicates. It does not infer a verdict from heterogeneous evidence. Contradiction-aware evidence resolution is intentionally reserved for ASSURE-003, and temporal proof that a plan was preregistered before evidence production is intentionally reserved for ASSURE-002.

Invalidation is value-preserving: `QualificationResult::invalidated_by` returns a new invalidated result and leaves the original content-addressed result unchanged. Explicit predecessor/currentness/requalification lineage remains ASSURE-004 work.

The crate declares Rust 1.96 because that is the exact minimum toolchain qualified by its focused lane. Qualification proves exact candidate HEAD, formatting, check, tests, strict Clippy, and committed root-lock parity. The workflow itself is read-only; dependency drift fails closed.

Negative outcomes (`NotDemonstrated`, `Contradicted`, `Inconclusive`, `Expired`, `Invalidated`) remain orthogonal to the positive support ladder.

ASSURE-000 is not a certification engine, evidence resolver, red-team runner, deployment authorizer, compliance mapper, logging proxy, runtime sandbox, successful-replication authority, or independent-audit authority. Later assurance tranches may consume external evidence and execute qualification campaigns on top of this kernel.
