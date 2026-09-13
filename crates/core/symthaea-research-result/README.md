# symthaea-research-result

Immutable result manifests and endpoint-bound evidence validation for Symthaea research.

## Purpose

`symthaea-research-protocol` freezes what an experiment said it would test and binds an exact run to source/data/environment/seed lineage. This crate records what the run actually produced without allowing inconvenient outcomes, missing values, deviations, amendments, or endpoint records to silently disappear.

The V1 result manifest remains a historical content-addressed evidence format. V2 is additive: it binds that exact V1 result to the canonical typed confirmatory protocol/run identity from `symthaea-research-protocol` and applies stricter endpoint/evidence validation without redefining V1 digests.

## V1 invariants

Every preregistered primary metric must have a `MetricResult` entry. An entry may be Numeric, Boolean, Categorical, Missing, or NotComputed. Missing data are therefore represented explicitly rather than omitted.

Claims reference reported metric/artifact identities rather than carrying ungrounded prose alone. Exploratory hypotheses cannot be relabeled confirmatory. Post-unblinding amendments and primary-analysis deviations downgrade the overall interpretation, while invalidation requires invalidated claim interpretations.

At least one digested Analysis artifact is required and the V1 manifest has a versioned BLAKE3 identity over the exact recorded result contents.

## V2 endpoint-bound result

`ConfirmatoryResearchResultV2` preserves the exact V1 result identity and additionally binds:

- the canonical V2 confirmatory-protocol digest;
- the canonical V2 run-binding digest;
- exact endpoint-to-terminal-claim bindings;
- frozen metric value schemas;
- exact endpoint hypothesis and metric input order;
- analysis-output references for externally frozen analysis rules when a confirmatory positive/negative/null conclusion is claimed.

A V2 object represents a campaign that began under a confirmatory endpoint contract. It does **not** imply that the final campaign interpretation remained confirmatory. A legitimate post-unblinding amendment, primary-analysis deviation, or invalidation remains representable and must preserve the corresponding Exploratory/Invalidated claim interpretation.

This distinction is intentional:

```text
confirmatory preregistration
!= confirmatory final interpretation
```

## Complete endpoint boundary

`CompleteEndpointResearchResultV2` is the stronger validation witness intended for consumers that require selection-complete endpoint evidence.

It adds two interpretation-independent rules:

1. every frozen V2 endpoint receives exactly one terminal claim binding, including safety and secondary endpoints;
2. a required metric recorded as Missing or NotComputed cannot support `ConsistentWithHypothesis`, `InconsistentWithHypothesis`, or `NullResult`, whether the terminal claim is Confirmatory or Exploratory.

The wrapper does not mint a new digest. The inner V2 result remains the evidence identity; the wrapper is a stronger validation boundary over that identity. Imported/deserialized wrappers must call `validate_against` again.

## Direct vs external decision rules

For direct Boolean/Categorical decision rules, a genuinely confirmatory claim disposition must agree with the observed endpoint value. Missing/NotComputed evidence may only produce an absence-compatible disposition such as Inconclusive or NotEvaluated.

For `ExternalFrozenAnalysisRule`, the generic result layer does not pretend to be a statistical engine. Confirmatory positive/negative/null conclusions require a bound Analysis output artifact, while the protocol separately commits the exact external rule identity and artifact digest.

Important non-equivalence:

```text
frozen analysis-rule identity
+ result Analysis artifact
!= proof the rule was faithfully executed
```

A later execution/verification receipt should close that theorem. Until then this layer establishes identity, typed inputs, endpoint completeness, and output binding—not faithful statistical execution.

## Event-lineage boundary

This crate validates every supplied amendment/deviation but does not prove that the supplied event census is complete or current. The append-only event-lineage/current-head theorem tracked separately by the research-integrity work remains required before a caller can claim complete protocol-history custody.

## Non-claims

This crate does not establish scientific truth, construct validity, statistical validity, causal identification, independent replication, faithful execution of an external analysis rule, amendment/deviation census completeness, operating-system custody, software qualification, deployment validity, or authority to act.

A successful source-level test or digest check is not executable qualification.

## Intended first consumers

- Alignment Crucible / hostile-agent causal experiments;
- Planetary Perception / Wetland Watch;
- semantic-downlink experiments;
- hidden-world subsurface inference;
- Futures Laboratory physical-world extensions;
- cognition/recurrence evidence campaigns;
- Symtropy/Symthaea controlled experiments.

## Required package gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-result --all-targets
cargo test -p symthaea-research-result
cargo clippy -p symthaea-research-result --all-targets -- -D warnings
```
