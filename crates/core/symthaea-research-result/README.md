# symthaea-research-result

Immutable result manifests for evidence-bearing Symthaea research.

## Purpose

`symthaea-research-protocol` freezes what an experiment said it would test and binds an exact run to source/data/environment/seed lineage. This crate closes the other end of that chain: it records what the run actually produced without allowing inconvenient primary outcomes, deviations, amendments, or null findings to disappear.

## Core invariants

### Primary metrics cannot disappear

Every preregistered `MetricRole::Primary` must have one `MetricResult` entry. That entry may be:

- a numeric/boolean/categorical observation;
- `Missing { reason }`;
- `NotComputed { reason }`.

Missing data are therefore represented as missing data, not silently removed from the result surface.

### Claims are references, not free prose

A `ResultClaim` must reference at least one reported metric or digested result artifact. Metric, hypothesis, and artifact ids are checked against the frozen protocol/result manifest.

### Exploratory remains exploratory

Claims against preregistered exploratory hypotheses cannot be labeled confirmatory. If the parent protocol reports a post-unblinding amendment or a deviation affecting the primary analysis, confirmatory claims are rejected.

### Null is a result

`ClaimDisposition::NullResult` and `Inconclusive` are first-class outcomes. The crate contains no API that converts absence of improvement into absence of a result record.

### Artifact provenance is explicit

The manifest requires at least one digested `Analysis` artifact and can bind raw outputs, metrics, tables, figures, models, forecast ledgers, verification records, and logs.

### Result identity is content-addressed

`ResearchResultManifest` carries a versioned BLAKE3 digest over the exact run registration, protocol digest, amendments, deviations, interpretation, artifacts, metric outcomes, and claims.

Mutation after construction is detectable with `verify_digest()`.

## Non-claims

This crate does not establish that:

- a claim is scientifically true;
- a p-value or effect estimate is valid;
- a model is causal;
- a result generalizes beyond its study population/regime;
- a confirmatory result is independently replicated;
- one metric should dominate another;
- a scientific conclusion grants authority to act.

Those remain separate evidence, replication, value, governance, and authority questions.

## Intended first consumers

- Wetland Watch / semantic-downlink experiments;
- hidden-world subsurface inference;
- Futures Laboratory physical-world extensions;
- consciousness/recurrence evidence campaigns;
- Symtropy/Symthaea controlled experiments.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-result --all-targets
cargo test -p symthaea-research-result
cargo clippy -p symthaea-research-result --all-targets -- -D warnings
```
