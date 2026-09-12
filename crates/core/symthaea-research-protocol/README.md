# symthaea-research-protocol

Outcome-blind preregistration, run binding, amendment, deviation, and typed confirmatory-endpoint contracts for Symthaea research.

## Why this exists

A strong experiment should make it difficult to change its hypothesis, primary metric, baseline, exclusion rule, stopping rule, endpoint mapping, or decision rule after seeing outcomes.

This crate provides a small reusable contract for that boundary while keeping historical evidence identities stable across schema evolution.

## Frozen V1 protocol

A `ResearchProtocol` declares:

- research question;
- primary/secondary/exploratory/safety hypotheses;
- primary/secondary/safety/exploratory metrics;
- explicit baselines;
- exclusion rules;
- stopping rule;
- multiplicity policy;
- digested analysis plan;
- dataset plan;
- seed plan;
- mandatory null-result retention policy.

`freeze()` produces a `FrozenProtocol` with a versioned BLAKE3 identity over the canonical protocol bytes **and the freeze-time boundary**. Later mutation without a corresponding identity change is detectable.

Digest validity and semantic validity are deliberately separate:

```text
verify_digest() = this record matches its content-addressed frozen identity
validate()       = verify_digest() + recursively valid protocol semantics
```

A caller can legitimately create a different record and recompute its digest, but the new digest does not make an invalid protocol valid and does not preserve the historical identity of the original frozen record. Imported/deserialized evidence should therefore be accepted through the semantic validation boundary, not by digest comparison alone.

## Typed V2 confirmatory envelope

V1 is deliberately preserved as a historical evidence schema rather than being silently redefined when richer confirmatory semantics are added.

`FrozenConfirmatoryProtocol` adds an explicit V2 envelope over one exact validated V1 frozen protocol. It binds:

- a `MetricValueSchema` for every preregistered metric;
- exact confirmatory endpoint IDs;
- hypothesis-to-metric mappings;
- baseline/comparator identities where relevant;
- typed direct decision rules where the complete input is structurally represented;
- exact frozen external-analysis rule identities for complex/numeric/comparator analyses.

The V2 digest therefore changes when endpoint mappings, metric kinds, decision rules, categorical vocabularies, external rule digests, or the underlying V1 frozen identity change.

`ConfirmatoryRunBinding` additionally commits an exact V1 run registration to the exact V2 confirmatory-protocol identity.

### Canonical promotion boundary

For evidence intended to carry a stable confirmatory identity, use `CanonicalConfirmatoryProtocol` and `CanonicalConfirmatoryRunBinding` rather than treating caller vector order as scientific meaning.

Canonical construction normalizes only collections whose ordering is semantically set-like:

- metric-schema bindings are sorted by metric ID;
- outer endpoint records are sorted by endpoint ID;
- categorical allowed-value vocabularies are sorted lexicographically.

Duplicate vocabulary values remain errors; canonicalization does not silently deduplicate evidence.

Endpoint-local `metric_ids` and `baseline_ids` are **not** reordered. An externally frozen analysis rule may assign positional meaning to its inputs, so their order remains identity-bearing until a rule explicitly declares set semantics.

Canonical validation is rerun after deserialization. A raw V2 envelope can therefore be digest-valid and semantically valid yet still fail the stronger canonical-evidence boundary if its set-like collections are not in canonical form.

## Decision-rule scope

The generic crate deliberately implements only direct rules whose evidence inputs are completely represented:

- `BooleanMustBe`;
- `CategoricalMustEqual`.

Those direct rules require exactly one metric and no baseline comparator.

Numeric comparisons, paired analyses, rates, equivalence/non-inferiority calculations, and baseline comparisons remain `ExternalFrozenAnalysisRule` in V2 unless/until their complete comparator observations receive exact typed identities in the generic result model. This avoids claiming that the protocol crate mechanically verified an input it does not structurally possess.

## Runs

`ResearchRunRegistration` binds a V1 run to:

- frozen protocol digest;
- source commit;
- dataset-manifest digest;
- reproducibility-capsule digest;
- seed-manifest digest.

Imported run records should be checked with `validate_against(&FrozenProtocol)`, which rechecks exact protocol binding, the freeze-time boundary, and required lineage fields rather than assuming successful deserialization implies a valid registered run.

A confirmatory V2 consumer should additionally require the V2 run binding; the inner V1 protocol/run digest alone does not prove that metric kinds or endpoint mappings were preregistered.

This is intended to complement, not replace, the existing Symthaea reproducibility-capsule work.

## Amendments and deviations

Protocols may be amended, but amendments are append-only records referencing the frozen parent digest.

Amendment timing is explicit:

- before data collection;
- before outcome unblinding;
- after outcome unblinding.

A post-unblinding amendment automatically prevents the result from being labelled confirmatory by `classify_result`.

Imported amendment records should be checked with `validate_against(&FrozenProtocol)`. Complete amendment/deviation chronology and census are a separate event-lineage responsibility; this crate does not claim that a caller-supplied list proves completeness merely because each supplied record is valid.

Likewise, a deviation affecting the primary analysis downgrades the result to exploratory unless the run is explicitly invalidated.

## Core principle

Null results and protocol failures are evidence.

The contract therefore does not provide an API for rewriting the frozen protocol or silently removing inconvenient outcomes.

## Non-claims

This crate does not establish external identity/authenticity, OS-level custody, amendment/deviation completeness, statistical validity, scientific truth, independent replication, or action authority. Content digests detect identity mismatch; they are not signatures, secrecy primitives, or substitutes for semantic validation.

The V2 endpoint contract also does not establish that a chosen metric or statistical method is scientifically appropriate. It prevents type and endpoint substitution after freeze; scientific construct validity remains a separate evidence question.

## Intended first users

- Alignment Crucible / hostile-agent causal experiments;
- Planetary Perception / Wetland Watch;
- semantic downlink compression benchmark;
- synthetic subsurface hidden-world campaign;
- Symthaea Futures Laboratory extensions;
- cognition/recurrence evidence campaigns;
- any later Symtropy/Symthaea cross-world experiment.
