# symthaea-research-protocol

Outcome-blind preregistration, run binding, amendment, and deviation contracts for Symthaea research.

## Why this exists

A strong experiment should make it difficult to change its hypothesis, primary metric, baseline, exclusion rule, stopping rule, or analysis plan after seeing outcomes.

This crate provides a small reusable contract for that boundary.

## Frozen protocol

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

## Runs

`ResearchRunRegistration` binds a run to:

- frozen protocol digest;
- source commit;
- dataset-manifest digest;
- reproducibility-capsule digest;
- seed-manifest digest.

Imported run records should be checked with `validate_against(&FrozenProtocol)`, which rechecks exact protocol binding, the freeze-time boundary, and required lineage fields rather than assuming successful deserialization implies a valid registered run.

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

This crate does not establish external identity/authenticity, OS-level custody, amendment/deviation completeness, statistical validity, scientific truth, or action authority. Content digests detect identity mismatch; they are not signatures, secrecy primitives, or substitutes for semantic validation.

## Intended first users

- Planetary Perception / Wetland Watch;
- semantic downlink compression benchmark;
- synthetic subsurface hidden-world campaign;
- Symthaea Futures Laboratory extensions;
- cognition/recurrence evidence campaigns;
- any later Symtropy/Symthaea cross-world experiment.
