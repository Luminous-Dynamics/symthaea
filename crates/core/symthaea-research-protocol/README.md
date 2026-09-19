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

`freeze()` produces a `FrozenProtocol` with a versioned BLAKE3 fingerprint over the serialized protocol. Later tampering is detectable.

## Runs

`ResearchRunRegistration` binds a run to:

- frozen protocol digest;
- source commit;
- dataset-manifest digest;
- reproducibility-capsule digest;
- seed-manifest digest.

This is intended to complement, not replace, the existing Symthaea reproducibility-capsule work.

## Amendments and deviations

Protocols may be amended, but amendments are append-only records referencing the frozen parent digest.

Amendment timing is explicit:

- before data collection;
- before outcome unblinding;
- after outcome unblinding.

A post-unblinding amendment automatically prevents the result from being labelled confirmatory by `classify_result`.

Likewise, a deviation affecting the primary analysis downgrades the result to exploratory unless the run is explicitly invalidated.

## Core principle

Null results and protocol failures are evidence.

The contract therefore does not provide an API for rewriting the frozen protocol or silently removing inconvenient outcomes.

## Intended first users

- Planetary Perception / Wetland Watch;
- semantic downlink compression benchmark;
- synthetic subsurface hidden-world campaign;
- Symthaea Futures Laboratory extensions;
- cognition/recurrence evidence campaigns;
- any later Symtropy/Symthaea cross-world experiment.
