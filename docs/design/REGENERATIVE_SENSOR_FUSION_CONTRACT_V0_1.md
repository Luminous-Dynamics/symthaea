# Regenerative Health: Deterministic Sensor Corroboration v0.1

## Purpose

Individual sensor qualification answers whether a channel is admissible. It does not answer whether independent channels corroborate one another.

This contract adds a second evidence boundary:

sensor qualification -> multi-sensor corroboration -> structural-health inference.

The fusion layer intentionally produces evidence-quality states rather than a safety verdict.

## States

- Corroborated: the configured quorum of independently identified trusted sensors agrees within the configured residual spread.
- InsufficientEvidence: too few trusted channels are available to establish the configured quorum.
- Conflicted: trusted channels disagree materially, duplicate identities are present, or trusted observations carry incompatible configuration digests.

A degraded or untrusted sensor does not satisfy the quorum.

## Determinism

The gate uses a fixed minimum trusted-sensor quorum and maximum normalized-residual spread. The consensus residual is the median of trusted residuals, avoiding dependence on input ordering and reducing sensitivity to a single extreme value.

This is deliberately simpler than probabilistic sensor fusion. More sophisticated Bayesian, learned, or physics-informed fusion can be layered above this boundary without changing its basic safety semantics.

## Failure principle

Sensor disagreement must remain visible.

The system must not average away a disagreement and then silently present the resulting value as corroborated evidence. If trusted channels materially disagree, the result is Conflicted and remains outside the normal downstream recovery path.

## Research basis

Recent structural-health-monitoring work emphasizes multi-source fusion, explicit sensor-failure isolation, and reliability-aware weighting. A 2026 edge/cloud digital-twin study describes conflict identification followed by sensor-failure detection, isolation, and refusion; its experiments report improved robustness after isolating a failed strain sensor. A 2026 spatiotemporal digital-twin study similarly identifies sensor redundancy, error accumulation, and multi-source integration as central challenges in complex SHM. These studies motivate the architecture, but their reported numerical performance is not treated as a guarantee for this implementation.

## Non-goals

This module does not certify structural safety, identify the physical failure mechanism, or authorize repair. It establishes whether a set of already-qualified sensing channels provides sufficient corroborating evidence for downstream reasoning.
