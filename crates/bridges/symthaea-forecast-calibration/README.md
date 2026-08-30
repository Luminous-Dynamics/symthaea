# symthaea-forecast-calibration

Physical-world forecast provenance adapter for Planetary Perception and the existing Symthaea Futures Laboratory.

## Core rule

Planetary Perception does **not** define a second forecast representation or a second scoring implementation.

Canonical forecast distributions, typed abstention, and proper scoring are reused from:

- `symthaea-futures-core`;
- `symthaea-futures-calibration`.

This bridge adds the pieces the simulation-focused Futures Laboratory does not own:

- explicit wall-clock ↔ forecast-tick binding;
- physical verification windows;
- Earth evidence references;
- model/scenario provenance for physical forecasts;
- append-only registration and resolution of physical forecasts.

## Hindsight resistance

A forecast is registered before later evidence resolves it. Bad forecasts, good forecasts, abstentions, and still-pending forecasts all remain visible.

Verification requires explicit `EvidenceStage::Verification` evidence.

## Explicit clocks

The Futures Laboratory is tick-indexed. Physical Earth observations are wall-clock-indexed. `PhysicalTimeBinding` makes the mapping explicit; no implicit convention such as “one tick = one day” is permitted.

A distribution's canonical `issued_at_tick` and `Horizon` must agree with the physical forecast record.

## Scoring

`ScoringRuleKind::{Brier, Crps, LogScore}` is delegated directly to `symthaea-futures-calibration`.

This crate intentionally does not reimplement Brier, CRPS, log score, reliability math, or probability validation.

Reports aggregate only within the same model, target, and scoring rule. Scores from incompatible rules are never averaged together.

## Abstention

`ForecastOutput::Abstain` remains first-class. An abstention can later be resolved against what happened so coverage/abstention rates remain auditable, but it is not converted into a numeric failure sentinel.

## Boundaries

- Historical calibration is evidence, not authority.
- A proper score does not prove a causal model.
- An abstention is not silently discarded.
- Verification does not rewrite the original forecast.
- Physical-world provenance remains separate from the simulation-specific `symthaea-futures-ledger::EvidenceRecord` until a generic cross-context ledger contract is deliberately designed.

## Next integration

Use this bridge in the Living Watershed witness so Sentinel-derived evidence can initialize a forecast, later Sentinel/local verification can resolve it, and the exact same proper-scoring machinery used by the Futures Laboratory judges the result.
