# Experience Signal Missingness Contract

**Status:** implementation draft; tests authored, not compiled or executed in this environment.

## Problem statement

A missing prediction-error comparison is not evidence of maximum surprise, zero surprise, or a measured neutral value. The previous `SignalComputer::compute_prediction_error(&[])` returned `1.0`, and the dormant ExperienceBus retrieval path assigned `1.0` when it found no similar experiences. The first path exposed no missingness marker; the second used a numeric surprise as a proxy for missing evidence.

This branch introduces an explicit validity bit for `PrincipledSignals`, with `prediction_error_value()` returning `Some(finite_value)` only when the bit and payload are both usable. The numeric field is retained for source compatibility and diagnostics, but must not drive decisions when the validity bit is false.

## Normative rules

1. No similarity candidates means `prediction_error_valid = false`. The numeric field is a neutral display placeholder (`0.5`), never a valid measurement.
2. Non-finite prediction error is unavailable even if a producer incorrectly sets the validity bit.
3. Legacy serialized `PrincipledSignals` payloads without `prediction_error_valid` deserialize fail-closed (`false`).
4. `should_clarify`, `action_readiness`, `learning_potential`, `dominant_signal`, primitive selection, experience learning-value calculation and Wisdom PE-dependent updates must not consume an invalid PE.
5. When PE is invalid, independent uncertainty and coherence channels remain active. Weighted scores renormalize over their available channels rather than substituting a fabricated PE.
6. Smoothing may retain the last numeric PE for diagnostics, but must clear validity for the current invalid event. The next valid PE starts a fresh PE series rather than smoothing against stale held evidence.
7. `compute_prediction_error` returns `None` for an empty input, non-finite input scores, or a non-finite result. Its established score convention (one minus arithmetic mean) is unchanged for valid inputs.
8. Downstream state updates that require measured PE must accept optional evidence and skip their PE-dependent branch when evidence is absent; other independent signal updates continue. In Wisdom, the autopoietic monitor must still record finite coherence and cycle accounting when PE is unavailable, use a neutral 0.5 PE prior only for aggregate display/closure calculation, expose PE observation count, and never insert that prior into the measured-error history.

## Compatibility boundary

`PrincipledSignals` and `ExperienceRecord` gain serde-defaulted boolean fields. Older serialized objects remain readable, but the missing validity field is considered invalid. This intentionally prioritizes safety over inferring provenance that was not serialized.

The return type of `SignalComputer::compute_prediction_error` changes from `f32` to `Option<f32>`. Repository search found no production call sites in the current source tree; test calls are migrated. This remains a public API change and should be reviewed before merging.

The live cognitive-loop bridge currently has only a scalar `prediction_error` argument, not the full typed provenance carried by `CycleSnapshot` on the PEV branch. This draft marks PE invalid on the first cycle and when non-finite, but that does **not** prove source-cycle consistency or detect every upstream unavailability status. A follow-up should pass real typed evidence across this boundary rather than treating finiteness as complete provenance.

## Required negative controls

- Empty candidate list must return `None`, not `Some(1.0)`.
- NaN or infinite similarity scores must not produce valid PE.
- Two signals that differ only in the numeric placeholder for invalid PE must produce identical action-readiness and learning-potential scores.
- Invalid high PE must not independently trigger `should_clarify`, `PROBE`, or a Wisdom boost.
- Invalid PE must not dominate the named signal or leak into experience record learning-value.
- Smoothing an invalid event after valid PE may hold the last numeric value, but the validity flag must be false; a subsequent valid value must be admitted without blending against stale PE.
- Old JSON without a validity field must deserialize with PE unavailable.
- Uncertainty/coherence decisions and coherence-only autopoietic cycle monitoring must remain active when PE is unavailable; PE history stays unchanged.
- Test per case and in a deterministic mixed trace; do not pool missingness cases with valid measurements.

## Proposed benchmark metrics

Report correctness first, utility second:

- **Missingness integrity:** invalid-to-valid promotion rate (target 0), false surprise triggers on missing PE (target 0), and invalid PE contributions to recorded learning value (target 0).
- **Decision invariance:** maximum difference in decisions/scores between paired inputs that differ only in an invalid PE payload (target 0).
- **Recovery:** valid PE after an unavailable cycle should be processed without requiring a restart, while stale PE must remain rejected where cycle identity exists.
- **Independent-channel preservation:** uncertainty/coherence decisions should match their valid-PE-free control.
- **Calibration and utility:** only after the hard gates pass, measure task success, adaptation after distribution shift, retention/forgetting, update count, wall-clock time, and memory/energy cost under matched seeds and budgets.

These are acceptance targets, not results. Passing them does not by itself demonstrate state-of-the-art learning performance. External comparisons must run the complete relevant benchmark protocol and report matched budgets, confidence intervals, and all failure traces.
