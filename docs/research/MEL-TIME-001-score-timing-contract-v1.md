# MEL-TIME-001-S1 — Score timing contract and exact symbolic verifier V1

**Status:** Rust implementation and unit tests authored; not compiled or executed  
**Date:** 2026-11-03  
**Tracking issue:** [#10262](https://github.com/Luminous-Dynamics/symthaea/issues/10262)  
**Implementation:** `crates/domains/symthaea-muse/src/score_timing.rs`

## Motivation

TimingBench (Bhake et al., August 2026) measures whether text-to-music systems place requested events such as instrument entrances or pauses inside instructed timestamp windows. It uses 250 prompts with machine-readable event schemas and task-specific detectors behind rejection-only gates; its detector localization error against human annotations varies by event family.

Reference: https://taan.ai/research/rushing-or-dragging/

The first Melothaea increment should not begin with an audio detector. The symbolic Score already contains exact onsets and durations, so we can establish a deterministic oracle for what the score itself requested and emitted before measuring whether a renderer faithfully realizes it.

## V1 contract

A `ScoreTimingContractV1` contains versioned requirements with stable requirement IDs. Each requirement selects events by:
- voice role;
- exact assigned part identity; or
- exact assigned part plus voice role.

V1 supports two checks:
- **OnsetCount:** count note onsets in a half-open window `[start, end)`, with explicit minimum and optional maximum.
- **Silence:** require that no selected note overlaps the half-open window. Notes that start before the window and sustain into it count as violations.

Windows may use:
- **Beats:** absolute beat positions from the start of the Score.
- **Seconds:** elapsed seconds from score start, converted using the Score's single constant tempo.

A requirement result retains exact matching note indices. Note indices are scoped to the exact serialized Score digest in the report; they are not persistent event IDs. The legacy Score has voice role and optional part identity but no instrument-identity field, so V1 cannot prove an instrument-specific entry/exit. The report binds the schema and explicit verifier version, contract digest, exact serialized Score digest, tempo and duration. V1 serialization hashes are over the repository's `serde_json` representation; they are not claimed as a universal cross-language canonical JSON format.

## Fail-closed behavior

The verifier rejects:
- unsupported schema versions, empty/duplicate requirement IDs or invalid windows;
- invalid min/max event-count ranges;
- part selectors without assigned part identity;
- invalid score tempo, duration or note timing;
- requirements whose windows exceed the score.

An unsupported timing map must not be silently approximated. V1 does not support piecewise tempo maps, expressive rubato, sample-frame clocks, performed MIDI scheduling, audio encoders or rendered-audio localization.

## Unit tests authored

1. Half-open beat windows select exact onset note indices.
2. Seconds convert under the exact constant score tempo.
3. Silence checks detect a note that sustains into the requested interval.
4. Unassigned part identity is rejected rather than guessed.
5. Dense decoy onsets exceeding the declared maximum fail rather than scoring as success.
6. Windows extending beyond the Score fail closed.
7. Report digests resolve to the exact serialized Score and contract, and the verifier version is explicit.
8. Duplicate requirement IDs and reversed windows fail closed.

These tests are authored only; no Cargo compilation, rustfmt or test pass is claimed.

## Qualification targets

- 100% correct symbolic event/window decisions on deterministic fixtures.
- Zero false silence passes for notes overlapping a silence window.
- Zero guessed identity for unassigned score parts.
- Every requirement returns a pass/fail result with exact matching note indices.
- Byte-identical reports for fixed contract, Score and environment.
- No cross-modality inference: passing the symbolic oracle is not evidence that the audio renderer delivered the same timing.

## Next layer

After the symbolic oracle is reviewed and executed, add a separate rendered-audio evaluator contract with:
- detector identity/version and calibration artifact digest;
- detector-specific localization uncertainty;
- positive/negative controls for dense decoys, wrong timbre, silence leakage and duplicated transients;
- p50/p95/p99 localization error in milliseconds and beat fractions;
- explicit ambiguous/not-estimable results.

Human timing tolerances should be derived per task/instrument/tempo from calibration and a musician pilot, not from a universal millisecond constant. The audio evaluator must not reuse the generator's own beat estimates as sole ground truth.

## Claim boundary

This module qualifies exact timing requirements against symbolic Scores at one constant tempo. It does not establish audio timing controllability, groove, expressive rubato, musician anticipation, collaboration quality or listener preference.
