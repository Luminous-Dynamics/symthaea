# MEL-PERCEPTION-002 — audio front-end uncertainty evidence v1

**Date:** 21 November 2026  
**Parent evaluator:** [MEL-PERCEPTION-001 — paired audio/symbolic perception parity](https://github.com/Luminous-Dynamics/symthaea/pull/13164)  
**Implementation:** \`crates/domains/symthaea-muse/src/audio_frontend_evidence.rs\`  
**Schema:** \`benchmarks/mel/schemas/audio-frontend-evidence-v1.schema.json\`

## Problem

A downstream symbolic solver can be correct and still produce a wrong musical answer because the audio front end extracted the wrong notes, attacks, chroma or beat grid. Without a front-end uncertainty record, these two failure classes become indistinguishable.

The 2026 MUSE audio/MIDI perception study shows why this matters: symbolic MIDI reasoning can be near-ceiling while audio recognition remains substantially lower for syncopation and chord quality. An audio-to-symbolic pipeline must therefore expose what it actually heard, what it failed to extract, and how confident those estimates are.

## v1 evidence contract

Each front-end record binds:
- exact benchmark case and source audio SHA-256;
- sample rate, channel count and sample count;
- front-end system/version and immutable identity digest;
- four named stages: pitch salience, onset detection, chroma extraction and beat-grid tracking;
- explicit status for every stage.

Each stage is one of:

| Status | Required evidence | Meaning |
|---|---|---|
| Extracted | estimate digest, confidence, uncertainty; optional calibration identity | A typed output exists. Confidence and uncertainty are separately recorded. |
| Abstained | non-empty reason; no estimate/confidence/uncertainty | The stage intentionally declined because the signal or model state was insufficient. |
| NotRun | non-empty reason; no output/confidence/uncertainty | The stage was not invoked, which is distinct from successful empty extraction. |

If a confidence score is calibrated, the calibration artifact/cohort identity is bound by digest. If no calibration identity is supplied, the score remains explicitly uncalibrated; consumers must not silently interpret it as a calibrated probability.

The validator also enforces digest binding, sample-rate/channel/count bounds, unique stage records and complete coverage of the four declared stages. No missing stage is interpreted as a successful zero-valued feature.

## Engineering implemented

The module implements typed evidence records, schema identifiers, digest verification and negative-control validation. Its six tests cover:
1. complete extracted evidence matching the exact audio input;
2. case and audio digest mismatches;
3. missing-stage rejection and explicit NotRun acceptance;
4. abstention with reason and no fabricated output;
5. duplicate-stage and invalid-confidence rejection;
6. distinction between calibrated and uncalibrated scores.

This is an evidence-contract layer. It does not itself run pitch tracking, onset detection, chroma estimation or beat tracking; it does not claim that a front end is accurate.

## Next research increments

1. Bind actual Muse pitch/onset/chroma/beat estimators to these records.
2. Build synthetic known-answer tests with controlled SNR, tuning, tempo, expressive timing, timbre and reverberation perturbations.
3. Measure calibration against held-out, rights-cleared annotations: reliability curves, Brier score, expected calibration error and selective risk/coverage.
4. Preserve per-stage failure propagation. For example, an uncertain onset grid must widen syncopation uncertainty rather than being hidden by a downstream symbolic solver.
5. Report end-to-end audio task accuracy and conditional solver accuracy given correct front-end representation separately.
6. Keep abstention useful: the system should ask for a clearer clip or user-confirmed beat grid rather than inventing certainty.

## Qualification boundary

A passing test suite establishes the evidence type and validator behavior on deterministic fixtures only. It does not establish pitch/onset/chroma/beat estimation accuracy, human musicality, or improved MUSE benchmark scores.


**Integration note:** the validator is designed to attach to the paired perception case by exact `audio_input_sha256`; this preserves a single content-addressed source between front-end extraction evidence and the modality-comparison report.
