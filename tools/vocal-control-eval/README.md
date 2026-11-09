# Vocal control-response evaluator

A dependency-free Rust evaluator for controlled vocal parameter interventions. It accepts already measured feature vectors from an audio-analysis pipeline; it does not perform DSP or infer features from WAV bytes.

## Core question

When the renderer receives "increase breathiness", did breathiness increase enough, and did vibrato, pitch, and loudness remain within pre-registered tolerances?

This is a causal intervention-style test over a fixed baseline and treatment render. It is not proof of causality in the philosophical sense: the evaluator measures the observed difference between two renders. Use matched seeds, fixed lyrics/score/voice/backend/model assets, and repeated cases to isolate the control intervention.

## Protocol

1. Freeze a baseline VocalPerformance and render configuration.
2. Clone it and change exactly one control (for example breathiness, vibrato rate, portamento, or consonant articulation).
3. Re-render with identical model/voice assets, random seeds, score, lyrics, sample rate, and output processing.
4. Extract pre-registered acoustic/perceptual features from both dry stems.
5. Normalize changes by frozen feature ranges; score the intended target effect and collateral guardrails.
6. Repeat over the benchmark subset and report pass fraction, mean target effect, and maximum spillover separately.

The first implementation uses BTreeMap feature vectors and explicit MetricRange definitions. Missing, non-finite, or out-of-range observations fail closed. A control passes only if the target changes in the requested direction by at least the declared fraction of its range and every unrelated guard metric stays within its allowed fraction.

## Initial intervention portfolio

| Control intervention | Primary target metric | Candidate collateral guards |
|---|---|---|
| Increase breathiness | breathiness index / aspiration ratio | F0 median, vibrato rate/depth, loudness, lyric intelligibility |
| Increase vibrato depth | F0 modulation depth (cents) | median F0, vibrato rate, phone error, loudness |
| Increase vibrato rate | modulation frequency (Hz) | modulation depth, median F0, note onset |
| Increase portamento | transition glide duration / slope | note-center pitch error, phone timing, vibrato |
| Sharpen consonant articulation | phone onset clarity / consonant spectral cue | vowel formants, note pitch, phrase timing |
| Change register | register classifier / spectral envelope | lyric accuracy, identity, target pitch, artifact rate |
| Shift breath location | detected inhale onset and phrase coverage | preceding/following phone timing, pitch and lyric omissions |

Do not use one model's own acoustic estimates as the only authority. Pair signal-derived measurements with independent analyzers and blinded listener ratings; calibrate thresholds on a pilot before confirmatory tests.

## Proposed gates

- 100% deterministic positive controls pass.
- 100% wrong-direction, insufficient-effect, missing-feature, NaN, and out-of-range negative controls fail.
- For repeated real renders, report per-control efficacy and collateral-guard pass rates with confidence intervals.
- No global weighted "control score" can hide a catastrophic spillover in lyric identity, pitch, or voice identity.

## Scope

The library scores feature-vector deltas only. It does not extract F0, vibrato, breathiness, spectral tilt, speaker identity or intelligibility, and it does not claim any real renderer currently passes these intervention tests.
