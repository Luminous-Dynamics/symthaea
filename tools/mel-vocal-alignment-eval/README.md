# Mel vocal phoneme-alignment evaluator

A dependency-free Rust scoring primitive for supplied singing phoneme alignments.

It compares an independently authored reference event sequence with the output of an ASR/phoneme recognizer or forced aligner. It does not inspect waveform bytes, perform recognition, run forced alignment, estimate F0, or judge whether a vocal sounds human. That boundary is deliberate: evaluator correctness must not be confused with acoustic-model quality.

## Input contract

- One event per canonical phone token, with a stable word index, syllable index, and half-open interval on the source-audio sample-frame clock.
- Predictions use the same sample rate and frame origin. Optional confidence is retained as optional evidence and rejected when non-finite or outside [0, 1].
- Language-specific IPA normalization must happen before scoring and must be versioned by the caller. Unsupported phones must remain explicit failures; no silent schwa or nearest-phone fallback is permitted.
- A melisma is represented as one vowel phone covering a span containing two or more note events. The same vowel must not be counted again for each note.

## Metrics

- Phoneme edit counts: exact, substitution, deletion, insertion.
- Phone error rate: (S + D + I) / reference-phone count.
- Exact phone identity requires symbol, word index, syllable index, and vowel/consonant class to agree. Onset, offset, and duration error are calculated only for those exact matches.
- Onset timing includes median absolute error and empirical fractions within 20 ms, 50 ms and 100 ms, alongside MAE/p95. Vowel-onset coverage within 50 ms is also reported separately. These are conditional on exact identity pairs; this is conditional timing fidelity, not a joint transcription-plus-alignment score.
- Duration ratio: sum of predicted durations / sum of reference durations for exact identity pairs. It detects systematic under- or over-allocation without allowing a substitution to claim accurate duration.
- Melisma coverage: whether the predicted vowel interval covers the complete declared multi-note span.
- Undefined timing or melisma support is None, never a synthetic zero.

Phone alignment uses deterministic Levenshtein sequence alignment. Tie-breaking is diagonal, deletion, insertion. The scorer deliberately does not perform fuzzy phonetic matching: dialect equivalence belongs in an explicit, reviewed language profile and must not silently change a benchmark's ground truth.

## Why this is distinct work

The existing singing release path has score-conditioned pitch/timing checks, waveform-integrity checks, and human ratings. Those do not, on their own, quantify whether each consonant, vowel, and syllable was actually realized at the intended time. Speech WER alone is also insufficient: singing contains long vowels, melisma, vibrato, and expressive timing. This scorer adds a narrow, auditable layer for the event-alignment boundary; it does not replace F0, listening, or voice-identity evaluation.

## Integration sequence

1. Export an immutable reference phone/word/syllable alignment from the exact VocalPerformance and authorized lyric source.
2. Run one or more independent singing-robust ASR/phoneme-alignment systems on the exact dry WAV bytes.
3. Normalize their outputs with a versioned language-specific phone map; keep raw outputs and mapping receipts.
4. Score each system with this crate and report per-language, per-register, per-phoneme, per-session, and per-phrase results.
5. Compare predicted phone events with the musical note map, preserving melisma as one sustained vowel and allowing explicitly represented portamento/vibrato.
6. Use the report as diagnostic evidence until an external held-out corpus and blinded listener study establish decision thresholds.

## Test status

Sixteen deterministic unit tests are authored, including same-symbol wrong-word/syllable and wrong vowel-class controls, median timing, 20/50/100 ms onset coverage, vowel-only 50 ms coverage, duration-ratio accounting, and undefined empty-support metrics. No local compile/test pass is claimed until the exact commit is built with the pinned Rust toolchain. The dedicated PR workflow checks out the exact commit and executes formatting plus tests. A queued or skipped workflow is not a pass.
