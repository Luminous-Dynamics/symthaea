# MUSE-OBS-001 — Provenance-bound audio event pitch observations (V1)

**Status:** research implementation slice; authored tests not yet compiled or executed  
**Date:** 11 November 2026  
**Depends on:** [MUSE-PERF-001, PR #11541](https://github.com/Luminous-Dynamics/symthaea/pull/11541)

## Why this exists

Muse already has WAV hashing and basic artifact audits (sample rate, channels, bit depth, duration, peak and clipped-sample count) in the blinded-study artifact path. It also has audio feature extraction and an active-inference surprise meter. Those facilities are useful, but neither answers a different question:

> For a particular score-bound MIDI event, is there measurable acoustic energy near the expected pitch during the event's scheduled time interval?

V1 adds a deliberately modest, independent spectral observation over the exact WAV bytes and the exact part-aware MIDI manifest. It does not duplicate the existing WAV integrity audit, and it does not treat the renderer's exit status as proof of musical correctness.

## What V1 does

1. Checks the MIDI artifact's byte digest and the domain-separated manifest digest.
2. Checks the schema identifiers, tempo/PPQ representation, note ordinals, and each note's part/segment/instrument/track binding.
3. Parses the WAV bytes directly and rejects malformed headers, invalid channel/frame layouts, and non-finite float samples.
4. Converts each note's integer MIDI ticks into expected audio sample frames using the manifest's integer microseconds-per-quarter and PPQ.
5. Selects a centered window fully inside the expected event interval, with a frequency-dependent minimum window length. Sample-frame conversion rounds half-frame ties upward using integer arithmetic.
6. Uses a Hann-windowed FFT to measure the fundamental and harmonic peak-to-local-sideband ratios.
7. Emits one observation per source-note ordinal, including the expected frame interval, analyzed window, measured salience, and an explicit abstention status when the note is too short, truncated, or below the available frequency resolution.
8. Seals the report with a domain-separated receipt digest bound to the source score digest, MIDI bytes, MIDI manifest, and WAV bytes.

The report is descriptive. The dB salience is not a calibrated probability, confidence interval, audibility threshold, or pass/fail score.

## Claims deliberately withheld

V1 does **not** establish:

- exact acoustic onset or note-off timing; it probes a window inside the expected event interval;
- that a detected partial originated from this note rather than another simultaneous note, a harmonic collision, or background material;
- acoustic attribution to PartId, instrument ID, segment, performer, or MIDI channel; those fields are copied from the trusted symbolic manifest, not inferred from the waveform;
- correct timbre, patch, articulation, dynamics, tuning, stereo position, mix quality, or perceptual musicality;
- that every note must be measured; short events and low-frequency events may explicitly abstain;
- independent authentication of the receipt producer; a digest detects later mutation relative to a retained copy, but it is not a signature or trusted timestamp.

A high salience value is not proof of note identity. Polyphonic overlap, reverb tails, sample resonances and harmonic collisions can produce false support. The V1 output must not authorize a release or certify a renderer.

## Test coverage authored

Eight focused tests cover:

1. a synthetic A4 tone produces a measurable expected-pitch observation;
2. an off-pitch synthetic control has lower expected-fundamental salience;
3. notes shorter than the analysis window abstain;
4. MIDI-byte tampering is rejected before analysis;
5. WAV replacement and receipt-field tampering are detected;
6. silent audio produces an explicit no-energy abstention;
7. MIDI tick-to-sample mapping uses checked integer arithmetic and deterministic rounding;
8. non-finite float audio is rejected.

The tests are authored but have not been compiled or run in this environment. The focused GitHub Actions workflow checks out the exact PR head and uses Rust 1.96.0.

## Benchmark programme

### B0 — Known-answer signal fixtures

Generate tones with known frequencies, onsets, durations, sample rates, channel layouts, SNRs and controlled harmonic content. Report pitch-salience distributions and abstention rate. Include exact-frequency-bin and off-bin fixtures; do not tune thresholds on the final holdout.

**Initial target:** all supported long-note fixtures produce a measured observation; planted off-pitch controls show a preregistered separation. The unit test currently asks for a 20 dB separation in its one pure-tone fixture only. That is not a general instrument threshold.

### B1 — Polyphony and harmonic collision panel

Include simultaneous intervals, unisons, octaves, fifths, dense chords, bass fundamentals with upper-part harmonics, repeated notes, overlapping releases, and reverberant tails. Measure false support, missed support and abstention separately by pitch/range and polyphony.

### B2 — Renderer and patch calibration

Freeze exact MIDI, renderer binary, soundfont/asset digests, sample rate, mastering configuration and audio bytes. Evaluate each supported renderer/patch/range combination. Report unsupported or uncalibrated combinations as such. A GM program number is not a patch qualification.

### B3 — Timing sensitivity

Shift a known note's rendered onset and release around the expected MIDI interval. V1 currently reports an in-event pitch-energy window; it is not an onset detector. A later version should add independent onset-envelope and note-offset observations before any event-timing claim is made.

### B4 — Independent transcription comparison

Compare the event probe with a separately implemented multi-pitch transcription or event detector on the same frozen renders. Keep the symbolic manifest hidden from the acoustic estimator except for evaluation. Report precision, recall, pitch error, onset error, offset error, and calibration/abstention. Human review is required for ambiguous polyphonic cases.

### B5 — Adversarial negative controls

Test silence, wrong pitch, octave-only energy, detuned tones, noise, wrong sample rate, truncated files, malformed WAVs, altered MIDI bytes, changed manifests, duplicate ordinals, missing tracks, and score/MIDI/WAV cross-binding. Every invalid input must be rejected or explicitly abstained; it must never receive a false successful certification.

## Promotion gates for later versions

- 100% rejection of tampered MIDI/WAV/manifest bindings in the frozen adversarial suite.
- 100% source-note ordinal coverage in the receipt, with unsupported observations represented as abstentions rather than dropped events.
- Report event-level precision/recall and abstention coverage with 95% confidence intervals on a frozen held-out render corpus.
- No renderer/patch family promoted until calibrated on independent known-answer and polyphonic fixtures.
- No claim of note timing correctness until an onset/offset detector is validated separately.
- No release qualification until an independent acoustic implementation and a human-reviewed error sample agree within preregistered tolerances.

## Next research steps

1. Add a calibrated onset/offset observation with uncertainty bounds.
2. Add renderer/patch-specific response profiles without confusing a profile with a universal instrument model.
3. Add independent audio-to-MIDI event extraction and cross-implementation agreement.
4. Where practical, render isolated stems or use per-part bounces; only then can acoustic observations begin to support part attribution.
5. Bind the observation receipt into the existing render/study artifact bundle rather than creating another parallel provenance system.

The intended paradigm shift is not “listen to the model and let it grade itself.” It is a split between authored musical intent, execution artifacts, independently measured acoustic observations, and separately qualified musical judgement.
