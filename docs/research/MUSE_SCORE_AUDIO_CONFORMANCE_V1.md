# MUSE Score-to-Audio Conformance v1

**Status:** first standalone observer and CLI authored. Rust qualification pending; no build/test/Clippy pass is claimed.

**Tracking:** [#11389 — Verify rendered audio against the symbolic score](https://github.com/Luminous-Dynamics/symthaea/issues/11389).

**Related evidence layers:**
- [#12927 — render execution receipt](https://github.com/Luminous-Dynamics/symthaea/pull/12927)
- [#12951 — paired symbolic edit benchmark](https://github.com/Luminous-Dynamics/symthaea/pull/12951)
- [#12977 — independent WAV gain observer](https://github.com/Luminous-Dynamics/symthaea/pull/12977)
- [#12985 — cross-layer evidence bundle](https://github.com/Luminous-Dynamics/symthaea/pull/12985)

## Paradigm shift: rendering as compilation with conformance checking

A valid render receipt proves certain execution/artifact relationships. A symbolically correct Score edit proves a score-space transformation. Neither alone establishes that the intended notes reached the audio.

This work treats the symbolic Score as a source program and the renderer as a compiler/realizer. An independent observer checks whether a bounded set of expected musical events is present in the rendered samples. The observer must be independent from the edit and render producers, publish uncertainty, and abstain when its scope is not identifiable.

The first slice is deliberately narrower than general audio-to-score transcription:

- one assigned Score `PartId`;
- no overlapping notes;
- mono WAV for an explicitly identified isolated stem;
- exact frozen Score tempo and sample rate;
- score-derived interior analysis windows;
- independent harmonic-salience FFT pitch estimation.

It does **not** fit a time offset, resample, time-warp, separate sources or infer instrument identity.

## Package and files

Standalone package: `tools/muse-score-audio-conformance/`

- `src/lib.rs`: typed Score validation, WAV decode, pitch observer, result receipt and seal verifier.
- `src/main.rs`: `verify SCORE.json AUDIO.wav CONTRACT.json` CLI.
- Contract schema: `docs/research/schemas/muse-score-audio-contract-v1.schema.json`.
- Result schema: `docs/research/schemas/muse-score-audio-conformance-v1.schema.json`.
- Synthetic typed Score, contract and result fixtures: `docs/research/fixtures/muse-score-audio-*.synthetic.json`.
- Focused workflow: `.github/workflows/muse-score-audio-conformance.yml`.

## V1 measurement semantics

For each Score note, V1 maps rational beat onset and duration to seconds using the exact Score tempo. It selects a fixed-size window centered inside the note, applies a Hann window, computes an FFT and scores candidate MIDI fundamentals using weighted harmonic salience. It reports:

- expected and observed MIDI pitch;
- estimated fundamental frequency;
- pitch deviation in cents;
- harmonic-salience confidence;
- analysis-window frame bounds and RMS dBFS;
- per-note pass/fail/abstain and a non-compensating overall result.

A note is a failure when the expected window is silent or a sufficiently confident observed pitch is wrong. Low signal, insufficient confidence, short notes, unsupported score pitch range, overlapping events, multiple parts or missing part identity abstain. A failed event dominates abstentions.

The result binds:
- the repository's typed-Score serde digest using `muse-score-serde-json-sha256-v1`;
- exact input WAV byte hash;
- canonical contract digest;
- observer identity/version;
- per-event measurements and final result seal.

The SHA-256 receipt is integrity evidence only; it does not establish that the WAV is the correct stem. `scope_id` is supplied by the caller and must be linked to an upstream stem/material lineage receipt before a part-local claim is accepted by the evidence bundle.

## Explicit limitations

- No onset or offset conformance yet. The interior window tests pitch presence at a score-derived time, not exact note boundaries.
- No velocity/dynamics conformance. Use the separate #12977 gain-envelope observer for bounded audio-level interventions.
- No polyphonic pitch estimation or source separation.
- No instrument or part attribution from a full mix.
- No timbre, articulation, tuning-system, expressive timing or listener-preference verdict.
- No confidence calibration on real instruments yet. The initial confidence value is a spectral best-vs-second salience margin, not a calibrated probability.
- Score timing is treated as fixed tempo. A renderer with declared expressive rubato needs a separately versioned performance map and conformance tolerance; V1 does not silently fit one.
- This is not an overall Score-to-audio pass, StrictEMR or bundle pass.

## Authored test coverage

Nineteen unit tests are authored at the current implementation stage:
1. exact monophonic sine pitch;
2. semitone corruption;
3. octave corruption;
4. silent expected event;
5. overlapping/polyphonic Score events;
6. multiple assigned parts;
7. unassigned PartId;
8. note shorter than the frozen analysis window;
9. source/candidate sample-rate mismatch;
10. low-signal abstention;
11. out-of-calibration pitch abstention;
12. explicit confidence-threshold abstention;
13. unknown Score fields;
14. receipt tampering;
15. malformed rational denominator and oversized-numerator rejection;
16. unknown result field rejection after re-sealing;
17. outcome re-derivation prevents a pass that hides abstention;
18. result tampering;
19. typed Score digest binding.

These tests have not been compiled or executed in the current environment, which lacks Cargo/rustc/rustfmt. The dedicated GitHub workflow is intended to qualify exact-head formatting, compilation, tests and Clippy.

## Benchmark programme and prospective targets

All figures below are objectives, not achieved measurements.

### Synthetic known-answer conformance

Build a rights-free renderer-generated panel with exact Score, renderer binary/config/asset, MIDI and WAV digests. Sweep:
- MIDI pitches 24–108, duration, velocity and sample rate;
- sine, harmonic-rich and inharmonic/timbre-like synthetic tones;
- high/low SNR, vibrato, detuning and partial suppression;
- wrong semitone, octave, missing note, silent/truncated window and timing shift;
- mono isolated stem, stereo and dense multi-part audio (unsupported controls must abstain).

Prospective goals after frozen calibration:
- 100% detection of deterministic silent, wrong-semitone and wrong-octave controls in the supported pitch/range cells;
- zero false pass on deterministic wrong-note and missing-note fixtures;
- zero false pass when score/artifact/contract bindings differ;
- at least 99% exact-pitch agreement on noiseless supported synthetic fixtures;
- report coverage and abstention separately; do not count abstentions as correct classifications.

### Real instrument / renderer qualification

Only after synthetic baselines pass, add rights-cleared measured isolated stems. Partition by instrument/patch, pitch range, velocity, articulation, sample rate, SNR and performer. Keep a composer/work/recording-disjoint held-out split. Report:
- exact-note pitch accuracy and cents-error distribution;
- octave-error rate;
- event coverage / abstention;
- confidence calibration (ECE, Brier score, reliability diagram);
- false-pass rate on corrupted render controls;
- runtime and memory per minute of audio.

### Multi-observer extension

Add independent onset/offset, pitch, spectral/timbre and dynamic-envelope observers with separate contracts and calibration. Use agreement to improve confidence, not to let correlated models vote themselves into a pass. Preserve a disagreement state and abstain when observers are incompatible.

For polyphonic mixtures, use known-answer synthetic mixtures and aligned ground-truth stems first; then independent transcription models and human annotations. A model's own transcription cannot be its ground truth.

## Competitive positioning

SOTA comparisons must be matched by task and denominator. Broad audio-editing EMR is not comparable to note-pitch conformance. YuE2's public editing study, SMEBench and the ICASSP 2027 MMAE benchmark measure different tasks. This observer should first prove a narrower property well: whether each expected Score pitch is present in the declared isolated audio window, with uncertainty and tamper-evident binding. The wider bundle can then combine this with symbolic, render, acoustic, score-to-audio and human receipts without a compensating aggregate score.

## Next implementation steps

1. Exact-head Rust workflow qualification and repair any compile/test/Clippy findings without weakening gates.
2. Add calibrated onset/offset observer on the same frozen timeline; no automatic alignment before its own alignment benchmark exists.
3. Add multi-window note and melody-contour observer, then controlled polyphonic synthetic mixtures.
4. Register the public score-audio seal verifier in the evidence-bundle tool only after its schema, tests and provenance semantics are stable.
5. Bind isolated-stem identity to a material/stem-lineage receipt before permitting the score-audio component to pass.
