# MUSE acoustic observer v1 — gain-envelope evidence

**Status:** implementation and fixture work in progress. No Rust qualification result is claimed.

## Objective

Provide an independent, deterministic WAV-domain observer for edits whose contract is an explicit change in signal level over frozen time windows. It complements, and must not be conflated with:

- #12927 — render execution integrity and artifact-chain verification;
- #12949 / PR #12951 — symbolic event correctness, protected Score invariants and correspondence;
- #11389 — independent score-to-audio event conformance;
- #12971 — broader acoustic observer and human-perceptibility programme.

## Paradigm shift

Measure the transformation actually present in the rendered samples, independently of the Score-edit path. Keep target effect, protected content, clipping, evidence identity and human acceptance as separate conjuncts.

The observer consumes:
1. source WAV bytes;
2. candidate WAV bytes;
3. a frozen `muse-gain-envelope-contract-v1` describing sample-aligned target and protected windows, expected gain in dB, tolerances, format and scope.

It emits `muse-acoustic-observation-v1` with source/candidate/contract SHA-256 digests, per-channel measurements, per-window pass/fail/abstain, clipping status and a domain-separated receipt seal.

## V1 measurement and boundaries

- Uses windowed RMS level in dBFS, calculated independently for each channel. This is a useful level-change proxy, **not** an ITU-R BS.1770 integrated loudness / LUFS meter. The reported numerical floor is -300 dBFS; low-source abstention is governed separately by the contract's bounded threshold.
- Uses the exact same frame coordinates on source and candidate. It performs no alignment, sample-rate conversion, time stretching or dynamic time warping.
- Requires identical sample rate, channel count, frame count, sample format and bit depth for the source/candidate pair.
- Supports PCM integer 16/24/32-bit and IEEE float32 WAV input. Non-finite or grossly out-of-range float samples are refused.
- A hard decoded-input limit of 32,000,000 interleaved samples prevents unbounded allocation. This corresponds to at most 256 MiB of decoded f64 samples per WAV, excluding input buffers. Source and candidate must also use the same sample format and bit depth.
- Contract target windows and protected windows must each be nonempty, nonoverlapping internally, and disjoint from each other. Adjacent target windows allow a piecewise gain envelope to be specified and checked one segment at a time.
- A target or protected source channel below the configured RMS floor produces `abstain`, never a free pass.
- Clipping is independently measured over the full candidate WAV (absolute sample magnitude >=0.999); a configured ceiling produces a separate gate.
- A full-mix scope proves only a mix-level measurement. An `isolated_stem` scope is a caller assertion: the observer does **not** independently authenticate that a WAV is the claimed stem. Stem identity must be bound by upstream separation/material-lineage receipts, and arbitrary source separation is outside this slice.
- The observer does not establish instruction-language compatibility, note/event correctness, musical validity, source attribution, audibility for all listeners, preference or artistic acceptance.

The Hound WAV API defines `WavReader::duration()` in samples per channel, independent of channel count. The implementation uses that as the frame coordinate and multiplies by the channel count only for interleaved decoding. See https://docs.rs/hound/latest/hound/struct.WavReader.html.

## Current implementation

- `crates/domains/symthaea-muse/src/audio_edit_observer.rs`
- Exported through `symthaea_muse::audio_edit_observer`.
- Contract schema: `docs/research/schemas/muse-gain-envelope-contract-v1.schema.json`.
- Result schema: `docs/research/schemas/muse-acoustic-observation-v1.schema.json`.
- Synthetic contract and result fixtures live under `docs/research/fixtures/`.

## Authored adversarial coverage

Fifteen unit tests are authored. They currently cover:
- expected gain edit with protected windows;
- piecewise -2/-4/-6 dB envelope trajectory across adjacent windows;
- 16-bit integer PCM decoding;
- mixed sample encoding rejection;
- every-channel target verification (one channel left unchanged must fail);
- no-op candidate;
- protected-region gain mutation, including per-channel protected preservation;
- low-signal abstention;
- sample/frame timeline mismatch;
- overlapping target/protected windows;
- missing isolated-stem identity;
- clipping as a separate gate;
- non-finite float sample rejection;
- receipt tampering.

These tests are source additions only; no Cargo/rustc/rustfmt is available in the current execution environment. The test count must be checked against the latest branch head before making any validation claim.

## Research implications and benchmark plan

SMEBench's human validation and its audio-domain gain-envelope implementation for crescendo/decrescendo show why symbolic MIDI velocity cannot be used as a proxy for perceptual dynamics. MUSE should report separate strata for:
- exact symbolic velocity edits;
- fixed score/track gain edits;
- audio-domain gain-envelope edits;
- isolated-stem versus full-mix claims;
- synthetic known-answer versus real audio;
- channel count, sample rate, source level and dynamic range.

Initial prospective targets, not results:
- >=99% correct target-direction detection on deterministic synthetic gain fixtures;
- 0 false passes on no-op, wrong-direction, wrong-window, wrong-channel and protected-region negative controls;
- <=0.1 dB absolute gain error on noiseless synthetic fixtures, with tolerance/precision reported by sample format;
- protected-window false-pass rate 0 on deterministic controls;
- ECE <=0.05 after calibration on a rights-cleared development cohort;
- >=90% human change-recognition on preregistered high-salience eligible cases, with low-salience cases reported separately.

A stepwise or piecewise envelope target is represented by multiple disjoint adjacent target windows. A future observer version should add continuous envelope-shape features (slope, monotonicity, ripple/pumping and local transition behaviour) after native WAV tests and musician validation.

## Industry comparison

Suno Studio 2.0 now offers MIDI, automation, effects, synth and conversational production support. Moises Studio offers real-time collaboration, stem-based workflows, editable MIDI and visible/reversible agent actions. MUSE must not claim differentiation merely from having chat, MIDI or automation.

The differentiator is independent evidence that the requested transformation occurred and protected content remained stable. Public challenge comparisons still require the native scorer and matching denominator: the ICASSP 2027 Audio Editing Challenge's public MMAE set (2,000 examples / 17,741 rubrics) and the two hidden 500-example tracks are broad audio-editing comparisons, not substitutes for this narrow gain-observer qualification.

References:
- SMEBench: https://arxiv.org/abs/2610.36715
- ICASSP 2027 Audio Editing Challenge: https://audio-editing-challenge.github.io/
- Suno Studio 2.0: https://help.suno.com/en/articles/13670529
- Moises Studio: https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/
- Hound WAV API: https://docs.rs/hound/latest/hound/struct.WavReader.html

## Next steps

1. Pin JSON Schema Draft 2020-12 validation in the dedicated workflow; structurally checked synthetic contract/result fixtures are present, but no pinned schema validator or exact-head CI pass is claimed. Add negative schema fixtures.
3. Add explicit source/candidate encoding mismatch, multichannel protected mutation, float-invalid and oversized-WAV negatives.
4. Add a CLI only after the contract is stable; a library-only API is appropriate for the first slice.
5. Bind isolated-stem identity to the upstream stem/separation receipt rather than trusting the caller's label.
6. Add the observer receipt to the paired benchmark result only as acoustic evidence; it must not authorize its own output or replace human confirmation.
