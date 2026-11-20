# MUSE Score-Audio Conformance Observer

A standalone Rust observer for the first narrow Score-to-audio conformance slice. It is intended to complement #12927 render integrity, #12951 symbolic edit scoring and #12977 acoustic gain-envelope observation.

## Run

```sh
cargo run --manifest-path tools/muse-score-audio-conformance/Cargo.toml -- \
  verify SCORE.json AUDIO.wav CONTRACT.json
```

Exit codes: `0` pass, `2` fail, `3` abstain, `4` invalid input or verifier error.

## V1 support

- One assigned `PartId`.
- No overlapping note events.
- Mono WAV representing an explicitly identified isolated stem.
- Integer PCM16/24/32 or float32 WAV.
- Frozen Score tempo; fixed score-derived interior windows.
- Harmonic-salience FFT pitch estimate, MIDI pitch, estimated frequency, cents deviation and confidence.
- A domain-separated receipt binding typed-Score serialization, exact WAV bytes, observer contract and per-event results.

## Explicit non-claims

This observer does not estimate automatic alignment, tempo drift, onset/offset, duration, instrument identity, timbre, musicality or listener preference. It does not perform source separation or dense polyphonic transcription. Multi-part, overlapping, unassigned-part, too-short and low-confidence cases abstain. The caller-provided stem identity is not itself proof of stem lineage.

A passing pitch result is one bounded conformance claim. It is not a complete Score-to-audio verdict or the evidence-bundle's overall pass.

## Tests and qualification

Fifteen Rust unit tests are authored, including exact pitch, semitone and octave corruption, silence, polyphonic overlap, multiple parts, unassigned identity, short notes, unknown fields and receipt tampering. They have not been compiled or executed yet. The focused workflow runs format, tests and Clippy on the exact PR head.
