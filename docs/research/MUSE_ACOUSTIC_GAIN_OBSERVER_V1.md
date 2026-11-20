# MUSE Acoustic Gain-Envelope Observer v1

**Status:** first deterministic WAV-domain observer authored; Rust tests and release qualification pending.

**Tracking:** [MUSE-AUDIO-EVAL-001 / #12971](https://github.com/Luminous-Dynamics/symthaea/issues/12971)  
**Related:** [render execution receipt #12927](https://github.com/Luminous-Dynamics/symthaea/pull/12927), [symbolic edit evidence #12951](https://github.com/Luminous-Dynamics/symthaea/pull/12951), [score-to-audio conformance #11389](https://github.com/Luminous-Dynamics/symthaea/issues/11389).

## Why this observer is separate

A Score/MIDI receipt can establish that the symbolic note velocities or durations changed. It cannot establish the actual rendered amplitude envelope. SMEBench reports that MIDI velocity alone performed poorly for crescendo/decrescendo, and uses an audio-domain gain envelope for those operations. This observer therefore evaluates WAV samples independently of the symbolic edit scorer.

Reference: [SMEBench](https://arxiv.org/abs/2610.36715).

## Contract and method

The contract is versioned as `muse-gain-envelope-contract-v1`; the observation receipt is `muse-acoustic-observation-v1`.

Inputs:
- Original and candidate WAV byte streams.
- Frozen sample rate, channel count and frame count.
- Explicit target windows with expected gain delta in dB and tolerance.
- Explicit protected windows with a maximum absolute gain delta.
- A minimum source RMS floor, candidate clipping threshold and scope declaration.

The observer decodes integer PCM (16/24/32-bit) and IEEE float32 WAV. It compares the source and candidate on the same sample-frame timeline. It refuses mismatched sample rates, channel counts or frame counts; it never time-warps or auto-aligns the candidate.

It computes windowed RMS dBFS independently for each channel. Every target channel must be within the target gain tolerance; every protected channel must remain within the protected gain tolerance. A quiet/near-silent source channel causes abstention rather than a guessed gain estimate. Candidate clipping is a separate failure gate. The receipt binds the contract digest and exact source/candidate WAV hashes.

`scope_kind = full_mix` supports a mix-level claim only. `scope_kind = isolated_stem` records a caller-supplied stem/bus identity; the string is provenance supplied by the caller, not proof that the media actually is that stem. A full mix cannot support an instrument-local claim.

## Deliberate limitations

This is a windowed RMS gain-envelope observer, **not** a BS.1770 LUFS/true-peak meter. It does not assess spectral balance, phase, stereo image beyond independent per-channel RMS, note/pitch correspondence, source identity, musical validity, instruction-language compatibility, or human perceptibility. It can establish that a measured envelope moved as contracted; it cannot establish that the edit sounds good or that listeners recognize the intended musical change.

A pure RMS observer can also miss equal-energy timbral or phase damage. Multi-resolution spectral/protected-content observers, pitch/onset observers, and blinded human evaluation remain separate follow-on layers.

## Executable API

The library entry point is:

```rust
observe_gain_envelope_v1(source_wav_bytes, candidate_wav_bytes, &contract)
```

The result has three independent acoustic gates:
- target-window gain adherence;
- protected-window gain stability;
- candidate clipping fraction.

The final `acoustic_observation_status` is pass, fail or abstain. It is not the benchmark's overall StrictEMR and does not replace instruction, integrity, musicality or human gates.

## Adversarial tests authored

Ten Rust tests are authored:
1. Correct -6 dB target envelope and unchanged protected windows pass.
2. No-op candidate fails the requested -6 dB target.
3. Protected-window attenuation fails preservation.
4. A silent source target window abstains.
5. Timeline/frame mismatch is rejected without time-warp alignment.
6. Clipping is reported as a separate failure gate.
7. Only one channel changed in a stereo target: the target fails.
8. Target/protected window overlap is rejected.
9. Isolated-stem scope without an explicit identity is rejected.
10. Receipt tampering is detected.

Synthetic fixture hashes are sentinels and do not refer to real WAV media. The authored tests have **not** been compiled or executed: this environment has no local Cargo/rustc/rustfmt, and the branch has no passing exact-head Rust qualification yet.

## Next steps

1. Run the pinned Rust test/format/Clippy workflow on the exact branch head.
2. Add known-answer integer-PCM fixtures in addition to float32 fixtures.
3. Add true-peak / loudness observers and frequency-domain protected-window comparisons.
4. Add independent pitch/onset/duration observers with coverage and polyphony abstention.
5. Run a blinded listener study with high-/low-salience strata, and preserve the human gate as independent evidence.
6. Integrate the acoustic observation into the paired result receipt without allowing it to authorize musical acceptance.
