# MEL-HEAR-011 — Physics-grounded auditory evidence substrate v0.1

**Status:** contract and metadata-fixture slice. No signal-processing implementation or real-audio qualification is claimed.

## Research thesis

Melothaea should be multi-representational rather than symbolic-only or embedding-only. A score is a compact description of intended musical events; a recording is a physical observation shaped by sources, performance, instruments, room acoustics, recording channels and processing. These objects are related, but they are not interchangeable.

The first useful causal approximation is

`x_c(t) \approx \sum_s h_{s,c}(t) * u_s(t) + n_c(t)`

where `u_s` is a source signal, `h_{s,c}` is the source-to-channel acoustic/production response, and `n_c` is noise. This is a starting model, not a universal law: nonlinear instruments, dynamic processing, time-varying effects and coupled resonances are possible model-mismatch sources.

## Representation stack

Keep independently addressable, time-aligned views rather than one irreversible embedding:

| View | Primary information | Important failure if omitted |
|---|---|---|
| Original waveform and channel metadata | physical samples, phase, clipping, clock and channel lineage | impossible to audit what downstream processing discarded |
| Complex STFT | local time-frequency energy and phase | magnitude-only analysis loses phase relations |
| ERB/gammatone-like bands | frequency-selective auditory-scale evidence | linear-frequency bins do not match perceptual frequency resolution |
| Constant-Q/log-frequency | pitch-related structure across octaves | fixed linear bins allocate resolution poorly for musical pitch |
| Modulation/scattering views | attacks, decays, vibrato, tremolo and playing-technique texture | a static spectrum misses temporal modulation |
| Spatial view | inter-channel timing, level, coherence and source position cues | stereo/ambisonic information disappears in mono |

Do not claim that any one of these is a complete cochlear, psychoacoustic or cognitive model. Their role is to retain complementary evidence and expose the loss incurred by each transform.

## Mathematical and physical work packages

1. **Auditory scene analysis.** Generate explicit evidence for harmonicity, common onset, comodulation, spectral continuity and spatial coherence. Retain competing event/source groupings; do not commit to one assignment when evidence is underdetermined.
2. **Psychoacoustics.** Separate loudness proxies from calibrated SPL; model masking, pitch salience, beating/roughness and critical-band interactions with declared calibration and operating limits.
3. **Performance dynamics.** Represent F0/multi-F0, onset/offset distributions, inter-onset intervals, tempo curves, rubato, vibrato, tremolo, articulation and amplitude envelopes as trajectories with uncertainty.
4. **Differentiable DSP.** Use known forward filters, resonators, envelopes, source mixers and room responses as inverse-problem baselines. Compare estimated parameters and counterfactual renders against known parameters and measured residuals.
5. **Room physics.** Consider a separate bounded room-impulse-response module with wave-equation or Helmholtz constraints. Do not require a neural PDE solver for everyday musical questions.
6. **Causal factorization.** Distinguish changes in musical content from performance, instrument, room/channel, production processing and corruption. Some mixtures are not identifiable; represent that explicitly.
7. **Geometry and information.** Preserve pitch class, register and pulse as separate coordinates; use event graphs and task-specific transport distances for voice-leading/motif alignment. Predictive surprise and minimum description length may propose structure, but they are not measures of artistic value or meaning.

## What exists in this first commit

- A strict JSON Schema for source identity, analysis configuration, independently named feature views, event hypotheses, scene hypotheses, causal-factor assessment, identifiability and query claims.
- One deterministic 16-sample, 1 kHz sinusoid fixture encoded as float32 little-endian hexadecimal text.
- A Rust validator that verifies the fixture's exact bytes, frame/sample count, representation declarations and the metadata-only claim ceiling.
- Negative controls for corrupted bytes, sample-count mismatch, uncalibrated SPL claims, phase-policy loss, invented event claims, artifacts on planned views and duplicated representation kinds.

The views are deliberately marked `planned`. The fixture does **not** claim that STFT, ERB or CQT features have been computed. This first slice establishes the wire and evidence contract before adding DSP.

## Next vertical slices

- Implement independent, pinned CPU feature extractors and known-answer fixtures for complex STFT, ERB filterbanks and CQT/log-frequency views.
- Add multi-source scenes with known onset synchrony, harmonicity, comodulation and stereo position; score grouping and all-occurrence event localization.
- Add room/channel counterfactuals (gain, EQ, reverb, noise, codec, microphone response) separately from musical edits.
- Connect audio-derived events to #7230 symbolic state through support intervals and explicit alignment receipts.
- Route property-relative FLIP/STAY judgments through #13908 and #14147; only #14151-certified exact symbolic actions may enable quotient logic, and no production pruning is authorized by this contract.

## Claim boundary

This code is a metadata validator, not an auditory model. It does not perform source separation, transcription, psychoacoustic loudness, event localization, room estimation, perceptual similarity, or musician-gesture inference. It makes no inference about performer emotion, intent, honesty or stable traits.
