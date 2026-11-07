# Voice acoustic faithfulness — fixture design v1

**Status:** design-only manifest and test protocol. No audio corpus is included and no benchmark has been run.

Parent research: issue [#10890](https://github.com/Luminous-Dynamics/symthaea/issues/10890).

## Purpose

Test whether a spoken interaction system uses acoustic evidence rather than merely reconstructing likely prosody, speaker attributes or background events from transcript semantics and prompt context.

The key design is **paired counterfactual intervention**: hold one information channel invariant while deliberately changing another. Evaluate not just accuracy but whether the model's output changes in the right dimensions and remains stable in the dimensions that should not change.

## Source audit and reusable baseline

- The current native `symthaea-stt` acoustic/HV path is not qualified lexical ASR. Its README reports poor word-recognition results and identifies acoustic/perceptual representation as the honest capability.
- `symthaea-vocal-tract::formant_extraction` already provides a dependency-free candidate DSP baseline: LPC formants, autocorrelation F0/periodicity, RMS energy and 200 Hz frames. It has synthetic tone, silence, vowel and frame-timing tests. This does not establish natural-speaker accuracy.
- `examples/voice_affect_gate.rs` checks whether Symthaea's *own generated voice* changes acoustically under a controlled valence/arousal grid and a small held-out inverse model. It is an outbound self-monitoring experiment, not inbound human prosody understanding.
- `src/perception/audio.rs` includes a separate placeholder prosody API with zero-crossing pitch proxy, zero pitch variance and zero speaking rate. Do not treat that output as qualified F0 or speaking-rate evidence.

Reuse the formant extractor as the first DSP comparator. Extend and independently validate its behavior before adding a second pitch tracker or learned paralinguistic model.

## Evaluation lanes

Keep these lanes distinct:

1. **Perception:** describe measurable acoustic observations, with frame spans and uncertainty.
2. **Lexical recognition:** identify words and preserve source-frame alignment; ASR quality is measured independently.
3. **Acoustic faithfulness:** resist misleading transcript/prompt priors when they conflict with measurable audio.
4. **Interaction:** use qualified observations to inform participation without granting an acoustic model authority to decide identity, truth or actions.
5. **Generation:** follow an authorized style plan while preserving semantic content; score output naturalness separately.

## Counterfactual fixture set

The companion `fixtures-v1.json` is a manifest of proposed pair families. Each family must eventually bind exact audio artifact digests, generation/recording parameters, native annotations, split identity and license/consent metadata.

| Family | Controlled change | Must remain invariant | Expected change |
|---|---|---|---|
| Same words / different delivery | Prosody, rate, intensity, voice quality | Lexical content | Measured delivery description |
| Same delivery / different words | Lexical sequence | Delivery features within tolerance | Lexical content |
| Transcript or prompt conflict | Misleading text/caption | Acoustic evidence | Explicit disagreement / audio-grounded answer |
| Background swap | Ambient scene | Foreground speech | Scene/event description |
| Speaker swap | Session-local voice source | Transcript and intended delivery where controlled | Session-local speaker hypothesis |
| NVV insertion/removal | Laughter, sigh, breath, gasp, cough | Neighboring lexical content | Event category and timed span |
| Channel/noise perturbation | SNR, codec, reverberation, clipping | Intended lexical content | Quality/uncertainty and acoustic conditions |
| Temporal shuffle | Chunk order or event timing | Component frames | Temporal interpretation should fail safely |

A benchmark pair is not valid solely because a TTS system emitted two files. Verify signal-level manipulation and human-listen plausibility on a held-out subset. For natural data, obtain consent and independent annotation.

## Required baselines

- Transcript-only language model (no audio features).
- General audio-language model with native audio input.
- ASR → text model pipeline.
- Numeric DSP features + transcript.
- Native Symthaea acoustic/HDC path + transcript.
- DSP + acoustic-HDC + transcript fusion.
- Shared HDC–CfC candidate from #10842.
- Each candidate with audio, DSP, transcript, and temporal-order ablations.

Match model/provider version, prompts, decoding, compute and allowed context. Include a no-context baseline and a misleading-text negative control.

## Scoring rules

Report independent dimensions:

- lexical WER/CER and exact semantic-field preservation;
- acoustic-grounded accuracy under transcript/prompt conflict;
- paired invariance/sensitivity scores (change only the affected output dimensions);
- F0/voicing errors on valid voiced reference regions, including octave-error rate;
- event category macro-F1 and onset/offset tolerance F1;
- scene-event detection;
- speaker attribution only on separately consented, annotated cohorts;
- calibration and selective-risk coverage for any probabilistic claim;
- generation style-following, intelligibility, naturalness and pairwise human preference;
- latency, memory and compute.

Do not combine these into one score. Do not infer a person's actual emotion, health, intent or protected traits from a voice. Prefer observable statements (e.g. measured pitch range, intensity, pause pattern, audible laughter) and keep any contextual interpretation tentative and non-authoritative.

## Candidate benchmark adapters

Use official versions and scorers, with splits and licenses pinned:

- DEAF — acoustic faithfulness under text/audio conflict.
- VoxParadox and LISTEN — lexical priors versus acoustic prosody.
- MMSU — broad spoken-language understanding.
- NVVSpeech — nonverbal vocal event understanding and generation.
- S2S-Arena and ParaS2SBench — speech-native content/style generation evaluation.
- MSU-Bench / MultiTalkBench — conversational multi-speaker who-says-what and speaker-relative context.

Synthetic conflict stimuli, scripted speech and naturally recorded conversations remain separate cohorts. English/Chinese results do not qualify South African languages; create a native-speaker annotated multilingual/code-switch cohort only after consent and licensing are in place.

## Entry criteria

Before building a learned model or ingesting any corpus:

1. Pin the exact benchmark release, scorer, license and split governance.
2. Reproduce at least one reference baseline.
3. Audit the existing formant extractor and placeholder duplicate API; no duplicate F0 implementation without evidence of need.
4. Freeze the counterfactual generation protocol and exact signal invariants.
5. Define native-speaker/human adjudication and blind evaluation.
6. Confirm the current #5352 program gate permits the proposed implementation stage.

## Nonclaims

This file is a design manifest only. It contains no audio labels, trained weights, benchmark scores, human annotations, validated emotion model or evidence of SOTA. It does not promote native ASR or Broca capabilities.
