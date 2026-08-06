# Backbone smoke test: Kokoro + WORLD-vocoder + real forced alignment (2026-07-28)

Per `SYMTHAEA_VOCAL_BACKBONE_SURVEY_2026-07-28.md`'s revised recommendation
(the only candidate left that needs no new training data/corpus). Tests
two specific, mechanistically-justified fixes to the already-known
`src/voice/kokoro_singing.rs` phase-vocoder failure (WER ~1.0-1.1,
described by the user as "random fart noises"):

1. **Real forced alignment** (torchaudio's MMS_FA model via
   `ctc_forced_aligner.get_word_stamps`) instead of an energy heuristic
   for word boundaries.
2. **WORLD-vocoder** (pyworld: `harvest` F0 + `cheaptrick` spectral
   envelope + `d4c` aperiodicity, frame-domain resampling for duration,
   direct F0 replacement for pitch) instead of an STFT phase vocoder,
   which avoids the overlap-add breakdown hypothesized to have caused the
   prior catastrophic failure.

## Pipeline

`01_kokoro_render.py` (voice-conversion venv): synthesize each phrase as
plain spoken `af_heart` Kokoro TTS + write a transcript.
`02_align.py` (ctc-align venv, `source env.sh` for the NixOS libstdc++
fix): real forced alignment of the KNOWN transcript against Kokoro's OWN
audio via MMS_FA -- a well-posed task (clean synthetic speech + known
text), unlike the earlier failed CTC-on-real-sung-audio attempt.
`03_reshape_pyworld.py` (diffsinger venv, already has pyworld): WORLD
analysis, per-word frame-domain retiming (natural duration x1.2, floor
0.35s) + repitching (every originally-voiced frame set to the word's
target note frequency; unvoiced frames left at 0), resynthesis.
`04_evaluate.py` (voice-conversion/whisper-eval venv, `pyworld`+
`soundfile` added): Whisper `base`/int8 WER (same method as every other
gate in this arc) on both the spoken and sung renders, plus a melody-
tracking check (median cents-error to the nearest target note, fraction
of voiced frames within 50 cents of a target note, observed vs target
pitch range in semitones).

Test phrases: `hello_world` and `sun_rises` are byte-identical to the
phrases used in the original `kokoro_singing` phase-vocoder gate
(`SINGING_INTELLIGIBILITY_GATE_KOKORO_V2_PHASELOCK_2026-07-22.json`),
enabling a direct, apples-to-apples comparison. `quiet_morning` is new.

## Result

| Phrase | Prior phase-vocoder WER | This attempt's sung WER | Melody tracking (median cents err / frac within 50c) |
|---|---|---|---|
| hello world | 1.5 | **0.0** | 4.5c / 0.825 |
| the sun rises over the valley | 0.667 | **0.0** | 6.5c / 0.836 |
| a quiet morning walk | (not tested before) | 0.25 (single "a"->"the" substitution) | 4.6c / 0.911 |

Overall sung WER: **0.083** (vs. the prior attempt's overall 1.03-1.1).
Full transcripts, per-phrase numbers: `results.json`.

A brief follow-up check on `hello_world_sung.wav`'s own independently
re-estimated F0 (not the internal array fed to synthesis) found 19/143
voiced frames (13%) sit more than 100 cents from the nearest target note,
clustered in the 164-368 Hz transition zone between the two target
pitches (261.63/392.00 Hz) -- consistent with brief onset/transition
artifacts at word and gap boundaries, not a broken pitch-setting
mechanism (the bulk of frames land at a median 4.5-6.5 cents, i.e.
essentially in tune).

## What this does and doesn't support

**Supported**: the two specific fixes (real alignment, WORLD vocoder)
produce a categorically different, much better result than the prior
phase-vocoder attempt on directly comparable phrases -- both
intelligibility (WER) and objective pitch-tracking are strong. This is a
real, meaningful positive result, not a marginal one.

**Not yet supported**:
- **No human listening check.** Every gate in this arc has held Whisper
  WER as a proxy, not a perceptual-quality substitute -- this result is
  no exception. Whether the audio actually sounds like acceptable
  singing (vs. correctly-pitched but otherwise odd-sounding recitation)
  is unverified. Claude has no audio-perception capability and makes no
  listening claim here.
- **n=3 phrases, single trial each, no seed variation.** A small, bounded
  pilot, not a statistically powered result.
- **Duration stretch was modest** (natural x1.2, floor 0.35s) -- this is
  closer to "deliberate, evenly-paced recitation" than full legato
  singing with sustained held notes; whether the same mechanism holds up
  under more aggressive note-holding (as `VocalPerformancePlan` would
  need for melismas/long sustained notes) is untested.
- **Word-level, not phoneme-level, alignment/reshaping.** Simpler than
  `VocalPerformancePlan`'s per-phoneme model; whether within-word
  consonant/vowel timing (`singing_bridge.rs`'s natural-duration formula)
  layers cleanly on top of this word-level scaffold is a follow-up, not
  yet attempted.
- **One voice** (`af_heart`), **one phrase register** (short, simple
  melodies, single-octave-ish range) -- generalization to wider melodic
  range, faster phrases, or different voices is untested.

## Revision (2026-07-28, same day): independent review found and fixed 3 real bugs

An external reviewer independently analyzed the raw v1 output files
(without running any code) and reported three specific, checkable
findings. All three were verified directly against the actual sample
data before acting on them, then fixed:

1. **10.5-13.1 dB loudness increase vs. the spoken source, all outputs
   peaking at exactly 0.92** — confirmed exactly (v1 blind-peak-normalized
   every output to a fixed 0.92 regardless of source loudness). Fixed:
   RMS-match the output to the spoken source's own RMS instead, with a
   0.98 peak safety cap.
2. **Severe boundary discontinuities** ("large sample discontinuities...
   approximately 0.63 and 0.46 full-scale amplitude" near word
   boundaries) — confirmed exactly (`sun_rises_sung.wav` had a 0.629
   full-scale sample jump at 0.428s, right at a word boundary). **Root
   cause**: v1 fed `f0=0` "gap" frames through the same `pw.synthesize()`
   call as real word content — in WORLD, `f0=0` means UNVOICED (noise-
   excited synthesis using whatever spectral envelope is supplied), not
   silence, so the gap frames were synthesizing a spurious noise burst
   (shaped by a copied previous-word spectral envelope) discontinuous
   with real neighboring audio. Fixed: each word is now synthesized in
   complete isolation (its own `pw.synthesize()` call), concatenated with
   genuine time-domain silence (actual zero samples) for gaps, with an
   8ms edge fade. Result: the worst jump dropped from 0.629 to 0.173 (a
   ~3.6x reduction), now in a range consistent with ordinary speech
   transients rather than a synthesis artifact.
3. **Word-level (not phoneme-aware) retiming, flagged as the largest
   structural limitation** — correct characterization of v1's design.
   Fixed: each word's phonemes (from Kokoro's OWN misaki G2P output,
   checked directly rather than assumed — Kokoro exposes the phoneme
   *symbol sequence* per the reviewer's suggestion, but not per-phoneme
   *durations*, so sub-word timing remains a proportional estimate, not
   true forced sub-alignment, disclosed not hidden) are classified as
   vowel/consonant; consonants keep a fixed ~60ms nominal duration in
   both the natural and target timeline, vowels absorb the entire
   difference — directly generalizing `singing_bridge.rs`'s established
   "consonants stay brief, vowels absorb the stretch" rule from one
   vowel per syllable to N vowels per word (evenly split).

**The reviewer's specific "~50% duration compression" reading of
`hello_world` (1.525s spoken -> 0.785s sung) was checked and found to be
a file-level artifact, not a per-word compression**: the spoken WAV
includes ~0.9s of Kokoro's natural lead-in/trail-out silence around only
~0.62s of actual word content (per the MMS_FA alignment), while the sung
output has no added silence padding at all — the actual per-word target
durations (`STRETCH=1.2`, floor 0.35s) *lengthen* the words slightly
(0.735s of word content vs. ~0.62s natural), consistent with singing
normally holding notes longer than speech. Total-file duration and
per-word duration were being conflated; corrected here rather than left
standing.

**Re-verification after all three fixes**: WER unchanged (overall sung
WER 0.083, same "a"->"the" substitution as before — the fixes did not
regress intelligibility) and melody-tracking held or slightly improved
(median cents error 3.4-4.3, vs. 4.5-6.5 before; 88.8-90.7% of frames
within 50 cents, vs. 82.5-91.1% before). Updated `results.json`/scripts
in this directory are the v2 (post-fix) versions; the numbers/audio
referenced above as "before" are preserved in this README and in git
history (commit `1ea84f7958`), not silently overwritten.

**Still not addressed** (reviewer's own longer-term roadmap, not
attempted this pass): natural pitch micro-variation/vibrato/portamento
(current F0 is a hard, exact target — "too quantized," per the review,
a fair characterization); the wider observed-vs-target pitch range
(13.7-21.5 semitones observed vs. 7-12 target) remains a real, disclosed
open question, most plausibly transient onset-frame estimation noise at
word boundaries (checked in the v1 pass, not rechecked here); a genuine
human listening check (still the real tiebreaker, not run).

## Next steps if this line continues

A genuine human listening check is the real tiebreaker (per this arc's
standing practice) before treating this as a viable exact-score backbone
candidate. If that holds up: scale to Gate D's phrase-category set (for
direct comparability with ACE-Step v1's known capability boundary),
per-phoneme (not per-word) alignment/reshaping to match
`VocalPerformancePlan`'s full schema, and a wider melodic range/duration
stretch test.
