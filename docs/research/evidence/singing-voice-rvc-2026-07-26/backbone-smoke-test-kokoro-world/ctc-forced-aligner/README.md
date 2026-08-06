# Gold-set-oriented CTC forced aligner (2026-07-28)

Per the reviewer's decision after the native-duration audit: `pred_dur`
is a free, deterministic, correctly-indexed prior, but too context-
dependent (per-token variance comparable to or exceeding the mean skew
in every phone class) to hand-correct. Build a real forced aligner
behind a replaceable interface, use `pred_dur` as a soft prior/search-
window (never a hard boundary), and validate before touching synthesis.

## What was built

`phone_aligner.py`: a `PhoneAligner` Protocol (`align(waveform,
sample_rate, expected_phones, prior_spans) -> AlignmentResult`), plus:

- `NativeDurationPrior` -- wraps `pred_dur` spans directly. Every span
  carries a standing `"NOT ACOUSTICALLY VALIDATED"` warning so it can
  never be silently mistaken for a real alignment.
- `CtcPhoneAligner` -- real acoustic forced alignment via
  `facebook/wav2vec2-lv-60-espeak-cv-ft` (a genuine IPA-phoneme-output
  CTC model, 392-phone vocabulary, confirmed network-downloadable) +
  `torchaudio.functional.forced_align` (the standard PyTorch CTC
  forced-alignment primitive -- no external `ctc_forced_aligner` package
  reused, since the installed version (1.0.2) turned out to be a fixed
  31-character English-grapheme-only ONNX aligner, not phone-aware, on
  inspection before use).

`misaki_to_espeak.py`: a phone transducer from misaki's phoneme
characters to this model's vocabulary, built from a VERIFIED character
inventory (every character misaki actually emitted across all 10 Gate-2
phrases, not assumed from external IPA docs) -- diphthong codes (`I`,
`A`, `O`, `W`, `Y`) map to the model's own 2-character diphthong tokens
(`aɪ`, `eɪ`, `oʊ`, `aʊ`, `ɔɪ`, all present in its vocab); misaki's `ɡ`
(IPA U+0261) matches the model's token exactly, no ASCII-g conversion
needed (verified, not assumed). Untested extras (covering codebase
phoneme classes not exercised by this specific phrase set) are marked
explicitly, not silently assumed correct.

**Environment note**: the model's own `AutoProcessor` requires a system
`espeak`/`espeak-ng` binary for its (unused here) text-to-phoneme
backend and fails without it; bypassed by loading
`Wav2Vec2ForCTC`/`Wav2Vec2FeatureExtractor` directly plus the raw
`vocab.json`, since misaki phonemes are already available and the
model's own phonemizer is never needed.

## Rule-outs before trusting any result

- **Indexing**: `phone_order_ok=True` and `len(ctc_spans) ==
  len(expected_phones)` exactly for all 6 test phrases (21/21, 25/25,
  19/19, 12/12, 17/17, 17/17), zero aligner warnings -- confirms
  `torchaudio.functional.forced_align`'s guarantee (the k-th non-blank
  collapsed segment corresponds exactly to the k-th target token) held
  in practice, not just in theory.
- **Confidence carries real signal**: correlation between CTC confidence
  and `|native_vs_ctc discrepancy|` across all 111 tokens = **-0.42** --
  a genuine, meaningful negative correlation (low confidence really does
  predict bigger disagreement with the native prior), not noise.
  Bucketed: confidence<0.1 (n=14) mean discrepancy 67.5ms; 0.1-0.7 (n=11)
  58.6ms; >=0.7 (n=86) 41.2ms.
- **Low confidence is concentrated almost entirely in vowels/sonorants**:
  all 14 sub-0.1-confidence tokens are vowels (11) or sonorants (3) --
  zero fricatives, stops, voiced obstruents, or affricates fell below
  0.1 confidence. This is favorable for the actual engineering goal
  (consonant-boundary placement for singing synthesis) but means vowel
  timing from this aligner should not yet be trusted.

## Result: real but partial improvement -- reduced variance, not reduced bias

| Class | CTC mean (ms) | CTC std | Native mean (ms) | Native std |
|---|---|---|---|---|
| Fricative | -40.0 | **19.9** | -41.3 | 54.8 |
| Vowel | -44.5 | 31.0 | -51.6 | 37.8 |
| Sonorant | -45.8 | 29.5 | -50.1 | 31.3 |
| Stop | **+17.0** | 36.7 | +2.7 | 43.1 |

(mean/std of landmark-detector offset from the phone's own nominal
start; native numbers reproduce the earlier native-duration-class audit
almost exactly, confirming both analyses used the same landmark code
correctly.)

**Fricatives**: CTC's mean bias is essentially IDENTICAL to native's
(-40.0 vs -41.3ms) but its variance is dramatically lower (19.9 vs
54.8) -- CTC did NOT eliminate the ~40ms early-frication-realization
skew found in the native-duration audit, but it IS far more consistent
about it. Two explanations, not distinguished by this data: (a) the
skew is a genuine acoustic property (frication really is realized ~40ms
before whatever mechanism assigns it a phone-boundary label, and two
independent mechanisms -- Kokoro's duration predictor and this separate
wav2vec2 CTC model -- both reflect that same reality), or (b) both
mechanisms share a similar "late emission" bias for continuant material
(a documented tendency of CTC-trained acoustic models generally, and
plausibly of duration-predictor-based TTS too). Not resolved here.

**Vowels/sonorants**: modest improvement in both mean and variance, but
undermined by the confidence problem above -- many of the largest
individual discrepancies (e.g. `-105.8ms`, `-102.0ms`, `-98.7ms`) occur
on tokens with confidence <0.01, meaning the aligner itself is signaling
it doesn't trust these specific placements.

**Stops**: a real, honest negative -- CTC's mean bias (+17.0ms) is
WORSE than native's (+2.7ms), though variance improved slightly
(36.7 vs 43.1). Plausible explanation (not confirmed further): the
landmark detector defines a stop's "onset" as the burst, but a CTC
aligner may be placing the phone boundary at closure onset (the silence
before the burst) instead -- a different, also-defensible definition of
"where the stop begins" that this landmark comparison isn't designed to
distinguish from a real placement error.

## Against the pre-registered pass gate: not yet cleared

The reviewer's gate requires the aligner to (among other things) "beat
native... boundaries," "improve vowel-onset timing," and "localize stop
bursts" before entering synthesis. On this evidence: fricatives improved
in consistency but not bias; vowels improved modestly but with a real,
disclosed confidence problem; stops got WORSE by this landmark's
definition. **This is a mixed result, not a clean pass** -- proceeding
to the 4-arm synthesis matrix now would risk repeating the exact mistake
v9 was a lesson in (a locally-measured acoustic improvement that doesn't
translate to a real synthesis win). Recommend investigating the
persistent fricative bias and the stop-definition question before
trusting these spans in a renderer.

## Not yet done

- Manual/human gold-set annotation in the sense the reviewer specified
  ("manually validating the important boundaries") -- what exists here
  is Claude's best available substitute (the same rigorous, previously
  cross-validated automated acoustic-landmark methodology used
  throughout this arc), not human perceptual judgment, which Claude
  cannot provide. Disclosed, not glossed over.
- Affricates (n=1) and voiced obstruents (landmark not yet defined for
  this class beyond the earlier crude RMS-dip proxy) remain under-
  measured relative to fricatives/stops/vowels/sonorants.
- Distinguishing "genuine acoustic property" from "shared CTC/duration-
  predictor late-emission bias" for the persistent ~40ms fricative skew.
- Resolving the stop-boundary-definition question (burst vs. closure
  onset).
- The 4-arm synthesis matrix (A/B/C/D) -- correctly not yet attempted,
  since the pass gate isn't cleared.
- The human listening check -- still the standing, most important item.

## Files

- `phone_aligner.py` -- `PhoneAligner` Protocol, `NativeDurationPrior`,
  `CtcPhoneAligner`.
- `misaki_to_espeak.py` -- the phone transducer.
- `12_validate_ctc_aligner.py` -- validation harness (reuses the
  native-duration-class audit's exact landmark code for direct
  comparability).
- `ctc_aligner_validation.json` -- raw per-token records, 111 tokens
  across 6 phrases.
