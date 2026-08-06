# Label-independent acoustic comparison (2026-07-26)

**This is NOT a human blind-listening result.** The intended blind
human-transcription test (see `README.md` in this directory) has not
yet been completed by anyone:
- The project owner's own attempt was contaminated — the answer key
  (`ANSWER_KEY_do_not_open_until_after_transcribing.json`, in the same
  audio_output folder as the clips) was auto-exposed by their tooling
  before they evaluated the clips, before they could transcribe blind.
- Claude (this session's AI collaborator) has no audio-perception
  capability at all — it cannot listen to a `.wav` file and form a
  genuine subjective judgment, only analyze it numerically via code. No
  claim of a Claude "blind listening judgment" is valid; if such a claim
  appeared anywhere in this project's history, it does not originate
  from anything Claude actually did and should not be trusted.

**Protocol lesson for next time**: don't ship a plaintext,
auto-previewable answer-key file in the same folder as the test clips —
that's what made the accidental exposure easy. A future attempt should
keep the mapping out-of-band (e.g. disclosed only after the tester
explicitly confirms completion, or encoded in a way that isn't
casually glanceable by a file browser/IDE preview).

## What this file records instead
A **label-independent acoustic comparison** — computed with the labels
hidden, using validated speech-quality/intelligibility-correlated
metrics rather than the raw silence-fraction/WER proxies used earlier
in this bundle. The real audio-to-condition mapping (confirmed via the
answer key, matching what this bundle's own generation script produced)
was: Clip_A = untuned RVC, Clip_B = tuned RVC, Clip_C = DiffSinger
source.

## Results (source as reference)

| Measure | A: untuned | B: tuned |
|---|---:|---:|
| STOI similarity | 0.533 | 0.495 |
| Extended STOI (ESTOI) | 0.382 | 0.362 |
| Phrase-envelope correlation | 0.891 | 0.843 |
| Spectral-transient correlation | 0.674 | 0.621 |
| Mel-spectrum similarity | 0.9861 | 0.9839 |
| Final ~5-8s region STOI | 0.481 | 0.448 |
| Exact-zero sample fraction | 2.70% | 1.52% (source: 0.30%) |
| Energy vs. source, 3-6kHz band (consonant/fricative range) | -3.11 dB | -4.45 dB |

**Every metric agrees on direction: untuned (A) preserves the source's
articulation more faithfully than tuned (B).** Tuned reduces exact-zero
gaps (consistent with this bundle's earlier finding) but loses more
energy specifically in the 3-6kHz band where consonant/fricative cues
live — a plausible mechanism for "sounds fuller, communicates less."

**Caveat carried over from the earlier addendum**: STOI is not validated
here as an absolute intelligibility score for sung, spelled-out-letter
content either — no metric in this bundle has been validated against
actual human perception of this specific content type yet. What's
different this time is that multiple independently-computed,
established metrics converge on the same ranking, which is stronger
evidence than any single metric alone, but it is still not a substitute
for the still-outstanding human listening test.

## Revised configuration status

| Configuration | Status |
|---|---|
| DiffSinger source | Reference |
| Epoch 200, untuned (RVC defaults) | Current RVC baseline |
| Epoch 200, tuned (rms_mix_rate=1.0, index on) | **Rejected as default** — measurably worse articulation preservation on every acoustic metric tested, though it may still be preferred for smoothness/fullness (unconfirmed — no valid human preference data exists yet) |
| Canonical production configuration | **None yet** |

## Recommended follow-up (not yet started)
See the parent conversation / project history for the full 5-point plan
(intelligibility ladder for DiffSinger itself, phoneme/duration table
inspection, mel-vs-vocoder isolation test, commercially-clean singing
corpus, and an isolated B-variant RVC sweep — index and rms_mix_rate
varied independently rather than combined, plus explicit consonant
`protect` values). The core finding driving that plan: **the tuned RVC
settings may be perceptually appealing but are not the current
bottleneck to fix — the DiffSinger source itself likely has the primary
intelligibility problem**, since even the best RVC output (untuned)
only reaches STOI ~0.53 against a source that itself was only partially
transcribable by Whisper.
