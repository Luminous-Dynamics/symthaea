# Kokoro native duration check (2026-07-28)

Per the reviewer's explicit recommendation after v9's negative result:
"before adding another model, inspect whether the Kokoro inference path
exposes... token-level durations... duration-predictor output... frame-
to-token mapping. If native durations exist, they may be the most
internally consistent source of phone boundaries. They still need
validation against the waveform."

## Finding 1: Kokoro DOES expose exact per-phoneme-character durations, for free

`kokoro.pipeline.KPipeline`'s underlying `KModel.forward_with_tokens`
already computes `pred_dur`: a per-input-token frame-count from the
model's own duration predictor (`kokoro/model.py:107-110`), used via
`torch.repeat_interleave` to literally construct the frame sequence fed
to the decoder. This is not an external estimate to be reconciled with
the audio -- it IS the exact generative process that produced the audio,
returned as `KPipeline.Result.pred_dur` from the SAME single call that
renders a phrase (confirmed: exactly 1 `Result` per short test phrase,
`len(pred_dur) == len(phonemes) + 2` [bos, phonemes..., eos], `sum(pred_dur)
* 600 == len(audio)` exactly, verified on `fricative_heavy`
["she sells seashells by the seashore"], 95 total frames * 600 =
57000 samples = the exact rendered length).

**Our existing `01_kokoro_render.py` discards this entirely** -- it
iterates the pipeline generator as `for _, _, chunk in gen`, using only
the 3rd (audio) element and throwing away `pred_dur`/tokens. Re-running
render while capturing `result.pred_dur` alongside `result.phonemes`
gives an EXACT per-phoneme-character frame boundary table for free, no
extra model or inference call needed.

The frame->sample conversion is exact: 1 pred_dur frame = 600 samples
at 24kHz (25ms hop), confirmed via `sum(pred_dur) * 600 == len(audio)`.

## Finding 2 (the important one): native durations FAIL waveform validation, systematically

Built the full per-phoneme-character boundary table for `fricative_heavy`
(cumulative-sum of `pred_dur`, skipping bos, one entry per `ps` character
including stress marks) and checked it against the spoken audio's
high-band (>=3kHz) energy fraction -- the same acoustic ground-truth
measurement used to root-cause the v9 regression.

**Every single /s/ and /S/ (ʃ) token in the phrase shows the SAME
systematic misalignment**: high-frequency frication energy is
concentrated in the ~50ms window immediately BEFORE the token's own
pred_dur-designated span, not within it.

| Token | Own pred_dur span: hf_frac | Preceding ~50ms: hf_frac |
|---|---|---|
| "she" /S/ | 0.155 | **0.814** |
| "sells" /s/ | 0.144 | **0.900** |
| "seashells" /s/ (sea's s) | 0.139 | **0.920** |
| "seashells" /S/ (shells' sh) | 0.100 | **0.794** |
| "seashore" /s/ (sea's s) | 0.135 | **0.932** |
| "seashore" /S/ (shore's sh) | 0.047 | **0.705** |

All 6 of 6 fricative tokens in this phrase show the identical pattern
(own span low-frequency-dominated, preceding window high-frequency-
dominated) -- not noise, a consistent, mechanistically explicable skew.
For "seashells"' /S/ specifically, this exactly reproduces (from an
independent measurement path) the v9 root-cause finding: the true
frication is absorbed into the PRECEDING vowel/consonant token's nominal
span, not the fricative token's own span.

**Plausible mechanism (not yet confirmed further)**: Kokoro's duration
predictor is a non-autoregressive regression head trained end-to-end
with the decoder; there is no requirement that a given token's assigned
frame-count line up with the acoustic content a listener would attribute
to that phoneme -- coarticulation smearing in the decoder can plausibly
shift a fricative's noise earlier than its own nominal token slot,
consistently, if the model learned to "front-load" upcoming frication
during the preceding segment's frames. This is a property of the model's
learned alignment, not a bug in our own duration-allocation code.

## What this changes about the recommended next step

The reviewer's own conditional applies exactly as stated: native
durations exist, are free, and are mechanistically exact -- but they
**fail validation against the waveform**, in the same specific way our
own heuristic did, and by a consistent, roughly one-token-wide (~50ms)
margin across every fricative tested here. Two live options, neither
attempted yet:

1. **A systematic offset correction** on top of `pred_dur`: since the
   skew looks consistent in direction and rough magnitude (~1 token's
   duration early), a fixed backward-shift correction for consonant
   (esp. fricative) token boundaries might recover most of the benefit
   pred_dur offers (free, exact, no external model) without needing a
   full forced-aligner. Cheap to test against this same 6-token dataset
   before building anything further.
2. **Real external forced alignment** (MMS_FA, already proven to work
   well for word-level boundaries in stage 2) at the phone level instead
   of word level -- the reviewer's originally proposed path, unaffected
   by this finding, still available as a fallback if (1) doesn't hold up
   across more phrases/phoneme classes.

Neither has been decided or implemented yet -- reporting this finding
before choosing, since it materially changes what "check Kokoro first"
should conclude (a qualified yes, not a clean yes).

## Not yet done

- Testing whether the ~50ms/1-token skew is consistent across other
  phoneme classes (stops, affricates, other fricatives /f/,/θ/) and
  other phrases, or specific to /s/,/ʃ/ in this one phrase.
- Deciding between a systematic-offset correction and full forced
  alignment.
- Any new rendering using either approach.
- The human listening check -- still the standing, most important item.

## Files

- No new scripts committed yet -- this was an interactive investigation
  using the existing `fricative_heavy_spoken.wav` (already in the
  v9 bundle's referenced audio) and a direct `KPipeline` call capturing
  `Result.phonemes`/`Result.pred_dur`. A reusable extraction script will
  be written once the offset-vs-forced-alignment decision is made, to
  avoid building throwaway tooling twice.

## Update (2026-07-28): extended across phoneme classes -- indexing bug ruled out, offset is real but too context-dependent to hand-correct

Per the reviewer's explicit next step: rule out an indexing bug first,
then extend the audit across phone classes/positions with class-
appropriate landmarks, before deciding between a correction model and
a full forced aligner.

### Indexing bug: ruled out

Checked all 6 test phrases (`fricative_heavy`, `consonant_clusters`,
`phrase_final_stops`, `repeated_syllables`, `long_sustained_vowels`,
`semantically_unusual`): every character of `ps` (including stress
marks and spaces) maps to a real vocab entry (`model.vocab.get(c) is
not None` for all c, 0 filtered in every phrase -- `KModel.forward`'s
`list(filter(...))` never drops anything here), and `len(pred_dur) ==
len(ps) + 2` exactly in every case. `ps[i]` <-> `pred_dur[i+1]` is a
correct, verified 1:1 mapping, not an off-by-one artifact.

**Determinism check**: `pred_dur` is bit-identical across repeated
calls with the same text/voice (verified). The audio waveform itself
has small floating-point-level nondeterminism between calls (max abs
sample diff ~0.08-0.10, correlation 0.996 against the canonical stage-1
`_spoken.wav` file) -- almost certainly multi-threaded floating-point
reduction-order noise in the vocoder, not a different rendering, and
irrelevant to the boundary analysis since `pred_dur` (what defines
nominal boundaries) doesn't move at all.

**Suggestive prior art found while reading the code**: Kokoro's own
`join_timestamps` (used for its word-level timestamp feature) already
applies an undocumented `-3 frame` correction to the leading `<bos>`
token specifically (`left = right = 2 * max(0, pred_dur[0] - 3)`, with
its own `# TODO: Is -3 an appropriate offset?` comment) -- the model's
authors already know raw `pred_dur` needs adjustment near a boundary,
though this existing fix doesn't cover interior token skew.

### Class-specific landmarks, across 6 phrases, ~90 phone tokens

Built class-appropriate detectors: fricatives via high-band (>=3kHz)
threshold-crossing (independently validated by the original 6-token
study); stops via steepest-RMS-rise (burst) location, architecturally
clean since a closure-to-burst transition is a genuine sharp transient
regardless of context; vowels/sonorants via periodicity-onset
threshold-crossing; voiced obstruents via a crude local-RMS-minimum
proxy (explicitly disclosed as the roughest of the five, not to be
trusted); affricates got only 1 data point in this phrase set
(insufficient n).

**Self-check before trusting the vowel/sonorant number**: periodicity-
onset could spuriously fire early if the PRECEDING phone is already
voiced (no real "onset" to detect). Split vowel/sonorant offsets by
whether the preceding phone was voiceless (clean, unambiguous onset) vs.
voiced (potentially corrupted): **-53.9ms (n=16, clean cases) vs.
-50.5ms (n=41, other cases) -- statistically indistinguishable.** This
rules out the artifact concern; the early-voicing effect is real, not
a detector bug.

| Class (trustworthy detector only) | n | mean offset (ms) | std (ms) |
|---|---|---|---|
| Fricative (hf_frac threshold) | 12 | **-41.3** | 54.8 |
| Stop (RMS burst) | 12 | **+2.5** | 43.1 |
| Vowel/sonorant (periodicity onset) | 57 | **-51.5** (combined clean+other) | ~34 |

(Negative = acoustic landmark occurs BEFORE Kokoro's own nominal
pred_dur boundary.)

### Revised picture: not "fricative coarticulation" -- continuant vs. transient

The pattern is broader than the original 6-fricative finding suggested:
**sustained/continuant material (frication noise, voicing) is
consistently realized ~40-55ms before its nominal pred_dur boundary,
while stop BURSTS -- a sharp, unavoidable acoustic transient -- show no
systematic bias (mean near zero).** Plausible mechanism (not confirmed
further, no literature check done): a duration predictor trained
end-to-end with the decoder has no acoustic pressure to precisely time
a phone boundary where the signal is smoothly continuous on both sides,
but IS pressured to place a stop burst correctly (misplacing it would
be an audible timing artifact) -- so continuants get to "drift," bursts
don't.

### Decision: too context-dependent for a hand correction; use as an aligner prior instead

Applying the reviewer's own decision tree: **std is comparable to (or
larger than, for fricatives) the mean effect size** -- fricative offsets
ranged from -87.5ms to **+102.5ms** (one token, "sells"' /s/, moved the
*opposite* direction), and vowel/sonorant std (~34-37ms) is not small
relative to its ~-51ms mean either. This is the reviewer's "errors vary
strongly by context or word" branch, not the "stable class-specific
offset" branch -- a single global or per-class fixed-ms correction is
not well-supported by this data; it would help on average but get
individual tokens wrong in either direction, including some by a lot.

**Recommendation, following the reviewer's own prescribed branch**:
native durations are not accurate enough to use as the sole extraction
source, but are valuable as a free, exact-by-construction PRIOR for a
real forced aligner -- narrowing its search window and providing an
independent discrepancy signal (a forced-alignment result that disagrees
with `pred_dur` by far more than this audit's typical spread would be
worth flagging as suspect). Proceed to the gold-set-validated forced
aligner next, per the original plan, using `pred_dur` as an
initialization rather than building a standalone correction model.

### Not yet done

- Affricates and voiced obstruents remain under-measured (n=1 and a
  disclosed-unreliable proxy respectively) -- would need dedicated
  phrases/detectors if this line of investigation continues.
- No check of whether the continuant-early-realization effect holds at
  other speeds/voices (only `af_heart`, speed=1.0 tested).
- Building the gold-set-validated forced aligner itself -- not started.
- The human listening check -- still the standing, most important item.

### Files

- `11_native_duration_class_audit.py` -- the audit script (indexing
  check, per-class landmark detectors, signed-offset aggregation).
- `native_duration_class_audit.json` -- raw per-token records across
  all 6 phrases.
