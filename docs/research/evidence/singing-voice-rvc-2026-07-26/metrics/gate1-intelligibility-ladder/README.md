# Gate 1: intelligibility ladder (2026-07-26)

Per the pre-agreed decision framework (see Gate 0's README): Gate 0 ruled out
near-zero/sub-frame consonant durations but found a real, phonetically-naive
uniform-duration heuristic. Gate 1 tests whether GENEROUS, UNIFORM durations
(one syllable per note, no compression) restore intelligibility, using the
same trained checkpoint (`csd-en-poc`, step 2000) as every other render in
this bundle — no retraining. Native DiffSinger acoustic inference only, no
RVC in this test.

Two rungs of 7 phrases each, of increasing length: "me" → "sing with me" →
"won't you sing with me" → "won't you sing along with me" → "now I know my
ABC" → the full closing phrase → the full alphabet.

## v1 (flat pitch) — confound identified after the fact

`gate1_intelligibility_ladder.py` gave every phoneme a **flat, constant
220Hz** pitch contour. All 7 renders were transcribed by Whisper (`small`,
CPU) as completely unrelated to the ground truth — see
`gate1_transcribe.log`.

This result was reported, but a real methodological problem was flagged
before drawing any conclusion from it: DiffSinger's training data (CSD)
never contains a flat/monotone pitch — every training example has a real,
varying melodic contour. A flat pitch is therefore **out-of-distribution**,
and could independently degrade output quality, confounding the
duration-generosity variable this test was actually designed to isolate.
Supporting data point: this bundle's main sample (`en001a-final`, rushed
70ms consonants but a REAL melodic pitch contour) transcribed perfectly
("Won't you sing along with me?" — see `CLAIMS.md`), while v1's rung 4
(identical words, generous durations, flat pitch) did not.

## v2 (real pitch) — confound corrected

`gate1_intelligibility_ladder_v2.py` keeps the exact same generous/uniform
duration policy (150ms consonants, 350ms vowels) but replaces the flat
220Hz contour with the **real CSD ground-truth pitch** (converted from the
actual MIDI note number in `en001a.csv`) for every syllable, held constant
across that syllable's own lengthened duration. All 7 rungs are literally
verbatim sub-phrases or the full alphabet from the same song used
throughout this bundle, so every pitch value used is a real, in-training-
distribution value, not an invented melody. Transcribed the same way (same
Whisper `small`/CPU config) — see `gate1_transcribe_v2.log`.

## Side-by-side result

| Rung | Ground truth | v1 (flat 220Hz) | v2 (real pitch) |
|---|---|---|---|
| 01 | me | "Okay" | "Yeah" |
| 02 | sing with me | "Skateboarding" | "See you next week, my dear friend!" |
| 03 | won't you sing with me | "All my life speaking my mind" | "All the days you see in my view" |
| 04 | won't you sing along with me | "Won't you please stay with me for a long time?" | "All my life's just in the moment **with me**" |
| 05 | now I know my ABC | "Well, I don't know why you need to see me." | "Why are you always here?" |
| 06 | now I know my ABC won't you sing along with me | "I don't know, but I think it's me, Mom, I guess you should've known, but it's me." | "I don't know why, she told me to see her love with me." |
| 07 | A B C ... Y and Z | "I hate this movie, it's just pretty good..." (repeats 3x) | "I think it's the easiest way to take your hand..." |

## Honest interpretation

**Fixing the pitch-flatness confound did not restore intelligibility.**
Both versions are total misses on all 7 rungs — same qualitative failure
mode in both (fluent-sounding, grammatically well-formed English that bears
no relation to the ground truth; classic Whisper hallucination-on-garbage
behavior). The one partial curiosity — v2 rung 4 ending in "with me",
matching the ground truth's last two words — is not a broader pattern (v1
rung 4 and v2 rung 3 do not show this) and should be read as coincidental
hallucination, not evidence of partial recognition.

This is a real, useful negative result: it does not support "flat/unnatural
pitch was masking a genuine duration-driven improvement." Generous,
phonetically-sane durations plus real, in-distribution pitch **still**
produce output Whisper cannot transcribe, even for the single-syllable
rung ("me"). Combined with Gate 0's finding (no near-zero durations, most
consonants acoustically plausible), this shifts weight away from "duration
heuristic is the bottleneck" and toward the next item in the pre-agreed
framework: **Gate 2, mel-spectrogram-vs-vocoder isolation** — distinguishing
whether the acoustic model's own output is fundamentally deficient for
intelligibility, or whether the vocoder step is where information is lost.

As with every ASR-based check in this bundle: Whisper is **not validated
for sung, especially spelled-letter, content** (this project's own prior
SING-3–SING-6a investigation found ~100% WER on an unrelated singing
pipeline in this same style of test). Total failure on both v1 and v2 is
a strong signal given that magnitude, but is not a substitute for a human
blind-listening test, which remains outstanding (protocol at
`symthaea/audio_output/blinded_listening_test_2026-07-26/`).

## Files

- `gate1_intelligibility_ladder.py` / `gate1_intelligibility_ladder_v2.py` — `.ds` generators (flat vs. real pitch)
- `gate1_transcribe.py` / `gate1_transcribe_v2.py` — Whisper transcription scripts
- `gate1_transcribe.log` / `gate1_transcribe_v2.log` — raw transcription output
- Audio: `symthaea/audio_output/gate1_intelligibility_ladder_2026-07-26/{v1_flat_pitch,v2_real_pitch}/*.wav`
