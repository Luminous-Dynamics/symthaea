# Singing-voice listening pack — 2026-07-28

> **Where the audio is**: `symthaea/audio_output/LISTENING_PACK_2026-07-28/` on
> this machine. That directory is gitignored, so the 16 WAVs plus the blinded
> copies are local-only and deliberately not committed (~9 MB of binaries).
> This index is the tracked record; §Provenance plus the seeded shuffle
> (`20260728`) make the pack rebuildable.


**Why this exists**: every quality claim in the singing-voice arc rests on
Whisper WER (an ASR proxy) plus F0 measurements. Both are now measured
carefully — see `SYMTHAEA_VOCAL_IMPROVEMENT_PLAN_2026-07-28.md` F8/F8c — and
both say the same thing: *the words are recoverable and the notes are exact.*

Neither can tell us whether it **sounds like singing**. That is the one open
question no further measurement will answer, and it gates all expressive work
(Phase 4/5).

**Audio duration: 54.8 seconds** across 16 clips (0.81s–14.95s; all but one are
under 3s). **Expected evaluation time ~10–15 minutes** with repeats and writing
— the two are different numbers and an earlier version of this index conflated
them, and also wrongly claimed no clip exceeded ~3s.

---

## Do this in two passes. Blind first.

### → Pass 1: `01_BLIND_PASS/`

Start here. 16 shuffled clips with neutral names, no system labels, no expected
text, no metric values. Fill in `01_BLIND_PASS/RESPONSE_SHEET.md`:
**transcribe exactly what you hear** for each clip, then rate singing/quality.

The transcription is the highest-value field in this whole pack. It only works
if you write it before knowing the target — a corrected guess is not data.

### → Pass 2: `02_KEY_DO_NOT_OPEN_UNTIL_JUDGED/`

Only after the sheet is filled. Contains the clip→source mapping, the true
target text, what Whisper heard, a partial-recovery scoring method (not WER —
at 2–7 words per phrase WER has ~1-word resolution), the decision routing
table, and the objective signal analysis.

The labelled folders `A_*`–`E_*` are the same 16 files with descriptive names.
They're the pass-2 view; **an earlier version of this pack shipped only those**,
which primed the listener with `current-best`, `OLD-phase-vocoder`, `WER-1.00`
and "random fart noises" — exactly what a first pass must not see.

---

## What the groups are (safe to read — no per-clip mapping)

| | |
|---|---|
| **A** | Three phrases, each as spoken TTS source + its sung render — the current best output of the arc |
| **B** | The same three through the old STFT phase vocoder — known-bad anchor |
| **C** | ACE-Step v1, a real generative singer that can't be told what notes to sing — known-good anchor |
| **D** | The hard suite: melody accuracy is 2–4 cents on *all* of these while WER runs 0.00 → 1.00 |
| **E** | One phrase through v8 vs v9 — the unexplained regression |

## The two results that actually matter

Everything else is largely settled by signal analysis already (see the key).
The consequential unknowns are **D3** and **D4**:

- **D3** (`moon over the blue lagoon`, WER 0.80) is the strongest ASR-artifact
  test. Understandable to you but not Whisper → WER is underestimating the
  backbone. Unintelligible to you too → vowel identity is being lost during
  sustain.
- **D4** (`a b c d e f g`, WER 1.00, Whisper heard "Oh, Lucy!") — signal
  analysis shows it has the *highest* high-frequency and transient energy in
  the suite, so this is **not** a missing-consonant-energy failure. If it's
  unintelligible to a human too, the priority is **segmentation and timing**,
  not more fricative preservation.

Together they decide whether the project has been overestimating a consonant
problem inferred from Whisper failures, or whether exact pitch has been
achieved while phone identity and timing remain genuinely inadequate.

**Stop condition**: if A doesn't sound like singing at all, WER 0.083 and
1.2-cent pitch accuracy are both true and both beside the point — reconsider
the backbone before spending anything on expression.

Record the verdict in
`symthaea/docs/research/evidence/singing-voice-rvc-2026-07-26/`. A negative
answer is as useful as a positive one, and far more useful than another proxy
metric.

---

## Provenance

Nothing was generated for this pack; all 16 files are recorded outputs already
under measurement. Blind shuffle is seeded (`20260728`) and reproducible.

- A: `/var/lib/symthaea/training-runs/kokoro-world-vocoder/audio/`
- B: `symthaea/audio_output/kokoro_singing_2026-07-22-offsetfix-full/`
- C: `symthaea/audio_output/ace_step_verification_2026-07-27/`
- D, E: `/var/lib/symthaea/training-runs/kokoro-world-vocoder/gate2_audio/`
