# Blind listening pack: Arm B vs. Vocos v12 vs. spoken Kokoro — 2026-07-29

**Why this exists**: Step 4 of the singing-voice naturalness plan
(`/home/tstoltz/.claude/plans/synthetic-tumbling-raccoon.md`) swapped WORLD's
`pw.synthesize()` for Vocos (`charactr/vocos-mel-24khz`) as a post-hoc
analysis-resynthesis pass over Arm B's existing render. The automated
result was genuinely mixed and self-contradictory: DNSMOS improved more
than any other experiment in this whole arc (+0.334 over Arm B, closing
21.8% of the gap to spoken), but WER got worse (0.284→0.385) and UTMOS
moved the *opposite* direction from DNSMOS (-0.181). Two naturalness
proxies disagreeing on direction, plus an intelligibility regression, means
neither automated number can settle whether Vocos is actually an
improvement — this pack exists to answer that with a real listener instead
of trusting either metric.

**No renderer changes were made to build this pack.** It only copies,
peak-normalizes, and shuffles already-rendered audio.

## Do this in two passes. Blind first.

### → Pass 1: `01_BLIND_PASS/`

Start here. 15 shuffled clips with neutral names, no system labels, no
expected text, no metric values. Fill in
`01_BLIND_PASS/RESPONSE_SHEET.md`: **transcribe exactly what you hear**
for each clip first, then rate intelligibility and naturalness
*separately* — the whole point of this pack is to find out whether those
two move together or apart for Vocos, the same way DNSMOS and UTMOS just
did.

### → Pass 2: `02_KEY_DO_NOT_OPEN_UNTIL_JUDGED/`

Only after the sheet is filled. Contains the clip→(phrase, condition)
mapping, the true target text for each phrase, and the peak-normalization
gain applied to each file.

## What the three conditions are (safe to read — no per-clip mapping)

| Code | What it is |
|---|---|
| **B** | Arm B baseline — the existing WER-winning render (event-informed masking, WORLD vocoder, no pitch naturalization) |
| **V** | v12 — Arm B's waveform resynthesized through Vocos (`charactr/vocos-mel-24khz`)'s own mel-extract-and-decode, F0/duration untouched |
| **K** | spoken Kokoro TTS — a quality anchor, not a singing candidate. Expected to sound clearly better and clearly not sung; not informative about B vs. V, included so you have a "what does clean audio from this same voice sound like" reference point |

5 phrases × 3 conditions = 15 clips: `positive_control`,
`fricative_heavy`, `consonant_clusters`, `short_unstressed`,
`long_sustained_vowels` (the same five singled out in the external review
that prompted this pack, as a mix of an intelligibility positive-control,
two consonant-heavy stress tests, and two vowel/prosody stress tests).

## The question this pack actually answers

Does Vocos sound like a real improvement over Arm B, a real regression, or
a genuine tradeoff (say, smoother/less harsh but less intelligible)? Any of
the three is a valid, useful answer — this is not a pack built to confirm a
predetermined conclusion.

**Decision rule going in** (recorded before judging, so it can't be
adjusted after seeing the result): if V is rated equal-or-better than B on
naturalness *and* not clearly worse on intelligibility across most of the 5
phrases, that's evidence to keep pursuing the Vocos direction (e.g. trying
NSF-HiFiGAN next). If V is worse on naturalness, or the WER regression is
audible as genuinely lost words rather than a metric artifact, that's
evidence to shelve the neural-vocoder-swap direction and prioritize the
separate, already-identified product-integration gap instead (`/sing`
wired to the worst-measured renderer).

## Provenance

Nothing was re-rendered for this pack — all 15 source files are already
recorded outputs under measurement (`v10_4arm_matrix_full10/` for B,
`v12_vocos_resynth/` for V, `gate2_audio/*_spoken.wav` for K). Peak
normalization to -1 dBFS was applied per clip (gain factors recorded in the
key). Shuffle is seeded (`20260729`, see
`kokoro-world-vocoder/20_build_v12_blind_pack.py`) and reproducible.

Record the verdict in
`symthaea/docs/research/evidence/singing-voice-rvc-2026-07-26/backbone-smoke-test-kokoro-world/vocos-resynthesis/`.
A negative or mixed answer is as useful as a positive one.
