# Blind pass — Arm B vs. Vocos v12 vs. spoken Kokoro

**Do not open `../02_KEY_DO_NOT_OPEN_UNTIL_JUDGED/` until this sheet is filled in.**

15 clips (5 phrases × 3 conditions), 1.8–2.6s each, ~33s of audio total.
Expect 10–15 minutes with repeats and writing time.

All clips are **peak-normalized to a shared -1 dBFS peak** so none is louder
purely from gain — this is peak matching, not true perceptual/LUFS loudness
matching, so a clip can still *sound* louder if its energy is more
concentrated. No other processing was applied; nothing about the renders
themselves was changed to make this pack.

Clip identities are shuffled and neutral (`clip_01.wav` … `clip_15.wav`) —
no system names, no expected text, no metric values appear anywhere in this
folder.

## For each clip, fill in four things

**1. Transcription — the most important field.** Write *exactly* what you
hear, verbatim, before thinking about anything else. Guess if it's marginal.
Write `(unintelligible)` if you genuinely can't. Do **not** try to work out
what it "should" be — a wrong guess is data, a corrected guess is not.

**2. Intelligibility** — 1 (couldn't make out any words) to 5 (every word
clear). This is about whether you can tell what's being said, independent of
whether it sounds good.

**3. Naturalness** — 1 (robotic/broken/not like a voice) to 5 (sounds like a
real voice). This is about how it sounds, independent of whether you could
transcribe it. **Rate this separately from intelligibility** — a clip can be
perfectly clear and still sound robotic, or mostly mumbled and still sound
warm/natural; that combination (or its absence) is exactly what this pack is
trying to find.

**4. Notes** — anything: robotic, buzzy, warbly, clicks, breathy, flat,
metallic, smeared/blurry.

```
clip_01   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_02   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_03   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_04   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_05   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_06   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_07   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_08   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_09   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_10   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_11   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_12   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_13   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_14   heard: "                                              "  intelligibility: __  naturalness: __  notes:
clip_15   heard: "                                              "  intelligibility: __  naturalness: __  notes:
```

## Two overall questions, after all 15

- **Best and worst clip numbers**, and roughly why.
- **Did any clips sound like the same phrase rendered differently?** If you
  notice groups that seem related, note which and which of each group you'd
  pick, before opening the key.

## Play them

```
cd symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_blind_pack_v12_vs_b_vs_kokoro/01_BLIND_PASS
for f in clip_*.wav; do echo "== $f"; ffplay -nodisp -autoexit -loglevel quiet "$f"; done
```

## Disclosed limitations of this blinding

- **Duration doesn't leak much here** (1.82–2.55s across all 15, a narrow
  band) — unlike the prior `LISTENING_PACK_2026-07-28` pack, duration is not
  a usable side-channel in this one.
- **The spoken-Kokoro clips (condition K, unknown to you at this stage) are
  a quality anchor, not a singing candidate** — they will very likely sound
  clearly better and clearly not sung. That's expected and not informative
  about Arm B vs. v12; the comparison this pack is actually for is the two
  *sung* conditions against each other.
- Peak normalization matches loudness only approximately (peak, not RMS/LUFS)
  — if one condition still sounds noticeably louder/softer, note it, but
  don't let it drive your naturalness/intelligibility scores if you can help
  it.

Shuffle is seeded (`20260729`) and reproducible; the mapping is in the key
folder, along with the true target text for scoring transcriptions.
