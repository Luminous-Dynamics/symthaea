# Blind pass — response sheet

**Do not open `02_KEY_DO_NOT_OPEN_UNTIL_JUDGED/` until this sheet is filled in.**

16 clips, **54.8 seconds of audio total** (longest single clip 14.9s, rest are
0.8–6.3s). Expect ~10–15 minutes with repeats and writing time.

The clips are shuffled across five groups you are not told the membership of.
No system names, no expected text, no metric values appear anywhere in this
folder.

## For each clip, fill in four things

**1. Transcription — the most important field.** Write *exactly* what you hear,
verbatim, before thinking about anything else. Guess if it's marginal. Write
`(unintelligible)` if you genuinely can't. Do **not** try to work out what it
"should" be — a wrong guess is data, a corrected guess is not.

**2. Singing?** — `yes` / `partly` / `no, pitch-bent speech` / `no, not speech`

**3. Quality** — 1 (unusable) to 5 (would ship)

**4. Notes** — anything: robotic, buzzy, warbly, clicks, breathy, flat.

```
clip_01   heard: "                                              "  singing: ____  quality: __  notes:
clip_02   heard: "                                              "  singing: ____  quality: __  notes:
clip_03   heard: "                                              "  singing: ____  quality: __  notes:
clip_04   heard: "                                              "  singing: ____  quality: __  notes:
clip_05   heard: "                                              "  singing: ____  quality: __  notes:
clip_06   heard: "                                              "  singing: ____  quality: __  notes:
clip_07   heard: "                                              "  singing: ____  quality: __  notes:
clip_08   heard: "                                              "  singing: ____  quality: __  notes:
clip_09   heard: "                                              "  singing: ____  quality: __  notes:
clip_10   heard: "                                              "  singing: ____  quality: __  notes:
clip_11   heard: "                                              "  singing: ____  quality: __  notes:
clip_12   heard: "                                              "  singing: ____  quality: __  notes:
clip_13   heard: "                                              "  singing: ____  quality: __  notes:
clip_14   heard: "                                              "  singing: ____  quality: __  notes:
clip_15   heard: "                                              "  singing: ____  quality: __  notes:
clip_16   heard: "                                              "  singing: ____  quality: __  notes:
```

## Two overall questions, after all 16

- **Best and worst clip numbers**, and roughly why.
- **Did any two clips sound like the same phrase rendered differently?** If so,
  which, and which of the pair was better? (This is the E comparison, blind.)

## Play them

```
cd symthaea/audio_output/LISTENING_PACK_2026-07-28/01_BLIND_PASS
for f in clip_*.wav; do echo "== $f"; ffplay -nodisp -autoexit -loglevel quiet "$f"; done
```

## Disclosed limitation of this blinding

Blinding is by filename only, so two cues survive and you should know about
them rather than be misled:

- **Duration partially leaks group membership.** One group sits at 3.6–6.3s and
  one clip is 14.9s, while the rest cluster at 0.8–2.8s.
- **One clip is a different phrase entirely** from every other clip, so its
  identity is obvious from content.

Those two are *anchors* (a known-bad baseline and a known-good reference),
not candidates. **The comparisons the decision actually turns on — the current
renders against each other, and the two versions of the same phrase — all sit
inside the 0.8–2.8s band and are properly blinded.** Transcription accuracy,
the single most important field, is immune to this priming either way.

Shuffle is seeded (`20260728`) and reproducible; the mapping is in the key
folder.
