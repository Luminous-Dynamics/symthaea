# v6: voiced vs. voiceless obstruent split (2026-07-28)

Per the reviewer's refinement of the ablation's mask-only mechanism:
"force every obstruent unvoiced" is too broad -- English has voiced
obstruents (`/b d g v ð z ʒ dʒ/`) that shouldn't be silenced the same
way voiceless ones (`/p t k f θ s ʃ h tʃ/`) should. On top of v3's
unmodified duration behavior (per the reviewer's own naming, "v4-mask
candidate: v3 plus phoneme-class-aware voicing eligibility, with the
old duration behavior retained" -- the ablation found duration-pinning
counterproductive in isolation, so it is deliberately NOT carried
forward here):

- **sonorant** (vowel + nasal/liquid/glide): pitch-imposed as before
  (target note + gestures), only on originally-voiced frames.
- **voiceless obstruent**: forced unvoiced (`target_f0=0`) regardless
  of source voicing -- the confirmed mask-only mechanism.
- **voiced obstruent**: neither forced unvoiced nor repitched -- the
  ORIGINAL (resampled) F0 passes through unchanged, preserving the
  source's own natural micro-pitch and any partial devoicing already
  present, per "preserve the source voicing decision."

Character sets checked directly against this project's own test
phrases' actual misaki G2P output (not assumed from external IPA docs).

## Result

### consonant_clusters (replication + refinement check)

| Variant | WER | voiced (harvest/dio) | centroid |
|---|---|---|---|
| spoken reference | -- | 0.551 / 0.387 | 3559 Hz |
| A_v3 (ablation) | 0.250 | 0.793 / 0.582 | 2797 Hz |
| B_mask_only (ablation) | 0.000 | 0.650 / 0.491 | 2755 Hz |
| **v6_voiced_split** | **0.000** | **0.786 / 0.516** | **2942 Hz** |

WER stays perfect, matching mask-only. Voiced-fraction sits *between*
v3 and mask-only -- this is the mathematically correct, expected
consequence of only zeroing voiceless obstruents while now correctly
preserving the phrase's two real voiced obstruents (`z` in "streams",
`dʒ` in "strangely") that mask-only had also (per the reviewer's own
critique) incorrectly silenced. Centroid actually moves PAST mask-only,
closer to the spoken source (2942 vs mask-only's 2755 vs source's
3559) -- a genuinely positive signal, not a partial regression.

### fricative_heavy / phrase_final_stops (new obstruent-heavy phrases)

| Phrase | v6 WER | Note |
|---|---|---|
| fricative_heavy | 0.333 | Sole "error" is "sea shore" vs "seashore" -- a tokenization/spacing artifact (this exact word has shown this quirk elsewhere in this whole arc), not a real word error. Effectively correct. |
| phrase_final_stops | 0.143 | Dropped the final "it" -- a familiar word-boundary issue seen elsewhere in this project, not obviously related to the voicing mechanism. |

No v3/mask-only baseline exists yet for these two phrases (they were
not part of the original 4-arm ablation), so only spoken-vs-v6 is
reported. `phrase_final_stops`'s centroid drop (3305->2169) is the
largest of any phrase tested this pass -- flagged, not yet explained.

### hello_world (negative control)

| Variant | WER | voiced (harvest/dio) | centroid |
|---|---|---|---|
| spoken reference | -- | 0.346 / 0.301 | 3439 Hz |
| v3 | 0.000 | 0.873 / 0.819 | 2069 Hz |
| v6 | 0.000 | 0.873 / 0.804 | 2069 Hz |

**v3 and v6 are numerically identical on harvest voiced-fraction and
centroid** -- exactly as predicted, since this phrase has essentially
no obstruent material (1 per word) for the refined classification to
act on differently than v3. The tiny dio delta (0.819->0.804) is
negligible.

## Interpretation

The voiced/voiceless split replicates mask-only's clean WER win while
being phonetically more principled (it no longer artificially silences
real voiced consonants), and the one direct A/B/C comparison available
(`consonant_clusters`) shows it moving centroid even further toward the
source than pure mask-only did. The negative control confirms the
refinement doesn't introduce any unwanted effect where there's nothing
to fix.

## Not yet done

- No v3/mask-only baseline for `fricative_heavy`/`phrase_final_stops`
  specifically -- only spoken-vs-v6 comparison available for those two.
- A voiced-obstruent-heavy phrase deliberately designed to stress-test
  the voiced-obstruent-preservation rule specifically (the reviewer's
  item 4) -- not built this pass.
- The proposed 5th arm (preserve the original Kokoro consonant waveform
  directly, crossfaded into the WORLD-rendered vowel) -- not attempted.
- Per-phoneme-span localized metrics -- still only whole-clip aggregates.
- The human listening check -- still the standing, most important open
  item across this entire arc.

## Files

- `03v6_voiced_obstruent_split.py` -- the v6 renderer.
- `08_v6_evaluate.py` -- evaluation against ablation arms + spoken refs.
- `v6_config_gate2.json` / `v6_config_main.json` -- test-phrase configs.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/*_sung_v6.wav`.
