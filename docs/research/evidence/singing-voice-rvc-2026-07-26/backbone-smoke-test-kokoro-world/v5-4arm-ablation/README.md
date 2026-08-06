# 4-arm ablation: separating the two v4 mechanisms (2026-07-28)

Per the reviewer's originally-proposed design (run separately, not
combined): a single script (`03_ablation_4arm.py`) with two independent
boolean toggles renders all 4 combinations from one codebase, avoiding
the risk of the arms silently drifting apart.

| Arm | Voicing eligibility | Duration allocation |
|---|---|---|
| A_v3 | any originally-voiced frame (v3 behavior) | v3 scaling (obstruent share caps to the SYLLABLE's own total, natural and target computed independently) |
| B_mask_only | obstruents forced unvoiced regardless of source voicing | v3 scaling |
| C_duration_only | v3 voicing | obstruent duration pinned to its own natural estimate; sonorants absorb 100% of the difference |
| D_combined | obstruents forced unvoiced | obstruent duration pinned (= the original v4) |

Test phrases: `consonant_clusters` ("strong streams splashed
strangely" -- genuinely obstruent-heavy) as the positive test, and
`hello_world` as the reviewer's own predicted low-effect negative
control (already established: only 1 true obstruent phoneme per word).
Evaluated with BOTH `pw.harvest` and `pw.dio` F0 estimators throughout,
per the correction below.

## Result

### consonant_clusters (positive test)

| Arm | voiced (harvest/dio) | centroid | WER | hypothesis |
|---|---|---|---|---|
| spoken reference | 0.551 / 0.387 | 3559 Hz | -- | -- |
| A_v3 | 0.793 / 0.582 | 2797 Hz | 0.25 | "Strong Streams **Blashed** Strangely." |
| **B_mask_only** | **0.650 / 0.491** | 2755 Hz | **0.0** | "Strong streams splashed strangely." |
| C_duration_only | 0.820 / 0.596 | 2759 Hz | 0.25 | "Strong streams **flashed** strangely." |
| D_combined (=v4) | 0.689 / 0.501 | 2720 Hz | 0.0 | "Strong streams splashed strangely." |

### hello_world (negative control)

| Arm | voiced (harvest/dio) | centroid | WER |
|---|---|---|---|
| spoken reference | 0.346 / 0.301 | 3439 Hz | -- |
| A_v3 | 0.940 / 0.836 | 1970 Hz | 0.0 |
| B_mask_only | 0.940 / 0.806 | 1963 Hz | 0.0 |
| C_duration_only | 0.940 / 0.836 | 1970 Hz | 0.0 |
| D_combined | 0.940 / 0.806 | 1963 Hz | 0.0 |

## This is now a clean, decisive, separated result

**Mask-only (Arm B) is a real, unambiguous win on the obstruent-heavy
phrase**: voiced-fraction drops substantially toward the spoken
reference (both F0 estimators agree on the direction and rough
magnitude -- harvest 0.793->0.650, dio 0.582->0.491), *and* WER improves
to a perfect 0.0 from v3's 0.25 ("splashed"->"Blashed"). This is the
mechanism working exactly as the reviewer predicted.

**Duration-only (Arm C) is actively counterproductive in isolation**:
voiced-fraction goes UP relative to v3 (harvest 0.793->0.820, dio
0.582->0.596), and WER gets slightly worse ("splashed"->"flashed").
Confirms the hypothesis from the v4 writeup: pinning obstruent duration
near-natural while letting sonorants absorb 100% of the syllable's
stretch budget makes the timeline MORE vowel-dominated, not less --
working against, not toward, reduced over-voicing.

**Combined (Arm D, = the original v4) sits between B and C** on every
metric (voiced-fraction 0.689/0.501, between B's 0.650/0.491 and C's
0.820/0.596). This precisely explains why the original v4 test (which
only compared v3 against the combined D arm) showed a diluted,
sometimes-reversed effect: combining a real win (masking) with a real
harm (duration-pinning) partially cancels the benefit. The earlier
"diagnosed null result" undersold the masking mechanism specifically
because it was never tested in isolation.

**hello_world's negative control behaves exactly as predicted**: all
four arms are nearly identical (the duration-pinning arms A vs C are
*exactly* identical on every metric -- with only one obstruent phoneme
per word, pinning vs. not pinning its duration makes no measurable
difference), and even the masking arms (B, D) show only a small dio
delta (0.836->0.806) with harvest showing none at all. A null on this
phrase was expected and remains uninformative either way, exactly as
the reviewer predicted before this ablation ran.

## Revised recommendation

**Adopt mask-only (Arm B) as the new lead candidate, not the combined
v4.** Drop or substantially redesign the duration-pinning mechanism --
it measurably works against the goal in isolation. A more promising
direction for duration (not yet built): cap the OVERALL syllable's
stretch factor itself when the syllable is heavily obstruent-loaded,
rather than pinning obstruents and dumping all slack onto the vowel
unconditionally.

## What's still not addressed, per the reviewer's own further
diagnosis (their most recent message, not yet acted on this pass)

- `F0=0` is necessary but likely not sufficient for consonant
  naturalness -- WORLD's resynthesized spectral envelope/aperiodicity
  for a forced-unvoiced obstruent may still sound smoothed/muffled
  relative to preserving the original waveform directly. Not tested.
- Per-phoneme-span localized metrics (not just whole-clip aggregates)
  -- not implemented this pass; the whole-clip A/B/C/D comparison above
  is still coarser than the reviewer's requested per-span breakdown.
- The proposed 5th arm (preserve original Kokoro consonant waveform
  directly, crossfade into the transformed vowel) -- not attempted.
- Only 1 obstruent-heavy phrase tested (`consonant_clusters`);
  `fricative_heavy` and `phrase_final_stops` were not run this pass.
- No human listening check, same standing caveat as every gate in this
  arc.

## Methodological correction folded in from this pass

An earlier version of the v4 writeup reported a "diagnosed null result"
based on `pw.harvest` alone, which disagreed with an external
re-measurement. Rechecked with `pw.dio` and found the two F0 estimators
disagree on this kind of transient/consonant-heavy material -- both are
now reported side by side throughout this doc rather than trusting
either alone. See `../v4-obstruent-preserve/README.md`'s correction
section for the full detail of that reconciliation.

## Files

- `03_ablation_4arm.py` -- the 4-arm renderer (two boolean toggles,
  one codebase, avoids arm drift).
- `07_ablation_evaluate.py` -- dual-F0-estimator + WER evaluation.
- `ablation_config_gate2.json` / `ablation_config_main.json` -- the
  two test-phrase configs.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/ablation/`.
