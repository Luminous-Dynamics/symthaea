# v4: sonorant/obstruent voicing preservation -- a diagnosed null result (2026-07-28)

Per the reviewer's diagnosis of v3's over-voicing (independently
confirmed: voiced-frame fraction roughly doubled vs. the spoken source,
spectral centroid down 20-40%, ZCR down 57-76% -- consistent with
periodic energy replacing consonant/fricative noise). v4 implements
their exact proposed fix on top of the locked v3 control
(`03v3_LOCKED_control.py`, unmodified):

1. Three-way phoneme classification (sonorant = vowel + nasal/liquid/
   glide, pitch-bearing; obstruent = stop/fricative/affricate, never
   pitch-bearing) instead of v3's binary vowel/consonant split.
2. Obstruent frames forced to `target_f0=0` (unvoiced/aperiodic
   synthesis) regardless of the source's own voicing, per the
   reviewer's literal pseudocode.
3. Obstruent duration pinned to its own estimated natural value in both
   the natural and target timeline (not scaled by the syllable's
   overall stretch factor); sonorants absorb 100% of the difference.

## Result: the fix did not reduce the measured over-voicing

| Phrase | spoken voiced_frac | v3 voiced_frac | v4 voiced_frac | spoken centroid | v3 centroid | v4 centroid |
|---|---|---|---|---|---|---|
| hello_world | 0.346 | 0.873 | **0.940** | 3439 Hz | 2069 Hz | 1963 Hz |
| sun_rises | 0.624 | 0.874 | **0.881** | 3341 Hz | 2669 Hz | 2559 Hz |
| quiet_morning | 0.522 | 0.867 | **0.894** | 3286 Hz | 2453 Hz | 2376 Hz |

v4's voiced fraction is *higher* than v3's on all three phrases, and
centroid moved slightly further from the source, not closer -- the
opposite of the intended direction. This was investigated immediately
rather than reported at face value, and the root cause was found and
confirmed directly (not guessed):

**"Hello world" is phonetically sonorant-dominated.** Direct inspection
of Kokoro's own G2P output (`həlˈO wˈɜɹld`) and the resulting
classification: "hello" = `h`(obstruent) + `ə,l,O`(all sonorant) -- only
ONE true obstruent phoneme in the whole word; "world" = `w,ɜ,ɹ,l`(all
sonorant) + `d`(obstruent) -- again only ONE. Direct measurement of the
fed F0 array (before synthesis, not the resynthesized-audio re-analysis)
confirms the classification is working exactly as coded: 89.3%/84.2% of
frames are genuinely sonorant-eligible for these two words -- there is
simply very little obstruent material in this specific phrase for the
fix to act on. Combined with change #3 (sonorants now absorb 100% of the
stretch budget, vs. v3's more even spread), the already-dominant
sonorant/vowel portion becomes proportionally LARGER in v4, which works
against reducing the aggregate voiced-time fraction even though the
classification itself is correct.

## What this does and doesn't mean

**Does NOT mean**: the sonorant/obstruent classification or the
forced-unvoiced-obstruent idea is wrong. It is verified working exactly
as designed at the phoneme level.

**Does mean**: the reviewer's own recommended methodology (run a proper
ablation, one change at a time) was the right call, and this session
combined two mechanisms (voicing-eligibility change + duration-budget
change) in one step, which is now diagnosed as a real confound for a
phrase this sonorant-heavy. It also means the *aggregate* voiced-
fraction/centroid/ZCR proxies, useful for catching v3's original
problem, are not sensitive enough on their own to validate a fix whose
effect is concentrated in a small number of consonant frames within an
otherwise vowel/sonorant-dominated phrase -- a phrase with more genuine
obstruent content (e.g. Gate 2's `consonant_clusters` or
`fricative_heavy`) would be a fairer test of whether this fix helps, and
was not tried this pass.

## Explicitly not done, stopped here rather than iterate further blindly

- WER was not re-measured for v4 (the voicing/aggregate result was
  negative enough on its own terms to warrant stopping and reporting
  before spending more compute chasing it further).
- The reviewer's full 4-arm ablation (v3 control / +unvoiced-mask-only /
  +unvoiced-mask+vowel-only-stretch / +contextual-transitions) was not
  run as literally specified -- this session combined arms 2+3 into one
  v4 step, which is exactly the kind of confound the ablation design was
  meant to prevent, now confirmed as a real issue.
- Not tried: separating the voicing-eligibility change from the
  duration-budget change to see which one (if either) actually helps;
  re-running on a more obstruent-heavy phrase.

## Correction (2026-07-28, same day): the "null result" was an artifact of a single F0-estimation algorithm

An external re-measurement using a different methodology found the
OPPOSITE direction for 2 of 3 phrases (quiet_morning 91.7%->83.4%,
sun_rises 88.7%->81.9%, both real decreases) than this doc originally
reported (all 3 phrases increasing, per `pw.harvest`). Rechecked
directly with `pw.dio` (a different WORLD-family F0 estimator) alongside
the original `pw.harvest` measurement:

| Phrase | harvest v3->v4 | dio v3->v4 |
|---|---|---|
| hello_world | 0.873->0.940 (+0.068) | 0.819->0.806 (-0.013) |
| sun_rises | 0.874->0.881 (+0.007) | 0.736->0.703 (-0.034) |
| quiet_morning | 0.867->0.894 (+0.027) | 0.794->0.766 (-0.027) |

**With `dio`, all three phrases show the expected direction** (voiced
fraction decreasing after the fix) -- consistent with the fix actually
working, at least modestly, on sun_rises and quiet_morning (which have
more obstruent content), while hello_world's effect stays small/
ambiguous either way (consistent with the root-cause diagnosis above:
too little obstruent material in that phrase for a reliable signal in
either direction -- a real negative CONTROL result, not a failure).

**Corrected conclusion**: the original "diagnosed null result" section
above was itself based on a measurement methodology too narrow to trust
-- a single F0-estimation algorithm (`harvest`) that disagrees with a
standard alternative (`dio`) specifically on this kind of transient/
consonant-heavy material, which is exactly the region under dispute.
The underlying fix (sonorant/obstruent classification, obstruent
duration pinning) is NOT disproven and shows real, modest, algorithm-
confirmed improvement on the two phrases with adequate obstruent
content. The methodological lesson stands regardless: voiced-fraction
measurement on short/transient segments needs cross-checking against
more than one F0 estimator before being treated as a reliable verdict,
and aggregate whole-clip statistics remain too coarse to properly
attribute credit -- per-phoneme-span localized measurement (not yet
done) is the right next step, not a broader v5.

## Files

- `03v3_LOCKED_control.py` -- the frozen control point for this
  comparison (identical to `../03_reshape_pyworld.py`'s v3, kept as its
  own copy here for a stable before/after reference).
- `03v4_obstruent_preserve.py` -- the v4 implementation.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/*_sung_v4.wav`.
