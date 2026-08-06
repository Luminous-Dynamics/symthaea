# v10: the 4-arm synthesis matrix -- first real test of the synthesis-event-record boundaries in a renderer

Per the reviewer's plan: only after the acoustic-event semantics audit
and synthesis-event-record work (both closed, both gaps fixed) does it
make sense to test whether event-aware boundaries actually help
synthesis. Four arms on the three designated phrases
(`fricative_heavy`, `consonant_clusters`, `phrase_final_stops`):

| Arm | Boundaries | Rendering |
|---|---|---|
| A | Existing proportional heuristic (v6, unmodified, frozen control) | Mask-only |
| B | Hybrid event alignment (synthesis-event-record) | Mask-only, event-informed partial masking |
| C | Hybrid event alignment | Raw-transient/consonant preservation |
| D | Hybrid event alignment | High-frequency residual preservation (new -- never attempted before in this arc) |

## Two real bugs found and fixed while building the renderer, before any evaluation

1. **Stress-mark indexing bug**: `classify()` strips stress marks
   (`ˈˌ`) before producing its phoneme list, so the N-th entry of that
   list is NOT the N-th character of the original phoneme string once a
   stress mark has been skipped. A naive running `char_cursor += 1` per
   phoneme silently misaligned every phoneme after the first stress mark
   in a word. Caught immediately: `fricative_heavy` (which has 7 real
   events, confirmed directly via `build_events_by_idx`) reported
   `events_used=0` on the first run. Fixed via `phoneme_index_map()`, a
   precomputed per-word map from post-strip phoneme index to true
   original-string index.
2. **Word-boundary-too-narrow bug**: even after fixing (1), only 2/7
   events were usable, because raw extraction was constrained to the
   MMS_FA word-level slice (`x_word`), and MMS_FA's word boundary is
   frequently narrower than the true event span -- e.g. "she"'s /S/
   event starts at 5ms, but MMS_FA places the word "she" starting at
   322ms. Fixed by extracting directly from the full-phrase audio at
   absolute sample coordinates instead of word-relative ones. Result:
   events_used rose to 6/7 (fricative_heavy), 5/5 (phrase_final_stops),
   10/10 (consonant_clusters) -- essentially every detected event is now
   actually usable by the renderer.

Both bugs are disclosed here because they're the same lesson this whole
arc keeps re-teaching: verify the pipeline is doing what it's supposed
to before trusting any output from it.

## Result: WER (secondary evidence, as scoped)

| Phrase | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| phrase_final_stops | 0.143 | 0.143 | 0.143 | **0.000** |
| fricative_heavy | 0.333 | 0.333 | 0.333 (different errors, see below) | 0.333 |
| consonant_clusters | **0.000** | 0.000 | **1.000** | **0.750** |

Hypotheses, verbatim:
- A/B (identical on all 3): `"Turn off the light and lock."` /
  `"She sells seashells by the sea shore."` / `"Strong streams splashed
  strangely."`
- C: `"Turn off the light and lock."` / `"She fell seashells by the
  seashells."` / `"Drong Stream's flash screened link."`
- D: `"Turn off the light and lock it."` / `"She sells seashells by the
  sea shore."` / `"Drongstream slash Strangely."`

## Arm B: neutral

Bit-for-bit-identical WER and hypotheses to Arm A on all 3 phrases. The
event-informed partial-masking refinement (mask only the fraction of a
proportional entry the real event covers, not the whole entry) didn't
change the transcribed outcome here -- plausibly because for most
tokens the event's span covers nearly the whole proportional entry
anyway, so partial masking rarely differs materially from full masking
in practice on this phrase set. Not harmful, not yet demonstrated
helpful by this metric.

## Arm C: net negative

`phrase_final_stops` unchanged. `fricative_heavy` scores the same
0.333 but with QUALITATIVELY DIFFERENT errors -- "sells"->"fell" is a
new, unrelated mishearing (not the same "sea shore" tokenization-only
artifact as A/B), a real degradation even though the raw score matches.
`consonant_clusters` -- the phrase with the densest consonant clusters
(`str-`, `spl-`) -- **collapses from perfect (0.000) to totally garbled
(1.000)**, "Drong Stream's flash screened link" bearing little
resemblance to the target.

## Arm D: mixed -- one genuine win, one severe regression

`phrase_final_stops` improves cleanly: 0.143 -> **0.000**, correctly
transcribing the previously-dropped final "it" ("Turn off the light and
lock it."). `fricative_heavy` unchanged. `consonant_clusters` also
collapses, though less catastrophically than C (0.750 vs 1.000 --
"Drongstream slash Strangely" is still badly garbled but marginally
less so).

## Sanity-checked: the consonant_clusters collapse is not a crude technical bug

Before writing this up as a real acoustic problem, checked for the
obvious alternative explanation: no NaN/inf in either regressed render,
RMS/peak levels consistent across all 4 arms (~0.049/~0.43), and max
sample-to-sample click magnitude is actually slightly LOWER for C/D
than A (0.202-0.203 vs 0.218). This rules out a gross technical failure
(clipping, corruption, a broken crossfade) -- the degradation is a real,
more subtle spectral/temporal-coherence problem specific to dense
consonant clusters, not investigated further this pass.

## What this means, honestly

**Not a clean win for either new mechanism.** Arm D's phrase_final_stops
result is a genuine, real improvement worth taking seriously -- it's
exactly the kind of case (an isolated, well-separated stop) the whole
acoustic-event-semantics investigation predicted should benefit most.
But BOTH Arm C and Arm D fail badly on dense consonant clusters, and
per the standing lesson from v9 (a local acoustic win doesn't guarantee
a synthesis win, and vice versa a local metric doesn't fully predict
listener-relevant failure), this asymmetry is itself the most important
finding here: **event-aware consonant preservation seems to help
isolated consonants and hurt tightly-clustered ones** -- plausibly
because extracting and re-stitching multiple short raw/residual
segments in rapid succession (several phones within ~100-150ms of each
other, per the acoustic-event audit's own neighbor-clamping findings)
compounds concatenation artifacts faster than it removes them.

**Arm B remains the safest choice if a decision were needed today** --
neutral on this evidence, not the source of any regression. Arms C/D
are not ready to promote as-is.

## Not yet done

- Root-causing WHY dense clusters specifically break down (candidate
  hypotheses: rapid raw/residual segment re-stitching, resampling
  distortion under unusual target-duration stretch ratios, or
  compounding boundary artifacts across closely-spaced extractions --
  none tested).
- Testing Arm D on a broader phrase set to see if the phrase_final_stops
  win generalizes to other isolated-consonant contexts.
- Any attempt to fix the cluster-collapse failure mode.
- The `MIN_SYLLABLE_DUR_S`-floor hypothesis (still open since v9).
- The human listening check -- still the standing, most important item,
  and arguably more necessary now than ever given WER alone shows a
  genuinely mixed, hard-to-interpret picture (does Arm D's
  phrase_final_stops win sound better? does Arm C/D's cluster garbling
  sound as bad as the WER implies, or is WER over-penalizing a real but
  survivable acoustic change? Only a human ear can answer this).

## Files

- `15_hybrid_event_synthesis_matrix.py` -- the Arm B/C/D renderer
  (Arm A is `03v6_LOCKED_control.py`, unmodified, run separately).
- `matrix3_config.json` -- the 3-phrase subset config.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix/*_sung_v10_{a,b,c,d}.wav`
  (gitignored, not duplicated here).

## Update (2026-07-29): root-caused the consonant_clusters collapse -- confirmed, partially fixed

Traced `consonant_clusters`' events directly (same method used throughout
this arc: dump `build_events_by_idx`'s output and inspect for anomalies
before trusting anything). Found two real, previously-undiscovered bugs
that the search-window neighbor-clamping (from the acoustic-event-
semantics pass) does NOT prevent:

1. **Phrase-initial extreme-duration span**: "strong"'s initial /s/ --
   phrase-initial, so its search legitimately extends to the true
   utterance start per the earlier phrase-boundary fix -- produced a
   **266ms preservation span**. Squeezing 266ms of real source audio
   into a ~30-60ms target consonant slot via `resample_waveform` (linear
   time-compression) would badly distort it -- likely the single
   largest contributor to the collapse, since it's the very FIRST
   sound in the phrase and severe distortion there could plausibly
   throw off Whisper's whole-utterance decoding.
2. **Cross-word preservation-span overlap**: neighbor-clamping bounds
   the SEARCH WINDOW against a neighbor's CTC span, but the resulting
   preservation spans are computed independently and can still overlap
   each other. Found exactly this: "splashed"'s final /t/
   `[1.4375,1.4516]`s vs "strangely"'s initial /s/ `[1.4325,1.5175]`s --
   /s/'s span starts BEFORE /t/'s own span ends. Extracting raw audio
   for both duplicates overlapping source content when concatenated.

**Fix**: cap preservation-span duration at 120ms (clipped from the
start side, since every observed case had the start erroneously too
early, never the end), and sequentially clip each event's start against
the immediately-preceding event's own end.

**Result**: Arm C's `consonant_clusters` WER improves substantially,
1.000 -> **0.500** ("Drong streams splash strangely" -- only "Strong"
and a minor "splashed"/"splash" error remain, a completely different
and far less severe failure mode than the original "Drong Stream's
flash screened link"). This confirms the root-cause diagnosis: fixing
the two bugs measurably improved the worst failure.

**But this is a partial fix, not a solved problem**: Arm D's
`consonant_clusters` WER is UNCHANGED (0.750, still "Drongstream slash
strangely") -- the same two fixes that helped Arm C didn't help Arm D,
suggesting the band-split residual mechanism has an ADDITIONAL,
separate problem these fixes don't address (candidate: the WORLD
low-band component still uses the OLD proportional-model frame slicing,
now potentially misaligned with the corrected, shorter raw high-band
content in a new way -- not confirmed). And Arm C's `fricative_heavy`
got a NEW, milder regression (0.333 -> 0.500, "She fell seashells by the
sea shore" -- the pre-existing "sells"->"fell" error persists plus a
new "seashore"->"sea shore" tokenization split) -- plausibly an
interaction between the new duration cap and this phrase's own
phrase-initial-fricative case, not investigated further.

## Updated verdict

Arm B remains WER-neutral throughout, on both the original and fixed
event tables -- still the safest choice on this evidence. Arm C is
now a clearer "real but incomplete win" for dense clusters. Arm D's
`phrase_final_stops` win still stands, but its cluster problem is
evidently not the same bug as Arm C's and remains open. Not chasing
further this pass -- the two bugs found and fixed were a well-scoped,
concrete unit of work; the residual-mechanism-specific issue is a
distinct, not-yet-diagnosed next question.

## Not yet done (updated)

- Root-causing Arm D's separate, unresolved consonant_clusters problem.
- The new Arm C fricative_heavy regression.
- Testing whether Arm D's phrase_final_stops win generalizes.
- The `MIN_SYLLABLE_DUR_S`-floor hypothesis (open since v9).
- The human listening check -- still the standing, most important item.

## Files (updated)

- `15_hybrid_event_synthesis_matrix.py` -- now includes the duration-cap
  + sequential-overlap-clip fix in `build_events_by_idx`.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_fix2/*_sung_v10c2_{b,c,d}.wav`.

## Update (2026-07-29, second pass): root-caused and fixed Arm D's remaining problem -- reshapes the whole verdict

Hypothesis for why the duration-cap/overlap fix helped Arm C but not
Arm D: Arm D's residual mechanism sums a WORLD-synthesized low band
with a raw-extracted high band, but only the raw extraction had been
corrected -- the WORLD component's own `sp`/`ap`/`f0` frames were STILL
being sliced from the OLD, uncorrected proportional-duration window.
The two bands could therefore represent different, temporally
mismatched material even after the raw-extraction fix.

**Fix**: slice the WORLD-domain frames from the event's own corrected
time span too (from the full-phrase `f0`/`sp`/`ap` arrays, mirroring
the raw-extraction fix exactly), whenever a real event exists --
applied uniformly to whichever arm reaches that code path (harmless for
Arm C, whose "raw" groups never read `sp`/`ap`/`f0` at all; a real
change for Arm B and Arm D, both of which do).

### Result: confirms the hypothesis, and reshapes the entire verdict

| Phrase | A (baseline) | B (final) | C (final) | D (final) |
|---|---|---|---|---|
| phrase_final_stops | 0.143 | **0.000** | 0.143 | **0.000** |
| fricative_heavy | 0.333 | 0.333 | 0.500 | 0.333 |
| consonant_clusters | **0.000** | 0.250 | 0.500 | 0.250 |

**Arm D's consonant_clusters WER drops from 0.750 to 0.250**
("Drongstream slash strangely" -> "Strong streams slash strangely" --
"Strong" and "streams" now both correct, only "splashed"->"slash"
remains wrong). This confirms the mismatched-time-window hypothesis
directly.

**Unexpected: Arm B's own audio changed too**, and non-uniformly --
`phrase_final_stops` improved (0.143 -> 0.000, now correctly says
"lock it"), but `consonant_clusters` REGRESSED (0.000 -> 0.250,
"splashed" -> "slashed"). Arm B is therefore no longer accurately
described as "purely neutral" -- it now shows a real win on one phrase
and a real, mild regression on another. Worth flagging since it means
this specific fix has a genuine, non-free tradeoff even for the
"safe" mask-only arm.

**Striking, not yet explained**: Arm D's final numbers (0.000/0.333/
0.250) are now IDENTICAL to Arm B's on all 3 phrases. Plausibly because
once the high-band residual is properly time-aligned with a coherent
low-band WORLD signal, the combined result transcribes equivalently to
a pure mask-only WORLD rendering for these phrases -- or this could be
a coincidence of a 3-phrase sample. Not investigated further.

**Arm C remains the worst-performing mechanism** even after this fix
(unaffected by it, since its "raw" groups never use the WORLD frames) --
net WORSE than the Arm A baseline on 2 of 3 phrases now
(`fricative_heavy` 0.333->0.500, `consonant_clusters` 0.000->0.500).
Pure raw-waveform replacement (no WORLD blending at all) looks like the
least robust of the three new mechanisms on this evidence.

### Updated verdict

**Arm D (band-split residual) is now the clearest candidate of the
three new mechanisms** -- it recovers the `phrase_final_stops` win,
ties Arm B (no longer clearly worse anywhere), and no longer collapses
on dense clusters. Arm B remains reasonable but is no longer risk-free.
Arm C should not be promoted -- worse than baseline on the majority of
tested phrases even after every fix applied this round.

## Not yet done (updated again)

- Explaining why Arm D and Arm B converge to identical WER on this
  3-phrase set.
- Investigating Arm B's new consonant_clusters regression
  ("splashed"->"slashed").
- Testing Arm D (and B) on a wider phrase set to see if this verdict
  holds beyond 3 phrases.
- The `MIN_SYLLABLE_DUR_S`-floor hypothesis (open since v9).
- The human listening check -- still the standing, most important item,
  and now the most decision-relevant it has been all arc: WER alone
  currently ranks D=B > A > C, but only a human ear can confirm whether
  that ranking matches perceived quality.

## Files (updated again)

- `15_hybrid_event_synthesis_matrix.py` -- now also slices WORLD-domain
  frames from the event's corrected span, not just the raw waveform.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_fix3/*_sung_v10c3_{b,c,d}.wav`.

## Update (2026-07-29, third pass): generalization test on all 10 Gate-2 phrases -- revises the verdict

Per the standing open question ("does the D=B>A>C ranking hold beyond
3 phrases?"): ran all 4 arms (Arm A = `03v6_LOCKED_control.py`,
Arms B/C/D = the fully-fixed `15_hybrid_event_synthesis_matrix.py`) on
the complete Gate-2 10-phrase set, 40 renders total.

### Result: mean WER across all 10 phrases

| Arm | Mean WER |
|---|---|
| A (baseline) | 0.358 |
| **B** | **0.284** (clear best) |
| C | 0.390 (worst -- net negative vs. baseline) |
| D | 0.333 |

**This revises the 3-phrase verdict.** On the larger, more diverse set,
Arm B is the clear overall winner, not tied with D -- D is only a
modest improvement over baseline, and C remains net-negative,
consistent with the 3-phrase finding.

### A genuine, disclosed methodological confound found while investigating the discrepancy

`phrase_final_stops` scored WER=0.000 on Arm D in the 3-phrase test but
WER=0.143 on Arm D in this 10-phrase test -- same phrase, same fully-
fixed code, same source audio. Traced and confirmed: NOT a
determinism bug. `build_config_with_syllable_melody`'s melody
assignment (`target_hz` per syllable) is a running counter (`note_idx`)
across the WHOLE phrase list passed to one invocation, not
phrase-independent. `phrase_final_stops` is index 0 in
`matrix3_config.json` (3-phrase subset, in original Gate-2 order) but
index 4 in `gate2_config.json` (full set) -- by the time the full-set
run reaches it, `note_idx` has already advanced past 4 prior phrases'
syllables, giving genuinely different pitch targets for the exact same
phrase. **Cross-config comparisons for the same phrase ID are therefore
confounded and not directly comparable** -- the 10-phrase run's
internally-consistent, same-config comparison across arms (this
section) is the more trustworthy generalization test; the earlier
3-phrase numbers should not be read as "the same experiment, more
phrases."

### Notable results within the 10-phrase set

- **`positive_control`: 0.500 (A) -> 0.000 (B, C, AND D all identical)**
  -- a clean, mechanism-independent win. All three new arms fix this
  phrase perfectly regardless of how they differ downstream, suggesting
  the shared ingredient (event-informed voicing/boundary awareness) is
  what matters here, not the specific rendering choice.
- **`rapid_letter_names`: badly broken on every arm** (0.857-1.000) --
  a letter-naming task ("a b c d e f g") is structurally unlike normal
  speech (short, disconnected, non-word units), and none of the tested
  mechanisms meaningfully help it. Out of scope for this investigation,
  flagged rather than chased.
- **`long_sustained_vowels`: Arm C actually wins here** (0.800 vs.
  1.000 for A/B/D) -- the one phrase where raw-preservation beats the
  others, though all scores are still poor (a vowel-heavy phrase with
  few obstruents to begin with, so this arm difference may not be
  meaningful).
- B and D still produce identical or near-identical hypotheses on
  several phrases (`positive_control`, `conversational`,
  `repeated_syllables`) but diverge on others (`rapid_letter_names`,
  `phrase_final_stops`) -- the earlier "B and D converge" observation
  does not hold universally.

## Final updated verdict

**Arm B (event-informed partial masking, no raw-waveform preservation
at all) is the best-supported choice on all evidence gathered this
arc.** It never showed a regression worse than a fraction of a point
on any phrase, won the `positive_control`/`phrase_final_stops` cases
cleanly, and has the lowest mean WER of any arm tested, including the
baseline. Arm D is a modest, inconsistent improvement. Arm C should not
be promoted -- worse than baseline on the full set's mean WER.

## Not yet done

- Investigating `rapid_letter_names`' universal failure (out of scope
  for this arc, a structurally different task).
- Deciding whether to normalize melody assignment to be phrase-
  independent (would remove the cross-config confound going forward,
  not attempted).
- The `MIN_SYLLABLE_DUR_S`-floor hypothesis (open since v9).
- The human listening check -- still the standing, most important item.
  WER now points clearly at Arm B as the best candidate for
  event-informed rendering, but WER remains a proxy; only a human ear
  can confirm this translates to perceived quality.

## Files (updated again)

- `v10full_wer_results.json` -- raw per-phrase, per-arm WER + hypothesis
  text for all 40 renders.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_full10/*_sung_v10full_{a,b,c,d}.wav`.
