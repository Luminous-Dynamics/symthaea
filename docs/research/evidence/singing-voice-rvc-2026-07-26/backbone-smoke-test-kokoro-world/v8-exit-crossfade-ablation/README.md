# v8: exit-crossfade-only ablation (2026-07-28)

Per the reviewer's precise next-step recommendation after v7b localized
v7's loss to the exit crossfade: "change nothing else... test four exit
policies." Implemented on top of the locked v7b behavior (entry-into-
consonant crossfade, source spans, F0, duration, gain all unchanged);
only the boundary where a raw (voiceless-obstruent) group EXITS into
the next WORLD-synthesized group uses a new policy. Sanity-checked:
Arm A (current policy) reproduces v7b byte-for-byte.

| Arm | Exit policy |
|---|---|
| A | Current fixed 10ms linear crossfade (unchanged from v7b) |
| B | Short (1.5ms stops / 5ms fricatives, by the exiting group's final phoneme) equal-power fade |
| C | Multiband: low band (<3kHz) keeps the full 10ms linear fade; high band holds the consonant's content until the final 25% of the window |
| D | (preserve high-frequency residual over the vowel onset) -- **not attempted this pass**, given as the most architecturally novel/risky arm and the size of what B+C already surfaced |

## Critical correction found before drawing any conclusion

Evaluating WER on all 3 obstruent-heavy phrases (not just
`consonant_clusters`, the only one v7/v7b's own writeups checked)
surfaced a real, previously-uncaught problem: **`fricative_heavy`
transcribes as "She sells T-shirts by the T-shirt" on Arm A** (and
identically on v7 and v7b -- confirmed byte-identical, confirmed
deterministic across 4 repeated Whisper calls) -- a genuine content
error, not the earlier-reported "sea shore" vs "seashore" tokenization
artifact. That tokenization-only result was v6's (pre-waveform-
preservation) render; **v7/v7b/v8-Arm-A were never actually WER-tested
on `fricative_heavy` before this pass**. This is a real regression
introduced by the waveform-preservation mechanism itself (present
identically in all of Arms A/B/C, since they share the same "preserve
original consonant waveform" core idea and differ only in exit-fade
policy) -- not something either earlier writeup got wrong, but
something neither writeup actually checked. See the correction sections
added to `../v7-waveform-preserved-obstruents/README.md` and
`../v7b-lineage-instrumentation/README.md`.

**Consequence for the pre-registered acceptance criteria**: the
reviewer's own draft threshold ("preserves perfect transcription")
implicitly assumed Arm A's own baseline WER was already 0.0 across all
3 phrases. It is not -- Arm A's real per-phrase WER is 0.0 /
0.333 ("T-shirts") / 0.143 (dropped final "it"), mean 0.159. The
correct comparison is therefore Arm B/C **against Arm A's own actual
per-phrase results**, not against an idealized 0.0.

## Result

| Arm | core centroid | exit centroid | centroid ratio | core high-band | exit high-band | high-band ratio | mean max click | mean WER |
|---|---|---|---|---|---|---|---|---|
| A (baseline) | 4160 Hz | 1843 Hz | 0.443 | 0.2883 | 0.0388 | 0.1346 | 0.152 | 0.159 |
| B | 4056 Hz | 1864 Hz | 0.460 | 0.2702 | 0.0537 | **0.1986** | 0.151 | **0.214** |
| C | 4160 Hz | 1895 Hz | 0.456 | 0.2883 | 0.0472 | 0.1637 | 0.152 | 0.159 |

Per-phrase WER (the number that actually matters here):

| Phrase | Arm A | Arm B | Arm C |
|---|---|---|---|
| consonant_clusters | 0.000 | 0.000 | 0.000 |
| fricative_heavy | 0.333 ("T-shirts"/"T-shirt") | **0.500** ("t-shirts"/"this t-shirt" -- worse) | 0.333 (identical to A) |
| phrase_final_stops | 0.143 | 0.143 | 0.143 |

**Neither B nor C clears the reviewer's exact pre-registered bar**
(double high-band retention vs. A's 0.1346 -> need >= 0.2693): B reaches
0.1986 (a real, meaningful improvement, ~48% relative gain, but short of
doubling); C reaches 0.1637 (a smaller improvement, ~22% relative gain).
Both improve centroid ratio slightly. **Click magnitude is unaffected
by either arm** (within noise of A's 0.152).

**But B and C diverge on WER, and this is the more decisive finding**:
Arm C's WER is IDENTICAL to Arm A's on every phrase (no regression at
all). Arm B's WER is WORSE on `fricative_heavy` specifically (0.500 vs
0.333) -- the short equal-power fade's bigger acoustic change comes at
a real transcription cost on this phrase. **Arm C is the safer choice
of the two**: a real, if modest, high-band retention improvement with
zero WER cost anywhere tested.

## Verdict against the pre-registered criteria

Applying the reviewer's exact rule ("accept only if it at least doubles
high-band retention, improves centroid retention, preserves perfect
transcription, and doesn't increase max click by more than 20%") **to
the corrected Arm-A baseline**: neither B nor C is accepted -- both fall
short of doubling high-band retention, and B additionally fails the
transcription-preservation clause (a real regression, not just short of
an idealized target). **This is an honest non-result relative to the
strict bar, not a null finding**: both arms move in the intended
direction (higher high-band retention, unchanged-or-better centroid
ratio), just not far enough, and C does so without cost while B trades
some intelligibility for a larger acoustic gain.

## Recommendation

- **If a v8 candidate must be chosen now**: Arm C (multiband), since it
  matches Arm A's WER exactly while delivering a real (if below-
  threshold) high-band retention gain.
- **Arm D (residual extension) remains the reviewer's most promising
  proposal and was not attempted this pass** -- given B/C's modest
  gains, it may be necessary to actually clear the pre-registered bar.
- **The newly-found `fricative_heavy` regression needs its own
  investigation**, independent of exit-crossfade policy (it's present
  identically in A/B/C) -- most likely related to this phrase's dense
  fricative content (`she SELLS SEAshells by the SEAshore`) combined
  with the proportional (non-forced-aligned) phoneme-boundary estimate
  possibly mis-locating short /s/-/ʃ/ spans, though not yet confirmed.

## Not yet done

- Arm D (preserve high-frequency residual over the vowel onset).
- Real forced-alignment-based source phone boundaries (still the
  standing accuracy limitation for the raw-waveform extraction).
- The human listening check -- still the standing, most important item.

## Update (2026-07-28): fricative_heavy root-cause investigated -- confirmed but doesn't fix WER

See `../v9-syllable-boundary-rootcause/README.md`. The equal-per-
syllable-count natural-duration split was confirmed (via direct spoken-
audio measurement) to misplace the /sh/ consonant/vowel boundary in
"seashells"/"seashore" -- the modeled raw-extraction span was mostly or
entirely vowel-onset material, not frication. A phoneme-count-weighted
fix corrects this precisely (frication content in the extracted span
rose from hf_frac 0.10/0.25 to 0.81/0.69 for seashells/seashore
respectively) but **does not fix, and mildly worsens, WER** on the
declared test set (fricative_heavy 0.333->0.667, consonant_clusters
0.000->0.250, phrase_final_stops unchanged). The "T-shirts"/"T-shirt"
mishearing persists even with the correct consonant captured -- so this
specific defect, while real, is not (or not solely) the cause of the
ASR confusion. Do not promote the v9 fix as-is; see that doc for a
plausible but unconfirmed secondary mechanism (`MIN_SYLLABLE_DUR_S`
floor interaction) and recommendation to prioritize real forced-
alignment over further heuristic tuning.

## Files

- `03v8_exit_crossfade_ablation.py` -- the 4-policy-capable renderer
  (Arm D not implemented, would slot in as a 4th `EXIT_POLICIES` entry).
- `10_v8_ablation_evaluate.py` -- the localized-metric + WER evaluation,
  including the pre-registered acceptance check.
- `*_lineage.json` -- exact lineage manifests per arm/phrase (tracked
  here).
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v8_ablation/`
  (gitignored, not duplicated here).
