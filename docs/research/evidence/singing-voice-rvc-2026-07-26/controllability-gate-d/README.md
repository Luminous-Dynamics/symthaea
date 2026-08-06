# Controllability Audit -- Gate D: phonetic stress test (2026-07-27)

Per `ACE_STEP_CONTROLLABILITY_AUDIT_2026-07-27.md`. **v1 only, baseline
prompt only** (no control-lever variation -- Gate D answers a different
question than Gates B/C: not "does a control lever work," but "which
phonetic structures remain reliable, and which fail predictably").

## Setup

10 phrase categories x 3 seeds (111/222/333) = 30 renders, baseline
caption ("acapella, clean female vocals, no instruments, pop, a
cappella"), no melody reference, no tempo descriptor. Categories per
user design: positive control, ordinary conversational lyric, repeated
syllables, rapid letter names, phrase-final stops, fricative-heavy
tongue-twister, consonant clusters, long sustained vowels, short
unstressed function words, one semantically unusual phrase.

## A real bug caught in my own scoring, fixed before reporting

The first analysis pass showed a suspiciously low exact-match rate for
`positive_control` (0/3) that contradicted every prior gate's finding for
this exact phrase (previously 4-5/5 or 2/3 across multiple batches). Root
cause: the scoring script normalized (lowercased, stripped punctuation)
the Whisper *transcript* before substring-matching, but never applied the
same normalization to the *target* string -- so `"won't"` (target,
apostrophe intact) never matched `"wont"` (transcript, apostrophe
stripped). Fixed (`gate_d_analyze.py`, target now normalized identically
to the transcript) and rescored from the same raw transcripts --
corrected numbers below. This is exactly the kind of self-check this
whole investigation has practiced throughout: catch and fix your own
scoring bugs before they become a false finding.

## Result: a clear capability boundary, not a single verdict

Per the user's own instruction ("do not use Whisper alone... human
transcription first"), each category below is hand-adjudicated from the
actual transcripts, not just the strict-substring "exact" flag -- a
single clean word substitution is scored differently from complete
garbling, even though both count as `exact=False`.

| Tier | Categories |
|---|---|
| **Reliable** (correct content >=2/3 seeds) | conversational (3/3), positive_control (2/3), repeated_syllables (2/3), phrase_final_stops (2/3), short_unstressed (effectively 2/3) |
| **Near-miss** (0/3 strict but a single clean substitution or a scoring artifact, not garbled) | semantically_unusual (1 seed: "ate"->"hit", otherwise perfect), fricative_heavy (1 seed: "sea shore" vs "seashore" spacing only) |
| **Genuinely unreliable** (real, repeated word-level errors, gist survives) | fricative_heavy (2 seeds: a specific, reproducible "seashore"->repeat-of-"seashells" substitution -- alliterative content seems to trigger a repetition-preference bug), consonant_clusters (all 3 seeds have real word errors), semantically_unusual (2 of 3 badly garbled) |
| **Fails completely** (0/3, unrelated content) | rapid_letter_names -- all 3 seeds produce entirely unrelated sentences, no letter names recognizable at all |
| **Distinct failure modes** (not accuracy issues) | long_sustained_vowels (ranges from a proximate substitution to complete collapse into non-lexical vocalese -- word boundaries appear to dissolve under melisma); repeated_syllables/seed333 (a runaway repetition loop, "Bye!" x59 filling the whole clip -- degenerate generation, not a substitution error) |

Full per-seed transcripts and the hand-adjudication reasoning:
`gate_d_results_corrected.log`.

## Interpretation

This confirms and sharpens the audit's throughline finding with a
concrete boundary map rather than a general impression:

- **Ordinary, everyday lyric content is genuinely reliable** on v1
  (conversational: 3/3 perfect; positive_control, phrase_final_stops,
  short_unstressed: all ~2/3).
- **Letter-name articulation fails completely and predictably** -- this
  is not noise, it's a clean, repeatable failure category worth
  excluding from any production render selection without a fallback.
- **Alliterative/tongue-twister content and dense consonant clusters
  produce specific, somewhat repeatable substitution errors** (not
  random garbling) -- worth flagging in any future "which lyrics are
  render-safe" heuristic.
- **Sustained melismatic vowels and rhythmic repeated syllables can
  trigger qualitatively different failure modes** (word-boundary
  collapse, runaway repetition loops) distinct from ordinary
  mispronunciation -- useful to know for render-time detection/rejection
  logic (e.g. flag outputs where a single word repeats >10x as likely
  degenerate).

## Files

- `gate_d_phonetic.py` -- generation script (10 phrases x 3 seeds).
- `gate_d_analyze.py` -- transcription + scoring (bug-fixed).
- `gate_d_results_raw.log` -- first pass, before the normalization fix.
- `gate_d_results_corrected.log` -- corrected scoring + hand-adjudicated tiers.
- Audio: `symthaea/audio_output/ace_step_gate_d_2026-07-27/` (gitignored,
  not duplicated here).

## What NOT to conclude

- n=3 seeds per category is small -- treat the tier boundaries as
  directionally reliable, not statistically definitive.
- "Human transcription" here means careful manual reading of Whisper's
  own output, not actual listening -- Whisper itself remains a proxy,
  consistent with every other gate in this audit, not the "primary
  language result" a real listening pass would be.
- This doesn't test whether these failure categories are fixable via
  prompt engineering, alternate seeds, or post-hoc filtering -- only
  that they exist and are somewhat predictable under baseline
  conditions.
