# Controllability Audit -- Gate A: repeatability and identity (2026-07-27)

Per `ACE_STEP_CONTROLLABILITY_AUDIT_2026-07-27.md`: render the same
lyrics + prompt across multiple seeds, measure lyric/timbre/melody/timing
consistency and failure rate. Answers whether ACE-Step's base model
behaves like a controllable singer or invents a new performer/
interpretation each render.

## Setup

5 seeds (111/222/333/444/555), identical lyrics ("Won't you sing along
with me" -- Gate 3/4's own target phrase), identical prompt tags
("acapella, clean female vocals, no instruments, pop, a cappella"),
identical duration (15s) and all other generation parameters.
`gate_a_repeatability.py` (renders), `gate_a_analyze.py` (measurement:
Whisper transcription, `librosa.pyin` F0 contour, onset proxy, MFCC
cosine similarity as a timbre-consistency proxy). Full raw output:
`gate_a_results.log`.

## Results

| Seed | Transcript | F0 mean (Hz) | F0 std | Onset (s) | Voiced dur (s) |
|---|---|---|---|---|---|
| 111 | "Won't you sing along with me? Won't you sing along with me?" | 324.6 | 164.6 | 0.54 | 8.89 |
| 222 | "Won't you sing along with me, won't you sing along with me?" | 346.6 | 83.3 | 0.67 | 12.46 |
| 333 | **"Won't you stay alone with me? Won't you stay alone with me?"** | 227.3 | 144.1 | 0.58 | 9.47 |
| 444 | "Won't you sing along, won't you sing along with me? Won't you sing along with me?" | 260.2 | 109.1 | 0.03 | 11.55 |
| 555 | "Won't you sing along with me?" | 447.5 | 126.5 | 0.03 | 12.13 |

- **Lyric consistency: 4/5 verbatim (80%).** Seed 333 substituted "sing
  along" -> "stay alone" -- a real content miss, not a minor
  mispronunciation (different words entirely). Phrase-repetition count
  within the fixed 15s window also varies (1x/2x/3x across seeds) --
  the model doesn't just re-sing the line at a fixed cadence, it decides
  a different musical structure each time.
- **Melody/register consistency: LOW.** F0 mean ranges 227-447Hz across
  seeds (range 220Hz, coefficient of variation 0.238) -- nearly an
  octave between the lowest and highest mean pitch, from *identical*
  text+tag conditioning. **This is the headline finding**: with no
  explicit melody control, ACE-Step's base model invents a materially
  different melodic register/interpretation on every render, not a
  fixed "score."
- **Timing consistency: moderate.** Onset ranges 0.03-0.67s (small in
  absolute terms) but voiced duration ranges 8.89-12.46s within the same
  15s window -- real variation in how much of the clip is actually sung
  vs. left as instrumental-style padding/silence.
- **Timbre-consistency proxy (MFCC cosine similarity): mean 0.9502,
  range 0.9054-0.9805** across all 10 pairs -- comparatively high and
  stable. Caveat, disclosed in the analysis script itself: MFCCs aren't
  fully pitch-invariant, so part of this similarity/dissimilarity is
  confounded with the F0 differences above, not a clean isolated
  timbre-identity measurement. Treat as a rough proxy, not a validated
  speaker-verification result (no speaker-embedding model was used for
  this bounded check).
- **Failure rate: 0/5** -- every render produced valid, substantially
  intelligible audio (per the sanity-check pattern established in the
  base verification).

## Verdict

**Lyric content is reasonably (not perfectly) stable across seeds;
melody/register is not stable at all.** This is quantified evidence, not
a guess, for the audit doc's outcome #2 over #1: ACE-Step's base model
"follows rough musical instructions but not exact scores" -- even basic
melodic identity doesn't hold without any explicit melody conditioning,
which is exactly what Gate B needs to test directly (can a *requested*
contour be imposed, given the base model's default behavior is to
invent one freely). Raises Gate B's priority rather than lowering it --
if Gate A had shown melody was already stable by default, Gate B would
mostly confirm a lower-stakes question; instead Gate A shows the model's
unconstrained default is high melodic variance, so Gate B needs to show
whether that variance can be *suppressed* by explicit control.

## Files

- `gate_a_repeatability.py`, `gate_a_analyze.py` -- generation + analysis (re-runnable)
- `gate_a_results.log` -- full raw output
- Audio: `symthaea/audio_output/ace_step_gate_a_2026-07-27/` (gitignored, not duplicated here)

## What NOT to conclude

- n=5 seeds, one phrase, one prompt -- not a statistically powered study,
  a bounded first look per the audit's own "don't add adapters before
  measuring deficiencies" discipline.
- The MFCC timbre proxy is not a validated identity metric; don't cite
  the 0.95 mean similarity as proof of stable voice identity without a
  real speaker-embedding follow-up if that question becomes load-bearing.
- Doesn't test whether melody INSTABILITY is fixable via prompt
  engineering (e.g. requesting a specific key/tempo in the tag string)
  -- untested here, would be a cheap thing to check before assuming a
  full control-bridge architecture is required.
