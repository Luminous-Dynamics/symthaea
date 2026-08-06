# Controllability Audit -- Gate C: rhythm and duration control (2026-07-27)

Per `ACE_STEP_CONTROLLABILITY_AUDIT_2026-07-27.md`, run on **v1 only**
(the confirmed standing baseline -- no reason to re-test 1.5 given its
consistent underperformance across every prior gate).

## Scoping note: v1 exposes no explicit duration/tempo parameter

Unlike 1.5's `bpm`/`timesignature` fields, v1's `ACEStepPipeline` has no
structured tempo, BPM, or per-word-duration control at all -- the only
available lever is a **prompt-based tempo descriptor** in the caption
text (e.g. "slow tempo, 60 bpm, ballad" vs "fast tempo, 150 bpm,
upbeat"). This gate tests whether that text-only lever has any causal
effect on pacing, or whether prosody stays entirely internally generated
regardless of the prompt.

## Setup

Fixed lyrics ("Won't you sing along with me," for continuity with prior
gates), no melody reference (isolating the tempo-prompt variable alone,
given `audio2audio`'s own established limitations). 3 conditions
(baseline / slow / fast caption) x 3 seeds (111/222/333) = 9 renders.
Measured: voice-activity onset, voiced duration, word count and
"sing along" phrase-repetition count (via Whisper), words-per-voiced-
second as a pacing proxy.

## Result: a small, directionally-plausible effect on onset timing; no clean effect on pacing; a real accuracy cost

| Condition | Mean onset | Mean voiced dur | Mean words | Words/voiced-sec |
|---|---|---|---|---|
| baseline | 0.60s | 10.27s | 12.0 | 1.17 |
| slow | 0.73s | 11.26s | 10.7 | **0.95** |
| fast | **0.31s** | 11.39s | 13.0 | 1.14 |

- **Onset timing shows a real, directionally sensible effect**: "fast"
  starts singing almost immediately (0.31s), "slow" waits longest
  (0.73s), baseline in between (0.60s) -- consistent with an
  "upbeat"/"ballad" framing difference.
- **Words-per-voiced-second does NOT show a clean tempo effect.** "Slow"
  is indeed the lowest (0.95), but "fast" (1.14) is actually *slightly
  lower* than baseline (1.17), not higher as the "fast tempo, 150 bpm"
  descriptor would predict. The prompt lever partially works (slows
  things down) but does not reliably speed things up.
- **Real accuracy cost, matching Gate B's pattern**: baseline got the
  lyrics fully correct in 2/3 seeds (the third substituted "stay alone"
  for "sing along," a pre-existing known error mode). Both tempo
  conditions did notably worse -- only 1/3 seeds each landed close to
  correct; the rest substituted unrelated words ("Won't you say the love
  with me?", "Won't you say the long with me", "Won't you see love with
  me? Won't you say no line with me?"). Full transcripts:
  `gate_c_results.log`.

## Interpretation

v1's only duration-control lever (prompt text) shows a **real but weak
and asymmetric** causal effect -- it can plausibly delay onset and
slightly slow pacing, but doesn't reliably speed pacing up, and adding
descriptive tempo language appears to cost lyric fidelity in this small
sample. This is the same shape of finding as Gate B (a control lever
that partially works but at a real content cost) rather than either a
clean pass or a clean failure. Small sample (n=3 per condition) --
treat directional signals as suggestive, not definitive.

## Files

- `gate_c_tempo.py` -- generation script.
- `gate_c_analyze.py` -- voice-activity + transcript analysis.
- `gate_c_results.log` -- full raw output.
- Audio: `symthaea/audio_output/ace_step_gate_c_2026-07-27/` (gitignored,
  not duplicated here).

## What NOT to conclude

- n=3 per condition is small -- the onset-timing effect is the cleanest
  signal here but hasn't been replicated at larger scale.
- Sustained-vowel duration, inter-word pause, and melisma-vs-syllabic
  control (all named in the audit's original Gate C design) were not
  tested in this pass -- only tempo framing was, as the single lever v1
  actually exposes.
- Doesn't test 1.5's explicit `bpm` parameter, which (unlike v1's
  prompt-only lever) is a structured field -- an asymmetry worth noting:
  1.5 has the more principled interface for this specific control even
  though it underperformed on every other axis tested.
