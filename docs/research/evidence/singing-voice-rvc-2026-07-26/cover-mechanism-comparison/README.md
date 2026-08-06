# Direct control-mechanism comparison: v1 unconditioned vs. v1 audio2audio vs. 1.5 cover (2026-07-27)

Per user direction, Task 2 of the two-part stability+comparison plan:
compare v1's `audio2audio` against 1.5's dedicated `cover` task using the
**same real vocal-shaped references** (no sine tones), same 3 seeds
(111/222/333), same 2 phrases ("won't you sing along with me", "I love
the summer breeze tonight"), same 2 unmistakable melodies (ascending,
alternating leap) built for Gate B (`espeak-ng` "laaa", pitch-shifted per
note). v1's unconditioned baseline rendered once per (seed, phrase) since
it doesn't depend on melody. 1.5's runs used `task_type="cover"`,
`src_audio=<vocal reference>`, `audio_cover_strength=0.5`, with the CoT
confound removed (per the controlled-replication methodology) and one
retry per failed render (per the stability gate's finding that failures
are non-deterministic, not seed-specific -- none of these 12 needed a
retry, all succeeded on attempt 1).

## Headline result: 1.5's cover mode failed catastrophically on every axis tested

### Lyric preservation

| Condition | Renders | Preserved intended lyrics |
|---|---|---|
| v1 unconditioned | 6 | 5/6 (1 substitution, "sing along"->"stay alone") |
| v1 audio2audio | 12 | ~10/12 core content preserved (some stutter/garble, matching Gate B's earlier pattern) |
| **1.5 cover** | 12 | **0/12** |

**Every single 1.5-cover render produced entirely unrelated, hallucinated
lyrical content** -- none contain "won't you sing along with me" or "I
love the summer breeze tonight" in any recognizable form. Examples:
"I'm sorry I'm sorry I'm sorry I'm sorry.", "The dream you know to be
your child...", "I'm standing in this pile of waste...", "Thank you very
much, sir, for coming here this afternoon...", plus 3 renders that
transcribed as empty (no detected speech) or a single stray word
("Music"). Full transcripts: `cover_15_transcripts.log` vs.
`cover_v1_transcripts.log`.

### Melody adherence (F0 vs. requested contour)

| Condition | Mean absolute error (semitones) | Correlation pattern |
|---|---|---|
| v1 audio2audio | 1.6-8.2st (most renders) | Mixed, no consistent sign this batch (differs from Gate B's original "consistently negative" ascending finding -- see caveat below) |
| **1.5 cover** | **7.8-25.7st** (several 2+ octaves off) | Scattered, no consistent sign or magnitude, several implausibly low registers (65-78Hz) unrelated to the requested "female vocals" caption |

1.5-cover's melody error is not just larger than v1's -- it's in a
different regime entirely (multiple full octaves off, vs. v1's errors
staying mostly within one octave). Combined with the complete lyric
failure, this indicates 1.5's cover mode with this specific
reference/strength combination produced output almost entirely
disconnected from **every** conditioning signal supplied (lyrics, melody,
and even the requested vocal timbre/register).

## Important caveat on v1 audio2audio's melody numbers, disclosed not glossed over

This batch's v1 audio2audio F0 correlations were **more mixed** than Gate
B's original finding (which reported "ascending's F0-contour correlation
... consistently negative across all 6 renders, -0.75 to -0.34, zero
exceptions"). Here, with a different set of seeds and phrases, ascending
shows both positive (0.89, 0.85, 0.73) and negative (-0.76, -0.72)
correlations, averaging weakly positive. This is worth stating plainly:
**Gate B's "consistently negative" finding was based on n=6 renders (one
phrase, one melody-type family) and does not fully replicate with a
different phrase/seed set** -- the underlying reality is closer to "v1's
melody-following is unreliable and noisy" than "v1's melody-following is
reliably wrong in one direction." This doesn't change Gate B's overall
verdict (v1's `audio2audio` doesn't deliver reliable score-following
control), but the *mechanism* of that unreliability is better described
as high variance than a consistent bias.

## Verdict

**Outcome C from the audit's decision framework**: control did not
improve, and it came at a much steeper cost than v1's already-imperfect
`audio2audio` -- 1.5's cover mode, as tested here, should not be
considered for the vocal critical path. This is a stronger, more
decisive negative result than "control remains weak" alone -- content
correctness itself broke down entirely, not just melody precision.

**Important scope limitation, not swept under the rug**: this tested
*one* reference construction (a pitch-shifted TTS "la" vocalization,
never a real sung/hummed performance or a genuine musical recording) at
*one* strength (0.5). Cover mode is explicitly designed to "transform
existing audio into a new style" -- it may be fundamentally
out-of-distribution for a synthetic, non-musical, phoneme-level probe
tone in a way that a real song excerpt would not be. **This result rules
out "cover mode with this specific probe construction," not "cover mode
in general."** A follow-up with a genuinely musical/sung reference (or a
strength sweep, mirroring the earlier v1 sine-vs-vocal / strength-sweep
methodology) would be needed before concluding cover mode is unusable
outright -- but per the user's own bounded-task framing, that follow-up
is not undertaken here.

## What this means for the overall audit

Combined with the controlled-replication finding (1.5-base behind v1 on
both diction and reliability for plain text2music) and this cover-mode
result, **ACE-Step 1.5 has not yet demonstrated any established advantage
over v1** for this project's needs, on any axis tested so far: not
diction, not reliability, not melody control via its dedicated cover
mechanism. v1 remains the standing baseline. Whether a better-constructed
reference could redeem 1.5's cover mode is a genuinely open question, not
a closed one -- but the burden of proof has not yet been met.

## Files

- `gate_cover_v1.py`, `gate_cover_15.py` -- generation scripts.
- `analyze_cover_v1.py`, `analyze_cover_15.py` -- F0 analysis scripts.
- `cover_v1_transcripts.log`, `cover_15_transcripts.log` -- Whisper transcripts.
- `cover_v1_f0.log`, `cover_15_f0.log` -- per-note F0 analysis.
- Audio: `symthaea/audio_output/cover_mechanism_comparison_2026-07-27/`
  (30 files: 18 v1 + 12 1.5-cover; gitignored, not duplicated here).

## What NOT to conclude

- Don't conclude cover mode is unusable in general -- only that this
  specific out-of-distribution reference type failed badly.
- Don't treat the v1 audio2audio melody-correlation sign flip (vs. Gate
  B) as evidence Gate B was wrong -- both results are real; the honest
  synthesis is "high variance," not "one of them is the true number."
- `audio_cover_strength=0.5` was not swept -- unlike the earlier v1
  strength sweep that found a sharp lyrics-vs-strength cliff, no
  equivalent sweep was run for 1.5's cover mechanism in this pass.
