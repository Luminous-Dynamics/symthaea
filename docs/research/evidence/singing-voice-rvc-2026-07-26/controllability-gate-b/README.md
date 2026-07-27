# Controllability Audit -- Gate B: melody control (2026-07-27)

Per `ACE_STEP_CONTROLLABILITY_AUDIT_2026-07-27.md`. Tests ACE-Step v1's
`audio2audio` mechanism (SDEdit-style: reference audio -> DCAE latent,
diffusion starts from `sigma_max=(1-ref_audio_strength)`, i.e. partial
noising toward the reference -- a coarse "how much to preserve" knob, not
a note-by-note score API). Two passes: a 4-melody pilot at one strength,
then a strength sweep after the pilot's headline finding demanded it.

## Setup

`make_melody_refs.py` synthesizes 4 unmistakable reference melodies as
pure sine-tone sequences (monotone/ascending/descending/leap, 5 notes,
1.2s each). `gate_b_melody.py` renders "one two three four five"
(one syllable per note) with each melody as an `audio2audio` reference
(`ref_audio_strength=0.35`), 2 seeds, plus one shared unconditioned
baseline per seed (rendered once, not once per melody, since it doesn't
depend on melody) -- 10 renders total. `gate_b_analyze.py` extracts a
per-note median F0 (`librosa.pyin`, 1.2s windows matching the reference
note structure) and transcribes with Whisper.

## Pilot result (strength=0.35): headline finding is an intelligibility collapse, not a melody-accuracy number

**Every single conditioned render (8/8) failed transcription entirely**
(empty or ". . . ." placeholder output). **Both unconditioned baselines
transcribed perfectly** ("1, 2, 3, 4, 5"). This is the most important
result from the pilot -- before even asking "how accurate is the
melody," the answer to "did conditioning preserve the lyrics at all" is
no, categorically, at this strength.

Pitch-tracking on the (unintelligible) conditioned renders showed a
genuine, if noisy and unreliable, directional signal:
- **Monotone**: near-perfect (0.0 and 0.6 semitone mean error) -- but
  trivial, matching a flat target requires no real contour-following.
- **Ascending/descending**: correct overall direction in 3/4 seed-melody
  pairs, but wildly inconsistent per-note detail (mean absolute error
  1.6-7.9 semitones) and **poor cross-seed consistency** (mean per-note
  seed-to-seed difference 10.2 semitones for ascending, 6.1 for
  descending) -- the signal that IS there isn't reliable.
- **Leap** (alternating low/high): the alternation direction is present
  but consistently **undershoots the requested interval by ~7-8
  semitones at the high notes** -- contour correlation only 0.20-0.34.
  Cross-seed consistency here is high (mean diff 0.4 semitones) but
  that's two seeds *agreeing on the same undershoot*, not two seeds
  reproducing the target.

Full numbers: `gate_b_pilot_results.log`.

## Strength sweep: resolves the ambiguity, but not in ACE-Step's favor

Before concluding "ACE-Step trades lyrics for melody control" (a real,
important finding if true, matching the audit's own named failure mode),
swept `ref_audio_strength` down on one melody (ascending, seed 111) to
rule out the simpler explanation that 0.35 was just too aggressive:

| Strength | Transcript | Per-note F0 (Hz) |
|---|---|---|
| 0.05 | "One, two, three, four, five, one, two, three, four, four, five" -- intact | 292.0, 449.0, 502.5, 441.3, 493.9 -- no ascending trend, just scattered/high |
| 0.10 | "One, two, three, four, five. One, two, three, four, five." -- intact | None, None, 245.5, 219.4, 246.9 -- 2/5 unvoiced, no clear trend |
| 0.15 | empty | all None (undetectable) |
| 0.20 | empty | all None (undetectable) |

(target ascending sequence: 261.63, 293.66, 329.63, 349.23, 392.00 Hz)

**There is a sharp cliff between 0.10 and 0.15**, not a gradual
tradeoff curve. Below it, lyrics survive but show **no discernible
melody-following at all** (the F0 pattern looks like Gate A's
unconditioned high-variance default, not a tracked ascending line).
At or above it, lyrics vanish entirely and F0 becomes undetectable.
**No strength tested gives both intact lyrics and requested-contour
adherence.**

## Real methodological finding, not just a negative result

The most likely explanation isn't "ACE-Step can't follow melody" in the
abstract -- it's that **a bare sine-tone reference is a poor conditioning
signal for this specific mechanism**. `audio2audio`'s SDEdit blend pulls
the output toward resembling the reference's *actual audio content*, not
an abstracted "pitch contour to sing." A pure, non-vocal sine tone has no
singing in it at all -- at higher strength, the mechanism does what it's
designed to do (make the output resemble the reference more), and the
reference contains no vocals, so vocals (and with them, the lyrics)
disappear. At low strength, the reference's influence is too weak to
imprint even the pitch trajectory, and the model falls back to its own
default (Gate A-style) unconditioned pitch invention.

**This means the pilot doesn't yet answer "can ACE-Step follow a
melody" cleanly** -- it answers "can a bare sine-tone `audio2audio`
reference make it follow a melody," and the answer to that narrower
question is no, at any strength tested. A properly designed follow-up
would use a reference that itself contains *sung or hummed* audio with
the desired contour (even low-quality), not a pure tone -- giving the
SDEdit blend something with real vocal characteristics to preserve while
nudging pitch, rather than asking it to preserve a non-vocal reference's
character.

## Verdict against the audit's pre-registered pass criteria

- Lyrics preserved in >=4/5 seeds: **failed** (0/8 conditioned renders at
  strength=0.35; the two working strengths, 0.05/0.10, preserve lyrics
  but show no melody-following, so they don't count as "conditioned
  successfully" either).
- Four contour classes distinguishable: partially (monotone clearly
  flat; leap distinguishable as *some* alternation but wrong magnitude;
  ascending/descending noisy).
- Correct conditioning beats unconditioned/mismatched controls: not
  cleanly established -- the mismatched-reference control wasn't run in
  this bounded pilot (deferred, as documented).
- Cross-seed melody variance drops >=50%: **failed** for
  ascending/descending (variance is large and inconsistent across seeds).
- No severe intelligibility regression: **failed decisively** -- this is
  the headline result.
- Leap pattern reflected in F0: **failed** -- consistent ~7-8 semitone
  undershoot.

## What this means for the audit's decision framework

Per `ACE_STEP_CONTROLLABILITY_AUDIT_2026-07-27.md`'s three outcomes, this
pilot's result (with its probe-design caveat) is most consistent with
**outcome 2/3 boundary, not outcome 1** -- `audio2audio` with a naive
reference does not currently look like a usable score-control mechanism
for this project without a fundamentally different reference-construction
approach (real sung/hummed reference audio, not synthetic tones). Doesn't
rule out outcome 1 for a *properly constructed* reference -- that's the
next thing to test, not yet done.

## Files

- `make_melody_refs.py` -- synthesizes the 4 sine-tone references.
- `gate_b_melody.py` -- the 10-render pilot (2 seeds x 4 melodies + 2
  shared unconditioned baselines).
- `gate_b_analyze.py` -- F0/transcription analysis for the pilot.
- `gate_b_strength_sweep.py` -- the 4-render strength sweep on ascending/seed111.
- `gate_b_pilot_results.log`, `strength_sweep_results.log` -- raw output.
- Audio: `symthaea/audio_output/ace_step_gate_b_2026-07-27/` (renders +
  the 4 synthetic reference tones themselves, gitignored, not duplicated
  here).

## Closing control: a VOCAL-shaped reference, not a sine tone (2026-07-27, later same day)

Per user direction: before spending more time on sine-tone strength
tuning, run one small, tightly-scoped closing check to find out whether
v1 needs a *vocal-shaped acoustic reference* specifically, rather than
merely an F0 contour in the abstract. Built a genuinely vocal reference
via `espeak-ng` (`"laaa"`, a real formant-bearing synthesized vocalization,
not a sine tone), pitch-shifted per note with `librosa.effects.pitch_shift`
(phase-vocoder, preserves timbre/formants reasonably well) to hit the
`ascending` and `leap` target frequencies (`make_vocal_melody_refs.py`;
pitch-shift accuracy verified directly against target Hz before spending
render compute). Rendered 2 melodies x 2 seeds x 3 strengths (0.05/0.10/
0.15) = 12 renders (`gate_b_vocal_ref.py`), analyzed the same way as the
sine-tone pilot (`gate_b_vocal_analyze.py`).

### Result: lyrics-collapse problem SOLVED, melody-following problem NOT solved

**12/12 conditioned renders transcribed successfully** ("1, 2, 3, 4, 5" or
"One, two, three, four, five!" in every case) -- a complete reversal from
the sine-tone pilot's 0/8. This directly confirms the diagnosis: the
sine-tone reference's *lack of vocal content* (not the strength value)
caused the earlier collapse. A vocal-shaped reference preserves lyrics
even at strengths (0.10, 0.15) that fully destroyed them with a bare tone.

**But melody adherence remains poor, and the failure mode is
informative.** `ascending`'s F0-contour correlation with the requested
rising trend was **consistently NEGATIVE across all 6 renders**
(-0.75, -0.48, -0.70, -0.34, -0.75, -0.75) -- not just weak or noisy, but
reliably *anti*-correlated with the request. `leap`'s correlation was
weak/inconsistent (-0.67, -0.24, -0.08, 0.59, 0.07, 0.07 -- one positive
value among six, not a reliable signal). Mean absolute pitch error stayed
large throughout (3.6-11.8 semitones). Full numbers:
`vocal_reference_control_results.log`.

### v1 closing verdict (Gate B v1 investigation now frozen)

**v1's `audio2audio` mechanism, at the strengths needed to preserve
lyrics, conveys coarse vocal-vs-nonvocal character but not fine-grained
pitch trajectory.** This is no longer attributable to a probe-design flaw
(the earlier sine-tone caveat) -- a proper vocal reference was used and
still failed to transfer melody, while cleanly succeeding at the
narrower "is this a vocal performance" signal. Likely explanation,
consistent with the mechanism's own SDEdit design: at low enough
strength to keep `sigma_max` high enough for the model to still generate
its own vocal content freely, the specific per-frame pitch information
in the reference latent doesn't survive the near-full-noise process,
while the reference's coarser spectral/vocal character partially does.

**Per the audit's decision framework, this rules out outcome 1 (v1 as a
sufficient-controls foundation) for melody specifically**, and supports
moving to ACE-Step 1.5's dedicated cover/reference-conditioning mechanism
(explicitly designed to preserve musical structure while regenerating
performance, a different mechanism than v1's generic resemble-this-audio
blend) as the next real test, per the audit doc's 1.5 section -- not
another v1 sine-tone or strength-tuning pass.

## Files (added this pass)

- `make_vocal_melody_refs.py` -- builds the vocal-shaped references.
- `gate_b_vocal_ref.py` -- the 12-render vocal-reference control.
- `gate_b_vocal_analyze.py` -- F0/transcription analysis.
- `vocal_reference_control_results.log` -- raw output.

## Next (not yet done)

Gate B on v1 is closed. Per the audit doc: move to ACE-Step 1.5's cover/
reference-conditioning mechanism as the primary next melody-control test
(a genuinely different mechanism than v1's `audio2audio`, not another
pass on the same one), using real sung/hummed references from the start.
Gates C and D remain unstarted on v1.
3. Gates C and D remain unstarted.
