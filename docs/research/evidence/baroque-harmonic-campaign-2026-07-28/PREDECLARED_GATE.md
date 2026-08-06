# Baroque harmonic-syntax campaign — predeclared success gate

**Written and committed before any comparative old-vs-new analysis of the
16-seed cohort.** At the time of writing, the only facts observed about
`campaign_summary.json` are: 32 entries (16 seeds x 2 variants), zero
FluidSynth/MIDI-export failures, all 32 WAVs valid RIFF headers. No
per-metric old-vs-new comparison has been made. This is the decision rule
that will be applied *after* this file is committed, not derived from
looking at the results first.

Cohort: seeds 1-16, `bars=8`, `BaroqueSuite` old (fixed I-IV-V-I,
`BAROQUE_SUITE_COMPATIBILITY_PROGRESSION`) vs. new (`ProgressionSpec::Grammar`).
Data: `audio_output/baroque_campaign_2026-07-27/` (git commit
`3bdb7c969a5baed16b2a816422a6e829dd4b5d70`, per `provenance.json`).

## Gate criteria

Each criterion below maps to a specific field in `harmony_report.json`
(written by `symthaea_music_theory::harmony_verifier::verify`). "Majority
of seeds" means `>= 9` of 16 unless stated otherwise.

1. **No catastrophic failures.** Operationalized as: zero crashes, zero
   empty `Score`s, zero WAV render failures, for either variant, across
   the full cohort. Already satisfied for the raw pipeline (confirmed:
   0/32 failures). Extended definition for THIS gate: no seed where a
   voice that has notes in the old variant has ZERO notes in the new
   variant (a sign the new progression broke that voice's realization
   entirely), or vice versa.

2. **Strong-beat harmonic conflicts: new <= old, majority of seeds.**
   Sum `strong_beat_chord_conflicts` across all 4 voices
   (`voice_metrics[*].strong_beat_chord_conflicts`) per seed per variant.
   Count how many of the 16 seeds have `new_total <= old_total`. Passes
   if `>= 9/16`.

3. **Cadence closure: new >= old, majority of seeds.** Compare the
   `"final"`-boundary entry in `cadences[].closure` (0.0-1.0) per seed.
   A seed with no detected final cadence in either variant is excluded
   from the count (denominator shrinks accordingly, but the pass
   threshold stays a strict majority of the seeds that HAVE a
   comparison). Passes if `new.closure >= old.closure` in a strict
   majority of comparable seeds.

4. **Progression diversity: new > old, in aggregate.** Compare
   `progression_diversity.distinct_transitions /
   progression_diversity.total_transitions` (a normalized diversity
   ratio, not the raw count, since old/new pieces can end up with
   different `inferred_bar_count` totals). Passes if the MEAN ratio
   across all 16 seeds is strictly higher for `new` than `old`. (Chosen
   as a whole-cohort aggregate rather than per-seed majority because
   diversity is exactly what `ProgressionSpec::Grammar` is designed to
   increase over a repeating fixed archetype — a consistent, even if
   individually small, aggregate lift is the relevant signal, not
   winning every single seed.)

5. **No substantial voice-leading degradation, Melody and CounterMelody,
   majority of seeds.** For each of these two voices, compare
   `mean_voice_leading_distance` (semitones) between variants per seed.
   "Substantial degradation" = new value is `> 25%` larger than old
   (a predeclared threshold, not derived from this cohort's data —
   chosen as a round number representing a clearly audible, not
   marginal, change in melodic smoothness). A voice absent in a given
   seed (no notes, `None`) is excluded from that seed's count for that
   voice. Passes if fewer than a majority of comparable seeds show
   substantial degradation, for BOTH voices independently.

6. **Blinded listening preference: functional (new) preferred in a
   meaningful majority of pairs.** Evaluated only after the user's blind
   listening session (see `HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md`
   and the campaign harness doc comment for the blinding protocol still
   to be built). "Meaningful majority" is read as `>= 10/16` pairs
   preferring the functional variant, with "no meaningful difference"
   and "both flawed" responses recorded but not counted toward either
   side (so the majority is computed over decided pairs, with the total
   decided-pair count also reported honestly rather than hidden).

## Overall verdict rule

The functional-harmony route is considered validated by this campaign if
criteria 1, 2, 4, and 6 all pass, AND at least one of {3, 5} passes for
each of its sub-parts. This is deliberately not "all 6 must pass
unanimously" — criterion 5 in particular (voice-leading) is exactly
where the earlier single-seed pilot found a real, disclosed trade-off
(seed 7's counter-voice becoming more angular even as the accompaniment
improved), so a gate that requires perfection on every axis would
likely reject a genuinely net-positive change over a genuinely net-flat
one. If the gate fails, the specific failing criterion becomes the input
to the "diagnose and fix the dominant repeated failure" step, not a
reason to abandon the functional-harmony direction outright.

## What this gate does NOT decide

- It does not decide whether any individual seed's rendering "sounds
  good" — that is what the blinded listening session is for.
- It does not average away or hide per-seed variance. The aggregate
  report (produced after this gate is applied) will show the full
  per-seed breakdown for every criterion, not just the majority-vote
  outcome, so a reader can see whether a "pass" was 9/16 (marginal) or
  16/16 (unanimous).
