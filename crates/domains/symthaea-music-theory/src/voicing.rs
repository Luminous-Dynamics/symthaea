// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Voice leading: choose SMOOTH per-chord voicings instead of restating each
//! chord in a fixed register.
//!
//! Without this, consecutive chords in a progression each get voiced
//! independently at a fixed octave — so a I→IV move can leap an octave for no
//! musical reason. Classical voice leading keeps common tones in place and
//! moves the rest by the smallest possible distance; that's the difference
//! between chords that *connect* and chords that just *restate*.

use crate::chord::Chord;
use crate::pitch::{Pitch, PitchClass};

/// Choose the bass pitch for `chord`.
///
/// - If `prev` is `None` (the first chord) or `force_root` is set (cadential
///   emphasis — the piece's opening and closing chords), the root is used.
/// - Otherwise every chord tone is a candidate: a genuinely smooth bass line
///   uses inversions (e.g. landing on the third when it's a step away rather
///   than leaping to the root), which is exactly what real basslines do.
///
/// Search stays within one octave of `target_octave` so the register never
/// drifts even when a nearer pitch class sits an octave off.
///
/// # Do NOT make this melody-aware without reading this first (2026-07-31)
///
/// This chooses with no reference to the melody, which sounds like an obvious
/// gap: `validate_parallel_motion` reports 461 Fatal Bass-vs-Melody /
/// Bass-vs-CounterMelody parallels, and the melody is already in the score by
/// the time this runs (`realize_melody` precedes `realize_bass`).
///
/// Adding parallel + consonance + crossing penalties here WORKS on the
/// measurements — total Fatal issues 4859 -> 4072, with StrongBeatConsonance
/// 3173 -> 2656, the largest single improvement measured across this whole arc.
/// It was still reverted, twice, because it breaks
/// `composer::tests::development_dna_gives_each_style_its_own_departure`:
/// Tango's sequential development stops travelling directionally
/// (`[83.0, 83.25, 80.0, 80.0]`).
///
/// THE MECHANISM, since it is not obvious: `apply_development_style` runs AFTER
/// the bass and treats "the section's real bass [as] the cantus the new line
/// must obey" (composer.rs:943-948), re-fitting the MELODY against it through
/// the species fitter. So changing the bass changes the melody, in exactly the
/// span that test measures. The melody is emitted before the bass but is not
/// final until after it.
///
/// Two things follow, both worth knowing before trying again:
///
/// 1. Weight tuning is not the lever. Penalties of 150 and 500 produce
///    BIT-IDENTICAL defect counts — with only ~3 candidate chord tones and
///    motion <= 12 semitones, anything above ~50 is already a hard filter.
/// 2. `counterpoint::fit_against` — the fitter that pass uses — ALREADY avoids
///    parallels against the bass, but only within developed spans. So a
///    melody-aware bass and the fitter would be two passes negotiating the same
///    relationship from opposite ends.
///
/// MEASURED 2026-07-31, settling that: **146 of the 461 parallels (32%) occur in
/// the 10 styles whose `DevelopmentDna` is `Classic`**, where
/// `apply_development_style` returns immediately (composer.rs:915) so the fitter
/// NEVER runs. The other 315 are in the 19 non-Classic styles — and even there
/// the fitter only touches `SectionRole::B`/`C` spans, so a further unmeasured
/// share of those is unfitted too.
///
/// So this is BLOCKED, not a dead end, and the block is narrower than it looked.
/// The conflict exists only where a later pass re-derives the melody FROM the
/// bass. The principled rule is "do not optimize the bass against the melody in
/// spans where something downstream will re-fit the melody against that bass" —
/// an ordering constraint, not a special case. At minimum the 10 Classic-DNA
/// styles can take melody-aware bass selection today, with no possibility of
/// disturbing a fitter that never runs for them.
///
/// What it costs: `realize_bass_measures` receives no spec or DNA today, so that
/// flag has to be threaded through 7 call sites — the same plumbing the reorder
/// in `e613958c14` was deliberately scoped to avoid. Worth doing on purpose; not
/// worth bolting on to a diagnostic pass.
///
/// Acceptance gate for any future attempt: that Tango test, unweakened.
pub fn lead_bass(prev: Option<Pitch>, chord: Chord, target_octave: i32, force_root: bool) -> Pitch {
    let Some(prev) = prev else {
        return Pitch::new(chord.root, target_octave);
    };
    if force_root {
        return nearest_octave_pitch(chord.root, prev, target_octave);
    }
    chord
        .pitch_classes()
        .into_iter()
        .map(|pc| nearest_octave_pitch(pc, prev, target_octave))
        .min_by_key(|p| (p.midi() as i32 - prev.midi() as i32).abs())
        .expect("chord has at least one tone")
}

/// [`lead_bass`], but aware of the MELODY it will sound against.
///
/// Only safe where nothing downstream will re-derive the melody FROM this bass —
/// see [`lead_bass`]'s doc for the full mechanism and the three reverted
/// attempts that established it. `realize_bass` applies it per section using
/// that rule; it is not a drop-in replacement.
///
/// Scores three things against the melody on the same footing as motion:
/// a parallel perfect fifth/octave, a dissonance, and crossing above it.
/// Weights are derived from `lead_upper`'s rather than guessed — there 500 sits
/// against a cost summing 2-3 voices (max ~24-36); here the cost is one voice's
/// motion (max ~12), so ~150 is the equivalent proportion. Measured: 150 and 500
/// give bit-identical results, because with ~3 candidate tones anything above
/// ~50 is already decisive. 150 is kept as the honest value, not the lucky one.
///
/// Passing `None` for either melody pitch reproduces [`lead_bass`] exactly.
pub fn lead_bass_against_melody(
    prev: Option<Pitch>,
    chord: Chord,
    target_octave: i32,
    force_root: bool,
    prev_melody: Option<Pitch>,
    melody: Option<Pitch>,
) -> Pitch {
    let Some(prev_pitch) = prev else {
        return Pitch::new(chord.root, target_octave);
    };
    if force_root {
        return nearest_octave_pitch(chord.root, prev_pitch, target_octave);
    }
    let (Some(pm), Some(m)) = (prev_melody, melody) else {
        return lead_bass(prev, chord, target_octave, force_root);
    };
    const PENALTY: i64 = 150;
    chord
        .pitch_classes()
        .into_iter()
        .map(|pc| nearest_octave_pitch(pc, prev_pitch, target_octave))
        .min_by_key(|p| {
            let motion = (p.midi() as i32 - prev_pitch.midi() as i32).abs() as i64;
            let parallel = if crate::counterpoint::has_parallel_perfect(prev_pitch, pm, *p, m) {
                PENALTY
            } else {
                0
            };
            // Escaping a parallel must not BUY a dissonance: without this term
            // the bass sidesteps a fifth onto a tone that clashes instead
            // (+79 StrongBeatConsonance, measured).
            let (lo, hi) = if *p <= m { (*p, m) } else { (m, *p) };
            let dissonant = if crate::counterpoint::is_consonant(lo, hi) {
                0
            } else {
                PENALTY
            };
            // The bass must stay the bottom voice.
            let crossing = if *p > m { 2 * PENALTY } else { 0 };
            // NOT DONE: a home-register preference. Tried 2026-07-31 to recover
            // the 90 bass-vs-Harmony crossings this function costs. It works on
            // the aggregate (total Fatal 4135 -> 4058) but gets there by trading
            // 26 melody-facing parallels (432 -> 458) for 78 harmony-facing
            // crossings (941 -> 863) — the exact reverse of the priority this
            // whole change is justified by, since bass-vs-melody defects are the
            // audible ones and bass-vs-Harmony sits in a mid-register pad.
            // Rejected on that basis, not on the total.
            //
            // Also inert as a knob: PENALTY/3 and PENALTY/6 give bit-identical
            // results, the second time weight-tuning has proved to be a switch
            // rather than a dial here. With ~3 candidates and small integer
            // costs these penalties do not interpolate.
            motion + parallel + dissonant + crossing
        })
        .expect("chord has at least one tone")
}

/// Voice-lead the upper (non-root) chord tones across the SAME number of
/// voice slots as `prev`, choosing whichever assignment-of-tones-to-slots and
/// per-slot octave minimizes total movement from `prev` — the "keep common
/// tones, move the rest by step" principle. Non-crossing voicings (each voice
/// staying above/below its neighbors, matching `prev`'s order) are strongly
/// preferred but not mandatory (a crossing is allowed if it's the only option
/// close to `prev`, rather than leaving no candidate at all).
///
/// If `prev` is empty, or its length doesn't match the number of non-root
/// tones (e.g. a seventh chord appears where only a triad led before), there
/// is nothing to lead FROM: voice ascending from `target_octave`.
pub fn lead_upper(prev: &[Pitch], chord: Chord, target_octave: i32) -> Vec<Pitch> {
    lead_upper_above_bass(prev, chord, target_octave, None)
}

/// # The residual 941 crossings, diagnosed 2026-07-31 (do not re-guess these)
///
/// This fixed 374 of them. What is left does NOT yield to the two obvious next
/// moves, both of which were tried and measured:
///
/// - **Voicing against the bass's bar PEAK instead of its downbeat value** is a
///   NO-OP. 803 of ~5,058 bars do have a peak differing from the onset pitch, so
///   the change is live, but the defect count did not move by a single unit —
///   where the peak differs, the harmony already clears both. Reverted rather
///   than shipped as unexercised machinery with a rationale that does not hold.
/// - **A home-register preference on the bass** wins on the aggregate and loses
///   on musical priority; see `lead_bass`'s doc.
///
/// The real shape, measured: **779 of 933 bass-vs-Harmony crossings are
/// OFF-DOWNBEAT** (worst at beat 3, 184). Harmony is voiced ONCE PER BAR, but
/// the accompaniment pattern then spreads those tones across the bar while the
/// bass moves independently — so a crossing at beat 3 involves a harmony tone
/// and a bass note that were never compared, and no per-bar voicing decision can
/// see it. Separately, **913 of ~5,058 bar onsets have no bass sounding at all**
/// (18%), so those bars get no constraint from this function whatsoever.
///
/// Fixing the remainder therefore means moving the check from per-bar voicing to
/// per-EVENT, after `realize_measure` has placed the tones — a different layer,
/// not a better penalty here.
///
/// [`lead_upper`], but kept ABOVE the bass.
///
/// `validate_voice_crossing` requires the bass to be the bottom voice, and
/// measured 2026-07-30 the overwhelming majority of crossings are bass-vs-
/// Harmony: 1542 of 1550. `lead_upper`'s existing crossing penalty only orders
/// the upper voices among THEMSELVES, so a harmony note falling below the bass
/// is invisible to it.
///
/// This adds a bass term to the RANKING ONLY. The candidate tone set is
/// unchanged (`.skip(1)` as before) — deliberately, because an earlier attempt
/// that also changed which tones the harmony voices regressed VoiceCrossing
/// 1225 -> 2034 by pushing Harmony out of its tight register. Ranking-only is
/// the minimal intervention that can fix the crossing without that side effect.
///
/// Passing `None` reproduces the previous behaviour exactly.
pub fn lead_upper_above_bass(
    prev: &[Pitch],
    chord: Chord,
    target_octave: i32,
    bass: Option<Pitch>,
) -> Vec<Pitch> {
    let tones: Vec<PitchClass> = chord.pitch_classes().into_iter().skip(1).collect();
    if prev.is_empty() || prev.len() != tones.len() {
        return tones
            .iter()
            .map(|&pc| Pitch::new(pc, target_octave))
            .collect();
    }

    let mut best: Option<(i64, Vec<Pitch>)> = None;
    for perm in permutations(&tones) {
        let candidates: Vec<Vec<Pitch>> = perm
            .iter()
            .map(|&pc| {
                (target_octave - 1..=target_octave + 1)
                    .map(|oct| Pitch::new(pc, oct))
                    .collect()
            })
            .collect();
        for combo in cartesian(&candidates) {
            let cost: i64 = combo
                .iter()
                .zip(prev)
                .map(|(p, pv)| (p.midi() as i64 - pv.midi() as i64).abs())
                .sum();
            let ascending = combo.windows(2).all(|w| w[0].midi() <= w[1].midi());
            // Soft preference for non-crossing: a large tie-break penalty, not
            // a hard filter, so we still return SOMETHING when every crossing-
            // free option is far away.
            let crossing_penalty = if ascending { 0 } else { 1000 };
            // Same soft-preference treatment for parallel perfect
            // fifths/octaves between any pair of voices (see counterpoint.rs)
            // -- smaller than the crossing penalty so a crossing-free,
            // slightly-parallel option still beats a crossing, parallel-free
            // one, but large enough to steer away from parallels whenever a
            // comparably-cheap alternative exists.
            //
            // SCOPE LIMIT, measured 2026-07-30: this penalty is blind to the
            // BASS. `parallel_perfect_violations` compares every pair WITHIN
            // the slice it is given, and `prev`/`combo` hold only the upper
            // voices -- `lead_upper` never receives the bass pitch (see this
            // function's `.skip(1)` and the `#[ignore]`d
            // `bass_and_uppers_together_must_contain_every_chord_tone`).
            //
            // CORRECTION, same day: an earlier version of this comment claimed
            // that blindness caused the 461 Fatal ParallelPerfectMotion issues
            // `score_validation` reports. IT DOES NOT, and the claim was
            // disproven by building the fix and measuring it.
            //
            // `lead_upper` emits `VoiceRole::Harmony` (accompaniment.rs:153),
            // but `validate_parallel_motion` compares the bass ONLY against
            // `Melody` and `CounterMelody` (score_validation.rs:386). It never
            // examines Harmony at all. Those 461 parallels are between the bass
            // and the MELODIC voices, which are generated by an entirely
            // different path -- `lead_upper` cannot cause them and cannot fix
            // them.
            //
            // The attempted fix (voice against the real emitted bass, and score
            // bass-vs-upper parallels) left ParallelPerfectMotion unchanged at
            // 461, exactly as that structure predicts, while regressing
            // VoiceCrossing 1225 -> 2034. It was reverted. The regression is
            // itself informative: giving Harmony the root whenever the bass
            // inverts pushes it out of its tight register, which is part of why
            // `.skip(1)` exists.
            //
            // What remains TRUE: this penalty is bass-blind, and the rootless
            // chord defect (`bass_and_uppers_together_must_contain_every_chord_tone`)
            // is real and unfixed. What is NOT true is that either explains the
            // measured parallel fifths. Whatever generates Melody and
            // CounterMelody against the bass is where that defect lives, and it
            // has not been located.
            let parallel_penalty =
                500 * crate::counterpoint::parallel_perfect_violations(prev, &combo) as i64;
            // Any voice at or below the bass is a Fatal VoiceCrossing. Weighted
            // like the internal crossing penalty, per voice.
            let below_bass_penalty = match bass {
                Some(b) => 1000 * combo.iter().filter(|p| p.midi() <= b.midi()).count() as i64,
                None => 0,
            };
            let scored = cost + crossing_penalty + parallel_penalty + below_bass_penalty;
            if best.as_ref().map(|(b, _)| scored < *b).unwrap_or(true) {
                best = Some((scored, combo));
            }
        }
    }
    best.map(|(_, v)| v).unwrap_or_else(|| {
        tones
            .iter()
            .map(|&pc| Pitch::new(pc, target_octave))
            .collect()
    })
}

/// Nearest absolute pitch of `pc` to `reference`, searched within one octave
/// of `target_octave` (keeps results in the intended register).
fn nearest_octave_pitch(pc: PitchClass, reference: Pitch, target_octave: i32) -> Pitch {
    (target_octave - 1..=target_octave + 1)
        .map(|oct| Pitch::new(pc, oct))
        .min_by_key(|p| (p.midi() as i32 - reference.midi() as i32).abs())
        .expect("non-empty octave range")
}

fn permutations(items: &[PitchClass]) -> Vec<Vec<PitchClass>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, head);
            out.push(p);
        }
    }
    out
}

fn cartesian(lists: &[Vec<Pitch>]) -> Vec<Vec<Pitch>> {
    lists.iter().fold(vec![Vec::new()], |acc, list| {
        acc.into_iter()
            .flat_map(|prefix| {
                list.iter().map(move |&p| {
                    let mut v = prefix.clone();
                    v.push(p);
                    v
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chord::ChordQuality;

    #[test]
    fn force_root_ignores_prev() {
        let g = Chord::new(PitchClass::G, ChordQuality::Major);
        let prev = Some(Pitch::new(PitchClass::B, 3));
        let bass = lead_bass(prev, g, 3, true);
        assert_eq!(bass.pitch_class(), PitchClass::G);
    }

    #[test]
    fn first_chord_with_no_prev_is_root() {
        let g = Chord::new(PitchClass::G, ChordQuality::Major);
        assert_eq!(lead_bass(None, g, 3, false).pitch_class(), PitchClass::G);
    }

    #[test]
    fn bass_prefers_smooth_inversion_over_root_leap() {
        // Previous bass on C4 (60). Next chord is G major (G,B,D). The root G
        // is a 5th away (G3=55, dist 5); the third B is a step away (B3=59,
        // dist 1). A smooth bass line should land on B (first inversion),
        // not leap to the root.
        let prev = Pitch::from_midi(60);
        let g = Chord::new(PitchClass::G, ChordQuality::Major);
        let bass = lead_bass(Some(prev), g, 3, false);
        assert_eq!(bass.pitch_class(), PitchClass::B);
        assert!((bass.midi() as i32 - prev.midi() as i32).abs() <= 2);
    }

    /// KNOWN GAP, not yet fixed — a runnable reproduction of the target
    /// invariant `bass ∪ uppers ⊇ chord.pitch_classes()`.
    ///
    /// `lead_upper` does `.skip(1)` on the chord's tones (voicing.rs:52) on the
    /// assumption that the bass carries the root, but `lead_bass` with
    /// `force_root == false` inverts freely — the test immediately above,
    /// `bass_prefers_smooth_inversion_over_root_leap`, asserts a G-major chord
    /// landing on B. Combine the two and the root is absent from the entire
    /// texture: bass B + uppers [B, D] is a bare third with the third doubled
    /// and no G anywhere.
    ///
    /// RETRACTED (2026-07-30, same day): an earlier version of this doc claimed
    /// the same bass-blindness also caused 461 Fatal parallel fifths/octaves.
    /// **It does not.** `lead_upper` emits `VoiceRole::Harmony`, while
    /// `validate_parallel_motion` compares the bass only against `Melody` and
    /// `CounterMelody` — it never looks at Harmony. The fix was built and
    /// measured: parallels stayed at exactly 461 while VoiceCrossing regressed
    /// 1225 → 2034, so it was reverted. This defect is real and unfixed, but it
    /// is ONE defect, not two, and it is worth only what one rootless-chord
    /// defect is worth.
    ///
    /// WHY THIS IS `#[ignore]` RATHER THAN FIXED. The two functions are called
    /// from `realize_harmony_measures` (composer.rs:4104) and
    /// `realize_bass_measures` (composer.rs:4386) — separate functions with no
    /// shared scope, and harmony is realized FIRST (call sites 4260/4939 precede
    /// 4670/4972), so the bass line does not exist yet when the upper voices are
    /// chosen. The cheapest real fix is to precompute the bass PITCH CLASSES
    /// first — `lead_bass` is deterministic given prev/chord/octave/force_root —
    /// pass them into `lead_upper`, then realize the bass fully as today. And because
    /// the uppers never contain the root, EVERY inversion breaks completeness,
    /// not just some: the current design is only sound when `force_root` is
    /// always true. A real fix has to let the upper voices choose among ALL
    /// chord tones instead of a fixed non-root set, which changes voicing for
    /// every style and needs listening validation — a design pass, not a patch.
    ///
    /// Measuring the rate from score output was attempted and ABANDONED as
    /// invalid: aggregating a bar's pitch classes flags ordinary scalar and
    /// passing motion ({C,D,E,F}, {E,G,G♯}) as "rootless", so it measures
    /// non-triadic bars rather than this defect. Sizing it properly needs the
    /// composer instrumented per harmony event, not output analysis.
    ///
    /// Remove the `#[ignore]` when the design pass lands; it should then pass.
    #[test]
    #[ignore = "known gap: lead_upper cannot see lead_bass's inversion; needs a design pass"]
    fn bass_and_uppers_together_must_contain_every_chord_tone() {
        let prev_bass = Pitch::from_midi(60); // C4
        let g = Chord::new(PitchClass::G, ChordQuality::Major);
        let bass = lead_bass(Some(prev_bass), g, 3, false);
        // Two upper slots, positioned so the leading is unremarkable.
        let prev_upper = vec![Pitch::from_midi(59), Pitch::from_midi(62)];
        let uppers = lead_upper(&prev_upper, g, 4);

        let mut sounding: Vec<PitchClass> = uppers.iter().map(|p| p.pitch_class()).collect();
        sounding.push(bass.pitch_class());
        for pc in g.pitch_classes() {
            assert!(
                sounding.contains(&pc),
                "chord tone {pc:?} is absent from bass {:?} + uppers {:?}",
                bass.pitch_class(),
                uppers.iter().map(|p| p.pitch_class()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn upper_voices_find_the_smoother_assignment() {
        // C major (C,E,G) -> F major (F,A,C), upper voices at E4=64, G4=67.
        // The two possible assignments cost 10 (A->slot0,C->slot1) vs 6
        // (C->slot0,A->slot1) — the algorithm must find the cheaper one.
        let prev = vec![Pitch::from_midi(64), Pitch::from_midi(67)]; // E4, G4
        let f = Chord::new(PitchClass::F, ChordQuality::Major);
        let next = lead_upper(&prev, f, 4);
        assert_eq!(next.len(), 2);
        assert_eq!(next[0], Pitch::from_midi(60)); // C4
        assert_eq!(next[1], Pitch::from_midi(69)); // A4
        let cost: i64 = next
            .iter()
            .zip(&prev)
            .map(|(p, pv)| (p.midi() as i64 - pv.midi() as i64).abs())
            .sum();
        assert_eq!(cost, 6);
    }

    #[test]
    fn upper_voicing_stays_ascending_when_possible() {
        let prev = vec![Pitch::from_midi(64), Pitch::from_midi(67)];
        let f = Chord::new(PitchClass::F, ChordQuality::Major);
        let next = lead_upper(&prev, f, 4);
        assert!(next[0].midi() <= next[1].midi());
    }

    #[test]
    fn lead_upper_avoids_parallel_fifths_when_an_alternative_exists() {
        // Cmaj7 (non-root E,G,B) -> Dm7 (non-root F,A,C). The naive
        // identity assignment (E->F, G->A, B->C) is the raw-movement-cost
        // minimum, but it preserves a perfect fifth between slot0/slot2
        // (E-B is a P5, F-C is a P5, both voices move up together) --
        // exactly the textbook parallel-fifths trap. A different
        // permutation avoids it at only slightly higher raw cost, and the
        // parallel-avoidance penalty must make lead_upper prefer it.
        let prev = vec![
            Pitch::from_midi(64), // E4
            Pitch::from_midi(67), // G4
            Pitch::from_midi(71), // B4
        ];
        let dm7 = Chord::new(PitchClass::D, ChordQuality::Minor7);
        let next = lead_upper(&prev, dm7, 4);
        assert_eq!(
            crate::counterpoint::parallel_perfect_violations(&prev, &next),
            0,
            "lead_upper picked a voicing with parallel fifths/octaves: {next:?}"
        );
    }

    #[test]
    fn empty_prev_voices_ascending_from_target_octave() {
        let c = Chord::new(PitchClass::C, ChordQuality::Major);
        let voiced = lead_upper(&[], c, 4);
        assert_eq!(
            voiced,
            vec![Pitch::new(PitchClass::E, 4), Pitch::new(PitchClass::G, 4)]
        );
    }

    #[test]
    fn count_mismatch_falls_back_to_ascending() {
        // A seventh chord has 3 non-root tones; a 2-voice `prev` can't lead
        // it — must fall back rather than panic or silently truncate.
        let prev = vec![Pitch::from_midi(64), Pitch::from_midi(67)];
        let g7 = Chord::new(PitchClass::G, ChordQuality::Dominant7);
        let voiced = lead_upper(&prev, g7, 4);
        assert_eq!(voiced.len(), 3);
    }

    #[test]
    fn voice_leading_reduces_total_movement_across_a_progression() {
        // ii-V-I in C: Dm, G, C. Compare total upper-voice movement WITH
        // leading vs a naive fixed-octave re-voicing every chord.
        let key_chords = [
            Chord::new(PitchClass::D, ChordQuality::Minor),
            Chord::new(PitchClass::G, ChordQuality::Major),
            Chord::new(PitchClass::C, ChordQuality::Major),
        ];
        let mut led_total = 0i64;
        let mut prev: Vec<Pitch> = Vec::new();
        for &c in &key_chords {
            let voiced = lead_upper(&prev, c, 4);
            if !prev.is_empty() && prev.len() == voiced.len() {
                led_total += voiced
                    .iter()
                    .zip(&prev)
                    .map(|(a, b)| (a.midi() as i64 - b.midi() as i64).abs())
                    .sum::<i64>();
            }
            prev = voiced;
        }

        let mut naive_total = 0i64;
        let mut naive_prev: Option<Vec<Pitch>> = None;
        for &c in &key_chords {
            let voiced: Vec<Pitch> = c
                .pitch_classes()
                .into_iter()
                .skip(1)
                .map(|pc| Pitch::new(pc, 4))
                .collect();
            if let Some(p) = &naive_prev {
                naive_total += voiced
                    .iter()
                    .zip(p)
                    .map(|(a, b)| (a.midi() as i64 - b.midi() as i64).abs())
                    .sum::<i64>();
            }
            naive_prev = Some(voiced);
        }

        assert!(
            led_total <= naive_total,
            "voice leading should not increase movement: led={led_total} naive={naive_total}"
        );
    }
}
