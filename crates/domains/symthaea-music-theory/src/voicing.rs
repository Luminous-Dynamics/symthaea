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
            let parallel_penalty =
                500 * crate::counterpoint::parallel_perfect_violations(prev, &combo) as i64;
            let scored = cost + crossing_penalty + parallel_penalty;
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
