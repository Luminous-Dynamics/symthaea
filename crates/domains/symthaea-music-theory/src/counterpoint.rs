// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counterpoint rules: constraints on how TWO voices may move relative to
//! each other, independent of what chord either belongs to. The first (and
//! for now only) rule implemented is the oldest and most universally cited
//! one in tonal counterpoint: **no parallel perfect fifths or octaves**.
//!
//! Two voices that sit a perfect fifth or octave apart, then BOTH move in
//! the same direction to another perfect fifth or octave, fuse into a
//! single implied voice — the polyphonic independence between them
//! momentarily vanishes. Oblique motion (one voice holds) or contrary
//! motion into the same perfect interval is fine; it's specifically
//! *similar motion between two perfect intervals of the same kind* that's
//! forbidden.
//!
//! [`crate::voicing::lead_upper`] uses [`has_parallel_perfect`] as a soft
//! cost penalty (same pattern as its existing voice-crossing penalty): a
//! preference against parallels, not a hard filter. **Honest limit**: when
//! every candidate voicing in the search space shares the same parallel
//! motion (e.g. transposing a whole chord uniformly by a fixed interval —
//! genuinely the cheapest and only nearby option), the penalty cannot
//! manufacture an alternative that doesn't exist; see the module's own
//! tests for a case where an alternative DOES exist and is correctly
//! preferred, and `voicing.rs` for the wiring.

use crate::motif::Motif;
use crate::pitch::Pitch;
use crate::scale::Scale;

/// Does moving voice A from `prev_a` to `next_a`, alongside voice B moving
/// from `prev_b` to `next_b`, create parallel perfect fifths or octaves?
///
/// True iff ALL of:
/// - the pitch-class interval between the voices is the same before and
///   after (mod 12),
/// - that interval is a perfect fifth (7 semitones) or unison/octave (0),
/// - BOTH voices actually moved (oblique motion — one voice static — is not
///   a violation), and
/// - both voices moved in the same direction (contrary motion into the same
///   interval is not a violation).
pub fn has_parallel_perfect(prev_a: Pitch, prev_b: Pitch, next_a: Pitch, next_b: Pitch) -> bool {
    let before = (prev_b.midi() as i32 - prev_a.midi() as i32).rem_euclid(12);
    let after = (next_b.midi() as i32 - next_a.midi() as i32).rem_euclid(12);
    let is_perfect = |d: i32| d == 0 || d == 7;
    if before != after || !is_perfect(before) {
        return false;
    }
    let move_a = next_a.midi() as i32 - prev_a.midi() as i32;
    let move_b = next_b.midi() as i32 - prev_b.midi() as i32;
    if move_a == 0 || move_b == 0 {
        return false; // oblique motion
    }
    (move_a > 0) == (move_b > 0) // similar/parallel motion
}

/// Is a two-voice vertical interval consonant, in strict species terms?
/// Consonances (mod 12): unison/octave (0), thirds (3, 4), perfect fifth
/// (7), sixths (8, 9). The perfect fourth (5) counts as a DISSONANCE —
/// the strict two-voice rule this module implements (against a bass it is
/// unambiguous; between upper voices real practice is laxer, and we accept
/// being stricter than necessary there).
pub fn is_consonant(lower: Pitch, upper: Pitch) -> bool {
    let d = (upper.midi() as i32 - lower.midi() as i32).rem_euclid(12);
    matches!(d, 0 | 3 | 4 | 7 | 8 | 9)
}

/// One sounding note of a fixed reference line (a cantus firmus): absolute
/// onset and duration in beats, and its realized pitch.
#[derive(Debug, Clone, Copy)]
pub struct CantusEvent {
    pub onset: f64,
    pub duration: f64,
    pub pitch: Pitch,
}

/// The cantus event sounding at time `t`, if any.
fn sounding_at(cantus: &[CantusEvent], t: f64) -> Option<CantusEvent> {
    cantus
        .iter()
        .copied()
        .find(|e| e.onset - 1e-9 <= t && t < e.onset + e.duration - 1e-9)
}

/// Fit `line` against a fixed `cantus` under two-voice species rules,
/// adjusting degrees MINIMALLY — the species-checked counterpoint pass.
///
/// The rules enforced, per pitched note of `line` (at absolute onset
/// `start + cumulative durations`, realized via `scale.degree_pitch(d,
/// line_octave)`):
///
/// 1. **On-beat notes must be consonant** with the cantus note sounding at
///    that moment ([`is_consonant`]; the perfect fourth counts as
///    dissonant). This is first-species grammar governing the strong
///    beats.
/// 2. **Off-beat notes pass through untouched** — florid off-beat
///    dissonance (passing, neighbor, échappée) is the surface texture of
///    real counterpoint, and bending it costs more identity than it buys.
///    (An earlier draft enforced a passing-only rule off the beat; the
///    Φ A/B showed the extra bending destroyed more motif-channel
///    integration than the consonance channel gained. Minimal
///    intervention won empirically, not just aesthetically.)
/// 3. **No parallel perfect fifths/octaves** with the cantus across
///    consecutive on-beat pairs ([`has_parallel_perfect`]).
///
/// A violating note is moved to the NEAREST scale degree (searching
/// ±1, ±2) that satisfies rules 1 and 3; if no candidate within two steps
/// works, the original is kept — the fitter is best-effort, never
/// destructive. Notes with no cantus sounding (the reference voice is
/// resting) pass through untouched, as do rests. Rhythm is never altered.
///
/// This is deliberately what a composer does when correcting a DERIVED
/// countersubject: keep its identity (contour, rhythm — the derivation),
/// bend individual tones until the verticals obey the species rules.
pub fn fit_against(
    cantus: &[CantusEvent],
    line: &Motif,
    scale: Scale,
    line_octave: i32,
    start: f64,
) -> Motif {
    let mut out = line.notes.to_vec();
    let mut t = start;
    let mut prev: Option<(Pitch, Pitch)> = None; // (line, cantus) at previous pitched note
    for i in 0..out.len() {
        let dur = out[i].duration.beats();
        let Some(d) = out[i].degree else {
            t += dur;
            continue;
        };
        let Some(cf) = sounding_at(cantus, t) else {
            prev = None; // reference resting: parallel chains break here
            t += dur;
            continue;
        };
        let on_beat = (t - t.round()).abs() < 1e-6;
        if !on_beat {
            // Off the beat: florid dissonance is the texture, not a fault.
            t += dur;
            continue;
        }
        let ok = |deg: i32, prev: &Option<(Pitch, Pitch)>| -> bool {
            let p = scale.degree_pitch(deg, line_octave);
            let consonant = if p.midi() >= cf.pitch.midi() {
                is_consonant(cf.pitch, p)
            } else {
                is_consonant(p, cf.pitch)
            };
            let no_parallel = prev
                .map(|(lp, cp)| !has_parallel_perfect(lp, cp, p, cf.pitch))
                .unwrap_or(true);
            consonant && no_parallel
        };
        let chosen = if ok(d, &prev) {
            d
        } else {
            [-1, 1, -2, 2]
                .iter()
                .map(|off| d + off)
                .find(|cand| ok(*cand, &prev))
                .unwrap_or(d)
        };
        out[i].degree = Some(chosen);
        prev = Some((scale.degree_pitch(chosen, line_octave), cf.pitch));
        t += dur;
    }
    Motif { notes: out }
}

/// Count how many pairs of voices in `prev`/`next` (matched by index) form
/// parallel perfect fifths/octaves. Used to score a whole candidate
/// voicing at once, not just one pair.
pub fn parallel_perfect_violations(prev: &[Pitch], next: &[Pitch]) -> usize {
    let n = prev.len().min(next.len());
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if has_parallel_perfect(prev[i], prev[j], next[i], next[j]) {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_fifths_up_a_step_is_a_violation() {
        // C4->D4 and G4->A4: a perfect fifth apart both times, both voices
        // move up a step together. THE textbook forbidden case.
        let (c4, d4) = (Pitch::from_midi(60), Pitch::from_midi(62));
        let (g4, a4) = (Pitch::from_midi(67), Pitch::from_midi(69));
        assert!(has_parallel_perfect(c4, g4, d4, a4));
    }

    #[test]
    fn parallel_octaves_up_an_octave_is_a_violation() {
        let (c4, d4) = (Pitch::from_midi(60), Pitch::from_midi(62));
        let (c5, d5) = (Pitch::from_midi(72), Pitch::from_midi(74));
        assert!(has_parallel_perfect(c4, c5, d4, d5));
    }

    #[test]
    fn oblique_motion_into_a_fifth_is_not_a_violation() {
        // Voice B stays put; only voice A moves. Not parallel motion.
        let (c4, d4) = (Pitch::from_midi(60), Pitch::from_midi(62));
        let g4 = Pitch::from_midi(67);
        assert!(!has_parallel_perfect(c4, g4, d4, g4));
    }

    #[test]
    fn similar_motion_into_a_fifth_is_flagged_but_contrary_motion_is_not() {
        // Sanity check first: G3-D4 (7 semitones = P5) -> A3-E4 (P5 again),
        // both voices move up a step together -- similar motion, flagged.
        let (g3, d4) = (Pitch::from_midi(55), Pitch::from_midi(62));
        let (a3, e4) = (Pitch::from_midi(57), Pitch::from_midi(64));
        assert!(has_parallel_perfect(g3, d4, a3, e4));

        // Now contrary motion INTO THE SAME interval type (P5 both times),
        // to isolate the direction check specifically: C4-G4 (P5) -> voice A
        // up to D4, voice B down to A3 -- the resulting interval (A3 to D4)
        // is STILL a P5 (57 to 62 = 7 mod 12), but the voices moved in
        // OPPOSITE directions, so this must not be flagged.
        let (c4b, g4b) = (Pitch::from_midi(60), Pitch::from_midi(67));
        let (d4b, a3b) = (Pitch::from_midi(62), Pitch::from_midi(57));
        assert!(!has_parallel_perfect(c4b, g4b, d4b, a3b));
    }

    #[test]
    fn interval_changing_from_perfect_to_imperfect_is_not_a_violation() {
        // C4-G4 (P5) -> D4-F#4... pick a case where the interval type
        // changes even though it stays vaguely "similar": C4->D4 (up 2),
        // G4->B4 (up 4) -- interval goes from P5 (7) to major sixth (9).
        let (c4, d4) = (Pitch::from_midi(60), Pitch::from_midi(62));
        let (g4, b4) = (Pitch::from_midi(67), Pitch::from_midi(71));
        assert!(!has_parallel_perfect(c4, g4, d4, b4));
    }

    #[test]
    fn static_interval_with_no_motion_is_not_a_violation() {
        let c4 = Pitch::from_midi(60);
        let g4 = Pitch::from_midi(67);
        assert!(!has_parallel_perfect(c4, g4, c4, g4));
    }

    #[test]
    fn violations_are_counted_across_all_voice_pairs() {
        // Three voices, uniform whole-step transposition: every pair that
        // started as a perfect interval stays parallel.
        let prev = [
            Pitch::from_midi(60), // C4
            Pitch::from_midi(67), // G4 (P5 above C4)
            Pitch::from_midi(72), // C5 (P8 above C4, P4 above G4 -- not perfect vs G4)
        ];
        let next = [
            Pitch::from_midi(62), // D4
            Pitch::from_midi(69), // A4 (P5 above D4)
            Pitch::from_midi(74), // D5 (P8 above D4)
        ];
        // Pairs: (0,1) P5->P5 parallel = violation. (0,2) P8->P8 parallel =
        // violation. (1,2) P4 (5 semitones, not perfect) -- not counted.
        assert_eq!(parallel_perfect_violations(&prev, &next), 2);
    }

    #[test]
    fn no_violations_when_voices_move_by_different_directions() {
        let prev = [Pitch::from_midi(60), Pitch::from_midi(67)];
        let next = [Pitch::from_midi(64), Pitch::from_midi(62)]; // contrary + interval changes
        assert_eq!(parallel_perfect_violations(&prev, &next), 0);
    }

    // ── species fitter ───────────────────────────────────────────────────

    use crate::harmony::Key;
    use crate::motif::Motif;
    use crate::pitch::PitchClass;
    use crate::rhythm::Duration;

    fn c_major() -> Scale {
        Key::major(PitchClass::C).scale()
    }

    /// A cantus holding one tone for `beats` from t=0.
    fn held(degree: i32, beats: f64) -> Vec<CantusEvent> {
        vec![CantusEvent {
            onset: 0.0,
            duration: beats,
            pitch: c_major().degree_pitch(degree, 4),
        }]
    }

    #[test]
    fn consonance_table_is_ground_truth() {
        let c4 = Pitch::from_midi(60);
        for (semis, expect) in [
            (0, true),   // unison
            (1, false),  // minor second
            (2, false),  // major second
            (3, true),   // minor third
            (4, true),   // major third
            (5, false),  // perfect fourth: dissonant in strict two-voice
            (6, false),  // tritone
            (7, true),   // perfect fifth
            (8, true),   // minor sixth
            (9, true),   // major sixth
            (10, false), // minor seventh
            (11, false), // major seventh
            (12, true),  // octave
        ] {
            assert_eq!(
                is_consonant(c4, Pitch::from_midi(60 + semis)),
                expect,
                "{semis} semitones"
            );
        }
    }

    #[test]
    fn on_beat_dissonance_is_bent_to_the_nearest_consonance() {
        // Cantus holds degree 1 (C). Line lands degree 2 (D) on the beat —
        // a second, dissonant. The fitter must move it to a neighbor that
        // is consonant (degree 1 or 3), not leave it.
        let line = Motif::from_degrees(&[(2, Duration::whole())]);
        let fitted = fit_against(&held(1, 4.0), &line, c_major(), 4, 0.0);
        let got = fitted.notes[0].degree.unwrap();
        assert!(got == 1 || got == 3, "got degree {got}");
    }

    #[test]
    fn consonant_notes_pass_through_untouched() {
        // Thirds and fifths against the cantus: nothing to fix.
        let line = Motif::from_degrees(&[
            (3, Duration::half()),
            (5, Duration::quarter()),
            (8, Duration::quarter()),
        ]);
        let fitted = fit_against(&held(1, 4.0), &line, c_major(), 4, 0.0);
        assert_eq!(fitted, line);
    }

    #[test]
    fn off_beat_passing_dissonance_is_kept() {
        // 3 - 4 - 5 in eighths over a held 1: the 4 (an off-beat fourth,
        // dissonant) is approached and left by step — a textbook passing
        // tone the fitter must NOT flatten.
        let line = Motif::from_degrees(&[
            (3, Duration::eighth()),
            (4, Duration::eighth()),
            (5, Duration::quarter()),
        ]);
        let fitted = fit_against(&held(1, 4.0), &line, c_major(), 4, 0.0);
        assert_eq!(fitted, line, "passing tone must survive");
    }

    #[test]
    fn off_beat_notes_are_out_of_scope_by_design() {
        // 1 - leap to 4 (off-beat, dissonant, approached and left by leap):
        // deliberately NOT corrected — the fitter governs strong beats
        // only. An earlier draft bent these; the Φ A/B showed the extra
        // bending cost more motif-channel integration than it bought in
        // consonance. See `fit_against` docs.
        let line = Motif::from_degrees(&[
            (1, Duration::eighth()),
            (4, Duration::eighth()),
            (8, Duration::quarter()),
        ]);
        let fitted = fit_against(&held(1, 4.0), &line, c_major(), 4, 0.0);
        assert_eq!(
            fitted.notes[1].degree,
            Some(4),
            "off-beat notes pass through untouched"
        );
    }

    #[test]
    fn fitter_avoids_creating_parallel_fifths() {
        // Cantus walks 1 -> 2 in whole beats; a naive fit of a line walking
        // 5 -> 6 keeps perfect fifths moving in parallel. The fitter's
        // second note must not be a parallel fifth even though 6-over-2 is
        // "consonant" in isolation.
        let cantus = vec![
            CantusEvent {
                onset: 0.0,
                duration: 1.0,
                pitch: c_major().degree_pitch(1, 3),
            },
            CantusEvent {
                onset: 1.0,
                duration: 1.0,
                pitch: c_major().degree_pitch(2, 3),
            },
        ];
        let line = Motif::from_degrees(&[(5, Duration::quarter()), (6, Duration::quarter())]);
        let fitted = fit_against(&cantus, &line, c_major(), 4, 0.0);
        let p0 = c_major().degree_pitch(fitted.notes[0].degree.unwrap(), 4);
        let p1 = c_major().degree_pitch(fitted.notes[1].degree.unwrap(), 4);
        assert!(
            !has_parallel_perfect(p0, cantus[0].pitch, p1, cantus[1].pitch),
            "fitted line still forms parallel perfects"
        );
    }

    #[test]
    fn rests_and_uncovered_spans_pass_through() {
        // Line notes where the cantus is silent (after its hold ends) and
        // rests are untouched; rhythm is never altered.
        let line = Motif::new(vec![
            crate::motif::MotifNote::rest(Duration::quarter()),
            crate::motif::MotifNote::new(2, Duration::quarter()),
        ]);
        // Cantus covers only the first beat (the rest).
        let fitted = fit_against(&held(1, 1.0), &line, c_major(), 4, 0.0);
        assert_eq!(fitted, line);
        let durs: Vec<_> = fitted.notes.iter().map(|n| n.duration).collect();
        let orig: Vec<_> = line.notes.iter().map(|n| n.duration).collect();
        assert_eq!(durs, orig);
    }
}
