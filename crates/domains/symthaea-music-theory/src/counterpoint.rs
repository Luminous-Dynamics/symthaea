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

use crate::pitch::Pitch;

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
}
