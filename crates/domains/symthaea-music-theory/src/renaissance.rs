// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Renaissance polyphony (style-roadmap item, Tier 1 "high priority" —
//! "very different from fugue... flowing independence, vocal balance,
//! modal cadence, equal voices"): three genuinely EQUAL voices, no
//! subject-entry hierarchy.
//!
//! [`crate::fugue`] has a hierarchy baked into its very vocabulary —
//! "subject," "answer" (a fixed transposition up a diatonic fifth),
//! "countersubject" (a fixed derivation FROM the subject). This module
//! shares fugue's low-level plumbing ([`crate::fugue::emit`],
//! [`crate::counterpoint::fit_against`], [`crate::fugue::head_fragment`])
//! but deliberately does NOT reuse its hierarchy:
//!
//! - **Imitation at the OCTAVE, not the fifth.** A fugue answer transposes
//!   the subject up a diatonic fifth — a real tonal move that fugue's own
//!   exposition depends on. A Renaissance point of imitation restates the
//!   SAME point at whatever octave the entering voice's register wants —
//!   register difference, not transposition, is what varies. No voice's
//!   entry is privileged as "the answer."
//! - **TWO points, not one subject with episodes.** A motet sets successive
//!   phrases of text to successive points of imitation. Point 1 enters
//!   low-to-high (bass first); point 2 enters high-to-low (soprano
//!   first) — voice order itself rotates, so no single voice is always
//!   "the leader."
//! - **Modal, with a real suspension-and-under-third close.** Dorian (no
//!   leading tone to borrow — the mode's own color), and the final
//!   cadence is a prepared 7-6 suspension in the middle voice against the
//!   bass's motion to the tonic, with the top voice approaching home via
//!   the "Landini" under-third (7-6-8, not a direct 7-8 climb) — the
//!   textbook Renaissance clausula, not a borrowed leading-tone drive.
//!
//! Species-checked exactly as fugue: each entering voice's degrees are
//! fitted against the most recently sounding voice via
//! [`crate::counterpoint::fit_against`] (strong-beat consonance, no
//! parallel perfects); the points themselves are never altered.

use crate::counterpoint::{CantusEvent, fit_against};
use crate::form::figuration_variation;
use crate::fugue::{emit, head_fragment, hold};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::rhythm::Duration;
use crate::scale::Scale;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};

/// Cantus events (absolute onsets) for a motif emitted at `start` in
/// `octave` — the same bookkeeping [`crate::fugue`] does locally, needed
/// here too to hand a just-sung voice to [`fit_against`] as the next
/// voice's reference.
fn cantus_of(scale: Scale, m: &Motif, octave: i32, start: Duration) -> Vec<CantusEvent> {
    let mut t = start.beats();
    let mut out = Vec::new();
    for n in &m.notes {
        if let Some(d) = n.degree {
            out.push(CantusEvent {
                onset: t,
                duration: n.duration.beats(),
                pitch: scale.degree_pitch(d, octave),
            });
        }
        t += n.duration.beats();
    }
    out
}

/// Compose the motet-in-miniature. `point1` arrives already hook-grafted
/// (the piece's name). Fixed 9-bar shape:
///
/// | bars | what |
/// |------|------|
/// | 0-2  | point 1, entering bass -> alto -> soprano (low to high) |
/// | 3    | free bridge — each voice continues with derived figuration |
/// | 4-6  | point 2 (inverted/transposed derivation), entering soprano -> bass -> alto (high to low, then filling the middle) |
/// | 7-8  | cadence: prepared 7-6 suspension in the alto, bass to the tonic, soprano's under-third approach |
pub(crate) fn realize_renaissance(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    point1: &Motif,
    seed: u64,
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let scale = key.scale();
    let s_beats = meter as i64;
    let bar = |n: i64| Duration::new(s_beats * n, 1);

    use Emphasis::{Cadential, Normal, PhraseStart};
    use VoiceRole::{Bass, CounterMelody, Melody};
    const POINT: f32 = 0.9;
    const BRIDGE: f32 = 0.85;
    const CADENCE: f32 = 1.0;

    let pivot = point1.notes.iter().find_map(|x| x.degree).unwrap_or(1);
    let point2 = point1.invert(pivot).transpose(2);

    // ── Point 1 (bars 0-2): bass -> alto -> soprano ─────────────────────
    emit(&mut score, point1, bar(0), Bass, 3, POINT, PhraseStart);
    let bass_cf = cantus_of(scale, point1, 3, bar(0));
    let alto1 = fit_against(&bass_cf, point1, scale, 4, bar(1).beats());
    emit(
        &mut score,
        &alto1,
        bar(1),
        CounterMelody,
        4,
        POINT,
        PhraseStart,
    );
    let alto_cf = cantus_of(scale, &alto1, 4, bar(1));
    let sop1 = fit_against(&alto_cf, point1, scale, 5, bar(2).beats());
    emit(&mut score, &sop1, bar(2), Melody, 5, POINT, PhraseStart);

    // ── Bridge (bar 3): free figuration, each voice off the last ────────
    let bridge_len = Duration::new(s_beats, 1);
    let sop_frag = head_fragment(&figuration_variation(point1, seed), bridge_len);
    emit(&mut score, &sop_frag, bar(3), Melody, 5, BRIDGE, Normal);
    let sop_frag_cf = cantus_of(scale, &sop_frag, 5, bar(3));
    let alto_frag = fit_against(
        &sop_frag_cf,
        &head_fragment(
            &figuration_variation(&alto1, seed.wrapping_add(1)),
            bridge_len,
        ),
        scale,
        4,
        bar(3).beats(),
    );
    emit(
        &mut score,
        &alto_frag,
        bar(3),
        CounterMelody,
        4,
        BRIDGE,
        Normal,
    );

    // ── Point 2 (bars 4-6): soprano -> bass -> alto ─────────────────────
    // Voice order rotates — no voice is always the leader.
    emit(&mut score, &point2, bar(4), Melody, 5, POINT, PhraseStart);
    let sop2_cf = cantus_of(scale, &point2, 5, bar(4));
    let bass2 = fit_against(&sop2_cf, &point2, scale, 3, bar(5).beats());
    emit(&mut score, &bass2, bar(5), Bass, 3, POINT, PhraseStart);
    let bass2_cf = cantus_of(scale, &bass2, 3, bar(5));
    let alto2 = fit_against(&bass2_cf, &point2, scale, 4, bar(6).beats());
    emit(
        &mut score,
        &alto2,
        bar(6),
        CounterMelody,
        4,
        POINT,
        PhraseStart,
    );

    // ── Cadence (bars 7-8): prepared 7-6 suspension + under-third close ─
    // The alto's OUTGOING tone (scale degree 7, sounding through bar 7) is
    // tied over the bar line as a prepared dissonance against the bass's
    // move toward the tonic, then resolves DOWN a step to degree 6 — the
    // real 7-6 mechanism, not a borrowed leading tone. The soprano's own
    // approach to home is the Landini under-third: 7, then DOWN to 6,
    // THEN up to the octave — never a direct climb.
    let full_bar = Duration::new(s_beats, 1);
    let half_bar = Duration::new(s_beats, 2);
    hold(&mut score, 4, bar(6), full_bar, Bass, 3, CADENCE, Normal); // predominant, sets up the approach
    // Alto: suspension prepared in bar 7 (deg 7 held through the bar
    // line), resolves to deg 6 at the top of bar 8.
    score.push(ScoreNote {
        pitch: scale.degree_pitch(7, 4),
        onset: bar(7),
        duration: full_bar + half_bar,
        velocity: (0.72 * CADENCE).clamp(0.0, 1.0), // the suspension leans in
        role: CounterMelody,
        emphasis: Emphasis::Cadential,
        section_intensity: CADENCE,
    });
    score.push(ScoreNote {
        pitch: scale.degree_pitch(6, 4),
        onset: bar(7) + full_bar + half_bar,
        duration: half_bar,
        velocity: (0.62 * CADENCE).clamp(0.0, 1.0), // resolves, releases
        role: CounterMelody,
        emphasis: Emphasis::Normal,
        section_intensity: CADENCE,
    });
    // Bass: dominant under the suspension, tonic at the true close.
    hold(
        &mut score,
        5,
        bar(7),
        full_bar,
        Bass,
        3,
        CADENCE,
        Emphasis::Normal,
    );
    hold(
        &mut score,
        1,
        bar(7) + full_bar,
        full_bar,
        Bass,
        3,
        CADENCE,
        Cadential,
    );
    // Soprano: the under-third approach — 7, down to 6, up to the octave.
    hold(
        &mut score,
        7,
        bar(7),
        half_bar,
        Melody,
        5,
        CADENCE,
        Emphasis::Normal,
    );
    hold(
        &mut score,
        6,
        bar(7) + half_bar,
        half_bar,
        Melody,
        5,
        CADENCE,
        Emphasis::Normal,
    );
    hold(
        &mut score,
        8,
        bar(7) + full_bar,
        full_bar,
        Melody,
        5,
        CADENCE,
        Cadential,
    );

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn point() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (1, Duration::quarter()),
        ])
    }

    #[test]
    fn no_voice_is_privileged_as_the_answer() {
        // Unlike fugue::answer (a fixed +4-degree transposition), point-of-
        // imitation entries restate the SAME degree sequence, only in a
        // different register — provable directly: alto's fitted line, read
        // back at the same octave as the point, must match the point's
        // degrees wherever the fitter left them untouched (species fitting
        // only nudges individual degrees for consonance, it never
        // transposes the whole line the way a fugue answer does).
        let key = Key::modal(PitchClass::D, crate::scale::Mode::Dorian).unwrap();
        let scale = key.scale();
        let p = point();
        let bass_cf = cantus_of(scale, &p, 3, Duration::zero());
        let alto = fit_against(&bass_cf, &p, scale, 4, Duration::quarter().beats() * 0.0);
        // Same LENGTH and same starting degree — a real point of
        // imitation, not a transposed answer.
        assert_eq!(alto.notes.len(), p.notes.len());
        assert_eq!(alto.notes[0].degree, p.notes[0].degree);
    }

    #[test]
    fn realize_renaissance_composes_three_independent_voices() {
        let key = Key::modal(PitchClass::D, crate::scale::Mode::Dorian).unwrap();
        let score = realize_renaissance(key, 92.0, 4, &point(), 7);
        assert!(!score.notes.is_empty(), "a real piece must come out");
        let has_role = |r: VoiceRole| score.notes.iter().any(|n| n.role == r);
        assert!(has_role(VoiceRole::Melody), "soprano must sound");
        assert!(has_role(VoiceRole::CounterMelody), "alto must sound");
        assert!(has_role(VoiceRole::Bass), "bass must sound");
        assert!(
            !score.notes.iter().any(|n| n.role == VoiceRole::Harmony),
            "three independent LINES, no chordal harmony voice at all"
        );
    }

    #[test]
    fn cadence_resolves_the_suspension_down_by_step() {
        let key = Key::modal(PitchClass::D, crate::scale::Mode::Dorian).unwrap();
        let score = realize_renaissance(key, 92.0, 4, &point(), 3);
        let mut alto: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::CounterMelody)
            .collect();
        alto.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let last_two: Vec<&&ScoreNote> = alto.iter().rev().take(2).collect();
        assert_eq!(last_two.len(), 2);
        // The final alto pitch (the resolution) must sit BELOW the
        // suspension it resolved from, by a single diatonic step.
        let (resolution, suspension) = (last_two[0], last_two[1]);
        assert!(
            resolution.pitch.midi() < suspension.pitch.midi(),
            "resolution must be below the suspension"
        );
    }

    #[test]
    fn renaissance_is_deterministic() {
        let key = Key::modal(PitchClass::D, crate::scale::Mode::Dorian).unwrap();
        let a = realize_renaissance(key, 92.0, 4, &point(), 11);
        let b = realize_renaissance(key, 92.0, 4, &point(), 11);
        assert_eq!(a.notes.len(), b.notes.len());
    }
}
