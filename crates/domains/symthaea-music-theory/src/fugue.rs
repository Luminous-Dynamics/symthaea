// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fugue (style-roadmap item 3): the subject IS the piece — three voices
//! that all speak it, in a texture where every device is a way of
//! remembering it differently.
//!
//! What this module honestly builds is a **fughetta** — a compact fugue
//! with every load-bearing fugal device present and real:
//!
//! - **Exposition**: soprano states the subject alone; alto answers at the
//!   diatonic fifth (see [`answer`]) against the soprano's countersubject;
//!   bass states the subject while the alto carries the countersubject.
//! - **Episodes**: sequences of the subject's HEAD ([`head_fragment`]) —
//!   descending after the exposition, ascending back before the stretto —
//!   over a stepwise bass. Episodes are where a fugue breathes between
//!   entries; sequencing the head keeps them ABOUT the subject.
//! - **Middle entry**: the subject INVERTED, on the submediant — the
//!   darkened, upside-down memory of the theme.
//! - **Stretto**: overlapping entries at HALF the subject's length — the
//!   subject crowding in on itself, the fugue's traditional peak.
//! - **Final entry**: the subject AUGMENTED (doubled note values) in the
//!   bass — the peroration — its tail bent to the tonic under a simple
//!   cadence in the upper voices.
//!
//! **Deliberately not claimed** (documented limits, not hidden ones):
//! species-checked counterpoint (the countersubject is DERIVED — retrograde
//! inversion, transposed to favor imperfect consonances — and collisions
//! are managed by register separation, not note-level voice-leading); tonal
//! answers (the answer is real/diatonic — every degree up a fifth — not the
//! tonal-answer alteration of the head); and modulation (the whole piece
//! stays in one diatonic collection; the "submediant entry" is a diatonic
//! transposition, not a key change). Each is a real upgrade path.

use crate::form::figuration_variation;
use crate::harmony::Key;
use crate::motif::{Motif, MotifNote};
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};

/// The answer: the subject transposed up a diatonic fifth (+4 degrees) —
/// the defining move of a fugue exposition. This is a REAL answer (every
/// interval preserved diatonically), not a tonal answer.
pub(crate) fn answer(subject: &Motif) -> Motif {
    subject.transpose(4)
}

/// The derived countersubject: retrograde inversion of the subject,
/// transposed up a third so that against the subject/answer it tends to
/// land on imperfect consonances. Retrograde inversion gives it the
/// opposite contour AND the reversed accent pattern — genuinely
/// complementary material that is still, secretly, the subject.
pub(crate) fn countersubject(subject: &Motif) -> Motif {
    let pivot = subject.notes.iter().find_map(|x| x.degree).unwrap_or(1);
    subject.invert(pivot).retrograde().transpose(2)
}

/// The subject's head, truncated to exactly `beats` — the episode
/// sequencing unit. Notes are taken in order; the note that crosses the
/// boundary is shortened to fit, so the fragment's total duration is exact
/// (episodes tile half-bar slots and must never bleed into the next slot).
pub(crate) fn head_fragment(subject: &Motif, beats: Duration) -> Motif {
    let target = beats.beats();
    let mut out = Vec::new();
    let mut acc = 0.0;
    for n in &subject.notes {
        if acc >= target - 1e-9 {
            break;
        }
        let remaining = target - acc;
        if n.duration.beats() <= remaining + 1e-9 {
            out.push(*n);
            acc += n.duration.beats();
        } else {
            // Shorten the crossing note to exactly fill the fragment.
            // `remaining` is always a sum/difference of the subject's own
            // rational durations and the rational target, so a denominator
            // large enough to be exact exists; 480 (the MIDI-tick-like
            // resolution used across the crate) covers every duration the
            // motif banks and hook cells produce.
            let num = (remaining * 480.0).round() as i64;
            out.push(MotifNote {
                degree: n.degree,
                duration: Duration::new(num, 480),
            });
            acc = target;
        }
    }
    Motif { notes: out }
}

/// Emit `motif` into `score` at absolute onset `start`, realizing degrees
/// in `octave`. The first pitched note carries `entry_emphasis` (subject
/// entries are phrase starts; the stretto's last entry is the climax).
fn emit(
    score: &mut Score,
    motif: &Motif,
    start: Duration,
    role: VoiceRole,
    octave: i32,
    intensity: f32,
    entry_emphasis: Emphasis,
) {
    let scale = score.key.scale();
    let mut t = start;
    let mut first_pitched = true;
    for n in &motif.notes {
        if let Some(d) = n.degree {
            let emphasis = if first_pitched {
                entry_emphasis
            } else {
                Emphasis::Normal
            };
            first_pitched = false;
            score.push(ScoreNote {
                pitch: scale.degree_pitch(d, octave),
                onset: t,
                duration: n.duration,
                velocity: (0.72 * intensity).clamp(0.0, 1.0),
                role,
                emphasis,
                section_intensity: intensity,
            });
        }
        t = t + n.duration;
    }
}

/// One held tone (degree in octave) — bass roots and cadence tones.
fn hold(
    score: &mut Score,
    degree: i32,
    start: Duration,
    duration: Duration,
    role: VoiceRole,
    octave: i32,
    intensity: f32,
    emphasis: Emphasis,
) {
    let scale = score.key.scale();
    score.push(ScoreNote {
        pitch: scale.degree_pitch(degree, octave),
        onset: start,
        duration,
        velocity: (0.68 * intensity).clamp(0.0, 1.0),
        role,
        emphasis,
        section_intensity: intensity,
    });
}

/// Compose the fughetta. `subject` arrives already hook-grafted (the main
/// pipeline's naming machinery — the subject is the piece's name, stated
/// by every voice). Fixed 11-bar shape:
///
/// | bars | what |
/// |------|------|
/// | 0-2  | exposition (S alone / A answers + S countersubject / B subject + A countersubject + S figured free part) |
/// | 3-4  | episode 1: head sequenced DOWN, bass walks 1-7-6-5 |
/// | 5    | middle entry: subject inverted on the submediant |
/// | 6    | episode 2: head sequenced UP, bass walks 4-5 |
/// | 7-8  | stretto: entries at half-subject stagger (S, A at +S/2, B at +S) |
/// | 9-10 | final: subject augmented in the bass, tail bent to the tonic, cadence above |
///
/// Long-range intensity arc: exposition 0.85 → episodes 0.9 → middle 1.0 →
/// stretto 1.15 (the peak) → final 0.95 — the same establish/depart/
/// peak/settle grammar as [`crate::form::SectionRole::intensity`].
pub(crate) fn realize_fugue(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let s_beats = meter as i64;
    let bar = |n: i64| Duration::new(s_beats * n, 1);
    let half = |n2: i64| Duration::new(s_beats * n2, 2); // n2 half-bars
    let half_bar = Duration::new(s_beats, 2);

    let ans = answer(subject);
    let cs = countersubject(subject);
    let frag = head_fragment(subject, half_bar);
    let pivot = subject.notes.iter().find_map(|x| x.degree).unwrap_or(1);

    use Emphasis::{Cadential, Climax, Normal, PhraseStart};
    use VoiceRole::{Bass, CounterMelody, Melody};
    const EXPO: f32 = 0.85;
    const EPISODE: f32 = 0.9;
    const MIDDLE: f32 = 1.0;
    const STRETTO: f32 = 1.15;
    const FINAL: f32 = 0.95;

    // ── Exposition (bars 0-2) ────────────────────────────────────────────
    emit(&mut score, subject, bar(0), Melody, 5, EXPO, PhraseStart);
    emit(
        &mut score,
        &ans,
        bar(1),
        CounterMelody,
        4,
        EXPO,
        PhraseStart,
    );
    emit(&mut score, &cs, bar(1), Melody, 5, EXPO, Normal);
    emit(&mut score, subject, bar(2), Bass, 3, EXPO, PhraseStart);
    emit(&mut score, &cs, bar(2), CounterMelody, 4, EXPO, Normal);
    // Soprano's free part over the bass entry: the countersubject a third
    // up, figured — parallel thirds against the alto's countersubject, the
    // one place this texture is allowed to be sweet.
    let free = figuration_variation(&cs.transpose(2), seed);
    emit(&mut score, &free, bar(2), Melody, 5, EXPO, Normal);

    // ── Episode 1 (bars 3-4): head sequenced down ────────────────────────
    for (i, step) in [0i32, -1, -2, -3].iter().enumerate() {
        let slot = bar(3) + half(i as i64);
        emit(
            &mut score,
            &frag.transpose(*step),
            slot,
            Melody,
            5,
            EPISODE,
            if i == 0 { PhraseStart } else { Normal },
        );
    }
    // Alto imitates a half-bar behind, a third below the soprano's steps
    // (three statements so it lands exactly on the middle entry's downbeat).
    for (i, step) in [-2i32, -3, -4].iter().enumerate() {
        let slot = bar(3) + half(i as i64 + 1);
        emit(
            &mut score,
            &frag.transpose(*step),
            slot,
            CounterMelody,
            4,
            EPISODE,
            Normal,
        );
    }
    for (i, deg) in [1i32, 7, 6, 5].iter().enumerate() {
        // Walking bass: 1-7-6-5, arriving on the dominant.
        hold(
            &mut score,
            *deg - 7, // stay below the tenor register: one octave down
            bar(3) + half(i as i64),
            half_bar,
            Bass,
            3,
            EPISODE,
            Normal,
        );
    }

    // ── Middle entry (bar 5): subject inverted, on the submediant ───────
    emit(
        &mut score,
        &subject.invert(pivot).transpose(5),
        bar(5),
        CounterMelody,
        4,
        MIDDLE,
        PhraseStart,
    );
    emit(
        &mut score,
        &cs.transpose(5),
        bar(5),
        Melody,
        5,
        MIDDLE,
        Normal,
    );
    hold(&mut score, 6 - 7, bar(5), bar(1), Bass, 3, MIDDLE, Normal);

    // ── Episode 2 (bar 6): head sequenced up, toward the stretto ────────
    for (i, step) in [0i32, 1].iter().enumerate() {
        emit(
            &mut score,
            &frag.transpose(*step),
            bar(6) + half(i as i64),
            Melody,
            5,
            EPISODE,
            if i == 0 { PhraseStart } else { Normal },
        );
    }
    emit(
        &mut score,
        &frag.transpose(5),
        bar(6) + half(1),
        CounterMelody,
        4,
        EPISODE,
        Normal,
    );
    for (i, deg) in [4i32, 5].iter().enumerate() {
        hold(
            &mut score,
            *deg - 7,
            bar(6) + half(i as i64),
            half_bar,
            Bass,
            3,
            EPISODE,
            Normal,
        );
    }

    // ── Stretto (bars 7-8): entries at HALF the subject's length ────────
    emit(&mut score, subject, bar(7), Melody, 5, STRETTO, PhraseStart);
    emit(
        &mut score,
        &ans,
        bar(7) + half_bar,
        CounterMelody,
        4,
        STRETTO,
        PhraseStart,
    );
    emit(&mut score, subject, bar(8), Bass, 3, STRETTO, Climax);

    // ── Final entry (bars 9-10): augmented, in the bass, tail to tonic ──
    let mut augmented = subject.scale_rhythm(2, 1);
    if let Some(last) = augmented
        .notes
        .iter_mut()
        .rev()
        .find(|n| n.degree.is_some())
    {
        // Bend the tail to the tonic — the compositional intervention every
        // final entry gets so the piece can actually cadence.
        last.degree = Some(1);
    }
    emit(&mut score, &augmented, bar(9), Bass, 2, FINAL, PhraseStart);
    // Simple cadence above: soprano 5 | 2-1, alto 3 | leading-tone-1.
    hold(&mut score, 5, bar(9), bar(1), Melody, 5, FINAL, Normal);
    hold(
        &mut score,
        3,
        bar(9),
        bar(1),
        CounterMelody,
        4,
        FINAL,
        Normal,
    );
    hold(&mut score, 2, bar(10), half_bar, Melody, 5, FINAL, Normal);
    hold(
        &mut score,
        1,
        bar(10) + half_bar,
        half_bar,
        Melody,
        5,
        FINAL,
        Cadential,
    );
    hold(
        &mut score,
        0,
        bar(10),
        half_bar,
        CounterMelody,
        4,
        FINAL,
        Normal,
    );
    hold(
        &mut score,
        1,
        bar(10) + half_bar,
        half_bar,
        CounterMelody,
        4,
        FINAL,
        Cadential,
    );

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn subject() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::eighth()),
            (2, Duration::eighth()),
            (5, Duration::half()),
        ])
    }

    fn fugue(seed: u64) -> Score {
        realize_fugue(Key::major(PitchClass::C), 90.0, 4, &subject(), seed)
    }

    #[test]
    fn answer_is_the_subject_at_the_diatonic_fifth() {
        let a = answer(&subject());
        for (s, n) in subject().notes.iter().zip(a.notes.iter()) {
            assert_eq!(n.degree, s.degree.map(|d| d + 4));
            assert_eq!(n.duration, s.duration);
        }
    }

    #[test]
    fn countersubject_is_derived_and_duration_preserving() {
        let cs = countersubject(&subject());
        assert_eq!(cs.total_duration(), subject().total_duration());
        // Retrograde inversion: genuinely different material...
        assert_ne!(cs, subject());
        // ...that is still, secretly, the subject: undoing the derivation
        // (un-transpose, un-retrograde, un-invert) recovers it exactly.
        assert_eq!(cs.transpose(-2).retrograde().invert(1), subject());
    }

    #[test]
    fn head_fragment_fills_exactly_the_requested_beats() {
        let frag = head_fragment(&subject(), Duration::new(2, 1));
        assert_eq!(frag.total_duration().beats(), 2.0);
        // First notes match the subject head verbatim until the boundary.
        assert_eq!(frag.notes[0], subject().notes[0]);
        // And a boundary that crosses mid-note truncates, never spills.
        let cross = head_fragment(&subject(), Duration::new(3, 2));
        assert_eq!(cross.total_duration().beats(), 1.5);
    }

    #[test]
    fn exposition_entries_are_staggered_one_bar_apart() {
        let s = fugue(1);
        let first_onset = |role: VoiceRole| {
            s.voice(role)
                .first()
                .map(|n| n.onset.beats())
                .expect("voice must exist")
        };
        assert_eq!(first_onset(VoiceRole::Melody), 0.0);
        assert_eq!(first_onset(VoiceRole::CounterMelody), 4.0);
        assert_eq!(first_onset(VoiceRole::Bass), 8.0);
    }

    #[test]
    fn all_three_voices_state_the_subject_rhythm() {
        // Every voice must carry a full subject statement somewhere — the
        // defining democracy of a fugue. Rhythm identity (the subject's
        // duration sequence appearing consecutively) is the check; the
        // stretto and exposition provide the statements.
        let s = fugue(1);
        let rhythm: Vec<f64> = subject().notes.iter().map(|n| n.duration.beats()).collect();
        for role in [VoiceRole::Melody, VoiceRole::CounterMelody, VoiceRole::Bass] {
            let notes = s.voice(role);
            let durs: Vec<f64> = notes.iter().map(|n| n.duration.beats()).collect();
            let found = durs.windows(rhythm.len()).any(|w| {
                w.iter()
                    .zip(rhythm.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-9)
            });
            assert!(found, "{role:?} never states the subject rhythm");
        }
    }

    #[test]
    fn stretto_entries_genuinely_overlap() {
        // The alto's stretto entry starts a HALF subject after the
        // soprano's — while the soprano's statement is still sounding.
        // That overlap is the definition of stretto.
        let s = fugue(1);
        let stretto_start = 7.0 * 4.0;
        let soprano_entry = s
            .voice(VoiceRole::Melody)
            .iter()
            .find(|n| n.onset.beats() >= stretto_start)
            .map(|n| n.onset.beats())
            .unwrap();
        let alto_entry = s
            .voice(VoiceRole::CounterMelody)
            .iter()
            .find(|n| n.onset.beats() >= stretto_start)
            .map(|n| n.onset.beats())
            .unwrap();
        let subject_len = subject().total_duration().beats();
        assert_eq!(alto_entry - soprano_entry, subject_len / 2.0);
        assert!(alto_entry - soprano_entry < subject_len);
    }

    #[test]
    fn final_entry_is_augmented_and_ends_on_the_tonic() {
        let s = fugue(1);
        let bass = s.voice(VoiceRole::Bass);
        // The final-entry notes (from bar 9) carry doubled durations.
        let final_notes: Vec<_> = bass
            .iter()
            .filter(|n| n.onset.beats() >= 9.0 * 4.0)
            .collect();
        let final_len: f64 = final_notes.iter().map(|n| n.duration.beats()).sum();
        assert_eq!(final_len, 2.0 * subject().total_duration().beats());
        // Its last note is the tonic.
        assert_eq!(
            final_notes.last().unwrap().pitch.pitch_class(),
            PitchClass::C
        );
        // And the soprano cadences onto the tonic too.
        assert_eq!(
            s.voice(VoiceRole::Melody)
                .last()
                .unwrap()
                .pitch
                .pitch_class(),
            PitchClass::C
        );
    }

    #[test]
    fn stretto_is_the_intensity_peak() {
        let s = fugue(1);
        let max = s
            .notes
            .iter()
            .map(|n| n.section_intensity)
            .fold(f32::MIN, f32::max);
        assert_eq!(max, 1.15);
        // And the exposition is the calmest.
        let min = s
            .notes
            .iter()
            .map(|n| n.section_intensity)
            .fold(f32::MAX, f32::min);
        assert_eq!(min, 0.85);
    }

    #[test]
    fn every_voice_is_monophonic() {
        // Fugue voices are LINES — within one voice, no note may sound
        // while the previous still does.
        let s = fugue(1);
        for role in [VoiceRole::Melody, VoiceRole::CounterMelody, VoiceRole::Bass] {
            let notes = s.voice(role);
            for pair in notes.windows(2) {
                assert!(
                    pair[1].onset.beats()
                        >= pair[0].onset.beats() + pair[0].duration.beats() - 1e-9,
                    "{role:?} overlaps itself at beat {}",
                    pair[1].onset.beats()
                );
            }
        }
    }

    #[test]
    fn fugue_is_deterministic_and_eleven_bars() {
        let a = fugue(7);
        let b = fugue(7);
        assert_eq!(a, b);
        assert_eq!(a.total_beats.beats(), 11.0 * 4.0);
    }
}
