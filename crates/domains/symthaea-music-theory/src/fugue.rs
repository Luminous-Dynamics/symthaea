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
//! **Species-checked counterpoint** (added one wave after the fugue
//! itself): accompanying lines — countersubjects, the figured free part —
//! are fitted against the lowest sounding voice
//! ([`crate::counterpoint::fit_against`]: strong-beat consonance, no
//! parallel perfects, minimal degree bending), and the episode walking
//! bass bends UNDER the sacrosanct soprano sequence
//! ([`fitted_bass_degree`]). Thematic material — subject entries and
//! episode head-fragments — is never altered. See
//! `species_fitting_improves_verticals_without_costing_integration` for
//! the falsification history that produced this exact division of roles.
//!
//! **Deliberately not claimed** (documented limits, not hidden ones):
//! tonal answers (the answer is real/diatonic — every degree up a fifth —
//! not the tonal-answer alteration of the head); modulation (the whole
//! piece stays in one diatonic collection; the "submediant entry" is a
//! diatonic transposition, not a key change); and off-beat dissonance
//! control (florid off-beat dissonance is deliberately out of the
//! fitter's scope). Each is a real upgrade path.

use crate::counterpoint::{CantusEvent, fit_against, has_parallel_perfect, is_consonant};
use crate::form::figuration_variation;
use crate::harmony::Key;
use crate::motif::{Motif, MotifNote};
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::scale::Scale;
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
pub(crate) fn emit(
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

/// Bend one walking-bass degree (realized at octave 3) to a consonance
/// with the upper line sounding at `onset` — the episode counterpart of
/// [`fit_against`], with the roles reversed: there the free upper line
/// bends under a fixed cantus; here the free BASS bends under the fixed
/// (thematic) soprano sequence. Same minimal-adjustment search, same
/// parallel-perfect guard via `prev` (the previous chosen (bass, soprano)
/// pitch pair). Returns the original degree when no upper note sounds or
/// no candidate within two steps works.
fn fitted_bass_degree(
    deg: i32,
    onset: Duration,
    uppers: &[CantusEvent],
    prev: &mut Option<(Pitch, Pitch)>,
    scale: Scale,
) -> i32 {
    let t = onset.beats();
    let Some(up) = uppers
        .iter()
        .copied()
        .find(|e| e.onset - 1e-9 <= t && t < e.onset + e.duration - 1e-9)
    else {
        *prev = None;
        return deg;
    };
    let ok = |d: i32, prev: &Option<(Pitch, Pitch)>| -> bool {
        let p = scale.degree_pitch(d, 3);
        is_consonant(p, up.pitch)
            && prev
                .map(|(bp, sp)| !has_parallel_perfect(bp, sp, p, up.pitch))
                .unwrap_or(true)
    };
    let chosen = if ok(deg, prev) {
        deg
    } else {
        [-1, 1, -2, 2]
            .iter()
            .map(|off| deg + off)
            .find(|cand| ok(*cand, prev))
            .unwrap_or(deg)
    };
    *prev = Some((scale.degree_pitch(chosen, 3), up.pitch));
    chosen
}

/// One held tone (degree in octave) — bass roots and cadence tones.
pub(crate) fn hold(
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
    realize_fugue_inner(key, tempo_bpm, meter, subject, seed, true)
}

/// The realization body, with the SPECIES-CHECKED COUNTERPOINT pass
/// switchable (`fit`). Production always fits (see [`realize_fugue`]);
/// tests A/B the two to keep the pass's value falsifiable — the claim
/// "voice-led verticals integrate the voices" is asserted against
/// [`crate::integration::musical_phi`], not taken on faith.
///
/// Fitting contract: **subject entries are never altered** (the theme is
/// sacrosanct — exposition statements, the inverted middle entry, the
/// stretto, the augmented final entry); every ACCOMPANYING line
/// (countersubjects, the figured free part, episode fragments) is fitted
/// against the lowest sounding voice via
/// [`crate::counterpoint::fit_against`] — strong-beat consonance,
/// passing-only weak-beat dissonance, no parallel perfects.
fn realize_fugue_inner(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
    fit: bool,
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let scale = key.scale();
    let s_beats = meter as i64;
    let bar = |n: i64| Duration::new(s_beats * n, 1);
    let half = |n2: i64| Duration::new(s_beats * n2, 2); // n2 half-bars
    let half_bar = Duration::new(s_beats, 2);

    // Cantus events (absolute onsets) for a motif emitted at `start`.
    let cantus_of = |m: &Motif, octave: i32, start: Duration| -> Vec<CantusEvent> {
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
    };
    let fitted = |line: &Motif, cantus: &[CantusEvent], octave: i32, start: Duration| -> Motif {
        if fit {
            fit_against(cantus, line, scale, octave, start.beats())
        } else {
            line.clone()
        }
    };

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
    // Bar 1's governing voice is the answer itself (no bass yet); bar 2's
    // is the bass subject entry. Countersubjects and the free part are
    // fitted; the entries they accompany are not.
    let bar1_cf = cantus_of(&ans, 4, bar(1));
    emit(
        &mut score,
        &fitted(&cs, &bar1_cf, 5, bar(1)),
        bar(1),
        Melody,
        5,
        EXPO,
        Normal,
    );
    emit(&mut score, subject, bar(2), Bass, 3, EXPO, PhraseStart);
    let bar2_cf = cantus_of(subject, 3, bar(2));
    emit(
        &mut score,
        &fitted(&cs, &bar2_cf, 4, bar(2)),
        bar(2),
        CounterMelody,
        4,
        EXPO,
        Normal,
    );
    // Soprano's free part over the bass entry: the countersubject a third
    // up, figured — parallel thirds against the alto's countersubject, the
    // one place this texture is allowed to be sweet.
    let free = figuration_variation(&cs.transpose(2), seed);
    emit(
        &mut score,
        &fitted(&free, &bar2_cf, 5, bar(2)),
        bar(2),
        Melody,
        5,
        EXPO,
        Normal,
    );

    // ── Episode 1 (bars 3-4): head sequenced down ────────────────────────
    // The fragments are the subject's HEAD — thematic, sacrosanct (the
    // first fitter draft bent them and the Φ A/B showed it destroyed more
    // motif-web integration than the consonance channel gained). The free
    // voice here is the walking BASS, so IT bends under the soprano's
    // sequence — which is also just what a composer does.
    let mut ep1_sop = Vec::new();
    for (i, step) in [0i32, -1, -2, -3].iter().enumerate() {
        let slot = bar(3) + half(i as i64);
        let m = frag.transpose(*step);
        ep1_sop.extend(cantus_of(&m, 5, slot));
        emit(
            &mut score,
            &m,
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
    let mut prev_pair = None;
    for (i, deg) in [1i32, 7, 6, 5].iter().enumerate() {
        // Walking bass: 1-7-6-5, arriving on the dominant — each step bent
        // (when fitting) to a consonance with the soprano fragment sounding
        // at its onset.
        let onset = bar(3) + half(i as i64);
        let chosen = if fit {
            fitted_bass_degree(*deg - 7, onset, &ep1_sop, &mut prev_pair, scale)
        } else {
            *deg - 7
        };
        hold(
            &mut score, chosen, onset, half_bar, Bass, 3, EPISODE, Normal,
        );
    }

    // ── Middle entry (bar 5): subject inverted, on the submediant ───────
    // The inverted entry is sacrosanct; the countersubject above it is
    // fitted against the bass pedal (the lowest sounding voice).
    let mid_cf = [CantusEvent {
        onset: bar(5).beats(),
        duration: bar(1).beats(),
        pitch: scale.degree_pitch(6 - 7, 3),
    }];
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
        &fitted(&cs.transpose(5), &mid_cf, 5, bar(5)),
        bar(5),
        Melody,
        5,
        MIDDLE,
        Normal,
    );
    hold(&mut score, 6 - 7, bar(5), bar(1), Bass, 3, MIDDLE, Normal);

    // ── Episode 2 (bar 6): head sequenced up, toward the stretto ────────
    // Same governance as episode 1: fragments sacrosanct, bass bends.
    let mut ep2_sop = Vec::new();
    for (i, step) in [0i32, 1].iter().enumerate() {
        let slot = bar(6) + half(i as i64);
        let m = frag.transpose(*step);
        ep2_sop.extend(cantus_of(&m, 5, slot));
        emit(
            &mut score,
            &m,
            slot,
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
    let mut prev_pair = None;
    for (i, deg) in [4i32, 5].iter().enumerate() {
        let onset = bar(6) + half(i as i64);
        let chosen = if fit {
            fitted_bass_degree(*deg - 7, onset, &ep2_sop, &mut prev_pair, scale)
        } else {
            *deg - 7
        };
        hold(
            &mut score, chosen, onset, half_bar, Bass, 3, EPISODE, Normal,
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

    #[test]
    fn fitted_lines_obey_the_species_contract_against_the_lowest_voice() {
        // The fitter's contract, checked on the realized score, scoped to
        // exactly what is enforced (thematic material — entries AND
        // episode fragments — is sacrosanct by design):
        // - exposition bars 1-2 (beats 4..12): fitted CS/free lines land
        //   on-beat consonant against the lowest sounding voice;
        // - middle entry (beats 20..24): the fitted soprano CS vs the bass
        //   pedal (the alto is the inverted subject entry — exempt);
        // - episode bass onsets (12,14,16,18,24,26): the FITTED walking
        //   bass vs the sacrosanct soprano fragment sounding there.
        let s = fugue(1);
        let sounding = |role: VoiceRole, t: f64| -> Option<crate::pitch::Pitch> {
            s.voice(role)
                .iter()
                .find(|n| n.onset.beats() - 1e-9 <= t && t < n.onset.beats() + n.duration.beats())
                .map(|n| n.pitch)
        };
        let mut checks: Vec<(f64, VoiceRole)> = Vec::new();
        for beat in 4..12 {
            checks.push((beat as f64, VoiceRole::Melody));
            checks.push((beat as f64, VoiceRole::CounterMelody));
        }
        for beat in 20..24 {
            checks.push((beat as f64, VoiceRole::Melody));
        }
        for (t, role) in checks {
            let Some(low) =
                sounding(VoiceRole::Bass, t).or_else(|| sounding(VoiceRole::CounterMelody, t))
            else {
                continue;
            };
            let landed = s
                .voice(role)
                .into_iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-6);
            if let Some(n) = landed {
                if n.pitch.midi() <= low.midi() {
                    continue; // this note IS the lowest (or crossed)
                }
                assert!(
                    crate::counterpoint::is_consonant(low, n.pitch),
                    "beat {t}: {role:?} lands {} against lowest {} — dissonant on-beat \
                     vertical survived the fitter",
                    n.pitch.midi(),
                    low.midi()
                );
            }
        }
        // Episode bass onsets vs the soprano fragment sounding there.
        for t in [12.0, 14.0, 16.0, 18.0, 24.0, 26.0] {
            let (Some(bass), Some(sop)) = (
                s.voice(VoiceRole::Bass)
                    .into_iter()
                    .find(|n| (n.onset.beats() - t).abs() < 1e-6)
                    .map(|n| n.pitch),
                sounding(VoiceRole::Melody, t),
            ) else {
                continue;
            };
            assert!(
                crate::counterpoint::is_consonant(bass, sop),
                "beat {t}: episode bass {} dissonant under soprano {}",
                bass.midi(),
                sop.midi()
            );
        }
    }

    #[test]
    fn species_fitting_improves_verticals_without_costing_integration() {
        // The falsification experiment this wave was built around, with
        // its ACTUAL result pinned. The original hypothesis — "species
        // fitting should RAISE Φ" — was run twice and falsified twice:
        //
        // 1. First draft (fragments bent too, strict off-beat passing
        //    rule): Φ FELL, 0.0185 → 0.0061. Bending thematic material
        //    destroyed more motif-channel integration than the consonance
        //    channel gained. Fix: thematic lines (entries AND episode
        //    fragments) sacrosanct; the walking bass bends under the
        //    soprano instead; strong-beats-only enforcement.
        // 2. Redesign: consonance channel +25% (0.330 → 0.413), motif
        //    channel preserved (0.104 → 0.105) — and Φ IDENTICAL to four
        //    decimals. Why: λ₂ is a bottleneck measure, and the fugue's
        //    minimum cut is TEMPORAL (cross-segment motif continuity,
        //    which only trigram edges span). Vertical consonance lives
        //    inside segments and cannot move a cross-segment bottleneck.
        //
        // So the honest, tested claim is the three-part one below:
        // fitting genuinely improves the verticals the metric measures,
        // preserves the motif web, and never costs integration. Raising
        // Φ itself would require improving TEMPORAL integration — motif
        // continuity across the piece's thirds — which is a composition
        // question (episodes, memory), not a voice-leading one. That
        // insight is what the falsification bought.
        let key = Key::major(PitchClass::C);
        for seed in [1u64, 5, 9] {
            let raw = realize_fugue_inner(key, 90.0, 4, &subject(), seed, false);
            let fit = realize_fugue_inner(key, 90.0, 4, &subject(), seed, true);
            let pr = crate::integration::musical_phi(&raw);
            let pf = crate::integration::musical_phi(&fit);
            let report = format!(
                "seed {seed}: fitted Φ={:.4} (cons {:.3}, motif {:.3}) vs raw \
                 Φ={:.4} (cons {:.3}, motif {:.3})",
                pf.phi,
                pf.mean_consonance_edge,
                pf.mean_trigram_edge,
                pr.phi,
                pr.mean_consonance_edge,
                pr.mean_trigram_edge
            );
            assert!(
                pf.mean_consonance_edge > pr.mean_consonance_edge,
                "verticals must measurably improve — {report}"
            );
            assert!(
                pf.mean_trigram_edge >= pr.mean_trigram_edge - 0.005,
                "the motif web must be preserved — {report}"
            );
            // f32 power-iteration noise: the two λ₂ values are identical
            // at display precision (0.0185 vs 0.0185) but differ in the
            // ~1e-5 digits. 5e-4 is far below any musically meaningful Φ
            // difference and far above the iteration noise.
            assert!(
                pf.phi >= pr.phi - 5e-4,
                "integration must never fall — {report}"
            );
        }
    }

    #[test]
    fn fitting_changes_notes_but_never_rhythm() {
        // The fitter bends degrees, never durations/onsets — the fugue's
        // rhythmic identity (and every entry-stagger/stretto test above)
        // is invariant under it.
        let key = Key::major(PitchClass::C);
        let raw = realize_fugue_inner(key, 90.0, 4, &subject(), 1, false);
        let fit = realize_fugue_inner(key, 90.0, 4, &subject(), 1, true);
        assert_ne!(raw, fit, "fitting must actually do something");
        let rhythm = |s: &Score| -> Vec<(f64, f64)> {
            let mut v: Vec<(f64, f64)> = s
                .notes
                .iter()
                .map(|n| (n.onset.beats(), n.duration.beats()))
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        assert_eq!(rhythm(&raw), rhythm(&fit));
    }
}
