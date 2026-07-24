// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Accompaniment patterns: how a voiced chord is laid out IN TIME within its
//! measure.
//!
//! Until this existed the harmony voice had exactly one texture: every voiced
//! tone struck at the bar line and held for the whole measure — a static pad
//! under the melody, whatever the style. A pattern changes only the RHYTHM of
//! the accompaniment: the pitches always come from the same voice-led chord
//! (`voicing::lead_upper`), so functional harmony, voice leading, and the
//! melody's chord-tone snapping are untouched. That containment is the safety
//! property (tested): a pattern can never introduce a wrong note, only a new
//! placement of right ones.

use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

/// A one-measure accompaniment figure. All variants realize the SAME voiced
/// chord; they differ only in when (and in which grouping) its tones sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accompaniment {
    /// The original texture: every tone at the bar line, held for the bar.
    Block,
    /// Ascending eighth-note arpeggio cycling through the chord tones —
    /// the guitar-fingerpicking / broken-chord texture.
    Arpeggio,
    /// Alberti figure (low, high, middle, high in eighths) — the classical
    /// keyboard accompaniment idiom.
    Alberti,
    /// Oom-pah-pah: beat 1 belongs to the bass alone; the full chord answers
    /// as a quarter-note stab on every later beat. THE waltz texture (and it
    /// generalizes to 4/4 as oom-pah-pah-pah).
    OomPah,
    /// Syncopated comping: two short full-chord stabs on offbeats — a
    /// playful, rhythmically active texture.
    Comp,
    /// The habanera — the first RHYTHM CELL: a figure whose identity is
    /// rhythm AND accent together (dotted anchor, pickup, two answering
    /// beats: DUM..da DUM DUM). Unlike the patterns above, its per-event
    /// velocities are part of the cell — accent hierarchy as musical
    /// identity, the habit the Tango style teaches the whole engine
    /// (blues shuffle, baroque dance cells, and minimalist pulses are the
    /// same mechanism with different tables). Assumes a duple bar; in
    /// shorter meters events beyond the bar are dropped.
    Habanera,
    /// The FIVE-GAIT — the second rhythm cell: quintuple meter's classic
    /// 3+2 grouping spelled as accents (anchor · pickup · step | second
    /// anchor · release). Without it, 5/4 was a 4/4 pattern looping
    /// arithmetically — metrically legal, gaitless. Assumes a 5-beat bar;
    /// shorter meters drop the tail events.
    FiveGait,
    /// The JIG-GAIT — the third rhythm cell, the Celtic style's habit:
    /// sextuple meter's 3+3 lilt (DUM-da-DA · dum-da-da — two mirrored
    /// dotted-anchor groups, the second lighter than the first, the exact
    /// FiveGait mechanism extended from 3+2 to 3+3). Without it, 6/4 would
    /// be a 4/4 pattern looping arithmetically the same way 5/4 was before
    /// FiveGait. Assumes a 6-beat bar; shorter meters drop the tail events.
    JigGait,
    /// The SHUFFLE — the fourth rhythm cell, the Blues style's habit and
    /// the literal "blues shuffle" the Habanera doc predicted this
    /// mechanism would generalize to. A boogie-woogie bass bounce: STRICT
    /// root-fifth alternation (not a walk through every voiced tone, the
    /// way Arpeggio does) in eighths, accented strong-weak-strong-weak —
    /// the pattern real shuffle timing (`TextureSpec::swing`) then swings
    /// at the performance layer. Identity here is the NOTE CHOICE (root,
    /// fifth, root, fifth — never the third), not the rhythm alone.
    Shuffle,
    /// The MONTUNO — the fifth rhythm cell, and the first that is not a
    /// single repeating bar: son clave (3-2) is a TWO-BAR cycle, and this
    /// pattern alternates a three-side (tresillo: three stabs at the
    /// classic 3-3-2 grouping) with a two-side (two backbeat-adjacent
    /// stabs) depending on which absolute bar it lands on. No prior cell
    /// crosses a bar boundary to know its own identity; the montuno's
    /// identity is only visible across two bars, and it never resets at a
    /// section boundary (real clave never restarts mid-tune either). Paired
    /// with a tumbao bass (`composer.rs::realize_bass_measures`) whose
    /// onsets are chosen to interlock with — never coincide with — this
    /// pattern's onsets on either side: the "rhythmic conversation" the
    /// Afro-Cuban style exists to teach, made into a checkable non-overlap
    /// invariant rather than a vibe.
    Montuno,
    /// The COMPÁS-GAIT — the sixth rhythm cell, and the first ASYMMETRIC
    /// twelve-count: flamenco's 12-beat compás groups 3+3+2+2+2, accenting
    /// only counts 3, 6, 8, 10, 12 (0-indexed onsets 2, 5, 7, 9, 11) and
    /// leaving every other beat SILENT — every prior cell fills its whole
    /// bar with some figure; this one is defined as much by its rests as
    /// its stabs. Accent hierarchy marks 12 (the cycle's close) strongest,
    /// 6 (its midpoint) second, and 3/8/10 lighter — a real hierarchy
    /// inside a single bar rather than a two-group split. Assumes a
    /// 12-beat meter; shorter meters drop the tail events, same contract
    /// as every earlier gait cell.
    CompasGait,
    /// BOSSA COMP — the seventh rhythm cell, and the first defined by an
    /// ABSENCE of silence rather than a presence of accent: every prior
    /// cell (Habanera through Compás-Gait) alternates hits with rests.
    /// This one syncopates its onsets (0, 1.5, 3.0 — the same tresillo
    /// timing lineage bossa genuinely shares with Afro-Cuban rhythm) but
    /// chains each stab's duration exactly into the next one's onset, so
    /// the chord NEVER stops ringing — "floating" legato harmony instead
    /// of punctuated stabs, at consistently soft velocity ("understated
    /// accompaniment"). Reusable habit: syncopated timing and continuous
    /// sustain aren't opposites — a groove can be both at once.
    BossaComp,
}

/// Which half of the two-bar son-clave cycle a given bar falls on. The
/// three-side carries the tresillo (3-3-2) grouping; the two-side answers
/// with two backbeat-adjacent stabs. Bar parity is computed from the
/// ABSOLUTE bar index (bar_onset / meter), so the cycle never resets at a
/// section or phrase boundary — matching how clave behaves in real practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaveSide {
    Three,
    Two,
}

pub(crate) fn clave_side(bar_onset: Duration, meter_beats: f64) -> ClaveSide {
    let bar_index = (bar_onset.beats() / meter_beats.max(1e-9)).round() as i64;
    if bar_index.rem_euclid(2) == 0 {
        ClaveSide::Three
    } else {
        ClaveSide::Two
    }
}

impl Accompaniment {
    /// Realize one measure of this pattern from an already voice-led chord.
    /// `voiced` comes straight from [`crate::voicing::lead_upper`]; the
    /// pattern never alters which pitches exist, only their onsets/durations.
    #[allow(clippy::too_many_arguments)] // mirrors realize_harmony_measures' state set
    pub(crate) fn realize_measure(
        self,
        score: &mut Score,
        voiced: &[Pitch],
        bar_onset: Duration,
        meter_beats: f64,
        velocity: f32,
        section_intensity: f32,
    ) {
        if voiced.is_empty() {
            return;
        }
        let bar = Duration::new(meter_beats as i64, 1);
        let push = |score: &mut Score, pitch: Pitch, onset: Duration, dur: Duration, vel: f32| {
            score.push(ScoreNote {
                pitch,
                onset,
                duration: dur,
                velocity: vel,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
        };
        // Patterns that walk single tones want them in register order; the
        // voice-led ordering (which optimizes movement, not register) stays
        // authoritative for which pitches exist.
        let mut ascending: Vec<Pitch> = voiced.to_vec();
        ascending.sort_by_key(|p| p.midi());

        match self {
            Accompaniment::Block => {
                for &pitch in voiced {
                    push(score, pitch, bar_onset, bar, velocity);
                }
            }
            Accompaniment::Arpeggio => {
                let eighth = Duration::eighth();
                let steps = (meter_beats * 2.0) as i64; // eighths per bar
                for i in 0..steps {
                    let pitch = ascending[(i as usize) % ascending.len()];
                    // Light metric shape: beat-start eighths slightly stronger.
                    let vel = if i % 2 == 0 {
                        velocity
                    } else {
                        velocity * 0.85
                    };
                    push(score, pitch, bar_onset + eighth.scale(i, 1), eighth, vel);
                }
            }
            Accompaniment::Alberti => {
                let eighth = Duration::eighth();
                let n = ascending.len();
                // low – high – middle – high (n=3: 0,2,1,2; n=4: 0,3,1,3).
                let figure = [0usize, n - 1, (n - 1) / 2, n - 1];
                let steps = (meter_beats * 2.0) as i64;
                for i in 0..steps {
                    let pitch = ascending[figure[(i as usize) % 4].min(n - 1)];
                    let vel = if i % 4 == 0 {
                        velocity
                    } else {
                        velocity * 0.85
                    };
                    push(score, pitch, bar_onset + eighth.scale(i, 1), eighth, vel);
                }
            }
            Accompaniment::OomPah => {
                let beat = Duration::new(1, 1);
                for b in 1..(meter_beats as i64) {
                    for &pitch in voiced {
                        push(
                            score,
                            pitch,
                            bar_onset + beat.scale(b, 1),
                            beat,
                            velocity * 0.9,
                        );
                    }
                }
            }
            Accompaniment::Comp => {
                let eighth = Duration::eighth();
                let m = meter_beats as i64;
                // Offbeat stabs: the "and" before each half-bar boundary
                // (4/4 → beats 1.5 and 3.5; 3/4 → 1.0 and 2.5).
                let stabs = [Duration::new(m - 1, 2), Duration::new(2 * m - 1, 2)];
                for onset in stabs {
                    for &pitch in voiced {
                        push(score, pitch, bar_onset + onset, eighth, velocity);
                    }
                }
            }
            Accompaniment::FiveGait => {
                // The 3+2 table: (onset, duration, accent, anchor?).
                // Beat 0 anchors the THREE group (lowest tone alone,
                // dotted); beat 3 anchors the TWO group (full chord,
                // near-full accent) — the grouping IS the accents.
                let events: [(i64, i64, i64, i64, f32, bool); 5] = [
                    (0, 1, 3, 2, 1.0, true),   // DUM — the three-group anchor
                    (3, 2, 1, 2, 0.65, false), // da — pickup
                    (2, 1, 1, 1, 0.8, false),  // step
                    (3, 1, 1, 1, 0.92, false), // DUM — the two-group anchor
                    (4, 1, 1, 1, 0.7, false),  // release
                ];
                for (on_n, on_d, d_n, d_d, accent, anchor) in events {
                    let onset = Duration::new(on_n, on_d);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue;
                    }
                    let dur = Duration::new(d_n, d_d);
                    if anchor {
                        push(
                            score,
                            ascending[0],
                            bar_onset + onset,
                            dur,
                            velocity * accent,
                        );
                    } else {
                        for &pitch in voiced {
                            push(score, pitch, bar_onset + onset, dur, velocity * accent);
                        }
                    }
                }
            }
            Accompaniment::JigGait => {
                // Two mirrored dotted-anchor groups (3+3): the first
                // group's anchor is the strongest event in the bar (the
                // jig's characteristic long note), the second group's
                // anchor is lighter — the same anchor-then-echo shape
                // FiveGait uses for 3+2, extended by one more group.
                let events: [(i64, i64, i64, i64, f32, bool); 6] = [
                    (0, 1, 3, 2, 1.0, true),   // DUM — first group's dotted anchor
                    (3, 2, 1, 2, 0.65, false), // da — pickup
                    (2, 1, 1, 1, 0.85, false), // DA — group arrives
                    (3, 1, 3, 2, 0.92, true),  // dum — second group's dotted anchor
                    (9, 2, 1, 2, 0.6, false),  // da — pickup
                    (5, 1, 1, 1, 0.78, false), // da — group arrives
                ];
                for (on_n, on_d, d_n, d_d, accent, anchor) in events {
                    let onset = Duration::new(on_n, on_d);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue; // shorter meters drop the tail events
                    }
                    let dur = Duration::new(d_n, d_d);
                    if anchor {
                        push(
                            score,
                            ascending[0],
                            bar_onset + onset,
                            dur,
                            velocity * accent,
                        );
                    } else {
                        for &pitch in voiced {
                            push(score, pitch, bar_onset + onset, dur, velocity * accent);
                        }
                    }
                }
            }
            Accompaniment::Shuffle => {
                // Strict root-fifth alternation in eighths — a boogie
                // bounce, not a walk through every voiced tone (that's
                // Arpeggio's job). Accent hierarchy: strongest on beats 1
                // and 3, a lighter on-beat on 2 and 4, weakest on every
                // offbeat — `texture.swing` does the actual swung timing
                // at the performance layer, so this table only needs to
                // carry the note choice and the accent shape.
                let eighth = Duration::eighth();
                let root = ascending[0];
                let fifth = *ascending.last().unwrap();
                let steps = (meter_beats * 2.0) as i64;
                for i in 0..steps {
                    let onset = eighth.scale(i, 1);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue;
                    }
                    let pitch = if i % 2 == 0 { root } else { fifth };
                    let accent = if i % 4 == 0 {
                        1.0
                    } else if i % 2 == 0 {
                        0.85
                    } else {
                        0.55
                    };
                    push(score, pitch, bar_onset + onset, eighth, velocity * accent);
                }
            }
            Accompaniment::Habanera => {
                // The cell table: (onset, duration, accent, anchor?). The
                // anchor is the LOWEST tone alone (the habanera's tango
                // gravity); the answers are full chords. Accents are the
                // point: the figure without its accent hierarchy is just
                // notes.
                let events: [(i64, i64, i64, i64, f32, bool); 4] = [
                    (0, 1, 3, 2, 1.0, true),  // DUM — dotted anchor
                    (3, 2, 1, 2, 0.7, false), // da — the pickup
                    (2, 1, 1, 1, 0.9, false), // DUM
                    (3, 1, 1, 1, 0.8, false), // DUM
                ];
                for (on_n, on_d, d_n, d_d, accent, anchor) in events {
                    let onset = Duration::new(on_n, on_d);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue; // shorter meters drop the tail events
                    }
                    let dur = Duration::new(d_n, d_d);
                    if anchor {
                        push(
                            score,
                            ascending[0],
                            bar_onset + onset,
                            dur,
                            velocity * accent,
                        );
                    } else {
                        for &pitch in voiced {
                            push(score, pitch, bar_onset + onset, dur, velocity * accent);
                        }
                    }
                }
            }
            Accompaniment::Montuno => {
                // Three-side: the tresillo (3-3-2 in eighths) — stabs at
                // 0, 1.5, 3.0. Two-side: two backbeat-adjacent stabs at
                // 1.0, 2.0. Both are full-chord stabs (montuno voices the
                // whole triad, unlike the anchor cells above which reduce
                // to a single tone) — the piano-montuno idiom is a chordal
                // pattern, not a bass-register anchor figure.
                let events: &[(i64, i64)] = match clave_side(bar_onset, meter_beats) {
                    ClaveSide::Three => &[(0, 1), (3, 2), (3, 1)],
                    ClaveSide::Two => &[(1, 1), (2, 1)],
                };
                let eighth = Duration::eighth();
                for &(on_n, on_d) in events {
                    let onset = Duration::new(on_n, on_d);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue;
                    }
                    for &pitch in voiced {
                        push(score, pitch, bar_onset + onset, eighth, velocity);
                    }
                }
            }
            Accompaniment::CompasGait => {
                // (onset, accent). Only 5 of 12 beats sound; the rest are
                // silence — the compás's identity is the GAP as much as
                // the hit. Quarter-note stabs (a rasgueado punctuation, not
                // a held chord).
                let events: [(i64, f32); 5] = [
                    (2, 0.75), // "3" — first group's close
                    (5, 0.85), // "6" — the cycle's midpoint
                    (7, 0.7),  // "8"
                    (9, 0.7),  // "10"
                    (11, 1.0), // "12" — the cycle's close, strongest
                ];
                let quarter = Duration::quarter();
                for (on_n, accent) in events {
                    let onset = Duration::new(on_n, 1);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue;
                    }
                    for &pitch in voiced {
                        push(score, pitch, bar_onset + onset, quarter, velocity * accent);
                    }
                }
            }
            Accompaniment::BossaComp => {
                // (onset, duration, accent) — durations chain EXACTLY
                // (0..1.5, 1.5..3.0, 3.0..4.0): zero silence, zero
                // overlap, a continuous legato chain across the bar.
                let events: [(i64, i64, i64, i64, f32); 3] = [
                    (0, 1, 3, 2, 0.75), // floating open, softest lean-in
                    (3, 2, 3, 2, 0.65), // the syncopated "and" — quietest
                    (3, 1, 1, 1, 0.72), // closing lean toward the next bar
                ];
                for (on_n, on_d, d_n, d_d, accent) in events {
                    let onset = Duration::new(on_n, on_d);
                    if onset.beats() >= meter_beats - 1e-9 {
                        continue;
                    }
                    let dur = Duration::new(d_n, d_d);
                    for &pitch in voiced {
                        push(score, pitch, bar_onset + onset, dur, velocity * accent);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn triad() -> Vec<Pitch> {
        // C4, E4, G4 (deliberately unsorted to prove register sorting).
        vec![
            Pitch::new(PitchClass::E, 4),
            Pitch::new(PitchClass::C, 4),
            Pitch::new(PitchClass::G, 4),
        ]
    }

    fn realize(pattern: Accompaniment, meter: f64) -> Vec<ScoreNote> {
        let mut score = Score::new(
            crate::harmony::Key::major(PitchClass::C),
            100.0,
            meter as u8,
        );
        pattern.realize_measure(&mut score, &triad(), Duration::zero(), meter, 0.5, 1.0);
        score.notes
    }

    #[test]
    fn block_is_the_original_texture() {
        let notes = realize(Accompaniment::Block, 4.0);
        assert_eq!(notes.len(), 3);
        for n in &notes {
            assert_eq!(n.onset, Duration::zero());
            assert_eq!(n.duration, Duration::new(4, 1));
        }
    }

    #[test]
    fn arpeggio_fills_the_bar_exactly_with_eighths() {
        let notes = realize(Accompaniment::Arpeggio, 4.0);
        assert_eq!(notes.len(), 8);
        // Contiguous eighths from bar start to bar end.
        for (i, n) in notes.iter().enumerate() {
            assert_eq!(n.onset, Duration::eighth().scale(i as i64, 1));
            assert_eq!(n.duration, Duration::eighth());
        }
        let last = notes.last().unwrap();
        assert_eq!(last.onset + last.duration, Duration::new(4, 1));
        // Ascends in register from the lowest tone.
        assert_eq!(notes[0].pitch.midi(), 60); // C4
        assert!(notes[1].pitch.midi() > notes[0].pitch.midi());
    }

    #[test]
    fn alberti_walks_low_high_middle_high() {
        let notes = realize(Accompaniment::Alberti, 4.0);
        let midis: Vec<u8> = notes.iter().map(|n| n.pitch.midi()).collect();
        // C4 E4 G4 sorted → low=60, mid=64, high=67.
        assert_eq!(&midis[..4], &[60, 67, 64, 67]);
        assert_eq!(&midis[4..8], &[60, 67, 64, 67]);
    }

    #[test]
    fn oom_pah_leaves_beat_one_to_the_bass() {
        let notes = realize(Accompaniment::OomPah, 3.0);
        // 3/4: chords on beats 2 and 3 only (onsets 1 and 2), quarter each.
        assert_eq!(notes.len(), 6);
        assert!(notes.iter().all(|n| n.onset != Duration::zero()));
        let onsets: std::collections::BTreeSet<i64> = notes
            .iter()
            .map(|n| (n.onset.beats() * 2.0) as i64)
            .collect();
        assert_eq!(onsets, [2i64, 4].into_iter().collect()); // beats 1.0, 2.0 in half-beat units
        assert!(notes.iter().all(|n| n.duration == Duration::new(1, 1)));
    }

    #[test]
    fn comp_stabs_land_on_offbeats() {
        let notes = realize(Accompaniment::Comp, 4.0);
        assert_eq!(notes.len(), 6); // 2 stabs × 3 tones
        for n in &notes {
            let b = n.onset.beats();
            assert!(
                (b - 1.5).abs() < 1e-9 || (b - 3.5).abs() < 1e-9,
                "stab off the expected offbeats: {b}"
            );
            assert_eq!(n.duration, Duration::eighth());
        }
    }

    #[test]
    fn every_pattern_only_uses_the_voiced_pitches() {
        // The safety property: a pattern may re-time chord tones but can
        // NEVER introduce a pitch the voice leading didn't produce.
        let allowed: std::collections::BTreeSet<u8> = triad().iter().map(|p| p.midi()).collect();
        for pattern in [
            Accompaniment::Block,
            Accompaniment::Arpeggio,
            Accompaniment::Alberti,
            Accompaniment::OomPah,
            Accompaniment::Comp,
        ] {
            for meter in [3.0, 4.0] {
                for n in realize(pattern, meter) {
                    assert!(
                        allowed.contains(&n.pitch.midi()),
                        "{pattern:?} produced a non-chord pitch {}",
                        n.pitch.midi()
                    );
                }
            }
        }
    }

    #[test]
    fn every_pattern_stays_within_the_bar() {
        for pattern in [
            Accompaniment::Block,
            Accompaniment::Arpeggio,
            Accompaniment::Alberti,
            Accompaniment::OomPah,
            Accompaniment::Comp,
        ] {
            for meter in [3.0, 4.0] {
                for n in realize(pattern, meter) {
                    assert!((n.onset + n.duration).beats() <= meter + 1e-9);
                }
            }
        }
    }

    #[test]
    fn habanera_is_a_rhythm_cell_with_accent_identity() {
        let notes = realize(Accompaniment::Habanera, 4.0);
        // Onsets: the anchor, the pickup, and the two answering beats.
        let mut onsets: Vec<f64> = notes.iter().map(|n| n.onset.beats()).collect();
        onsets.sort_by(f64::total_cmp);
        onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert_eq!(onsets, vec![0.0, 1.5, 2.0, 3.0]);
        // The anchor is a SINGLE tone — the lowest — not a chord.
        let anchors: Vec<_> = notes.iter().filter(|n| n.onset.beats() == 0.0).collect();
        assert_eq!(anchors.len(), 1);
        assert_eq!(
            anchors[0].pitch.midi(),
            notes.iter().map(|n| n.pitch.midi()).min().unwrap()
        );
        // Accent hierarchy IS the cell: anchor strongest, pickup weakest.
        let vel_at = |t: f64| {
            notes
                .iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-9)
                .unwrap()
                .velocity
        };
        assert!(vel_at(0.0) > vel_at(2.0));
        assert!(vel_at(2.0) > vel_at(3.0));
        assert!(
            vel_at(3.0) > vel_at(1.5),
            "the pickup must be the lightest touch"
        );
    }

    #[test]
    fn habanera_truncates_gracefully_in_shorter_meters() {
        let notes = realize(Accompaniment::Habanera, 3.0);
        assert!(!notes.is_empty());
        for n in &notes {
            assert!((n.onset.beats()) < 3.0, "no event may start beyond the bar");
        }
    }

    #[test]
    fn five_gait_spells_three_plus_two_in_accents() {
        let notes = realize(Accompaniment::FiveGait, 5.0);
        let mut onsets: Vec<f64> = notes.iter().map(|n| n.onset.beats()).collect();
        onsets.sort_by(f64::total_cmp);
        onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert_eq!(onsets, vec![0.0, 1.5, 2.0, 3.0, 4.0]);
        let vel_at = |t: f64| {
            notes
                .iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-9)
                .unwrap()
                .velocity
        };
        // Two anchors, three-group's strongest, two-group's second.
        assert!(vel_at(0.0) > vel_at(3.0));
        assert!(vel_at(3.0) > vel_at(2.0));
        assert!(vel_at(2.0) > vel_at(1.5), "the pickup is the lightest");
        // The first anchor is the lowest tone alone.
        let anchors: Vec<_> = notes.iter().filter(|n| n.onset.beats() == 0.0).collect();
        assert_eq!(anchors.len(), 1);
    }

    #[test]
    fn jig_gait_spells_three_plus_three_in_accents() {
        let notes = realize(Accompaniment::JigGait, 6.0);
        let mut onsets: Vec<f64> = notes.iter().map(|n| n.onset.beats()).collect();
        onsets.sort_by(f64::total_cmp);
        onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert_eq!(onsets, vec![0.0, 1.5, 2.0, 3.0, 4.5, 5.0]);
        let vel_at = |t: f64| {
            notes
                .iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-9)
                .unwrap()
                .velocity
        };
        // Two dotted anchors, the first group's stronger than the second's
        // — mirrors FiveGait's anchor hierarchy, extended to 3+3.
        assert!(vel_at(0.0) > vel_at(3.0));
        assert!(vel_at(0.0) > vel_at(2.0));
        assert!(vel_at(3.0) > vel_at(4.5));
        assert!(
            vel_at(2.0) > vel_at(1.5),
            "the first pickup is the lightest"
        );
        assert!(
            vel_at(5.0) > vel_at(4.5),
            "the second pickup is the lightest in its own group"
        );
        // Both anchors are the lowest tone ALONE, not a chord — the
        // habanera/five-gait gravity, mirrored twice.
        for t in [0.0, 3.0] {
            let anchors: Vec<_> = notes
                .iter()
                .filter(|n| (n.onset.beats() - t).abs() < 1e-9)
                .collect();
            assert_eq!(anchors.len(), 1, "anchor at {t} must be a single tone");
        }
    }

    #[test]
    fn shuffle_alternates_root_and_fifth_with_a_strong_weak_accent() {
        let notes = realize(Accompaniment::Shuffle, 4.0);
        assert_eq!(notes.len(), 8); // one voice per eighth, not a full chord
        let mut sorted = notes.clone();
        sorted.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let root = 60; // C4, the lowest voiced tone
        let fifth = 67; // G4, the highest voiced tone
        for (i, n) in sorted.iter().enumerate() {
            let expected = if i % 2 == 0 { root } else { fifth };
            assert_eq!(
                n.pitch.midi(),
                expected,
                "eighth {i} must alternate root/fifth, never the third"
            );
        }
        // Beat-1 and beat-3 onsets are the strongest; every offbeat is the
        // weakest — the accent shape a shuffle needs before swing timing
        // is even applied.
        let vel_at = |t: f64| {
            sorted
                .iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-9)
                .unwrap()
                .velocity
        };
        assert!(vel_at(0.0) > vel_at(1.0));
        assert!(vel_at(1.0) > vel_at(0.5));
        assert!(vel_at(2.0) > vel_at(3.0));
        assert!(vel_at(3.0) > vel_at(2.5));
    }

    fn realize_at_bar(pattern: Accompaniment, meter: f64, bar_index: i64) -> Vec<ScoreNote> {
        let mut score = Score::new(
            crate::harmony::Key::major(PitchClass::C),
            100.0,
            meter as u8,
        );
        let bar_onset = Duration::new(meter as i64, 1).scale(bar_index, 1);
        pattern.realize_measure(&mut score, &triad(), bar_onset, meter, 0.5, 1.0);
        score.notes
    }

    fn onsets_within_bar(notes: &[ScoreNote], bar_index: i64, meter: f64) -> Vec<f64> {
        let bar_start = bar_index as f64 * meter;
        let mut out: Vec<f64> = notes.iter().map(|n| n.onset.beats() - bar_start).collect();
        out.sort_by(f64::total_cmp);
        out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        out
    }

    #[test]
    fn montuno_alternates_tresillo_and_two_side_across_two_bars() {
        // Bar 0: the three-side (tresillo: 0, 1.5, 3.0). Bar 1: the
        // two-side (two backbeat-adjacent stabs: 1.0, 2.0). This is the
        // first pattern in the whole engine whose identity only appears
        // ACROSS two bars — every prior cell repeats identically forever.
        let three_side = realize_at_bar(Accompaniment::Montuno, 4.0, 0);
        assert_eq!(onsets_within_bar(&three_side, 0, 4.0), vec![0.0, 1.5, 3.0]);
        let two_side = realize_at_bar(Accompaniment::Montuno, 4.0, 1);
        assert_eq!(onsets_within_bar(&two_side, 1, 4.0), vec![1.0, 2.0]);
        // Both sides voice the full chord (a chordal comping pattern),
        // unlike the single-tone anchor cells above.
        assert_eq!(three_side.len(), 9); // 3 onsets × 3 tones
        assert_eq!(two_side.len(), 6); // 2 onsets × 3 tones
    }

    #[test]
    fn montuno_cycle_never_resets_and_keeps_absolute_bar_parity() {
        // Real clave never restarts at a phrase boundary — the cycle is
        // defined by ABSOLUTE bar index, not a local counter. Bar 4 (even)
        // must be the three-side again; bar 5 (odd) the two-side, exactly
        // like bars 0 and 1.
        let bar4 = realize_at_bar(Accompaniment::Montuno, 4.0, 4);
        assert_eq!(onsets_within_bar(&bar4, 4, 4.0), vec![0.0, 1.5, 3.0]);
        let bar5 = realize_at_bar(Accompaniment::Montuno, 4.0, 5);
        assert_eq!(onsets_within_bar(&bar5, 5, 4.0), vec![1.0, 2.0]);
    }

    #[test]
    fn compas_gait_hits_only_the_five_counted_beats_of_twelve() {
        let notes = realize_at_bar(Accompaniment::CompasGait, 12.0, 0);
        assert_eq!(
            onsets_within_bar(&notes, 0, 12.0),
            vec![2.0, 5.0, 7.0, 9.0, 11.0],
            "the other 7 of 12 beats must stay silent"
        );
        assert_eq!(notes.len(), 15); // 5 onsets × 3 tones, full chord stabs
    }

    #[test]
    fn compas_gait_accents_the_cycle_close_strongest_then_its_midpoint() {
        let notes = realize_at_bar(Accompaniment::CompasGait, 12.0, 0);
        let vel_at = |t: f64| {
            notes
                .iter()
                .find(|n| (n.onset.beats() - t).abs() < 1e-9)
                .unwrap()
                .velocity
        };
        assert!(
            vel_at(11.0) > vel_at(5.0),
            "the '12' must be the strongest hit in the cycle"
        );
        assert!(
            vel_at(5.0) > vel_at(2.0),
            "the '6' midpoint outranks the lighter counts"
        );
    }

    #[test]
    fn compas_gait_truncates_gracefully_in_shorter_meters() {
        let notes = realize_at_bar(Accompaniment::CompasGait, 6.0, 0);
        assert!(!notes.is_empty());
        for n in &notes {
            assert!(n.onset.beats() < 6.0, "no event may start beyond the bar");
        }
    }

    #[test]
    fn bossa_comp_tiles_the_bar_with_zero_silence_and_zero_overlap() {
        let notes = realize_at_bar(Accompaniment::BossaComp, 4.0, 0);
        let mut spans: Vec<(f64, f64)> = notes
            .iter()
            .map(|n| (n.onset.beats(), (n.onset + n.duration).beats()))
            .collect();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        spans.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
        assert_eq!(spans, vec![(0.0, 1.5), (1.5, 3.0), (3.0, 4.0)]);
        for pair in spans.windows(2) {
            assert!(
                (pair[0].1 - pair[1].0).abs() < 1e-9,
                "gap or overlap between {:?} and {:?} — bossa must chain \
                 exactly, no silence and no collision",
                pair[0],
                pair[1]
            );
        }
        // Understated: every stab softer than a Block chord at the same
        // input velocity (accent multipliers are all < 1.0).
        assert!(notes.iter().all(|n| n.velocity < 0.5));
    }

    #[test]
    fn bossa_comp_truncates_gracefully_in_shorter_meters() {
        let notes = realize_at_bar(Accompaniment::BossaComp, 3.0, 0);
        assert!(!notes.is_empty());
        for n in &notes {
            assert!(n.onset.beats() < 3.0, "no event may start beyond the bar");
        }
    }
}
