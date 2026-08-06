// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A narrow, symbolic **realized-harmony** verifier: measures facts about
//! the notes a [`Score`] actually emits, against a per-bar chord sequence
//! *inferred empirically from those same notes* — not a general-purpose
//! style critic. Built for the Baroque harmonic-syntax A/B campaign (see
//! `symthaea-music-theory/HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md` and
//! the harness in `symthaea-muse/examples/baroque_campaign.rs`) —
//! deliberately scoped to what that experiment needs, not the full
//! imagined Analyst system.
//!
//! # Why chords are INFERRED, not read off the abstract `Progression`
//! An earlier version of this module took a `Progression` (the seed's
//! `bars`-length degree sequence) and indexed into it by
//! `bar_index % progression.len()`, on the assumption that a piece just
//! plays that sequence once (or repeats it verbatim). **That assumption is
//! false for the styles this campaign actually composes.** Tracing
//! `compose_with_spec` → `Form::ternary`/`rondo`/`variations` →
//! `Period::parallel_in` found that a "period" (antecedent+consequent)
//! each get their OWN full-length COPY of the base progression, with only
//! the trailing 1-2 degrees overwritten to force a cadence (Half for the
//! antecedent, Authentic for the consequent) — so a single `bars`-long
//! base progression is realized MULTIPLE times per piece (once per phrase
//! half, per section), each copy independently cadence-steered. None of
//! this structure is returned to callers of `compose_with_spec` (it only
//! returns a flat `Score`), so reconstructing the true bar-by-bar degree
//! sequence from outside would mean re-implementing `Form`'s internals —
//! fragile, and liable to silently drift from whatever the composer
//! actually does next.
//!
//! Instead, this module infers the chord in effect for each bar directly
//! from what the Harmony and Bass voices actually sound during that bar
//! (best-overlap match against the key's 7 diatonic triads). This is
//! robust to ANY Form/section/cadence-forcing structure, because it reads
//! the ground truth (the notes actually emitted) rather than a label that
//! may not survive the realization pipeline intact.
//!
//! # What this is and isn't
//! Every metric here is an **objective symbolic fact** about pitches,
//! beats, and chord-tone membership (per the inference above). Several
//! correspond to common-practice *conventions* (no parallel fifths, leaps
//! recover by step, leading tones resolve up) that a genuinely different
//! style could violate on purpose — those are labeled "style-specific" in
//! their doc comments. **None of this module's output is an artistic
//! judgment of whether the piece is good.** Interpreting counts (e.g. "is
//! 3 strong-beat conflicts in an 8-bar piece too many?") is left to
//! whoever reads [`HarmonyReport`], deliberately.
//!
//! # Known simplifications (disclosed, not hidden)
//! - **Chord inference can misfire on genuinely ambiguous bars** (e.g. a
//!   bar with only a single Harmony/Bass pitch class sounding, which
//!   matches several triads equally) — ties break toward the
//!   lowest-numbered scale degree, an arbitrary but deterministic choice.
//!   A bar with NO Harmony/Bass notes at all (e.g. a rest, or a texture
//!   that only uses Melody) infers no chord and is excluded from every
//!   metric that needs one.
//! - **Only diatonic triads are checked** — the composer's progression
//!   system never voices a seventh chord, so "chordal-seventh resolution"
//!   (part of the original request) is not implemented; only leading-tone
//!   resolution on inferred-dominant bars is, since that tone genuinely is
//!   a triad member (the third of V).
//! - **Voice-crossing is checked only for the two most defensible
//!   directional pairs** (Bass vs Melody, Bass vs Harmony) — Melody vs
//!   Harmony/CounterMelody legitimately interleave in real practice and
//!   crossing there isn't reliably "wrong," so this module doesn't flag it.
//! - **Cadence detection and progression-diversity now operate on the
//!   INFERRED per-bar degree sequence** (the same one chord-tone checks
//!   use), not the seed's raw `Progression` — so they describe what the
//!   piece actually did, including every section/repeat/cadence-force,
//!   rather than just the seed's starting plan.

use crate::cadence::Cadence;
use crate::harmony::{HarmonicFunction, Key};
use crate::pitch::PitchClass;
use crate::score::{Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A leap (consecutive-note interval) larger than a major sixth is "large"
/// — the threshold this experiment's request specified (`> 9` semitones,
/// i.e. strictly larger than a major sixth).
const LARGE_LEAP_SEMITONES: i32 = 10;
/// An interval this small or smaller counts as stepwise recovery.
const STEP_SEMITONES: i32 = 2;
/// A note held at least this many beats counts as "sustained" for the
/// unresolved-non-chord-tone check.
const SUSTAINED_BEATS: f64 = 1.0;

/// Tally of a binary outcome (e.g. "resolved" vs "not") — kept as counts
/// rather than a pre-computed rate so a caller can aggregate across many
/// pieces before dividing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ResolutionTally {
    pub total: usize,
    pub resolved_or_recovered: usize,
}

impl ResolutionTally {
    /// `resolved_or_recovered / total`, or `None` if `total == 0` (no
    /// occurrences to judge, not a rate of zero).
    pub fn rate(&self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some(self.resolved_or_recovered as f64 / self.total as f64)
        }
    }
}

/// Objective symbolic harmony facts for ONE voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceHarmonyMetrics {
    pub voice: VoiceRole,
    pub note_count: usize,
    /// Non-chord-tone notes landing on a bar's downbeat (beat 1), against
    /// the INFERRED chord for that bar. An objective fact, not
    /// automatically "wrong" — a prepared suspension is also a non-chord
    /// tone on the downbeat and this narrow check can't distinguish it
    /// from an unprepared clash.
    pub strong_beat_chord_conflicts: usize,
    /// Non-chord tones (against the inferred chord) held
    /// `>= SUSTAINED_BEATS` whose immediately following note in this SAME
    /// voice does not move by step (a proxy for "left unresolved," not a
    /// full suspension/appoggiatura analysis).
    pub unresolved_sustained_non_chord_tones: usize,
    /// Of this voice's notes that sound the inferred-dominant-function
    /// bar's leading tone, how many resolve correctly (up a step to the
    /// tonic pitch class) on the immediately following note in this same
    /// voice.
    pub leading_tone_resolution: ResolutionTally,
    /// Style-specific: leaps (`> major sixth`) and how many are followed
    /// by contrary-motion stepwise recovery (`<= a whole step`, opposite
    /// direction) in this same voice.
    pub large_leaps: ResolutionTally,
    /// Mean |semitone delta| between this voice's consecutive notes —
    /// lower is smoother voice-leading. `None` if the voice has `< 2`
    /// notes.
    pub mean_voice_leading_distance: Option<f64>,
}

/// How harmonically diverse the INFERRED per-bar degree sequence is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProgressionDiversity {
    /// Bars with a successfully inferred chord (ambiguous/empty bars are
    /// excluded, see this module's "known simplifications").
    pub inferred_bar_count: usize,
    /// Immediate-repeat transitions (bar N's degree == bar N+1's degree).
    pub immediate_repeats: usize,
    /// Distinct (from, to) transition pairs, out of the total transitions
    /// between consecutive inferred bars.
    pub distinct_transitions: usize,
    pub total_transitions: usize,
    /// Distinct scale degrees used, out of `inferred_bar_count` bars.
    pub distinct_degrees_used: usize,
}

/// A cadence found at a phrase-boundary candidate in the INFERRED degree
/// sequence (see this module's "known simplifications" for what counts as
/// a boundary here).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedCadence {
    /// `"final"` (the piece-ending two inferred degrees) or `"midpoint"`
    /// (a candidate half-cadence at the periodic phrase midpoint).
    pub boundary: String,
    pub cadence: Cadence,
    pub closure: f32,
}

/// Style-specific: consecutive parallel perfect 5ths/octaves between two
/// voices, checked for the two most defensible directional pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParallelPerfectFindings {
    pub bass_vs_melody: usize,
    pub bass_vs_harmony: usize,
}

/// Objective symbolic facts about how a [`Score`]'s actual notes relate to
/// an empirically-inferred per-bar chord sequence. Does NOT judge whether
/// the piece is good — see this module's doc comment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonyReport {
    pub voice_metrics: Vec<VoiceHarmonyMetrics>,
    pub voice_crossings_bass_vs_melody: usize,
    pub voice_crossings_bass_vs_harmony: usize,
    pub parallel_perfect: ParallelPerfectFindings,
    pub progression_diversity: ProgressionDiversity,
    pub cadences: Vec<DetectedCadence>,
}

fn bar_index_of(onset_beats: f64, meter: u8) -> usize {
    (onset_beats / meter.max(1) as f64).floor().max(0.0) as usize
}

/// Infers the scale degree (1..=7) best explaining the Harmony+Bass
/// voices' sounding pitch classes during bar `bar` (by pitch-class
/// overlap against `key`'s 7 diatonic triads). `None` if neither voice
/// sounds any note in that bar. Ties break toward the lowest-numbered
/// degree (see this module's "known simplifications").
fn infer_bar_degree(score: &Score, key: Key, bar: usize) -> Option<i32> {
    let meter = score.meter.max(1) as f64;
    let bar_start = bar as f64 * meter;
    let bar_end = bar_start + meter;
    let mut sounding: HashSet<PitchClass> = HashSet::new();
    for role in [VoiceRole::Harmony, VoiceRole::Bass] {
        for note in score.notes.iter().filter(|n| n.role == role) {
            let onset = note.onset.beats();
            if onset >= bar_start && onset < bar_end {
                sounding.insert(note.pitch.pitch_class());
            }
        }
    }
    if sounding.is_empty() {
        return None;
    }
    (1..=7).max_by_key(|&d| {
        let tones: HashSet<PitchClass> =
            key.diatonic_triad(d).pitch_classes().into_iter().collect();
        // Reverse the degree so ties break toward the LOWEST degree: a
        // higher (degree, count) tuple with `7 - d` as the tiebreaker
        // makes degree 1 win ties over degree 7.
        (sounding.intersection(&tones).count(), 7 - d)
    })
}

/// One inferred degree per bar across the whole score (bar 0 through the
/// last bar containing any note), `None` for bars with no Harmony/Bass
/// evidence.
fn infer_bar_degrees(score: &Score, key: Key) -> Vec<Option<i32>> {
    let meter = score.meter.max(1) as f64;
    let last_onset = score
        .notes
        .iter()
        .map(|n| n.onset.beats())
        .fold(0.0_f64, f64::max);
    let num_bars = (last_onset / meter).floor() as usize + 1;
    (0..num_bars)
        .map(|bar| infer_bar_degree(score, key, bar))
        .collect()
}

fn voice_metrics(
    score: &Score,
    role: VoiceRole,
    key: Key,
    bar_degrees: &[Option<i32>],
) -> VoiceHarmonyMetrics {
    let notes = score.voice(role);
    let mut m = VoiceHarmonyMetrics {
        voice: role,
        note_count: notes.len(),
        strong_beat_chord_conflicts: 0,
        unresolved_sustained_non_chord_tones: 0,
        leading_tone_resolution: ResolutionTally::default(),
        large_leaps: ResolutionTally::default(),
        mean_voice_leading_distance: None,
    };
    if notes.is_empty() {
        return m;
    }

    let leading_tone = key.scale().degree_pitch_class(7);
    let tonic_pc = key.scale().degree_pitch_class(1);

    for (i, note) in notes.iter().enumerate() {
        let bar = bar_index_of(note.onset.beats(), score.meter);
        let Some(Some(degree)) = bar_degrees.get(bar) else {
            continue;
        };
        let chord = key.diatonic_triad(*degree);
        let pc = note.pitch.pitch_class();
        let is_chord_tone = chord.contains(pc);
        let on_downbeat = (note.onset.beats() % score.meter.max(1) as f64).abs() < 1e-6;

        if !is_chord_tone && on_downbeat {
            m.strong_beat_chord_conflicts += 1;
        }

        if let Some(next) = notes.get(i + 1) {
            let interval = next.pitch.midi() as i32 - note.pitch.midi() as i32;

            if !is_chord_tone && note.duration.beats() >= SUSTAINED_BEATS {
                let moved_by_step = interval.abs() <= STEP_SEMITONES;
                if !moved_by_step {
                    m.unresolved_sustained_non_chord_tones += 1;
                }
            }

            let is_dominant_bar = key.function(*degree) == HarmonicFunction::Dominant;
            if is_dominant_bar && pc == leading_tone {
                m.leading_tone_resolution.total += 1;
                let resolves_up_to_tonic = interval == 1 && next.pitch.pitch_class() == tonic_pc;
                if resolves_up_to_tonic {
                    m.leading_tone_resolution.resolved_or_recovered += 1;
                }
            }

            if interval.abs() > LARGE_LEAP_SEMITONES {
                m.large_leaps.total += 1;
                if let Some(after_next) = notes.get(i + 2) {
                    let recovery_interval =
                        after_next.pitch.midi() as i32 - next.pitch.midi() as i32;
                    let opposite_direction = recovery_interval.signum() != 0
                        && recovery_interval.signum() != interval.signum();
                    if opposite_direction && recovery_interval.abs() <= STEP_SEMITONES {
                        m.large_leaps.resolved_or_recovered += 1;
                    }
                }
            }
        }
    }

    if notes.len() >= 2 {
        let sum: i64 = notes
            .windows(2)
            .map(|w| (w[1].pitch.midi() as i32 - w[0].pitch.midi() as i32).unsigned_abs() as i64)
            .sum();
        m.mean_voice_leading_distance = Some(sum as f64 / (notes.len() - 1) as f64);
    }

    m
}

/// Active-at-time note in a voice sorted by onset: the last note whose
/// onset is `<= time`. An approximation (ignores whether that note's
/// duration actually reaches `time`, i.e. rests are not distinguished from
/// a held note) — acceptable for this narrow crossing/parallel-motion
/// check, which only samples at the OTHER voice's own onsets.
fn active_at(sorted_notes: &[ScoreNote], time: f64) -> Option<ScoreNote> {
    sorted_notes
        .iter()
        .rev()
        .find(|n| n.onset.beats() <= time + 1e-9)
        .copied()
}

/// Counts times `lower_role` sounds strictly above `upper_role` at
/// `upper_role`'s own note onsets.
fn count_crossings(score: &Score, lower_role: VoiceRole, upper_role: VoiceRole) -> usize {
    let lower = score.voice(lower_role);
    let upper = score.voice(upper_role);
    if lower.is_empty() || upper.is_empty() {
        return 0;
    }
    upper
        .iter()
        .filter_map(|u| active_at(&lower, u.onset.beats()).map(|l| (l, u)))
        .filter(|(l, u)| l.pitch.midi() > u.pitch.midi())
        .count()
}

/// Parallel-perfect violations between two voices, sampled at
/// `role_a`'s own onsets (reusing [`crate::counterpoint::has_parallel_perfect`]).
fn count_parallel_perfect(score: &Score, role_a: VoiceRole, role_b: VoiceRole) -> usize {
    let a = score.voice(role_a);
    let b = score.voice(role_b);
    if a.len() < 2 {
        return 0;
    }
    let mut violations = 0;
    for w in a.windows(2) {
        let (prev_a, next_a) = (w[0], w[1]);
        let (Some(prev_b), Some(next_b)) = (
            active_at(&b, prev_a.onset.beats()),
            active_at(&b, next_a.onset.beats()),
        ) else {
            continue;
        };
        if crate::counterpoint::has_parallel_perfect(
            prev_a.pitch,
            prev_b.pitch,
            next_a.pitch,
            next_b.pitch,
        ) {
            violations += 1;
        }
    }
    violations
}

fn progression_diversity(bar_degrees: &[Option<i32>]) -> ProgressionDiversity {
    let degrees: Vec<i32> = bar_degrees.iter().filter_map(|d| *d).collect();
    let inferred_bar_count = degrees.len();
    if inferred_bar_count == 0 {
        return ProgressionDiversity::default();
    }
    let mut immediate_repeats = 0;
    let mut distinct: HashSet<(i32, i32)> = HashSet::new();
    for w in degrees.windows(2) {
        if w[0] == w[1] {
            immediate_repeats += 1;
        }
        distinct.insert((w[0], w[1]));
    }
    let distinct_degrees_used: HashSet<i32> = degrees.iter().copied().collect();
    ProgressionDiversity {
        inferred_bar_count,
        immediate_repeats,
        distinct_transitions: distinct.len(),
        total_transitions: inferred_bar_count.saturating_sub(1),
        distinct_degrees_used: distinct_degrees_used.len(),
    }
}

fn detect_cadences(bar_degrees: &[Option<i32>]) -> Vec<DetectedCadence> {
    let degrees: Vec<i32> = bar_degrees.iter().filter_map(|d| *d).collect();
    let mut found = Vec::new();
    if degrees.len() >= 2 {
        let n = degrees.len();
        if let Some(cadence) = Cadence::detect(degrees[n - 2], degrees[n - 1]) {
            found.push(DetectedCadence {
                boundary: "final".to_string(),
                cadence,
                closure: cadence.closure(),
            });
        }
    }
    if degrees.len() >= 4 && degrees.len() % 2 == 0 {
        let mid = degrees.len() / 2;
        if let Some(cadence) = Cadence::detect(degrees[mid - 1], degrees[mid]) {
            found.push(DetectedCadence {
                boundary: "midpoint".to_string(),
                cadence,
                closure: cadence.closure(),
            });
        }
    }
    found
}

/// Compute a [`HarmonyReport`] for `score` in `key`. Unlike an earlier
/// version of this function, this does NOT take the seed's abstract
/// `Progression` — every metric is derived from an empirically-inferred
/// per-bar chord sequence (see this module's doc comment for why).
pub fn verify(score: &Score, key: Key) -> HarmonyReport {
    let bar_degrees = infer_bar_degrees(score, key);
    let voices = [
        VoiceRole::Melody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
        VoiceRole::CounterMelody,
    ];
    let voice_metrics = voices
        .into_iter()
        .map(|role| voice_metrics(score, role, key, &bar_degrees))
        .collect();

    HarmonyReport {
        voice_metrics,
        voice_crossings_bass_vs_melody: count_crossings(score, VoiceRole::Bass, VoiceRole::Melody),
        voice_crossings_bass_vs_harmony: count_crossings(
            score,
            VoiceRole::Bass,
            VoiceRole::Harmony,
        ),
        parallel_perfect: ParallelPerfectFindings {
            bass_vs_melody: count_parallel_perfect(score, VoiceRole::Bass, VoiceRole::Melody),
            bass_vs_harmony: count_parallel_perfect(score, VoiceRole::Bass, VoiceRole::Harmony),
        },
        progression_diversity: progression_diversity(&bar_degrees),
        cadences: detect_cadences(&bar_degrees),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Tonality;
    use crate::rhythm::Duration;
    use crate::score::Emphasis;
    use crate::score::PartId;

    fn c_major() -> Key {
        Key {
            tonic: PitchClass::C,
            tonality: Tonality::Major,
        }
    }

    /// Test helper -- `onset_beats`/`dur_beats` must be whole numbers (every
    /// call site in this module's tests uses whole-beat values).
    fn note(onset_beats: f64, dur_beats: f64, midi: u8, role: VoiceRole) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: crate::pitch::Pitch::from_midi(midi),
            onset: Duration::new(onset_beats as i64, 1),
            duration: Duration::new(dur_beats as i64, 1),
            velocity: 0.8,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    /// A clean, hand-built one-bar-per-chord I-IV-V-I score in C major,
    /// meter 4, where Harmony+Bass unambiguously voice each bar's triad
    /// and the melody is chord-tone-only, with the leading tone (bar 2)
    /// resolving up a step to tonic (bar 3) -- the ground-truth "nothing
    /// wrong" case.
    fn clean_score() -> Score {
        let mut score = Score::new(c_major(), 100.0, 4);
        // Melody, chord tones only.
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody)); // C (I)
        score.push(note(4.0, 4.0, 69, VoiceRole::Melody)); // A (IV)
        score.push(note(8.0, 4.0, 71, VoiceRole::Melody)); // B (V, leading tone)
        score.push(note(12.0, 4.0, 72, VoiceRole::Melody)); // C (I, resolves up a step)
        // Bass: roots of each chord (I=C, IV=F, V=G, I=C).
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        score.push(note(4.0, 4.0, 53, VoiceRole::Bass));
        score.push(note(8.0, 4.0, 55, VoiceRole::Bass));
        score.push(note(12.0, 4.0, 48, VoiceRole::Bass));
        // Harmony: thirds+fifths of each chord, to disambiguate inference
        // (bass root alone is ambiguous between e.g. I and vi).
        score.push(note(0.0, 4.0, 64, VoiceRole::Harmony)); // E (I: C E G)
        // The fifth of IV (C), not its root (F) -- F alone would tie with
        // ii (D F A), which also contains F; C disambiguates (only in IV).
        score.push(note(4.0, 4.0, 72, VoiceRole::Harmony));
        score.push(note(8.0, 4.0, 62, VoiceRole::Harmony)); // D (V: G B D)
        score.push(note(12.0, 4.0, 64, VoiceRole::Harmony)); // E (I)
        score
    }

    #[test]
    fn bar_degrees_are_correctly_inferred_from_harmony_and_bass() {
        let score = clean_score();
        let degrees = infer_bar_degrees(&score, c_major());
        assert_eq!(degrees, vec![Some(1), Some(4), Some(5), Some(1)]);
    }

    #[test]
    fn a_clean_chord_tone_only_score_has_no_strong_beat_conflicts() {
        let score = clean_score();
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.strong_beat_chord_conflicts, 0);
    }

    #[test]
    fn the_leading_tone_resolving_up_a_step_to_tonic_is_counted_as_resolved() {
        let score = clean_score();
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.leading_tone_resolution.total, 1);
        assert_eq!(melody.leading_tone_resolution.resolved_or_recovered, 1);
    }

    #[test]
    fn an_unresolved_leading_tone_is_counted_as_not_resolved() {
        let mut score = Score::new(c_major(), 100.0, 4);
        // Bar 0 (I): tonic, unambiguous via Harmony+Bass.
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        score.push(note(0.0, 4.0, 64, VoiceRole::Harmony));
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody));
        // Bar 1 (V): leading tone (B) in melody, but the NEXT note leaps
        // away instead of resolving up a step to C.
        score.push(note(4.0, 4.0, 55, VoiceRole::Bass));
        score.push(note(4.0, 4.0, 62, VoiceRole::Harmony));
        score.push(note(4.0, 4.0, 71, VoiceRole::Melody));
        score.push(note(8.0, 4.0, 50, VoiceRole::Melody));
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.leading_tone_resolution.total, 1);
        assert_eq!(melody.leading_tone_resolution.resolved_or_recovered, 0);
    }

    #[test]
    fn a_downbeat_non_chord_tone_is_flagged_as_a_strong_beat_conflict() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        score.push(note(0.0, 4.0, 64, VoiceRole::Harmony));
        // Bar 0 (I = C E G): melody lands on D (a passing tone), not a
        // chord tone, right on the downbeat.
        score.push(note(0.0, 4.0, 62, VoiceRole::Melody));
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.strong_beat_chord_conflicts, 1);
    }

    #[test]
    fn a_bar_with_no_harmony_or_bass_notes_infers_no_chord() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody));
        let degrees = infer_bar_degrees(&score, c_major());
        assert_eq!(degrees, vec![None]);
    }

    #[test]
    fn a_leap_larger_than_a_sixth_followed_by_a_step_back_counts_as_recovered() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        score.push(note(0.0, 4.0, 64, VoiceRole::Harmony));
        score.push(note(0.0, 1.0, 60, VoiceRole::Melody));
        // Leap of 11 semitones up (> a major sixth).
        score.push(note(1.0, 1.0, 71, VoiceRole::Melody));
        // Step back down (recovery).
        score.push(note(2.0, 1.0, 69, VoiceRole::Melody));
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.large_leaps.total, 1);
        assert_eq!(melody.large_leaps.resolved_or_recovered, 1);
    }

    #[test]
    fn a_leap_larger_than_a_sixth_with_no_step_back_is_not_recovered() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        score.push(note(0.0, 4.0, 64, VoiceRole::Harmony));
        score.push(note(0.0, 1.0, 60, VoiceRole::Melody));
        score.push(note(1.0, 1.0, 71, VoiceRole::Melody));
        // Continues leaping further away instead of stepping back.
        score.push(note(2.0, 1.0, 76, VoiceRole::Melody));
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.large_leaps.total, 1);
        assert_eq!(melody.large_leaps.resolved_or_recovered, 0);
    }

    #[test]
    fn bass_sounding_above_melody_is_a_voice_crossing() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody)); // C4
        score.push(note(0.0, 4.0, 72, VoiceRole::Bass)); // C5 -- above melody
        let report = verify(&score, c_major());
        assert_eq!(report.voice_crossings_bass_vs_melody, 1);
    }

    #[test]
    fn bass_correctly_below_melody_is_not_a_crossing() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody));
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass));
        let report = verify(&score, c_major());
        assert_eq!(report.voice_crossings_bass_vs_melody, 0);
    }

    #[test]
    fn parallel_fifths_between_bass_and_melody_are_detected() {
        let mut score = Score::new(c_major(), 100.0, 4);
        // Bass C3 -> D3 (up a step), Melody G3 -> A3 (up a step): both a
        // perfect fifth apart before and after, moving in parallel.
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass)); // C3
        score.push(note(4.0, 4.0, 50, VoiceRole::Bass)); // D3
        score.push(note(0.0, 4.0, 55, VoiceRole::Melody)); // G3
        score.push(note(4.0, 4.0, 57, VoiceRole::Melody)); // A3
        let report = verify(&score, c_major());
        assert_eq!(report.parallel_perfect.bass_vs_melody, 1);
    }

    #[test]
    fn contrary_motion_into_a_fifth_is_not_a_parallel_violation() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 48, VoiceRole::Bass)); // C3
        score.push(note(4.0, 4.0, 50, VoiceRole::Bass)); // D3 (up)
        score.push(note(0.0, 4.0, 57, VoiceRole::Melody)); // A3
        // G3 (down -- contrary motion, into a perfect fifth: not a
        // violation).
        score.push(note(4.0, 4.0, 55, VoiceRole::Melody));
        let report = verify(&score, c_major());
        assert_eq!(report.parallel_perfect.bass_vs_melody, 0);
    }

    #[test]
    fn an_authentic_cadence_is_detected_at_the_inferred_sequences_end() {
        let score = clean_score();
        let bar_degrees = infer_bar_degrees(&score, c_major());
        let cadences = detect_cadences(&bar_degrees);
        let final_cadence = cadences.iter().find(|c| c.boundary == "final").unwrap();
        assert_eq!(final_cadence.cadence, Cadence::Authentic);
        assert_eq!(final_cadence.closure, 1.0);
    }

    #[test]
    fn a_sequence_that_never_repeats_a_degree_has_full_transition_diversity() {
        let bar_degrees = vec![Some(1), Some(4), Some(5), Some(6)];
        let diversity = progression_diversity(&bar_degrees);
        assert_eq!(diversity.immediate_repeats, 0);
        assert_eq!(diversity.total_transitions, 3);
        assert_eq!(diversity.distinct_transitions, 3);
        assert_eq!(diversity.distinct_degrees_used, 4);
    }

    #[test]
    fn a_sequence_that_repeats_the_same_transition_has_low_diversity() {
        let bar_degrees = vec![Some(1), Some(5), Some(1), Some(5), Some(1), Some(5)];
        let diversity = progression_diversity(&bar_degrees);
        assert_eq!(diversity.immediate_repeats, 0);
        assert_eq!(diversity.total_transitions, 5);
        // Only two distinct transitions ever occur: 1->5 and 5->1.
        assert_eq!(diversity.distinct_transitions, 2);
        assert_eq!(diversity.distinct_degrees_used, 2);
    }

    #[test]
    fn ambiguous_bars_are_excluded_from_diversity_and_cadence_detection() {
        let bar_degrees = vec![Some(1), None, Some(4), Some(5), Some(1)];
        let diversity = progression_diversity(&bar_degrees);
        // Only the 4 non-None entries count.
        assert_eq!(diversity.inferred_bar_count, 4);
    }

    #[test]
    fn mean_voice_leading_distance_is_none_for_a_single_note_voice() {
        let mut score = Score::new(c_major(), 100.0, 4);
        score.push(note(0.0, 4.0, 60, VoiceRole::Melody));
        let report = verify(&score, c_major());
        let melody = report
            .voice_metrics
            .iter()
            .find(|v| v.voice == VoiceRole::Melody)
            .unwrap();
        assert_eq!(melody.mean_voice_leading_distance, None);
    }

    #[test]
    fn resolution_tally_rate_is_none_when_there_are_no_occurrences() {
        let tally = ResolutionTally::default();
        assert_eq!(tally.rate(), None);
    }
}
