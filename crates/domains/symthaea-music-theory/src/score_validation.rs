// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, inspectable validation for completed symbolic scores.
//!
//! Generators may use soft costs internally, but product and experiment paths
//! need one independent report describing which invariants were actually
//! satisfied by the completed score. The report never repairs a score and does
//! not compress failures into an opaque quality number.

use crate::counterpoint::{has_parallel_perfect, is_consonant};
use crate::score::{Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

pub const THEORY_VALIDATION_VERSION: &str = "theory-validation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Warning,
    Fatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreValidationRule {
    ScoreMetadata,
    NoteBounds,
    VoiceMonophony,
    VoiceCrossing,
    StrongBeatConsonance,
    ParallelPerfectMotion,
    MelodicLeap,
    FinalTonicArrival,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreValidationIssue {
    pub rule: ScoreValidationRule,
    pub severity: ValidationSeverity,
    pub note_indices: Vec<usize>,
    pub start_beat: Option<f64>,
    pub end_beat: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreValidationConfig {
    pub min_midi: u8,
    pub max_midi: u8,
    pub max_melodic_leap_semitones: u8,
    pub require_final_tonic: bool,
    pub check_strong_beat_consonance: bool,
    pub check_parallel_perfect_motion: bool,
}

impl Default for ScoreValidationConfig {
    fn default() -> Self {
        Self {
            min_midi: 24,
            max_midi: 108,
            max_melodic_leap_semitones: 12,
            require_final_tonic: true,
            check_strong_beat_consonance: true,
            check_parallel_perfect_motion: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TheoryValidationReport {
    pub validation_version: String,
    pub valid: bool,
    pub issues: Vec<ScoreValidationIssue>,
}

impl TheoryValidationReport {
    pub fn fatal_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == ValidationSeverity::Fatal)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == ValidationSeverity::Warning)
            .count()
    }
}

pub fn validate_score(score: &Score, config: &ScoreValidationConfig) -> TheoryValidationReport {
    let mut issues = Vec::new();
    validate_metadata(score, &mut issues);
    validate_notes(score, config, &mut issues);
    validate_voice_monophony(score, &mut issues);
    validate_voice_crossing(score, &mut issues);
    validate_melodic_leaps(score, config, &mut issues);
    if config.check_strong_beat_consonance {
        validate_strong_beats(score, &mut issues);
    }
    if config.check_parallel_perfect_motion {
        validate_parallel_motion(score, &mut issues);
    }
    if config.require_final_tonic {
        validate_final_tonic(score, &mut issues);
    }

    let valid = !issues
        .iter()
        .any(|issue| issue.severity == ValidationSeverity::Fatal);
    TheoryValidationReport {
        validation_version: THEORY_VALIDATION_VERSION.into(),
        valid,
        issues,
    }
}

/// Rules that represent a genuinely MALFORMED score, regardless of style or
/// grammar family — collisions, out-of-range notes, missing metadata.
/// Deliberately excludes `StrongBeatConsonance`/`ParallelPerfectMotion`/
/// `VoiceCrossing`/`FinalTonicArrival`/`MelodicLeap`: those are real
/// music-theoretic judgments, but they're style-dependent (a raga arc, a
/// blues turnaround, or a groove cycle has no obligation to satisfy
/// classical-tonal voice-leading rules it was never designed against), so
/// gating production on them would either mask genuine stylistic idioms as
/// "fatal" or force every grammar-family engine to a tonal common
/// denominator it doesn't share.
pub fn is_universal_invariant(rule: ScoreValidationRule) -> bool {
    matches!(
        rule,
        ScoreValidationRule::ScoreMetadata
            | ScoreValidationRule::NoteBounds
            | ScoreValidationRule::VoiceMonophony
    )
}

/// Debug-only production gate: panics if `score` has any Fatal issue among
/// the universal-invariant rules (see [`is_universal_invariant`]), under the
/// default validation config. Compiles to nothing in release builds — a
/// development/test-time safety net against a score that collides or
/// malforms itself, not a claim that the score is musically "correct" (the
/// excluded voice-leading rules can and do still fail; see that function's
/// doc for why they're deliberately not gated here).
pub fn debug_assert_no_structural_defects(score: &Score, label: &str) {
    if !cfg!(debug_assertions) {
        return;
    }
    let report = validate_score(score, &ScoreValidationConfig::default());
    let defects: Vec<&ScoreValidationIssue> = report
        .issues
        .iter()
        .filter(|i| i.severity == ValidationSeverity::Fatal && is_universal_invariant(i.rule))
        .collect();
    assert!(
        defects.is_empty(),
        "{label}: composed score has {} structural defect(s) (collision/malformation \
         within a single voice — not a voice-leading judgment call):\n{:#?}",
        defects.len(),
        defects
    );
}

fn validate_metadata(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        issue(
            issues,
            ScoreValidationRule::ScoreMetadata,
            ValidationSeverity::Fatal,
            Vec::new(),
            None,
            None,
            "tempo must be finite and positive",
        );
    }
    if score.meter == 0 {
        issue(
            issues,
            ScoreValidationRule::ScoreMetadata,
            ValidationSeverity::Fatal,
            Vec::new(),
            None,
            None,
            "meter must contain at least one beat",
        );
    }
    if !score.total_beats.beats().is_finite() || score.total_beats.beats() <= 0.0 {
        issue(
            issues,
            ScoreValidationRule::ScoreMetadata,
            ValidationSeverity::Fatal,
            Vec::new(),
            None,
            None,
            "score duration must be finite and positive",
        );
    }
    if score.notes.is_empty() {
        issue(
            issues,
            ScoreValidationRule::ScoreMetadata,
            ValidationSeverity::Fatal,
            Vec::new(),
            None,
            None,
            "score contains no notes",
        );
    }
}

fn validate_notes(
    score: &Score,
    config: &ScoreValidationConfig,
    issues: &mut Vec<ScoreValidationIssue>,
) {
    for (index, note) in score.notes.iter().enumerate() {
        let onset = note.onset.beats();
        let duration = note.duration.beats();
        let end = onset + duration;
        let finite = onset.is_finite()
            && duration.is_finite()
            && note.velocity.is_finite()
            && note.section_intensity.is_finite();
        let bounded = onset >= 0.0
            && duration > 0.0
            && end <= score.total_beats.beats() + 1e-9
            && (0.0..=1.0).contains(&note.velocity)
            && note.section_intensity >= 0.0
            && (config.min_midi..=config.max_midi).contains(&note.pitch.midi());
        if !finite || !bounded {
            issue(
                issues,
                ScoreValidationRule::NoteBounds,
                ValidationSeverity::Fatal,
                vec![index],
                Some(onset),
                Some(end),
                format!("note {index} has invalid timing, dynamics, intensity, or pitch range"),
            );
        }
    }
}

fn validate_voice_monophony(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    for role in [VoiceRole::Melody, VoiceRole::Bass, VoiceRole::CounterMelody] {
        let mut notes: Vec<(usize, &ScoreNote)> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, note)| note.role == role)
            .collect();
        notes.sort_by(|left, right| left.1.onset.beats().total_cmp(&right.1.onset.beats()));
        for pair in notes.windows(2) {
            let previous_end = pair[0].1.onset.beats() + pair[0].1.duration.beats();
            if previous_end > pair[1].1.onset.beats() + 1e-9 {
                issue(
                    issues,
                    ScoreValidationRule::VoiceMonophony,
                    ValidationSeverity::Fatal,
                    vec![pair[0].0, pair[1].0],
                    Some(pair[1].1.onset.beats()),
                    Some(previous_end),
                    format!("{} voice overlaps itself", role_name(role)),
                );
            }
        }
    }
}

fn validate_voice_crossing(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    let mut times: Vec<f64> = score.notes.iter().map(|note| note.onset.beats()).collect();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
    for time in times {
        let Some((bass_index, bass)) = sounding(score, VoiceRole::Bass, time) else {
            continue;
        };
        for role in [
            VoiceRole::Harmony,
            VoiceRole::CounterMelody,
            VoiceRole::Melody,
        ] {
            for (upper_index, upper) in sounding_all(score, role, time) {
                if bass.pitch.midi() > upper.pitch.midi() {
                    issue(
                        issues,
                        ScoreValidationRule::VoiceCrossing,
                        ValidationSeverity::Fatal,
                        vec![bass_index, upper_index],
                        Some(time),
                        None,
                        format!("bass crosses above the {} voice", role_name(role)),
                    );
                }
            }
        }
    }
}

fn validate_melodic_leaps(
    score: &Score,
    config: &ScoreValidationConfig,
    issues: &mut Vec<ScoreValidationIssue>,
) {
    for role in [VoiceRole::Melody, VoiceRole::CounterMelody] {
        let mut notes: Vec<(usize, &ScoreNote)> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, note)| note.role == role)
            .collect();
        notes.sort_by(|left, right| left.1.onset.beats().total_cmp(&right.1.onset.beats()));
        for pair in notes.windows(2) {
            let leap = pair[0].1.pitch.semitones_to(pair[1].1.pitch).unsigned_abs() as u8;
            if leap > config.max_melodic_leap_semitones {
                issue(
                    issues,
                    ScoreValidationRule::MelodicLeap,
                    ValidationSeverity::Warning,
                    vec![pair[0].0, pair[1].0],
                    Some(pair[0].1.onset.beats()),
                    Some(pair[1].1.onset.beats()),
                    format!("{} voice leaps {leap} semitones", role_name(role)),
                );
            }
        }
    }
}

fn validate_strong_beats(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    let last = score.total_beats.beats().floor().max(0.0) as usize;
    // Only METRICALLY strong beats — the downbeat and the mid-measure accent
    // (beats 1 and 3 of 4/4), per the crate's canonical
    // [`crate::phrase::is_strong_beat`].
    //
    // This used to iterate EVERY integer beat, so beats 2 and 4 of 4/4 counted
    // as strong and an ordinary passing or neighbour tone there was reported
    // Fatal. Measured before the fix: 7,336 StrongBeatConsonance issues across
    // 29 styles x 12 seeds, with every single style implicated — including
    // RenaissancePolyphony, whose entire identity is species counterpoint.
    // A rule that flags everything measures nothing, which is why that count
    // could not be used as evidence about musical quality.
    //
    // Sharing `phrase::is_strong_beat` rather than reimplementing it is
    // deliberate: a second, silently-divergable definition of "strong beat" is
    // the same defect class as the duplicate progression source removed from
    // `Style::progression` on 2026-07-30.
    let meter = f64::from(score.meter.max(1));
    for beat in 0..=last {
        let time = beat as f64;
        if !crate::phrase::is_strong_beat(time, meter) {
            continue;
        }
        let Some((bass_index, bass)) = sounding(score, VoiceRole::Bass, time) else {
            continue;
        };
        for role in [VoiceRole::Melody, VoiceRole::CounterMelody] {
            if let Some((upper_index, upper)) = sounding(score, role, time) {
                let (lower, higher) = if bass.pitch <= upper.pitch {
                    (bass.pitch, upper.pitch)
                } else {
                    (upper.pitch, bass.pitch)
                };
                if !is_consonant(lower, higher) {
                    issue(
                        issues,
                        ScoreValidationRule::StrongBeatConsonance,
                        ValidationSeverity::Fatal,
                        vec![bass_index, upper_index],
                        Some(time),
                        Some(time + 1.0),
                        format!(
                            "{} and bass are dissonant on strong beat {beat}",
                            role_name(role)
                        ),
                    );
                }
            }
        }
    }
}

fn validate_parallel_motion(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    let last = score.total_beats.beats().floor().max(0.0) as usize;
    for beat in 0..last {
        let before = beat as f64;
        let after = (beat + 1) as f64;
        for role in [VoiceRole::Melody, VoiceRole::CounterMelody] {
            let Some((bass_before_index, bass_before)) = sounding(score, VoiceRole::Bass, before)
            else {
                continue;
            };
            let Some((upper_before_index, upper_before)) = sounding(score, role, before) else {
                continue;
            };
            let Some((bass_after_index, bass_after)) = sounding(score, VoiceRole::Bass, after)
            else {
                continue;
            };
            let Some((upper_after_index, upper_after)) = sounding(score, role, after) else {
                continue;
            };
            if has_parallel_perfect(
                bass_before.pitch,
                upper_before.pitch,
                bass_after.pitch,
                upper_after.pitch,
            ) {
                issue(
                    issues,
                    ScoreValidationRule::ParallelPerfectMotion,
                    ValidationSeverity::Fatal,
                    vec![
                        bass_before_index,
                        upper_before_index,
                        bass_after_index,
                        upper_after_index,
                    ],
                    Some(before),
                    Some(after),
                    format!(
                        "{} and bass move in parallel perfect intervals",
                        role_name(role)
                    ),
                );
            }
        }
    }
}

fn validate_final_tonic(score: &Score, issues: &mut Vec<ScoreValidationIssue>) {
    let final_note = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| note.role == VoiceRole::Melody)
        .max_by(|left, right| {
            let left_end = left.1.onset.beats() + left.1.duration.beats();
            let right_end = right.1.onset.beats() + right.1.duration.beats();
            left_end.total_cmp(&right_end)
        });
    match final_note {
        Some((index, note)) if note.pitch.pitch_class() == score.key.tonic => {}
        Some((index, note)) => issue(
            issues,
            ScoreValidationRule::FinalTonicArrival,
            ValidationSeverity::Fatal,
            vec![index],
            Some(note.onset.beats()),
            Some(note.onset.beats() + note.duration.beats()),
            "final melody note does not arrive on the tonic",
        ),
        None => issue(
            issues,
            ScoreValidationRule::FinalTonicArrival,
            ValidationSeverity::Fatal,
            Vec::new(),
            None,
            None,
            "score has no melody note to verify at the ending",
        ),
    }
}

/// The pitch of `role` sounding at `time`, if any.
///
/// `pub(crate)` so the composer can choose the bass against the melody that is
/// ALREADY in the score (`realize_melody` runs before `realize_bass`) — see
/// [`crate::voicing::lead_bass_against_melody`]. Shares this module's
/// `sounding` rather than reimplementing the overlap test, for the same reason
/// `validate_strong_beats` shares `phrase::is_strong_beat`.
pub(crate) fn sounding_pitch(
    score: &Score,
    role: VoiceRole,
    time: f64,
) -> Option<crate::pitch::Pitch> {
    sounding(score, role, time).map(|(_, note)| note.pitch)
}

fn sounding(score: &Score, role: VoiceRole, time: f64) -> Option<(usize, &ScoreNote)> {
    sounding_all(score, role, time)
        .into_iter()
        .max_by(|left, right| left.1.onset.beats().total_cmp(&right.1.onset.beats()))
}

fn sounding_all(score: &Score, role: VoiceRole, time: f64) -> Vec<(usize, &ScoreNote)> {
    score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            note.role == role
                && note.onset.beats() <= time + 1e-9
                && time < note.onset.beats() + note.duration.beats() - 1e-9
        })
        .collect()
}

fn role_name(role: VoiceRole) -> &'static str {
    match role {
        VoiceRole::Melody => "melody",
        VoiceRole::Harmony => "harmony",
        VoiceRole::Bass => "bass",
        VoiceRole::CounterMelody => "countermelody",
    }
}

#[allow(clippy::too_many_arguments)]
fn issue(
    issues: &mut Vec<ScoreValidationIssue>,
    rule: ScoreValidationRule,
    severity: ValidationSeverity,
    note_indices: Vec<usize>,
    start_beat: Option<f64>,
    end_beat: Option<f64>,
    message: impl Into<String>,
) {
    issues.push(ScoreValidationIssue {
        rule,
        severity,
        note_indices,
        start_beat,
        end_beat,
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::PartId;
    use crate::{Duration, Emphasis, Key, Pitch, PitchClass, ScoreNote};

    fn note(midi: u8, onset: i64, duration: i64, role: VoiceRole) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration: Duration::new(duration, 1),
            velocity: 0.7,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn valid_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(48, 0, 1, VoiceRole::Bass));
        score.push(note(60, 0, 1, VoiceRole::Melody));
        score.push(note(50, 1, 1, VoiceRole::Bass));
        score.push(note(65, 1, 1, VoiceRole::Melody));
        score.push(note(55, 2, 1, VoiceRole::Bass));
        score.push(note(64, 2, 1, VoiceRole::Melody));
        score.push(note(48, 3, 1, VoiceRole::Bass));
        score.push(note(60, 3, 1, VoiceRole::Melody));
        score
    }

    #[test]
    fn canonical_report_accepts_a_bounded_consonant_score() {
        let report = validate_score(&valid_score(), &ScoreValidationConfig::default());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.fatal_count(), 0);
    }

    #[test]
    fn overlap_and_missing_tonic_are_independently_reported() {
        let mut score = valid_score();
        score.notes[2].role = VoiceRole::Melody;
        score.notes[2].onset = Duration::new(0, 1);
        score.notes[2].duration = Duration::new(2, 1);
        score.notes.last_mut().unwrap().pitch = Pitch::from_midi(62);
        let report = validate_score(&score, &ScoreValidationConfig::default());
        assert!(!report.valid);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| { issue.rule == ScoreValidationRule::VoiceMonophony })
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| { issue.rule == ScoreValidationRule::FinalTonicArrival })
        );
    }

    #[test]
    fn universal_invariant_excludes_style_dependent_voice_leading_rules() {
        assert!(is_universal_invariant(ScoreValidationRule::ScoreMetadata));
        assert!(is_universal_invariant(ScoreValidationRule::NoteBounds));
        assert!(is_universal_invariant(ScoreValidationRule::VoiceMonophony));
        assert!(!is_universal_invariant(ScoreValidationRule::VoiceCrossing));
        assert!(!is_universal_invariant(
            ScoreValidationRule::StrongBeatConsonance
        ));
        assert!(!is_universal_invariant(
            ScoreValidationRule::ParallelPerfectMotion
        ));
        assert!(!is_universal_invariant(ScoreValidationRule::MelodicLeap));
        assert!(!is_universal_invariant(
            ScoreValidationRule::FinalTonicArrival
        ));
    }

    #[test]
    fn debug_gate_panics_on_a_same_voice_overlap_but_not_on_a_missing_tonic() {
        // A genuine same-voice collision (VoiceMonophony, a universal
        // invariant) must panic -- this is exactly the bug class fixed in
        // composer.rs's apply_counter_hook_echoes/CounterShift.
        let mut colliding = valid_score();
        colliding.notes[2].role = VoiceRole::Melody;
        colliding.notes[2].onset = Duration::new(0, 1);
        colliding.notes[2].duration = Duration::new(2, 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            debug_assert_no_structural_defects(&colliding, "test");
        }));
        assert!(result.is_err(), "expected a panic on a colliding score");

        // A score whose ONLY fatal issue is a non-universal, style-dependent
        // rule (missing final-tonic arrival) must NOT panic -- see
        // `is_universal_invariant`'s doc for why.
        let mut no_tonic = valid_score();
        no_tonic.notes.last_mut().unwrap().pitch = Pitch::from_midi(62);
        let report = validate_score(&no_tonic, &ScoreValidationConfig::default());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.rule == ScoreValidationRule::FinalTonicArrival
                    && i.severity == ValidationSeverity::Fatal),
            "test setup must actually produce a FinalTonicArrival fatal: {:?}",
            report.issues
        );
        let result2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            debug_assert_no_structural_defects(&no_tonic, "test");
        }));
        assert!(
            result2.is_ok(),
            "must not panic on a non-universal-rule fatal issue"
        );
    }
}
