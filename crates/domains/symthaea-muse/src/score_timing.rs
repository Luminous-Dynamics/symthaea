// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact score-level timing conformance for declared musical event windows.
//!
//! V1 operates on a symbolic Score with one constant tempo. Beat windows are
//! evaluated directly; second windows are converted through that tempo. It
//! verifies onsets and note overlap, not rendered-audio localization or live
//! MIDI/audio I/O latency. Those require separate calibrated evaluators.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;
use symthaea_music_theory::{PartId, Score, VoiceRole};

pub const SCORE_TIMING_CONTRACT_SCHEMA_VERSION_V1: u32 = 1;
pub const SCORE_TIMING_VERIFIER_VERSION_V1: &str = "score-timing-verifier-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "basis")]
pub enum ScoreTimeWindowV1 {
    /// Beat positions measured from the start of the Score.
    Beats { start: f64, end: f64 },
    /// Elapsed seconds from the start of a constant-tempo Score.
    Seconds { start: f64, end: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "selector")]
pub enum ScoreEventSelectorV1 {
    VoiceRole { role: VoiceRole },
    Part { part: PartId },
    PartAndRole { part: PartId, role: VoiceRole },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "check")]
pub enum ScoreTimingCheckV1 {
    /// Count note onsets in the half-open window [start, end).
    OnsetCount {
        selector: ScoreEventSelectorV1,
        window: ScoreTimeWindowV1,
        min_count: u32,
        max_count: Option<u32>,
    },
    /// Require no selected note to overlap the half-open window [start, end).
    /// A note that begins before the window and sustains into it is a violation.
    Silence {
        selector: ScoreEventSelectorV1,
        window: ScoreTimeWindowV1,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreTimingRequirementV1 {
    pub requirement_id: String,
    pub check: ScoreTimingCheckV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreTimingContractV1 {
    pub schema_version: u32,
    pub contract_id: String,
    pub requirements: Vec<ScoreTimingRequirementV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreTimingIssueV1 {
    UnsupportedSchemaVersion { found: u32 },
    EmptyContractId,
    EmptyRequirements,
    EmptyRequirementId,
    DuplicateRequirementId { requirement_id: String },
    InvalidWindow { requirement_id: String },
    InvalidEventCountRange { requirement_id: String },
    UnassignedPartSelector { requirement_id: String },
    InvalidScoreTempo,
    InvalidScoreDuration,
    InvalidScoreEventTiming { note_index: usize },
    WindowExceedsScore { requirement_id: String },
    SerializationFailure { reason: String },
}

impl ScoreTimingContractV1 {
    pub fn validate(&self) -> Vec<ScoreTimingIssueV1> {
        let mut issues = Vec::new();
        if self.schema_version != SCORE_TIMING_CONTRACT_SCHEMA_VERSION_V1 {
            issues.push(ScoreTimingIssueV1::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self.contract_id.trim().is_empty() {
            issues.push(ScoreTimingIssueV1::EmptyContractId);
        }
        if self.requirements.is_empty() {
            issues.push(ScoreTimingIssueV1::EmptyRequirements);
        }

        let mut ids = BTreeSet::new();
        for requirement in &self.requirements {
            if requirement.requirement_id.trim().is_empty() {
                issues.push(ScoreTimingIssueV1::EmptyRequirementId);
            }
            if !ids.insert(requirement.requirement_id.clone()) {
                issues.push(ScoreTimingIssueV1::DuplicateRequirementId {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }

            let (selector, window, invalid_range) = match &requirement.check {
                ScoreTimingCheckV1::OnsetCount {
                    selector,
                    window,
                    min_count,
                    max_count,
                } => (
                    selector,
                    window,
                    max_count.is_some_and(|maximum| maximum < *min_count),
                ),
                ScoreTimingCheckV1::Silence { selector, window } => (selector, window, false),
            };
            if invalid_range {
                issues.push(ScoreTimingIssueV1::InvalidEventCountRange {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
            let (start, end) = window.bounds();
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
                issues.push(ScoreTimingIssueV1::InvalidWindow {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
            if selector.has_unassigned_part() {
                issues.push(ScoreTimingIssueV1::UnassignedPartSelector {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
        }
        issues
    }
}

impl ScoreTimeWindowV1 {
    fn bounds(self) -> (f64, f64) {
        match self {
            Self::Beats { start, end } | Self::Seconds { start, end } => (start, end),
        }
    }

    fn as_beats(self, tempo_bpm: f64) -> (f64, f64) {
        match self {
            Self::Beats { start, end } => (start, end),
            Self::Seconds { start, end } => {
                let beats_per_second = tempo_bpm / 60.0;
                (start * beats_per_second, end * beats_per_second)
            }
        }
    }
}

impl ScoreEventSelectorV1 {
    fn has_unassigned_part(self) -> bool {
        match self {
            Self::VoiceRole { .. } => false,
            Self::Part { part } | Self::PartAndRole { part, .. } => !part.is_assigned(),
        }
    }

    fn matches(self, note: &symthaea_music_theory::ScoreNote) -> bool {
        match self {
            Self::VoiceRole { role } => note.role == role,
            Self::Part { part } => note.part == part,
            Self::PartAndRole { part, role } => note.part == part && note.role == role,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreTimingVerdictV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreTimingRequirementResultV1 {
    pub requirement_id: String,
    pub window_start_beats: f64,
    pub window_end_beats: f64,
    pub matching_note_indices: Vec<usize>,
    pub observed_event_count: u32,
    pub expected_min_count: u32,
    pub expected_max_count: Option<u32>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreTimingReportV1 {
    pub schema_version: u32,
    pub verifier_version: String,
    pub contract_id: String,
    pub contract_sha256: String,
    pub score_sha256: String,
    pub score_tempo_bpm: f32,
    pub score_duration_beats: f64,
    pub score_duration_seconds: f64,
    pub verdict: ScoreTimingVerdictV1,
    pub passed_requirements: usize,
    pub failed_requirements: usize,
    pub results: Vec<ScoreTimingRequirementResultV1>,
}

/// Evaluate a timing contract against the exact symbolic Score supplied.
///
/// Seconds are measured from score start using the Score's single constant
/// tempo. V1 explicitly does not claim to support a piecewise tempo map,
/// performed rubato, MIDI output latency or rendered-audio detector accuracy.
pub fn verify_score_timing_v1(
    score: &Score,
    contract: &ScoreTimingContractV1,
) -> Result<ScoreTimingReportV1, Vec<ScoreTimingIssueV1>> {
    let mut issues = contract.validate();
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        issues.push(ScoreTimingIssueV1::InvalidScoreTempo);
    }
    let duration_beats = score.total_beats.beats();
    if !duration_beats.is_finite() || duration_beats <= 0.0 {
        issues.push(ScoreTimingIssueV1::InvalidScoreDuration);
    }
    for (note_index, note) in score.notes.iter().enumerate() {
        let onset = note.onset.beats();
        let duration = note.duration.beats();
        if !onset.is_finite()
            || !duration.is_finite()
            || onset < 0.0
            || duration <= 0.0
            || onset + duration > duration_beats + 1e-9
        {
            issues.push(ScoreTimingIssueV1::InvalidScoreEventTiming { note_index });
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let tempo = score.tempo_bpm as f64;
    let duration_seconds = score.seconds();
    let mut results = Vec::with_capacity(contract.requirements.len());

    for requirement in &contract.requirements {
        let (selector, window, expected_min, expected_max, is_silence) =
            match &requirement.check {
                ScoreTimingCheckV1::OnsetCount {
                    selector,
                    window,
                    min_count,
                    max_count,
                } => (*selector, *window, *min_count, *max_count, false),
                ScoreTimingCheckV1::Silence { selector, window } => {
                    (*selector, *window, 0, Some(0), true)
                }
            };
        let (start_beats, end_beats) = window.as_beats(tempo);
        if end_beats > duration_beats + 1e-9 {
            issues.push(ScoreTimingIssueV1::WindowExceedsScore {
                requirement_id: requirement.requirement_id.clone(),
            });
            continue;
        }

        let matching_note_indices: Vec<usize> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, note)| {
                if !selector.matches(note) {
                    return false;
                }
                let onset = note.onset.beats();
                if is_silence {
                    let note_end = onset + note.duration.beats();
                    onset < end_beats && note_end > start_beats
                } else {
                    onset >= start_beats && onset < end_beats
                }
            })
            .map(|(index, _)| index)
            .collect();
        let observed = matching_note_indices.len().min(u32::MAX as usize) as u32;
        let passed = observed >= expected_min
            && expected_max.is_none_or(|maximum| observed <= maximum);

        results.push(ScoreTimingRequirementResultV1 {
            requirement_id: requirement.requirement_id.clone(),
            window_start_beats: start_beats,
            window_end_beats: end_beats,
            matching_note_indices,
            observed_event_count: observed,
            expected_min_count: expected_min,
            expected_max_count: expected_max,
            passed,
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let passed_requirements = results.iter().filter(|result| result.passed).count();
    let failed_requirements = results.len() - passed_requirements;
    let verdict = if failed_requirements == 0 {
        ScoreTimingVerdictV1::Pass
    } else {
        ScoreTimingVerdictV1::Fail
    };
    let contract_sha256 = digest_json_v1(contract)
        .map_err(|reason| vec![ScoreTimingIssueV1::SerializationFailure { reason }])?;
    let score_sha256 = digest_json_v1(score)
        .map_err(|reason| vec![ScoreTimingIssueV1::SerializationFailure { reason }])?;

    Ok(ScoreTimingReportV1 {
        schema_version: SCORE_TIMING_CONTRACT_SCHEMA_VERSION_V1,
        verifier_version: SCORE_TIMING_VERIFIER_VERSION_V1.to_owned(),
        contract_id: contract.contract_id.clone(),
        contract_sha256,
        score_sha256,
        score_tempo_bpm: score.tempo_bpm,
        score_duration_beats: duration_beats,
        score_duration_seconds: duration_seconds,
        verdict,
        passed_requirements,
        failed_requirements,
        results,
    })
}

fn digest_json_v1<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{Duration, Emphasis, Key, Pitch, PitchClass, ScoreNote};

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (onset, role, midi, part) in [
            (0, VoiceRole::Melody, 60, PartId(0)),
            (1, VoiceRole::Melody, 64, PartId(0)),
            (2, VoiceRole::Bass, 36, PartId(1)),
        ] {
            score.push(ScoreNote {
                part,
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::quarter(),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn contract(check: ScoreTimingCheckV1) -> ScoreTimingContractV1 {
        ScoreTimingContractV1 {
            schema_version: SCORE_TIMING_CONTRACT_SCHEMA_VERSION_V1,
            contract_id: "timing-fixture-v1".into(),
            requirements: vec![ScoreTimingRequirementV1 {
                requirement_id: "event-window-1".into(),
                check,
            }],
        }
    }

    #[test]
    fn beat_window_is_half_open_and_returns_exact_note_indices() {
        let report = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::OnsetCount {
                selector: ScoreEventSelectorV1::VoiceRole {
                    role: VoiceRole::Melody,
                },
                window: ScoreTimeWindowV1::Beats { start: 1.0, end: 2.0 },
                min_count: 1,
                max_count: Some(1),
            }),
        )
        .unwrap();
        assert_eq!(report.verdict, ScoreTimingVerdictV1::Pass);
        assert_eq!(report.results[0].matching_note_indices, vec![1]);
        assert_eq!(report.results[0].observed_event_count, 1);
    }

    #[test]
    fn second_window_uses_the_pinned_constant_score_tempo() {
        let report = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::OnsetCount {
                selector: ScoreEventSelectorV1::VoiceRole {
                    role: VoiceRole::Melody,
                },
                window: ScoreTimeWindowV1::Seconds { start: 0.5, end: 1.0 },
                min_count: 1,
                max_count: Some(1),
            }),
        )
        .unwrap();
        assert_eq!(report.results[0].window_start_beats, 1.0);
        assert_eq!(report.results[0].window_end_beats, 2.0);
        assert_eq!(report.results[0].matching_note_indices, vec![1]);
    }

    #[test]
    fn silence_window_detects_a_note_that_sustains_into_it() {
        let report = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::Silence {
                selector: ScoreEventSelectorV1::VoiceRole {
                    role: VoiceRole::Melody,
                },
                window: ScoreTimeWindowV1::Beats { start: 0.5, end: 1.5 },
            }),
        )
        .unwrap();
        assert_eq!(report.verdict, ScoreTimingVerdictV1::Fail);
        assert_eq!(report.results[0].matching_note_indices, vec![0, 1]);
    }

    #[test]
    fn dense_decoy_onsets_exceed_the_declared_maximum() {
        let report = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::OnsetCount {
                selector: ScoreEventSelectorV1::VoiceRole {
                    role: VoiceRole::Melody,
                },
                window: ScoreTimeWindowV1::Beats { start: 0.0, end: 2.0 },
                min_count: 1,
                max_count: Some(1),
            }),
        )
        .unwrap();
        assert_eq!(report.verdict, ScoreTimingVerdictV1::Fail);
        assert_eq!(report.results[0].matching_note_indices, vec![0, 1]);
        assert_eq!(report.results[0].observed_event_count, 2);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn requested_event_window_beyond_score_is_rejected() {
        let result = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::OnsetCount {
                selector: ScoreEventSelectorV1::VoiceRole {
                    role: VoiceRole::Melody,
                },
                window: ScoreTimeWindowV1::Beats { start: 2.5, end: 3.5 },
                min_count: 1,
                max_count: Some(1),
            }),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.contains(&ScoreTimingIssueV1::WindowExceedsScore {
                requirement_id: "event-window-1".into()
            })
        ));
    }

    #[test]
    fn unassigned_part_identity_is_rejected() {
        let result = verify_score_timing_v1(
            &score(),
            &contract(ScoreTimingCheckV1::OnsetCount {
                selector: ScoreEventSelectorV1::Part { part: PartId::UNASSIGNED },
                window: ScoreTimeWindowV1::Beats { start: 0.0, end: 1.0 },
                min_count: 1,
                max_count: None,
            }),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.contains(&ScoreTimingIssueV1::UnassignedPartSelector {
                requirement_id: "event-window-1".into()
            })
        ));
    }

    #[test]
    fn report_binds_exact_score_contract_and_verifier_version() {
        let source_score = score();
        let source_contract = contract(ScoreTimingCheckV1::OnsetCount {
            selector: ScoreEventSelectorV1::VoiceRole {
                role: VoiceRole::Melody,
            },
            window: ScoreTimeWindowV1::Beats { start: 0.0, end: 1.0 },
            min_count: 1,
            max_count: Some(1),
        });
        let report = verify_score_timing_v1(&source_score, &source_contract).unwrap();
        assert_eq!(report.verifier_version, SCORE_TIMING_VERIFIER_VERSION_V1);
        assert_eq!(report.score_sha256, digest_json_v1(&source_score).unwrap());
        assert_eq!(report.contract_sha256, digest_json_v1(&source_contract).unwrap());
        assert_eq!(report.score_sha256.len(), 64);
        assert_eq!(report.contract_sha256.len(), 64);
    }

    #[test]
    fn duplicate_ids_and_reversed_windows_fail_closed() {
        let mut invalid = contract(ScoreTimingCheckV1::OnsetCount {
            selector: ScoreEventSelectorV1::VoiceRole {
                role: VoiceRole::Melody,
            },
            window: ScoreTimeWindowV1::Beats { start: 2.0, end: 1.0 },
            min_count: 1,
            max_count: Some(1),
        });
        invalid.requirements.push(invalid.requirements[0].clone());
        let result = verify_score_timing_v1(&score(), &invalid);
        assert!(matches!(
            result,
            Err(issues)
                if issues.contains(&ScoreTimingIssueV1::InvalidWindow {
                    requirement_id: "event-window-1".into()
                })
                && issues.contains(&ScoreTimingIssueV1::DuplicateRequirementId {
                    requirement_id: "event-window-1".into()
                })
        ));
    }
}
