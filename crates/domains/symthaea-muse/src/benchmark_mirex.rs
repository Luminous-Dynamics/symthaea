// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MIREX 2026 symbolic piano-continuation interchange.
//!
//! MIREX's wire representation is intentionally much smaller than Melothaea's
//! native score. This adapter never rounds native timing to the benchmark grid
//! and never pretends the wire format preserves key, tempo, part/role,
//! dynamics, or structural annotations. Loss is explicit and provenance-bound.

use crate::evidence_digest::benchmark_manifest::{
    BenchmarkSamplingPolicyV1, BenchmarkSeedPolicyV1, BenchmarkSelectionPolicyV1,
    BenchmarkSelectorV1,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{Duration, Score, ScoreNote};

pub const MIREX_2026_INTERCHANGE_VERSION: &str = "mirex-2026-symbolic-piano-continuation-v1";
pub const MIREX_2026_PROMPT_START_MIN: u16 = 0;
pub const MIREX_2026_PROMPT_START_MAX: u16 = 79;
pub const MIREX_2026_GENERATION_START_MIN: u16 = 80;
pub const MIREX_2026_GENERATION_START_MAX: u16 = 271;
pub const MIREX_2026_GENERATION_OFFSET_SIXTEENTHS: u16 = 80;
pub const MIREX_2026_SUBJECTIVE_SAMPLES_PER_PROMPT: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexNoteV2026 {
    /// Absolute timeline position in sixteenth-note units.
    pub start: u16,
    /// MIDI note number. Wire validation enforces the benchmark's 0..=127 range.
    pub pitch: u8,
    /// Positive duration in sixteenth-note units.
    pub duration: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexPromptV2026 {
    pub prompt: Vec<MirexNoteV2026>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexGenerationV2026 {
    pub generation: Vec<MirexNoteV2026>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirexWireFieldV2026 {
    Start,
    Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirexWireIssueV2026 {
    EmptyNoteSet,
    ZeroDuration { note_index: usize },
    PitchOutsideMidiRange { note_index: usize, pitch: u8 },
    StartOutsidePrompt { note_index: usize, start: u16 },
    StartOutsideGeneration { note_index: usize, start: u16 },
}

impl MirexPromptV2026 {
    pub fn validate(&self) -> Vec<MirexWireIssueV2026> {
        validate_note_set(&self.prompt, WireDomain::Prompt)
    }

    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

impl MirexGenerationV2026 {
    pub fn validate(&self) -> Vec<MirexWireIssueV2026> {
        validate_note_set(&self.generation, WireDomain::Generation)
    }

    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireDomain {
    Prompt,
    Generation,
}

fn validate_note_set(notes: &[MirexNoteV2026], domain: WireDomain) -> Vec<MirexWireIssueV2026> {
    let mut issues = Vec::new();
    if notes.is_empty() {
        issues.push(MirexWireIssueV2026::EmptyNoteSet);
    }
    for (note_index, note) in notes.iter().enumerate() {
        if note.duration == 0 {
            issues.push(MirexWireIssueV2026::ZeroDuration { note_index });
        }
        if note.pitch > 127 {
            issues.push(MirexWireIssueV2026::PitchOutsideMidiRange {
                note_index,
                pitch: note.pitch,
            });
        }
        match domain {
            WireDomain::Prompt
                if !(MIREX_2026_PROMPT_START_MIN..=MIREX_2026_PROMPT_START_MAX)
                    .contains(&note.start) =>
            {
                issues.push(MirexWireIssueV2026::StartOutsidePrompt {
                    note_index,
                    start: note.start,
                });
            }
            WireDomain::Generation
                if !(MIREX_2026_GENERATION_START_MIN..=MIREX_2026_GENERATION_START_MAX)
                    .contains(&note.start) =>
            {
                issues.push(MirexWireIssueV2026::StartOutsideGeneration {
                    note_index,
                    start: note.start,
                });
            }
            _ => {}
        }
    }
    issues
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirexScoreTimelineV2026 {
    /// The score occupies the complete MIREX timeline. Prompt-region notes are
    /// omitted while generation notes retain absolute positions.
    FullPieceGlobal,
    /// The score contains only the continuation starting at local beat zero.
    /// The fixed MIREX generation offset is added exactly.
    ContinuationLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MirexRepresentationLossV2026 {
    KeyContext,
    TempoContext,
    PartIdentity,
    VoiceRole,
    Velocity,
    StructuralEmphasis,
    SectionIntensity,
}

pub const MIREX_REQUIRED_REPRESENTATION_LOSSES_V2026: [MirexRepresentationLossV2026; 7] = [
    MirexRepresentationLossV2026::KeyContext,
    MirexRepresentationLossV2026::TempoContext,
    MirexRepresentationLossV2026::PartIdentity,
    MirexRepresentationLossV2026::VoiceRole,
    MirexRepresentationLossV2026::Velocity,
    MirexRepresentationLossV2026::StructuralEmphasis,
    MirexRepresentationLossV2026::SectionIntensity,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirexProjectionEvidenceV2026 {
    pub interchange_version: String,
    pub timeline: MirexScoreTimelineV2026,
    pub source_score_sha256: String,
    pub generation_sha256: String,
    pub source_note_count: usize,
    pub prompt_region_notes_omitted: usize,
    pub generation_notes_written: usize,
    pub representation_losses: Vec<MirexRepresentationLossV2026>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirexGenerationProjectionV2026 {
    pub generation: MirexGenerationV2026,
    pub evidence: MirexProjectionEvidenceV2026,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirexProjectionErrorV2026 {
    UnsupportedMeter { found: u8 },
    InvalidTempo,
    InvalidNoteScalar { note_index: usize, field: String },
    NegativeOnset { note_index: usize },
    NonPositiveDuration { note_index: usize },
    NonSixteenthAligned {
        note_index: usize,
        field: MirexWireFieldV2026,
        numerator: i64,
        denominator: i64,
    },
    TimeOverflow {
        note_index: usize,
        field: MirexWireFieldV2026,
    },
    StartOutsideGeneration { note_index: usize, start: u32 },
    EmptyGeneration,
    Serialization,
}

/// Project a native score into MIREX's note-list format without quantization,
/// clipping, inferred meter conversion, deduplication, or metadata synthesis.
pub fn project_score_to_mirex_generation(
    score: &Score,
    timeline: MirexScoreTimelineV2026,
) -> Result<MirexGenerationProjectionV2026, MirexProjectionErrorV2026> {
    if score.meter != 4 {
        return Err(MirexProjectionErrorV2026::UnsupportedMeter { found: score.meter });
    }
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        return Err(MirexProjectionErrorV2026::InvalidTempo);
    }

    let source_score_sha256 =
        canonical_json_sha256(score).map_err(|_| MirexProjectionErrorV2026::Serialization)?;
    let mut notes = Vec::new();
    let mut prompt_region_notes_omitted = 0usize;

    for (note_index, note) in score.notes.iter().enumerate() {
        validate_note_scalars(note_index, note)?;
        let onset = exact_sixteenths(note_index, MirexWireFieldV2026::Start, note.onset)?;
        let duration = exact_sixteenths(note_index, MirexWireFieldV2026::Duration, note.duration)?;
        if note.onset.num() < 0 {
            return Err(MirexProjectionErrorV2026::NegativeOnset { note_index });
        }
        if note.duration.num() <= 0 {
            return Err(MirexProjectionErrorV2026::NonPositiveDuration { note_index });
        }

        let global_start = match timeline {
            MirexScoreTimelineV2026::FullPieceGlobal => u32::from(onset),
            MirexScoreTimelineV2026::ContinuationLocal => {
                u32::from(onset) + u32::from(MIREX_2026_GENERATION_OFFSET_SIXTEENTHS)
            }
        };

        if timeline == MirexScoreTimelineV2026::FullPieceGlobal
            && global_start <= u32::from(MIREX_2026_PROMPT_START_MAX)
        {
            prompt_region_notes_omitted += 1;
            continue;
        }
        if !(u32::from(MIREX_2026_GENERATION_START_MIN)
            ..=u32::from(MIREX_2026_GENERATION_START_MAX))
            .contains(&global_start)
        {
            return Err(MirexProjectionErrorV2026::StartOutsideGeneration {
                note_index,
                start: global_start,
            });
        }

        notes.push(MirexNoteV2026 {
            start: u16::try_from(global_start).map_err(|_| MirexProjectionErrorV2026::TimeOverflow {
                note_index,
                field: MirexWireFieldV2026::Start,
            })?,
            pitch: note.pitch.midi(),
            duration,
        });
    }

    if notes.is_empty() {
        return Err(MirexProjectionErrorV2026::EmptyGeneration);
    }
    notes.sort();
    let generation = MirexGenerationV2026 { generation: notes };
    debug_assert!(generation.validate().is_empty());

    let generation_sha256 = generation
        .canonical_sha256()
        .map_err(|_| MirexProjectionErrorV2026::Serialization)?;
    let generation_notes_written = generation.generation.len();

    Ok(MirexGenerationProjectionV2026 {
        generation,
        evidence: MirexProjectionEvidenceV2026 {
            interchange_version: MIREX_2026_INTERCHANGE_VERSION.into(),
            timeline,
            source_score_sha256,
            generation_sha256,
            source_note_count: score.notes.len(),
            prompt_region_notes_omitted,
            generation_notes_written,
            representation_losses: MIREX_REQUIRED_REPRESENTATION_LOSSES_V2026.to_vec(),
        },
    })
}

fn validate_note_scalars(
    note_index: usize,
    note: &ScoreNote,
) -> Result<(), MirexProjectionErrorV2026> {
    for (field, value) in [
        ("velocity", note.velocity),
        ("section_intensity", note.section_intensity),
    ] {
        if !value.is_finite() {
            return Err(MirexProjectionErrorV2026::InvalidNoteScalar {
                note_index,
                field: field.into(),
            });
        }
    }
    Ok(())
}

fn exact_sixteenths(
    note_index: usize,
    field: MirexWireFieldV2026,
    value: Duration,
) -> Result<u16, MirexProjectionErrorV2026> {
    let scaled = value
        .num()
        .checked_mul(4)
        .ok_or(MirexProjectionErrorV2026::TimeOverflow { note_index, field })?;
    if scaled % value.den() != 0 {
        return Err(MirexProjectionErrorV2026::NonSixteenthAligned {
            note_index,
            field,
            numerator: value.num(),
            denominator: value.den(),
        });
    }
    let sixteenths = scaled / value.den();
    u16::try_from(sixteenths)
        .map_err(|_| MirexProjectionErrorV2026::TimeOverflow { note_index, field })
}

/// The 2026 subjective evaluation generates eight samples per prompt and then
/// permits submitter-side selection. Binding both policies makes that selection
/// bias explicit in the BENCH-000 manifest.
pub fn mirex_2026_subjective_sampling_profile() -> (
    BenchmarkSamplingPolicyV1,
    BenchmarkSelectionPolicyV1,
) {
    (
        BenchmarkSamplingPolicyV1 {
            samples_per_subject: MIREX_2026_SUBJECTIVE_SAMPLES_PER_PROMPT,
            seed_policy: BenchmarkSeedPolicyV1::ExternalProtocol,
        },
        BenchmarkSelectionPolicyV1::BestOfN {
            n: MIREX_2026_SUBJECTIVE_SAMPLES_PER_PROMPT,
            selector: BenchmarkSelectorV1::SystemSubmitterManual,
        },
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirexSampleFilenameErrorV2026 {
    ZeroSampleCount,
    IndexOutOfRange { index: u32, n_sample: u32 },
}

pub fn mirex_sample_filename(
    index: u32,
    n_sample: u32,
) -> Result<String, MirexSampleFilenameErrorV2026> {
    if n_sample == 0 {
        return Err(MirexSampleFilenameErrorV2026::ZeroSampleCount);
    }
    if index == 0 || index > n_sample {
        return Err(MirexSampleFilenameErrorV2026::IndexOutOfRange { index, n_sample });
    }
    Ok(format!("sample_{index:02}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{Emphasis, Key, PartId, Pitch, PitchClass, VoiceRole};

    fn note(onset: Duration, duration: Duration, pitch: u8) -> ScoreNote {
        ScoreNote {
            part: PartId(1),
            pitch: Pitch::from_midi(pitch),
            onset,
            duration,
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn continuation_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(Duration::zero(), Duration::quarter(), 60));
        score.push(note(Duration::quarter(), Duration::new(3, 2), 64));
        score
    }

    #[test]
    fn wire_ranges_and_midi_pitch_are_fail_closed() {
        let valid = MirexGenerationV2026 {
            generation: vec![MirexNoteV2026 {
                start: 80,
                pitch: 127,
                duration: 4,
            }],
        };
        assert!(valid.validate().is_empty());

        let invalid = MirexGenerationV2026 {
            generation: vec![MirexNoteV2026 {
                start: 79,
                pitch: 128,
                duration: 4,
            }],
        };
        let issues = invalid.validate();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            MirexWireIssueV2026::PitchOutsideMidiRange { pitch: 128, .. }
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue,
            MirexWireIssueV2026::StartOutsideGeneration { .. }
        )));
    }

    #[test]
    fn prompt_and_generation_domains_remain_distinct() {
        let prompt = MirexPromptV2026 {
            prompt: vec![MirexNoteV2026 {
                start: 79,
                pitch: 60,
                duration: 4,
            }],
        };
        assert!(prompt.validate().is_empty());

        let wrong = MirexPromptV2026 {
            prompt: vec![MirexNoteV2026 {
                start: 80,
                pitch: 60,
                duration: 4,
            }],
        };
        assert!(wrong.validate().iter().any(|issue| matches!(
            issue,
            MirexWireIssueV2026::StartOutsidePrompt { .. }
        )));
    }

    #[test]
    fn local_continuation_projects_exactly_without_quantization() {
        let projection = project_score_to_mirex_generation(
            &continuation_score(),
            MirexScoreTimelineV2026::ContinuationLocal,
        )
        .unwrap();
        assert_eq!(
            projection.generation.generation,
            vec![
                MirexNoteV2026 {
                    start: 80,
                    pitch: 60,
                    duration: 4,
                },
                MirexNoteV2026 {
                    start: 84,
                    pitch: 64,
                    duration: 6,
                },
            ]
        );
        assert_eq!(
            projection.evidence.representation_losses.as_slice(),
            MIREX_REQUIRED_REPRESENTATION_LOSSES_V2026.as_slice()
        );
    }

    #[test]
    fn full_piece_projection_omits_prompt_but_does_not_retime_generation() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(Duration::new(19, 1), Duration::quarter(), 55)); // 76
        score.push(note(Duration::new(20, 1), Duration::quarter(), 60)); // 80
        score.push(note(Duration::new(21, 1), Duration::quarter(), 64)); // 84
        let projection = project_score_to_mirex_generation(
            &score,
            MirexScoreTimelineV2026::FullPieceGlobal,
        )
        .unwrap();
        assert_eq!(projection.evidence.prompt_region_notes_omitted, 1);
        assert_eq!(
            projection
                .generation
                .generation
                .iter()
                .map(|note| note.start)
                .collect::<Vec<_>>(),
            vec![80, 84]
        );
    }

    #[test]
    fn triplet_time_fails_instead_of_rounding_to_sixteenths() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(Duration::zero(), Duration::triplet_eighth(), 60));
        assert!(matches!(
            project_score_to_mirex_generation(
                &score,
                MirexScoreTimelineV2026::ContinuationLocal
            ),
            Err(MirexProjectionErrorV2026::NonSixteenthAligned {
                field: MirexWireFieldV2026::Duration,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_external_notes_are_preserved() {
        let generation = MirexGenerationV2026 {
            generation: vec![
                MirexNoteV2026 {
                    start: 80,
                    pitch: 40,
                    duration: 4,
                },
                MirexNoteV2026 {
                    start: 80,
                    pitch: 40,
                    duration: 4,
                },
            ],
        };
        assert!(generation.validate().is_empty());
        assert_eq!(generation.generation.len(), 2);
    }

    #[test]
    fn lossy_native_fields_remain_bound_through_source_identity() {
        let baseline = continuation_score();
        let mut changed_velocity = baseline.clone();
        changed_velocity.notes[0].velocity = 0.2;
        let left = project_score_to_mirex_generation(
            &baseline,
            MirexScoreTimelineV2026::ContinuationLocal,
        )
        .unwrap();
        let right = project_score_to_mirex_generation(
            &changed_velocity,
            MirexScoreTimelineV2026::ContinuationLocal,
        )
        .unwrap();

        assert_eq!(left.generation, right.generation);
        assert_ne!(left.evidence.source_score_sha256, right.evidence.source_score_sha256);
        assert_eq!(left.evidence.generation_sha256, right.evidence.generation_sha256);
    }

    #[test]
    fn subjective_profile_discloses_best_of_eight() {
        let (sampling, selection) = mirex_2026_subjective_sampling_profile();
        assert_eq!(sampling.samples_per_subject, 8);
        assert_eq!(sampling.seed_policy, BenchmarkSeedPolicyV1::ExternalProtocol);
        assert_eq!(
            selection,
            BenchmarkSelectionPolicyV1::BestOfN {
                n: 8,
                selector: BenchmarkSelectorV1::SystemSubmitterManual,
            }
        );
    }

    #[test]
    fn sample_file_names_match_submission_contract() {
        assert_eq!(mirex_sample_filename(1, 8).unwrap(), "sample_01.json");
        assert_eq!(mirex_sample_filename(8, 8).unwrap(), "sample_08.json");
        assert!(mirex_sample_filename(0, 8).is_err());
        assert!(mirex_sample_filename(1, 0).is_err());
    }
}
