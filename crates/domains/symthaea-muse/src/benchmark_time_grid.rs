// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact symbolic rasterization contract for external-comparable metrics.
//!
//! Melothaea stores beat time as exact rational values. External toolkits such
//! as MusPy evaluate several metrics on an integer time-step grid. This module
//! freezes that bridge without silently rounding native music to a convenient
//! resolution or accepting a forged external-compatibility identity.

use crate::evidence_digest::benchmark_symbolic_parity::{
    MUSPY_METRICS_SOURCE_BLOB, MUSPY_REFERENCE_REVISION,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{Duration, Score};

pub const SYMBOLIC_TIME_GRID_VERSION: &str = "melothaea-symbolic-time-grid-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalTimeGridFamilyV1 {
    MusPyResolution,
    InternalExactRaster,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyPolyphonyRateComparatorV1 {
    /// Audited MusPy source uses `active_pitch_count > threshold`.
    StrictlyGreaterThanThreshold,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicTimeGridPolicyV1 {
    pub version: String,
    pub family: ExternalTimeGridFamilyV1,
    /// Integer time steps per quarter-note beat.
    pub steps_per_quarter: u32,
    /// Current `Score::meter` uses quarter-note beats per measure, so this is
    /// `meter * steps_per_quarter` and is committed explicitly.
    pub steps_per_measure: u32,
    pub muspy_polyphony_rate_comparator: MuspyPolyphonyRateComparatorV1,
    pub muspy_reference_revision: String,
    pub muspy_metrics_source_blob: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RasterizedNoteV1 {
    pub start_step: u64,
    pub duration_steps: u64,
    pub pitch: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactSymbolicRasterV1 {
    pub policy: SymbolicTimeGridPolicyV1,
    pub source_score_sha256: String,
    pub total_steps: u64,
    /// Canonically ordered by `(start_step, duration_steps, pitch)` according
    /// to the derived struct ordering. Duplicate notes remain separate events.
    pub notes: Vec<RasterizedNoteV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RasterTimeFieldV1 {
    Onset,
    Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicTimeGridErrorV1 {
    WrongVersion { found: String },
    WrongFamily,
    WrongMuspyComparator,
    WrongMuspyReferenceRevision { found: String },
    WrongMuspyMetricsSourceBlob { found: String },
    ZeroResolution,
    ZeroMeter,
    StepsPerMeasureOverflow,
    StepsPerMeasureMismatch { expected: u32, found: u32 },
    NegativeOnset { note_index: usize },
    NonPositiveDuration { note_index: usize },
    NonRepresentableTime {
        note_index: usize,
        field: RasterTimeFieldV1,
        numerator: i64,
        denominator: i64,
        steps_per_quarter: u32,
    },
    StepOverflow { note_index: usize, field: RasterTimeFieldV1 },
    ScoreEndOverflow,
    Serialization,
}

impl SymbolicTimeGridPolicyV1 {
    pub fn muspy_source_exact(
        meter: u8,
        steps_per_quarter: u32,
    ) -> Result<Self, SymbolicTimeGridErrorV1> {
        if steps_per_quarter == 0 {
            return Err(SymbolicTimeGridErrorV1::ZeroResolution);
        }
        if meter == 0 {
            return Err(SymbolicTimeGridErrorV1::ZeroMeter);
        }
        let steps_per_measure = u32::from(meter)
            .checked_mul(steps_per_quarter)
            .ok_or(SymbolicTimeGridErrorV1::StepsPerMeasureOverflow)?;
        Ok(Self {
            version: SYMBOLIC_TIME_GRID_VERSION.into(),
            family: ExternalTimeGridFamilyV1::MusPyResolution,
            steps_per_quarter,
            steps_per_measure,
            muspy_polyphony_rate_comparator:
                MuspyPolyphonyRateComparatorV1::StrictlyGreaterThanThreshold,
            muspy_reference_revision: MUSPY_REFERENCE_REVISION.into(),
            muspy_metrics_source_blob: MUSPY_METRICS_SOURCE_BLOB.into(),
        })
    }

    pub fn validate_for_meter(&self, meter: u8) -> Result<(), SymbolicTimeGridErrorV1> {
        if self.version != SYMBOLIC_TIME_GRID_VERSION {
            return Err(SymbolicTimeGridErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.family != ExternalTimeGridFamilyV1::MusPyResolution {
            return Err(SymbolicTimeGridErrorV1::WrongFamily);
        }
        if self.muspy_polyphony_rate_comparator
            != MuspyPolyphonyRateComparatorV1::StrictlyGreaterThanThreshold
        {
            return Err(SymbolicTimeGridErrorV1::WrongMuspyComparator);
        }
        if self.muspy_reference_revision != MUSPY_REFERENCE_REVISION {
            return Err(SymbolicTimeGridErrorV1::WrongMuspyReferenceRevision {
                found: self.muspy_reference_revision.clone(),
            });
        }
        if self.muspy_metrics_source_blob != MUSPY_METRICS_SOURCE_BLOB {
            return Err(SymbolicTimeGridErrorV1::WrongMuspyMetricsSourceBlob {
                found: self.muspy_metrics_source_blob.clone(),
            });
        }
        if self.steps_per_quarter == 0 {
            return Err(SymbolicTimeGridErrorV1::ZeroResolution);
        }
        if meter == 0 {
            return Err(SymbolicTimeGridErrorV1::ZeroMeter);
        }
        let expected = u32::from(meter)
            .checked_mul(self.steps_per_quarter)
            .ok_or(SymbolicTimeGridErrorV1::StepsPerMeasureOverflow)?;
        if self.steps_per_measure != expected {
            return Err(SymbolicTimeGridErrorV1::StepsPerMeasureMismatch {
                expected,
                found: self.steps_per_measure,
            });
        }
        Ok(())
    }

    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

impl ExactSymbolicRasterV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

/// Rasterize only when every native onset/duration is exactly representable at
/// the declared resolution. No rounding, truncation, clipping, or snapping is
/// permitted, and the serialized policy is fully revalidated before use.
pub fn rasterize_score_exact(
    score: &Score,
    policy: SymbolicTimeGridPolicyV1,
) -> Result<ExactSymbolicRasterV1, SymbolicTimeGridErrorV1> {
    policy.validate_for_meter(score.meter)?;

    let source_score_sha256 =
        canonical_json_sha256(score).map_err(|_| SymbolicTimeGridErrorV1::Serialization)?;
    let mut notes = Vec::with_capacity(score.notes.len());
    let mut total_steps = 0u64;

    for (note_index, note) in score.notes.iter().enumerate() {
        if note.onset.num() < 0 {
            return Err(SymbolicTimeGridErrorV1::NegativeOnset { note_index });
        }
        if note.duration.num() <= 0 {
            return Err(SymbolicTimeGridErrorV1::NonPositiveDuration { note_index });
        }
        let start_step = exact_steps(
            note_index,
            RasterTimeFieldV1::Onset,
            note.onset,
            policy.steps_per_quarter,
        )?;
        let duration_steps = exact_steps(
            note_index,
            RasterTimeFieldV1::Duration,
            note.duration,
            policy.steps_per_quarter,
        )?;
        let end = start_step
            .checked_add(duration_steps)
            .ok_or(SymbolicTimeGridErrorV1::ScoreEndOverflow)?;
        total_steps = total_steps.max(end);
        notes.push(RasterizedNoteV1 {
            start_step,
            duration_steps,
            pitch: note.pitch.midi(),
        });
    }

    notes.sort();
    Ok(ExactSymbolicRasterV1 {
        policy,
        source_score_sha256,
        total_steps,
        notes,
    })
}

fn exact_steps(
    note_index: usize,
    field: RasterTimeFieldV1,
    value: Duration,
    steps_per_quarter: u32,
) -> Result<u64, SymbolicTimeGridErrorV1> {
    let scaled = i128::from(value.num())
        .checked_mul(i128::from(steps_per_quarter))
        .ok_or(SymbolicTimeGridErrorV1::StepOverflow { note_index, field })?;
    let denominator = i128::from(value.den());
    if scaled % denominator != 0 {
        return Err(SymbolicTimeGridErrorV1::NonRepresentableTime {
            note_index,
            field,
            numerator: value.num(),
            denominator: value.den(),
            steps_per_quarter,
        });
    }
    let steps = scaled / denominator;
    u64::try_from(steps).map_err(|_| SymbolicTimeGridErrorV1::StepOverflow { note_index, field })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        Emphasis, Key, PartId, Pitch, PitchClass, ScoreNote, VoiceRole,
    };

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

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(Duration::zero(), Duration::quarter(), 60));
        score.push(note(Duration::quarter(), Duration::eighth(), 64));
        score
    }

    #[test]
    fn policy_binds_resolution_meter_and_audited_external_source() {
        let policy = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 24).unwrap();
        assert_eq!(policy.steps_per_quarter, 24);
        assert_eq!(policy.steps_per_measure, 96);
        assert_eq!(policy.muspy_reference_revision, MUSPY_REFERENCE_REVISION);
        assert_eq!(policy.muspy_metrics_source_blob, MUSPY_METRICS_SOURCE_BLOB);
        assert_eq!(
            policy.muspy_polyphony_rate_comparator,
            MuspyPolyphonyRateComparatorV1::StrictlyGreaterThanThreshold
        );
        assert!(policy.validate_for_meter(4).is_ok());
    }

    #[test]
    fn deserialized_policy_identity_is_revalidated() {
        let mut policy = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 24).unwrap();
        policy.muspy_reference_revision = "0000000000000000000000000000000000000000".into();
        assert!(matches!(
            rasterize_score_exact(&score(), policy),
            Err(SymbolicTimeGridErrorV1::WrongMuspyReferenceRevision { .. })
        ));
    }

    #[test]
    fn exact_quarter_and_eighth_rasterize_without_loss() {
        let policy = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 4).unwrap();
        let raster = rasterize_score_exact(&score(), policy).unwrap();
        assert_eq!(
            raster.notes,
            vec![
                RasterizedNoteV1 {
                    start_step: 0,
                    duration_steps: 4,
                    pitch: 60,
                },
                RasterizedNoteV1 {
                    start_step: 4,
                    duration_steps: 2,
                    pitch: 64,
                },
            ]
        );
        assert_eq!(raster.total_steps, 6);
    }

    #[test]
    fn resolution_is_load_bearing_for_tuplets() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(Duration::zero(), Duration::triplet_eighth(), 60));

        let coarse = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 4).unwrap();
        assert!(matches!(
            rasterize_score_exact(&score, coarse),
            Err(SymbolicTimeGridErrorV1::NonRepresentableTime {
                field: RasterTimeFieldV1::Duration,
                ..
            })
        ));

        let divisible = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 12).unwrap();
        let raster = rasterize_score_exact(&score, divisible).unwrap();
        assert_eq!(raster.notes[0].duration_steps, 4);
    }

    #[test]
    fn changing_resolution_changes_policy_and_raster_identity() {
        let p4 = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 4).unwrap();
        let p8 = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 8).unwrap();
        assert_ne!(p4.canonical_sha256().unwrap(), p8.canonical_sha256().unwrap());

        let r4 = rasterize_score_exact(&score(), p4).unwrap();
        let r8 = rasterize_score_exact(&score(), p8).unwrap();
        assert_ne!(r4.canonical_sha256().unwrap(), r8.canonical_sha256().unwrap());
    }

    #[test]
    fn duplicate_notes_remain_distinct_raster_events() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        let n = note(Duration::zero(), Duration::quarter(), 60);
        score.push(n);
        score.push(n);
        let policy = SymbolicTimeGridPolicyV1::muspy_source_exact(4, 4).unwrap();
        let raster = rasterize_score_exact(&score, policy).unwrap();
        assert_eq!(raster.notes.len(), 2);
        assert_eq!(raster.notes[0], raster.notes[1]);
    }
}
