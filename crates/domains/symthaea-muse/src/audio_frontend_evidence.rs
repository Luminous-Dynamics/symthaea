// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit confidence and provenance records for audio-to-symbolic front ends.
//!
//! This module does not implement pitch tracking, onset detection, chroma or
//! beat tracking. It defines the evidence boundary those implementations must
//! satisfy before their estimates are passed to a symbolic solver. The purpose
//! is to keep "the waveform was misheard" distinguishable from "the notes were
//! reasoned about incorrectly".

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const AUDIO_FRONTEND_EVIDENCE_SCHEMA_V1: &str = "mel-audio-frontend-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFrontEndStageV1 {
    PitchSalience,
    OnsetDetection,
    ChromaExtraction,
    BeatGridTracking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFrontEndStatusV1 {
    Extracted,
    Abstained,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFrontEndEstimateV1 {
    pub stage: AudioFrontEndStageV1,
    pub status: AudioFrontEndStatusV1,
    /// Content digest of the typed estimate payload (pitch curve, onset list,
    /// chroma sequence or beat grid). Never substitute a digest of the audio.
    pub estimate_sha256: Option<String>,
    /// Confidence and uncertainty are separate values and both use [0, 1].
    pub confidence: Option<f32>,
    pub uncertainty: Option<f32>,
    /// When confidence is a calibrated probability, bind the calibrator and
    /// calibration cohort. None means the score is explicitly uncalibrated.
    pub calibration_identity_sha256: Option<String>,
    /// Required for explicit abstention or a stage that was not run.
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFrontEndEvidenceV1 {
    pub schema: String,
    pub case_id: String,
    pub audio_input_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub sample_count: u64,
    pub front_end_id: String,
    pub front_end_version: String,
    pub front_end_identity_sha256: String,
    /// All four stages must be represented, including explicit abstentions or
    /// not-run statuses. Omission is not interpreted as a confident success.
    pub estimates: Vec<AudioFrontEndEstimateV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFrontEndEvidenceErrorV1 {
    InvalidSchema,
    CaseIdMismatch,
    AudioInputDigestMismatch,
    InvalidDigest { field: String },
    InvalidAudioFormat,
    EmptyFrontEndIdentity,
    MissingStage { stage: AudioFrontEndStageV1 },
    DuplicateStage { stage: AudioFrontEndStageV1 },
    InvalidConfidence { stage: AudioFrontEndStageV1 },
    InvalidUncertainty { stage: AudioFrontEndStageV1 },
    MissingEstimateDigest { stage: AudioFrontEndStageV1 },
    EstimateDigestPresentWithoutOutput { stage: AudioFrontEndStageV1 },
    CalibrationDigestPresentWithoutEstimate { stage: AudioFrontEndStageV1 },
    AbstentionHasConfidence { stage: AudioFrontEndStageV1 },
    AbstentionMissingReason { stage: AudioFrontEndStageV1 },
    NotRunHasOutput { stage: AudioFrontEndStageV1 },
    NotRunMissingReason { stage: AudioFrontEndStageV1 },
    ExtractedHasReason { stage: AudioFrontEndStageV1 },
}

const REQUIRED_STAGES: [AudioFrontEndStageV1; 4] = [
    AudioFrontEndStageV1::PitchSalience,
    AudioFrontEndStageV1::OnsetDetection,
    AudioFrontEndStageV1::ChromaExtraction,
    AudioFrontEndStageV1::BeatGridTracking,
];

/// Validate one audio front-end report against the exact paired benchmark case.
///
/// The function validates provenance, shape and uncertainty semantics. It
/// does not assert that an estimate is musically correct or perceptually good.
pub fn validate_audio_frontend_evidence(
    expected_case_id: &str,
    expected_audio_sha256: &str,
    evidence: &AudioFrontEndEvidenceV1,
) -> Result<(), AudioFrontEndEvidenceErrorV1> {
    if evidence.schema != AUDIO_FRONTEND_EVIDENCE_SCHEMA_V1 {
        return Err(AudioFrontEndEvidenceErrorV1::InvalidSchema);
    }
    if evidence.case_id != expected_case_id {
        return Err(AudioFrontEndEvidenceErrorV1::CaseIdMismatch);
    }
    validate_digest("expected_audio_sha256", expected_audio_sha256)?;
    validate_digest("audio_input_sha256", &evidence.audio_input_sha256)?;
    if evidence.audio_input_sha256 != expected_audio_sha256 {
        return Err(AudioFrontEndEvidenceErrorV1::AudioInputDigestMismatch);
    }
    validate_digest(
        "front_end_identity_sha256",
        &evidence.front_end_identity_sha256,
    )?;
    if evidence.sample_rate_hz < 8_000
        || evidence.sample_rate_hz > 192_000
        || evidence.channels == 0
        || evidence.channels > 8
        || evidence.sample_count == 0
    {
        return Err(AudioFrontEndEvidenceErrorV1::InvalidAudioFormat);
    }
    if evidence.front_end_id.trim().is_empty() || evidence.front_end_version.trim().is_empty() {
        return Err(AudioFrontEndEvidenceErrorV1::EmptyFrontEndIdentity);
    }

    let mut seen = HashSet::with_capacity(REQUIRED_STAGES.len());
    for estimate in &evidence.estimates {
        let stage = estimate.stage;
        if !seen.insert(stage) {
            return Err(AudioFrontEndEvidenceErrorV1::DuplicateStage { stage });
        }
        match estimate.status {
            AudioFrontEndStatusV1::Extracted => {
                let Some(estimate_sha256) = &estimate.estimate_sha256 else {
                    return Err(AudioFrontEndEvidenceErrorV1::MissingEstimateDigest { stage });
                };
                validate_digest("estimate_sha256", estimate_sha256)?;
                if let Some(confidence) = estimate.confidence {
                    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                        return Err(AudioFrontEndEvidenceErrorV1::InvalidConfidence { stage });
                    }
                } else {
                    return Err(AudioFrontEndEvidenceErrorV1::InvalidConfidence { stage });
                }
                if let Some(uncertainty) = estimate.uncertainty {
                    if !uncertainty.is_finite() || !(0.0..=1.0).contains(&uncertainty) {
                        return Err(AudioFrontEndEvidenceErrorV1::InvalidUncertainty { stage });
                    }
                } else {
                    return Err(AudioFrontEndEvidenceErrorV1::InvalidUncertainty { stage });
                }
                if let Some(calibration_sha256) = &estimate.calibration_identity_sha256 {
                    validate_digest("calibration_identity_sha256", calibration_sha256)?;
                }
                if estimate.reason.is_some() {
                    return Err(AudioFrontEndEvidenceErrorV1::ExtractedHasReason { stage });
                }
            }
            AudioFrontEndStatusV1::Abstained => {
                if estimate.estimate_sha256.is_some() {
                    return Err(AudioFrontEndEvidenceErrorV1::EstimateDigestPresentWithoutOutput {
                        stage,
                    });
                }
                if estimate.confidence.is_some() || estimate.uncertainty.is_some() {
                    return Err(AudioFrontEndEvidenceErrorV1::AbstentionHasConfidence { stage });
                }
                if estimate.calibration_identity_sha256.is_some() {
                    return Err(
                        AudioFrontEndEvidenceErrorV1::CalibrationDigestPresentWithoutEstimate {
                            stage,
                        },
                    );
                }
                if estimate.reason.as_deref().is_none_or(str::is_empty) {
                    return Err(AudioFrontEndEvidenceErrorV1::AbstentionMissingReason { stage });
                }
            }
            AudioFrontEndStatusV1::NotRun => {
                if estimate.estimate_sha256.is_some()
                    || estimate.confidence.is_some()
                    || estimate.uncertainty.is_some()
                    || estimate.calibration_identity_sha256.is_some()
                {
                    return Err(AudioFrontEndEvidenceErrorV1::NotRunHasOutput { stage });
                }
                if estimate.reason.as_deref().is_none_or(str::is_empty) {
                    return Err(AudioFrontEndEvidenceErrorV1::NotRunMissingReason { stage });
                }
            }
        }
    }

    for stage in REQUIRED_STAGES {
        if !seen.contains(&stage) {
            return Err(AudioFrontEndEvidenceErrorV1::MissingStage { stage });
        }
    }
    Ok(())
}

fn validate_digest(
    field: &str,
    value: &str,
) -> Result<(), AudioFrontEndEvidenceErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AudioFrontEndEvidenceErrorV1::InvalidDigest {
            field: field.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> String {
        format!("{value:02x}{}", "b".repeat(62))
    }

    fn extracted(stage: AudioFrontEndStageV1, seed: u8) -> AudioFrontEndEstimateV1 {
        AudioFrontEndEstimateV1 {
            stage,
            status: AudioFrontEndStatusV1::Extracted,
            estimate_sha256: Some(digest(seed)),
            confidence: Some(0.85),
            uncertainty: Some(0.15),
            calibration_identity_sha256: Some(digest(seed + 1)),
            reason: None,
        }
    }

    fn evidence() -> AudioFrontEndEvidenceV1 {
        AudioFrontEndEvidenceV1 {
            schema: AUDIO_FRONTEND_EVIDENCE_SCHEMA_V1.into(),
            case_id: "paired-case-001".into(),
            audio_input_sha256: digest(1),
            sample_rate_hz: 48_000,
            channels: 2,
            sample_count: 480_000,
            front_end_id: "fixture-front-end".into(),
            front_end_version: "0.1.0".into(),
            front_end_identity_sha256: digest(2),
            estimates: vec![
                extracted(AudioFrontEndStageV1::PitchSalience, 3),
                extracted(AudioFrontEndStageV1::OnsetDetection, 5),
                extracted(AudioFrontEndStageV1::ChromaExtraction, 7),
                extracted(AudioFrontEndStageV1::BeatGridTracking, 9),
            ],
        }
    }

    #[test]
    fn complete_frontend_evidence_validates_against_exact_audio_input() {
        let record = evidence();
        validate_audio_frontend_evidence(
            "paired-case-001",
            &record.audio_input_sha256,
            &record,
        )
        .unwrap();
    }

    #[test]
    fn audio_or_case_digest_mismatch_is_rejected() {
        let record = evidence();
        assert_eq!(
            validate_audio_frontend_evidence("different-case", &record.audio_input_sha256, &record),
            Err(AudioFrontEndEvidenceErrorV1::CaseIdMismatch)
        );
        assert_eq!(
            validate_audio_frontend_evidence("paired-case-001", &digest(99), &record),
            Err(AudioFrontEndEvidenceErrorV1::AudioInputDigestMismatch)
        );
    }

    #[test]
    fn all_frontend_stages_must_be_explicit_even_when_not_run() {
        let mut record = evidence();
        record.estimates.pop();
        assert!(matches!(
            validate_audio_frontend_evidence(
                "paired-case-001",
                &record.audio_input_sha256,
                &record
            ),
            Err(AudioFrontEndEvidenceErrorV1::MissingStage {
                stage: AudioFrontEndStageV1::BeatGridTracking
            })
        ));

        record.estimates.push(AudioFrontEndEstimateV1 {
            stage: AudioFrontEndStageV1::BeatGridTracking,
            status: AudioFrontEndStatusV1::NotRun,
            estimate_sha256: None,
            confidence: None,
            uncertainty: None,
            calibration_identity_sha256: None,
            reason: Some("beat grid not requested by this task".into()),
        });
        validate_audio_frontend_evidence(
            "paired-case-001",
            &record.audio_input_sha256,
            &record,
        )
        .unwrap();
    }

    #[test]
    fn abstention_requires_a_reason_and_may_not_claim_an_estimate() {
        let mut record = evidence();
        let onset = record
            .estimates
            .iter_mut()
            .find(|estimate| estimate.stage == AudioFrontEndStageV1::OnsetDetection)
            .unwrap();
        onset.status = AudioFrontEndStatusV1::Abstained;
        onset.estimate_sha256 = None;
        onset.confidence = None;
        onset.uncertainty = None;
        onset.calibration_identity_sha256 = None;
        onset.reason = Some("low signal-to-noise ratio".into());
        validate_audio_frontend_evidence(
            "paired-case-001",
            &record.audio_input_sha256,
            &record,
        )
        .unwrap();

        let onset = record
            .estimates
            .iter_mut()
            .find(|estimate| estimate.stage == AudioFrontEndStageV1::OnsetDetection)
            .unwrap();
        onset.reason = None;
        assert_eq!(
            validate_audio_frontend_evidence(
                "paired-case-001",
                &record.audio_input_sha256,
                &record
            ),
            Err(AudioFrontEndEvidenceErrorV1::AbstentionMissingReason {
                stage: AudioFrontEndStageV1::OnsetDetection
            })
        );
    }

    #[test]
    fn duplicate_stages_and_invalid_confidence_are_rejected() {
        let mut record = evidence();
        record.estimates.push(record.estimates[0].clone());
        assert_eq!(
            validate_audio_frontend_evidence(
                "paired-case-001",
                &record.audio_input_sha256,
                &record
            ),
            Err(AudioFrontEndEvidenceErrorV1::DuplicateStage {
                stage: AudioFrontEndStageV1::PitchSalience
            })
        );

        let mut record = evidence();
        record.estimates[0].confidence = Some(f32::NAN);
        assert_eq!(
            validate_audio_frontend_evidence(
                "paired-case-001",
                &record.audio_input_sha256,
                &record
            ),
            Err(AudioFrontEndEvidenceErrorV1::InvalidConfidence {
                stage: AudioFrontEndStageV1::PitchSalience
            })
        );
    }

    #[test]
    fn uncalibrated_scores_are_explicitly_distinguishable_from_calibrated_ones() {
        let mut calibrated = evidence();
        assert!(calibrated.estimates[0]
            .calibration_identity_sha256
            .is_some());
        calibrated.estimates[0].calibration_identity_sha256 = None;
        validate_audio_frontend_evidence(
            "paired-case-001",
            &calibrated.audio_input_sha256,
            &calibrated,
        )
        .unwrap();
    }
}
