// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural validation for serialized MEL-003P1CR qualification receipts.
//!
//! Raw WAV bytes are required to *produce* a P1CR qualification. Downstream
//! stages should not need those bytes merely to detect receipt pruning,
//! substitution, relabeling, or resealing. This module therefore re-binds a
//! serialized receipt to the exact P1A/P1C subjects and re-derives every
//! manifest-level invariant that remains checkable without reopening audio.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
        LOUDNESS_REMEASUREMENT_TOLERANCE_LU, MAX_RESIDUAL_PAIR_DELTA_LU_V1,
        NUMERIC_TOLERANCE_DB, REQUIRED_CHANNEL_COUNT, REQUIRED_SAMPLE_RATE_HZ,
        StimulusArmV1, StimulusAudioAssetV1,
    },
    perceptual_stimulus_qualification::{
        ATTENUATION_SAMPLE_TRANSFORM_PROFILE_V1, CANONICAL_PCM_DIGEST_PROFILE_V1,
        CANONICAL_WAV_PROFILE_V1, FrozenPerceptualStimulusByteQualificationV1,
        QualifiedStimulusAssetV1, QualifiedStimulusPairV1,
        STIMULUS_BYTE_QUALIFICATION_VERSION, qualification_commitment,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimulusQualificationReceiptIssueV1 {
    InvalidProtocol,
    InvalidRenderSubjectBinding,
    InvalidStimulusPack,
    ProtocolSerializationFailed,
    RenderSubjectBindingSerializationFailed,
    StimulusPackSerializationFailed,
    ProtocolDigestMismatch,
    RenderSubjectBindingDigestMismatch,
    StimulusPackDigestMismatch,
    WrongQualificationVersion,
    WrongWavProfile,
    WrongPcmDigestProfile,
    WrongTransformProfile,
    WrongPairCount { found: usize, expected: usize },
    WrongPairIdentity { index: usize },
    AssetIdentityMismatch { item_id: String, arm: StimulusArmV1 },
    InvalidDigest { item_id: String, arm: StimulusArmV1, field: String },
    GeometryMismatch { item_id: String, arm: StimulusArmV1 },
    GainMismatch { item_id: String, arm: StimulusArmV1 },
    LoudnessMismatch { item_id: String, arm: StimulusArmV1, field: String },
    ReconstructedFileDigestMismatch { item_id: String, arm: StimulusArmV1 },
    ReconstructedPcmDigestMismatch { item_id: String, arm: StimulusArmV1 },
    ReconstructionNotExact { item_id: String, arm: StimulusArmV1 },
    UnchangedArmNotIdentical { item_id: String, arm: StimulusArmV1 },
    PairDeltaMismatch { item_id: String },
    PairDeltaExceedsTolerance { item_id: String },
    QualificationSerializationFailed,
    QualificationDigestMismatch,
}

pub fn validate_stimulus_byte_qualification_receipt(
    receipt: &FrozenPerceptualStimulusByteQualificationV1,
    protocol: &FrozenPerceptualStudyProtocolV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    pack: &FrozenPerceptualStimulusPackV1,
) -> Vec<StimulusQualificationReceiptIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(StimulusQualificationReceiptIssueV1::InvalidProtocol);
    }
    if !render_binding.validate(protocol).is_empty() {
        issues.push(StimulusQualificationReceiptIssueV1::InvalidRenderSubjectBinding);
    }
    if !pack.validate(protocol, render_binding).is_empty() {
        issues.push(StimulusQualificationReceiptIssueV1::InvalidStimulusPack);
    }

    match canonical_json_sha256(protocol) {
        Ok(value) if value == receipt.protocol_sha256 => {}
        Ok(_) => issues.push(StimulusQualificationReceiptIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(StimulusQualificationReceiptIssueV1::ProtocolSerializationFailed),
    }
    match canonical_json_sha256(render_binding) {
        Ok(value) if value == receipt.render_subject_binding_sha256 => {}
        Ok(_) => {
            issues.push(StimulusQualificationReceiptIssueV1::RenderSubjectBindingDigestMismatch)
        }
        Err(_) => issues.push(
            StimulusQualificationReceiptIssueV1::RenderSubjectBindingSerializationFailed,
        ),
    }
    match canonical_json_sha256(pack) {
        Ok(value) if value == receipt.stimulus_pack_sha256 => {}
        Ok(_) => issues.push(StimulusQualificationReceiptIssueV1::StimulusPackDigestMismatch),
        Err(_) => {
            issues.push(StimulusQualificationReceiptIssueV1::StimulusPackSerializationFailed)
        }
    }

    if receipt.qualification_version != STIMULUS_BYTE_QUALIFICATION_VERSION {
        issues.push(StimulusQualificationReceiptIssueV1::WrongQualificationVersion);
    }
    if receipt.canonical_wav_profile != CANONICAL_WAV_PROFILE_V1 {
        issues.push(StimulusQualificationReceiptIssueV1::WrongWavProfile);
    }
    if receipt.canonical_pcm_digest_profile != CANONICAL_PCM_DIGEST_PROFILE_V1 {
        issues.push(StimulusQualificationReceiptIssueV1::WrongPcmDigestProfile);
    }
    if receipt.attenuation_sample_transform_profile != ATTENUATION_SAMPLE_TRANSFORM_PROFILE_V1 {
        issues.push(StimulusQualificationReceiptIssueV1::WrongTransformProfile);
    }

    if receipt.pairs.len() != pack.items.len() {
        issues.push(StimulusQualificationReceiptIssueV1::WrongPairCount {
            found: receipt.pairs.len(),
            expected: pack.items.len(),
        });
    }

    for (index, manifest_pair) in pack.items.iter().enumerate() {
        let Some(pair) = receipt.pairs.get(index) else {
            continue;
        };
        if pair.item_id != manifest_pair.item_id || pair.seed != manifest_pair.seed {
            issues.push(StimulusQualificationReceiptIssueV1::WrongPairIdentity { index });
            continue;
        }

        validate_asset(
            pair,
            &pair.baseline,
            StimulusArmV1::Baseline,
            &manifest_pair.baseline,
            manifest_pair.attenuated_arm,
            &mut issues,
        );
        validate_asset(
            pair,
            &pair.intervention,
            StimulusArmV1::Intervention,
            &manifest_pair.intervention,
            manifest_pair.attenuated_arm,
            &mut issues,
        );

        let recomputed_delta =
            (pair.baseline.measured_final_lufs - pair.intervention.measured_final_lufs).abs();
        if !same_number(
            recomputed_delta,
            pair.independently_measured_pair_delta_lu,
            LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
        ) || !same_number(
            recomputed_delta,
            manifest_pair.post_match_pair_delta_lu,
            LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
        ) {
            issues.push(StimulusQualificationReceiptIssueV1::PairDeltaMismatch {
                item_id: pair.item_id.clone(),
            });
        }
        if !recomputed_delta.is_finite() || recomputed_delta > MAX_RESIDUAL_PAIR_DELTA_LU_V1 {
            issues.push(StimulusQualificationReceiptIssueV1::PairDeltaExceedsTolerance {
                item_id: pair.item_id.clone(),
            });
        }
    }

    match qualification_commitment(receipt) {
        Ok(value) if value == receipt.qualification_sha256 => {}
        Ok(_) => issues.push(StimulusQualificationReceiptIssueV1::QualificationDigestMismatch),
        Err(_) => {
            issues.push(StimulusQualificationReceiptIssueV1::QualificationSerializationFailed)
        }
    }

    issues
}

fn validate_asset(
    pair: &QualifiedStimulusPairV1,
    asset: &QualifiedStimulusAssetV1,
    arm: StimulusArmV1,
    manifest: &StimulusAudioAssetV1,
    attenuated_arm: Option<StimulusArmV1>,
    issues: &mut Vec<StimulusQualificationReceiptIssueV1>,
) {
    if asset.item_id != pair.item_id || asset.seed != pair.seed || asset.arm != arm {
        issues.push(StimulusQualificationReceiptIssueV1::AssetIdentityMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }

    for (field, digest) in [
        ("source_file_sha256", asset.source_file_sha256.as_str()),
        ("source_pcm_sha256", asset.source_pcm_sha256.as_str()),
        ("output_file_sha256", asset.output_file_sha256.as_str()),
        ("output_pcm_sha256", asset.output_pcm_sha256.as_str()),
        (
            "reconstructed_output_file_sha256",
            asset.reconstructed_output_file_sha256.as_str(),
        ),
        (
            "reconstructed_output_pcm_sha256",
            asset.reconstructed_output_pcm_sha256.as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(StimulusQualificationReceiptIssueV1::InvalidDigest {
                item_id: pair.item_id.clone(),
                arm,
                field: field.into(),
            });
        }
    }

    if asset.source_file_sha256 != manifest.source_sha256
        || asset.output_file_sha256 != manifest.output_sha256
    {
        issues.push(StimulusQualificationReceiptIssueV1::AssetIdentityMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if asset.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ
        || asset.channel_count != REQUIRED_CHANNEL_COUNT
        || asset.frame_count != manifest.frame_count
        || manifest.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ
        || manifest.channel_count != REQUIRED_CHANNEL_COUNT
    {
        issues.push(StimulusQualificationReceiptIssueV1::GeometryMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !same_number(asset.applied_gain_db, manifest.applied_gain_db, NUMERIC_TOLERANCE_DB) {
        issues.push(StimulusQualificationReceiptIssueV1::GainMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !same_number(
        asset.measured_initial_lufs,
        manifest.initial_integrated_lufs,
        LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
    ) {
        issues.push(StimulusQualificationReceiptIssueV1::LoudnessMismatch {
            item_id: pair.item_id.clone(),
            arm,
            field: "initial".into(),
        });
    }
    if !same_number(
        asset.measured_final_lufs,
        manifest.final_integrated_lufs,
        LOUDNESS_REMEASUREMENT_TOLERANCE_LU,
    ) {
        issues.push(StimulusQualificationReceiptIssueV1::LoudnessMismatch {
            item_id: pair.item_id.clone(),
            arm,
            field: "final".into(),
        });
    }

    if asset.reconstructed_output_file_sha256 != asset.output_file_sha256 {
        issues.push(StimulusQualificationReceiptIssueV1::ReconstructedFileDigestMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if asset.reconstructed_output_pcm_sha256 != asset.output_pcm_sha256 {
        issues.push(StimulusQualificationReceiptIssueV1::ReconstructedPcmDigestMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !asset.reconstructed_output_exact {
        issues.push(StimulusQualificationReceiptIssueV1::ReconstructionNotExact {
            item_id: pair.item_id.clone(),
            arm,
        });
    }

    if attenuated_arm != Some(arm)
        && (!asset.source_output_file_identical
            || asset.source_file_sha256 != asset.output_file_sha256
            || asset.source_pcm_sha256 != asset.output_pcm_sha256)
    {
        issues.push(StimulusQualificationReceiptIssueV1::UnchangedArmNotIdentical {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
}

fn same_number(left: f64, right: f64, tolerance: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= tolerance
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha_shape_guard_rejects_non_digest_strings() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256("abc"));
        assert!(!is_sha256(&"g".repeat(64)));
    }

    #[test]
    fn receipt_numeric_comparison_fails_closed_on_non_finite_values() {
        assert!(same_number(-20.0, -20.005, 0.01));
        assert!(!same_number(f64::NAN, f64::NAN, 0.01));
        assert!(!same_number(f64::INFINITY, f64::INFINITY, 0.01));
    }
}
