// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen-holdout receipt binding for psych-bench calibration evidence.
//!
//! The base calibration contract establishes parameter origin and manifest
//! identity. This layer preserves the full pre-scoring commitment in the final
//! receipt so downstream consumers cannot lose the code/task/holdout/lineage
//! identities while retaining only a parameter digest.

use crate::calibration_contract::{
    CalibrationContractError, CalibrationEvaluationReceipt, CalibrationFreezeStatus,
    CalibrationManifest, EvaluationCalibrationMode, FreezeBindingAuthority,
    FrozenCalibrationCommitment, evaluate_calibration_contract,
};
use serde::{Deserialize, Serialize};

pub const FROZEN_CALIBRATION_RECEIPT_SCHEMA_VERSION: &str = "psych-frozen-calibration-receipt-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenCalibrationEvidenceReceipt {
    pub schema_version: String,
    pub evaluation: CalibrationEvaluationReceipt,
    pub commitment: FrozenCalibrationCommitment,
    pub commitment_digest: String,
}

impl FrozenCalibrationEvidenceReceipt {
    pub fn new(
        manifest: &CalibrationManifest,
        commitment: &FrozenCalibrationCommitment,
    ) -> Result<Self, FrozenCalibrationReceiptError> {
        let evaluation = evaluate_calibration_contract(
            EvaluationCalibrationMode::FrozenHoldout,
            manifest,
            Some(commitment),
        )?;
        let commitment_digest = frozen_commitment_digest_hex(commitment)?;

        Ok(Self {
            schema_version: FROZEN_CALIBRATION_RECEIPT_SCHEMA_VERSION.to_string(),
            evaluation,
            commitment: commitment.clone(),
            commitment_digest,
        })
    }

    /// Revalidate all nested identities before authority-bearing consumption.
    pub fn validate(
        &self,
        manifest: &CalibrationManifest,
    ) -> Result<(), FrozenCalibrationReceiptError> {
        if self.schema_version != FROZEN_CALIBRATION_RECEIPT_SCHEMA_VERSION {
            return Err(FrozenCalibrationReceiptError::UnsupportedReceiptSchema);
        }

        let actual_digest = frozen_commitment_digest_hex(&self.commitment)?;
        if actual_digest != self.commitment_digest {
            return Err(FrozenCalibrationReceiptError::CommitmentDigestMismatch);
        }

        let expected = evaluate_calibration_contract(
            EvaluationCalibrationMode::FrozenHoldout,
            manifest,
            Some(&self.commitment),
        )?;
        if expected != self.evaluation {
            return Err(FrozenCalibrationReceiptError::EvaluationMismatch);
        }

        if self.evaluation.evidence_profile.freeze_status
            != CalibrationFreezeStatus::VerifiedFrozen
        {
            return Err(FrozenCalibrationReceiptError::EvaluationMismatch);
        }
        Ok(())
    }
}

/// Canonical commitment digest. Every identity that defines the frozen
/// experiment is included in fixed field order.
pub fn frozen_commitment_digest_hex(
    commitment: &FrozenCalibrationCommitment,
) -> Result<String, CalibrationContractError> {
    commitment.validate()?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.psych.frozen-calibration-commitment.v1\0");
    hash_field(&mut hasher, commitment.schema_version.as_bytes());
    hash_field(
        &mut hasher,
        commitment.parameter_manifest_digest.as_bytes(),
    );
    hash_field(&mut hasher, commitment.code_subject.as_bytes());
    hash_field(&mut hasher, commitment.task_set_id.as_bytes());
    hash_field(
        &mut hasher,
        commitment.baseline_or_holdout_id.as_bytes(),
    );
    hash_field(&mut hasher, commitment.lineage_id.as_bytes());
    hash_field(&mut hasher, commitment.freeze_artifact_id.as_bytes());
    hasher.update(&[match commitment.binding_authority {
        FreezeBindingAuthority::DeclaredOnly => 1,
        FreezeBindingAuthority::PreScoringEvidenceBound => 2,
    }]);

    Ok(hasher.finalize().to_hex().to_string())
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenCalibrationReceiptError {
    Contract(CalibrationContractError),
    UnsupportedReceiptSchema,
    CommitmentDigestMismatch,
    EvaluationMismatch,
}

impl From<CalibrationContractError> for FrozenCalibrationReceiptError {
    fn from(value: CalibrationContractError) -> Self {
        Self::Contract(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration_contract::{
        CALIBRATION_FREEZE_SCHEMA_VERSION, CalibrationClass, CalibrationParameter,
        CalibrationParameterSource,
    };

    fn manifest() -> CalibrationManifest {
        CalibrationManifest::new(
            "NBack",
            1,
            vec![CalibrationParameter::new(
                "base_threshold",
                "0.5",
                CalibrationClass::APriori,
                CalibrationParameterSource::BenchmarkLocal,
            )],
        )
    }

    fn commitment(manifest: &CalibrationManifest) -> FrozenCalibrationCommitment {
        FrozenCalibrationCommitment {
            schema_version: CALIBRATION_FREEZE_SCHEMA_VERSION.to_string(),
            parameter_manifest_digest: manifest.digest_hex().unwrap(),
            code_subject: "0123456789abcdef".to_string(),
            task_set_id: "nback-holdout-v1".to_string(),
            baseline_or_holdout_id: "human-baseline-v2".to_string(),
            lineage_id: "psych-freeze-001".to_string(),
            freeze_artifact_id: "freeze-receipt-001".to_string(),
            binding_authority: FreezeBindingAuthority::PreScoringEvidenceBound,
        }
    }

    #[test]
    fn frozen_receipt_preserves_full_experiment_identity() {
        let manifest = manifest();
        let commitment = commitment(&manifest);
        let receipt = FrozenCalibrationEvidenceReceipt::new(&manifest, &commitment).unwrap();

        assert_eq!(receipt.commitment, commitment);
        assert_eq!(
            receipt.evaluation.evidence_profile.freeze_status,
            CalibrationFreezeStatus::VerifiedFrozen
        );
        assert!(receipt.validate(&manifest).is_ok());
    }

    #[test]
    fn changing_task_set_changes_commitment_digest() {
        let manifest = manifest();
        let first = commitment(&manifest);
        let mut second = first.clone();
        second.task_set_id = "nback-holdout-v2".to_string();
        assert_ne!(
            frozen_commitment_digest_hex(&first).unwrap(),
            frozen_commitment_digest_hex(&second).unwrap()
        );
    }

    #[test]
    fn changing_holdout_identity_changes_commitment_digest() {
        let manifest = manifest();
        let first = commitment(&manifest);
        let mut second = first.clone();
        second.baseline_or_holdout_id = "different-holdout".to_string();
        assert_ne!(
            frozen_commitment_digest_hex(&first).unwrap(),
            frozen_commitment_digest_hex(&second).unwrap()
        );
    }

    #[test]
    fn changing_code_subject_changes_commitment_digest() {
        let manifest = manifest();
        let first = commitment(&manifest);
        let mut second = first.clone();
        second.code_subject = "fedcba9876543210".to_string();
        assert_ne!(
            frozen_commitment_digest_hex(&first).unwrap(),
            frozen_commitment_digest_hex(&second).unwrap()
        );
    }

    #[test]
    fn tampered_nested_commitment_is_detected() {
        let manifest = manifest();
        let commitment = commitment(&manifest);
        let mut receipt = FrozenCalibrationEvidenceReceipt::new(&manifest, &commitment).unwrap();
        receipt.commitment.lineage_id = "tampered-lineage".to_string();
        assert_eq!(
            receipt.validate(&manifest),
            Err(FrozenCalibrationReceiptError::CommitmentDigestMismatch)
        );
    }

    #[test]
    fn tampered_evaluation_is_detected() {
        let manifest = manifest();
        let commitment = commitment(&manifest);
        let mut receipt = FrozenCalibrationEvidenceReceipt::new(&manifest, &commitment).unwrap();
        receipt.evaluation.benchmark = "OtherBenchmark".to_string();
        assert_eq!(
            receipt.validate(&manifest),
            Err(FrozenCalibrationReceiptError::EvaluationMismatch)
        );
    }

    #[test]
    fn declared_only_commitment_cannot_create_frozen_receipt() {
        let manifest = manifest();
        let mut commitment = commitment(&manifest);
        commitment.binding_authority = FreezeBindingAuthority::DeclaredOnly;
        assert_eq!(
            FrozenCalibrationEvidenceReceipt::new(&manifest, &commitment),
            Err(FrozenCalibrationReceiptError::Contract(
                CalibrationContractError::FreezeTimingNotEstablished
            ))
        );
    }

    #[test]
    fn serialization_round_trip_keeps_full_binding() {
        let manifest = manifest();
        let commitment = commitment(&manifest);
        let receipt = FrozenCalibrationEvidenceReceipt::new(&manifest, &commitment).unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: FrozenCalibrationEvidenceReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
        assert!(decoded.validate(&manifest).is_ok());
    }
}
