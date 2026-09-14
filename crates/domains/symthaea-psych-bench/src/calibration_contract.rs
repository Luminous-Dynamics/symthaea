// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-verifiable calibration provenance and frozen-parameter commitments.
//!
//! A boolean `frozen=true` cannot prove which scientific parameters were
//! frozen, and a parameter digest cannot by itself prove that the freeze
//! happened before outcome observation. This contract keeps those dimensions
//! separate and fails closed when either is missing.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CALIBRATION_MANIFEST_SCHEMA_VERSION: &str = "psych-calibration-manifest-v1";
pub const CALIBRATION_FREEZE_SCHEMA_VERSION: &str = "psych-calibration-freeze-v1";

/// How one scientific parameter was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationClass {
    APriori,
    Literature,
    PostHoc,
    Theoretical,
    Ambiguous,
}

/// Where the effective runtime value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationParameterSource {
    BenchmarkLocal,
    SharedConfig,
    Learned,
    External,
}

/// One scientifically relevant benchmark parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationParameter {
    pub name: String,
    /// Canonical, reviewable text encoding of the effective value.
    pub canonical_value: String,
    pub class: CalibrationClass,
    pub source: CalibrationParameterSource,
    pub citation: Option<String>,
    pub rationale: Option<String>,
}

impl CalibrationParameter {
    pub fn new(
        name: impl Into<String>,
        canonical_value: impl Into<String>,
        class: CalibrationClass,
        source: CalibrationParameterSource,
    ) -> Self {
        Self {
            name: name.into(),
            canonical_value: canonical_value.into(),
            class,
            source,
            citation: None,
            rationale: None,
        }
    }
}

/// Complete calibration surface for one benchmark revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationManifest {
    pub schema_version: String,
    pub benchmark: String,
    pub revision: u32,
    pub parameters: Vec<CalibrationParameter>,
}

impl CalibrationManifest {
    pub fn new(
        benchmark: impl Into<String>,
        revision: u32,
        parameters: Vec<CalibrationParameter>,
    ) -> Self {
        Self {
            schema_version: CALIBRATION_MANIFEST_SCHEMA_VERSION.to_string(),
            benchmark: benchmark.into(),
            revision,
            parameters,
        }
    }

    pub fn validate(&self) -> Result<(), CalibrationContractError> {
        if self.schema_version != CALIBRATION_MANIFEST_SCHEMA_VERSION {
            return Err(CalibrationContractError::UnsupportedManifestSchema);
        }
        if self.benchmark.trim().is_empty() {
            return Err(CalibrationContractError::EmptyBenchmarkIdentity);
        }
        if self.revision == 0 {
            return Err(CalibrationContractError::InvalidManifestRevision);
        }
        if self.parameters.is_empty() {
            return Err(CalibrationContractError::EmptyParameterManifest);
        }

        let mut names = BTreeSet::new();
        for parameter in &self.parameters {
            if parameter.name.trim().is_empty() {
                return Err(CalibrationContractError::EmptyParameterName);
            }
            if parameter.canonical_value.is_empty() {
                return Err(CalibrationContractError::EmptyCanonicalValue);
            }
            if !names.insert(parameter.name.clone()) {
                return Err(CalibrationContractError::DuplicateParameterName);
            }
        }
        Ok(())
    }

    /// Canonical digest independent of parameter declaration order.
    pub fn digest_hex(&self) -> Result<String, CalibrationContractError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.name.cmp(&b.name));

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.calibration-manifest.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.benchmark.as_bytes());
        hasher.update(&self.revision.to_le_bytes());
        hasher.update(&(parameters.len() as u64).to_le_bytes());

        for parameter in parameters {
            hash_field(&mut hasher, parameter.name.as_bytes());
            hash_field(&mut hasher, parameter.canonical_value.as_bytes());
            hasher.update(&[calibration_class_tag(parameter.class)]);
            hasher.update(&[parameter_source_tag(parameter.source)]);
            hash_option(&mut hasher, parameter.citation.as_deref());
            hash_option(&mut hasher, parameter.rationale.as_deref());
        }

        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn classes_present(&self) -> Result<BTreeSet<CalibrationClass>, CalibrationContractError> {
        self.validate()?;
        Ok(self.parameters.iter().map(|p| p.class).collect())
    }

    pub fn parameter_authority(
        &self,
    ) -> Result<CalibrationParameterAuthority, CalibrationContractError> {
        let classes = self.classes_present()?;
        if classes.contains(&CalibrationClass::Ambiguous) {
            return Ok(CalibrationParameterAuthority::MixedOrAmbiguous);
        }
        if classes.contains(&CalibrationClass::Theoretical) {
            return if classes.len() == 1 {
                Ok(CalibrationParameterAuthority::TheoreticalComparisonOnly)
            } else {
                Ok(CalibrationParameterAuthority::MixedOrAmbiguous)
            };
        }
        if classes.contains(&CalibrationClass::PostHoc) {
            return Ok(CalibrationParameterAuthority::CalibratedReproductionOnly);
        }
        Ok(CalibrationParameterAuthority::APrioriOrLiterature)
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_option(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_field(hasher, value.as_bytes());
        }
        None => hasher.update(&[0]),
    }
}

const fn calibration_class_tag(class: CalibrationClass) -> u8 {
    match class {
        CalibrationClass::APriori => 1,
        CalibrationClass::Literature => 2,
        CalibrationClass::PostHoc => 3,
        CalibrationClass::Theoretical => 4,
        CalibrationClass::Ambiguous => 5,
    }
}

const fn parameter_source_tag(source: CalibrationParameterSource) -> u8 {
    match source {
        CalibrationParameterSource::BenchmarkLocal => 1,
        CalibrationParameterSource::SharedConfig => 2,
        CalibrationParameterSource::Learned => 3,
        CalibrationParameterSource::External => 4,
    }
}

/// Claim ceiling implied by parameter origin alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationParameterAuthority {
    APrioriOrLiterature,
    CalibratedReproductionOnly,
    TheoreticalComparisonOnly,
    MixedOrAmbiguous,
}

/// Evaluation regime requested by an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationCalibrationMode {
    Exploratory,
    CalibratedReproduction,
    FrozenHoldout,
    APrioriOnly,
}

/// Whether the freeze timing/lineage has external pre-scoring evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreezeBindingAuthority {
    DeclaredOnly,
    PreScoringEvidenceBound,
}

/// Immutable identity of one frozen calibration experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenCalibrationCommitment {
    pub schema_version: String,
    pub parameter_manifest_digest: String,
    pub code_subject: String,
    pub task_set_id: String,
    pub baseline_or_holdout_id: String,
    pub lineage_id: String,
    pub freeze_artifact_id: String,
    pub binding_authority: FreezeBindingAuthority,
}

impl FrozenCalibrationCommitment {
    pub fn validate(&self) -> Result<(), CalibrationContractError> {
        if self.schema_version != CALIBRATION_FREEZE_SCHEMA_VERSION {
            return Err(CalibrationContractError::UnsupportedFreezeSchema);
        }
        for value in [
            &self.parameter_manifest_digest,
            &self.code_subject,
            &self.task_set_id,
            &self.baseline_or_holdout_id,
            &self.lineage_id,
            &self.freeze_artifact_id,
        ] {
            if value.trim().is_empty() {
                return Err(CalibrationContractError::IncompleteFreezeBinding);
            }
        }
        Ok(())
    }
}

/// Two-dimensional evidence profile. Parameter origin and freeze status are
/// intentionally not collapsed into one scalar authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvidenceProfile {
    pub parameter_authority: CalibrationParameterAuthority,
    pub freeze_status: CalibrationFreezeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationFreezeStatus {
    Unfrozen,
    DeclaredOnly,
    VerifiedFrozen,
}

/// Receipt produced after checking the runtime manifest against a commitment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvaluationReceipt {
    pub mode: EvaluationCalibrationMode,
    pub benchmark: String,
    pub runtime_manifest_digest: String,
    pub committed_manifest_digest: Option<String>,
    pub parameter_authority: CalibrationParameterAuthority,
    pub freeze_status: CalibrationFreezeStatus,
}

pub fn evaluate_calibration_contract(
    mode: EvaluationCalibrationMode,
    manifest: &CalibrationManifest,
    commitment: Option<&FrozenCalibrationCommitment>,
) -> Result<CalibrationEvaluationReceipt, CalibrationContractError> {
    let runtime_manifest_digest = manifest.digest_hex()?;
    let parameter_authority = manifest.parameter_authority()?;

    if mode == EvaluationCalibrationMode::APrioriOnly
        && parameter_authority != CalibrationParameterAuthority::APrioriOrLiterature
    {
        return Err(CalibrationContractError::APrioriModeContainsNonAPrioriParameters);
    }

    let (committed_manifest_digest, freeze_status) = match mode {
        EvaluationCalibrationMode::FrozenHoldout => {
            let commitment = commitment.ok_or(CalibrationContractError::MissingFreezeCommitment)?;
            commitment.validate()?;
            if commitment.parameter_manifest_digest != runtime_manifest_digest {
                return Err(CalibrationContractError::FrozenManifestDrift);
            }
            if commitment.binding_authority != FreezeBindingAuthority::PreScoringEvidenceBound {
                return Err(CalibrationContractError::FreezeTimingNotEstablished);
            }
            (
                Some(commitment.parameter_manifest_digest.clone()),
                CalibrationFreezeStatus::VerifiedFrozen,
            )
        }
        _ => {
            let status = match commitment {
                Some(commitment) => {
                    commitment.validate()?;
                    if commitment.parameter_manifest_digest != runtime_manifest_digest {
                        return Err(CalibrationContractError::FrozenManifestDrift);
                    }
                    match commitment.binding_authority {
                        FreezeBindingAuthority::DeclaredOnly => CalibrationFreezeStatus::DeclaredOnly,
                        FreezeBindingAuthority::PreScoringEvidenceBound => {
                            CalibrationFreezeStatus::VerifiedFrozen
                        }
                    }
                }
                None => CalibrationFreezeStatus::Unfrozen,
            };
            (
                commitment.map(|c| c.parameter_manifest_digest.clone()),
                status,
            )
        }
    };

    Ok(CalibrationEvaluationReceipt {
        mode,
        benchmark: manifest.benchmark.clone(),
        runtime_manifest_digest,
        committed_manifest_digest,
        parameter_authority,
        freeze_status,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationContractError {
    UnsupportedManifestSchema,
    UnsupportedFreezeSchema,
    EmptyBenchmarkIdentity,
    InvalidManifestRevision,
    EmptyParameterManifest,
    EmptyParameterName,
    EmptyCanonicalValue,
    DuplicateParameterName,
    MissingFreezeCommitment,
    IncompleteFreezeBinding,
    FrozenManifestDrift,
    FreezeTimingNotEstablished,
    APrioriModeContainsNonAPrioriParameters,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter(
        name: &str,
        value: &str,
        class: CalibrationClass,
    ) -> CalibrationParameter {
        CalibrationParameter::new(
            name,
            value,
            class,
            CalibrationParameterSource::BenchmarkLocal,
        )
    }

    fn apriori_manifest() -> CalibrationManifest {
        CalibrationManifest::new(
            "NBack",
            1,
            vec![parameter(
                "base_threshold",
                "0.5",
                CalibrationClass::APriori,
            )],
        )
    }

    fn bound_commitment(digest: String) -> FrozenCalibrationCommitment {
        FrozenCalibrationCommitment {
            schema_version: CALIBRATION_FREEZE_SCHEMA_VERSION.to_string(),
            parameter_manifest_digest: digest,
            code_subject: "deadbeef".to_string(),
            task_set_id: "holdout-v1".to_string(),
            baseline_or_holdout_id: "baseline-v1".to_string(),
            lineage_id: "lineage-a".to_string(),
            freeze_artifact_id: "freeze-receipt-a".to_string(),
            binding_authority: FreezeBindingAuthority::PreScoringEvidenceBound,
        }
    }

    #[test]
    fn manifest_digest_is_independent_of_parameter_order() {
        let a = parameter("alpha", "0.1", CalibrationClass::APriori);
        let b = parameter("beta", "0.2", CalibrationClass::Literature);
        let first = CalibrationManifest::new("Bench", 1, vec![a.clone(), b.clone()]);
        let second = CalibrationManifest::new("Bench", 1, vec![b, a]);
        assert_eq!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn changing_one_parameter_changes_manifest_digest() {
        let first = CalibrationManifest::new(
            "Stroop",
            1,
            vec![parameter("temperature", "0.25", CalibrationClass::PostHoc)],
        );
        let second = CalibrationManifest::new(
            "Stroop",
            1,
            vec![parameter("temperature", "0.30", CalibrationClass::PostHoc)],
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn duplicate_parameter_names_fail_closed() {
        let manifest = CalibrationManifest::new(
            "Bench",
            1,
            vec![
                parameter("threshold", "0.1", CalibrationClass::APriori),
                parameter("threshold", "0.2", CalibrationClass::PostHoc),
            ],
        );
        assert_eq!(
            manifest.digest_hex(),
            Err(CalibrationContractError::DuplicateParameterName)
        );
    }

    #[test]
    fn posthoc_parameter_caps_authority_at_calibrated_reproduction() {
        let manifest = CalibrationManifest::new(
            "Stroop",
            1,
            vec![
                parameter("temperature", "0.25", CalibrationClass::PostHoc),
                parameter("structure", "fixed", CalibrationClass::APriori),
            ],
        );
        assert_eq!(
            manifest.parameter_authority().unwrap(),
            CalibrationParameterAuthority::CalibratedReproductionOnly
        );
    }

    #[test]
    fn theoretical_only_does_not_become_human_validation() {
        let manifest = CalibrationManifest::new(
            "SubstrateTransfer",
            1,
            vec![parameter(
                "comparison_target",
                "iit-gwt-derived",
                CalibrationClass::Theoretical,
            )],
        );
        assert_eq!(
            manifest.parameter_authority().unwrap(),
            CalibrationParameterAuthority::TheoreticalComparisonOnly
        );
    }

    #[test]
    fn theoretical_mixed_with_empirical_origin_remains_noncollapsed() {
        let manifest = CalibrationManifest::new(
            "MixedBench",
            1,
            vec![
                parameter("a", "1", CalibrationClass::Theoretical),
                parameter("b", "2", CalibrationClass::APriori),
            ],
        );
        assert_eq!(
            manifest.parameter_authority().unwrap(),
            CalibrationParameterAuthority::MixedOrAmbiguous
        );
    }

    #[test]
    fn frozen_holdout_requires_pre_scoring_binding() {
        let manifest = apriori_manifest();
        let digest = manifest.digest_hex().unwrap();
        let mut commitment = bound_commitment(digest);
        commitment.binding_authority = FreezeBindingAuthority::DeclaredOnly;
        assert_eq!(
            evaluate_calibration_contract(
                EvaluationCalibrationMode::FrozenHoldout,
                &manifest,
                Some(&commitment),
            ),
            Err(CalibrationContractError::FreezeTimingNotEstablished)
        );
    }

    #[test]
    fn frozen_holdout_rejects_parameter_drift() {
        let manifest = apriori_manifest();
        let commitment = bound_commitment("wrong-digest".to_string());
        assert_eq!(
            evaluate_calibration_contract(
                EvaluationCalibrationMode::FrozenHoldout,
                &manifest,
                Some(&commitment),
            ),
            Err(CalibrationContractError::FrozenManifestDrift)
        );
    }

    #[test]
    fn frozen_holdout_preserves_two_dimensional_evidence_profile() {
        let manifest = apriori_manifest();
        let commitment = bound_commitment(manifest.digest_hex().unwrap());
        let receipt = evaluate_calibration_contract(
            EvaluationCalibrationMode::FrozenHoldout,
            &manifest,
            Some(&commitment),
        )
        .unwrap();
        assert_eq!(
            receipt.parameter_authority,
            CalibrationParameterAuthority::APrioriOrLiterature
        );
        assert_eq!(receipt.freeze_status, CalibrationFreezeStatus::VerifiedFrozen);
    }

    #[test]
    fn apriori_only_mode_rejects_posthoc_parameters() {
        let manifest = CalibrationManifest::new(
            "ArcFluid",
            1,
            vec![parameter(
                "noise_weight",
                "0.008",
                CalibrationClass::PostHoc,
            )],
        );
        assert_eq!(
            evaluate_calibration_contract(EvaluationCalibrationMode::APrioriOnly, &manifest, None),
            Err(CalibrationContractError::APrioriModeContainsNonAPrioriParameters)
        );
    }

    #[test]
    fn missing_calibration_class_fails_deserialization() {
        let json = r#"{
            "name":"temperature",
            "canonical_value":"0.25",
            "source":"benchmark_local",
            "citation":null,
            "rationale":null
        }"#;
        assert!(serde_json::from_str::<CalibrationParameter>(json).is_err());
    }

    #[test]
    fn receipt_serialization_preserves_freeze_and_origin_axes() {
        let manifest = apriori_manifest();
        let commitment = bound_commitment(manifest.digest_hex().unwrap());
        let receipt = evaluate_calibration_contract(
            EvaluationCalibrationMode::FrozenHoldout,
            &manifest,
            Some(&commitment),
        )
        .unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: CalibrationEvaluationReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
    }
}
