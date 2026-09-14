// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-verifiable calibration provenance and frozen-parameter commitments.
//!
//! Parameter-selection provenance and comparison-target provenance are separate
//! evidence axes. A benchmark can, for example, use post-hoc tuned parameters
//! while comparing against a theoretical target. Neither axis may silently
//! strengthen the other.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CALIBRATION_MANIFEST_SCHEMA_VERSION: &str = "psych-calibration-manifest-v2";
pub const CALIBRATION_FREEZE_SCHEMA_VERSION: &str = "psych-calibration-freeze-v1";

/// How one scientific parameter was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationClass {
    APriori,
    Literature,
    PostHoc,
    Ambiguous,
}

/// What kind of external/reference target the benchmark compares against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonTargetClass {
    HumanEmpirical,
    ExternalEmpirical,
    TheoreticalModel,
    InternalReference,
    NoExternalBaseline,
    Ambiguous,
}

/// Where the effective runtime parameter value comes from.
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
    /// Canonical, human-reviewable text encoding of the effective value.
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
    pub comparison_target: ComparisonTargetClass,
    pub parameters: Vec<CalibrationParameter>,
}

impl CalibrationManifest {
    pub fn new(
        benchmark: impl Into<String>,
        revision: u32,
        comparison_target: ComparisonTargetClass,
        parameters: Vec<CalibrationParameter>,
    ) -> Self {
        Self {
            schema_version: CALIBRATION_MANIFEST_SCHEMA_VERSION.to_string(),
            benchmark: benchmark.into(),
            revision,
            comparison_target,
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

    /// Canonical digest independent of parameter declaration order. Target
    /// provenance is part of the commitment, so reinterpreting the same numbers
    /// against a different reference target changes the digest.
    pub fn digest_hex(&self) -> Result<String, CalibrationContractError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.name.cmp(&b.name));

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.calibration-manifest.v2\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.benchmark.as_bytes());
        hasher.update(&self.revision.to_le_bytes());
        hasher.update(&[comparison_target_tag(self.comparison_target)]);
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
        Ok(self.parameters.iter().map(|parameter| parameter.class).collect())
    }

    /// Conservative claim ceiling implied only by parameter-selection origin.
    pub fn parameter_authority(
        &self,
    ) -> Result<CalibrationParameterAuthority, CalibrationContractError> {
        let classes = self.classes_present()?;

        if classes.contains(&CalibrationClass::Ambiguous) {
            return Ok(CalibrationParameterAuthority::MixedOrAmbiguous);
        }
        if classes.contains(&CalibrationClass::PostHoc) {
            return Ok(CalibrationParameterAuthority::CalibratedReproductionOnly);
        }
        Ok(CalibrationParameterAuthority::APrioriOrLiterature)
    }

    /// Target provenance is descriptive authority, not a numeric strength rank.
    pub const fn target_authority(&self) -> ComparisonTargetAuthority {
        match self.comparison_target {
            ComparisonTargetClass::HumanEmpirical => ComparisonTargetAuthority::HumanEmpiricalTarget,
            ComparisonTargetClass::ExternalEmpirical => {
                ComparisonTargetAuthority::ExternalEmpiricalTarget
            }
            ComparisonTargetClass::TheoreticalModel => {
                ComparisonTargetAuthority::TheoreticalModelOnly
            }
            ComparisonTargetClass::InternalReference => {
                ComparisonTargetAuthority::InternalReferenceOnly
            }
            ComparisonTargetClass::NoExternalBaseline => {
                ComparisonTargetAuthority::NoExternalBaseline
            }
            ComparisonTargetClass::Ambiguous => ComparisonTargetAuthority::Ambiguous,
        }
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
        None => {
            hasher.update(&[0]);
        }
    }
}

const fn calibration_class_tag(class: CalibrationClass) -> u8 {
    match class {
        CalibrationClass::APriori => 1,
        CalibrationClass::Literature => 2,
        CalibrationClass::PostHoc => 3,
        CalibrationClass::Ambiguous => 4,
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

const fn comparison_target_tag(target: ComparisonTargetClass) -> u8 {
    match target {
        ComparisonTargetClass::HumanEmpirical => 1,
        ComparisonTargetClass::ExternalEmpirical => 2,
        ComparisonTargetClass::TheoreticalModel => 3,
        ComparisonTargetClass::InternalReference => 4,
        ComparisonTargetClass::NoExternalBaseline => 5,
        ComparisonTargetClass::Ambiguous => 6,
    }
}

/// Claim ceiling implied by parameter selection. No cross-axis scalar is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationParameterAuthority {
    APrioriOrLiterature,
    CalibratedReproductionOnly,
    MixedOrAmbiguous,
}

/// Provenance category of the benchmark's comparison target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonTargetAuthority {
    HumanEmpiricalTarget,
    ExternalEmpiricalTarget,
    TheoreticalModelOnly,
    InternalReferenceOnly,
    NoExternalBaseline,
    Ambiguous,
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

/// Whether the timing/lineage of the freeze is externally established.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationFreezeStatus {
    Unfrozen,
    DeclaredOnly,
    VerifiedFrozen,
}

/// Three independent evidence axes. They must remain separate downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvidenceProfile {
    pub parameter_authority: CalibrationParameterAuthority,
    pub target_authority: ComparisonTargetAuthority,
    pub freeze_status: CalibrationFreezeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvaluationReceipt {
    pub mode: EvaluationCalibrationMode,
    pub benchmark: String,
    pub runtime_manifest_digest: String,
    pub committed_manifest_digest: Option<String>,
    pub evidence_profile: CalibrationEvidenceProfile,
}

pub fn evaluate_calibration_contract(
    mode: EvaluationCalibrationMode,
    manifest: &CalibrationManifest,
    commitment: Option<&FrozenCalibrationCommitment>,
) -> Result<CalibrationEvaluationReceipt, CalibrationContractError> {
    let runtime_manifest_digest = manifest.digest_hex()?;
    let parameter_authority = manifest.parameter_authority()?;
    let target_authority = manifest.target_authority();

    if mode == EvaluationCalibrationMode::APrioriOnly
        && parameter_authority != CalibrationParameterAuthority::APrioriOrLiterature
    {
        return Err(CalibrationContractError::APrioriModeContainsNonAPrioriParameters);
    }

    let (committed_manifest_digest, freeze_status) = match commitment {
        Some(commitment) => {
            commitment.validate()?;
            if commitment.parameter_manifest_digest != runtime_manifest_digest {
                return Err(CalibrationContractError::FrozenManifestDrift);
            }
            let status = match commitment.binding_authority {
                FreezeBindingAuthority::DeclaredOnly => CalibrationFreezeStatus::DeclaredOnly,
                FreezeBindingAuthority::PreScoringEvidenceBound => {
                    CalibrationFreezeStatus::VerifiedFrozen
                }
            };
            (Some(commitment.parameter_manifest_digest.clone()), status)
        }
        None => (None, CalibrationFreezeStatus::Unfrozen),
    };

    if mode == EvaluationCalibrationMode::FrozenHoldout {
        if commitment.is_none() {
            return Err(CalibrationContractError::MissingFreezeCommitment);
        }
        if freeze_status != CalibrationFreezeStatus::VerifiedFrozen {
            return Err(CalibrationContractError::FreezeTimingNotEstablished);
        }
    }

    Ok(CalibrationEvaluationReceipt {
        mode,
        benchmark: manifest.benchmark.clone(),
        runtime_manifest_digest,
        committed_manifest_digest,
        evidence_profile: CalibrationEvidenceProfile {
            parameter_authority,
            target_authority,
            freeze_status,
        },
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

    fn parameter(name: &str, value: &str, class: CalibrationClass) -> CalibrationParameter {
        CalibrationParameter::new(
            name,
            value,
            class,
            CalibrationParameterSource::BenchmarkLocal,
        )
    }

    fn apriori_manifest(target: ComparisonTargetClass) -> CalibrationManifest {
        CalibrationManifest::new(
            "NBack",
            1,
            target,
            vec![parameter(
                "base_threshold",
                "0.5",
                CalibrationClass::APriori,
            )],
        )
    }

    fn bound_commitment(manifest: &CalibrationManifest) -> FrozenCalibrationCommitment {
        FrozenCalibrationCommitment {
            schema_version: CALIBRATION_FREEZE_SCHEMA_VERSION.to_string(),
            parameter_manifest_digest: manifest.digest_hex().unwrap(),
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
        let first = CalibrationManifest::new(
            "Bench",
            1,
            ComparisonTargetClass::HumanEmpirical,
            vec![a.clone(), b.clone()],
        );
        let second = CalibrationManifest::new(
            "Bench",
            1,
            ComparisonTargetClass::HumanEmpirical,
            vec![b, a],
        );
        assert_eq!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn changing_target_provenance_changes_manifest_digest() {
        let human = apriori_manifest(ComparisonTargetClass::HumanEmpirical);
        let theory = apriori_manifest(ComparisonTargetClass::TheoreticalModel);
        assert_ne!(human.digest_hex().unwrap(), theory.digest_hex().unwrap());
    }

    #[test]
    fn changing_one_parameter_changes_manifest_digest() {
        let first = CalibrationManifest::new(
            "Stroop",
            1,
            ComparisonTargetClass::HumanEmpirical,
            vec![parameter("temperature", "0.25", CalibrationClass::PostHoc)],
        );
        let second = CalibrationManifest::new(
            "Stroop",
            1,
            ComparisonTargetClass::HumanEmpirical,
            vec![parameter("temperature", "0.30", CalibrationClass::PostHoc)],
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn duplicate_parameter_names_fail_closed() {
        let manifest = CalibrationManifest::new(
            "Bench",
            1,
            ComparisonTargetClass::HumanEmpirical,
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
    fn posthoc_and_human_target_remain_separate_axes() {
        let manifest = CalibrationManifest::new(
            "Stroop",
            1,
            ComparisonTargetClass::HumanEmpirical,
            vec![parameter("temperature", "0.25", CalibrationClass::PostHoc)],
        );
        assert_eq!(
            manifest.parameter_authority().unwrap(),
            CalibrationParameterAuthority::CalibratedReproductionOnly
        );
        assert_eq!(
            manifest.target_authority(),
            ComparisonTargetAuthority::HumanEmpiricalTarget
        );
    }

    #[test]
    fn posthoc_and_theoretical_target_remain_separate_axes() {
        let manifest = CalibrationManifest::new(
            "SubstrateTransfer",
            1,
            ComparisonTargetClass::TheoreticalModel,
            vec![parameter("noise_level", "0.010", CalibrationClass::PostHoc)],
        );
        assert_eq!(
            manifest.parameter_authority().unwrap(),
            CalibrationParameterAuthority::CalibratedReproductionOnly
        );
        assert_eq!(
            manifest.target_authority(),
            ComparisonTargetAuthority::TheoreticalModelOnly
        );
    }

    #[test]
    fn frozen_holdout_requires_pre_scoring_binding() {
        let manifest = apriori_manifest(ComparisonTargetClass::HumanEmpirical);
        let mut commitment = bound_commitment(&manifest);
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
    fn frozen_holdout_rejects_parameter_or_target_drift() {
        let manifest = apriori_manifest(ComparisonTargetClass::HumanEmpirical);
        let other_manifest = apriori_manifest(ComparisonTargetClass::TheoreticalModel);
        let commitment = bound_commitment(&other_manifest);
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
    fn frozen_receipt_preserves_three_independent_axes() {
        let manifest = apriori_manifest(ComparisonTargetClass::HumanEmpirical);
        let commitment = bound_commitment(&manifest);
        let receipt = evaluate_calibration_contract(
            EvaluationCalibrationMode::FrozenHoldout,
            &manifest,
            Some(&commitment),
        )
        .unwrap();
        assert_eq!(
            receipt.evidence_profile,
            CalibrationEvidenceProfile {
                parameter_authority: CalibrationParameterAuthority::APrioriOrLiterature,
                target_authority: ComparisonTargetAuthority::HumanEmpiricalTarget,
                freeze_status: CalibrationFreezeStatus::VerifiedFrozen,
            }
        );
    }

    #[test]
    fn apriori_only_mode_rejects_posthoc_even_with_human_target() {
        let manifest = CalibrationManifest::new(
            "ArcFluid",
            1,
            ComparisonTargetClass::HumanEmpirical,
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
    fn missing_parameter_class_fails_deserialization() {
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
    fn missing_target_class_fails_manifest_deserialization() {
        let json = r#"{
            "schema_version":"psych-calibration-manifest-v2",
            "benchmark":"NBack",
            "revision":1,
            "parameters":[{
                "name":"base_threshold",
                "canonical_value":"0.5",
                "class":"a_priori",
                "source":"benchmark_local",
                "citation":null,
                "rationale":null
            }]
        }"#;
        assert!(serde_json::from_str::<CalibrationManifest>(json).is_err());
    }

    #[test]
    fn receipt_serialization_preserves_all_evidence_axes() {
        let manifest = apriori_manifest(ComparisonTargetClass::HumanEmpirical);
        let commitment = bound_commitment(&manifest);
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
