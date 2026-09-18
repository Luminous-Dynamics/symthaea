// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::identity::{GitObjectId, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactRole {
    Dataset,
    Configuration,
    ReferenceResult,
    EnvironmentLock,
    ResultBundle,
    Other(String),
}

/// Immutable identity for any external scientific artifact consumed or emitted
/// by a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableArtifact {
    pub role: ArtifactRole,
    pub label: String,
    pub locator: String,
    pub sha256: Sha256Digest,
}

impl ImmutableArtifact {
    pub fn validate(&self) -> Result<(), ReproductionSpecError> {
        if self.label.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyArtifactLabel);
        }
        if self.locator.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyArtifactLocator);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    Cobaya,
    Desilike,
    Class,
    Camb,
    Other(String),
}

/// Identity for a numerical/inference backend.
///
/// `environment_lock` should identify a complete lock, image, or closure rather
/// than only a human-readable version string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendIdentity {
    pub kind: BackendKind,
    pub release: String,
    pub source_commit: Option<GitObjectId>,
    pub environment_lock: ImmutableArtifact,
}

impl BackendIdentity {
    pub fn validate(&self) -> Result<(), ReproductionSpecError> {
        if self.release.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyBackendRelease);
        }
        self.environment_lock.validate()?;
        if self.environment_lock.role != ArtifactRole::EnvironmentLock {
            return Err(ReproductionSpecError::WrongEnvironmentArtifactRole);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineModel {
    FlatLambdaCdm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonCriterion {
    /// Machine-readable or human-readable statistic name, e.g. `H0.mean` or
    /// `best_fit.minus_2_loglike`.
    pub statistic: String,
    /// Maximum accepted absolute difference from the frozen reference.
    pub max_absolute_delta: Option<f64>,
    /// Maximum accepted fractional difference from the frozen reference.
    pub max_relative_delta: Option<f64>,
}

impl ComparisonCriterion {
    pub fn validate(&self) -> Result<(), ReproductionSpecError> {
        if self.statistic.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyComparisonStatistic);
        }

        for tolerance in [self.max_absolute_delta, self.max_relative_delta]
            .into_iter()
            .flatten()
        {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(ReproductionSpecError::InvalidTolerance);
            }
        }

        if self.max_absolute_delta.is_none() && self.max_relative_delta.is_none() {
            return Err(ReproductionSpecError::MissingValidTolerance);
        }
        Ok(())
    }
}

/// DE-001A cannot license an anomaly claim. Its only authority is reproduction
/// of an already-published frozen target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum De001aClaimPolicy {
    ReproductionOnly,
}

impl De001aClaimPolicy {
    pub const fn allows_observational_anomaly_claim(self) -> bool {
        false
    }
}

/// Frozen contract for DE-001A exact reproduction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct De001aReproductionSpec {
    pub experiment_id: String,
    pub protocol_version: String,
    pub subject_commit: GitObjectId,
    pub model: BaselineModel,
    pub datasets: Vec<ImmutableArtifact>,
    pub backends: Vec<BackendIdentity>,
    pub configuration: ImmutableArtifact,
    pub reference_result: ImmutableArtifact,
    pub criteria: Vec<ComparisonCriterion>,
    pub claim_policy: De001aClaimPolicy,
}

impl De001aReproductionSpec {
    pub fn validate(&self) -> Result<(), ReproductionSpecError> {
        if self.experiment_id.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyExperimentId);
        }
        if self.protocol_version.trim().is_empty() {
            return Err(ReproductionSpecError::EmptyProtocolVersion);
        }
        if self.datasets.is_empty() {
            return Err(ReproductionSpecError::NoDatasets);
        }
        if self.backends.is_empty() {
            return Err(ReproductionSpecError::NoBackends);
        }
        if self.criteria.is_empty() {
            return Err(ReproductionSpecError::NoComparisonCriteria);
        }

        for dataset in &self.datasets {
            dataset.validate()?;
            if dataset.role != ArtifactRole::Dataset {
                return Err(ReproductionSpecError::WrongDatasetArtifactRole);
            }
        }
        for backend in &self.backends {
            backend.validate()?;
        }
        self.configuration.validate()?;
        if self.configuration.role != ArtifactRole::Configuration {
            return Err(ReproductionSpecError::WrongConfigurationArtifactRole);
        }
        self.reference_result.validate()?;
        if self.reference_result.role != ArtifactRole::ReferenceResult {
            return Err(ReproductionSpecError::WrongReferenceArtifactRole);
        }
        for criterion in &self.criteria {
            criterion.validate()?;
        }
        Ok(())
    }
}

/// Classification of a DE-001A execution.
///
/// `Negative` means the frozen reproduction executed but did not satisfy one or
/// more preregistered criteria. It is not evidence against ΛCDM. `Invalid`
/// means the evidence lineage itself is not admissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum De001aResultClass {
    Pass,
    Negative,
    Indeterminate,
    Invalid,
}

impl De001aResultClass {
    pub const fn is_completed_reproduction_outcome(self) -> bool {
        matches!(self, Self::Pass | Self::Negative)
    }
}

/// Receipt for an executed DE-001A reproduction.
///
/// The reported result class is never trusted by itself. Provenance or
/// preregistration violations force the effective class to `Invalid`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct De001aExecutionReceipt {
    pub subject_commit: GitObjectId,
    pub spec_digest: Sha256Digest,
    pub result_bundle: ImmutableArtifact,
    pub reported_result_class: De001aResultClass,
    pub postflight_immutable: bool,
    pub preregistration_intact: bool,
    /// True if thresholds, model choices, data cuts, or analysis logic were
    /// changed after the reproduction result was exposed.
    pub development_after_result_exposure: bool,
}

impl De001aExecutionReceipt {
    pub fn effective_result_class(&self) -> De001aResultClass {
        let result_identity_valid = self.result_bundle.validate().is_ok()
            && self.result_bundle.role == ArtifactRole::ResultBundle;
        if !result_identity_valid
            || !self.postflight_immutable
            || !self.preregistration_intact
            || self.development_after_result_exposure
        {
            De001aResultClass::Invalid
        } else {
            self.reported_result_class
        }
    }

    pub fn is_qualified_reproduction_pass(&self) -> bool {
        self.effective_result_class() == De001aResultClass::Pass
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproductionSpecError {
    EmptyExperimentId,
    EmptyProtocolVersion,
    NoDatasets,
    NoBackends,
    NoComparisonCriteria,
    EmptyArtifactLabel,
    EmptyArtifactLocator,
    EmptyBackendRelease,
    EmptyComparisonStatistic,
    MissingValidTolerance,
    InvalidTolerance,
    WrongDatasetArtifactRole,
    WrongConfigurationArtifactRole,
    WrongReferenceArtifactRole,
    WrongEnvironmentArtifactRole,
}

impl fmt::Display for ReproductionSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid DE-001A reproduction specification: {self:?}")
    }
}

impl std::error::Error for ReproductionSpecError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Digest {
        Sha256Digest::parse(&byte.to_string().repeat(64)).unwrap()
    }

    fn artifact(role: ArtifactRole, label: &str, byte: char) -> ImmutableArtifact {
        ImmutableArtifact {
            role,
            label: label.into(),
            locator: format!("artifact://{label}"),
            sha256: digest(byte),
        }
    }

    fn valid_spec() -> De001aReproductionSpec {
        De001aReproductionSpec {
            experiment_id: "DE-001A".into(),
            protocol_version: "v1".into(),
            subject_commit: GitObjectId::parse(&"a".repeat(40)).unwrap(),
            model: BaselineModel::FlatLambdaCdm,
            datasets: vec![artifact(ArtifactRole::Dataset, "desi-dr2-bao", 'b')],
            backends: vec![BackendIdentity {
                kind: BackendKind::Cobaya,
                release: "frozen-release".into(),
                source_commit: None,
                environment_lock: artifact(ArtifactRole::EnvironmentLock, "environment", 'c'),
            }],
            configuration: artifact(ArtifactRole::Configuration, "configuration", 'd'),
            reference_result: artifact(ArtifactRole::ReferenceResult, "reference", 'e'),
            criteria: vec![ComparisonCriterion {
                statistic: "H0.mean".into(),
                max_absolute_delta: Some(0.1),
                max_relative_delta: None,
            }],
            claim_policy: De001aClaimPolicy::ReproductionOnly,
        }
    }

    fn valid_receipt() -> De001aExecutionReceipt {
        De001aExecutionReceipt {
            subject_commit: GitObjectId::parse(&"a".repeat(40)).unwrap(),
            spec_digest: digest('f'),
            result_bundle: artifact(ArtifactRole::ResultBundle, "result", '1'),
            reported_result_class: De001aResultClass::Pass,
            postflight_immutable: true,
            preregistration_intact: true,
            development_after_result_exposure: false,
        }
    }

    #[test]
    fn valid_reproduction_spec_is_accepted() {
        assert_eq!(valid_spec().validate(), Ok(()));
    }

    #[test]
    fn spec_fails_closed_without_datasets_or_criteria() {
        let mut no_data = valid_spec();
        no_data.datasets.clear();
        assert_eq!(no_data.validate(), Err(ReproductionSpecError::NoDatasets));

        let mut no_criteria = valid_spec();
        no_criteria.criteria.clear();
        assert_eq!(
            no_criteria.validate(),
            Err(ReproductionSpecError::NoComparisonCriteria)
        );
    }

    #[test]
    fn roles_are_not_interchangeable() {
        let mut spec = valid_spec();
        spec.configuration.role = ArtifactRole::Dataset;
        assert_eq!(
            spec.validate(),
            Err(ReproductionSpecError::WrongConfigurationArtifactRole)
        );
    }

    #[test]
    fn malformed_tolerance_is_explicitly_invalid() {
        let mut spec = valid_spec();
        spec.criteria[0].max_absolute_delta = Some(f64::NAN);
        assert_eq!(
            spec.validate(),
            Err(ReproductionSpecError::InvalidTolerance)
        );
    }

    #[test]
    fn absent_tolerance_is_distinct_from_invalid_tolerance() {
        let mut spec = valid_spec();
        spec.criteria[0].max_absolute_delta = None;
        assert_eq!(
            spec.validate(),
            Err(ReproductionSpecError::MissingValidTolerance)
        );
    }

    #[test]
    fn de001a_never_licenses_an_anomaly_claim() {
        assert!(!De001aClaimPolicy::ReproductionOnly.allows_observational_anomaly_claim());
    }

    #[test]
    fn clean_pass_receipt_qualifies_reproduction_only() {
        let receipt = valid_receipt();
        assert_eq!(receipt.effective_result_class(), De001aResultClass::Pass);
        assert!(receipt.is_qualified_reproduction_pass());
    }

    #[test]
    fn broken_postflight_forces_invalid_even_if_reported_pass() {
        let mut receipt = valid_receipt();
        receipt.postflight_immutable = false;
        assert_eq!(
            receipt.effective_result_class(),
            De001aResultClass::Invalid
        );
        assert!(!receipt.is_qualified_reproduction_pass());
    }

    #[test]
    fn post_result_tuning_forces_invalid() {
        let mut receipt = valid_receipt();
        receipt.development_after_result_exposure = true;
        assert_eq!(
            receipt.effective_result_class(),
            De001aResultClass::Invalid
        );
    }

    #[test]
    fn wrong_result_artifact_role_forces_invalid() {
        let mut receipt = valid_receipt();
        receipt.result_bundle.role = ArtifactRole::ReferenceResult;
        assert_eq!(
            receipt.effective_result_class(),
            De001aResultClass::Invalid
        );
    }

    #[test]
    fn negative_reproduction_is_not_an_invalid_lineage() {
        let mut receipt = valid_receipt();
        receipt.reported_result_class = De001aResultClass::Negative;
        assert_eq!(
            receipt.effective_result_class(),
            De001aResultClass::Negative
        );
        assert!(!receipt.is_qualified_reproduction_pass());
        assert!(receipt
            .effective_result_class()
            .is_completed_reproduction_outcome());
    }
}
