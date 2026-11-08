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
    Other(String),
}

/// Immutable identity for any external scientific artifact consumed by a run.
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
        let abs_ok = self
            .max_absolute_delta
            .is_some_and(|value| value.is_finite() && value >= 0.0);
        let rel_ok = self
            .max_relative_delta
            .is_some_and(|value| value.is_finite() && value >= 0.0);
        if !abs_ok && !rel_ok {
            return Err(ReproductionSpecError::MissingValidTolerance);
        }
        if self
            .max_absolute_delta
            .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self
                .max_relative_delta
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(ReproductionSpecError::InvalidTolerance);
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
    fn malformed_tolerance_is_rejected() {
        let mut spec = valid_spec();
        spec.criteria[0].max_absolute_delta = Some(f64::NAN);
        assert_eq!(
            spec.validate(),
            Err(ReproductionSpecError::MissingValidTolerance)
        );
    }

    #[test]
    fn de001a_never_licenses_an_anomaly_claim() {
        assert!(!De001aClaimPolicy::ReproductionOnly.allows_observational_anomaly_claim());
    }
}
