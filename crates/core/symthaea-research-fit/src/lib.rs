//! Fit provenance and evaluation-leakage guards for Symthaea research.
//!
//! A leakage-resistant split is not enough if evaluation samples influence normalization,
//! dimensionality reduction, feature selection, representation learning, model fitting,
//! calibration, threshold tuning, or other learned preprocessing. This crate binds each fitted
//! artifact to an already-frozen [`symthaea_research_split::ResearchSplitManifest`] and fails
//! closed when Evaluation data appears in the fit influence set.
//!
//! Applying a frozen artifact to Evaluation data is allowed and recorded separately through
//! [`TransformReceipt`]. Fitting on Evaluation data is not.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use symthaea_research_split::{PartitionRole, ResearchSplitManifest};
use thiserror::Error;

const FIT_SCHEMA: &str = "symthaea-research-fit/v1";
const TRANSFORM_SCHEMA: &str = "symthaea-research-transform/v1";

pub type Result<T> = std::result::Result<T, FitError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FitError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("fit influence set must not be empty")]
    EmptyInfluenceSet,
    #[error("duplicate fit influence sample id: {0}")]
    DuplicateInfluence(String),
    #[error("fit sample is not present in split manifest: {0}")]
    UnknownSample(String),
    #[error("evaluation sample must not influence fitting: {0}")]
    EvaluationLeakage(String),
    #[error("calibration sample is not allowed by TrainingOnly policy: {0}")]
    CalibrationLeakage(String),
    #[error("split manifest digest does not match fit manifest")]
    SplitManifestMismatch,
    #[error("fit influence content digest changed for sample {sample_id}")]
    SampleDigestMismatch { sample_id: String },
    #[error("fit influence role changed for sample {sample_id}: recorded={recorded:?}, actual={actual:?}")]
    SampleRoleMismatch {
        sample_id: String,
        recorded: PartitionRole,
        actual: PartitionRole,
    },
    #[error("fit manifest digest mismatch")]
    FitDigestMismatch,
    #[error("transform receipt digest mismatch")]
    TransformDigestMismatch,
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("split manifest validation failed: {0}")]
    Split(String),
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(FitError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitStage {
    Preprocessing,
    FeatureSelection,
    RepresentationLearning,
    ModelTraining,
    Calibration,
    ThresholdSelection,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitRolePolicy {
    /// Only Training samples may influence the fitted artifact.
    TrainingOnly,
    /// Training and Calibration samples may influence the artifact. Evaluation remains forbidden.
    TrainingAndCalibration,
}

impl FitRolePolicy {
    fn allows(self, role: PartitionRole) -> bool {
        match self {
            Self::TrainingOnly => role == PartitionRole::Training,
            Self::TrainingAndCalibration => {
                matches!(role, PartitionRole::Training | PartitionRole::Calibration)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitInfluence {
    pub sample_id: String,
    pub content_digest: String,
    pub role: PartitionRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FitArtifactManifest {
    pub artifact_id: String,
    pub stage: FitStage,
    pub role_policy: FitRolePolicy,
    pub split_manifest_digest: String,
    pub implementation_digest: String,
    pub hyperparameters_digest: String,
    pub influences: Vec<FitInfluence>,
    pub output_artifact_digest: String,
    pub fitted_at_unix_ms: i64,
    pub manifest_digest: String,
}

#[derive(Deserialize)]
struct FitArtifactManifestRepr {
    artifact_id: String,
    stage: FitStage,
    role_policy: FitRolePolicy,
    split_manifest_digest: String,
    implementation_digest: String,
    hyperparameters_digest: String,
    influences: Vec<FitInfluence>,
    output_artifact_digest: String,
    fitted_at_unix_ms: i64,
    manifest_digest: String,
}

#[derive(Serialize)]
struct FitDigestView<'a> {
    schema: &'static str,
    artifact_id: &'a str,
    stage: FitStage,
    role_policy: FitRolePolicy,
    split_manifest_digest: &'a str,
    implementation_digest: &'a str,
    hyperparameters_digest: &'a str,
    influences: &'a [FitInfluence],
    output_artifact_digest: &'a str,
    fitted_at_unix_ms: i64,
}

impl FitArtifactManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        split: &ResearchSplitManifest,
        artifact_id: impl Into<String>,
        stage: FitStage,
        role_policy: FitRolePolicy,
        implementation_digest: impl Into<String>,
        hyperparameters_digest: impl Into<String>,
        influence_sample_ids: Vec<String>,
        output_artifact_digest: impl Into<String>,
        fitted_at_unix_ms: i64,
    ) -> Result<Self> {
        split
            .verify_digest()
            .map_err(|error| FitError::Split(error.to_string()))?;
        let artifact_id = artifact_id.into();
        let implementation_digest = implementation_digest.into();
        let hyperparameters_digest = hyperparameters_digest.into();
        let output_artifact_digest = output_artifact_digest.into();
        non_empty(&artifact_id, "fit artifact id")?;
        non_empty(&implementation_digest, "implementation digest")?;
        non_empty(&hyperparameters_digest, "hyperparameters digest")?;
        non_empty(&output_artifact_digest, "output artifact digest")?;
        if influence_sample_ids.is_empty() {
            return Err(FitError::EmptyInfluenceSet);
        }

        let mut seen = HashSet::new();
        let mut influences = Vec::with_capacity(influence_sample_ids.len());
        for sample_id in influence_sample_ids {
            non_empty(&sample_id, "fit influence sample id")?;
            if !seen.insert(sample_id.clone()) {
                return Err(FitError::DuplicateInfluence(sample_id));
            }
            let assignment = split
                .assignments
                .iter()
                .find(|assignment| assignment.unit.sample_id == sample_id)
                .ok_or_else(|| FitError::UnknownSample(sample_id.clone()))?;
            if assignment.role == PartitionRole::Evaluation {
                return Err(FitError::EvaluationLeakage(sample_id));
            }
            if !role_policy.allows(assignment.role) {
                return Err(FitError::CalibrationLeakage(sample_id));
            }
            influences.push(FitInfluence {
                sample_id: assignment.unit.sample_id.clone(),
                content_digest: assignment.unit.content_digest.clone(),
                role: assignment.role,
            });
        }

        influences.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
        let mut result = Self {
            artifact_id,
            stage,
            role_policy,
            split_manifest_digest: split.manifest_digest.clone(),
            implementation_digest,
            hyperparameters_digest,
            influences,
            output_artifact_digest,
            fitted_at_unix_ms,
            manifest_digest: String::new(),
        };
        result.manifest_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> FitDigestView<'_> {
        FitDigestView {
            schema: FIT_SCHEMA,
            artifact_id: &self.artifact_id,
            stage: self.stage,
            role_policy: self.role_policy,
            split_manifest_digest: &self.split_manifest_digest,
            implementation_digest: &self.implementation_digest,
            hyperparameters_digest: &self.hyperparameters_digest,
            influences: &self.influences,
            output_artifact_digest: &self.output_artifact_digest,
            fitted_at_unix_ms: self.fitted_at_unix_ms,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| FitError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.manifest_digest {
            return Err(FitError::FitDigestMismatch);
        }
        Ok(())
    }

    /// Revalidate this persisted fit artifact against the exact frozen split manifest.
    ///
    /// Internal digest validity is not sufficient: this check ensures the recorded role/content
    /// facts still match the referenced split and that no Evaluation sample appears in fitting.
    pub fn verify_against_split(&self, split: &ResearchSplitManifest) -> Result<()> {
        self.verify_digest()?;
        split
            .verify_digest()
            .map_err(|error| FitError::Split(error.to_string()))?;
        if self.split_manifest_digest != split.manifest_digest {
            return Err(FitError::SplitManifestMismatch);
        }
        for influence in &self.influences {
            let assignment = split
                .assignments
                .iter()
                .find(|assignment| assignment.unit.sample_id == influence.sample_id)
                .ok_or_else(|| FitError::UnknownSample(influence.sample_id.clone()))?;
            if assignment.unit.content_digest != influence.content_digest {
                return Err(FitError::SampleDigestMismatch {
                    sample_id: influence.sample_id.clone(),
                });
            }
            if assignment.role != influence.role {
                return Err(FitError::SampleRoleMismatch {
                    sample_id: influence.sample_id.clone(),
                    recorded: influence.role,
                    actual: assignment.role,
                });
            }
            if assignment.role == PartitionRole::Evaluation {
                return Err(FitError::EvaluationLeakage(influence.sample_id.clone()));
            }
            if !self.role_policy.allows(assignment.role) {
                return Err(FitError::CalibrationLeakage(influence.sample_id.clone()));
            }
        }
        Ok(())
    }
}

impl TryFrom<FitArtifactManifestRepr> for FitArtifactManifest {
    type Error = FitError;

    fn try_from(value: FitArtifactManifestRepr) -> Result<Self> {
        let result = Self {
            artifact_id: value.artifact_id,
            stage: value.stage,
            role_policy: value.role_policy,
            split_manifest_digest: value.split_manifest_digest,
            implementation_digest: value.implementation_digest,
            hyperparameters_digest: value.hyperparameters_digest,
            influences: value.influences,
            output_artifact_digest: value.output_artifact_digest,
            fitted_at_unix_ms: value.fitted_at_unix_ms,
            manifest_digest: value.manifest_digest,
        };
        non_empty(&result.artifact_id, "fit artifact id")?;
        non_empty(&result.split_manifest_digest, "split manifest digest")?;
        non_empty(&result.implementation_digest, "implementation digest")?;
        non_empty(&result.hyperparameters_digest, "hyperparameters digest")?;
        non_empty(&result.output_artifact_digest, "output artifact digest")?;
        if result.influences.is_empty() {
            return Err(FitError::EmptyInfluenceSet);
        }
        let mut seen = HashSet::new();
        for influence in &result.influences {
            non_empty(&influence.sample_id, "fit influence sample id")?;
            non_empty(&influence.content_digest, "fit influence content digest")?;
            if !seen.insert(influence.sample_id.as_str()) {
                return Err(FitError::DuplicateInfluence(influence.sample_id.clone()));
            }
            if influence.role == PartitionRole::Evaluation {
                return Err(FitError::EvaluationLeakage(influence.sample_id.clone()));
            }
            if !result.role_policy.allows(influence.role) {
                return Err(FitError::CalibrationLeakage(influence.sample_id.clone()));
            }
        }
        result.verify_digest()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for FitArtifactManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = FitArtifactManifestRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransformReceipt {
    pub receipt_id: String,
    pub fit_manifest_digest: String,
    pub split_manifest_digest: String,
    pub sample_id: String,
    pub sample_role: PartitionRole,
    pub input_content_digest: String,
    pub output_content_digest: String,
    pub applied_at_unix_ms: i64,
    pub receipt_digest: String,
}

#[derive(Serialize)]
struct TransformDigestView<'a> {
    schema: &'static str,
    receipt_id: &'a str,
    fit_manifest_digest: &'a str,
    split_manifest_digest: &'a str,
    sample_id: &'a str,
    sample_role: PartitionRole,
    input_content_digest: &'a str,
    output_content_digest: &'a str,
    applied_at_unix_ms: i64,
}

impl TransformReceipt {
    pub fn new(
        receipt_id: impl Into<String>,
        fit: &FitArtifactManifest,
        split: &ResearchSplitManifest,
        sample_id: &str,
        output_content_digest: impl Into<String>,
        applied_at_unix_ms: i64,
    ) -> Result<Self> {
        fit.verify_against_split(split)?;
        non_empty(sample_id, "transform sample id")?;
        let receipt_id = receipt_id.into();
        let output_content_digest = output_content_digest.into();
        non_empty(&receipt_id, "transform receipt id")?;
        non_empty(&output_content_digest, "transform output content digest")?;
        let assignment = split
            .assignments
            .iter()
            .find(|assignment| assignment.unit.sample_id == sample_id)
            .ok_or_else(|| FitError::UnknownSample(sample_id.to_string()))?;

        let mut result = Self {
            receipt_id,
            fit_manifest_digest: fit.manifest_digest.clone(),
            split_manifest_digest: split.manifest_digest.clone(),
            sample_id: assignment.unit.sample_id.clone(),
            sample_role: assignment.role,
            input_content_digest: assignment.unit.content_digest.clone(),
            output_content_digest,
            applied_at_unix_ms,
            receipt_digest: String::new(),
        };
        result.receipt_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> TransformDigestView<'_> {
        TransformDigestView {
            schema: TRANSFORM_SCHEMA,
            receipt_id: &self.receipt_id,
            fit_manifest_digest: &self.fit_manifest_digest,
            split_manifest_digest: &self.split_manifest_digest,
            sample_id: &self.sample_id,
            sample_role: self.sample_role,
            input_content_digest: &self.input_content_digest,
            output_content_digest: &self.output_content_digest,
            applied_at_unix_ms: self.applied_at_unix_ms,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| FitError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.receipt_digest {
            return Err(FitError::TransformDigestMismatch);
        }
        Ok(())
    }
}
