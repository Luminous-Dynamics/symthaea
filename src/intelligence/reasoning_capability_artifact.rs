// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content-bound publication envelope for the reasoning capability matrix.
//!
//! The capability matrix contains derived evidence. Publishing it without a deterministic
//! identity would allow a score, baseline, resource field, or contamination label to change
//! after qualification without detection. This envelope binds the exact serialized matrix.

use super::reasoning_capability_matrix::{
    ReasoningCapabilityMatrix, REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const REASONING_CAPABILITY_ARTIFACT_SCHEMA_VERSION: u32 = 1;
const DIGEST_DOMAIN: &[u8] = b"symthaea:reasoning-capability-matrix-artifact:v1\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningCapabilityArtifact {
    pub artifact_schema_version: u32,
    pub serialization: String,
    pub matrix_digest: String,
    pub matrix: ReasoningCapabilityMatrix,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityArtifactError {
    UnsupportedArtifactSchema(u32),
    UnsupportedMatrixSchema(u32),
    EmptySubjectRevision,
    EmptyMatrix,
    SubjectRevisionMismatch { expected: String, found: String },
    Serialization(String),
    DigestMismatch { expected: String, found: String },
}

impl fmt::Display for CapabilityArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArtifactSchema(version) => {
                write!(f, "unsupported capability artifact schema version {version}")
            }
            Self::UnsupportedMatrixSchema(version) => {
                write!(f, "unsupported capability matrix schema version {version}")
            }
            Self::EmptySubjectRevision => write!(f, "capability matrix subject revision is empty"),
            Self::EmptyMatrix => write!(f, "capability matrix contains no lanes"),
            Self::SubjectRevisionMismatch { expected, found } => write!(
                f,
                "capability lane subject revision mismatch: expected `{expected}`, found `{found}`"
            ),
            Self::Serialization(err) => write!(f, "failed to serialize capability matrix: {err}"),
            Self::DigestMismatch { expected, found } => write!(
                f,
                "capability artifact digest mismatch: expected `{expected}`, found `{found}`"
            ),
        }
    }
}

impl std::error::Error for CapabilityArtifactError {}

impl ReasoningCapabilityArtifact {
    pub fn new(matrix: ReasoningCapabilityMatrix) -> Result<Self, CapabilityArtifactError> {
        validate_matrix_shape(&matrix)?;
        let matrix_digest = matrix_digest(&matrix)?;
        Ok(Self {
            artifact_schema_version: REASONING_CAPABILITY_ARTIFACT_SCHEMA_VERSION,
            serialization: "serde-json-v1".into(),
            matrix_digest,
            matrix,
        })
    }

    /// Recompute the content identity so persisted or transported artifacts fail closed after
    /// any post-qualification mutation.
    pub fn validate(&self) -> Result<(), CapabilityArtifactError> {
        if self.artifact_schema_version != REASONING_CAPABILITY_ARTIFACT_SCHEMA_VERSION {
            return Err(CapabilityArtifactError::UnsupportedArtifactSchema(
                self.artifact_schema_version,
            ));
        }
        validate_matrix_shape(&self.matrix)?;
        let expected = matrix_digest(&self.matrix)?;
        if self.matrix_digest != expected {
            return Err(CapabilityArtifactError::DigestMismatch {
                expected,
                found: self.matrix_digest.clone(),
            });
        }
        Ok(())
    }
}

fn validate_matrix_shape(matrix: &ReasoningCapabilityMatrix) -> Result<(), CapabilityArtifactError> {
    if matrix.schema_version != REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION {
        return Err(CapabilityArtifactError::UnsupportedMatrixSchema(
            matrix.schema_version,
        ));
    }
    if matrix.subject_revision.trim().is_empty() {
        return Err(CapabilityArtifactError::EmptySubjectRevision);
    }
    if matrix.lanes.is_empty() {
        return Err(CapabilityArtifactError::EmptyMatrix);
    }
    for lane in &matrix.lanes {
        if lane.subject_revision != matrix.subject_revision {
            return Err(CapabilityArtifactError::SubjectRevisionMismatch {
                expected: matrix.subject_revision.clone(),
                found: lane.subject_revision.clone(),
            });
        }
    }
    Ok(())
}

fn matrix_digest(matrix: &ReasoningCapabilityMatrix) -> Result<String, CapabilityArtifactError> {
    let encoded = serde_json::to_vec(matrix)
        .map_err(|err| CapabilityArtifactError::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::{
        build_capability_lane, build_capability_matrix, evaluate_episode, AbstentionReason,
        CapabilityLaneDescriptor, ContaminationStatus, EpisodeJudgment, HoldoutPolicy,
        ReasoningDomain, ReasoningEpisode, ReasoningOutcome, ReasoningProblemRef, ResourceBudget,
        ResourceUsage,
    };

    fn matrix() -> ReasoningCapabilityMatrix {
        let episode = match ReasoningEpisode::new(
            "subject-a",
            "config-a",
            ReasoningDomain::Logic,
            ReasoningProblemRef {
                benchmark: "unit".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: "p1".into(),
                problem_hash: "blake3:p1".into(),
            },
            vec![],
            vec![],
            vec![],
            ReasoningOutcome::Asserted {
                value: "true".into(),
                confidence: 0.8,
            },
            ResourceUsage::default(),
        ) {
            Ok(value) => value,
            Err(err) => panic!("episode must validate: {err}"),
        };
        let receipt = match evaluate_episode(
            &episode,
            "eval:p1",
            &EpisodeJudgment {
                exact_correct: Some(true),
                task_score: None,
            },
        ) {
            Ok(value) => value,
            Err(err) => panic!("receipt must validate: {err}"),
        };
        let descriptor = CapabilityLaneDescriptor {
            lane_id: "logic".into(),
            domain: ReasoningDomain::Logic,
            benchmark: "unit".into(),
            benchmark_version: "v1".into(),
            split: "holdout".into(),
            holdout_policy: HoldoutPolicy::FrozenUnseen,
            contamination_policy_id: "ledger:v1".into(),
            contamination_status: ContaminationStatus::Controlled,
            resource_budget: ResourceBudget::default(),
        };
        let lane = match build_capability_lane(descriptor, &[episode], &[receipt], vec![]) {
            Ok(value) => value,
            Err(err) => panic!("lane must validate: {err}"),
        };
        match build_capability_matrix("subject-a", vec![lane]) {
            Ok(value) => value,
            Err(err) => panic!("matrix must validate: {err}"),
        }
    }

    #[test]
    fn artifact_identity_is_deterministic() {
        let left = ReasoningCapabilityArtifact::new(matrix());
        let right = ReasoningCapabilityArtifact::new(matrix());
        assert!(left.is_ok());
        assert_eq!(left, right);
    }

    #[test]
    fn post_qualification_metric_mutation_is_detected() {
        let mut artifact = match ReasoningCapabilityArtifact::new(matrix()) {
            Ok(value) => value,
            Err(err) => panic!("artifact must validate: {err}"),
        };
        artifact.matrix.lanes[0].capability.coverage = 0.0;
        assert!(matches!(
            artifact.validate(),
            Err(CapabilityArtifactError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn post_qualification_contamination_relabel_is_detected() {
        let mut artifact = match ReasoningCapabilityArtifact::new(matrix()) {
            Ok(value) => value,
            Err(err) => panic!("artifact must validate: {err}"),
        };
        artifact.matrix.lanes[0].descriptor.contamination_status = ContaminationStatus::Exposed;
        assert!(matches!(
            artifact.validate(),
            Err(CapabilityArtifactError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn empty_matrix_is_not_publishable() {
        let empty = ReasoningCapabilityMatrix {
            schema_version: REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION,
            subject_revision: "subject-a".into(),
            lanes: vec![],
        };
        assert!(matches!(
            ReasoningCapabilityArtifact::new(empty),
            Err(CapabilityArtifactError::EmptyMatrix)
        ));
    }

    #[test]
    fn abstention_type_remains_available_to_artifact_consumers() {
        let reason = AbstentionReason::Unidentified;
        assert!(matches!(reason, AbstentionReason::Unidentified));
    }
}
