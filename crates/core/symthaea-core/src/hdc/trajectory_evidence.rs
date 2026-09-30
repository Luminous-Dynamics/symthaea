//! Fail-closed dependency gate from operator evidence into trajectory evidence.
//!
//! This is intentionally feature-neutral. It does not execute HDC kernels and
//! it does not assert task quality, latency, memory, or energy. It only answers
//! whether the numerical/semantic operator preconditions required by a declared
//! trajectory transition have been satisfied by a specific evidence artifact.

use serde::{Deserialize, Serialize};

use super::resource_evidence::{qualify_resource, ResourceEvidenceError, ResourceEvidenceRecord, RESOURCE_EVIDENCE_SCHEMA_VERSION};
use super::operator_evidence_contract::{
    OperatorEvidenceArtifact, OperatorEvidenceRecord, OPERATOR_EVIDENCE_SCHEMA_VERSION,
    QUALIFIED_STATUS, CONTINUOUS_F32_REPRESENTATION,
};

pub const TRAJECTORY_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredOperator {
    pub resolution: usize,
    pub operation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEvidenceDependency {
    pub schema_version: u32,
    pub representation: String,
    pub artifact_sha256: String,
    pub required_operators: Vec<RequiredOperator>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryMetrics {
    pub terminal_state_error: Option<f32>,
    pub mean_state_error: Option<f32>,
    pub terminal_tau_error: Option<f32>,
    pub mean_tau_error: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceEvidenceDependency {
    pub schema_version: u32,
    pub resolution: usize,
    pub representation: String,
    pub provenance_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryEvidenceRecord {
    pub schema_version: u32,
    pub source_resolution: usize,
    pub target_resolution: usize,
    pub operator_dependency: OperatorEvidenceDependency,
    pub resource_dependency: ResourceEvidenceDependency,
    pub qualification_disposition: String,
    pub metrics: TrajectoryMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryQualificationError {
    UnsupportedOperatorSchema(u32),
    RepresentationMismatch {
        expected: String,
        observed: String,
    },
    ArtifactDigestMismatch {
        expected: String,
        observed: String,
    },
    UnsupportedDigestAlgorithm(String),
    MissingOperator {
        resolution: usize,
        operation: String,
    },
    DuplicateOperator {
        resolution: usize,
        operation: String,
    },
    FailedOperator {
        resolution: usize,
        operation: String,
    },
    Resource(ResourceEvidenceError),
    ResourceDependencyMismatch,
}

impl std::fmt::Display for TrajectoryQualificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOperatorSchema(v) => write!(f, "unsupported operator evidence schema: {v}"),
            Self::RepresentationMismatch { expected, observed } => {
                write!(f, "operator representation mismatch: expected {expected}, observed {observed}")
            }
            Self::ArtifactDigestMismatch { expected, observed } => {
                write!(f, "operator artifact digest mismatch: expected {expected}, observed {observed}")
            }
            Self::UnsupportedDigestAlgorithm(algorithm) => {
                write!(f, "unsupported operator evidence digest algorithm: {algorithm}")
            }
            Self::MissingOperator { resolution, operation } => {
                write!(f, "required operator evidence missing: {resolution}/{operation}")
            }
            Self::DuplicateOperator { resolution, operation } => {
                write!(f, "required operator evidence duplicated: {resolution}/{operation}")
            }
            Self::FailedOperator { resolution, operation } => {
                write!(f, "required operator evidence failed: {resolution}/{operation}")
            }
            Self::Resource(error) => write!(f, "resource evidence rejected: {error}"),
            Self::ResourceDependencyMismatch => {
                write!(f, "trajectory resource dependency does not match verified resource evidence")
            }
        }
    }
}

impl std::error::Error for TrajectoryQualificationError {}

pub fn qualify_transition(
    artifact: &OperatorEvidenceArtifact,
    dependency: &OperatorEvidenceDependency,
    resource: &ResourceEvidenceRecord,
    resource_dependency: &ResourceEvidenceDependency,
) -> Result<(), TrajectoryQualificationError> {
    if dependency.schema_version != OPERATOR_EVIDENCE_SCHEMA_VERSION {
        return Err(TrajectoryQualificationError::UnsupportedOperatorSchema(
            dependency.schema_version,
        ));
    }
    if artifact.schema_version != OPERATOR_EVIDENCE_SCHEMA_VERSION {
        return Err(TrajectoryQualificationError::UnsupportedOperatorSchema(
            artifact.schema_version,
        ));
    }
    if dependency.representation != CONTINUOUS_F32_REPRESENTATION
        || artifact.representation != CONTINUOUS_F32_REPRESENTATION
    {
        return Err(TrajectoryQualificationError::RepresentationMismatch {
            expected: CONTINUOUS_F32_REPRESENTATION.to_owned(),
            observed: if artifact.representation != CONTINUOUS_F32_REPRESENTATION {
                artifact.representation.clone()
            } else {
                dependency.representation.clone()
            },
        });
    }

    if resource_dependency.schema_version != RESOURCE_EVIDENCE_SCHEMA_VERSION
        || resource_dependency.resolution != resource.workload.resolution
        || resource_dependency.representation != resource.workload.representation
        || resource_dependency.provenance_id != resource.provenance_id
    {
        return Err(TrajectoryQualificationError::ResourceDependencyMismatch);
    }
    qualify_resource(
        resource,
        resource_dependency.resolution,
        &resource_dependency.representation,
    )
    .map_err(TrajectoryQualificationError::Resource)?;

    if artifact.digest.algorithm != "sha256" {
        return Err(TrajectoryQualificationError::UnsupportedDigestAlgorithm(
            artifact.digest.algorithm.clone(),
        ));
    }

    let observed_digest = &artifact.digest.value;
    if !observed_digest.eq_ignore_ascii_case(&dependency.artifact_sha256) {
        return Err(TrajectoryQualificationError::ArtifactDigestMismatch {
            expected: dependency.artifact_sha256.clone(),
            observed: observed_digest.clone(),
        });
    }

    for required in &dependency.required_operators {
        let matches: Vec<&OperatorEvidenceRecord> = artifact
            .records
            .iter()
            .filter(|record| {
                record.resolution == required.resolution
                    && record.operation == required.operation
            })
            .collect();

        match matches.as_slice() {
            [] => {
                return Err(TrajectoryQualificationError::MissingOperator {
                    resolution: required.resolution,
                    operation: required.operation.clone(),
                });
            }
            [_first, _second, ..] => {
                return Err(TrajectoryQualificationError::DuplicateOperator {
                    resolution: required.resolution,
                    operation: required.operation.clone(),
                });
            }
            [record] if record.schema_version != OPERATOR_EVIDENCE_SCHEMA_VERSION
                || record.representation != CONTINUOUS_F32_REPRESENTATION => {
                return Err(TrajectoryQualificationError::FailedOperator {
                    resolution: required.resolution,
                    operation: required.operation.clone(),
                });
            }
            [record] if record.qualification_status != QUALIFIED_STATUS => {
                return Err(TrajectoryQualificationError::FailedOperator {
                    resolution: required.resolution,
                    operation: required.operation.clone(),
                });
            }
            [_record] => {}
        }
    }

    Ok(())
}

pub fn transition_record(
    source_resolution: usize,
    target_resolution: usize,
    dependency: OperatorEvidenceDependency,
    resource_dependency: ResourceEvidenceDependency,
    metrics: TrajectoryMetrics,
) -> TrajectoryEvidenceRecord {
    TrajectoryEvidenceRecord {
        schema_version: TRAJECTORY_EVIDENCE_SCHEMA_VERSION,
        source_resolution,
        target_resolution,
        operator_dependency: dependency,
        resource_dependency,
        qualification_disposition: "qualified".to_owned(),
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::operator_evidence_contract::{
        sha256_hex, OperatorEvidenceDigest,
    };

    fn artifact() -> OperatorEvidenceArtifact {
        let records = [16_384usize, 32_768, 65_536, 131_072, 262_144]
            .into_iter()
            .flat_map(|resolution| {
                ["dot", "bind", "bundle", "norm", "similarity"].into_iter().map(
                    move |operation| OperatorEvidenceRecord {
                        schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
                        representation: CONTINUOUS_F32_REPRESENTATION.to_owned(),
                        resolution,
                        operation: operation.to_owned(),
                        seed_a: 42,
                        seed_b: 43,
                        scalar_reference: 0.0,
                        simd_result: 0.0,
                        abs_error: 0.0,
                        relative_error: 0.0,
                        max_abs_error: 0.0,
                        max_relative_error: 0.0,
                        tolerance: 1e-4,
                        logical_bytes: 1,
                        qualification_status: QUALIFIED_STATUS.to_owned(),
                    },
                )
            })
            .collect();

        OperatorEvidenceArtifact {
            schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
            representation: CONTINUOUS_F32_REPRESENTATION.to_owned(),
            digest: OperatorEvidenceDigest {
                algorithm: "sha256".to_owned(),
                value: "fixture-digest".to_owned(),
            },
            records,
        }
    }

    fn dependency(artifact: &OperatorEvidenceArtifact) -> OperatorEvidenceDependency {
        OperatorEvidenceDependency {
            schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
            representation: CONTINUOUS_F32_REPRESENTATION.to_owned(),
            artifact_sha256: artifact.digest.value.clone(),
            required_operators: vec![
                RequiredOperator { resolution: 65_536, operation: "dot".to_owned() },
                RequiredOperator { resolution: 131_072, operation: "bind".to_owned() },
            ],
        }
    }

    #[test]
    fn qualified_operator_matrix_unlocks_transition() {
        let artifact = artifact();
        qualify_transition(&artifact, &dependency(&artifact)).expect("transition should qualify");
    }

    #[test]
    fn missing_operator_fails_closed() {
        let mut artifact = artifact();
        artifact.records.retain(|r| !(r.resolution == 131_072 && r.operation == "bind"));
        let error = qualify_transition(&artifact, &dependency(&artifact)).expect_err("missing must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::MissingOperator { resolution: 131_072, .. }
        ));
    }

    #[test]
    fn failed_operator_fails_closed() {
        let mut artifact = artifact();
        artifact.records.iter_mut().find(|r| r.resolution == 65_536 && r.operation == "dot").unwrap().qualification_status = "failed".to_owned();
        let error = qualify_transition(&artifact, &dependency(&artifact)).expect_err("failed must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::FailedOperator { resolution: 65_536, .. }
        ));
    }

    #[test]
    fn duplicate_operator_fails_closed() {
        let mut artifact = artifact();
        artifact.records.push(artifact.records[0].clone());
        let error = qualify_transition(&artifact, &dependency(&artifact)).expect_err("duplicate must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::DuplicateOperator { resolution: 65_536, .. }
        ));
    }

    #[test]
    fn representation_mismatch_fails_closed() {
        let mut artifact = artifact();
        artifact.representation = "binary".to_owned();
        let error = qualify_transition(&artifact, &dependency(&artifact)).expect_err("representation must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::RepresentationMismatch { .. }
        ));
    }

    #[test]
    fn digest_mismatch_fails_closed() {
        let artifact = artifact();
        let mut dependency = dependency(&artifact);
        dependency.artifact_sha256 = "different".to_owned();
        let error = qualify_transition(&artifact, &dependency).expect_err("digest must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::ArtifactDigestMismatch { .. }
        ));
    }

    #[test]
    fn unsupported_digest_algorithm_fails_closed() {
        let mut artifact = artifact();
        artifact.digest.algorithm = "sha1".to_owned();
        let error = qualify_transition(&artifact, &dependency(&artifact))
            .expect_err("unsupported algorithm must reject");
        assert!(matches!(
            error,
            TrajectoryQualificationError::UnsupportedDigestAlgorithm(_)
        ));
    }

    #[test]
    fn transition_record_preserves_dependency_and_metrics() {
        let artifact = artifact();
        let dependency = dependency(&artifact);
        let record = transition_record(
            65_536,
            131_072,
            dependency.clone(),
            TrajectoryMetrics {
                terminal_state_error: Some(0.01),
                mean_state_error: Some(0.005),
                terminal_tau_error: Some(0.001),
                mean_tau_error: Some(0.0005),
            },
        );
        assert_eq!(record.source_resolution, 65_536);
        assert_eq!(record.target_resolution, 131_072);
        assert_eq!(record.operator_dependency, dependency);
        assert_eq!(record.qualification_disposition, "qualified");
    }

    #[test]
    fn real_digest_can_be_carried_as_dependency() {
        let bytes = serde_json::to_vec(&artifact()).expect("fixture serializes");
        let digest = sha256_hex(&bytes);
        let mut artifact = artifact();
        artifact.digest.value = digest.clone();
        let dependency = OperatorEvidenceDependency {
            schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
            representation: CONTINUOUS_F32_REPRESENTATION.to_owned(),
            artifact_sha256: digest,
            required_operators: vec![RequiredOperator {
                resolution: 16_384,
                operation: "dot".to_owned(),
            }],
        };
        qualify_transition(&artifact, &dependency).expect("digest reference qualifies");
    }
}
