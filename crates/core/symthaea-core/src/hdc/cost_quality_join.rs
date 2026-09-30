//! Exact join manifest for HDC cost-quality analysis.
//!
//! This module does not recompute or rank evidence. It binds already-produced
//! evidence artifacts to one explicit experimental identity so downstream
//! analysis cannot silently mix tasks, resolutions, representations, or model
//! revisions.
//!
//! A join is intentionally a manifest of references rather than a copied set
//! of measurements. Consumers resolve the referenced artifacts by digest and
//! perform analysis over their original measurements. This prevents duplicated
//! numeric fields from drifting apart.

use serde::{Deserialize, Serialize};

use super::evidence_identity::{derive_experiment_key, verify_join_identity, verify_trajectory_target};
use super::performance_evidence::PerformanceEvidenceRecord;
use super::resource_evidence::ResourceEvidenceRecord;
use super::task_quality_evidence::TaskQualityEvidenceRecord;
use super::trajectory_evidence::TrajectoryEvidenceRecord;

pub const COST_QUALITY_JOIN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinIdentity {
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub split: String,
    pub protocol: String,
    pub resolution: usize,
    pub representation: String,
    pub model_revision: String,
    pub workload: String,
    pub benchmark: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub kind: String,
    pub schema_version: u32,
    pub artifact_digest: String,
    pub artifact_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostQualityJoinRecord {
    pub schema_version: u32,
    pub identity: JoinIdentity,
    pub trajectory: EvidenceReference,
    pub resource: EvidenceReference,
    pub performance: EvidenceReference,
    pub task_quality: EvidenceReference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostQualityJoinError {
    UnsupportedSchema(u32),
    EmptyIdentity(&'static str),
    InvalidResolution,
    InvalidReference {
        field: &'static str,
        reason: &'static str,
    },
    Identity(String),
    UnexpectedReferenceKind {
        field: &'static str,
        expected: &'static str,
        observed: String,
    },
}

impl std::fmt::Display for CostQualityJoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => {
                write!(f, "unsupported cost-quality join schema: {v}")
            }
            Self::EmptyIdentity(field) => {
                write!(f, "cost-quality join identity field is empty: {field}")
            }
            Self::InvalidResolution => {
                write!(f, "cost-quality join resolution must be non-zero")
            }
            Self::InvalidReference { field, reason } => {
                write!(f, "invalid {field} evidence reference: {reason}")
            }
            Self::Identity(error) => write!(f, "evidence identity reconciliation failed: {error}"),
            Self::UnexpectedReferenceKind {
                field,
                expected,
                observed,
            } => write!(
                f,
                "{field} evidence reference kind mismatch: expected {expected}, observed {observed}"
            ),
        }
    }
}

impl std::error::Error for CostQualityJoinError {}

impl CostQualityJoinRecord {
    /// Validate the manifest against the actual upstream evidence identities.
    ///
    /// This closes the gap between a structurally valid manifest and a
    /// semantically valid join: the duplicated JoinIdentity is checked against
    /// identities derived from the referenced source records.
    pub fn validate_against_evidence(
        &self,
        task_quality: &TaskQualityEvidenceRecord,
        performance: &PerformanceEvidenceRecord,
        resource: &ResourceEvidenceRecord,
        trajectory: &TrajectoryEvidenceRecord,
    ) -> Result<(), CostQualityJoinError> {
        self.validate()?;
        let key = derive_experiment_key(task_quality, performance, resource)
            .map_err(|error| CostQualityJoinError::Identity(error.to_string()))?;
        verify_join_identity(&key, &self.identity)
            .map_err(|error| CostQualityJoinError::Identity(error.to_string()))?;
        verify_trajectory_target(&key, trajectory)
            .map_err(|error| CostQualityJoinError::Identity(error.to_string()))?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), CostQualityJoinError> {
        if self.schema_version != COST_QUALITY_JOIN_SCHEMA_VERSION {
            return Err(CostQualityJoinError::UnsupportedSchema(self.schema_version));
        }

        for (value, name) in [
            (&self.identity.task, "task"),
            (&self.identity.scenario_set, "scenario_set"),
            (&self.identity.scenario_revision, "scenario_revision"),
            (&self.identity.split, "split"),
            (&self.identity.protocol, "protocol"),
            (&self.identity.representation, "representation"),
            (&self.identity.model_revision, "model_revision"),
            (&self.identity.workload, "workload"),
            (&self.identity.benchmark, "benchmark"),
        ] {
            if value.is_empty() {
                return Err(CostQualityJoinError::EmptyIdentity(name));
            }
        }

        if self.identity.resolution == 0 {
            return Err(CostQualityJoinError::InvalidResolution);
        }

        validate_reference(&self.trajectory, "trajectory", "trajectory")?;
        validate_reference(&self.resource, "resource", "resource")?;
        validate_reference(&self.performance, "performance", "performance")?;
        validate_reference(&self.task_quality, "task_quality", "task_quality")?;

        Ok(())
    }
}

fn validate_reference(
    reference: &EvidenceReference,
    field: &'static str,
    expected_kind: &'static str,
) -> Result<(), CostQualityJoinError> {
    if reference.kind != expected_kind {
        return Err(CostQualityJoinError::UnexpectedReferenceKind {
            field,
            expected: expected_kind,
            observed: reference.kind.clone(),
        });
    }
    if reference.schema_version == 0 {
        return Err(CostQualityJoinError::InvalidReference {
            field,
            reason: "schema_version must be non-zero",
        });
    }
    if reference.artifact_digest.is_empty() {
        return Err(CostQualityJoinError::InvalidReference {
            field,
            reason: "artifact_digest is empty",
        });
    }
    if !reference.artifact_digest.starts_with("sha256:") {
        return Err(CostQualityJoinError::InvalidReference {
            field,
            reason: "artifact_digest must use sha256: prefix",
        });
    }
    if reference.artifact_id.is_empty() {
        return Err(CostQualityJoinError::InvalidReference {
            field,
            reason: "artifact_id is empty",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::operator_evidence_contract::OPERATOR_EVIDENCE_SCHEMA_VERSION;
    use crate::hdc::performance_evidence::{BenchmarkIdentity, ExecutionProvenance as PerformanceExecutionProvenance, Measurement, PerformanceEvidenceRecord, PERFORMANCE_EVIDENCE_SCHEMA_VERSION};
    use crate::hdc::resource_evidence::{ResourceBudget, ResourceEvidenceRecord, ResourceWorkload, RESOURCE_EVIDENCE_SCHEMA_VERSION, RESOURCE_QUALIFIED_STATUS};
    use crate::hdc::task_quality_evidence::{ExecutionProvenance as TaskExecutionProvenance, MetricDirection, QualityMeasurement, QualityMetric, TaskIdentity, TaskQualityEvidenceRecord, TASK_QUALITY_EVIDENCE_SCHEMA_VERSION};
    use crate::hdc::trajectory_evidence::{OperatorEvidenceDependency, ResourceEvidenceDependency, TrajectoryEvidenceRecord, TrajectoryMetrics, TRAJECTORY_EVIDENCE_SCHEMA_VERSION};

    fn reference(kind: &str) -> EvidenceReference {
        EvidenceReference {
            kind: kind.to_owned(),
            schema_version: 1,
            artifact_digest: "sha256:fixture".to_owned(),
            artifact_id: format!("{kind}-fixture-v1"),
        }
    }

    fn record() -> CostQualityJoinRecord {
        CostQualityJoinRecord {
            schema_version: COST_QUALITY_JOIN_SCHEMA_VERSION,
            identity: JoinIdentity {
                task: "hdc_retrieval".to_owned(),
                scenario_set: "retrieval-fixture-v1".to_owned(),
                scenario_revision: "sha256:scenario".to_owned(),
                split: "held-out".to_owned(),
                protocol: "top-k-10".to_owned(),
                resolution: 131_072,
                representation: "continuous_f32".to_owned(),
                model_revision: "model-fixture-v1".to_owned(),
                workload: "dot".to_owned(),
                benchmark: "simd_continuous".to_owned(),
            },
            trajectory: reference("trajectory"),
            resource: reference("resource"),
            performance: reference("performance"),
            task_quality: reference("task_quality"),
        }
    }

    #[test]
    fn complete_join_validates() {
        record().validate().expect("fixture should validate");
    }

    #[test]
    fn missing_reference_digest_fails_closed() {
        let mut r = record();
        r.performance.artifact_digest.clear();
        assert!(matches!(
            r.validate(),
            Err(CostQualityJoinError::InvalidReference {
                field: "performance",
                reason: "artifact_digest is empty"
            })
        ));
    }

    #[test]
    fn non_sha256_reference_fails_closed() {
        let mut r = record();
        r.resource.artifact_digest = "blake3:fixture".to_owned();
        assert!(matches!(
            r.validate(),
            Err(CostQualityJoinError::InvalidReference {
                field: "resource",
                reason: "artifact_digest must use sha256: prefix"
            })
        ));
    }

    #[test]
    fn wrong_reference_kind_fails_closed() {
        let mut r = record();
        r.task_quality.kind = "performance".to_owned();
        assert!(matches!(
            r.validate(),
            Err(CostQualityJoinError::UnexpectedReferenceKind {
                field: "task_quality",
                expected: "task_quality",
                ..
            })
        ));
    }

    #[test]
    fn empty_identity_fails_closed() {
        let mut r = record();
        r.identity.model_revision.clear();
        assert!(matches!(
            r.validate(),
            Err(CostQualityJoinError::EmptyIdentity("model_revision"))
        ));
    }


    fn upstream_task() -> TaskQualityEvidenceRecord {
        TaskQualityEvidenceRecord {
            schema_version: TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
            identity: TaskIdentity {
                task: "hdc_retrieval".into(),
                scenario_set: "retrieval-fixture-v1".into(),
                scenario_revision: "sha256:scenario".into(),
                split: "held-out".into(),
                protocol: "top-k-10".into(),
                resolution: 131_072,
                representation: "continuous_f32".into(),
                model_revision: "model-fixture-v1".into(),
                seed: 42,
            },
            metric: QualityMetric {
                name: "recall".into(),
                direction: MetricDirection::HigherIsBetter,
                unit: "fraction".into(),
            },
            measurement: QualityMeasurement {
                score: 0.9,
                sample_count: 10,
                uncertainty: None,
            },
            provenance: TaskExecutionProvenance {
                commit_sha: "task-commit".into(),
                toolchain: "rust".into(),
                compiler: "rustc".into(),
                target: "target".into(),
                operating_system: "linux".into(),
                hardware: "task-hardware".into(),
                runner: "task-runner".into(),
            },
            upstream_evidence: vec![],
            execution_status: "executed".into(),
            qualification_status: "qualified".into(),
        }
    }

    fn upstream_performance() -> PerformanceEvidenceRecord {
        let bytes = 2 * 131_072 * 4;
        PerformanceEvidenceRecord {
            schema_version: PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
            identity: BenchmarkIdentity {
                benchmark: "simd_continuous".into(),
                workload: "dot".into(),
                resolution: 131_072,
                representation: "continuous_f32".into(),
                operation: "dot".into(),
                implementation: "avx2".into(),
                seed: 43,
            },
            provenance: PerformanceExecutionProvenance {
                commit_sha: "performance-commit".into(),
                toolchain: "rust".into(),
                compiler: "rustc".into(),
                target: "target".into(),
                operating_system: "linux".into(),
                hardware: "performance-hardware".into(),
                runner: "performance-runner".into(),
            },
            measurement: Measurement {
                sample_count: 10,
                warmup_count: 2,
                iterations: 10,
                batch_size: 2,
                logical_bytes_per_iteration: bytes,
                total_logical_bytes: bytes * 10,
                elapsed_seconds: 1.0,
                throughput_bytes_per_second: None,
                allocations: None,
                peak_resident_bytes: None,
                physical_memory_bytes: None,
                energy_joules: None,
            },
            execution_status: "executed".into(),
        }
    }

    fn upstream_resource() -> ResourceEvidenceRecord {
        let workload = ResourceWorkload {
            resolution: 131_072,
            representation: "continuous_f32".into(),
            element_size_bytes: 4,
            resident_vectors: 1,
        };
        let bytes = workload.vector_bytes().unwrap();
        ResourceEvidenceRecord {
            schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION,
            workload,
            budget: Some(ResourceBudget::new(bytes, bytes, None)),
            vector_bytes: bytes,
            resident_bytes: bytes,
            peak_temporary_bytes: None,
            conversion_bytes: None,
            provenance_id: "resource-fixture".into(),
            qualification_status: RESOURCE_QUALIFIED_STATUS.into(),
        }
    }

    fn upstream_trajectory() -> TrajectoryEvidenceRecord {
        TrajectoryEvidenceRecord {
            schema_version: TRAJECTORY_EVIDENCE_SCHEMA_VERSION,
            source_resolution: 65_536,
            target_resolution: 131_072,
            operator_dependency: OperatorEvidenceDependency {
                schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
                representation: "continuous_f32".into(),
                artifact_sha256: "fixture".into(),
                required_operators: vec![],
            },
            resource_dependency: ResourceEvidenceDependency {
                schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION,
                resolution: 131_072,
                representation: "continuous_f32".into(),
                provenance_id: "resource-fixture".into(),
            },
            qualification_disposition: "qualified".into(),
            metrics: TrajectoryMetrics {
                terminal_state_error: None,
                mean_state_error: None,
                terminal_tau_error: None,
                mean_tau_error: None,
            },
        }
    }

    #[test]
    fn evidence_derived_identity_validates_join() {
        let r = record();
        r.validate_against_evidence(
            &upstream_task(),
            &upstream_performance(),
            &upstream_resource(),
            &upstream_trajectory(),
        )
        .expect("source identities should agree with the join");
    }

    #[test]
    fn declared_join_identity_mismatch_fails_closed() {
        let mut r = record();
        r.identity.scenario_revision = "sha256:other-scenario".into();
        assert!(matches!(
            r.validate_against_evidence(
                &upstream_task(),
                &upstream_performance(),
                &upstream_resource(),
                &upstream_trajectory(),
            ),
            Err(CostQualityJoinError::Identity(_))
        ));
    }

    #[test]
    fn unknown_cost_dimensions_are_not_encoded_as_zero() {
        // The join contains references only. Numeric cost/quality values remain
        // in their source evidence and therefore cannot silently become zero.
        let r = record();
        assert!(r.validate().is_ok());
    }
}
