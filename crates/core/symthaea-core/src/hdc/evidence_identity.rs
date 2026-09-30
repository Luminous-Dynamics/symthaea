//! Canonical semantic identity for joining independently produced HDC evidence.
//!
//! The experiment key answers whether evidence describes the same experiment.
//! Execution provenance is intentionally not part of this key: performance and
//! task-quality evidence may be produced by different jobs or machines while
//! remaining semantically comparable. Exact execution provenance stays in the
//! source evidence records.

use super::cost_quality_join::JoinIdentity;
use super::performance_evidence::PerformanceEvidenceRecord;
use super::resource_evidence::ResourceEvidenceRecord;
use super::task_quality_evidence::TaskQualityEvidenceRecord;
use super::trajectory_evidence::TrajectoryEvidenceRecord;

pub const EVIDENCE_IDENTITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentKey {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceIdentityError {
    InvalidTaskQuality(String),
    InvalidPerformance(String),
    InvalidResource(String),
    InvalidTrajectory(String),
    MissingIdentity(&'static str),
    Mismatch {
        field: &'static str,
        expected: String,
        observed: String,
    },
}

impl std::fmt::Display for EvidenceIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTaskQuality(e) => write!(f, "task-quality evidence is invalid: {e}"),
            Self::InvalidPerformance(e) => write!(f, "performance evidence is invalid: {e}"),
            Self::InvalidResource(e) => write!(f, "resource evidence is invalid: {e}"),
            Self::InvalidTrajectory(e) => write!(f, "trajectory evidence is invalid: {e}"),
            Self::MissingIdentity(field) => write!(f, "required experiment identity is missing: {field}"),
            Self::Mismatch { field, expected, observed } => write!(
                f,
                "experiment identity mismatch for {field}: expected {expected}, observed {observed}"
            ),
        }
    }
}

impl std::error::Error for EvidenceIdentityError {}

fn require_equal(
    field: &'static str,
    expected: &str,
    observed: &str,
) -> Result<(), EvidenceIdentityError> {
    if expected.is_empty() || observed.is_empty() {
        return Err(EvidenceIdentityError::MissingIdentity(field));
    }
    if expected != observed {
        return Err(EvidenceIdentityError::Mismatch {
            field,
            expected: expected.to_owned(),
            observed: observed.to_owned(),
        });
    }
    Ok(())
}

fn require_resolution(
    field: &'static str,
    expected: usize,
    observed: usize,
) -> Result<(), EvidenceIdentityError> {
    if expected == 0 || observed == 0 {
        return Err(EvidenceIdentityError::MissingIdentity(field));
    }
    if expected != observed {
        return Err(EvidenceIdentityError::Mismatch {
            field,
            expected: expected.to_string(),
            observed: observed.to_string(),
        });
    }
    Ok(())
}

/// Derive the semantic experiment key from independently validated evidence.
///
/// This is deliberately not a constructor accepting a caller-supplied key:
/// the key is derived from source evidence so duplicated identity metadata
/// cannot silently disagree with its referenced artifacts.
pub fn derive_experiment_key(
    task_quality: &TaskQualityEvidenceRecord,
    performance: &PerformanceEvidenceRecord,
    resource: &ResourceEvidenceRecord,
) -> Result<ExperimentKey, EvidenceIdentityError> {
    task_quality
        .validate()
        .map_err(|e| EvidenceIdentityError::InvalidTaskQuality(e.to_string()))?;
    performance
        .validate()
        .map_err(|e| EvidenceIdentityError::InvalidPerformance(e.to_string()))?;
    resource
        .workload
        .validate()
        .map_err(|e| EvidenceIdentityError::InvalidResource(e.to_string()))?;

    require_resolution(
        "resolution",
        task_quality.identity.resolution,
        performance.identity.resolution,
    )
    .map_err(|error| error)?;
    require_resolution(
        "resolution",
        task_quality.identity.resolution,
        resource.workload.resolution,
    )
    .map_err(|error| error)?;
    require_equal(
        "representation",
        &task_quality.identity.representation,
        &performance.identity.representation,
    )?;
    require_equal(
        "representation",
        &task_quality.identity.representation,
        &resource.workload.representation,
    )?;

    Ok(ExperimentKey {
        task: task_quality.identity.task.clone(),
        scenario_set: task_quality.identity.scenario_set.clone(),
        scenario_revision: task_quality.identity.scenario_revision.clone(),
        split: task_quality.identity.split.clone(),
        protocol: task_quality.identity.protocol.clone(),
        resolution: task_quality.identity.resolution,
        representation: task_quality.identity.representation.clone(),
        model_revision: task_quality.identity.model_revision.clone(),
        workload: performance.identity.workload.clone(),
        benchmark: performance.identity.benchmark.clone(),
    })
}

/// Verify a caller-supplied join identity against the key derived from source evidence.
pub fn verify_join_identity(
    key: &ExperimentKey,
    declared: &JoinIdentity,
) -> Result<(), EvidenceIdentityError> {
    require_equal("task", &key.task, &declared.task)?;
    require_equal("scenario_set", &key.scenario_set, &declared.scenario_set)?;
    require_equal("scenario_revision", &key.scenario_revision, &declared.scenario_revision)?;
    require_equal("split", &key.split, &declared.split)?;
    require_equal("protocol", &key.protocol, &declared.protocol)?;
    require_resolution("resolution", key.resolution, declared.resolution)?;
    require_equal("representation", &key.representation, &declared.representation)?;
    require_equal("model_revision", &key.model_revision, &declared.model_revision)?;
    require_equal("workload", &key.workload, &declared.workload)?;
    require_equal("benchmark", &key.benchmark, &declared.benchmark)?;
    Ok(())
}

/// Verify that a qualified transition terminates at the experiment resolution.
/// A transition may legitimately originate at another resolution.
pub fn verify_trajectory_target(
    key: &ExperimentKey,
    trajectory: &TrajectoryEvidenceRecord,
) -> Result<(), EvidenceIdentityError> {
    if trajectory.schema_version == 0 {
        return Err(EvidenceIdentityError::InvalidTrajectory("schema version is zero".into()));
    }
    require_resolution("trajectory.target_resolution", key.resolution, trajectory.target_resolution)?;
    require_equal(
        "trajectory.representation",
        &key.representation,
        &trajectory.operator_dependency.representation,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::performance_evidence::{BenchmarkIdentity, ExecutionProvenance, Measurement, PERFORMANCE_EVIDENCE_SCHEMA_VERSION};
    use crate::hdc::resource_evidence::{ResourceBudget, ResourceWorkload, RESOURCE_EVIDENCE_SCHEMA_VERSION, RESOURCE_QUALIFIED_STATUS};
    use crate::hdc::task_quality_evidence::{ExecutionProvenance as TaskProvenance, MetricDirection, QualityMeasurement, QualityMetric, TaskIdentity, TaskQualityEvidenceRecord, UncertaintyInterval, TASK_QUALITY_EVIDENCE_SCHEMA_VERSION};

    fn task() -> TaskQualityEvidenceRecord {
        TaskQualityEvidenceRecord {
            schema_version: TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
            identity: TaskIdentity { task: "retrieval".into(), scenario_set: "set-v1".into(), scenario_revision: "sha256:scenario".into(), split: "held-out".into(), protocol: "top-k-10".into(), resolution: 131_072, representation: "continuous_f32".into(), model_revision: "model-v1".into(), seed: 42 },
            metric: QualityMetric { name: "recall".into(), direction: MetricDirection::HigherIsBetter, unit: "fraction".into() },
            measurement: QualityMeasurement { score: 0.9, sample_count: 10, uncertainty: Some(UncertaintyInterval { method: "fixture".into(), lower: 0.8, upper: 0.95 }) },
            provenance: TaskProvenance { commit_sha: "task-commit".into(), toolchain: "rust".into(), compiler: "rustc".into(), target: "target".into(), operating_system: "linux".into(), hardware: "hw".into(), runner: "runner".into() },
            upstream_evidence: vec![], execution_status: "executed".into(), qualification_status: "qualified".into(),
        }
    }

    fn performance() -> PerformanceEvidenceRecord {
        let bytes = 2 * 131_072 * 4;
        let iterations = 10;
        PerformanceEvidenceRecord {
            schema_version: PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
            identity: BenchmarkIdentity { benchmark: "simd-v1".into(), workload: "dot".into(), resolution: 131_072, representation: "continuous_f32".into(), operation: "dot".into(), implementation: "avx2".into(), seed: 7 },
            provenance: ExecutionProvenance { commit_sha: "perf-commit".into(), toolchain: "rust".into(), compiler: "rustc".into(), target: "target".into(), operating_system: "linux".into(), hardware: "other-hw".into(), runner: "other-runner".into() },
            measurement: Measurement { sample_count: 10, warmup_count: 2, iterations, batch_size: 2, logical_bytes_per_iteration: bytes, total_logical_bytes: bytes * iterations, elapsed_seconds: 1.0, throughput_bytes_per_second: None, allocations: None, peak_resident_bytes: None, physical_memory_bytes: None, energy_joules: None },
            execution_status: "executed".into(),
        }
    }

    fn resource() -> ResourceEvidenceRecord {
        let workload = ResourceWorkload { resolution: 131_072, representation: "continuous_f32".into(), element_size_bytes: 4, resident_vectors: 1 };
        let vector_bytes = workload.vector_bytes().unwrap();
        ResourceEvidenceRecord { schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION, vector_bytes, resident_bytes: vector_bytes, peak_temporary_bytes: None, conversion_bytes: None, provenance_id: "resource-v1".into(), qualification_status: RESOURCE_QUALIFIED_STATUS.into(), budget: Some(ResourceBudget::new(vector_bytes, vector_bytes, None)) }
    }

    #[test]
    fn same_experiment_different_execution_provenance_is_joinable() {
        let key = derive_experiment_key(&task(), &performance(), &resource()).unwrap();
        assert_eq!(key.resolution, 131_072);
        assert_eq!(key.representation, "continuous_f32");
    }

    #[test]
    fn declared_scenario_revision_mismatch_is_rejected() {
        let key = derive_experiment_key(&task(), &performance(), &resource()).unwrap();
        let mut declared = JoinIdentity {
            task: key.task.clone(),
            scenario_set: key.scenario_set.clone(),
            scenario_revision: key.scenario_revision.clone(),
            split: key.split.clone(),
            protocol: key.protocol.clone(),
            resolution: key.resolution,
            representation: key.representation.clone(),
            model_revision: key.model_revision.clone(),
            workload: key.workload.clone(),
            benchmark: key.benchmark.clone(),
        };
        declared.scenario_revision = "sha256:other".into();
        assert!(matches!(verify_join_identity(&key, &declared), Err(EvidenceIdentityError::Mismatch { field: "scenario_revision", .. })));
    }

    #[test]
    fn declared_model_revision_mismatch_is_rejected() {
        let key = derive_experiment_key(&task(), &performance(), &resource()).unwrap();
        let mut declared = JoinIdentity {
            task: key.task.clone(), scenario_set: key.scenario_set.clone(), scenario_revision: key.scenario_revision.clone(),
            split: key.split.clone(), protocol: key.protocol.clone(), resolution: key.resolution,
            representation: key.representation.clone(), model_revision: "model-other".into(),
            workload: key.workload.clone(), benchmark: key.benchmark.clone(),
        };
        assert!(matches!(verify_join_identity(&key, &declared), Err(EvidenceIdentityError::Mismatch { field: "model_revision", .. })));
        declared.model_revision = key.model_revision.clone();
        assert!(verify_join_identity(&key, &declared).is_ok());
    }

    #[test]
    fn resolution_mismatch_is_rejected() {
        let mut p = performance();
        p.identity.resolution = 65_536;
        assert!(matches!(derive_experiment_key(&task(), &p, &resource()), Err(EvidenceIdentityError::Mismatch { field: "resolution", .. })));
    }

    #[test]
    fn representation_mismatch_is_rejected() {
        let mut p = performance();
        p.identity.representation = "binary".into();
        assert!(matches!(derive_experiment_key(&task(), &p, &resource()), Err(EvidenceIdentityError::Mismatch { field: "representation", .. })));
    }

    #[test]
    fn resource_resolution_mismatch_is_rejected() {
        let mut r = resource();
        r.workload.resolution = 65_536;
        assert!(matches!(derive_experiment_key(&task(), &performance(), &r), Err(EvidenceIdentityError::Mismatch { field: "resolution", .. })));
    }
}
