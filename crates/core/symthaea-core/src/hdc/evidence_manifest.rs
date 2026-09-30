//! First-class evidence manifest with independently verifiable experiment identity.
//!
//! The manifest carries the declared semantic identity alongside the join
//! artifact. The declaration is descriptive, not authoritative: validation
//! derives the experiment key from source evidence and compares its digest to
//! the declaration. Execution provenance remains in the source records.

use serde::{Deserialize, Serialize};

use super::cost_quality_join::CostQualityJoinRecord;
use super::evidence_identity::{derive_experiment_key, EvidenceIdentityError, ExperimentKey};
use super::performance_evidence::PerformanceEvidenceRecord;
use super::resource_evidence::ResourceEvidenceRecord;
use super::task_quality_evidence::TaskQualityEvidenceRecord;
use super::trajectory_evidence::TrajectoryEvidenceRecord;

pub const EVIDENCE_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const EXPERIMENT_IDENTITY_PREFIX: &str = "sha256:";
pub const EVIDENCE_MANIFEST_ARTIFACT_DOMAIN: &[u8] = b"symthaea:evidence-manifest";

/// Artifact-level declaration of a semantic experiment identity.
///
/// The digest is intentionally stored as a string so it can be transported
/// through JSON/CBOR/etc. without coupling the wire contract to a hash type.
/// It is never trusted without recomputation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredExperimentIdentity {
    pub schema_version: u32,
    pub digest: String,
}

fn validate_sha256_identity(digest: &str) -> bool {
    let Some(hex) = digest.strip_prefix(EXPERIMENT_IDENTITY_PREFIX) else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl DeclaredExperimentIdentity {
    pub fn from_key(key: &ExperimentKey) -> Self {
        Self {
            schema_version: super::evidence_identity::EVIDENCE_IDENTITY_SCHEMA_VERSION,
            digest: format!("{EXPERIMENT_IDENTITY_PREFIX}{}", key.identity_digest()),
        }
    }

    pub fn validate_against_key(
        &self,
        key: &ExperimentKey,
    ) -> Result<(), EvidenceManifestError> {
        if self.schema_version != super::evidence_identity::EVIDENCE_IDENTITY_SCHEMA_VERSION {
            return Err(EvidenceManifestError::UnsupportedIdentitySchema(
                self.schema_version,
            ));
        }
        if !validate_sha256_identity(&self.digest) {
            return Err(EvidenceManifestError::InvalidIdentity(EvidenceIdentityError::InvalidTaskQuality(
                "identity digest must be sha256: followed by exactly 64 hexadecimal characters".into(),
            )));
        }
        let expected = format!("{EXPERIMENT_IDENTITY_PREFIX}{}", key.identity_digest());
        if self.digest != expected {
            return Err(EvidenceManifestError::IdentityMismatch {
                expected,
                observed: self.digest.clone(),
            });
        }
        Ok(())
    }
}

/// Artifact boundary joining the semantic identity declaration to its join.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceManifest {
    pub schema_version: u32,
    pub experiment_identity: DeclaredExperimentIdentity,
    pub join: CostQualityJoinRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceManifestError {
    UnsupportedSchema(u32),
    UnsupportedIdentitySchema(u32),
    EmptyIdentityDigest,
    InvalidIdentity(EvidenceIdentityError),
    IdentityMismatch { expected: String, observed: String },
    Join(String),
}

impl std::fmt::Display for EvidenceManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported evidence manifest schema: {v}"),
            Self::UnsupportedIdentitySchema(v) => {
                write!(f, "unsupported experiment identity schema: {v}")
            }
            Self::EmptyIdentityDigest => write!(f, "experiment identity digest is empty"),
            Self::InvalidIdentity(e) => write!(f, "experiment identity derivation failed: {e}"),
            Self::IdentityMismatch { expected, observed } => write!(
                f,
                "declared experiment identity mismatch: expected {expected}, observed {observed}"
            ),
            Self::Join(e) => write!(f, "join validation failed: {e}"),
        }
    }
}

impl std::error::Error for EvidenceManifestError {}

impl EvidenceManifest {
    /// Validate the complete artifact boundary against source evidence.
    ///
    /// The declared digest is checked only after the semantic key has been
    /// independently derived from the source records. This makes the manifest
    /// identity a verifiable claim rather than caller-controlled identity.

    /// Canonical bytes for the exact manifest artifact.
    ///
    /// This is deliberately separate from the experiment identity: changing a
    /// referenced artifact, join declaration, or manifest schema changes the
    /// artifact digest without changing the semantic experiment identity.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn push_field(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value);
        }

        fn push_reference(bytes: &mut Vec<u8>, reference: &super::cost_quality_join::EvidenceReference) {
            push_field(bytes, reference.kind.as_bytes());
            bytes.extend_from_slice(&reference.schema_version.to_be_bytes());
            push_field(bytes, reference.artifact_digest.as_bytes());
            push_field(bytes, reference.artifact_id.as_bytes());
        }

        let mut bytes = Vec::with_capacity(1024);
        bytes.extend_from_slice(EVIDENCE_MANIFEST_ARTIFACT_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.experiment_identity.schema_version.to_be_bytes());
        push_field(&mut bytes, self.experiment_identity.digest.as_bytes());

        let identity = &self.join.identity;
        for field in [
            identity.task.as_str(),
            identity.scenario_set.as_str(),
            identity.scenario_revision.as_str(),
            identity.split.as_str(),
            identity.protocol.as_str(),
            identity.representation.as_str(),
            identity.model_revision.as_str(),
            identity.workload.as_str(),
            identity.benchmark.as_str(),
        ] {
            push_field(&mut bytes, field.as_bytes());
        }
        bytes.extend_from_slice(&(identity.resolution as u64).to_be_bytes());
        push_reference(&mut bytes, &self.join.trajectory);
        push_reference(&mut bytes, &self.join.resource);
        push_reference(&mut bytes, &self.join.performance);
        push_reference(&mut bytes, &self.join.task_quality);
        bytes
    }

    /// SHA-256 content identity of the exact manifest artifact.
    pub fn artifact_digest(&self) -> String {
        use sha2::{Digest, Sha256};

        let digest = Sha256::digest(self.canonical_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    pub fn validate_against_evidence(
        &self,
        task_quality: &TaskQualityEvidenceRecord,
        performance: &PerformanceEvidenceRecord,
        resource: &ResourceEvidenceRecord,
        trajectory: &TrajectoryEvidenceRecord,
    ) -> Result<(), EvidenceManifestError> {
        self.validate_shape()?;
        if self.experiment_identity.digest.is_empty() {
            return Err(EvidenceManifestError::EmptyIdentityDigest);
        }

        let key = derive_experiment_key(task_quality, performance, resource)
            .map_err(EvidenceManifestError::InvalidIdentity)?;
        self.experiment_identity.validate_against_key(&key)?;
        self.join
            .validate_against_evidence(task_quality, performance, resource, trajectory)
            .map_err(|error| EvidenceManifestError::Join(error.to_string()))?;
        Ok(())
    }

    pub fn validate_shape(&self) -> Result<(), EvidenceManifestError> {
        if self.schema_version != EVIDENCE_MANIFEST_SCHEMA_VERSION {
            return Err(EvidenceManifestError::UnsupportedSchema(self.schema_version));
        }
        if self.experiment_identity.digest.is_empty() {
            return Err(EvidenceManifestError::EmptyIdentityDigest);
        }
        if !validate_sha256_identity(&self.experiment_identity.digest) {
            return Err(EvidenceManifestError::InvalidIdentity(EvidenceIdentityError::InvalidTaskQuality(
                "identity digest must be sha256: followed by exactly 64 hexadecimal characters".into(),
            )));
        }
        self.join
            .validate()
            .map_err(|error| EvidenceManifestError::Join(error.to_string()))
    }

    pub fn from_join_and_key(join: CostQualityJoinRecord, key: &ExperimentKey) -> Self {
        Self {
            schema_version: EVIDENCE_MANIFEST_SCHEMA_VERSION,
            experiment_identity: DeclaredExperimentIdentity::from_key(key),
            join,
        }
    }
}

/// Stable content-addressed subject for publishing or attesting a manifest.
///
/// The envelope intentionally carries the manifest digest separately from the
/// manifest bytes. This avoids circular self-hashing while still making the
/// subject claim independently verifiable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceManifestEnvelope {
    pub schema_version: u32,
    pub subject_digest: String,
    pub manifest: EvidenceManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceManifestEnvelopeError {
    UnsupportedSchema(u32),
    InvalidSubjectDigest,
    SubjectMismatch { expected: String, observed: String },
    Manifest(EvidenceManifestError),
}

impl std::fmt::Display for EvidenceManifestEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported evidence manifest envelope schema: {v}"),
            Self::InvalidSubjectDigest => write!(f, "subject digest must be sha256: followed by exactly 64 hexadecimal characters"),
            Self::SubjectMismatch { expected, observed } => {
                write!(f, "manifest subject digest mismatch: expected {expected}, observed {observed}")
            }
            Self::Manifest(error) => write!(f, "manifest validation failed: {error}"),
        }
    }
}

impl std::error::Error for EvidenceManifestEnvelopeError {}

pub const EVIDENCE_MANIFEST_ENVELOPE_SCHEMA_VERSION: u32 = 1;

impl EvidenceManifest {
    /// Canonical subject digest used by external attestations.
    pub fn subject_digest(&self) -> String {
        format!("{EXPERIMENT_IDENTITY_PREFIX}{}", self.artifact_digest())
    }
}

impl EvidenceManifestEnvelope {
    pub fn from_manifest(manifest: EvidenceManifest) -> Self {
        let subject_digest = manifest.subject_digest();
        Self {
            schema_version: EVIDENCE_MANIFEST_ENVELOPE_SCHEMA_VERSION,
            subject_digest,
            manifest,
        }
    }

    /// Validate the envelope's declared subject against independently derived
    /// manifest content, then validate the manifest against source evidence.
    pub fn validate(
        &self,
        task_quality: &TaskQualityEvidenceRecord,
        performance: &PerformanceEvidenceRecord,
        resource: &ResourceEvidenceRecord,
        trajectory: &TrajectoryEvidenceRecord,
    ) -> Result<(), EvidenceManifestEnvelopeError> {
        if self.schema_version != EVIDENCE_MANIFEST_ENVELOPE_SCHEMA_VERSION {
            return Err(EvidenceManifestEnvelopeError::UnsupportedSchema(self.schema_version));
        }
        if !validate_sha256_identity(&self.subject_digest) {
            return Err(EvidenceManifestEnvelopeError::InvalidSubjectDigest);
        }

        let expected = self.manifest.subject_digest();
        if self.subject_digest != expected {
            return Err(EvidenceManifestEnvelopeError::SubjectMismatch {
                expected,
                observed: self.subject_digest.clone(),
            });
        }

        self.manifest
            .validate_against_evidence(task_quality, performance, resource, trajectory)
            .map_err(EvidenceManifestEnvelopeError::Manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::cost_quality_join::{EvidenceReference, JoinIdentity};
    use crate::hdc::operator_evidence_contract::OPERATOR_EVIDENCE_SCHEMA_VERSION;
    use crate::hdc::performance_evidence::{BenchmarkIdentity, ExecutionProvenance as PerformanceProvenance, Measurement, PERFORMANCE_EVIDENCE_SCHEMA_VERSION};
    use crate::hdc::resource_evidence::{ResourceBudget, ResourceWorkload, RESOURCE_EVIDENCE_SCHEMA_VERSION, RESOURCE_QUALIFIED_STATUS};
    use crate::hdc::task_quality_evidence::{ExecutionProvenance as TaskProvenance, MetricDirection, QualityMeasurement, QualityMetric, TaskIdentity, TaskQualityEvidenceRecord, TASK_QUALITY_EVIDENCE_SCHEMA_VERSION};
    use crate::hdc::trajectory_evidence::{OperatorEvidenceDependency, ResourceEvidenceDependency, TrajectoryEvidenceRecord, TrajectoryMetrics, TRAJECTORY_EVIDENCE_SCHEMA_VERSION};

    fn evidence() -> (TaskQualityEvidenceRecord, PerformanceEvidenceRecord, ResourceEvidenceRecord, TrajectoryEvidenceRecord) {
        let task = TaskQualityEvidenceRecord {
            schema_version: TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
            identity: TaskIdentity { task: "retrieval".into(), scenario_set: "set-v1".into(), scenario_revision: "sha256:scenario".into(), split: "held-out".into(), protocol: "top-k-10".into(), resolution: 131_072, representation: "continuous_f32".into(), model_revision: "model-v1".into(), seed: 42 },
            metric: QualityMetric { name: "recall".into(), direction: MetricDirection::HigherIsBetter, unit: "fraction".into() },
            measurement: QualityMeasurement { score: 0.9, sample_count: 10, uncertainty: None },
            provenance: TaskProvenance { commit_sha: "task".into(), toolchain: "rust".into(), compiler: "rustc".into(), target: "target".into(), operating_system: "linux".into(), hardware: "task-hw".into(), runner: "task-runner".into() },
            upstream_evidence: vec![], execution_status: "executed".into(), qualification_status: "qualified".into(),
        };
        let bytes = 2 * 131_072 * 4;
        let performance = PerformanceEvidenceRecord {
            schema_version: PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
            identity: BenchmarkIdentity { benchmark: "simd-v1".into(), workload: "dot".into(), resolution: 131_072, representation: "continuous_f32".into(), operation: "dot".into(), implementation: "avx2".into(), seed: 7 },
            provenance: PerformanceProvenance { commit_sha: "perf".into(), toolchain: "rust".into(), compiler: "rustc".into(), target: "target".into(), operating_system: "linux".into(), hardware: "perf-hw".into(), runner: "perf-runner".into() },
            measurement: Measurement { sample_count: 10, warmup_count: 2, iterations: 10, batch_size: 2, logical_bytes_per_iteration: bytes, total_logical_bytes: bytes * 10, elapsed_seconds: 1.0, throughput_bytes_per_second: None, allocations: None, peak_resident_bytes: None, physical_memory_bytes: None, energy_joules: None },
            execution_status: "executed".into(),
        };
        let workload = ResourceWorkload { resolution: 131_072, representation: "continuous_f32".into(), element_size_bytes: 4, resident_vectors: 1 };
        let vector_bytes = workload.vector_bytes().unwrap();
        let resource = ResourceEvidenceRecord { schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION, workload, budget: Some(ResourceBudget::new(vector_bytes, vector_bytes, None)), vector_bytes, resident_bytes: vector_bytes, peak_temporary_bytes: None, conversion_bytes: None, provenance_id: "resource".into(), qualification_status: RESOURCE_QUALIFIED_STATUS.into() };
        let trajectory = TrajectoryEvidenceRecord {
            schema_version: TRAJECTORY_EVIDENCE_SCHEMA_VERSION,
            source_resolution: 65_536,
            target_resolution: 131_072,
            operator_dependency: OperatorEvidenceDependency { schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION, representation: "continuous_f32".into(), artifact_sha256: "fixture".into(), required_operators: vec![] },
            resource_dependency: ResourceEvidenceDependency { schema_version: RESOURCE_EVIDENCE_SCHEMA_VERSION, resolution: 131_072, representation: "continuous_f32".into(), provenance_id: "resource".into() },
            qualification_disposition: "qualified".into(),
            metrics: TrajectoryMetrics { terminal_state_error: None, mean_state_error: None, terminal_tau_error: None, mean_tau_error: None },
        };
        (task, performance, resource, trajectory)
    }

    fn join(key: &ExperimentKey) -> CostQualityJoinRecord {
        let reference = |kind: &str| EvidenceReference { kind: kind.into(), schema_version: 1, artifact_digest: format!("sha256:{}", "0".repeat(64)), artifact_id: format!("{kind}-v1") };
        CostQualityJoinRecord {
            schema_version: crate::hdc::cost_quality_join::COST_QUALITY_JOIN_SCHEMA_VERSION,
            identity: JoinIdentity { task: key.task.clone(), scenario_set: key.scenario_set.clone(), scenario_revision: key.scenario_revision.clone(), split: key.split.clone(), protocol: key.protocol.clone(), resolution: key.resolution, representation: key.representation.clone(), model_revision: key.model_revision.clone(), workload: key.workload.clone(), benchmark: key.benchmark.clone() },
            trajectory: reference("trajectory"), resource: reference("resource"), performance: reference("performance"), task_quality: reference("task_quality"),
        }
    }

    #[test]
    fn artifact_digest_changes_when_manifest_content_changes() {
        let (task, performance, resource, trajectory) = evidence();
        let key = derive_experiment_key(&task, &performance, &resource).unwrap();
        let mut manifest = EvidenceManifest::from_join_and_key(join(&key), &key);
        let original = manifest.artifact_digest();
        manifest.join.performance.artifact_id = "performance-v2".into();
        assert_ne!(original, manifest.artifact_digest());
        assert_eq!(manifest.experiment_identity.digest, EvidenceManifest::from_join_and_key(join(&key), &key).experiment_identity.digest);
        manifest.validate_against_evidence(&task, &performance, &resource, &trajectory).unwrap();
    }

    #[test]
    fn manifest_identity_is_derived_not_authoritative() {
        let (task, performance, resource, trajectory) = evidence();
        let key = derive_experiment_key(&task, &performance, &resource).unwrap();
        let manifest = EvidenceManifest::from_join_and_key(join(&key), &key);
        manifest.validate_against_evidence(&task, &performance, &resource, &trajectory).unwrap();
    }

    #[test]
    fn declared_digest_mismatch_fails_closed() {
        let (task, performance, resource, trajectory) = evidence();
        let key = derive_experiment_key(&task, &performance, &resource).unwrap();
        let mut manifest = EvidenceManifest::from_join_and_key(join(&key), &key);
        manifest.experiment_identity.digest = "sha256:attacker-selected".into();
        assert!(matches!(manifest.validate_against_evidence(&task, &performance, &resource, &trajectory), Err(EvidenceManifestError::IdentityMismatch { .. })));
    }

    #[test]
    fn malformed_digest_fails_closed() {
        let (task, performance, resource, trajectory) = evidence();
        let key = derive_experiment_key(&task, &performance, &resource).unwrap();
        let mut manifest = EvidenceManifest::from_join_and_key(join(&key), &key);
        manifest.experiment_identity.digest = "sha256:not-a-digest".into();
        assert!(matches!(
            manifest.validate_against_evidence(&task, &performance, &resource, &trajectory),
            Err(EvidenceManifestError::InvalidIdentity(_))
        ));
    }

    #[test]
    fn semantic_source_change_invalidates_declared_manifest_identity() {
        let (mut task, performance, resource, trajectory) = evidence();
        let original_key = derive_experiment_key(&task, &performance, &resource).unwrap();
        let manifest = EvidenceManifest::from_join_and_key(join(&original_key), &original_key);
        task.identity.model_revision = "model-v2".into();
        assert!(matches!(manifest.validate_against_evidence(&task, &performance, &resource, &trajectory), Err(EvidenceManifestError::IdentityMismatch { .. } | EvidenceManifestError::Join(_))));
    }
}
