//! Feature-neutral measured task-quality evidence.
//!
//! This contract records task outcomes under an explicit evaluation protocol.
//! It does not infer quality from benchmark names, performance measurements,
//! resource admissibility, or trajectory qualification.

use serde::{Deserialize, Serialize};

pub const TASK_QUALITY_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskIdentity {
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub split: String,
    pub protocol: String,
    pub resolution: usize,
    pub representation: String,
    pub model_revision: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityMetric {
    pub name: String,
    pub direction: MetricDirection,
    pub unit: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
    TargetValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMeasurement {
    pub score: f64,
    pub sample_count: u64,
    pub uncertainty: Option<UncertaintyInterval>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyInterval {
    pub method: String,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub kind: String,
    pub schema_version: u32,
    pub artifact_digest: String,
    pub artifact_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProvenance {
    pub commit_sha: String,
    pub toolchain: String,
    pub compiler: String,
    pub target: String,
    pub operating_system: String,
    pub hardware: String,
    pub runner: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskQualityEvidenceRecord {
    pub schema_version: u32,
    pub identity: TaskIdentity,
    pub metric: QualityMetric,
    pub measurement: QualityMeasurement,
    pub provenance: ExecutionProvenance,
    pub upstream_evidence: Vec<EvidenceReference>,
    pub execution_status: String,
    pub qualification_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskQualityEvidenceError {
    UnsupportedSchema(u32),
    EmptyIdentity(&'static str),
    InvalidResolution,
    EmptyMetric(&'static str),
    InvalidScore,
    ZeroSamples,
    InvalidUncertainty(&'static str),
    EmptyProvenance(&'static str),
    EmptyCommitSha,
    InvalidEvidenceReference(&'static str),
    BenchmarkNotExecuted,
    InvalidQualificationStatus,
}

impl std::fmt::Display for TaskQualityEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported task-quality evidence schema: {v}"),
            Self::EmptyIdentity(field) => write!(f, "task identity field is empty: {field}"),
            Self::InvalidResolution => write!(f, "task resolution must be non-zero"),
            Self::EmptyMetric(field) => write!(f, "quality metric field is empty: {field}"),
            Self::InvalidScore => write!(f, "quality score must be finite"),
            Self::ZeroSamples => write!(f, "quality sample count must be non-zero"),
            Self::InvalidUncertainty(field) => write!(f, "invalid uncertainty: {field}"),
            Self::EmptyProvenance(field) => write!(f, "provenance field is empty: {field}"),
            Self::EmptyCommitSha => write!(f, "commit SHA is empty"),
            Self::InvalidEvidenceReference(field) => {
                write!(f, "invalid upstream evidence reference: {field}")
            }
            Self::BenchmarkNotExecuted => write!(f, "task-quality evidence is not an executed evaluation"),
            Self::InvalidQualificationStatus => write!(f, "qualification status must be explicit"),
        }
    }
}

impl std::error::Error for TaskQualityEvidenceError {}

impl TaskQualityEvidenceRecord {
    pub fn validate(&self) -> Result<(), TaskQualityEvidenceError> {
        if self.schema_version != TASK_QUALITY_EVIDENCE_SCHEMA_VERSION {
            return Err(TaskQualityEvidenceError::UnsupportedSchema(self.schema_version));
        }

        for (value, name) in [
            (&self.identity.task, "task"),
            (&self.identity.scenario_set, "scenario_set"),
            (&self.identity.scenario_revision, "scenario_revision"),
            (&self.identity.split, "split"),
            (&self.identity.protocol, "protocol"),
            (&self.identity.representation, "representation"),
            (&self.identity.model_revision, "model_revision"),
        ] {
            if value.is_empty() {
                return Err(TaskQualityEvidenceError::EmptyIdentity(name));
            }
        }

        if self.identity.resolution == 0 {
            return Err(TaskQualityEvidenceError::InvalidResolution);
        }

        for (value, name) in [
            (&self.metric.name, "name"),
            (&self.metric.unit, "unit"),
        ] {
            if value.is_empty() {
                return Err(TaskQualityEvidenceError::EmptyMetric(name));
            }
        }

        if !self.measurement.score.is_finite() {
            return Err(TaskQualityEvidenceError::InvalidScore);
        }

        if self.measurement.sample_count == 0 {
            return Err(TaskQualityEvidenceError::ZeroSamples);
        }

        if let Some(interval) = &self.measurement.uncertainty {
            if interval.method.is_empty() {
                return Err(TaskQualityEvidenceError::InvalidUncertainty("method"));
            }
            if !interval.lower.is_finite() || !interval.upper.is_finite() {
                return Err(TaskQualityEvidenceError::InvalidUncertainty("bounds"));
            }
            if interval.lower > interval.upper {
                return Err(TaskQualityEvidenceError::InvalidUncertainty("lower_above_upper"));
            }
            if self.measurement.score < interval.lower || self.measurement.score > interval.upper {
                return Err(TaskQualityEvidenceError::InvalidUncertainty("score_outside_interval"));
            }
        }

        if self.provenance.commit_sha.is_empty() {
            return Err(TaskQualityEvidenceError::EmptyCommitSha);
        }

        for (value, name) in [
            (&self.provenance.toolchain, "toolchain"),
            (&self.provenance.compiler, "compiler"),
            (&self.provenance.target, "target"),
            (&self.provenance.operating_system, "operating_system"),
            (&self.provenance.hardware, "hardware"),
            (&self.provenance.runner, "runner"),
        ] {
            if value.is_empty() {
                return Err(TaskQualityEvidenceError::EmptyProvenance(name));
            }
        }

        for reference in &self.upstream_evidence {
            if reference.kind.is_empty() {
                return Err(TaskQualityEvidenceError::InvalidEvidenceReference("kind"));
            }
            if reference.schema_version == 0 {
                return Err(TaskQualityEvidenceError::InvalidEvidenceReference("schema_version"));
            }
            if reference.artifact_digest.is_empty() {
                return Err(TaskQualityEvidenceError::InvalidEvidenceReference("artifact_digest"));
            }
            if reference.artifact_id.is_empty() {
                return Err(TaskQualityEvidenceError::InvalidEvidenceReference("artifact_id"));
            }
        }

        if self.execution_status != "executed" {
            return Err(TaskQualityEvidenceError::BenchmarkNotExecuted);
        }

        if self.qualification_status.is_empty() {
            return Err(TaskQualityEvidenceError::InvalidQualificationStatus);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> TaskQualityEvidenceRecord {
        TaskQualityEvidenceRecord {
            schema_version: TASK_QUALITY_EVIDENCE_SCHEMA_VERSION,
            identity: TaskIdentity {
                task: "hdc_retrieval".to_owned(),
                scenario_set: "retrieval-fixture-v1".to_owned(),
                scenario_revision: "sha256:scenario-fixture".to_owned(),
                split: "held-out".to_owned(),
                protocol: "top-k-10".to_owned(),
                resolution: 131_072,
                representation: "continuous_f32".to_owned(),
                model_revision: "model-fixture-v1".to_owned(),
                seed: 42,
            },
            metric: QualityMetric {
                name: "top_k_recall".to_owned(),
                direction: MetricDirection::HigherIsBetter,
                unit: "fraction".to_owned(),
            },
            measurement: QualityMeasurement {
                score: 0.92,
                sample_count: 100,
                uncertainty: Some(UncertaintyInterval {
                    method: "fixture_interval".to_owned(),
                    lower: 0.88,
                    upper: 0.95,
                }),
            },
            provenance: ExecutionProvenance {
                commit_sha: "deadbeef".to_owned(),
                toolchain: "rustc 1.96.0".to_owned(),
                compiler: "rustc".to_owned(),
                target: "x86_64-unknown-linux-gnu".to_owned(),
                operating_system: "linux".to_owned(),
                hardware: "fixture-hardware".to_owned(),
                runner: "declared-runner".to_owned(),
            },
            upstream_evidence: vec![EvidenceReference {
                kind: "performance".to_owned(),
                schema_version: 1,
                artifact_digest: "sha256:fixture".to_owned(),
                artifact_id: "performance-fixture-v1".to_owned(),
            }],
            execution_status: "executed".to_owned(),
            qualification_status: "qualified".to_owned(),
        }
    }

    #[test]
    fn executed_record_validates() {
        record().validate().expect("fixture should validate");
    }

    #[test]
    fn compile_only_fails_closed() {
        let mut r = record();
        r.execution_status = "compile-only".to_owned();
        assert!(matches!(
            r.validate(),
            Err(TaskQualityEvidenceError::BenchmarkNotExecuted)
        ));
    }

    #[test]
    fn missing_uncertainty_remains_valid_and_unknown() {
        let mut r = record();
        r.measurement.uncertainty = None;
        r.validate().expect("uncertainty is optional");
        assert!(r.measurement.uncertainty.is_none());
    }

    #[test]
    fn invalid_uncertainty_bounds_fail_closed() {
        let mut r = record();
        r.measurement.uncertainty.as_mut().unwrap().lower = 0.99;
        assert!(matches!(
            r.validate(),
            Err(TaskQualityEvidenceError::InvalidUncertainty("lower_above_upper"))
        ));
    }

    #[test]
    fn score_outside_uncertainty_fails_closed() {
        let mut r = record();
        r.measurement.score = 0.5;
        assert!(matches!(
            r.validate(),
            Err(TaskQualityEvidenceError::InvalidUncertainty(
                "score_outside_interval"
            ))
        ));
    }

    #[test]
    fn malformed_upstream_reference_fails_closed() {
        let mut r = record();
        r.upstream_evidence[0].artifact_digest.clear();
        assert!(matches!(
            r.validate(),
            Err(TaskQualityEvidenceError::InvalidEvidenceReference(
                "artifact_digest"
            ))
        ));
    }

    #[test]
    fn provenance_is_required() {
        let mut r = record();
        r.provenance.hardware.clear();
        assert!(matches!(
            r.validate(),
            Err(TaskQualityEvidenceError::EmptyProvenance("hardware"))
        ));
    }
}
