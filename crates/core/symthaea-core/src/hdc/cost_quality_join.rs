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

    #[test]
    fn unknown_cost_dimensions_are_not_encoded_as_zero() {
        // The join contains references only. Numeric cost/quality values remain
        // in their source evidence and therefore cannot silently become zero.
        let r = record();
        assert!(r.validate().is_ok());
    }
}
