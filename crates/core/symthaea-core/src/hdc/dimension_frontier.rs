//! Dimension-quality frontier manifest.
//!
//! This is a reference-only aggregation contract. It does not rank dimensions,
//! choose a winner, or copy task/resource/performance measurements. Each row
//! must point at already-produced evidence artifacts for exactly one resolution.
//! Analysis can therefore construct Pareto/frontier views without losing the
//! provenance and identity of the underlying records.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DIMENSION_FRONTIER_SCHEMA_VERSION: u32 = 1;
pub const DIMENSION_FRONTIER_DOMAIN: &[u8] = b"symthaea:hdc-dimension-frontier";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierReference {
    pub kind: String,
    pub schema_version: u32,
    pub artifact_digest: String,
    pub artifact_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionFrontierRow {
    pub resolution: usize,
    pub task_quality: FrontierReference,
    pub performance: FrontierReference,
    pub resource: FrontierReference,
    pub structural_sweep: Option<FrontierReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionFrontierManifest {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub split: String,
    pub protocol: String,
    pub representation: String,
    pub model_revision: String,
    pub rows: Vec<DimensionFrontierRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionFrontierError {
    UnsupportedSchema(u32),
    EmptyField(&'static str),
    EmptyRows,
    ZeroResolution,
    UnsortedDimensions,
    DuplicateDimension(usize),
    InvalidReference { field: &'static str, reason: &'static str },
    ReferenceResolutionUnavailable,
}

impl std::fmt::Display for DimensionFrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported dimension frontier schema: {v}"),
            Self::EmptyField(field) => write!(f, "dimension frontier field is empty: {field}"),
            Self::EmptyRows => f.write_str("dimension frontier has no rows"),
            Self::ZeroResolution => f.write_str("dimension frontier resolution must be non-zero"),
            Self::UnsortedDimensions => f.write_str("dimension frontier rows must be strictly ascending"),
            Self::DuplicateDimension(d) => write!(f, "duplicate frontier dimension: {d}"),
            Self::InvalidReference { field, reason } => {
                write!(f, "invalid {field} reference: {reason}")
            }
            Self::ReferenceResolutionUnavailable => {
                f.write_str("reference does not encode resolution; source artifact must be resolved before numeric analysis")
            }
        }
    }
}

impl std::error::Error for DimensionFrontierError {}

impl DimensionFrontierManifest {
    pub fn validate(&self) -> Result<(), DimensionFrontierError> {
        if self.schema_version != DIMENSION_FRONTIER_SCHEMA_VERSION {
            return Err(DimensionFrontierError::UnsupportedSchema(self.schema_version));
        }

        for (value, field) in [
            (&self.task, "task"),
            (&self.scenario_set, "scenario_set"),
            (&self.scenario_revision, "scenario_revision"),
            (&self.split, "split"),
            (&self.protocol, "protocol"),
            (&self.representation, "representation"),
            (&self.model_revision, "model_revision"),
        ] {
            if value.is_empty() {
                return Err(DimensionFrontierError::EmptyField(field));
            }
        }

        if self.rows.is_empty() {
            return Err(DimensionFrontierError::EmptyRows);
        }

        let mut previous = 0usize;
        for (index, row) in self.rows.iter().enumerate() {
            if row.resolution == 0 {
                return Err(DimensionFrontierError::ZeroResolution);
            }
            if index > 0 && row.resolution <= previous {
                if row.resolution == previous {
                    return Err(DimensionFrontierError::DuplicateDimension(row.resolution));
                }
                return Err(DimensionFrontierError::UnsortedDimensions);
            }
            previous = row.resolution;

            validate_reference(&row.task_quality, "task_quality", "task_quality")?;
            validate_reference(&row.performance, "performance", "performance")?;
            validate_reference(&row.resource, "resource", "resource")?;
            if let Some(reference) = &row.structural_sweep {
                validate_reference(reference, "structural_sweep", "dimension_sweep")?;
            }
        }

        Ok(())
    }

    /// Canonical bytes contain semantic identity only; referenced artifact
    /// digests are included, while execution provenance stays in the artifacts.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(DIMENSION_FRONTIER_DOMAIN);
        push_u32(&mut bytes, self.schema_version);
        for value in [
            &self.task,
            &self.scenario_set,
            &self.scenario_revision,
            &self.split,
            &self.protocol,
            &self.representation,
            &self.model_revision,
        ] {
            push_string(&mut bytes, value);
        }
        push_u32(&mut bytes, self.rows.len() as u32);
        for row in &self.rows {
            push_u64(&mut bytes, row.resolution as u64);
            for reference in [
                &row.task_quality,
                &row.performance,
                &row.resource,
            ] {
                push_reference(&mut bytes, reference);
            }
            match &row.structural_sweep {
                Some(reference) => {
                    bytes.push(1);
                    push_reference(&mut bytes, reference);
                }
                None => bytes.push(0),
            }
        }
        bytes
    }

    pub fn artifact_digest(&self) -> String {
        let digest = Sha256::digest(self.canonical_bytes());
        let mut out = String::from("sha256:");
        for byte in digest {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

fn validate_reference(
    reference: &FrontierReference,
    field: &'static str,
    expected_kind: &'static str,
) -> Result<(), DimensionFrontierError> {
    if reference.kind != expected_kind {
        return Err(DimensionFrontierError::InvalidReference {
            field,
            reason: "unexpected reference kind",
        });
    }
    if reference.schema_version == 0 {
        return Err(DimensionFrontierError::InvalidReference {
            field,
            reason: "schema_version must be non-zero",
        });
    }
    if !is_sha256(&reference.artifact_digest) {
        return Err(DimensionFrontierError::InvalidReference {
            field,
            reason: "artifact_digest must be sha256: followed by 64 lowercase hex characters",
        });
    }
    if reference.artifact_id.is_empty() {
        return Err(DimensionFrontierError::InvalidReference {
            field,
            reason: "artifact_id is empty",
        });
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value.as_bytes());
}

fn push_reference(bytes: &mut Vec<u8>, reference: &FrontierReference) {
    push_string(bytes, &reference.kind);
    push_u32(bytes, reference.schema_version);
    push_string(bytes, &reference.artifact_digest);
    push_string(bytes, &reference.artifact_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(kind: &str, id: &str) -> FrontierReference {
        FrontierReference {
            kind: kind.into(),
            schema_version: 1,
            artifact_digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
            artifact_id: id.into(),
        }
    }

    fn manifest() -> DimensionFrontierManifest {
        DimensionFrontierManifest {
            schema_version: DIMENSION_FRONTIER_SCHEMA_VERSION,
            task: "hdc_retrieval".into(),
            scenario_set: "fixture-v1".into(),
            scenario_revision: "sha256:scenario".into(),
            split: "held-out".into(),
            protocol: "top-k-10".into(),
            representation: "continuous_f32".into(),
            model_revision: "model-v1".into(),
            rows: vec![
                DimensionFrontierRow {
                    resolution: 1_024,
                    task_quality: reference("task_quality", "quality-1k"),
                    performance: reference("performance", "perf-1k"),
                    resource: reference("resource", "resource-1k"),
                    structural_sweep: Some(reference("dimension_sweep", "sweep-v1")),
                },
                DimensionFrontierRow {
                    resolution: 16_384,
                    task_quality: reference("task_quality", "quality-16k"),
                    performance: reference("performance", "perf-16k"),
                    resource: reference("resource", "resource-16k"),
                    structural_sweep: None,
                },
            ],
        }
    }

    #[test]
    fn complete_manifest_validates() {
        manifest().validate().unwrap();
        assert_eq!(manifest().artifact_digest().len(), 71);
    }

    #[test]
    fn identity_changes_when_an_artifact_changes() {
        let a = manifest().artifact_digest();
        let mut b = manifest();
        b.rows[0].task_quality.artifact_digest =
            "sha256:1111111111111111111111111111111111111111111111111111111111111111".into();
        assert_ne!(a, b.artifact_digest());
    }

    #[test]
    fn duplicate_dimensions_fail_closed() {
        let mut m = manifest();
        m.rows[1].resolution = 1_024;
        assert_eq!(
            m.validate(),
            Err(DimensionFrontierError::DuplicateDimension(1_024))
        );
    }

    #[test]
    fn descending_dimensions_fail_closed() {
        let mut m = manifest();
        m.rows[1].resolution = 512;
        assert_eq!(m.validate(), Err(DimensionFrontierError::UnsortedDimensions));
    }

    #[test]
    fn malformed_reference_fails_closed() {
        let mut m = manifest();
        m.rows[0].resource.artifact_digest = "blake3:fixture".into();
        assert!(matches!(
            m.validate(),
            Err(DimensionFrontierError::InvalidReference {
                field: "resource",
                ..
            })
        ));
    }

    #[test]
    fn structural_sweep_is_optional() {
        let mut m = manifest();
        m.rows[0].structural_sweep = None;
        m.validate().unwrap();
    }
}
