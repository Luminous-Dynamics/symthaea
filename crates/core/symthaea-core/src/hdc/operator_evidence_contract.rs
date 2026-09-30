//! Feature-neutral evidence contract for HDC operator qualification.
//!
//! This module deliberately contains no SIMD implementation. It is the stable
//! boundary consumed by trajectory qualification: numerical operator evidence
//! can be produced by one implementation and consumed by another without
//! making the trajectory layer depend on a particular execution backend.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const OPERATOR_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const CONTINUOUS_F32_REPRESENTATION: &str = "continuous_f32";
pub const QUALIFIED_STATUS: &str = "qualified";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEvidenceRecord {
    pub schema_version: u32,
    pub representation: String,
    pub resolution: usize,
    pub operation: String,
    pub seed_a: u64,
    pub seed_b: u64,
    pub scalar_reference: f64,
    pub simd_result: f64,
    pub abs_error: f64,
    pub relative_error: f64,
    pub max_abs_error: f64,
    pub max_relative_error: f64,
    pub tolerance: f64,
    pub logical_bytes: usize,
    pub qualification_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEvidenceSummary {
    pub schema_version: u32,
    pub expected_records: usize,
    pub observed_records: usize,
    pub qualified_records: usize,
    pub failed_records: usize,
    pub missing_records: usize,
    pub duplicate_records: usize,
    pub qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEvidenceDigest {
    pub algorithm: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEvidenceArtifact {
    pub schema_version: u32,
    pub representation: String,
    pub digest: OperatorEvidenceDigest,
    pub records: Vec<OperatorEvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorEvidenceError {
    DigestMismatch { expected: String, observed: String },
    InvalidDigestFormat(String),
    Json(String),
    UnsupportedSchema(u32),
}

impl std::fmt::Display for OperatorEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DigestMismatch { expected, observed } => {
                write!(f, "operator evidence digest mismatch: expected {expected}, observed {observed}")
            }
            Self::InvalidDigestFormat(value) => write!(f, "invalid SHA-256 digest: {value}"),
            Self::Json(error) => write!(f, "invalid operator evidence JSON: {error}"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported operator evidence schema: {version}")
            }
        }
    }
}

impl std::error::Error for OperatorEvidenceError {}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn verify_artifact_bytes(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<OperatorEvidenceArtifact, OperatorEvidenceError> {
    validate_digest(expected_sha256)?;
    let observed = sha256_hex(bytes);
    if !observed.eq_ignore_ascii_case(expected_sha256) {
        return Err(OperatorEvidenceError::DigestMismatch {
            expected: expected_sha256.to_ascii_lowercase(),
            observed,
        });
    }

    let records: Vec<OperatorEvidenceRecord> =
        serde_json::from_slice(bytes).map_err(|error| OperatorEvidenceError::Json(error.to_string()))?;
    if let Some(record) = records.iter().find(|record| record.schema_version != OPERATOR_EVIDENCE_SCHEMA_VERSION) {
        return Err(OperatorEvidenceError::UnsupportedSchema(record.schema_version));
    }
    Ok(OperatorEvidenceArtifact {
        schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
        representation: CONTINUOUS_F32_REPRESENTATION.to_owned(),
        digest: OperatorEvidenceDigest {
            algorithm: "sha256".to_owned(),
            value: observed,
        },
        records,
    })
}

fn validate_digest(value: &str) -> Result<(), OperatorEvidenceError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OperatorEvidenceError::InvalidDigestFormat(value.to_owned()));
    }
    Ok(())
}

pub fn qualify_records(records: &[OperatorEvidenceRecord]) -> OperatorEvidenceSummary {
    const DIMS: &[usize] = &[16_384, 32_768, 65_536, 131_072, 262_144];
    const OPS: &[&str] = &["dot", "bind", "bundle", "norm", "similarity"];

    let expected = DIMS.len() * OPS.len();
    let mut seen = BTreeSet::new();
    let mut qualified = 0;
    let mut failed = 0;
    let mut duplicate = 0;

    for record in records {
        let key = (record.resolution, record.operation);
        if !seen.insert(key) {
            duplicate += 1;
        }

        if record.schema_version == OPERATOR_EVIDENCE_SCHEMA_VERSION
            && record.representation == CONTINUOUS_F32_REPRESENTATION
            && record.qualification_status == QUALIFIED_STATUS
        {
            qualified += 1;
        } else {
            failed += 1;
        }
    }

    let observed = seen.len();
    let missing = expected.saturating_sub(observed);

    OperatorEvidenceSummary {
        schema_version: OPERATOR_EVIDENCE_SCHEMA_VERSION,
        expected_records: expected,
        observed_records: observed,
        qualified_records: qualified,
        failed_records: failed,
        missing_records: missing,
        duplicate_records: duplicate,
        qualified: records.len() == expected
            && observed == expected
            && qualified == expected
            && failed == 0
            && duplicate == 0
            && missing == 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records() -> Vec<OperatorEvidenceRecord> {
        [16_384usize, 32_768, 65_536, 131_072, 262_144]
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
            .collect()
    }

    fn artifact_bytes() -> Vec<u8> {
        serde_json::to_vec(&records()).expect("fixture serializes")
    }

    #[test]
    fn digest_round_trip_verifies_exact_bytes() {
        let bytes = artifact_bytes();
        let digest = sha256_hex(&bytes);
        let artifact = verify_artifact_bytes(&bytes, &digest).expect("digest verifies");
        assert_eq!(artifact.records.len(), 25);
        assert_eq!(qualify_records(&artifact.records).qualified, true);
    }

    #[test]
    fn digest_mismatch_fails_closed() {
        let bytes = artifact_bytes();
        let error = verify_artifact_bytes(&bytes, &"0".repeat(64)).expect_err("mismatch must reject");
        assert!(matches!(error, OperatorEvidenceError::DigestMismatch { .. }));
    }

    #[test]
    fn malformed_digest_fails_before_parsing() {
        let bytes = artifact_bytes();
        let error = verify_artifact_bytes(&bytes, "not-a-digest").expect_err("format must reject");
        assert!(matches!(error, OperatorEvidenceError::InvalidDigestFormat(_)));
    }

    #[test]
    fn incomplete_matrix_is_not_qualified() {
        let mut rows = records();
        rows.pop();
        let summary = qualify_records(&rows);
        assert_eq!(summary.missing_records, 1);
        assert!(!summary.qualified);
    }

    #[test]
    fn duplicate_matrix_cell_is_not_qualified() {
        let mut rows = records();
        rows.push(rows[0].clone());
        let summary = qualify_records(&rows);
        assert_eq!(summary.duplicate_records, 1);
        assert!(!summary.qualified);
    }
}
