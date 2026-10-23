// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed identity and consistency contract for external benchmark qualification.
//!
//! Developer integration tests may legitimately skip unavailable external assets.
//! Qualification may not: every requested benchmark must have a complete, content-
//! addressed record. Even then, this module proves only **record consistency**.
//! It does not prove that an evaluator process actually ran. Process execution,
//! materialized-asset verification, invocation identity, and terminal status are
//! the separate attestation layer tracked by #3384.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const EXTERNAL_QUALIFICATION_SCHEMA_VERSION: u32 = 1;
const MANIFEST_DOMAIN: &[u8] = b"symthaea-external-qualification-manifest-v1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea-external-qualification-consistency-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalAssetIdentity {
    /// BLAKE3 hex digest of the exact dataset/input corpus or canonical archive.
    pub dataset_digest: String,
    /// BLAKE3 hex digest of the exact evaluator implementation/package.
    pub evaluator_digest: String,
    /// BLAKE3 hex digest of the exact Symthaea adapter/protocol wrapper.
    pub adapter_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalBenchmarkRequirement {
    pub benchmark_id: String,
    pub assets: ExternalAssetIdentity,
    pub result_schema_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalQualificationManifest {
    pub schema_version: u32,
    pub qualification_id: String,
    pub benchmarks: Vec<ExternalBenchmarkRequirement>,
}

/// A caller-reported process disposition. This enum grants no execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalReportedDisposition {
    Completed,
    Skipped,
    NotRun,
    Failed,
}

/// A content-addressed *claim* about one requested benchmark result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalBenchmarkExecutionClaim {
    pub benchmark_id: String,
    pub reported_disposition: ExternalReportedDisposition,
    pub result_schema_id: String,
    /// BLAKE3 hex digest of the exact result artifact/canonical result bundle.
    pub result_digest: String,
}

/// Serializable consistency receipt for one declared external campaign.
///
/// The name is deliberate: this is not an execution attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalQualificationConsistencyReceipt {
    pub schema_version: u32,
    pub manifest_digest: String,
    /// Intended Symthaea code subject; actual process binding is a later layer.
    pub code_subject: String,
    pub execution_claims: Vec<ExternalBenchmarkExecutionClaim>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalQualificationError {
    UnsupportedSchema {
        found: u32,
    },
    EmptyManifest,
    NonCanonicalIdentifier {
        field: &'static str,
        value: String,
    },
    DuplicateBenchmarkId(String),
    InvalidDigest {
        field: &'static str,
        benchmark_id: Option<String>,
    },
    ManifestDigestMismatch,
    MissingExecutionClaim(String),
    UnknownExecutionClaim(String),
    DuplicateExecutionClaim(String),
    ExecutionNotReportedCompleted {
        benchmark_id: String,
        disposition: ExternalReportedDisposition,
    },
    ResultSchemaMismatch(String),
    ContentDigestMismatch,
}

impl fmt::Display for ExternalQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { found } => {
                write!(
                    f,
                    "unsupported external qualification schema version: {found}"
                )
            }
            Self::EmptyManifest => {
                write!(
                    f,
                    "external qualification manifest must request at least one benchmark"
                )
            }
            Self::NonCanonicalIdentifier { field, value } => {
                write!(f, "non-canonical {field}: {value:?}")
            }
            Self::DuplicateBenchmarkId(id) => write!(f, "duplicate external benchmark id: {id}"),
            Self::InvalidDigest {
                field,
                benchmark_id,
            } => {
                if let Some(id) = benchmark_id {
                    write!(f, "invalid BLAKE3 digest in {field} for benchmark {id}")
                } else {
                    write!(f, "invalid BLAKE3 digest in {field}")
                }
            }
            Self::ManifestDigestMismatch => {
                write!(
                    f,
                    "consistency receipt manifest digest does not match manifest"
                )
            }
            Self::MissingExecutionClaim(id) => {
                write!(f, "missing execution claim for requested benchmark: {id}")
            }
            Self::UnknownExecutionClaim(id) => {
                write!(f, "execution claim references unrequested benchmark: {id}")
            }
            Self::DuplicateExecutionClaim(id) => {
                write!(f, "duplicate execution claim for benchmark: {id}")
            }
            Self::ExecutionNotReportedCompleted {
                benchmark_id,
                disposition,
            } => write!(
                f,
                "benchmark {benchmark_id} is not reported completed: {disposition:?}"
            ),
            Self::ResultSchemaMismatch(id) => {
                write!(
                    f,
                    "result schema does not match manifest for benchmark: {id}"
                )
            }
            Self::ContentDigestMismatch => {
                write!(f, "materialized bytes do not match committed digest")
            }
        }
    }
}

impl std::error::Error for ExternalQualificationError {}

impl ExternalQualificationManifest {
    pub fn validate(&self) -> Result<(), ExternalQualificationError> {
        validate_schema(self.schema_version)?;
        validate_identifier("qualification_id", &self.qualification_id)?;
        if self.benchmarks.is_empty() {
            return Err(ExternalQualificationError::EmptyManifest);
        }

        let mut seen = BTreeSet::new();
        for benchmark in &self.benchmarks {
            validate_identifier("benchmark_id", &benchmark.benchmark_id)?;
            validate_identifier("result_schema_id", &benchmark.result_schema_id)?;
            if !seen.insert(benchmark.benchmark_id.as_str()) {
                return Err(ExternalQualificationError::DuplicateBenchmarkId(
                    benchmark.benchmark_id.clone(),
                ));
            }
            validate_digest(
                "dataset_digest",
                Some(&benchmark.benchmark_id),
                &benchmark.assets.dataset_digest,
            )?;
            validate_digest(
                "evaluator_digest",
                Some(&benchmark.benchmark_id),
                &benchmark.assets.evaluator_digest,
            )?;
            validate_digest(
                "adapter_digest",
                Some(&benchmark.benchmark_id),
                &benchmark.assets.adapter_digest,
            )?;
        }
        Ok(())
    }

    /// Order-independent, language-neutral canonical manifest encoding.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ExternalQualificationError> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MANIFEST_DOMAIN);
        push_u32(&mut bytes, self.schema_version);
        push_str(&mut bytes, &self.qualification_id);
        push_u64(&mut bytes, self.benchmarks.len() as u64);

        let mut benchmarks: Vec<&ExternalBenchmarkRequirement> = self.benchmarks.iter().collect();
        benchmarks.sort_by(|left, right| left.benchmark_id.cmp(&right.benchmark_id));
        for entry in benchmarks {
            push_str(&mut bytes, &entry.benchmark_id);
            push_str(&mut bytes, &entry.assets.dataset_digest);
            push_str(&mut bytes, &entry.assets.evaluator_digest);
            push_str(&mut bytes, &entry.assets.adapter_digest);
            push_str(&mut bytes, &entry.result_schema_id);
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<String, ExternalQualificationError> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }
}

impl ExternalQualificationConsistencyReceipt {
    pub fn validate_against(
        &self,
        manifest: &ExternalQualificationManifest,
    ) -> Result<(), ExternalQualificationError> {
        validate_schema(self.schema_version)?;
        validate_identifier("code_subject", &self.code_subject)?;
        validate_digest("manifest_digest", None, &self.manifest_digest)?;
        if self.manifest_digest != manifest.digest()? {
            return Err(ExternalQualificationError::ManifestDigestMismatch);
        }

        let required: BTreeMap<&str, &ExternalBenchmarkRequirement> = manifest
            .benchmarks
            .iter()
            .map(|entry| (entry.benchmark_id.as_str(), entry))
            .collect();
        let mut seen = BTreeSet::new();

        for claim in &self.execution_claims {
            validate_identifier("execution benchmark_id", &claim.benchmark_id)?;
            validate_identifier("execution result_schema_id", &claim.result_schema_id)?;
            if !seen.insert(claim.benchmark_id.as_str()) {
                return Err(ExternalQualificationError::DuplicateExecutionClaim(
                    claim.benchmark_id.clone(),
                ));
            }
            let requirement = required.get(claim.benchmark_id.as_str()).ok_or_else(|| {
                ExternalQualificationError::UnknownExecutionClaim(claim.benchmark_id.clone())
            })?;
            if claim.reported_disposition != ExternalReportedDisposition::Completed {
                return Err(ExternalQualificationError::ExecutionNotReportedCompleted {
                    benchmark_id: claim.benchmark_id.clone(),
                    disposition: claim.reported_disposition,
                });
            }
            if claim.result_schema_id != requirement.result_schema_id {
                return Err(ExternalQualificationError::ResultSchemaMismatch(
                    claim.benchmark_id.clone(),
                ));
            }
            validate_digest(
                "result_digest",
                Some(&claim.benchmark_id),
                &claim.result_digest,
            )?;
        }

        for benchmark_id in required.keys() {
            if !seen.contains(benchmark_id) {
                return Err(ExternalQualificationError::MissingExecutionClaim(
                    (*benchmark_id).to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(
        &self,
        manifest: &ExternalQualificationManifest,
    ) -> Result<Vec<u8>, ExternalQualificationError> {
        self.validate_against(manifest)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(RECEIPT_DOMAIN);
        push_u32(&mut bytes, self.schema_version);
        push_str(&mut bytes, &self.manifest_digest);
        push_str(&mut bytes, &self.code_subject);
        push_u64(&mut bytes, self.execution_claims.len() as u64);

        let mut claims: Vec<&ExternalBenchmarkExecutionClaim> =
            self.execution_claims.iter().collect();
        claims.sort_by(|left, right| left.benchmark_id.cmp(&right.benchmark_id));
        for claim in claims {
            push_str(&mut bytes, &claim.benchmark_id);
            bytes.push(match claim.reported_disposition {
                ExternalReportedDisposition::Completed => 0,
                ExternalReportedDisposition::Skipped => 1,
                ExternalReportedDisposition::NotRun => 2,
                ExternalReportedDisposition::Failed => 3,
            });
            push_str(&mut bytes, &claim.result_schema_id);
            push_str(&mut bytes, &claim.result_digest);
        }
        Ok(bytes)
    }

    pub fn digest(
        &self,
        manifest: &ExternalQualificationManifest,
    ) -> Result<String, ExternalQualificationError> {
        Ok(blake3::hash(&self.canonical_bytes(manifest)?)
            .to_hex()
            .to_string())
    }
}

/// Verify exact bytes against one digest committed by a manifest or receipt.
///
/// This helper proves byte equality only. It does not prove process execution.
pub fn verify_content_digest(
    expected_digest: &str,
    bytes: &[u8],
) -> Result<(), ExternalQualificationError> {
    validate_digest("expected_digest", None, expected_digest)?;
    let actual = blake3::hash(bytes).to_hex().to_string();
    if actual != expected_digest {
        return Err(ExternalQualificationError::ContentDigestMismatch);
    }
    Ok(())
}

fn validate_schema(schema_version: u32) -> Result<(), ExternalQualificationError> {
    if schema_version != EXTERNAL_QUALIFICATION_SCHEMA_VERSION {
        return Err(ExternalQualificationError::UnsupportedSchema {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), ExternalQualificationError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ExternalQualificationError::NonCanonicalIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_digest(
    field: &'static str,
    benchmark_id: Option<&str>,
    digest: &str,
) -> Result<(), ExternalQualificationError> {
    let valid = digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(ExternalQualificationError::InvalidDigest {
            field,
            benchmark_id: benchmark_id.map(str::to_owned),
        });
    }
    Ok(())
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn requirement(id: &str) -> ExternalBenchmarkRequirement {
        ExternalBenchmarkRequirement {
            benchmark_id: id.into(),
            assets: ExternalAssetIdentity {
                dataset_digest: digest(&format!("{id}:dataset")),
                evaluator_digest: digest(&format!("{id}:evaluator")),
                adapter_digest: digest(&format!("{id}:adapter")),
            },
            result_schema_id: "external-result-v1".into(),
        }
    }

    fn manifest() -> ExternalQualificationManifest {
        ExternalQualificationManifest {
            schema_version: EXTERNAL_QUALIFICATION_SCHEMA_VERSION,
            qualification_id: "external-campaign-v1".into(),
            benchmarks: vec![requirement("arc-agi-2"), requirement("gaia-dev")],
        }
    }

    fn receipt(
        manifest: &ExternalQualificationManifest,
    ) -> ExternalQualificationConsistencyReceipt {
        ExternalQualificationConsistencyReceipt {
            schema_version: EXTERNAL_QUALIFICATION_SCHEMA_VERSION,
            manifest_digest: manifest.digest().unwrap(),
            code_subject: "git:0123456789abcdef".into(),
            execution_claims: manifest
                .benchmarks
                .iter()
                .map(|entry| ExternalBenchmarkExecutionClaim {
                    benchmark_id: entry.benchmark_id.clone(),
                    reported_disposition: ExternalReportedDisposition::Completed,
                    result_schema_id: entry.result_schema_id.clone(),
                    result_digest: digest(&format!("{}:result", entry.benchmark_id)),
                })
                .collect(),
        }
    }

    #[test]
    fn manifest_is_order_independent_and_asset_bound() {
        let first = manifest();
        let mut reordered = first.clone();
        reordered.benchmarks.reverse();
        assert_eq!(first.digest().unwrap(), reordered.digest().unwrap());

        let mut changed = first.clone();
        changed.benchmarks[0].assets.dataset_digest = digest("different-dataset");
        assert_ne!(first.digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn duplicate_and_noncanonical_identifiers_fail_closed() {
        let mut duplicate = manifest();
        duplicate.benchmarks.push(requirement("arc-agi-2"));
        assert!(matches!(
            duplicate.validate(),
            Err(ExternalQualificationError::DuplicateBenchmarkId(id)) if id == "arc-agi-2"
        ));

        for bad in [" arc-agi-2", "arc-agi-2 ", "arc\nagi-2", "arc\tagi-2"] {
            let mut campaign = manifest();
            campaign.benchmarks[0].benchmark_id = bad.into();
            assert!(matches!(
                campaign.validate(),
                Err(ExternalQualificationError::NonCanonicalIdentifier {
                    field: "benchmark_id",
                    ..
                })
            ));
        }

        let mut campaign = manifest();
        campaign.qualification_id = "external\ncampaign".into();
        assert!(matches!(
            campaign.validate(),
            Err(ExternalQualificationError::NonCanonicalIdentifier {
                field: "qualification_id",
                ..
            })
        ));
    }

    #[test]
    fn receipt_rejects_missing_unknown_and_duplicate_claims() {
        let campaign = manifest();

        let mut missing = receipt(&campaign);
        missing.execution_claims.pop();
        assert!(matches!(
            missing.validate_against(&campaign),
            Err(ExternalQualificationError::MissingExecutionClaim(id)) if id == "gaia-dev"
        ));

        let mut unknown = receipt(&campaign);
        unknown
            .execution_claims
            .push(ExternalBenchmarkExecutionClaim {
                benchmark_id: "unknown-benchmark".into(),
                reported_disposition: ExternalReportedDisposition::Completed,
                result_schema_id: "external-result-v1".into(),
                result_digest: digest("unknown-result"),
            });
        assert!(matches!(
            unknown.validate_against(&campaign),
            Err(ExternalQualificationError::UnknownExecutionClaim(id)) if id == "unknown-benchmark"
        ));

        let mut duplicate = receipt(&campaign);
        duplicate
            .execution_claims
            .push(duplicate.execution_claims[0].clone());
        assert!(matches!(
            duplicate.validate_against(&campaign),
            Err(ExternalQualificationError::DuplicateExecutionClaim(id)) if id == "arc-agi-2"
        ));
    }

    #[test]
    fn every_noncompleted_reported_disposition_fails_closed() {
        let campaign = manifest();
        for disposition in [
            ExternalReportedDisposition::Skipped,
            ExternalReportedDisposition::NotRun,
            ExternalReportedDisposition::Failed,
        ] {
            let mut record = receipt(&campaign);
            record.execution_claims[0].reported_disposition = disposition;
            assert!(matches!(
                record.validate_against(&campaign),
                Err(ExternalQualificationError::ExecutionNotReportedCompleted {
                    benchmark_id,
                    disposition: found,
                }) if benchmark_id == "arc-agi-2" && found == disposition
            ));
        }
    }

    #[test]
    fn receipt_binds_manifest_result_schema_result_digest_and_code_subject() {
        let campaign = manifest();
        let record = receipt(&campaign);
        record.validate_against(&campaign).unwrap();

        let mut wrong_manifest = campaign.clone();
        wrong_manifest.benchmarks[0].assets.adapter_digest = digest("changed-adapter");
        assert_eq!(
            record.validate_against(&wrong_manifest),
            Err(ExternalQualificationError::ManifestDigestMismatch)
        );

        let mut wrong_schema = receipt(&campaign);
        wrong_schema.execution_claims[0].result_schema_id = "external-result-v2".into();
        assert!(matches!(
            wrong_schema.validate_against(&campaign),
            Err(ExternalQualificationError::ResultSchemaMismatch(id)) if id == "arc-agi-2"
        ));

        let mut invalid_result = receipt(&campaign);
        invalid_result.execution_claims[0].result_digest = "not-a-digest".into();
        assert!(matches!(
            invalid_result.validate_against(&campaign),
            Err(ExternalQualificationError::InvalidDigest {
                field: "result_digest",
                ..
            })
        ));

        let mut changed_subject = record.clone();
        changed_subject.code_subject = "git:fedcba9876543210".into();
        assert_eq!(record.manifest_digest, changed_subject.manifest_digest);
        assert_ne!(
            record.digest(&campaign).unwrap(),
            changed_subject.digest(&campaign).unwrap()
        );
    }

    #[test]
    fn content_digest_verification_is_exact_but_not_execution_attestation() {
        let bytes = b"exact external artifact bytes";
        let expected = blake3::hash(bytes).to_hex().to_string();
        verify_content_digest(&expected, bytes).unwrap();
        assert_eq!(
            verify_content_digest(&expected, b"tampered"),
            Err(ExternalQualificationError::ContentDigestMismatch)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_consistency_identity() {
        let campaign = manifest();
        let record = receipt(&campaign);
        let manifest_json = serde_json::to_vec(&campaign).unwrap();
        let receipt_json = serde_json::to_vec(&record).unwrap();
        let decoded_manifest: ExternalQualificationManifest =
            serde_json::from_slice(&manifest_json).unwrap();
        let decoded_receipt: ExternalQualificationConsistencyReceipt =
            serde_json::from_slice(&receipt_json).unwrap();
        assert_eq!(
            campaign.digest().unwrap(),
            decoded_manifest.digest().unwrap()
        );
        assert_eq!(
            record.digest(&campaign).unwrap(),
            decoded_receipt.digest(&decoded_manifest).unwrap()
        );
    }
}
