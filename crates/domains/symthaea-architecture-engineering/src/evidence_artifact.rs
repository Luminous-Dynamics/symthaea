//! Content-addressed evidence artifact manifests and byte-verification receipts.
//!
//! This module verifies that bytes retrieved by an external content-addressed store match
//! a frozen manifest. It does not retrieve bytes, authenticate a remote store, validate
//! signatures, or decide whether an artifact proves a scientific claim. Those are separate
//! boundaries. A byte-verification receipt only establishes exact byte/digest/manifest
//! correspondence for the bytes supplied to this verifier.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.architecture-engineering.evidence-artifact";
const VERSION: &str = "0.1.0";
const RECEIPT_SCHEMA: &str = "symthaea.architecture-engineering.evidence-artifact-byte-receipt";
const RECEIPT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceArtifactError {
    pub path: String,
    pub message: String,
}

impl EvidenceArtifactError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for EvidenceArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for EvidenceArtifactError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceArtifactRole {
    CandidateSource,
    BuildProvenance,
    EvaluationOutput,
    EvaluationLog,
    ConstraintReport,
    VerifierReceipt,
    HeldOutResult,
    ReproductionBundle,
    Other,
}

/// Reference metadata committed before byte retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceArtifactReference {
    /// Stable identifier within one candidate manifest.
    pub artifact_id: String,
    /// Role explains why this artifact is retained; it does not authenticate the content.
    pub role: EvidenceArtifactRole,
    /// Stable media type such as application/json or application/octet-stream.
    pub media_type: String,
    /// Store-specific content-addressed locator. It is metadata, not a trusted URL.
    pub storage_locator: String,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceArtifactManifest {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub source_commit: String,
    /// Sorted by artifact_id by the constructor; order is part of the canonical manifest.
    pub artifacts: Vec<EvidenceArtifactReference>,
}

#[derive(Serialize)]
struct ManifestDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    run_id: &'a str,
    candidate_id: &'a str,
    source_commit: &'a str,
    artifacts: &'a [EvidenceArtifactReference],
}

impl EvidenceArtifactManifest {
    pub fn new(
        run_id: impl Into<String>,
        candidate_id: impl Into<String>,
        source_commit: impl Into<String>,
        mut artifacts: Vec<EvidenceArtifactReference>,
    ) -> Result<Self, EvidenceArtifactError> {
        artifacts.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
        let manifest = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            run_id: run_id.into(),
            candidate_id: candidate_id.into(),
            source_commit: source_commit.into(),
            artifacts,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), EvidenceArtifactError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(EvidenceArtifactError::new(
                "manifest.schema",
                "unsupported evidence-artifact schema/version",
            ));
        }
        for (path, value) in [
            ("run_id", self.run_id.as_str()),
            ("candidate_id", self.candidate_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(EvidenceArtifactError::new(
                    format!("manifest.{path}"),
                    "value must be non-empty",
                ));
            }
        }
        validate_commit(self.source_commit.as_str(), "manifest.source_commit")?;
        if self.artifacts.is_empty() {
            return Err(EvidenceArtifactError::new(
                "manifest.artifacts",
                "at least one retained artifact is required",
            ));
        }

        let mut artifact_ids = BTreeSet::new();
        let mut previous_id: Option<&str> = None;
        for (index, artifact) in self.artifacts.iter().enumerate() {
            let path = format!("manifest.artifacts[{index}]");
            if artifact.artifact_id.trim().is_empty()
                || !artifact_ids.insert(artifact.artifact_id.as_str())
            {
                return Err(EvidenceArtifactError::new(
                    format!("{path}.artifact_id"),
                    "artifact IDs must be non-empty and unique",
                ));
            }
            if previous_id.is_some_and(|previous| previous >= artifact.artifact_id.as_str()) {
                return Err(EvidenceArtifactError::new(
                    format!("{path}.artifact_id"),
                    "artifacts must be sorted by artifact_id",
                ));
            }
            previous_id = Some(artifact.artifact_id.as_str());
            if artifact.media_type.trim().is_empty() || artifact.storage_locator.trim().is_empty() {
                return Err(EvidenceArtifactError::new(
                    &path,
                    "media_type and storage_locator must be non-empty",
                ));
            }
            validate_sha256(&artifact.sha256, &format!("{path}.sha256"))?;
        }
        Ok(())
    }

    pub fn manifest_sha256(&self) -> Result<String, EvidenceArtifactError> {
        self.validate()?;
        digest_serializable(&ManifestDigestPayload {
            schema: &self.schema,
            version: &self.version,
            run_id: &self.run_id,
            candidate_id: &self.candidate_id,
            source_commit: &self.source_commit,
            artifacts: &self.artifacts,
        })
    }
}

/// Resolved bytes handed to the verifier by a CAS/object-store adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEvidenceArtifact {
    pub artifact_id: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedArtifactBytes {
    pub artifact_id: String,
    pub role: EvidenceArtifactRole,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceArtifactByteReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub source_commit: String,
    pub verifier_id: String,
    pub verifier_build_sha256: String,
    pub manifest_sha256: String,
    pub resolved_artifacts: Vec<VerifiedArtifactBytes>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ReceiptDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    run_id: &'a str,
    candidate_id: &'a str,
    source_commit: &'a str,
    verifier_id: &'a str,
    verifier_build_sha256: &'a str,
    manifest_sha256: &'a str,
    resolved_artifacts: &'a [VerifiedArtifactBytes],
}

/// Verify the exact bytes supplied by a resolver against every artifact in a frozen manifest.
///
/// The resolved set must be complete and exact: no missing references, duplicate artifact IDs,
/// or unmanifested extra bytes. The returned receipt attests byte correspondence only.
pub fn verify_evidence_artifact_bytes(
    manifest: &EvidenceArtifactManifest,
    verifier_id: impl Into<String>,
    verifier_build_sha256: impl Into<String>,
    resolved: &[ResolvedEvidenceArtifact],
) -> Result<EvidenceArtifactByteReceipt, EvidenceArtifactError> {
    manifest.validate()?;
    let verifier_id = verifier_id.into();
    let verifier_build_sha256 = verifier_build_sha256.into();
    if verifier_id.trim().is_empty() {
        return Err(EvidenceArtifactError::new("verifier_id", "verifier identity must be non-empty"));
    }
    validate_sha256(&verifier_build_sha256, "verifier_build_sha256")?;
    if resolved.len() != manifest.artifacts.len() {
        return Err(EvidenceArtifactError::new(
            "resolved",
            "resolved artifact count must exactly match the manifest",
        ));
    }

    let mut resolved_by_id = BTreeMap::new();
    for (index, artifact) in resolved.iter().enumerate() {
        if artifact.artifact_id.trim().is_empty()
            || resolved_by_id.insert(artifact.artifact_id.as_str(), artifact).is_some()
        {
            return Err(EvidenceArtifactError::new(
                format!("resolved[{index}].artifact_id"),
                "resolved artifact IDs must be non-empty and unique",
            ));
        }
    }

    let mut verified_artifacts = Vec::with_capacity(manifest.artifacts.len());
    for (index, reference) in manifest.artifacts.iter().enumerate() {
        let bytes = resolved_by_id.get(reference.artifact_id.as_str()).ok_or_else(|| {
            EvidenceArtifactError::new(
                format!("manifest.artifacts[{index}].artifact_id"),
                "manifested artifact bytes were not supplied",
            )
        })?;
        let actual_length = u64::try_from(bytes.bytes.len())
            .map_err(|_| EvidenceArtifactError::new(
                format!("resolved.{}.byte_length", reference.artifact_id),
                "byte length exceeds u64",
            ))?;
        if actual_length != reference.byte_length {
            return Err(EvidenceArtifactError::new(
                format!("resolved.{}.byte_length", reference.artifact_id),
                "resolved byte length differs from the frozen manifest",
            ));
        }
        let actual_sha256 = digest_bytes(&bytes.bytes);
        if actual_sha256 != reference.sha256 {
            return Err(EvidenceArtifactError::new(
                format!("resolved.{}.sha256", reference.artifact_id),
                "resolved bytes do not match the manifested SHA-256 digest",
            ));
        }
        verified_artifacts.push(VerifiedArtifactBytes {
            artifact_id: reference.artifact_id.clone(),
            role: reference.role,
            byte_length: actual_length,
            sha256: actual_sha256,
        });
    }

    let manifest_sha256 = manifest.manifest_sha256()?;
    let payload = ReceiptDigestPayload {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        run_id: &manifest.run_id,
        candidate_id: &manifest.candidate_id,
        source_commit: &manifest.source_commit,
        verifier_id: &verifier_id,
        verifier_build_sha256: &verifier_build_sha256,
        manifest_sha256: &manifest_sha256,
        resolved_artifacts: &verified_artifacts,
    };
    let receipt_sha256 = digest_serializable(&payload)?;

    Ok(EvidenceArtifactByteReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: RECEIPT_VERSION.to_owned(),
        run_id: manifest.run_id.clone(),
        candidate_id: manifest.candidate_id.clone(),
        source_commit: manifest.source_commit.clone(),
        verifier_id,
        verifier_build_sha256,
        manifest_sha256,
        resolved_artifacts: verified_artifacts,
        receipt_sha256,
    })
}

pub fn verify_evidence_artifact_byte_receipt(
    manifest: &EvidenceArtifactManifest,
    verifier_id: &str,
    verifier_build_sha256: &str,
    resolved: &[ResolvedEvidenceArtifact],
    receipt: &EvidenceArtifactByteReceipt,
) -> Result<(), EvidenceArtifactError> {
    let reconstructed = verify_evidence_artifact_bytes(
        manifest,
        verifier_id,
        verifier_build_sha256,
        resolved,
    )?;
    if &reconstructed != receipt {
        return Err(EvidenceArtifactError::new(
            "receipt",
            "byte-verification receipt does not reconstruct from the manifest and supplied bytes",
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), EvidenceArtifactError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(EvidenceArtifactError::new(
            path,
            "expected lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_commit(value: &str, path: &str) -> Result<(), EvidenceArtifactError> {
    if !matches!(value.len(), 40 | 64)
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EvidenceArtifactError::new(
            path,
            "expected lowercase 40- or 64-character source commit ID",
        ));
    }
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, EvidenceArtifactError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        EvidenceArtifactError::new("serialization", format!("cannot serialize artifact receipt: {error}"))
    })?;
    Ok(digest_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const VERIFIER_BUILD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn reference(artifact_id: &str, role: EvidenceArtifactRole, bytes: &[u8]) -> EvidenceArtifactReference {
        EvidenceArtifactReference {
            artifact_id: artifact_id.to_owned(),
            role,
            media_type: "application/octet-stream".to_owned(),
            storage_locator: format!("sha256:{}", digest_bytes(bytes)),
            byte_length: bytes.len() as u64,
            sha256: digest_bytes(bytes),
        }
    }

    fn manifest() -> EvidenceArtifactManifest {
        EvidenceArtifactManifest::new(
            "run-1",
            "candidate-1",
            COMMIT,
            vec![
                reference("candidate-source", EvidenceArtifactRole::CandidateSource, b"fn main() {}"),
                reference("test-log", EvidenceArtifactRole::EvaluationLog, b"all tests passed"),
            ],
        ).unwrap()
    }

    fn resolved() -> Vec<ResolvedEvidenceArtifact> {
        vec![
            ResolvedEvidenceArtifact {
                artifact_id: "candidate-source".to_owned(),
                bytes: b"fn main() {}".to_vec(),
            },
            ResolvedEvidenceArtifact {
                artifact_id: "test-log".to_owned(),
                bytes: b"all tests passed".to_vec(),
            },
        ]
    }

    #[test]
    fn manifests_are_canonical_and_byte_receipts_replay() {
        let first = manifest();
        let second = EvidenceArtifactManifest::new(
            "run-1",
            "candidate-1",
            COMMIT,
            vec![
                reference("test-log", EvidenceArtifactRole::EvaluationLog, b"all tests passed"),
                reference("candidate-source", EvidenceArtifactRole::CandidateSource, b"fn main() {}"),
            ],
        ).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.manifest_sha256().unwrap(), second.manifest_sha256().unwrap());

        let bytes = resolved();
        let receipt = verify_evidence_artifact_bytes(
            &first,
            "independent-verifier",
            VERIFIER_BUILD,
            &bytes,
        ).unwrap();
        assert_eq!(receipt.resolved_artifacts.len(), 2);
        verify_evidence_artifact_byte_receipt(
            &first,
            "independent-verifier",
            VERIFIER_BUILD,
            &bytes,
            &receipt,
        ).unwrap();

        let mut tampered = receipt.clone();
        tampered.resolved_artifacts[0].byte_length += 1;
        assert!(verify_evidence_artifact_byte_receipt(
            &first,
            "independent-verifier",
            VERIFIER_BUILD,
            &bytes,
            &tampered,
        ).is_err());
    }

    #[test]
    fn modified_bytes_are_rejected_even_when_length_is_unchanged() {
        let manifest = manifest();
        let mut bytes = resolved();
        bytes[0].bytes[0] = b'F';
        assert!(verify_evidence_artifact_bytes(
            &manifest,
            "independent-verifier",
            VERIFIER_BUILD,
            &bytes,
        ).is_err());
    }

    #[test]
    fn missing_extra_and_duplicate_resolved_artifacts_are_rejected() {
        let manifest = manifest();
        let mut bytes = resolved();
        bytes.pop();
        assert!(verify_evidence_artifact_bytes(
            &manifest, "verifier", VERIFIER_BUILD, &bytes,
        ).is_err());

        let mut bytes = resolved();
        bytes.push(ResolvedEvidenceArtifact {
            artifact_id: "unmanifested".to_owned(),
            bytes: Vec::new(),
        });
        assert!(verify_evidence_artifact_bytes(
            &manifest, "verifier", VERIFIER_BUILD, &bytes,
        ).is_err());

        let mut bytes = resolved();
        bytes.push(bytes[0].clone());
        assert!(verify_evidence_artifact_bytes(
            &manifest, "verifier", VERIFIER_BUILD, &bytes,
        ).is_err());
    }

    #[test]
    fn invalid_manifests_and_verifier_build_identity_are_rejected() {
        let mut invalid = manifest();
        invalid.artifacts.swap(0, 1);
        assert!(invalid.validate().is_err());

        assert!(verify_evidence_artifact_bytes(
            &manifest(), "verifier", "not-a-digest", &resolved(),
        ).is_err());
    }
}
