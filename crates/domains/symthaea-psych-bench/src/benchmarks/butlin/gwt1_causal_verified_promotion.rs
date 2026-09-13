// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cryptographic authority boundary for GWT-1 causal promotion capsules.
//!
//! A [`Gwt1CausalPromotionCapsuleV1`] is only structured data. This module
//! produces an opaque [`VerifiedGwt1CausalPromotionV1`] only after invoking a
//! trusted GitHub attestation verifier against the frozen repository, signer
//! workflow and signer digest. No public bool, deserializable token, or ambient
//! `PATH` lookup can substitute for that verification step.
//!
//! The authority-minting function is intentionally crate-internal. External
//! callers may consume the opaque capability through the resolved-view API, but
//! cannot manufacture it from a locally selected executable. A trusted internal
//! workflow/binary must supply explicit absolute regular-file verifier paths and
//! an isolated GitHub CLI configuration directory.

use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use super::gwt1_causal_promotion_capsule::{
    GWT1_CAUSAL_TRUSTED_REPOSITORY_V1, Gwt1CausalPromotionCapsuleFailureV1,
    Gwt1CausalPromotionCapsuleV1, validate_gwt1_causal_promotion_capsule_v1,
};

pub const GWT1_CAUSAL_PROMOTION_WORKFLOW_V1: &str =
    ".github/workflows/butlin-gwt1-causal-trusted-promotion.yml";

/// Bootstrap candidate only. Before activation this must be updated to the
/// reviewed/landed #2510 promotion-workflow revision.
pub const GWT1_CAUSAL_APPROVED_PROMOTION_WORKFLOW_SHA_V1: &str =
    "8e8100bee2c1f850d7abc2ffee1d903600e2a0ab";

/// Bootstrap candidate only. Before activation this must be updated to the
/// reviewed/landed #2507 causal-builder revision.
pub const GWT1_CAUSAL_APPROVED_BUILDER_SHA_V1: &str =
    "afabae61fa9b93d033410fb58af56f41ec47f60f";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gwt1CausalPromotionVerificationErrorV1 {
    Io(String),
    CapsuleJson(String),
    CapsuleShape(Vec<Gwt1CausalPromotionCapsuleFailureV1>),
    WrongApprovedBuilder { observed: String },
    EmptyAttestationBundle,
    VerifierPathNotAbsolute { role: &'static str, observed: String },
    VerifierPathNotRegularFile { role: &'static str, observed: String },
    VerifierPathIsSymlink { role: &'static str, observed: String },
    VerifierConfigPathNotAbsolute { observed: String },
    VerifierConfigPathInvalid { observed: String },
    MissingGithubToken,
    AttestationCommandFailed { stderr: String },
    VerificationJson(String),
    MalformedSha256 { observed: String },
}

impl std::fmt::Display for Gwt1CausalPromotionVerificationErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O failure: {error}"),
            Self::CapsuleJson(error) => write!(f, "invalid promotion capsule JSON: {error}"),
            Self::CapsuleShape(failures) => {
                write!(f, "promotion capsule failed shape validation: {failures:?}")
            }
            Self::WrongApprovedBuilder { observed } => write!(
                f,
                "promotion capsule names causal builder {observed:?}, not the approved V1 builder"
            ),
            Self::EmptyAttestationBundle => write!(f, "promotion attestation bundle is empty"),
            Self::VerifierPathNotAbsolute { role, observed } => write!(
                f,
                "trusted {role} verifier path must be absolute, observed {observed:?}"
            ),
            Self::VerifierPathNotRegularFile { role, observed } => write!(
                f,
                "trusted {role} verifier path must be a regular file, observed {observed:?}"
            ),
            Self::VerifierPathIsSymlink { role, observed } => write!(
                f,
                "trusted {role} verifier path must not be a symlink, observed {observed:?}"
            ),
            Self::VerifierConfigPathNotAbsolute { observed } => write!(
                f,
                "trusted GitHub CLI config path must be absolute, observed {observed:?}"
            ),
            Self::VerifierConfigPathInvalid { observed } => write!(
                f,
                "trusted GitHub CLI config path must be a real directory, observed {observed:?}"
            ),
            Self::MissingGithubToken => write!(f, "trusted verifier requires non-empty GH_TOKEN"),
            Self::AttestationCommandFailed { stderr } => {
                write!(f, "promotion attestation verification failed: {stderr}")
            }
            Self::VerificationJson(error) => {
                write!(f, "promotion attestation verifier returned invalid JSON: {error}")
            }
            Self::MalformedSha256 { observed } => {
                write!(f, "malformed SHA-256 digest {observed:?}")
            }
        }
    }
}

impl std::error::Error for Gwt1CausalPromotionVerificationErrorV1 {}

/// Opaque proof that a promotion capsule passed the frozen V1 attestation gate.
///
/// Fields are private and this type is deliberately not serializable. Persisted
/// authority is the attestation bundle itself, not an instance of this struct.
/// There is intentionally no public constructor.
#[derive(Debug, Clone)]
pub struct VerifiedGwt1CausalPromotionV1 {
    capsule: Gwt1CausalPromotionCapsuleV1,
    capsule_sha256: String,
    capsule_byte_len: u64,
    promotion_attestation_bundle_sha256: String,
    promotion_attestation_verification_sha256: String,
    // Preserve the exact verifier stdout whose digest is carried into the V2
    // causal authority identity. This stays crate-private so retaining replay
    // evidence does not widen the authority-construction surface.
    promotion_attestation_verification_bytes: Vec<u8>,
}

impl VerifiedGwt1CausalPromotionV1 {
    pub fn capsule(&self) -> &Gwt1CausalPromotionCapsuleV1 {
        &self.capsule
    }

    pub fn capsule_sha256(&self) -> &str {
        &self.capsule_sha256
    }

    pub fn capsule_byte_len(&self) -> u64 {
        self.capsule_byte_len
    }

    pub fn promotion_attestation_bundle_sha256(&self) -> &str {
        &self.promotion_attestation_bundle_sha256
    }

    pub fn promotion_attestation_verification_sha256(&self) -> &str {
        &self.promotion_attestation_verification_sha256
    }

    pub(crate) fn promotion_attestation_verification_bytes(&self) -> &[u8] {
        &self.promotion_attestation_verification_bytes
    }
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_trusted_executable(
    path: &Path,
    role: &'static str,
) -> Result<(), Gwt1CausalPromotionVerificationErrorV1> {
    if !path.is_absolute() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::VerifierPathNotAbsolute {
            role,
            observed: path.display().to_string(),
        });
    }

    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;
    if metadata.file_type().is_symlink() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::VerifierPathIsSymlink {
            role,
            observed: path.display().to_string(),
        });
    }
    if !metadata.is_file() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::VerifierPathNotRegularFile {
            role,
            observed: path.display().to_string(),
        });
    }
    Ok(())
}

fn validate_trusted_config_dir(
    path: &Path,
) -> Result<(), Gwt1CausalPromotionVerificationErrorV1> {
    if !path.is_absolute() {
        return Err(
            Gwt1CausalPromotionVerificationErrorV1::VerifierConfigPathNotAbsolute {
                observed: path.display().to_string(),
            },
        );
    }
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::VerifierConfigPathInvalid {
            observed: path.display().to_string(),
        });
    }
    Ok(())
}

fn sha256_bytes(
    sha256sum_executable: &Path,
    bytes: &[u8],
) -> Result<String, Gwt1CausalPromotionVerificationErrorV1> {
    let mut child = Command::new(sha256sum_executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| {
            Gwt1CausalPromotionVerificationErrorV1::Io("missing sha256sum stdin".to_string())
        })?
        .write_all(bytes)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;

    let output = child
        .wait_with_output()
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;
    if !output.status.success() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::Io(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let digest = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| Gwt1CausalPromotionVerificationErrorV1::MalformedSha256 {
            observed: stdout.to_string(),
        })?
        .to_string();
    if !is_lower_hex_64(&digest) {
        return Err(Gwt1CausalPromotionVerificationErrorV1::MalformedSha256 {
            observed: digest,
        });
    }
    Ok(digest)
}

fn validate_capsule_authority_v1(
    capsule: &Gwt1CausalPromotionCapsuleV1,
) -> Result<(), Gwt1CausalPromotionVerificationErrorV1> {
    let failures = validate_gwt1_causal_promotion_capsule_v1(capsule);
    if !failures.is_empty() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::CapsuleShape(failures));
    }
    if capsule.trusted_builder_sha != GWT1_CAUSAL_APPROVED_BUILDER_SHA_V1 {
        return Err(Gwt1CausalPromotionVerificationErrorV1::WrongApprovedBuilder {
            observed: capsule.trusted_builder_sha.clone(),
        });
    }
    Ok(())
}

/// Verify a promotion capsule against the frozen V1 GitHub/Sigstore authority.
///
/// This is deliberately crate-internal: arbitrary library consumers cannot
/// select a local verifier and mint the opaque authority token. A trusted
/// internal workflow/binary must supply explicit absolute non-symlink regular
/// files for both GitHub CLI and SHA-256 plus an isolated GitHub CLI config
/// directory. The caller's `PATH` is therefore not part of authority resolution.
pub(crate) fn verify_gwt1_causal_promotion_package_v1(
    capsule_path: &Path,
    attestation_bundle_path: &Path,
    gh_executable: &Path,
    sha256sum_executable: &Path,
    gh_config_dir: &Path,
) -> Result<VerifiedGwt1CausalPromotionV1, Gwt1CausalPromotionVerificationErrorV1> {
    validate_trusted_executable(gh_executable, "GitHub CLI")?;
    validate_trusted_executable(sha256sum_executable, "SHA-256")?;
    validate_trusted_config_dir(gh_config_dir)?;

    let github_token = std::env::var("GH_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty())
        .ok_or(Gwt1CausalPromotionVerificationErrorV1::MissingGithubToken)?;

    let capsule_bytes = std::fs::read(capsule_path)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;
    let capsule: Gwt1CausalPromotionCapsuleV1 = serde_json::from_slice(&capsule_bytes)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::CapsuleJson(error.to_string()))?;
    validate_capsule_authority_v1(&capsule)?;

    let bundle_bytes = std::fs::read(attestation_bundle_path)
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;
    if bundle_bytes.is_empty() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::EmptyAttestationBundle);
    }

    let signer_workflow = format!(
        "{}/{}",
        GWT1_CAUSAL_TRUSTED_REPOSITORY_V1, GWT1_CAUSAL_PROMOTION_WORKFLOW_V1
    );
    let output = Command::new(gh_executable)
        .arg("attestation")
        .arg("verify")
        .arg(capsule_path)
        .arg("--repo")
        .arg(GWT1_CAUSAL_TRUSTED_REPOSITORY_V1)
        .arg("--bundle")
        .arg(attestation_bundle_path)
        .arg("--signer-workflow")
        .arg(signer_workflow)
        .arg("--signer-digest")
        .arg(GWT1_CAUSAL_APPROVED_PROMOTION_WORKFLOW_SHA_V1)
        .arg("--deny-self-hosted-runners")
        .arg("--format")
        .arg("json")
        .env("GH_TOKEN", github_token)
        .env("GH_CONFIG_DIR", gh_config_dir)
        .env("GH_NO_UPDATE_NOTIFIER", "1")
        .env("GH_PROMPT_DISABLED", "1")
        .env_remove("GH_HOST")
        .env_remove("GH_ENTERPRISE_TOKEN")
        .output()
        .map_err(|error| Gwt1CausalPromotionVerificationErrorV1::Io(error.to_string()))?;

    if !output.status.success() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::AttestationCommandFailed {
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    let verification: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| {
            Gwt1CausalPromotionVerificationErrorV1::VerificationJson(error.to_string())
        })?;
    if verification.is_null() {
        return Err(Gwt1CausalPromotionVerificationErrorV1::VerificationJson(
            "verification record is JSON null".to_string(),
        ));
    }

    let verification_bytes = output.stdout;
    let verification_sha256 = sha256_bytes(sha256sum_executable, &verification_bytes)?;

    Ok(VerifiedGwt1CausalPromotionV1 {
        capsule,
        capsule_sha256: sha256_bytes(sha256sum_executable, &capsule_bytes)?,
        capsule_byte_len: capsule_bytes.len() as u64,
        promotion_attestation_bundle_sha256: sha256_bytes(
            sha256sum_executable,
            &bundle_bytes,
        )?,
        promotion_attestation_verification_sha256: verification_sha256,
        promotion_attestation_verification_bytes: verification_bytes,
    })
}

#[cfg(test)]
pub(crate) fn verified_gwt1_causal_promotion_for_test(
    capsule: Gwt1CausalPromotionCapsuleV1,
) -> VerifiedGwt1CausalPromotionV1 {
    VerifiedGwt1CausalPromotionV1 {
        capsule,
        capsule_sha256: "a".repeat(64),
        capsule_byte_len: 256,
        promotion_attestation_bundle_sha256: "b".repeat(64),
        promotion_attestation_verification_sha256: "c".repeat(64),
        promotion_attestation_verification_bytes: b"{}".to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use super::*;
    use crate::benchmarks::butlin::{
        EvidenceOutcome, GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1,
        GWT1_CAUSAL_PROMOTION_POLICY_V1, GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1,
        Gwt1CausalQualificationOutcomeV1, Gwt1ExecutionIdentityV1, SupportTier,
    };

    fn capsule(builder_sha: &str) -> Gwt1CausalPromotionCapsuleV1 {
        Gwt1CausalPromotionCapsuleV1 {
            schema: GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1.to_string(),
            policy: GWT1_CAUSAL_PROMOTION_POLICY_V1.to_string(),
            indicator_id: "GWT-1".to_string(),
            repository: GWT1_CAUSAL_TRUSTED_REPOSITORY_V1.to_string(),
            trusted_builder_workflow: GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1.to_string(),
            trusted_builder_sha: builder_sha.to_string(),
            trusted_builder_ref: "trusted-ref".to_string(),
            causal_archive_sha256: "1".repeat(64),
            archive_attestation_bundle_sha256: "2".repeat(64),
            archive_attestation_verification_sha256: "3".repeat(64),
            evidence_subject: Gwt1ExecutionIdentityV1 {
                source_commit_sha: "4".repeat(40),
                source_tree_sha: "5".repeat(40),
                execution_run_id: "123/1".to_string(),
                toolchain: "rustc test".to_string(),
                specialist_blob_shas: BTreeMap::from([
                    ("drive_manager".to_string(), "6".repeat(40)),
                    ("memory_manager".to_string(), "7".repeat(40)),
                    ("learning_manager".to_string(), "8".repeat(40)),
                    ("perception_manager".to_string(), "9".repeat(40)),
                ]),
            },
            scientific_outcome: Gwt1CausalQualificationOutcomeV1::Qualified,
            eligible_evidence_outcome: EvidenceOutcome::Supported(SupportTier::CausallySupported),
            tier_ceiling: SupportTier::CausallySupported,
        }
    }

    #[test]
    fn wrong_causal_builder_is_rejected_before_crypto_verification() {
        let wrong = capsule(&"f".repeat(40));
        assert!(matches!(
            validate_capsule_authority_v1(&wrong),
            Err(Gwt1CausalPromotionVerificationErrorV1::WrongApprovedBuilder { .. })
        ));
    }

    #[test]
    fn approved_builder_passes_precrypto_authority_shape() {
        assert!(validate_capsule_authority_v1(&capsule(
            GWT1_CAUSAL_APPROVED_BUILDER_SHA_V1
        ))
        .is_ok());
    }

    #[test]
    fn relative_verifier_paths_are_rejected_before_execution() {
        assert!(matches!(
            validate_trusted_executable(Path::new("gh"), "GitHub CLI"),
            Err(Gwt1CausalPromotionVerificationErrorV1::VerifierPathNotAbsolute { .. })
        ));
        assert!(matches!(
            validate_trusted_executable(Path::new("sha256sum"), "SHA-256"),
            Err(Gwt1CausalPromotionVerificationErrorV1::VerifierPathNotAbsolute { .. })
        ));
    }

    #[test]
    fn relative_config_path_is_rejected() {
        assert!(matches!(
            validate_trusted_config_dir(Path::new("gh-config"),),
            Err(Gwt1CausalPromotionVerificationErrorV1::VerifierConfigPathNotAbsolute { .. })
        ));
    }
}
