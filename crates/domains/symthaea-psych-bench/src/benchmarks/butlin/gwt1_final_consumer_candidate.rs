// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-authoritative reconstruction adapter for the trusted final consumer.
//!
//! The surrounding consumer workflow establishes external authority by
//! verifying the final archive attestation and bounded-admitting the final and
//! nested direct archives. This module then replays the scientific result from
//! those admitted bytes and returns a one-way reporting projection. Calling this
//! function outside the trusted workflow does not mint authority; only the
//! separately attested consumer/report artifact is durable external authority.

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Serialize, de::DeserializeOwned};

use super::gwt1_evidence_disposition::Gwt1EvidenceDispositionSummaryV1;
use super::gwt1_trusted_resolution::{
    Gwt1TrustedResolutionCandidateErrorV1, Gwt1TrustedResolutionCandidateV1,
    generate_gwt1_trusted_resolution_candidate_v1,
};
use super::gwt1_verified_final_artifact::{
    Gwt1FinalArtifactReconstructionErrorV1, verify_gwt1_final_reconstruction_v1,
};
use super::gwt1_verified_report_projection::{
    Gwt1VerifiedReportProjectionV1, project_gwt1_verified_report_v1,
};
use super::report::ButlinIndicatorReport;
use super::resolution_view_v2::ButlinResolvedEvidenceViewV2;

pub const GWT1_CONSUMER_GH_VERSION_V1: &str = "2.100.0";
pub const GWT1_CONSUMER_GH_ARCHIVE_SHA256_V1: &str =
    "e4d4bb4498e8d007abe545b6568926793ace1b6447da598294a610018cb164be";
pub const GWT1_CONSUMER_GH_BINARY_SHA256_V1: &str =
    "553949e2efa12842771efe6012aa4de21f1d591530ec17fc435f610f10e017ee";
pub const GWT1_CONSUMER_GH_VERSION_OUTPUT_SHA256_V1: &str =
    "2ec8b2f6e0e8e915f7e0c8d7b14b77fe93e33225872ea1d98041228950c620a9";

#[derive(Debug)]
pub enum Gwt1FinalConsumerCandidateErrorV1 {
    Io(String),
    Json {
        component: &'static str,
        error: String,
    },
    NonCanonicalJson {
        component: &'static str,
    },
    InvalidFinalRoot {
        observed: String,
    },
    EmptyInternalPromotionVerification,
    VerifierPathNotAbsolute {
        role: &'static str,
        observed: String,
    },
    VerifierPathInvalid {
        role: &'static str,
        observed: String,
    },
    MalformedSha256 {
        observed: String,
    },
    VerifierBinaryIdentityMismatch {
        observed: String,
    },
    VerifierVersionOutputIdentityMismatch {
        observed: String,
    },
    VerifierVersionCommandFailed {
        stderr: String,
    },
    Recompute(Gwt1TrustedResolutionCandidateErrorV1),
    Reconstruction(Gwt1FinalArtifactReconstructionErrorV1),
}

impl std::fmt::Display for Gwt1FinalConsumerCandidateErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O failure: {error}"),
            Self::Json { component, error } => {
                write!(f, "invalid JSON for {component}: {error}")
            }
            Self::NonCanonicalJson { component } => write!(
                f,
                "stored {component} bytes are not the producer's canonical serde_json encoding"
            ),
            Self::InvalidFinalRoot { observed } => write!(
                f,
                "admitted final root must be an absolute non-symlink directory named gwt1-final-resolution, observed {observed:?}"
            ),
            Self::EmptyInternalPromotionVerification => write!(
                f,
                "stored inner promotion-verification transcript is empty"
            ),
            Self::VerifierPathNotAbsolute { role, observed } => {
                write!(f, "trusted {role} path must be absolute, observed {observed:?}")
            }
            Self::VerifierPathInvalid { role, observed } => write!(
                f,
                "trusted {role} path must be an ordinary non-symlink file, observed {observed:?}"
            ),
            Self::MalformedSha256 { observed } => {
                write!(f, "malformed SHA-256 digest {observed:?}")
            }
            Self::VerifierBinaryIdentityMismatch { observed } => write!(
                f,
                "GitHub CLI executable SHA-256 {observed:?} does not match the frozen consumer root"
            ),
            Self::VerifierVersionOutputIdentityMismatch { observed } => write!(
                f,
                "GitHub CLI version-output SHA-256 {observed:?} does not match the frozen consumer root"
            ),
            Self::VerifierVersionCommandFailed { stderr } => {
                write!(f, "GitHub CLI version command failed: {stderr}")
            }
            Self::Recompute(error) => write!(f, "final scientific reconstruction failed: {error}"),
            Self::Reconstruction(error) => {
                write!(f, "stored final artifact disagrees with reconstruction: {error}")
            }
        }
    }
}

impl std::error::Error for Gwt1FinalConsumerCandidateErrorV1 {}

impl From<Gwt1TrustedResolutionCandidateErrorV1> for Gwt1FinalConsumerCandidateErrorV1 {
    fn from(value: Gwt1TrustedResolutionCandidateErrorV1) -> Self {
        Self::Recompute(value)
    }
}

impl From<Gwt1FinalArtifactReconstructionErrorV1> for Gwt1FinalConsumerCandidateErrorV1 {
    fn from(value: Gwt1FinalArtifactReconstructionErrorV1) -> Self {
        Self::Reconstruction(value)
    }
}

fn validate_regular_executable(
    path: &Path,
    role: &'static str,
) -> Result<(), Gwt1FinalConsumerCandidateErrorV1> {
    if !path.is_absolute() {
        return Err(Gwt1FinalConsumerCandidateErrorV1::VerifierPathNotAbsolute {
            role,
            observed: path.display().to_string(),
        });
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Gwt1FinalConsumerCandidateErrorV1::VerifierPathInvalid {
            role,
            observed: path.display().to_string(),
        });
    }
    Ok(())
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(
    sha256sum_executable: &Path,
    bytes: &[u8],
) -> Result<String, Gwt1FinalConsumerCandidateErrorV1> {
    validate_regular_executable(sha256sum_executable, "SHA-256")?;
    let mut child = Command::new(sha256sum_executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| Gwt1FinalConsumerCandidateErrorV1::Io("missing sha256sum stdin".into()))?
        .write_all(bytes)
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    let output = child
        .wait_with_output()
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    if !output.status.success() {
        return Err(Gwt1FinalConsumerCandidateErrorV1::Io(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let digest = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| Gwt1FinalConsumerCandidateErrorV1::MalformedSha256 {
            observed: stdout.to_string(),
        })?
        .to_string();
    if !is_lower_hex_64(&digest) {
        return Err(Gwt1FinalConsumerCandidateErrorV1::MalformedSha256 {
            observed: digest,
        });
    }
    Ok(digest)
}

fn sha256_file(
    sha256sum_executable: &Path,
    path: &Path,
) -> Result<String, Gwt1FinalConsumerCandidateErrorV1> {
    let bytes = fs::read(path)
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    sha256_bytes(sha256sum_executable, &bytes)
}

fn validate_frozen_gh_root(
    gh_executable: &Path,
    sha256sum_executable: &Path,
) -> Result<(), Gwt1FinalConsumerCandidateErrorV1> {
    validate_regular_executable(gh_executable, "GitHub CLI")?;
    let binary_sha256 = sha256_file(sha256sum_executable, gh_executable)?;
    if binary_sha256 != GWT1_CONSUMER_GH_BINARY_SHA256_V1 {
        return Err(
            Gwt1FinalConsumerCandidateErrorV1::VerifierBinaryIdentityMismatch {
                observed: binary_sha256,
            },
        );
    }

    let output = Command::new(gh_executable)
        .arg("--version")
        .output()
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    if !output.status.success() {
        return Err(
            Gwt1FinalConsumerCandidateErrorV1::VerifierVersionCommandFailed {
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            },
        );
    }
    let mut version_bytes = output.stdout;
    while version_bytes.last() == Some(&b'\n') {
        version_bytes.pop();
    }
    let version_output_sha256 = sha256_bytes(sha256sum_executable, &version_bytes)?;
    if version_output_sha256 != GWT1_CONSUMER_GH_VERSION_OUTPUT_SHA256_V1 {
        return Err(
            Gwt1FinalConsumerCandidateErrorV1::VerifierVersionOutputIdentityMismatch {
                observed: version_output_sha256,
            },
        );
    }
    Ok(())
}

fn validate_admitted_final_root(
    path: &Path,
) -> Result<(), Gwt1FinalConsumerCandidateErrorV1> {
    if !path.is_absolute() || path.file_name().and_then(|name| name.to_str()) != Some("gwt1-final-resolution") {
        return Err(Gwt1FinalConsumerCandidateErrorV1::InvalidFinalRoot {
            observed: path.display().to_string(),
        });
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Gwt1FinalConsumerCandidateErrorV1::InvalidFinalRoot {
            observed: path.display().to_string(),
        });
    }
    Ok(())
}

fn read_canonical_json<T>(
    path: &Path,
    component: &'static str,
) -> Result<T, Gwt1FinalConsumerCandidateErrorV1>
where
    T: DeserializeOwned + Serialize,
{
    let bytes = fs::read(path)
        .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    let value: T = serde_json::from_slice(&bytes).map_err(|error| {
        Gwt1FinalConsumerCandidateErrorV1::Json {
            component,
            error: error.to_string(),
        }
    })?;
    let canonical = serde_json::to_vec(&value).map_err(|error| {
        Gwt1FinalConsumerCandidateErrorV1::Json {
            component,
            error: error.to_string(),
        }
    })?;
    if canonical != bytes {
        return Err(Gwt1FinalConsumerCandidateErrorV1::NonCanonicalJson { component });
    }
    Ok(value)
}

/// Reconstruct a report projection candidate from a bounded-admitted final root.
///
/// **Not an authority boundary.** The trusted consumer workflow must first
/// verify the final archive attestation and perform bounded archive admission,
/// and must separately attest the bytes emitted from this result before those
/// bytes are externally authoritative.
pub fn reconstruct_gwt1_verified_report_projection_candidate_v1(
    admitted_final_root: &Path,
    admitted_direct_evidence_dir: &Path,
    final_archive_sha256: &str,
    gh_executable: &Path,
    sha256sum_executable: &Path,
    gh_config_dir: &Path,
) -> Result<Gwt1VerifiedReportProjectionV1, Gwt1FinalConsumerCandidateErrorV1> {
    validate_admitted_final_root(admitted_final_root)?;
    validate_frozen_gh_root(gh_executable, sha256sum_executable)?;

    let stored_base: ButlinIndicatorReport = read_canonical_json(
        &admitted_final_root.join("base_report.json"),
        "base_report.json",
    )?;
    let stored_view: ButlinResolvedEvidenceViewV2 = read_canonical_json(
        &admitted_final_root.join("resolved_view_v2.json"),
        "resolved_view_v2.json",
    )?;
    let stored_disposition: Gwt1EvidenceDispositionSummaryV1 = read_canonical_json(
        &admitted_final_root.join("gwt1_evidence_disposition.json"),
        "gwt1_evidence_disposition.json",
    )?;
    let stored_candidate: Gwt1TrustedResolutionCandidateV1 = read_canonical_json(
        &admitted_final_root.join("resolution_candidate.json"),
        "resolution_candidate.json",
    )?;
    let stored_internal_verification = fs::read(
        admitted_final_root.join("authority/promotion/internal_verification.json"),
    )
    .map_err(|error| Gwt1FinalConsumerCandidateErrorV1::Io(error.to_string()))?;
    if stored_internal_verification.is_empty() {
        return Err(Gwt1FinalConsumerCandidateErrorV1::EmptyInternalPromotionVerification);
    }

    let recomputed = generate_gwt1_trusted_resolution_candidate_v1(
        admitted_direct_evidence_dir,
        &admitted_final_root.join("sources/promotion/promotion_capsule.json"),
        &admitted_final_root.join("authority/promotion/capsule_attestation.json"),
        gh_executable,
        sha256sum_executable,
        gh_config_dir,
    )?;

    let verified = verify_gwt1_final_reconstruction_v1(
        final_archive_sha256,
        &stored_base,
        &stored_view,
        &stored_disposition,
        &stored_candidate,
        &stored_internal_verification,
        &recomputed,
    )?;
    Ok(project_gwt1_verified_report_v1(&verified))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumer_verifier_root_constants_are_frozen() {
        assert_eq!(GWT1_CONSUMER_GH_VERSION_V1, "2.100.0");
        assert_eq!(GWT1_CONSUMER_GH_ARCHIVE_SHA256_V1.len(), 64);
        assert_eq!(GWT1_CONSUMER_GH_BINARY_SHA256_V1.len(), 64);
        assert_eq!(GWT1_CONSUMER_GH_VERSION_OUTPUT_SHA256_V1.len(), 64);
    }

    #[test]
    fn frozen_root_digests_are_lower_hex() {
        assert!(is_lower_hex_64(GWT1_CONSUMER_GH_ARCHIVE_SHA256_V1));
        assert!(is_lower_hex_64(GWT1_CONSUMER_GH_BINARY_SHA256_V1));
        assert!(is_lower_hex_64(GWT1_CONSUMER_GH_VERSION_OUTPUT_SHA256_V1));
    }
}
