// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing HTTPS acquisition adapter for the historical OQMD snapshot.
//!
//! This crate executes only the download half of the historical-data pipeline. It
//! composes the raw `symthaea-process-capture` boundary with MAG-DATA-006 receipts.
//! It deliberately does not claim provider authenticity merely because HTTPS succeeds.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use symthaea_materials_historical_extraction::OqmdExtractionProtocol;
use symthaea_materials_snapshot_acquisition::{
    AcquisitionRoute, HistoricalSnapshotAcquisitionReceipt, StreamedSnapshotIdentity,
    hash_snapshot_stream,
};
use symthaea_process_capture::{
    EnvironmentPolicy, ProcessCapture, ProcessSpec, capture_process,
};
use thiserror::Error;

const OFFICIAL_DOWNLOAD_PAGE: &str = "https://oqmd.org/download/";
const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_DIAGNOSTIC_BYTES: usize = 4 * 1024 * 1024;

/// Preregistered local/runtime inputs for one OQMD HTTPS acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdHttpsAcquisitionProfile {
    /// Profile schema version.
    pub schema_version: u32,
    /// Exact MAG-DATA-003 historical extraction protocol.
    pub protocol_sha256: String,
    /// Canonical OQMD download-page locator whose captured bytes informed this run.
    pub source_listing_url: String,
    /// Local path to the captured official download-page artifact.
    pub source_listing_artifact_path: String,
    /// SHA-256 of those exact captured source-listing bytes.
    pub source_listing_artifact_sha256: String,
    /// Exact HTTPS archive URL selected from the source listing.
    pub requested_url: String,
    /// Absolute path to the curl executable.
    pub curl_executable_path: String,
    /// Exact curl executable SHA-256.
    pub curl_executable_sha256: String,
    /// Absolute path to the CA bundle used by curl.
    pub ca_bundle_path: String,
    /// Exact CA bundle SHA-256.
    pub ca_bundle_sha256: String,
    /// Local path to the exact Nix/container/environment manifest.
    pub execution_environment_manifest_path: String,
    /// Exact execution-environment manifest SHA-256.
    pub execution_environment_manifest_sha256: String,
    /// Direct-child wall-clock bound.
    pub timeout_ms: u64,
    /// Per-stream diagnostic capture bound.
    pub max_output_bytes: usize,
}

impl OqmdHttpsAcquisitionProfile {
    /// Validate deterministic profile fields against the frozen OQMD protocol.
    pub fn validate_against(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<(), OqmdAcquisitionError> {
        protocol
            .validate()
            .map_err(|error| OqmdAcquisitionError::Protocol(error.to_string()))?;
        if self.schema_version != 1 {
            return Err(OqmdAcquisitionError::UnsupportedProfileSchema(
                self.schema_version,
            ));
        }
        let expected = protocol
            .protocol_sha256()
            .map_err(|error| OqmdAcquisitionError::Protocol(error.to_string()))?;
        if !self.protocol_sha256.eq_ignore_ascii_case(&expected) {
            return Err(OqmdAcquisitionError::ProtocolDigestMismatch);
        }
        if self.source_listing_url != OFFICIAL_DOWNLOAD_PAGE {
            return Err(OqmdAcquisitionError::UnexpectedSourceListingUrl(
                self.source_listing_url.clone(),
            ));
        }
        validate_sha256(&self.source_listing_artifact_sha256)?;
        validate_sha256(&self.curl_executable_sha256)?;
        validate_sha256(&self.ca_bundle_sha256)?;
        validate_sha256(&self.execution_environment_manifest_sha256)?;
        validate_absolute_path("source_listing_artifact_path", &self.source_listing_artifact_path)?;
        validate_absolute_path("curl_executable_path", &self.curl_executable_path)?;
        validate_absolute_path("ca_bundle_path", &self.ca_bundle_path)?;
        validate_absolute_path(
            "execution_environment_manifest_path",
            &self.execution_environment_manifest_path,
        )?;
        validate_oqmd_https_dump_url(&self.requested_url, &protocol.dump_filename)?;
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(OqmdAcquisitionError::InvalidTimeout(self.timeout_ms));
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > MAX_DIAGNOSTIC_BYTES {
            return Err(OqmdAcquisitionError::InvalidOutputLimit(
                self.max_output_bytes,
            ));
        }
        Ok(())
    }

    /// Rehash every preregistered local input before process launch.
    pub fn validate_local_artifacts(&self) -> Result<(), OqmdAcquisitionError> {
        verify_exact_file(
            Path::new(&self.source_listing_artifact_path),
            &self.source_listing_artifact_sha256,
        )?;
        verify_exact_file(
            Path::new(&self.curl_executable_path),
            &self.curl_executable_sha256,
        )?;
        verify_exact_file(Path::new(&self.ca_bundle_path), &self.ca_bundle_sha256)?;
        verify_exact_file(
            Path::new(&self.execution_environment_manifest_path),
            &self.execution_environment_manifest_sha256,
        )?;
        Ok(())
    }

    /// Deterministic profile identity after validation.
    pub fn profile_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<String, OqmdAcquisitionError> {
        self.validate_against(protocol)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Post-process assessment of one captured curl invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferAssessment {
    /// Process capture exists but curl did not exit successfully.
    ProcessDidNotSucceed,
    /// Curl exited successfully but its bound metadata/output could not qualify.
    EvidenceRejected {
        /// Stable human-readable rejection reason.
        reason: String,
    },
    /// Complete transfer evidence was assembled successfully.
    Complete {
        /// Final effective HTTPS URL after redirects.
        final_url: String,
        /// Final HTTP response status.
        http_status: u16,
        /// Exact response-header artifact SHA-256.
        response_headers_sha256: String,
        /// Exact downloaded compressed archive SHA-256.
        snapshot_sha256: String,
        /// Exact downloaded byte count.
        snapshot_bytes: u64,
    },
}

/// Full attempt evidence, including failed transfers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdHttpsAcquisitionAttempt {
    /// Exact acquisition profile identity.
    pub profile_sha256: String,
    /// Exact process request used for curl.
    pub process_spec: ProcessSpec,
    /// Raw bounded direct-child process evidence.
    pub process_capture: ProcessCapture,
    /// Partial/final archive identity when a non-empty output file exists.
    pub observed_snapshot: Option<StreamedSnapshotIdentity>,
    /// Response-header artifact digest when the file exists and is non-empty.
    pub observed_response_headers_sha256: Option<String>,
    /// Post-process transfer assessment.
    pub assessment: TransferAssessment,
}

impl OqmdHttpsAcquisitionAttempt {
    /// Deterministic identity over the complete attempt evidence.
    pub fn attempt_sha256(&self) -> Result<String, OqmdAcquisitionError> {
        self.process_spec
            .validate()
            .map_err(|error| OqmdAcquisitionError::Process(error.to_string()))?;
        let spec_sha = self
            .process_spec
            .manifest_sha256()
            .map_err(|error| OqmdAcquisitionError::Process(error.to_string()))?;
        if !spec_sha.eq_ignore_ascii_case(&self.process_capture.command_manifest_sha256) {
            return Err(OqmdAcquisitionError::ProcessManifestMismatch);
        }
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Execute one preregistered HTTPS acquisition while retaining failed-attempt evidence.
pub fn execute_https_acquisition(
    protocol: &OqmdExtractionProtocol,
    profile: &OqmdHttpsAcquisitionProfile,
    destination: &Path,
    response_headers_path: &Path,
) -> Result<OqmdHttpsAcquisitionAttempt, OqmdAcquisitionError> {
    profile.validate_against(protocol)?;
    profile.validate_local_artifacts()?;
    require_new_absolute_output(destination, "destination")?;
    require_new_absolute_output(response_headers_path, "response_headers")?;

    let profile_sha = profile.profile_sha256(protocol)?;
    let process_spec = build_curl_spec(profile, destination, response_headers_path)?;
    let process_capture = capture_process(&process_spec)
        .map_err(|error| OqmdAcquisitionError::Process(error.to_string()))?;

    let observed_snapshot = try_hash_nonempty_file(destination)?;
    let observed_response_headers_sha256 = try_hash_nonempty_file(response_headers_path)?
        .map(|identity| identity.sha256);

    let assessment = if !process_capture.process_success() {
        TransferAssessment::ProcessDidNotSucceed
    } else {
        assess_successful_process(
            protocol,
            &process_capture,
            observed_snapshot.as_ref(),
            observed_response_headers_sha256.as_deref(),
        )
    };

    let attempt = OqmdHttpsAcquisitionAttempt {
        profile_sha256: profile_sha,
        process_spec,
        process_capture,
        observed_snapshot,
        observed_response_headers_sha256,
        assessment,
    };
    attempt.attempt_sha256()?;
    Ok(attempt)
}

/// Promote a complete attempt into the narrower MAG-DATA-006 acquisition receipt.
pub fn bind_successful_acquisition_receipt(
    protocol: &OqmdExtractionProtocol,
    profile: &OqmdHttpsAcquisitionProfile,
    attempt: &OqmdHttpsAcquisitionAttempt,
    acquired_at_utc: &str,
) -> Result<HistoricalSnapshotAcquisitionReceipt, OqmdAcquisitionError> {
    profile.validate_against(protocol)?;
    profile.validate_local_artifacts()?;
    let expected_profile = profile.profile_sha256(protocol)?;
    if !attempt.profile_sha256.eq_ignore_ascii_case(&expected_profile) {
        return Err(OqmdAcquisitionError::ProfileDigestMismatch);
    }
    let attempt_sha = attempt.attempt_sha256()?;

    let TransferAssessment::Complete {
        final_url,
        http_status,
        response_headers_sha256,
        snapshot_sha256,
        snapshot_bytes,
    } = &attempt.assessment
    else {
        return Err(OqmdAcquisitionError::AttemptNotComplete);
    };

    if attempt
        .observed_snapshot
        .as_ref()
        .is_none_or(|identity| {
            !identity.sha256.eq_ignore_ascii_case(snapshot_sha256)
                || identity.bytes != *snapshot_bytes
        })
    {
        return Err(OqmdAcquisitionError::AttemptSnapshotMismatch);
    }
    if attempt
        .observed_response_headers_sha256
        .as_ref()
        .is_none_or(|digest| !digest.eq_ignore_ascii_case(response_headers_sha256))
    {
        return Err(OqmdAcquisitionError::AttemptHeadersMismatch);
    }

    let receipt = HistoricalSnapshotAcquisitionReceipt {
        schema_version: 1,
        protocol_sha256: protocol
            .protocol_sha256()
            .map_err(|error| OqmdAcquisitionError::Protocol(error.to_string()))?,
        provider: protocol.provider.clone(),
        database_version: protocol.database_version.clone(),
        dump_filename: protocol.dump_filename.clone(),
        source_license: protocol.source_license.clone(),
        acquired_at_utc: acquired_at_utc.to_string(),
        compressed_snapshot_sha256: snapshot_sha256.to_ascii_lowercase(),
        compressed_snapshot_bytes: *snapshot_bytes,
        acquisition_tool_sha256: profile.curl_executable_sha256.to_ascii_lowercase(),
        execution_environment_sha256: profile
            .execution_environment_manifest_sha256
            .to_ascii_lowercase(),
        transfer_log_sha256: attempt_sha,
        route: AcquisitionRoute::ProviderHttps {
            requested_url: profile.requested_url.clone(),
            final_url: final_url.clone(),
            http_status: *http_status,
            response_headers_sha256: response_headers_sha256.to_ascii_lowercase(),
        },
    };
    receipt
        .validate_against(protocol)
        .map_err(|error| OqmdAcquisitionError::Receipt(error.to_string()))?;
    Ok(receipt)
}

fn build_curl_spec(
    profile: &OqmdHttpsAcquisitionProfile,
    destination: &Path,
    response_headers_path: &Path,
) -> Result<ProcessSpec, OqmdAcquisitionError> {
    let destination = destination
        .to_str()
        .ok_or(OqmdAcquisitionError::NonUtf8Path)?;
    let response_headers = response_headers_path
        .to_str()
        .ok_or(OqmdAcquisitionError::NonUtf8Path)?;
    let mut environment = BTreeMap::new();
    environment.insert("LANG".to_string(), "C".to_string());
    environment.insert("LC_ALL".to_string(), "C".to_string());
    environment.insert("TZ".to_string(), "UTC".to_string());
    environment.insert("SSL_CERT_FILE".to_string(), profile.ca_bundle_path.clone());

    let spec = ProcessSpec {
        command: profile.curl_executable_path.clone(),
        args: vec![
            "--disable".to_string(),
            "--fail".to_string(),
            "--location".to_string(),
            "--proto".to_string(),
            "=https".to_string(),
            "--proto-redir".to_string(),
            "=https".to_string(),
            "--tlsv1.2".to_string(),
            "--cacert".to_string(),
            profile.ca_bundle_path.clone(),
            "--silent".to_string(),
            "--show-error".to_string(),
            "--output".to_string(),
            destination.to_string(),
            "--dump-header".to_string(),
            response_headers.to_string(),
            "--write-out".to_string(),
            "%{http_code}\n%{url_effective}\n".to_string(),
            profile.requested_url.clone(),
        ],
        environment,
        environment_policy: EnvironmentPolicy::ClearAndSet,
        timeout_ms: profile.timeout_ms,
        max_output_bytes: profile.max_output_bytes,
    };
    spec.validate()
        .map_err(|error| OqmdAcquisitionError::Process(error.to_string()))?;
    Ok(spec)
}

fn assess_successful_process(
    protocol: &OqmdExtractionProtocol,
    capture: &ProcessCapture,
    snapshot: Option<&StreamedSnapshotIdentity>,
    headers_sha: Option<&str>,
) -> TransferAssessment {
    let Ok(stdout) = std::str::from_utf8(&capture.stdout) else {
        return rejected("curl metadata stdout is not UTF-8");
    };
    let mut lines = stdout.lines();
    let Some(status_text) = lines.next() else {
        return rejected("curl metadata is missing HTTP status");
    };
    let Some(final_url) = lines.next() else {
        return rejected("curl metadata is missing final URL");
    };
    if lines.next().is_some() {
        return rejected("curl metadata contains unexpected extra lines");
    }
    let Ok(http_status) = status_text.parse::<u16>() else {
        return rejected("curl metadata HTTP status is not numeric");
    };
    if !(200..=299).contains(&http_status) {
        return rejected("final HTTP status is not 2xx");
    }
    if validate_oqmd_https_dump_url(final_url, &protocol.dump_filename).is_err() {
        return rejected("final URL is not an accepted OQMD HTTPS dump URL");
    }
    let Some(snapshot) = snapshot else {
        return rejected("downloaded archive is missing or empty");
    };
    let Some(headers_sha) = headers_sha else {
        return rejected("response-header artifact is missing or empty");
    };
    TransferAssessment::Complete {
        final_url: final_url.to_string(),
        http_status,
        response_headers_sha256: headers_sha.to_ascii_lowercase(),
        snapshot_sha256: snapshot.sha256.to_ascii_lowercase(),
        snapshot_bytes: snapshot.bytes,
    }
}

fn rejected(reason: &str) -> TransferAssessment {
    TransferAssessment::EvidenceRejected {
        reason: reason.to_string(),
    }
}

fn try_hash_nonempty_file(path: &Path) -> Result<Option<StreamedSnapshotIdentity>, OqmdAcquisitionError> {
    if !path.exists() {
        return Ok(None);
    }
    let metadata = std::fs::metadata(path).map_err(OqmdAcquisitionError::Io)?;
    if metadata.len() == 0 {
        return Ok(None);
    }
    let file = File::open(path).map_err(OqmdAcquisitionError::Io)?;
    let identity = hash_snapshot_stream(BufReader::new(file))
        .map_err(|error| OqmdAcquisitionError::Snapshot(error.to_string()))?;
    Ok(Some(identity))
}

fn verify_exact_file(path: &Path, expected_sha: &str) -> Result<(), OqmdAcquisitionError> {
    let file = File::open(path).map_err(OqmdAcquisitionError::Io)?;
    let identity = hash_snapshot_stream(BufReader::new(file))
        .map_err(|error| OqmdAcquisitionError::Snapshot(error.to_string()))?;
    if !identity.sha256.eq_ignore_ascii_case(expected_sha) {
        return Err(OqmdAcquisitionError::LocalArtifactDigestMismatch(
            path.display().to_string(),
        ));
    }
    Ok(())
}

fn require_new_absolute_output(path: &Path, name: &'static str) -> Result<(), OqmdAcquisitionError> {
    if !path.is_absolute() {
        return Err(OqmdAcquisitionError::OutputPathNotAbsolute(name));
    }
    if path.exists() {
        return Err(OqmdAcquisitionError::OutputAlreadyExists(
            path.display().to_string(),
        ));
    }
    Ok(())
}

fn validate_absolute_path(name: &'static str, value: &str) -> Result<(), OqmdAcquisitionError> {
    if value.trim().is_empty() || !Path::new(value).is_absolute() {
        return Err(OqmdAcquisitionError::ProfilePathNotAbsolute(name));
    }
    Ok(())
}

fn validate_oqmd_https_dump_url(url: &str, dump_filename: &str) -> Result<(), OqmdAcquisitionError> {
    if !url.is_ascii()
        || url
            .chars()
            .any(|ch| matches!(ch, '\n' | '\r' | '\0' | '?' | '#' | '@'))
    {
        return Err(OqmdAcquisitionError::InvalidOqmdUrl(url.to_string()));
    }
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| OqmdAcquisitionError::InvalidOqmdUrl(url.to_string()))?;
    let host = rest
        .split('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if host.is_empty()
        || host.contains(':')
        || !(host == "oqmd.org" || host.ends_with(".oqmd.org"))
    {
        return Err(OqmdAcquisitionError::InvalidOqmdUrl(url.to_string()));
    }
    let last = rest.rsplit('/').next().unwrap_or_default();
    if last != dump_filename {
        return Err(OqmdAcquisitionError::UnexpectedDumpFilename {
            expected: dump_filename.to_string(),
            observed: last.to_string(),
        });
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), OqmdAcquisitionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OqmdAcquisitionError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// HTTPS acquisition execution/validation failure.
#[derive(Debug, Error)]
pub enum OqmdAcquisitionError {
    /// Historical protocol rejected input.
    #[error("historical protocol rejected input: {0}")]
    Protocol(String),
    /// MAG-DATA-006 receipt rejected input.
    #[error("snapshot acquisition receipt rejected input: {0}")]
    Receipt(String),
    /// Raw process-capture layer failed to assemble trustworthy evidence.
    #[error("process-capture failure: {0}")]
    Process(String),
    /// Snapshot hashing/validation failed.
    #[error("snapshot identity failure: {0}")]
    Snapshot(String),
    /// Profile schema unsupported.
    #[error("unsupported OQMD acquisition profile schema {0}")]
    UnsupportedProfileSchema(u32),
    /// Profile names another historical protocol.
    #[error("acquisition profile protocol digest mismatch")]
    ProtocolDigestMismatch,
    /// Profile digest differs from the attempt profile digest.
    #[error("acquisition profile digest mismatch")]
    ProfileDigestMismatch,
    /// Source listing locator differs from the canonical OQMD download page.
    #[error("unexpected source listing URL: {0}")]
    UnexpectedSourceListingUrl(String),
    /// Path required to be absolute.
    #[error("profile path is not absolute: {0}")]
    ProfilePathNotAbsolute(&'static str),
    /// Output path required to be absolute.
    #[error("output path is not absolute: {0}")]
    OutputPathNotAbsolute(&'static str),
    /// Output path already exists and will not be overwritten.
    #[error("output already exists: {0}")]
    OutputAlreadyExists(String),
    /// Path is not UTF-8 and cannot enter deterministic curl args.
    #[error("non-UTF-8 path cannot enter deterministic acquisition command")]
    NonUtf8Path,
    /// SHA-256 text malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Preregistered local artifact bytes changed before launch.
    #[error("local preregistered artifact digest mismatch: {0}")]
    LocalArtifactDigestMismatch(String),
    /// OQMD URL malformed/not accepted.
    #[error("invalid OQMD HTTPS dump URL: {0}")]
    InvalidOqmdUrl(String),
    /// URL names another dump filename.
    #[error("unexpected dump filename: expected {expected}, observed {observed}")]
    UnexpectedDumpFilename {
        /// Frozen expected filename.
        expected: String,
        /// Filename present in URL.
        observed: String,
    },
    /// Timeout outside allowed bounds.
    #[error("invalid acquisition timeout {0} ms")]
    InvalidTimeout(u64),
    /// Diagnostic-output bound outside allowed range.
    #[error("invalid acquisition diagnostic-output bound {0} bytes")]
    InvalidOutputLimit(usize),
    /// Capture manifest does not match its stored ProcessSpec.
    #[error("captured process manifest differs from stored ProcessSpec")]
    ProcessManifestMismatch,
    /// Attempt did not reach complete-transfer authority.
    #[error("acquisition attempt is not complete")]
    AttemptNotComplete,
    /// Attempt's observed archive identity disagrees with complete assessment.
    #[error("attempt archive identity disagrees with complete assessment")]
    AttemptSnapshotMismatch,
    /// Attempt's header artifact identity disagrees with complete assessment.
    #[error("attempt header identity disagrees with complete assessment")]
    AttemptHeadersMismatch,
    /// I/O failure reading local evidence/artifact bytes.
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol;
    use symthaea_process_capture::ProcessTermination;

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn profile(protocol: &OqmdExtractionProtocol) -> OqmdHttpsAcquisitionProfile {
        OqmdHttpsAcquisitionProfile {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            source_listing_url: OFFICIAL_DOWNLOAD_PAGE.to_string(),
            source_listing_artifact_path: "/tmp/download-page.html".to_string(),
            source_listing_artifact_sha256: hex('1'),
            requested_url: format!(
                "https://static.oqmd.org/static/downloads/{}",
                protocol.dump_filename
            ),
            curl_executable_path: "/nix/store/example/bin/curl".to_string(),
            curl_executable_sha256: hex('2'),
            ca_bundle_path: "/nix/store/example/etc/ca-bundle.crt".to_string(),
            ca_bundle_sha256: hex('3'),
            execution_environment_manifest_path: "/tmp/environment.json".to_string(),
            execution_environment_manifest_sha256: hex('4'),
            timeout_ms: 60_000,
            max_output_bytes: 1024,
        }
    }

    #[test]
    fn profile_accepts_oqmd_https_filename_and_rejects_query_smuggling() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let valid = profile(&protocol);
        valid.validate_against(&protocol).unwrap();
        let mut bad = valid;
        bad.requested_url.push_str("?other=1");
        assert!(matches!(
            bad.validate_against(&protocol),
            Err(OqmdAcquisitionError::InvalidOqmdUrl(_))
        ));
    }

    #[test]
    fn successful_curl_metadata_requires_exact_oqmd_https_final_url() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let capture = ProcessCapture {
            command_manifest_sha256: hex('a'),
            termination: ProcessTermination::Exited {
                exit_code: Some(0),
                success: true,
            },
            stdout: format!(
                "200\nhttps://static.oqmd.org/static/downloads/{}\n",
                protocol.dump_filename
            )
            .into_bytes(),
            stderr: Vec::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            elapsed_ms: 1,
        };
        let snapshot = StreamedSnapshotIdentity {
            sha256: hex('b'),
            bytes: 42,
        };
        assert!(matches!(
            assess_successful_process(&protocol, &capture, Some(&snapshot), Some(&hex('c'))),
            TransferAssessment::Complete { http_status: 200, .. }
        ));
    }

    #[test]
    fn redirect_to_non_oqmd_host_cannot_qualify() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let capture = ProcessCapture {
            command_manifest_sha256: hex('a'),
            termination: ProcessTermination::Exited {
                exit_code: Some(0),
                success: true,
            },
            stdout: format!("200\nhttps://example.com/{}\n", protocol.dump_filename).into_bytes(),
            stderr: Vec::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            elapsed_ms: 1,
        };
        let snapshot = StreamedSnapshotIdentity {
            sha256: hex('b'),
            bytes: 42,
        };
        assert!(matches!(
            assess_successful_process(&protocol, &capture, Some(&snapshot), Some(&hex('c'))),
            TransferAssessment::EvidenceRejected { .. }
        ));
    }
}
