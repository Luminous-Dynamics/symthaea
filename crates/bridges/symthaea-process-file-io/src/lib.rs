// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed stdin plus create-new stdout-file execution.
//!
//! This is the large-artifact companion to SIM-PROC-003. The launcher verifies an
//! exact open stdin file descriptor, creates stdout with create-new semantics, then
//! `exec`s the target. SIM-PROC-001 continues to own timeout, stderr bounds, and
//! termination evidence while large stdout bypasses the diagnostic capture buffer.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use symthaea_process_capture::{ProcessCapture, ProcessSpec, capture_process};
use symthaea_process_stdin::ContentAddressedStdinFile;
use thiserror::Error;

/// Exact launcher artifact for file-bound stdin/stdout execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileIoLauncherArtifact {
    /// Absolute UTF-8 launcher path.
    pub path: String,
    /// Exact launcher SHA-256.
    pub sha256: String,
}

impl FileIoLauncherArtifact {
    /// Observe one local launcher executable.
    pub fn observe(path: &Path) -> Result<Self, FileIoCaptureError> {
        if !path.is_absolute() {
            return Err(FileIoCaptureError::LauncherPathNotAbsolute);
        }
        let path_text = path.to_str().ok_or(FileIoCaptureError::NonUtf8Path)?;
        let (sha256, _) = hash_file(path)?;
        Ok(Self {
            path: path_text.to_string(),
            sha256,
        })
    }

    fn validate_and_rehash(&self) -> Result<(), FileIoCaptureError> {
        if self.path.trim().is_empty() || !Path::new(&self.path).is_absolute() {
            return Err(FileIoCaptureError::LauncherPathNotAbsolute);
        }
        validate_sha256(&self.sha256)?;
        let (actual, _) = hash_file(Path::new(&self.path))?;
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(FileIoCaptureError::LauncherDigestMismatch);
        }
        Ok(())
    }
}

/// Requested create-new stdout path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewStdoutFile {
    /// Absolute UTF-8 destination path.
    pub path: String,
}

impl NewStdoutFile {
    /// Construct an output request; the path must not already exist at execution.
    pub fn new(path: &Path) -> Result<Self, FileIoCaptureError> {
        if !path.is_absolute() {
            return Err(FileIoCaptureError::OutputPathNotAbsolute);
        }
        let path = path.to_str().ok_or(FileIoCaptureError::NonUtf8Path)?;
        Ok(Self {
            path: path.to_string(),
        })
    }

    fn validate_new(&self) -> Result<(), FileIoCaptureError> {
        if self.path.trim().is_empty() || !Path::new(&self.path).is_absolute() {
            return Err(FileIoCaptureError::OutputPathNotAbsolute);
        }
        if Path::new(&self.path).exists() {
            return Err(FileIoCaptureError::OutputAlreadyExists(self.path.clone()));
        }
        Ok(())
    }
}

/// Observed exact bytes written to a requested stdout file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedOutputFile {
    /// Absolute output path.
    pub path: String,
    /// SHA-256 of all bytes currently present.
    pub sha256: String,
    /// Exact byte count.
    pub bytes: u64,
}

/// One exact file-I/O execution request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundFileIoProcessRequest {
    /// Exact file-I/O launcher artifact.
    pub launcher: FileIoLauncherArtifact,
    /// Exact stdin identity to verify on the descriptor used by the target.
    pub stdin: ContentAddressedStdinFile,
    /// Create-new stdout destination.
    pub stdout: NewStdoutFile,
    /// Logical target process specification.
    pub target: ProcessSpec,
}

impl BoundFileIoProcessRequest {
    /// Validate all static/local request constraints.
    pub fn validate(&self) -> Result<(), FileIoCaptureError> {
        self.launcher.validate_and_rehash()?;
        self.stdin
            .validate()
            .map_err(|error| FileIoCaptureError::Stdin(error.to_string()))?;
        self.stdout.validate_new()?;
        self.target
            .validate()
            .map_err(|error| FileIoCaptureError::Process(error.to_string()))?;
        Ok(())
    }

    /// Deterministic request identity.
    pub fn request_sha256(&self) -> Result<String, FileIoCaptureError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Exact outer SIM-PROC request for the launcher.
    pub fn outer_process_spec(&self) -> Result<ProcessSpec, FileIoCaptureError> {
        self.validate()?;
        let mut args = vec![
            self.stdin.path.clone(),
            self.stdin.sha256.to_ascii_lowercase(),
            self.stdin.bytes.to_string(),
            self.stdout.path.clone(),
            self.target.command.clone(),
        ];
        args.extend(self.target.args.iter().cloned());
        let spec = ProcessSpec {
            command: self.launcher.path.clone(),
            args,
            environment: self.target.environment.clone(),
            environment_policy: self.target.environment_policy,
            timeout_ms: self.target.timeout_ms,
            max_output_bytes: self.target.max_output_bytes,
        };
        spec.validate()
            .map_err(|error| FileIoCaptureError::Process(error.to_string()))?;
        Ok(spec)
    }
}

/// Evidence from one file-I/O invocation, including partial output on failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundFileIoProcessCapture {
    /// Exact request identity.
    pub request_sha256: String,
    /// Logical target process manifest identity.
    pub target_process_manifest_sha256: String,
    /// Outer launcher process manifest identity.
    pub launcher_process_manifest_sha256: String,
    /// Bound stdin identity.
    pub stdin: ContentAddressedStdinFile,
    /// Requested stdout path.
    pub stdout_path: String,
    /// Exact observed output when any output file was created.
    pub observed_stdout: Option<ObservedOutputFile>,
    /// Raw SIM-PROC evidence. Target stdout is file-bound; stderr remains bounded capture.
    pub process_capture: ProcessCapture,
}

impl BoundFileIoProcessCapture {
    /// Whether the exec'd target exited successfully and a stdout file exists.
    pub fn complete_output(&self) -> bool {
        self.process_capture.process_success() && self.observed_stdout.is_some()
    }

    /// Deterministic complete evidence identity.
    pub fn capture_sha256(&self) -> Result<String, FileIoCaptureError> {
        validate_sha256(&self.request_sha256)?;
        validate_sha256(&self.target_process_manifest_sha256)?;
        validate_sha256(&self.launcher_process_manifest_sha256)?;
        if let Some(output) = &self.observed_stdout {
            validate_sha256(&output.sha256)?;
        }
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Execute one target with verified file stdin and create-new file stdout.
pub fn capture_process_with_file_io(
    request: &BoundFileIoProcessRequest,
) -> Result<BoundFileIoProcessCapture, FileIoCaptureError> {
    request.validate()?;
    let request_sha = request.request_sha256()?;
    let target_sha = request
        .target
        .manifest_sha256()
        .map_err(|error| FileIoCaptureError::Process(error.to_string()))?;
    let outer = request.outer_process_spec()?;
    let outer_sha = outer
        .manifest_sha256()
        .map_err(|error| FileIoCaptureError::Process(error.to_string()))?;
    let process_capture = capture_process(&outer)
        .map_err(|error| FileIoCaptureError::Process(error.to_string()))?;
    if !process_capture
        .command_manifest_sha256
        .eq_ignore_ascii_case(&outer_sha)
    {
        return Err(FileIoCaptureError::OuterManifestMismatch);
    }
    let observed_stdout = observe_optional(Path::new(&request.stdout.path))?;
    let capture = BoundFileIoProcessCapture {
        request_sha256: request_sha,
        target_process_manifest_sha256: target_sha,
        launcher_process_manifest_sha256: outer_sha,
        stdin: request.stdin.clone(),
        stdout_path: request.stdout.path.clone(),
        observed_stdout,
        process_capture,
    };
    capture.capture_sha256()?;
    Ok(capture)
}

fn observe_optional(path: &Path) -> Result<Option<ObservedOutputFile>, FileIoCaptureError> {
    if !path.exists() {
        return Ok(None);
    }
    let (sha256, bytes) = hash_file(path)?;
    Ok(Some(ObservedOutputFile {
        path: path
            .to_str()
            .ok_or(FileIoCaptureError::NonUtf8Path)?
            .to_string(),
        sha256,
        bytes,
    }))
}

fn hash_file(path: &Path) -> Result<(String, u64), FileIoCaptureError> {
    let file = File::open(path).map_err(FileIoCaptureError::Io)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut chunk).map_err(FileIoCaptureError::Io)?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or(FileIoCaptureError::ByteCountOverflow)?;
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn validate_sha256(value: &str) -> Result<(), FileIoCaptureError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FileIoCaptureError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// File-bound process execution failure.
#[derive(Debug, Error)]
pub enum FileIoCaptureError {
    /// Launcher path invalid.
    #[error("file-I/O launcher path must be absolute")]
    LauncherPathNotAbsolute,
    /// Output path invalid.
    #[error("stdout file path must be absolute")]
    OutputPathNotAbsolute,
    /// Output path already exists.
    #[error("stdout destination already exists: {0}")]
    OutputAlreadyExists(String),
    /// Required path not UTF-8.
    #[error("file-I/O path must be UTF-8")]
    NonUtf8Path,
    /// SHA text malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Launcher bytes differ from expected identity.
    #[error("file-I/O launcher digest mismatch")]
    LauncherDigestMismatch,
    /// Input identity layer rejected its object.
    #[error("stdin identity failure: {0}")]
    Stdin(String),
    /// Raw process layer failed to capture invocation.
    #[error("raw process-capture failure: {0}")]
    Process(String),
    /// Raw capture names another outer command manifest.
    #[error("raw process capture manifest differs from bound file-I/O launcher request")]
    OuterManifestMismatch,
    /// Byte count overflowed.
    #[error("file byte count overflowed u64")]
    ByteCountOverflow,
    /// File I/O failure.
    #[error("file I/O failure: {0}")]
    Io(#[source] std::io::Error),
    /// Serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
