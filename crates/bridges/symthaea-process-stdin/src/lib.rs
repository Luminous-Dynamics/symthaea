// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed file-to-stdin execution above SIM-PROC raw process capture.
//!
//! The launcher opens the preregistered file once, hashes/counts that exact file
//! descriptor, rewinds it, attaches the same descriptor to stdin, and `exec`s the
//! requested target. Timeout/output/termination evidence remains owned by
//! `symthaea-process-capture`; this crate adds only stdin identity and exec binding.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use symthaea_process_capture::{ProcessCapture, ProcessSpec, capture_process};
use thiserror::Error;

/// Exact identity of a regular file intended to become child stdin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentAddressedStdinFile {
    /// Absolute UTF-8 path used to open the file.
    pub path: String,
    /// SHA-256 of exact file bytes.
    pub sha256: String,
    /// Exact byte count.
    pub bytes: u64,
}

impl ContentAddressedStdinFile {
    /// Observe and content-address one local file.
    pub fn observe(path: &Path) -> Result<Self, StdinCaptureError> {
        if !path.is_absolute() {
            return Err(StdinCaptureError::PathNotAbsolute);
        }
        let path_text = path.to_str().ok_or(StdinCaptureError::NonUtf8Path)?;
        let (sha256, bytes) = hash_file(path)?;
        Ok(Self {
            path: path_text.to_string(),
            sha256,
            bytes,
        })
    }

    /// Validate shape without re-reading bytes.
    pub fn validate(&self) -> Result<(), StdinCaptureError> {
        if self.path.trim().is_empty() || !Path::new(&self.path).is_absolute() {
            return Err(StdinCaptureError::PathNotAbsolute);
        }
        validate_sha256(&self.sha256)?;
        Ok(())
    }
}

/// Exact launcher artifact used to verify stdin and then `exec` the target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StdinLauncherArtifact {
    /// Absolute UTF-8 launcher path.
    pub path: String,
    /// SHA-256 of launcher bytes expected before invocation.
    pub sha256: String,
}

impl StdinLauncherArtifact {
    /// Observe one launcher executable artifact.
    pub fn observe(path: &Path) -> Result<Self, StdinCaptureError> {
        if !path.is_absolute() {
            return Err(StdinCaptureError::LauncherPathNotAbsolute);
        }
        let path_text = path.to_str().ok_or(StdinCaptureError::NonUtf8Path)?;
        let (sha256, _) = hash_file(path)?;
        Ok(Self {
            path: path_text.to_string(),
            sha256,
        })
    }

    fn validate_and_rehash(&self) -> Result<(), StdinCaptureError> {
        if self.path.trim().is_empty() || !Path::new(&self.path).is_absolute() {
            return Err(StdinCaptureError::LauncherPathNotAbsolute);
        }
        validate_sha256(&self.sha256)?;
        let (actual, _) = hash_file(Path::new(&self.path))?;
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(StdinCaptureError::LauncherDigestMismatch);
        }
        Ok(())
    }
}

/// One content-addressed stdin invocation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundStdinProcessRequest {
    /// Exact helper executable that performs file verification + fd binding + exec.
    pub launcher: StdinLauncherArtifact,
    /// Exact expected stdin file identity.
    pub stdin: ContentAddressedStdinFile,
    /// Logical target process specification.
    pub target: ProcessSpec,
}

impl BoundStdinProcessRequest {
    /// Validate the request without executing it.
    pub fn validate(&self) -> Result<(), StdinCaptureError> {
        self.stdin.validate()?;
        self.launcher.validate_and_rehash()?;
        self.target
            .validate()
            .map_err(|error| StdinCaptureError::Process(error.to_string()))?;
        Ok(())
    }

    /// Deterministic identity over launcher, stdin, and logical target request.
    pub fn request_sha256(&self) -> Result<String, StdinCaptureError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Materialize the exact outer SIM-PROC request.
    ///
    /// Target environment, timeout, and output bounds remain unchanged. The target
    /// command/args and stdin identity are encoded as ordinary ordered launcher args.
    pub fn outer_process_spec(&self) -> Result<ProcessSpec, StdinCaptureError> {
        self.validate()?;
        let mut args = vec![
            self.stdin.path.clone(),
            self.stdin.sha256.to_ascii_lowercase(),
            self.stdin.bytes.to_string(),
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
            .map_err(|error| StdinCaptureError::Process(error.to_string()))?;
        Ok(spec)
    }
}

/// Captured evidence for one stdin-bound process invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundStdinProcessCapture {
    /// Exact request identity.
    pub request_sha256: String,
    /// Exact logical target `ProcessSpec` identity.
    pub target_process_manifest_sha256: String,
    /// Exact outer launcher `ProcessSpec` identity.
    pub launcher_process_manifest_sha256: String,
    /// Exact stdin identity claimed and verified by the launcher before exec.
    pub stdin: ContentAddressedStdinFile,
    /// Raw SIM-PROC capture. A successful target has replaced the launcher by exec.
    pub process_capture: ProcessCapture,
}

impl BoundStdinProcessCapture {
    /// Whether the exec'd target process ultimately exited successfully.
    ///
    /// This remains only an OS process fact, never a scientific-validity claim.
    pub fn process_success(&self) -> bool {
        self.process_capture.process_success()
    }

    /// Deterministic identity of the complete stdin-bound process evidence.
    pub fn capture_sha256(&self) -> Result<String, StdinCaptureError> {
        validate_sha256(&self.request_sha256)?;
        validate_sha256(&self.target_process_manifest_sha256)?;
        validate_sha256(&self.launcher_process_manifest_sha256)?;
        self.stdin.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Execute one exact file as stdin to one exact logical target process.
pub fn capture_process_with_stdin_file(
    request: &BoundStdinProcessRequest,
) -> Result<BoundStdinProcessCapture, StdinCaptureError> {
    request.validate()?;
    let request_sha = request.request_sha256()?;
    let target_sha = request
        .target
        .manifest_sha256()
        .map_err(|error| StdinCaptureError::Process(error.to_string()))?;
    let outer = request.outer_process_spec()?;
    let outer_sha = outer
        .manifest_sha256()
        .map_err(|error| StdinCaptureError::Process(error.to_string()))?;
    let process_capture = capture_process(&outer)
        .map_err(|error| StdinCaptureError::Process(error.to_string()))?;
    if !process_capture
        .command_manifest_sha256
        .eq_ignore_ascii_case(&outer_sha)
    {
        return Err(StdinCaptureError::OuterManifestMismatch);
    }
    let capture = BoundStdinProcessCapture {
        request_sha256: request_sha,
        target_process_manifest_sha256: target_sha,
        launcher_process_manifest_sha256: outer_sha,
        stdin: request.stdin.clone(),
        process_capture,
    };
    capture.capture_sha256()?;
    Ok(capture)
}

fn hash_file(path: &Path) -> Result<(String, u64), StdinCaptureError> {
    let file = File::open(path).map_err(StdinCaptureError::Io)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut chunk).map_err(StdinCaptureError::Io)?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or(StdinCaptureError::ByteCountOverflow)?;
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn validate_sha256(value: &str) -> Result<(), StdinCaptureError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(StdinCaptureError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Content-addressed stdin invocation failure.
#[derive(Debug, Error)]
pub enum StdinCaptureError {
    /// Stdin path must be absolute.
    #[error("stdin file path must be absolute")]
    PathNotAbsolute,
    /// Launcher path must be absolute.
    #[error("stdin launcher path must be absolute")]
    LauncherPathNotAbsolute,
    /// Required path is not UTF-8.
    #[error("stdin/launcher path must be UTF-8")]
    NonUtf8Path,
    /// SHA-256 text malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Exact launcher bytes differ from preregistered digest.
    #[error("stdin launcher bytes differ from preregistered digest")]
    LauncherDigestMismatch,
    /// Byte count overflowed u64.
    #[error("file byte count overflowed u64")]
    ByteCountOverflow,
    /// Underlying raw process layer rejected/could not capture the invocation.
    #[error("raw process-capture failure: {0}")]
    Process(String),
    /// Raw capture names another outer command manifest.
    #[error("raw process capture command manifest differs from bound launcher request")]
    OuterManifestMismatch,
    /// File I/O failed.
    #[error("file I/O failure: {0}")]
    Io(#[source] std::io::Error),
    /// Serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
