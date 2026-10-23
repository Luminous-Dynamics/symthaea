// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded raw subprocess capture for reproducible scientific solver evidence.
//!
//! This crate deliberately stops at the operating-system process boundary. It can
//! establish which command was requested, which environment policy was used, how
//! the direct child terminated, how long it ran, and which bounded stdout/stderr
//! bytes were observed. It cannot establish numerical convergence or scientific
//! validity; domain adapters/parsers retain that authority.
//!
//! Timeout, output-limit termination, and non-zero exit are returned as captured
//! process evidence rather than discarded as generic errors. Spawn/pipe/I/O/wait
//! failures remain errors because no trustworthy process record can be completed.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};
use thiserror::Error;

const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const POST_EXIT_DRAIN: Duration = Duration::from_secs(2);

const fn default_timeout_ms() -> u64 {
    5 * 60 * 1000
}

const fn default_output_limit() -> usize {
    1024 * 1024
}

/// Whether the child inherits unspecified variables from the parent process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentPolicy {
    /// Preserve the parent's environment and override only explicitly supplied keys.
    InheritAndOverride,
    /// Clear the parent environment and expose only explicitly supplied keys.
    ///
    /// This is the preferred scientific-qualification policy because the process
    /// environment is then closed and reproducible from the command manifest.
    ClearAndSet,
}

impl Default for EnvironmentPolicy {
    fn default() -> Self {
        Self::InheritAndOverride
    }
}

/// Reproducible subprocess request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessSpec {
    /// Executable name or path.
    pub command: String,
    /// Ordered command-line arguments.
    pub args: Vec<String>,
    /// Explicit environment variables in deterministic key order.
    pub environment: BTreeMap<String, String>,
    /// Environment inheritance policy.
    #[serde(default)]
    pub environment_policy: EnvironmentPolicy,
    /// Wall-clock limit in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Maximum retained bytes from stdout and stderr independently.
    #[serde(default = "default_output_limit")]
    pub max_output_bytes: usize,
}

impl ProcessSpec {
    /// Construct a request using compatibility-oriented parent-environment inheritance.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            environment: BTreeMap::new(),
            environment_policy: EnvironmentPolicy::InheritAndOverride,
            timeout_ms: default_timeout_ms(),
            max_output_bytes: default_output_limit(),
        }
    }

    /// Append one ordered argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set one explicit environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }

    /// Select whether unspecified parent variables are inherited.
    pub fn environment_policy(mut self, policy: EnvironmentPolicy) -> Self {
        self.environment_policy = policy;
        self
    }

    /// Select a closed environment containing only variables explicitly supplied here.
    pub fn clear_environment(self) -> Self {
        self.environment_policy(EnvironmentPolicy::ClearAndSet)
    }

    /// Set the wall-clock timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = timeout.as_millis().min(u64::MAX as u128) as u64;
        self
    }

    /// Set the per-stream retained-output limit.
    pub fn max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Validate bounded execution parameters.
    pub fn validate(&self) -> Result<(), ProcessCaptureError> {
        if self.command.trim().is_empty() {
            return Err(ProcessCaptureError::EmptyCommand);
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(ProcessCaptureError::InvalidTimeout {
                timeout_ms: self.timeout_ms,
                maximum_ms: MAX_TIMEOUT_MS,
            });
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > MAX_OUTPUT_BYTES {
            return Err(ProcessCaptureError::InvalidOutputLimit {
                output_bytes: self.max_output_bytes,
                maximum_bytes: MAX_OUTPUT_BYTES,
            });
        }
        if self
            .environment
            .keys()
            .any(|key| key.is_empty() || key.contains('=') || key.contains('\0'))
        {
            return Err(ProcessCaptureError::InvalidEnvironmentKey);
        }
        if self.environment.values().any(|value| value.contains('\0')) {
            return Err(ProcessCaptureError::InvalidEnvironmentValue);
        }
        Ok(())
    }

    /// Deterministic SHA-256 of the exact serialized command/environment policy.
    pub fn manifest_sha256(&self) -> Result<String, ProcessCaptureError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Why the direct child stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessTermination {
    /// Direct child exited before timeout/output-limit enforcement.
    Exited {
        /// Portable exit code when available. `None` commonly means signal termination on Unix.
        exit_code: Option<i32>,
        /// Whether the OS exit status reported success.
        success: bool,
    },
    /// Wall-clock timeout elapsed and the child was killed.
    TimedOut,
    /// At least one output stream exceeded the configured retained-output bound.
    OutputLimitExceeded,
}

/// Raw bounded process evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessCapture {
    /// SHA-256 of the exact [`ProcessSpec`] used to launch the process.
    pub command_manifest_sha256: String,
    /// Direct-child termination classification.
    pub termination: ProcessTermination,
    /// Captured stdout prefix, bounded by the request.
    pub stdout: Vec<u8>,
    /// Captured stderr prefix, bounded by the request.
    pub stderr: Vec<u8>,
    /// Whether stdout exceeded the retention limit.
    pub stdout_truncated: bool,
    /// Whether stderr exceeded the retention limit.
    pub stderr_truncated: bool,
    /// Monotonic elapsed wall-clock duration in milliseconds.
    pub elapsed_ms: u64,
}

impl ProcessCapture {
    /// Whether the direct process exited successfully without enforcement termination.
    ///
    /// This is **not** a scientific/numerical convergence predicate.
    pub fn process_success(&self) -> bool {
        matches!(
            &self.termination,
            ProcessTermination::Exited { success: true, .. }
        )
    }

    /// Decode captured stdout lossily for legacy text-oriented adapters.
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Decode captured stderr lossily for diagnostics.
    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }

    /// Deterministic identity over the exact bounded process record.
    pub fn capture_sha256(&self) -> Result<String, ProcessCaptureError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Execute one bounded subprocess and preserve its raw process evidence.
///
/// Timeout, output overflow, and non-zero direct-child exits are successful
/// captures with an explicit [`ProcessTermination`]. Only failures that prevent a
/// trustworthy capture from being assembled return [`ProcessCaptureError`].
pub fn capture_process(spec: &ProcessSpec) -> Result<ProcessCapture, ProcessCaptureError> {
    spec.validate()?;
    let manifest_sha = spec.manifest_sha256()?;

    let mut command = std::process::Command::new(&spec.command);
    command.args(&spec.args);
    match spec.environment_policy {
        EnvironmentPolicy::InheritAndOverride => {
            command.envs(&spec.environment);
        }
        EnvironmentPolicy::ClearAndSet => {
            command.env_clear().envs(&spec.environment);
        }
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| ProcessCaptureError::Spawn {
            command: spec.command.clone(),
            source,
        })?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or(ProcessCaptureError::MissingStdoutPipe)?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or(ProcessCaptureError::MissingStderrPipe)?;

    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        set_nonblocking(stdout.as_raw_fd())?;
        set_nonblocking(stderr.as_raw_fd())?;
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err(ProcessCaptureError::UnsupportedPlatform);
    }

    let output_limit = spec.max_output_bytes;
    let mut stdout_buf = Vec::with_capacity(output_limit.min(8192));
    let mut stderr_buf = Vec::with_capacity(output_limit.min(8192));
    let mut stdout_exceeded = false;
    let mut stderr_exceeded = false;
    let mut stdout_eof = false;
    let mut stderr_eof = false;

    let started = Instant::now();
    let timeout = Duration::from_millis(spec.timeout_ms);
    let termination = loop {
        drain_streams(
            &mut child,
            &mut stdout,
            &mut stderr,
            &mut stdout_buf,
            &mut stderr_buf,
            output_limit,
            &mut stdout_exceeded,
            &mut stderr_exceeded,
            &mut stdout_eof,
            &mut stderr_eof,
        )?;

        match child.try_wait() {
            Ok(Some(status)) => {
                break ProcessTermination::Exited {
                    exit_code: status.code(),
                    success: status.success(),
                };
            }
            Ok(None) if stdout_exceeded || stderr_exceeded => {
                let _ = child.kill();
                child.wait().map_err(ProcessCaptureError::Wait)?;
                break ProcessTermination::OutputLimitExceeded;
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                child.wait().map_err(ProcessCaptureError::Wait)?;
                break ProcessTermination::TimedOut;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(source) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ProcessCaptureError::Poll(source));
            }
        }
    };

    // A descendant can inherit stdout/stderr pipes after the direct child exits.
    // Bound post-exit draining so such descendants cannot keep this API alive forever.
    let drain_deadline = Instant::now() + POST_EXIT_DRAIN;
    while Instant::now() < drain_deadline && (!stdout_eof || !stderr_eof) {
        drain_streams(
            &mut child,
            &mut stdout,
            &mut stderr,
            &mut stdout_buf,
            &mut stderr_buf,
            output_limit,
            &mut stdout_exceeded,
            &mut stderr_exceeded,
            &mut stdout_eof,
            &mut stderr_eof,
        )?;
        if !stdout_eof || !stderr_eof {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // A process may exit immediately after writing enough bytes to cross the bound,
    // before the poll loop notices the overflow. Preserve the stronger termination.
    let termination = if stdout_exceeded || stderr_exceeded {
        ProcessTermination::OutputLimitExceeded
    } else {
        termination
    };

    Ok(ProcessCapture {
        command_manifest_sha256: manifest_sha,
        termination,
        stdout: stdout_buf,
        stderr: stderr_buf,
        stdout_truncated: stdout_exceeded,
        stderr_truncated: stderr_exceeded,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
    })
}

fn drain_streams<R1: Read, R2: Read>(
    child: &mut std::process::Child,
    stdout: &mut R1,
    stderr: &mut R2,
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    output_limit: usize,
    stdout_exceeded: &mut bool,
    stderr_exceeded: &mut bool,
    stdout_eof: &mut bool,
    stderr_eof: &mut bool,
) -> Result<(), ProcessCaptureError> {
    if !*stdout_eof {
        match drain_nonblocking(stdout, stdout_buf, output_limit, stdout_exceeded) {
            Ok(eof) => *stdout_eof = eof,
            Err(source) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ProcessCaptureError::ReadStdout(source));
            }
        }
    }
    if !*stderr_eof {
        match drain_nonblocking(stderr, stderr_buf, output_limit, stderr_exceeded) {
            Ok(eof) => *stderr_eof = eof,
            Err(source) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ProcessCaptureError::ReadStderr(source));
            }
        }
    }
    Ok(())
}

fn drain_nonblocking<R: Read>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    limit: usize,
    exceeded: &mut bool,
) -> std::io::Result<bool> {
    let mut chunk = [0_u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => return Ok(true),
            Ok(read) => {
                let retain = read.min(limit.saturating_sub(buffer.len()));
                buffer.extend_from_slice(&chunk[..retain]);
                if retain < read {
                    *exceeded = true;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(error) => return Err(error),
        }
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn set_nonblocking(fd: std::os::fd::RawFd) -> Result<(), ProcessCaptureError> {
    // SAFETY: callers pass an AsRawFd borrow of a live owned ChildStdout/ChildStderr.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(ProcessCaptureError::NonBlocking(
                std::io::Error::last_os_error(),
            ));
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(ProcessCaptureError::NonBlocking(
                std::io::Error::last_os_error(),
            ));
        }
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Failure preventing a trustworthy raw process record from being completed.
#[derive(Debug, Error)]
pub enum ProcessCaptureError {
    /// Executable field empty.
    #[error("process command cannot be empty")]
    EmptyCommand,
    /// Timeout outside supported bounds.
    #[error("process timeout {timeout_ms} ms outside 1..={maximum_ms}")]
    InvalidTimeout {
        /// Requested timeout.
        timeout_ms: u64,
        /// Maximum supported timeout.
        maximum_ms: u64,
    },
    /// Per-stream retention limit outside supported bounds.
    #[error("process output limit {output_bytes} outside 1..={maximum_bytes} bytes")]
    InvalidOutputLimit {
        /// Requested limit.
        output_bytes: usize,
        /// Maximum supported limit.
        maximum_bytes: usize,
    },
    /// Environment key invalid for an OS process environment.
    #[error("invalid process environment key")]
    InvalidEnvironmentKey,
    /// Environment value contains an embedded NUL.
    #[error("invalid process environment value")]
    InvalidEnvironmentValue,
    /// Platform does not support the audited non-blocking pipe implementation.
    #[error("raw process capture currently requires a Unix platform")]
    UnsupportedPlatform,
    /// Child spawn failed.
    #[error("failed to spawn process '{command}': {source}")]
    Spawn {
        /// Requested command.
        command: String,
        /// OS error.
        #[source]
        source: std::io::Error,
    },
    /// Stdout pipe unexpectedly unavailable.
    #[error("child stdout pipe unavailable")]
    MissingStdoutPipe,
    /// Stderr pipe unexpectedly unavailable.
    #[error("child stderr pipe unavailable")]
    MissingStderrPipe,
    /// Failed to configure non-blocking pipe I/O.
    #[error("failed to configure non-blocking child pipe: {0}")]
    NonBlocking(#[source] std::io::Error),
    /// Failed to read stdout.
    #[error("failed to read child stdout: {0}")]
    ReadStdout(#[source] std::io::Error),
    /// Failed to read stderr.
    #[error("failed to read child stderr: {0}")]
    ReadStderr(#[source] std::io::Error),
    /// Failed while polling the direct child.
    #[error("failed to poll child process: {0}")]
    Poll(#[source] std::io::Error),
    /// Failed to reap the direct child after enforcement termination.
    #[error("failed to wait for child process: {0}")]
    Wait(#[source] std::io::Error),
    /// Serialization failure while constructing content-addressed identities.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn successful_capture_preserves_stdout_and_manifest_identity() {
        let spec = ProcessSpec::new("printf").arg("hello-capture");
        let capture = capture_process(&spec).unwrap();
        assert!(capture.process_success());
        assert_eq!(capture.stdout, b"hello-capture");
        assert!(capture.stderr.is_empty());
        assert_eq!(capture.command_manifest_sha256, spec.manifest_sha256().unwrap());
        assert_eq!(capture.capture_sha256().unwrap().len(), 64);
    }

    #[test]
    fn nonzero_exit_is_preserved_as_evidence_not_discarded() {
        let spec = ProcessSpec::new("sh")
            .arg("-c")
            .arg("printf problem >&2; exit 7");
        let capture = capture_process(&spec).unwrap();
        assert_eq!(
            capture.termination,
            ProcessTermination::Exited {
                exit_code: Some(7),
                success: false,
            }
        );
        assert_eq!(capture.stderr, b"problem");
        assert!(!capture.process_success());
    }

    #[test]
    fn timeout_preserves_partial_output_and_reason() {
        let spec = ProcessSpec::new("sh")
            .arg("-c")
            .arg("printf before-timeout; sleep 1")
            .timeout(Duration::from_millis(30));
        let started = Instant::now();
        let capture = capture_process(&spec).unwrap();
        assert_eq!(capture.termination, ProcessTermination::TimedOut);
        assert_eq!(capture.stdout, b"before-timeout");
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn output_limit_preserves_prefix_and_truncation_state() {
        let spec = ProcessSpec::new("printf")
            .arg("0123456789abcdef")
            .max_output_bytes(8);
        let capture = capture_process(&spec).unwrap();
        assert_eq!(capture.termination, ProcessTermination::OutputLimitExceeded);
        assert_eq!(capture.stdout, b"01234567");
        assert!(capture.stdout_truncated);
        assert!(!capture.process_success());
    }

    #[test]
    fn clear_environment_is_part_of_command_identity_and_behavior() {
        let inherited = ProcessSpec::new("/bin/sh")
            .arg("-c")
            .arg("printf %s \"${PATH:+present}\"");
        let closed = inherited.clone().clear_environment();
        assert_ne!(inherited.manifest_sha256().unwrap(), closed.manifest_sha256().unwrap());
        let capture = capture_process(&closed).unwrap();
        assert!(capture.process_success());
        assert!(capture.stdout.is_empty());
    }

    #[test]
    fn explicit_closed_environment_is_reproducible() {
        let spec = ProcessSpec::new("/bin/sh")
            .arg("-c")
            .arg("printf %s \"$MAG_TEST\"")
            .clear_environment()
            .env("MAG_TEST", "bound-value");
        let capture = capture_process(&spec).unwrap();
        assert!(capture.process_success());
        assert_eq!(capture.stdout, b"bound-value");
    }

    #[test]
    fn missing_binary_is_a_capture_error() {
        let spec = ProcessSpec::new("symthaea-nonexistent-process-capture-binary-xyz");
        assert!(matches!(
            capture_process(&spec),
            Err(ProcessCaptureError::Spawn { .. })
        ));
    }

    #[test]
    fn elapsed_time_is_process_evidence_not_convergence_evidence() {
        let capture = capture_process(&ProcessSpec::new("true")).unwrap();
        assert!(capture.process_success());
        // No convergence field exists by design: only a domain parser may grant it.
        assert!(capture.elapsed_ms < 10_000);
    }
}
