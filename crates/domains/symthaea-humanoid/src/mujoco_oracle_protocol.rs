// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned one-shot protocol for an independently built MuJoCo oracle worker.
//!
//! The candidate process never imports an oracle snapshot from its own runtime.
//! Instead it launches a separately identified executable, supplies an exact
//! generalized state, and independently revalidates the returned structure and
//! state fingerprint before dataset admission.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::floating_base::FloatingBaseDynamicsSnapshot;
use crate::morphology::HumanoidMorphology;
use crate::oracle_dataset::fingerprint_state;

pub const MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MujocoOracleWorkerRequest {
    pub protocol_version: u32,
    pub request_id: u64,
    pub generator_id: String,
    pub generator_build_id: String,
    pub candidate_build_id: String,
    pub engine_id: String,
    pub morphology: HumanoidMorphology,
    pub generalized_position: Vec<f64>,
    pub generalized_velocity: Vec<f64>,
    pub actuator_command: Vec<f64>,
    pub state_fingerprint: u64,
}

impl MujocoOracleWorkerRequest {
    pub fn validate(&self) -> bool {
        let actuators = self.morphology.num_actuators();
        self.protocol_version == MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION
            && self.request_id != 0
            && !self.generator_id.trim().is_empty()
            && !self.generator_build_id.trim().is_empty()
            && !self.candidate_build_id.trim().is_empty()
            && self.generator_build_id != self.candidate_build_id
            && !self.engine_id.trim().is_empty()
            && self.generalized_position.len() == 7 + actuators
            && self.generalized_velocity.len() == 6 + actuators
            && self.actuator_command.len() == actuators
            && self
                .generalized_position
                .iter()
                .chain(self.generalized_velocity.iter())
                .chain(self.actuator_command.iter())
                .all(|value| value.is_finite())
            && fingerprint_state(
                &self.generalized_position,
                &self.generalized_velocity,
                &self.actuator_command,
            ) == self.state_fingerprint
            && root_quaternion_is_unit(&self.generalized_position)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MujocoOracleWorkerResponse {
    pub protocol_version: u32,
    pub request_id: u64,
    pub generator_id: String,
    pub generator_build_id: String,
    pub engine_id: String,
    pub state_fingerprint: u64,
    pub oracle: Option<FloatingBaseDynamicsSnapshot>,
    pub error: Option<String>,
}

impl MujocoOracleWorkerResponse {
    pub fn success(
        request: &MujocoOracleWorkerRequest,
        oracle: FloatingBaseDynamicsSnapshot,
    ) -> Self {
        Self {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: request.request_id,
            generator_id: request.generator_id.clone(),
            generator_build_id: request.generator_build_id.clone(),
            engine_id: request.engine_id.clone(),
            state_fingerprint: request.state_fingerprint,
            oracle: Some(oracle),
            error: None,
        }
    }

    pub fn failure(request: &MujocoOracleWorkerRequest, error: impl Into<String>) -> Self {
        Self {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: request.request_id,
            generator_id: request.generator_id.clone(),
            generator_build_id: request.generator_build_id.clone(),
            engine_id: request.engine_id.clone(),
            state_fingerprint: request.state_fingerprint,
            oracle: None,
            error: Some(error.into()),
        }
    }

    pub fn validate_against(&self, request: &MujocoOracleWorkerRequest) -> bool {
        if !request.validate()
            || self.protocol_version != MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION
            || self.request_id != request.request_id
            || self.generator_id != request.generator_id
            || self.generator_build_id != request.generator_build_id
            || self.engine_id != request.engine_id
            || self.state_fingerprint != request.state_fingerprint
            || self.error.is_some()
        {
            return false;
        }
        self.oracle
            .as_ref()
            .map(|oracle| {
                oracle.validate()
                    && oracle.morphology == request.morphology
                    && oracle.generalized_velocity_count == request.generalized_velocity.len()
            })
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct ProcessMujocoOracleGenerator {
    executable: PathBuf,
    arguments: Vec<String>,
    maximum_response_bytes: usize,
    /// Expected executable bytes, provisioned from a protected build manifest.
    expected_executable_sha256: Option<String>,
    maximum_execution_time: Duration,
}

impl ProcessMujocoOracleGenerator {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            maximum_response_bytes: 64 * 1024 * 1024,
            expected_executable_sha256: None,
            maximum_execution_time: Duration::from_secs(5),
        }
    }

    pub fn with_arguments(mut self, arguments: impl IntoIterator<Item = String>) -> Self {
        self.arguments = arguments.into_iter().collect();
        self
    }

    pub fn with_maximum_response_bytes(mut self, maximum_response_bytes: usize) -> Self {
        self.maximum_response_bytes = maximum_response_bytes.max(1024);
        self
    }

    /// Pin the worker binary to an externally trusted SHA-256 digest.
    ///
    /// The digest should come from a protected build/qualification manifest.
    /// This path-level check detects persistent artifact drift before or after
    /// invocation; it is not OS-level process-image attestation or sandboxing.
    pub fn with_expected_executable_sha256(mut self, digest: impl Into<String>) -> Self {
        self.expected_executable_sha256 = Some(digest.into());
        self
    }

    pub fn expected_executable_sha256(&self) -> Option<&str> {
        self.expected_executable_sha256.as_deref()
    }

    /// Limit wall time for each oracle process. Values below 1 ms are clamped.
    pub fn with_maximum_execution_time(mut self, maximum_execution_time: Duration) -> Self {
        self.maximum_execution_time = maximum_execution_time.max(Duration::from_millis(1));
        self
    }

    pub fn executable(&self) -> &Path {
        &self.executable
    }

    pub fn generate(
        &self,
        request: &MujocoOracleWorkerRequest,
    ) -> Option<MujocoOracleWorkerResponse> {
        if self.executable.as_os_str().is_empty() || !request.validate() {
            return None;
        }
        let expected_digest = self.expected_executable_sha256.as_deref()?;
        if !is_lower_hex_sha256(expected_digest)
            || executable_file_sha256(&self.executable).as_deref() != Some(expected_digest)
        {
            return None;
        }
        let payload = serde_json::to_vec(request).ok()?;
        let deadline = Instant::now().checked_add(self.maximum_execution_time)?;
        let mut command = Command::new(&self.executable);
        command
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn().ok()?;
        {
            let Some(mut stdin) = child.stdin.take() else {
                terminate_worker_process_group(&mut child);
                let _ = child.wait();
                return None;
            };
            if stdin.write_all(&payload).is_err() || stdin.write_all(b"\n").is_err() {
                terminate_worker_process_group(&mut child);
                let _ = child.wait();
                return None;
            }
        }

        // Drain stdout concurrently so the child cannot deadlock on a full pipe.
        // Retain at most maximum_response_bytes, and signal overflow immediately.
        let Some(stdout) = child.stdout.take() else {
            terminate_worker_process_group(&mut child);
            let _ = child.wait();
            return None;
        };
        let response_limit = self.maximum_response_bytes;
        let overflow = Arc::new(AtomicBool::new(false));
        let reader_overflow = Arc::clone(&overflow);
        let reader_done = Arc::new(AtomicBool::new(false));
        let reader_done_flag = Arc::clone(&reader_done);
        let (reader_tx, reader_rx) = mpsc::channel();
        let _reader = thread::spawn(move || {
            let result = (|| -> std::io::Result<Vec<u8>> {
                let mut stream = stdout;
                let mut bytes = Vec::with_capacity(response_limit.min(64 * 1024));
                let mut buffer = [0u8; 8192];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    let remaining = response_limit.saturating_sub(bytes.len());
                    if read > remaining {
                        bytes.extend_from_slice(&buffer[..remaining]);
                        reader_overflow.store(true, Ordering::Release);
                        return Ok(bytes);
                    }
                    bytes.extend_from_slice(&buffer[..read]);
                }
                Ok(bytes)
            })();
            let _ = reader_tx.send(result);
            reader_done_flag.store(true, Ordering::Release);
        });

        let mut timed_out = false;
        let mut wait_failed = false;
        let mut status: Option<ExitStatus> = None;
        loop {
            if overflow.load(Ordering::Acquire) {
                terminate_worker_process_group(&mut child);
                status = child.wait().ok();
                break;
            }
            if status.is_none() {
                match child.try_wait() {
                    Ok(Some(exit_status)) => status = Some(exit_status),
                    Ok(None) => {}
                    Err(_) => {
                        wait_failed = true;
                        terminate_worker_process_group(&mut child);
                        status = child.wait().ok();
                        break;
                    }
                }
            }
            // Fail closed at the deadline before accepting a result. Because
            // process status is polled, this can reject a completion observed
            // just after the deadline even if it exited moments earlier.
            if Instant::now() >= deadline {
                timed_out = true;
                terminate_worker_process_group(&mut child);
                if status.is_none() {
                    status = child.wait().ok();
                }
                break;
            }
            // Process exit is insufficient: a forked descendant may still hold
            // stdout open. Wait for EOF as well, under the same hard deadline.
            if status.is_some() && reader_done.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        let status = status?;
        // Never block forever waiting for EOF from a descendant that inherited
        // stdout. On platforms without process-group termination, a malicious
        // descendant may leave this bounded reader thread alive until it exits.
        let stdout_bytes = reader_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .ok()?
            .ok()?;
        if timed_out
            || wait_failed
            || overflow.load(Ordering::Acquire)
            || !status.success()
            || stdout_bytes.is_empty()
            || stdout_bytes.len() > self.maximum_response_bytes
            || executable_file_sha256(&self.executable).as_deref() != Some(expected_digest)
        {
            return None;
        }
        let response: MujocoOracleWorkerResponse = serde_json::from_slice(&stdout_bytes).ok()?;
        response.validate_against(request).then_some(response)
    }
}

fn terminate_worker_process_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let process_group_id = child.id() as libc::pid_t;
        // The child was launched as leader of a fresh process group. Killing
        // the group also closes inherited stdout handles held by worker children.
        unsafe {
            let _ = libc::kill(-process_group_id, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn executable_file_sha256(path: &Path) -> Option<String> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(&bytes)))
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn root_quaternion_is_unit(position: &[f64]) -> bool {
    if position.len() < 7 {
        return false;
    }
    let norm_squared = position[3..7]
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    norm_squared.is_finite() && (norm_squared - 1.0).abs() <= 1.0e-6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> MujocoOracleWorkerRequest {
        let morphology = HumanoidMorphology::Dmc21;
        let mut generalized_position = vec![0.0; 7 + morphology.num_actuators()];
        generalized_position[2] = 1.35;
        generalized_position[3] = 1.0;
        let generalized_velocity = vec![0.0; 6 + morphology.num_actuators()];
        let actuator_command = vec![0.0; morphology.num_actuators()];
        let state_fingerprint = fingerprint_state(
            &generalized_position,
            &generalized_velocity,
            &actuator_command,
        );
        MujocoOracleWorkerRequest {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: 1,
            generator_id: "independent-mujoco-oracle".to_string(),
            generator_build_id: "oracle-build".to_string(),
            candidate_build_id: "candidate-build".to_string(),
            engine_id: "mujoco-3.x".to_string(),
            morphology,
            generalized_position,
            generalized_velocity,
            actuator_command,
            state_fingerprint,
        }
    }

    #[test]
    fn executable_digest_must_be_explicit_and_match_file_bytes() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-oracle-worker-{}-digest-test",
            std::process::id()
        ));
        std::fs::write(&path, b"not an executable").unwrap();

        let unpinned = ProcessMujocoOracleGenerator::new(&path);
        assert!(unpinned.generate(&request()).is_none());

        let mismatched = ProcessMujocoOracleGenerator::new(&path)
            .with_expected_executable_sha256("a".repeat(64));
        assert!(mismatched.generate(&request()).is_none());

        let expected = format!("{:x}", Sha256::digest(b"not an executable"));
        let pinned = ProcessMujocoOracleGenerator::new(&path)
            .with_expected_executable_sha256(expected.clone());
        assert_eq!(pinned.expected_executable_sha256(), Some(expected.as_str()));
        assert_eq!(executable_file_sha256(&path).as_deref(), Some(expected.as_str()));

        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn oracle_worker_timeout_terminates_the_process() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "symthaea-oracle-worker-{}-timeout-test",
            std::process::id()
        ));
        std::fs::write(&path, b"#!/bin/sh\nsleep 10 &\nwait\n").unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        let digest = executable_file_sha256(&path).unwrap();

        let worker = ProcessMujocoOracleGenerator::new(&path)
            .with_expected_executable_sha256(digest)
            .with_maximum_execution_time(Duration::from_millis(25));
        let started = Instant::now();
        assert!(worker.generate(&request()).is_none());
        assert!(started.elapsed() < Duration::from_secs(2));

        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn worker_parent_exit_does_not_bypass_deadline_while_descendant_holds_stdout() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "symthaea-oracle-worker-{}-orphan-pipe-test",
            std::process::id()
        ));
        std::fs::write(&path, b"#!/bin/sh\nsleep 10 &\nprintf 'not-json\\n'\nexit 0\n")
            .unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        let digest = executable_file_sha256(&path).unwrap();

        let worker = ProcessMujocoOracleGenerator::new(&path)
            .with_expected_executable_sha256(digest)
            .with_maximum_execution_time(Duration::from_millis(40));
        let started = Instant::now();
        assert!(worker.generate(&request()).is_none());
        assert!(started.elapsed() < Duration::from_secs(2));

        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn oracle_worker_output_is_bounded_during_read() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "symthaea-oracle-worker-{}-output-limit-test",
            std::process::id()
        ));
        std::fs::write(&path, b"#!/bin/sh\nprintf '%2048s' x\n").unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        let digest = executable_file_sha256(&path).unwrap();

        let worker = ProcessMujocoOracleGenerator::new(&path)
            .with_expected_executable_sha256(digest)
            .with_maximum_response_bytes(1024)
            .with_maximum_execution_time(Duration::from_secs(1));
        assert!(worker.generate(&request()).is_none());

        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn executable_digest_rejects_symlink_paths() {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "symthaea-oracle-worker-{}-symlink-test",
            std::process::id()
        ));
        let target = base.with_extension("target");
        let alias = base.with_extension("alias");
        std::fs::write(&target, b"oracle bytes").unwrap();
        symlink(&target, &alias).unwrap();
        assert!(executable_file_sha256(&alias).is_none());
        std::fs::remove_file(alias).unwrap();
        std::fs::remove_file(target).unwrap();
    }

    #[test]
    fn self_oracle_identity_is_rejected() {
        let mut request = request();
        request.candidate_build_id = request.generator_build_id.clone();
        assert!(!request.validate());
    }

    #[test]
    fn non_unit_root_quaternion_is_rejected() {
        let mut request = request();
        request.generalized_position[3] = 2.0;
        request.state_fingerprint = fingerprint_state(
            &request.generalized_position,
            &request.generalized_velocity,
            &request.actuator_command,
        );
        assert!(!request.validate());
    }
}
