// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! One-shot process boundary for simulation Component execution.
//!
//! This crate intentionally separates three claims:
//!
//! 1. **wire correctness** — exact manifest bytes, Component bytes and one typed
//!    request cross a length-bounded one-shot protocol;
//! 2. **technical execution** — the child invokes `SimulationComponentHost` and
//!    returns a technical result whose `SimulationEvidence` is still empty;
//! 3. **process supervision** — on Linux, the parent applies conservative
//!    rlimits, `PR_SET_NO_NEW_PRIVS`, an empty environment and a wall-clock kill.
//!
//! This is not yet a complete hostile-code sandbox. In particular this tranche
//! does not claim seccomp or network-namespace containment. Admission, signer
//! trust, routing authority and engineering-evidence promotion remain outside
//! the worker entirely.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_extension_host::CONTROL_WASM_PROFILE_V1;
use symthaea_extension_simulation_host::{
    SIMULATION_ADAPTER_V1, SIMULATION_WASM_PROFILE_V1, SIMULATION_WIT_V1,
};
use symthaea_sim_bridge::{SimulationEvidence, SimulationRequest, SimulationResult};
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use thiserror::Error;

pub const WORKER_PROTOCOL_V1: &str = "symthaea.simulation.worker.v1";
pub const SUPERVISOR_PROFILE_V1: &str = "symthaea.simulation.worker-supervisor.linux-rlimit-v1";

const REQUEST_MAGIC: &[u8; 8] = b"SYMWRQ01";
const RESPONSE_MAGIC: &[u8; 8] = b"SYMWRS01";
const FRAME_VERSION: u32 = 1;
const MAX_WALL_TIME_MS: u64 = 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerFrameLimits {
    pub max_manifest_bytes: u64,
    pub max_component_bytes: u64,
    pub max_request_json_bytes: u64,
    pub max_response_json_bytes: u64,
}

impl Default for WorkerFrameLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 1024 * 1024,
            max_component_bytes: 128 * 1024 * 1024,
            max_request_json_bytes: 1024 * 1024,
            max_response_json_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkerRequestFrame {
    pub manifest_bytes: Vec<u8>,
    pub component_bytes: Vec<u8>,
    pub request: SimulationRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerFailureKind {
    InvalidFrame,
    InvalidRequest,
    HostFailure,
    CanonicalizationFailure,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerFailure {
    pub kind: WorkerFailureKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSuccess {
    pub profile: String,
    pub manifest_sha256: String,
    pub component_sha256: String,
    pub request_sha256: String,
    pub output_sha256: String,
    pub extension_id: String,
    pub extension_version: String,
    pub control_wasm_profile: String,
    pub simulation_wasm_profile: String,
    pub wit_version: String,
    pub adapter_version: String,
    pub result: SimulationResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WorkerResponse {
    Ok(WorkerSuccess),
    Err(WorkerFailure),
}

#[derive(Debug, Error)]
pub enum WorkerProtocolError {
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
    #[error("unexpected worker frame magic")]
    BadMagic,
    #[error("unsupported worker frame version {0}")]
    UnsupportedVersion(u32),
    #[error("{field} length {actual} exceeds limit {limit}")]
    FrameTooLarge {
        field: &'static str,
        actual: u64,
        limit: u64,
    },
    #[error("frame length cannot be represented on this host")]
    LengthOverflow,
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn write_request_frame(
    mut writer: impl Write,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
    limits: WorkerFrameLimits,
) -> Result<(), WorkerProtocolError> {
    let request_json = serde_json::to_vec(request)?;
    check_len("manifest", manifest_bytes.len(), limits.max_manifest_bytes)?;
    check_len("component", component_bytes.len(), limits.max_component_bytes)?;
    check_len("request", request_json.len(), limits.max_request_json_bytes)?;

    writer.write_all(REQUEST_MAGIC)?;
    writer.write_all(&FRAME_VERSION.to_le_bytes())?;
    write_len(&mut writer, manifest_bytes.len())?;
    write_len(&mut writer, component_bytes.len())?;
    write_len(&mut writer, request_json.len())?;
    writer.write_all(manifest_bytes)?;
    writer.write_all(component_bytes)?;
    writer.write_all(&request_json)?;
    writer.flush()?;
    Ok(())
}

pub fn read_request_frame(
    mut reader: impl Read,
    limits: WorkerFrameLimits,
) -> Result<WorkerRequestFrame, WorkerProtocolError> {
    require_magic(&mut reader, REQUEST_MAGIC)?;
    let version = read_u32(&mut reader)?;
    if version != FRAME_VERSION {
        return Err(WorkerProtocolError::UnsupportedVersion(version));
    }
    let manifest_len = read_u64(&mut reader)?;
    let component_len = read_u64(&mut reader)?;
    let request_len = read_u64(&mut reader)?;
    check_u64("manifest", manifest_len, limits.max_manifest_bytes)?;
    check_u64("component", component_len, limits.max_component_bytes)?;
    check_u64("request", request_len, limits.max_request_json_bytes)?;

    let manifest_bytes = read_exact_vec(&mut reader, manifest_len)?;
    let component_bytes = read_exact_vec(&mut reader, component_len)?;
    let request_json = read_exact_vec(&mut reader, request_len)?;
    let request = serde_json::from_slice(&request_json)?;
    Ok(WorkerRequestFrame {
        manifest_bytes,
        component_bytes,
        request,
    })
}

pub fn write_response_frame(
    mut writer: impl Write,
    response: &WorkerResponse,
    limits: WorkerFrameLimits,
) -> Result<(), WorkerProtocolError> {
    let json = serde_json::to_vec(response)?;
    check_len("response", json.len(), limits.max_response_json_bytes)?;
    writer.write_all(RESPONSE_MAGIC)?;
    writer.write_all(&FRAME_VERSION.to_le_bytes())?;
    write_len(&mut writer, json.len())?;
    writer.write_all(&json)?;
    writer.flush()?;
    Ok(())
}

pub fn read_response_frame(
    mut reader: impl Read,
    limits: WorkerFrameLimits,
) -> Result<WorkerResponse, WorkerProtocolError> {
    require_magic(&mut reader, RESPONSE_MAGIC)?;
    let version = read_u32(&mut reader)?;
    if version != FRAME_VERSION {
        return Err(WorkerProtocolError::UnsupportedVersion(version));
    }
    let response_len = read_u64(&mut reader)?;
    check_u64("response", response_len, limits.max_response_json_bytes)?;
    let json = read_exact_vec(&mut reader, response_len)?;
    Ok(serde_json::from_slice(&json)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupervisorLimits {
    pub address_space_bytes: u64,
    pub cpu_seconds: u64,
    pub file_size_bytes: u64,
    pub open_files: u64,
    pub process_count: u64,
    pub wall_time_ms: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
}

impl Default for SupervisorLimits {
    fn default() -> Self {
        Self {
            address_space_bytes: 2 * 1024 * 1024 * 1024,
            cpu_seconds: 15,
            file_size_bytes: 1024 * 1024,
            open_files: 32,
            process_count: 64,
            wall_time_ms: 10_000,
            max_stdout_bytes: 16 * 1024 * 1024,
            max_stderr_bytes: 256 * 1024,
        }
    }
}

impl SupervisorLimits {
    pub fn validate(self) -> Result<Self, SupervisorError> {
        if self.address_space_bytes == 0
            || self.cpu_seconds == 0
            || self.open_files < 3
            || self.process_count == 0
            || self.wall_time_ms == 0
            || self.wall_time_ms > MAX_WALL_TIME_MS
            || self.max_stdout_bytes == 0
            || self.max_stderr_bytes == 0
        {
            return Err(SupervisorError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub struct SupervisedWorker {
    executable: PathBuf,
    limits: SupervisorLimits,
    frame_limits: WorkerFrameLimits,
}

impl SupervisedWorker {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            limits: SupervisorLimits::default(),
            frame_limits: WorkerFrameLimits::default(),
        }
    }

    pub fn with_limits(mut self, limits: SupervisorLimits) -> Result<Self, SupervisorError> {
        self.limits = limits.validate()?;
        Ok(self)
    }

    pub fn with_frame_limits(mut self, frame_limits: WorkerFrameLimits) -> Self {
        self.frame_limits = frame_limits;
        self
    }

    pub fn execute(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
        request: &SimulationRequest,
    ) -> Result<SupervisedInvocation, SupervisorError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (manifest_bytes, component_bytes, request);
            return Err(SupervisorError::UnsupportedPlatform);
        }

        #[cfg(target_os = "linux")]
        self.execute_linux(manifest_bytes, component_bytes, request)
    }

    #[cfg(target_os = "linux")]
    fn execute_linux(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
        request: &SimulationRequest,
    ) -> Result<SupervisedInvocation, SupervisorError> {
        self.limits.validate()?;
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
            .map_err(SupervisorError::ManifestJson)?;
        manifest
            .validate()
            .map_err(|problems| SupervisorError::ManifestInvalid(format!("{problems:?}")))?;
        if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
            return Err(SupervisorError::UnexpectedManifest);
        }

        let worker_before = sha256_file(&self.executable)?;
        let expected_manifest: [u8; 32] = Sha256::digest(manifest_bytes).into();
        let expected_component: [u8; 32] = Sha256::digest(component_bytes).into();
        let expected_request = canonical_request_sha256_v1(request)
            .map_err(|error| SupervisorError::Canonical(error.to_string()))?;

        // Encode the bounded frame before spawning. The actual pipe write occurs
        // on a separate thread so a child that never reads stdin cannot block
        // the parent before wall-clock supervision begins.
        let mut request_frame = Vec::new();
        write_request_frame(
            &mut request_frame,
            manifest_bytes,
            component_bytes,
            request,
            self.frame_limits,
        )?;

        let mut command = Command::new(&self.executable);
        command
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_linux_supervision(&mut command, self.limits);

        let mut child = command.spawn().map_err(SupervisorError::Spawn)?;
        let mut stdin = child.stdin.take().ok_or(SupervisorError::MissingPipe("stdin"))?;
        let stdout = child.stdout.take().ok_or(SupervisorError::MissingPipe("stdout"))?;
        let stderr = child.stderr.take().ok_or(SupervisorError::MissingPipe("stderr"))?;

        let input_writer = thread::spawn(move || -> io::Result<()> {
            stdin.write_all(&request_frame)?;
            stdin.flush()?;
            Ok(())
        });
        let stdout_limit = self.limits.max_stdout_bytes;
        let stderr_limit = self.limits.max_stderr_bytes;
        let stdout_reader =
            thread::spawn(move || read_limited(stdout, stdout_limit, "stdout"));
        let stderr_reader =
            thread::spawn(move || read_limited(stderr, stderr_limit, "stderr"));

        let deadline = Instant::now() + Duration::from_millis(self.limits.wall_time_ms);
        let status = loop {
            if let Some(status) = child.try_wait().map_err(SupervisorError::Wait)? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                let _ = input_writer.join();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(SupervisorError::WallTimeExceeded(self.limits.wall_time_ms));
            }
            thread::sleep(Duration::from_millis(5));
        };

        let input_result = input_writer
            .join()
            .map_err(|_| SupervisorError::WriterPanicked)?;
        let stdout = stdout_reader
            .join()
            .map_err(|_| SupervisorError::ReaderPanicked("stdout"))??;
        let stderr = stderr_reader
            .join()
            .map_err(|_| SupervisorError::ReaderPanicked("stderr"))??;

        if !status.success() {
            return Err(SupervisorError::WorkerExited {
                code: status.code(),
                stderr: bounded_utf8(&stderr),
            });
        }
        input_result.map_err(SupervisorError::InputWrite)?;

        let worker_after = sha256_file(&self.executable)?;
        if worker_before != worker_after {
            return Err(SupervisorError::WorkerExecutableChanged);
        }

        let mut cursor = Cursor::new(stdout.as_slice());
        let response = read_response_frame(&mut cursor, self.frame_limits)?;
        if cursor.position() != u64::try_from(stdout.len()).unwrap_or(u64::MAX) {
            return Err(SupervisorError::TrailingResponseBytes);
        }
        let success = match response {
            WorkerResponse::Ok(success) => success,
            WorkerResponse::Err(failure) => return Err(SupervisorError::WorkerFailure(failure)),
        };
        verify_success(
            &success,
            expected_manifest,
            expected_component,
            expected_request,
            request,
            &manifest,
        )?;

        Ok(SupervisedInvocation {
            success,
            worker_sha256: worker_after,
            supervisor_profile: SUPERVISOR_PROFILE_V1,
            limits: self.limits,
        })
    }
}

#[derive(Debug)]
pub struct SupervisedInvocation {
    success: WorkerSuccess,
    worker_sha256: [u8; 32],
    supervisor_profile: &'static str,
    limits: SupervisorLimits,
}

impl SupervisedInvocation {
    pub fn result(&self) -> &SimulationResult {
        &self.success.result
    }

    pub fn into_result(self) -> SimulationResult {
        self.success.result
    }

    pub fn worker(&self) -> &WorkerSuccess {
        &self.success
    }

    pub const fn worker_sha256(&self) -> [u8; 32] {
        self.worker_sha256
    }

    pub const fn supervisor_profile(&self) -> &'static str {
        self.supervisor_profile
    }

    pub const fn limits(&self) -> SupervisorLimits {
        self.limits
    }
}

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("supervised simulation worker is currently supported only on Linux")]
    UnsupportedPlatform,
    #[error("supervisor limits are invalid")]
    InvalidLimits,
    #[error("manifest json is invalid: {0}")]
    ManifestJson(serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("supervised worker requires a wasm simulation manifest")]
    UnexpectedManifest,
    #[error("failed to read worker executable: {0}")]
    WorkerExecutable(#[source] io::Error),
    #[error("failed to spawn simulation worker: {0}")]
    Spawn(#[source] io::Error),
    #[error("worker pipe {0} is unavailable")]
    MissingPipe(&'static str),
    #[error(transparent)]
    Protocol(#[from] WorkerProtocolError),
    #[error("canonicalization failed: {0}")]
    Canonical(String),
    #[error("failed while waiting for worker: {0}")]
    Wait(#[source] io::Error),
    #[error("failed writing worker input: {0}")]
    InputWrite(#[source] io::Error),
    #[error("worker stdin writer thread panicked")]
    WriterPanicked,
    #[error("worker exceeded wall-clock limit of {0} ms")]
    WallTimeExceeded(u64),
    #[error("{0} reader thread panicked")]
    ReaderPanicked(&'static str),
    #[error("failed reading worker {stream}: {source}")]
    OutputRead {
        stream: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("worker output exceeded configured {stream} limit")]
    OutputTooLarge { stream: &'static str },
    #[error("worker exited unsuccessfully: code={code:?}, stderr={stderr}")]
    WorkerExited { code: Option<i32>, stderr: String },
    #[error("worker executable changed between preflight and postflight hashing")]
    WorkerExecutableChanged,
    #[error("worker response contained trailing bytes")]
    TrailingResponseBytes,
    #[error("worker returned typed failure: {0:?}")]
    WorkerFailure(WorkerFailure),
    #[error("worker response profile mismatch")]
    ProfileMismatch,
    #[error("worker response {0} does not match parent-computed value")]
    CommitmentMismatch(&'static str),
    #[error("worker attempted to mint simulation evidence")]
    WorkerMintedEvidence,
}

fn verify_success(
    success: &WorkerSuccess,
    expected_manifest: [u8; 32],
    expected_component: [u8; 32],
    expected_request: [u8; 32],
    request: &SimulationRequest,
    manifest: &ExtensionManifest,
) -> Result<(), SupervisorError> {
    if success.profile != WORKER_PROTOCOL_V1 {
        return Err(SupervisorError::ProfileMismatch);
    }
    require_hex("manifest", &success.manifest_sha256, expected_manifest)?;
    require_hex("component", &success.component_sha256, expected_component)?;
    require_hex("request", &success.request_sha256, expected_request)?;
    if success.result.request_id != request.id {
        return Err(SupervisorError::CommitmentMismatch("request id"));
    }
    if success.result.evidence != SimulationEvidence::default() {
        return Err(SupervisorError::WorkerMintedEvidence);
    }
    let output = canonical_output_sha256_v1(&success.result)
        .map_err(|error| SupervisorError::Canonical(error.to_string()))?;
    require_hex("output", &success.output_sha256, output)?;
    if success.extension_id != manifest.id.as_str() {
        return Err(SupervisorError::CommitmentMismatch("extension id"));
    }
    if success.extension_version != manifest.version {
        return Err(SupervisorError::CommitmentMismatch("extension version"));
    }
    if success.control_wasm_profile != CONTROL_WASM_PROFILE_V1 {
        return Err(SupervisorError::CommitmentMismatch("control runtime profile"));
    }
    if success.simulation_wasm_profile != SIMULATION_WASM_PROFILE_V1 {
        return Err(SupervisorError::CommitmentMismatch("simulation runtime profile"));
    }
    if success.wit_version != SIMULATION_WIT_V1 {
        return Err(SupervisorError::CommitmentMismatch("wit version"));
    }
    if success.adapter_version != SIMULATION_ADAPTER_V1 {
        return Err(SupervisorError::CommitmentMismatch("adapter version"));
    }
    Ok(())
}

fn require_hex(
    field: &'static str,
    actual: &str,
    expected: [u8; 32],
) -> Result<(), SupervisorError> {
    if actual != hex_digest(expected) {
        return Err(SupervisorError::CommitmentMismatch(field));
    }
    Ok(())
}

fn check_len(field: &'static str, actual: usize, limit: u64) -> Result<(), WorkerProtocolError> {
    let actual = u64::try_from(actual).map_err(|_| WorkerProtocolError::LengthOverflow)?;
    check_u64(field, actual, limit)
}

fn check_u64(field: &'static str, actual: u64, limit: u64) -> Result<(), WorkerProtocolError> {
    if actual > limit {
        return Err(WorkerProtocolError::FrameTooLarge {
            field,
            actual,
            limit,
        });
    }
    Ok(())
}

fn write_len(writer: &mut impl Write, len: usize) -> Result<(), WorkerProtocolError> {
    let len = u64::try_from(len).map_err(|_| WorkerProtocolError::LengthOverflow)?;
    writer.write_all(&len.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32, WorkerProtocolError> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, WorkerProtocolError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact_vec(reader: &mut impl Read, len: u64) -> Result<Vec<u8>, WorkerProtocolError> {
    let len = usize::try_from(len).map_err(|_| WorkerProtocolError::LengthOverflow)?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn require_magic(reader: &mut impl Read, expected: &[u8; 8]) -> Result<(), WorkerProtocolError> {
    let mut actual = [0u8; 8];
    reader.read_exact(&mut actual)?;
    if &actual != expected {
        return Err(WorkerProtocolError::BadMagic);
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<[u8; 32], SupervisorError> {
    let bytes = fs::read(path).map_err(SupervisorError::WorkerExecutable)?;
    Ok(Sha256::digest(bytes).into())
}

fn read_limited(
    mut reader: impl Read,
    limit: u64,
    stream: &'static str,
) -> Result<Vec<u8>, SupervisorError> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| SupervisorError::OutputRead { stream, source })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(SupervisorError::OutputTooLarge { stream });
    }
    Ok(bytes)
}

fn bounded_utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(target_os = "linux")]
fn apply_linux_supervision(command: &mut Command, limits: SupervisorLimits) {
    use std::os::unix::process::CommandExt;

    // SAFETY: `pre_exec` runs in the forked child immediately before exec. The
    // closure performs only direct process-attribute/resource-limit syscalls and
    // creates no Rust synchronization or background state.
    unsafe {
        command.pre_exec(move || {
            set_limit(libc::RLIMIT_AS, limits.address_space_bytes)?;
            set_limit(libc::RLIMIT_CPU, limits.cpu_seconds)?;
            set_limit(libc::RLIMIT_FSIZE, limits.file_size_bytes)?;
            set_limit(libc::RLIMIT_NOFILE, limits.open_files)?;
            set_limit(libc::RLIMIT_NPROC, limits.process_count)?;
            set_limit(libc::RLIMIT_CORE, 0)?;
            // SAFETY: prctl is called with the documented PR_SET_NO_NEW_PRIVS
            // operation and integer arguments only.
            if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: setsid takes no pointer arguments and affects only the
            // forked child session/process-group state.
            if unsafe { libc::setsid() } == -1 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: umask takes a plain mode value and affects only the child.
            unsafe { libc::umask(0o077) };
            Ok(())
        });
    }
}

#[cfg(target_os = "linux")]
fn set_limit(resource: libc::__rlimit_resource_t, value: u64) -> io::Result<()> {
    let value: libc::rlim_t = value
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "rlimit overflow"))?;
    let limit = libc::rlimit {
        rlim_cur: value,
        rlim_max: value,
    };
    // SAFETY: `limit` is initialized and `resource` is a libc RLIMIT constant.
    if unsafe { libc::setrlimit(resource, &limit) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{EngineeringDomain, SolverKind};

    fn request() -> SimulationRequest {
        SimulationRequest::new(
            "worker-protocol",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "worker protocol test",
        )
        .with_parameter("x", 1.25, "1", "test")
    }

    #[test]
    fn request_frame_round_trips_exact_package_bytes() {
        let manifest = br#"{"runtime":"wasm"}"#;
        let component = b"component-bytes";
        let mut bytes = Vec::new();
        write_request_frame(
            &mut bytes,
            manifest,
            component,
            &request(),
            WorkerFrameLimits::default(),
        )
        .unwrap();
        let decoded = read_request_frame(bytes.as_slice(), WorkerFrameLimits::default()).unwrap();
        assert_eq!(decoded.manifest_bytes, manifest);
        assert_eq!(decoded.component_bytes, component);
        assert_eq!(decoded.request, request());
    }

    #[test]
    fn oversized_component_fails_before_allocation_on_decode() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(REQUEST_MAGIC);
        bytes.extend_from_slice(&FRAME_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(
            &(WorkerFrameLimits::default().max_component_bytes + 1).to_le_bytes(),
        );
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            read_request_frame(bytes.as_slice(), WorkerFrameLimits::default()),
            Err(WorkerProtocolError::FrameTooLarge {
                field: "component",
                ..
            })
        ));
    }

    #[test]
    fn invalid_supervisor_limits_fail_closed() {
        assert!(matches!(
            SupervisorLimits {
                wall_time_ms: 0,
                ..SupervisorLimits::default()
            }
            .validate(),
            Err(SupervisorError::InvalidLimits)
        ));
    }
}
