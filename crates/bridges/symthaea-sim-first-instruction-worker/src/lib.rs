// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Technical simulation-worker protocol over an exact sealed ELF whose process
//! is created directly in a cgroup-v2 leaf before `execveat`.
//!
//! This remains deliberately below admission/routing/release authority. The
//! child returns a technical `SimulationResult` whose `SimulationEvidence` must
//! remain empty. The parent independently recomputes all package/request/output
//! commitments and binds the exact first-instruction launch evidence.
//!
//! The intended production subject is the filesystem-contained worker. Profile
//! labels recorded here are expectations only: this crate does not upgrade an
//! arbitrary protocol-speaking ELF into a seccomp/Landlock-qualified worker.
//! That claim remains owned by live worker qualification.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Cursor, Read, Write};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_extension_host::CONTROL_WASM_PROFILE_V1;
use symthaea_extension_simulation_host::{
    SIMULATION_ADAPTER_V1, SIMULATION_WASM_PROFILE_V1, SIMULATION_WIT_V1,
};
use symthaea_sim_bridge::{SimulationEvidence, SimulationRequest, SimulationResult};
use symthaea_sim_clone_cgroup_exec::{
    spawn_sealed_image_in_cgroup, SealedExecveatCgroupError, SealedExecveatCgroupEvidence,
    SEALED_EXECVEAT_CGROUP_PROFILE_V1,
};
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use symthaea_sim_worker::{
    read_response_frame, write_request_frame, WorkerFrameLimits, WorkerProtocolError, WorkerResponse,
    WorkerSuccess, WORKER_PROTOCOL_V1,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::SealedWorkerImage;
use thiserror::Error;

pub const FIRST_INSTRUCTION_WORKER_PROFILE_V1: &str =
    "symthaea.simulation.first-instruction-worker.v1";
const EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.simulation.first-instruction-worker.v1\0";
const MAX_WALL_TIME_MS: u64 = 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionWorkerLimits {
    pub wall_time_ms: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
}

impl Default for FirstInstructionWorkerLimits {
    fn default() -> Self {
        Self {
            wall_time_ms: 10_000,
            max_stdout_bytes: 16 * 1024 * 1024,
            max_stderr_bytes: 256 * 1024,
        }
    }
}

impl FirstInstructionWorkerLimits {
    pub fn validate(self) -> Result<Self, FirstInstructionWorkerError> {
        if self.wall_time_ms == 0
            || self.wall_time_ms > MAX_WALL_TIME_MS
            || self.max_stdout_bytes == 0
            || self.max_stderr_bytes == 0
        {
            return Err(FirstInstructionWorkerError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionWorkerEvidence {
    pub profile: String,
    pub launch: SealedExecveatCgroupEvidence,
    pub worker_protocol: String,
    pub expected_process_containment_profile: String,
    pub expected_filesystem_containment_profile: String,
    pub containment_profiles_established_by_this_layer: bool,
    pub extension_id: String,
    pub extension_version: String,
    pub manifest_sha256: String,
    pub component_sha256: String,
    pub request_sha256: String,
    pub output_sha256: String,
    pub control_wasm_profile: String,
    pub simulation_wasm_profile: String,
    pub wit_version: String,
    pub adapter_version: String,
    pub wall_time_ms: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
    pub worker_born_in_target_cgroup: bool,
    pub worker_exec_in_target_cgroup: bool,
    pub worker_runtime_allocations_begin_after_exec_in_target_cgroup: bool,
    pub inherited_parent_memory_recharged: bool,
    pub legacy_preexec_rlimits_applied: bool,
    pub evidence_sha256: String,
}

impl FirstInstructionWorkerEvidence {
    pub fn verify(&self) -> Result<(), FirstInstructionWorkerError> {
        if self.profile != FIRST_INSTRUCTION_WORKER_PROFILE_V1
            || self.worker_protocol != WORKER_PROTOCOL_V1
            || self.expected_process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
            || self.expected_filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
            || self.containment_profiles_established_by_this_layer
            || self.control_wasm_profile != CONTROL_WASM_PROFILE_V1
            || self.simulation_wasm_profile != SIMULATION_WASM_PROFILE_V1
            || self.wit_version != SIMULATION_WIT_V1
            || self.adapter_version != SIMULATION_ADAPTER_V1
            || !self.worker_born_in_target_cgroup
            || !self.worker_exec_in_target_cgroup
            || !self.worker_runtime_allocations_begin_after_exec_in_target_cgroup
            || self.inherited_parent_memory_recharged
            || self.legacy_preexec_rlimits_applied
        {
            return Err(FirstInstructionWorkerError::InvalidEvidence);
        }
        FirstInstructionWorkerLimits {
            wall_time_ms: self.wall_time_ms,
            max_stdout_bytes: self.max_stdout_bytes,
            max_stderr_bytes: self.max_stderr_bytes,
        }
        .validate()?;
        self.launch.verify()?;
        if self.launch.profile != SEALED_EXECVEAT_CGROUP_PROFILE_V1 {
            return Err(FirstInstructionWorkerError::InvalidEvidence);
        }
        for digest in [
            &self.manifest_sha256,
            &self.component_sha256,
            &self.request_sha256,
            &self.output_sha256,
        ] {
            parse_hex_32(digest)?;
        }
        let expected = hex_digest(evidence_sha256_v1(self)?);
        if self.evidence_sha256 != expected {
            return Err(FirstInstructionWorkerError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FirstInstructionWorkerInvocation {
    success: WorkerSuccess,
    evidence: FirstInstructionWorkerEvidence,
}

impl FirstInstructionWorkerInvocation {
    pub fn result(&self) -> &SimulationResult {
        &self.success.result
    }

    pub fn worker(&self) -> &WorkerSuccess {
        &self.success
    }

    pub fn evidence(&self) -> &FirstInstructionWorkerEvidence {
        &self.evidence
    }

    pub fn into_result(self) -> SimulationResult {
        self.success.result
    }
}

#[derive(Debug)]
enum IoEvent {
    Writer(io::Result<()>),
    Stdout(Result<Vec<u8>, FirstInstructionWorkerError>),
    Stderr(Result<Vec<u8>, FirstInstructionWorkerError>),
}

#[allow(clippy::too_many_arguments)]
pub fn execute_first_instruction_worker(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
    frame_limits: WorkerFrameLimits,
    limits: FirstInstructionWorkerLimits,
) -> Result<FirstInstructionWorkerInvocation, FirstInstructionWorkerError> {
    let limits = limits.validate()?;
    validate_frame_limits(frame_limits)?;
    request
        .validate()
        .map_err(|error| FirstInstructionWorkerError::InvalidRequest(error.to_string()))?;

    let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
        .map_err(FirstInstructionWorkerError::ManifestJson)?;
    manifest
        .validate()
        .map_err(|problems| FirstInstructionWorkerError::ManifestInvalid(format!("{problems:?}")))?;
    if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
        return Err(FirstInstructionWorkerError::UnexpectedManifest);
    }

    let expected_manifest: [u8; 32] = Sha256::digest(manifest_bytes).into();
    let expected_component: [u8; 32] = Sha256::digest(component_bytes).into();
    let expected_request = canonical_request_sha256_v1(request)
        .map_err(|error| FirstInstructionWorkerError::Canonical(error.to_string()))?;

    let mut request_frame = Vec::new();
    write_request_frame(
        &mut request_frame,
        manifest_bytes,
        component_bytes,
        request,
        frame_limits,
    )?;

    let mut child = spawn_sealed_image_in_cgroup(image, cgroup)?;
    let launch = child.evidence().clone();
    launch.verify()?;

    let mut stdin = child
        .stdin_mut()
        .ok_or(FirstInstructionWorkerError::MissingPipe("stdin"))?
        .try_clone()
        .map_err(|source| FirstInstructionWorkerError::PipeClone {
            stream: "stdin",
            source,
        })?;
    let stdout = child
        .stdout_mut()
        .try_clone()
        .map_err(|source| FirstInstructionWorkerError::PipeClone {
            stream: "stdout",
            source,
        })?;
    let stderr = child
        .stderr_mut()
        .try_clone()
        .map_err(|source| FirstInstructionWorkerError::PipeClone {
            stream: "stderr",
            source,
        })?;
    child.close_stdin();

    let (tx, rx) = mpsc::channel();
    let writer_tx = tx.clone();
    let writer = thread::spawn(move || {
        let result = (|| {
            stdin.write_all(&request_frame)?;
            stdin.flush()?;
            Ok(())
        })();
        let _ = writer_tx.send(IoEvent::Writer(result));
    });

    let stdout_tx = tx.clone();
    let stdout_limit = limits.max_stdout_bytes;
    let stdout_reader = thread::spawn(move || {
        let result = read_limited(stdout, stdout_limit, "stdout");
        let _ = stdout_tx.send(IoEvent::Stdout(result));
    });

    let stderr_tx = tx.clone();
    let stderr_limit = limits.max_stderr_bytes;
    let stderr_reader = thread::spawn(move || {
        let result = read_limited(stderr, stderr_limit, "stderr");
        let _ = stderr_tx.send(IoEvent::Stderr(result));
    });
    drop(tx);

    let deadline = Instant::now() + Duration::from_millis(limits.wall_time_ms);
    let mut writer_result: Option<io::Result<()>> = None;
    let mut stdout_bytes: Option<Vec<u8>> = None;
    let mut stderr_bytes: Option<Vec<u8>> = None;

    while writer_result.is_none() || stdout_bytes.is_none() || stderr_bytes.is_none() {
        let now = Instant::now();
        if now >= deadline {
            let _ = child.terminate_and_wait();
            join_threads(writer, stdout_reader, stderr_reader);
            return Err(FirstInstructionWorkerError::WallTimeExceeded(
                limits.wall_time_ms,
            ));
        }
        let remaining = deadline.saturating_duration_since(now);
        match rx.recv_timeout(remaining) {
            Ok(IoEvent::Writer(result)) => writer_result = Some(result),
            Ok(IoEvent::Stdout(Ok(bytes))) => stdout_bytes = Some(bytes),
            Ok(IoEvent::Stderr(Ok(bytes))) => stderr_bytes = Some(bytes),
            Ok(IoEvent::Stdout(Err(error))) | Ok(IoEvent::Stderr(Err(error))) => {
                let _ = child.terminate_and_wait();
                join_threads(writer, stdout_reader, stderr_reader);
                return Err(error);
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.terminate_and_wait();
                join_threads(writer, stdout_reader, stderr_reader);
                return Err(FirstInstructionWorkerError::WallTimeExceeded(
                    limits.wall_time_ms,
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = child.terminate_and_wait();
                join_threads(writer, stdout_reader, stderr_reader);
                return Err(FirstInstructionWorkerError::IoChannelClosed);
            }
        }
    }

    let writer_result = writer_result.expect("loop requires writer result");
    let stdout_bytes = stdout_bytes.expect("loop requires stdout");
    let stderr_bytes = stderr_bytes.expect("loop requires stderr");
    let (status, released_launch) = child.wait()?;
    join_threads(writer, stdout_reader, stderr_reader);
    released_launch.verify()?;
    if released_launch != launch {
        return Err(FirstInstructionWorkerError::LaunchEvidenceChanged);
    }
    if !status.success() {
        return Err(FirstInstructionWorkerError::WorkerExited {
            code: status.code(),
            stderr: bounded_utf8(&stderr_bytes),
        });
    }
    writer_result.map_err(FirstInstructionWorkerError::InputWrite)?;

    let mut cursor = Cursor::new(stdout_bytes.as_slice());
    let response = read_response_frame(&mut cursor, frame_limits)?;
    if cursor.position() != u64::try_from(stdout_bytes.len()).unwrap_or(u64::MAX) {
        return Err(FirstInstructionWorkerError::TrailingResponseBytes);
    }
    let success = match response {
        WorkerResponse::Ok(success) => success,
        WorkerResponse::Err(failure) => {
            return Err(FirstInstructionWorkerError::WorkerFailure(failure.message));
        }
    };

    let output_sha = verify_success(
        &success,
        expected_manifest,
        expected_component,
        expected_request,
        request,
        &manifest,
    )?;

    let mut evidence = FirstInstructionWorkerEvidence {
        profile: FIRST_INSTRUCTION_WORKER_PROFILE_V1.into(),
        launch,
        worker_protocol: WORKER_PROTOCOL_V1.into(),
        expected_process_containment_profile: WORKER_CONTAINMENT_PROFILE_V1.into(),
        expected_filesystem_containment_profile: WORKER_FILESYSTEM_PROFILE_V1.into(),
        containment_profiles_established_by_this_layer: false,
        extension_id: manifest.id.as_str().to_owned(),
        extension_version: manifest.version.clone(),
        manifest_sha256: hex_digest(expected_manifest),
        component_sha256: hex_digest(expected_component),
        request_sha256: hex_digest(expected_request),
        output_sha256: hex_digest(output_sha),
        control_wasm_profile: CONTROL_WASM_PROFILE_V1.into(),
        simulation_wasm_profile: SIMULATION_WASM_PROFILE_V1.into(),
        wit_version: SIMULATION_WIT_V1.into(),
        adapter_version: SIMULATION_ADAPTER_V1.into(),
        wall_time_ms: limits.wall_time_ms,
        max_stdout_bytes: limits.max_stdout_bytes,
        max_stderr_bytes: limits.max_stderr_bytes,
        worker_born_in_target_cgroup: true,
        worker_exec_in_target_cgroup: true,
        worker_runtime_allocations_begin_after_exec_in_target_cgroup: true,
        inherited_parent_memory_recharged: false,
        legacy_preexec_rlimits_applied: false,
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
    evidence.verify()?;

    Ok(FirstInstructionWorkerInvocation { success, evidence })
}

fn verify_success(
    success: &WorkerSuccess,
    expected_manifest: [u8; 32],
    expected_component: [u8; 32],
    expected_request: [u8; 32],
    request: &SimulationRequest,
    manifest: &ExtensionManifest,
) -> Result<[u8; 32], FirstInstructionWorkerError> {
    if success.profile != WORKER_PROTOCOL_V1 {
        return Err(FirstInstructionWorkerError::ProfileMismatch(
            "worker protocol",
        ));
    }
    require_digest("manifest", &success.manifest_sha256, expected_manifest)?;
    require_digest("component", &success.component_sha256, expected_component)?;
    require_digest("request", &success.request_sha256, expected_request)?;
    if success.result.request_id != request.id {
        return Err(FirstInstructionWorkerError::CommitmentMismatch(
            "request id",
        ));
    }
    if success.result.evidence != SimulationEvidence::default() {
        return Err(FirstInstructionWorkerError::WorkerMintedEvidence);
    }
    let output = canonical_output_sha256_v1(&success.result)
        .map_err(|error| FirstInstructionWorkerError::Canonical(error.to_string()))?;
    require_digest("output", &success.output_sha256, output)?;
    if success.extension_id != manifest.id.as_str() {
        return Err(FirstInstructionWorkerError::CommitmentMismatch(
            "extension id",
        ));
    }
    if success.extension_version != manifest.version {
        return Err(FirstInstructionWorkerError::CommitmentMismatch(
            "extension version",
        ));
    }
    if success.control_wasm_profile != CONTROL_WASM_PROFILE_V1 {
        return Err(FirstInstructionWorkerError::ProfileMismatch(
            "control runtime",
        ));
    }
    if success.simulation_wasm_profile != SIMULATION_WASM_PROFILE_V1 {
        return Err(FirstInstructionWorkerError::ProfileMismatch(
            "simulation runtime",
        ));
    }
    if success.wit_version != SIMULATION_WIT_V1 {
        return Err(FirstInstructionWorkerError::ProfileMismatch("wit"));
    }
    if success.adapter_version != SIMULATION_ADAPTER_V1 {
        return Err(FirstInstructionWorkerError::ProfileMismatch("adapter"));
    }
    Ok(output)
}

fn validate_frame_limits(limits: WorkerFrameLimits) -> Result<(), FirstInstructionWorkerError> {
    if limits.max_manifest_bytes == 0
        || limits.max_component_bytes == 0
        || limits.max_request_json_bytes == 0
        || limits.max_response_json_bytes == 0
    {
        return Err(FirstInstructionWorkerError::InvalidFrameLimits);
    }
    Ok(())
}

fn read_limited(
    mut reader: impl Read,
    limit: u64,
    stream: &'static str,
) -> Result<Vec<u8>, FirstInstructionWorkerError> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| FirstInstructionWorkerError::OutputRead { stream, source })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(FirstInstructionWorkerError::OutputTooLarge { stream, limit });
    }
    Ok(bytes)
}

fn join_threads(
    writer: thread::JoinHandle<()>,
    stdout: thread::JoinHandle<()>,
    stderr: thread::JoinHandle<()>,
) {
    let _ = writer.join();
    let _ = stdout.join();
    let _ = stderr.join();
}

fn require_digest(
    field: &'static str,
    actual: &str,
    expected: [u8; 32],
) -> Result<(), FirstInstructionWorkerError> {
    if actual != hex_digest(expected) {
        return Err(FirstInstructionWorkerError::CommitmentMismatch(field));
    }
    Ok(())
}

fn evidence_sha256_v1(
    evidence: &FirstInstructionWorkerEvidence,
) -> Result<[u8; 32], FirstInstructionWorkerError> {
    evidence.launch.verify()?;
    let launch = parse_hex_32(&evidence.launch.evidence_sha256)?;
    let manifest = parse_hex_32(&evidence.manifest_sha256)?;
    let component = parse_hex_32(&evidence.component_sha256)?;
    let request = parse_hex_32(&evidence.request_sha256)?;
    let output = parse_hex_32(&evidence.output_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    hasher.update(launch);
    put_string(&mut hasher, &evidence.worker_protocol)?;
    put_string(
        &mut hasher,
        &evidence.expected_process_containment_profile,
    )?;
    put_string(
        &mut hasher,
        &evidence.expected_filesystem_containment_profile,
    )?;
    hasher.update([u8::from(
        evidence.containment_profiles_established_by_this_layer,
    )]);
    put_string(&mut hasher, &evidence.extension_id)?;
    put_string(&mut hasher, &evidence.extension_version)?;
    hasher.update(manifest);
    hasher.update(component);
    hasher.update(request);
    hasher.update(output);
    put_string(&mut hasher, &evidence.control_wasm_profile)?;
    put_string(&mut hasher, &evidence.simulation_wasm_profile)?;
    put_string(&mut hasher, &evidence.wit_version)?;
    put_string(&mut hasher, &evidence.adapter_version)?;
    hasher.update(evidence.wall_time_ms.to_le_bytes());
    hasher.update(evidence.max_stdout_bytes.to_le_bytes());
    hasher.update(evidence.max_stderr_bytes.to_le_bytes());
    hasher.update([u8::from(evidence.worker_born_in_target_cgroup)]);
    hasher.update([u8::from(evidence.worker_exec_in_target_cgroup)]);
    hasher.update([u8::from(
        evidence.worker_runtime_allocations_begin_after_exec_in_target_cgroup,
    )]);
    hasher.update([u8::from(evidence.inherited_parent_memory_recharged)]);
    hasher.update([u8::from(evidence.legacy_preexec_rlimits_applied)]);
    Ok(hasher.finalize().into())
}

fn put_string(
    hasher: &mut Sha256,
    value: &str,
) -> Result<(), FirstInstructionWorkerError> {
    let len = u64::try_from(value.len()).map_err(|_| FirstInstructionWorkerError::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], FirstInstructionWorkerError> {
    if value.len() != 64 {
        return Err(FirstInstructionWorkerError::InvalidDigest);
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, FirstInstructionWorkerError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(FirstInstructionWorkerError::InvalidDigest),
    }
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

fn bounded_utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[derive(Debug, Error)]
pub enum FirstInstructionWorkerError {
    #[error("first-instruction worker limits are invalid")]
    InvalidLimits,
    #[error("worker frame limits are invalid")]
    InvalidFrameLimits,
    #[error("simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error("manifest json is invalid: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("first-instruction worker requires a wasm simulation manifest")]
    UnexpectedManifest,
    #[error(transparent)]
    Protocol(#[from] WorkerProtocolError),
    #[error(transparent)]
    Launch(#[from] SealedExecveatCgroupError),
    #[error("worker pipe {0} is unavailable")]
    MissingPipe(&'static str),
    #[error("failed to clone worker {stream} pipe: {source}")]
    PipeClone {
        stream: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("failed writing worker input: {0}")]
    InputWrite(#[source] io::Error),
    #[error("worker {stream} read failed: {source}")]
    OutputRead {
        stream: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("worker {stream} exceeded configured limit {limit}")]
    OutputTooLarge { stream: &'static str, limit: u64 },
    #[error("worker exceeded wall-clock limit of {0} ms")]
    WallTimeExceeded(u64),
    #[error("worker i/o event channel closed unexpectedly")]
    IoChannelClosed,
    #[error("worker process exited unsuccessfully: code={code:?}, stderr={stderr}")]
    WorkerExited { code: Option<i32>, stderr: String },
    #[error("worker returned typed failure: {0}")]
    WorkerFailure(String),
    #[error("worker response contained trailing bytes")]
    TrailingResponseBytes,
    #[error("worker launch evidence changed between spawn and reap")]
    LaunchEvidenceChanged,
    #[error("canonicalization failed: {0}")]
    Canonical(String),
    #[error("worker response profile mismatch: {0}")]
    ProfileMismatch(&'static str),
    #[error("worker response commitment mismatch: {0}")]
    CommitmentMismatch(&'static str),
    #[error("worker attempted to mint simulation evidence")]
    WorkerMintedEvidence,
    #[error("serialized first-instruction worker evidence is structurally invalid")]
    InvalidEvidence,
    #[error("serialized first-instruction worker evidence digest does not match")]
    EvidenceDigestMismatch,
    #[error("invalid canonical lowercase SHA-256 digest")]
    InvalidDigest,
    #[error("string length cannot be represented by evidence profile v1")]
    LengthOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_fail_closed() {
        assert!(matches!(
            FirstInstructionWorkerLimits {
                wall_time_ms: 0,
                ..FirstInstructionWorkerLimits::default()
            }
            .validate(),
            Err(FirstInstructionWorkerError::InvalidLimits)
        ));
    }

    #[test]
    fn profile_is_frozen() {
        assert_eq!(
            FIRST_INSTRUCTION_WORKER_PROFILE_V1,
            "symthaea.simulation.first-instruction-worker.v1"
        );
    }
}
