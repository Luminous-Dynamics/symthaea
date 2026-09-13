// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Technical simulation-worker protocol over an exact sealed ELF whose process
//! is born in cgroup v2 and receives the reviewed legacy-style pre-exec rlimit
//! envelope before sealed `execveat`.
//!
//! This layer remains technical-only. It does not mint admission, routing,
//! worker qualification, deployment authority, engineering evidence or release
//! authority. The child must return `SimulationEvidence::default()`.

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
use symthaea_sim_clone_cgroup_exec_rlimit::{
    RlimitExecveatCgroupError, SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1,
    SealedExecveatCgroupRlimitEvidence, spawn_sealed_image_in_cgroup_with_rlimits,
};
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use symthaea_sim_worker::{
    SUPERVISOR_PROFILE_V1, SupervisorError, SupervisorLimits, WorkerFailure,
    WorkerFrameLimits, WorkerProtocolError, WorkerResponse, WorkerSuccess, WORKER_PROTOCOL_V1,
    read_response_frame, write_request_frame,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::SealedWorkerImage;
use thiserror::Error;

pub const FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1: &str =
    "symthaea.simulation.first-instruction-worker.rlimit-v1";
const EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.first-instruction-worker.rlimit-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionWorkerRlimitEvidence {
    pub profile: String,
    pub launch: SealedExecveatCgroupRlimitEvidence,
    pub supervisor_profile: String,
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
    pub parent_wall_time_ms: u64,
    pub parent_max_stdout_bytes: u64,
    pub parent_max_stderr_bytes: u64,
    pub worker_born_in_target_cgroup: bool,
    pub worker_exec_in_target_cgroup: bool,
    pub worker_runtime_allocations_begin_after_exec_in_target_cgroup: bool,
    pub inherited_parent_memory_recharged: bool,
    pub legacy_preexec_rlimits_applied: bool,
    pub parent_wall_time_enforced: bool,
    pub parent_output_limits_enforced: bool,
    pub evidence_sha256: String,
}

impl FirstInstructionWorkerRlimitEvidence {
    pub fn verify(&self) -> Result<(), FirstInstructionWorkerRlimitError> {
        if self.profile != FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1
            || self.supervisor_profile != SUPERVISOR_PROFILE_V1
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
            || !self.legacy_preexec_rlimits_applied
            || !self.parent_wall_time_enforced
            || !self.parent_output_limits_enforced
        {
            return Err(FirstInstructionWorkerRlimitError::InvalidEvidence);
        }
        self.launch.verify()?;
        if self.launch.profile != SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1
            || self.launch.supervisor_profile != SUPERVISOR_PROFILE_V1
            || !self.launch.legacy_preexec_rlimits_applied
            || self.launch.parent_wall_time_enforced_by_this_launcher
            || self.launch.parent_output_limits_enforced_by_this_launcher
            || self.launch.limits.expected_parent_wall_time_ms != self.parent_wall_time_ms
            || self.launch.limits.expected_parent_max_stdout_bytes != self.parent_max_stdout_bytes
            || self.launch.limits.expected_parent_max_stderr_bytes != self.parent_max_stderr_bytes
        {
            return Err(FirstInstructionWorkerRlimitError::InvalidEvidence);
        }
        SupervisorLimits {
            address_space_bytes: self.launch.limits.address_space_bytes,
            cpu_seconds: self.launch.limits.cpu_seconds,
            file_size_bytes: self.launch.limits.file_size_bytes,
            open_files: self.launch.limits.open_files,
            process_count: self.launch.limits.process_count,
            wall_time_ms: self.parent_wall_time_ms,
            max_stdout_bytes: self.parent_max_stdout_bytes,
            max_stderr_bytes: self.parent_max_stderr_bytes,
        }
        .validate()?;
        for digest in [
            &self.manifest_sha256,
            &self.component_sha256,
            &self.request_sha256,
            &self.output_sha256,
        ] {
            parse_hex_32(digest)?;
        }
        if self.evidence_sha256 != hex_digest(evidence_sha256_v1(self)?) {
            return Err(FirstInstructionWorkerRlimitError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FirstInstructionWorkerRlimitInvocation {
    success: WorkerSuccess,
    evidence: FirstInstructionWorkerRlimitEvidence,
}

impl FirstInstructionWorkerRlimitInvocation {
    pub fn result(&self) -> &SimulationResult {
        &self.success.result
    }

    pub fn worker(&self) -> &WorkerSuccess {
        &self.success
    }

    pub fn evidence(&self) -> &FirstInstructionWorkerRlimitEvidence {
        &self.evidence
    }

    pub fn into_result(self) -> SimulationResult {
        self.success.result
    }
}

#[derive(Debug)]
enum IoEvent {
    Writer(io::Result<()>),
    Stdout(Result<Vec<u8>, FirstInstructionWorkerRlimitError>),
    Stderr(Result<Vec<u8>, FirstInstructionWorkerRlimitError>),
}

#[allow(clippy::too_many_arguments)]
pub fn execute_first_instruction_worker_with_rlimits(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
    frame_limits: WorkerFrameLimits,
    limits: SupervisorLimits,
) -> Result<FirstInstructionWorkerRlimitInvocation, FirstInstructionWorkerRlimitError> {
    let limits = limits.validate()?;
    validate_frame_limits(frame_limits)?;
    request
        .validate()
        .map_err(|error| FirstInstructionWorkerRlimitError::InvalidRequest(error.to_string()))?;

    let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
        .map_err(FirstInstructionWorkerRlimitError::ManifestJson)?;
    manifest
        .validate()
        .map_err(|problems| FirstInstructionWorkerRlimitError::ManifestInvalid(format!("{problems:?}")))?;
    if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
        return Err(FirstInstructionWorkerRlimitError::UnexpectedManifest);
    }

    let expected_manifest: [u8; 32] = Sha256::digest(manifest_bytes).into();
    let expected_component: [u8; 32] = Sha256::digest(component_bytes).into();
    let expected_request = canonical_request_sha256_v1(request)
        .map_err(|error| FirstInstructionWorkerRlimitError::Canonical(error.to_string()))?;

    let mut request_frame = Vec::new();
    write_request_frame(
        &mut request_frame,
        manifest_bytes,
        component_bytes,
        request,
        frame_limits,
    )?;

    let mut child = spawn_sealed_image_in_cgroup_with_rlimits(image, cgroup, limits)?;
    let launch = child.evidence().clone();
    launch.verify()?;

    let mut stdin = child
        .stdin_mut()
        .ok_or(FirstInstructionWorkerRlimitError::MissingPipe("stdin"))?
        .try_clone()
        .map_err(|source| FirstInstructionWorkerRlimitError::PipeClone {
            stream: "stdin",
            source,
        })?;
    let stdout = child
        .stdout_mut()
        .try_clone()
        .map_err(|source| FirstInstructionWorkerRlimitError::PipeClone {
            stream: "stdout",
            source,
        })?;
    let stderr = child
        .stderr_mut()
        .try_clone()
        .map_err(|source| FirstInstructionWorkerRlimitError::PipeClone {
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
            return Err(FirstInstructionWorkerRlimitError::WallTimeExceeded(
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
                return Err(FirstInstructionWorkerRlimitError::WallTimeExceeded(
                    limits.wall_time_ms,
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = child.terminate_and_wait();
                join_threads(writer, stdout_reader, stderr_reader);
                return Err(FirstInstructionWorkerRlimitError::IoChannelClosed);
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
        return Err(FirstInstructionWorkerRlimitError::LaunchEvidenceChanged);
    }
    if !status.success() {
        return Err(FirstInstructionWorkerRlimitError::WorkerExited {
            code: status.code(),
            stderr: bounded_utf8(&stderr_bytes),
        });
    }
    writer_result.map_err(FirstInstructionWorkerRlimitError::InputWrite)?;

    let mut cursor = Cursor::new(stdout_bytes.as_slice());
    let response = read_response_frame(&mut cursor, frame_limits)?;
    if cursor.position() != u64::try_from(stdout_bytes.len()).unwrap_or(u64::MAX) {
        return Err(FirstInstructionWorkerRlimitError::TrailingResponseBytes);
    }
    let success = match response {
        WorkerResponse::Ok(success) => success,
        WorkerResponse::Err(failure) => {
            return Err(FirstInstructionWorkerRlimitError::WorkerFailure(failure));
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

    let mut evidence = FirstInstructionWorkerRlimitEvidence {
        profile: FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1.into(),
        launch,
        supervisor_profile: SUPERVISOR_PROFILE_V1.into(),
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
        parent_wall_time_ms: limits.wall_time_ms,
        parent_max_stdout_bytes: limits.max_stdout_bytes,
        parent_max_stderr_bytes: limits.max_stderr_bytes,
        worker_born_in_target_cgroup: true,
        worker_exec_in_target_cgroup: true,
        worker_runtime_allocations_begin_after_exec_in_target_cgroup: true,
        inherited_parent_memory_recharged: false,
        legacy_preexec_rlimits_applied: true,
        parent_wall_time_enforced: true,
        parent_output_limits_enforced: true,
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
    evidence.verify()?;

    Ok(FirstInstructionWorkerRlimitInvocation { success, evidence })
}

fn verify_success(
    success: &WorkerSuccess,
    expected_manifest: [u8; 32],
    expected_component: [u8; 32],
    expected_request: [u8; 32],
    request: &SimulationRequest,
    manifest: &ExtensionManifest,
) -> Result<[u8; 32], FirstInstructionWorkerRlimitError> {
    if success.profile != WORKER_PROTOCOL_V1 {
        return Err(FirstInstructionWorkerRlimitError::ProfileMismatch("worker protocol"));
    }
    require_digest("manifest", &success.manifest_sha256, expected_manifest)?;
    require_digest("component", &success.component_sha256, expected_component)?;
    require_digest("request", &success.request_sha256, expected_request)?;
    if success.result.request_id != request.id {
        return Err(FirstInstructionWorkerRlimitError::CommitmentMismatch("request id"));
    }
    if success.result.evidence != SimulationEvidence::default() {
        return Err(FirstInstructionWorkerRlimitError::WorkerMintedEvidence);
    }
    let output = canonical_output_sha256_v1(&success.result)
        .map_err(|error| FirstInstructionWorkerRlimitError::Canonical(error.to_string()))?;
    require_digest("output", &success.output_sha256, output)?;
    if success.extension_id != manifest.id.as_str() {
        return Err(FirstInstructionWorkerRlimitError::CommitmentMismatch("extension id"));
    }
    if success.extension_version != manifest.version {
        return Err(FirstInstructionWorkerRlimitError::CommitmentMismatch("extension version"));
    }
    if success.control_wasm_profile != CONTROL_WASM_PROFILE_V1 {
        return Err(FirstInstructionWorkerRlimitError::ProfileMismatch("control runtime"));
    }
    if success.simulation_wasm_profile != SIMULATION_WASM_PROFILE_V1 {
        return Err(FirstInstructionWorkerRlimitError::ProfileMismatch("simulation runtime"));
    }
    if success.wit_version != SIMULATION_WIT_V1 {
        return Err(FirstInstructionWorkerRlimitError::ProfileMismatch("wit"));
    }
    if success.adapter_version != SIMULATION_ADAPTER_V1 {
        return Err(FirstInstructionWorkerRlimitError::ProfileMismatch("adapter"));
    }
    Ok(output)
}

fn validate_frame_limits(limits: WorkerFrameLimits) -> Result<(), FirstInstructionWorkerRlimitError> {
    if limits.max_manifest_bytes == 0
        || limits.max_component_bytes == 0
        || limits.max_request_json_bytes == 0
        || limits.max_response_json_bytes == 0
    {
        return Err(FirstInstructionWorkerRlimitError::InvalidFrameLimits);
    }
    Ok(())
}

fn read_limited(
    mut reader: impl Read,
    limit: u64,
    stream: &'static str,
) -> Result<Vec<u8>, FirstInstructionWorkerRlimitError> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| FirstInstructionWorkerRlimitError::OutputRead { stream, source })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(FirstInstructionWorkerRlimitError::OutputTooLarge { stream, limit });
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
) -> Result<(), FirstInstructionWorkerRlimitError> {
    if actual != hex_digest(expected) {
        return Err(FirstInstructionWorkerRlimitError::CommitmentMismatch(field));
    }
    Ok(())
}

fn evidence_sha256_v1(
    evidence: &FirstInstructionWorkerRlimitEvidence,
) -> Result<[u8; 32], FirstInstructionWorkerRlimitError> {
    evidence.launch.verify()?;
    let launch = parse_hex_32(&evidence.launch.evidence_sha256)?;
    let manifest = parse_hex_32(&evidence.manifest_sha256)?;
    let component = parse_hex_32(&evidence.component_sha256)?;
    let request = parse_hex_32(&evidence.request_sha256)?;
    let output = parse_hex_32(&evidence.output_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    hasher.update(launch);
    put_string(&mut hasher, &evidence.supervisor_profile)?;
    put_string(&mut hasher, &evidence.worker_protocol)?;
    put_string(&mut hasher, &evidence.expected_process_containment_profile)?;
    put_string(&mut hasher, &evidence.expected_filesystem_containment_profile)?;
    hasher.update([u8::from(evidence.containment_profiles_established_by_this_layer)]);
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
    hasher.update(evidence.parent_wall_time_ms.to_le_bytes());
    hasher.update(evidence.parent_max_stdout_bytes.to_le_bytes());
    hasher.update(evidence.parent_max_stderr_bytes.to_le_bytes());
    for value in [
        evidence.worker_born_in_target_cgroup,
        evidence.worker_exec_in_target_cgroup,
        evidence.worker_runtime_allocations_begin_after_exec_in_target_cgroup,
        evidence.inherited_parent_memory_recharged,
        evidence.legacy_preexec_rlimits_applied,
        evidence.parent_wall_time_enforced,
        evidence.parent_output_limits_enforced,
    ] {
        hasher.update([u8::from(value)]);
    }
    Ok(hasher.finalize().into())
}

fn put_string(hasher: &mut Sha256, value: &str) -> Result<(), FirstInstructionWorkerRlimitError> {
    let len = u64::try_from(value.len()).map_err(|_| FirstInstructionWorkerRlimitError::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], FirstInstructionWorkerRlimitError> {
    if value.len() != 64 {
        return Err(FirstInstructionWorkerRlimitError::InvalidDigest);
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, FirstInstructionWorkerRlimitError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(FirstInstructionWorkerRlimitError::InvalidDigest),
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
pub enum FirstInstructionWorkerRlimitError {
    #[error("invalid parent supervisor limits")]
    Supervisor(#[from] SupervisorError),
    #[error("invalid worker frame limits")]
    InvalidFrameLimits,
    #[error("invalid simulation request: {0}")]
    InvalidRequest(String),
    #[error("manifest JSON is invalid: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("manifest failed structural validation: {0}")]
    ManifestInvalid(String),
    #[error("manifest must describe one Wasm simulation extension")]
    UnexpectedManifest,
    #[error(transparent)]
    WorkerProtocol(#[from] WorkerProtocolError),
    #[error(transparent)]
    Launcher(#[from] RlimitExecveatCgroupError),
    #[error("missing worker {0} pipe")]
    MissingPipe(&'static str),
    #[error("failed to clone worker {stream} pipe: {source}")]
    PipeClone {
        stream: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("worker exceeded wall-clock limit of {0} ms")]
    WallTimeExceeded(u64),
    #[error("worker I/O event channel closed unexpectedly")]
    IoChannelClosed,
    #[error("failed writing worker input: {0}")]
    InputWrite(#[source] io::Error),
    #[error("failed reading worker {stream}: {source}")]
    OutputRead {
        stream: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("worker {stream} exceeded configured limit {limit}")]
    OutputTooLarge { stream: &'static str, limit: u64 },
    #[error("worker exited unsuccessfully: code={code:?}, stderr={stderr}")]
    WorkerExited { code: Option<i32>, stderr: String },
    #[error("worker launch evidence changed between spawn and reap")]
    LaunchEvidenceChanged,
    #[error("worker response contained trailing bytes")]
    TrailingResponseBytes,
    #[error("worker returned typed failure: {0:?}")]
    WorkerFailure(WorkerFailure),
    #[error("worker response profile mismatch: {0}")]
    ProfileMismatch(&'static str),
    #[error("worker response commitment mismatch: {0}")]
    CommitmentMismatch(&'static str),
    #[error("worker attempted to mint simulation evidence")]
    WorkerMintedEvidence,
    #[error("canonical simulation digest failed: {0}")]
    Canonical(String),
    #[error("serialized stronger worker evidence is structurally invalid")]
    InvalidEvidence,
    #[error("serialized stronger worker evidence digest does not match")]
    EvidenceDigestMismatch,
    #[error("invalid canonical lowercase SHA-256 digest")]
    InvalidDigest,
    #[error("string length overflow while hashing evidence")]
    LengthOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_is_frozen() {
        assert_eq!(
            FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1,
            "symthaea.simulation.first-instruction-worker.rlimit-v1"
        );
    }

    #[test]
    fn zero_frame_limit_fails_closed() {
        let limits = WorkerFrameLimits {
            max_manifest_bytes: 0,
            ..WorkerFrameLimits::default()
        };
        assert!(matches!(
            validate_frame_limits(limits),
            Err(FirstInstructionWorkerRlimitError::InvalidFrameLimits)
        ));
    }
}
