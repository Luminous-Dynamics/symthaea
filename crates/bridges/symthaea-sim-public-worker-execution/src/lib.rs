// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen technical resource profile for public/community simulation workers.
//!
//! This layer composes the bounded pre-spawn public input theorem with one exact
//! supervisor/cgroup resource envelope. Callers do not supply arbitrary resource
//! limits. The profile also binds the v1 worker response framing overhead to the
//! parent stdout ceiling so every protocol-valid response size fits the parent
//! reader by construction.
//!
//! This remains technical/audit state. It does not mint admission, routing,
//! worker qualification, deployment authority, engineering evidence or release
//! authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_sim_bridge::{SimulationRequest, SimulationResult};
use symthaea_sim_public_worker_input::{
    PUBLIC_MAX_RESPONSE_JSON_BYTES_V1, PublicWorkerInputError, PublicWorkerInputEvidence,
    PublicWorkerInvocation, execute_prepared_public_first_instruction_worker_with_rlimits,
    prepare_public_simulation_input,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerSuccess};
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_image::SealedWorkerImage;
use thiserror::Error;

pub const PUBLIC_WORKER_EXECUTION_PROFILE_V1: &str =
    "symthaea.simulation.public-worker-execution.v1";

pub const PUBLIC_ADDRESS_SPACE_BYTES_V1: u64 = 16 * 1024 * 1024 * 1024;
pub const PUBLIC_CPU_SECONDS_V1: u64 = 15;
pub const PUBLIC_FILE_SIZE_BYTES_V1: u64 = 1024 * 1024;
pub const PUBLIC_OPEN_FILES_V1: u64 = 32;
pub const PUBLIC_PROCESS_COUNT_V1: u64 = 64;
pub const PUBLIC_WALL_TIME_MS_V1: u64 = 20_000;
pub const PUBLIC_STDERR_BYTES_V1: u64 = 256 * 1024;

pub const PUBLIC_CGROUP_MEMORY_MAX_BYTES_V1: u64 = 2 * 1024 * 1024 * 1024;
pub const PUBLIC_CGROUP_PIDS_MAX_V1: u64 = 64;
pub const PUBLIC_CGROUP_CPU_QUOTA_US_V1: u64 = 100_000;
pub const PUBLIC_CGROUP_CPU_PERIOD_US_V1: u64 = 100_000;

/// v1 worker response framing is 8-byte magic + 4-byte version + 8-byte JSON length.
pub const PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1: u64 = 20;
pub const PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1: u64 =
    PUBLIC_MAX_RESPONSE_JSON_BYTES_V1 + PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1;

const EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.simulation.public-worker-execution.v1\0";

pub const fn public_supervisor_limits_v1() -> SupervisorLimits {
    SupervisorLimits {
        address_space_bytes: PUBLIC_ADDRESS_SPACE_BYTES_V1,
        cpu_seconds: PUBLIC_CPU_SECONDS_V1,
        file_size_bytes: PUBLIC_FILE_SIZE_BYTES_V1,
        open_files: PUBLIC_OPEN_FILES_V1,
        process_count: PUBLIC_PROCESS_COUNT_V1,
        wall_time_ms: PUBLIC_WALL_TIME_MS_V1,
        max_stdout_bytes: PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1,
        max_stderr_bytes: PUBLIC_STDERR_BYTES_V1,
    }
}

pub const fn public_cgroup_limits_v1() -> CgroupV2Limits {
    CgroupV2Limits {
        memory_max_bytes: PUBLIC_CGROUP_MEMORY_MAX_BYTES_V1,
        pids_max: PUBLIC_CGROUP_PIDS_MAX_V1,
        cpu_quota_us: PUBLIC_CGROUP_CPU_QUOTA_US_V1,
        cpu_period_us: PUBLIC_CGROUP_CPU_PERIOD_US_V1,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWorkerExecutionEvidence {
    pub profile: String,
    pub inner: PublicWorkerInputEvidence,
    pub address_space_bytes: u64,
    pub cpu_seconds: u64,
    pub file_size_bytes: u64,
    pub open_files: u64,
    pub process_count: u64,
    pub wall_time_ms: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
    pub cgroup_memory_max_bytes: u64,
    pub cgroup_pids_max: u64,
    pub cgroup_cpu_quota_us: u64,
    pub cgroup_cpu_period_us: u64,
    pub response_frame_overhead_bytes: u64,
    pub max_response_frame_bytes: u64,
    pub exact_supervisor_profile: bool,
    pub exact_cgroup_profile: bool,
    pub framed_response_fits_stdout_exactly: bool,
    pub evidence_sha256: String,
}

impl PublicWorkerExecutionEvidence {
    pub fn verify(&self) -> Result<(), PublicWorkerExecutionError> {
        self.inner.verify()?;
        if self.profile != PUBLIC_WORKER_EXECUTION_PROFILE_V1
            || self.address_space_bytes != PUBLIC_ADDRESS_SPACE_BYTES_V1
            || self.cpu_seconds != PUBLIC_CPU_SECONDS_V1
            || self.file_size_bytes != PUBLIC_FILE_SIZE_BYTES_V1
            || self.open_files != PUBLIC_OPEN_FILES_V1
            || self.process_count != PUBLIC_PROCESS_COUNT_V1
            || self.wall_time_ms != PUBLIC_WALL_TIME_MS_V1
            || self.max_stdout_bytes != PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
            || self.max_stderr_bytes != PUBLIC_STDERR_BYTES_V1
            || self.cgroup_memory_max_bytes != PUBLIC_CGROUP_MEMORY_MAX_BYTES_V1
            || self.cgroup_pids_max != PUBLIC_CGROUP_PIDS_MAX_V1
            || self.cgroup_cpu_quota_us != PUBLIC_CGROUP_CPU_QUOTA_US_V1
            || self.cgroup_cpu_period_us != PUBLIC_CGROUP_CPU_PERIOD_US_V1
            || self.response_frame_overhead_bytes != PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1
            || self.max_response_frame_bytes != PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
            || !self.exact_supervisor_profile
            || !self.exact_cgroup_profile
            || !self.framed_response_fits_stdout_exactly
        {
            return Err(PublicWorkerExecutionError::InvalidEvidence);
        }

        let inner_worker = &self.inner.inner;
        let launch_limits = &inner_worker.launch.limits;
        if launch_limits.address_space_bytes != self.address_space_bytes
            || launch_limits.cpu_seconds != self.cpu_seconds
            || launch_limits.file_size_bytes != self.file_size_bytes
            || launch_limits.open_files != self.open_files
            || launch_limits.process_count != self.process_count
            || inner_worker.parent_wall_time_ms != self.wall_time_ms
            || inner_worker.parent_max_stdout_bytes != self.max_stdout_bytes
            || inner_worker.parent_max_stderr_bytes != self.max_stderr_bytes
        {
            return Err(PublicWorkerExecutionError::InvalidEvidence);
        }

        let target = &inner_worker.launch.base.target;
        if target.memory_max_bytes != self.cgroup_memory_max_bytes
            || target.pids_max != self.cgroup_pids_max
            || target.cpu_quota_us != self.cgroup_cpu_quota_us
            || target.cpu_period_us != self.cgroup_cpu_period_us
        {
            return Err(PublicWorkerExecutionError::InvalidEvidence);
        }

        let framed = self
            .inner
            .max_response_json_bytes
            .checked_add(self.response_frame_overhead_bytes)
            .ok_or(PublicWorkerExecutionError::ResponseFrameOverflow)?;
        if framed != self.max_response_frame_bytes || framed != self.max_stdout_bytes {
            return Err(PublicWorkerExecutionError::InvalidEvidence);
        }

        if self.evidence_sha256 != hex_digest(evidence_sha256_v1(self)) {
            return Err(PublicWorkerExecutionError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PublicWorkerExecutionInvocation {
    inner: PublicWorkerInvocation,
    evidence: PublicWorkerExecutionEvidence,
}

impl PublicWorkerExecutionInvocation {
    pub fn result(&self) -> &SimulationResult {
        self.inner.result()
    }

    pub fn worker(&self) -> &WorkerSuccess {
        self.inner.worker()
    }

    pub fn input_evidence(&self) -> &PublicWorkerInputEvidence {
        self.inner.evidence()
    }

    pub fn evidence(&self) -> &PublicWorkerExecutionEvidence {
        &self.evidence
    }

    pub fn into_result(self) -> SimulationResult {
        self.inner.into_result()
    }
}

#[derive(Debug, Error)]
pub enum PublicWorkerExecutionError {
    #[error(transparent)]
    Input(#[from] PublicWorkerInputError),
    #[error(transparent)]
    Cgroup(#[from] CgroupV2Error),
    #[error("public worker requires the exact v1 cgroup resource envelope")]
    CgroupLimitsMismatch,
    #[error("public worker response-frame arithmetic overflow")]
    ResponseFrameOverflow,
    #[error("public worker execution evidence is invalid")]
    InvalidEvidence,
    #[error("public worker execution evidence digest mismatch")]
    EvidenceDigestMismatch,
}

pub fn execute_public_worker_v1(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
) -> Result<PublicWorkerExecutionInvocation, PublicWorkerExecutionError> {
    // Hostile/public input is bounded before any live cgroup inspection or child
    // process creation. The prepared value has private fields and cannot be
    // forged by downstream callers.
    let prepared = prepare_public_simulation_input(manifest_bytes, component_bytes, request)?;

    require_public_cgroup_v1(cgroup)?;
    let inner = execute_prepared_public_first_instruction_worker_with_rlimits(
        image,
        cgroup,
        prepared,
        public_supervisor_limits_v1(),
    )?;

    let mut evidence = PublicWorkerExecutionEvidence {
        profile: PUBLIC_WORKER_EXECUTION_PROFILE_V1.into(),
        inner: inner.evidence().clone(),
        address_space_bytes: PUBLIC_ADDRESS_SPACE_BYTES_V1,
        cpu_seconds: PUBLIC_CPU_SECONDS_V1,
        file_size_bytes: PUBLIC_FILE_SIZE_BYTES_V1,
        open_files: PUBLIC_OPEN_FILES_V1,
        process_count: PUBLIC_PROCESS_COUNT_V1,
        wall_time_ms: PUBLIC_WALL_TIME_MS_V1,
        max_stdout_bytes: PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1,
        max_stderr_bytes: PUBLIC_STDERR_BYTES_V1,
        cgroup_memory_max_bytes: PUBLIC_CGROUP_MEMORY_MAX_BYTES_V1,
        cgroup_pids_max: PUBLIC_CGROUP_PIDS_MAX_V1,
        cgroup_cpu_quota_us: PUBLIC_CGROUP_CPU_QUOTA_US_V1,
        cgroup_cpu_period_us: PUBLIC_CGROUP_CPU_PERIOD_US_V1,
        response_frame_overhead_bytes: PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1,
        max_response_frame_bytes: PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1,
        exact_supervisor_profile: true,
        exact_cgroup_profile: true,
        framed_response_fits_stdout_exactly: true,
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence));
    evidence.verify()?;

    Ok(PublicWorkerExecutionInvocation { inner, evidence })
}

fn require_public_cgroup_v1(cgroup: &CgroupV2Lease) -> Result<(), PublicWorkerExecutionError> {
    validate_public_cgroup_limits_v1(cgroup.limits())?;
    cgroup.verify_configuration()?;
    Ok(())
}

fn validate_public_cgroup_limits_v1(
    limits: CgroupV2Limits,
) -> Result<(), PublicWorkerExecutionError> {
    if limits != public_cgroup_limits_v1() {
        return Err(PublicWorkerExecutionError::CgroupLimitsMismatch);
    }
    Ok(())
}

fn evidence_sha256_v1(evidence: &PublicWorkerExecutionEvidence) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    update_string(&mut hasher, &evidence.profile);
    update_string(&mut hasher, &evidence.inner.evidence_sha256);
    hasher.update(evidence.address_space_bytes.to_le_bytes());
    hasher.update(evidence.cpu_seconds.to_le_bytes());
    hasher.update(evidence.file_size_bytes.to_le_bytes());
    hasher.update(evidence.open_files.to_le_bytes());
    hasher.update(evidence.process_count.to_le_bytes());
    hasher.update(evidence.wall_time_ms.to_le_bytes());
    hasher.update(evidence.max_stdout_bytes.to_le_bytes());
    hasher.update(evidence.max_stderr_bytes.to_le_bytes());
    hasher.update(evidence.cgroup_memory_max_bytes.to_le_bytes());
    hasher.update(evidence.cgroup_pids_max.to_le_bytes());
    hasher.update(evidence.cgroup_cpu_quota_us.to_le_bytes());
    hasher.update(evidence.cgroup_cpu_period_us.to_le_bytes());
    hasher.update(evidence.response_frame_overhead_bytes.to_le_bytes());
    hasher.update(evidence.max_response_frame_bytes.to_le_bytes());
    hasher.update([u8::from(evidence.exact_supervisor_profile)]);
    hasher.update([u8::from(evidence.exact_cgroup_profile)]);
    hasher.update([u8::from(evidence.framed_response_fits_stdout_exactly)]);
    hasher.finalize().into()
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_worker::{
        WorkerFailure, WorkerFailureKind, WorkerFrameLimits, WorkerResponse, write_response_frame,
    };

    #[test]
    fn public_resource_profile_is_frozen() {
        assert_eq!(
            public_supervisor_limits_v1(),
            SupervisorLimits {
                address_space_bytes: 16 * 1024 * 1024 * 1024,
                cpu_seconds: 15,
                file_size_bytes: 1024 * 1024,
                open_files: 32,
                process_count: 64,
                wall_time_ms: 20_000,
                max_stdout_bytes: 16 * 1024 * 1024 + 20,
                max_stderr_bytes: 256 * 1024,
            }
        );
        assert_eq!(public_cgroup_limits_v1(), CgroupV2Limits::default());
    }

    #[test]
    fn v1_response_writer_proves_twenty_byte_frame_overhead() {
        let response = WorkerResponse::Err(WorkerFailure {
            kind: WorkerFailureKind::Internal,
            message: "probe".into(),
        });
        let mut bytes = Vec::new();
        write_response_frame(&mut bytes, &response, WorkerFrameLimits::default()).unwrap();
        assert!(bytes.len() >= PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1 as usize);
        let payload_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        assert_eq!(
            bytes.len() as u64,
            payload_len + PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1
        );
    }

    #[test]
    fn public_stdout_limit_exactly_covers_maximum_v1_response_frame() {
        assert_eq!(
            PUBLIC_MAX_RESPONSE_JSON_BYTES_V1 + PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1,
            PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
        );
        assert_eq!(
            public_supervisor_limits_v1().max_stdout_bytes,
            PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
        );
    }

    #[test]
    fn alternate_cgroup_limits_are_rejected() {
        let mut limits = public_cgroup_limits_v1();
        limits.memory_max_bytes += 1;
        assert!(matches!(
            validate_public_cgroup_limits_v1(limits),
            Err(PublicWorkerExecutionError::CgroupLimitsMismatch)
        ));
    }
}
