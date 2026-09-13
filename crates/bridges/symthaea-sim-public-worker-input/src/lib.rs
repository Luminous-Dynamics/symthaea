// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pre-spawn input envelope for public/community simulation Components.
//!
//! The generic worker protocol intentionally remains reusable and comparatively
//! permissive. This layer is narrower: it freezes the public raw-manifest and
//! Component ceilings to the default zero-authority Component host, adds a
//! separately versioned request-JSON ceiling, and performs those checks before
//! the stronger first-instruction worker can spawn a process.
//!
//! This is a resource/transport theorem only. It does not mint admission,
//! routing, worker qualification, deployment authority, engineering evidence or
//! release authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use symthaea_extension_host::ControlHostPolicy;
use symthaea_sim_bridge::{SimulationRequest, SimulationResult};
use symthaea_sim_first_instruction_worker_rlimit::{
    FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1, FirstInstructionWorkerRlimitError,
    FirstInstructionWorkerRlimitEvidence, FirstInstructionWorkerRlimitInvocation,
    execute_first_instruction_worker_with_rlimits,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits, WorkerSuccess};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_image::SealedWorkerImage;
use thiserror::Error;

pub const PUBLIC_WORKER_INPUT_PROFILE_V1: &str =
    "symthaea.simulation.public-worker-input-envelope.v1";
pub const PUBLIC_MAX_MANIFEST_BYTES_V1: u64 = 256 * 1024;
pub const PUBLIC_MAX_COMPONENT_BYTES_V1: u64 = 16 * 1024 * 1024;
pub const PUBLIC_MAX_REQUEST_JSON_BYTES_V1: u64 = 256 * 1024;
/// Wire/protocol ceiling only. This is deliberately not described as the
/// simulation host's semantic output ceiling.
pub const PUBLIC_MAX_RESPONSE_JSON_BYTES_V1: u64 = 16 * 1024 * 1024;
pub const RESPONSE_LIMIT_SEMANTICS_V1: &str = "protocol-cap-only";

const EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.simulation.public-worker-input-envelope.v1\0";

pub const fn public_worker_frame_limits_v1() -> WorkerFrameLimits {
    WorkerFrameLimits {
        max_manifest_bytes: PUBLIC_MAX_MANIFEST_BYTES_V1,
        max_component_bytes: PUBLIC_MAX_COMPONENT_BYTES_V1,
        max_request_json_bytes: PUBLIC_MAX_REQUEST_JSON_BYTES_V1,
        max_response_json_bytes: PUBLIC_MAX_RESPONSE_JSON_BYTES_V1,
    }
}

#[derive(Debug)]
pub struct PreparedPublicSimulationInput<'a> {
    manifest_bytes: &'a [u8],
    component_bytes: &'a [u8],
    request: &'a SimulationRequest,
    manifest_len: u64,
    component_len: u64,
    request_json_len: u64,
}

impl<'a> PreparedPublicSimulationInput<'a> {
    pub const fn manifest_bytes(&self) -> &'a [u8] {
        self.manifest_bytes
    }

    pub const fn component_bytes(&self) -> &'a [u8] {
        self.component_bytes
    }

    pub const fn request(&self) -> &'a SimulationRequest {
        self.request
    }

    pub const fn manifest_len(&self) -> u64 {
        self.manifest_len
    }

    pub const fn component_len(&self) -> u64 {
        self.component_len
    }

    pub const fn request_json_len(&self) -> u64 {
        self.request_json_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWorkerInputEvidence {
    pub profile: String,
    pub inner: FirstInstructionWorkerRlimitEvidence,
    pub max_manifest_bytes: u64,
    pub max_component_bytes: u64,
    pub max_request_json_bytes: u64,
    pub max_response_json_bytes: u64,
    pub observed_manifest_bytes: u64,
    pub observed_component_bytes: u64,
    pub observed_request_json_bytes: u64,
    pub pre_spawn_raw_input_gate: bool,
    pub bounded_request_json_counter: bool,
    pub host_input_ceiling_parity: bool,
    pub response_limit_semantics: String,
    pub evidence_sha256: String,
}

impl PublicWorkerInputEvidence {
    pub fn verify(&self) -> Result<(), PublicWorkerInputError> {
        self.inner.verify()?;
        if self.profile != PUBLIC_WORKER_INPUT_PROFILE_V1
            || self.inner.profile != FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1
            || self.max_manifest_bytes != PUBLIC_MAX_MANIFEST_BYTES_V1
            || self.max_component_bytes != PUBLIC_MAX_COMPONENT_BYTES_V1
            || self.max_request_json_bytes != PUBLIC_MAX_REQUEST_JSON_BYTES_V1
            || self.max_response_json_bytes != PUBLIC_MAX_RESPONSE_JSON_BYTES_V1
            || self.observed_manifest_bytes > self.max_manifest_bytes
            || self.observed_component_bytes > self.max_component_bytes
            || self.observed_request_json_bytes > self.max_request_json_bytes
            || !self.pre_spawn_raw_input_gate
            || !self.bounded_request_json_counter
            || !self.host_input_ceiling_parity
            || self.response_limit_semantics != RESPONSE_LIMIT_SEMANTICS_V1
        {
            return Err(PublicWorkerInputError::InvalidEvidence);
        }
        if self.evidence_sha256 != hex_digest(evidence_sha256_v1(self)) {
            return Err(PublicWorkerInputError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PublicWorkerInvocation {
    inner: FirstInstructionWorkerRlimitInvocation,
    evidence: PublicWorkerInputEvidence,
}

impl PublicWorkerInvocation {
    pub fn result(&self) -> &SimulationResult {
        self.inner.result()
    }

    pub fn worker(&self) -> &WorkerSuccess {
        self.inner.worker()
    }

    pub fn inner_evidence(&self) -> &FirstInstructionWorkerRlimitEvidence {
        self.inner.evidence()
    }

    pub fn evidence(&self) -> &PublicWorkerInputEvidence {
        &self.evidence
    }

    pub fn into_result(self) -> SimulationResult {
        self.inner.into_result()
    }
}

#[derive(Debug, Error)]
pub enum PublicWorkerInputError {
    #[error("public worker host-policy ceiling drift for {field}: expected {expected}, actual {actual}")]
    HostPolicyDrift {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("{field} length cannot be represented by the public worker profile")]
    LengthOverflow { field: &'static str },
    #[error("public manifest length {actual} exceeds pre-spawn limit {limit}")]
    ManifestTooLarge { actual: u64, limit: u64 },
    #[error("public Component length {actual} exceeds pre-spawn limit {limit}")]
    ComponentTooLarge { actual: u64, limit: u64 },
    #[error("public simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error("public request JSON exceeds pre-spawn limit {limit}")]
    RequestJsonTooLarge { limit: u64 },
    #[error("public request JSON measurement failed: {0}")]
    RequestJson(serde_json::Error),
    #[error(transparent)]
    Inner(#[from] FirstInstructionWorkerRlimitError),
    #[error("public worker input evidence is invalid")]
    InvalidEvidence,
    #[error("public worker input evidence digest mismatch")]
    EvidenceDigestMismatch,
}

/// Validate hostile/public input before the worker path may parse the manifest,
/// hash the Component, create a cgroup child, or write request-frame bytes.
pub fn prepare_public_simulation_input<'a>(
    manifest_bytes: &'a [u8],
    component_bytes: &'a [u8],
    request: &'a SimulationRequest,
) -> Result<PreparedPublicSimulationInput<'a>, PublicWorkerInputError> {
    require_host_policy_parity()?;

    // Raw length checks deliberately precede manifest JSON parsing and Component
    // hashing performed by the inner technical worker layer.
    let manifest_len = checked_len("manifest", manifest_bytes.len())?;
    if manifest_len > PUBLIC_MAX_MANIFEST_BYTES_V1 {
        return Err(PublicWorkerInputError::ManifestTooLarge {
            actual: manifest_len,
            limit: PUBLIC_MAX_MANIFEST_BYTES_V1,
        });
    }
    let component_len = checked_len("component", component_bytes.len())?;
    if component_len > PUBLIC_MAX_COMPONENT_BYTES_V1 {
        return Err(PublicWorkerInputError::ComponentTooLarge {
            actual: component_len,
            limit: PUBLIC_MAX_COMPONENT_BYTES_V1,
        });
    }

    request
        .validate()
        .map_err(|error| PublicWorkerInputError::InvalidRequest(error.to_string()))?;
    let request_json_len = measure_request_json_bounded(request)?;

    Ok(PreparedPublicSimulationInput {
        manifest_bytes,
        component_bytes,
        request,
        manifest_len,
        component_len,
        request_json_len,
    })
}

pub fn execute_public_first_instruction_worker_with_rlimits(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    request: &SimulationRequest,
    limits: SupervisorLimits,
) -> Result<PublicWorkerInvocation, PublicWorkerInputError> {
    let prepared = prepare_public_simulation_input(manifest_bytes, component_bytes, request)?;
    execute_prepared_public_first_instruction_worker_with_rlimits(image, cgroup, prepared, limits)
}

pub fn execute_prepared_public_first_instruction_worker_with_rlimits(
    image: &SealedWorkerImage,
    cgroup: &CgroupV2Lease,
    prepared: PreparedPublicSimulationInput<'_>,
    limits: SupervisorLimits,
) -> Result<PublicWorkerInvocation, PublicWorkerInputError> {
    // The inner path only becomes reachable after the non-forgeable prepared
    // value has established the public pre-spawn input envelope.
    let inner = execute_first_instruction_worker_with_rlimits(
        image,
        cgroup,
        prepared.manifest_bytes,
        prepared.component_bytes,
        prepared.request,
        public_worker_frame_limits_v1(),
        limits,
    )?;

    let mut evidence = PublicWorkerInputEvidence {
        profile: PUBLIC_WORKER_INPUT_PROFILE_V1.into(),
        inner: inner.evidence().clone(),
        max_manifest_bytes: PUBLIC_MAX_MANIFEST_BYTES_V1,
        max_component_bytes: PUBLIC_MAX_COMPONENT_BYTES_V1,
        max_request_json_bytes: PUBLIC_MAX_REQUEST_JSON_BYTES_V1,
        max_response_json_bytes: PUBLIC_MAX_RESPONSE_JSON_BYTES_V1,
        observed_manifest_bytes: prepared.manifest_len,
        observed_component_bytes: prepared.component_len,
        observed_request_json_bytes: prepared.request_json_len,
        pre_spawn_raw_input_gate: true,
        bounded_request_json_counter: true,
        host_input_ceiling_parity: true,
        response_limit_semantics: RESPONSE_LIMIT_SEMANTICS_V1.into(),
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence));
    evidence.verify()?;

    Ok(PublicWorkerInvocation { inner, evidence })
}

fn require_host_policy_parity() -> Result<(), PublicWorkerInputError> {
    let policy = ControlHostPolicy::default();
    let manifest = u64::try_from(policy.max_manifest_bytes)
        .map_err(|_| PublicWorkerInputError::LengthOverflow { field: "host manifest policy" })?;
    let component = u64::try_from(policy.max_component_bytes)
        .map_err(|_| PublicWorkerInputError::LengthOverflow { field: "host Component policy" })?;
    if manifest != PUBLIC_MAX_MANIFEST_BYTES_V1 {
        return Err(PublicWorkerInputError::HostPolicyDrift {
            field: "max_manifest_bytes",
            expected: PUBLIC_MAX_MANIFEST_BYTES_V1,
            actual: manifest,
        });
    }
    if component != PUBLIC_MAX_COMPONENT_BYTES_V1 {
        return Err(PublicWorkerInputError::HostPolicyDrift {
            field: "max_component_bytes",
            expected: PUBLIC_MAX_COMPONENT_BYTES_V1,
            actual: component,
        });
    }
    Ok(())
}

fn checked_len(field: &'static str, len: usize) -> Result<u64, PublicWorkerInputError> {
    u64::try_from(len).map_err(|_| PublicWorkerInputError::LengthOverflow { field })
}

#[derive(Debug)]
struct BoundedCountWriter {
    count: u64,
    limit: u64,
    exceeded: bool,
}

impl BoundedCountWriter {
    const fn new(limit: u64) -> Self {
        Self {
            count: 0,
            limit,
            exceeded: false,
        }
    }
}

impl Write for BoundedCountWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let add = u64::try_from(bytes.len())
            .map_err(|_| io::Error::other("JSON write length overflow"))?;
        let next = self
            .count
            .checked_add(add)
            .ok_or_else(|| io::Error::other("JSON byte count overflow"))?;
        if next > self.limit {
            self.exceeded = true;
            return Err(io::Error::other("bounded request JSON limit exceeded"));
        }
        self.count = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn measure_request_json_bounded(request: &SimulationRequest) -> Result<u64, PublicWorkerInputError> {
    let mut writer = BoundedCountWriter::new(PUBLIC_MAX_REQUEST_JSON_BYTES_V1);
    if let Err(error) = serde_json::to_writer(&mut writer, request) {
        if writer.exceeded {
            return Err(PublicWorkerInputError::RequestJsonTooLarge {
                limit: PUBLIC_MAX_REQUEST_JSON_BYTES_V1,
            });
        }
        return Err(PublicWorkerInputError::RequestJson(error));
    }
    Ok(writer.count)
}

fn evidence_sha256_v1(evidence: &PublicWorkerInputEvidence) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    update_string(&mut hasher, &evidence.profile);
    update_string(&mut hasher, &evidence.inner.evidence_sha256);
    hasher.update(evidence.max_manifest_bytes.to_le_bytes());
    hasher.update(evidence.max_component_bytes.to_le_bytes());
    hasher.update(evidence.max_request_json_bytes.to_le_bytes());
    hasher.update(evidence.max_response_json_bytes.to_le_bytes());
    hasher.update(evidence.observed_manifest_bytes.to_le_bytes());
    hasher.update(evidence.observed_component_bytes.to_le_bytes());
    hasher.update(evidence.observed_request_json_bytes.to_le_bytes());
    hasher.update([u8::from(evidence.pre_spawn_raw_input_gate)]);
    hasher.update([u8::from(evidence.bounded_request_json_counter)]);
    hasher.update([u8::from(evidence.host_input_ceiling_parity)]);
    update_string(&mut hasher, &evidence.response_limit_semantics);
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
    use symthaea_sim_bridge::{EngineeringDomain, SolverKind};

    fn request(objective: impl Into<String>) -> SimulationRequest {
        SimulationRequest::new(
            "public-input-test",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            objective,
        )
    }

    #[test]
    fn profile_is_exactly_aligned_with_default_control_host_input_ceilings() {
        require_host_policy_parity().unwrap();
        let policy = ControlHostPolicy::default();
        assert_eq!(policy.max_manifest_bytes as u64, PUBLIC_MAX_MANIFEST_BYTES_V1);
        assert_eq!(policy.max_component_bytes as u64, PUBLIC_MAX_COMPONENT_BYTES_V1);
        assert_eq!(public_worker_frame_limits_v1().max_request_json_bytes, 256 * 1024);
        assert_eq!(
            public_worker_frame_limits_v1().max_response_json_bytes,
            16 * 1024 * 1024
        );
    }

    #[test]
    fn oversized_manifest_is_rejected_before_json_parsing() {
        let manifest = vec![b'{'; (PUBLIC_MAX_MANIFEST_BYTES_V1 + 1) as usize];
        let error = prepare_public_simulation_input(&manifest, &[0], &request("bounded"))
            .expect_err("oversized raw manifest must fail before inner JSON parsing");
        assert!(matches!(
            error,
            PublicWorkerInputError::ManifestTooLarge { .. }
        ));
    }

    #[test]
    fn oversized_component_is_rejected_by_raw_preflight() {
        let component = vec![0u8; (PUBLIC_MAX_COMPONENT_BYTES_V1 + 1) as usize];
        let error = prepare_public_simulation_input(b"{}", &component, &request("bounded"))
            .expect_err("oversized raw Component must fail before inner hashing/spawn");
        assert!(matches!(
            error,
            PublicWorkerInputError::ComponentTooLarge { .. }
        ));
    }

    #[test]
    fn huge_request_is_counted_without_building_a_json_vec() {
        let huge = "x".repeat((PUBLIC_MAX_REQUEST_JSON_BYTES_V1 + 4096) as usize);
        let error = prepare_public_simulation_input(b"{}", &[0], &request(huge))
            .expect_err("oversized request JSON must fail in bounded counter");
        assert!(matches!(
            error,
            PublicWorkerInputError::RequestJsonTooLarge { .. }
        ));
    }

    #[test]
    fn ordinary_input_prepares_with_exact_observed_sizes() {
        let manifest = br#"{"id":"fixture"}"#;
        let component = [0u8; 32];
        let request = request("bounded");
        let prepared = prepare_public_simulation_input(manifest, &component, &request).unwrap();
        assert_eq!(prepared.manifest_len(), manifest.len() as u64);
        assert_eq!(prepared.component_len(), component.len() as u64);
        assert!(prepared.request_json_len() > 0);
        assert!(prepared.request_json_len() <= PUBLIC_MAX_REQUEST_JSON_BYTES_V1);
    }
}
