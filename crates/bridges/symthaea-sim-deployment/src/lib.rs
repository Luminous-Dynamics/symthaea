// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact deployment bindings for admitted simulation Components.
//!
//! An extension manifest describes package truth. A deployment binding describes
//! how one exact admitted package will be executed by this host. These are
//! deliberately distinct facts: an admitted `runtime = wasm` manifest is never
//! rewritten to `remote` merely because the host chooses an out-of-process
//! contained worker.
//!
//! A worker profile label is not enough. Deployment issuance additionally
//! requires a live [`ActiveWorkerQualification`] whose exact worker digest equals
//! the exact sealed image. Persisted qualification/deployment evidence cannot
//! recreate either admission or worker authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem, Sha256Digest,
};
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_sim_bridge::SimulationRequest;
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_worker::{
    SUPERVISOR_PROFILE_V1, SupervisorError, SupervisorLimits, WORKER_PROTOCOL_V1,
    WorkerFrameLimits,
};
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::{
    SEALED_WORKER_IMAGE_PROFILE_V1, SealedWorkerImage, SealedWorkerImageError,
    SealedWorkerInvocation,
};
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
    WORKER_QUALIFICATION_PROFILE_V1,
};
use thiserror::Error;

pub const SIMULATION_DEPLOYMENT_PROFILE_V1: &str = "symthaea.simulation.deployment.v1";
pub const SIMULATION_DEPLOYMENT_TRANSPORT_V1: &str =
    "symthaea.simulation.deployment.local-contained-worker.v1";

const DEPLOYMENT_DOMAIN_V1: &[u8] = b"symthaea.simulation.deployment.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentSupervisorEnvelope {
    pub address_space_bytes: u64,
    pub cpu_seconds: u64,
    pub file_size_bytes: u64,
    pub open_files: u64,
    pub process_count: u64,
    pub wall_time_ms: u64,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
}

impl From<SupervisorLimits> for DeploymentSupervisorEnvelope {
    fn from(value: SupervisorLimits) -> Self {
        Self {
            address_space_bytes: value.address_space_bytes,
            cpu_seconds: value.cpu_seconds,
            file_size_bytes: value.file_size_bytes,
            open_files: value.open_files,
            process_count: value.process_count,
            wall_time_ms: value.wall_time_ms,
            max_stdout_bytes: value.max_stdout_bytes,
            max_stderr_bytes: value.max_stderr_bytes,
        }
    }
}

impl From<DeploymentSupervisorEnvelope> for SupervisorLimits {
    fn from(value: DeploymentSupervisorEnvelope) -> Self {
        Self {
            address_space_bytes: value.address_space_bytes,
            cpu_seconds: value.cpu_seconds,
            file_size_bytes: value.file_size_bytes,
            open_files: value.open_files,
            process_count: value.process_count,
            wall_time_ms: value.wall_time_ms,
            max_stdout_bytes: value.max_stdout_bytes,
            max_stderr_bytes: value.max_stderr_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentFrameEnvelope {
    pub max_manifest_bytes: u64,
    pub max_component_bytes: u64,
    pub max_request_json_bytes: u64,
    pub max_response_json_bytes: u64,
}

impl From<WorkerFrameLimits> for DeploymentFrameEnvelope {
    fn from(value: WorkerFrameLimits) -> Self {
        Self {
            max_manifest_bytes: value.max_manifest_bytes,
            max_component_bytes: value.max_component_bytes,
            max_request_json_bytes: value.max_request_json_bytes,
            max_response_json_bytes: value.max_response_json_bytes,
        }
    }
}

impl From<DeploymentFrameEnvelope> for WorkerFrameLimits {
    fn from(value: DeploymentFrameEnvelope) -> Self {
        Self {
            max_manifest_bytes: value.max_manifest_bytes,
            max_component_bytes: value.max_component_bytes,
            max_request_json_bytes: value.max_request_json_bytes,
            max_response_json_bytes: value.max_response_json_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationDeploymentEvidence {
    pub profile: String,
    pub transport: String,
    pub selected_extension: String,
    pub selected_version: String,
    pub package_runtime: RuntimeKind,
    pub admission_generation: u64,
    pub trust_generation: u64,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub policy_sha256: String,
    pub worker_qualification_profile: String,
    pub worker_qualification_generation: u64,
    pub worker_qualification_evidence_sha256: String,
    pub worker_image_sha256: String,
    pub worker_image_size_bytes: u64,
    pub worker_image_seal_mask: u32,
    pub worker_image_profile: String,
    pub supervisor_profile: String,
    pub worker_protocol: String,
    pub process_containment_profile: String,
    pub filesystem_containment_profile: String,
    pub supervisor: DeploymentSupervisorEnvelope,
    pub frame: DeploymentFrameEnvelope,
    pub deployment_sha256: String,
}

/// Non-serializable host binding between one exact active package admission and
/// one exact actively-qualified sealed worker image.
#[derive(Debug)]
pub struct BoundSimulationDeployment {
    manifest: ExtensionManifest,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
    worker_image: SealedWorkerImage,
    selected_extension: String,
    selected_version: String,
    admission_generation: u64,
    trust_generation: u64,
    manifest_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
    worker_image_size_bytes: u64,
    worker_image_seal_mask: u32,
    supervisor: DeploymentSupervisorEnvelope,
    frame: DeploymentFrameEnvelope,
    deployment_sha256: [u8; 32],
}

impl BoundSimulationDeployment {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        admission: &ActiveAdmission,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_qualification: &ActiveWorkerQualification,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        manifest_bytes: impl Into<Arc<[u8]>>,
        component_bytes: impl Into<Arc<[u8]>>,
        worker_image: SealedWorkerImage,
        supervisor_limits: SupervisorLimits,
        frame_limits: WorkerFrameLimits,
    ) -> Result<Self, SimulationDeploymentError> {
        admission
            .recheck_currentness(admission_currentness)
            .map_err(SimulationDeploymentError::IssuanceAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(SimulationDeploymentError::IssuanceWorkerCurrentness)?;
        if !worker_qualification.matches_image(&worker_image) {
            return Err(SimulationDeploymentError::WorkerQualificationMismatch);
        }
        supervisor_limits
            .validate()
            .map_err(SimulationDeploymentError::Supervisor)?;
        validate_frame_limits(frame_limits)?;

        let manifest_bytes = manifest_bytes.into();
        let component_bytes = component_bytes.into();
        let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(SimulationDeploymentError::ManifestJson)?;
        manifest
            .validate()
            .map_err(|problems| SimulationDeploymentError::ManifestInvalid(format!("{problems:?}")))?;

        if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
            return Err(SimulationDeploymentError::UnexpectedManifest);
        }
        if !admission.matches_manifest(&manifest) {
            return Err(SimulationDeploymentError::AdmissionManifestMismatch);
        }
        if admission.extension() != &manifest.id
            || admission.subject().extension_version() != manifest.version
        {
            return Err(SimulationDeploymentError::AdmissionIdentityMismatch);
        }

        let manifest_sha256: [u8; 32] = Sha256::digest(manifest_bytes.as_ref()).into();
        let payload_sha256: [u8; 32] = Sha256::digest(component_bytes.as_ref()).into();
        require_digest("manifest", admission.manifest_sha256(), manifest_sha256)?;
        require_digest("payload", admission.payload_sha256(), payload_sha256)?;

        if manifest_bytes.len() as u128 > frame_limits.max_manifest_bytes as u128 {
            return Err(SimulationDeploymentError::PackageExceedsFrame("manifest"));
        }
        if component_bytes.len() as u128 > frame_limits.max_component_bytes as u128 {
            return Err(SimulationDeploymentError::PackageExceedsFrame("component"));
        }

        let selected_extension = manifest.id.as_str().to_owned();
        let selected_version = manifest.version.clone();
        let admission_generation = admission.generation();
        let trust_generation = admission.trust_generation();
        let policy_sha256 = admission.policy_sha256().0;
        let worker_qualification_generation = worker_qualification.generation();
        let worker_qualification_evidence_sha256 =
            worker_qualification.qualification_evidence_sha256();
        let worker_image_sha256 = worker_image.image_sha256();
        let worker_image_size_bytes = worker_image.image_size_bytes();
        let worker_image_seal_mask = worker_image.seal_mask();
        let supervisor = supervisor_limits.into();
        let frame = frame_limits.into();

        let deployment_sha256 = deployment_sha256_v1(
            &selected_extension,
            &selected_version,
            admission_generation,
            trust_generation,
            manifest_sha256,
            payload_sha256,
            policy_sha256,
            worker_qualification_generation,
            worker_qualification_evidence_sha256,
            worker_image_sha256,
            worker_image_size_bytes,
            worker_image_seal_mask,
            supervisor,
            frame,
        )?;

        Ok(Self {
            manifest,
            manifest_bytes,
            component_bytes,
            worker_image,
            selected_extension,
            selected_version,
            admission_generation,
            trust_generation,
            manifest_sha256,
            payload_sha256,
            policy_sha256,
            worker_qualification_generation,
            worker_qualification_evidence_sha256,
            worker_image_sha256,
            worker_image_size_bytes,
            worker_image_seal_mask,
            supervisor,
            frame,
            deployment_sha256,
        })
    }

    pub fn evidence(&self) -> SimulationDeploymentEvidence {
        SimulationDeploymentEvidence {
            profile: SIMULATION_DEPLOYMENT_PROFILE_V1.into(),
            transport: SIMULATION_DEPLOYMENT_TRANSPORT_V1.into(),
            selected_extension: self.selected_extension.clone(),
            selected_version: self.selected_version.clone(),
            package_runtime: RuntimeKind::Wasm,
            admission_generation: self.admission_generation,
            trust_generation: self.trust_generation,
            manifest_sha256: hex_digest(self.manifest_sha256),
            payload_sha256: hex_digest(self.payload_sha256),
            policy_sha256: hex_digest(self.policy_sha256),
            worker_qualification_profile: WORKER_QUALIFICATION_PROFILE_V1.into(),
            worker_qualification_generation: self.worker_qualification_generation,
            worker_qualification_evidence_sha256: hex_digest(
                self.worker_qualification_evidence_sha256,
            ),
            worker_image_sha256: hex_digest(self.worker_image_sha256),
            worker_image_size_bytes: self.worker_image_size_bytes,
            worker_image_seal_mask: self.worker_image_seal_mask,
            worker_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            supervisor_profile: SUPERVISOR_PROFILE_V1.into(),
            worker_protocol: WORKER_PROTOCOL_V1.into(),
            process_containment_profile: WORKER_CONTAINMENT_PROFILE_V1.into(),
            filesystem_containment_profile: WORKER_FILESYSTEM_PROFILE_V1.into(),
            supervisor: self.supervisor,
            frame: self.frame,
            deployment_sha256: hex_digest(self.deployment_sha256),
        }
    }

    pub const fn deployment_sha256(&self) -> [u8; 32] {
        self.deployment_sha256
    }

    pub const fn worker_image_sha256(&self) -> [u8; 32] {
        self.worker_image_sha256
    }

    pub fn selected_extension(&self) -> &str {
        &self.selected_extension
    }

    pub fn selected_version(&self) -> &str {
        &self.selected_version
    }

    pub fn verify(
        &self,
        admission: &ActiveAdmission,
        worker_qualification: &ActiveWorkerQualification,
    ) -> Result<(), SimulationDeploymentError> {
        self.verify_binding(admission, worker_qualification)
    }

    /// Execute the already-bound exact package through the already-bound exact
    /// sealed worker image. This does not perform provider discovery/selection
    /// and does not mint a `SimulationReleaseReceipt`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        admission: &ActiveAdmission,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_qualification: &ActiveWorkerQualification,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        request: &SimulationRequest,
    ) -> Result<BoundDeploymentInvocation, SimulationDeploymentError> {
        self.verify_binding(admission, worker_qualification)?;
        let capability = solver_capability(request.solver);
        if !admission.allows_capability(&capability) {
            return Err(SimulationDeploymentError::CapabilityNotAdmitted);
        }
        admission
            .recheck_currentness(admission_currentness)
            .map_err(SimulationDeploymentError::PreExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(SimulationDeploymentError::PreExecutionWorkerCurrentness)?;

        let invocation = self.worker_image.execute_with_limits(
            &self.manifest_bytes,
            &self.component_bytes,
            request,
            self.supervisor.into(),
            self.frame.into(),
        )?;

        admission
            .recheck_currentness(admission_currentness)
            .map_err(SimulationDeploymentError::PostExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(SimulationDeploymentError::PostExecutionWorkerCurrentness)?;

        if invocation.invocation().worker_sha256() != self.worker_image_sha256 {
            return Err(SimulationDeploymentError::WorkerImageMismatch);
        }
        if invocation.invocation().worker().extension_id != self.selected_extension
            || invocation.invocation().worker().extension_version != self.selected_version
        {
            return Err(SimulationDeploymentError::WorkerPackageIdentityMismatch);
        }

        Ok(BoundDeploymentInvocation {
            invocation,
            deployment_sha256: self.deployment_sha256,
            worker_qualification_evidence_sha256: self.worker_qualification_evidence_sha256,
        })
    }

    fn verify_binding(
        &self,
        admission: &ActiveAdmission,
        worker_qualification: &ActiveWorkerQualification,
    ) -> Result<(), SimulationDeploymentError> {
        if !admission.matches_manifest(&self.manifest)
            || admission.extension().as_str() != self.selected_extension
            || admission.subject().extension_version() != self.selected_version
        {
            return Err(SimulationDeploymentError::AdmissionManifestMismatch);
        }
        if admission.generation() != self.admission_generation
            || admission.trust_generation() != self.trust_generation
            || admission.manifest_sha256().0 != self.manifest_sha256
            || admission.payload_sha256().0 != self.payload_sha256
            || admission.policy_sha256().0 != self.policy_sha256
        {
            return Err(SimulationDeploymentError::AdmissionCommitmentMismatch);
        }
        if worker_qualification.generation() != self.worker_qualification_generation
            || worker_qualification.qualification_evidence_sha256()
                != self.worker_qualification_evidence_sha256
            || worker_qualification.worker_sha256() != self.worker_image_sha256
            || !worker_qualification.matches_image(&self.worker_image)
        {
            return Err(SimulationDeploymentError::WorkerQualificationMismatch);
        }

        let current_manifest: [u8; 32] = Sha256::digest(self.manifest_bytes.as_ref()).into();
        let current_payload: [u8; 32] = Sha256::digest(self.component_bytes.as_ref()).into();
        if current_manifest != self.manifest_sha256 || current_payload != self.payload_sha256 {
            return Err(SimulationDeploymentError::PackageDigestDrift);
        }
        if self.worker_image.image_sha256() != self.worker_image_sha256
            || self.worker_image.image_size_bytes() != self.worker_image_size_bytes
            || self.worker_image.seal_mask() != self.worker_image_seal_mask
        {
            return Err(SimulationDeploymentError::WorkerImageMismatch);
        }

        let expected = deployment_sha256_v1(
            &self.selected_extension,
            &self.selected_version,
            self.admission_generation,
            self.trust_generation,
            self.manifest_sha256,
            self.payload_sha256,
            self.policy_sha256,
            self.worker_qualification_generation,
            self.worker_qualification_evidence_sha256,
            self.worker_image_sha256,
            self.worker_image_size_bytes,
            self.worker_image_seal_mask,
            self.supervisor,
            self.frame,
        )?;
        if expected != self.deployment_sha256 {
            return Err(SimulationDeploymentError::DeploymentDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct BoundDeploymentInvocation {
    invocation: SealedWorkerInvocation,
    deployment_sha256: [u8; 32],
    worker_qualification_evidence_sha256: [u8; 32],
}

impl BoundDeploymentInvocation {
    pub fn invocation(&self) -> &SealedWorkerInvocation {
        &self.invocation
    }

    pub const fn deployment_sha256(&self) -> [u8; 32] {
        self.deployment_sha256
    }

    pub const fn worker_qualification_evidence_sha256(&self) -> [u8; 32] {
        self.worker_qualification_evidence_sha256
    }
}

#[derive(Debug, Error)]
pub enum SimulationDeploymentError {
    #[error("deployment issuance admission currentness failed: {0:?}")]
    IssuanceAdmissionCurrentness(AdmissionProblem),
    #[error("deployment issuance worker qualification currentness failed: {0}")]
    IssuanceWorkerCurrentness(WorkerQualificationError),
    #[error("deployment pre-execution admission currentness failed: {0:?}")]
    PreExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("deployment pre-execution worker qualification currentness failed: {0}")]
    PreExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("deployment post-execution admission currentness failed: {0:?}")]
    PostExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("deployment post-execution worker qualification currentness failed: {0}")]
    PostExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("manifest json is invalid: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("deployment requires an admitted simulation Wasm manifest")]
    UnexpectedManifest,
    #[error("active admission does not match the exact deployment manifest")]
    AdmissionManifestMismatch,
    #[error("active admission identity/version does not match the deployment manifest")]
    AdmissionIdentityMismatch,
    #[error("active admission commitments/generations do not match deployment binding")]
    AdmissionCommitmentMismatch,
    #[error("admission {field} digest does not match exact deployment bytes")]
    AdmissionDigestMismatch { field: &'static str },
    #[error("active worker qualification does not match exact deployment worker image/evidence")]
    WorkerQualificationMismatch,
    #[error("deployment package {0} exceeds the configured worker frame envelope")]
    PackageExceedsFrame(&'static str),
    #[error("deployment request capability is not granted by the exact admission")]
    CapabilityNotAdmitted,
    #[error("exact deployment package bytes changed")]
    PackageDigestDrift,
    #[error("sealed deployment worker image no longer matches its binding")]
    WorkerImageMismatch,
    #[error("worker technical receipt does not match bound package identity/version")]
    WorkerPackageIdentityMismatch,
    #[error("deployment binding digest does not match")]
    DeploymentDigestMismatch,
    #[error("deployment frame envelope is invalid")]
    InvalidFrameEnvelope,
    #[error("string length cannot be represented by deployment profile v1")]
    LengthOverflow,
    #[error(transparent)]
    Supervisor(#[from] SupervisorError),
    #[error(transparent)]
    WorkerImage(#[from] SealedWorkerImageError),
}

fn require_digest(
    field: &'static str,
    admitted: Sha256Digest,
    exact: [u8; 32],
) -> Result<(), SimulationDeploymentError> {
    if admitted.0 != exact {
        return Err(SimulationDeploymentError::AdmissionDigestMismatch { field });
    }
    Ok(())
}

fn validate_frame_limits(limits: WorkerFrameLimits) -> Result<(), SimulationDeploymentError> {
    if limits.max_manifest_bytes == 0
        || limits.max_component_bytes == 0
        || limits.max_request_json_bytes == 0
        || limits.max_response_json_bytes == 0
    {
        return Err(SimulationDeploymentError::InvalidFrameEnvelope);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn deployment_sha256_v1(
    selected_extension: &str,
    selected_version: &str,
    admission_generation: u64,
    trust_generation: u64,
    manifest_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
    worker_image_size_bytes: u64,
    worker_image_seal_mask: u32,
    supervisor: DeploymentSupervisorEnvelope,
    frame: DeploymentFrameEnvelope,
) -> Result<[u8; 32], SimulationDeploymentError> {
    let mut hasher = Sha256::new();
    hasher.update(DEPLOYMENT_DOMAIN_V1);
    put_string(&mut hasher, SIMULATION_DEPLOYMENT_TRANSPORT_V1)?;
    put_string(&mut hasher, selected_extension)?;
    put_string(&mut hasher, selected_version)?;
    hasher.update([1]); // exact admitted package runtime = Wasm
    hasher.update(admission_generation.to_le_bytes());
    hasher.update(trust_generation.to_le_bytes());
    hasher.update(manifest_sha256);
    hasher.update(payload_sha256);
    hasher.update(policy_sha256);
    put_string(&mut hasher, WORKER_QUALIFICATION_PROFILE_V1)?;
    hasher.update(worker_qualification_generation.to_le_bytes());
    hasher.update(worker_qualification_evidence_sha256);
    hasher.update(worker_image_sha256);
    hasher.update(worker_image_size_bytes.to_le_bytes());
    hasher.update(worker_image_seal_mask.to_le_bytes());
    put_string(&mut hasher, SEALED_WORKER_IMAGE_PROFILE_V1)?;
    put_string(&mut hasher, SUPERVISOR_PROFILE_V1)?;
    put_string(&mut hasher, WORKER_PROTOCOL_V1)?;
    put_string(&mut hasher, WORKER_CONTAINMENT_PROFILE_V1)?;
    put_string(&mut hasher, WORKER_FILESYSTEM_PROFILE_V1)?;
    put_supervisor(&mut hasher, supervisor);
    put_frame(&mut hasher, frame);
    Ok(hasher.finalize().into())
}

fn put_supervisor(hasher: &mut Sha256, value: DeploymentSupervisorEnvelope) {
    hasher.update(value.address_space_bytes.to_le_bytes());
    hasher.update(value.cpu_seconds.to_le_bytes());
    hasher.update(value.file_size_bytes.to_le_bytes());
    hasher.update(value.open_files.to_le_bytes());
    hasher.update(value.process_count.to_le_bytes());
    hasher.update(value.wall_time_ms.to_le_bytes());
    hasher.update(value.max_stdout_bytes.to_le_bytes());
    hasher.update(value.max_stderr_bytes.to_le_bytes());
}

fn put_frame(hasher: &mut Sha256, value: DeploymentFrameEnvelope) {
    hasher.update(value.max_manifest_bytes.to_le_bytes());
    hasher.update(value.max_component_bytes.to_le_bytes());
    hasher.update(value.max_request_json_bytes.to_le_bytes());
    hasher.update(value.max_response_json_bytes.to_le_bytes());
}

fn put_string(hasher: &mut Sha256, value: &str) -> Result<(), SimulationDeploymentError> {
    let length = u64::try_from(value.len()).map_err(|_| SimulationDeploymentError::LengthOverflow)?;
    hasher.update(length.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
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
