// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authority-preserving wrapper that strengthens an existing deployment-v1
//! binding with exact first-instruction cgroup execution.
//!
//! The underlying `BoundSimulationDeployment` remains the package/worker
//! qualification theorem. This layer does not rewrite that profile. It binds a
//! concrete cgroup resource policy, reuses the deployment's exact frame and
//! watchdog limits, executes the exact package/image through the
//! first-instruction worker path, and requires both package admission and worker
//! qualification to remain current before and after execution.
//!
//! Persisted evidence is audit-only. Only the private-field live wrapper and
//! invocation carry execution authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem,
};
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_sim_bridge::{SimulationRequest, SimulationResult};
use symthaea_sim_deployment::{
    BoundSimulationDeployment, DeploymentFrameEnvelope, SimulationDeploymentError,
    SimulationDeploymentEvidence, SIMULATION_DEPLOYMENT_PROFILE_V1,
    SIMULATION_DEPLOYMENT_TRANSPORT_V1,
};
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_first_instruction_worker::{
    execute_first_instruction_worker, FirstInstructionWorkerError, FirstInstructionWorkerEvidence,
    FirstInstructionWorkerInvocation, FirstInstructionWorkerLimits,
    FIRST_INSTRUCTION_WORKER_PROFILE_V1,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits, SUPERVISOR_PROFILE_V1, WORKER_PROTOCOL_V1};
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Evidence, CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::{SealedWorkerImage, SEALED_WORKER_IMAGE_PROFILE_V1};
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
    WORKER_QUALIFICATION_PROFILE_V1,
};
use thiserror::Error;

pub const FIRST_INSTRUCTION_DEPLOYMENT_PROFILE_V1: &str =
    "symthaea.simulation.deployment.first-instruction-cgroup.v1";
const BINDING_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.deployment.first-instruction-cgroup.v1\0";
const EXECUTION_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.deployment.first-instruction-cgroup.execution.v1\0";
const BASE_DEPLOYMENT_DOMAIN_V1: &[u8] = b"symthaea.simulation.deployment.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionResourcePolicy {
    pub cgroup: CgroupV2Limits,
    pub worker: FirstInstructionWorkerLimits,
}

impl FirstInstructionResourcePolicy {
    pub fn validate(
        self,
        base: &SimulationDeploymentEvidence,
    ) -> Result<Self, FirstInstructionDeploymentError> {
        self.cgroup.validate()?;
        self.worker.validate()?;
        if self.worker.wall_time_ms != base.supervisor.wall_time_ms
            || self.worker.max_stdout_bytes != base.supervisor.max_stdout_bytes
            || self.worker.max_stderr_bytes != base.supervisor.max_stderr_bytes
        {
            return Err(FirstInstructionDeploymentError::WatchdogEnvelopeMismatch);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionDeploymentEvidence {
    pub profile: String,
    pub base_deployment: SimulationDeploymentEvidence,
    pub resource_policy: FirstInstructionResourcePolicy,
    pub worker_qualification_profile: String,
    pub worker_qualification_generation: u64,
    pub worker_qualification_evidence_sha256: String,
    pub exact_worker_image_sha256: String,
    pub exact_worker_image_profile: String,
    pub first_instruction_worker_profile: String,
    pub required_process_containment_profile: String,
    pub required_filesystem_containment_profile: String,
    pub containment_claim_requires_live_worker_qualification: bool,
    pub legacy_preexec_rlimits_applied: bool,
    pub base_frame_and_watchdog_reused_exactly: bool,
    pub binding_sha256: String,
}

impl FirstInstructionDeploymentEvidence {
    pub fn verify(&self) -> Result<(), FirstInstructionDeploymentError> {
        if self.profile != FIRST_INSTRUCTION_DEPLOYMENT_PROFILE_V1
            || self.worker_qualification_profile != WORKER_QUALIFICATION_PROFILE_V1
            || self.exact_worker_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
            || self.first_instruction_worker_profile != FIRST_INSTRUCTION_WORKER_PROFILE_V1
            || self.required_process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
            || self.required_filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
            || !self.containment_claim_requires_live_worker_qualification
            || self.legacy_preexec_rlimits_applied
            || !self.base_frame_and_watchdog_reused_exactly
            || self.worker_qualification_generation == 0
        {
            return Err(FirstInstructionDeploymentError::InvalidEvidence);
        }
        verify_base_deployment_evidence(&self.base_deployment)?;
        self.resource_policy.validate(&self.base_deployment)?;
        let base_deployment = parse_hex_32(&self.base_deployment.deployment_sha256)?;
        let worker_evidence = parse_hex_32(&self.worker_qualification_evidence_sha256)?;
        let worker_image = parse_hex_32(&self.exact_worker_image_sha256)?;
        if self.base_deployment.worker_image_sha256 != self.exact_worker_image_sha256
            || self.base_deployment.worker_qualification_evidence_sha256
                != self.worker_qualification_evidence_sha256
            || self.base_deployment.worker_qualification_generation
                != self.worker_qualification_generation
        {
            return Err(FirstInstructionDeploymentError::BaseDeploymentMismatch);
        }
        let expected = binding_sha256_v1(
            base_deployment,
            self.resource_policy,
            self.worker_qualification_generation,
            worker_evidence,
            worker_image,
        );
        if self.binding_sha256 != hex_digest(expected) {
            return Err(FirstInstructionDeploymentError::BindingDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstInstructionDeploymentExecutionEvidence {
    pub deployment: FirstInstructionDeploymentEvidence,
    pub worker: FirstInstructionWorkerEvidence,
    pub cgroup: CgroupV2Evidence,
    pub request_sha256: String,
    pub output_sha256: String,
    pub execution_sha256: String,
}

impl FirstInstructionDeploymentExecutionEvidence {
    pub fn verify(&self) -> Result<(), FirstInstructionDeploymentError> {
        self.deployment.verify()?;
        self.worker.verify()?;
        self.cgroup.verify()?;
        if self.worker.launch.target != self.cgroup
            || self.worker.request_sha256 != self.request_sha256
            || self.worker.output_sha256 != self.output_sha256
            || self.worker.launch.image_sha256 != self.deployment.exact_worker_image_sha256
            || self.worker.expected_process_containment_profile
                != self.deployment.required_process_containment_profile
            || self.worker.expected_filesystem_containment_profile
                != self.deployment.required_filesystem_containment_profile
            || self.worker.containment_profiles_established_by_this_layer
        {
            return Err(FirstInstructionDeploymentError::ExecutionEvidenceMismatch);
        }
        let deployment = parse_hex_32(&self.deployment.binding_sha256)?;
        let worker = parse_hex_32(&self.worker.evidence_sha256)?;
        let cgroup = parse_hex_32(&self.cgroup.evidence_sha256)?;
        let request = parse_hex_32(&self.request_sha256)?;
        let output = parse_hex_32(&self.output_sha256)?;
        let expected = execution_sha256_v1(deployment, worker, cgroup, request, output);
        if self.execution_sha256 != hex_digest(expected) {
            return Err(FirstInstructionDeploymentError::ExecutionDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct BoundFirstInstructionDeployment {
    base_deployment_sha256: [u8; 32],
    base_evidence: SimulationDeploymentEvidence,
    manifest: ExtensionManifest,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
    worker_image: SealedWorkerImage,
    resource_policy: FirstInstructionResourcePolicy,
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
    binding_sha256: [u8; 32],
}

impl BoundFirstInstructionDeployment {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        base: &BoundSimulationDeployment,
        admission: &ActiveAdmission,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_qualification: &ActiveWorkerQualification,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        manifest_bytes: impl Into<Arc<[u8]>>,
        component_bytes: impl Into<Arc<[u8]>>,
        worker_image: SealedWorkerImage,
        resource_policy: FirstInstructionResourcePolicy,
    ) -> Result<Self, FirstInstructionDeploymentError> {
        base.verify(admission, worker_qualification)?;
        admission
            .recheck_currentness(admission_currentness)
            .map_err(FirstInstructionDeploymentError::IssuanceAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(FirstInstructionDeploymentError::IssuanceWorkerCurrentness)?;

        let base_evidence = base.evidence();
        verify_base_deployment_evidence(&base_evidence)?;
        let resource_policy = resource_policy.validate(&base_evidence)?;
        let base_deployment_sha256 = parse_hex_32(&base_evidence.deployment_sha256)?;
        if base_deployment_sha256 != base.deployment_sha256() {
            return Err(FirstInstructionDeploymentError::BaseDeploymentMismatch);
        }

        let manifest_bytes = manifest_bytes.into();
        let component_bytes = component_bytes.into();
        let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(FirstInstructionDeploymentError::ManifestJson)?;
        manifest
            .validate()
            .map_err(|problems| FirstInstructionDeploymentError::ManifestInvalid(format!("{problems:?}")))?;
        if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
            return Err(FirstInstructionDeploymentError::UnexpectedManifest);
        }
        if !admission.matches_manifest(&manifest)
            || admission.extension() != &manifest.id
            || admission.subject().extension_version() != manifest.version
            || base.selected_extension() != manifest.id.as_str()
            || base.selected_version() != manifest.version
        {
            return Err(FirstInstructionDeploymentError::PackageIdentityMismatch);
        }

        let manifest_sha256: [u8; 32] = Sha256::digest(manifest_bytes.as_ref()).into();
        let component_sha256: [u8; 32] = Sha256::digest(component_bytes.as_ref()).into();
        if base_evidence.manifest_sha256 != hex_digest(manifest_sha256)
            || base_evidence.payload_sha256 != hex_digest(component_sha256)
            || admission.manifest_sha256().0 != manifest_sha256
            || admission.payload_sha256().0 != component_sha256
        {
            return Err(FirstInstructionDeploymentError::PackageDigestMismatch);
        }

        let worker_image_sha256 = worker_image.image_sha256();
        let worker_qualification_evidence_sha256 =
            worker_qualification.qualification_evidence_sha256();
        if base.worker_image_sha256() != worker_image_sha256
            || base_evidence.worker_image_sha256 != hex_digest(worker_image_sha256)
            || base_evidence.worker_image_size_bytes != worker_image.image_size_bytes()
            || base_evidence.worker_image_seal_mask != worker_image.seal_mask()
            || worker_qualification.worker_sha256() != worker_image_sha256
            || !worker_qualification.matches_image(&worker_image)
            || base_evidence.worker_qualification_generation != worker_qualification.generation()
            || base_evidence.worker_qualification_evidence_sha256
                != hex_digest(worker_qualification_evidence_sha256)
        {
            return Err(FirstInstructionDeploymentError::WorkerQualificationMismatch);
        }

        validate_frame_against_package(base_evidence.frame, &manifest_bytes, &component_bytes)?;
        let binding_sha256 = binding_sha256_v1(
            base_deployment_sha256,
            resource_policy,
            worker_qualification.generation(),
            worker_qualification_evidence_sha256,
            worker_image_sha256,
        );

        Ok(Self {
            base_deployment_sha256,
            base_evidence,
            manifest,
            manifest_bytes,
            component_bytes,
            worker_image,
            resource_policy,
            worker_qualification_generation: worker_qualification.generation(),
            worker_qualification_evidence_sha256,
            worker_image_sha256,
            binding_sha256,
        })
    }

    pub fn evidence(&self) -> FirstInstructionDeploymentEvidence {
        FirstInstructionDeploymentEvidence {
            profile: FIRST_INSTRUCTION_DEPLOYMENT_PROFILE_V1.into(),
            base_deployment: self.base_evidence.clone(),
            resource_policy: self.resource_policy,
            worker_qualification_profile: WORKER_QUALIFICATION_PROFILE_V1.into(),
            worker_qualification_generation: self.worker_qualification_generation,
            worker_qualification_evidence_sha256: hex_digest(
                self.worker_qualification_evidence_sha256,
            ),
            exact_worker_image_sha256: hex_digest(self.worker_image_sha256),
            exact_worker_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            first_instruction_worker_profile: FIRST_INSTRUCTION_WORKER_PROFILE_V1.into(),
            required_process_containment_profile: WORKER_CONTAINMENT_PROFILE_V1.into(),
            required_filesystem_containment_profile: WORKER_FILESYSTEM_PROFILE_V1.into(),
            containment_claim_requires_live_worker_qualification: true,
            legacy_preexec_rlimits_applied: false,
            base_frame_and_watchdog_reused_exactly: true,
            binding_sha256: hex_digest(self.binding_sha256),
        }
    }

    pub const fn binding_sha256(&self) -> [u8; 32] {
        self.binding_sha256
    }

    pub const fn base_deployment_sha256(&self) -> [u8; 32] {
        self.base_deployment_sha256
    }

    pub fn verify(
        &self,
        base: &BoundSimulationDeployment,
        admission: &ActiveAdmission,
        worker_qualification: &ActiveWorkerQualification,
    ) -> Result<(), FirstInstructionDeploymentError> {
        base.verify(admission, worker_qualification)?;
        let manifest_sha256: [u8; 32] = Sha256::digest(self.manifest_bytes.as_ref()).into();
        let payload_sha256: [u8; 32] = Sha256::digest(self.component_bytes.as_ref()).into();
        if base.deployment_sha256() != self.base_deployment_sha256
            || base.worker_image_sha256() != self.worker_image_sha256
            || base.selected_extension() != self.manifest.id.as_str()
            || base.selected_version() != self.manifest.version
            || admission.manifest_sha256().0 != manifest_sha256
            || admission.payload_sha256().0 != payload_sha256
            || worker_qualification.generation() != self.worker_qualification_generation
            || worker_qualification.qualification_evidence_sha256()
                != self.worker_qualification_evidence_sha256
            || worker_qualification.worker_sha256() != self.worker_image_sha256
            || !worker_qualification.matches_image(&self.worker_image)
        {
            return Err(FirstInstructionDeploymentError::BindingMismatch);
        }
        self.evidence().verify()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        base: &BoundSimulationDeployment,
        admission: &ActiveAdmission,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_qualification: &ActiveWorkerQualification,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        cgroup: &CgroupV2Lease,
        request: &SimulationRequest,
    ) -> Result<FirstInstructionDeploymentInvocation, FirstInstructionDeploymentError> {
        self.verify(base, admission, worker_qualification)?;
        let capability = solver_capability(request.solver);
        if !admission.allows_capability(&capability) {
            return Err(FirstInstructionDeploymentError::CapabilityNotAdmitted);
        }
        admission
            .recheck_currentness(admission_currentness)
            .map_err(FirstInstructionDeploymentError::PreExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(FirstInstructionDeploymentError::PreExecutionWorkerCurrentness)?;

        cgroup.verify_configuration()?;
        if cgroup.limits() != self.resource_policy.cgroup {
            return Err(FirstInstructionDeploymentError::CgroupPolicyMismatch);
        }
        if cgroup.populated()? {
            return Err(FirstInstructionDeploymentError::CgroupAlreadyPopulated);
        }
        let cgroup_evidence_before = cgroup.evidence()?;
        cgroup_evidence_before.verify()?;

        let technical = execute_first_instruction_worker(
            &self.worker_image,
            cgroup,
            &self.manifest_bytes,
            &self.component_bytes,
            request,
            self.base_evidence.frame.into(),
            self.resource_policy.worker,
        )?;

        admission
            .recheck_currentness(admission_currentness)
            .map_err(FirstInstructionDeploymentError::PostExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(FirstInstructionDeploymentError::PostExecutionWorkerCurrentness)?;

        let technical_evidence = technical.evidence().clone();
        technical_evidence.verify()?;
        let cgroup_evidence_after = cgroup.evidence()?;
        cgroup_evidence_after.verify()?;
        if cgroup_evidence_before != cgroup_evidence_after
            || technical_evidence.launch.target != cgroup_evidence_after
            || technical_evidence.launch.image_sha256 != hex_digest(self.worker_image_sha256)
            || technical_evidence.extension_id != self.manifest.id.as_str()
            || technical_evidence.extension_version != self.manifest.version
            || technical_evidence.containment_profiles_established_by_this_layer
            || technical_evidence.expected_process_containment_profile
                != WORKER_CONTAINMENT_PROFILE_V1
            || technical_evidence.expected_filesystem_containment_profile
                != WORKER_FILESYSTEM_PROFILE_V1
            || worker_qualification.worker_sha256()
                != parse_hex_32(&technical_evidence.launch.image_sha256)?
        {
            return Err(FirstInstructionDeploymentError::ExecutionEvidenceMismatch);
        }

        let deployment_evidence = self.evidence();
        deployment_evidence.verify()?;
        let request_sha256 = technical_evidence.request_sha256.clone();
        let output_sha256 = technical_evidence.output_sha256.clone();
        let execution_sha256 = execution_sha256_v1(
            self.binding_sha256,
            parse_hex_32(&technical_evidence.evidence_sha256)?,
            parse_hex_32(&cgroup_evidence_after.evidence_sha256)?,
            parse_hex_32(&request_sha256)?,
            parse_hex_32(&output_sha256)?,
        );
        let evidence = FirstInstructionDeploymentExecutionEvidence {
            deployment: deployment_evidence,
            worker: technical_evidence,
            cgroup: cgroup_evidence_after,
            request_sha256,
            output_sha256,
            execution_sha256: hex_digest(execution_sha256),
        };
        evidence.verify()?;

        Ok(FirstInstructionDeploymentInvocation {
            technical,
            evidence,
            binding_sha256: self.binding_sha256,
            base_deployment_sha256: self.base_deployment_sha256,
            worker_qualification_evidence_sha256: self.worker_qualification_evidence_sha256,
        })
    }
}

#[derive(Debug)]
pub struct FirstInstructionDeploymentInvocation {
    technical: FirstInstructionWorkerInvocation,
    evidence: FirstInstructionDeploymentExecutionEvidence,
    binding_sha256: [u8; 32],
    base_deployment_sha256: [u8; 32],
    worker_qualification_evidence_sha256: [u8; 32],
}

impl FirstInstructionDeploymentInvocation {
    pub fn result(&self) -> &SimulationResult {
        self.technical.result()
    }

    pub fn worker_invocation(&self) -> &FirstInstructionWorkerInvocation {
        &self.technical
    }

    pub fn evidence(&self) -> &FirstInstructionDeploymentExecutionEvidence {
        &self.evidence
    }

    pub const fn binding_sha256(&self) -> [u8; 32] {
        self.binding_sha256
    }

    pub const fn base_deployment_sha256(&self) -> [u8; 32] {
        self.base_deployment_sha256
    }

    pub const fn worker_qualification_evidence_sha256(&self) -> [u8; 32] {
        self.worker_qualification_evidence_sha256
    }

    pub fn into_result(self) -> SimulationResult {
        self.technical.into_result()
    }
}

#[derive(Debug, Error)]
pub enum FirstInstructionDeploymentError {
    #[error("first-instruction deployment issuance admission currentness failed: {0:?}")]
    IssuanceAdmissionCurrentness(AdmissionProblem),
    #[error("first-instruction deployment issuance worker qualification currentness failed: {0}")]
    IssuanceWorkerCurrentness(WorkerQualificationError),
    #[error("first-instruction deployment pre-execution admission currentness failed: {0:?}")]
    PreExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("first-instruction deployment pre-execution worker qualification currentness failed: {0}")]
    PreExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("first-instruction deployment post-execution admission currentness failed: {0:?}")]
    PostExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("first-instruction deployment post-execution worker qualification currentness failed: {0}")]
    PostExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("manifest json is invalid: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("first-instruction deployment requires a simulation Wasm manifest")]
    UnexpectedManifest,
    #[error("first-instruction package identity does not match base deployment/admission")]
    PackageIdentityMismatch,
    #[error("first-instruction package digest does not match base deployment/admission")]
    PackageDigestMismatch,
    #[error("first-instruction worker qualification/image does not match base deployment")]
    WorkerQualificationMismatch,
    #[error("base deployment commitments do not match")]
    BaseDeploymentMismatch,
    #[error("first-instruction deployment binding no longer matches live authorities/material")]
    BindingMismatch,
    #[error("first-instruction deployment watchdog/output envelope must exactly match deployment-v1")]
    WatchdogEnvelopeMismatch,
    #[error("first-instruction request capability is not admitted")]
    CapabilityNotAdmitted,
    #[error("live cgroup limits do not match bound first-instruction resource policy")]
    CgroupPolicyMismatch,
    #[error("first-instruction execution requires a fresh unpopulated cgroup leaf")]
    CgroupAlreadyPopulated,
    #[error("first-instruction execution evidence does not match bound authority/material")]
    ExecutionEvidenceMismatch,
    #[error("persisted first-instruction deployment evidence is structurally invalid")]
    InvalidEvidence,
    #[error("nested deployment-v1 evidence digest mismatch")]
    BaseDeploymentDigestMismatch,
    #[error("first-instruction deployment binding digest mismatch")]
    BindingDigestMismatch,
    #[error("first-instruction execution digest mismatch")]
    ExecutionDigestMismatch,
    #[error("deployment package {0} exceeds the deployment-v1 frame envelope")]
    PackageExceedsFrame(&'static str),
    #[error("string length cannot be represented by deployment-v1 commitment")]
    LengthOverflow,
    #[error("hex digest is invalid")]
    InvalidHexDigest,
    #[error("nested deployment supervisor envelope is invalid: {0}")]
    BaseSupervisor(symthaea_sim_worker::SupervisorError),
    #[error(transparent)]
    BaseDeployment(#[from] SimulationDeploymentError),
    #[error(transparent)]
    Cgroup(#[from] CgroupV2Error),
    #[error(transparent)]
    Worker(#[from] FirstInstructionWorkerError),
}

fn verify_base_deployment_evidence(
    evidence: &SimulationDeploymentEvidence,
) -> Result<(), FirstInstructionDeploymentError> {
    if evidence.profile != SIMULATION_DEPLOYMENT_PROFILE_V1
        || evidence.transport != SIMULATION_DEPLOYMENT_TRANSPORT_V1
        || evidence.package_runtime != RuntimeKind::Wasm
        || evidence.selected_extension.is_empty()
        || evidence.selected_version.is_empty()
        || evidence.admission_generation == 0
        || evidence.trust_generation == 0
        || evidence.worker_qualification_profile != WORKER_QUALIFICATION_PROFILE_V1
        || evidence.worker_qualification_generation == 0
        || evidence.worker_image_size_bytes == 0
        || evidence.worker_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
        || evidence.supervisor_profile != SUPERVISOR_PROFILE_V1
        || evidence.worker_protocol != WORKER_PROTOCOL_V1
        || evidence.process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
        || evidence.filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
    {
        return Err(FirstInstructionDeploymentError::InvalidEvidence);
    }
    let supervisor: SupervisorLimits = evidence.supervisor.into();
    supervisor
        .validate()
        .map_err(FirstInstructionDeploymentError::BaseSupervisor)?;
    let frame: WorkerFrameLimits = evidence.frame.into();
    if frame.max_manifest_bytes == 0
        || frame.max_component_bytes == 0
        || frame.max_request_json_bytes == 0
        || frame.max_response_json_bytes == 0
    {
        return Err(FirstInstructionDeploymentError::InvalidEvidence);
    }

    let manifest = parse_hex_32(&evidence.manifest_sha256)?;
    let payload = parse_hex_32(&evidence.payload_sha256)?;
    let policy = parse_hex_32(&evidence.policy_sha256)?;
    let worker_evidence = parse_hex_32(&evidence.worker_qualification_evidence_sha256)?;
    let worker_image = parse_hex_32(&evidence.worker_image_sha256)?;
    let expected = base_deployment_sha256_v1(
        evidence,
        manifest,
        payload,
        policy,
        worker_evidence,
        worker_image,
    )?;
    if evidence.deployment_sha256 != hex_digest(expected) {
        return Err(FirstInstructionDeploymentError::BaseDeploymentDigestMismatch);
    }
    Ok(())
}

fn validate_frame_against_package(
    frame: DeploymentFrameEnvelope,
    manifest: &[u8],
    component: &[u8],
) -> Result<(), FirstInstructionDeploymentError> {
    if manifest.len() as u128 > frame.max_manifest_bytes as u128 {
        return Err(FirstInstructionDeploymentError::PackageExceedsFrame("manifest"));
    }
    if component.len() as u128 > frame.max_component_bytes as u128 {
        return Err(FirstInstructionDeploymentError::PackageExceedsFrame("component"));
    }
    let limits: WorkerFrameLimits = frame.into();
    if limits.max_manifest_bytes == 0
        || limits.max_component_bytes == 0
        || limits.max_request_json_bytes == 0
        || limits.max_response_json_bytes == 0
    {
        return Err(FirstInstructionDeploymentError::InvalidEvidence);
    }
    Ok(())
}

fn base_deployment_sha256_v1(
    evidence: &SimulationDeploymentEvidence,
    manifest_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
) -> Result<[u8; 32], FirstInstructionDeploymentError> {
    let mut hasher = Sha256::new();
    hasher.update(BASE_DEPLOYMENT_DOMAIN_V1);
    put_string(&mut hasher, SIMULATION_DEPLOYMENT_TRANSPORT_V1)?;
    put_string(&mut hasher, &evidence.selected_extension)?;
    put_string(&mut hasher, &evidence.selected_version)?;
    hasher.update([1]);
    hasher.update(evidence.admission_generation.to_le_bytes());
    hasher.update(evidence.trust_generation.to_le_bytes());
    hasher.update(manifest_sha256);
    hasher.update(payload_sha256);
    hasher.update(policy_sha256);
    put_string(&mut hasher, WORKER_QUALIFICATION_PROFILE_V1)?;
    hasher.update(evidence.worker_qualification_generation.to_le_bytes());
    hasher.update(worker_qualification_evidence_sha256);
    hasher.update(worker_image_sha256);
    hasher.update(evidence.worker_image_size_bytes.to_le_bytes());
    hasher.update(evidence.worker_image_seal_mask.to_le_bytes());
    put_string(&mut hasher, SEALED_WORKER_IMAGE_PROFILE_V1)?;
    put_string(&mut hasher, SUPERVISOR_PROFILE_V1)?;
    put_string(&mut hasher, WORKER_PROTOCOL_V1)?;
    put_string(&mut hasher, WORKER_CONTAINMENT_PROFILE_V1)?;
    put_string(&mut hasher, WORKER_FILESYSTEM_PROFILE_V1)?;
    hasher.update(evidence.supervisor.address_space_bytes.to_le_bytes());
    hasher.update(evidence.supervisor.cpu_seconds.to_le_bytes());
    hasher.update(evidence.supervisor.file_size_bytes.to_le_bytes());
    hasher.update(evidence.supervisor.open_files.to_le_bytes());
    hasher.update(evidence.supervisor.process_count.to_le_bytes());
    hasher.update(evidence.supervisor.wall_time_ms.to_le_bytes());
    hasher.update(evidence.supervisor.max_stdout_bytes.to_le_bytes());
    hasher.update(evidence.supervisor.max_stderr_bytes.to_le_bytes());
    hasher.update(evidence.frame.max_manifest_bytes.to_le_bytes());
    hasher.update(evidence.frame.max_component_bytes.to_le_bytes());
    hasher.update(evidence.frame.max_request_json_bytes.to_le_bytes());
    hasher.update(evidence.frame.max_response_json_bytes.to_le_bytes());
    Ok(hasher.finalize().into())
}

fn binding_sha256_v1(
    base_deployment_sha256: [u8; 32],
    resource_policy: FirstInstructionResourcePolicy,
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(BINDING_DOMAIN_V1);
    hasher.update(base_deployment_sha256);
    hasher.update(resource_policy.cgroup.memory_max_bytes.to_le_bytes());
    hasher.update(resource_policy.cgroup.pids_max.to_le_bytes());
    hasher.update(resource_policy.cgroup.cpu_quota_us.to_le_bytes());
    hasher.update(resource_policy.cgroup.cpu_period_us.to_le_bytes());
    hasher.update(resource_policy.worker.wall_time_ms.to_le_bytes());
    hasher.update(resource_policy.worker.max_stdout_bytes.to_le_bytes());
    hasher.update(resource_policy.worker.max_stderr_bytes.to_le_bytes());
    hasher.update(worker_qualification_generation.to_le_bytes());
    hasher.update(worker_qualification_evidence_sha256);
    hasher.update(worker_image_sha256);
    hasher.update([1]);
    hasher.update([0]);
    hasher.update([1]);
    hasher.finalize().into()
}

fn execution_sha256_v1(
    deployment_binding: [u8; 32],
    worker_evidence: [u8; 32],
    cgroup_evidence: [u8; 32],
    request_sha256: [u8; 32],
    output_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(EXECUTION_DOMAIN_V1);
    hasher.update(deployment_binding);
    hasher.update(worker_evidence);
    hasher.update(cgroup_evidence);
    hasher.update(request_sha256);
    hasher.update(output_sha256);
    hasher.finalize().into()
}

fn put_string(
    hasher: &mut Sha256,
    value: &str,
) -> Result<(), FirstInstructionDeploymentError> {
    let len = u64::try_from(value.len()).map_err(|_| FirstInstructionDeploymentError::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], FirstInstructionDeploymentError> {
    if value.len() != 64 {
        return Err(FirstInstructionDeploymentError::InvalidHexDigest);
    }
    let bytes = value.as_bytes();
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, FirstInstructionDeploymentError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(FirstInstructionDeploymentError::InvalidHexDigest),
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
