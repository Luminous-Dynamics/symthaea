// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authority-preserving deployment binding for the frozen public/community
//! simulation-worker execution profile.
//!
//! `BoundSimulationDeployment` remains the package + qualified-worker waist.
//! This layer accepts only a base deployment already issued with the exact
//! public frame/supervisor envelopes, binds the frozen public execution profile,
//! executes only through that profile, and brackets execution with both package
//! admission and worker-qualification currentness checks.
//!
//! Persisted evidence is audit-only. Deserializing it cannot recreate a live
//! admission, worker qualification, sealed image, cgroup lease or deployment.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem,
};
use symthaea_extension_core::{ExtensionKind, ExtensionManifest, RuntimeKind};
use symthaea_sim_bridge::{SimulationEvidence, SimulationRequest, SimulationResult};
use symthaea_sim_deployment::{
    BoundSimulationDeployment, DeploymentFrameEnvelope, DeploymentSupervisorEnvelope,
    SimulationDeploymentError, SimulationDeploymentEvidence, SIMULATION_DEPLOYMENT_PROFILE_V1,
    SIMULATION_DEPLOYMENT_TRANSPORT_V1,
};
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_public_worker_execution::{
    execute_public_worker_v1, public_cgroup_limits_v1, public_supervisor_limits_v1,
    PublicWorkerExecutionError, PublicWorkerExecutionEvidence, PublicWorkerExecutionInvocation,
    PUBLIC_WORKER_EXECUTION_PROFILE_V1,
};
use symthaea_sim_public_worker_input::{
    public_worker_frame_limits_v1, PUBLIC_WORKER_INPUT_PROFILE_V1,
};
use symthaea_sim_worker::{
    SupervisorLimits, WorkerFrameLimits, SUPERVISOR_PROFILE_V1, WORKER_PROTOCOL_V1,
};
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Evidence, CgroupV2Lease};
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::{SealedWorkerImage, SEALED_WORKER_IMAGE_PROFILE_V1};
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
    WORKER_QUALIFICATION_PROFILE_V1,
};
use thiserror::Error;

pub const PUBLIC_WORKER_DEPLOYMENT_PROFILE_V1: &str =
    "symthaea.simulation.deployment.public-worker-execution.v1";
const BINDING_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.deployment.public-worker-execution.v1\0";
const EXECUTION_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.deployment.public-worker-execution.execution.v1\0";
const BASE_DEPLOYMENT_DOMAIN_V1: &[u8] = b"symthaea.simulation.deployment.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWorkerDeploymentEvidence {
    pub profile: String,
    pub base_deployment: SimulationDeploymentEvidence,
    pub public_execution_profile: String,
    pub public_input_profile: String,
    pub worker_qualification_profile: String,
    pub worker_qualification_generation: u64,
    pub worker_qualification_evidence_sha256: String,
    pub exact_worker_image_sha256: String,
    pub exact_worker_image_profile: String,
    pub supervisor_profile: String,
    pub worker_protocol: String,
    pub required_process_containment_profile: String,
    pub required_filesystem_containment_profile: String,
    pub live_worker_qualification_scope_bound: bool,
    pub frozen_public_supervisor_envelope: bool,
    pub frozen_public_frame_envelope: bool,
    pub frozen_public_cgroup_profile: bool,
    pub legacy_preexec_rlimits_applied: bool,
    pub binding_sha256: String,
}

impl PublicWorkerDeploymentEvidence {
    pub fn verify(&self) -> Result<(), PublicWorkerDeploymentError> {
        if self.profile != PUBLIC_WORKER_DEPLOYMENT_PROFILE_V1
            || self.public_execution_profile != PUBLIC_WORKER_EXECUTION_PROFILE_V1
            || self.public_input_profile != PUBLIC_WORKER_INPUT_PROFILE_V1
            || self.worker_qualification_profile != WORKER_QUALIFICATION_PROFILE_V1
            || self.exact_worker_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
            || self.supervisor_profile != SUPERVISOR_PROFILE_V1
            || self.worker_protocol != WORKER_PROTOCOL_V1
            || self.required_process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
            || self.required_filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
            || !self.live_worker_qualification_scope_bound
            || !self.frozen_public_supervisor_envelope
            || !self.frozen_public_frame_envelope
            || !self.frozen_public_cgroup_profile
            || !self.legacy_preexec_rlimits_applied
            || self.worker_qualification_generation == 0
        {
            return Err(PublicWorkerDeploymentError::InvalidEvidence);
        }
        verify_base_deployment_evidence(&self.base_deployment)?;
        require_public_base_envelopes(&self.base_deployment)?;
        if self.base_deployment.worker_qualification_generation
            != self.worker_qualification_generation
            || self.base_deployment.worker_qualification_evidence_sha256
                != self.worker_qualification_evidence_sha256
            || self.base_deployment.worker_image_sha256 != self.exact_worker_image_sha256
        {
            return Err(PublicWorkerDeploymentError::BaseDeploymentMismatch);
        }

        let base = parse_hex_32(&self.base_deployment.deployment_sha256)?;
        let worker_evidence = parse_hex_32(&self.worker_qualification_evidence_sha256)?;
        let worker_image = parse_hex_32(&self.exact_worker_image_sha256)?;
        let expected = binding_sha256_v1(
            base,
            self.worker_qualification_generation,
            worker_evidence,
            worker_image,
        );
        if self.binding_sha256 != hex_digest(expected) {
            return Err(PublicWorkerDeploymentError::BindingDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicWorkerDeploymentExecutionEvidence {
    pub deployment: PublicWorkerDeploymentEvidence,
    pub technical: PublicWorkerExecutionEvidence,
    pub cgroup: CgroupV2Evidence,
    pub request_sha256: String,
    pub output_sha256: String,
    pub execution_sha256: String,
}

impl PublicWorkerDeploymentExecutionEvidence {
    pub fn verify(&self) -> Result<(), PublicWorkerDeploymentError> {
        self.deployment.verify()?;
        self.technical.verify()?;
        self.cgroup.verify()?;
        if self.technical.profile != PUBLIC_WORKER_EXECUTION_PROFILE_V1 {
            return Err(PublicWorkerDeploymentError::ExecutionEvidenceMismatch);
        }
        let worker = &self.technical.inner.inner;
        if worker.request_sha256 != self.request_sha256
            || worker.output_sha256 != self.output_sha256
            || worker.launch.base.target != self.cgroup
            || worker.launch.base.image_sha256 != self.deployment.exact_worker_image_sha256
            || worker.expected_process_containment_profile
                != self.deployment.required_process_containment_profile
            || worker.expected_filesystem_containment_profile
                != self.deployment.required_filesystem_containment_profile
            || worker.containment_profiles_established_by_this_layer
            || !worker.legacy_preexec_rlimits_applied
        {
            return Err(PublicWorkerDeploymentError::ExecutionEvidenceMismatch);
        }
        let deployment = parse_hex_32(&self.deployment.binding_sha256)?;
        let technical = parse_hex_32(&self.technical.evidence_sha256)?;
        let cgroup = parse_hex_32(&self.cgroup.evidence_sha256)?;
        let request = parse_hex_32(&self.request_sha256)?;
        let output = parse_hex_32(&self.output_sha256)?;
        let expected = execution_sha256_v1(deployment, technical, cgroup, request, output);
        if self.execution_sha256 != hex_digest(expected) {
            return Err(PublicWorkerDeploymentError::ExecutionDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct BoundPublicWorkerDeployment {
    base_deployment_sha256: [u8; 32],
    base_evidence: SimulationDeploymentEvidence,
    manifest: ExtensionManifest,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
    worker_image: SealedWorkerImage,
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
    binding_sha256: [u8; 32],
}

impl BoundPublicWorkerDeployment {
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
    ) -> Result<Self, PublicWorkerDeploymentError> {
        base.verify(admission, worker_qualification)?;
        admission
            .recheck_currentness(admission_currentness)
            .map_err(PublicWorkerDeploymentError::IssuanceAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(PublicWorkerDeploymentError::IssuanceWorkerCurrentness)?;

        let base_evidence = base.evidence();
        verify_base_deployment_evidence(&base_evidence)?;
        require_public_base_envelopes(&base_evidence)?;
        let base_deployment_sha256 = parse_hex_32(&base_evidence.deployment_sha256)?;
        if base_deployment_sha256 != base.deployment_sha256() {
            return Err(PublicWorkerDeploymentError::BaseDeploymentMismatch);
        }

        let manifest_bytes = manifest_bytes.into();
        let component_bytes = component_bytes.into();
        validate_public_package_sizes(&manifest_bytes, &component_bytes)?;
        let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(PublicWorkerDeploymentError::ManifestJson)?;
        manifest
            .validate()
            .map_err(|problems| PublicWorkerDeploymentError::ManifestInvalid(format!("{problems:?}")))?;
        if manifest.kind != ExtensionKind::Simulation || manifest.runtime != RuntimeKind::Wasm {
            return Err(PublicWorkerDeploymentError::UnexpectedManifest);
        }
        if !admission.matches_manifest(&manifest)
            || admission.extension() != &manifest.id
            || admission.subject().extension_version() != manifest.version
            || base.selected_extension() != manifest.id.as_str()
            || base.selected_version() != manifest.version
        {
            return Err(PublicWorkerDeploymentError::PackageIdentityMismatch);
        }

        let manifest_sha256: [u8; 32] = Sha256::digest(manifest_bytes.as_ref()).into();
        let component_sha256: [u8; 32] = Sha256::digest(component_bytes.as_ref()).into();
        if base_evidence.manifest_sha256 != hex_digest(manifest_sha256)
            || base_evidence.payload_sha256 != hex_digest(component_sha256)
            || admission.manifest_sha256().0 != manifest_sha256
            || admission.payload_sha256().0 != component_sha256
        {
            return Err(PublicWorkerDeploymentError::PackageDigestMismatch);
        }

        let worker_image_sha256 = worker_image.image_sha256();
        let worker_qualification_evidence_sha256 =
            worker_qualification.qualification_evidence_sha256();
        verify_live_worker_qualification_scope(
            worker_qualification,
            worker_image_sha256,
            worker_qualification_evidence_sha256,
            worker_qualification.generation(),
        )?;
        if base.worker_image_sha256() != worker_image_sha256
            || base_evidence.worker_image_sha256 != hex_digest(worker_image_sha256)
            || base_evidence.worker_image_size_bytes != worker_image.image_size_bytes()
            || base_evidence.worker_image_seal_mask != worker_image.seal_mask()
            || !worker_qualification.matches_image(&worker_image)
            || base_evidence.worker_qualification_generation != worker_qualification.generation()
            || base_evidence.worker_qualification_evidence_sha256
                != hex_digest(worker_qualification_evidence_sha256)
        {
            return Err(PublicWorkerDeploymentError::WorkerQualificationMismatch);
        }

        let binding_sha256 = binding_sha256_v1(
            base_deployment_sha256,
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
            worker_qualification_generation: worker_qualification.generation(),
            worker_qualification_evidence_sha256,
            worker_image_sha256,
            binding_sha256,
        })
    }

    pub fn evidence(&self) -> PublicWorkerDeploymentEvidence {
        PublicWorkerDeploymentEvidence {
            profile: PUBLIC_WORKER_DEPLOYMENT_PROFILE_V1.into(),
            base_deployment: self.base_evidence.clone(),
            public_execution_profile: PUBLIC_WORKER_EXECUTION_PROFILE_V1.into(),
            public_input_profile: PUBLIC_WORKER_INPUT_PROFILE_V1.into(),
            worker_qualification_profile: WORKER_QUALIFICATION_PROFILE_V1.into(),
            worker_qualification_generation: self.worker_qualification_generation,
            worker_qualification_evidence_sha256: hex_digest(
                self.worker_qualification_evidence_sha256,
            ),
            exact_worker_image_sha256: hex_digest(self.worker_image_sha256),
            exact_worker_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            supervisor_profile: SUPERVISOR_PROFILE_V1.into(),
            worker_protocol: WORKER_PROTOCOL_V1.into(),
            required_process_containment_profile: WORKER_CONTAINMENT_PROFILE_V1.into(),
            required_filesystem_containment_profile: WORKER_FILESYSTEM_PROFILE_V1.into(),
            live_worker_qualification_scope_bound: true,
            frozen_public_supervisor_envelope: true,
            frozen_public_frame_envelope: true,
            frozen_public_cgroup_profile: true,
            legacy_preexec_rlimits_applied: true,
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
    ) -> Result<(), PublicWorkerDeploymentError> {
        base.verify(admission, worker_qualification)?;
        require_public_base_envelopes(&base.evidence())?;
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
            return Err(PublicWorkerDeploymentError::BindingMismatch);
        }
        verify_live_worker_qualification_scope(
            worker_qualification,
            self.worker_image_sha256,
            self.worker_qualification_evidence_sha256,
            self.worker_qualification_generation,
        )?;
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
    ) -> Result<PublicWorkerDeploymentInvocation, PublicWorkerDeploymentError> {
        self.verify(base, admission, worker_qualification)?;
        let capability = solver_capability(request.solver);
        if !admission.allows_capability(&capability) {
            return Err(PublicWorkerDeploymentError::CapabilityNotAdmitted);
        }
        admission
            .recheck_currentness(admission_currentness)
            .map_err(PublicWorkerDeploymentError::PreExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(PublicWorkerDeploymentError::PreExecutionWorkerCurrentness)?;

        cgroup.verify_configuration()?;
        if cgroup.limits() != public_cgroup_limits_v1() {
            return Err(PublicWorkerDeploymentError::CgroupPolicyMismatch);
        }
        if cgroup.populated()? {
            return Err(PublicWorkerDeploymentError::CgroupAlreadyPopulated);
        }
        let cgroup_before = cgroup.evidence()?;
        cgroup_before.verify()?;

        let technical = execute_public_worker_v1(
            &self.worker_image,
            cgroup,
            &self.manifest_bytes,
            &self.component_bytes,
            request,
        )?;

        admission
            .recheck_currentness(admission_currentness)
            .map_err(PublicWorkerDeploymentError::PostExecutionAdmissionCurrentness)?;
        worker_qualification
            .recheck_currentness(worker_currentness)
            .map_err(PublicWorkerDeploymentError::PostExecutionWorkerCurrentness)?;
        verify_live_worker_qualification_scope(
            worker_qualification,
            self.worker_image_sha256,
            self.worker_qualification_evidence_sha256,
            self.worker_qualification_generation,
        )?;

        let technical_evidence = technical.evidence().clone();
        technical_evidence.verify()?;
        let cgroup_after = cgroup.evidence()?;
        cgroup_after.verify()?;
        let worker = &technical_evidence.inner.inner;
        if cgroup_before != cgroup_after
            || worker.launch.base.target != cgroup_after
            || worker.launch.base.image_sha256 != hex_digest(self.worker_image_sha256)
            || worker.extension_id != self.manifest.id.as_str()
            || worker.extension_version != self.manifest.version
            || worker.containment_profiles_established_by_this_layer
            || worker.expected_process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
            || worker.expected_filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
            || !worker.legacy_preexec_rlimits_applied
            || technical.result().evidence != SimulationEvidence::default()
        {
            return Err(PublicWorkerDeploymentError::ExecutionEvidenceMismatch);
        }

        let deployment_evidence = self.evidence();
        deployment_evidence.verify()?;
        let request_sha256 = worker.request_sha256.clone();
        let output_sha256 = worker.output_sha256.clone();
        let execution_sha256 = execution_sha256_v1(
            self.binding_sha256,
            parse_hex_32(&technical_evidence.evidence_sha256)?,
            parse_hex_32(&cgroup_after.evidence_sha256)?,
            parse_hex_32(&request_sha256)?,
            parse_hex_32(&output_sha256)?,
        );
        let evidence = PublicWorkerDeploymentExecutionEvidence {
            deployment: deployment_evidence,
            technical: technical_evidence,
            cgroup: cgroup_after,
            request_sha256,
            output_sha256,
            execution_sha256: hex_digest(execution_sha256),
        };
        evidence.verify()?;

        Ok(PublicWorkerDeploymentInvocation {
            technical,
            evidence,
            binding_sha256: self.binding_sha256,
            base_deployment_sha256: self.base_deployment_sha256,
            worker_qualification_evidence_sha256: self.worker_qualification_evidence_sha256,
        })
    }
}

#[derive(Debug)]
pub struct PublicWorkerDeploymentInvocation {
    technical: PublicWorkerExecutionInvocation,
    evidence: PublicWorkerDeploymentExecutionEvidence,
    binding_sha256: [u8; 32],
    base_deployment_sha256: [u8; 32],
    worker_qualification_evidence_sha256: [u8; 32],
}

impl PublicWorkerDeploymentInvocation {
    pub fn result(&self) -> &SimulationResult {
        self.technical.result()
    }

    pub fn technical_invocation(&self) -> &PublicWorkerExecutionInvocation {
        &self.technical
    }

    pub fn evidence(&self) -> &PublicWorkerDeploymentExecutionEvidence {
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
pub enum PublicWorkerDeploymentError {
    #[error("public deployment issuance admission currentness failed: {0:?}")]
    IssuanceAdmissionCurrentness(AdmissionProblem),
    #[error("public deployment issuance worker qualification currentness failed: {0}")]
    IssuanceWorkerCurrentness(WorkerQualificationError),
    #[error("public deployment pre-execution admission currentness failed: {0:?}")]
    PreExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("public deployment pre-execution worker qualification currentness failed: {0}")]
    PreExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("public deployment post-execution admission currentness failed: {0:?}")]
    PostExecutionAdmissionCurrentness(AdmissionProblem),
    #[error("public deployment post-execution worker qualification currentness failed: {0}")]
    PostExecutionWorkerCurrentness(WorkerQualificationError),
    #[error("manifest json is invalid: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("public deployment requires a simulation Wasm manifest")]
    UnexpectedManifest,
    #[error("public deployment package identity does not match base deployment/admission")]
    PackageIdentityMismatch,
    #[error("public deployment package digest does not match base deployment/admission")]
    PackageDigestMismatch,
    #[error("public deployment worker qualification/image does not match base deployment")]
    WorkerQualificationMismatch,
    #[error("public deployment live worker qualification scope does not match execution profiles")]
    WorkerQualificationScopeMismatch,
    #[error("public deployment base commitments do not match")]
    BaseDeploymentMismatch,
    #[error("public deployment binding no longer matches live authorities/material")]
    BindingMismatch,
    #[error("base deployment must already use the exact frozen public supervisor/frame envelopes")]
    BaseEnvelopeMismatch,
    #[error("public deployment request capability is not admitted")]
    CapabilityNotAdmitted,
    #[error("live cgroup limits do not match the frozen public execution profile")]
    CgroupPolicyMismatch,
    #[error("public deployment execution requires a fresh unpopulated cgroup leaf")]
    CgroupAlreadyPopulated,
    #[error("public deployment execution evidence does not match bound authority/material")]
    ExecutionEvidenceMismatch,
    #[error("persisted public deployment evidence is structurally invalid")]
    InvalidEvidence,
    #[error("nested deployment-v1 evidence digest mismatch")]
    BaseDeploymentDigestMismatch,
    #[error("public deployment binding digest mismatch")]
    BindingDigestMismatch,
    #[error("public deployment execution digest mismatch")]
    ExecutionDigestMismatch,
    #[error("public deployment package {0} exceeds the frozen frame envelope")]
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
    Worker(#[from] PublicWorkerExecutionError),
}

fn verify_live_worker_qualification_scope(
    worker_qualification: &ActiveWorkerQualification,
    expected_worker_sha256: [u8; 32],
    expected_evidence_sha256: [u8; 32],
    expected_generation: u64,
) -> Result<(), PublicWorkerDeploymentError> {
    let scope = worker_qualification.evidence();
    if scope.profile != WORKER_QUALIFICATION_PROFILE_V1
        || scope.worker_sha256 != hex_digest(expected_worker_sha256)
        || scope.qualification_evidence_sha256 != hex_digest(expected_evidence_sha256)
        || scope.generation != expected_generation
        || scope.worker_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
        || scope.supervisor_profile != SUPERVISOR_PROFILE_V1
        || scope.worker_protocol != WORKER_PROTOCOL_V1
        || scope.process_containment_profile != WORKER_CONTAINMENT_PROFILE_V1
        || scope.filesystem_containment_profile != WORKER_FILESYSTEM_PROFILE_V1
    {
        return Err(PublicWorkerDeploymentError::WorkerQualificationScopeMismatch);
    }
    Ok(())
}

fn require_public_base_envelopes(
    evidence: &SimulationDeploymentEvidence,
) -> Result<(), PublicWorkerDeploymentError> {
    let expected_supervisor = DeploymentSupervisorEnvelope::from(public_supervisor_limits_v1());
    let expected_frame = DeploymentFrameEnvelope::from(public_worker_frame_limits_v1());
    if evidence.supervisor != expected_supervisor || evidence.frame != expected_frame {
        return Err(PublicWorkerDeploymentError::BaseEnvelopeMismatch);
    }
    Ok(())
}

fn validate_public_package_sizes(
    manifest: &[u8],
    component: &[u8],
) -> Result<(), PublicWorkerDeploymentError> {
    let frame = public_worker_frame_limits_v1();
    if manifest.len() as u128 > frame.max_manifest_bytes as u128 {
        return Err(PublicWorkerDeploymentError::PackageExceedsFrame("manifest"));
    }
    if component.len() as u128 > frame.max_component_bytes as u128 {
        return Err(PublicWorkerDeploymentError::PackageExceedsFrame("component"));
    }
    Ok(())
}

fn verify_base_deployment_evidence(
    evidence: &SimulationDeploymentEvidence,
) -> Result<(), PublicWorkerDeploymentError> {
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
        return Err(PublicWorkerDeploymentError::InvalidEvidence);
    }
    let supervisor: SupervisorLimits = evidence.supervisor.into();
    supervisor
        .validate()
        .map_err(PublicWorkerDeploymentError::BaseSupervisor)?;
    let frame: WorkerFrameLimits = evidence.frame.into();
    if frame.max_manifest_bytes == 0
        || frame.max_component_bytes == 0
        || frame.max_request_json_bytes == 0
        || frame.max_response_json_bytes == 0
    {
        return Err(PublicWorkerDeploymentError::InvalidEvidence);
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
        return Err(PublicWorkerDeploymentError::BaseDeploymentDigestMismatch);
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
) -> Result<[u8; 32], PublicWorkerDeploymentError> {
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
    worker_qualification_generation: u64,
    worker_qualification_evidence_sha256: [u8; 32],
    worker_image_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(BINDING_DOMAIN_V1);
    hasher.update(base_deployment_sha256);
    update_string(&mut hasher, PUBLIC_WORKER_EXECUTION_PROFILE_V1);
    update_string(&mut hasher, PUBLIC_WORKER_INPUT_PROFILE_V1);
    hasher.update(worker_qualification_generation.to_le_bytes());
    hasher.update(worker_qualification_evidence_sha256);
    hasher.update(worker_image_sha256);
    hasher.update([1, 1, 1, 1]);
    hasher.finalize().into()
}

fn execution_sha256_v1(
    deployment_binding: [u8; 32],
    technical_evidence: [u8; 32],
    cgroup_evidence: [u8; 32],
    request_sha256: [u8; 32],
    output_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(EXECUTION_DOMAIN_V1);
    hasher.update(deployment_binding);
    hasher.update(technical_evidence);
    hasher.update(cgroup_evidence);
    hasher.update(request_sha256);
    hasher.update(output_sha256);
    hasher.finalize().into()
}

fn put_string(
    hasher: &mut Sha256,
    value: &str,
) -> Result<(), PublicWorkerDeploymentError> {
    let len = u64::try_from(value.len()).map_err(|_| PublicWorkerDeploymentError::LengthOverflow)?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], PublicWorkerDeploymentError> {
    if value.len() != 64 {
        return Err(PublicWorkerDeploymentError::InvalidHexDigest);
    }
    let bytes = value.as_bytes();
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, PublicWorkerDeploymentError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(PublicWorkerDeploymentError::InvalidHexDigest),
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
