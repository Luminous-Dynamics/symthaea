// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic provider routing composed with exact first-instruction cgroup
//! deployment authority.
//!
//! Routing still selects the admitted `runtime = wasm` package. This crate does
//! not rewrite execution topology into `RuntimeKind::Remote`. It binds the
//! selected routing decision and canonical request digest to one exact base
//! deployment, one exact first-instruction deployment, and one live worker
//! qualification.

#![deny(unsafe_code)]

use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem,
};
use symthaea_extension_core::{ExtensionId, RuntimeKind};
use symthaea_extension_registry::ExtensionRegistry;
use symthaea_extension_router::{
    ExtensionRouter, ProviderObservation, RoutingConstraints, RoutingDecision, RoutingError,
    RoutingRequest,
};
use symthaea_sim_bridge::SimulationRequest;
use symthaea_sim_deployment::{BoundSimulationDeployment, SimulationDeploymentError};
use symthaea_sim_digest::{CanonicalDigestError, canonical_request_sha256_v1};
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_first_instruction_deployment::{
    BoundFirstInstructionDeployment, FirstInstructionDeploymentError,
    FirstInstructionDeploymentInvocation,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct FirstInstructionExecutionAuthority<'a> {
    base_deployment: &'a BoundSimulationDeployment,
    deployment: &'a BoundFirstInstructionDeployment,
    worker_qualification: &'a ActiveWorkerQualification,
}

impl<'a> FirstInstructionExecutionAuthority<'a> {
    pub fn new(
        base_deployment: &'a BoundSimulationDeployment,
        deployment: &'a BoundFirstInstructionDeployment,
        worker_qualification: &'a ActiveWorkerQualification,
    ) -> Self {
        Self {
            base_deployment,
            deployment,
            worker_qualification,
        }
    }

    pub fn base_deployment(&self) -> &'a BoundSimulationDeployment {
        self.base_deployment
    }

    pub fn deployment(&self) -> &'a BoundFirstInstructionDeployment {
        self.deployment
    }

    pub fn worker_qualification(&self) -> &'a ActiveWorkerQualification {
        self.worker_qualification
    }
}

#[derive(Debug)]
pub struct RoutedFirstInstructionSelection<'a> {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    admission: &'a ActiveAdmission,
    authority: FirstInstructionExecutionAuthority<'a>,
}

impl<'a> RoutedFirstInstructionSelection<'a> {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub fn deployment(&self) -> &'a BoundFirstInstructionDeployment {
        self.authority.deployment
    }

    pub fn base_deployment(&self) -> &'a BoundSimulationDeployment {
        self.authority.base_deployment
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        request: &SimulationRequest,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        cgroup: &CgroupV2Lease,
    ) -> Result<RoutedFirstInstructionInvocation, RoutedFirstInstructionError> {
        let actual_request_sha256 = canonical_request_sha256_v1(request)?;
        if actual_request_sha256 != self.request_sha256 {
            return Err(RoutedFirstInstructionError::RequestSubstitution);
        }
        if solver_capability(request.solver) != self.decision.capability {
            return Err(RoutedFirstInstructionError::CapabilitySubstitution);
        }

        require_decision_matches_admission(&self.decision, self.admission)?;
        verify_authority(self.admission, self.authority)?;

        let invocation = self.authority.deployment.execute(
            self.authority.base_deployment,
            self.admission,
            admission_currentness,
            self.authority.worker_qualification,
            worker_currentness,
            cgroup,
            request,
        )?;

        require_decision_matches_admission(&self.decision, self.admission)?;
        if invocation.base_deployment_sha256()
            != self.authority.base_deployment.deployment_sha256()
            || invocation.binding_sha256() != self.authority.deployment.binding_sha256()
            || invocation.worker_qualification_evidence_sha256()
                != self
                    .authority
                    .worker_qualification
                    .qualification_evidence_sha256()
        {
            return Err(RoutedFirstInstructionError::DeploymentCommitmentMismatch);
        }

        Ok(RoutedFirstInstructionInvocation {
            decision: self.decision.clone(),
            request_sha256: self.request_sha256,
            invocation,
        })
    }
}

#[derive(Debug)]
pub struct RoutedFirstInstructionInvocation {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    invocation: FirstInstructionDeploymentInvocation,
}

impl RoutedFirstInstructionInvocation {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub fn invocation(&self) -> &FirstInstructionDeploymentInvocation {
        &self.invocation
    }
}

#[derive(Debug, Error)]
pub enum RoutedFirstInstructionError {
    #[error("simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Canonical(#[from] CanonicalDigestError),
    #[error("routing failed: {0:?}")]
    Routing(RoutingError),
    #[error("router selected provider missing from registry: {0:?}")]
    SelectedManifestMissing(ExtensionId),
    #[error("first-instruction deployment routing requires selected manifest runtime = wasm")]
    SelectedRuntimeNotWasm,
    #[error("selected admission is missing or duplicated")]
    SelectedAdmissionMissingOrDuplicate,
    #[error("selected admission commitments do not equal routing decision")]
    SelectedAdmissionMismatch,
    #[error("selected admission currentness failed after routing: {0:?}")]
    SelectedAdmissionCurrentness(AdmissionProblem),
    #[error("selected first-instruction deployment authority is missing or duplicated")]
    SelectedDeploymentMissingOrDuplicate,
    #[error("selected worker qualification currentness failed after routing: {0}")]
    SelectedWorkerCurrentness(WorkerQualificationError),
    #[error("selected first-instruction deployment does not match its base deployment or live authorities")]
    SelectedDeploymentMismatch,
    #[error("route selected for one canonical request was used for a different request")]
    RequestSubstitution,
    #[error("route capability no longer matches request solver")]
    CapabilitySubstitution,
    #[error("executed first-instruction deployment commitment no longer matches selected authority")]
    DeploymentCommitmentMismatch,
    #[error(transparent)]
    BaseDeployment(#[from] SimulationDeploymentError),
    #[error(transparent)]
    FirstInstructionDeployment(#[from] FirstInstructionDeploymentError),
}

impl From<RoutingError> for RoutedFirstInstructionError {
    fn from(value: RoutingError) -> Self {
        Self::Routing(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn select_routed_first_instruction<'a>(
    registry: &ExtensionRegistry,
    request: &SimulationRequest,
    constraints: RoutingConstraints,
    admissions: &'a [ActiveAdmission],
    observations: &[ProviderObservation],
    authorities: &'a [FirstInstructionExecutionAuthority<'a>],
    admission_currentness: &dyn AdmissionCurrentnessSource,
    worker_currentness: &dyn WorkerQualificationCurrentnessSource,
) -> Result<RoutedFirstInstructionSelection<'a>, RoutedFirstInstructionError> {
    request
        .validate()
        .map_err(|error| RoutedFirstInstructionError::InvalidRequest(error.to_string()))?;
    let request_sha256 = canonical_request_sha256_v1(request)?;
    let capability = solver_capability(request.solver);
    let routing_request = RoutingRequest {
        capability: capability.clone(),
        constraints,
    };
    let decision = ExtensionRouter::route(registry, &routing_request, admissions, observations)?;
    if decision.capability != capability {
        return Err(RoutedFirstInstructionError::CapabilitySubstitution);
    }

    let manifest = registry
        .get(&decision.selected)
        .ok_or_else(|| RoutedFirstInstructionError::SelectedManifestMissing(decision.selected.clone()))?;
    if manifest.runtime != RuntimeKind::Wasm {
        return Err(RoutedFirstInstructionError::SelectedRuntimeNotWasm);
    }

    let admission = exact_selected_admission(&decision, admissions)?;
    if !admission.matches_manifest(manifest) {
        return Err(RoutedFirstInstructionError::SelectedAdmissionMismatch);
    }
    admission
        .recheck_currentness(admission_currentness)
        .map_err(RoutedFirstInstructionError::SelectedAdmissionCurrentness)?;

    let authority = exact_selected_authority(&decision, authorities)?;
    authority
        .worker_qualification
        .recheck_currentness(worker_currentness)
        .map_err(RoutedFirstInstructionError::SelectedWorkerCurrentness)?;
    verify_authority(admission, authority)?;

    Ok(RoutedFirstInstructionSelection {
        decision,
        request_sha256,
        admission,
        authority,
    })
}

fn exact_selected_admission<'a>(
    decision: &RoutingDecision,
    admissions: &'a [ActiveAdmission],
) -> Result<&'a ActiveAdmission, RoutedFirstInstructionError> {
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or(RoutedFirstInstructionError::SelectedAdmissionMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedFirstInstructionError::SelectedAdmissionMissingOrDuplicate);
    }
    require_decision_matches_admission(decision, admission)?;
    Ok(admission)
}

fn require_decision_matches_admission(
    decision: &RoutingDecision,
    admission: &ActiveAdmission,
) -> Result<(), RoutedFirstInstructionError> {
    if admission.extension() != &decision.selected
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
        || !admission.allows_capability(&decision.capability)
    {
        return Err(RoutedFirstInstructionError::SelectedAdmissionMismatch);
    }
    Ok(())
}

fn exact_selected_authority<'a>(
    decision: &RoutingDecision,
    authorities: &'a [FirstInstructionExecutionAuthority<'a>],
) -> Result<FirstInstructionExecutionAuthority<'a>, RoutedFirstInstructionError> {
    let mut matches = authorities.iter().copied().filter(|authority| {
        authority.base_deployment.selected_extension() == decision.selected.as_str()
    });
    let authority = matches
        .next()
        .ok_or(RoutedFirstInstructionError::SelectedDeploymentMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedFirstInstructionError::SelectedDeploymentMissingOrDuplicate);
    }
    Ok(authority)
}

fn verify_authority(
    admission: &ActiveAdmission,
    authority: FirstInstructionExecutionAuthority<'_>,
) -> Result<(), RoutedFirstInstructionError> {
    authority
        .base_deployment
        .verify(admission, authority.worker_qualification)?;
    authority.deployment.verify(
        authority.base_deployment,
        admission,
        authority.worker_qualification,
    )?;
    if authority.deployment.base_deployment_sha256()
        != authority.base_deployment.deployment_sha256()
        || authority.base_deployment.worker_image_sha256()
            != authority.worker_qualification.worker_sha256()
    {
        return Err(RoutedFirstInstructionError::SelectedDeploymentMismatch);
    }
    Ok(())
}
