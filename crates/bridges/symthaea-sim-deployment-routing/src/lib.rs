// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic provider routing composed with exact contained-worker deployment.
//!
//! Routing selects package authority; deployment selects execution authority.
//! Neither layer rewrites an admitted `runtime = wasm` manifest into `remote`.
//! This crate binds the two decisions and additionally commits the canonical
//! request digest so a route selected for request A cannot execute request B.

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
use symthaea_sim_deployment::{
    BoundDeploymentInvocation, BoundSimulationDeployment, SimulationDeploymentError,
};
use symthaea_sim_digest::{CanonicalDigestError, canonical_request_sha256_v1};
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
};
use thiserror::Error;

/// Pair one exact deployment with the live worker qualification required to use
/// that deployment. Fields remain private so selection always passes through the
/// validation in this crate.
#[derive(Debug, Clone, Copy)]
pub struct DeploymentExecutionAuthority<'a> {
    deployment: &'a BoundSimulationDeployment,
    worker_qualification: &'a ActiveWorkerQualification,
}

impl<'a> DeploymentExecutionAuthority<'a> {
    pub fn new(
        deployment: &'a BoundSimulationDeployment,
        worker_qualification: &'a ActiveWorkerQualification,
    ) -> Self {
        Self {
            deployment,
            worker_qualification,
        }
    }

    pub fn deployment(&self) -> &'a BoundSimulationDeployment {
        self.deployment
    }

    pub fn worker_qualification(&self) -> &'a ActiveWorkerQualification {
        self.worker_qualification
    }
}

/// Non-serializable point-in-time route/deployment selection.
///
/// The selected admission is rechecked before this object is returned. Actual
/// execution performs additional admission and worker-qualification currentness
/// checks inside [`BoundSimulationDeployment::execute`].
#[derive(Debug)]
pub struct RoutedDeploymentSelection<'a> {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    admission: &'a ActiveAdmission,
    authority: DeploymentExecutionAuthority<'a>,
}

impl<'a> RoutedDeploymentSelection<'a> {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub fn deployment(&self) -> &'a BoundSimulationDeployment {
        self.authority.deployment
    }

    /// Execute only the exact canonical request for which this route was chosen.
    pub fn execute(
        &self,
        request: &SimulationRequest,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
    ) -> Result<RoutedDeploymentInvocation, RoutedDeploymentError> {
        let actual_request_sha256 = canonical_request_sha256_v1(request)?;
        if actual_request_sha256 != self.request_sha256 {
            return Err(RoutedDeploymentError::RequestSubstitution);
        }
        if solver_capability(request.solver) != self.decision.capability {
            return Err(RoutedDeploymentError::CapabilitySubstitution);
        }

        require_decision_matches_admission(&self.decision, self.admission)?;
        self.authority
            .deployment
            .verify(self.admission, self.authority.worker_qualification)?;

        let invocation = self.authority.deployment.execute(
            self.admission,
            admission_currentness,
            self.authority.worker_qualification,
            worker_currentness,
            request,
        )?;

        // Reassert immutable selection commitments after execution. Currentness
        // itself is rechecked by the deployment immediately before returning.
        require_decision_matches_admission(&self.decision, self.admission)?;
        if invocation.deployment_sha256() != self.authority.deployment.deployment_sha256() {
            return Err(RoutedDeploymentError::DeploymentCommitmentMismatch);
        }

        Ok(RoutedDeploymentInvocation {
            decision: self.decision.clone(),
            request_sha256: self.request_sha256,
            invocation,
        })
    }
}

#[derive(Debug)]
pub struct RoutedDeploymentInvocation {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    invocation: BoundDeploymentInvocation,
}

impl RoutedDeploymentInvocation {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub fn invocation(&self) -> &BoundDeploymentInvocation {
        &self.invocation
    }
}

#[derive(Debug, Error)]
pub enum RoutedDeploymentError {
    #[error("simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Canonical(#[from] CanonicalDigestError),
    #[error("routing failed: {0:?}")]
    Routing(RoutingError),
    #[error("router selected provider missing from registry: {0:?}")]
    SelectedManifestMissing(ExtensionId),
    #[error("contained deployment routing currently requires selected manifest runtime = wasm")]
    SelectedRuntimeNotWasm,
    #[error("selected admission is missing or duplicated")]
    SelectedAdmissionMissingOrDuplicate,
    #[error("selected admission commitments do not equal routing decision")]
    SelectedAdmissionMismatch,
    #[error("selected admission currentness failed after routing: {0:?}")]
    SelectedAdmissionCurrentness(AdmissionProblem),
    #[error("selected deployment authority is missing or duplicated")]
    SelectedDeploymentMissingOrDuplicate,
    #[error("selected worker qualification currentness failed after routing: {0}")]
    SelectedWorkerCurrentness(WorkerQualificationError),
    #[error("route selected for one canonical request was used for a different request")]
    RequestSubstitution,
    #[error("route capability no longer matches request solver")]
    CapabilitySubstitution,
    #[error("executed deployment commitment no longer matches selected deployment")]
    DeploymentCommitmentMismatch,
    #[error(transparent)]
    Deployment(#[from] SimulationDeploymentError),
}

impl From<RoutingError> for RoutedDeploymentError {
    fn from(value: RoutingError) -> Self {
        Self::Routing(value)
    }
}

/// Deterministically select one admitted provider and bind that exact route to
/// one exact deployment + worker qualification.
#[allow(clippy::too_many_arguments)]
pub fn select_routed_deployment<'a>(
    registry: &ExtensionRegistry,
    request: &SimulationRequest,
    constraints: RoutingConstraints,
    admissions: &'a [ActiveAdmission],
    observations: &[ProviderObservation],
    deployment_authorities: &'a [DeploymentExecutionAuthority<'a>],
    admission_currentness: &dyn AdmissionCurrentnessSource,
    worker_currentness: &dyn WorkerQualificationCurrentnessSource,
) -> Result<RoutedDeploymentSelection<'a>, RoutedDeploymentError> {
    request
        .validate()
        .map_err(|error| RoutedDeploymentError::InvalidRequest(error.to_string()))?;
    let request_sha256 = canonical_request_sha256_v1(request)?;
    let capability = solver_capability(request.solver);
    let routing_request = RoutingRequest {
        capability: capability.clone(),
        constraints,
    };
    let decision = ExtensionRouter::route(registry, &routing_request, admissions, observations)?;
    if decision.capability != capability {
        return Err(RoutedDeploymentError::CapabilitySubstitution);
    }

    let manifest = registry
        .get(&decision.selected)
        .ok_or_else(|| RoutedDeploymentError::SelectedManifestMissing(decision.selected.clone()))?;
    if manifest.runtime != RuntimeKind::Wasm {
        return Err(RoutedDeploymentError::SelectedRuntimeNotWasm);
    }

    let admission = exact_selected_admission(&decision, admissions)?;
    if !admission.matches_manifest(manifest) {
        return Err(RoutedDeploymentError::SelectedAdmissionMismatch);
    }
    admission
        .recheck_currentness(admission_currentness)
        .map_err(RoutedDeploymentError::SelectedAdmissionCurrentness)?;

    let authority = exact_selected_deployment(&decision, deployment_authorities)?;
    authority
        .worker_qualification
        .recheck_currentness(worker_currentness)
        .map_err(RoutedDeploymentError::SelectedWorkerCurrentness)?;
    authority
        .deployment
        .verify(admission, authority.worker_qualification)?;

    Ok(RoutedDeploymentSelection {
        decision,
        request_sha256,
        admission,
        authority,
    })
}

fn exact_selected_admission<'a>(
    decision: &RoutingDecision,
    admissions: &'a [ActiveAdmission],
) -> Result<&'a ActiveAdmission, RoutedDeploymentError> {
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or(RoutedDeploymentError::SelectedAdmissionMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedDeploymentError::SelectedAdmissionMissingOrDuplicate);
    }
    require_decision_matches_admission(decision, admission)?;
    Ok(admission)
}

fn require_decision_matches_admission(
    decision: &RoutingDecision,
    admission: &ActiveAdmission,
) -> Result<(), RoutedDeploymentError> {
    if admission.extension() != &decision.selected
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
        || !admission.allows_capability(&decision.capability)
    {
        return Err(RoutedDeploymentError::SelectedAdmissionMismatch);
    }
    Ok(())
}

fn exact_selected_deployment<'a>(
    decision: &RoutingDecision,
    authorities: &'a [DeploymentExecutionAuthority<'a>],
) -> Result<DeploymentExecutionAuthority<'a>, RoutedDeploymentError> {
    let mut matches = authorities.iter().copied().filter(|authority| {
        authority.deployment.selected_extension() == decision.selected.as_str()
    });
    let authority = matches
        .next()
        .ok_or(RoutedDeploymentError::SelectedDeploymentMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedDeploymentError::SelectedDeploymentMissingOrDuplicate);
    }
    Ok(authority)
}
