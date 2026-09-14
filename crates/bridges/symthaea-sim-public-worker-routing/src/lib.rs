// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic provider routing into the frozen public/community worker deployment.
//!
//! Routing continues to select the admitted package as `RuntimeKind::Wasm`.
//! Out-of-process execution does not rewrite package truth to `Remote`. A routed
//! selection binds one canonical request digest, one exact routing decision, one
//! exact active admission, one exact public deployment, and one live worker
//! qualification. The resulting invocation is a private-field capability for a
//! later release layer; this crate mints no release receipt.

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
use symthaea_sim_digest::{canonical_request_sha256_v1, CanonicalDigestError};
use symthaea_sim_extension_routing::solver_capability;
use symthaea_sim_public_worker_deployment::{
    BoundPublicWorkerDeployment, PublicWorkerDeploymentError, PublicWorkerDeploymentInvocation,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_qualification::{
    ActiveWorkerQualification, WorkerQualificationCurrentnessSource, WorkerQualificationError,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct PublicWorkerExecutionAuthority<'a> {
    base_deployment: &'a BoundSimulationDeployment,
    deployment: &'a BoundPublicWorkerDeployment,
    worker_qualification: &'a ActiveWorkerQualification,
}

impl<'a> PublicWorkerExecutionAuthority<'a> {
    pub fn new(
        base_deployment: &'a BoundSimulationDeployment,
        deployment: &'a BoundPublicWorkerDeployment,
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

    pub fn deployment(&self) -> &'a BoundPublicWorkerDeployment {
        self.deployment
    }

    pub fn worker_qualification(&self) -> &'a ActiveWorkerQualification {
        self.worker_qualification
    }
}

#[derive(Debug)]
pub struct RoutedPublicWorkerSelection<'a> {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    admission: &'a ActiveAdmission,
    authority: PublicWorkerExecutionAuthority<'a>,
}

impl<'a> RoutedPublicWorkerSelection<'a> {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub fn deployment(&self) -> &'a BoundPublicWorkerDeployment {
        self.authority.deployment
    }

    pub fn base_deployment(&self) -> &'a BoundSimulationDeployment {
        self.authority.base_deployment
    }

    pub fn worker_qualification(&self) -> &'a ActiveWorkerQualification {
        self.authority.worker_qualification
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        request: &SimulationRequest,
        admission_currentness: &dyn AdmissionCurrentnessSource,
        worker_currentness: &dyn WorkerQualificationCurrentnessSource,
        cgroup: &CgroupV2Lease,
    ) -> Result<RoutedPublicWorkerInvocation, RoutedPublicWorkerError> {
        require_request_matches(self.request_sha256, &self.decision, request)?;
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
        invocation.evidence().verify()?;
        if invocation.base_deployment_sha256()
            != self.authority.base_deployment.deployment_sha256()
            || invocation.binding_sha256() != self.authority.deployment.binding_sha256()
            || invocation.worker_qualification_evidence_sha256()
                != self
                    .authority
                    .worker_qualification
                    .qualification_evidence_sha256()
        {
            return Err(RoutedPublicWorkerError::DeploymentCommitmentMismatch);
        }

        Ok(RoutedPublicWorkerInvocation {
            decision: self.decision.clone(),
            request_sha256: self.request_sha256,
            public_deployment_sha256: self.authority.deployment.binding_sha256(),
            worker_qualification_evidence_sha256: self
                .authority
                .worker_qualification
                .qualification_evidence_sha256(),
            invocation,
        })
    }
}

#[derive(Debug)]
pub struct RoutedPublicWorkerInvocation {
    decision: RoutingDecision,
    request_sha256: [u8; 32],
    public_deployment_sha256: [u8; 32],
    worker_qualification_evidence_sha256: [u8; 32],
    invocation: PublicWorkerDeploymentInvocation,
}

impl RoutedPublicWorkerInvocation {
    pub fn decision(&self) -> &RoutingDecision {
        &self.decision
    }

    pub const fn request_sha256(&self) -> [u8; 32] {
        self.request_sha256
    }

    pub const fn public_deployment_sha256(&self) -> [u8; 32] {
        self.public_deployment_sha256
    }

    pub const fn worker_qualification_evidence_sha256(&self) -> [u8; 32] {
        self.worker_qualification_evidence_sha256
    }

    pub fn invocation(&self) -> &PublicWorkerDeploymentInvocation {
        &self.invocation
    }
}

#[derive(Debug, Error)]
pub enum RoutedPublicWorkerError {
    #[error("simulation request is invalid: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Canonical(#[from] CanonicalDigestError),
    #[error("routing failed: {0:?}")]
    Routing(RoutingError),
    #[error("router selected provider missing from registry: {0:?}")]
    SelectedManifestMissing(ExtensionId),
    #[error("public worker routing requires selected manifest runtime = wasm")]
    SelectedRuntimeNotWasm,
    #[error("selected admission is missing or duplicated")]
    SelectedAdmissionMissingOrDuplicate,
    #[error("selected admission commitments do not equal routing decision")]
    SelectedAdmissionMismatch,
    #[error("selected admission currentness failed after routing: {0:?}")]
    SelectedAdmissionCurrentness(AdmissionProblem),
    #[error("selected public deployment authority is missing or duplicated")]
    SelectedDeploymentMissingOrDuplicate,
    #[error("selected worker qualification currentness failed after routing: {0}")]
    SelectedWorkerCurrentness(WorkerQualificationError),
    #[error("selected public deployment does not match base deployment/live authorities")]
    SelectedDeploymentMismatch,
    #[error("route selected for one canonical request was used for a different request")]
    RequestSubstitution,
    #[error("route capability no longer matches request solver")]
    CapabilitySubstitution,
    #[error("executed public deployment commitment no longer matches selected authority")]
    DeploymentCommitmentMismatch,
    #[error(transparent)]
    BaseDeployment(#[from] SimulationDeploymentError),
    #[error(transparent)]
    PublicDeployment(#[from] PublicWorkerDeploymentError),
}

impl From<RoutingError> for RoutedPublicWorkerError {
    fn from(value: RoutingError) -> Self {
        Self::Routing(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn select_routed_public_worker<'a>(
    registry: &ExtensionRegistry,
    request: &SimulationRequest,
    constraints: RoutingConstraints,
    admissions: &'a [ActiveAdmission],
    observations: &[ProviderObservation],
    authorities: &'a [PublicWorkerExecutionAuthority<'a>],
    admission_currentness: &dyn AdmissionCurrentnessSource,
    worker_currentness: &dyn WorkerQualificationCurrentnessSource,
) -> Result<RoutedPublicWorkerSelection<'a>, RoutedPublicWorkerError> {
    request
        .validate()
        .map_err(|error| RoutedPublicWorkerError::InvalidRequest(error.to_string()))?;
    let request_sha256 = canonical_request_sha256_v1(request)?;
    let capability = solver_capability(request.solver);
    let routing_request = RoutingRequest {
        capability: capability.clone(),
        constraints,
    };
    let decision = ExtensionRouter::route(registry, &routing_request, admissions, observations)?;
    if decision.capability != capability {
        return Err(RoutedPublicWorkerError::CapabilitySubstitution);
    }

    let manifest = registry
        .get(&decision.selected)
        .ok_or_else(|| RoutedPublicWorkerError::SelectedManifestMissing(decision.selected.clone()))?;
    if manifest.runtime != RuntimeKind::Wasm {
        return Err(RoutedPublicWorkerError::SelectedRuntimeNotWasm);
    }

    let admission = exact_selected_admission(&decision, admissions)?;
    if !admission.matches_manifest(manifest) {
        return Err(RoutedPublicWorkerError::SelectedAdmissionMismatch);
    }
    admission
        .recheck_currentness(admission_currentness)
        .map_err(RoutedPublicWorkerError::SelectedAdmissionCurrentness)?;

    let authority = exact_selected_authority(&decision, authorities)?;
    authority
        .worker_qualification
        .recheck_currentness(worker_currentness)
        .map_err(RoutedPublicWorkerError::SelectedWorkerCurrentness)?;
    verify_authority(admission, authority)?;

    Ok(RoutedPublicWorkerSelection {
        decision,
        request_sha256,
        admission,
        authority,
    })
}

fn require_request_matches(
    expected_request_sha256: [u8; 32],
    decision: &RoutingDecision,
    request: &SimulationRequest,
) -> Result<(), RoutedPublicWorkerError> {
    let actual_request_sha256 = canonical_request_sha256_v1(request)?;
    if actual_request_sha256 != expected_request_sha256 {
        return Err(RoutedPublicWorkerError::RequestSubstitution);
    }
    if solver_capability(request.solver) != decision.capability {
        return Err(RoutedPublicWorkerError::CapabilitySubstitution);
    }
    Ok(())
}

fn exact_selected_admission<'a>(
    decision: &RoutingDecision,
    admissions: &'a [ActiveAdmission],
) -> Result<&'a ActiveAdmission, RoutedPublicWorkerError> {
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or(RoutedPublicWorkerError::SelectedAdmissionMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedPublicWorkerError::SelectedAdmissionMissingOrDuplicate);
    }
    require_decision_matches_admission(decision, admission)?;
    Ok(admission)
}

fn require_decision_matches_admission(
    decision: &RoutingDecision,
    admission: &ActiveAdmission,
) -> Result<(), RoutedPublicWorkerError> {
    if admission.extension() != &decision.selected
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
        || !admission.allows_capability(&decision.capability)
    {
        return Err(RoutedPublicWorkerError::SelectedAdmissionMismatch);
    }
    Ok(())
}

fn exact_selected_authority<'a>(
    decision: &RoutingDecision,
    authorities: &'a [PublicWorkerExecutionAuthority<'a>],
) -> Result<PublicWorkerExecutionAuthority<'a>, RoutedPublicWorkerError> {
    let mut matches = authorities.iter().copied().filter(|authority| {
        authority.base_deployment.selected_extension() == decision.selected.as_str()
    });
    let authority = matches
        .next()
        .ok_or(RoutedPublicWorkerError::SelectedDeploymentMissingOrDuplicate)?;
    if matches.next().is_some() {
        return Err(RoutedPublicWorkerError::SelectedDeploymentMissingOrDuplicate);
    }
    Ok(authority)
}

fn verify_authority(
    admission: &ActiveAdmission,
    authority: PublicWorkerExecutionAuthority<'_>,
) -> Result<(), RoutedPublicWorkerError> {
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
        || !authority.deployment.evidence().legacy_preexec_rlimits_applied
    {
        return Err(RoutedPublicWorkerError::SelectedDeploymentMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionRecord, AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
    };
    use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
    use symthaea_extension_router::{ProviderState, RejectionReason};
    use symthaea_sim_bridge::{EngineeringDomain, SolverKind};
    use symthaea_sim_public_worker_execution::public_supervisor_limits_v1;
    use symthaea_sim_public_worker_input::public_worker_frame_limits_v1;
    use symthaea_sim_worker_image::SealedWorkerImage;
    use symthaea_sim_worker_qualification::{
        WorkerQualificationContext, WorkerQualificationRecord,
    };

    #[derive(Debug)]
    struct AdmissionCurrentness;

    impl AdmissionCurrentnessSource for AdmissionCurrentness {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            Some(AdmissionContext::active(3, 4))
        }
    }

    #[derive(Debug)]
    struct WorkerCurrentness;

    impl WorkerQualificationCurrentnessSource for WorkerCurrentness {
        fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
            Some(WorkerQualificationContext::active(9))
        }
    }

    fn manifest() -> ExtensionManifest {
        serde_json::from_str(include_str!(
            "../../../../examples/extensions/hello-simulation/manifest.json"
        ))
        .unwrap()
    }

    fn request(id: &str) -> SimulationRequest {
        SimulationRequest::new(
            id,
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "route public worker test",
        )
        .with_parameter("alpha", 2.5, "1", "routing")
        .with_parameter("beta", 1.5, "1", "routing")
    }

    #[cfg(target_os = "linux")]
    fn subject() -> (
        ExtensionRegistry,
        Vec<ActiveAdmission>,
        Vec<ProviderObservation>,
        BoundSimulationDeployment,
        BoundPublicWorkerDeployment,
        ActiveWorkerQualification,
    ) {
        let manifest = manifest();
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let component = b"synthetic-component-for-routing".to_vec();
        let admission_currentness = AdmissionCurrentness;
        let worker_currentness = WorkerCurrentness;
        let admission_record = AdmissionRecord::issue(
            manifest.id.clone(),
            manifest.version.clone(),
            Sha256Digest::new(Sha256::digest(&manifest_bytes).into()),
            Sha256Digest::new(Sha256::digest(&component).into()),
            Sha256Digest::new([0x55; 32]),
            PrincipalId::new("test.public-worker-routing").unwrap(),
            None,
            TrustLevel::Community,
            vec![CapabilityId::new("engineering.simulation.custom")],
            PermissionSet::default(),
            3,
            4,
        )
        .unwrap();
        let admission = admission_record
            .activate(&manifest, &admission_currentness)
            .unwrap();

        let qualification_image =
            SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
        let qualification_record = WorkerQualificationRecord::issue(
            qualification_image.image_sha256(),
            [0xa7; 32],
            9,
        )
        .unwrap();
        let qualification = qualification_record
            .activate(&qualification_image, &worker_currentness)
            .unwrap();

        let base = BoundSimulationDeployment::issue(
            &admission,
            &admission_currentness,
            &qualification,
            &worker_currentness,
            manifest_bytes.clone(),
            component.clone(),
            SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap(),
            public_supervisor_limits_v1(),
            public_worker_frame_limits_v1(),
        )
        .unwrap();
        let deployment = BoundPublicWorkerDeployment::issue(
            &base,
            &admission,
            &admission_currentness,
            &qualification,
            &worker_currentness,
            manifest_bytes,
            component,
            SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap(),
        )
        .unwrap();

        let mut registry = ExtensionRegistry::new();
        registry.register(manifest.clone()).unwrap();
        let observations = vec![ProviderObservation {
            extension: manifest.id.clone(),
            state: ProviderState::Ready,
            evidence_grade: 5,
            reliability_bps: 10_000,
            estimated_latency_ms: Some(1),
        }];
        (registry, vec![admission], observations, base, deployment, qualification)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selection_binds_request_and_exact_public_deployment() {
        let (registry, admissions, observations, base, deployment, qualification) = subject();
        let authorities = [PublicWorkerExecutionAuthority::new(
            &base,
            &deployment,
            &qualification,
        )];
        let selected = select_routed_public_worker(
            &registry,
            &request("public-route"),
            RoutingConstraints::default(),
            &admissions,
            &observations,
            &authorities,
            &AdmissionCurrentness,
            &WorkerCurrentness,
        )
        .unwrap();
        assert_eq!(&selected.decision().selected, admissions[0].extension());
        assert_eq!(selected.deployment().binding_sha256(), deployment.binding_sha256());
        assert_eq!(selected.base_deployment().deployment_sha256(), base.deployment_sha256());
        assert_eq!(selected.worker_qualification().worker_sha256(), qualification.worker_sha256());
        assert_eq!(
            selected.request_sha256(),
            canonical_request_sha256_v1(&request("public-route")).unwrap()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn contained_public_wasm_is_not_remote_runtime() {
        let (registry, admissions, observations, base, deployment, qualification) = subject();
        let authorities = [PublicWorkerExecutionAuthority::new(
            &base,
            &deployment,
            &qualification,
        )];
        let constraints = RoutingConstraints {
            allowed_runtimes: vec![RuntimeKind::Remote],
            ..RoutingConstraints::default()
        };
        let error = select_routed_public_worker(
            &registry,
            &request("remote-only"),
            constraints,
            &admissions,
            &observations,
            &authorities,
            &AdmissionCurrentness,
            &WorkerCurrentness,
        )
        .expect_err("runtime=wasm package must not satisfy Remote-only routing");
        match error {
            RoutedPublicWorkerError::Routing(RoutingError::NoEligibleProvider {
                assessments,
                ..
            }) => assert!(assessments.iter().any(|assessment| {
                assessment
                    .rejection_reasons
                    .contains(&RejectionReason::RuntimeNotAllowed)
            })),
            other => panic!("unexpected routing error: {other:?}"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn canonical_request_substitution_is_rejected_before_execution() {
        let (registry, admissions, observations, base, deployment, qualification) = subject();
        let authorities = [PublicWorkerExecutionAuthority::new(
            &base,
            &deployment,
            &qualification,
        )];
        let selected = select_routed_public_worker(
            &registry,
            &request("request-a"),
            RoutingConstraints::default(),
            &admissions,
            &observations,
            &authorities,
            &AdmissionCurrentness,
            &WorkerCurrentness,
        )
        .unwrap();
        let error = require_request_matches(
            selected.request_sha256(),
            selected.decision(),
            &request("request-b"),
        )
        .expect_err("different canonical request must be rejected");
        assert!(matches!(error, RoutedPublicWorkerError::RequestSubstitution));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn duplicate_public_authority_is_rejected() {
        let (registry, admissions, observations, base, deployment, qualification) = subject();
        let authority = PublicWorkerExecutionAuthority::new(&base, &deployment, &qualification);
        let authorities = [authority, authority];
        let error = select_routed_public_worker(
            &registry,
            &request("duplicate-authority"),
            RoutingConstraints::default(),
            &admissions,
            &observations,
            &authorities,
            &AdmissionCurrentness,
            &WorkerCurrentness,
        )
        .expect_err("duplicate exact deployment authority must fail closed");
        assert!(matches!(
            error,
            RoutedPublicWorkerError::SelectedDeploymentMissingOrDuplicate
        ));
    }
}
