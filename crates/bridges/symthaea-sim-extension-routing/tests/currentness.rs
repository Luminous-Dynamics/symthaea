// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_authority::{AdmissionAuthority, ScopedAdmissionSetError};
use symthaea_extension_core::{
    AbiVersion, CapabilityDescriptor, EffectClass, ExtensionId, ExtensionKind, ExtensionManifest,
    PermissionSet, ResourceBudget, RuntimeKind,
};
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_sim_bridge::{
    EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest, SimulationResult,
    SolverKind,
};
use symthaea_sim_extension_routing::{
    LazySimulationError, LazySimulationRegistry, SimulationBackendFactory,
    SimulationProviderDescriptor, solver_capability,
};

#[derive(Debug)]
struct Backend;

impl SimulationBackend for Backend {
    fn name(&self) -> &'static str {
        "currentness-test"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Circuit]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        Ok(SimulationResult::dry_run(
            request.id.clone(),
            self.name(),
            0.9,
        ))
    }
}

#[derive(Debug)]
struct Factory {
    descriptor: SimulationProviderDescriptor,
    constructions: Arc<AtomicUsize>,
}

impl SimulationBackendFactory for Factory {
    fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError> {
        self.constructions.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Backend))
    }
}

#[derive(Debug, Clone, Copy)]
struct FixedCurrentness(Option<AdmissionContext>);

impl AdmissionCurrentnessSource for FixedCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        self.0
    }
}

fn registry() -> (AdmissionAuthority, LazySimulationRegistry, Arc<AtomicUsize>) {
    let authority = AdmissionAuthority::new();
    let constructions = Arc::new(AtomicUsize::new(0));
    let manifest = ExtensionManifest {
        id: ExtensionId::new("org.example.currentness"),
        name: "currentness test provider".into(),
        version: "1.0.0".into(),
        abi: AbiVersion::V1,
        kind: ExtensionKind::Simulation,
        runtime: RuntimeKind::Native,
        description: String::new(),
        provides: vec![CapabilityDescriptor {
            id: solver_capability(SolverKind::Circuit),
            description: "test circuit simulation".into(),
            effect: EffectClass::Pure,
        }],
        requires: vec![],
        permissions: PermissionSet::default(),
        resources: ResourceBudget::default(),
    };
    let descriptor = SimulationProviderDescriptor {
        manifest,
        backend_name: "currentness-test".into(),
        supported_solvers: vec![SolverKind::Circuit],
    };
    let mut registry = LazySimulationRegistry::new(authority.scope());
    registry
        .register(Factory {
            descriptor,
            constructions: constructions.clone(),
        })
        .unwrap();
    registry
        .set_observation(ProviderObservation {
            extension: ExtensionId::new("org.example.currentness"),
            state: ProviderState::Ready,
            evidence_grade: 5,
            reliability_bps: 10_000,
            estimated_latency_ms: Some(1),
        })
        .unwrap();
    (authority, registry, constructions)
}

fn scoped_admission(
    authority: &AdmissionAuthority,
    registry: &LazySimulationRegistry,
) -> symthaea_extension_authority::ScopedAdmission {
    let extension = ExtensionId::new("org.example.currentness");
    let manifest = registry.catalog().get(&extension).unwrap();
    let source = FixedCurrentness(Some(AdmissionContext::active(7, 11)));
    let record = AdmissionRecord::issue(
        extension,
        manifest.version.clone(),
        Sha256Digest::new([1; 32]),
        Sha256Digest::new([2; 32]),
        Sha256Digest::new([3; 32]),
        PrincipalId::new("local:test-authority").unwrap(),
        Some(PrincipalId::new("did:example:test-publisher").unwrap()),
        TrustLevel::Trusted,
        vec![solver_capability(SolverKind::Circuit)],
        PermissionSet::default(),
        7,
        11,
    )
    .unwrap();
    authority.activate(&record, manifest, &source).unwrap()
}

fn request() -> SimulationRequest {
    SimulationRequest::new(
        "currentness-test",
        EngineeringDomain::Electrical,
        SolverKind::Circuit,
        "test",
    )
}

fn constraints() -> RoutingConstraints {
    RoutingConstraints {
        maximum_effect: EffectClass::Pure,
        minimum_trust: TrustLevel::Trusted,
        ..RoutingConstraints::default()
    }
}

#[test]
fn unavailable_currentness_fails_before_backend_construction() {
    let (authority, registry, constructions) = registry();
    let admissions = authority
        .scope()
        .bundle(vec![scoped_admission(&authority, &registry)])
        .unwrap();
    let error = registry
        .run(
            &request(),
            constraints(),
            &admissions,
            &FixedCurrentness(None),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        LazySimulationError::AdmissionAuthority(ScopedAdmissionSetError::Currentness {
            problem: AdmissionProblem::CurrentnessUnavailable,
            ..
        })
    ));
    assert_eq!(constructions.load(Ordering::SeqCst), 0);
}

#[test]
fn stale_policy_generation_fails_before_backend_construction() {
    let (authority, registry, constructions) = registry();
    let admissions = authority
        .scope()
        .bundle(vec![scoped_admission(&authority, &registry)])
        .unwrap();
    let error = registry
        .run(
            &request(),
            constraints(),
            &admissions,
            &FixedCurrentness(Some(AdmissionContext::active(8, 11))),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        LazySimulationError::AdmissionAuthority(ScopedAdmissionSetError::Currentness {
            problem: AdmissionProblem::GenerationMismatch {
                admitted: 7,
                current: 8
            },
            ..
        })
    ));
    assert_eq!(constructions.load(Ordering::SeqCst), 0);
}
