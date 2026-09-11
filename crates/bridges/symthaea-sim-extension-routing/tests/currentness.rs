// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

#[derive(Debug)]
struct RevokingBackend {
    loser_revoked: Arc<AtomicBool>,
}

impl SimulationBackend for RevokingBackend {
    fn name(&self) -> &'static str {
        "currentness-test"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Circuit]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        self.loser_revoked.store(true, Ordering::SeqCst);
        Ok(SimulationResult::dry_run(
            request.id.clone(),
            self.name(),
            0.9,
        ))
    }
}

#[derive(Debug)]
struct RevokingFactory {
    descriptor: SimulationProviderDescriptor,
    constructions: Arc<AtomicUsize>,
    loser_revoked: Arc<AtomicBool>,
}

impl SimulationBackendFactory for RevokingFactory {
    fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError> {
        self.constructions.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(RevokingBackend {
            loser_revoked: self.loser_revoked.clone(),
        }))
    }
}

#[derive(Debug, Clone, Copy)]
struct FixedCurrentness(Option<AdmissionContext>);

impl AdmissionCurrentnessSource for FixedCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        self.0
    }
}

#[derive(Debug)]
struct RevokeOnFourthCheck {
    calls: AtomicUsize,
}

impl AdmissionCurrentnessSource for RevokeOnFourthCheck {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Some(if call < 3 {
            AdmissionContext::active(7, 11)
        } else {
            AdmissionContext::revoked(7, 11)
        })
    }
}

#[derive(Debug)]
struct LoserRevokedDuringWinnerExecution {
    loser: ExtensionId,
    loser_revoked: Arc<AtomicBool>,
}

impl AdmissionCurrentnessSource for LoserRevokedDuringWinnerExecution {
    fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let revoked = subject.extension() == &self.loser
            && self.loser_revoked.load(Ordering::SeqCst);
        Some(if revoked {
            AdmissionContext::revoked(7, 11)
        } else {
            AdmissionContext::active(7, 11)
        })
    }
}

fn manifest(id: &str) -> ExtensionManifest {
    ExtensionManifest {
        id: ExtensionId::new(id),
        name: format!("{id} test provider"),
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
    }
}

fn descriptor(id: &str) -> SimulationProviderDescriptor {
    SimulationProviderDescriptor {
        manifest: manifest(id),
        backend_name: "currentness-test".into(),
        supported_solvers: vec![SolverKind::Circuit],
    }
}

fn registry() -> (AdmissionAuthority, LazySimulationRegistry, Arc<AtomicUsize>) {
    let authority = AdmissionAuthority::new();
    let constructions = Arc::new(AtomicUsize::new(0));
    let mut registry = LazySimulationRegistry::new(authority.scope());
    registry
        .register(Factory {
            descriptor: descriptor("org.example.currentness"),
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

fn scoped_admission_for(
    authority: &AdmissionAuthority,
    registry: &LazySimulationRegistry,
    id: &str,
) -> symthaea_extension_authority::ScopedAdmission {
    let extension = ExtensionId::new(id);
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

fn scoped_admission(
    authority: &AdmissionAuthority,
    registry: &LazySimulationRegistry,
) -> symthaea_extension_authority::ScopedAdmission {
    scoped_admission_for(authority, registry, "org.example.currentness")
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

#[test]
fn revocation_during_selected_execution_withholds_result_and_reports_post_use_failure() {
    let (authority, registry, constructions) = registry();
    let admissions = authority
        .scope()
        .bundle(vec![scoped_admission(&authority, &registry)])
        .unwrap();
    let source = RevokeOnFourthCheck {
        calls: AtomicUsize::new(0),
    };

    let error = registry
        .run(&request(), constraints(), &admissions, &source)
        .unwrap_err();

    assert!(matches!(
        error,
        LazySimulationError::AdmissionAuthority(
            ScopedAdmissionSetError::PostUseCurrentness {
                problem: AdmissionProblem::Revoked,
                ..
            }
        )
    ));
    // Candidate routing consumed two checks; selected execution consumed the
    // third pre-check, then the fourth post-check observed revocation.
    assert_eq!(source.calls.load(Ordering::SeqCst), 4);
    assert_eq!(constructions.load(Ordering::SeqCst), 1);
}

#[test]
fn losing_provider_revoked_during_winner_execution_does_not_invalidate_winner_result() {
    let authority = AdmissionAuthority::new();
    let scope = authority.scope();
    let loser_count = Arc::new(AtomicUsize::new(0));
    let winner_count = Arc::new(AtomicUsize::new(0));
    let loser_revoked = Arc::new(AtomicBool::new(false));
    let loser = ExtensionId::new("org.example.loser");
    let winner = ExtensionId::new("org.example.winner");

    let mut registry = LazySimulationRegistry::new(scope.clone());
    registry
        .register(Factory {
            descriptor: descriptor(loser.as_str()),
            constructions: loser_count.clone(),
        })
        .unwrap();
    registry
        .register(RevokingFactory {
            descriptor: descriptor(winner.as_str()),
            constructions: winner_count.clone(),
            loser_revoked: loser_revoked.clone(),
        })
        .unwrap();
    registry
        .set_observation(ProviderObservation {
            extension: loser.clone(),
            state: ProviderState::Ready,
            evidence_grade: 1,
            reliability_bps: 8_000,
            estimated_latency_ms: Some(10),
        })
        .unwrap();
    registry
        .set_observation(ProviderObservation {
            extension: winner.clone(),
            state: ProviderState::Ready,
            evidence_grade: 5,
            reliability_bps: 10_000,
            estimated_latency_ms: Some(1),
        })
        .unwrap();

    let admissions = scope
        .bundle(vec![
            scoped_admission_for(&authority, &registry, loser.as_str()),
            scoped_admission_for(&authority, &registry, winner.as_str()),
        ])
        .unwrap();
    let source = LoserRevokedDuringWinnerExecution {
        loser: loser.clone(),
        loser_revoked: loser_revoked.clone(),
    };

    let (result, decision) = registry
        .run(&request(), constraints(), &admissions, &source)
        .unwrap();

    assert_eq!(decision.selected, winner);
    assert_eq!(result.evidence.backend.as_deref(), Some("currentness-test"));
    assert!(loser_revoked.load(Ordering::SeqCst));
    assert_eq!(loser_count.load(Ordering::SeqCst), 0);
    assert_eq!(winner_count.load(Ordering::SeqCst), 1);
}
