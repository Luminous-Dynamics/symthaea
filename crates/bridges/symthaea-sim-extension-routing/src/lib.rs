// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lazy, extension-aware routing for simulation backends.
//!
//! This bridge preserves `symthaea-sim-bridge` as the numerical contract while
//! moving provider discovery/routing ahead of backend instantiation. Expensive
//! solver adapters remain dormant until selected. Process-local authority scope
//! and live currentness are both checked before routing or construction.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use symthaea_extension_admission::{ActiveAdmission, AdmissionCurrentnessSource};
use symthaea_extension_authority::{
    AuthorityScope, ScopedAdmissionSet, ScopedAdmissionSetError,
};
use symthaea_extension_core::{CapabilityId, ExtensionId, ExtensionManifest};
use symthaea_extension_registry::{ExtensionRegistry, RegistryError};
use symthaea_extension_router::{
    ExtensionRouter, ProviderObservation, RoutingConstraints, RoutingDecision, RoutingError,
    RoutingRequest,
};
use symthaea_sim_bridge::{
    SimulationBackend, SimulationError, SimulationRegistry, SimulationRequest, SimulationResult,
    SolverKind,
};

pub fn solver_capability(solver: SolverKind) -> CapabilityId {
    let suffix = match solver {
        SolverKind::FiniteElement => "finite_element",
        SolverKind::ComputationalFluidDynamics => "computational_fluid_dynamics",
        SolverKind::MultibodyDynamics => "multibody_dynamics",
        SolverKind::Circuit => "circuit",
        SolverKind::Process => "process",
        SolverKind::CadGeometry => "cad_geometry",
        SolverKind::MultiPhysics => "multi_physics",
        SolverKind::Custom => "custom",
    };
    CapabilityId::new(format!("engineering.simulation.{suffix}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationProviderDescriptor {
    pub manifest: ExtensionManifest,
    pub backend_name: String,
    pub supported_solvers: Vec<SolverKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorProblem {
    InvalidManifest,
    EmptyBackendName,
    EmptySolverSet,
    DuplicateSolver(SolverKind),
    MissingSolverCapability(SolverKind),
}

impl SimulationProviderDescriptor {
    pub fn validate(&self) -> Result<(), DescriptorProblem> {
        self.manifest
            .validate()
            .map_err(|_| DescriptorProblem::InvalidManifest)?;
        if self.backend_name.trim().is_empty() {
            return Err(DescriptorProblem::EmptyBackendName);
        }
        if self.supported_solvers.is_empty() {
            return Err(DescriptorProblem::EmptySolverSet);
        }
        let mut seen = BTreeSet::new();
        for solver in &self.supported_solvers {
            if !seen.insert(solver_sort_key(*solver)) {
                return Err(DescriptorProblem::DuplicateSolver(*solver));
            }
            let capability = solver_capability(*solver);
            if !self
                .manifest
                .provides
                .iter()
                .any(|provided| provided.id == capability)
            {
                return Err(DescriptorProblem::MissingSolverCapability(*solver));
            }
        }
        Ok(())
    }
}

fn solver_sort_key(solver: SolverKind) -> u8 {
    match solver {
        SolverKind::FiniteElement => 0,
        SolverKind::ComputationalFluidDynamics => 1,
        SolverKind::MultibodyDynamics => 2,
        SolverKind::Circuit => 3,
        SolverKind::Process => 4,
        SolverKind::CadGeometry => 5,
        SolverKind::MultiPhysics => 6,
        SolverKind::Custom => 7,
    }
}

/// Cheap factory boundary. `create()` is called only after routing selects it.
pub trait SimulationBackendFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &SimulationProviderDescriptor;
    fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError>;
}

#[derive(Debug)]
pub enum LazySimulationError {
    Descriptor(DescriptorProblem),
    Registry(RegistryError),
    DuplicateFactory(ExtensionId),
    ObservationForUnknownProvider(ExtensionId),
    AdmissionForUnknownProvider(ExtensionId),
    AdmissionAuthority(ScopedAdmissionSetError),
    Route(RoutingError),
    SelectedFactoryMissing(ExtensionId),
    BackendConstruction {
        extension: ExtensionId,
        source: SimulationError,
    },
    BackendNameMismatch {
        extension: ExtensionId,
        expected: String,
        actual: String,
    },
    BackendSolverMismatch {
        extension: ExtensionId,
        solver: SolverKind,
    },
    Simulation(SimulationError),
}

/// Factories and quality telemetry may be cached. Admission authority is bound to
/// one process-local host scope, while scoped admission sets are supplied per use.
pub struct LazySimulationRegistry {
    authority_scope: AuthorityScope,
    catalog: ExtensionRegistry,
    factories: BTreeMap<ExtensionId, Box<dyn SimulationBackendFactory>>,
    observations: BTreeMap<ExtensionId, ProviderObservation>,
}

impl Debug for LazySimulationRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazySimulationRegistry")
            .field("provider_count", &self.factories.len())
            .field("observation_count", &self.observations.len())
            .finish_non_exhaustive()
    }
}

impl LazySimulationRegistry {
    /// Create a lazy solver registry bound to one process-local extension
    /// authority. There is intentionally no authority-free default constructor.
    pub fn new(authority_scope: AuthorityScope) -> Self {
        Self {
            authority_scope,
            catalog: ExtensionRegistry::new(),
            factories: BTreeMap::new(),
            observations: BTreeMap::new(),
        }
    }

    pub fn register(
        &mut self,
        factory: impl SimulationBackendFactory + 'static,
    ) -> Result<(), LazySimulationError> {
        let descriptor = factory.descriptor();
        descriptor
            .validate()
            .map_err(LazySimulationError::Descriptor)?;
        let id = descriptor.manifest.id.clone();
        if self.factories.contains_key(&id) {
            return Err(LazySimulationError::DuplicateFactory(id));
        }
        self.catalog
            .register(descriptor.manifest.clone())
            .map_err(LazySimulationError::Registry)?;
        self.factories.insert(id, Box::new(factory));
        Ok(())
    }

    pub fn set_observation(
        &mut self,
        observation: ProviderObservation,
    ) -> Result<(), LazySimulationError> {
        if !self.catalog.contains(&observation.extension) {
            return Err(LazySimulationError::ObservationForUnknownProvider(
                observation.extension,
            ));
        }
        self.observations
            .insert(observation.extension.clone(), observation);
        Ok(())
    }

    pub fn catalog(&self) -> &ExtensionRegistry {
        &self.catalog
    }

    /// Verify process-local authority and live currentness, then perform routing
    /// and backend selection entirely inside the checked authority callback.
    ///
    /// No raw admission slice escapes the callback, and no backend factory is
    /// invoked before the whole scoped set passes its authority checks.
    pub fn run(
        &self,
        request: &SimulationRequest,
        constraints: RoutingConstraints,
        admissions: &ScopedAdmissionSet,
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<(SimulationResult, RoutingDecision), LazySimulationError> {
        request
            .validate()
            .map_err(LazySimulationError::Simulation)?;

        self.authority_scope
            .with_rechecked_set(admissions, currentness, |checked| {
                self.run_checked(request, constraints, checked)
            })
            .map_err(LazySimulationError::AdmissionAuthority)?
    }

    fn run_checked(
        &self,
        request: &SimulationRequest,
        constraints: RoutingConstraints,
        admissions: &[ActiveAdmission],
    ) -> Result<(SimulationResult, RoutingDecision), LazySimulationError> {
        for admission in admissions {
            if !self.catalog.contains(admission.extension()) {
                return Err(LazySimulationError::AdmissionForUnknownProvider(
                    admission.extension().clone(),
                ));
            }
        }

        let routing_request = RoutingRequest {
            capability: solver_capability(request.solver),
            constraints,
        };
        let observations: Vec<_> = self.observations.values().cloned().collect();
        let decision = ExtensionRouter::route(
            &self.catalog,
            &routing_request,
            admissions,
            &observations,
        )
        .map_err(LazySimulationError::Route)?;

        let factory = self
            .factories
            .get(&decision.selected)
            .ok_or_else(|| LazySimulationError::SelectedFactoryMissing(decision.selected.clone()))?;
        let descriptor = factory.descriptor();
        let backend = factory
            .create()
            .map_err(|source| LazySimulationError::BackendConstruction {
                extension: decision.selected.clone(),
                source,
            })?;

        if backend.name() != descriptor.backend_name {
            return Err(LazySimulationError::BackendNameMismatch {
                extension: decision.selected.clone(),
                expected: descriptor.backend_name.clone(),
                actual: backend.name().to_string(),
            });
        }
        if !backend.supported_solvers().contains(&request.solver) {
            return Err(LazySimulationError::BackendSolverMismatch {
                extension: decision.selected.clone(),
                solver: request.solver,
            });
        }

        let mut legacy = SimulationRegistry::new();
        legacy.register(BoxedBackend(backend));
        let result = legacy
            .run(request)
            .map_err(LazySimulationError::Simulation)?;
        Ok((result, decision))
    }
}

struct BoxedBackend(Box<dyn SimulationBackend>);

impl Debug for BoxedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BoxedBackend").field(&self.0.name()).finish()
    }
}

impl SimulationBackend for BoxedBackend {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        self.0.supported_solvers()
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        self.0.run(request)
    }

    fn spawn_daemon(
        &self,
        request: &SimulationRequest,
    ) -> Result<Option<std::process::Child>, SimulationError> {
        self.0.spawn_daemon(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionRecord, AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
    };
    use symthaea_extension_authority::{
        AdmissionAuthority, AuthorityScopeError, ScopedAdmission, ScopedAdmissionSetError,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionKind, PermissionSet,
        ResourceBudget, RuntimeKind,
    };
    use symthaea_extension_router::ProviderState;
    use symthaea_sim_bridge::EngineeringDomain;

    #[derive(Debug)]
    struct MockBackend {
        name: &'static str,
        solvers: Vec<SolverKind>,
    }

    impl SimulationBackend for MockBackend {
        fn name(&self) -> &'static str {
            self.name
        }
        fn supported_solvers(&self) -> &[SolverKind] {
            &self.solvers
        }
        fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
            Ok(SimulationResult::dry_run(
                request.id.clone(),
                self.name,
                0.8,
            ))
        }
    }

    #[derive(Debug)]
    struct CountingFactory {
        descriptor: SimulationProviderDescriptor,
        count: Arc<AtomicUsize>,
        backend_name: &'static str,
        backend_solvers: Option<Vec<SolverKind>>,
    }

    impl SimulationBackendFactory for CountingFactory {
        fn descriptor(&self) -> &SimulationProviderDescriptor {
            &self.descriptor
        }
        fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError> {
            self.count.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(Box::new(MockBackend {
                name: self.backend_name,
                solvers: self
                    .backend_solvers
                    .clone()
                    .unwrap_or_else(|| self.descriptor.supported_solvers.clone()),
            }))
        }
    }

    #[derive(Debug, Default)]
    struct StaticCurrentness {
        contexts: BTreeMap<ExtensionId, AdmissionContext>,
    }

    impl StaticCurrentness {
        fn set(&mut self, extension: &str, context: AdmissionContext) {
            self.contexts.insert(ExtensionId::new(extension), context);
        }
    }

    impl AdmissionCurrentnessSource for StaticCurrentness {
        fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            self.contexts.get(subject.extension()).copied()
        }
    }

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::new([byte; 32])
    }

    fn descriptor(id: &str, name: &str, solver: SolverKind) -> SimulationProviderDescriptor {
        SimulationProviderDescriptor {
            manifest: ExtensionManifest {
                id: ExtensionId::new(id),
                name: id.into(),
                version: "1.0.0".into(),
                abi: AbiVersion::V1,
                kind: ExtensionKind::Simulation,
                runtime: RuntimeKind::Native,
                description: String::new(),
                provides: vec![CapabilityDescriptor {
                    id: solver_capability(solver),
                    description: "test simulation provider".into(),
                    effect: EffectClass::Pure,
                }],
                requires: vec![],
                permissions: PermissionSet::default(),
                resources: ResourceBudget::default(),
            },
            backend_name: name.into(),
            supported_solvers: vec![solver],
        }
    }

    fn factory(
        desc: SimulationProviderDescriptor,
        count: Arc<AtomicUsize>,
        name: &'static str,
    ) -> CountingFactory {
        CountingFactory {
            descriptor: desc,
            count,
            backend_name: name,
            backend_solvers: None,
        }
    }

    fn observation(id: &str, evidence: u8, reliability: u16) -> ProviderObservation {
        ProviderObservation {
            extension: ExtensionId::new(id),
            state: ProviderState::Ready,
            evidence_grade: evidence,
            reliability_bps: reliability,
            estimated_latency_ms: Some(10),
        }
    }

    fn currentness_for(entries: &[(&str, u64, u8)]) -> StaticCurrentness {
        let mut source = StaticCurrentness::default();
        for (id, generation, seed) in entries {
            source.set(
                id,
                AdmissionContext::active(*generation, 10_000 + u64::from(*seed)),
            );
        }
        source
    }

    fn scoped_admission(
        registry: &LazySimulationRegistry,
        authority: &AdmissionAuthority,
        id: &str,
        generation: u64,
        seed: u8,
    ) -> ScopedAdmission {
        let extension = ExtensionId::new(id);
        let manifest = registry.catalog().get(&extension).unwrap();
        let trust_generation = 10_000 + u64::from(seed);
        let record = AdmissionRecord::issue(
            extension.clone(),
            manifest.version.clone(),
            digest(seed),
            digest(seed.wrapping_add(1)),
            digest(seed.wrapping_add(2)),
            PrincipalId::new("local:test-authority").unwrap(),
            Some(PrincipalId::new("did:example:test-publisher").unwrap()),
            TrustLevel::Trusted,
            manifest
                .provides
                .iter()
                .map(|capability| capability.id.clone())
                .collect(),
            PermissionSet::default(),
            generation,
            trust_generation,
        )
        .unwrap();
        let source = StaticCurrentness {
            contexts: BTreeMap::from([(
                extension,
                AdmissionContext::active(generation, trust_generation),
            )]),
        };
        authority.activate(&record, manifest, &source).unwrap()
    }

    #[test]
    fn registration_is_cold_and_only_winner_is_instantiated() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::ComputationalFluidDynamics;
        let low = Arc::new(AtomicUsize::new(0));
        let high = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(factory(
                descriptor("org.example.low", "low", solver),
                low.clone(),
                "low",
            ))
            .unwrap();
        registry
            .register(factory(
                descriptor("org.example.high", "high", solver),
                high.clone(),
                "high",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.low", 2, 8_000))
            .unwrap();
        registry
            .set_observation(observation("org.example.high", 4, 9_900))
            .unwrap();
        let admissions = scope
            .bundle(vec![
                scoped_admission(&registry, &authority, "org.example.low", 1, 10),
                scoped_admission(&registry, &authority, "org.example.high", 1, 20),
            ])
            .unwrap();
        let currentness = currentness_for(&[
            ("org.example.low", 1, 10),
            ("org.example.high", 1, 20),
        ]);
        let request =
            SimulationRequest::new("run-1", EngineeringDomain::Mechanical, solver, "test");
        let (result, decision) = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &currentness,
            )
            .unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.high"));
        assert_eq!(result.evidence.backend.as_deref(), Some("high"));
        assert_eq!(low.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(high.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn missing_admission_prevents_instantiation() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(factory(
                descriptor("org.example.circuit", "circuit", solver),
                count.clone(),
                "circuit",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.circuit", 5, 10_000))
            .unwrap();
        let admissions = scope.bundle(Vec::new()).unwrap();
        let request =
            SimulationRequest::new("no-auth", EngineeringDomain::Electrical, solver, "test");
        assert!(registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &StaticCurrentness::default(),
            )
            .is_err());
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn foreign_authority_set_prevents_instantiation() {
        let authority = AdmissionAuthority::new();
        let foreign = AdmissionAuthority::new();
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(authority.scope());
        registry
            .register(factory(
                descriptor("org.example.foreign", "foreign", solver),
                count.clone(),
                "foreign",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.foreign", 5, 10_000))
            .unwrap();
        let token = scoped_admission(&registry, &foreign, "org.example.foreign", 1, 30);
        let admissions = foreign.scope().bundle(vec![token]).unwrap();
        let currentness = currentness_for(&[("org.example.foreign", 1, 30)]);
        let request =
            SimulationRequest::new("foreign", EngineeringDomain::Electrical, solver, "test");

        let error = registry
            .run(&request, RoutingConstraints::default(), &admissions, &currentness)
            .unwrap_err();
        assert!(matches!(
            error,
            LazySimulationError::AdmissionAuthority(ScopedAdmissionSetError::Scope(
                AuthorityScopeError::ForeignAuthority
            ))
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn revoked_admission_prevents_instantiation() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(factory(
                descriptor("org.example.revoked", "revoked", solver),
                count.clone(),
                "revoked",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.revoked", 5, 10_000))
            .unwrap();
        let admissions = scope
            .bundle(vec![scoped_admission(
                &registry,
                &authority,
                "org.example.revoked",
                1,
                31,
            )])
            .unwrap();
        let mut currentness = currentness_for(&[("org.example.revoked", 1, 31)]);
        currentness.set(
            "org.example.revoked",
            AdmissionContext::revoked(1, 10_031),
        );
        let request =
            SimulationRequest::new("revoked", EngineeringDomain::Electrical, solver, "test");
        let error = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &currentness,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LazySimulationError::AdmissionAuthority(ScopedAdmissionSetError::Currentness {
                problem: symthaea_extension_admission::AdmissionProblem::Revoked,
                ..
            })
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn stale_trust_generation_prevents_instantiation() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(factory(
                descriptor("org.example.stale", "stale", solver),
                count.clone(),
                "stale",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.stale", 5, 10_000))
            .unwrap();
        let admissions = scope
            .bundle(vec![scoped_admission(
                &registry,
                &authority,
                "org.example.stale",
                1,
                32,
            )])
            .unwrap();
        let mut currentness = currentness_for(&[("org.example.stale", 1, 32)]);
        currentness.set(
            "org.example.stale",
            AdmissionContext::active(1, 10_033),
        );
        let request =
            SimulationRequest::new("stale", EngineeringDomain::Electrical, solver, "test");
        let error = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &currentness,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LazySimulationError::AdmissionAuthority(ScopedAdmissionSetError::Currentness {
                problem: symthaea_extension_admission::AdmissionProblem::TrustGenerationMismatch {
                    ..
                },
                ..
            })
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn backend_must_match_cheap_descriptor() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(factory(
                descriptor("org.example.circuit", "declared", solver),
                count,
                "actual",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.circuit", 5, 10_000))
            .unwrap();
        let admissions = scope
            .bundle(vec![scoped_admission(
                &registry,
                &authority,
                "org.example.circuit",
                1,
                30,
            )])
            .unwrap();
        let currentness = currentness_for(&[("org.example.circuit", 1, 30)]);
        let request =
            SimulationRequest::new("run-2", EngineeringDomain::Electrical, solver, "test");
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &currentness,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::BackendNameMismatch { .. }
        ));
    }

    #[test]
    fn backend_solver_claim_must_match_descriptor() {
        let authority = AdmissionAuthority::new();
        let scope = authority.scope();
        let solver = SolverKind::FiniteElement;
        let count = Arc::new(AtomicUsize::new(0));
        let desc = descriptor("org.example.structure", "structure", solver);
        let mut registry = LazySimulationRegistry::new(scope.clone());
        registry
            .register(CountingFactory {
                descriptor: desc,
                count,
                backend_name: "structure",
                backend_solvers: Some(vec![SolverKind::Circuit]),
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.structure", 5, 10_000))
            .unwrap();
        let admissions = scope
            .bundle(vec![scoped_admission(
                &registry,
                &authority,
                "org.example.structure",
                1,
                40,
            )])
            .unwrap();
        let currentness = currentness_for(&[("org.example.structure", 1, 40)]);
        let request = SimulationRequest::new("run-3", EngineeringDomain::Civil, solver, "test");
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &currentness,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::BackendSolverMismatch { .. }
        ));
    }

    #[test]
    fn descriptor_must_advertise_capability_for_each_solver() {
        let solver = SolverKind::Process;
        let mut desc = descriptor("org.example.process", "process", solver);
        desc.manifest.provides.clear();
        assert_eq!(
            desc.validate(),
            Err(DescriptorProblem::MissingSolverCapability(solver))
        );
    }
}
