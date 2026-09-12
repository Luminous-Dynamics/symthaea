// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lazy, extension-aware routing for simulation backends.
//!
//! This bridge preserves `symthaea-sim-bridge` as the numerical contract while
//! moving provider discovery/routing ahead of backend instantiation. Expensive
//! solver adapters remain dormant until selected, and authority is supplied as
//! fresh point-of-use [`ActiveAdmission`] values rather than cached booleans.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionCurrentnessSource, AdmissionProblem, Sha256Digest,
};
use symthaea_extension_core::{CapabilityId, ExtensionId, ExtensionManifest, RuntimeKind};
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

/// Unforgeable safe-Rust call capability for selected backend construction.
///
/// The constructor is private to this crate. A permit is minted only after the
/// router has recovered the exact selected admission, rechecked live currentness,
/// and validated the factory's package commitments against that admission. The
/// permit carries no authority data itself and is intentionally neither Clone,
/// Copy, Default, Serialize, nor Deserialize.
#[derive(Debug)]
pub struct SelectedExecutionPermit {
    _private: (),
}

impl SelectedExecutionPermit {
    fn new() -> Self {
        Self { _private: () }
    }
}

/// Cheap factory boundary. Construction requires a router-minted selected permit.
pub trait SimulationBackendFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &SimulationProviderDescriptor;

    /// SHA-256 of the exact manifest bytes this factory will present to the
    /// execution host.
    ///
    /// Wasm factories must return `Some` and compute it from the immutable raw
    /// manifest bytes later passed to control inspection. Native built-ins may
    /// return `None` when their deployment model has no byte-addressed manifest.
    fn manifest_sha256(&self) -> Option<Sha256Digest> {
        None
    }

    /// SHA-256 of the exact executable payload bytes this factory will use.
    ///
    /// Wasm factories must return `Some` and compute it from the exact Component
    /// bytes later passed to Wasmtime. Native built-ins may return `None` when
    /// their admission/deployment model does not identify one payload blob.
    fn executable_payload_sha256(&self) -> Option<Sha256Digest> {
        None
    }

    fn create(
        &self,
        permit: &SelectedExecutionPermit,
    ) -> Result<Box<dyn SimulationBackend>, SimulationError>;
}

/// Point in the selected-provider lifecycle where live authority was rechecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentnessPhase {
    BeforeInstantiation,
    BeforeExecution,
    AfterExecution,
}

#[derive(Debug)]
pub enum LazySimulationError {
    Descriptor(DescriptorProblem),
    Registry(RegistryError),
    DuplicateFactory(ExtensionId),
    ObservationForUnknownProvider(ExtensionId),
    AdmissionForUnknownProvider(ExtensionId),
    Route(RoutingError),
    SelectedFactoryMissing(ExtensionId),
    SelectedAdmissionMissing(ExtensionId),
    SelectedAdmissionMismatch(ExtensionId),
    MissingManifestDigest(ExtensionId),
    ManifestDigestMismatch {
        extension: ExtensionId,
        admitted: Sha256Digest,
        executable: Sha256Digest,
    },
    MissingExecutablePayloadDigest(ExtensionId),
    ExecutablePayloadMismatch {
        extension: ExtensionId,
        admitted: Sha256Digest,
        executable: Sha256Digest,
    },
    AdmissionCurrentness {
        extension: ExtensionId,
        phase: CurrentnessPhase,
        source: AdmissionProblem,
    },
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

/// Factories and quality telemetry may be cached. Active authority is not.
#[derive(Default)]
pub struct LazySimulationRegistry {
    catalog: ExtensionRegistry,
    factories: BTreeMap<ExtensionId, Box<dyn SimulationBackendFactory>>,
    observations: BTreeMap<ExtensionId, ProviderObservation>,
}

impl Debug for LazySimulationRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazySimulationRegistry")
            .field("provider_count", &self.factories.len())
            .field("observation_count", &self.observations.len())
            .finish()
    }
}

impl LazySimulationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        factory: impl SimulationBackendFactory + 'static,
    ) -> Result<(), LazySimulationError> {
        let descriptor = factory.descriptor();
        descriptor.validate().map_err(LazySimulationError::Descriptor)?;
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

    /// Route, recheck the selected authority, instantiate only the winner,
    /// execute, then recheck authority again before releasing the result.
    ///
    /// Active admissions are never cached here. `currentness` is host-owned and
    /// is queried after selection before construction, immediately before
    /// execution, and after execution before release.
    pub fn run(
        &self,
        request: &SimulationRequest,
        constraints: RoutingConstraints,
        admissions: &[ActiveAdmission],
        currentness: &dyn AdmissionCurrentnessSource,
    ) -> Result<(SimulationResult, RoutingDecision), LazySimulationError> {
        request.validate().map_err(LazySimulationError::Simulation)?;
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

        let selected_admission = selected_admission(&decision, admissions)?;
        recheck_selected(
            selected_admission,
            &decision.selected,
            currentness,
            CurrentnessPhase::BeforeInstantiation,
        )?;

        let factory = self
            .factories
            .get(&decision.selected)
            .ok_or_else(|| LazySimulationError::SelectedFactoryMissing(decision.selected.clone()))?;
        let descriptor = factory.descriptor();
        validate_package_bindings(
            factory.as_ref(),
            descriptor,
            selected_admission,
            &decision.selected,
        )?;
        let permit = SelectedExecutionPermit::new();
        let backend = factory
            .create(&permit)
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

        recheck_selected(
            selected_admission,
            &decision.selected,
            currentness,
            CurrentnessPhase::BeforeExecution,
        )?;

        let mut legacy = SimulationRegistry::new();
        legacy.register(BoxedBackend(backend));
        let result = legacy.run(request).map_err(LazySimulationError::Simulation)?;

        recheck_selected(
            selected_admission,
            &decision.selected,
            currentness,
            CurrentnessPhase::AfterExecution,
        )?;
        Ok((result, decision))
    }
}

fn selected_admission<'a>(
    decision: &RoutingDecision,
    admissions: &'a [ActiveAdmission],
) -> Result<&'a ActiveAdmission, LazySimulationError> {
    let mut matches = admissions
        .iter()
        .filter(|admission| admission.extension() == &decision.selected);
    let admission = matches
        .next()
        .ok_or_else(|| LazySimulationError::SelectedAdmissionMissing(decision.selected.clone()))?;
    if matches.next().is_some()
        || admission.generation() != decision.selected_admission_generation
        || admission.trust_generation() != decision.selected_trust_generation
        || admission.manifest_sha256() != decision.selected_manifest_sha256
        || admission.payload_sha256() != decision.selected_payload_sha256
        || admission.policy_sha256() != decision.selected_policy_sha256
    {
        return Err(LazySimulationError::SelectedAdmissionMismatch(
            decision.selected.clone(),
        ));
    }
    Ok(admission)
}

fn validate_package_bindings(
    factory: &dyn SimulationBackendFactory,
    descriptor: &SimulationProviderDescriptor,
    admission: &ActiveAdmission,
    extension: &ExtensionId,
) -> Result<(), LazySimulationError> {
    let manifest = factory.manifest_sha256();
    if descriptor.manifest.runtime == RuntimeKind::Wasm && manifest.is_none() {
        return Err(LazySimulationError::MissingManifestDigest(extension.clone()));
    }
    if let Some(executable) = manifest {
        let admitted = admission.manifest_sha256();
        if executable != admitted {
            return Err(LazySimulationError::ManifestDigestMismatch {
                extension: extension.clone(),
                admitted,
                executable,
            });
        }
    }

    let executable = factory.executable_payload_sha256();
    if descriptor.manifest.runtime == RuntimeKind::Wasm && executable.is_none() {
        return Err(LazySimulationError::MissingExecutablePayloadDigest(
            extension.clone(),
        ));
    }
    if let Some(executable) = executable {
        let admitted = admission.payload_sha256();
        if executable != admitted {
            return Err(LazySimulationError::ExecutablePayloadMismatch {
                extension: extension.clone(),
                admitted,
                executable,
            });
        }
    }
    Ok(())
}

fn recheck_selected(
    admission: &ActiveAdmission,
    extension: &ExtensionId,
    currentness: &dyn AdmissionCurrentnessSource,
    phase: CurrentnessPhase,
) -> Result<(), LazySimulationError> {
    admission
        .recheck_currentness(currentness)
        .map_err(|source| LazySimulationError::AdmissionCurrentness {
            extension: extension.clone(),
            phase,
            source,
        })
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
        AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
        Sha256Digest, TrustLevel,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionKind, PermissionSet,
        ResourceBudget, RuntimeKind,
    };
    use symthaea_extension_router::ProviderState;
    use symthaea_sim_bridge::EngineeringDomain;

    #[derive(Debug, Clone, Copy)]
    struct TestCurrentness(AdmissionContext);

    impl AdmissionCurrentnessSource for TestCurrentness {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            Some(self.0)
        }
    }

    #[derive(Debug)]
    struct RevokeOnThirdCheck {
        checks: AtomicUsize,
        generation: u64,
        trust_generation: u64,
    }

    impl AdmissionCurrentnessSource for RevokeOnThirdCheck {
        fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            let check = self.checks.fetch_add(1, AtomicOrdering::SeqCst);
            if check < 2 {
                Some(AdmissionContext::active(self.generation, self.trust_generation))
            } else {
                Some(AdmissionContext::revoked(self.generation, self.trust_generation))
            }
        }
    }

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
            Ok(SimulationResult::dry_run(request.id.clone(), self.name, 0.8))
        }
    }

    #[derive(Debug)]
    struct CountingFactory {
        descriptor: SimulationProviderDescriptor,
        count: Arc<AtomicUsize>,
        backend_name: &'static str,
        backend_solvers: Option<Vec<SolverKind>>,
        manifest_sha256: Option<Sha256Digest>,
        executable_payload_sha256: Option<Sha256Digest>,
    }

    impl SimulationBackendFactory for CountingFactory {
        fn descriptor(&self) -> &SimulationProviderDescriptor {
            &self.descriptor
        }
        fn manifest_sha256(&self) -> Option<Sha256Digest> {
            self.manifest_sha256
        }
        fn executable_payload_sha256(&self) -> Option<Sha256Digest> {
            self.executable_payload_sha256
        }
        fn create(
            &self,
            _permit: &SelectedExecutionPermit,
        ) -> Result<Box<dyn SimulationBackend>, SimulationError> {
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
            manifest_sha256: None,
            executable_payload_sha256: None,
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

    fn active_admission(
        registry: &LazySimulationRegistry,
        id: &str,
        generation: u64,
        seed: u8,
    ) -> ActiveAdmission {
        let extension = ExtensionId::new(id);
        let manifest = registry.catalog().get(&extension).unwrap();
        let trust_generation = 10_000 + u64::from(seed);
        let currentness = TestCurrentness(AdmissionContext::active(generation, trust_generation));
        AdmissionRecord::issue(
            extension,
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
        .unwrap()
        .activate(manifest, &currentness)
        .unwrap()
    }

    fn currentness(generation: u64, seed: u8) -> TestCurrentness {
        TestCurrentness(AdmissionContext::active(
            generation,
            10_000 + u64::from(seed),
        ))
    }

    #[test]
    fn registration_is_cold_and_only_winner_is_instantiated() {
        let solver = SolverKind::ComputationalFluidDynamics;
        let low = Arc::new(AtomicUsize::new(0));
        let high = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
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
        assert_eq!(low.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(high.load(AtomicOrdering::SeqCst), 0);
        registry
            .set_observation(observation("org.example.low", 2, 8_000))
            .unwrap();
        registry
            .set_observation(observation("org.example.high", 4, 9_900))
            .unwrap();
        let admissions = vec![
            active_admission(&registry, "org.example.low", 1, 10),
            active_admission(&registry, "org.example.high", 1, 20),
        ];
        let request = SimulationRequest::new(
            "run-1",
            EngineeringDomain::Mechanical,
            solver,
            "test",
        );
        let source = currentness(1, 20);
        let (result, decision) = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.high"));
        assert_eq!(result.evidence.backend.as_deref(), Some("high"));
        assert_eq!(low.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(high.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn missing_admission_prevents_instantiation() {
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
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
        let request = SimulationRequest::new(
            "no-auth",
            EngineeringDomain::Electrical,
            solver,
            "test",
        );
        let source = currentness(1, 30);
        assert!(
            registry
                .run(
                    &request,
                    RoutingConstraints {
                        maximum_effect: EffectClass::Pure,
                        minimum_trust: TrustLevel::Trusted,
                        ..RoutingConstraints::default()
                    },
                    &[],
                    &source,
                )
                .is_err()
        );
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn selected_revocation_before_instantiation_fails_before_factory_creation() {
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
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
        let admissions = vec![active_admission(
            &registry,
            "org.example.circuit",
            1,
            30,
        )];
        let request = SimulationRequest::new(
            "revoked-before-create",
            EngineeringDomain::Electrical,
            solver,
            "test",
        );
        let revoked = TestCurrentness(AdmissionContext::revoked(1, 10_030));
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &revoked,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::AdmissionCurrentness {
                phase: CurrentnessPhase::BeforeInstantiation,
                source: AdmissionProblem::Revoked,
                ..
            }
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn revocation_during_execution_withholds_result() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(factory(
                descriptor("org.example.custom", "custom", solver),
                count.clone(),
                "custom",
            ))
            .unwrap();
        registry
            .set_observation(observation("org.example.custom", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.custom",
            7,
            50,
        )];
        let request = SimulationRequest::new(
            "revoked-after-run",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = RevokeOnThirdCheck {
            checks: AtomicUsize::new(0),
            generation: 7,
            trust_generation: 10_050,
        };
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::AdmissionCurrentness {
                phase: CurrentnessPhase::AfterExecution,
                source: AdmissionProblem::Revoked,
                ..
            }
        ));
        assert_eq!(source.checks.load(AtomicOrdering::SeqCst), 3);
        assert_eq!(count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn wasm_factory_requires_a_manifest_digest() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut desc = descriptor("org.example.wasm-manifest-missing", "wasm-manifest-missing", solver);
        desc.manifest.runtime = RuntimeKind::Wasm;
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count: count.clone(),
                backend_name: "wasm-manifest-missing",
                backend_solvers: None,
                manifest_sha256: None,
                executable_payload_sha256: Some(digest(61)),
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.wasm-manifest-missing", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.wasm-manifest-missing",
            3,
            60,
        )];
        let request = SimulationRequest::new(
            "wasm-missing-manifest-digest",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = currentness(3, 60);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::MissingManifestDigest(extension)
                if extension == ExtensionId::new("org.example.wasm-manifest-missing")
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn wasm_factory_manifest_digest_must_match_selected_admission() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut desc = descriptor("org.example.wasm-manifest-mismatch", "wasm-manifest-mismatch", solver);
        desc.manifest.runtime = RuntimeKind::Wasm;
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count: count.clone(),
                backend_name: "wasm-manifest-mismatch",
                backend_solvers: None,
                manifest_sha256: Some(digest(99)),
                executable_payload_sha256: Some(digest(71)),
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.wasm-manifest-mismatch", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.wasm-manifest-mismatch",
            4,
            70,
        )];
        let request = SimulationRequest::new(
            "wasm-manifest-mismatch-digest",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = currentness(4, 70);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(err, LazySimulationError::ManifestDigestMismatch { .. }));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn wasm_factory_requires_an_executable_payload_digest() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut desc = descriptor("org.example.wasm-missing", "wasm-missing", solver);
        desc.manifest.runtime = RuntimeKind::Wasm;
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count: count.clone(),
                backend_name: "wasm-missing",
                backend_solvers: None,
                manifest_sha256: Some(digest(60)),
                executable_payload_sha256: None,
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.wasm-missing", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.wasm-missing",
            3,
            60,
        )];
        let request = SimulationRequest::new(
            "wasm-missing-digest",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = currentness(3, 60);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            LazySimulationError::MissingExecutablePayloadDigest(extension)
                if extension == ExtensionId::new("org.example.wasm-missing")
        ));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn wasm_factory_payload_digest_must_match_selected_admission() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut desc = descriptor("org.example.wasm-mismatch", "wasm-mismatch", solver);
        desc.manifest.runtime = RuntimeKind::Wasm;
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count: count.clone(),
                backend_name: "wasm-mismatch",
                backend_solvers: None,
                manifest_sha256: Some(digest(70)),
                executable_payload_sha256: Some(digest(99)),
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.wasm-mismatch", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.wasm-mismatch",
            4,
            70,
        )];
        let request = SimulationRequest::new(
            "wasm-mismatch-digest",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = currentness(4, 70);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(err, LazySimulationError::ExecutablePayloadMismatch { .. }));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn wasm_factory_exact_manifest_and_payload_digests_are_allowed() {
        let solver = SolverKind::Custom;
        let count = Arc::new(AtomicUsize::new(0));
        let mut desc = descriptor("org.example.wasm-exact", "wasm-exact", solver);
        desc.manifest.runtime = RuntimeKind::Wasm;
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count: count.clone(),
                backend_name: "wasm-exact",
                backend_solvers: None,
                manifest_sha256: Some(digest(80)),
                executable_payload_sha256: Some(digest(81)),
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.wasm-exact", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.wasm-exact",
            5,
            80,
        )];
        let request = SimulationRequest::new(
            "wasm-exact-digest",
            EngineeringDomain::Systems,
            solver,
            "test",
        );
        let source = currentness(5, 80);
        let (result, decision) = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.wasm-exact"));
        assert_eq!(result.evidence.backend.as_deref(), Some("wasm-exact"));
        assert_eq!(count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn backend_must_match_cheap_descriptor() {
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
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
        let admissions = vec![active_admission(
            &registry,
            "org.example.circuit",
            1,
            30,
        )];
        let request = SimulationRequest::new(
            "run-2",
            EngineeringDomain::Electrical,
            solver,
            "test",
        );
        let source = currentness(1, 30);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(err, LazySimulationError::BackendNameMismatch { .. }));
    }

    #[test]
    fn backend_solver_claim_must_match_descriptor() {
        let solver = SolverKind::FiniteElement;
        let count = Arc::new(AtomicUsize::new(0));
        let desc = descriptor("org.example.structure", "structure", solver);
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: desc,
                count,
                backend_name: "structure",
                backend_solvers: Some(vec![SolverKind::Circuit]),
                manifest_sha256: None,
                executable_payload_sha256: None,
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.structure", 5, 10_000))
            .unwrap();
        let admissions = vec![active_admission(
            &registry,
            "org.example.structure",
            1,
            40,
        )];
        let request = SimulationRequest::new(
            "run-3",
            EngineeringDomain::Civil,
            solver,
            "test",
        );
        let source = currentness(1, 40);
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
                &admissions,
                &source,
            )
            .unwrap_err();
        assert!(matches!(err, LazySimulationError::BackendSolverMismatch { .. }));
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
