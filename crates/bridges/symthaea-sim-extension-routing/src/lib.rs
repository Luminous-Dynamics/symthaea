// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lazy, extension-aware routing for simulation backends.
//!
//! This bridge preserves `symthaea-sim-bridge` as the numerical contract while
//! moving provider discovery/routing ahead of backend instantiation. Expensive
//! solver adapters therefore remain dormant until selected.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
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

/// Stable semantic capability used to route one solver family.
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

/// Cheap metadata describing one lazily-instantiated simulation provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationProviderDescriptor {
    /// Shared extension identity/capability/permission/resource declaration.
    pub manifest: ExtensionManifest,
    /// Expected `SimulationBackend::name()` after factory instantiation.
    pub backend_name: String,
    /// Solver families this provider advertises without instantiating the backend.
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
            let sort_key = solver_sort_key(*solver);
            if !seen.insert(sort_key) {
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

/// Factory for an expensive or externally connected simulation backend.
///
/// `descriptor()` must be cheap and side-effect free. `create()` is called only
/// after the extension router selects this provider for an invocation.
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

/// Lazy simulation catalog. Factories are registered, not live backends.
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

    /// Register provider metadata and its lazy factory without constructing a
    /// solver backend.
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

    /// Update host-observed readiness/evidence/reliability for an admitted
    /// provider. Observation is runtime state, not extension-controlled metadata.
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

    /// Route, lazily instantiate only the winner, then delegate execution and
    /// result/evidence validation to the existing `SimulationRegistry` contract.
    pub fn run(
        &self,
        request: &SimulationRequest,
        constraints: RoutingConstraints,
    ) -> Result<(SimulationResult, RoutingDecision), LazySimulationError> {
        request
            .validate()
            .map_err(LazySimulationError::Simulation)?;

        let routing_request = RoutingRequest {
            capability: solver_capability(request.solver),
            constraints,
        };
        let observations: Vec<_> = self.observations.values().cloned().collect();
        let decision = ExtensionRouter::route(&self.catalog, &routing_request, &observations)
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

        // Preserve every request/result/provenance invariant in the existing
        // bridge by dispatching through a single-backend legacy registry.
        let mut legacy = SimulationRegistry::new();
        legacy.register(BoxedBackend(backend));
        let result = legacy.run(request).map_err(LazySimulationError::Simulation)?;
        Ok((result, decision))
    }
}

/// Thin delegation wrapper allowing an already-boxed dynamic backend to pass
/// through `SimulationRegistry::register`, which accepts a concrete backend.
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
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionKind, PermissionSet,
        ResourceBudget, RuntimeKind,
    };
    use symthaea_extension_router::{ProviderState, TrustLevel};

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
        create_count: Arc<AtomicUsize>,
        backend_name: &'static str,
    }

    impl SimulationBackendFactory for CountingFactory {
        fn descriptor(&self) -> &SimulationProviderDescriptor {
            &self.descriptor
        }

        fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError> {
            self.create_count.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(Box::new(MockBackend {
                name: self.backend_name,
                solvers: self.descriptor.supported_solvers.clone(),
            }))
        }
    }

    fn descriptor(id: &str, backend_name: &str, solver: SolverKind) -> SimulationProviderDescriptor {
        let capability = solver_capability(solver);
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
                    id: capability,
                    description: "test simulation provider".into(),
                    effect: EffectClass::Pure,
                }],
                requires: vec![],
                permissions: PermissionSet::default(),
                resources: ResourceBudget::default(),
            },
            backend_name: backend_name.into(),
            supported_solvers: vec![solver],
        }
    }

    fn observation(
        id: &str,
        evidence_grade: u8,
        reliability_bps: u16,
    ) -> ProviderObservation {
        ProviderObservation {
            extension: ExtensionId::new(id),
            admitted: true,
            state: ProviderState::Ready,
            trust: TrustLevel::Trusted,
            evidence_grade,
            reliability_bps,
            estimated_latency_ms: Some(10),
        }
    }

    #[test]
    fn registration_does_not_instantiate_backend() {
        let count = Arc::new(AtomicUsize::new(0));
        let factory = CountingFactory {
            descriptor: descriptor("org.example.fea", "mock-fea", SolverKind::FiniteElement),
            create_count: count.clone(),
            backend_name: "mock-fea",
        };
        let mut registry = LazySimulationRegistry::new();
        registry.register(factory).unwrap();
        assert_eq!(count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn only_selected_provider_is_instantiated() {
        let solver = SolverKind::ComputationalFluidDynamics;
        let low_count = Arc::new(AtomicUsize::new(0));
        let high_count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();

        registry
            .register(CountingFactory {
                descriptor: descriptor("org.example.low", "low", solver),
                create_count: low_count.clone(),
                backend_name: "low",
            })
            .unwrap();
        registry
            .register(CountingFactory {
                descriptor: descriptor("org.example.high", "high", solver),
                create_count: high_count.clone(),
                backend_name: "high",
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.low", 2, 8_000))
            .unwrap();
        registry
            .set_observation(observation("org.example.high", 4, 9_900))
            .unwrap();

        let request = SimulationRequest::new("run-1", EngineeringDomain::Mechanical, solver, "test");
        let (result, decision) = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
            )
            .unwrap();

        assert_eq!(decision.selected, ExtensionId::new("org.example.high"));
        assert_eq!(result.evidence.backend.as_deref(), Some("high"));
        assert_eq!(low_count.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(high_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn instantiated_backend_name_must_match_descriptor() {
        let solver = SolverKind::Circuit;
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = LazySimulationRegistry::new();
        registry
            .register(CountingFactory {
                descriptor: descriptor("org.example.circuit", "declared", solver),
                create_count: count,
                backend_name: "actual",
            })
            .unwrap();
        registry
            .set_observation(observation("org.example.circuit", 5, 10_000))
            .unwrap();

        let request = SimulationRequest::new("run-2", EngineeringDomain::Electrical, solver, "test");
        let err = registry
            .run(
                &request,
                RoutingConstraints {
                    maximum_effect: EffectClass::Pure,
                    minimum_trust: TrustLevel::Trusted,
                    ..RoutingConstraints::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, LazySimulationError::BackendNameMismatch { .. }));
    }

    #[test]
    fn descriptor_must_advertise_capability_for_each_solver() {
        let solver = SolverKind::Process;
        let mut descriptor = descriptor("org.example.process", "process", solver);
        descriptor.manifest.provides.clear();
        assert_eq!(
            descriptor.validate(),
            Err(DescriptorProblem::MissingSolverCapability(solver))
        );
    }
}
