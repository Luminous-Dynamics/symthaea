// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physical-agency composition primitives for Symthaea.
//!
//! PA-11 remains deliberately pre-execution. This crate can negotiate declared
//! simulator capabilities, preserve multi-objective candidate frontiers, bind
//! deliberation to immutable world snapshots, and validate strict typed-context
//! simulation lineage, but it cannot construct actuator commands, depend on HAL,
//! or mint physical execution authority.
//!
//! Capability manifests are declarations used to choose a suitable modelling
//! path. They are **not** safety evidence and cannot discharge execution gates.
//!
//! The loose PA-04 qualifier and its narrower evidence binding are crate-private.
//! Public qualification must consume a non-serializable selection receipt
//! produced from an evaluated Pareto frontier plus a binding to the exact world
//! snapshot used for deliberation.

#![deny(unsafe_code)]

pub mod deliberation;
pub mod portfolio;
pub mod strict_context;
mod qualification;
mod qualification_lineage;

pub use deliberation::WorldSnapshotRef;
pub use qualification::{
    SimulationQualificationError,
    VerifiedSimulationEvidence as RegistryValidatedSimulationEvidence,
    execute_verified_simulation as execute_registry_validated_simulation,
};
pub use qualification_lineage::{
    DeliberationBoundSimulationCandidate, DeliberationQualificationError,
    DeliberationSimulationBinding, qualify_selected_simulation_candidate,
};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_sim_bridge::{SimulationBackend, SolverKind};
use thiserror::Error;

/// Planning-relevant capabilities a simulator backend may explicitly declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendCapability {
    UncertaintyQuantification,
    Gradients,
    Jacobians,
    Adjoints,
    SystemIdentification,
    BatchedCounterfactuals,
    MultiPhysics,
}

/// Fail-closed capability set.
///
/// Every field defaults to false so an old, absent, malformed, or unknown
/// declaration never silently acquires a new modelling capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BackendCapabilities {
    pub uncertainty_quantification: bool,
    pub gradients: bool,
    pub jacobians: bool,
    pub adjoints: bool,
    pub system_identification: bool,
    pub batched_counterfactuals: bool,
    pub multiphysics: bool,
}

impl BackendCapabilities {
    /// Whether this declaration claims support for a capability.
    pub fn supports(self, capability: BackendCapability) -> bool {
        match capability {
            BackendCapability::UncertaintyQuantification => self.uncertainty_quantification,
            BackendCapability::Gradients => self.gradients,
            BackendCapability::Jacobians => self.jacobians,
            BackendCapability::Adjoints => self.adjoints,
            BackendCapability::SystemIdentification => self.system_identification,
            BackendCapability::BatchedCounterfactuals => self.batched_counterfactuals,
            BackendCapability::MultiPhysics => self.multiphysics,
        }
    }
}

/// Serializable planning-time declaration for one simulator backend.
///
/// This is intentionally named a *manifest*, not an attestation or permit.
/// A positive bit means "declared available for planning" only. Later safety
/// qualification must rely on independently checked evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilityManifest {
    /// Must match [`SimulationBackend::name`].
    pub backend_name: String,
    /// Solver families this declaration applies to.
    pub supported_solvers: Vec<SolverKind>,
    /// Explicitly declared capabilities. Undeclared fields remain false.
    pub capabilities: BackendCapabilities,
    /// Human/audit-readable origin for the declaration, for example a checked
    /// adapter contract, solver manual version, or qualification record path.
    pub declaration_provenance: String,
}

impl BackendCapabilityManifest {
    pub fn validate(&self) -> Result<(), CapabilityError> {
        if self.backend_name.trim().is_empty() {
            return Err(CapabilityError::EmptyField("manifest.backend_name"));
        }
        if self.supported_solvers.is_empty() {
            return Err(CapabilityError::NoSupportedSolvers);
        }
        if self.declaration_provenance.trim().is_empty() {
            return Err(CapabilityError::EmptyField(
                "manifest.declaration_provenance",
            ));
        }
        Ok(())
    }

    /// Validate that the declaration is attached to the backend it names and
    /// does not claim a solver family the backend itself does not expose.
    ///
    /// This checks identity/shape only. It does not turn the manifest into
    /// engineering or execution evidence.
    pub fn validate_against_backend(
        &self,
        backend: &dyn SimulationBackend,
    ) -> Result<(), CapabilityError> {
        self.validate()?;
        if self.backend_name != backend.name() {
            return Err(CapabilityError::BackendIdentityMismatch {
                manifest: self.backend_name.clone(),
                backend: backend.name().to_string(),
            });
        }
        for solver in &self.supported_solvers {
            if !backend.supported_solvers().contains(solver) {
                return Err(CapabilityError::SolverClaimMismatch(*solver));
            }
        }
        Ok(())
    }
}

/// Capabilities required by a planning operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequirement {
    pub solver: SolverKind,
    pub required: Vec<BackendCapability>,
}

impl CapabilityRequirement {
    pub fn new(solver: SolverKind) -> Self {
        Self {
            solver,
            required: Vec::new(),
        }
    }

    pub fn requiring(mut self, capability: BackendCapability) -> Self {
        if !self.required.contains(&capability) {
            self.required.push(capability);
        }
        self
    }
}

/// Result of planning-time capability negotiation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDecision {
    pub backend_name: String,
    pub solver: SolverKind,
    pub accepted: bool,
    pub missing: Vec<BackendCapability>,
}

impl CapabilityDecision {
    pub fn is_accepted(&self) -> bool {
        self.accepted
    }
}

/// Registry of explicit backend declarations.
///
/// The registry refuses duplicate names instead of silently replacing an
/// existing declaration. This prevents a later untrusted/accidental manifest
/// from widening planning assumptions by overwrite.
#[derive(Debug, Default)]
pub struct CapabilityCatalog {
    manifests: BTreeMap<String, BackendCapabilityManifest>,
}

impl CapabilityCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, manifest: BackendCapabilityManifest) -> Result<(), CapabilityError> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.backend_name) {
            return Err(CapabilityError::DuplicateBackend(manifest.backend_name));
        }
        self.manifests.insert(manifest.backend_name.clone(), manifest);
        Ok(())
    }

    pub fn manifest(&self, backend_name: &str) -> Option<&BackendCapabilityManifest> {
        self.manifests.get(backend_name)
    }

    /// Negotiate against an explicit manifest and a live backend identity.
    ///
    /// Unknown backends, identity mismatches, undeclared solver families, and
    /// missing capabilities all fail closed. This function never calls
    /// [`SimulationBackend::run`].
    pub fn negotiate(
        &self,
        backend: &dyn SimulationBackend,
        requirement: &CapabilityRequirement,
    ) -> Result<CapabilityDecision, CapabilityError> {
        let manifest = self
            .manifests
            .get(backend.name())
            .ok_or_else(|| CapabilityError::UnknownBackend(backend.name().to_string()))?;

        manifest.validate_against_backend(backend)?;

        if !manifest.supported_solvers.contains(&requirement.solver) {
            return Ok(CapabilityDecision {
                backend_name: backend.name().to_string(),
                solver: requirement.solver,
                accepted: false,
                missing: requirement.required.clone(),
            });
        }

        let missing = requirement
            .required
            .iter()
            .copied()
            .filter(|capability| !manifest.capabilities.supports(*capability))
            .collect::<Vec<_>>();

        Ok(CapabilityDecision {
            backend_name: backend.name().to_string(),
            solver: requirement.solver,
            accepted: missing.is_empty(),
            missing,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CapabilityError {
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("capability manifest must declare at least one solver family")]
    NoSupportedSolvers,
    #[error("capability catalog already contains backend {0:?}")]
    DuplicateBackend(String),
    #[error("no capability manifest is registered for backend {0:?}")]
    UnknownBackend(String),
    #[error("manifest backend {manifest:?} does not match live backend {backend:?}")]
    BackendIdentityMismatch { manifest: String, backend: String },
    #[error("manifest claims solver family not exposed by live backend: {0:?}")]
    SolverClaimMismatch(SolverKind),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{SimulationError, SimulationRequest, SimulationResult};

    #[derive(Debug)]
    struct MockFeaBackend;

    impl SimulationBackend for MockFeaBackend {
        fn name(&self) -> &'static str {
            "mock-fea"
        }

        fn supported_solvers(&self) -> &[SolverKind] {
            &[SolverKind::FiniteElement]
        }

        fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
            Ok(SimulationResult::dry_run(&request.id, self.name(), 1.0))
        }
    }

    fn manifest(capabilities: BackendCapabilities) -> BackendCapabilityManifest {
        BackendCapabilityManifest {
            backend_name: "mock-fea".into(),
            supported_solvers: vec![SolverKind::FiniteElement],
            capabilities,
            declaration_provenance: "unit-test adapter contract".into(),
        }
    }

    #[test]
    fn default_capabilities_are_all_fail_closed() {
        let capabilities = BackendCapabilities::default();
        for capability in [
            BackendCapability::UncertaintyQuantification,
            BackendCapability::Gradients,
            BackendCapability::Jacobians,
            BackendCapability::Adjoints,
            BackendCapability::SystemIdentification,
            BackendCapability::BatchedCounterfactuals,
            BackendCapability::MultiPhysics,
        ] {
            assert!(!capabilities.supports(capability));
        }
    }

    #[test]
    fn unknown_backend_is_rejected() {
        let catalog = CapabilityCatalog::new();
        let requirement = CapabilityRequirement::new(SolverKind::FiniteElement);
        let error = catalog.negotiate(&MockFeaBackend, &requirement).unwrap_err();
        assert_eq!(error, CapabilityError::UnknownBackend("mock-fea".into()));
    }

    #[test]
    fn explicit_capability_can_satisfy_planning_requirement() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register(manifest(BackendCapabilities {
                gradients: true,
                ..BackendCapabilities::default()
            }))
            .unwrap();

        let requirement = CapabilityRequirement::new(SolverKind::FiniteElement)
            .requiring(BackendCapability::Gradients);
        let decision = catalog.negotiate(&MockFeaBackend, &requirement).unwrap();
        assert!(decision.is_accepted());
        assert!(decision.missing.is_empty());
    }

    #[test]
    fn missing_capability_is_explicitly_reported() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register(manifest(BackendCapabilities::default()))
            .unwrap();

        let requirement = CapabilityRequirement::new(SolverKind::FiniteElement)
            .requiring(BackendCapability::SystemIdentification)
            .requiring(BackendCapability::UncertaintyQuantification);
        let decision = catalog.negotiate(&MockFeaBackend, &requirement).unwrap();

        assert!(!decision.is_accepted());
        assert_eq!(decision.missing.len(), 2);
        assert!(decision
            .missing
            .contains(&BackendCapability::SystemIdentification));
        assert!(decision
            .missing
            .contains(&BackendCapability::UncertaintyQuantification));
    }

    #[test]
    fn duplicate_manifest_cannot_silently_widen_capabilities() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register(manifest(BackendCapabilities::default()))
            .unwrap();

        let widened = manifest(BackendCapabilities {
            gradients: true,
            adjoints: true,
            ..BackendCapabilities::default()
        });
        assert_eq!(
            catalog.register(widened),
            Err(CapabilityError::DuplicateBackend("mock-fea".into()))
        );
    }

    #[test]
    fn manifest_must_match_live_backend_solver_surface() {
        let bad = BackendCapabilityManifest {
            backend_name: "mock-fea".into(),
            supported_solvers: vec![SolverKind::ComputationalFluidDynamics],
            capabilities: BackendCapabilities::default(),
            declaration_provenance: "bad test declaration".into(),
        };

        assert_eq!(
            bad.validate_against_backend(&MockFeaBackend),
            Err(CapabilityError::SolverClaimMismatch(
                SolverKind::ComputationalFluidDynamics
            ))
        );
    }

    #[test]
    fn capability_negotiation_never_runs_backend() {
        #[derive(Debug)]
        struct PanicBackend;

        impl SimulationBackend for PanicBackend {
            fn name(&self) -> &'static str {
                "panic-backend"
            }

            fn supported_solvers(&self) -> &[SolverKind] {
                &[SolverKind::FiniteElement]
            }

            fn run(
                &self,
                _request: &SimulationRequest,
            ) -> Result<SimulationResult, SimulationError> {
                panic!("capability negotiation must not execute the simulator")
            }
        }

        let mut catalog = CapabilityCatalog::new();
        catalog
            .register(BackendCapabilityManifest {
                backend_name: "panic-backend".into(),
                supported_solvers: vec![SolverKind::FiniteElement],
                capabilities: BackendCapabilities::default(),
                declaration_provenance: "test".into(),
            })
            .unwrap();

        let requirement = CapabilityRequirement::new(SolverKind::FiniteElement);
        let decision = catalog.negotiate(&PanicBackend, &requirement).unwrap();
        assert!(decision.is_accepted());
    }
}
