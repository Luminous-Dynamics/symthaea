// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Solver-neutral contracts for binding typed passive-device interfaces to a
//! concrete solver boundary.
//!
//! The adapter layer deliberately owns the vendor/solver-specific handle.
//! Symthaea owns the invariant that the handle was bound to the exact
//! PortInterface identity and the exact realized boundary geometry identity.
//!
//! This crate does not run a solver and does not certify physical transport.

use blake3::Hasher;
use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_passive_void_compiler::{BoundaryConditionDomain, PortInterface, SolverBoundaryIdentity};
use symthaea_passive_void_graph::PortId;

/// Opaque identity for the realized boundary patch selected from the candidate
/// mesh. A solver adapter creates this from its actual boundary entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RealizedBoundaryIdentity {
    pub candidate_geometry_digest: [u8; 32],
    pub boundary_patch_digest: [u8; 32],
}

/// Solver-specific adapter binding to an external boundary entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverBoundaryBinding {
    pub port: PortId,
    pub interface_digest: [u8; 32],
    pub solver_boundary: SolverBoundaryIdentity,
    pub adapter_id: String,
    pub external_boundary_handle: String,
    pub realized_boundary: RealizedBoundaryIdentity,
    pub solver_binding_verified: bool,
    pub physical_transport_unproven: bool,
}

impl SolverBoundaryBinding {
    pub fn verified(
        interface: &PortInterface,
        adapter_id: impl Into<String>,
        external_boundary_handle: impl Into<String>,
        realized_boundary: RealizedBoundaryIdentity,
    ) -> Result<Self, SolverBindingError> {
        let adapter_id = adapter_id.into();
        let external_boundary_handle = external_boundary_handle.into();

        interface
            .validate(0.001)
            .map_err(|_| SolverBindingError::InvalidInterface)?;

        if adapter_id.trim().is_empty() {
            return Err(SolverBindingError::EmptyAdapterId);
        }
        if external_boundary_handle.trim().is_empty() {
            return Err(SolverBindingError::EmptyExternalBoundaryHandle);
        }

        Ok(Self {
            port: interface.port,
            interface_digest: interface.digest(),
            solver_boundary: interface.solver_boundary,
            adapter_id,
            external_boundary_handle,
            realized_boundary,
            solver_binding_verified: true,
            physical_transport_unproven: true,
        })
    }

    /// Re-check that a binding still corresponds to the exact interface.
    pub fn validate_against(
        &self,
        interface: &PortInterface,
    ) -> Result<(), SolverBindingError> {
        if !self.solver_binding_verified {
            return Err(SolverBindingError::UnverifiedBinding);
        }
        if !self.physical_transport_unproven {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        if self.port != interface.port {
            return Err(SolverBindingError::PortMismatch);
        }
        if self.interface_digest != interface.digest() {
            return Err(SolverBindingError::InterfaceDigestMismatch);
        }
        if self.solver_boundary != interface.solver_boundary {
            return Err(SolverBindingError::SolverBoundaryMismatch);
        }
        Ok(())
    }

    /// Stable identity of the complete binding statement.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"passive-solver-boundary-binding:v1");
        hasher.update(&self.port.0.to_le_bytes());
        hasher.update(&self.interface_digest);
        hasher.update(&[domain_byte(self.solver_boundary.domain)]);
        hasher.update(&self.solver_boundary.id.to_le_bytes());
        hasher.update(self.adapter_id.as_bytes());
        hasher.update(&[0]);
        hasher.update(self.external_boundary_handle.as_bytes());
        hasher.update(&[0]);
        hasher.update(&self.realized_boundary.candidate_geometry_digest);
        hasher.update(&self.realized_boundary.boundary_patch_digest);
        hasher.update(&[u8::from(self.solver_binding_verified)]);
        hasher.update(&[u8::from(self.physical_transport_unproven)]);
        *hasher.finalize().as_bytes()
    }
}

/// Contract implemented by concrete solver adapters.
///
/// The adapter must resolve the typed interface against the actual candidate
/// geometry and return an opaque handle plus the realized-boundary digest.
/// It must not reinterpret geometry based on a stale port position or name.
pub trait SolverBoundaryBindingAdapter {
    fn adapter_id(&self) -> &str;

    fn bind(
        &self,
        interface: &PortInterface,
        candidate: &TriangleMesh,
        candidate_geometry_digest: [u8; 32],
    ) -> Result<SolverBoundaryBinding, SolverBindingError>;
}

/// Deterministic registry-level validation for a complete set of bindings.
pub fn validate_binding_set(
    interfaces: &[PortInterface],
    bindings: &[SolverBoundaryBinding],
) -> Result<(), SolverBindingError> {
    if interfaces.len() != bindings.len() {
        return Err(SolverBindingError::BindingCountMismatch);
    }

    let mut interface_digests = std::collections::BTreeSet::new();
    let mut binding_digests = std::collections::BTreeSet::new();
    let mut handles = std::collections::BTreeSet::new();

    for interface in interfaces {
        if !interface_digests.insert(interface.digest()) {
            return Err(SolverBindingError::DuplicateInterface(interface.port));
        }
    }

    for binding in bindings {
        if !binding.solver_binding_verified {
            return Err(SolverBindingError::UnverifiedBinding);
        }
        if !binding.physical_transport_unproven {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        if !binding_digests.insert(binding.digest()) {
            return Err(SolverBindingError::DuplicateBinding(binding.port));
        }
        if !handles.insert((
            binding.solver_boundary,
            binding.external_boundary_handle.clone(),
        )) {
            return Err(SolverBindingError::DuplicateExternalBoundaryHandle(
                binding.external_boundary_handle.clone(),
            ));
        }

        let interface = interfaces
            .iter()
            .find(|interface| interface.port == binding.port)
            .ok_or(SolverBindingError::PortMismatch)?;
        binding.validate_against(interface)?;
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverBindingError {
    InvalidInterface,
    EmptyAdapterId,
    EmptyExternalBoundaryHandle,
    UnverifiedBinding,
    InvalidEvidenceState,
    EmptyCandidateGeometryDigest,
    EmptyBoundaryPatchDigest,
    PortMismatch,
    InterfaceDigestMismatch,
    SolverBoundaryMismatch,
    BindingCountMismatch,
    DuplicateInterface(PortId),
    DuplicateBinding(PortId),
    DuplicateExternalBoundaryHandle(String),
}

const fn domain_byte(domain: BoundaryConditionDomain) -> u8 {
    match domain {
        BoundaryConditionDomain::Mechanical => 0,
        BoundaryConditionDomain::Fluidic => 1,
        BoundaryConditionDomain::Thermal => 2,
        BoundaryConditionDomain::Acoustic => 3,
        BoundaryConditionDomain::Electromagnetic => 4,
        BoundaryConditionDomain::Optical => 5,
        BoundaryConditionDomain::Chemical => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_passive_void_compiler::{
        BoundaryConditionDomain, InterfacePlane, PortAperture, SolverBoundaryIdentity,
    };

    fn interface(port: PortId, solver_id: u32) -> PortInterface {
        PortInterface::new(
            port,
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: solver_id,
            },
        )
        .unwrap()
    }

    fn realized(tag: u8) -> RealizedBoundaryIdentity {
        RealizedBoundaryIdentity {
            candidate_geometry_digest: [tag; 32],
            boundary_patch_digest: [tag.wrapping_add(1); 32],
        }
    }

    #[test]
    fn adapter_contract_binds_against_the_actual_candidate_mesh() {
        let interface = interface(PortId(10), 7);
        let binding = FixtureAdapter
            .bind(&interface, &candidate(), [7; 32])
            .unwrap();

        assert_eq!(binding.port, PortId(10));
        assert_eq!(binding.external_boundary_handle, "fixture:boundary-7");
        assert_eq!(binding.realized_boundary.candidate_geometry_digest, [7; 32]);
    }

    #[test]
    fn verified_binding_carries_exact_interface_identity() {
        let interface = interface(PortId(10), 7);
        let binding =
            SolverBoundaryBinding::verified(&interface, "test-adapter/v1", "patch:inlet", realized(1))
                .unwrap();

        assert!(binding.validate_against(&interface).is_ok());
        assert!(binding.solver_binding_verified);
        assert!(binding.physical_transport_unproven);
    }

    #[test]
    fn interface_drift_is_rejected() {
        let interface = interface(PortId(10), 7);
        let binding =
            SolverBoundaryBinding::verified(&interface, "test-adapter/v1", "patch:inlet", realized(1))
                .unwrap();

        let drifted = interface(PortId(10), 8);
        assert_eq!(
            binding.validate_against(&drifted),
            Err(SolverBindingError::InterfaceDigestMismatch)
        );
    }

    #[test]
    fn duplicate_external_handle_is_rejected() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let bindings = vec![
            SolverBoundaryBinding::verified(&a, "test-adapter/v1", "patch:shared", realized(1))
                .unwrap(),
            SolverBoundaryBinding::verified(&b, "test-adapter/v1", "patch:shared", realized(2))
                .unwrap(),
        ];
        assert_eq!(
            validate_binding_set(&[a, b], &bindings),
            Err(SolverBindingError::DuplicateExternalBoundaryHandle(
                "patch:shared".into()
            ))
        );
    }

    #[test]
    fn binding_set_rejects_wrong_solver_identity() {
        let a = interface(PortId(10), 7);
        let binding =
            SolverBoundaryBinding::verified(&a, "test-adapter/v1", "patch:inlet", realized(1))
                .unwrap();
        let b = interface(PortId(10), 8);
        assert_eq!(
            binding.validate_against(&b),
            Err(SolverBindingError::InterfaceDigestMismatch)
        );
    }
}
