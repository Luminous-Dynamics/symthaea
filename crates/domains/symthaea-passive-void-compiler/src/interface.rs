// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed geometric identities for external passive-device interfaces.
//!
//! A port anchor says where a functional region is represented. A port
//! interface goes further: it defines the aperture, interface plane, outward
//! normal, and the stable solver-boundary identity that will eventually bind
//! the geometry to a physical simulation boundary condition.
//!
//! Geometry matching remains conservative. The types here do not prove that a
//! solver applied the declared boundary condition, nor do they prove transport.

use blake3::Hasher;

use symthaea_passive_void_graph::PortId;

/// Physical solver domain associated with an external boundary identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BoundaryConditionDomain {
    Mechanical,
    Fluidic,
    Thermal,
    Acoustic,
    Electromagnetic,
    Optical,
    Chemical,
}

/// Stable identity for a solver boundary-condition binding.
///
/// The numeric id is intentionally solver-neutral. A solver adapter is
/// responsible for mapping this identity to its concrete boundary-condition
/// handle without changing the design artifact's geometric identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SolverBoundaryIdentity {
    pub domain: BoundaryConditionDomain,
    pub id: u32,
}

/// A geometric plane carrying the interface boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfacePlane {
    pub origin_mm: [f32; 3],
    pub normal_unit: [f32; 3],
}

impl InterfacePlane {
    pub fn new(origin_mm: [f32; 3], normal: [f32; 3]) -> Result<Self, PortInterfaceError> {
        Ok(Self {
            origin_mm,
            normal_unit: normalize(normal)
                .ok_or(PortInterfaceError::InvalidNormal)?,
        })
    }
}

/// Aperture geometry supported by the first interface contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PortAperture {
    Circular { radius_mm: f32 },
}

impl PortAperture {
    pub fn radius_mm(self) -> f32 {
        match self {
            Self::Circular { radius_mm } => radius_mm,
        }
    }
}

/// Complete typed geometric interface for one external port.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortInterface {
    pub port: PortId,
    pub position_mm: [f32; 3],
    pub aperture: PortAperture,
    pub outward_normal_unit: [f32; 3],
    pub interface_plane: InterfacePlane,
    pub solver_boundary: SolverBoundaryIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortInterfaceError {
    NonFinitePosition,
    InvalidAperture,
    InvalidNormal,
    PlaneNormalMismatch,
    PositionOffPlane,
}

impl PortInterface {
    pub fn new(
        port: PortId,
        position_mm: [f32; 3],
        aperture: PortAperture,
        outward_normal: [f32; 3],
        interface_plane: InterfacePlane,
        solver_boundary: SolverBoundaryIdentity,
    ) -> Result<Self, PortInterfaceError> {
        let outward_normal_unit =
            normalize(outward_normal).ok_or(PortInterfaceError::InvalidNormal)?;

        let interface = Self {
            port,
            position_mm,
            aperture,
            outward_normal_unit,
            interface_plane,
            solver_boundary,
        };
        interface.validate(0.001)?;
        Ok(interface)
    }

    pub fn radius_mm(&self) -> f32 {
        self.aperture.radius_mm()
    }

    /// Validate geometry consistency using a positional tolerance in millimetres.
    pub fn validate(&self, tolerance_mm: f32) -> Result<(), PortInterfaceError> {
        if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
            return Err(PortInterfaceError::PositionOffPlane);
        }
        if !self.position_mm.iter().all(|value| value.is_finite())
            || !self.interface_plane.origin_mm.iter().all(|value| value.is_finite())
        {
            return Err(PortInterfaceError::NonFinitePosition);
        }

        let radius = self.radius_mm();
        if !radius.is_finite() || radius <= 0.0 {
            return Err(PortInterfaceError::InvalidAperture);
        }

        let outward = normalize(self.outward_normal_unit)
            .ok_or(PortInterfaceError::InvalidNormal)?;
        let plane_normal = normalize(self.interface_plane.normal_unit)
            .ok_or(PortInterfaceError::InvalidNormal)?;
        let alignment = dot(outward, plane_normal);
        if alignment < 0.999 {
            return Err(PortInterfaceError::PlaneNormalMismatch);
        }

        let delta = sub(self.position_mm, self.interface_plane.origin_mm);
        if dot(delta, plane_normal).abs() > tolerance_mm.max(0.001) {
            return Err(PortInterfaceError::PositionOffPlane);
        }

        Ok(())
    }

    /// Deterministic identity digest for provenance and downstream solver binding.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"passive-port-interface:v1");
        hasher.update(&self.port.0.to_le_bytes());

        for value in self.position_mm {
            hasher.update(&quantize_mm(value).to_le_bytes());
        }
        for value in self.outward_normal_unit {
            hasher.update(&quantize_unit(value).to_le_bytes());
        }
        for value in self.interface_plane.origin_mm {
            hasher.update(&quantize_mm(value).to_le_bytes());
        }
        for value in self.interface_plane.normal_unit {
            hasher.update(&quantize_unit(value).to_le_bytes());
        }

        match self.aperture {
            PortAperture::Circular { radius_mm } => {
                hasher.update(&[0]);
                hasher.update(&quantize_mm(radius_mm).to_le_bytes());
            }
        }

        hasher.update(&[boundary_domain_byte(self.solver_boundary.domain)]);
        hasher.update(&self.solver_boundary.id.to_le_bytes());

        *hasher.finalize().as_bytes()
    }
}

fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    if !vector.iter().all(|value| value.is_finite()) {
        return None;
    }
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !length.is_finite() || length <= 1.0e-9 {
        return None;
    }
    Some([
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    ])
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn quantize_mm(value: f32) -> i64 {
    (value as f64 * 1_000_000.0).round() as i64
}

fn quantize_unit(value: f32) -> i32 {
    (value.clamp(-1.0, 1.0) as f64 * 1_000_000.0).round() as i32
}

const fn boundary_domain_byte(domain: BoundaryConditionDomain) -> u8 {
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

    fn interface() -> PortInterface {
        PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap()
    }

    #[test]
    fn interface_normalizes_normals_and_is_valid() {
        let value = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 4.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap();

        assert!((value.outward_normal_unit[2] - 1.0).abs() < 1.0e-6);
        assert_eq!(value.radius_mm(), 2.0);
    }

    #[test]
    fn interface_rejects_plane_normal_flip() {
        let result = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        );
        assert_eq!(result, Err(PortInterfaceError::PlaneNormalMismatch));
    }

    #[test]
    fn digest_is_stable() {
        assert_eq!(interface().digest(), interface().digest());
    }
}
