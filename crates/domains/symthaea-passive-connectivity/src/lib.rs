// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Mesh-topology realization evidence for passive void candidates.
//!
//! This layer answers only a geometric question: did the compiled void candidate
//! resolve to a valid, watertight, single connected mesh? It does not prove
//! that any fluid, heat, acoustic wave, or field can traverse the structure.

use symthaea_fabrication_kernel::mesh::{resolve_to_mesh, TriangleMesh};
use symthaea_fabrication_kernel::validate::validate_mesh;
use symthaea_fabrication_kernel::csg::CSGNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectivityStatus {
    InvalidMesh,
    Disconnected { components: usize },
    Connected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshConnectivityEvidence {
    pub status: ConnectivityStatus,
    pub connected_components: usize,
    pub watertight: bool,
    pub mesh_valid: bool,
    /// Always true: this observation does not establish physical transport.
    pub physical_transport_unproven: bool,
}

impl MeshConnectivityEvidence {
    pub fn from_mesh(mesh: &TriangleMesh) -> Self {
        let report = validate_mesh(mesh);
        let status = if !report.is_valid() || !report.is_watertight {
            ConnectivityStatus::InvalidMesh
        } else if report.connected_components == 1 {
            ConnectivityStatus::Connected
        } else {
            ConnectivityStatus::Disconnected {
                components: report.connected_components,
            }
        };

        Self {
            status,
            connected_components: report.connected_components,
            watertight: report.is_watertight,
            mesh_valid: report.is_valid(),
            physical_transport_unproven: true,
        }
    }

    pub fn from_csg(candidate: &CSGNode) -> Self {
        let mesh = resolve_to_mesh(candidate);
        Self::from_mesh(&mesh)
    }

    pub fn is_geometrically_connected(&self) -> bool {
        matches!(self.status, ConnectivityStatus::Connected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::Transform3D;

    #[test]
    fn cube_is_one_connected_closed_component() {
        let evidence = MeshConnectivityEvidence::from_csg(&CSGNode::cube());
        assert_eq!(evidence.status, ConnectivityStatus::Connected);
        assert_eq!(evidence.connected_components, 1);
        assert!(evidence.watertight);
        assert!(evidence.mesh_valid);
        assert!(evidence.physical_transport_unproven);
    }

    #[test]
    fn separated_solids_are_reported_as_disconnected() {
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [3.0, 0.0, 0.0],
            ..Default::default()
        }));
        left.merge(&right);
        let evidence = MeshConnectivityEvidence::from_mesh(&left);
        assert_eq!(
            evidence.status,
            ConnectivityStatus::Disconnected { components: 2 }
        );
        assert!(!evidence.is_geometrically_connected());
    }

    #[test]
    fn open_surface_is_not_connectivity_evidence() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let evidence = MeshConnectivityEvidence::from_mesh(&mesh);
        assert_eq!(evidence.status, ConnectivityStatus::InvalidMesh);
        assert!(!evidence.watertight);
    }
}

pub mod port_path;
