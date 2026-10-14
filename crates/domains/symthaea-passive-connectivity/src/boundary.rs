// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit boundary-policy evidence for passive ported geometries.
//!
//! The baseline fabrication validator is intentionally a closed-solid gate.
//! Ported devices are different because an inlet or outlet can be an intentional
//! boundary. The typed path below therefore matches the observed boundary to an
//! explicit aperture and interface plane instead of accepting any opening that
//! merely falls inside a large anchor radius.
//!
//! The solver-boundary identity and outward normal remain intent/provenance fields;
//! geometry alone does not prove that a solver applied the boundary condition.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_passive_solver_binding::{select_boundary_patch, BoundaryEdgeKey};
use symthaea_fabrication_kernel::validate::validate_mesh;
use symthaea_passive_void_compiler::{GeometryEmbedding, PortInterface};
use symthaea_passive_void_graph::PortId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryValidationStatus {
    InvalidMesh,
    Closed,
    ExpectedOpeningsOnly,
    UnexpectedOpenings,
    MissingPortAnchor(PortId),
    AmbiguousOpening,
    MissingPortOpening(PortId),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortBoundaryPolicy {
    allowed_open_ports: BTreeSet<PortId>,
    required_open_ports: BTreeSet<PortId>,
    typed_interfaces: BTreeMap<PortId, PortInterface>,
    tolerance_micrometers: u64,
}

impl Default for PortBoundaryPolicy {
    fn default() -> Self {
        Self {
            allowed_open_ports: BTreeSet::new(),
            required_open_ports: BTreeSet::new(),
            typed_interfaces: BTreeMap::new(),
            tolerance_micrometers: 50,
        }
    }
}

impl PortBoundaryPolicy {
    pub fn closed() -> Self {
        Self::default()
    }

    pub fn with_allowed_open_port(mut self, port: PortId) -> Self {
        self.allowed_open_ports.insert(port);
        self
    }

    pub fn with_required_open_port(mut self, port: PortId) -> Self {
        self.allowed_open_ports.insert(port);
        self.required_open_ports.insert(port);
        self
    }

    /// Permit an opening described by the complete typed interface identity.
    pub fn with_allowed_port_interface(mut self, interface: PortInterface) -> Self {
        self.allowed_open_ports.insert(interface.port);
        self.typed_interfaces.insert(interface.port, interface);
        self
    }

    /// Require an opening described by the complete typed interface identity.
    pub fn with_required_port_interface(mut self, interface: PortInterface) -> Self {
        self.allowed_open_ports.insert(interface.port);
        self.required_open_ports.insert(interface.port);
        self.typed_interfaces.insert(interface.port, interface);
        self
    }

    pub fn required_open_ports(&self) -> impl Iterator<Item = PortId> + '_ {
        self.required_open_ports.iter().copied()
    }

    pub fn allowed_open_ports(&self) -> impl Iterator<Item = PortId> + '_ {
        self.allowed_open_ports.iter().copied()
    }

    pub fn typed_interfaces(&self) -> impl Iterator<Item = &PortInterface> {
        self.typed_interfaces.values()
    }

    pub fn tolerance_mm(mut self, tolerance_mm: f32) -> Option<Self> {
        if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
            return None;
        }
        let tolerance_micrometers = tolerance_mm as f64 * 1_000.0;
        if tolerance_micrometers > u64::MAX as f64 {
            return None;
        }
        self.tolerance_micrometers = tolerance_micrometers.round() as u64;
        Some(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortBoundaryEvidence {
    pub status: BoundaryValidationStatus,
    pub boundary_edges: usize,
    pub matched_boundary_edges: usize,
    pub unexpected_boundary_edges: usize,
    pub ambiguous_boundary_edges: usize,
    pub missing_port_anchors: Vec<PortId>,
    /// Geometry/topology evidence only; transport remains unproven.
    pub physical_transport_unproven: bool,
}

impl PortBoundaryEvidence {
    pub fn evaluate(
        candidate: &TriangleMesh,
        embedding: &GeometryEmbedding,
        policy: &PortBoundaryPolicy,
    ) -> Self {
        let report = validate_mesh(candidate);
        if !report.is_valid() {
            return Self {
                status: BoundaryValidationStatus::InvalidMesh,
                boundary_edges: report.boundary_edges,
                matched_boundary_edges: 0,
                unexpected_boundary_edges: report.boundary_edges,
                ambiguous_boundary_edges: 0,
                missing_port_anchors: Vec::new(),
                physical_transport_unproven: true,
            };
        }

        if report.boundary_edges == 0 {
            return Self {
                status: BoundaryValidationStatus::Closed,
                boundary_edges: 0,
                matched_boundary_edges: 0,
                unexpected_boundary_edges: 0,
                ambiguous_boundary_edges: 0,
                missing_port_anchors: Vec::new(),
                physical_transport_unproven: true,
            };
        }

        if policy.allowed_open_ports.is_empty() {
            return Self {
                status: BoundaryValidationStatus::UnexpectedOpenings,
                boundary_edges: report.boundary_edges,
                matched_boundary_edges: 0,
                unexpected_boundary_edges: report.boundary_edges,
                ambiguous_boundary_edges: 0,
                missing_port_anchors: Vec::new(),
                physical_transport_unproven: true,
            };
        }

        let boundary_edges = collect_boundary_edges(candidate);
        let mut missing_port_anchors = Vec::new();
        let mut legacy_ports = Vec::new();
        let mut typed_edge_owners: BTreeMap<BoundaryEdgeKey, BTreeSet<PortId>> = BTreeMap::new();
        let tolerance_mm = policy.tolerance_micrometers as f64 / 1_000.0;

        for port in &policy.allowed_open_ports {
            match policy
                .typed_interfaces
                .get(port)
                .or_else(|| embedding.interfaces.get(port))
            {
                Some(interface) => match interface.validate(0.001) {
                    Ok(()) => {
                        if let Ok(selection) =
                            select_boundary_patch(interface, candidate, tolerance_mm)
                        {
                            for edge in selection.edges() {
                                typed_edge_owners
                                    .entry(*edge)
                                    .or_default()
                                    .insert(*port);
                            }
                        }
                    }
                    Err(_) => missing_port_anchors.push(*port),
                },
                None => match embedding.ports.get(port) {
                    Some(anchor)
                        if anchor.center_mm.iter().all(|value| value.is_finite())
                            && anchor.radius_mm.is_finite()
                            && anchor.radius_mm > 0.0 =>
                    {
                        legacy_ports.push(PortBoundaryGeometry::Legacy {
                            port: *port,
                            center_mm: anchor.center_mm,
                            radius_mm: anchor.radius_mm,
                        });
                    }
                    None => missing_port_anchors.push(*port),
                    Some(_) => missing_port_anchors.push(*port),
                },
            }
        }

        if let Some(port) = missing_port_anchors.first().copied() {
            return Self {
                status: BoundaryValidationStatus::MissingPortAnchor(port),
                boundary_edges: boundary_edges.len(),
                matched_boundary_edges: 0,
                unexpected_boundary_edges: boundary_edges.len(),
                ambiguous_boundary_edges: 0,
                missing_port_anchors,
                physical_transport_unproven: true,
            };
        }

        let mut matched = 0usize;
        let mut unexpected = 0usize;
        let mut ambiguous = 0usize;
        let mut matched_by_port = BTreeSet::new();

        for (edge_key, midpoint) in boundary_edges {
            let mut matches = typed_edge_owners
                .get(&edge_key)
                .cloned()
                .unwrap_or_default();
            for port_geometry in &legacy_ports {
                if port_geometry.matches(midpoint, tolerance_mm) {
                    matches.insert(port_geometry.port());
                }
            }

            match matches.len() {
                0 => unexpected += 1,
                1 => {
                    matched += 1;
                    matched_by_port.extend(matches);
                }
                _ => ambiguous += 1,
            }
        }

        if let Some(port) = policy
            .required_open_ports
            .iter()
            .find(|port| !matched_by_port.contains(port))
            .copied()
        {
            return Self {
                status: BoundaryValidationStatus::MissingPortOpening(port),
                boundary_edges: matched + unexpected + ambiguous,
                matched_boundary_edges: matched,
                unexpected_boundary_edges: unexpected,
                ambiguous_boundary_edges: ambiguous,
                missing_port_anchors,
                physical_transport_unproven: true,
            };
        }

        let status = if ambiguous > 0 {
            BoundaryValidationStatus::AmbiguousOpening
        } else if unexpected > 0 {
            BoundaryValidationStatus::UnexpectedOpenings
        } else {
            BoundaryValidationStatus::ExpectedOpeningsOnly
        };

        Self {
            status,
            boundary_edges: matched + unexpected + ambiguous,
            matched_boundary_edges: matched,
            unexpected_boundary_edges: unexpected,
            ambiguous_boundary_edges: ambiguous,
            missing_port_anchors,
            physical_transport_unproven: true,
        }
    }

    pub fn is_admissible(&self) -> bool {
        matches!(
            self.status,
            BoundaryValidationStatus::Closed | BoundaryValidationStatus::ExpectedOpeningsOnly
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum PortBoundaryGeometry {
    Legacy {
        port: PortId,
        center_mm: [f32; 3],
        radius_mm: f32,
    },
}

impl PortBoundaryGeometry {
    fn port(self) -> PortId {
        match self {
            Self::Legacy { port, .. } => port,
        }
    }

    fn matches(self, midpoint: [f64; 3], tolerance_mm: f64) -> bool {
        match self {
            Self::Legacy {
                center_mm,
                radius_mm,
                ..
            } => distance_mm(midpoint, center_mm) <= radius_mm as f64 + tolerance_mm,
        }
    }
}

type QuantizedPoint = [i64; 3];
type QuantizedEdge = (QuantizedPoint, QuantizedPoint);

fn quantize(point: [f32; 3]) -> QuantizedPoint {
    [
        (point[0] as f64 * 1_000_000.0).round() as i64,
        (point[1] as f64 * 1_000_000.0).round() as i64,
        (point[2] as f64 * 1_000_000.0).round() as i64,
    ]
}

fn edge_key(a: QuantizedPoint, b: QuantizedPoint) -> QuantizedEdge {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn collect_boundary_edges(mesh: &TriangleMesh) -> Vec<(BoundaryEdgeKey, [f64; 3])> {
    let mut edges: HashMap<QuantizedEdge, [f64; 3]> = HashMap::new();
    let mut counts: HashMap<QuantizedEdge, usize> = HashMap::new();

    for triangle in &mesh.indices {
        if triangle
            .iter()
            .any(|index| (*index as usize) >= mesh.vertices.len())
        {
            continue;
        }

        let vertices = [
            mesh.vertices[triangle[0] as usize],
            mesh.vertices[triangle[1] as usize],
            mesh.vertices[triangle[2] as usize],
        ];

        for (a, b) in [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ] {
            let qa = quantize(a);
            let qb = quantize(b);
            let key = edge_key(qa, qb);
            edges.entry(key).or_insert(midpoint(a, b));
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut boundary = Vec::new();
    for ((a, b), center) in edges {
        if counts.get(&(a, b)).copied().unwrap_or(0) == 1 {
            boundary.push((BoundaryEdgeKey { a, b }, center));
        }
    }
    boundary
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        (a[0] as f64 + b[0] as f64) / 2.0,
        (a[1] as f64 + b[1] as f64) / 2.0,
        (a[2] as f64 + b[2] as f64) / 2.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::CSGNode;
    use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
    use symthaea_passive_void_compiler::{
        BoundaryConditionDomain, InterfacePlane, PortAperture, SolverBoundaryIdentity,
    };

    fn typed_interface(port: PortId, center_mm: [f32; 3], normal: [f32; 3], radius_mm: f32) -> PortInterface {
        PortInterface::new(
            port,
            center_mm,
            PortAperture::Circular { radius_mm },
            normal,
            InterfacePlane::new(center_mm, normal).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: port.0,
            },
        )
        .unwrap()
    }

    #[test]
    fn closed_cube_passes_closed_policy() {
        let embedding = GeometryEmbedding::default();
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &embedding,
            &PortBoundaryPolicy::closed(),
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::Closed);
        assert!(evidence.is_admissible());
    }

    #[test]
    fn open_surface_is_rejected_without_explicit_port_policy() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &GeometryEmbedding::default(),
            &PortBoundaryPolicy::closed(),
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::UnexpectedOpenings);
        assert!(!evidence.is_admissible());
    }

    #[test]
    fn typed_circular_interface_matches_its_rim() {
        let mesh = TriangleMesh {
            vertices: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let interface = typed_interface(
            PortId(10),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let policy = PortBoundaryPolicy::closed()
            .with_required_port_interface(interface)
            .tolerance_mm(0.05)
            .unwrap();
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &GeometryEmbedding::default().with_port_interface(interface),
            &policy,
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::ExpectedOpeningsOnly);
        assert!(evidence.is_admissible());
    }

    #[test]
    fn typed_interface_rejects_near_size_hole_inside_declared_aperture() {
        let mesh = TriangleMesh {
            vertices: vec![
                [0.85, 0.0, 0.0],
                [0.0, 0.85, 0.0],
                [-0.85, 0.0, 0.0],
                [0.0, -0.85, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            indices: vec![
                [0, 1, 2],
                [0, 2, 3],
            ],
        };
        let interface = typed_interface(
            PortId(10),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let policy = PortBoundaryPolicy::closed()
            .with_required_port_interface(interface)
            .tolerance_mm(0.05)
            .unwrap();
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &GeometryEmbedding::default().with_port_interface(interface),
            &policy,
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::UnexpectedOpenings);
        assert!(!evidence.is_admissible());
    }

    #[test]
    fn typed_interface_rejects_smaller_hole_inside_large_aperture() {
        let mesh = TriangleMesh {
            vertices: vec![
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [-0.5, 0.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let interface = typed_interface(
            PortId(10),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let policy = PortBoundaryPolicy::closed()
            .with_required_port_interface(interface)
            .tolerance_mm(0.05)
            .unwrap();
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &GeometryEmbedding::default().with_port_interface(interface),
            &policy,
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::UnexpectedOpenings);
        assert!(!evidence.is_admissible());
    }

    #[test]
    fn typed_interface_rejects_wrong_plane() {
        let mesh = TriangleMesh {
            vertices: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let mut interface = typed_interface(
            PortId(10),
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        interface.position_mm = [0.0, 0.0, 1.0];
        let policy = PortBoundaryPolicy::closed()
            .with_required_port_interface(interface)
            .tolerance_mm(0.05)
            .unwrap();
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &GeometryEmbedding::default().with_port_interface(interface),
            &policy,
        );
        assert_eq!(evidence.status, BoundaryValidationStatus::MissingPortOpening(PortId(10)));
    }

    #[test]
    fn default_tolerance_is_fifty_micrometers() {
        assert_eq!(
            PortBoundaryPolicy::closed().tolerance_micrometers,
            50
        );
    }

    #[test]
    fn tolerance_builder_rejects_overflow() {
        assert!(PortBoundaryPolicy::closed().tolerance_mm(f32::MAX).is_none());
    }

    #[test]
    fn tolerance_builder_rejects_invalid_values() {
        assert!(PortBoundaryPolicy::closed().tolerance_mm(-1.0).is_none());
        assert!(PortBoundaryPolicy::closed().tolerance_mm(f32::NAN).is_none());
    }
}
