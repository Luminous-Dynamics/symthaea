// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Explicit boundary-policy evidence for passive ported geometries.
//!
//! The baseline fabrication validator is intentionally a closed-solid gate.
//! Fluidic devices can legitimately contain openings, so this module provides
//! a narrower alternative: open boundary edges are admitted only when they can
//! be geometrically associated with explicitly allowed port anchors.

use std::collections::{BTreeSet, HashMap};

use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_fabrication_kernel::validate::validate_mesh;
use symthaea_passive_void_compiler::GeometryEmbedding;
use symthaea_passive_void_graph::PortId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryValidationStatus {
    InvalidMesh,
    Closed,
    ExpectedOpeningsOnly,
    UnexpectedOpenings,
    MissingPortAnchor(PortId),
    AmbiguousOpening(PortId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortBoundaryPolicy {
    allowed_open_ports: BTreeSet<PortId>,
    tolerance_micrometers: u64,
}

impl Default for PortBoundaryPolicy {
    fn default() -> Self {
        Self {
            allowed_open_ports: BTreeSet::new(),
            tolerance_micrometers: 50_000,
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

    pub fn tolerance_mm(mut self, tolerance_mm: f32) -> Option<Self> {
        if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
            return None;
        }
        self.tolerance_micrometers = (tolerance_mm as f64 * 1_000.0).round() as u64;
        Some(self)
    }

    pub fn allowed_open_ports(&self) -> impl Iterator<Item = PortId> + '_ {
        self.allowed_open_ports.iter().copied()
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
        let mut valid_ports = Vec::new();

        for port in &policy.allowed_open_ports {
            match embedding.ports.get(port) {
                Some(anchor)
                    if anchor.center_mm.iter().all(|value| value.is_finite())
                        && anchor.radius_mm.is_finite()
                        && anchor.radius_mm > 0.0 =>
                {
                    valid_ports.push((*port, anchor.center_mm, anchor.radius_mm));
                }
                None => missing_port_anchors.push(*port),
                Some(_) => missing_port_anchors.push(*port),
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
        let tolerance_mm = policy.tolerance_micrometers as f64 / 1_000.0;

        for (_, _, midpoint) in boundary_edges {
            let mut matches = Vec::new();
            for (port, center, radius) in &valid_ports {
                let distance = distance_mm(midpoint, *center);
                if distance <= *radius as f64 + tolerance_mm {
                    matches.push(*port);
                }
            }

            match matches.as_slice() {
                [] => unexpected += 1,
                [port] => {
                    let _ = port;
                    matched += 1;
                }
                ports => {
                    ambiguous += 1;
                    if let Some(port) = ports.first().copied() {
                        let _ = port;
                    }
                }
            }
        }

        let status = if ambiguous > 0 {
            BoundaryValidationStatus::AmbiguousOpening(PortId(0))
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
    if a <= b { (a, b) } else { (b, a) }
}

fn collect_boundary_edges(mesh: &TriangleMesh) -> Vec<(QuantizedPoint, QuantizedPoint, [f64; 3])> {
    let mut edges: HashMap<QuantizedEdge, (usize, [f64; 3])> = HashMap::new();
    let mut counts: HashMap<QuantizedEdge, usize> = HashMap::new();

    for triangle in &mesh.indices {
        if triangle.iter().any(|index| (*index as usize) >= mesh.vertices.len()) {
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
            edges.entry(key).or_insert((1, midpoint(a, b)));
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut boundary = Vec::new();
    for ((a, b), (_, center)) in edges {
        if counts.get(&(a, b)).copied().unwrap_or(0) == 1 {
            boundary.push((a, b, center));
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

fn distance_mm(a: [f64; 3], b: [f32; 3]) -> f64 {
    let dx = a[0] - b[0] as f64;
    let dy = a[1] - b[1] as f64;
    let dz = a[2] - b[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::CSGNode;
    use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
    use symthaea_passive_void_compiler::PortAnchor;

    fn embedding() -> GeometryEmbedding {
        GeometryEmbedding::default().with_port(
            PortId(10),
            PortAnchor {
                center_mm: [0.0, 0.0, 0.0],
                radius_mm: 1.0,
            },
        )
    }

    #[test]
    fn closed_cube_passes_closed_policy() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let evidence = PortBoundaryEvidence::evaluate(
            &mesh,
            &embedding(),
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
        let evidence =
            PortBoundaryEvidence::evaluate(&mesh, &embedding(), &PortBoundaryPolicy::closed());
        assert_eq!(evidence.status, BoundaryValidationStatus::UnexpectedOpenings);
        assert!(!evidence.is_admissible());
    }

    #[test]
    fn tolerance_builder_rejects_invalid_values() {
        assert!(PortBoundaryPolicy::closed().tolerance_mm(-1.0).is_none());
        assert!(PortBoundaryPolicy::closed().tolerance_mm(f32::NAN).is_none());
    }
}
