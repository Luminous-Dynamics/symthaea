use std::collections::{HashMap, HashSet};

use super::boundary::{PortBoundaryEvidence, PortBoundaryPolicy};
use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_fabrication_kernel::validate::validate_mesh;
use symthaea_passive_void_compiler::GeometryEmbedding;
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortPathStatus {
    InvalidGraph,
    InvalidMesh,
    BoundaryPolicyRejected,
    UndeclaredPath,
    AnchorNotRepresented(PortId),
    AmbiguousAnchor(PortId),
    Disconnected,
    Connected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortPathEvidence {
    pub status: PortPathStatus,
    pub from_component: Option<usize>,
    pub to_component: Option<usize>,
    /// Geometry/topology evidence only; transport remains unproven.
    pub physical_transport_unproven: bool,
}

pub fn evaluate_port_path(
    graph: &FunctionalVoidGraph,
    embedding: &GeometryEmbedding,
    candidate: &TriangleMesh,
    from: PortId,
    to: PortId,
) -> PortPathEvidence {
    evaluate_port_path_inner(graph, embedding, candidate, from, to, None)
}

/// Evaluate a declared flow path with an explicit boundary policy.
pub fn evaluate_port_path_with_boundary_policy(
    graph: &FunctionalVoidGraph,
    embedding: &GeometryEmbedding,
    candidate: &TriangleMesh,
    from: PortId,
    to: PortId,
    boundary_policy: &PortBoundaryPolicy,
) -> PortPathEvidence {
    evaluate_port_path_inner(
        graph,
        embedding,
        candidate,
        from,
        to,
        Some(boundary_policy),
    )
}

fn evaluate_port_path_inner(
    graph: &FunctionalVoidGraph,
    embedding: &GeometryEmbedding,
    candidate: &TriangleMesh,
    from: PortId,
    to: PortId,
    boundary_policy: Option<&PortBoundaryPolicy>,
) -> PortPathEvidence {
    if graph.validate().is_err() {
        return PortPathEvidence { status: PortPathStatus::InvalidGraph, from_component: None, to_component: None, physical_transport_unproven: true };
    }
    if !graph.declares_path(from, to, symthaea_passive_void_graph::VoidRelation::FlowPath) {
        return PortPathEvidence { status: PortPathStatus::UndeclaredPath, from_component: None, to_component: None, physical_transport_unproven: true };
    }
    let report = validate_mesh(candidate);
    if !report.is_valid() {
        return PortPathEvidence { status: PortPathStatus::InvalidMesh, from_component: None, to_component: None, physical_transport_unproven: true };
    }
    if let Some(policy) = boundary_policy {
        let boundary = PortBoundaryEvidence::evaluate(candidate, embedding, policy);
        if !boundary.is_admissible() {
            return PortPathEvidence { status: PortPathStatus::BoundaryPolicyRejected, from_component: None, to_component: None, physical_transport_unproven: true };
        }
    } else if !report.is_watertight {
        return PortPathEvidence { status: PortPathStatus::InvalidMesh, from_component: None, to_component: None, physical_transport_unproven: true };
    }
    let (from_point, to_point) = match (embedding.ports.get(&from), embedding.ports.get(&to)) {
        (Some(a), Some(b)) => (a.center_mm, b.center_mm),
        (None, _) => return missing_anchor(from),
        (_, None) => return missing_anchor(to),
    };
    let labels = triangle_components(candidate);
    let from_radius = embedding.ports.get(&from).map(|port| port.radius_mm).unwrap_or(0.0);
    let to_radius = embedding.ports.get(&to).map(|port| port.radius_mm).unwrap_or(0.0);
    let from_component = nearest_component(candidate, &labels, from_point, from_radius);
    let to_component = nearest_component(candidate, &labels, to_point, to_radius);
    if matches!(from_component, AnchorResolution::Ambiguous) {
        return PortPathEvidence { status: PortPathStatus::AmbiguousAnchor(from), from_component: None, to_component: component_value(to_component), physical_transport_unproven: true };
    }
    if matches!(to_component, AnchorResolution::Ambiguous) {
        return PortPathEvidence { status: PortPathStatus::AmbiguousAnchor(to), from_component: component_value(from_component), to_component: None, physical_transport_unproven: true };
    }
    let from_component = component_value(from_component);
    let to_component = component_value(to_component);
    match (from_component, to_component) {
        (Some(a), Some(b)) if a == b => PortPathEvidence { status: PortPathStatus::Connected, from_component: Some(a), to_component: Some(b), physical_transport_unproven: true },
        (Some(a), Some(b)) => PortPathEvidence { status: PortPathStatus::Disconnected, from_component: Some(a), to_component: Some(b), physical_transport_unproven: true },
        (None, _) => PortPathEvidence { status: PortPathStatus::AnchorNotRepresented(from), from_component, to_component, physical_transport_unproven: true },
        (_, None) => PortPathEvidence { status: PortPathStatus::AnchorNotRepresented(to), from_component, to_component, physical_transport_unproven: true },
    }
}

fn missing_anchor(port: PortId) -> PortPathEvidence {
    PortPathEvidence {
        status: PortPathStatus::AnchorNotRepresented(port),
        from_component: None,
        to_component: None,
        physical_transport_unproven: true,
    }
}

pub(crate) fn triangle_components(mesh: &TriangleMesh) -> Vec<usize> {
    let valid: Vec<usize> = mesh
        .indices
        .iter()
        .enumerate()
        .filter(|(_, tri)| tri.iter().all(|i| (*i as usize) < mesh.vertices.len()))
        .map(|(i, _)| i)
        .collect();
    let mut parent: Vec<usize> = (0..mesh.indices.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb { parent[rb] = ra; }
    }
    let mut edges: HashMap<([i64;3], [i64;3]), usize> = HashMap::new();
    for i in valid.iter().copied() {
        let tri = mesh.indices[i];
        let points = [
            quantize(mesh.vertices[tri[0] as usize]),
            quantize(mesh.vertices[tri[1] as usize]),
            quantize(mesh.vertices[tri[2] as usize]),
        ];
        for edge in [(points[0], points[1]), (points[1], points[2]), (points[2], points[0])] {
            let edge = if edge.0 <= edge.1 { edge } else { (edge.1, edge.0) };
            if let Some(other) = edges.insert(edge, i) { union(&mut parent, i, other); }
        }
    }
    (0..mesh.indices.len()).map(|i| find(&mut parent, i)).collect()
}

pub(crate) enum AnchorResolution {
    Missing,
    Ambiguous,
    Found(usize),
}

fn component_value(resolution: AnchorResolution) -> Option<usize> {
    match resolution {
        AnchorResolution::Found(component) => Some(component),
        AnchorResolution::Missing | AnchorResolution::Ambiguous => None,
    }
}

pub(crate) fn resolve_port_component(
    mesh: &TriangleMesh,
    embedding: &GeometryEmbedding,
    labels: &[usize],
    port: PortId,
) -> AnchorResolution {
    let Some(anchor) = embedding.ports.get(&port) else {
        return AnchorResolution::Missing;
    };
    nearest_component(mesh, labels, anchor.center_mm, anchor.radius_mm)
}

fn nearest_component(
    mesh: &TriangleMesh,
    labels: &[usize],
    point: [f32; 3],
    radius_mm: f32,
) -> AnchorResolution {
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return AnchorResolution::Missing;
    }

    let mut vertex_component: HashMap<[i64; 3], HashSet<usize>> = HashMap::new();
    for (tri_index, tri) in mesh.indices.iter().enumerate() {
        if !tri.iter().all(|i| (*i as usize) < mesh.vertices.len()) {
            continue;
        }
        let label = labels[tri_index];
        for vertex in tri {
            vertex_component
                .entry(quantize(mesh.vertices[*vertex as usize]))
                .or_default()
                .insert(label);
        }
    }

    let tolerance = (radius_mm as f64 * 1.05).powi(2);
    let mut best: Option<(f64, Vec<usize>)> = None;

    for (vertex, components) in vertex_component {
        let dx = vertex[0] as f64 / 1_000_000.0 - point[0] as f64;
        let dy = vertex[1] as f64 / 1_000_000.0 - point[1] as f64;
        let dz = vertex[2] as f64 / 1_000_000.0 - point[2] as f64;
        let distance = dx * dx + dy * dy + dz * dz;
        if distance > tolerance {
            continue;
        }

        let mut component_ids = components.into_iter().collect::<Vec<_>>();
        component_ids.sort_unstable();

        match &mut best {
            Some((best_distance, best_components)) if distance < *best_distance - 1.0e-18 => {
                *best_distance = distance;
                *best_components = component_ids;
            }
            Some((best_distance, best_components))
                if (distance - *best_distance).abs() <= 1.0e-18 =>
            {
                for component in component_ids {
                    if !best_components.contains(&component) {
                        best_components.push(component);
                    }
                }
                best_components.sort_unstable();
            }
            None => {
                best = Some((distance, component_ids));
            }
            _ => {}
        }
    }

    match best {
        None => AnchorResolution::Missing,
        Some((_, components)) if components.len() == 1 => AnchorResolution::Found(components[0]),
        Some(_) => AnchorResolution::Ambiguous,
    }
}

fn quantize(point: [f32;3]) -> [i64;3] {
    [
        (point[0] as f64 * 1_000_000.0).round() as i64,
        (point[1] as f64 * 1_000_000.0).round() as i64,
        (point[2] as f64 * 1_000_000.0).round() as i64,
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::{CSGNode, Transform3D};
    use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
    use symthaea_passive_void_compiler::{GeometryEmbedding, PortAnchor};

    fn graph() -> FunctionalVoidGraph {
        let mut graph = FunctionalVoidGraph::new();
        graph
            .add_region(symthaea_passive_void_graph::VoidRegion {
                id: RegionId(1),
                role: symthaea_passive_void_graph::VoidRegionRole::Inlet,
            })
            .unwrap();
        graph
            .add_region(symthaea_passive_void_graph::VoidRegion {
                id: RegionId(2),
                role: symthaea_passive_void_graph::VoidRegionRole::Outlet,
            })
            .unwrap();
        graph
            .add_port(symthaea_passive_void_graph::VoidPort {
                id: PortId(10),
                region: RegionId(1),
            })
            .unwrap();
        graph
            .add_port(symthaea_passive_void_graph::VoidPort {
                id: PortId(20),
                region: RegionId(2),
            })
            .unwrap();
        graph
            .connect(symthaea_passive_void_graph::VoidConnection {
                from: PortId(10),
                to: PortId(20),
                relation: symthaea_passive_void_graph::VoidRelation::FlowPath,
                bidirectional: false,
            })
            .unwrap();
        graph
    }

    fn embedding() -> GeometryEmbedding {
        GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-0.5, -0.5, -0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [0.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            )
    }

    #[test]
    fn undeclared_port_pair_is_rejected() {
        let mut graph = graph();
        graph.connections.clear();
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let evidence = evaluate_port_path(&graph, &embedding(), &mesh, PortId(10), PortId(20));
        assert_eq!(evidence.status, PortPathStatus::UndeclaredPath);
    }

    #[test]
    fn connected_ports_are_detected_in_mesh_component() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let evidence = evaluate_port_path(&graph(), &embedding(), &mesh, PortId(10), PortId(20));
        assert_eq!(evidence.status, PortPathStatus::Connected);
        assert!(evidence.physical_transport_unproven);
    }

    #[test]
    fn far_away_port_is_not_falsely_attached_to_mesh() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let e = GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [100.0, 100.0, 100.0],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [0.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            );
        let evidence = evaluate_port_path(&graph(), &e, &mesh, PortId(10), PortId(20));
        assert_eq!(
            evidence.status,
            PortPathStatus::AnchorNotRepresented(PortId(10))
        );
    }

    #[test]
    fn disconnected_ports_are_detected() {
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [3.0, 0.0, 0.0],
            ..Default::default()
        }));
        left.merge(&right);

        let e = GeometryEmbedding::default()
            .with_port(
                PortId(10),
                PortAnchor {
                    center_mm: [-0.5, -0.5, -0.5],
                    radius_mm: 1.0,
                },
            )
            .with_port(
                PortId(20),
                PortAnchor {
                    center_mm: [2.5, 0.5, 0.5],
                    radius_mm: 1.0,
                },
            );
        let evidence = evaluate_port_path(&graph(), &e, &left, PortId(10), PortId(20));
        assert_eq!(evidence.status, PortPathStatus::Disconnected);
    }
}

