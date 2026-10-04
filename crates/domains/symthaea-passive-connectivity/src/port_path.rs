use std::collections::{HashMap, HashSet};

use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_fabrication_kernel::validate::validate_mesh;
use symthaea_passive_void_compiler::GeometryEmbedding;
use symthaea_passive_void_graph::{FunctionalVoidGraph, PortId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortPathStatus {
    InvalidMesh,
    AnchorNotRepresented(PortId),
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
    if graph.validate().is_err() {
        return PortPathEvidence {
            status: PortPathStatus::InvalidMesh,
            from_component: None,
            to_component: None,
            physical_transport_unproven: true,
        };
    }
    let report = validate_mesh(candidate);
    if !report.is_valid() || !report.is_watertight {
        return PortPathEvidence {
            status: PortPathStatus::InvalidMesh,
            from_component: None,
            to_component: None,
            physical_transport_unproven: true,
        };
    }
    let (from_point, to_point) = match (embedding.ports.get(&from), embedding.ports.get(&to)) {
        (Some(a), Some(b)) => (a.center_mm, b.center_mm),
        (None, _) => return missing_anchor(from),
        (_, None) => return missing_anchor(to),
    };
    let labels = triangle_components(candidate);
    let from_component = nearest_component(candidate, &labels, from_point);
    let to_component = nearest_component(candidate, &labels, to_point);
    match (from_component, to_component) {
        (Some(a), Some(b)) if a == b => PortPathEvidence {
            status: PortPathStatus::Connected,
            from_component: Some(a),
            to_component: Some(b),
            physical_transport_unproven: true,
        },
        (Some(a), Some(b)) => PortPathEvidence {
            status: PortPathStatus::Disconnected,
            from_component: Some(a),
            to_component: Some(b),
            physical_transport_unproven: true,
        },
        (None, _) => PortPathEvidence {
            status: PortPathStatus::AnchorNotRepresented(from),
            from_component,
            to_component,
            physical_transport_unproven: true,
        },
        (_, None) => PortPathEvidence {
            status: PortPathStatus::AnchorNotRepresented(to),
            from_component,
            to_component,
            physical_transport_unproven: true,
        },
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

fn triangle_components(mesh: &TriangleMesh) -> Vec<usize> {
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

fn nearest_component(mesh: &TriangleMesh, labels: &[usize], point: [f32;3]) -> Option<usize> {
    let mut vertex_component: HashMap<[i64;3], HashSet<usize>> = HashMap::new();
    for (tri_index, tri) in mesh.indices.iter().enumerate() {
        if !tri.iter().all(|i| (*i as usize) < mesh.vertices.len()) { continue; }
        let label = labels[tri_index];
        for vertex in tri {
            vertex_component.entry(quantize(mesh.vertices[*vertex as usize])).or_default().insert(label);
        }
    }
    let mut best: Option<(f64, usize)> = None;
    for (vertex, components) in vertex_component {
        let dx = vertex[0] as f64 / 1_000_000.0 - point[0] as f64;
        let dy = vertex[1] as f64 / 1_000_000.0 - point[1] as f64;
        let dz = vertex[2] as f64 / 1_000_000.0 - point[2] as f64;
        let distance = dx*dx + dy*dy + dz*dz;
        if let Some(&component) = components.iter().next() {
            if best.map(|(d, _)| distance < d).unwrap_or(true) { best = Some((distance, component)); }
        }
    }
    best.map(|(_, component)| component)
}

fn quantize(point: [f32;3]) -> [i64;3] {
    [
        (point[0] as f64 * 1_000_000.0).round() as i64,
        (point[1] as f64 * 1_000_000.0).round() as i64,
        (point[2] as f64 * 1_000_000.0).round() as i64,
    ]
}

pub(crate) fn embedding_port(_embedding: &GeometryEmbedding, _port: PortId) -> Option<PortAnchor> { None }

#[allow(dead_code)]
fn _keep_report_type(_: &ValidationReport, _: &ConnectivityStatus) {}