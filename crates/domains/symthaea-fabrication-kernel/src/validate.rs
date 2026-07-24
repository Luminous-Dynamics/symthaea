// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh validation: watertight checks, degenerate triangle detection,
//! normal consistency, and index bounds validation.

use crate::intersection::{DEFAULT_SELF_INTERSECTION_PAIR_BUDGET, find_self_intersections};
use crate::mesh::TriangleMesh;
use std::collections::{HashMap, HashSet};

/// Full validation report for a mesh
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub triangle_count: usize,
    pub vertex_count: usize,
    pub is_empty: bool,
    pub normal_count_matches: bool,
    pub is_watertight: bool,
    /// Edges referenced by exactly one triangle.
    pub boundary_edges: usize,
    /// Edges referenced by more than two triangles.
    pub non_manifold_edges: usize,
    /// Triangle-connected components under shared quantized edges.
    pub connected_components: usize,
    /// Duplicate geometric triangles after position quantization.
    pub duplicate_triangles: Vec<usize>,
    pub degenerate_triangles: Vec<usize>,
    pub inconsistent_normals: Vec<usize>,
    pub out_of_bounds_indices: Vec<usize>,
    pub non_finite_vertices: Vec<usize>,
    pub non_finite_normals: Vec<usize>,
    /// Non-adjacent triangle pairs whose interiors intersect.
    pub self_intersections: Vec<[usize; 2]>,
    /// True when the bounded self-intersection scan examined every candidate.
    pub self_intersection_scan_complete: bool,
    pub signed_volume: f32,
}

/// Owned mesh that has crossed the baseline closed-solid fabrication gate.
#[derive(Debug, Clone)]
pub struct FabricationReadyMesh {
    mesh: TriangleMesh,
    report: ValidationReport,
}

impl FabricationReadyMesh {
    /// Validate and grant the fabrication-ready capability.
    pub fn try_new(mesh: TriangleMesh) -> Result<Self, ValidationReport> {
        let report = validate_mesh(&mesh);
        if !report.is_printable() {
            return Err(report);
        }
        Ok(Self { mesh, report })
    }

    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    pub fn report(&self) -> &ValidationReport {
        &self.report
    }

    pub fn into_mesh(self) -> TriangleMesh {
        self.mesh
    }
}

impl ValidationReport {
    /// True if the mesh is a non-empty, finite, internally consistent triangle set.
    pub fn is_valid(&self) -> bool {
        !self.is_empty
            && self.normal_count_matches
            && self.out_of_bounds_indices.is_empty()
            && self.degenerate_triangles.is_empty()
            && self.duplicate_triangles.is_empty()
            && self.non_manifold_edges == 0
            && self.non_finite_vertices.is_empty()
            && self.non_finite_normals.is_empty()
            && self.self_intersection_scan_complete
            && self.self_intersections.is_empty()
    }

    /// True if the mesh passes the minimum closed-solid gate for fabrication.
    ///
    /// This remains a baseline gate rather than proof of manufacturability: it
    /// does not yet establish self-intersection freedom, minimum wall thickness,
    /// process clearances, or support adequacy.
    pub fn is_printable(&self) -> bool {
        self.is_valid()
            && self.is_watertight
            && self.inconsistent_normals.is_empty()
            && self.signed_volume > 1.0e-9
    }
}

/// Run all validation checks on a mesh
pub fn validate_mesh(mesh: &TriangleMesh) -> ValidationReport {
    let degenerate_triangles = find_degenerate_triangles(mesh, 1e-10);
    let out_of_bounds_indices = check_index_bounds(mesh);
    let topology = analyze_edge_topology(mesh);
    let is_watertight = topology.boundary_edges == 0 && topology.non_manifold_edges == 0;
    let duplicate_triangles = find_duplicate_triangles(mesh);
    let inconsistent_normals = check_normal_consistency(mesh);
    let non_finite_vertices = find_non_finite_vectors(&mesh.vertices);
    let non_finite_normals = find_non_finite_vectors(&mesh.normals);
    let self_intersection_report =
        find_self_intersections(mesh, 1.0e-5, DEFAULT_SELF_INTERSECTION_PAIR_BUDGET);
    let signed_volume = compute_signed_volume(mesh);

    ValidationReport {
        triangle_count: mesh.triangle_count(),
        vertex_count: mesh.vertices.len(),
        is_empty: mesh.vertices.is_empty() || mesh.indices.is_empty(),
        normal_count_matches: mesh.normals.len() == mesh.vertices.len(),
        is_watertight,
        boundary_edges: topology.boundary_edges,
        non_manifold_edges: topology.non_manifold_edges,
        connected_components: topology.connected_components,
        duplicate_triangles,
        degenerate_triangles,
        inconsistent_normals,
        out_of_bounds_indices,
        non_finite_vertices,
        non_finite_normals,
        self_intersections: self_intersection_report.triangle_pairs,
        self_intersection_scan_complete: !self_intersection_report.truncated,
        signed_volume,
    }
}

/// Return indices of vectors containing NaN or infinity.
pub fn find_non_finite_vectors(values: &[[f32; 3]]) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            (!value.iter().all(|component| component.is_finite())).then_some(index)
        })
        .collect()
}

/// Find triangles with near-zero area (degenerate)
pub fn find_degenerate_triangles(mesh: &TriangleMesh, epsilon: f64) -> Vec<usize> {
    let mut degenerate = Vec::new();
    for (i, tri) in mesh.indices.iter().enumerate() {
        if tri[0] as usize >= mesh.vertices.len()
            || tri[1] as usize >= mesh.vertices.len()
            || tri[2] as usize >= mesh.vertices.len()
        {
            continue; // Skip invalid indices — caught by bounds check
        }
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let area = triangle_area(v0, v1, v2);
        if (area as f64) < epsilon {
            degenerate.push(i);
        }
    }
    degenerate
}

/// Check that all indices reference valid vertices
pub fn check_index_bounds(mesh: &TriangleMesh) -> Vec<usize> {
    let n = mesh.vertices.len() as u32;
    let mut bad = Vec::new();
    for (i, tri) in mesh.indices.iter().enumerate() {
        if tri[0] >= n || tri[1] >= n || tri[2] >= n {
            bad.push(i);
        }
    }
    bad
}

type QuantizedPoint = [i64; 3];
type QuantizedEdge = (QuantizedPoint, QuantizedPoint);

fn quantize_position(value: [f32; 3]) -> QuantizedPoint {
    const SCALE: f64 = 1_000_000.0;
    [
        (value[0] as f64 * SCALE).round() as i64,
        (value[1] as f64 * SCALE).round() as i64,
        (value[2] as f64 * SCALE).round() as i64,
    ]
}

fn canonical_edge(a: QuantizedPoint, b: QuantizedPoint) -> QuantizedEdge {
    if a < b { (a, b) } else { (b, a) }
}

/// Edge-level topology evidence for a triangle mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeTopologyReport {
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub connected_components: usize,
}

/// Analyze edge multiplicity and triangle connectivity.
pub fn analyze_edge_topology(mesh: &TriangleMesh) -> EdgeTopologyReport {
    let valid_triangles: Vec<(usize, [QuantizedPoint; 3])> = mesh
        .indices
        .iter()
        .enumerate()
        .filter_map(|(triangle_index, triangle)| {
            let indices = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if indices.iter().any(|index| *index >= mesh.vertices.len()) {
                return None;
            }
            Some((
                triangle_index,
                [
                    quantize_position(mesh.vertices[indices[0]]),
                    quantize_position(mesh.vertices[indices[1]]),
                    quantize_position(mesh.vertices[indices[2]]),
                ],
            ))
        })
        .collect();

    let mut edge_faces: HashMap<QuantizedEdge, Vec<usize>> = HashMap::new();
    for (triangle_index, vertices) in &valid_triangles {
        for edge in [
            canonical_edge(vertices[0], vertices[1]),
            canonical_edge(vertices[1], vertices[2]),
            canonical_edge(vertices[2], vertices[0]),
        ] {
            edge_faces.entry(edge).or_default().push(*triangle_index);
        }
    }

    let boundary_edges = edge_faces.values().filter(|faces| faces.len() == 1).count();
    let non_manifold_edges = edge_faces.values().filter(|faces| faces.len() > 2).count();

    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for (triangle_index, _) in &valid_triangles {
        adjacency.entry(*triangle_index).or_default();
    }
    for faces in edge_faces.values() {
        if let Some((&first, rest)) = faces.split_first() {
            for &other in rest {
                adjacency.entry(first).or_default().push(other);
                adjacency.entry(other).or_default().push(first);
            }
        }
    }

    let mut visited = HashSet::new();
    let mut connected_components = 0usize;
    for &(triangle_index, _) in &valid_triangles {
        if !visited.insert(triangle_index) {
            continue;
        }
        connected_components += 1;
        let mut stack = vec![triangle_index];
        while let Some(current) = stack.pop() {
            if let Some(neighbors) = adjacency.get(&current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    EdgeTopologyReport {
        boundary_edges,
        non_manifold_edges,
        connected_components,
    }
}

/// Return all but the first instance of each duplicate geometric triangle.
pub fn find_duplicate_triangles(mesh: &TriangleMesh) -> Vec<usize> {
    let mut seen: HashMap<[QuantizedPoint; 3], usize> = HashMap::new();
    let mut duplicates = Vec::new();
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        let indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let mut key = [
            quantize_position(mesh.vertices[indices[0]]),
            quantize_position(mesh.vertices[indices[1]]),
            quantize_position(mesh.vertices[indices[2]]),
        ];
        key.sort();
        if seen.insert(key, triangle_index).is_some() {
            duplicates.push(triangle_index);
        }
    }
    duplicates
}

/// Check mesh is watertight (every edge shared by exactly 2 faces)
///
/// Returns (is_watertight, boundary_edge_count).
/// A watertight mesh has 0 boundary edges.
///
/// Uses position-based edge matching (quantized to avoid floating-point issues)
/// since tessellated meshes may have separate vertex instances per face.
pub fn check_watertight(mesh: &TriangleMesh) -> (bool, usize) {
    let topology = analyze_edge_topology(mesh);
    (
        topology.boundary_edges == 0 && topology.non_manifold_edges == 0,
        topology.boundary_edges + topology.non_manifold_edges,
    )
}

/// Check that stored normals are consistent with face winding direction.
///
/// For each triangle, compute the face normal from the cross product
/// and compare with the average stored vertex normal. If the dot product
/// is negative, the normal is inconsistent.
pub fn check_normal_consistency(mesh: &TriangleMesh) -> Vec<usize> {
    let mut inconsistent = Vec::new();
    for (i, tri) in mesh.indices.iter().enumerate() {
        if tri[0] as usize >= mesh.vertices.len()
            || tri[1] as usize >= mesh.vertices.len()
            || tri[2] as usize >= mesh.vertices.len()
        {
            continue;
        }
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        // Face normal from cross product
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let face_n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let face_len =
            (face_n[0] * face_n[0] + face_n[1] * face_n[1] + face_n[2] * face_n[2]).sqrt();
        if face_len < 1e-10 {
            continue; // Degenerate — skip
        }

        // Average stored normal for this triangle's vertices
        if tri[0] as usize >= mesh.normals.len()
            || tri[1] as usize >= mesh.normals.len()
            || tri[2] as usize >= mesh.normals.len()
        {
            continue;
        }
        let n0 = mesh.normals[tri[0] as usize];
        let n1 = mesh.normals[tri[1] as usize];
        let n2 = mesh.normals[tri[2] as usize];
        let avg_n = [
            (n0[0] + n1[0] + n2[0]) / 3.0,
            (n0[1] + n1[1] + n2[1]) / 3.0,
            (n0[2] + n1[2] + n2[2]) / 3.0,
        ];

        // Dot product: negative means inconsistent
        let dot = face_n[0] * avg_n[0] + face_n[1] * avg_n[1] + face_n[2] * avg_n[2];
        if dot < 0.0 {
            inconsistent.push(i);
        }
    }
    inconsistent
}

/// Compute signed volume of mesh (positive = outward-facing normals by convention)
///
/// Uses the divergence theorem: V = (1/6) Σ v0 · (v1 × v2)
pub fn compute_signed_volume(mesh: &TriangleMesh) -> f32 {
    let mut volume = 0.0f32;
    for tri in &mesh.indices {
        if tri[0] as usize >= mesh.vertices.len()
            || tri[1] as usize >= mesh.vertices.len()
            || tri[2] as usize >= mesh.vertices.len()
        {
            continue;
        }
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        // Signed volume contribution: v0 · (v1 × v2)
        let cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        volume += v0[0] * cross[0] + v0[1] * cross[1] + v0[2] * cross[2];
    }
    volume / 6.0
}

fn triangle_area(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> f32 {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSGNode;
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn fabrication_ready_mesh_accepts_closed_cube() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let ready = FabricationReadyMesh::try_new(mesh).unwrap();
        assert!(ready.report().is_printable());
    }

    #[test]
    fn fabrication_ready_mesh_rejects_open_surface() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let report = FabricationReadyMesh::try_new(mesh)
            .err()
            .expect("open surface must not gain fabrication authority");
        assert!(!report.is_printable());
    }

    #[test]
    fn cube_is_watertight() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight, "cube should be watertight");
        assert_eq!(report.boundary_edges, 0);
    }

    #[test]
    fn duplicate_triangle_is_reported_and_breaks_manifoldness() {
        let mut mesh = resolve_to_mesh(&CSGNode::cube());
        mesh.indices.push(mesh.indices[0]);
        let report = validate_mesh(&mesh);
        assert_eq!(report.duplicate_triangles, vec![12]);
        assert!(report.non_manifold_edges > 0);
        assert!(!report.is_printable());
    }

    #[test]
    fn disconnected_closed_solids_are_counted_separately() {
        use crate::csg::Transform3D;
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [2.0, 0.0, 0.0],
            ..Default::default()
        }));
        left.merge(&right);
        let report = validate_mesh(&left);
        assert_eq!(report.connected_components, 2);
        assert_eq!(report.non_manifold_edges, 0);
        assert_eq!(report.boundary_edges, 0);
    }

    #[test]
    fn overlapping_closed_shells_fail_self_intersection_gate() {
        use crate::csg::Transform3D;
        let mut left = resolve_to_mesh(&CSGNode::cube());
        let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [0.25, 0.25, 0.25],
            ..Default::default()
        }));
        left.merge(&right);
        let report = validate_mesh(&left);
        assert!(report.self_intersection_scan_complete);
        assert!(!report.self_intersections.is_empty());
        assert!(!report.is_printable());
    }

    #[test]
    fn open_triangle_has_boundary_but_no_non_manifold_edges() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let report = validate_mesh(&mesh);
        assert_eq!(report.boundary_edges, 3);
        assert_eq!(report.non_manifold_edges, 0);
        assert_eq!(report.connected_components, 1);
    }

    #[test]
    fn cube_no_degenerate() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = validate_mesh(&mesh);
        assert!(report.degenerate_triangles.is_empty());
    }

    #[test]
    fn cube_consistent_normals() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = validate_mesh(&mesh);
        assert!(report.inconsistent_normals.is_empty());
    }

    #[test]
    fn cube_positive_volume() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let vol = compute_signed_volume(&mesh);
        // Unit cube: volume = 1.0
        assert!(
            (vol - 1.0).abs() < 0.01,
            "cube volume should be ~1.0, got {}",
            vol
        );
    }

    #[test]
    fn cube_is_printable() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = validate_mesh(&mesh);
        assert!(report.is_printable(), "cube should be printable");
    }

    #[test]
    fn sphere_is_watertight_without_pole_degeneracy() {
        let mesh = resolve_to_mesh(&CSGNode::sphere());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight);
        assert!(report.degenerate_triangles.is_empty());
    }

    #[test]
    fn sphere_positive_volume() {
        let mesh = resolve_to_mesh(&CSGNode::sphere());
        let vol = compute_signed_volume(&mesh);
        // Unit sphere r=0.5: V = (4/3)π(0.5)³ ≈ 0.524
        assert!(
            vol > 0.4 && vol < 0.6,
            "sphere volume should be ~0.524, got {}",
            vol
        );
    }

    #[test]
    fn cylinder_is_watertight() {
        let mesh = resolve_to_mesh(&CSGNode::cylinder());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight);
    }

    #[test]
    fn cone_is_watertight() {
        let mesh = resolve_to_mesh(&CSGNode::cone());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight);
    }

    #[test]
    fn torus_is_watertight() {
        let mesh = resolve_to_mesh(&CSGNode::torus());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight);
    }

    #[test]
    fn empty_mesh_is_not_a_fabrication_candidate() {
        let mesh = TriangleMesh::empty();
        let report = validate_mesh(&mesh);
        assert!(report.is_empty);
        assert!(!report.is_valid());
        assert!(!report.is_printable());
        assert!(report.is_watertight); // topologically vacuous, but not valid
        assert_eq!(report.signed_volume, 0.0);
    }

    #[test]
    fn rejects_non_finite_geometry() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let report = validate_mesh(&mesh);
        assert_eq!(report.non_finite_vertices, vec![1]);
        assert!(!report.is_valid());
    }

    #[test]
    fn rejects_missing_normals() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![],
            indices: vec![[0, 1, 2]],
        };
        let report = validate_mesh(&mesh);
        assert!(!report.normal_count_matches);
        assert!(!report.is_valid());
    }

    #[test]
    fn detect_degenerate() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0; 3], [0.0; 3], [0.0; 3]], // All same point
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };
        let degen = find_degenerate_triangles(&mesh, 1e-10);
        assert_eq!(degen.len(), 1);
    }

    #[test]
    fn detect_bad_indices() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0; 3], [1.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 2],
            indices: vec![[0, 1, 99]], // Index 99 out of bounds
        };
        let bad = check_index_bounds(&mesh);
        assert_eq!(bad.len(), 1);
        assert!(!validate_mesh(&mesh).is_valid());
    }

    #[test]
    fn roundtrip_validate_cube() {
        // Export → import → validate
        let cube = resolve_to_mesh(&CSGNode::cube());
        let stl = crate::export::export_stl(&cube);
        let imported = crate::import::parse_binary_stl(&stl).unwrap();
        let report = validate_mesh(&imported);
        assert!(report.is_valid());
        assert_eq!(report.triangle_count, cube.triangle_count());
        // Imported mesh won't share vertices, so watertight check uses per-triangle vertices
        // The edge count will be different (non-shared vertices → each edge appears once per face)
    }

    #[test]
    fn stl_roundtrip_preserves_volume() {
        let cube = resolve_to_mesh(&CSGNode::cube());
        let original_vol = compute_signed_volume(&cube);
        let stl = crate::export::export_stl(&cube);
        let imported = crate::import::parse_binary_stl(&stl).unwrap();
        let imported_vol = compute_signed_volume(&imported);
        assert!(
            (original_vol - imported_vol).abs() < 0.01,
            "volume should be preserved: {} vs {}",
            original_vol,
            imported_vol
        );
    }
}
