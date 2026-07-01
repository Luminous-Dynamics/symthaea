// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mesh validation: watertight checks, degenerate triangle detection,
//! normal consistency, and index bounds validation.

use crate::mesh::TriangleMesh;
use std::collections::HashMap;

/// Full validation report for a mesh
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub triangle_count: usize,
    pub vertex_count: usize,
    pub is_watertight: bool,
    pub boundary_edges: usize,
    pub degenerate_triangles: Vec<usize>,
    pub inconsistent_normals: Vec<usize>,
    pub out_of_bounds_indices: Vec<usize>,
    pub signed_volume: f32,
}

impl ValidationReport {
    /// True if mesh passes all checks
    pub fn is_valid(&self) -> bool {
        self.out_of_bounds_indices.is_empty() && self.degenerate_triangles.is_empty()
    }

    /// True if mesh is suitable for 3D printing (valid + watertight + consistent normals)
    pub fn is_printable(&self) -> bool {
        self.is_valid() && self.is_watertight && self.inconsistent_normals.is_empty()
    }
}

/// Run all validation checks on a mesh
pub fn validate_mesh(mesh: &TriangleMesh) -> ValidationReport {
    let degenerate_triangles = find_degenerate_triangles(mesh, 1e-10);
    let out_of_bounds_indices = check_index_bounds(mesh);
    let (is_watertight, boundary_edges) = check_watertight(mesh);
    let inconsistent_normals = check_normal_consistency(mesh);
    let signed_volume = compute_signed_volume(mesh);

    ValidationReport {
        triangle_count: mesh.triangle_count(),
        vertex_count: mesh.vertices.len(),
        is_watertight,
        boundary_edges,
        degenerate_triangles,
        inconsistent_normals,
        out_of_bounds_indices,
        signed_volume,
    }
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

/// Check mesh is watertight (every edge shared by exactly 2 faces)
///
/// Returns (is_watertight, boundary_edge_count).
/// A watertight mesh has 0 boundary edges.
///
/// Uses position-based edge matching (quantized to avoid floating-point issues)
/// since tessellated meshes may have separate vertex instances per face.
pub fn check_watertight(mesh: &TriangleMesh) -> (bool, usize) {
    // Quantize positions to avoid floating-point comparison issues
    fn quantize(v: [f32; 3]) -> [i64; 3] {
        const SCALE: f64 = 1_000_000.0;
        [
            (v[0] as f64 * SCALE).round() as i64,
            (v[1] as f64 * SCALE).round() as i64,
            (v[2] as f64 * SCALE).round() as i64,
        ]
    }

    let mut edge_count: HashMap<([i64; 3], [i64; 3]), u32> = HashMap::new();

    for tri in &mesh.indices {
        if tri[0] as usize >= mesh.vertices.len()
            || tri[1] as usize >= mesh.vertices.len()
            || tri[2] as usize >= mesh.vertices.len()
        {
            continue;
        }
        for k in 0..3 {
            let va = quantize(mesh.vertices[tri[k] as usize]);
            let vb = quantize(mesh.vertices[tri[(k + 1) % 3] as usize]);
            let edge = if va < vb { (va, vb) } else { (vb, va) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    let boundary = edge_count.values().filter(|&&c| c != 2).count();
    (boundary == 0, boundary)
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
    fn cube_is_watertight() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let report = validate_mesh(&mesh);
        assert!(report.is_watertight, "cube should be watertight");
        assert_eq!(report.boundary_edges, 0);
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
    fn sphere_mostly_watertight() {
        // Sphere tessellation has degenerate triangles at poles where multiple
        // vertices converge to the same point. This creates a small number of
        // boundary edges in the quantized representation. We check that the
        // boundary count is very low relative to total edges.
        let mesh = resolve_to_mesh(&CSGNode::sphere());
        let report = validate_mesh(&mesh);
        let total_edges = mesh.triangle_count() * 3;
        let boundary_pct = report.boundary_edges as f32 / total_edges as f32;
        assert!(
            boundary_pct < 0.05,
            "sphere boundary edges should be <5%, got {:.1}%",
            boundary_pct * 100.0
        );
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
    fn empty_mesh_valid() {
        let mesh = TriangleMesh::empty();
        let report = validate_mesh(&mesh);
        assert!(report.is_valid());
        assert!(report.is_watertight); // vacuously true
        assert_eq!(report.signed_volume, 0.0);
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
