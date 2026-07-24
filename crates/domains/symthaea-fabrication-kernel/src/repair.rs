// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative, bounded triangle-mesh sanitation.
//!
//! Repair is deliberately narrower than geometric healing. It may weld
//! coincident vertices, remove unusable triangles, rebuild normals, and correct
//! global winding. It never fills holes, resolves self-intersections, or grants
//! fabrication authority by itself; callers must validate the result again.

use crate::mesh::TriangleMesh;
use crate::validate::compute_signed_volume;
use std::collections::{HashMap, HashSet};

/// Limits and tolerances for conservative mesh repair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshRepairPolicy {
    /// Vertices within the same quantized cell are welded, in millimetres.
    pub weld_tolerance_mm: f32,
    /// Triangles at or below this area are removed, in square millimetres.
    pub minimum_triangle_area_mm2: f32,
    /// Hard input-work bound.
    pub max_input_triangles: usize,
    pub remove_duplicate_triangles: bool,
    pub orient_positive_volume: bool,
}

impl Default for MeshRepairPolicy {
    fn default() -> Self {
        Self {
            weld_tolerance_mm: 1.0e-5,
            minimum_triangle_area_mm2: 1.0e-10,
            max_input_triangles: 2_000_000,
            remove_duplicate_triangles: true,
            orient_positive_volume: true,
        }
    }
}

/// Evidence describing exactly what conservative repair changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshRepairReport {
    pub input_vertices: usize,
    pub input_triangles: usize,
    pub output_vertices: usize,
    pub output_triangles: usize,
    pub welded_vertices: usize,
    pub removed_out_of_bounds_triangles: usize,
    pub removed_non_finite_triangles: usize,
    pub removed_degenerate_triangles: usize,
    pub removed_duplicate_triangles: usize,
    pub rebuilt_normals: bool,
    pub reversed_global_winding: bool,
}

impl MeshRepairReport {
    pub fn changed(&self) -> bool {
        self.input_vertices != self.output_vertices
            || self.input_triangles != self.output_triangles
            || self.rebuilt_normals
            || self.reversed_global_winding
    }
}

/// Conservative repair failure. No partially repaired mesh is returned.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshRepairError {
    InvalidPolicy(&'static str),
    TriangleBudgetExceeded { actual: usize, maximum: usize },
    CoordinateOutOfRange { vertex: usize },
    VertexIndexOverflow,
}

/// Result of a bounded sanitation pass.
#[derive(Debug, Clone)]
pub struct MeshRepairResult {
    pub mesh: TriangleMesh,
    pub report: MeshRepairReport,
}

/// Sanitize a triangle mesh without claiming topological or fabrication repair.
pub fn repair_mesh(
    mesh: &TriangleMesh,
    policy: MeshRepairPolicy,
) -> Result<MeshRepairResult, MeshRepairError> {
    validate_policy(policy)?;
    if mesh.indices.len() > policy.max_input_triangles {
        return Err(MeshRepairError::TriangleBudgetExceeded {
            actual: mesh.indices.len(),
            maximum: policy.max_input_triangles,
        });
    }

    let mut vertices = Vec::<[f32; 3]>::new();
    let mut indices = Vec::<[u32; 3]>::new();
    let mut quantized_vertices = HashMap::<[i64; 3], u32>::new();
    let mut duplicate_keys = HashSet::<[u32; 3]>::new();
    let mut removed_out_of_bounds_triangles = 0usize;
    let mut removed_non_finite_triangles = 0usize;
    let mut removed_degenerate_triangles = 0usize;
    let mut removed_duplicate_triangles = 0usize;

    for triangle in &mesh.indices {
        let source_indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if source_indices
            .iter()
            .any(|index| *index >= mesh.vertices.len())
        {
            removed_out_of_bounds_triangles += 1;
            continue;
        }

        let source_vertices = [
            mesh.vertices[source_indices[0]],
            mesh.vertices[source_indices[1]],
            mesh.vertices[source_indices[2]],
        ];
        if source_vertices
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            removed_non_finite_triangles += 1;
            continue;
        }
        if triangle_area(source_vertices) <= policy.minimum_triangle_area_mm2 {
            removed_degenerate_triangles += 1;
            continue;
        }

        let mut repaired_triangle = [0u32; 3];
        for corner in 0..3 {
            let source_index = source_indices[corner];
            let key = quantize_vertex(
                source_vertices[corner],
                policy.weld_tolerance_mm,
                source_index,
            )?;
            let repaired_index = if let Some(index) = quantized_vertices.get(&key) {
                *index
            } else {
                let index = u32::try_from(vertices.len())
                    .map_err(|_| MeshRepairError::VertexIndexOverflow)?;
                vertices.push(source_vertices[corner]);
                quantized_vertices.insert(key, index);
                index
            };
            repaired_triangle[corner] = repaired_index;
        }

        if repaired_triangle[0] == repaired_triangle[1]
            || repaired_triangle[1] == repaired_triangle[2]
            || repaired_triangle[2] == repaired_triangle[0]
        {
            removed_degenerate_triangles += 1;
            continue;
        }

        if policy.remove_duplicate_triangles {
            let mut duplicate_key = repaired_triangle;
            duplicate_key.sort_unstable();
            if !duplicate_keys.insert(duplicate_key) {
                removed_duplicate_triangles += 1;
                continue;
            }
        }
        indices.push(repaired_triangle);
    }

    let mut repaired = TriangleMesh {
        normals: vec![[0.0, 0.0, 0.0]; vertices.len()],
        vertices,
        indices,
    };
    let mut reversed_global_winding = false;
    if policy.orient_positive_volume && compute_signed_volume(&repaired) < 0.0 {
        for triangle in &mut repaired.indices {
            triangle.swap(1, 2);
        }
        reversed_global_winding = true;
    }
    // Welding changes vertex ownership, so source vertex normals cannot be
    // retained safely. Rebuild them unconditionally from repaired winding.
    rebuild_vertex_normals(&mut repaired);

    let report = MeshRepairReport {
        input_vertices: mesh.vertices.len(),
        input_triangles: mesh.indices.len(),
        output_vertices: repaired.vertices.len(),
        output_triangles: repaired.indices.len(),
        welded_vertices: mesh.vertices.len().saturating_sub(repaired.vertices.len()),
        removed_out_of_bounds_triangles,
        removed_non_finite_triangles,
        removed_degenerate_triangles,
        removed_duplicate_triangles,
        rebuilt_normals: true,
        reversed_global_winding,
    };
    Ok(MeshRepairResult {
        mesh: repaired,
        report,
    })
}

fn validate_policy(policy: MeshRepairPolicy) -> Result<(), MeshRepairError> {
    if !policy.weld_tolerance_mm.is_finite() || policy.weld_tolerance_mm <= 0.0 {
        return Err(MeshRepairError::InvalidPolicy("weld_tolerance_mm"));
    }
    if !policy.minimum_triangle_area_mm2.is_finite() || policy.minimum_triangle_area_mm2 < 0.0 {
        return Err(MeshRepairError::InvalidPolicy("minimum_triangle_area_mm2"));
    }
    if policy.max_input_triangles == 0 {
        return Err(MeshRepairError::InvalidPolicy("max_input_triangles"));
    }
    Ok(())
}

fn quantize_vertex(
    vertex: [f32; 3],
    tolerance: f32,
    vertex_index: usize,
) -> Result<[i64; 3], MeshRepairError> {
    let mut result = [0i64; 3];
    for axis in 0..3 {
        let scaled = (vertex[axis] as f64 / tolerance as f64).round();
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err(MeshRepairError::CoordinateOutOfRange {
                vertex: vertex_index,
            });
        }
        result[axis] = scaled as i64;
    }
    Ok(result)
}

fn triangle_area(vertices: [[f32; 3]; 3]) -> f32 {
    let a = vertices[0];
    let b = vertices[1];
    let c = vertices[2];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

fn rebuild_vertex_normals(mesh: &mut TriangleMesh) {
    mesh.normals.clear();
    mesh.normals.resize(mesh.vertices.len(), [0.0, 0.0, 0.0]);
    for triangle in &mesh.indices {
        let a = mesh.vertices[triangle[0] as usize];
        let b = mesh.vertices[triangle[1] as usize];
        let c = mesh.vertices[triangle[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let face = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for index in triangle {
            let normal = &mut mesh.normals[*index as usize];
            normal[0] += face[0];
            normal[1] += face[1];
            normal[2] += face[2];
        }
    }
    for normal in &mut mesh.normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > f32::EPSILON {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSGNode;
    use crate::mesh::resolve_to_mesh;
    use crate::validate::validate_mesh;

    #[test]
    fn conservative_repair_welds_cube_and_removes_bad_triangles() {
        let mut dirty = resolve_to_mesh(&CSGNode::cube());
        dirty.indices.push(dirty.indices[0]);
        dirty.indices.push([0, 0, 1]);
        dirty.indices.push([0, 1, u32::MAX]);

        let result = repair_mesh(&dirty, MeshRepairPolicy::default()).unwrap();
        assert_eq!(result.mesh.vertices.len(), 8);
        assert_eq!(result.mesh.indices.len(), 12);
        assert_eq!(result.report.removed_duplicate_triangles, 1);
        assert_eq!(result.report.removed_degenerate_triangles, 1);
        assert_eq!(result.report.removed_out_of_bounds_triangles, 1);
        assert!(validate_mesh(&result.mesh).is_printable());
    }

    #[test]
    fn repair_corrects_globally_reversed_winding() {
        let mut mesh = resolve_to_mesh(&CSGNode::cube());
        for triangle in &mut mesh.indices {
            triangle.swap(1, 2);
        }
        let result = repair_mesh(&mesh, MeshRepairPolicy::default()).unwrap();
        assert!(result.report.reversed_global_winding);
        assert!(compute_signed_volume(&result.mesh) > 0.0);
    }

    #[test]
    fn repair_budget_fails_closed() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let error = repair_mesh(
            &mesh,
            MeshRepairPolicy {
                max_input_triangles: 1,
                ..MeshRepairPolicy::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MeshRepairError::TriangleBudgetExceeded { .. }
        ));
    }
}
