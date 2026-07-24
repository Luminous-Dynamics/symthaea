// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Triangle mesh generation from CSG trees
//!
//! Tessellates geometric primitives into triangle meshes and resolves closed
//! solid CSG operations through the BSP backend.

use crate::bsp;
use crate::csg::{BooleanOp, CSGNode, Primitive, Transform3D};
use serde::{Deserialize, Serialize};

/// A triangle mesh with vertices, normals, and triangle indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
}

/// Controls curved primitive tessellation. Geometry coordinates are millimetres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TessellationPolicy {
    /// Maximum radial sagitta error for curved primitive facets, in millimetres.
    pub max_chord_error_mm: f32,
    /// Lower bound for a complete circular sweep.
    pub min_segments: usize,
    /// Upper bound preventing unbounded mesh growth.
    pub max_segments: usize,
}

impl Default for TessellationPolicy {
    fn default() -> Self {
        Self {
            max_chord_error_mm: 0.01,
            min_segments: 12,
            max_segments: 256,
        }
    }
}

impl TessellationPolicy {
    fn circular_segments(self, radius_mm: f32) -> usize {
        let min_segments = self.min_segments.max(3);
        let max_segments = self.max_segments.max(min_segments);
        if !radius_mm.is_finite() || radius_mm <= 0.0 {
            return min_segments;
        }
        let error = if self.max_chord_error_mm.is_finite() {
            self.max_chord_error_mm.max(f32::EPSILON)
        } else {
            0.01
        };
        if error >= radius_mm {
            return min_segments;
        }
        let angle = (1.0 - error / radius_mm).clamp(-1.0, 1.0).acos();
        if !angle.is_finite() || angle <= f32::EPSILON {
            return max_segments;
        }
        ((std::f32::consts::PI / angle).ceil() as usize).clamp(min_segments, max_segments)
    }
}

impl TriangleMesh {
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Append another mesh as an additional disconnected shell.
    pub fn merge(&mut self, other: &TriangleMesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.normals.extend_from_slice(&other.normals);
        for tri in &other.indices {
            self.indices
                .push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
    }

    /// Apply an affine transform to all vertices and normals.
    ///
    /// Normals use the inverse-transpose transform so non-uniform scaling does
    /// not silently corrupt lighting or validation. Reflections also reverse
    /// triangle winding to preserve outward orientation.
    pub fn apply_transform(&mut self, transform: &Transform3D) {
        for v in &mut self.vertices {
            *v = transform.apply(*v);
        }
        for n in &mut self.normals {
            *n = transform.apply_normal(*n).unwrap_or([0.0, 0.0, 0.0]);
        }
        if transform.reverses_orientation() {
            for tri in &mut self.indices {
                tri.swap(1, 2);
            }
        }
    }
}

/// Resolve a CSG tree into a triangle mesh using the default adaptive policy.
pub fn resolve_to_mesh(node: &CSGNode) -> TriangleMesh {
    resolve_to_mesh_with_policy(node, TessellationPolicy::default())
}

/// Resolve a CSG tree using an explicit bounded tessellation policy.
pub fn resolve_to_mesh_with_policy(node: &CSGNode, policy: TessellationPolicy) -> TriangleMesh {
    resolve_with_scale(node, policy, [1.0, 1.0, 1.0])
}

fn resolve_with_scale(
    node: &CSGNode,
    policy: TessellationPolicy,
    inherited_scale: [f32; 3],
) -> TriangleMesh {
    match node {
        CSGNode::Primitive(primitive) => tessellate_primitive(*primitive, policy, inherited_scale),
        CSGNode::Transform { node, transform } => {
            let detail_scale = [
                inherited_scale[0] * transform.scale[0].abs(),
                inherited_scale[1] * transform.scale[1].abs(),
                inherited_scale[2] * transform.scale[2].abs(),
            ];
            let mut mesh = resolve_with_scale(node, policy, detail_scale);
            mesh.apply_transform(transform);
            mesh
        }
        CSGNode::Boolean { op, left, right } => {
            let left_mesh = resolve_with_scale(left, policy, inherited_scale);
            let right_mesh = resolve_with_scale(right, policy, inherited_scale);
            match op {
                BooleanOp::Union => bsp::csg_union(&left_mesh, &right_mesh),
                BooleanOp::Subtract => bsp::csg_subtract(&left_mesh, &right_mesh),
                BooleanOp::Intersect => bsp::csg_intersect(&left_mesh, &right_mesh),
            }
        }
    }
}

/// Tessellate a primitive into triangles at its effective world-space scale.
fn tessellate_primitive(
    primitive: Primitive,
    policy: TessellationPolicy,
    scale: [f32; 3],
) -> TriangleMesh {
    let radial_scale = scale[0].max(scale[1]).max(f32::EPSILON);
    let maximum_scale = radial_scale.max(scale[2]).max(f32::EPSILON);
    match primitive {
        Primitive::Cube => tessellate_cube(),
        Primitive::Cylinder => tessellate_cylinder(policy.circular_segments(0.5 * radial_scale)),
        Primitive::Sphere => {
            let longitude = policy.circular_segments(0.5 * maximum_scale);
            tessellate_sphere(longitude, (longitude / 2).max(6))
        }
        Primitive::Cone => tessellate_cone(policy.circular_segments(0.5 * radial_scale)),
        Primitive::Torus => {
            let major = policy.circular_segments(0.7 * radial_scale);
            let minor = policy.circular_segments(0.2 * maximum_scale);
            tessellate_torus(major, minor)
        }
    }
}

fn tessellate_cube() -> TriangleMesh {
    let h = 0.5;
    let vertices = vec![
        // Front face
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
        // Back face
        [h, -h, -h],
        [-h, -h, -h],
        [-h, h, -h],
        [h, h, -h],
        // Top face
        [-h, h, h],
        [h, h, h],
        [h, h, -h],
        [-h, h, -h],
        // Bottom face
        [-h, -h, -h],
        [h, -h, -h],
        [h, -h, h],
        [-h, -h, h],
        // Right face
        [h, -h, h],
        [h, -h, -h],
        [h, h, -h],
        [h, h, h],
        // Left face
        [-h, -h, -h],
        [-h, -h, h],
        [-h, h, h],
        [-h, h, -h],
    ];
    let normals = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let indices = vec![
        [0, 1, 2],
        [0, 2, 3], // Front
        [4, 5, 6],
        [4, 6, 7], // Back
        [8, 9, 10],
        [8, 10, 11], // Top
        [12, 13, 14],
        [12, 14, 15], // Bottom
        [16, 17, 18],
        [16, 18, 19], // Right
        [20, 21, 22],
        [20, 22, 23], // Left
    ];
    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

fn tessellate_cylinder(segments: usize) -> TriangleMesh {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let r = 0.5f32;
    let h = 0.5f32;

    // Side faces
    for i in 0..segments {
        let a0 = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let a1 = ((i + 1) as f32) / (segments as f32) * std::f32::consts::TAU;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            [r * c0, r * s0, -h],
            [r * c1, r * s1, -h],
            [r * c1, r * s1, h],
            [r * c0, r * s0, h],
        ]);
        let nx0 = c0;
        let ny0 = s0;
        let nx1 = c1;
        let ny1 = s1;
        normals.extend_from_slice(&[
            [nx0, ny0, 0.0],
            [nx1, ny1, 0.0],
            [nx1, ny1, 0.0],
            [nx0, ny0, 0.0],
        ]);
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
    }

    // Top cap
    let center_top = vertices.len() as u32;
    vertices.push([0.0, 0.0, h]);
    normals.push([0.0, 0.0, 1.0]);
    for i in 0..segments {
        let a = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        vertices.push([r * c, r * s, h]);
        normals.push([0.0, 0.0, 1.0]);
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push([
            center_top,
            center_top + 1 + i as u32,
            center_top + 1 + next as u32,
        ]);
    }

    // Bottom cap
    let center_bot = vertices.len() as u32;
    vertices.push([0.0, 0.0, -h]);
    normals.push([0.0, 0.0, -1.0]);
    for i in 0..segments {
        let a = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        vertices.push([r * c, r * s, -h]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push([
            center_bot,
            center_bot + 1 + next as u32,
            center_bot + 1 + i as u32,
        ]);
    }

    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

fn tessellate_sphere(lon_segments: usize, lat_segments: usize) -> TriangleMesh {
    let lon_segments = lon_segments.max(3);
    let lat_segments = lat_segments.max(3);
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let radius = 0.5f32;

    let top = vertices.len() as u32;
    vertices.push([0.0, 0.0, radius]);
    normals.push([0.0, 0.0, 1.0]);

    for latitude in 1..lat_segments {
        let theta = latitude as f32 / lat_segments as f32 * std::f32::consts::PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for longitude in 0..lon_segments {
            let phi = longitude as f32 / lon_segments as f32 * std::f32::consts::TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let normal = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta];
            vertices.push([radius * normal[0], radius * normal[1], radius * normal[2]]);
            normals.push(normal);
        }
    }

    let bottom = vertices.len() as u32;
    vertices.push([0.0, 0.0, -radius]);
    normals.push([0.0, 0.0, -1.0]);

    let first_ring = 1u32;
    for longitude in 0..lon_segments {
        let next = (longitude + 1) % lon_segments;
        indices.push([top, first_ring + longitude as u32, first_ring + next as u32]);
    }

    for latitude in 0..lat_segments - 2 {
        let current_ring = first_ring + (latitude * lon_segments) as u32;
        let next_ring = current_ring + lon_segments as u32;
        for longitude in 0..lon_segments {
            let next = (longitude + 1) % lon_segments;
            let a = current_ring + longitude as u32;
            let b = current_ring + next as u32;
            let c = next_ring + longitude as u32;
            let d = next_ring + next as u32;
            indices.push([a, c, b]);
            indices.push([b, c, d]);
        }
    }

    let last_ring = first_ring + ((lat_segments - 2) * lon_segments) as u32;
    for longitude in 0..lon_segments {
        let next = (longitude + 1) % lon_segments;
        indices.push([
            bottom,
            last_ring + next as u32,
            last_ring + longitude as u32,
        ]);
    }

    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

fn tessellate_cone(segments: usize) -> TriangleMesh {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let r = 0.5f32;
    let h = 1.0f32;

    // Side faces
    let slope = r / (r * r + h * h).sqrt();
    let height_comp = h / (r * r + h * h).sqrt();
    for i in 0..segments {
        let a0 = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let a1 = ((i + 1) as f32) / (segments as f32) * std::f32::consts::TAU;
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            [0.0, 0.0, h],         // Apex
            [r * c0, r * s0, 0.0], // Base edge 0
            [r * c1, r * s1, 0.0], // Base edge 1
        ]);
        normals.extend_from_slice(&[
            [
                height_comp * (c0 + c1) / 2.0,
                height_comp * (s0 + s1) / 2.0,
                slope,
            ],
            [height_comp * c0, height_comp * s0, slope],
            [height_comp * c1, height_comp * s1, slope],
        ]);
        indices.push([base, base + 1, base + 2]);
    }

    // Base cap
    let center = vertices.len() as u32;
    vertices.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 0.0, -1.0]);
    for i in 0..segments {
        let a = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        vertices.push([r * c, r * s, 0.0]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push([center, center + 1 + next as u32, center + 1 + i as u32]);
    }

    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

fn tessellate_torus(major_segments: usize, minor_segments: usize) -> TriangleMesh {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let major_r = 0.5f32;
    let minor_r = 0.2f32;

    for i in 0..=major_segments {
        let u = (i as f32) / (major_segments as f32) * std::f32::consts::TAU;
        let (su, cu) = u.sin_cos();
        for j in 0..=minor_segments {
            let v = (j as f32) / (minor_segments as f32) * std::f32::consts::TAU;
            let (sv, cv) = v.sin_cos();

            let x = (major_r + minor_r * cv) * cu;
            let y = (major_r + minor_r * cv) * su;
            let z = minor_r * sv;
            vertices.push([x, y, z]);

            let nx = cv * cu;
            let ny = cv * su;
            let nz = sv;
            normals.push([nx, ny, nz]);
        }
    }

    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = (i * (minor_segments + 1) + j) as u32;
            let b = a + (minor_segments + 1) as u32;
            indices.push([a, b, a + 1]);
            indices.push([a + 1, b, b + 1]);
        }
    }

    TriangleMesh {
        vertices,
        normals,
        indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSGNode;

    #[test]
    fn test_cube_mesh() {
        let mesh = tessellate_cube();
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_cylinder_mesh() {
        let mesh = tessellate_cylinder(24);
        assert!(mesh.triangle_count() > 0);
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
    }

    #[test]
    fn test_sphere_mesh() {
        let mesh = tessellate_sphere(16, 12);
        assert!(mesh.triangle_count() > 100);
    }

    #[test]
    fn test_cone_mesh() {
        let mesh = tessellate_cone(24);
        assert!(mesh.triangle_count() >= 24);
    }

    #[test]
    fn test_torus_mesh() {
        let mesh = tessellate_torus(24, 12);
        assert!(mesh.triangle_count() > 400);
    }

    #[test]
    fn test_resolve_primitive() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_resolve_union() {
        let tree = CSGNode::cube().union(CSGNode::cube().translate(0.25, 0.25, 0.25));
        let mesh = resolve_to_mesh(&tree);
        let volume = crate::validate::compute_signed_volume(&mesh).abs();
        assert!((volume - 1.578_125).abs() < 1.0e-3);
    }

    #[test]
    fn non_uniform_scale_uses_inverse_transpose_normals() {
        let inv_sqrt_two = 1.0 / 2.0_f32.sqrt();
        let mut mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0]],
            normals: vec![[inv_sqrt_two, inv_sqrt_two, 0.0]],
            indices: vec![],
        };
        mesh.apply_transform(&Transform3D {
            scale: [2.0, 1.0, 1.0],
            ..Default::default()
        });
        let normal = mesh.normals[0];
        assert!((normal[0] - 0.447_213_6).abs() < 1.0e-5);
        assert!((normal[1] - 0.894_427_2).abs() < 1.0e-5);
    }

    #[test]
    fn reflection_preserves_positive_solid_orientation() {
        let mesh = resolve_to_mesh(&CSGNode::cube().scale(-1.0, 1.0, 1.0));
        let report = crate::validate::validate_mesh(&mesh);
        assert!(report.inconsistent_normals.is_empty());
        assert!((report.signed_volume - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = TriangleMesh::empty();
        assert_eq!(mesh.triangle_count(), 0);
        assert!(mesh.vertices.is_empty());
    }
}
