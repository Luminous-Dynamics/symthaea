// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Triangle mesh generation from CSG trees
//!
//! Tessellates geometric primitives into triangle meshes and resolves
//! CSG operations via mesh merging (union) or placeholder pass-through
//! (subtract/intersect — full CSG mesh boolean is future work).

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

    /// Merge another mesh into this one (for CSG union approximation)
    pub fn merge(&mut self, other: &TriangleMesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.normals.extend_from_slice(&other.normals);
        for tri in &other.indices {
            self.indices
                .push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
    }

    /// Apply a transform to all vertices and normals
    pub fn apply_transform(&mut self, transform: &Transform3D) {
        for v in &mut self.vertices {
            *v = transform.apply(*v);
        }
        // For normals, we should use the inverse transpose of the rotation,
        // but for uniform scaling this simplification works
        let normal_transform = Transform3D {
            scale: [1.0, 1.0, 1.0], // Don't scale normals
            rotate: transform.rotate,
            translate: [0.0, 0.0, 0.0], // Don't translate normals
        };
        for n in &mut self.normals {
            *n = normal_transform.apply(*n);
            // Renormalize
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-10 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }
    }
}

/// Resolve a CSG tree into a triangle mesh
pub fn resolve_to_mesh(node: &CSGNode) -> TriangleMesh {
    match node {
        CSGNode::Primitive(prim) => tessellate_primitive(*prim),
        CSGNode::Transform { node, transform } => {
            let mut mesh = resolve_to_mesh(node);
            mesh.apply_transform(transform);
            mesh
        }
        CSGNode::Boolean { op, left, right } => {
            let left_mesh = resolve_to_mesh(left);
            let right_mesh = resolve_to_mesh(right);
            match op {
                BooleanOp::Union => {
                    let mut result = left_mesh;
                    result.merge(&right_mesh);
                    result
                }
                BooleanOp::Subtract => bsp::csg_subtract(&left_mesh, &right_mesh),
                BooleanOp::Intersect => bsp::csg_intersect(&left_mesh, &right_mesh),
            }
        }
    }
}

/// Tessellate a primitive into triangles
fn tessellate_primitive(prim: Primitive) -> TriangleMesh {
    match prim {
        Primitive::Cube => tessellate_cube(),
        Primitive::Cylinder => tessellate_cylinder(24),
        Primitive::Sphere => tessellate_sphere(16, 12),
        Primitive::Cone => tessellate_cone(24),
        Primitive::Torus => tessellate_torus(24, 12),
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
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let r = 0.5f32;

    for lat in 0..=lat_segments {
        let theta = (lat as f32) / (lat_segments as f32) * std::f32::consts::PI;
        let (st, ct) = theta.sin_cos();
        for lon in 0..=lon_segments {
            let phi = (lon as f32) / (lon_segments as f32) * std::f32::consts::TAU;
            let (sp, cp) = phi.sin_cos();
            let nx = st * cp;
            let ny = st * sp;
            let nz = ct;
            vertices.push([r * nx, r * ny, r * nz]);
            normals.push([nx, ny, nz]);
        }
    }

    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let a = (lat * (lon_segments + 1) + lon) as u32;
            let b = a + (lon_segments + 1) as u32;
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
        let tree = CSGNode::cube().union(CSGNode::sphere());
        let mesh = resolve_to_mesh(&tree);
        let cube_tris = tessellate_cube().triangle_count();
        let sphere_tris = tessellate_sphere(16, 12).triangle_count();
        assert_eq!(mesh.triangle_count(), cube_tris + sphere_tris);
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = TriangleMesh::empty();
        assert_eq!(mesh.triangle_count(), 0);
        assert!(mesh.vertices.is_empty());
    }
}
