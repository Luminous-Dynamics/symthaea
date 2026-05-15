// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Triangle mesh collider and BVH-based narrowphase.

pub mod narrowphase;

use nalgebra::SVector;
use std::any::Any;
use symtropy_math::{Point, Shape};
use symtropy_physics::broadphase::Aabb;

pub use narrowphase::generate_mesh_contacts;

/// A single triangle in 3D.
#[derive(Clone, Debug)]
pub struct Triangle {
    pub vertices: [SVector<f64, 3>; 3],
}

impl Shape<3> for Triangle {
    fn support(&self, direction: &SVector<f64, 3>) -> SVector<f64, 3> {
        let mut best_dist = f64::MIN;
        let mut best_v = self.vertices[0];
        for v in &self.vertices {
            let dist = v.dot(direction);
            if dist > best_dist {
                best_dist = dist;
                best_v = *v;
            }
        }
        best_v
    }

    fn bounding_sphere(&self) -> (Point<3>, f64) {
        let center_v = (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3.0;
        let mut max_r2 = 0.0;
        for v in &self.vertices {
            let r2 = (v - center_v).norm_squared();
            if r2 > max_r2 {
                max_r2 = r2;
            }
        }
        (
            Point::new([center_v[0], center_v[1], center_v[2]]),
            max_r2.sqrt(),
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Shape<3>> {
        Box::new(self.clone())
    }
}

/// Node in a Bounding Volume Hierarchy for meshes.
#[derive(Clone, Debug)]
pub struct MeshBvhNode {
    pub aabb: Aabb<3>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub triangle_idx: Option<usize>,
}

/// Concave triangle mesh.
#[derive(Clone)]
pub struct TriangleMesh {
    pub triangles: Vec<Triangle>,
    pub bvh_nodes: Vec<MeshBvhNode>,
    pub root_idx: Option<usize>,
    pub center: Point<3>,
    pub radius: f64,
}

impl TriangleMesh {
    pub fn new(triangles: Vec<Triangle>) -> Self {
        let mut world_min = SVector::<f64, 3>::from_element(f64::MAX);
        let mut world_max = SVector::<f64, 3>::from_element(f64::MIN);

        for tri in &triangles {
            for v in &tri.vertices {
                for i in 0..3 {
                    if v[i] < world_min[i] {
                        world_min[i] = v[i];
                    }
                    if v[i] > world_max[i] {
                        world_max[i] = v[i];
                    }
                }
            }
        }

        let center_v = (world_min + world_max) * 0.5;
        let center = Point::new([center_v[0], center_v[1], center_v[2]]);
        let mut max_r2 = 0.0;
        for tri in &triangles {
            for v in &tri.vertices {
                let r2 = (v - center_v).norm_squared();
                if r2 > max_r2 {
                    max_r2 = r2;
                }
            }
        }

        let mut bvh_nodes = Vec::new();
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let root_idx = if !triangles.is_empty() {
            Some(Self::build_bvh_rec(
                &triangles,
                &mut indices,
                &mut bvh_nodes,
            ))
        } else {
            None
        };

        Self {
            triangles,
            bvh_nodes,
            root_idx,
            center,
            radius: max_r2.sqrt(),
        }
    }

    fn build_bvh_rec(
        triangles: &[Triangle],
        indices: &mut [usize],
        nodes: &mut Vec<MeshBvhNode>,
    ) -> usize {
        let n = indices.len();
        let mut aabb = Aabb::empty();
        for &idx in indices.iter() {
            let (c, r) = triangles[idx].bounding_sphere();
            aabb = aabb.union(&Aabb::from_sphere(&c.0, r));
        }

        if n == 1 {
            let idx = nodes.len();
            nodes.push(MeshBvhNode {
                aabb,
                left: None,
                right: None,
                triangle_idx: Some(indices[0]),
            });
            return idx;
        }

        let extents = aabb.max - aabb.min;
        let axis = if extents[0] > extents[1] && extents[0] > extents[2] {
            0
        } else if extents[1] > extents[2] {
            1
        } else {
            2
        };

        indices.sort_by(|&a, &b| {
            let ca = triangles[a].bounding_sphere().0[axis];
            let cb = triangles[b].bounding_sphere().0[axis];
            ca.partial_cmp(&cb).unwrap()
        });

        let mid = n / 2;
        let left_idx = Self::build_bvh_rec(triangles, &mut indices[..mid], nodes);
        let right_idx = Self::build_bvh_rec(triangles, &mut indices[mid..], nodes);

        let idx = nodes.len();
        nodes.push(MeshBvhNode {
            aabb,
            left: Some(left_idx),
            right: Some(right_idx),
            triangle_idx: None,
        });
        idx
    }

    pub fn query_overlap(&self, query_aabb: &Aabb<3>) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        if let Some(root) = self.root_idx {
            stack.push(root);
        }

        while let Some(idx) = stack.pop() {
            let node = &self.bvh_nodes[idx];
            if !node.aabb.overlaps(query_aabb) {
                continue;
            }

            if let Some(tri_idx) = node.triangle_idx {
                result.push(tri_idx);
            } else {
                if let Some(l) = node.left {
                    stack.push(l);
                }
                if let Some(r) = node.right {
                    stack.push(r);
                }
            }
        }
        result
    }
}

impl symtropy_physics::manifold_gen::MeshColliderMetadata<3> for TriangleMesh {
    fn generate_mesh_contacts(
        &self,
        self_handle: symtropy_physics::body::BodyHandle,
        self_pos: &SVector<f64, 3>,
        other_shape: &dyn Shape<3>,
        other_handle: symtropy_physics::body::BodyHandle,
        other_pos: &SVector<f64, 3>,
    ) -> Vec<symtropy_physics::contact::ContactManifold<3>> {
        crate::narrowphase::generate_mesh_contacts(
            self,
            self_handle,
            self_pos,
            other_shape,
            other_handle,
            other_pos,
        )
    }
}

impl Shape<3> for TriangleMesh {
    fn support(&self, direction: &SVector<f64, 3>) -> SVector<f64, 3> {
        let mut best_dist = f64::MIN;
        let mut best_v = self.triangles[0].vertices[0];
        for tri in &self.triangles {
            for v in &tri.vertices {
                let dist = v.dot(direction);
                if dist > best_dist {
                    best_dist = dist;
                    best_v = *v;
                }
            }
        }
        best_v
    }

    fn bounding_sphere(&self) -> (Point<3>, f64) {
        (self.center, self.radius)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Shape<3>> {
        Box::new(self.clone())
    }
}
