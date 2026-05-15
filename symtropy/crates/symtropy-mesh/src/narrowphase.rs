// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! BVH-based narrowphase for triangle meshes.

use crate::TriangleMesh;
use arrayvec::ArrayVec;
use nalgebra::SVector;
use symtropy_math::Shape;
use symtropy_physics::body::BodyHandle;
use symtropy_physics::broadphase::Aabb;
use symtropy_physics::contact::{ContactManifold, ContactPoint};
use symtropy_physics::{epa, gjk};

/// Generate contacts between a triangle mesh and another convex shape.
pub fn generate_mesh_contacts(
    mesh: &TriangleMesh,
    mesh_handle: BodyHandle,
    mesh_pos: &SVector<f64, 3>,
    other_shape: &dyn Shape<3>,
    other_handle: BodyHandle,
    other_pos: &SVector<f64, 3>,
) -> Vec<ContactManifold<3>> {
    let mut manifolds = Vec::new();

    // 1. Compute AABB for the other shape
    let (center, radius) = other_shape.bounding_sphere();
    let center_v = SVector::<f64, 3>::from([center.0[0], center.0[1], center.0[2]]);
    let other_aabb_world = Aabb {
        min: center_v + *other_pos - SVector::from_element(radius),
        max: center_v + *other_pos + SVector::from_element(radius),
    };

    // 2. Query BVH for overlapping triangles
    // Need AABB in mesh local space
    let other_aabb_local = Aabb {
        min: other_aabb_world.min - *mesh_pos,
        max: other_aabb_world.max - *mesh_pos,
    };

    let tri_indices = mesh.query_overlap(&other_aabb_local);

    // 3. Perform GJK/EPA for each candidate triangle
    for idx in tri_indices {
        let tri = &mesh.triangles[idx];

        // GJK in world space
        let gjk_res = gjk::intersects(tri, mesh_pos, other_shape, other_pos);

        if gjk_res.intersecting {
            // EPA for penetration
            if let Some(epa_res) =
                epa::penetration(tri, mesh_pos, other_shape, other_pos, &gjk_res.simplex)
            {
                // Generate multi-point manifold (simplified: single point for now)
                let mut points = ArrayVec::new();

                // Contact point is tri.support(normal) + mesh_pos
                let contact_pos = tri.support(&epa_res.normal) + mesh_pos;

                points.push(ContactPoint {
                    position: contact_pos,
                    depth: epa_res.depth,
                    lambda: 0.0,
                });

                manifolds.push(ContactManifold {
                    body_a: mesh_handle,
                    body_b: other_handle,
                    normal: epa_res.normal,
                    points,
                });
            }
        }
    }

    manifolds
}
