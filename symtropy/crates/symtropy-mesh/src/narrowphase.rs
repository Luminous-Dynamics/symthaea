// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! BVH-based narrowphase for triangle meshes and multi-physics meshlet meshes.

use crate::Triangle;
use crate::TriangleMesh;
use crate::meshlet_physics::MultiPhysicsMeshletMesh;
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
                    restitution_bias: 0.0,
                });

                manifolds.push(ContactManifold {
                    body_a: mesh_handle,
                    body_b: other_handle,
                    normal: epa_res.normal,
                    points,
                    elasticity: None,
                });
            }
        }
    }

    manifolds
}

/// Generate contacts between a multi-physics virtual geometry meshlet mesh and another convex shape.
///
/// Accelerates collision detection using a midphase sphere-sphere check at the meshlet
/// cluster level, running GJK/EPA only on triangles belonging to overlapping meshlets.
pub fn generate_meshlet_contacts(
    mesh: &MultiPhysicsMeshletMesh,
    mesh_handle: BodyHandle,
    mesh_pos: &SVector<f64, 3>,
    other_shape: &dyn Shape<3>,
    other_handle: BodyHandle,
    other_pos: &SVector<f64, 3>,
) -> Vec<ContactManifold<3>> {
    let mut manifolds = Vec::new();

    // 1. Compute bounding sphere for the other shape in world space
    let (other_center, other_radius) = other_shape.bounding_sphere();
    let other_center_v =
        SVector::<f64, 3>::from([other_center.0[0], other_center.0[1], other_center.0[2]]);
    let other_sphere_world_center = other_center_v + *other_pos;

    // 2. Iterate through meshlets and perform midphase sphere-sphere check
    for meshlet in &mesh.meshlets {
        // Calculate max distance from center of mass to vertices in the cluster to find the cluster radius
        let mut max_r2 = 0.0f32;
        let start_v = meshlet.vertex_offset as usize;
        let end_v = (meshlet.vertex_offset + meshlet.vertex_count) as usize;

        for v in &mesh.vertices[start_v..end_v] {
            let dx = v.position[0] - meshlet.center_of_mass[0];
            let dy = v.position[1] - meshlet.center_of_mass[1];
            let dz = v.position[2] - meshlet.center_of_mass[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq > max_r2 {
                max_r2 = dist_sq;
            }
        }
        let meshlet_radius = (max_r2.sqrt() as f64).max(0.001);
        let meshlet_center_local = SVector::<f64, 3>::from([
            meshlet.center_of_mass[0] as f64,
            meshlet.center_of_mass[1] as f64,
            meshlet.center_of_mass[2] as f64,
        ]);
        let meshlet_center_world = meshlet_center_local + *mesh_pos;

        // Check sphere-sphere intersection
        let dist = (meshlet_center_world - other_sphere_world_center).norm();
        if dist > meshlet_radius + other_radius {
            continue; // No overlap with this meshlet cluster
        }

        // 3. Perform GJK/EPA on each triangle inside this overlapping meshlet
        let start_idx = meshlet.index_offset as usize;
        let num_triangles = meshlet.index_count as usize / 3;

        for t in 0..num_triangles {
            let i0 = meshlet.vertex_offset as usize + mesh.indices[start_idx + t * 3] as usize;
            let i1 = meshlet.vertex_offset as usize + mesh.indices[start_idx + t * 3 + 1] as usize;
            let i2 = meshlet.vertex_offset as usize + mesh.indices[start_idx + t * 3 + 2] as usize;

            let v0 = &mesh.vertices[i0];
            let v1 = &mesh.vertices[i1];
            let v2 = &mesh.vertices[i2];

            let tri = Triangle {
                vertices: [
                    SVector::<f64, 3>::from([
                        v0.position[0] as f64,
                        v0.position[1] as f64,
                        v0.position[2] as f64,
                    ]),
                    SVector::<f64, 3>::from([
                        v1.position[0] as f64,
                        v1.position[1] as f64,
                        v1.position[2] as f64,
                    ]),
                    SVector::<f64, 3>::from([
                        v2.position[0] as f64,
                        v2.position[1] as f64,
                        v2.position[2] as f64,
                    ]),
                ],
            };

            let gjk_res = gjk::intersects(&tri, mesh_pos, other_shape, other_pos);

            if gjk_res.intersecting {
                if let Some(epa_res) =
                    epa::penetration(&tri, mesh_pos, other_shape, other_pos, &gjk_res.simplex)
                {
                    let mut points = ArrayVec::new();
                    let contact_pos = tri.support(&epa_res.normal) + mesh_pos;

                    points.push(ContactPoint {
                        position: contact_pos,
                        depth: epa_res.depth,
                        lambda: 0.0,
                        restitution_bias: 0.0,
                    });

                    manifolds.push(ContactManifold {
                        body_a: mesh_handle,
                        body_b: other_handle,
                        normal: epa_res.normal,
                        points,
                        elasticity: Some(meshlet.average_elasticity as f64),
                    });
                }
            }
        }
    }

    manifolds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet_physics::MultiPhysicsVertex;
    use symtropy_math::{Point, Sphere};

    #[test]
    fn test_meshlet_narrowphase_collision() {
        // Construct simple triangle representing a flat floor
        let v1 = MultiPhysicsVertex {
            position: [-5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };
        let v2 = MultiPhysicsVertex {
            position: [5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };
        let v3 = MultiPhysicsVertex {
            position: [0.0, 0.0, 5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.5, 1.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };

        let vertices = vec![v1, v2, v3];
        let indices = vec![0, 1, 2];

        let mesh = MultiPhysicsMeshletMesh::build_prototype(vertices, indices);

        // A sphere colliding downward into the floor
        let collider_sphere = Sphere::new(Point::new([0.0, 0.5, 0.0]), 1.0); // center: [0, 0.5, 0], radius: 1.0 (penetration depth = 0.5)
        let mesh_pos = SVector::from([0.0, 0.0, 0.0]);
        let other_pos = SVector::from([0.0, 0.0, 0.0]);

        let manifolds = generate_meshlet_contacts(
            &mesh,
            BodyHandle(0),
            &mesh_pos,
            &collider_sphere,
            BodyHandle(1),
            &other_pos,
        );

        assert!(
            !manifolds.is_empty(),
            "Collision should have generated contact manifolds"
        );
        let manifold = &manifolds[0];
        assert_eq!(manifold.body_a, BodyHandle(0));
        assert_eq!(manifold.body_b, BodyHandle(1));
        assert!(
            manifold.normal[1] > 0.0,
            "Normal should point upward from the mesh triangle"
        );
        assert!(
            manifold.points[0].depth > 0.1,
            "Penetration depth should be positive, got {}",
            manifold.points[0].depth
        );
    }

    #[test]
    fn test_material_aware_compliance() {
        // Construct flat floor with steel-like modulus (stiff: 200e9)
        let v1_stiff = MultiPhysicsVertex {
            position: [-5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };
        let v2_stiff = MultiPhysicsVertex {
            position: [5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };
        let v3_stiff = MultiPhysicsVertex {
            position: [0.0, 0.0, 5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.5, 1.0],
            mass_density: 7800.0,
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        };
        let mesh_stiff = MultiPhysicsMeshletMesh::build_prototype(
            vec![v1_stiff, v2_stiff, v3_stiff],
            vec![0, 1, 2],
        );

        // Construct flat floor with rubber-like modulus (soft: 100e3)
        let v1_soft = MultiPhysicsVertex {
            position: [-5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            mass_density: 1100.0,
            elastic_modulus: 100e3,
            thermal_conductivity: 0.15,
            acoustic_impedance: 1.5e6,
        };
        let v2_soft = MultiPhysicsVertex {
            position: [5.0, 0.0, -5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            mass_density: 1100.0,
            elastic_modulus: 100e3,
            thermal_conductivity: 0.15,
            acoustic_impedance: 1.5e6,
        };
        let v3_soft = MultiPhysicsVertex {
            position: [0.0, 0.0, 5.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.5, 1.0],
            mass_density: 1100.0,
            elastic_modulus: 100e3,
            thermal_conductivity: 0.15,
            acoustic_impedance: 1.5e6,
        };
        let mesh_soft = MultiPhysicsMeshletMesh::build_prototype(
            vec![v1_soft, v2_soft, v3_soft],
            vec![0, 1, 2],
        );

        // Generate contact manifolds
        let collider_sphere = Sphere::new(Point::new([0.0, 0.5, 0.0]), 1.0);
        let mesh_pos = SVector::from([0.0, 0.0, 0.0]);
        let other_pos = SVector::from([0.0, 0.0, 0.0]);

        let stiff_manifolds = generate_meshlet_contacts(
            &mesh_stiff,
            BodyHandle(0),
            &mesh_pos,
            &collider_sphere,
            BodyHandle(1),
            &other_pos,
        );

        let soft_manifolds = generate_meshlet_contacts(
            &mesh_soft,
            BodyHandle(0),
            &mesh_pos,
            &collider_sphere,
            BodyHandle(2),
            &other_pos,
        );

        assert!(!stiff_manifolds.is_empty());
        assert!(!soft_manifolds.is_empty());

        let stiff_m = &stiff_manifolds[0];
        let soft_m = &soft_manifolds[0];

        // Check that elasticity values are correctly propagated to manifolds (with float tolerance)
        assert!((stiff_m.elasticity.unwrap() - 200e9).abs() < 1e5);
        assert!((soft_m.elasticity.unwrap() - 100e3).abs() < 1e1);
    }
}
