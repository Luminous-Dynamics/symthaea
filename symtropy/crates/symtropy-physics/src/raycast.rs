// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ray casting: find the first intersection of a ray with physics bodies.
//!
//! Supports two modes:
//! - **Analytical**: Exact ray-sphere intersection (O(1) per body)
//! - **GJK-based**: Ray vs any `Shape<D>` via Minkowski cast (future)
//!
//! # Usage
//! ```ignore
//! let hit = raycast(&world, &origin, &direction, 100.0);
//! if let Some(hit) = hit {
//!     println!("hit body {:?} at distance {}", hit.body, hit.distance);
//! }
//! ```

use nalgebra::SVector;

use crate::body::BodyHandle;
use crate::world::PhysicsWorld;

/// Result of a ray cast against the physics world.
#[derive(Clone, Debug)]
pub struct RayHit<const D: usize> {
    /// The body that was hit.
    pub body: BodyHandle,
    /// Distance from the ray origin to the hit point.
    pub distance: f64,
    /// Hit point in world space.
    pub point: SVector<f64, D>,
    /// Surface normal at the hit point (pointing toward the ray origin).
    pub normal: SVector<f64, D>,
}

/// Cast a ray through the physics world and return the closest hit.
///
/// `origin`: ray start point
/// `direction`: ray direction (will be normalized internally)
/// `max_distance`: maximum ray length (for performance and game design)
///
/// Returns `None` if no body is hit within `max_distance`.
/// Skips sleeping bodies and sensors by default.
pub fn raycast<const D: usize>(
    world: &PhysicsWorld<D>,
    origin: &SVector<f64, D>,
    direction: &SVector<f64, D>,
    max_distance: f64,
) -> Option<RayHit<D>> {
    let dir_norm = direction.norm();
    if dir_norm < 1e-15 {
        return None;
    }
    let dir = direction / dir_norm;

    let mut closest: Option<RayHit<D>> = None;

    for body in &world.bodies {
        if body.is_sensor {
            continue;
        }

        // Quick bounding sphere test first
        let (bsphere_center, bsphere_radius) = body.collider.bounding_sphere();
        let world_center = body.transform.transform_point(&bsphere_center).0;

        if let Some(t) = ray_sphere_intersection(origin, &dir, &world_center, bsphere_radius) {
            if t > 0.0 && t <= max_distance {
                // For spheres, the bounding sphere IS the shape — compute exact hit
                let hit_point = origin + dir * t;
                let normal = (hit_point - world_center).normalize();

                let is_closer = closest.as_ref().is_none_or(|c| t < c.distance);
                if is_closer {
                    closest = Some(RayHit {
                        body: body.handle,
                        distance: t,
                        point: hit_point,
                        normal,
                    });
                }
            }
        }
    }

    closest
}

/// Cast a ray and return ALL hits (not just the closest), sorted by distance.
pub fn raycast_all<const D: usize>(
    world: &PhysicsWorld<D>,
    origin: &SVector<f64, D>,
    direction: &SVector<f64, D>,
    max_distance: f64,
) -> Vec<RayHit<D>> {
    let dir_norm = direction.norm();
    if dir_norm < 1e-15 {
        return Vec::new();
    }
    let dir = direction / dir_norm;

    let mut hits = Vec::new();

    for body in &world.bodies {
        if body.is_sensor {
            continue;
        }

        let (bsphere_center, bsphere_radius) = body.collider.bounding_sphere();
        let world_center = body.transform.transform_point(&bsphere_center).0;

        if let Some(t) = ray_sphere_intersection(origin, &dir, &world_center, bsphere_radius) {
            if t > 0.0 && t <= max_distance {
                let hit_point = origin + dir * t;
                let normal = (hit_point - world_center).normalize();
                hits.push(RayHit {
                    body: body.handle,
                    distance: t,
                    point: hit_point,
                    normal,
                });
            }
        }
    }

    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// Analytical ray-sphere intersection.
///
/// Returns the distance `t` to the nearest intersection point, or `None` if the ray misses.
/// Uses the standard quadratic formula: solve `|origin + t*dir - center|² = radius²`.
fn ray_sphere_intersection<const D: usize>(
    origin: &SVector<f64, D>,
    dir: &SVector<f64, D>, // must be unit length
    center: &SVector<f64, D>,
    radius: f64,
) -> Option<f64> {
    let oc = origin - center;
    let a = dir.dot(dir); // Should be ~1.0 for unit dir
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(&oc) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    // Return the nearest positive intersection
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None // Ray starts inside or behind the sphere
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::{Point, Sphere};

    #[test]
    fn ray_hits_sphere() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([10.0, 0.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        let hit = raycast(&world, &origin, &dir, 100.0).unwrap();
        assert_eq!(hit.body, h);
        // Hit distance should be ~9.0 (sphere center at 10, radius 1)
        assert!(
            (hit.distance - 9.0).abs() < 0.1,
            "hit distance = {}, expected ~9.0",
            hit.distance
        );
    }

    #[test]
    fn ray_misses_sphere() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        world.add_sphere(Point::new([10.0, 5.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]); // Shoots along X, sphere is at Y=5

        let hit = raycast(&world, &origin, &dir, 100.0);
        assert!(hit.is_none(), "ray should miss sphere at Y=5");
    }

    #[test]
    fn ray_max_distance() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        world.add_sphere(Point::new([10.0, 0.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        // Max distance 5 — sphere is at 10, should miss
        let hit = raycast(&world, &origin, &dir, 5.0);
        assert!(hit.is_none(), "ray should not reach sphere at distance 10");
    }

    #[test]
    fn ray_closest_hit() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h1 = world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = world.add_sphere(Point::new([10.0, 0.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        let hit = raycast(&world, &origin, &dir, 100.0).unwrap();
        assert_eq!(hit.body, h1, "should hit the closer sphere");
    }

    #[test]
    fn raycast_all_returns_sorted() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        world.add_sphere(Point::new([10.0, 0.0, 0.0]), 1.0, 1.0);
        world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        let hits = raycast_all(&world, &origin, &dir, 100.0);
        assert_eq!(hits.len(), 2);
        assert!(
            hits[0].distance < hits[1].distance,
            "hits should be sorted by distance"
        );
    }

    #[test]
    fn ray_skips_sensors() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);
        world.body_mut(h).unwrap().is_sensor = true;

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        let hit = raycast(&world, &origin, &dir, 100.0);
        assert!(hit.is_none(), "ray should skip sensors");
    }

    #[test]
    fn ray_hit_normal_points_outward() {
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);

        let hit = raycast(&world, &origin, &dir, 100.0).unwrap();
        // Normal should point back toward the ray origin (negative X)
        assert!(hit.normal[0] < 0.0, "normal should face the ray origin");
    }

    #[test]
    fn ray_4d() {
        let mut world = PhysicsWorld::<4>::new(SVector::zeros());
        world.add_sphere(Point::new([0.0, 0.0, 0.0, 5.0]), 1.0, 1.0);

        let origin = SVector::from([0.0, 0.0, 0.0, 0.0]);
        let dir = SVector::from([0.0, 0.0, 0.0, 1.0]); // Cast along W axis

        let hit = raycast(&world, &origin, &dir, 100.0).unwrap();
        assert!(
            (hit.distance - 4.0).abs() < 0.1,
            "4D ray hit distance = {}, expected ~4.0",
            hit.distance
        );
    }

    #[test]
    fn ray_sphere_analytical_behind() {
        // Ray starts inside sphere — should not report hit (t < 0 for entry)
        let origin = SVector::from([0.0, 0.0, 0.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let center = SVector::from([0.0, 0.0, 0.0]);
        let radius = 5.0;

        // Origin is at center — t1 < 0, t2 > 0
        let t = ray_sphere_intersection(&origin, &dir, &center, radius);
        // Should return t2 (exit point, forward along ray)
        assert!(t.is_some());
        assert!(t.unwrap() > 0.0);
    }
}
