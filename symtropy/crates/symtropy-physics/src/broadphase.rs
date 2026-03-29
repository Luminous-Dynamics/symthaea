//! Broadphase collision detection using bounding sphere overlap.
//!
//! AABB broadphase generalizes to ND, but bounding spheres are simpler
//! and sufficient for moderate body counts (<1000). Upgrade to BVH if needed.

use crate::body::{BodyHandle, BodyType, RigidBody};

/// Pair of body handles that may be colliding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BroadphasePair(pub BodyHandle, pub BodyHandle);

/// Find all potentially colliding pairs using bounding sphere overlap.
///
/// O(n²) brute force. For <500 bodies this is faster than a BVH due to
/// cache locality and zero allocation.
pub fn find_pairs<const D: usize>(bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
    let mut pairs = Vec::new();

    for i in 0..bodies.len() {
        // Skip static-vs-static (never need collision response)
        let type_i = bodies[i].body_type;

        for j in (i + 1)..bodies.len() {
            let type_j = bodies[j].body_type;

            // Skip static-static pairs
            if type_i == BodyType::Static && type_j == BodyType::Static {
                continue;
            }

            // Bounding sphere overlap test
            let (center_i, radius_i) = bodies[i].collider.bounding_sphere();
            let (center_j, radius_j) = bodies[j].collider.bounding_sphere();

            // Translate bounding sphere centers to world space
            let world_center_i = bodies[i].transform.transform_point(&center_i);
            let world_center_j = bodies[j].transform.transform_point(&center_j);

            let dist = world_center_i.distance(&world_center_j);
            if dist <= radius_i + radius_j {
                pairs.push(BroadphasePair(bodies[i].handle, bodies[j].handle));
            }
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use symtropy_math::{Point, Sphere};

    #[test]
    fn overlapping_pair_detected() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        let pairs = find_pairs(&bodies);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn separated_pair_not_detected() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([5.0, 0.0, 0.0]), 1.0, 1.0),
        ];
        let pairs = find_pairs(&bodies);
        assert!(pairs.is_empty());
    }

    #[test]
    fn static_static_ignored() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::static_body(BodyHandle(1), Point::new([0.5, 0.0, 0.0]), Box::new(Sphere::unit())),
        ];
        let pairs = find_pairs(&bodies);
        assert!(pairs.is_empty());
    }

    #[test]
    fn static_dynamic_detected() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        let pairs = find_pairs(&bodies);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn multiple_pairs() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(2), Point::new([10.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(3), Point::new([1.0, 1.0, 0.0]), 1.0, 1.0),
        ];
        let pairs = find_pairs(&bodies);
        // 0-1 overlap (dist=1.5 < 2.0), 0-3 overlap (dist=√2≈1.41 < 2.0), 1-3 overlap (dist=√(0.25+1)≈1.12)
        assert!(pairs.len() >= 2);
    }
}
