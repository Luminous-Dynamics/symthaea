// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::point::Point;
use crate::shape::Shape;
use nalgebra::SVector;

/// D-dimensional convex hull defined by a set of vertices.
///
/// The support function iterates over all vertices, which is O(n) per query.
/// For large vertex counts, use a spatial acceleration structure. For physics
/// colliders with <100 vertices, linear scan is faster due to cache locality.
#[derive(Clone, Debug)]
pub struct ConvexHull<const D: usize> {
    pub vertices: Vec<SVector<f64, D>>,
    /// Cached bounding sphere for broadphase.
    center: SVector<f64, D>,
    radius: f64,
}

impl<const D: usize> ConvexHull<D> {
    /// Create from a set of vertices.
    pub fn new(vertices: Vec<SVector<f64, D>>) -> Self {
        assert!(
            !vertices.is_empty(),
            "convex hull needs at least one vertex"
        );

        // Compute centroid
        let mut center = SVector::zeros();
        for v in &vertices {
            center += v;
        }
        center /= vertices.len() as f64;

        // Compute bounding radius
        let radius = vertices
            .iter()
            .map(|v| (v - center).norm())
            .fold(0.0f64, f64::max);

        Self {
            vertices,
            center,
            radius,
        }
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Create an axis-aligned hyperbox centered at origin with given half-extents.
    pub fn hyperbox(half_extents: [f64; D]) -> Self {
        // 2^D vertices
        let n = 1 << D;
        let mut vertices = Vec::with_capacity(n);
        for mask in 0..n {
            let mut v = SVector::zeros();
            for axis in 0..D {
                v[axis] = if (mask >> axis) & 1 == 0 {
                    -half_extents[axis]
                } else {
                    half_extents[axis]
                };
            }
            vertices.push(v);
        }
        Self::new(vertices)
    }

    /// Unit hypercube centered at origin: side length 2, half-extent 1 per axis.
    pub fn unit_cube() -> Self {
        Self::hyperbox([1.0; D])
    }
}

impl<const D: usize> Shape<D> for ConvexHull<D> {
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        let mut best = self.vertices[0];
        let mut best_dot = direction.dot(&best);
        for v in &self.vertices[1..] {
            let d = direction.dot(v);
            if d > best_dot {
                best_dot = d;
                best = *v;
            }
        }
        best
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        (Point(self.center), self.radius)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Shape<D>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube_2d() {
        let cube = ConvexHull::<2>::unit_cube();
        assert_eq!(cube.num_vertices(), 4); // 2^2 = 4 vertices
    }

    #[test]
    fn unit_cube_3d() {
        let cube = ConvexHull::<3>::unit_cube();
        assert_eq!(cube.num_vertices(), 8); // 2^3 = 8
    }

    #[test]
    fn unit_cube_4d() {
        let cube = ConvexHull::<4>::unit_cube();
        assert_eq!(cube.num_vertices(), 16); // 2^4 = 16 (tesseract)
    }

    #[test]
    fn support_returns_correct_vertex() {
        let cube = ConvexHull::<3>::unit_cube();
        let dir = SVector::from([1.0, 1.0, 1.0]);
        let sp = cube.support(&dir);
        // Should be the (1,1,1) vertex
        assert!((sp[0] - 1.0).abs() < 1e-12);
        assert!((sp[1] - 1.0).abs() < 1e-12);
        assert!((sp[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn support_negative_direction() {
        let cube = ConvexHull::<3>::unit_cube();
        let dir = SVector::from([-1.0, 0.0, 0.0]);
        let sp = cube.support(&dir);
        assert!((sp[0] + 1.0).abs() < 1e-12); // -1 in x
    }

    #[test]
    fn bounding_sphere_contains_all_vertices() {
        let cube = ConvexHull::<4>::unit_cube();
        let (center, radius) = cube.bounding_sphere();
        for v in &cube.vertices {
            let dist = Point(*v).distance(&center);
            assert!(dist <= radius + 1e-10);
        }
    }

    #[test]
    fn custom_hyperbox() {
        let box3 = ConvexHull::<3>::hyperbox([2.0, 3.0, 4.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = box3.support(&dir);
        assert!((sp[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn triangle_support() {
        let tri = ConvexHull::<2>::new(vec![
            SVector::from([0.0, 0.0]),
            SVector::from([1.0, 0.0]),
            SVector::from([0.5, 1.0]),
        ]);
        let dir = SVector::from([0.0, 1.0]);
        let sp = tri.support(&dir);
        assert!((sp[1] - 1.0).abs() < 1e-12);
    }
}
