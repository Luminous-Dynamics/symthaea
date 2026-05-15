// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::point::Point;
use crate::shape::Shape;
use nalgebra::SVector;

/// D-dimensional hypersphere.
#[derive(Clone, Copy, Debug)]
pub struct Sphere<const D: usize> {
    pub center: Point<D>,
    pub radius: f64,
}

impl<const D: usize> Sphere<D> {
    pub fn unit() -> Self {
        Self {
            center: Point::origin(),
            radius: 1.0,
        }
    }

    pub fn new(center: Point<D>, radius: f64) -> Self {
        Self { center, radius }
    }

    #[inline]
    pub fn contains(&self, point: &Point<D>) -> bool {
        self.center.distance_squared(point) <= self.radius * self.radius
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.center.distance(&other.center) <= self.radius + other.radius
    }
}

impl<const D: usize> Shape<D> for Sphere<D> {
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        let norm = direction.norm();
        if norm < 1e-15 {
            return self.center.0;
        }
        self.center.0 + direction * (self.radius / norm)
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        (self.center, self.radius)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Shape<D>> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_origin() {
        let s = Sphere::<3>::unit();
        assert!(s.contains(&Point::origin()));
    }

    #[test]
    fn not_contains_far() {
        let s = Sphere::<3>::unit();
        assert!(!s.contains(&Point::new([2.0, 0.0, 0.0])));
    }

    #[test]
    fn support_x() {
        let s = Sphere::<3>::unit();
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = s.support(&dir);
        assert!((sp[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn intersection() {
        let a = Sphere::new(Point::new([0.0, 0.0]), 1.0);
        let b = Sphere::new(Point::new([1.5, 0.0]), 1.0);
        assert!(a.intersects(&b));
        let c = Sphere::new(Point::new([3.0, 0.0]), 1.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn support_4d() {
        let s = Sphere::new(Point::new([1.0, 2.0, 3.0, 4.0]), 2.0);
        let dir = SVector::from([0.0, 0.0, 0.0, 1.0]);
        let sp = s.support(&dir);
        assert!((sp[3] - 6.0).abs() < 1e-12);
    }
}
