// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::point::Point;
use nalgebra::SVector;

/// D-dimensional hyperplane: { x : normal · x = offset }.
/// Used for BSP partitioning and 4D cross-section slicing.
#[derive(Clone, Copy, Debug)]
pub struct Hyperplane<const D: usize> {
    pub normal: SVector<f64, D>,
    pub offset: f64,
}

impl<const D: usize> Hyperplane<D> {
    pub fn from_normal_and_point(normal: SVector<f64, D>, point: &Point<D>) -> Self {
        let n = normal.normalize();
        Self {
            normal: n,
            offset: n.dot(&point.0),
        }
    }

    pub fn new(normal: SVector<f64, D>, offset: f64) -> Self {
        Self { normal, offset }
    }

    #[inline]
    pub fn signed_distance(&self, point: &Point<D>) -> f64 {
        self.normal.dot(&point.0) - self.offset
    }

    #[inline]
    pub fn classify(&self, point: &Point<D>) -> Side {
        let d = self.signed_distance(point);
        if d > 1e-10 {
            Side::Front
        } else if d < -1e-10 {
            Side::Back
        } else {
            Side::On
        }
    }

    pub fn project(&self, point: &Point<D>) -> Point<D> {
        let d = self.signed_distance(point);
        Point(point.0 - self.normal * d)
    }

    pub fn intersect_segment(&self, a: &Point<D>, b: &Point<D>) -> Option<f64> {
        let da = self.signed_distance(a);
        let db = self.signed_distance(b);
        if da * db > 0.0 {
            return None;
        }
        let denom = da - db;
        if denom.abs() < 1e-15 {
            return None;
        }
        Some(da / denom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Front,
    Back,
    On,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_3d() {
        let hp = Hyperplane::new(SVector::from([0.0, 1.0, 0.0]), 0.0);
        assert_eq!(hp.classify(&Point::new([0.0, 1.0, 0.0])), Side::Front);
        assert_eq!(hp.classify(&Point::new([0.0, -1.0, 0.0])), Side::Back);
        assert_eq!(hp.classify(&Point::new([5.0, 0.0, 3.0])), Side::On);
    }

    #[test]
    fn signed_distance() {
        let hp = Hyperplane::new(SVector::from([1.0, 0.0]), 5.0);
        let p = Point::new([8.0, 3.0]);
        assert!((hp.signed_distance(&p) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn project() {
        let hp = Hyperplane::new(SVector::from([0.0, 1.0, 0.0]), 0.0);
        let p = Point::new([3.0, 7.0, 2.0]);
        let proj = hp.project(&p);
        assert!((proj.coord(0) - 3.0).abs() < 1e-12);
        assert!(proj.coord(1).abs() < 1e-12);
    }

    #[test]
    fn segment_intersection() {
        let hp = Hyperplane::new(SVector::from([0.0, 1.0, 0.0]), 0.0);
        let a = Point::new([0.0, -1.0, 0.0]);
        let b = Point::new([0.0, 1.0, 0.0]);
        let t = hp.intersect_segment(&a, &b).unwrap();
        assert!((t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn no_intersection_same_side() {
        let hp = Hyperplane::new(SVector::from([0.0, 1.0]), 0.0);
        assert!(hp
            .intersect_segment(&Point::new([0.0, 1.0]), &Point::new([0.0, 2.0]))
            .is_none());
    }

    #[test]
    fn hyperplane_4d_slicing() {
        let hp = Hyperplane::new(SVector::from([0.0, 0.0, 0.0, 1.0]), 0.5);
        assert_eq!(hp.classify(&Point::new([1.0, 2.0, 3.0, 0.5])), Side::On);
        assert_eq!(hp.classify(&Point::new([1.0, 2.0, 3.0, 1.0])), Side::Front);
        assert_eq!(hp.classify(&Point::new([1.0, 2.0, 3.0, 0.0])), Side::Back);
    }
}
