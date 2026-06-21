// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use nalgebra::SVector;
use std::ops::{Add, Sub};

/// A point in D-dimensional space. Stack-allocated via nalgebra::SVector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point<const D: usize>(pub SVector<f64, D>);

impl<const D: usize> Point<D> {
    /// Point at the origin.
    pub fn origin() -> Self {
        Self(SVector::zeros())
    }

    /// Create from a fixed-size array.
    pub fn new(coords: [f64; D]) -> Self {
        Self(SVector::from(coords))
    }

    /// Euclidean distance to another point.
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        (self.0 - other.0).norm()
    }

    /// Squared distance (avoids sqrt — use for comparisons).
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> f64 {
        (self.0 - other.0).norm_squared()
    }

    /// Linear interpolation: self*(1-t) + other*t.
    #[inline]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self(self.0 * (1.0 - t) + other.0 * t)
    }

    /// Access coordinate by index.
    #[inline]
    pub fn coord(&self, i: usize) -> f64 {
        self.0[i]
    }

    /// Mutable coordinate access.
    #[inline]
    pub fn coord_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }

    /// Convert to direction vector.
    #[inline]
    pub fn to_vector(&self) -> SVector<f64, D> {
        self.0
    }
}

impl<const D: usize> std::ops::Index<usize> for Point<D> {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> std::ops::IndexMut<usize> for Point<D> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize> Add<SVector<f64, D>> for Point<D> {
    type Output = Point<D>;
    #[inline]
    fn add(self, rhs: SVector<f64, D>) -> Point<D> {
        Point(self.0 + rhs)
    }
}

impl<const D: usize> Sub for Point<D> {
    type Output = SVector<f64, D>;
    #[inline]
    fn sub(self, rhs: Point<D>) -> SVector<f64, D> {
        self.0 - rhs.0
    }
}

impl<const D: usize> Default for Point<D> {
    fn default() -> Self {
        Self::origin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_zero() {
        let p = Point::<4>::origin();
        for i in 0..4 {
            assert_eq!(p.coord(i), 0.0);
        }
    }

    #[test]
    fn distance_2d() {
        let a = Point::new([0.0, 0.0]);
        let b = Point::new([3.0, 4.0]);
        assert!((a.distance(&b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn distance_4d() {
        let a = Point::new([1.0, 2.0, 3.0, 4.0]);
        let b = Point::new([5.0, 6.0, 7.0, 8.0]);
        assert!((a.distance(&b) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn lerp_midpoint() {
        let a = Point::new([0.0, 0.0, 0.0]);
        let b = Point::new([10.0, 20.0, 30.0]);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.coord(0) - 5.0).abs() < 1e-12);
        assert!((mid.coord(1) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn subtraction() {
        let a = Point::new([1.0, 2.0]);
        let b = Point::new([4.0, 6.0]);
        let v = b - a;
        assert!((v[0] - 3.0).abs() < 1e-12);
        assert!((v[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn is_copy() {
        let a = Point::new([1.0, 2.0]);
        let b = a;
        assert_eq!(a, b);
    }
}
