// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use nalgebra::{SMatrix, SVector};

/// Maximum supported dimension for bivectors.
/// D=4 → 6, D=5 → 10, D=6 → 15, D=7 → 21, D=8 → 28, D=9 → 36.
/// Supports up to 9D physics for dimensional sweep experiments.
const MAX_BIVECTOR_COMPONENTS: usize = 36;

/// A bivector in D-dimensional space (grade-2 element of the geometric algebra).
///
/// Bivectors represent oriented planes. In 3D they are dual to pseudovectors.
/// In ND they properly represent rotations: a rotation happens IN a plane.
///
/// **Stack-allocated**: uses `[f64; 6]` fixed array (max for D=4).
/// Zero heap allocation — safe for physics hot loops at 60+ Hz.
///
/// Component ordering: lexicographic (i,j) with i < j.
/// For D=4: [e01, e02, e03, e12, e13, e23].
#[derive(Clone, Copy, Debug)]
pub struct Bivector<const D: usize> {
    components: [f64; MAX_BIVECTOR_COMPONENTS],
}

// Compile-time assertion: D must be ≤ 9 (36 bivector components max)
const fn _assert_d_le_9<const D: usize>() {
    assert!(
        D * (D - 1) / 2 <= MAX_BIVECTOR_COMPONENTS,
        "Bivector supports D <= 9 (max 36 components)"
    );
}

impl<const D: usize> Bivector<D> {
    /// Number of independent components: D*(D-1)/2.
    pub const fn num_components() -> usize {
        D * (D - 1) / 2
    }

    /// Zero bivector.
    #[inline]
    pub fn zero() -> Self {
        _assert_d_le_9::<D>();
        Self {
            components: [0.0; MAX_BIVECTOR_COMPONENTS],
        }
    }

    /// Unit bivector in the plane spanned by axes i and j (i < j).
    pub fn unit_plane(i: usize, j: usize) -> Self {
        assert!(i < j && j < D, "requires i < j < D");
        let mut bv = Self::zero();
        bv.set(i, j, 1.0);
        bv
    }

    /// Flat index for the (i,j) component with i < j.
    #[inline]
    fn index(i: usize, j: usize) -> usize {
        debug_assert!(i < j && j < D);
        i * D - i * (i + 1) / 2 + j - i - 1
    }

    /// Get the component for the (i, j) plane (i < j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.components[Self::index(i, j)]
    }

    /// Set the component for the (i, j) plane (i < j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.components[Self::index(i, j)] = value;
    }

    /// Convert to antisymmetric matrix representation.
    pub fn to_matrix(&self) -> SMatrix<f64, D, D> {
        let mut mat = SMatrix::zeros();
        for i in 0..D {
            for j in (i + 1)..D {
                let val = self.get(i, j);
                mat[(i, j)] = val;
                mat[(j, i)] = -val;
            }
        }
        mat
    }

    /// Construct from an antisymmetric matrix.
    pub fn from_matrix(mat: &SMatrix<f64, D, D>) -> Self {
        let mut bv = Self::zero();
        for i in 0..D {
            for j in (i + 1)..D {
                bv.set(i, j, (mat[(i, j)] - mat[(j, i)]) / 2.0);
            }
        }
        bv
    }

    /// Returns true if all components are finite (not NaN or infinite).
    #[inline]
    pub fn is_finite(&self) -> bool {
        let n = Self::num_components();
        for i in 0..n {
            if !self.components[i].is_finite() {
                return false;
            }
        }
        true
    }

    /// Squared norm.
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        let n = Self::num_components();
        let mut sum = 0.0;
        for i in 0..n {
            sum += self.components[i] * self.components[i];
        }
        sum
    }

    /// Norm.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit bivector. Returns None if zero.
    pub fn normalized(&self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-15 {
            return None;
        }
        Some(self.scale(1.0 / n))
    }

    /// Scale all components by a scalar.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let mut result = *self;
        let n = Self::num_components();
        for i in 0..n {
            result.components[i] *= s;
        }
        result
    }

    /// Add two bivectors.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = *self;
        let n = Self::num_components();
        for i in 0..n {
            result.components[i] += other.components[i];
        }
        result
    }

    /// Apply this bivector (as angular velocity) to a vector.
    /// Returns B · v (antisymmetric matrix × vector).
    #[inline]
    pub fn apply_to_vector(&self, v: &SVector<f64, D>) -> SVector<f64, D> {
        self.to_matrix() * v
    }
}

impl<const D: usize> Default for Bivector<D> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize> PartialEq for Bivector<D> {
    fn eq(&self, other: &Self) -> bool {
        let n = Self::num_components();
        for i in 0..n {
            if (self.components[i] - other.components[i]).abs() > 1e-14 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_components() {
        assert_eq!(Bivector::<2>::num_components(), 1);
        assert_eq!(Bivector::<3>::num_components(), 3);
        assert_eq!(Bivector::<4>::num_components(), 6);
    }

    #[test]
    fn unit_plane_3d() {
        let bv = Bivector::<3>::unit_plane(0, 1);
        assert!((bv.get(0, 1) - 1.0).abs() < 1e-15);
        assert!(bv.get(0, 2).abs() < 1e-15);
        assert!(bv.get(1, 2).abs() < 1e-15);
    }

    #[test]
    fn matrix_roundtrip() {
        let mut bv = Bivector::<4>::zero();
        bv.set(0, 1, 1.0);
        bv.set(0, 3, -0.5);
        bv.set(2, 3, 0.7);
        let mat = bv.to_matrix();
        let recovered = Bivector::<4>::from_matrix(&mat);
        assert_eq!(bv, recovered);
    }

    #[test]
    fn antisymmetric_matrix() {
        let bv = Bivector::<3>::unit_plane(0, 2);
        let mat = bv.to_matrix();
        assert!((mat[(0, 2)] - 1.0).abs() < 1e-15);
        assert!((mat[(2, 0)] + 1.0).abs() < 1e-15);
    }

    #[test]
    fn angular_velocity_3d() {
        let bv = Bivector::<3>::unit_plane(0, 1);
        let v = SVector::from([1.0, 0.0, 0.0]);
        let result = bv.apply_to_vector(&v);
        assert!(result[0].abs() < 1e-15);
        assert!((result[1] + 1.0).abs() < 1e-15);
        assert!(result[2].abs() < 1e-15);
    }

    #[test]
    fn norm_and_normalize() {
        let mut bv = Bivector::<3>::zero();
        bv.set(0, 1, 3.0);
        bv.set(0, 2, 4.0);
        assert!((bv.norm() - 5.0).abs() < 1e-14);
        let unit = bv.normalized().unwrap();
        assert!((unit.norm() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn four_d_double_rotation() {
        let mut bv = Bivector::<4>::zero();
        bv.set(0, 1, 1.0);
        bv.set(2, 3, 1.0);
        assert!((bv.norm_squared() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn is_copy() {
        let a = Bivector::<3>::unit_plane(0, 1);
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn scale_no_alloc() {
        let bv = Bivector::<4>::unit_plane(0, 1);
        let scaled = bv.scale(3.0);
        assert!((scaled.get(0, 1) - 3.0).abs() < 1e-14);
        // Original unchanged (Copy semantics)
        assert!((bv.get(0, 1) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn add_no_alloc() {
        let a = Bivector::<3>::unit_plane(0, 1);
        let b = Bivector::<3>::unit_plane(0, 2);
        let c = a.add(&b);
        assert!((c.get(0, 1) - 1.0).abs() < 1e-14);
        assert!((c.get(0, 2) - 1.0).abs() < 1e-14);
    }
}
