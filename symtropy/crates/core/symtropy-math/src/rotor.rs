// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::bivector::Bivector;
use crate::point::Point;
use nalgebra::{SMatrix, SVector};

/// A rotor — performs rotations in D dimensions.
///
/// Internally stores the rotation matrix. This is optimal for the hot path
/// (applying rotations to many points) and avoids lossy rotor↔matrix round-trips.
/// Construction uses geometric algebra (bivectors, plane-angle), but once built
/// the matrix is the canonical representation.
///
/// Stack-allocated via `SMatrix<f64, D, D>`.
#[derive(Clone, Debug)]
pub struct Rotor<const D: usize> {
    mat: SMatrix<f64, D, D>,
}

impl<const D: usize> Rotor<D> {
    /// Identity rotor (no rotation).
    pub fn identity() -> Self {
        Self {
            mat: SMatrix::identity(),
        }
    }

    /// Rotation by `angle` radians in the plane defined by `plane` bivector.
    ///
    /// Positive angle rotates from the first axis toward the second axis
    /// of the plane (e.g., `unit_plane(0,1)` with positive angle rotates x→y).
    ///
    /// Uses Rodrigues formula: M = I - sin(θ)*B̂ + (1-cos(θ))*B̂²
    /// (The sign on sin is negative because our bivector convention has
    /// B̂[i,j]=+1, B̂[j,i]=-1, producing clockwise rotation. The negation
    /// makes positive angles go counterclockwise: axis_i → axis_j.)
    pub fn from_plane_angle(plane: &Bivector<D>, angle: f64) -> Self {
        let norm = plane.norm();
        if norm < 1e-15 || angle.abs() < 1e-15 {
            return Self::identity();
        }

        let b_hat = plane.to_matrix() / norm;
        let b_hat_sq = b_hat * b_hat;
        let mat = SMatrix::identity() - b_hat * angle.sin() + b_hat_sq * (1.0 - angle.cos());

        Self { mat }
    }

    /// Rotation from unit vector `from` to unit vector `to`.
    pub fn from_vectors(from: &SVector<f64, D>, to: &SVector<f64, D>) -> Self {
        let dot = from.dot(to).clamp(-1.0, 1.0);
        let angle = dot.acos();

        if angle.abs() < 1e-14 {
            return Self::identity();
        }

        // Exterior product (antisymmetric part)
        let mut ext = SMatrix::<f64, D, D>::zeros();
        for i in 0..D {
            for j in 0..D {
                ext[(i, j)] = from[i] * to[j] - to[i] * from[j];
            }
        }
        let plane = Bivector::from_matrix(&ext);
        Self::from_plane_angle(&plane, angle)
    }

    /// Direct construction from a rotation matrix (must be orthogonal, det=+1).
    pub fn from_matrix(mat: SMatrix<f64, D, D>) -> Self {
        Self { mat }
    }

    /// Get the rotation matrix.
    #[inline]
    pub fn to_matrix(&self) -> &SMatrix<f64, D, D> {
        &self.mat
    }

    /// Reverse (inverse): R† = R^T for rotation matrices.
    #[inline]
    pub fn reverse(&self) -> Self {
        Self {
            mat: self.mat.transpose(),
        }
    }

    /// Compose: apply `other` first, then `self`.
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            mat: self.mat * other.mat,
        }
    }

    /// Rotate a point.
    #[inline]
    pub fn rotate_point(&self, point: &Point<D>) -> Point<D> {
        Point(self.mat * point.0)
    }

    /// Rotate a vector.
    #[inline]
    pub fn rotate_vector(&self, v: &SVector<f64, D>) -> SVector<f64, D> {
        self.mat * v
    }

    /// Slerp: identity at t=0, this rotation at t=1.
    ///
    /// Extracts the rotation angle and plane from the matrix,
    /// then builds a partial rotation.
    pub fn slerp(&self, t: f64) -> Self {
        if t <= 1e-14 {
            return Self::identity();
        }
        if (t - 1.0).abs() <= 1e-14 {
            return self.clone();
        }

        // The antisymmetric part of M is -sin(θ)*B̂ (due to our sign convention)
        let antisym = (self.mat - self.mat.transpose()) / 2.0;
        // Negate to get sin(θ)*B̂, then extract plane
        let plane = Bivector::from_matrix(&(-antisym));
        let sin_theta = plane.norm();

        if sin_theta < 1e-14 {
            return Self::identity();
        }

        let trace = self.mat.trace();
        let cos_theta = ((trace - (D as f64 - 2.0)) / 2.0).clamp(-1.0, 1.0);
        let theta = f64::atan2(sin_theta, cos_theta);

        Self::from_plane_angle(&plane, theta * t)
    }
}

impl<const D: usize> Default for Rotor<D> {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn approx_vec<const N: usize>(a: &SVector<f64, N>, b: &SVector<f64, N>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10)
    }

    #[test]
    fn identity_preserves_point() {
        let r = Rotor::<4>::identity();
        let p = Point::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.rotate_point(&p), p);
    }

    #[test]
    fn rotation_2d_90() {
        let plane = Bivector::<2>::unit_plane(0, 1);
        let r = Rotor::from_plane_angle(&plane, FRAC_PI_2);
        let p = Point::new([1.0, 0.0]);
        let q = r.rotate_point(&p);
        assert!(approx_eq(q.coord(0), 0.0));
        assert!(approx_eq(q.coord(1), 1.0));
    }

    #[test]
    fn rotation_3d_xy_plane() {
        let plane = Bivector::<3>::unit_plane(0, 1);
        let r = Rotor::from_plane_angle(&plane, FRAC_PI_2);
        let p = Point::new([1.0, 0.0, 0.0]);
        let q = r.rotate_point(&p);
        assert!(approx_eq(q.coord(0), 0.0));
        assert!(approx_eq(q.coord(1), 1.0));
        assert!(approx_eq(q.coord(2), 0.0));
    }

    #[test]
    fn preserves_distance() {
        let plane = Bivector::<4>::unit_plane(1, 3);
        let r = Rotor::from_plane_angle(&plane, 1.0);
        let p = Point::new([1.0, 2.0, 3.0, 4.0]);
        let q = r.rotate_point(&p);
        assert!(approx_eq(p.0.norm(), q.0.norm()));
    }

    #[test]
    fn full_rotation() {
        let plane = Bivector::<3>::unit_plane(0, 2);
        let r = Rotor::from_plane_angle(&plane, 2.0 * PI);
        let p = Point::new([1.0, 2.0, 3.0]);
        let q = r.rotate_point(&p);
        assert!(approx_vec(&p.0, &q.0));
    }

    #[test]
    fn compose_additive() {
        let plane = Bivector::<3>::unit_plane(0, 1);
        let r1 = Rotor::from_plane_angle(&plane, FRAC_PI_4);
        let r2 = Rotor::from_plane_angle(&plane, FRAC_PI_4);
        let composed = r1.compose(&r2);
        let direct = Rotor::from_plane_angle(&plane, FRAC_PI_2);

        let p = Point::new([1.0, 0.0, 0.0]);
        assert!(approx_vec(
            &composed.rotate_point(&p).0,
            &direct.rotate_point(&p).0
        ));
    }

    #[test]
    fn reverse_undoes() {
        let plane = Bivector::<4>::unit_plane(0, 3);
        let r = Rotor::from_plane_angle(&plane, 1.7);
        let p = Point::new([1.0, 2.0, 3.0, 4.0]);
        let q = r.rotate_point(&p);
        let back = r.reverse().rotate_point(&q);
        assert!(approx_vec(&p.0, &back.0));
    }

    #[test]
    fn from_vectors_3d() {
        let from = SVector::from([1.0, 0.0, 0.0]);
        let to = SVector::from([0.0, 1.0, 0.0]);
        let r = Rotor::<3>::from_vectors(&from, &to);
        assert!(approx_vec(&r.rotate_vector(&from), &to));
    }

    #[test]
    fn from_vectors_4d() {
        let from = SVector::from([1.0, 0.0, 0.0, 0.0]);
        let to = SVector::from([0.0, 0.0, 0.0, 1.0]);
        let r = Rotor::<4>::from_vectors(&from, &to);
        assert!(approx_vec(&r.rotate_vector(&from), &to));
    }

    #[test]
    fn orthogonal_matrix() {
        let plane = Bivector::<4>::unit_plane(0, 2);
        let r = Rotor::from_plane_angle(&plane, 1.23);
        let mat = r.to_matrix();
        let product = mat * mat.transpose();
        let identity = SMatrix::<f64, 4, 4>::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (product[(i, j)] - identity[(i, j)]).abs() < 1e-10,
                    "not orthogonal at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn det_is_one() {
        let plane = Bivector::<3>::unit_plane(0, 1);
        let r = Rotor::from_plane_angle(&plane, 0.5);
        assert!(approx_eq(r.to_matrix().determinant(), 1.0));
    }

    #[test]
    fn slerp_endpoints() {
        let plane = Bivector::<3>::unit_plane(0, 1);
        let r = Rotor::from_plane_angle(&plane, 1.0);
        let p = Point::new([1.0, 0.0, 0.0]);

        let at_0 = r.slerp(0.0).rotate_point(&p);
        assert!(approx_vec(&at_0.0, &p.0));

        let at_1 = r.slerp(1.0).rotate_point(&p);
        assert!(approx_vec(&at_1.0, &r.rotate_point(&p).0));
    }

    #[test]
    fn compose_different_planes() {
        // Compose rotations in different planes
        let r1 = Rotor::from_plane_angle(&Bivector::<4>::unit_plane(0, 1), 0.3);
        let r2 = Rotor::from_plane_angle(&Bivector::<4>::unit_plane(2, 3), 0.5);
        let composed = r2.compose(&r1);

        let p = Point::new([1.0, 1.0, 1.0, 1.0]);
        let sequential = r2.rotate_point(&r1.rotate_point(&p));
        let via_compose = composed.rotate_point(&p);
        assert!(approx_vec(&sequential.0, &via_compose.0));
    }

    #[test]
    fn slerp_midpoint() {
        let plane = Bivector::<3>::unit_plane(0, 1);
        let r = Rotor::from_plane_angle(&plane, FRAC_PI_2);
        let half = r.slerp(0.5);
        let expected = Rotor::from_plane_angle(&plane, FRAC_PI_4);

        let p = Point::new([1.0, 0.0, 0.0]);
        assert!(approx_vec(
            &half.rotate_point(&p).0,
            &expected.rotate_point(&p).0
        ));
    }
}
