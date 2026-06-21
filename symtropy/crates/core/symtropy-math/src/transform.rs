// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::point::Point;
use crate::rotor::Rotor;
use nalgebra::SVector;

/// N-dimensional rigid body transform: rotation + translation.
/// Applied as: p' = R(p) + t
#[derive(Clone, Debug)]
pub struct Transform<const D: usize> {
    pub translation: Point<D>,
    pub rotation: Rotor<D>,
}

impl<const D: usize> Transform<D> {
    pub fn identity() -> Self {
        Self {
            translation: Point::origin(),
            rotation: Rotor::identity(),
        }
    }

    pub fn from_translation(t: Point<D>) -> Self {
        Self {
            translation: t,
            rotation: Rotor::identity(),
        }
    }

    pub fn from_rotation(r: Rotor<D>) -> Self {
        Self {
            translation: Point::origin(),
            rotation: r,
        }
    }

    #[inline]
    pub fn transform_point(&self, point: &Point<D>) -> Point<D> {
        let rotated = self.rotation.rotate_point(point);
        Point(rotated.0 + self.translation.0)
    }

    #[inline]
    pub fn transform_vector(&self, v: &SVector<f64, D>) -> SVector<f64, D> {
        self.rotation.rotate_vector(v)
    }

    pub fn compose(&self, other: &Self) -> Self {
        let rotation = self.rotation.compose(&other.rotation);
        let t = self.rotation.rotate_vector(&other.translation.0);
        Self {
            translation: Point(self.translation.0 + t),
            rotation,
        }
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.reverse();
        let neg_t = inv_rot.rotate_vector(&self.translation.0) * -1.0;
        Self {
            translation: Point(neg_t),
            rotation: inv_rot,
        }
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let translation = self.translation.lerp(&other.translation, t);
        let relative = other.rotation.compose(&self.rotation.reverse());
        let rotation = relative.slerp(t).compose(&self.rotation);
        Self {
            translation,
            rotation,
        }
    }
}

impl<const D: usize> Default for Transform<D> {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivector::Bivector;
    use std::f64::consts::FRAC_PI_2;

    fn approx_vec<const N: usize>(a: &SVector<f64, N>, b: &SVector<f64, N>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10)
    }

    #[test]
    fn identity_preserves() {
        let t = Transform::<3>::identity();
        let p = Point::new([1.0, 2.0, 3.0]);
        assert!(approx_vec(&p.0, &t.transform_point(&p).0));
    }

    #[test]
    fn pure_translation() {
        let t = Transform::from_translation(Point::new([10.0, 20.0]));
        let p = Point::new([1.0, 2.0]);
        let q = t.transform_point(&p);
        assert!((q.coord(0) - 11.0).abs() < 1e-10);
        assert!((q.coord(1) - 22.0).abs() < 1e-10);
    }

    #[test]
    fn inverse_roundtrip() {
        let plane = Bivector::<4>::unit_plane(0, 2);
        let t = Transform {
            translation: Point::new([1.0, 2.0, 3.0, 4.0]),
            rotation: Rotor::from_plane_angle(&plane, 1.0),
        };
        let p = Point::new([5.0, 6.0, 7.0, 8.0]);
        let back = t.inverse().transform_point(&t.transform_point(&p));
        assert!(approx_vec(&p.0, &back.0));
    }

    #[test]
    fn compose_equals_sequential() {
        let t1 = Transform {
            translation: Point::new([1.0, 0.0, 0.0]),
            rotation: Rotor::from_plane_angle(&Bivector::<3>::unit_plane(0, 1), 0.5),
        };
        let t2 = Transform {
            translation: Point::new([0.0, 2.0, 0.0]),
            rotation: Rotor::from_plane_angle(&Bivector::<3>::unit_plane(1, 2), 0.3),
        };
        let p = Point::new([1.0, 1.0, 1.0]);
        let seq = t2.transform_point(&t1.transform_point(&p));
        let composed = t2.compose(&t1).transform_point(&p);
        assert!(approx_vec(&seq.0, &composed.0));
    }

    #[test]
    fn rotate_then_translate() {
        let t = Transform {
            translation: Point::new([5.0, 0.0, 0.0]),
            rotation: Rotor::from_plane_angle(&Bivector::<3>::unit_plane(0, 1), FRAC_PI_2),
        };
        let p = Point::new([1.0, 0.0, 0.0]);
        let q = t.transform_point(&p);
        assert!((q.coord(0) - 5.0).abs() < 1e-10);
        assert!((q.coord(1) - 1.0).abs() < 1e-10);
    }
}
