// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use nalgebra::SVector;
use symtropy_math::{Bivector, Point, Shape, Sphere, Transform};

/// Opaque handle to a rigid body in the physics world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle(pub usize);

/// Stable network identifier for a body, used for cross-machine replay determinism.
///
/// Unlike `BodyHandle` (which depends on insertion order), `NetId` is assigned by
/// the application and survives serialization/deserialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NetId(pub u64);

/// Whether a body is affected by forces and collisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyType {
    /// Affected by forces and collisions.
    Dynamic,
    /// Not affected by forces, but participates in collisions (infinite mass).
    Static,
    /// Moved by user code, pushes dynamic bodies but isn't affected by them.
    Kinematic,
}

/// Invalid parameters supplied while constructing a rigid body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyBuildError {
    /// Dynamic bodies require a finite, strictly positive mass.
    InvalidMass,
    /// Sphere colliders require a finite, strictly positive radius.
    InvalidRadius,
    /// Dynamic bodies require finite, strictly positive principal inertia.
    InvalidInertia,
}

/// A rigid body in D-dimensional space.
pub struct RigidBody<const D: usize> {
    pub handle: BodyHandle,
    /// Stable network identifier for cross-machine replay. `None` for local-only bodies.
    pub net_id: Option<NetId>,
    pub body_type: BodyType,
    pub transform: Transform<D>,
    pub linear_velocity: SVector<f64, D>,
    pub angular_velocity: Bivector<D>,
    pub mass: f64,
    pub inv_mass: f64,
    /// Principal moments of inertia along each axis [I_x, I_y, ...].
    /// For spheres: all equal (isotropic). For asymmetric bodies: different per axis.
    /// Angular acceleration in bivector plane (i,j) uses avg(I_i, I_j).
    pub inertia: SVector<f64, D>,
    /// Inverse principal moments (1/I per axis). Zero for static bodies.
    pub inv_inertia: SVector<f64, D>,
    pub restitution: f64,
    pub friction: f64,
    /// Collider shape. Boxed because Shape<D> is a trait object.
    pub collider: Box<dyn Shape<D>>,
    /// Accumulated force this frame (cleared each step).
    pub force_accumulator: SVector<f64, D>,
    /// Accumulated torque this frame (as bivector, cleared each step).
    pub torque_accumulator: Bivector<D>,
    /// Linear damping factor (0.0 = no damping, 1.0 = full damping).
    pub linear_damping: f64,
    /// Angular damping factor.
    pub angular_damping: f64,
    /// Whether this body is sleeping (deactivated due to low velocity).
    pub sleeping: bool,
    /// Ticks since velocity dropped below sleep threshold.
    pub sleep_counter: u32,
    /// Collision group membership (bitmask). Body belongs to these groups.
    /// Default: `0xFFFF_FFFF` (all groups).
    pub collision_group: u32,
    /// Collision filter mask. Body collides with bodies whose `collision_group`
    /// overlaps this mask. Default: `0xFFFF_FFFF` (collides with everything).
    pub collision_mask: u32,
    /// Whether this body is a sensor/trigger (detects overlaps but doesn't resolve collisions).
    pub is_sensor: bool,
}

impl<const D: usize> RigidBody<D> {
    /// Create a dynamic body with a sphere collider.
    ///
    /// Panics if `radius` or `mass` are not finite and strictly positive. Use
    /// [`Self::try_dynamic_sphere`] when accepting untrusted input.
    pub fn dynamic_sphere(handle: BodyHandle, position: Point<D>, radius: f64, mass: f64) -> Self {
        Self::try_dynamic_sphere(handle, position, radius, mass)
            .expect("dynamic sphere requires finite positive radius and mass")
    }

    /// Create a dynamic sphere, returning an error instead of constructing a
    /// body with infinite or NaN inverse mass/inertia.
    pub fn try_dynamic_sphere(
        handle: BodyHandle,
        position: Point<D>,
        radius: f64,
        mass: f64,
    ) -> Result<Self, BodyBuildError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(BodyBuildError::InvalidRadius);
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(BodyBuildError::InvalidMass);
        }
        let inertia = 0.4 * mass * radius * radius; // 2/5 * m * r² (sphere, isotropic)

        Ok(Self {
            handle,
            net_id: None,
            body_type: BodyType::Dynamic,
            transform: Transform::from_translation(position),
            linear_velocity: SVector::zeros(),
            angular_velocity: Bivector::zero(),
            mass,
            inv_mass: 1.0 / mass,
            inertia: SVector::from_element(inertia),
            inv_inertia: SVector::from_element(1.0 / inertia),
            restitution: 0.5,
            friction: 0.3,
            collider: Box::new(Sphere::new(Point::origin(), radius)),
            force_accumulator: SVector::zeros(),
            torque_accumulator: Bivector::zero(),
            linear_damping: 0.01,
            angular_damping: 0.05,
            sleeping: false,
            sleep_counter: 0,
            collision_group: 0xFFFF_FFFF,
            collision_mask: 0xFFFF_FFFF,
            is_sensor: false,
        })
    }

    /// Create a static body (infinite mass, zero velocity).
    pub fn static_body(
        handle: BodyHandle,
        position: Point<D>,
        collider: Box<dyn Shape<D>>,
    ) -> Self {
        Self {
            handle,
            net_id: None,
            body_type: BodyType::Static,
            transform: Transform::from_translation(position),
            linear_velocity: SVector::zeros(),
            angular_velocity: Bivector::zero(),
            mass: f64::INFINITY,
            inv_mass: 0.0,
            inertia: SVector::from_element(f64::INFINITY),
            inv_inertia: SVector::zeros(),
            restitution: 0.5,
            friction: 0.5,
            collider,
            force_accumulator: SVector::zeros(),
            torque_accumulator: Bivector::zero(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            sleeping: false,
            sleep_counter: 0,
            collision_group: 0xFFFF_FFFF,
            collision_mask: 0xFFFF_FFFF,
            is_sensor: false,
        }
    }

    /// Create a dynamic body with a custom collider (isotropic inertia).
    ///
    /// Panics if `mass` or `inertia` are not finite and strictly positive. Use
    /// [`Self::try_dynamic`] when accepting untrusted input.
    pub fn dynamic(
        handle: BodyHandle,
        position: Point<D>,
        mass: f64,
        inertia: f64,
        collider: Box<dyn Shape<D>>,
    ) -> Self {
        Self::try_dynamic(handle, position, mass, inertia, collider)
            .expect("dynamic body requires finite positive mass and inertia")
    }

    /// Create a dynamic body with a custom collider, returning an error instead
    /// of constructing invalid inverse mass/inertia values.
    pub fn try_dynamic(
        handle: BodyHandle,
        position: Point<D>,
        mass: f64,
        inertia: f64,
        collider: Box<dyn Shape<D>>,
    ) -> Result<Self, BodyBuildError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(BodyBuildError::InvalidMass);
        }
        if !inertia.is_finite() || inertia <= 0.0 {
            return Err(BodyBuildError::InvalidInertia);
        }

        Ok(Self {
            handle,
            net_id: None,
            body_type: BodyType::Dynamic,
            transform: Transform::from_translation(position),
            linear_velocity: SVector::zeros(),
            angular_velocity: Bivector::zero(),
            mass,
            inv_mass: 1.0 / mass,
            inertia: SVector::from_element(inertia),
            inv_inertia: SVector::from_element(1.0 / inertia),
            restitution: 0.5,
            friction: 0.3,
            collider,
            force_accumulator: SVector::zeros(),
            torque_accumulator: Bivector::zero(),
            linear_damping: 0.01,
            angular_damping: 0.05,
            sleeping: false,
            sleep_counter: 0,
            collision_group: 0xFFFF_FFFF,
            collision_mask: 0xFFFF_FFFF,
            is_sensor: false,
        })
    }

    /// Apply a force at the center of mass.
    #[inline]
    pub fn apply_force(&mut self, force: SVector<f64, D>) {
        self.force_accumulator += force;
    }

    /// Apply a torque (as a bivector).
    #[inline]
    pub fn apply_torque(&mut self, torque: Bivector<D>) {
        self.torque_accumulator = self.torque_accumulator.add(&torque);
    }

    /// Clear accumulated forces and torques.
    #[inline]
    pub fn clear_accumulators(&mut self) {
        self.force_accumulator = SVector::zeros();
        self.torque_accumulator = Bivector::zero();
    }

    /// Position of the body's center of mass.
    #[inline]
    pub fn position(&self) -> &Point<D> {
        &self.transform.translation
    }

    /// Whether this body can be moved by forces.
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    /// Wake this body from sleep.
    #[inline]
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_counter = 0;
    }

    /// Check if this body should go to sleep (low velocity for many ticks).
    /// Returns true if the body just fell asleep.
    pub fn try_sleep(&mut self, threshold: f64, ticks_required: u32) -> bool {
        if !self.is_dynamic() || self.sleeping {
            return false;
        }
        let speed = self.linear_velocity.norm() + self.angular_velocity.norm();
        if speed < threshold {
            self.sleep_counter += 1;
            if self.sleep_counter >= ticks_required {
                self.sleeping = true;
                self.linear_velocity = SVector::zeros();
                self.angular_velocity = Bivector::zero();
                return true;
            }
        } else {
            self.sleep_counter = 0;
        }
        false
    }

    /// Kinetic energy: 0.5 * m * v² + 0.5 * I_avg * ω²
    /// Uses mean principal moment for angular KE (exact for isotropic bodies).
    pub fn kinetic_energy(&self) -> f64 {
        let linear = 0.5 * self.mass * self.linear_velocity.norm_squared();
        let i_avg = self.inertia.sum() / D as f64;
        let angular = 0.5 * i_avg * self.angular_velocity.norm_squared();
        linear + angular
    }

    /// Support point in world space (transforms collider support by body pose).
    pub fn world_support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        // Transform direction to local space
        let local_dir = self.transform.rotation.reverse().rotate_vector(direction);
        // Get local support
        let local_support = self.collider.support(&local_dir);
        // Transform back to world space
        self.transform.transform_point(&Point(local_support)).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_sphere_creation() {
        let body =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([1.0, 2.0, 3.0]), 1.0, 10.0);
        assert!(body.is_dynamic());
        assert!((body.mass - 10.0).abs() < 1e-12);
        assert!((body.inv_mass - 0.1).abs() < 1e-12);
    }

    #[test]
    fn dynamic_sphere_rejects_invalid_dimensions() {
        assert_eq!(
            RigidBody::<3>::try_dynamic_sphere(BodyHandle(0), Point::origin(), 0.0, 1.0).err(),
            Some(BodyBuildError::InvalidRadius)
        );
        assert_eq!(
            RigidBody::<3>::try_dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, f64::NAN).err(),
            Some(BodyBuildError::InvalidMass)
        );
    }

    #[test]
    fn dynamic_rejects_invalid_mass_and_inertia() {
        assert_eq!(
            RigidBody::<3>::try_dynamic(
                BodyHandle(0),
                Point::origin(),
                0.0,
                1.0,
                Box::new(Sphere::<3>::unit()),
            )
            .err(),
            Some(BodyBuildError::InvalidMass)
        );
        assert_eq!(
            RigidBody::<3>::try_dynamic(
                BodyHandle(0),
                Point::origin(),
                1.0,
                f64::INFINITY,
                Box::new(Sphere::<3>::unit()),
            )
            .err(),
            Some(BodyBuildError::InvalidInertia)
        );
    }

    #[test]
    fn static_body_infinite_mass() {
        let body = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::origin(),
            Box::new(Sphere::<3>::unit()),
        );
        assert!(!body.is_dynamic());
        assert_eq!(body.inv_mass, 0.0);
    }

    #[test]
    fn force_accumulation() {
        let mut body = RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0);
        body.apply_force(SVector::from([1.0, 0.0, 0.0]));
        body.apply_force(SVector::from([0.0, 2.0, 0.0]));
        assert!((body.force_accumulator[0] - 1.0).abs() < 1e-12);
        assert!((body.force_accumulator[1] - 2.0).abs() < 1e-12);
        body.clear_accumulators();
        assert!(body.force_accumulator.norm() < 1e-12);
    }

    #[test]
    fn kinetic_energy_at_rest_is_zero() {
        let body = RigidBody::<4>::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0);
        assert!(body.kinetic_energy() < 1e-12);
    }

    #[test]
    fn world_support_translated() {
        let body =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([10.0, 0.0, 0.0]), 1.0, 1.0);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = body.world_support(&dir);
        // Sphere of radius 1 centered at (10,0,0), support in +x = (11,0,0)
        assert!((sp[0] - 11.0).abs() < 1e-10);
    }
}
