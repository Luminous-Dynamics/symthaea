// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semi-implicit Euler integrator for ND rigid body dynamics.
//!
//! Angular velocity is a bivector (element of so(n)), integrated via the
//! rotation matrix exponential. For small timesteps, the Rodrigues formula
//! approximation is sufficient and avoids matrix exponential computation.

use crate::body::RigidBody;
use nalgebra::SVector;
use symtropy_math::{Bivector, Point, Rotor};

/// Integrate a single rigid body forward by dt seconds.
///
/// Semi-implicit Euler: update velocity first, then position.
/// This is unconditionally stable for Hamiltonian systems (symplectic).
pub fn integrate<const D: usize>(body: &mut RigidBody<D>, gravity: &SVector<f64, D>, dt: f64) {
    if !body.is_dynamic() {
        body.clear_accumulators();
        return;
    }

    // --- NaN/Inf guards on inputs ---
    // Sanitize force/torque accumulators: replace non-finite values with zero
    // to prevent NaN propagation from bad external forces.
    if !body.force_accumulator.iter().all(|v| v.is_finite()) {
        body.force_accumulator = SVector::zeros();
    }
    if !body.torque_accumulator.is_finite() {
        body.torque_accumulator = Bivector::zero();
    }

    // --- Linear dynamics ---

    // Acceleration = F/m + gravity
    let accel = body.force_accumulator * body.inv_mass + gravity;

    // Semi-implicit: velocity first
    body.linear_velocity += accel * dt;

    // LTC-based damping: v(t+dt) = v(t) × exp(-dt/τ) where τ = 1/damping.
    // Frame-rate independent, never reverses velocity (exp > 0), exact for any dt.
    if body.linear_damping > 1e-15 {
        let tau = 1.0 / body.linear_damping;
        body.linear_velocity *= (-dt / tau).exp();
    }

    // Clamp velocity to prevent explosion
    let speed = body.linear_velocity.norm();
    if speed > 1000.0 {
        body.linear_velocity *= 1000.0 / speed;
    }

    // Update position
    let displacement = body.linear_velocity * dt;
    body.transform.translation = Point(body.transform.translation.0 + displacement);

    // --- Angular dynamics ---

    // Angular acceleration = torque / inertia (scalar approximation)
    // Per-axis inertia: use mean of principal moments for each bivector plane.
    // For spheres (isotropic), this is exact. For asymmetric bodies, it's the
    // average inertia of the two axes spanning each rotation plane.
    //
    // TODO(isotropic-inertia): this scalar mean makes ALL rotation isotropic,
    // which is only physically correct for spheres. A non-spherical body
    // (box, capsule, compound shape) will rotate as if it had a sphere's mass
    // distribution — e.g. a long thin rod will resist tumbling about its long
    // axis exactly as much as about a short axis, which is wrong. Fixing this
    // properly requires per-bivector-plane inertia (a full D×D inertia tensor
    // or at least per-plane scalars derived from the collider's actual mass
    // distribution) plumbed through `RigidBody`'s `inertia`/`inv_inertia`
    // fields (currently `SVector<f64, D>`, one scalar per *axis*, then
    // averaged pairwise here — not one scalar per *bivector plane*, and not
    // tracked as the body rotates, i.e. no world-space inertia tensor
    // update). That's a larger data-layout change to `body.rs` than is safe
    // to make alongside the contact-solver fixes in this change; deferred.
    let inv_i_avg = body.inv_inertia.sum() / D as f64;
    let ang_accel = body.torque_accumulator.scale(inv_i_avg);

    // Update angular velocity
    body.angular_velocity = body.angular_velocity.add(&ang_accel.scale(dt));

    // LTC angular damping (same closed-form)
    if body.angular_damping > 1e-15 {
        let ang_tau = 1.0 / body.angular_damping;
        body.angular_velocity = body.angular_velocity.scale((-dt / ang_tau).exp());
    }

    // Clamp angular velocity
    let ang_speed = body.angular_velocity.norm();
    if ang_speed > 100.0 {
        body.angular_velocity = body.angular_velocity.scale(100.0 / ang_speed);
    }

    // Integrate rotation: small angle approximation via Rodrigues
    if ang_speed > 1e-12 {
        let delta_angle = ang_speed * dt;
        let delta_rotation = Rotor::from_plane_angle(&body.angular_velocity, delta_angle);
        body.transform.rotation = delta_rotation.compose(&body.transform.rotation);
    }

    // --- Post-integration NaN/Inf assertions ---
    // If any state became non-finite despite clamping, zero the offending values
    // rather than propagating corruption through the simulation.
    if !body.linear_velocity.iter().all(|v| v.is_finite()) {
        debug_assert!(
            false,
            "NaN/Inf detected in linear_velocity after integration"
        );
        body.linear_velocity = SVector::zeros();
    }
    if !body.angular_velocity.is_finite() {
        debug_assert!(
            false,
            "NaN/Inf detected in angular_velocity after integration"
        );
        body.angular_velocity = Bivector::zero();
    }
    if !body.transform.translation.0.iter().all(|v| v.is_finite()) {
        debug_assert!(false, "NaN/Inf detected in position after integration");
        // Don't zero position — that would teleport. Just freeze velocity.
        body.linear_velocity = SVector::zeros();
    }

    // Clear accumulators for next frame
    body.clear_accumulators();
}

/// Apply a position correction to separate overlapping bodies.
pub fn separate<const D: usize>(body: &mut RigidBody<D>, correction: &SVector<f64, D>) {
    if body.is_dynamic() {
        body.transform.translation = Point(body.transform.translation.0 + correction);
    }
}

/// Apply an impulse to a body at its center of mass.
pub fn apply_impulse<const D: usize>(body: &mut RigidBody<D>, impulse: &SVector<f64, D>) {
    if body.is_dynamic() {
        body.linear_velocity += impulse * body.inv_mass;
    }
}

/// Apply an angular impulse (as a bivector) to a body.
pub fn apply_angular_impulse<const D: usize>(body: &mut RigidBody<D>, impulse: &Bivector<D>) {
    if body.is_dynamic() {
        // TODO(isotropic-inertia): same scalar-mean-inertia limitation as in
        // `integrate` above — see that TODO for why this is only exact for
        // spheres and what a real fix would require.
        let inv_i_avg = body.inv_inertia.sum() / D as f64;
        body.angular_velocity = body.angular_velocity.add(&impulse.scale(inv_i_avg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyHandle;

    #[test]
    fn free_fall_3d() {
        let mut body = RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0);
        let gravity = SVector::from([0.0, -9.81, 0.0]);
        let dt = 0.01;

        // Simulate 1 second of free fall (100 steps)
        for _ in 0..100 {
            integrate(&mut body, &gravity, dt);
        }

        // After 1s: y ≈ -0.5 * 9.81 * 1² ≈ -4.905 (with damping, less)
        let y = body.transform.translation.coord(1);
        assert!(y < -3.0, "y = {y}, expected < -3.0");
        assert!(y > -6.0, "y = {y}, expected > -6.0 (damping bounds)");
    }

    #[test]
    fn stationary_without_forces() {
        let mut body =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([5.0, 5.0, 5.0]), 1.0, 1.0);
        let zero_gravity = SVector::zeros();
        let dt = 0.01;

        for _ in 0..100 {
            integrate(&mut body, &zero_gravity, dt);
        }

        // Should barely move (damping causes slight drift toward zero vel)
        assert!((body.transform.translation.coord(0) - 5.0).abs() < 0.1);
    }

    #[test]
    fn applied_force_accelerates() {
        let mut body = RigidBody::<3>::dynamic_sphere(
            BodyHandle(0),
            Point::origin(),
            1.0,
            2.0, // 2 kg
        );
        let zero_gravity = SVector::zeros();
        let dt = 0.01;

        // Apply 10N in x direction for 1 second
        for _ in 0..100 {
            body.apply_force(SVector::from([10.0, 0.0, 0.0]));
            integrate(&mut body, &zero_gravity, dt);
        }

        // v ≈ F/m * t = 10/2 * 1 = 5 m/s (minus damping)
        let vx = body.linear_velocity[0];
        assert!(vx > 3.0, "vx = {vx}, expected > 3.0");
        assert!(vx < 6.0, "vx = {vx}, expected < 6.0");
    }

    #[test]
    fn angular_velocity_rotates_body() {
        let mut body = RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0);
        // Set angular velocity in xy plane
        body.angular_velocity = Bivector::<3>::unit_plane(0, 1).scale(1.0);

        let zero_gravity = SVector::zeros();
        let dt = 0.01;

        for _ in 0..100 {
            integrate(&mut body, &zero_gravity, dt);
        }

        // Body should have rotated — rotation matrix should not be identity
        let mat = body.transform.rotation.to_matrix();
        let trace = mat.trace();
        // Identity has trace = 3, rotated will have trace < 3
        assert!(trace < 2.99, "trace = {trace}, body didn't rotate");
    }

    #[test]
    fn impulse_changes_velocity() {
        let mut body = RigidBody::<4>::dynamic_sphere(
            BodyHandle(0),
            Point::origin(),
            1.0,
            5.0, // 5 kg
        );
        apply_impulse(&mut body, &SVector::from([50.0, 0.0, 0.0, 0.0]));
        // Δv = impulse / mass = 50/5 = 10
        assert!((body.linear_velocity[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn energy_conservation_no_damping() {
        let mut body =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 100.0, 0.0]), 1.0, 1.0);
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;

        let gravity = SVector::from([0.0, -9.81, 0.0]);
        let dt = 0.001; // Small timestep for accuracy

        let initial_energy =
            body.kinetic_energy() + body.mass * 9.81 * body.transform.translation.coord(1);

        for _ in 0..1000 {
            integrate(&mut body, &gravity, dt);
        }

        let final_energy =
            body.kinetic_energy() + body.mass * 9.81 * body.transform.translation.coord(1);
        let drift = ((final_energy - initial_energy) / initial_energy).abs();
        assert!(drift < 0.01, "energy drift {drift} > 1%");
    }

    #[test]
    fn static_body_not_affected() {
        use symtropy_math::Sphere;
        let mut body = RigidBody::<3>::static_body(
            BodyHandle(0),
            Point::new([5.0, 5.0, 5.0]),
            Box::new(Sphere::unit()),
        );
        body.apply_force(SVector::from([1000.0, 0.0, 0.0]));
        let gravity = SVector::from([0.0, -100.0, 0.0]);
        integrate(&mut body, &gravity, 1.0);
        assert!((body.transform.translation.coord(0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nan_force_is_sanitized() {
        let mut body =
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([1.0, 2.0, 3.0]), 1.0, 1.0);
        // Apply a NaN force — should be zeroed, not propagated
        body.apply_force(SVector::from([f64::NAN, 0.0, 0.0]));
        let gravity = SVector::zeros();
        integrate(&mut body, &gravity, 0.01);

        // Position and velocity must remain finite
        assert!(
            body.linear_velocity.iter().all(|v| v.is_finite()),
            "NaN force propagated to velocity"
        );
        assert!(
            body.transform.translation.0.iter().all(|v| v.is_finite()),
            "NaN force propagated to position"
        );
    }

    #[test]
    fn inf_force_is_sanitized() {
        let mut body = RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::origin(), 1.0, 1.0);
        body.apply_force(SVector::from([f64::INFINITY, 0.0, 0.0]));
        let gravity = SVector::zeros();
        integrate(&mut body, &gravity, 0.01);

        assert!(
            body.linear_velocity.iter().all(|v| v.is_finite()),
            "Inf force propagated to velocity"
        );
    }
}
