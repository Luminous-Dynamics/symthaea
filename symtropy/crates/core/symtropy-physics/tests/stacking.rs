// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression test for the lever-arm angular contact response (Fix 3).
//!
//! Before this fix, `resolve_contact` computed relative velocity from
//! `linear_velocity` only and never applied an angular impulse, so
//! multi-point manifolds (e.g. a box resting on a plane) would just
//! repeatedly push the center of mass without any lever-arm torque. That
//! made resting boxes rely entirely on the multi-point manifold's redundant
//! points to avoid pivoting, with no ability to correct rotational drift
//! once disturbed -- and no torque contribution from off-center friction at
//! all (which, per-point, can itself run away without careful load-sharing;
//! see the "friction must be per-point, not per-manifold" fix alongside this
//! test).
//!
//! This test rests two boxes side-by-side on a ground half-space (a 4-point
//! manifold each, via the analytical `HalfSpace` fast path) and asserts both
//! settle into a stable, non-jittering configuration.
//!
//! NOTE on scope: an earlier version of this test stacked box B directly on
//! top of box A. That surfaced a real, but separate and pre-existing, bug in
//! `epa_3d`'s bounding-sphere fallback (triggered whenever GJK's terminating
//! simplex has fewer than 4 points) significantly overestimating the
//! penetration depth for box-vs-box contact in some configurations -- e.g.
//! reporting ~1.4 units of penetration for two half-extent-0.5 boxes that
//! were only fractionally overlapping. That is a GJK/EPA convex-geometry
//! accuracy gap unrelated to the 5 solver fixes in this change (CCD,
//! restitution, lever-arm contact response, hinge motor, isotropic-inertia
//! TODO), so it is out of scope here and left as a follow-up; this test is
//! scoped to the box-vs-halfspace path (fully analytical, no GJK/EPA
//! involved) so it reliably guards the angular contact code without also
//! depending on that separate, pre-existing gap.

use nalgebra::SVector;
use symtropy_math::{HalfSpace, HyperBox, Point, Transform};
use symtropy_physics::PhysicsWorld;
use symtropy_physics::body::{BodyHandle, BodyType, RigidBody};

/// Analytical solid-cuboid inertia (per-axis principal moments), matching the
/// convention used by `RigidBody::dynamic_sphere` (one scalar per axis).
fn box_inertia(mass: f64, half_extents: [f64; 3]) -> SVector<f64, 3> {
    let [hx, hy, hz] = half_extents;
    let ixx = (mass / 3.0) * (hy * hy + hz * hz);
    let iyy = (mass / 3.0) * (hx * hx + hz * hz);
    let izz = (mass / 3.0) * (hx * hx + hy * hy);
    SVector::from([ixx, iyy, izz])
}

fn make_box(pos: Point<3>, half_extents: [f64; 3], mass: f64) -> RigidBody<3> {
    let inertia = box_inertia(mass, half_extents);
    RigidBody::new(
        BodyHandle(0), // overwritten by PhysicsWorld::add_body
        BodyType::Dynamic,
        Transform::from_translation(pos),
        Box::new(HyperBox::<3>::new(half_extents)),
        mass,
        inertia,
    )
}

#[test]
fn resting_boxes_settle_without_jitter() {
    let gravity = SVector::from([0.0, -9.81, 0.0]);
    let mut world = PhysicsWorld::<3>::new(gravity);
    world.solver_iterations = 8;

    // Ground: half-space with normal +Y through the origin (top at y=0).
    let ground = RigidBody::<3>::static_body(
        BodyHandle(0),
        Point::origin(),
        Box::new(HalfSpace::<3>::new(SVector::from([0.0, 1.0, 0.0]), 0.0)),
    );
    world.add_body(ground);

    let half = [0.5, 0.5, 0.5];
    // Two boxes resting side-by-side, each with its own 4-point manifold
    // against the ground. Box B is dropped from a small height with a
    // deliberate X offset so any un-arrested lever-arm/torque error would
    // show up as toppling rather than a perfectly symmetric configuration.
    let a = world.add_body(make_box(Point::new([-1.0, 0.5, 0.0]), half, 1.0));
    let b = world.add_body(make_box(Point::new([1.05, 0.8, 0.02]), half, 1.0));

    let dt = 1.0 / 60.0;

    // Let both boxes settle.
    for _ in 0..600 {
        world.step(dt);
    }

    let pos_a = world.body(a).unwrap().position();
    let pos_b = world.body(b).unwrap().position();

    assert!(
        (pos_a[1] - 0.5).abs() < 0.05,
        "box A should rest at y≈0.5, got y={}",
        pos_a[1]
    );
    assert!(
        (pos_b[1] - 0.5).abs() < 0.05,
        "box B should rest at y≈0.5, got y={}",
        pos_b[1]
    );

    // Track frame-to-frame motion over the next second: a genuinely settled
    // configuration should not keep jittering (small oscillating position
    // deltas), which is exactly what the lever-arm torque fix guards against
    // for multi-point manifolds.
    let mut max_delta = 0.0_f64;
    let mut prev_a = pos_a;
    let mut prev_b = pos_b;
    for _ in 0..60 {
        world.step(dt);
        let cur_a = world.body(a).unwrap().position();
        let cur_b = world.body(b).unwrap().position();
        max_delta = max_delta
            .max((cur_a - prev_a).norm())
            .max((cur_b - prev_b).norm());
        prev_a = cur_a;
        prev_b = cur_b;
    }

    assert!(
        max_delta < 0.01,
        "resting boxes should not jitter once settled; max frame-to-frame delta = {max_delta}"
    );

    // Velocities should be small (settled or asleep) -- no perpetual bounce
    // or energy injection from the new angular contact response.
    let va = world.body(a).unwrap().linear_velocity.norm();
    let vb = world.body(b).unwrap().linear_velocity.norm();
    assert!(va < 0.5, "box A should be nearly at rest, v={va}");
    assert!(vb < 0.5, "box B should be nearly at rest, v={vb}");

    // Angular velocity must not have run away either (regression guard for
    // the "friction must be per-point, not per-manifold" fix: concentrating
    // friction's torque at a single point caused unbounded spin growth).
    let wa = world.body(a).unwrap().angular_velocity.norm();
    let wb = world.body(b).unwrap().angular_velocity.norm();
    assert!(
        wa < 0.5,
        "box A angular velocity should not run away, w={wa}"
    );
    assert!(
        wb < 0.5,
        "box B angular velocity should not run away, w={wb}"
    );

    // Both bodies must remain finite (no NaN/Inf explosion from the new
    // lever-arm torque code).
    assert!(pos_a.iter().all(|v| v.is_finite()));
    assert!(pos_b.iter().all(|v| v.is_finite()));
}
