// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Regression tests protecting load-bearing empirical constants in
// `examples/pendulum_swarm.md`. Each test guards one physics primitive
// the demo depends on. If an upstream change breaks one of these, the
// test failure points at which spike to re-run, not just "demo broken."
//
// Spike constants protected:
//   - DistanceConstraint produces a clean pendulum arc (Step 0 spike)
//   - linear_damping range 0.001 ↔ 0.5 gives visible motion contrast (Step 5 spike)
//   - MasterConsciousnessEquation response to uniform unit inputs (landmine 0)
//   - 100-body scene fits the 60 Hz budget (Success Criterion #1)
//
// Tolerances are chosen generously (within 2× the measured value) so
// cosmetic upstream changes don't trigger false alarms — only
// substantial-enough changes to break the demo's visual hook.

use nalgebra::SVector;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_math::{Point, Sphere};
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, PhysicsWorld, RigidBody};

/// Step 0 spike: DistanceConstraint between a dynamic bob and a static pivot
/// produces a clean pendulum arc. Arm length stays at rest_length (no drift)
/// and the bob sweeps through the full ±L range within the first period.
#[test]
fn distance_constraint_behaves_as_pendulum() {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
    let pivot = world.add_body(RigidBody::<2>::static_body(
        BodyHandle(0),
        Point::new([0.0, 0.0]),
        Box::new(Sphere::new(Point::origin(), 0.01)),
    ));
    let bob = world.add_sphere(Point::new([1.0, 0.0]), 0.05, 1.0);
    world.body_mut(bob).unwrap().linear_damping = 0.0;
    world.body_mut(bob).unwrap().collision_mask = 0;
    world.body_mut(pivot).unwrap().collision_mask = 0;
    world.add_constraint(Box::new(DistanceConstraint::<2> {
        body_a: pivot,
        body_b: bob,
        rest_length: 1.0,
        stiffness: 1.0,
    }));

    let dt = 1.0 / 240.0;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut max_dist_error = 0.0f64;

    // Simulate 4 seconds — roughly two large-amplitude periods (~2.37 s each).
    for _ in 0..(4 * 240) {
        world.step(dt);
        let p = world.body(bob).unwrap().position();
        let x = p[0];
        let y = p[1];
        if x < x_min {
            x_min = x;
        }
        if x > x_max {
            x_max = x;
        }
        let dist = (x * x + y * y).sqrt();
        max_dist_error = max_dist_error.max((dist - 1.0).abs());
    }

    // Constraint must hold: distance drift ≤ 1% of arm length.
    assert!(
        max_dist_error < 0.01,
        "DistanceConstraint drift too large: {} > 0.01. \
         Re-run pendulum_swarm.md Step 0 spike and update the doc.",
        max_dist_error
    );

    // Full ±L swing within 4 s — guards against the bob getting stuck
    // in a small oscillation (which would indicate energy loss too
    // fast to produce the visible hook).
    assert!(
        x_max > 0.9,
        "Bob didn't reach +L during swing: x_max={}. Demo visual hook compromised.",
        x_max
    );
    assert!(
        x_min < -0.9,
        "Bob didn't reach -L during swing: x_min={}. Demo visual hook compromised.",
        x_min
    );
}

/// Total mechanical energy (KE + PE) of a pendulum about its pivot.
/// PE referenced to the lowest point (bob hanging straight down).
fn total_energy(world: &PhysicsWorld<2>, bob: BodyHandle, pivot_y: f64) -> f64 {
    let body = world.body(bob).unwrap();
    let p = body.position();
    let dy = p[1] - pivot_y; // negative while bob is below pivot
    let height_above_lowest = 1.0 + dy; // arm length = 1.0
    let ke = 0.5 * body.mass * body.linear_velocity.norm_squared();
    let pe = body.mass * 9.81 * height_above_lowest;
    ke + pe
}

fn build_pendulum(world: &mut PhysicsWorld<2>, pivot_x: f64, damping: f64) -> BodyHandle {
    let pivot = world.add_body(RigidBody::<2>::static_body(
        BodyHandle(0),
        Point::new([pivot_x, 0.0]),
        Box::new(Sphere::new(Point::origin(), 0.01)),
    ));
    let bob = world.add_sphere(Point::new([pivot_x, -1.0]), 0.05, 1.0);
    {
        let b = world.body_mut(bob).unwrap();
        b.linear_damping = damping;
        b.angular_damping = 0.0;
        b.linear_velocity[0] = 3.0; // shock: horizontal kick at the bottom
        b.collision_mask = 0;
    }
    world.body_mut(pivot).unwrap().collision_mask = 0;
    world.add_constraint(Box::new(DistanceConstraint::<2> {
        body_a: pivot,
        body_b: bob,
        rest_length: 1.0,
        stiffness: 1.0,
    }));
    bob
}

/// Step 5 spike: LOW_DAMP=0.001 and HIGH_DAMP=0.5 produce visible motion
/// contrast. After 5 seconds of identical shocking, the low-damp bob
/// retains substantially more mechanical energy than the high-damp bob.
/// Measured values: 78.7% vs 13.0% at t=5s. Tolerances here are loose
/// (allow 50% low, 30% high as the pass band) because we only need to
/// guarantee the VISUAL distinction survives, not reproduce the exact
/// numbers.
#[test]
fn damping_range_produces_visible_contrast() {
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
    let bob_low = build_pendulum(&mut world, 0.0, 0.001);
    let bob_high = build_pendulum(&mut world, 3.0, 0.5);

    let initial_low = total_energy(&world, bob_low, 0.0);
    let initial_high = total_energy(&world, bob_high, 0.0);

    let dt = 1.0 / 240.0;
    for _ in 0..(5 * 240) {
        world.step(dt);
    }

    let final_low = total_energy(&world, bob_low, 0.0);
    let final_high = total_energy(&world, bob_high, 0.0);
    let low_retention = final_low / initial_low;
    let high_retention = final_high / initial_high;

    assert!(
        low_retention > 0.5,
        "LOW_DAMP pendulum lost too much energy: retention={:.3} < 0.5. \
         Measured 0.787 at writing. Re-run pendulum_swarm.md Step 5 spike.",
        low_retention
    );
    assert!(
        high_retention < 0.3,
        "HIGH_DAMP pendulum retained too much energy: retention={:.3} > 0.3. \
         Measured 0.130 at writing. Re-run pendulum_swarm.md Step 5 spike.",
        high_retention
    );
    // The ratio guard catches "both damping values suddenly behave
    // similarly" — the exact failure mode that silently breaks the
    // visual hook.
    assert!(
        low_retention / high_retention > 2.5,
        "Damping contrast collapsed: ratio={:.2} < 2.5. \
         low={:.3}, high={:.3}. Visual hook is at risk. \
         Measured 6.1× at writing.",
        low_retention / high_retention,
        low_retention,
        high_retention
    );
}

/// Landmine 0: the empirical PHI_NORMALIZE constant matches the current
/// `MasterConsciousnessEquation` response to sustained uniform unit
/// `ConsciousnessInputs`. Note: one-shot single-call measurement is
/// NOT steady-state — the equation carries `EntityConsciousness` memory
/// that warms up over several frames. The demo runs update_entity at
/// 60 Hz for many frames, so steady-state is the right reference.
///
/// If this drifts, the demo's `phi → damping` mapping loses dynamic
/// range and the visual hook silently breaks.
#[test]
fn phi_max_at_uniform_unit_inputs_matches_pendulum_swarm_constant() {
    // The doc records a settled measurement of 0.314. We allow a
    // ±33% band — tight enough to catch the equation being retuned
    // from "single digits" to "mostly saturated," loose enough that
    // a cosmetic reweighting doesn't false-alarm.
    const PHI_NORMALIZE_EXPECTED: f64 = 0.314;
    const TOLERANCE: f64 = 0.10;

    let mut field = ConsciousnessField::<2>::new();
    let h = BodyHandle(42);
    field.register(h, 100.0, 10.0);

    let inputs = ConsciousnessInputs {
        phi: 1.0,
        broadcast: 1.0,
        working_memory: 1.0,
        attention: 1.0,
        recurrence: 1.0,
        embodiment: 1.0,
        knowledge: 1.0,
        synchrony: 1.0,
    };
    // Warm up for 1 second of simulated 60 Hz updates before measuring.
    // This matches the demo's runtime context — it will never observe
    // a one-shot phi value.
    for _ in 0..60 {
        field.update_entity(h, &inputs, Point::origin());
    }
    let measured = field.phi(h);

    assert!(
        (measured - PHI_NORMALIZE_EXPECTED).abs() < TOLERANCE,
        "PHI_NORMALIZE in pendulum_swarm.md (landmine 0) is stale. \
         Expected ~{:.3}, measured {:.4} after 60-frame warmup. \
         Update the constant in the doc + the demo's \
         `phi_modulates_damping` system, and re-run the phi-mapping \
         spike to record the new full table.",
        PHI_NORMALIZE_EXPECTED,
        measured
    );

    // Negative case from the original spike: single-channel input
    // (only inputs.phi=1.0) SHOULD still produce ~zero phi output,
    // because the demo's "uniform coherence scalar" strategy relies
    // on multi-channel interaction being the dominant term. If a
    // single channel starts producing non-trivial phi, the design
    // premise weakens.
    //
    // Fresh field for independent measurement.
    let mut field2 = ConsciousnessField::<2>::new();
    field2.register(h, 100.0, 10.0);
    let only_phi = ConsciousnessInputs {
        phi: 1.0,
        broadcast: 0.0,
        working_memory: 0.0,
        attention: 0.0,
        recurrence: 0.0,
        embodiment: 0.0,
        knowledge: 0.0,
        synchrony: 0.0,
    };
    for _ in 0..60 {
        field2.update_entity(h, &only_phi, Point::origin());
    }
    let single_channel = field2.phi(h);
    assert!(
        single_channel < 0.1,
        "Single-channel phi input produced non-trivial output {:.4} \
         after warmup. The 'uniform coherence scalar' strategy in \
         pendulum_swarm.md assumes multi-channel interaction is \
         required for phi > 0. Re-evaluate the design.",
        single_channel
    );
}

/// Success Criterion #1: 100 pendulums + constraints step fast enough to
/// hold 60 Hz. We build the full 10×10 grid (100 bobs, 100 static pivots,
/// 100 DistanceConstraints, all with collision disabled per the demo
/// spec) and measure how long 60 physics ticks take. The 60 Hz budget
/// is 16.7 ms per tick; we allow 10× that (167 ms / 60 ticks = 2.8 ms
/// mean per tick on unloaded hardware) as a loose upper bound that
/// still catches catastrophic regression (e.g. O(N²) where it used
/// to be O(N log N)).
///
/// Release builds only — debug is uselessly slow for physics.
#[test]
fn scene_100_pendulums_fits_60hz_budget_release() {
    // Skip the perf guard in debug; it's not meaningful there.
    if cfg!(debug_assertions) {
        eprintln!("[perf] skipped in debug build (run with --release for meaningful numbers)");
        return;
    }

    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));

    // 10×10 grid, pivots 1 unit apart
    for row in 0..10 {
        for col in 0..10 {
            let pivot_x = col as f64 * 2.0;
            let pivot_y = row as f64 * 2.0;
            let pivot = world.add_body(RigidBody::<2>::static_body(
                BodyHandle(0),
                Point::new([pivot_x, pivot_y]),
                Box::new(Sphere::new(Point::origin(), 0.01)),
            ));
            let bob = world.add_sphere(Point::new([pivot_x + 1.0, pivot_y]), 0.05, 1.0);
            {
                let b = world.body_mut(bob).unwrap();
                b.linear_damping = 0.01;
                b.collision_mask = 0;
            }
            world.body_mut(pivot).unwrap().collision_mask = 0;
            world.add_constraint(Box::new(DistanceConstraint::<2> {
                body_a: pivot,
                body_b: bob,
                rest_length: 1.0,
                stiffness: 1.0,
            }));
        }
    }

    let dt = 1.0 / 60.0;
    // Warmup: let broadphase trees build without counting first-tick cost.
    for _ in 0..10 {
        world.step(dt);
    }

    let n_ticks = 60usize;
    let start = std::time::Instant::now();
    for _ in 0..n_ticks {
        world.step(dt);
    }
    let elapsed = start.elapsed();
    let mean_ms = elapsed.as_secs_f64() * 1000.0 / n_ticks as f64;

    // Hard upper bound — 10× the 60 Hz budget. Any regression that
    // blows past this is a real problem. Log mean so CI output shows
    // the current number for tracking.
    eprintln!(
        "[perf] 100-pendulum step: {:.3} ms mean over {} ticks ({:.1}% of 16.67 ms budget)",
        mean_ms,
        n_ticks,
        100.0 * mean_ms / 16.67
    );
    assert!(
        mean_ms < 16.67,
        "100-pendulum step mean {:.3} ms exceeds 60 Hz budget 16.67 ms. \
         The demo can't hit Success Criterion #1 on this hardware. \
         Either the physics engine regressed (broadphase, constraint \
         solver) or the demo needs to drop pendulum count. Profile \
         with `cargo flamegraph --example _spike_100_pendulums` \
         after writing that spike.",
        mean_ms
    );
}
