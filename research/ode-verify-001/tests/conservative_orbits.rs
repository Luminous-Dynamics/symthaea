//! Long-horizon invariant checks for a non-circular two-body orbit.
//!
//! These tests are numerical verification of the stated softened Newtonian
//! model, not experimental validation or a structural/mission safety claim.

use std::f64::consts::PI;

use ode_verify_001_harness::dynamical_system::{
    try_simulate, NBodyGravity, SimulationResult, VerletIntegrator,
};

const MASS: f64 = 1.0;
const GRAVITATIONAL_CONSTANT: f64 = 1.0;
const INITIAL_TANGENTIAL_SPEED: f64 = 0.42;
const ORBIT_COUNT: f64 = 50.0;

#[derive(Debug)]
struct InvariantAudit {
    steps: usize,
    max_relative_energy_drift: f64,
    max_scaled_momentum_drift: f64,
    max_relative_angular_momentum_drift: f64,
}

fn initial_state() -> [f64; 8] {
    [
        -1.0, 0.0, // body 1 position
        1.0, 0.0, // body 2 position
        0.0, -INITIAL_TANGENTIAL_SPEED, // body 1 velocity
        0.0, INITIAL_TANGENTIAL_SPEED, // body 2 velocity
    ]
}

fn reference_kepler_period() -> f64 {
    // The point-mass Kepler period is used only to set a repeatable observation
    // horizon. NBodyGravity has a tiny softening length, so this is not asserted
    // as an exact period of the implemented softened potential.
    let relative_speed = 2.0 * INITIAL_TANGENTIAL_SPEED;
    let gravitational_parameter = GRAVITATIONAL_CONSTANT * (MASS + MASS);
    let specific_orbital_energy =
        0.5 * relative_speed * relative_speed - gravitational_parameter / 2.0;
    let semi_major_axis = -gravitational_parameter / (2.0 * specific_orbital_energy);
    2.0 * PI * (semi_major_axis.powi(3) / gravitational_parameter).sqrt()
}

fn angular_momentum_z(state: &[f64]) -> f64 {
    // State order: [all positions, all velocities] for two bodies in 2D.
    state[0] * state[5] - state[1] * state[4]
        + state[2] * state[7]
        - state[3] * state[6]
}

fn max_relative_energy_drift(system: &NBodyGravity, result: &SimulationResult, e0: f64) -> f64 {
    result
        .states
        .iter()
        .map(|state| (system.energy(state) - e0).abs() / e0.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

fn audit_orbit(dt: f64) -> InvariantAudit {
    let system = NBodyGravity::new(vec![MASS, MASS], GRAVITATIONAL_CONSTANT, 2);
    let initial = initial_state();
    let end_time = ORBIT_COUNT * reference_kepler_period();
    let result = try_simulate(
        &system,
        &VerletIntegrator,
        &initial,
        0.0,
        end_time,
        dt,
    )
    .expect("the finite, bounded orbit request should be accepted");

    let e0 = system.energy(&initial);
    let p0 = system.momentum(&initial);
    let l0 = angular_momentum_z(&initial);
    let momentum_scale = 2.0 * MASS * INITIAL_TANGENTIAL_SPEED;

    let max_scaled_momentum_drift = result
        .states
        .iter()
        .map(|state| {
            let momentum = system.momentum(state);
            ((momentum[0] - p0[0]).powi(2) + (momentum[1] - p0[1]).powi(2)).sqrt()
                / momentum_scale
        })
        .fold(0.0, f64::max);

    let max_relative_angular_momentum_drift = result
        .states
        .iter()
        .map(|state| (angular_momentum_z(state) - l0).abs() / l0.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max);

    assert_eq!(result.times.len(), result.states.len());
    assert!((result.times[0]).abs() < f64::EPSILON);
    assert!(
        (result.times[result.times.len() - 1] - end_time).abs() < 1e-10,
        "simulation must reach the declared horizon; actual={}, expected={end_time}",
        result.times[result.times.len() - 1]
    );

    InvariantAudit {
        steps: result.steps,
        max_relative_energy_drift: max_relative_energy_drift(&system, &result, e0),
        max_scaled_momentum_drift,
        max_relative_angular_momentum_drift,
    }
}

#[test]
fn verlet_keeps_two_body_invariants_bounded_over_fifty_eccentric_orbits() {
    // The sub-circular initial speed yields an eccentric orbit. This avoids
    // relying only on the unusually forgiving perfectly circular fixture.
    let audit = audit_orbit(0.01);

    assert!(
        audit.max_relative_energy_drift < 5e-5,
        "relative energy drift {:.3e} exceeded 5e-5 after {} steps",
        audit.max_relative_energy_drift,
        audit.steps
    );
    assert!(
        audit.max_scaled_momentum_drift < 1e-10,
        "scaled momentum drift {:.3e} exceeded 1e-10",
        audit.max_scaled_momentum_drift
    );
    assert!(
        audit.max_relative_angular_momentum_drift < 1e-10,
        "relative angular-momentum drift {:.3e} exceeded 1e-10",
        audit.max_relative_angular_momentum_drift
    );
}

#[test]
fn verlet_energy_error_improves_under_timestep_refinement() {
    let coarse = audit_orbit(0.02);
    let fine = audit_orbit(0.01);

    assert!(
        coarse.max_relative_energy_drift < 2e-4,
        "coarse relative energy drift {:.3e} exceeded 2e-4",
        coarse.max_relative_energy_drift
    );
    assert!(
        fine.max_relative_energy_drift < coarse.max_relative_energy_drift * 0.4,
        "halving dt should materially reduce energy error: coarse={:.3e}, fine={:.3e}",
        coarse.max_relative_energy_drift,
        fine.max_relative_energy_drift
    );
    assert!(
        fine.max_scaled_momentum_drift < 1e-10
            && fine.max_relative_angular_momentum_drift < 1e-10,
        "fine-step invariants out of tolerance: momentum={:.3e}, angular momentum={:.3e}",
        fine.max_scaled_momentum_drift,
        fine.max_relative_angular_momentum_drift
    );
}
