//! Geometric-structure checks for fixed-step integrators on a canonical oscillator.
//!
//! These tests exercise the production integrator implementation through the
//! isolated harness. They test a numerical map property, not physical-model
//! validity or general multidimensional symplecticity.

use ode_verify_001_harness::dynamical_system::{
    DynamicalSystem, Integrator, Rk4Integrator, VerletIntegrator,
};

struct UnitHarmonicOscillator;

impl DynamicalSystem for UnitHarmonicOscillator {
    fn state_dim(&self) -> usize {
        2
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        // Canonical state [q, v], with q' = v and v' = -q.
        vec![state[1], -state[0]]
    }

    fn name(&self) -> &str {
        "unit harmonic oscillator"
    }
}

fn step(integrator: &dyn Integrator, state: [f64; 2], t: f64, dt: f64) -> [f64; 2] {
    let next = integrator.step(&UnitHarmonicOscillator, &state, t, dt);
    [next[0], next[1]]
}

/// Compute the determinant of the one-step linear map using basis vectors.
///
/// For this linear test system, applying the map to the origin and two basis
/// states recovers the exact affine map up to floating-point rounding; no
/// finite-difference epsilon is needed.
fn one_step_area_determinant(integrator: &dyn Integrator, dt: f64) -> f64 {
    let origin = step(integrator, [0.0, 0.0], 0.0, dt);
    let q_basis = step(integrator, [1.0, 0.0], 0.0, dt);
    let v_basis = step(integrator, [0.0, 1.0], 0.0, dt);

    let dq_dq = q_basis[0] - origin[0];
    let dv_dq = q_basis[1] - origin[1];
    let dq_dv = v_basis[0] - origin[0];
    let dv_dv = v_basis[1] - origin[1];

    dq_dq * dv_dv - dq_dv * dv_dq
}

fn round_trip_error(integrator: &dyn Integrator, initial: [f64; 2], dt: f64) -> f64 {
    let forward = step(integrator, initial, 0.0, dt);
    let backward = step(integrator, forward, dt, -dt);
    (backward[0] - initial[0])
        .abs()
        .max((backward[1] - initial[1]).abs())
}

#[test]
fn verlet_preserves_canonical_phase_space_area_in_one_degree_of_freedom() {
    // In one canonical degree of freedom, the linear map is symplectic iff
    // its 2x2 matrix has determinant one. This is an exact-map property, not
    // an assertion that the numerical trajectory has zero energy error.
    let verlet_det = one_step_area_determinant(&VerletIntegrator, 1.0);
    assert!(
        (verlet_det - 1.0).abs() <= 1e-12,
        "Verlet area determinant {verlet_det:.16e} should be one"
    );
}

#[test]
fn rk4_is_a_negative_control_for_exact_phase_space_area_preservation() {
    // Classical RK4 is high-order but not symplectic. For this oscillator at
    // dt=1, its linear map has determinant 569/576, about 0.98785.
    let rk4_det = one_step_area_determinant(&Rk4Integrator, 1.0);
    assert!(
        (rk4_det - 1.0).abs() > 1e-3,
        "RK4 negative control unexpectedly preserved area: det={rk4_det:.16e}"
    );
}

#[test]
fn verlet_step_is_time_reversible_under_a_signed_step_round_trip() {
    // This deliberately calls the single-step map with +dt then -dt. The
    // checked trajectory API rejects reversed time bounds by design; this
    // test isolates the fixed-step integrator's algebraic reversibility.
    let initial = [0.7, -0.2];
    let error = round_trip_error(&VerletIntegrator, initial, 0.4);
    assert!(
        error <= 1e-12,
        "Verlet round-trip error {error:.16e} exceeded 1e-12"
    );
}

#[test]
fn rk4_round_trip_is_a_negative_control_for_time_reversibility() {
    let error = round_trip_error(&Rk4Integrator, [1.0, 0.0], 1.0);
    assert!(
        error > 1e-3,
        "RK4 negative control unexpectedly reversed exactly: error={error:.16e}"
    );
}
