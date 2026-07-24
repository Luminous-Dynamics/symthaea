// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Computational check of the weak equivalence principle's mathematical content.
//!
//! The weak equivalence principle (WEP) says inertial mass and gravitational
//! mass are the same parameter. Its precise, checkable consequence is that a
//! test particle's Newtonian gravitational acceleration doesn't contain its
//! own mass at all: `F = GMm/r²`, `a = F/m = GM/r²` — the `m` cancels exactly
//! *because* the same symbol appears in both `F=ma` and Newton's gravity law.
//! This module simulates that cancellation directly and contrasts it against
//! a non-gravitational force, where no such cancellation occurs and identical
//! initial conditions at different masses genuinely diverge.
//!
//! ## What this does and does not demonstrate
//!
//! This confirms the code's own model of gravity respects mass-independence
//! by construction — a sanity/regression check, not an independent
//! measurement of nature. The actual experimental bounds on WEP violation
//! come from real torsion-balance and satellite experiments, cited below as
//! [`EOT_WASH_ETA_BOUND`] and [`MICROSCOPE_ETA_BOUND`] — those numbers are
//! many orders of magnitude tighter than anything a numerical simulation's
//! floating-point/integration error could speak to, and this module makes no
//! claim to reproduce them.

/// Newtonian gravitational constant, m³ kg⁻¹ s⁻².
const G: f64 = 6.674_30e-11;

fn vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn vec_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn vec_norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Newtonian gravitational acceleration on a test particle at `position` due
/// to a point mass `big_m_kg` at the origin. Note there is no `mass_kg`
/// parameter here at all — that omission IS the weak equivalence principle:
/// the test particle's own mass has no lever on its gravitational
/// acceleration.
pub fn newtonian_gravity_acceleration(position: [f64; 3], big_m_kg: f64) -> [f64; 3] {
    let r = vec_norm(position);
    if r < 1e-6 {
        return [0.0; 3];
    }
    let mag = -G * big_m_kg / (r * r);
    vec_scale(position, mag / r)
}

fn rk4_step<F: Fn([f64; 3]) -> [f64; 3]>(
    pos: [f64; 3],
    vel: [f64; 3],
    accel_of: &F,
    dt: f64,
) -> ([f64; 3], [f64; 3]) {
    let a1 = accel_of(pos);
    let v1 = vel;

    let p2 = vec_add(pos, vec_scale(v1, 0.5 * dt));
    let u2 = vec_add(vel, vec_scale(a1, 0.5 * dt));
    let a2 = accel_of(p2);

    let p3 = vec_add(pos, vec_scale(u2, 0.5 * dt));
    let u3 = vec_add(vel, vec_scale(a2, 0.5 * dt));
    let a3 = accel_of(p3);

    let p4 = vec_add(pos, vec_scale(u3, dt));
    let u4 = vec_add(vel, vec_scale(a3, dt));
    let a4 = accel_of(p4);

    let new_pos = vec_add(
        pos,
        vec_scale(
            vec_add(
                v1,
                vec_add(vec_scale(u2, 2.0), vec_add(vec_scale(u3, 2.0), u4)),
            ),
            dt / 6.0,
        ),
    );
    let new_vel = vec_add(
        vel,
        vec_scale(
            vec_add(
                a1,
                vec_add(vec_scale(a2, 2.0), vec_add(vec_scale(a3, 2.0), a4)),
            ),
            dt / 6.0,
        ),
    );
    (new_pos, new_vel)
}

/// Simulate a free-falling test particle's trajectory under pure Newtonian
/// gravity from a central mass `big_m_kg`. `_mass_kg` is accepted only so
/// callers can vary it across runs and confirm the trajectory doesn't change
/// — it plays no role in [`newtonian_gravity_acceleration`].
pub fn simulate_gravity_trajectory(
    initial_pos: [f64; 3],
    initial_vel: [f64; 3],
    big_m_kg: f64,
    _mass_kg: f64,
    steps: usize,
    dt: f64,
) -> Vec<[f64; 3]> {
    let mut pos = initial_pos;
    let mut vel = initial_vel;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(pos);
    let accel_of = |p: [f64; 3]| newtonian_gravity_acceleration(p, big_m_kg);
    for _ in 0..steps {
        let (p, v) = rk4_step(pos, vel, &accel_of, dt);
        pos = p;
        vel = v;
        out.push(pos);
    }
    out
}

/// Contrast case: a constant external non-gravitational force (e.g. thrust,
/// or the Coulomb force on a charge in a uniform field) genuinely depends on
/// mass through `a = F/m` — no cancellation occurs, so trajectories at
/// different masses under the same force must diverge. This is the control
/// that proves the gravity case's mass-independence isn't a trivial artifact
/// of the simulator (e.g. an integrator that just ignores the mass argument
/// everywhere).
pub fn simulate_external_force_trajectory(
    initial_pos: [f64; 3],
    initial_vel: [f64; 3],
    force_n: [f64; 3],
    mass_kg: f64,
    steps: usize,
    dt: f64,
) -> Vec<[f64; 3]> {
    let mut pos = initial_pos;
    let mut vel = initial_vel;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(pos);
    let accel_of = |_p: [f64; 3]| vec_scale(force_n, 1.0 / mass_kg);
    for _ in 0..steps {
        let (p, v) = rk4_step(pos, vel, &accel_of, dt);
        pos = p;
        vel = v;
        out.push(pos);
    }
    out
}

/// Maximum pointwise Euclidean divergence between two equal-length trajectories.
pub fn trajectory_divergence(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| vec_norm(vec_sub(*p, *q)))
        .fold(0.0, f64::max)
}

/// Eötvös-parameter bound, ground-based torsion balance (Be–Ti test masses,
/// Earth as the attractor). Schlamminger, Choi, Wagner, Gundlach & Adelberger,
/// *Phys. Rev. Lett.* **100**, 041101 (2008): η = (0.3 ± 1.8) × 10⁻¹³.
/// This constant is the 1σ magnitude of that bound.
pub const EOT_WASH_ETA_BOUND: f64 = 1.8e-13;

/// Eötvös-parameter bound, MICROSCOPE satellite (Ti–Pt test masses), the most
/// precise WEP test to date. Touboul et al., *Phys. Rev. Lett.* **129**,
/// 121102 (2022): η(Ti,Pt) = (−1.5 ± 2.3 stat ± 1.5 syst) × 10⁻¹⁵. This
/// constant is the combined 1σ magnitude of that bound (stat and syst errors
/// added in quadrature: sqrt(2.3² + 1.5²) ≈ 2.746).
pub const MICROSCOPE_ETA_BOUND: f64 = 2.746e-15;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_trajectory_is_mass_independent() {
        let pos = [7_000_000.0, 0.0, 0.0]; // ~LEO altitude, meters
        let vel = [0.0, 7_500.0, 0.0]; // ~orbital speed, m/s
        let big_m = 5.972e24; // Earth mass, kg

        let light = simulate_gravity_trajectory(pos, vel, big_m, 1.0, 2000, 1.0);
        let heavy = simulate_gravity_trajectory(pos, vel, big_m, 1.0e6, 2000, 1.0);

        let divergence = trajectory_divergence(&light, &heavy);
        // This is a numerical-integration-noise-floor check, NOT a physics
        // bound — RK4 with identical inputs should reproduce bit-for-bit
        // determinism, so any divergence here would indicate `mass_kg` is
        // accidentally leaking into the dynamics somewhere.
        assert!(
            divergence < 1e-6,
            "gravity trajectory diverged by {} m across a 1e6x mass change — \
             mass must not appear in the dynamics",
            divergence
        );
    }

    #[test]
    fn external_force_trajectory_depends_on_mass() {
        let pos = [0.0, 0.0, 0.0];
        let vel = [0.0, 0.0, 0.0];
        let force = [1.0, 0.0, 0.0]; // 1 Newton, constant

        let light = simulate_external_force_trajectory(pos, vel, force, 1.0, 2000, 1.0);
        let heavy = simulate_external_force_trajectory(pos, vel, force, 1.0e6, 2000, 1.0);

        let divergence = trajectory_divergence(&light, &heavy);
        // Contrast case: a real, substantial divergence is expected and
        // required here — it's what proves the gravity case's cancellation
        // is a genuine feature of that force law, not a simulator bug that
        // ignores mass everywhere.
        assert!(
            divergence > 1.0,
            "external-force trajectories should diverge substantially across \
             a 1e6x mass change, got only {} m",
            divergence
        );
    }

    #[test]
    fn newtonian_acceleration_matches_known_earth_surface_gravity() {
        // At Earth's surface (r = 6.371e6 m), |a| should be ~9.8 m/s^2.
        let earth_mass = 5.972e24;
        let earth_radius = 6.371e6;
        let a = newtonian_gravity_acceleration([earth_radius, 0.0, 0.0], earth_mass);
        let g = vec_norm(a);
        assert!((g - 9.8).abs() < 0.1, "g = {} m/s^2, expected ~9.8", g);
    }

    #[test]
    fn microscope_bound_is_tighter_than_eot_wash() {
        // Sanity check on the two cited constants themselves: MICROSCOPE
        // (satellite, 2022) is the stated improvement over Eöt-Wash
        // (ground-based, 2008).
        assert!(MICROSCOPE_ETA_BOUND < EOT_WASH_ETA_BOUND);
    }
}
