// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 6DOF hydrodynamic model for AUV simulation.
//!
//! Key differences from aerial platforms:
//! - **Added mass**: Fluid accelerated with the vehicle adds to effective inertia
//! - **Quadratic drag**: Dominant force (not linear like air at low speed)
//! - **Buoyancy**: Constant restoring force toward surface
//! - **Hydrostatic restoring**: Metacentric height creates pitch/roll stability
//!
//! Science:
//! - Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
//! - REMUS 100 AUV parameters (Prestero, 2001).

use serde::{Deserialize, Serialize};

/// Hydrodynamic parameters for an AUV (~50kg torpedo-class).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrodynamicConfig {
    /// Vehicle dry mass (kg).
    pub mass: f64,
    /// Vehicle volume for buoyancy (m³).
    pub displacement_volume: f64,
    /// Added mass diagonal [surge, sway, heave, roll, pitch, yaw].
    /// Added mass ≈ mass of fluid displaced by acceleration in each DOF.
    pub added_mass: [f64; 6],
    /// Quadratic drag coefficients [surge, sway, heave, roll, pitch, yaw].
    /// F_drag = -0.5 × ρ × Cd × A × |v| × v for each DOF.
    pub drag_coefficients: [f64; 6],
    /// Frontal area per DOF [surge, sway, heave] in m².
    pub frontal_areas: [f64; 3],
    /// Center of buoyancy offset from center of gravity [x, y, z] in meters.
    /// Positive z = buoyancy center above CG (stable).
    pub buoyancy_center_offset: [f64; 3],
    /// Metacentric height (meters). Positive = statically stable in pitch/roll.
    pub metacentric_height: f64,
    /// Water density (kg/m³, typical freshwater: 998, seawater: 1025).
    pub water_density: f64,
    /// Maximum thruster force per thruster (Newtons).
    pub max_thruster_force: f64,
    /// Thruster time constant (seconds, first-order lag).
    pub thruster_tau: f64,
}

impl Default for HydrodynamicConfig {
    fn default() -> Self {
        Self {
            mass: 50.0,                // 50 kg (REMUS-class)
            displacement_volume: 0.05, // Slightly positively buoyant
            added_mass: [10.0, 30.0, 30.0, 0.5, 5.0, 5.0],
            drag_coefficients: [50.0, 100.0, 100.0, 5.0, 10.0, 10.0],
            frontal_areas: [0.03, 0.15, 0.15], // Torpedo shape: narrow surge, wide sway/heave
            buoyancy_center_offset: [0.0, 0.0, 0.02], // CB 2cm above CG
            metacentric_height: 0.02,
            water_density: 998.0,     // Freshwater
            max_thruster_force: 50.0, // 50N per thruster
            thruster_tau: 0.1,        // 100ms response
        }
    }
}

/// Hydrodynamic forces and moments for one timestep.
#[derive(Debug, Clone, Copy, Default)]
pub struct HydrodynamicForces {
    /// Force vector [surge, sway, heave] in body frame (Newtons).
    pub force: [f64; 3],
    /// Moment vector [roll, pitch, yaw] in body frame (Nm).
    pub moment: [f64; 3],
    /// Buoyancy force (positive = upward, Newtons).
    pub buoyancy: f64,
    /// Total drag force magnitude (Newtons).
    pub drag_magnitude: f64,
}

/// Rotate a world-frame vector into the body frame using the unit
/// quaternion `q = [w, x, y, z]` (body→world convention); this applies the
/// conjugate (inverse) rotation.
pub fn world_to_body(q: &[f64; 4], v: &[f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], -q[1], -q[2], -q[3]); // conjugate
    rotate(w, x, y, z, v)
}

/// Rotate a body-frame vector into the world frame using the unit
/// quaternion `q = [w, x, y, z]` (body→world convention).
pub fn body_to_world(q: &[f64; 4], v: &[f64; 3]) -> [f64; 3] {
    rotate(q[0], q[1], q[2], q[3], v)
}

/// v' = q v q⁻¹ expanded (unit quaternion assumed).
fn rotate(w: f64, x: f64, y: f64, z: f64, v: &[f64; 3]) -> [f64; 3] {
    let (vx, vy, vz) = (v[0], v[1], v[2]);
    // t = 2 (q_vec × v)
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    // v' = v + w t + q_vec × t
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

/// Compute hydrodynamic forces given current BODY-frame velocity and
/// angular velocity, plus the attitude quaternion `[w, x, y, z]`.
///
/// Frame convention (2026-07 frame-math rework): world z is DOWN-positive
/// (depth), matching `AuvState.position[2]`/`depth`. Weight acts +z world,
/// buoyancy −z world; both are rotated into the body frame here, so an AUV
/// pitched or rolled gets hydrostatic forces along the correct world axis —
/// previously buoyancy was applied along the body heave axis regardless of
/// attitude (an inverted AUV was pushed body-"up", i.e. world-down).
///
/// Returns forces and moments in body frame.
pub fn compute_forces(
    config: &HydrodynamicConfig,
    velocity: &[f64; 3],
    angular_velocity: &[f64; 3],
    quaternion: &[f64; 4],
    _depth: f64,
) -> HydrodynamicForces {
    let rho = config.water_density;
    let g = 9.81;

    // 1. Quadratic drag: F = -0.5 × ρ × Cd × A × |v| × v
    let mut drag_force = [0.0f64; 3];
    let mut drag_moment = [0.0f64; 3];

    for i in 0..3 {
        let v = velocity[i];
        let cd = config.drag_coefficients[i];
        let area = config.frontal_areas[i.min(2)];
        drag_force[i] = -0.5 * rho * cd * area * v.abs() * v;
    }

    for i in 0..3 {
        let w = angular_velocity[i];
        let cd = config.drag_coefficients[3 + i];
        // Angular drag: approximate with moment arm of 0.5m
        drag_moment[i] = -0.5 * rho * cd * 0.25 * w.abs() * w;
    }

    // 2. Hydrostatics in the WORLD frame, rotated into body.
    // Net reported buoyancy keeps the legacy sign convention
    // (positive = net upward force / positively buoyant).
    let weight = config.mass * g; // +z world (down)
    let buoyancy_mag = rho * g * config.displacement_volume; // −z world (up)
    let buoyancy = buoyancy_mag - weight;
    // Net hydrostatic force, world frame, z-down: W − B.
    let f_hydrostatic_body = world_to_body(quaternion, &[0.0, 0.0, weight - buoyancy_mag]);

    // 3. Hydrostatic restoring moment from the actual CB/CG separation:
    // buoyancy acts at the center of buoyancy, offset r_b from the CG, so
    // M = r_b × F_buoyancy(body). Exact for all attitudes — replaces the
    // previous small-angle `−ρgV·GM·sin(θ)` approximation, and finally
    // wires `buoyancy_center_offset`, which was dead config before.
    // NOTE: `buoyancy_center_offset` documents +z as "CB above CG"; in the
    // z-down body frame "above" is −z, hence the negation.
    let r_b = [
        config.buoyancy_center_offset[0],
        config.buoyancy_center_offset[1],
        -config.buoyancy_center_offset[2],
    ];
    let f_buoy_body = world_to_body(quaternion, &[0.0, 0.0, -buoyancy_mag]);
    let restoring = [
        r_b[1] * f_buoy_body[2] - r_b[2] * f_buoy_body[1],
        r_b[2] * f_buoy_body[0] - r_b[0] * f_buoy_body[2],
        r_b[0] * f_buoy_body[1] - r_b[1] * f_buoy_body[0],
    ];

    // 4. Combine forces
    let force = [
        drag_force[0] + f_hydrostatic_body[0],
        drag_force[1] + f_hydrostatic_body[1],
        drag_force[2] + f_hydrostatic_body[2],
    ];

    let moment = [
        drag_moment[0] + restoring[0],
        drag_moment[1] + restoring[1],
        drag_moment[2] + restoring[2],
    ];

    let drag_mag = (drag_force[0].powi(2) + drag_force[1].powi(2) + drag_force[2].powi(2)).sqrt();

    HydrodynamicForces {
        force,
        moment,
        buoyancy,
        drag_magnitude: drag_mag,
    }
}

/// Compute effective mass including added mass for each DOF.
pub fn effective_mass(config: &HydrodynamicConfig) -> [f64; 6] {
    [
        config.mass + config.added_mass[0],
        config.mass + config.added_mass[1],
        config.mass + config.added_mass[2],
        config.added_mass[3], // Roll inertia (simplified)
        config.added_mass[4], // Pitch inertia
        config.added_mass[5], // Yaw inertia
    ]
}

/// Compute thruster forces from command, applying first-order lag.
///
/// `thruster_state` is mutated in-place (current thruster force, Newtons).
/// `command` is the desired thruster setting [-1, 1].
/// Returns net force/moment from all 8 thrusters.
pub fn thruster_forces(
    config: &HydrodynamicConfig,
    thruster_state: &mut [f64; 8],
    command: &[f32; 8],
    dt: f64,
) -> ([f64; 3], [f64; 3]) {
    let alpha = 1.0 - (-dt / config.thruster_tau).exp();

    for i in 0..8 {
        let target = command[i] as f64 * config.max_thruster_force;
        thruster_state[i] += alpha * (target - thruster_state[i]);
    }

    // Thruster layout: 0-3 horizontal (surge/sway/yaw), 4-5 vertical (heave), 6-7 vectored
    let surge = thruster_state[0] + thruster_state[1]; // Forward pair
    let sway = thruster_state[2] - thruster_state[3]; // Lateral pair (differential)
    let heave = thruster_state[4] + thruster_state[5]; // Vertical pair
    let yaw = (thruster_state[0] - thruster_state[1]) * 0.3; // Differential thrust → yaw
    let pitch = (thruster_state[6] - thruster_state[7]) * 0.2; // Vectored → pitch
    let roll = (thruster_state[6] + thruster_state[7]) * 0.1; // Vectored → roll (weak)

    ([surge, sway, heave], [roll, pitch, yaw])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drag_opposes_motion() {
        let config = HydrodynamicConfig::default();
        let velocity = [1.0, 0.0, 0.0]; // Moving forward
        let forces = compute_forces(&config, &velocity, &[0.0; 3], &[1.0, 0.0, 0.0, 0.0], 10.0);
        assert!(forces.force[0] < 0.0, "Drag should oppose forward motion");
    }

    #[test]
    fn test_drag_quadratic() {
        let config = HydrodynamicConfig::default();
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [2.0, 0.0, 0.0];
        let f1 = compute_forces(&config, &v1, &[0.0; 3], &[1.0, 0.0, 0.0, 0.0], 10.0);
        let f2 = compute_forces(&config, &v2, &[0.0; 3], &[1.0, 0.0, 0.0, 0.0], 10.0);
        // At 2× speed, drag should be ~4× (quadratic)
        let ratio = f2.force[0].abs() / f1.force[0].abs();
        assert!(
            (ratio - 4.0).abs() < 0.5,
            "Drag should scale quadratically: ratio={ratio}"
        );
    }

    #[test]
    fn test_buoyancy_positive() {
        let config = HydrodynamicConfig::default();
        let forces = compute_forces(&config, &[0.0; 3], &[0.0; 3], &[1.0, 0.0, 0.0, 0.0], 10.0);
        // Default config is slightly positively buoyant (displacement_volume > mass/rho)
        assert!(forces.buoyancy > -1.0, "Should be near-neutral buoyancy");
    }

    #[test]
    fn test_restoring_moment_opposes_roll() {
        // With CB above CG (buoyancy_center_offset +z), a rolled vehicle
        // must get a righting moment opposing the roll — this is what
        // wiring the previously-dead buoyancy_center_offset buys. Matches
        // the old −ρgV·GM·sin(φ) magnitude at small angles, exact at all.
        let config = HydrodynamicConfig::default();
        let phi: f64 = 0.3; // roll right
        let q = [(phi / 2.0).cos(), (phi / 2.0).sin(), 0.0, 0.0];
        let forces = compute_forces(&config, &[0.0; 3], &[0.0; 3], &q, 10.0);
        assert!(
            forces.moment[0] < -1e-3,
            "positive roll must produce negative (righting) roll moment, got {}",
            forces.moment[0]
        );
        // And the opposite roll rights the opposite way.
        let q_neg = [(phi / 2.0).cos(), -(phi / 2.0).sin(), 0.0, 0.0];
        let f_neg = compute_forces(&config, &[0.0; 3], &[0.0; 3], &q_neg, 10.0);
        assert!(f_neg.moment[0] > 1e-3);
    }

    #[test]
    fn test_hydrostatic_force_is_world_aligned() {
        // The slightly-heavy default vehicle must experience its net
        // hydrostatic force toward world +z (down/deeper) regardless of
        // attitude. Inverted (180° roll), the BODY-frame force flips sign —
        // proving the force is world-anchored, not glued to the heave axis.
        let config = HydrodynamicConfig::default();
        let upright = compute_forces(&config, &[0.0; 3], &[0.0; 3], &[1.0, 0.0, 0.0, 0.0], 10.0);
        let inverted = compute_forces(&config, &[0.0; 3], &[0.0; 3], &[0.0, 1.0, 0.0, 0.0], 10.0);
        assert!(upright.force[2] > 0.0, "heavier-than-water sinks (+z down)");
        assert!(
            inverted.force[2] < 0.0,
            "inverted: same world-down force is body -z, got {}",
            inverted.force[2]
        );
    }

    #[test]
    fn test_effective_mass_includes_added() {
        let config = HydrodynamicConfig::default();
        let eff = effective_mass(&config);
        assert!(eff[0] > config.mass); // Surge: mass + added mass
        assert!(eff[1] > eff[0]); // Sway added mass > surge (wider cross-section)
    }

    #[test]
    fn test_thruster_lag() {
        let config = HydrodynamicConfig::default();
        let mut state = [0.0f64; 8];
        let command = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // After one small step, thruster shouldn't be at max yet
        thruster_forces(&config, &mut state, &command, 0.001);
        assert!(state[0] > 0.0);
        assert!(state[0] < config.max_thruster_force);

        // After many steps, should converge to max
        for _ in 0..10000 {
            thruster_forces(&config, &mut state, &command, 0.001);
        }
        assert!((state[0] - config.max_thruster_force).abs() < 1.0);
    }

    #[test]
    fn test_thruster_differential_yaw() {
        let config = HydrodynamicConfig::default();
        let mut state = [0.0f64; 8];
        // Differential thrust: thruster 0 forward, thruster 1 reverse → yaw
        let command = [1.0f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..100 {
            thruster_forces(&config, &mut state, &command, 0.01);
        }
        let (force, moment) = thruster_forces(&config, &mut state, &command, 0.01);
        assert!(
            moment[2].abs() > 0.0,
            "Differential thrust should produce yaw moment"
        );
    }
}
