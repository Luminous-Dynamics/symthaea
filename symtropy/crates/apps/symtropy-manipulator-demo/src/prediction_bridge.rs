// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prediction error bridge: human proximity → external force on EE.
//!
//! The force is physically modeled (repulsive field from human proximity).
//! This force modifies `ManipulatorState.end_effector_force`, which the
//! HDC encoder picks up at channel weight 2.5 (highest), causing a
//! prediction error spike → free energy rise → Phi drop.
//!
//! NO values are injected into Phi directly. The entire chain is emergent.

use crate::human_obstacle::HumanObstacle;
use symthaea_manipulator::ManipulatorState;

/// Compute repulsive forces on the end-effector due to human proximity.
///
/// Returns [Fx, Fy, Fz] in Newtons. Direction pushes EE away from human.
/// Force is zero when human is outside the interaction radius.
///
/// The force profile is quadratic: `F = penetration² × base_magnitude`.
/// This models the combination of:
/// - Air pressure changes from fast-moving human (small)
/// - Physical contact forces if human reaches arm (large)
/// - Safety repulsion field (virtual, for demo purposes represents sensor uncertainty)
pub fn compute_human_interaction_forces(
    human: &HumanObstacle,
    arm_state: &ManipulatorState,
) -> [f64; 3] {
    let ee = arm_state.end_effector_position;

    let dx = ee[0] - human.position[0];
    let dy = ee[1] - human.position[1];
    let dz = ee[2] - human.position[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    // Interaction radius: human reach + 15cm safety margin
    let interaction_radius = human.reach_radius + 0.15;

    if dist > interaction_radius || dist < 1e-6 {
        return [0.0; 3];
    }

    // Penetration depth [0, 1]: 0 at boundary, 1 at human center
    let penetration = (interaction_radius - dist) / interaction_radius;

    // Quadratic force profile: soft onset, strong near contact
    // Base magnitude: 40N at full penetration (below Yellow 20N cap at half penetration)
    let base_magnitude = 40.0;
    let force_magnitude = penetration * penetration * base_magnitude;

    // Direction: push EE away from human center
    let inv_dist = 1.0 / dist;
    [
        force_magnitude * dx * inv_dist,
        force_magnitude * dy * inv_dist,
        force_magnitude * dz * inv_dist,
    ]
}

/// Compute danger level [0, 1] for consciousness input.
///
/// This is NOT the Phi value — it feeds into `ConsciousnessInputs` as `danger_level`.
/// The actual Phi is computed by `MasterConsciousnessEquation`.
pub fn danger_level_from_human(human: &HumanObstacle, ee_pos: &[f64; 3]) -> f64 {
    let dist = human.distance_to(ee_pos);
    let interaction_radius = human.reach_radius + 0.15;

    if dist > interaction_radius * 2.0 {
        0.0
    } else {
        let normalized = 1.0 - (dist / (interaction_radius * 2.0)).min(1.0);
        (normalized * normalized).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state_at(x: f64, y: f64, z: f64) -> ManipulatorState {
        let mut s = ManipulatorState::home();
        s.end_effector_position = [x, y, z];
        s
    }

    #[test]
    fn no_force_when_far() {
        let human = HumanObstacle::new(42);
        // Human at [0, -1.2, 0.5], EE at [0.5, 0.5, 0.5] — well outside reach
        let state = test_state_at(0.5, 0.5, 0.5);
        let force = compute_human_interaction_forces(&human, &state);
        assert_eq!(force, [0.0; 3]);
    }

    #[test]
    fn force_pushes_away() {
        let mut human = HumanObstacle::new(42);
        human.position = [0.3, 0.0, 0.5]; // Near workspace center
        let state = test_state_at(0.35, 0.0, 0.5); // EE very close to human

        let force = compute_human_interaction_forces(&human, &state);
        // Force should push EE in +x direction (away from human at x=0.3)
        assert!(force[0] > 0.0, "Force should push EE away: {:?}", force);
    }

    #[test]
    fn force_increases_with_proximity() {
        let mut human = HumanObstacle::new(42);
        human.position = [0.3, 0.0, 0.5];

        let state_far = test_state_at(0.8, 0.0, 0.5);
        let state_near = test_state_at(0.35, 0.0, 0.5);

        let f_far = compute_human_interaction_forces(&human, &state_far);
        let f_near = compute_human_interaction_forces(&human, &state_near);

        let mag_far = (f_far[0].powi(2) + f_far[1].powi(2) + f_far[2].powi(2)).sqrt();
        let mag_near = (f_near[0].powi(2) + f_near[1].powi(2) + f_near[2].powi(2)).sqrt();

        assert!(
            mag_near > mag_far,
            "Closer = stronger force: near={mag_near}, far={mag_far}"
        );
    }

    #[test]
    fn danger_level_zero_when_far() {
        let human = HumanObstacle::new(42);
        let dl = danger_level_from_human(&human, &[0.5, 0.5, 0.5]);
        assert!(dl < 0.01);
    }

    #[test]
    fn danger_level_high_when_close() {
        let mut human = HumanObstacle::new(42);
        human.position = [0.3, 0.0, 0.5];
        let dl = danger_level_from_human(&human, &[0.35, 0.0, 0.5]);
        assert!(dl > 0.5, "Danger should be high when close: {dl}");
    }
}
