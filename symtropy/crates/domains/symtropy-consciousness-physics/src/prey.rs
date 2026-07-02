// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prey system for complementary task experiments.
//!
//! Prey entities yield energy ONLY when two agents with complementary
//! capabilities are co-located. This enforces genuine coordination —
//! by-product mutualism cannot solve it because different agent types
//! must arrive at the same place at the same time.

use crate::harmony_field::{HarmonyField, NUM_HARMONIES};
use nalgebra::SVector;

/// A prey entity that requires complementary agents to capture.
#[derive(Debug, Clone)]
pub struct Prey<const D: usize> {
    /// Spatial position of the prey.
    pub position: SVector<f64, D>,
    /// Energy yielded on successful capture.
    pub energy_value: f64,
    /// Interaction radius (agents must be within this distance).
    pub radius: f64,
    /// Ticks remaining until respawn (0 = active).
    pub respawn_countdown: usize,
    /// Ticks between captures before respawn.
    pub respawn_delay: usize,
}

/// Result of a hunt check.
pub struct HuntResult {
    /// Whether the hunt succeeded.
    pub success: bool,
    /// Indices of the driver and ambusher (if successful).
    pub driver_idx: Option<usize>,
    pub ambusher_idx: Option<usize>,
}

impl<const D: usize> Prey<D> {
    /// Create a new active prey.
    pub fn new(
        position: SVector<f64, D>,
        energy_value: f64,
        radius: f64,
        respawn_delay: usize,
    ) -> Self {
        Self {
            position,
            energy_value,
            radius,
            respawn_countdown: 0,
            respawn_delay,
        }
    }

    /// Whether the prey is available for capture.
    pub fn is_active(&self) -> bool {
        self.respawn_countdown == 0
    }

    /// Tick the respawn timer.
    pub fn tick(&mut self) {
        if self.respawn_countdown > 0 {
            self.respawn_countdown -= 1;
        }
    }

    /// Capture the prey (starts respawn timer).
    pub fn capture(&mut self) {
        self.respawn_countdown = self.respawn_delay;
    }
}

/// Check whether a complementary hunt can succeed at a prey location.
///
/// Requires at least one DRIVER (harmony[driver_channel] > threshold)
/// and one AMBUSHER (harmony[ambusher_channel] > threshold) within
/// the prey's radius. The two agents must be COMPLEMENTARY (low resonance),
/// ensuring genuinely different types cooperate.
///
/// Returns the indices of the successful pair, or None.
pub fn check_complementary_hunt(
    agents_at_prey: &[(usize, [f64; NUM_HARMONIES])], // (agent_index, harmony_profile)
    driver_channel: usize,
    ambusher_channel: usize,
    capability_threshold: f64,
    max_resonance: f64, // must be BELOW this to count as complementary
) -> Option<(usize, usize)> {
    // Find all drivers and ambushers
    let drivers: Vec<usize> = agents_at_prey
        .iter()
        .filter(|(_, h)| h[driver_channel] > capability_threshold)
        .map(|(idx, _)| *idx)
        .collect();
    let ambushers: Vec<usize> = agents_at_prey
        .iter()
        .filter(|(_, h)| h[ambusher_channel] > capability_threshold)
        .map(|(idx, _)| *idx)
        .collect();

    // Find a complementary pair (low resonance = different types)
    for &d in &drivers {
        let d_harm = &agents_at_prey.iter().find(|(i, _)| *i == d).unwrap().1;
        for &a in &ambushers {
            if a == d {
                continue;
            } // can't be both
            let a_harm = &agents_at_prey.iter().find(|(i, _)| *i == a).unwrap().1;
            let res = HarmonyField::<2>::resonance(d_harm, a_harm);
            if res < max_resonance {
                return Some((d, a)); // complementary pair found
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prey_respawn_cycle() {
        let mut prey = Prey::<2>::new(SVector::from([0.0, 0.0]), 50.0, 30.0, 100);
        assert!(prey.is_active());
        prey.capture();
        assert!(!prey.is_active());
        for _ in 0..99 {
            prey.tick();
        }
        assert!(!prey.is_active());
        prey.tick();
        assert!(prey.is_active());
    }

    #[test]
    fn complementary_hunt_requires_both_types() {
        // Only drivers present — should fail
        let agents = vec![(0, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])];
        assert!(check_complementary_hunt(&agents, 0, 2, 0.6, 0.3).is_none());

        // Only ambushers present — should fail
        let agents = vec![(0, [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])];
        assert!(check_complementary_hunt(&agents, 0, 2, 0.6, 0.3).is_none());

        // Both types present with low resonance — should succeed
        let agents = vec![
            (0, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]), // driver
            (1, [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]), // ambusher
        ];
        let result = check_complementary_hunt(&agents, 0, 2, 0.6, 0.3);
        assert!(result.is_some());
        let (d, a) = result.unwrap();
        assert_eq!(d, 0);
        assert_eq!(a, 1);
    }

    #[test]
    fn similar_agents_cannot_hunt() {
        // Two agents with similar profiles (high resonance) — should fail
        let agents = vec![
            (0, [0.8, 0.3, 0.7, 0.2, 0.3, 0.2, 0.2, 0.6, 0.0]),
            (1, [0.7, 0.4, 0.8, 0.1, 0.4, 0.3, 0.3, 0.5, 0.0]),
        ];
        // Both have high harmony[0] AND harmony[2], but resonance is high
        let result = check_complementary_hunt(&agents, 0, 2, 0.6, 0.3);
        assert!(
            result.is_none(),
            "Similar agents should not form complementary pair"
        );
    }
}
