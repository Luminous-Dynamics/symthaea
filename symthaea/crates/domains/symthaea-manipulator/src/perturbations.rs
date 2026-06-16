// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Perturbation types for manipulator robustness testing.
//!
//! Each perturbation modifies simulator state at a specified step,
//! testing the arm's ability to maintain safety under disturbances.

use serde::{Deserialize, Serialize};

/// A single perturbation event applied during a simulation episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManipulatorPerturbation {
    /// External force push on the end-effector (Newtons, 3D).
    ExternalForce { force: [f64; 3], at_step: usize },
    /// Sudden joint friction increase (multiplier on damping).
    JointFriction {
        joint: usize,
        factor: f64,
        at_step: usize,
    },
    /// Payload mass change (added mass at end-effector, kg).
    PayloadMass { mass_kg: f64, at_step: usize },
    /// Single joint failure (locked at current angle).
    JointFailure { joint: usize, at_step: usize },
    /// Collision force from obstacle contact.
    CollisionForce {
        force: [f64; 3],
        duration_steps: usize,
        at_step: usize,
    },
}

/// Schedule of perturbations for an episode.
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    pub perturbations: Vec<ManipulatorPerturbation>,
}

impl PerturbationSchedule {
    /// Standard robustness test: push, friction change, payload addition.
    pub fn standard_robustness() -> Self {
        Self {
            perturbations: vec![
                ManipulatorPerturbation::ExternalForce {
                    force: [5.0, 0.0, 0.0],
                    at_step: 100,
                },
                ManipulatorPerturbation::JointFriction {
                    joint: 2,
                    factor: 3.0,
                    at_step: 300,
                },
                ManipulatorPerturbation::PayloadMass {
                    mass_kg: 2.0,
                    at_step: 500,
                },
            ],
        }
    }

    /// Severe test: joint failure + collision.
    pub fn severe_test() -> Self {
        Self {
            perturbations: vec![
                ManipulatorPerturbation::JointFailure {
                    joint: 3,
                    at_step: 200,
                },
                ManipulatorPerturbation::CollisionForce {
                    force: [0.0, 10.0, -5.0],
                    duration_steps: 50,
                    at_step: 400,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_robustness_schedule() {
        let schedule = PerturbationSchedule::standard_robustness();
        assert_eq!(schedule.perturbations.len(), 3);
    }

    #[test]
    fn test_severe_schedule() {
        let schedule = PerturbationSchedule::severe_test();
        assert_eq!(schedule.perturbations.len(), 2);
    }

    #[test]
    fn test_serialization() {
        let schedule = PerturbationSchedule::standard_robustness();
        let json = serde_json::to_string(&schedule.perturbations).unwrap();
        let _: Vec<ManipulatorPerturbation> = serde_json::from_str(&json).unwrap();
    }
}
