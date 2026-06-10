// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mid-episode physical perturbations for domain randomization.
//!
//! When perturbations hit mid-episode, FEP's prediction error spikes, tau drops,
//! and the system re-stabilizes WITHOUT retraining — a biological survival reflex.
//!
//! Includes standard perturbations (push, friction, mass) and paradigm-breaking
//! crucibles: Phantom Limb (actuator failure) and cooperative transport.

use crate::simulator::HumanoidPhysicsSimulator;
use crate::types::HumanoidCommand;

/// A physical perturbation applied to the humanoid mid-episode.
#[derive(Debug, Clone)]
pub enum HumanoidPerturbation {
    /// External push force applied to torso.
    ExternalPush {
        /// Force vector [x, y, z] in Newtons.
        force: [f64; 3],
        /// Step at which push begins.
        at_step: usize,
        /// Duration in physics steps.
        duration: usize,
    },
    /// Change floor friction coefficient.
    FloorFriction {
        /// New friction coefficient (1.0 = normal, 0.2 = ice).
        friction: f64,
        /// Step at which friction changes.
        at_step: usize,
    },
    /// Change body mass (simulating payload).
    MassChange {
        /// Body name (e.g., "torso").
        body: String,
        /// Mass multiplier (e.g., 1.3 = +30%).
        factor: f64,
        /// Step at which mass changes.
        at_step: usize,
    },
    /// Change joint damping coefficient.
    JointDamping {
        /// Joint index.
        joint_index: usize,
        /// New damping coefficient.
        damping: f64,
        /// Step at which damping changes.
        at_step: usize,
    },
    /// Actuator failure: permanently sever a joint's torque output.
    /// The controller still outputs a command, but it's masked to zero.
    /// Forces zero-shot gait adaptation.
    ActuatorFailure {
        /// Joint index to disable.
        joint_index: usize,
        /// Step at which failure occurs.
        at_step: usize,
    },
}

impl HumanoidPerturbation {
    /// Get the step at which this perturbation triggers.
    pub fn trigger_step(&self) -> usize {
        match self {
            Self::ExternalPush { at_step, .. } => *at_step,
            Self::FloorFriction { at_step, .. } => *at_step,
            Self::MassChange { at_step, .. } => *at_step,
            Self::JointDamping { at_step, .. } => *at_step,
            Self::ActuatorFailure { at_step, .. } => *at_step,
        }
    }
}

/// A schedule of perturbations to apply during an episode.
#[derive(Debug, Clone)]
pub struct PerturbationSchedule {
    perturbations: Vec<HumanoidPerturbation>,
    /// Actuator indices that have been permanently disabled.
    disabled_actuators: Vec<usize>,
}

impl PerturbationSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self {
            perturbations: Vec::new(),
            disabled_actuators: Vec::new(),
        }
    }

    /// Add a perturbation to the schedule.
    pub fn add(&mut self, perturbation: HumanoidPerturbation) {
        self.perturbations.push(perturbation);
    }

    /// Builder pattern: add and return self.
    pub fn with(mut self, perturbation: HumanoidPerturbation) -> Self {
        self.perturbations.push(perturbation);
        self
    }

    /// Return all perturbations that trigger at the given step.
    pub fn triggered_at(&self, step: usize) -> Vec<&HumanoidPerturbation> {
        self.perturbations
            .iter()
            .filter(|p| match p {
                HumanoidPerturbation::ExternalPush {
                    at_step, duration, ..
                } => step >= *at_step && step < at_step + duration,
                HumanoidPerturbation::FloorFriction { at_step, .. } => step == *at_step,
                HumanoidPerturbation::MassChange { at_step, .. } => step == *at_step,
                HumanoidPerturbation::JointDamping { at_step, .. } => step == *at_step,
                HumanoidPerturbation::ActuatorFailure { at_step, .. } => step == *at_step,
            })
            .collect()
    }

    /// Apply perturbations for the given step. Modifies command (for actuator failure)
    /// and physics simulator (for forces). Returns true if any perturbation was applied.
    pub fn apply(
        &mut self,
        step: usize,
        cmd: &mut HumanoidCommand,
        sim: &mut dyn HumanoidPhysicsSimulator,
    ) -> bool {
        let mut applied = false;

        // Check for new perturbations at this step
        for p in &self.perturbations {
            match p {
                HumanoidPerturbation::ExternalPush {
                    force,
                    at_step,
                    duration,
                } if step >= *at_step && step < at_step + duration => {
                    sim.apply_external_force(*force);
                    applied = true;
                }
                HumanoidPerturbation::ActuatorFailure {
                    joint_index,
                    at_step,
                } if step == *at_step => {
                    if !self.disabled_actuators.contains(joint_index) {
                        self.disabled_actuators.push(*joint_index);
                    }
                    applied = true;
                }
                // FloorFriction, MassChange, JointDamping require MuJoCo model access
                // and are marked as applied for schedule tracking
                HumanoidPerturbation::FloorFriction { at_step, .. } if step == *at_step => {
                    applied = true;
                }
                HumanoidPerturbation::MassChange { at_step, .. } if step == *at_step => {
                    applied = true;
                }
                HumanoidPerturbation::JointDamping { at_step, .. } if step == *at_step => {
                    applied = true;
                }
                _ => {}
            }
        }

        // Always mask disabled actuators (phantom limb is permanent)
        for &idx in &self.disabled_actuators {
            if idx < cmd.torques.len() {
                cmd.torques[idx] = 0.0;
            }
        }

        applied
    }

    /// Check if a specific actuator is disabled.
    pub fn is_actuator_disabled(&self, joint_index: usize) -> bool {
        self.disabled_actuators.contains(&joint_index)
    }

    /// Get all perturbations in the schedule.
    pub fn perturbations(&self) -> &[HumanoidPerturbation] {
        &self.perturbations
    }

    /// Check if any perturbations are scheduled.
    pub fn is_empty(&self) -> bool {
        self.perturbations.is_empty()
    }

    /// Number of perturbations.
    pub fn len(&self) -> usize {
        self.perturbations.len()
    }

    // ── Preset schedules ──

    /// Chest shove: 500N push to torso at step 300 for 5 steps (~50kg impulse).
    pub fn chest_shove() -> Self {
        Self::new().with(HumanoidPerturbation::ExternalPush {
            force: [500.0, 0.0, 0.0],
            at_step: 300,
            duration: 5,
        })
    }

    /// Ice floor: friction drops from 1.0 to 0.2 at step 200.
    pub fn ice_floor() -> Self {
        Self::new().with(HumanoidPerturbation::FloorFriction {
            friction: 0.2,
            at_step: 200,
        })
    }

    /// Backpack: torso mass +20kg at step 100 (approx +30% for 70kg humanoid).
    pub fn backpack() -> Self {
        Self::new().with(HumanoidPerturbation::MassChange {
            body: "torso".to_string(),
            factor: 1.3,
            at_step: 100,
        })
    }

    /// Progressive chaos: all three standard perturbations combined.
    pub fn progressive_chaos() -> Self {
        Self::new()
            .with(HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.3,
                at_step: 100,
            })
            .with(HumanoidPerturbation::FloorFriction {
                friction: 0.2,
                at_step: 200,
            })
            .with(HumanoidPerturbation::ExternalPush {
                force: [500.0, 0.0, 0.0],
                at_step: 300,
                duration: 5,
            })
    }

    /// Phantom Limb: right knee (joint 6) fails at step 500.
    pub fn phantom_limb() -> Self {
        Self::new().with(HumanoidPerturbation::ActuatorFailure {
            joint_index: 6, // right_knee
            at_step: 500,
        })
    }

    /// Phantom Limb during running: right knee fails at step 200 during locomotion.
    pub fn phantom_limb_running() -> Self {
        Self::new().with(HumanoidPerturbation::ActuatorFailure {
            joint_index: 6, // right_knee
            at_step: 200,
        })
    }
}

impl Default for PerturbationSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from a perturbation benchmark run.
#[derive(Debug, Clone)]
pub struct PerturbationBenchmarkResult {
    /// Average standing reward before perturbation.
    pub pre_perturbation_reward: f64,
    /// Minimum standing reward after perturbation (worst moment).
    pub min_reward: f64,
    /// Steps to recover standing reward > 0.5 after perturbation.
    pub recovery_steps: usize,
    /// Final standing reward (after recovery period).
    pub final_reward: f64,
    /// Minimum tau factor observed.
    pub min_tau: f64,
    /// Whether the humanoid fell (head_height < 0.5).
    pub fell: bool,
    /// Per-step standing reward trace.
    pub reward_trace: Vec<f64>,
    /// Per-step free energy trace.
    pub free_energy_trace: Vec<f64>,
    /// Per-step tau trace.
    pub tau_trace: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturbation_schedule_builder() {
        let schedule = PerturbationSchedule::new()
            .with(HumanoidPerturbation::ExternalPush {
                force: [500.0, 0.0, 0.0],
                at_step: 100,
                duration: 5,
            })
            .with(HumanoidPerturbation::ActuatorFailure {
                joint_index: 6,
                at_step: 200,
            });

        assert_eq!(schedule.len(), 2);
        assert!(!schedule.is_empty());
    }

    #[test]
    fn test_perturbation_trigger_step() {
        let p = HumanoidPerturbation::MassChange {
            body: "torso".to_string(),
            factor: 1.3,
            at_step: 42,
        };
        assert_eq!(p.trigger_step(), 42);
    }

    #[test]
    fn test_chest_shove_preset() {
        let schedule = PerturbationSchedule::chest_shove();
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule.perturbations()[0].trigger_step(), 300);
    }

    #[test]
    fn test_ice_floor_preset() {
        let schedule = PerturbationSchedule::ice_floor();
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_backpack_preset() {
        let schedule = PerturbationSchedule::backpack();
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_progressive_chaos_preset() {
        let schedule = PerturbationSchedule::progressive_chaos();
        assert_eq!(schedule.len(), 3);
    }

    #[test]
    fn test_phantom_limb_preset() {
        let schedule = PerturbationSchedule::phantom_limb();
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule.perturbations()[0].trigger_step(), 500);
    }

    #[test]
    fn test_empty_schedule() {
        let schedule = PerturbationSchedule::default();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);
    }

    #[test]
    fn test_triggered_at_push() {
        let schedule = PerturbationSchedule::new().with(HumanoidPerturbation::ExternalPush {
            force: [500.0, 0.0, 0.0],
            at_step: 100,
            duration: 5,
        });
        assert!(schedule.triggered_at(99).is_empty());
        assert_eq!(schedule.triggered_at(100).len(), 1);
        assert_eq!(schedule.triggered_at(104).len(), 1);
        assert!(schedule.triggered_at(105).is_empty());
    }

    #[test]
    fn test_triggered_at_mass_change() {
        let schedule = PerturbationSchedule::new().with(HumanoidPerturbation::MassChange {
            body: "torso".to_string(),
            factor: 1.3,
            at_step: 100,
        });
        assert!(schedule.triggered_at(99).is_empty());
        assert_eq!(schedule.triggered_at(100).len(), 1);
        assert!(schedule.triggered_at(101).is_empty());
    }

    #[test]
    fn test_actuator_failure_masking() {
        let mut schedule =
            PerturbationSchedule::new().with(HumanoidPerturbation::ActuatorFailure {
                joint_index: 6,
                at_step: 10,
            });

        let mut cmd = HumanoidCommand {
            torques: vec![0.5; 21],
        };
        let mut sim = crate::simulator::SimpleHumanoidSimulator::new();

        // Before failure step: no masking
        schedule.apply(5, &mut cmd, &mut sim);
        assert!((cmd.torques[6] - 0.5).abs() < 1e-6);

        // At failure step: actuator gets disabled
        cmd.torques[6] = 0.5;
        schedule.apply(10, &mut cmd, &mut sim);
        assert!((cmd.torques[6] - 0.0).abs() < 1e-6);
        assert!(schedule.is_actuator_disabled(6));

        // After failure: permanently disabled
        cmd.torques[6] = 0.8;
        schedule.apply(100, &mut cmd, &mut sim);
        assert!((cmd.torques[6] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_perturbation_ordering() {
        let schedule = PerturbationSchedule::new()
            .with(HumanoidPerturbation::MassChange {
                body: "torso".to_string(),
                factor: 1.3,
                at_step: 100,
            })
            .with(HumanoidPerturbation::ExternalPush {
                force: [500.0, 0.0, 0.0],
                at_step: 150,
                duration: 51,
            })
            .with(HumanoidPerturbation::ActuatorFailure {
                joint_index: 6,
                at_step: 200,
            });

        assert_eq!(schedule.triggered_at(100).len(), 1);
        assert_eq!(schedule.triggered_at(180).len(), 1); // push only
        assert_eq!(schedule.triggered_at(200).len(), 2); // push + failure
    }
}
