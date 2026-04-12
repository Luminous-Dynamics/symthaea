// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motor trajectory dreaming: counterfactual replay of embodied motor commands.
//!
//! Implements `DreamableAction` for motor trajectories, enabling the dream
//! engine to replay motor commands with perturbation injection and discover
//! fragile states that the robot never encountered but could have.
//!
//! Science: Rasch & Born (2013) — targeted memory reactivation during sleep
//! consolidates motor procedural knowledge. Rats replay maze navigation
//! trajectories during REM sleep (Wilson & McNaughton, 1994).

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment::{EmbodimentPlatform, MotorSafetyLevel};

use super::DreamableAction;

/// A recorded motor trajectory segment from an embodied platform.
///
/// Captures a sequence of motor commands alongside consciousness and safety
/// state. When the dream engine replays this, it perturbs the commands to
/// discover what could have gone wrong (or better).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorTrajectory {
    /// Which platform produced this trajectory.
    pub platform: EmbodimentPlatform,
    /// Sequence of motor command vectors (one per timestep).
    /// Length of inner vec = num_actuators for this platform.
    pub joint_commands: Vec<Vec<f32>>,
    /// Phi trace during this trajectory segment.
    pub phi_trace: Vec<f64>,
    /// Safety levels during this trajectory.
    pub safety_levels: Vec<MotorSafetyLevel>,
    /// Prediction errors at each step.
    pub prediction_errors: Vec<f32>,
    /// Timestep dt used during recording.
    pub dt: f32,
}

impl MotorTrajectory {
    /// Create from recorded embodiment data.
    pub fn new(platform: EmbodimentPlatform, dt: f32) -> Self {
        Self {
            platform,
            joint_commands: Vec::new(),
            phi_trace: Vec::new(),
            safety_levels: Vec::new(),
            prediction_errors: Vec::new(),
            dt,
        }
    }

    /// Record one timestep.
    pub fn record_step(
        &mut self,
        commands: Vec<f32>,
        phi: f64,
        safety: MotorSafetyLevel,
        prediction_error: f32,
    ) {
        self.joint_commands.push(commands);
        self.phi_trace.push(phi);
        self.safety_levels.push(safety);
        self.prediction_errors.push(prediction_error);
    }

    /// Number of recorded timesteps.
    pub fn len(&self) -> usize {
        self.joint_commands.len()
    }

    /// Whether the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.joint_commands.is_empty()
    }

    /// Mean prediction error across the trajectory.
    pub fn mean_prediction_error(&self) -> f32 {
        if self.prediction_errors.is_empty() {
            return 0.0;
        }
        self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32
    }

    /// Maximum prediction error (most surprising moment).
    pub fn max_prediction_error(&self) -> f32 {
        self.prediction_errors
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
    }

    /// Whether any safety level reached Orange or Red.
    pub fn had_safety_event(&self) -> bool {
        self.safety_levels
            .iter()
            .any(|s| matches!(s, MotorSafetyLevel::Orange | MotorSafetyLevel::Red))
    }
}

impl DreamableAction for MotorTrajectory {
    /// Perturb the motor trajectory by adding noise to commands.
    ///
    /// Uses deterministic noise from the seed. The perturbation magnitude
    /// scales with the step index — later steps get more perturbation,
    /// simulating accumulating uncertainty.
    fn perturb(&self, seed: u64) -> Self {
        let mut perturbed = self.clone();
        for (step, commands) in perturbed.joint_commands.iter_mut().enumerate() {
            let step_scale = 1.0 + (step as f32 * 0.01); // Accumulating noise
            for (j, cmd) in commands.iter_mut().enumerate() {
                // Deterministic pseudo-random perturbation
                let mut s = seed
                    .wrapping_add(step as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(j as u64);
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                let noise = ((s as f64 / u64::MAX as f64) - 0.5) * 0.2 * step_scale as f64;
                *cmd = (*cmd + noise as f32).clamp(-1.0, 1.0);
            }
        }
        perturbed
    }

    /// Predict the outcome of this trajectory as a state vector.
    ///
    /// Returns a simplified state estimate: [mean_PE, max_PE, had_safety_event,
    /// mean_phi, min_phi, trajectory_length]. This is sufficient for the dream
    /// engine to compare counterfactual outcomes.
    fn predict_outcome(&self, _state: &[f32]) -> Vec<f32> {
        let mean_pe = self.mean_prediction_error();
        let max_pe = self.max_prediction_error();
        let safety_event = if self.had_safety_event() { 1.0 } else { 0.0 };
        let mean_phi = if self.phi_trace.is_empty() {
            0.0
        } else {
            (self.phi_trace.iter().sum::<f64>() / self.phi_trace.len() as f64) as f32
        };
        let min_phi = self
            .phi_trace
            .iter()
            .copied()
            .fold(f64::MAX, f64::min) as f32;
        let len = self.len() as f32;

        vec![mean_pe, max_pe, safety_event, mean_phi, min_phi, len]
    }

    /// Total control effort across the trajectory.
    fn magnitude(&self) -> f32 {
        self.joint_commands
            .iter()
            .flat_map(|cmds| cmds.iter())
            .map(|c| c.abs())
            .sum::<f32>()
    }
}

/// Wisdom entry produced by dreaming about motor trajectories.
///
/// Contains the original trajectory, the better counterfactual,
/// and the improvement metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorWisdom {
    /// Platform this wisdom applies to.
    pub platform: EmbodimentPlatform,
    /// The original recorded trajectory.
    pub original: MotorTrajectory,
    /// The better counterfactual trajectory discovered by dreaming.
    pub better: MotorTrajectory,
    /// Improvement in mean prediction error (positive = better).
    pub pe_improvement: f32,
    /// Improvement in Phi (positive = higher consciousness during maneuver).
    pub phi_improvement: f64,
    /// Confidence in this wisdom (0.0–1.0).
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(n_steps: usize) -> MotorTrajectory {
        let mut traj = MotorTrajectory::new(EmbodimentPlatform::Manipulator, 0.002);
        for i in 0..n_steps {
            traj.record_step(
                vec![0.1 * (i as f32), -0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
                0.7,
                MotorSafetyLevel::Green,
                0.05 + 0.001 * i as f32,
            );
        }
        traj
    }

    #[test]
    fn test_trajectory_recording() {
        let traj = make_trajectory(50);
        assert_eq!(traj.len(), 50);
        assert!(!traj.is_empty());
        assert!(!traj.had_safety_event());
    }

    #[test]
    fn test_prediction_error_stats() {
        let traj = make_trajectory(100);
        assert!(traj.mean_prediction_error() > 0.0);
        assert!(traj.max_prediction_error() >= traj.mean_prediction_error());
    }

    #[test]
    fn test_perturb_changes_commands() {
        let traj = make_trajectory(20);
        let perturbed = traj.perturb(42);
        assert_eq!(perturbed.len(), traj.len());
        // Commands should differ
        let different = traj
            .joint_commands
            .iter()
            .zip(perturbed.joint_commands.iter())
            .any(|(a, b)| a != b);
        assert!(different, "Perturbation should change at least one command");
    }

    #[test]
    fn test_perturbed_commands_bounded() {
        let traj = make_trajectory(50);
        let perturbed = traj.perturb(123);
        for cmds in &perturbed.joint_commands {
            for &c in cmds {
                assert!(
                    c >= -1.0 && c <= 1.0,
                    "Perturbed command {c} out of [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_predict_outcome() {
        let traj = make_trajectory(30);
        let outcome = traj.predict_outcome(&[]);
        assert_eq!(outcome.len(), 6);
        assert!(outcome.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_magnitude() {
        let traj = make_trajectory(10);
        assert!(traj.magnitude() > 0.0);
        assert!(traj.magnitude().is_finite());
    }

    #[test]
    fn test_safety_event_detection() {
        let mut traj = make_trajectory(5);
        assert!(!traj.had_safety_event());
        traj.record_step(vec![0.0; 7], 0.05, MotorSafetyLevel::Red, 0.9);
        assert!(traj.had_safety_event());
    }

    #[test]
    fn test_perturb_deterministic() {
        let traj = make_trajectory(20);
        let a = traj.perturb(42);
        let b = traj.perturb(42);
        assert_eq!(a.joint_commands, b.joint_commands);
    }

    #[test]
    fn test_different_seeds_different_perturbations() {
        let traj = make_trajectory(20);
        let a = traj.perturb(42);
        let b = traj.perturb(99);
        assert_ne!(a.joint_commands, b.joint_commands);
    }
}
