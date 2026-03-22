// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motor system for embodied active inference.

use std::collections::VecDeque;

use super::types::{MotorCommand, MotorCommandStats, MotorCommandType, MotorOutcome};

/// Simple random number generator for motor noise (deterministic for testing)
pub(crate) fn rand_f64() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(12345);

    let mut s = SEED.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    SEED.store(s, Ordering::Relaxed);

    (s as f64) / (u64::MAX as f64)
}

/// Motor system interface for embodied active inference
#[derive(Debug, Clone)]
pub struct MotorSystem {
    /// Last executed command
    pub(crate) last_command: Option<MotorCommand>,
    /// Command history
    command_history: VecDeque<MotorCommand>,
    /// Maximum history size
    max_history: usize,
    /// Prediction error for motor outcomes
    motor_prediction_errors: VecDeque<f64>,
    /// Proprioceptive feedback (for embodied systems)
    proprioceptive_state: Vec<f64>,
}

impl Default for MotorSystem {
    fn default() -> Self {
        Self {
            last_command: None,
            command_history: VecDeque::with_capacity(100),
            max_history: 100,
            motor_prediction_errors: VecDeque::with_capacity(50),
            proprioceptive_state: vec![0.5; 4], // Default 4-dimensional state
        }
    }
}

impl MotorSystem {
    /// Create a new motor system with specified state dimension
    pub fn new(state_dim: usize) -> Self {
        Self {
            last_command: None,
            command_history: VecDeque::with_capacity(100),
            max_history: 100,
            motor_prediction_errors: VecDeque::with_capacity(50),
            proprioceptive_state: vec![0.5; state_dim],
        }
    }

    /// Execute a motor command
    pub fn execute(&mut self, command: MotorCommand) -> MotorOutcome {
        // Store command
        self.last_command = Some(command.clone());
        if self.command_history.len() >= self.max_history {
            self.command_history.pop_front();
        }
        self.command_history.push_back(command.clone());

        // Simulate execution (in real system, this would actuate)
        let executed_intensity = command.intensity * (0.9 + rand_f64() * 0.2); // Add noise
        let execution_success = rand_f64() < (0.8 + command.confidence * 0.2);

        // Update proprioceptive state based on command
        self.update_proprioception(&command);

        MotorOutcome {
            command_type: command.command_type,
            executed_intensity: executed_intensity.clamp(0.0, 1.0),
            success: execution_success,
            proprioceptive_feedback: self.proprioceptive_state.clone(),
            prediction_error: if let Some(ref predicted) = command.predicted_outcome {
                self.compute_prediction_error(predicted)
            } else {
                0.0
            },
        }
    }

    /// Update proprioceptive state based on executed command
    fn update_proprioception(&mut self, command: &MotorCommand) {
        let delta = command.intensity * 0.1;
        match command.command_type {
            MotorCommandType::AttentionShift => {
                // Shift attention dimension
                if !self.proprioceptive_state.is_empty() {
                    self.proprioceptive_state[0] += delta * (rand_f64() - 0.5);
                    self.proprioceptive_state[0] = self.proprioceptive_state[0].clamp(0.0, 1.0);
                }
            }
            MotorCommandType::ExplorationTrigger => {
                // Increase variability in all dimensions
                for dim in &mut self.proprioceptive_state {
                    *dim += delta * (rand_f64() - 0.5) * 2.0;
                    *dim = dim.clamp(0.0, 1.0);
                }
            }
            MotorCommandType::MemoryConsolidate => {
                // Stabilize state (reduce change)
                // No change to proprioception
            }
            _ => {
                // Small random drift
                for dim in &mut self.proprioceptive_state {
                    *dim += 0.01 * (rand_f64() - 0.5);
                    *dim = dim.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Compute prediction error between predicted and actual outcome
    fn compute_prediction_error(&mut self, predicted: &[f64]) -> f64 {
        let error: f64 = predicted
            .iter()
            .zip(self.proprioceptive_state.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            .sqrt()
            / predicted.len().max(1) as f64;

        if self.motor_prediction_errors.len() >= 50 {
            self.motor_prediction_errors.pop_front();
        }
        self.motor_prediction_errors.push_back(error);

        error
    }

    /// Get average motor prediction error
    pub fn average_prediction_error(&self) -> f64 {
        if self.motor_prediction_errors.is_empty() {
            0.0
        } else {
            self.motor_prediction_errors.iter().sum::<f64>()
                / self.motor_prediction_errors.len() as f64
        }
    }

    /// Get current proprioceptive state
    pub fn proprioceptive_state(&self) -> &[f64] {
        &self.proprioceptive_state
    }

    /// Set proprioceptive state (for embodied systems receiving sensor input)
    pub fn set_proprioceptive_state(&mut self, state: Vec<f64>) {
        self.proprioceptive_state = state;
    }

    /// Get command statistics
    pub fn command_stats(&self) -> MotorCommandStats {
        let total_commands = self.command_history.len();
        let meaningful_commands = self
            .command_history
            .iter()
            .filter(|c| c.is_meaningful())
            .count();

        let avg_intensity = if total_commands > 0 {
            self.command_history
                .iter()
                .map(|c| c.intensity)
                .sum::<f64>()
                / total_commands as f64
        } else {
            0.0
        };

        let avg_confidence = if total_commands > 0 {
            self.command_history
                .iter()
                .map(|c| c.confidence)
                .sum::<f64>()
                / total_commands as f64
        } else {
            0.0
        };

        MotorCommandStats {
            total_commands,
            meaningful_commands,
            avg_intensity,
            avg_confidence,
            avg_prediction_error: self.average_prediction_error(),
        }
    }

    /// Reset motor system
    pub fn reset(&mut self) {
        self.last_command = None;
        self.command_history.clear();
        self.motor_prediction_errors.clear();
        for dim in &mut self.proprioceptive_state {
            *dim = 0.5;
        }
    }
}
