// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Goal system and world model bridges for the cognitive loop.
//!
//! - `GoalSystemBridge`: Goal-directed attention with priority-based weighting
//! - `WorldModelBridge`: Hierarchical world model predictions with multi-level state

use serde::{Deserialize, Serialize};

/// Goal representation for the cognitive loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveGoal {
    /// Goal ID
    pub id: String,
    /// Goal description
    pub description: String,
    /// Priority (0.0 to 1.0)
    pub priority: f32,
    /// Progress (0.0 to 1.0)
    pub progress: f32,
    /// Whether actively pursued
    pub is_active: bool,
    /// Attention weight (how much to bias attention toward this goal)
    pub attention_weight: f32,
}

impl CognitiveGoal {
    /// Create a new goal
    pub fn new(id: impl Into<String>, description: impl Into<String>, priority: f32) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            priority: priority.clamp(0.0, 1.0),
            progress: 0.0,
            is_active: true,
            attention_weight: priority, // Initially weight by priority
        }
    }
}

/// Goal System Bridge for goal-directed attention
#[derive(Debug, Clone, Default)]
pub struct GoalSystemBridge {
    /// Active goals
    goals: Vec<CognitiveGoal>,
    /// Maximum concurrent goals
    max_goals: usize,
}

impl GoalSystemBridge {
    /// Create with default capacity
    pub fn new() -> Self {
        Self {
            goals: Vec::with_capacity(10),
            max_goals: 10,
        }
    }

    /// Add a goal
    pub fn add_goal(&mut self, goal: CognitiveGoal) {
        if self.goals.len() < self.max_goals {
            self.goals.push(goal);
        }
    }

    /// Add a goal derived from a high-level mission narrative.
    pub fn add_narrative_goal(&mut self, narrative: &str, priority: f32) {
        let id = format!("mission_{}", self.goals.len());
        let goal = CognitiveGoal::new(id, narrative, priority);
        self.add_goal(goal);
    }

    /// Get attention bias based on goals
    ///
    /// Returns a multiplier for attention based on goal priorities
    pub fn attention_bias(&self) -> f32 {
        if self.goals.is_empty() {
            return 1.0;
        }
        let active_weight: f32 = self
            .goals
            .iter()
            .filter(|g| g.is_active)
            .map(|g| g.attention_weight)
            .sum();
        1.0 + active_weight * 0.2 // Up to 20% boost per unit of goal weight
    }

    /// Update goal progress
    pub fn update_progress(&mut self, goal_id: &str, delta: f32) {
        if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
            goal.progress = (goal.progress + delta).clamp(0.0, 1.0);
            if goal.progress >= 1.0 {
                goal.is_active = false;
            }
        }
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&CognitiveGoal> {
        self.goals.iter().filter(|g| g.is_active).collect()
    }

    /// Get highest priority active goal
    pub fn top_goal(&self) -> Option<&CognitiveGoal> {
        self.goals.iter().filter(|g| g.is_active).max_by(|a, b| {
            a.priority
                .partial_cmp(&b.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Clear completed goals
    pub fn clear_completed(&mut self) {
        self.goals.retain(|g| g.progress < 1.0);
    }

    /// Reset all goals
    pub fn reset(&mut self) {
        self.goals.clear();
    }
}

/// World Model Bridge for grounded prediction
///
/// Lightweight interface to hierarchical world model predictions
#[derive(Debug, Clone)]
pub struct WorldModelBridge {
    /// Multi-level state representations
    level_states: Vec<Vec<f32>>,
    /// Level dimensions
    level_dims: Vec<usize>,
    /// Prediction error at each level
    level_errors: Vec<f32>,
    /// Total predictions made
    pub total_predictions: u64,
    /// Average prediction error across levels
    pub avg_error: f32,
}

impl Default for WorldModelBridge {
    fn default() -> Self {
        // Default 4-level hierarchy
        let level_dims = vec![64, 128, 256, 128];
        Self {
            level_states: level_dims.iter().map(|&d| vec![0.0; d]).collect(),
            level_dims,
            level_errors: vec![0.0; 4],
            total_predictions: 0,
            avg_error: 0.0,
        }
    }
}

impl WorldModelBridge {
    /// Update with sensory input (level 0)
    pub fn update_sensory(&mut self, input: &[f32]) {
        if input.len() >= self.level_dims[0] {
            // Compute prediction error at level 0
            let error: f32 = self.level_states[0]
                .iter()
                .zip(input.iter().take(self.level_dims[0]))
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f32>()
                .sqrt();
            self.level_errors[0] = error;

            // Update level 0 state
            for (i, &val) in input.iter().take(self.level_dims[0]).enumerate() {
                self.level_states[0][i] = val;
            }

            // Propagate up (simplified: just average to higher levels)
            self.propagate_up();

            self.total_predictions += 1;
            // Safe division: use max(1) to prevent division by zero
            self.avg_error =
                self.level_errors.iter().sum::<f32>() / self.level_errors.len().max(1) as f32;
        }
    }

    /// Propagate state up the hierarchy
    fn propagate_up(&mut self) {
        for level in 1..self.level_states.len() {
            let prev_level = level - 1;
            let prev_dim = self.level_dims[prev_level];
            let curr_dim = self.level_dims[level];

            // Simple projection: chunk and average
            // Safe division: use max(1) to prevent division by zero
            let chunk_size = (prev_dim + curr_dim - 1) / curr_dim.max(1);
            for i in 0..curr_dim {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(prev_dim);
                if start < prev_dim {
                    let sum: f32 = self.level_states[prev_level][start..end].iter().sum();
                    // Safe cast via f64 to prevent precision loss on large counts
                    let count = end.saturating_sub(start) as f64;
                    self.level_states[level][i] = (sum as f64 / count.max(1.0)) as f32;
                }
            }
        }
    }

    /// Get prediction at a specific level
    pub fn get_level_state(&self, level: usize) -> Option<&[f32]> {
        self.level_states.get(level).map(|v| v.as_slice())
    }

    /// Get prediction error at each level
    pub fn level_errors(&self) -> &[f32] {
        &self.level_errors
    }

    /// Get abstract level state (highest level - for planning)
    pub fn abstract_state(&self) -> &[f32] {
        self.level_states
            .last()
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Reset the world model
    pub fn reset(&mut self) {
        for state in &mut self.level_states {
            state.fill(0.0);
        }
        self.level_errors.fill(0.0);
        self.total_predictions = 0;
        self.avg_error = 0.0;
    }

    /// Incorporate discovered causal structure into the world model.
    pub fn incorporate_causal_structure(&mut self, causal_edges: &[(usize, usize, f32)]) {
        if causal_edges.is_empty() {
            return;
        }

        // Count how many causal edges touch each level-0 dimension.
        let dim0 = self.level_dims[0];
        let mut causal_strength = vec![0.0f32; dim0];
        for &(from, to, strength) in causal_edges {
            if from < dim0 {
                causal_strength[from] += strength;
            }
            if to < dim0 {
                causal_strength[to] += strength;
            }
        }

        let causal_dims = causal_strength.iter().filter(|&&cs| cs > 0.0).count();
        let causal_fraction = causal_dims as f32 / dim0.max(1) as f32;
        let mean_strength = if causal_dims > 0 {
            causal_strength.iter().filter(|&&cs| cs > 0.0).sum::<f32>() / causal_dims as f32
        } else {
            0.0
        };
        // Reduction = fraction_of_known_dims × strength_confidence, capped at 30%
        let reduction = (causal_fraction * mean_strength.min(1.0) * 0.5).min(0.3);
        if reduction > 0.001 {
            self.level_errors[0] *= 1.0 - reduction;
        }
    }

    /// Increase plasticity in the world model (triggered by high learning signals)
    pub fn increase_plasticity(&mut self, plasticity_signal: f32) {
        let decay = 1.0 - (plasticity_signal * 0.1).clamp(0.0, 0.3);
        for level_state in &mut self.level_states {
            for val in level_state.iter_mut() {
                *val *= decay;
            }
        }
    }
}
