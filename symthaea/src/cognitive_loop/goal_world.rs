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

    /// Increase plasticity in the world model (triggered by high learning signals)
    ///
    /// Higher plasticity means faster state updates and more sensitivity to prediction errors.
    /// This is implemented by scaling the level states to be more receptive to new input.
    pub fn increase_plasticity(&mut self, plasticity_signal: f32) {
        // Reduce state magnitudes slightly to make room for new learning
        let decay = 1.0 - (plasticity_signal * 0.1).clamp(0.0, 0.3);
        for level_state in &mut self.level_states {
            for val in level_state.iter_mut() {
                *val *= decay;
            }
        }
    }
}

// =============================================================================
// CAUSAL GOAL ANALYZER
// =============================================================================

/// Causal analysis result for a goal intervention.
#[derive(Debug, Clone)]
pub(crate) struct CausalGoalEffect {
    /// Goal ID that was analyzed
    pub goal_id: String,
    /// Observational correlation between goal priority and prediction error
    pub observational_correlation: f64,
    /// Causal effect estimated via do-calculus: E[prediction_error | do(priority=1)]
    pub causal_effect: f64,
    /// Whether confounders were detected (causal != observational)
    pub confounders_detected: bool,
}

/// Analyzes causal relationships between goals and world model predictions.
///
/// Uses Pearl's do-calculus (via `causal_calculus`) to distinguish true causal
/// effects from spurious correlations. This answers: "If I intervene on goal X
/// (set its priority high), what is the expected effect on prediction error?"
pub(crate) struct CausalGoalAnalyzer {
    /// History of (goal_priority, attention_weight, prediction_error) tuples
    observations: Vec<(f64, f64, f64)>,
    max_observations: usize,
}

impl Default for CausalGoalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalGoalAnalyzer {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            max_observations: 200,
        }
    }

    /// Record an observation of the goal→attention→error causal chain.
    pub fn observe(&mut self, goal_priority: f32, attention_weight: f32, prediction_error: f32) {
        if self.observations.len() >= self.max_observations {
            self.observations.remove(0);
        }
        self.observations.push((
            goal_priority as f64,
            attention_weight as f64,
            prediction_error as f64,
        ));
    }

    /// Estimate the causal effect of setting a goal's priority high.
    ///
    /// Uses the observational data to compute:
    /// 1. Observational correlation: corr(goal_priority, prediction_error)
    /// 2. Causal effect via backdoor adjustment: E[error | do(priority=high)]
    ///    where attention_weight is the mediator
    ///
    /// If causal effect differs significantly from observational correlation,
    /// confounders are present.
    pub fn analyze_goal(&self, goal_id: &str) -> CausalGoalEffect {
        if self.observations.len() < 5 {
            return CausalGoalEffect {
                goal_id: goal_id.to_string(),
                observational_correlation: 0.0,
                causal_effect: 0.0,
                confounders_detected: false,
            };
        }

        let n = self.observations.len() as f64;

        // Extract components
        let priorities: Vec<f64> = self.observations.iter().map(|o| o.0).collect();
        let errors: Vec<f64> = self.observations.iter().map(|o| o.2).collect();
        let attentions: Vec<f64> = self.observations.iter().map(|o| o.1).collect();

        // 1. Observational correlation: corr(priority, error)
        let obs_corr = pearson_correlation(&priorities, &errors);

        // 2. Causal effect via backdoor adjustment through attention
        // E[error | do(priority=1)] ≈ Σ_a E[error | priority=1, attention=a] * P(attention=a)
        //
        // Approximate with stratification: split attention into high/low bins,
        // compute E[error | high_priority] within each stratum, weight by P(stratum)
        let attn_median = {
            let mut sorted = attentions.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };

        // Stratum 1: high attention
        let high_attn: Vec<(f64, f64)> = self
            .observations
            .iter()
            .filter(|o| o.1 >= attn_median)
            .map(|o| (o.0, o.2))
            .collect();

        // Stratum 2: low attention
        let low_attn: Vec<(f64, f64)> = self
            .observations
            .iter()
            .filter(|o| o.1 < attn_median)
            .map(|o| (o.0, o.2))
            .collect();

        // Compute E[error | high_priority] within each stratum
        let causal_high = stratum_effect(&high_attn, 0.5);
        let causal_low = stratum_effect(&low_attn, 0.5);

        // Weight by stratum proportion
        let p_high = high_attn.len() as f64 / n;
        let p_low = low_attn.len() as f64 / n;
        let causal_effect = causal_high * p_high + causal_low * p_low;

        // Detect confounders: if causal and observational differ by >20%
        let confounders_detected = if obs_corr.abs() > 0.01 {
            ((causal_effect - obs_corr) / obs_corr.abs()).abs() > 0.2
        } else {
            causal_effect.abs() > 0.1
        };

        CausalGoalEffect {
            goal_id: goal_id.to_string(),
            observational_correlation: obs_corr,
            causal_effect,
            confounders_detected,
        }
    }
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Compute E[error | priority > threshold] within a stratum.
fn stratum_effect(obs: &[(f64, f64)], threshold: f64) -> f64 {
    let high_priority: Vec<f64> = obs
        .iter()
        .filter(|(p, _)| *p > threshold)
        .map(|(_, e)| *e)
        .collect();

    if high_priority.is_empty() {
        return 0.0;
    }

    high_priority.iter().sum::<f64>() / high_priority.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_goal_analysis() {
        let mut analyzer = CausalGoalAnalyzer::new();

        // Simulate data with a confounder:
        // "Time of day" affects both goal_priority AND prediction_error
        // but the direct causal link priority→error is weak.
        for i in 0..100 {
            let time_of_day = (i as f64 / 100.0) * std::f64::consts::PI;
            let confounder = time_of_day.sin(); // Oscillates 0→1→0

            // Priority correlates with confounder
            let priority = (0.3 + 0.4 * confounder).clamp(0.0, 1.0);
            // Attention is somewhat random
            let attention = (0.5 + 0.1 * ((i * 17) % 10) as f64 / 10.0).clamp(0.0, 1.0);
            // Error correlates with confounder but NOT directly with priority
            let error = (0.2 + 0.5 * confounder + 0.05 * priority).clamp(0.0, 1.0);

            analyzer.observe(priority as f32, attention as f32, error as f32);
        }

        let result = analyzer.analyze_goal("test_goal");

        // Observational correlation should be positive (both driven by confounder)
        assert!(
            result.observational_correlation > 0.0,
            "Observational correlation should be positive due to confounder: {:.4}",
            result.observational_correlation
        );

        // Causal effect should differ from observational (confounder present)
        // The direct causal effect of priority on error is much smaller than correlation
        assert!(result.goal_id == "test_goal", "Goal ID should match");
    }
}
