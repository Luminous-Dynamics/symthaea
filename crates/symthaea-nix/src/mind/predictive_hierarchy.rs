// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Hierarchy — 4-Level Prediction Stack
//!
//! Each level generates predictions about the level below. Prediction errors
//! propagate upward — if a config change surprises Level 1, that surprise
//! propagates to Level 2 and potentially changes concept understanding.
//!
//! - Level 0 (Sensory): Raw config lines, log entries, package lists
//! - Level 1 (Features): Option patterns, service dependencies, package groups
//! - Level 2 (Concepts): "web server stack", "GPU acceleration", "dev environment"
//! - Level 3 (Goals): "make the system production-ready", "set up for ML work"

use symthaea_core::hdc::ContinuousHV;

/// A single prediction level in the hierarchy.
#[derive(Debug, Clone)]
pub struct PredictionLevel {
    /// Current state representation at this level.
    pub state: ContinuousHV,
    /// Prediction of the level below's state.
    pub prediction: ContinuousHV,
    /// Current prediction error (surprise).
    pub prediction_error: f64,
    /// Learning rate for this level (higher levels learn slower).
    pub learning_rate: f64,
    /// Name of this level.
    pub name: &'static str,
}

impl PredictionLevel {
    /// Create a new prediction level.
    fn new(dim: usize, name: &'static str, learning_rate: f64) -> Self {
        Self {
            state: ContinuousHV::zero(dim),
            prediction: ContinuousHV::zero(dim),
            prediction_error: 0.0,
            learning_rate,
            name,
        }
    }

    /// Update this level with a new observation from below.
    ///
    /// Returns the prediction error (surprise).
    fn update(&mut self, observation: &ContinuousHV) -> f64 {
        // Compute prediction error
        self.prediction_error = 1.0 - self.prediction.similarity(observation).max(0.0) as f64;

        // Update state: blend current state with observation
        let lr = self.learning_rate as f32;
        let refs = [&self.state, observation];
        let weights = [1.0 - lr, lr];
        self.state = ContinuousHV::weighted_bundle(&refs, &weights);

        // Update prediction toward observation
        let pred_refs = [&self.prediction, observation];
        let pred_weights = [1.0 - lr * 0.5, lr * 0.5];
        self.prediction = ContinuousHV::weighted_bundle(&pred_refs, &pred_weights);

        self.prediction_error
    }
}

/// The 4-level predictive hierarchy for NixOS understanding.
pub struct PredictiveHierarchy {
    /// Level 0: Sensory (raw config/log data)
    pub sensory: PredictionLevel,
    /// Level 1: Features (option patterns, service deps)
    pub features: PredictionLevel,
    /// Level 2: Concepts (web stack, dev env, etc.)
    pub concepts: PredictionLevel,
    /// Level 3: Goals (production-ready, ML setup, etc.)
    pub goals: PredictionLevel,
    /// HDC dimension.
    dim: usize,
}

impl PredictiveHierarchy {
    /// Create a new predictive hierarchy.
    pub fn new(dim: usize) -> Self {
        Self {
            sensory: PredictionLevel::new(dim, "sensory", 0.3),
            features: PredictionLevel::new(dim, "features", 0.2),
            concepts: PredictionLevel::new(dim, "concepts", 0.1),
            goals: PredictionLevel::new(dim, "goals", 0.05),
            dim,
        }
    }

    /// Process a new observation through the hierarchy.
    ///
    /// The observation enters at Level 0 (sensory) and prediction errors
    /// propagate upward through all levels.
    ///
    /// Returns the total prediction error across all levels.
    pub fn process(&mut self, observation: &ContinuousHV) -> f64 {
        // Level 0: Direct observation
        let error_0 = self.sensory.update(observation);

        // Level 1: Observes Level 0's state
        let error_1 = self.features.update(&self.sensory.state);

        // Level 2: Observes Level 1's state
        let error_2 = self.concepts.update(&self.features.state);

        // Level 3: Observes Level 2's state
        let error_3 = self.goals.update(&self.concepts.state);

        // Total weighted prediction error (higher levels weighted more)
        error_0 * 0.1 + error_1 * 0.2 + error_2 * 0.3 + error_3 * 0.4
    }

    /// Get the current understanding at a specific level.
    pub fn state_at_level(&self, level: usize) -> &ContinuousHV {
        match level {
            0 => &self.sensory.state,
            1 => &self.features.state,
            2 => &self.concepts.state,
            _ => &self.goals.state,
        }
    }

    /// Get prediction error at each level.
    pub fn errors(&self) -> [f64; 4] {
        [
            self.sensory.prediction_error,
            self.features.prediction_error,
            self.concepts.prediction_error,
            self.goals.prediction_error,
        ]
    }

    /// Total free energy (weighted prediction error across all levels).
    pub fn free_energy(&self) -> f64 {
        let [e0, e1, e2, e3] = self.errors();
        e0 * 0.1 + e1 * 0.2 + e2 * 0.3 + e3 * 0.4
    }

    /// Whether the system is "surprised" (high prediction error).
    pub fn is_surprised(&self) -> bool {
        self.free_energy() > 0.5
    }

    /// Get the goal-level state (highest abstraction).
    pub fn goal_state(&self) -> &ContinuousHV {
        &self.goals.state
    }

    /// Reset all levels.
    pub fn reset(&mut self) {
        self.sensory = PredictionLevel::new(self.dim, "sensory", 0.3);
        self.features = PredictionLevel::new(self.dim, "features", 0.2);
        self.concepts = PredictionLevel::new(self.dim, "concepts", 0.1);
        self.goals = PredictionLevel::new(self.dim, "goals", 0.05);
    }
}

impl Default for PredictiveHierarchy {
    fn default() -> Self {
        Self::new(symthaea_core::hdc::HDC_DIMENSION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_error_decreases() {
        let dim = 1024;
        let mut hierarchy = PredictiveHierarchy::new(dim);
        let observation = ContinuousHV::random(dim, 42);

        // First observation — high prediction error (everything is zero)
        let error1 = hierarchy.process(&observation);

        // Same observation again — error should decrease
        let error2 = hierarchy.process(&observation);

        // And again
        let error3 = hierarchy.process(&observation);

        assert!(
            error3 <= error2 || error2 <= error1,
            "Prediction error should generally decrease with repeated observations: {:.3} → {:.3} → {:.3}",
            error1,
            error2,
            error3,
        );
    }

    #[test]
    fn test_novel_observation_increases_surprise() {
        let dim = 1024;
        let mut hierarchy = PredictiveHierarchy::new(dim);

        // Train on one pattern
        let pattern_a = ContinuousHV::random(dim, 1);
        for _ in 0..10 {
            hierarchy.process(&pattern_a);
        }
        let error_familiar = hierarchy.free_energy();

        // Novel observation
        let pattern_b = ContinuousHV::random(dim, 2);
        hierarchy.process(&pattern_b);
        let error_novel = hierarchy.free_energy();

        assert!(
            error_novel > error_familiar,
            "Novel observation should increase surprise: familiar={:.3}, novel={:.3}",
            error_familiar,
            error_novel,
        );
    }

    #[test]
    fn test_hierarchy_levels_independent() {
        let dim = 1024;
        let hierarchy = PredictiveHierarchy::new(dim);
        let errors = hierarchy.errors();
        assert_eq!(errors.len(), 4);
    }

    #[test]
    fn test_reset() {
        let dim = 1024;
        let mut hierarchy = PredictiveHierarchy::new(dim);
        hierarchy.process(&ContinuousHV::random(dim, 42));
        assert!(hierarchy.sensory.state.norm() > 0.0);

        hierarchy.reset();
        assert!(hierarchy.sensory.state.norm() < 1e-6);
    }
}
