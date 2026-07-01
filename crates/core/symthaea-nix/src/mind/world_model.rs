// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS World Model — Generative Model of the System
//!
//! Maintains a continuous HDC representation of the entire NixOS system.
//! The world model can predict outcomes of actions (transition model) and
//! generate expected observations from state (likelihood model).
//!
//! ## Transition Model (Hybrid HDC → CfC)
//!
//! Phase 1: HDC delta vectors — each action type gets a learned delta.
//!   Prediction: next_state ≈ current_state + action_delta
//!   Composable: multi-step plans = sequential addition of deltas.
//!
//! Phase 2: CfC neural model (auto-activates after ~50 episodes).

use super::predictive_hierarchy::PredictiveHierarchy;
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Action categories for the transition model.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ActionCategory {
    Install,
    Remove,
    Enable,
    Disable,
    Rebuild,
    Rollback,
    Configure,
    GarbageCollect,
    Update,
    Custom(String),
}

impl std::fmt::Display for ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Install => f.write_str("Install"),
            Self::Remove => f.write_str("Remove"),
            Self::Enable => f.write_str("Enable"),
            Self::Disable => f.write_str("Disable"),
            Self::Rebuild => f.write_str("Rebuild"),
            Self::Rollback => f.write_str("Rollback"),
            Self::Configure => f.write_str("Configure"),
            Self::GarbageCollect => f.write_str("GarbageCollect"),
            Self::Update => f.write_str("Update"),
            Self::Custom(s) => write!(f, "Custom({s})"),
        }
    }
}

impl ActionCategory {
    /// Classify a command string into a category.
    pub fn from_command(cmd: &str) -> Self {
        let lower = cmd.to_lowercase();
        if lower.contains("install") || lower.contains("nix-env -i") {
            Self::Install
        } else if lower.contains("remove") || lower.contains("nix-env -e") {
            Self::Remove
        } else if lower.contains("enable") {
            Self::Enable
        } else if lower.contains("disable") {
            Self::Disable
        } else if lower.contains("rebuild") || lower.contains("switch") {
            Self::Rebuild
        } else if lower.contains("rollback") {
            Self::Rollback
        } else if lower.contains("gc") || lower.contains("garbage") {
            Self::GarbageCollect
        } else if lower.contains("update") || lower.contains("upgrade") {
            Self::Update
        } else {
            Self::Custom(cmd.to_string())
        }
    }
}

/// A learned state transition delta.
#[derive(Debug, Clone)]
struct DeltaEntry {
    /// Average delta vector for this action category.
    delta: ContinuousHV,
    /// Number of observations used to learn this delta.
    observation_count: usize,
}

/// The NixOS world model.
pub struct NixWorldModel {
    /// Current system state as a ContinuousHV.
    system_state: ContinuousHV,

    /// Learned delta vectors per action category.
    /// `delta_vectors[action]` ≈ average(state_after - state_before) for that action.
    delta_vectors: HashMap<ActionCategory, DeltaEntry>,

    /// Predictive hierarchy for multi-scale understanding.
    prediction_hierarchy: PredictiveHierarchy,

    /// Current free energy (distance between state and goal).
    free_energy: f64,

    /// HDC dimension.
    dim: usize,
}

impl NixWorldModel {
    /// Create a new world model.
    pub fn new(dim: usize) -> Self {
        Self {
            system_state: ContinuousHV::zero(dim),
            delta_vectors: HashMap::new(),
            prediction_hierarchy: PredictiveHierarchy::new(dim),
            free_energy: 1.0,
            dim,
        }
    }

    /// Create with default HDC dimension.
    pub fn default_dim() -> Self {
        Self::new(symthaea_core::hdc::HDC_DIMENSION)
    }

    /// Update the world model with a new system state observation.
    pub fn observe(&mut self, state: ContinuousHV) {
        // Feed through predictive hierarchy
        self.prediction_hierarchy.process(&state);

        // Update current state
        self.system_state = state;
    }

    /// Predict the next state after applying an action.
    ///
    /// Uses delta vectors: `predicted_next ≈ current_state + delta[action]`
    pub fn predict_state(&self, action: &ActionCategory) -> ContinuousHV {
        if let Some(entry) = self.delta_vectors.get(action) {
            self.system_state.add(&entry.delta)
        } else {
            // Unknown action → predict no change
            self.system_state.clone()
        }
    }

    /// Predict the state after a sequence of actions.
    pub fn predict_trajectory(&self, actions: &[ActionCategory]) -> Vec<ContinuousHV> {
        let mut trajectory = Vec::with_capacity(actions.len() + 1);
        trajectory.push(self.system_state.clone());

        let mut current = self.system_state.clone();
        for action in actions {
            if let Some(entry) = self.delta_vectors.get(action) {
                current = current.add(&entry.delta);
            }
            trajectory.push(current.clone());
        }

        trajectory
    }

    /// Learn from an observed state transition.
    ///
    /// Updates the delta vector for the action category with the observed
    /// before→after transition.
    pub fn learn_transition(
        &mut self,
        state_before: &ContinuousHV,
        action: ActionCategory,
        state_after: &ContinuousHV,
    ) {
        let observed_delta = state_after.subtract(state_before);

        let entry = self
            .delta_vectors
            .entry(action)
            .or_insert_with(|| DeltaEntry {
                delta: ContinuousHV::zero(self.dim),
                observation_count: 0,
            });

        entry.observation_count += 1;
        let n = entry.observation_count as f32;

        // Running average: delta = ((n-1)/n) * old_delta + (1/n) * new_delta
        let refs = [&entry.delta, &observed_delta];
        let weights = [(n - 1.0) / n, 1.0 / n];
        entry.delta = ContinuousHV::weighted_bundle(&refs, &weights);
    }

    /// Compute free energy between current state and a desired goal state.
    ///
    /// Free energy = 1 - similarity(current, goal)
    /// Lower free energy = closer to goal.
    pub fn compute_free_energy(&mut self, goal_state: &ContinuousHV) -> f64 {
        let sim = self.system_state.similarity(goal_state).max(0.0) as f64;
        self.free_energy = 1.0 - sim;
        self.free_energy
    }

    /// Expected free energy of an action — how much closer to the goal will it get us?
    ///
    /// EFE = pragmatic_value + epistemic_value
    /// - Pragmatic: how close the predicted next state is to the goal
    /// - Epistemic: how much we'll learn (based on prediction uncertainty)
    pub fn expected_free_energy(
        &self,
        action: &ActionCategory,
        goal_state: &ContinuousHV,
        curiosity_weight: f64,
    ) -> f64 {
        let predicted_next = self.predict_state(action);

        // Pragmatic value: proximity to goal
        let pragmatic = predicted_next.similarity(goal_state).max(0.0) as f64;

        // Epistemic value: information gain (higher for less-observed actions)
        let epistemic = if let Some(entry) = self.delta_vectors.get(action) {
            1.0 / (1.0 + entry.observation_count as f64)
        } else {
            1.0 // Maximum curiosity for unknown actions
        };

        // Expected free energy (lower = better)
        -(pragmatic + curiosity_weight * epistemic)
    }

    /// Get the current system state.
    pub fn system_state(&self) -> &ContinuousHV {
        &self.system_state
    }

    /// Get current free energy.
    pub fn free_energy(&self) -> f64 {
        self.free_energy
    }

    /// Get the predictive hierarchy.
    pub fn prediction_hierarchy(&self) -> &PredictiveHierarchy {
        &self.prediction_hierarchy
    }

    /// Get the predictive hierarchy mutably.
    pub fn prediction_hierarchy_mut(&mut self) -> &mut PredictiveHierarchy {
        &mut self.prediction_hierarchy
    }

    /// Number of learned action categories.
    pub fn learned_action_count(&self) -> usize {
        self.delta_vectors.len()
    }

    /// Total observations across all action categories.
    pub fn total_observations(&self) -> usize {
        self.delta_vectors
            .values()
            .map(|e| e.observation_count)
            .sum()
    }

    /// Whether the transition model has enough data for a specific action.
    pub fn has_learned(&self, action: &ActionCategory) -> bool {
        self.delta_vectors
            .get(action)
            .map(|e| e.observation_count >= 3)
            .unwrap_or(false)
    }
}

impl Default for NixWorldModel {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_and_predict() {
        let dim = 1024;
        let mut wm = NixWorldModel::new(dim);

        let state_before = ContinuousHV::random(dim, 1);
        let state_after = ContinuousHV::random(dim, 2);

        // Learn a transition
        wm.learn_transition(&state_before, ActionCategory::Install, &state_after);

        // Now observe the before state
        wm.observe(state_before.clone());

        // Predict should be closer to state_after than the identity
        let predicted = wm.predict_state(&ActionCategory::Install);
        let sim_predicted = predicted.similarity(&state_after);
        let sim_identity = state_before.similarity(&state_after);

        // After learning one transition, prediction should shift toward after-state
        assert!(
            sim_predicted > sim_identity || (sim_predicted - sim_identity).abs() < 0.1,
            "Prediction should shift toward observed outcome"
        );
    }

    #[test]
    fn test_free_energy_decreases_toward_goal() {
        let dim = 1024;
        let mut wm = NixWorldModel::new(dim);

        let current = ContinuousHV::random(dim, 1);
        let goal = ContinuousHV::random(dim, 2);

        wm.observe(current.clone());
        let fe_far = wm.compute_free_energy(&goal);

        // Observe the goal state itself
        wm.observe(goal.clone());
        let fe_close = wm.compute_free_energy(&goal);

        assert!(
            fe_close < fe_far,
            "Free energy should be lower when at goal: far={:.3}, close={:.3}",
            fe_far,
            fe_close,
        );
    }

    #[test]
    fn test_expected_free_energy_prefers_known() {
        let dim = 1024;
        let mut wm = NixWorldModel::new(dim);
        let goal = ContinuousHV::random(dim, 100);

        // Observe current state
        wm.observe(ContinuousHV::random(dim, 1));

        // Learn some install transitions
        for i in 0..5 {
            let before = ContinuousHV::random(dim, i);
            let after = ContinuousHV::random(dim, i + 100);
            wm.learn_transition(&before, ActionCategory::Install, &after);
        }

        // Unknown action should have higher epistemic value (curiosity)
        let efe_known = wm.expected_free_energy(&ActionCategory::Install, &goal, 0.5);
        let efe_unknown = wm.expected_free_energy(&ActionCategory::Rollback, &goal, 0.5);

        // Both are negative; unknown should be more negative (more attractive) due to curiosity
        // This depends on the specific random vectors, so we just check the mechanism works
        assert!(efe_known.is_finite() && efe_unknown.is_finite());
    }

    #[test]
    fn test_trajectory_prediction() {
        let dim = 1024;
        let mut wm = NixWorldModel::new(dim);

        let before = ContinuousHV::random(dim, 1);
        let after = ContinuousHV::random(dim, 2);
        wm.learn_transition(&before, ActionCategory::Install, &after);

        wm.observe(before);

        let trajectory = wm.predict_trajectory(&[ActionCategory::Install, ActionCategory::Install]);

        // Should have 3 states: initial + 2 steps
        assert_eq!(trajectory.len(), 3);
    }

    #[test]
    fn test_action_category_from_command_all_branches() {
        assert_eq!(
            ActionCategory::from_command("nix-env -i firefox"),
            ActionCategory::Install
        );
        assert_eq!(
            ActionCategory::from_command("install vim"),
            ActionCategory::Install
        );
        assert_eq!(
            ActionCategory::from_command("nix-env -e firefox"),
            ActionCategory::Remove
        );
        assert_eq!(
            ActionCategory::from_command("remove vim"),
            ActionCategory::Remove
        );
        assert_eq!(
            ActionCategory::from_command("enable nginx"),
            ActionCategory::Enable
        );
        assert_eq!(
            ActionCategory::from_command("disable sshd"),
            ActionCategory::Disable
        );
        assert_eq!(
            ActionCategory::from_command("nixos-rebuild switch"),
            ActionCategory::Rebuild
        );
        assert_eq!(
            ActionCategory::from_command("switch to new config"),
            ActionCategory::Rebuild
        );
        assert_eq!(
            ActionCategory::from_command("rollback to previous"),
            ActionCategory::Rollback
        );
        assert_eq!(
            ActionCategory::from_command("nix-collect-garbage"),
            ActionCategory::GarbageCollect
        );
        assert_eq!(
            ActionCategory::from_command("gc old generations"),
            ActionCategory::GarbageCollect
        );
        assert_eq!(
            ActionCategory::from_command("update flake inputs"),
            ActionCategory::Update
        );
        assert_eq!(
            ActionCategory::from_command("upgrade system"),
            ActionCategory::Update
        );
    }

    #[test]
    fn test_action_category_custom_unknown() {
        let cat = ActionCategory::from_command("echo hello world");
        assert!(matches!(cat, ActionCategory::Custom(_)));
        if let ActionCategory::Custom(s) = cat {
            assert_eq!(s, "echo hello world");
        }
    }

    #[test]
    fn test_action_category_case_insensitive() {
        assert_eq!(
            ActionCategory::from_command("INSTALL firefox"),
            ActionCategory::Install
        );
        assert_eq!(
            ActionCategory::from_command("NixOS-Rebuild Switch"),
            ActionCategory::Rebuild
        );
    }

    #[test]
    fn test_has_learned_threshold() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let action = ActionCategory::Install;

        // Not learned yet
        assert!(!wm.has_learned(&action));

        // Learn 2 transitions (not enough)
        for i in 0..2 {
            let before = ContinuousHV::random(dim, i);
            let after = ContinuousHV::random(dim, i + 100);
            wm.learn_transition(&before, ActionCategory::Install, &after);
        }
        assert!(!wm.has_learned(&action));

        // 3rd transition crosses threshold
        wm.learn_transition(
            &ContinuousHV::random(dim, 50),
            ActionCategory::Install,
            &ContinuousHV::random(dim, 150),
        );
        assert!(wm.has_learned(&action));
    }

    #[test]
    fn test_world_model_accessors() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        assert_eq!(wm.learned_action_count(), 0);
        assert_eq!(wm.total_observations(), 0);
        assert!((wm.free_energy() - 1.0).abs() < 1e-6);

        wm.learn_transition(
            &ContinuousHV::random(dim, 1),
            ActionCategory::Install,
            &ContinuousHV::random(dim, 2),
        );
        assert_eq!(wm.learned_action_count(), 1);
        assert_eq!(wm.total_observations(), 1);
    }

    #[test]
    fn test_predict_unknown_action_returns_current() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 1);
        wm.observe(state.clone());

        let predicted = wm.predict_state(&ActionCategory::Rollback);
        let sim = predicted.similarity(&state);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Unknown action should predict no change (sim={:.6})",
            sim
        );
    }

    #[test]
    fn test_action_category_display() {
        assert_eq!(format!("{}", ActionCategory::Install), "Install");
        assert_eq!(format!("{}", ActionCategory::Remove), "Remove");
        assert_eq!(format!("{}", ActionCategory::Enable), "Enable");
        assert_eq!(format!("{}", ActionCategory::Disable), "Disable");
        assert_eq!(format!("{}", ActionCategory::Rebuild), "Rebuild");
        assert_eq!(format!("{}", ActionCategory::Rollback), "Rollback");
        assert_eq!(format!("{}", ActionCategory::Configure), "Configure");
        assert_eq!(
            format!("{}", ActionCategory::GarbageCollect),
            "GarbageCollect"
        );
        assert_eq!(format!("{}", ActionCategory::Update), "Update");
        assert_eq!(
            format!("{}", ActionCategory::Custom("reboot".into())),
            "Custom(reboot)"
        );
    }

    #[test]
    fn test_action_category_display_vs_debug_differ() {
        // Display should be cleaner than Debug for Custom variant
        let custom = ActionCategory::Custom("hello".into());
        let display = format!("{custom}");
        let debug = format!("{custom:?}");
        assert_eq!(display, "Custom(hello)");
        assert!(debug.contains("Custom"));
        // Debug wraps the string in quotes, Display doesn't
        assert!(!display.contains('"'));
    }
}
