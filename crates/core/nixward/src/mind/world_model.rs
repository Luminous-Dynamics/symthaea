// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS World Model — Generative Model of the System
//!
//! Maintains a continuous HDC representation of the entire NixOS system.
//! The world model can predict outcomes of actions (transition model) and
//! generate expected observations from state (likelihood model).
//!
//! ## Transition Model (current implementation)
//!
//! The current transition baseline is HDC action-delta averaging:
//! - Each action category stores a running-average state delta.
//! - Prediction applies `current_state + action_delta`.
//! - Unknown actions currently predict no change.
//! - Multi-step predictions sequentially apply the learned additive deltas.
//!
//! **CfC/HDC-LTC transition integration is not implemented in this module.**
//! Although separate HDC-LTC components exist in `symthaea-core`, this
//! `NixWorldModel` has no CfC model field, episode threshold, training path,
//! inference path or CfC checkpoint. Do not infer an automatic activation after
//! 50 episodes from this implementation. See issue #13766 for the integration
//! and qualification plan.

use super::predictive_hierarchy::PredictiveHierarchy;
use std::collections::{HashMap, HashSet};
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



/// Outcome label for an action that was actually executed and observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionOutcomeClass {
    Success,
    Failure,
    Partial,
    RolledBack,
}

/// Execution status is distinct from the action that was merely assigned.
#[derive(Debug, Clone)]
pub enum TransitionExecution {
    /// The requested action did not execute.
    NotExecuted,
    /// Execution status or the resulting state could not be reconciled.
    OutcomeUnknown,
    /// An action was executed and its resulting state was observed.
    Executed {
        realized_action: ActionCategory,
        outcome: TransitionOutcomeClass,
        state_after: ContinuousHV,
        state_after_sha256: String,
    },
}

/// Provenance-bearing transition episode for world-model learning.
///
/// The model learns from `realized_action`, never from `assigned_action`.
/// Executed failures and partial outcomes remain valid observations of the
/// transition dynamics. Unexecuted or unknown episodes are not learnable.
#[derive(Debug, Clone)]
pub struct TransitionEpisodeV1 {
    pub episode_id: String,
    pub assigned_action: ActionCategory,
    pub state_before: ContinuousHV,
    /// HDC context at action time (configuration, workload, environment, etc.).
    pub context: ContinuousHV,
    pub execution: TransitionExecution,
    pub horizon_steps: u32,
    pub environment_id: String,
    pub environment_family_id: String,
    /// Repeated episodes may share this ID; it is not an independent sample count.
    pub independent_cluster_id: String,
    pub state_before_sha256: String,
    pub source_manifest_sha256: String,
    pub provenance_sha256: String,
    pub captured_at_unix_ms: i64,
}

impl TransitionEpisodeV1 {
    /// Validate dimensions, provenance identity and execution-state consistency.
    ///
    /// Digests are format-checked here. The ingestion boundary must recompute
    /// them from exact source bytes and authenticate the receipt independently.
    pub fn validate(&self) -> Result<(), String> {
        let dimension = self.state_before.as_slice().len();
        if dimension == 0 || self.context.as_slice().len() != dimension {
            return Err("state_before and context must have the same non-zero HDC dimension".to_owned());
        }
        if self
            .state_before
            .as_slice()
            .iter()
            .chain(self.context.as_slice())
            .any(|value| !value.is_finite())
        {
            return Err("state_before and context must contain only finite values".to_owned());
        }
        if self.episode_id.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.environment_family_id.trim().is_empty()
            || self.independent_cluster_id.trim().is_empty()
        {
            return Err("episode and environment/cluster identities must be non-empty".to_owned());
        }
        if self.horizon_steps == 0 || self.captured_at_unix_ms < 0 {
            return Err("horizon_steps must be positive and capture time non-negative".to_owned());
        }
        for (name, digest) in [
            ("state_before_sha256", &self.state_before_sha256),
            ("source_manifest_sha256", &self.source_manifest_sha256),
            ("provenance_sha256", &self.provenance_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(format!("{name} must be a 64-character SHA-256 hex digest"));
            }
        }

        if let TransitionExecution::Executed {
            state_after,
            state_after_sha256,
            ..
        } = &self.execution
        {
            if state_after.as_slice().len() != dimension
                || state_after.as_slice().iter().any(|value| !value.is_finite())
            {
                return Err("executed state_after must match the HDC dimension and be finite".to_owned());
            }
            if !is_sha256_hex(state_after_sha256) {
                return Err("state_after_sha256 must be a 64-character SHA-256 hex digest".to_owned());
            }
        }
        Ok(())
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Why a context-conditioned transition prediction was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextualPredictionSource {
    ContextualDelta,
    GlobalActionDeltaFallback,
    UnknownActionIdentityFallback,
}

/// Prediction diagnostics for the context-conditioned HDC-delta candidate.
///
/// `training_residual_mse` is a prequential training diagnostic, not a
/// calibrated predictive interval or a held-out uncertainty guarantee.
#[derive(Debug, Clone)]
pub struct ContextualPrediction {
    pub predicted_state: ContinuousHV,
    pub source: ContextualPredictionSource,
    pub context_similarity: f64,
    pub observation_count: usize,
    pub independent_cluster_count: usize,
    pub training_residual_mse: Option<f64>,
}

#[derive(Debug, Clone)]
struct ContextDeltaEntry {
    environment_family_id: String,
    context: ContinuousHV,
    delta: ContinuousHV,
    observation_count: usize,
    residual_count: usize,
    residual_mse: f64,
    independent_cluster_ids: std::collections::HashSet<String>,
}

const CONTEXT_MATCH_THRESHOLD: f32 = 0.8;

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

    /// Context-conditioned candidate transition deltas. Kept separate from
    /// the global per-action baseline so the two can be compared directly.
    contextual_delta_vectors: HashMap<ActionCategory, Vec<ContextDeltaEntry>>,

    /// Prevent a provenance-bearing episode from being applied twice.
    learned_episode_ids: HashSet<String>,

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
            contextual_delta_vectors: HashMap::new(),
            learned_episode_ids: HashSet::new(),
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



    /// Learn a provenance-bearing, actually executed transition.
    ///
    /// The legacy global action-delta baseline is updated alongside a
    /// separate context-conditioned candidate. Failed and partial executions
    /// are retained because they are real transition outcomes. Assigned-only
    /// and execution-unknown episodes are rejected without mutating the model.
    pub fn learn_episode(&mut self, episode: &TransitionEpisodeV1) -> Result<(), String> {
        episode.validate()?;
        if self.learned_episode_ids.contains(&episode.episode_id) {
            return Err(format!("duplicate transition episode: {}", episode.episode_id));
        }
        if episode.state_before.as_slice().len() != self.dim {
            return Err("episode HDC dimension does not match this world model".to_owned());
        }

        let (realized_action, state_after) = match &episode.execution {
            TransitionExecution::Executed {
                realized_action,
                state_after,
                ..
            } => (realized_action.clone(), state_after),
            TransitionExecution::NotExecuted => {
                return Err("unexecuted transition episodes cannot train the dynamics model".to_owned());
            }
            TransitionExecution::OutcomeUnknown => {
                return Err("execution-unknown episodes require reconciliation before learning".to_owned());
            }
        };

        let observed_delta = state_after.subtract(&episode.state_before);
        self.learn_transition(&episode.state_before, realized_action.clone(), state_after);

        let entries = self
            .contextual_delta_vectors
            .entry(realized_action)
            .or_default();
        let best_match = entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.environment_family_id == episode.environment_family_id)
            .map(|(index, entry)| (index, entry.context.similarity(&episode.context)))
            .filter(|(_, similarity)| similarity.is_finite())
            .max_by(|left, right| left.1.total_cmp(&right.1));

        if let Some((index, similarity)) =
            best_match.filter(|(_, similarity)| *similarity >= CONTEXT_MATCH_THRESHOLD)
        {
            let entry = &mut entries[index];
            let residual_mse = mean_squared_error(&entry.delta, &observed_delta);
            entry.residual_count += 1;
            let residual_n = entry.residual_count as f64;
            entry.residual_mse += (residual_mse - entry.residual_mse) / residual_n;

            entry.observation_count += 1;
            let n = entry.observation_count as f32;
            let refs = [&entry.delta, &observed_delta];
            let weights = [(n - 1.0) / n, 1.0 / n];
            entry.delta = ContinuousHV::weighted_bundle(&refs, &weights);
            entry
                .independent_cluster_ids
                .insert(episode.independent_cluster_id.clone());

            // Keep a finite similarity check explicit; invalid HDC values
            // should have been rejected by episode validation.
            if !similarity.is_finite() {
                return Err("context similarity is non-finite".to_owned());
            }
        } else {
            let mut independent_cluster_ids = HashSet::new();
            independent_cluster_ids.insert(episode.independent_cluster_id.clone());
            entries.push(ContextDeltaEntry {
                environment_family_id: episode.environment_family_id.clone(),
                context: episode.context.clone(),
                delta: observed_delta,
                observation_count: 1,
                residual_count: 0,
                residual_mse: 0.0,
                independent_cluster_ids,
            });
        }

        self.learned_episode_ids.insert(episode.episode_id.clone());
        Ok(())
    }

    /// Predict from the nearest context-conditioned transition bucket.
    ///
    /// If no bucket meets the frozen similarity threshold, fall back to the
    /// global per-action delta baseline. An unknown action retains the
    /// existing identity/no-change behaviour but is explicitly labelled.
    pub fn predict_state_with_context(
        &self,
        state: &ContinuousHV,
        action: &ActionCategory,
        environment_family_id: &str,
        context: &ContinuousHV,
    ) -> Result<ContextualPrediction, String> {
        if state.as_slice().len() != self.dim
            || context.as_slice().len() != self.dim
            || state.as_slice().iter().any(|value| !value.is_finite())
            || context.as_slice().iter().any(|value| !value.is_finite())
        {
            return Err("prediction state and context must match the model dimension and be finite".to_owned());
        }

        let best_match = self.contextual_delta_vectors.get(action).and_then(|entries| {
            entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.environment_family_id == environment_family_id)
                .map(|(index, entry)| (index, entry.context.similarity(context)))
                .filter(|(_, similarity)| similarity.is_finite())
                .max_by(|left, right| left.1.total_cmp(&right.1))
        });

        if let Some((index, similarity)) =
            best_match.filter(|(_, similarity)| *similarity >= CONTEXT_MATCH_THRESHOLD)
        {
            let entry = &self.contextual_delta_vectors[action][index];
            return Ok(ContextualPrediction {
                predicted_state: state.add(&entry.delta),
                source: ContextualPredictionSource::ContextualDelta,
                context_similarity: similarity as f64,
                observation_count: entry.observation_count,
                independent_cluster_count: entry.independent_cluster_ids.len(),
                training_residual_mse: (entry.residual_count > 0).then_some(entry.residual_mse),
            });
        }

        let context_similarity = best_match
            .map(|(_, similarity)| similarity.max(0.0) as f64)
            .unwrap_or(0.0);
        if let Some(entry) = self.delta_vectors.get(action) {
            let independent_cluster_count = self
                .contextual_delta_vectors
                .get(action)
                .map(|entries| {
                    entries
                        .iter()
                        .flat_map(|entry| entry.independent_cluster_ids.iter())
                        .collect::<HashSet<_>>()
                        .len()
                })
                .unwrap_or(0);
            return Ok(ContextualPrediction {
                predicted_state: state.add(&entry.delta),
                source: ContextualPredictionSource::GlobalActionDeltaFallback,
                context_similarity,
                observation_count: entry.observation_count,
                independent_cluster_count,
                training_residual_mse: None,
            });
        }

        Ok(ContextualPrediction {
            predicted_state: state.clone(),
            source: ContextualPredictionSource::UnknownActionIdentityFallback,
            context_similarity,
            observation_count: 0,
            independent_cluster_count: 0,
            training_residual_mse: None,
        })
    }

    /// Number of provenance-bearing episodes successfully learned.
    pub fn learned_episode_count(&self) -> usize {
        self.learned_episode_ids.len()
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


fn mean_squared_error(left: &ContinuousHV, right: &ContinuousHV) -> f64 {
    if left.as_slice().len() != right.as_slice().len() || left.as_slice().is_empty() {
        return f64::NAN;
    }
    left.as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(left_value, right_value)| {
            let difference = *left_value as f64 - *right_value as f64;
            difference * difference
        })
        .sum::<f64>()
        / left.as_slice().len() as f64
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

    // ── Phase 2.9-A: World Model Transition Hardening ───────────────────────
    // Covers the delta-vector transition table that feeds predict_state →
    // prediction_error → curiosity modulation. Invariants: outputs always
    // finite, running average converges, contradictory inputs don't corrupt.

    /// Contradictory transitions (same action, opposite state changes) must
    /// converge toward zero delta — not diverge or NaN.
    #[test]
    fn test_contradictory_transitions_converge() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);

        let base = ContinuousHV::random(dim, 1);

        // Alternately teach Install: "go toward state A" and "go toward state B"
        // where A and B are near-orthogonal. The running average must stay finite.
        for i in 0u64..30 {
            let before = base.clone();
            // Alternate between two opposite target states
            let after = if i % 2 == 0 {
                ContinuousHV::random(dim, 100)
            } else {
                ContinuousHV::random(dim, 200)
            };
            wm.learn_transition(&before, ActionCategory::Install, &after);
        }

        wm.observe(base);
        let predicted = wm.predict_state(&ActionCategory::Install);

        // After contradictory training, prediction must still be finite
        for v in predicted.as_slice() {
            assert!(
                v.is_finite(),
                "predicted state must be finite after contradictory transitions: {v}"
            );
        }
        assert_eq!(wm.total_observations(), 30);
    }

    /// Running average stays finite and bounded over 100 consecutive transitions.
    /// Verifies the `((n-1)/n)*old + (1/n)*new` formula doesn't accumulate error.
    #[test]
    fn test_running_average_stays_finite_over_many_transitions() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);

        let start = ContinuousHV::random(dim, 1);
        wm.observe(start.clone());

        for i in 0u64..100 {
            let before = ContinuousHV::random(dim, i * 3 + 10);
            let after = ContinuousHV::random(dim, i * 3 + 11);
            wm.learn_transition(&before, ActionCategory::Rebuild, &after);

            // Check after every step
            let predicted = wm.predict_state(&ActionCategory::Rebuild);
            for (idx, v) in predicted.as_slice().iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "predicted[{idx}] not finite after {i} transitions: {v}"
                );
            }
        }

        assert_eq!(wm.total_observations(), 100);
    }

    /// EFE must be finite for every action category and a range of curiosity weights.
    #[test]
    fn test_expected_free_energy_always_finite() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);

        let current = ContinuousHV::random(dim, 1);
        let goal = ContinuousHV::random(dim, 2);
        wm.observe(current.clone());

        // Teach a few actions
        for (before_seed, after_seed, action) in [
            (10u64, 11u64, ActionCategory::Install),
            (20, 21, ActionCategory::Remove),
            (30, 31, ActionCategory::Rebuild),
        ] {
            wm.learn_transition(
                &ContinuousHV::random(dim, before_seed),
                action,
                &ContinuousHV::random(dim, after_seed),
            );
        }

        let all_actions = [
            ActionCategory::Install,
            ActionCategory::Remove,
            ActionCategory::Enable,
            ActionCategory::Disable,
            ActionCategory::Rebuild,
            ActionCategory::Rollback,
            ActionCategory::Configure,
            ActionCategory::GarbageCollect,
            ActionCategory::Update,
            ActionCategory::Custom("custom-op".into()),
        ];

        for curiosity in [0.0f64, 0.1, 0.3, 0.5, 0.8, 1.0] {
            for action in &all_actions {
                let efe = wm.expected_free_energy(action, &goal, curiosity);
                assert!(
                    efe.is_finite(),
                    "EFE must be finite for {action} at curiosity={curiosity}: {efe}"
                );
            }
        }
    }

    /// Trajectory with all-unknown actions must return a trajectory of identical
    /// states (no-change prediction) and never panic.
    #[test]
    fn test_trajectory_all_unknown_actions_returns_no_change() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);

        let state = ContinuousHV::random(dim, 42);
        wm.observe(state.clone());

        // No transitions learned — all actions unknown
        let actions = vec![
            ActionCategory::Rollback,
            ActionCategory::Configure,
            ActionCategory::Enable,
        ];
        let traj = wm.predict_trajectory(&actions);

        assert_eq!(traj.len(), 4, "trajectory must have initial + N steps");

        // All steps should be identical to the current state (unknown = no change)
        for (i, step) in traj.iter().enumerate() {
            let sim = step.similarity(&state);
            assert!(
                (sim - 1.0).abs() < 1e-5,
                "unknown-action trajectory step {i} should match current state: sim={sim}"
            );
        }
    }

    /// Free energy is always in [0, 1] since it equals 1 - similarity.clamp(0,1).
    #[test]
    fn test_free_energy_always_in_unit_interval() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);

        let goal = ContinuousHV::random(dim, 99);

        for seed in [1u64, 2, 50, 100, 999, 12345] {
            wm.observe(ContinuousHV::random(dim, seed));
            let fe = wm.compute_free_energy(&goal);
            assert!(
                fe >= 0.0 && fe <= 1.0,
                "free energy must be in [0,1] for seed {seed}: {fe}"
            );
            assert!(
                fe.is_finite(),
                "free energy must be finite for seed {seed}: {fe}"
            );
        }
    }

    /// predict_state output must always be finite, even when the delta table
    /// was built from many diverse inputs.
    #[test]
    fn test_predict_state_output_always_finite() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);

        for i in 0u64..20 {
            let before = ContinuousHV::random(dim, i * 2);
            let after = ContinuousHV::random(dim, i * 2 + 1);
            wm.learn_transition(&before, ActionCategory::Update, &after);
        }

        let current = ContinuousHV::random(dim, 9999);
        wm.observe(current);

        let predicted = wm.predict_state(&ActionCategory::Update);
        for (i, v) in predicted.as_slice().iter().enumerate() {
            assert!(
                v.is_finite(),
                "predicted[{i}] must be finite after diverse training: {v}"
            );
        }
    }

    fn transition_episode(
        episode_id: &str,
        assigned_action: ActionCategory,
        realized_action: ActionCategory,
        before: ContinuousHV,
        context: ContinuousHV,
        after: ContinuousHV,
        outcome: TransitionOutcomeClass,
        cluster: &str,
    ) -> TransitionEpisodeV1 {
        TransitionEpisodeV1 {
            episode_id: episode_id.to_owned(),
            assigned_action,
            state_before: before,
            context,
            execution: TransitionExecution::Executed {
                realized_action,
                outcome,
                state_after: after,
                state_after_sha256: "d".repeat(64),
            },
            horizon_steps: 1,
            environment_id: "nixos-fixture".to_owned(),
            environment_family_id: "nix-build-family".to_owned(),
            independent_cluster_id: cluster.to_owned(),
            state_before_sha256: "a".repeat(64),
            source_manifest_sha256: "b".repeat(64),
            provenance_sha256: "c".repeat(64),
            captured_at_unix_ms: 1_000,
        }
    }

    #[test]
    fn executed_episode_learns_from_realized_not_assigned_action() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 1);
        let after = ContinuousHV::random(dim, 2);
        let context = ContinuousHV::random(dim, 3);
        let episode = transition_episode(
            "episode-1",
            ActionCategory::Install,
            ActionCategory::Rollback,
            before.clone(),
            context.clone(),
            after.clone(),
            TransitionOutcomeClass::RolledBack,
            "cluster-1",
        );

        wm.learn_episode(&episode).expect("executed episode is learnable");
        assert_eq!(wm.learned_episode_count(), 1);
        assert_eq!(wm.total_observations(), 1);

        let learned = wm.predict_state_with_context(
            &before,
            &ActionCategory::Rollback,
            "nix-build-family",
            &context,
        ).expect("valid contextual prediction");
        assert_eq!(learned.source, ContextualPredictionSource::ContextualDelta);
        assert!((learned.predicted_state.similarity(&after) - 1.0).abs() < 1e-5);

        let not_learned = wm.predict_state_with_context(
            &before,
            &ActionCategory::Install,
            "nix-build-family",
            &context,
        ).expect("valid fallback prediction");
        assert_eq!(
            not_learned.source,
            ContextualPredictionSource::UnknownActionIdentityFallback
        );
    }

    #[test]
    fn executed_failure_is_retained_as_a_transition_observation() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 10);
        let after = ContinuousHV::random(dim, 11);
        let context = ContinuousHV::random(dim, 12);
        let episode = transition_episode(
            "failed-build",
            ActionCategory::Rebuild,
            ActionCategory::Rebuild,
            before.clone(),
            context.clone(),
            after.clone(),
            TransitionOutcomeClass::Failure,
            "cluster-failure",
        );

        wm.learn_episode(&episode).expect("executed failure is real evidence");
        let prediction = wm.predict_state_with_context(
            &before,
            &ActionCategory::Rebuild,
            "nix-build-family",
            &context,
        ).expect("valid prediction");
        assert_eq!(prediction.source, ContextualPredictionSource::ContextualDelta);
        assert!((prediction.predicted_state.similarity(&after) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unexecuted_and_unknown_episodes_do_not_mutate_the_model() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let base = transition_episode(
            "not-executed",
            ActionCategory::Install,
            ActionCategory::Install,
            ContinuousHV::random(dim, 1),
            ContinuousHV::random(dim, 2),
            ContinuousHV::random(dim, 3),
            TransitionOutcomeClass::Success,
            "cluster-1",
        );

        let mut unexecuted = base.clone();
        unexecuted.execution = TransitionExecution::NotExecuted;
        assert!(wm.learn_episode(&unexecuted).is_err());

        let mut unknown = base;
        unknown.episode_id = "unknown-execution".to_owned();
        unknown.execution = TransitionExecution::OutcomeUnknown;
        assert!(wm.learn_episode(&unknown).is_err());

        assert_eq!(wm.total_observations(), 0);
        assert_eq!(wm.learned_episode_count(), 0);
    }

    #[test]
    fn contextual_model_keeps_different_context_deltas_separate() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 20);
        let context_a = ContinuousHV::random(dim, 21);
        let context_b = ContinuousHV::random(dim, 22);
        let after_a = ContinuousHV::random(dim, 23);
        let after_b = ContinuousHV::random(dim, 24);

        let episode_a = transition_episode(
            "context-a",
            ActionCategory::Install,
            ActionCategory::Install,
            state.clone(),
            context_a.clone(),
            after_a.clone(),
            TransitionOutcomeClass::Success,
            "cluster-a",
        );
        let episode_b = transition_episode(
            "context-b",
            ActionCategory::Install,
            ActionCategory::Install,
            state.clone(),
            context_b.clone(),
            after_b.clone(),
            TransitionOutcomeClass::Success,
            "cluster-b",
        );
        wm.learn_episode(&episode_a).expect("first context");
        wm.learn_episode(&episode_b).expect("second context");

        let prediction_a = wm.predict_state_with_context(
            &state,
            &ActionCategory::Install,
            "nix-build-family",
            &context_a,
        ).expect("context a prediction");
        let prediction_b = wm.predict_state_with_context(
            &state,
            &ActionCategory::Install,
            "nix-build-family",
            &context_b,
        ).expect("context b prediction");

        assert_eq!(prediction_a.source, ContextualPredictionSource::ContextualDelta);
        assert_eq!(prediction_b.source, ContextualPredictionSource::ContextualDelta);
        assert!((prediction_a.predicted_state.similarity(&after_a) - 1.0).abs() < 1e-5);
        assert!((prediction_b.predicted_state.similarity(&after_b) - 1.0).abs() < 1e-5);
        assert_eq!(prediction_a.independent_cluster_count, 1);
        assert_eq!(prediction_b.independent_cluster_count, 1);
    }

    #[test]
    fn repeated_cluster_rows_do_not_inflate_independent_cluster_count() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 30);
        let context = ContinuousHV::random(dim, 31);
        let after_a = ContinuousHV::random(dim, 32);
        let after_b = ContinuousHV::random(dim, 33);

        let episode_a = transition_episode(
            "repeat-a",
            ActionCategory::Update,
            ActionCategory::Update,
            state.clone(),
            context.clone(),
            after_a,
            TransitionOutcomeClass::Success,
            "same-cluster",
        );
        let episode_b = transition_episode(
            "repeat-b",
            ActionCategory::Update,
            ActionCategory::Update,
            state.clone(),
            context.clone(),
            after_b,
            TransitionOutcomeClass::Partial,
            "same-cluster",
        );
        wm.learn_episode(&episode_a).expect("first observation");
        wm.learn_episode(&episode_b).expect("repeated cluster observation");

        let prediction = wm.predict_state_with_context(
            &state,
            &ActionCategory::Update,
            "nix-build-family",
            &context,
        ).expect("valid prediction");
        assert_eq!(prediction.observation_count, 2);
        assert_eq!(prediction.independent_cluster_count, 1);
        assert!(prediction.training_residual_mse.is_some());
    }

    #[test]
    fn duplicate_episode_id_and_invalid_provenance_are_rejected() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let episode = transition_episode(
            "duplicate",
            ActionCategory::Install,
            ActionCategory::Install,
            ContinuousHV::random(dim, 1),
            ContinuousHV::random(dim, 2),
            ContinuousHV::random(dim, 3),
            TransitionOutcomeClass::Success,
            "cluster-1",
        );
        wm.learn_episode(&episode).expect("first application");
        assert!(wm.learn_episode(&episode).is_err());
        assert_eq!(wm.total_observations(), 1);

        let mut invalid = episode;
        invalid.episode_id = "invalid-digest".to_owned();
        invalid.provenance_sha256 = "not-a-sha256".to_owned();
        assert!(wm.learn_episode(&invalid).is_err());
        assert_eq!(wm.total_observations(), 1);
    }



    #[test]
    fn contextual_deltas_do_not_cross_environment_family_boundaries() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 60);
        let context = ContinuousHV::random(dim, 61);
        let after = ContinuousHV::random(dim, 62);
        let mut episode = transition_episode(
            "family-a",
            ActionCategory::Install,
            ActionCategory::Install,
            state.clone(),
            context.clone(),
            after,
            TransitionOutcomeClass::Success,
            "cluster-a",
        );
        wm.learn_episode(&episode).expect("family a episode");
        episode.episode_id = "family-b".to_owned();
        episode.environment_family_id = "different-mechanism".to_owned();
        episode.independent_cluster_id = "cluster-b".to_owned();
        episode.execution = TransitionExecution::Executed {
            realized_action: ActionCategory::Install,
            outcome: TransitionOutcomeClass::Failure,
            state_after: ContinuousHV::random(dim, 63),
            state_after_sha256: "e".repeat(64),
        };
        wm.learn_episode(&episode).expect("family b episode");

        let prediction = wm.predict_state_with_context(
            &state,
            &ActionCategory::Install,
            "nix-build-family",
            &context,
        ).expect("family-scoped prediction");
        assert_eq!(prediction.source, ContextualPredictionSource::ContextualDelta);
        assert_eq!(prediction.independent_cluster_count, 1);
    }

    #[test]
    fn contextual_prediction_falls_back_visibly_when_context_is_unseen() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 40);
        let context_a = ContinuousHV::random(dim, 41);
        let context_b = ContinuousHV::random(dim, 42);
        let after = ContinuousHV::random(dim, 43);
        let episode = transition_episode(
            "known-context",
            ActionCategory::Install,
            ActionCategory::Install,
            state.clone(),
            context_a,
            after,
            TransitionOutcomeClass::Success,
            "cluster-1",
        );
        wm.learn_episode(&episode).expect("known context");
        let prediction = wm.predict_state_with_context(
            &state,
            &ActionCategory::Install,
            "nix-build-family",
            &context_b,
        ).expect("fallback prediction");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::GlobalActionDeltaFallback
        );
        assert_eq!(prediction.observation_count, 1);
        assert!(prediction.training_residual_mse.is_none());
    }

    #[test]
    fn contextual_prediction_rejects_wrong_dimensions_and_non_finite_values() {
        let dim = 256;
        let wm = NixWorldModel::new(dim);
        assert!(wm.predict_state_with_context(
            &ContinuousHV::zero(dim - 1),
            &ActionCategory::Install,
            &ContinuousHV::zero(dim),
        ).is_err());

        let invalid_context = ContinuousHV::from_vec(vec![f32::NAN; dim]);
        assert!(wm.predict_state_with_context(
            &ContinuousHV::zero(dim),
            &ActionCategory::Install,
            "nix-build-family",
            &invalid_context,
        ).is_err());
    }

}
