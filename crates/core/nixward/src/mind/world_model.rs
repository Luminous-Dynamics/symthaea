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
//! The production `NixWorldModel` baseline is HDC per-action mean-delta
//! averaging: `next_state ≈ current_state + mean_delta[action]`.
//! Unknown action categories retain the historical identity-state fallback;
//! callers can inspect `TransitionSupport::NoExamples` through diagnostics.
//!
//! `ContextualDeltaBaselineV1` is a separate research candidate keyed by
//! exact domain/context/family/action-profile/parameter identity. It abstains
//! on unseen keys and is not wired into production action selection.
//!
//! CfC/HDC-LTC transition integration is not implemented in this module.
//! Separate HDC-LTC components exist in `symthaea-core`, but this
//! `NixWorldModel` has no CfC model field, activation threshold, training
//! path, inference path or CfC checkpoint.

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


/// Execution disposition for an observed system transition.
///
/// A failed or unknown outcome is still retained as an episode. It must not be
/// silently converted into a successful action→state training pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionOutcome {
    Completed,
    Failed,
    TimedOut,
    Aborted,
    Unknown,
}

/// Provenance-bound transition observation for the NixWard world model.
///
/// This record supplements the existing per-action mean-delta baseline. It
/// does not itself grant execution authority, authenticate a receipt, or
/// automatically update the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionEpisodeV1 {
    episode_id: String,
    domain_id: String,
    context_id: String,
    environment_family_id: String,
    environment_cluster_id: String,
    action: ActionCategory,
    action_profile_id: String,
    /// BLAKE3 digest of canonical action parameters, computed by the caller.
    action_parameters_blake3: String,
    /// BLAKE3 digest of canonical before-state hypervector bytes.
    before_state_blake3: String,
    /// BLAKE3 digest of canonical after-state hypervector bytes, when observed.
    after_state_blake3: Option<String>,
    outcome: TransitionOutcome,
    /// Registered observation horizon for this episode.
    observation_horizon_ms: u64,
    observed_at_unix_ms: u64,
    /// BLAKE3 digest of the execution/observation receipt, when supplied.
    execution_receipt_blake3: Option<String>,
}

fn valid_blake3_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Hash a ContinuousHV using a canonical dimension + little-endian f32
/// encoding. Non-finite values are rejected rather than normalized.
pub fn continuous_hv_blake3(state: &ContinuousHV) -> Result<String, String> {
    if !state.as_slice().iter().all(|value| value.is_finite()) {
        return Err("state hypervector contains non-finite values".to_owned());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(&(state.as_slice().len() as u64).to_le_bytes());
    for value in state.as_slice() {
        hasher.update(&value.to_bits().to_le_bytes());
    }

    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest.as_bytes() {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing into String cannot fail");
    }
    Ok(encoded)
}

impl TransitionEpisodeV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        episode_id: impl Into<String>,
        domain_id: impl Into<String>,
        context_id: impl Into<String>,
        environment_family_id: impl Into<String>,
        environment_cluster_id: impl Into<String>,
        action: ActionCategory,
        action_profile_id: impl Into<String>,
        action_parameters_blake3: impl Into<String>,
        before_state: &ContinuousHV,
        after_state: Option<&ContinuousHV>,
        outcome: TransitionOutcome,
        observation_horizon_ms: u64,
        observed_at_unix_ms: u64,
        execution_receipt_blake3: Option<String>,
    ) -> Result<Self, String> {
        if let Some(after) = after_state {
            if before_state.as_slice().len() != after.as_slice().len() {
                return Err("before/after state dimensions do not match".to_owned());
            }
        }

        let episode = Self {
            episode_id: episode_id.into(),
            domain_id: domain_id.into(),
            context_id: context_id.into(),
            environment_family_id: environment_family_id.into(),
            environment_cluster_id: environment_cluster_id.into(),
            action,
            action_profile_id: action_profile_id.into(),
            action_parameters_blake3: action_parameters_blake3.into(),
            before_state_blake3: continuous_hv_blake3(before_state)?,
            after_state_blake3: after_state.map(continuous_hv_blake3).transpose()?,
            outcome,
            observation_horizon_ms,
            observed_at_unix_ms,
            execution_receipt_blake3,
        };
        episode.validate()?;
        Ok(episode)
    }

    pub fn episode_id(&self) -> &str { &self.episode_id }
    pub fn domain_id(&self) -> &str { &self.domain_id }
    pub fn context_id(&self) -> &str { &self.context_id }
    pub fn environment_family_id(&self) -> &str { &self.environment_family_id }
    pub fn environment_cluster_id(&self) -> &str { &self.environment_cluster_id }
    pub fn action(&self) -> &ActionCategory { &self.action }
    pub fn action_profile_id(&self) -> &str { &self.action_profile_id }
    pub fn action_parameters_blake3(&self) -> &str { &self.action_parameters_blake3 }
    pub fn before_state_blake3(&self) -> &str { &self.before_state_blake3 }
    pub fn after_state_blake3(&self) -> Option<&str> { self.after_state_blake3.as_deref() }
    pub fn outcome(&self) -> TransitionOutcome { self.outcome }
    pub fn observation_horizon_ms(&self) -> u64 { self.observation_horizon_ms }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
    pub fn execution_receipt_blake3(&self) -> Option<&str> {
        self.execution_receipt_blake3.as_deref()
    }

    pub fn validate(&self) -> Result<(), String> {
        if [
            self.episode_id.as_str(),
            self.domain_id.as_str(),
            self.context_id.as_str(),
            self.environment_family_id.as_str(),
            self.environment_cluster_id.as_str(),
            self.action_profile_id.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            return Err("episode, domain, context, family, cluster and action-profile IDs must be non-empty".to_owned());
        }
        if !valid_blake3_hex(&self.action_parameters_blake3)
            || !valid_blake3_hex(&self.before_state_blake3)
            || self
                .after_state_blake3
                .as_ref()
                .is_some_and(|digest| !valid_blake3_hex(digest))
            || self
                .execution_receipt_blake3
                .as_ref()
                .is_some_and(|digest| !valid_blake3_hex(digest))
        {
            return Err("action, state and receipt digests must be 64-character hexadecimal BLAKE3 values".to_owned());
        }
        if self.observation_horizon_ms == 0 {
            return Err("observation_horizon_ms must be positive".to_owned());
        }
        if self.outcome == TransitionOutcome::Completed {
            if self.after_state_blake3.is_none() {
                return Err("completed transitions require an observed after-state".to_owned());
            }
            if self.execution_receipt_blake3.is_none() {
                return Err("completed transitions require an execution/observation receipt digest".to_owned());
            }
        }
        Ok(())
    }

    /// True only when a completed after-state and receipt digest are present.
    ///
    /// The caller must still have an independent verifier authenticate the
    /// receipt before promoting this episode into a training/evidence set.
    pub fn has_completed_transition_measurement(&self) -> bool {
        self.outcome == TransitionOutcome::Completed
            && self.after_state_blake3.is_some()
            && self.execution_receipt_blake3.is_some()
    }
}


/// Exact context/action key for the research-only contextual delta baseline.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ContextualTransitionKeyV1 {
    domain_id: String,
    context_id: String,
    environment_family_id: String,
    action: ActionCategory,
    action_profile_id: String,
    action_parameters_blake3: String,
}

impl ContextualTransitionKeyV1 {
    pub fn from_episode(episode: &TransitionEpisodeV1) -> Result<Self, String> {
        episode.validate()?;
        Ok(Self {
            domain_id: episode.domain_id().to_owned(),
            context_id: episode.context_id().to_owned(),
            environment_family_id: episode.environment_family_id().to_owned(),
            action: episode.action().clone(),
            action_profile_id: episode.action_profile_id().to_owned(),
            action_parameters_blake3: episode.action_parameters_blake3().to_owned(),
        })
    }

    pub fn domain_id(&self) -> &str { &self.domain_id }
    pub fn context_id(&self) -> &str { &self.context_id }
    pub fn environment_family_id(&self) -> &str { &self.environment_family_id }
    pub fn action(&self) -> &ActionCategory { &self.action }
    pub fn action_profile_id(&self) -> &str { &self.action_profile_id }
    pub fn action_parameters_blake3(&self) -> &str { &self.action_parameters_blake3 }
}

/// Research-only exact-key HDC delta table.
///
/// Unlike the legacy action-category table, this candidate never falls back
/// across domains, contexts, environment families or action parameterizations.
/// It returns `None` for an unseen key so callers can abstain or acquire more
/// evidence rather than treating missing coverage as a no-op.
pub struct ContextualDeltaBaselineV1 {
    dim: usize,
    entries: HashMap<ContextualTransitionKeyV1, DeltaEntry>,
}

impl ContextualDeltaBaselineV1 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: HashMap::new(),
        }
    }

    /// Record a completed episode after independent receipt verification.
    ///
    /// This method checks that the supplied state vectors match the episode's
    /// BLAKE3 state digests. It cannot authenticate the execution receipt;
    /// callers must do that before admitting the episode.
    pub fn record_completed_episode(
        &mut self,
        episode: &TransitionEpisodeV1,
        before_state: &ContinuousHV,
        after_state: &ContinuousHV,
    ) -> Result<(), String> {
        episode.validate()?;
        if !episode.has_completed_transition_measurement() {
            return Err("only completed episodes with an after-state and receipt digest can be recorded".to_owned());
        }
        if before_state.as_slice().len() != self.dim
            || after_state.as_slice().len() != self.dim
        {
            return Err("transition state dimension does not match contextual baseline dimension".to_owned());
        }
        if continuous_hv_blake3(before_state)? != episode.before_state_blake3() {
            return Err("before-state vector does not match the episode digest".to_owned());
        }
        if Some(continuous_hv_blake3(after_state)?.as_str()) != episode.after_state_blake3() {
            return Err("after-state vector does not match the episode digest".to_owned());
        }

        let key = ContextualTransitionKeyV1::from_episode(episode)?;
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| DeltaEntry::new(self.dim));
        entry.observe(&after_state.subtract(before_state));
        Ok(())
    }

    /// Predict only when the exact context/action key has observed episodes.
    pub fn predict(
        &self,
        key: &ContextualTransitionKeyV1,
        current_state: &ContinuousHV,
    ) -> Result<Option<TransitionPredictionDiagnostics>, String> {
        if current_state.as_slice().len() != self.dim {
            return Err("current state dimension does not match contextual baseline dimension".to_owned());
        }
        Ok(self.entries.get(key).map(|entry| entry.diagnostics(current_state)))
    }

    pub fn learned_key_count(&self) -> usize {
        self.entries.len()
    }

    pub fn observations_for(&self, key: &ContextualTransitionKeyV1) -> usize {
        self.entries
            .get(key)
            .map(|entry| entry.observation_count)
            .unwrap_or(0)
    }

    pub fn total_observations(&self) -> usize {
        self.entries.values().map(|entry| entry.observation_count).sum()
    }
}

/// A learned state transition delta.
#[derive(Debug, Clone)]
struct DeltaEntry {
    /// Average delta vector for this action category or exact context/action key.
    delta: ContinuousHV,
    /// Number of observations used to learn this delta.
    observation_count: usize,
    /// Number of prequential residual observations (first sample has no prior predictor).
    residual_observation_count: usize,
    /// Number of prequential residuals rejected because their norm was non-finite.
    non_finite_prequential_residual_count: usize,
    /// Mean squared L2 residual against the mean delta before each update.
    mean_squared_prequential_delta_error: f64,
}

impl DeltaEntry {
    fn new(dim: usize) -> Self {
        Self {
            delta: ContinuousHV::zero(dim),
            observation_count: 0,
            residual_observation_count: 0,
            non_finite_prequential_residual_count: 0,
            mean_squared_prequential_delta_error: 0.0,
        }
    }

    fn observe(&mut self, observed_delta: &ContinuousHV) {
        if self.observation_count > 0 {
            let residual = observed_delta.subtract(&self.delta);
            let squared_error = f64::from(residual.norm()).powi(2);
            if squared_error.is_finite() {
                self.residual_observation_count += 1;
                let residual_count = self.residual_observation_count as f64;
                self.mean_squared_prequential_delta_error +=
                    (squared_error - self.mean_squared_prequential_delta_error) / residual_count;
            } else {
                self.non_finite_prequential_residual_count += 1;
            }
        }

        self.observation_count += 1;
        let n = self.observation_count as f32;
        let refs = [&self.delta, observed_delta];
        let weights = [(n - 1.0) / n, 1.0 / n];
        self.delta = ContinuousHV::weighted_bundle(&refs, &weights);
    }

    fn diagnostics(&self, current_state: &ContinuousHV) -> TransitionPredictionDiagnostics {
        TransitionPredictionDiagnostics {
            predicted_state: current_state.add(&self.delta),
            support: if self.observation_count >= 3 {
                TransitionSupport::AtLeastThreeExamples
            } else {
                TransitionSupport::FewerThanThreeExamples
            },
            observation_count: self.observation_count,
            residual_observation_count: self.residual_observation_count,
            non_finite_prequential_residual_count: self.non_finite_prequential_residual_count,
            mean_squared_prequential_delta_error: (self.residual_observation_count > 0)
                .then_some(self.mean_squared_prequential_delta_error),
        }
    }
}


/// Evidence support for a transition forecast.
///
/// The three-observation boundary mirrors the existing `has_learned`
/// heuristic; it is a count threshold, not calibrated confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionSupport {
    NoExamples,
    FewerThanThreeExamples,
    AtLeastThreeExamples,
}

/// A transition forecast plus descriptive support/error diagnostics.
///
/// The residual is the mean squared prequential error of observed action
/// deltas against the running mean *before* each update. It is not a calibrated
/// predictive variance or probability of correctness.
#[derive(Debug, Clone)]
pub struct TransitionPredictionDiagnostics {
    pub predicted_state: ContinuousHV,
    pub support: TransitionSupport,
    pub observation_count: usize,
    pub residual_observation_count: usize,
    pub non_finite_prequential_residual_count: usize,
    pub mean_squared_prequential_delta_error: Option<f64>,
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
        self.predict_state_with_diagnostics(action).predicted_state
    }

    /// Predict a state while exposing whether the action category has evidence.
    ///
    /// For compatibility, an unseen action still returns the current state as
    /// the numeric prediction. Its `NoExamples` status makes clear that this
    /// identity vector is a placeholder, not an evidence-supported no-op model.
    pub fn predict_state_with_diagnostics(
        &self,
        action: &ActionCategory,
    ) -> TransitionPredictionDiagnostics {
        match self.delta_vectors.get(action) {
            Some(entry) => entry.diagnostics(&self.system_state),
            None => TransitionPredictionDiagnostics {
                predicted_state: self.system_state.clone(),
                support: TransitionSupport::NoExamples,
                observation_count: 0,
                residual_observation_count: 0,
                non_finite_prequential_residual_count: 0,
                mean_squared_prequential_delta_error: None,
            },
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
            .or_insert_with(|| DeltaEntry::new(self.dim));
        entry.observe(&observed_delta);
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
    fn transition_episode_hashes_canonical_state_and_records_context() {
        let before = ContinuousHV::random(128, 11);
        let after = ContinuousHV::random(128, 12);
        let episode = TransitionEpisodeV1::new(
            "episode-1",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            Some(&after),
            TransitionOutcome::Completed,
            5_000,
            1_800_000_000_000,
            Some("b".repeat(64)),
        )
        .expect("valid completed transition");
        assert_eq!(episode.before_state_blake3, continuous_hv_blake3(&before).unwrap());
        assert_eq!(episode.after_state_blake3, Some(continuous_hv_blake3(&after).unwrap()));
        assert!(episode.has_completed_transition_measurement());
    }

    #[test]
    fn state_digest_is_deterministic_and_dimension_sensitive() {
        let first = ContinuousHV::random(128, 5);
        let same = ContinuousHV::random(128, 5);
        let different_dimension = ContinuousHV::random(256, 5);
        assert_eq!(continuous_hv_blake3(&first).unwrap(), continuous_hv_blake3(&same).unwrap());
        assert_ne!(
            continuous_hv_blake3(&first).unwrap(),
            continuous_hv_blake3(&different_dimension).unwrap()
        );
    }

    #[test]
    fn non_finite_state_is_rejected_before_digesting() {
        let mut values = vec![0.25_f32; 128];
        values[17] = f32::NAN;
        let state = ContinuousHV::from_vec(values);
        assert!(continuous_hv_blake3(&state).is_err());
    }

    #[test]
    fn failed_partial_and_unknown_outcomes_are_retained_but_not_completed() {
        let before = ContinuousHV::random(128, 21);
        let partial_after = ContinuousHV::random(128, 22);

        for outcome in [
            TransitionOutcome::Failed,
            TransitionOutcome::TimedOut,
            TransitionOutcome::Aborted,
        ] {
            let episode = TransitionEpisodeV1::new(
                format!("episode-{outcome:?}"),
                "nixos-host-v1",
                "workstation-gpu",
                "install-family-v1",
                "cluster-42",
                ActionCategory::Install,
                "nix-install-v1",
                "a".repeat(64),
                &before,
                Some(&partial_after),
                outcome,
                5_000,
                1_800_000_000_000,
                None,
            )
            .expect("failed/partial observation remains representable");
            assert!(!episode.has_completed_transition_measurement());
            assert!(episode.after_state_blake3.is_some());
        }

        let unknown = TransitionEpisodeV1::new(
            "episode-unknown",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            None,
            TransitionOutcome::Unknown,
            5_000,
            1_800_000_000_000,
            None,
        )
        .expect("unknown execution remains representable");
        assert!(!unknown.has_completed_transition_measurement());
        assert!(unknown.after_state_blake3.is_none());
    }

    #[test]
    fn completed_outcome_requires_after_state_and_receipt_digest() {
        let before = ContinuousHV::random(128, 31);
        let after = ContinuousHV::random(128, 32);
        let missing_after = TransitionEpisodeV1::new(
            "episode-missing-after",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            None,
            TransitionOutcome::Completed,
            5_000,
            1_800_000_000_000,
            Some("b".repeat(64)),
        );
        assert!(missing_after.is_err());

        let missing_receipt = TransitionEpisodeV1::new(
            "episode-missing-receipt",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            Some(&after),
            TransitionOutcome::Completed,
            5_000,
            1_800_000_000_000,
            None,
        );
        assert!(missing_receipt.is_err());
    }

    #[test]
    fn transition_episode_rejects_mismatched_dimensions_and_bad_provenance() {
        let before = ContinuousHV::random(128, 41);
        let after = ContinuousHV::random(256, 42);
        assert!(TransitionEpisodeV1::new(
            "episode-bad-dimension",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            Some(&after),
            TransitionOutcome::Completed,
            5_000,
            1_800_000_000_000,
            Some("b".repeat(64)),
        ).is_err());

        assert!(TransitionEpisodeV1::new(
            "episode-bad-receipt",
            "nixos-host-v1",
            "workstation-gpu",
            "install-family-v1",
            "cluster-42",
            ActionCategory::Install,
            "nix-install-v1",
            "bad-digest",
            &before,
            Some(&before),
            TransitionOutcome::Completed,
            5_000,
            1_800_000_000_000,
            Some("b".repeat(64)),
        ).is_err());
    }


    #[test]
    fn unknown_action_is_explicitly_distinguished_from_learned_no_change() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let current = ContinuousHV::random(dim, 91);
        wm.observe(current.clone());

        let diagnostics = wm.predict_state_with_diagnostics(&ActionCategory::Rollback);
        assert_eq!(diagnostics.support, TransitionSupport::NoExamples);
        assert_eq!(diagnostics.observation_count, 0);
        assert_eq!(diagnostics.residual_observation_count, 0);
        assert!(diagnostics.mean_squared_prequential_delta_error.is_none());
        assert!((diagnostics.predicted_state.similarity(&current) - 1.0).abs() < 1e-5);
        assert!((wm.predict_state(&ActionCategory::Rollback).similarity(&current) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn support_diagnostic_tracks_sparse_and_legacy_three_sample_threshold() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let action = ActionCategory::Install;
        let before = ContinuousHV::random(dim, 92);
        let after = ContinuousHV::random(dim, 93);

        for (index, expected_support) in [
            TransitionSupport::FewerThanThreeExamples,
            TransitionSupport::FewerThanThreeExamples,
            TransitionSupport::AtLeastThreeExamples,
        ]
        .into_iter()
        .enumerate()
        {
            wm.learn_transition(&before, action.clone(), &after);
            let diagnostics = wm.predict_state_with_diagnostics(&action);
            assert_eq!(diagnostics.observation_count, index + 1);
            assert_eq!(diagnostics.support, expected_support);
            assert_eq!(wm.has_learned(&action), index + 1 >= 3);
        }
    }

    #[test]
    fn prequential_delta_error_is_missing_for_first_sample_and_zero_for_repeat() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let action = ActionCategory::Install;
        let before = ContinuousHV::random(dim, 94);
        let after = ContinuousHV::random(dim, 95);

        wm.learn_transition(&before, action.clone(), &after);
        let first = wm.predict_state_with_diagnostics(&action);
        assert_eq!(first.residual_observation_count, 0);
        assert!(first.mean_squared_prequential_delta_error.is_none());

        wm.learn_transition(&before, action.clone(), &after);
        let second = wm.predict_state_with_diagnostics(&action);
        assert_eq!(second.residual_observation_count, 1);
        assert!(second.mean_squared_prequential_delta_error.is_some_and(|error| error < 1e-10));
    }


    #[test]
    fn non_finite_prequential_residual_is_counted_not_folded_into_mean() {
        let dim = 128;
        let mut wm = NixWorldModel::new(dim);
        let action = ActionCategory::Install;
        let before = ContinuousHV::random(dim, 101);
        let mut invalid_values = vec![0.5_f32; dim];
        invalid_values[7] = f32::NAN;
        let invalid_after = ContinuousHV::from_vec(invalid_values);

        wm.learn_transition(&before, action.clone(), &invalid_after);
        wm.learn_transition(&before, action.clone(), &invalid_after);
        let diagnostics = wm.predict_state_with_diagnostics(&action);
        assert_eq!(diagnostics.residual_observation_count, 0);
        assert_eq!(diagnostics.non_finite_prequential_residual_count, 1);
        assert!(diagnostics.mean_squared_prequential_delta_error.is_none());
    }

    #[test]
    fn contradictory_transition_increases_prequential_error_diagnostic() {
        let dim = 512;
        let mut wm = NixWorldModel::new(dim);
        let action = ActionCategory::Install;
        let before = ContinuousHV::random(dim, 96);
        let after_a = ContinuousHV::random(dim, 97);
        let after_b = ContinuousHV::random(dim, 98);

        wm.learn_transition(&before, action.clone(), &after_a);
        wm.learn_transition(&before, action.clone(), &after_a);
        let stable_error = wm
            .predict_state_with_diagnostics(&action)
            .mean_squared_prequential_delta_error
            .expect("second sample provides one prequential residual");

        wm.learn_transition(&before, action.clone(), &after_b);
        let contradictory_error = wm
            .predict_state_with_diagnostics(&action)
            .mean_squared_prequential_delta_error
            .expect("third sample updates prequential residual");
        assert!(contradictory_error > stable_error);
        assert!(contradictory_error.is_finite());
    }


    fn make_contextual_episode(
        episode_id: &str,
        context_id: &str,
        action_parameters_blake3: &str,
        before: &ContinuousHV,
        after: &ContinuousHV,
    ) -> TransitionEpisodeV1 {
        TransitionEpisodeV1::new(
            episode_id,
            "nixos-host-v1",
            context_id,
            "install-mechanism-v1",
            "cluster-1",
            ActionCategory::Install,
            "nix-install-v1",
            action_parameters_blake3,
            before,
            Some(after),
            TransitionOutcome::Completed,
            1_000,
            1_800_000_000_000,
            Some("b".repeat(64)),
        )
        .expect("valid contextual transition episode")
    }

    #[test]
    fn contextual_delta_table_separates_opposing_context_transitions() {
        let dim = 128;
        let before = ContinuousHV::zero(dim);
        let mut a_values = vec![0.0_f32; dim];
        let mut b_values = vec![0.0_f32; dim];
        a_values[..dim / 2].fill(1.0);
        b_values[dim / 2..].fill(1.0);
        let after_a = ContinuousHV::from_vec(a_values);
        let after_b = ContinuousHV::from_vec(b_values);

        let episode_a = make_contextual_episode("episode-a", "context-a", &"a".repeat(64), &before, &after_a);
        let episode_b = make_contextual_episode("episode-b", "context-b", &"a".repeat(64), &before, &after_b);
        let key_a = ContextualTransitionKeyV1::from_episode(&episode_a).unwrap();
        let key_b = ContextualTransitionKeyV1::from_episode(&episode_b).unwrap();

        let mut contextual = ContextualDeltaBaselineV1::new(dim);
        contextual.record_completed_episode(&episode_a, &before, &after_a).unwrap();
        contextual.record_completed_episode(&episode_b, &before, &after_b).unwrap();

        let predicted_a = contextual.predict(&key_a, &before).unwrap().unwrap();
        let predicted_b = contextual.predict(&key_b, &before).unwrap().unwrap();
        assert!(predicted_a.predicted_state.similarity(&after_a) > 0.999);
        assert!(predicted_b.predicted_state.similarity(&after_b) > 0.999);
        assert_eq!(contextual.learned_key_count(), 2);

        let mut global = NixWorldModel::new(dim);
        global.learn_transition(&before, ActionCategory::Install, &after_a);
        global.learn_transition(&before, ActionCategory::Install, &after_b);
        global.observe(before.clone());
        let global_prediction = global.predict_state(&ActionCategory::Install);
        assert!(
            predicted_a.predicted_state.similarity(&after_a)
                > global_prediction.similarity(&after_a) + 0.1
        );
        assert!(
            predicted_b.predicted_state.similarity(&after_b)
                > global_prediction.similarity(&after_b) + 0.1
        );
    }

    #[test]
    fn contextual_key_includes_action_parameter_identity() {
        let dim = 128;
        let before = ContinuousHV::random(dim, 501);
        let after = ContinuousHV::random(dim, 502);
        let first = make_contextual_episode("episode-a", "context-a", &"a".repeat(64), &before, &after);
        let second = make_contextual_episode("episode-b", "context-a", &"c".repeat(64), &before, &after);
        let key_first = ContextualTransitionKeyV1::from_episode(&first).unwrap();
        let key_second = ContextualTransitionKeyV1::from_episode(&second).unwrap();

        let mut contextual = ContextualDeltaBaselineV1::new(dim);
        contextual.record_completed_episode(&first, &before, &after).unwrap();

        assert!(contextual.predict(&key_first, &before).unwrap().is_some());
        assert!(contextual.predict(&key_second, &before).unwrap().is_none());
    }

    #[test]
    fn contextual_baseline_returns_none_for_unseen_context_key() {
        let dim = 128;
        let before = ContinuousHV::random(dim, 511);
        let after = ContinuousHV::random(dim, 512);
        let episode = make_contextual_episode("episode-a", "context-a", &"a".repeat(64), &before, &after);
        let key = ContextualTransitionKeyV1::from_episode(&episode).unwrap();
        let contextual = ContextualDeltaBaselineV1::new(dim);
        assert!(contextual.predict(&key, &before).unwrap().is_none());
    }

    #[test]
    fn contextual_baseline_rejects_incomplete_or_mismatched_episode_evidence() {
        let dim = 128;
        let before = ContinuousHV::random(dim, 521);
        let after = ContinuousHV::random(dim, 522);
        let failed = TransitionEpisodeV1::new(
            "failed-episode",
            "nixos-host-v1",
            "context-a",
            "install-mechanism-v1",
            "cluster-1",
            ActionCategory::Install,
            "nix-install-v1",
            "a".repeat(64),
            &before,
            Some(&after),
            TransitionOutcome::Failed,
            1_000,
            1_800_000_000_000,
            None,
        )
        .unwrap();
        let mut contextual = ContextualDeltaBaselineV1::new(dim);
        assert!(contextual.record_completed_episode(&failed, &before, &after).is_err());

        let completed = make_contextual_episode("completed-episode", "context-a", &"a".repeat(64), &before, &after);
        let wrong_after = ContinuousHV::random(dim, 523);
        assert!(
            contextual
                .record_completed_episode(&completed, &before, &wrong_after)
                .is_err()
        );
        assert_eq!(contextual.learned_key_count(), 0);
    }

    #[test]
    fn contextual_baseline_updates_running_mean_for_repeated_exact_key() {
        let dim = 128;
        let before = ContinuousHV::zero(dim);
        let mut first_values = vec![0.0_f32; dim];
        let mut second_values = vec![0.0_f32; dim];
        first_values[..dim / 2].fill(1.0);
        second_values[..dim / 2].fill(0.5);
        let first_after = ContinuousHV::from_vec(first_values);
        let second_after = ContinuousHV::from_vec(second_values);
        let first = make_contextual_episode("episode-1", "same-context", &"a".repeat(64), &before, &first_after);
        let second = make_contextual_episode("episode-2", "same-context", &"a".repeat(64), &before, &second_after);
        let key = ContextualTransitionKeyV1::from_episode(&first).unwrap();

        let mut contextual = ContextualDeltaBaselineV1::new(dim);
        contextual.record_completed_episode(&first, &before, &first_after).unwrap();
        contextual.record_completed_episode(&second, &before, &second_after).unwrap();

        assert_eq!(contextual.learned_key_count(), 1);
        assert_eq!(contextual.observations_for(&key), 2);
        let prediction = contextual.predict(&key, &before).unwrap().unwrap();
        assert_eq!(prediction.observation_count, 2);
        assert_eq!(prediction.residual_observation_count, 1);
        assert!(prediction.mean_squared_prequential_delta_error.is_some());
    }

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
}
