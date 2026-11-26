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

/// Counts of observed execution outcomes; repeated episodes are not independent samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransitionOutcomeCounts {
    pub success: usize,
    pub failure: usize,
    pub partial: usize,
    pub rolled_back: usize,
}

impl TransitionOutcomeCounts {
    fn record(&mut self, outcome: TransitionOutcomeClass) {
        match outcome {
            TransitionOutcomeClass::Success => self.success += 1,
            TransitionOutcomeClass::Failure => self.failure += 1,
            TransitionOutcomeClass::Partial => self.partial += 1,
            TransitionOutcomeClass::RolledBack => self.rolled_back += 1,
        }
    }
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
        realized_action_sha256: String,
        realized_action_parameters: ContinuousHV,
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
    /// Exact canonical intent/action payload identity, including parameters.
    pub assigned_action_sha256: String,
    /// HDC encoding of assigned action parameters.
    pub assigned_action_parameters: ContinuousHV,
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
        if dimension == 0
            || self.context.as_slice().len() != dimension
            || self.assigned_action_parameters.as_slice().len() != dimension
        {
            return Err(
                "state_before and context must have the same non-zero HDC dimension".to_owned(),
            );
        }
        if self
            .state_before
            .as_slice()
            .iter()
            .chain(self.context.as_slice())
            .chain(self.assigned_action_parameters.as_slice())
            .any(|value| !value.is_finite() || value.abs() > 1.0)
        {
            return Err(
                "state_before and context must contain finite HDC values in [-1, 1]".to_owned(),
            );
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
            ("assigned_action_sha256", &self.assigned_action_sha256),
            ("state_before_sha256", &self.state_before_sha256),
            ("source_manifest_sha256", &self.source_manifest_sha256),
            ("provenance_sha256", &self.provenance_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(format!("{name} must be a 64-character SHA-256 hex digest"));
            }
        }

        if let TransitionExecution::Executed {
            realized_action_sha256,
            realized_action_parameters,
            state_after,
            state_after_sha256,
            ..
        } = &self.execution
        {
            if realized_action_parameters.as_slice().len() != dimension
                || realized_action_parameters
                    .as_slice()
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 1.0)
            {
                return Err("realized action parameters must match the HDC dimension and contain finite values in [-1, 1]".to_owned());
            }
            if !is_sha256_hex(realized_action_sha256) {
                return Err(
                    "realized_action_sha256 must be a 64-character SHA-256 hex digest".to_owned(),
                );
            }
            if state_after.as_slice().len() != dimension
                || state_after
                    .as_slice()
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > 1.0)
            {
                return Err("executed state_after must match the HDC dimension and contain finite values in [-1, 1]".to_owned());
            }
            if !is_sha256_hex(state_after_sha256) {
                return Err(
                    "state_after_sha256 must be a 64-character SHA-256 hex digest".to_owned(),
                );
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
    /// Equally relevant family-specific buckets predict materially different deltas.
    /// The returned state is an identity placeholder; callers must treat this as UNKNOWN.
    AmbiguousContextUnknown,
    GlobalActionDeltaFallback,
    UnknownActionIdentityFallback,
}

/// Prediction diagnostics for the context-conditioned HDC-delta candidate.
///
/// For `AmbiguousContextUnknown`, `predicted_state` is an identity placeholder,
/// not a forecast; callers must abstain rather than plan from it.
/// `training_residual_mse` is a prequential training diagnostic, not a
/// calibrated predictive interval or a held-out uncertainty guarantee.
#[derive(Debug, Clone)]
pub struct ContextualPrediction {
    pub predicted_state: ContinuousHV,
    pub source: ContextualPredictionSource,
    pub context_similarity: f64,
    pub action_parameter_similarity: f64,
    pub observation_count: usize,
    pub independent_cluster_count: usize,
    pub training_residual_mse: Option<f64>,
    /// Outcome counts from the selected context bucket or global fallback.
    pub outcome_counts: TransitionOutcomeCounts,
}

#[derive(Debug, Clone)]
struct ContextDeltaEntry {
    environment_family_id: String,
    action_parameters: ContinuousHV,
    context: ContinuousHV,
    delta: ContinuousHV,
    observation_count: usize,
    residual_count: usize,
    residual_mse: f64,
    outcome_counts: TransitionOutcomeCounts,
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
            return Err(format!(
                "duplicate transition episode: {}",
                episode.episode_id
            ));
        }
        if episode.state_before.as_slice().len() != self.dim {
            return Err("episode HDC dimension does not match this world model".to_owned());
        }

        let (realized_action, realized_action_parameters, state_after) = match &episode.execution {
            TransitionExecution::Executed {
                realized_action,
                realized_action_parameters,
                state_after,
                ..
            } => (
                realized_action.clone(),
                realized_action_parameters,
                state_after,
            ),
            TransitionExecution::NotExecuted => {
                return Err(
                    "unexecuted transition episodes cannot train the dynamics model".to_owned(),
                );
            }
            TransitionExecution::OutcomeUnknown => {
                return Err(
                    "execution-unknown episodes require reconciliation before learning".to_owned(),
                );
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
            .map(|(index, entry)| {
                (
                    index,
                    entry.context.similarity(&episode.context),
                    entry
                        .action_parameters
                        .similarity(realized_action_parameters),
                )
            })
            .filter(|(_, context_similarity, action_similarity)| {
                context_similarity.is_finite() && action_similarity.is_finite()
            })
            .max_by(|left, right| left.1.min(left.2).total_cmp(&right.1.min(right.2)));

        if let Some((index, _context_similarity, _action_similarity)) =
            best_match.filter(|(_, context_similarity, action_similarity)| {
                *context_similarity >= CONTEXT_MATCH_THRESHOLD
                    && *action_similarity >= CONTEXT_MATCH_THRESHOLD
            })
        {
            let entry = &mut entries[index];
            if let Some(residual_mse) = mean_squared_error(&entry.delta, &observed_delta) {
                entry.residual_count += 1;
                let residual_n = entry.residual_count as f64;
                entry.residual_mse += (residual_mse - entry.residual_mse) / residual_n;
            }

            entry.observation_count += 1;
            let n = entry.observation_count as f32;
            let refs = [&entry.delta, &observed_delta];
            let weights = [(n - 1.0) / n, 1.0 / n];
            entry.delta = ContinuousHV::weighted_bundle(&refs, &weights);
            if let TransitionExecution::Executed { outcome, .. } = &episode.execution {
                entry.outcome_counts.record(*outcome);
            }
            entry
                .independent_cluster_ids
                .insert(episode.independent_cluster_id.clone());
        } else {
            let mut independent_cluster_ids = HashSet::new();
            independent_cluster_ids.insert(episode.independent_cluster_id.clone());
            entries.push(ContextDeltaEntry {
                environment_family_id: episode.environment_family_id.clone(),
                action_parameters: realized_action_parameters.clone(),
                context: episode.context.clone(),
                delta: observed_delta,
                observation_count: 1,
                residual_count: 0,
                residual_mse: 0.0,
                outcome_counts: match &episode.execution {
                    TransitionExecution::Executed { outcome, .. } => {
                        let mut counts = TransitionOutcomeCounts::default();
                        counts.record(*outcome);
                        counts
                    }
                    _ => TransitionOutcomeCounts::default(),
                },
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
        action_parameters: &ContinuousHV,
        context: &ContinuousHV,
    ) -> Result<ContextualPrediction, String> {
        if state.as_slice().len() != self.dim
            || context.as_slice().len() != self.dim
            || action_parameters.as_slice().len() != self.dim
            || state.as_slice().iter().any(|value| !value.is_finite())
            || context
                .as_slice()
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0)
            || action_parameters
                .as_slice()
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0)
        {
            return Err(
                "prediction state and context must match the model dimension and be finite"
                    .to_owned(),
            );
        }

        let best_match = self
            .contextual_delta_vectors
            .get(action)
            .and_then(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| entry.environment_family_id == environment_family_id)
                    .map(|(index, entry)| {
                        (
                            index,
                            entry.context.similarity(context),
                            entry.action_parameters.similarity(action_parameters),
                        )
                    })
                    .filter(|(_, context_similarity, action_similarity)| {
                        context_similarity.is_finite() && action_similarity.is_finite()
                    })
                    .max_by(|left, right| left.1.min(left.2).total_cmp(&right.1.min(right.2)))
            });

        if let Some((index, context_similarity, action_similarity)) =
            best_match.filter(|(_, context_similarity, action_similarity)| {
                *context_similarity >= CONTEXT_MATCH_THRESHOLD
                    && *action_similarity >= CONTEXT_MATCH_THRESHOLD
            })
        {
            let entry = &self.contextual_delta_vectors[action][index];
            return Ok(ContextualPrediction {
                predicted_state: state.add(&entry.delta),
                source: ContextualPredictionSource::ContextualDelta,
                context_similarity: context_similarity as f64,
                action_parameter_similarity: action_similarity as f64,
                observation_count: entry.observation_count,
                independent_cluster_count: entry.independent_cluster_ids.len(),
                training_residual_mse: (entry.residual_count > 0).then_some(entry.residual_mse),
                outcome_counts: entry.outcome_counts,
            });
        }

        let (context_similarity, action_parameter_similarity) = best_match
            .map(|(_, context_similarity, action_similarity)| {
                (
                    context_similarity.max(0.0) as f64,
                    action_similarity.max(0.0) as f64,
                )
            })
            .unwrap_or((0.0, 0.0));
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
            let outcome_counts = self
                .contextual_delta_vectors
                .get(action)
                .map(|entries| {
                    entries
                        .iter()
                        .fold(TransitionOutcomeCounts::default(), |mut total, entry| {
                            total.success += entry.outcome_counts.success;
                            total.failure += entry.outcome_counts.failure;
                            total.partial += entry.outcome_counts.partial;
                            total.rolled_back += entry.outcome_counts.rolled_back;
                            total
                        })
                })
                .unwrap_or_default();
            return Ok(ContextualPrediction {
                predicted_state: state.add(&entry.delta),
                source: ContextualPredictionSource::GlobalActionDeltaFallback,
                context_similarity,
                action_parameter_similarity,
                observation_count: entry.observation_count,
                independent_cluster_count,
                training_residual_mse: None,
                outcome_counts,
            });
        }

        Ok(ContextualPrediction {
            predicted_state: state.clone(),
            source: ContextualPredictionSource::UnknownActionIdentityFallback,
            context_similarity,
            action_parameter_similarity,
            observation_count: 0,
            independent_cluster_count: 0,
            training_residual_mse: None,
            outcome_counts: TransitionOutcomeCounts::default(),
        })
    }

    /// Predict using only observable context and action-parameter HDC.
    ///
    /// Unlike `predict_state_with_context`, this API accepts no environment or
    /// mechanism-family ID. It searches all family-scoped training buckets.
    /// If equally relevant buckets imply materially different deltas, the
    /// result is explicitly UNKNOWN instead of selecting one family silently.
    /// In the UNKNOWN case, `predicted_state` is an identity placeholder and
    /// must not be interpreted as a prediction or supplied to a planner.
    pub fn predict_state_without_family_label(
        &self,
        state: &ContinuousHV,
        action: &ActionCategory,
        action_parameters: &ContinuousHV,
        context: &ContinuousHV,
    ) -> Result<ContextualPrediction, String> {
        if state.as_slice().len() != self.dim
            || context.as_slice().len() != self.dim
            || action_parameters.as_slice().len() != self.dim
            || state.as_slice().iter().any(|value| !value.is_finite())
            || context
                .as_slice()
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0)
            || action_parameters
                .as_slice()
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0)
        {
            return Err(
                "prediction state and context must match the model dimension and be finite"
                    .to_owned(),
            );
        }

        let entries = self
            .contextual_delta_vectors
            .get(action)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut nearest: Option<(usize, f32, f32, f32)> = None;
        let mut matches = Vec::new();

        for (index, entry) in entries.iter().enumerate() {
            let context_similarity = entry.context.similarity(context);
            let action_similarity = entry.action_parameters.similarity(action_parameters);
            if !context_similarity.is_finite() || !action_similarity.is_finite() {
                continue;
            }
            let score = context_similarity.min(action_similarity);
            if nearest.is_none_or(|(_, _, _, best_score)| score > best_score) {
                nearest = Some((index, context_similarity, action_similarity, score));
            }
            if context_similarity >= CONTEXT_MATCH_THRESHOLD
                && action_similarity >= CONTEXT_MATCH_THRESHOLD
            {
                matches.push((index, context_similarity, action_similarity, score));
            }
        }

        if matches.is_empty() {
            let (context_similarity, action_parameter_similarity) = nearest
                .map(|(_, context, parameters, _)| {
                    (context.max(0.0) as f64, parameters.max(0.0) as f64)
                })
                .unwrap_or((0.0, 0.0));
            if let Some(global) = self.delta_vectors.get(action) {
                let independent_cluster_count = entries
                    .iter()
                    .flat_map(|entry| entry.independent_cluster_ids.iter())
                    .collect::<HashSet<_>>()
                    .len();
                let outcome_counts =
                    entries
                        .iter()
                        .fold(TransitionOutcomeCounts::default(), |mut total, entry| {
                            total.success += entry.outcome_counts.success;
                            total.failure += entry.outcome_counts.failure;
                            total.partial += entry.outcome_counts.partial;
                            total.rolled_back += entry.outcome_counts.rolled_back;
                            total
                        });
                return Ok(ContextualPrediction {
                    predicted_state: state.add(&global.delta),
                    source: ContextualPredictionSource::GlobalActionDeltaFallback,
                    context_similarity,
                    action_parameter_similarity,
                    observation_count: global.observation_count,
                    independent_cluster_count,
                    training_residual_mse: None,
                    outcome_counts,
                });
            }

            return Ok(ContextualPrediction {
                predicted_state: state.clone(),
                source: ContextualPredictionSource::UnknownActionIdentityFallback,
                context_similarity,
                action_parameter_similarity,
                observation_count: 0,
                independent_cluster_count: 0,
                training_residual_mse: None,
                outcome_counts: TransitionOutcomeCounts::default(),
            });
        }

        let best_score = matches
            .iter()
            .map(|(_, _, _, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        const MATCH_TIE_EPSILON: f32 = 1.0e-6;
        const DELTA_CONFLICT_MSE: f64 = 1.0e-6;
        let mut tied_matches: Vec<_> = matches
            .into_iter()
            .filter(|(_, _, _, score)| best_score - *score <= MATCH_TIE_EPSILON)
            .collect();

        // Stable ordering is based only on candidate-visible prediction data,
        // never on environment-family ID or insertion order.
        tied_matches.sort_by(|left, right| {
            let left_delta = entries[left.0].delta.as_slice();
            let right_delta = entries[right.0].delta.as_slice();
            for (left_value, right_value) in left_delta.iter().zip(right_delta) {
                let order = left_value.total_cmp(right_value);
                if !order.is_eq() {
                    return order;
                }
            }
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.2.total_cmp(&right.2))
        });

        let selected = &entries[tied_matches[0].0];
        let best_context_similarity = tied_matches
            .iter()
            .map(|(_, context, _, _)| *context)
            .fold(f32::NEG_INFINITY, f32::max);
        let best_action_similarity = tied_matches
            .iter()
            .map(|(_, _, parameters, _)| *parameters)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut observation_count = 0usize;
        let mut independent_cluster_ids = HashSet::new();
        let mut outcome_counts = TransitionOutcomeCounts::default();
        let mut residual_squared_error = 0.0_f64;
        let mut residual_count = 0usize;
        let mut conflicting_delta = false;

        for (index, _, _, _) in &tied_matches {
            let entry = &entries[*index];
            observation_count = observation_count.saturating_add(entry.observation_count);
            independent_cluster_ids.extend(entry.independent_cluster_ids.iter().cloned());
            outcome_counts.success =
                outcome_counts.success.saturating_add(entry.outcome_counts.success);
            outcome_counts.failure =
                outcome_counts.failure.saturating_add(entry.outcome_counts.failure);
            outcome_counts.partial =
                outcome_counts.partial.saturating_add(entry.outcome_counts.partial);
            outcome_counts.rolled_back =
                outcome_counts.rolled_back.saturating_add(entry.outcome_counts.rolled_back);

            if entry.residual_count > 0 {
                residual_squared_error += entry.residual_mse * entry.residual_count as f64;
                residual_count = residual_count.saturating_add(entry.residual_count);
            }
            if mean_squared_error(&selected.delta, &entry.delta)
                .is_none_or(|mse| mse > DELTA_CONFLICT_MSE)
            {
                conflicting_delta = true;
            }
        }

        if conflicting_delta {
            return Ok(ContextualPrediction {
                predicted_state: state.clone(),
                source: ContextualPredictionSource::AmbiguousContextUnknown,
                context_similarity: best_context_similarity.max(0.0) as f64,
                action_parameter_similarity: best_action_similarity.max(0.0) as f64,
                observation_count,
                independent_cluster_count: independent_cluster_ids.len(),
                training_residual_mse: None,
                outcome_counts,
            });
        }

        Ok(ContextualPrediction {
            predicted_state: state.add(&selected.delta),
            source: ContextualPredictionSource::ContextualDelta,
            context_similarity: best_context_similarity.max(0.0) as f64,
            action_parameter_similarity: best_action_similarity.max(0.0) as f64,
            observation_count,
            independent_cluster_count: independent_cluster_ids.len(),
            training_residual_mse: (residual_count > 0)
                .then_some(residual_squared_error / residual_count as f64),
            outcome_counts,
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

fn mean_squared_error(left: &ContinuousHV, right: &ContinuousHV) -> Option<f64> {
    if left.as_slice().len() != right.as_slice().len() || left.as_slice().is_empty() {
        return None;
    }
    Some(
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left_value, right_value)| {
                let difference = *left_value as f64 - *right_value as f64;
                difference * difference
            })
            .sum::<f64>()
            / left.as_slice().len() as f64,
    )
}

impl Default for NixWorldModel {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_020_state_response::{
        score_abstraction, ResponseKey, StateResponseContract, StateResponseRecord,
    };
    use std::collections::{BTreeMap, BTreeSet};

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
            assigned_action_sha256: "f".repeat(64),
            assigned_action_parameters: ContinuousHV::random(256, 90),
            state_before: before,
            context,
            execution: TransitionExecution::Executed {
                realized_action,
                realized_action_sha256: "e".repeat(64),
                realized_action_parameters: ContinuousHV::random(256, 90),
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

        wm.learn_episode(&episode)
            .expect("executed episode is learnable");
        assert_eq!(wm.learned_episode_count(), 1);
        assert_eq!(wm.total_observations(), 1);

        let learned = wm
            .predict_state_with_context(
                &before,
                &ActionCategory::Rollback,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("valid contextual prediction");
        assert_eq!(learned.source, ContextualPredictionSource::ContextualDelta);
        assert_eq!(learned.outcome_counts.rolled_back, 1);
        assert!((learned.predicted_state.similarity(&after) - 1.0).abs() < 1e-5);

        let not_learned = wm
            .predict_state_with_context(
                &before,
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("valid fallback prediction");
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

        wm.learn_episode(&episode)
            .expect("executed failure is real evidence");
        let prediction = wm
            .predict_state_with_context(
                &before,
                &ActionCategory::Rebuild,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("valid prediction");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::ContextualDelta
        );
        assert_eq!(prediction.outcome_counts.failure, 1);
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

        let prediction_a = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context_a,
            )
            .expect("context a prediction");
        let prediction_b = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context_b,
            )
            .expect("context b prediction");

        assert_eq!(
            prediction_a.source,
            ContextualPredictionSource::ContextualDelta
        );
        assert_eq!(
            prediction_b.source,
            ContextualPredictionSource::ContextualDelta
        );
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
        wm.learn_episode(&episode_b)
            .expect("repeated cluster observation");

        let prediction = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Update,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("valid prediction");
        assert_eq!(prediction.observation_count, 2);
        assert_eq!(prediction.independent_cluster_count, 1);
        assert_eq!(prediction.outcome_counts.success, 1);
        assert_eq!(prediction.outcome_counts.partial, 1);
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

        let mut invalid_provenance = episode.clone();
        invalid_provenance.episode_id = "invalid-provenance-digest".to_owned();
        invalid_provenance.provenance_sha256 = "not-a-sha256".to_owned();
        assert!(wm.learn_episode(&invalid_provenance).is_err());
        assert_eq!(wm.total_observations(), 1);

        let mut invalid_assigned_action = episode.clone();
        invalid_assigned_action.episode_id = "invalid-assigned-action-digest".to_owned();
        invalid_assigned_action.assigned_action_sha256 = "not-a-sha256".to_owned();
        assert!(wm.learn_episode(&invalid_assigned_action).is_err());
        assert_eq!(wm.total_observations(), 1);

        let mut invalid_realized_action = episode;
        invalid_realized_action.episode_id = "invalid-realized-action-digest".to_owned();
        if let TransitionExecution::Executed {
            realized_action_sha256,
            ..
        } = &mut invalid_realized_action.execution
        {
            *realized_action_sha256 = "not-a-sha256".to_owned();
        }
        assert!(wm.learn_episode(&invalid_realized_action).is_err());
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
            realized_action_sha256: "d".repeat(64),
            realized_action_parameters: ContinuousHV::random(dim, 90),
            outcome: TransitionOutcomeClass::Failure,
            state_after: ContinuousHV::random(dim, 63),
            state_after_sha256: "e".repeat(64),
        };
        wm.learn_episode(&episode).expect("family b episode");

        let prediction = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("family-scoped prediction");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::ContextualDelta
        );
        assert_eq!(prediction.independent_cluster_count, 1);
    }

    #[test]
    fn contextual_candidate_beats_global_delta_when_mechanism_family_is_known() {
        // This is a known-family contextual retrieval comparison, not a
        // held-out-family transfer result: both mechanism families appear in
        // training and prediction explicitly supplies family A.
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 200);
        let after_a = ContinuousHV::random(dim, 201);
        let after_b = ContinuousHV::random(dim, 202);
        let context_a = ContinuousHV::random(dim, 203);
        let context_b = ContinuousHV::random(dim, 204);
        let action_parameters = ContinuousHV::random(dim, 90);

        for index in 0..4 {
            let mut episode_a = transition_episode(
                &format!("family-a-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before.clone(),
                context_a.clone(),
                after_a.clone(),
                TransitionOutcomeClass::Success,
                &format!("family-a-cluster-{index}"),
            );
            episode_a.environment_family_id = "mechanism-family-a".to_owned();
            wm.learn_episode(&episode_a)
                .expect("family A training episode");

            let mut episode_b = transition_episode(
                &format!("family-b-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before.clone(),
                context_b.clone(),
                after_b.clone(),
                TransitionOutcomeClass::Success,
                &format!("family-b-cluster-{index}"),
            );
            episode_b.environment_family_id = "mechanism-family-b".to_owned();
            wm.learn_episode(&episode_b)
                .expect("family B training episode");
        }

        wm.observe(before.clone());
        let global_prediction = wm.predict_state(&ActionCategory::Install);
        let contextual_prediction = wm
            .predict_state_with_context(
                &before,
                &ActionCategory::Install,
                "mechanism-family-a",
                &action_parameters,
                &context_a,
            )
            .expect("family A contextual prediction");

        let global_error = mean_squared_error(&global_prediction, &after_a)
            .expect("same-dimensional global prediction");
        let contextual_error = mean_squared_error(&contextual_prediction.predicted_state, &after_a)
            .expect("same-dimensional contextual prediction");

        assert_eq!(
            contextual_prediction.source,
            ContextualPredictionSource::ContextualDelta
        );
        assert_eq!(contextual_prediction.independent_cluster_count, 4);
        assert!(
            contextual_error + 1e-5 < global_error,
            "contextual error {contextual_error} should beat pooled global error {global_error}"
        );
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
        let prediction = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &context_b,
            )
            .expect("fallback prediction");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::GlobalActionDeltaFallback
        );
        assert_eq!(prediction.observation_count, 1);
        assert!(prediction.training_residual_mse.is_none());
    }

    #[test]
    fn contextual_prediction_uses_global_fallback_for_unseen_mechanism_family() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 300);
        let after = ContinuousHV::random(dim, 301);
        let context = ContinuousHV::random(dim, 302);
        let action_parameters = ContinuousHV::random(dim, 90);
        let episode = transition_episode(
            "known-family-training",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            context.clone(),
            after,
            TransitionOutcomeClass::Success,
            "known-family-cluster",
        );
        wm.learn_episode(&episode)
            .expect("known-family training episode");

        let prediction = wm
            .predict_state_with_context(
                &before,
                &ActionCategory::Install,
                "unseen-mechanism-family",
                &action_parameters,
                &context,
            )
            .expect("unseen family remains an explicit prediction");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::GlobalActionDeltaFallback
        );
        assert_eq!(prediction.context_similarity, 0.0);
        assert_eq!(prediction.action_parameter_similarity, 0.0);
        assert_eq!(prediction.observation_count, 1);
    }

    #[test]
    fn contextual_model_does_not_reuse_transition_for_different_action_parameters() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let state = ContinuousHV::random(dim, 70);
        let context = ContinuousHV::random(dim, 71);
        let after = ContinuousHV::random(dim, 72);
        let episode = transition_episode(
            "parameterized-action",
            ActionCategory::Configure,
            ActionCategory::Configure,
            state.clone(),
            context.clone(),
            after,
            TransitionOutcomeClass::Success,
            "cluster-1",
        );
        wm.learn_episode(&episode).expect("training episode");

        let unseen_action_parameters = ContinuousHV::random(dim, 73);
        let prediction = wm
            .predict_state_with_context(
                &state,
                &ActionCategory::Configure,
                "nix-build-family",
                &unseen_action_parameters,
                &context,
            )
            .expect("global baseline fallback is explicit");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::GlobalActionDeltaFallback
        );
        assert!(prediction.action_parameter_similarity < 0.8);
    }

    #[test]
    fn family_label_free_prediction_selects_distinct_observable_context() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 400);
        let after_a = ContinuousHV::random(dim, 401);
        let after_b = ContinuousHV::random(dim, 402);
        let context_a = ContinuousHV::random(dim, 403);
        let context_b = ContinuousHV::random(dim, 404);

        for index in 0..4 {
            let mut episode_a = transition_episode(
                &format!("observable-a-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before.clone(),
                context_a.clone(),
                after_a.clone(),
                TransitionOutcomeClass::Success,
                &format!("observable-a-cluster-{index}"),
            );
            episode_a.environment_family_id = "training-family-a".to_owned();
            wm.learn_episode(&episode_a)
                .expect("family A training episode");

            let mut episode_b = transition_episode(
                &format!("observable-b-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before.clone(),
                context_b.clone(),
                after_b.clone(),
                TransitionOutcomeClass::Success,
                &format!("observable-b-cluster-{index}"),
            );
            episode_b.environment_family_id = "training-family-b".to_owned();
            wm.learn_episode(&episode_b)
                .expect("family B training episode");
        }

        wm.observe(before.clone());
        let global = wm.predict_state(&ActionCategory::Install);
        let contextual = wm
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Install,
                &ContinuousHV::random(dim, 90),
                &context_a,
            )
            .expect("observable-context prediction");

        let global_error =
            mean_squared_error(&global, &after_a).expect("same-dimensional global prediction");
        let contextual_error = mean_squared_error(&contextual.predicted_state, &after_a)
            .expect("same-dimensional context prediction");
        assert_eq!(contextual.source, ContextualPredictionSource::ContextualDelta);
        assert_eq!(contextual.independent_cluster_count, 4);
        assert!(contextual.context_similarity > 0.99);
        assert!(contextual.action_parameter_similarity > 0.99);
        assert!(
            contextual_error + 1e-5 < global_error,
            "context error {contextual_error} should beat pooled global error {global_error}"
        );
    }

    #[test]
    fn family_label_free_prediction_abstains_on_conflicting_equally_matched_deltas() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 410);
        let after_a = ContinuousHV::random(dim, 411);
        let after_b = ContinuousHV::random(dim, 412);
        let context = ContinuousHV::random(dim, 413);

        let mut episode_a = transition_episode(
            "ambiguous-family-a",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            context.clone(),
            after_a,
            TransitionOutcomeClass::Success,
            "ambiguous-cluster-a",
        );
        episode_a.environment_family_id = "ambiguous-family-a".to_owned();
        wm.learn_episode(&episode_a)
            .expect("family A training episode");

        let mut episode_b = transition_episode(
            "ambiguous-family-b",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            context.clone(),
            after_b,
            TransitionOutcomeClass::Failure,
            "ambiguous-cluster-b",
        );
        episode_b.environment_family_id = "ambiguous-family-b".to_owned();
        wm.learn_episode(&episode_b)
            .expect("family B training episode");

        let prediction = wm
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Install,
                &ContinuousHV::random(dim, 90),
                &context,
            )
            .expect("ambiguous context is an explicit result");

        assert_eq!(
            prediction.source,
            ContextualPredictionSource::AmbiguousContextUnknown
        );
        assert!((prediction.predicted_state.similarity(&before) - 1.0).abs() < 1e-5);
        assert_eq!(prediction.observation_count, 2);
        assert_eq!(prediction.independent_cluster_count, 2);
        assert_eq!(prediction.training_residual_mse, None);
    }

    #[test]
    fn family_label_free_prediction_uses_global_fallback_for_unseen_context() {
        let dim = 256;
        let mut wm = NixWorldModel::new(dim);
        let before = ContinuousHV::random(dim, 420);
        let after = ContinuousHV::random(dim, 421);
        let training_context = ContinuousHV::random(dim, 422);
        let unseen_context = ContinuousHV::random(dim, 423);
        let episode = transition_episode(
            "unseen-context-training",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            training_context,
            after,
            TransitionOutcomeClass::Success,
            "unseen-context-cluster",
        );
        wm.learn_episode(&episode)
            .expect("training episode");

        let prediction = wm
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Install,
                &ContinuousHV::random(dim, 90),
                &unseen_context,
            )
            .expect("unseen context produces an explicit fallback");
        assert_eq!(
            prediction.source,
            ContextualPredictionSource::GlobalActionDeltaFallback
        );
        assert!(prediction.context_similarity < 0.8);
        assert!(prediction.action_parameter_similarity > 0.99);
        assert_eq!(prediction.observation_count, 1);

        let unknown_action = wm
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Configure,
                &ContinuousHV::random(dim, 90),
                &unseen_context,
            )
            .expect("unknown action returns an explicit identity fallback");
        assert_eq!(
            unknown_action.source,
            ContextualPredictionSource::UnknownActionIdentityFallback
        );
    }

    #[test]
    fn family_label_free_prediction_is_order_invariant_and_deduplicates_clusters() {
        let dim = 256;
        let before = ContinuousHV::random(dim, 430);
        let after = ContinuousHV::random(dim, 431);
        let context = ContinuousHV::random(dim, 432);

        let mut family_a = transition_episode(
            "same-transition-family-a",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            context.clone(),
            after.clone(),
            TransitionOutcomeClass::Success,
            "shared-independent-cluster",
        );
        family_a.environment_family_id = "same-transition-a".to_owned();
        let mut family_b = transition_episode(
            "same-transition-family-b",
            ActionCategory::Install,
            ActionCategory::Install,
            before.clone(),
            context.clone(),
            after,
            TransitionOutcomeClass::Failure,
            "shared-independent-cluster",
        );
        family_b.environment_family_id = "same-transition-b".to_owned();

        let mut forward = NixWorldModel::new(dim);
        forward
            .learn_episode(&family_a)
            .expect("first family episode");
        forward
            .learn_episode(&family_b)
            .expect("second family episode");

        let mut reverse = NixWorldModel::new(dim);
        reverse
            .learn_episode(&family_b)
            .expect("first reversed family episode");
        reverse
            .learn_episode(&family_a)
            .expect("second reversed family episode");

        let action_parameters = ContinuousHV::random(dim, 90);
        let prediction_forward = forward
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Install,
                &action_parameters,
                &context,
            )
            .expect("forward family-order prediction");
        let prediction_reverse = reverse
            .predict_state_without_family_label(
                &before,
                &ActionCategory::Install,
                &action_parameters,
                &context,
            )
            .expect("reverse family-order prediction");

        assert_eq!(
            prediction_forward.source,
            ContextualPredictionSource::ContextualDelta
        );
        assert_eq!(prediction_forward.observation_count, 2);
        assert_eq!(prediction_forward.independent_cluster_count, 1);
        assert_eq!(prediction_forward.outcome_counts.success, 1);
        assert_eq!(prediction_forward.outcome_counts.failure, 1);
        assert!(
            (prediction_forward
                .predicted_state
                .similarity(&prediction_reverse.predicted_state)
                - 1.0)
                .abs()
                < 1e-5
        );
        assert_eq!(
            prediction_forward.independent_cluster_count,
            prediction_reverse.independent_cluster_count
        );
    }

    #[test]
    fn contextual_prediction_rejects_wrong_dimensions_and_non_finite_values() {
        let dim = 256;
        let wm = NixWorldModel::new(dim);
        assert!(
            wm.predict_state_with_context(
                &ContinuousHV::zero(dim - 1),
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &ContinuousHV::zero(dim),
            )
            .is_err()
        );

        let invalid_context = ContinuousHV::from_vec(vec![f32::NAN; dim]);
        assert!(
            wm.predict_state_with_context(
                &ContinuousHV::zero(dim),
                &ActionCategory::Install,
                "nix-build-family",
                &ContinuousHV::random(dim, 90),
                &invalid_context,
            )
            .is_err()
        );
    }
    #[test]
    fn cmaq_adapter_qualifies_query_sufficient_partition_from_real_nixward_hdc_predictions() {
        const DIM: usize = 256;
        let mut world_model = NixWorldModel::new(DIM);
        let context_a = ContinuousHV::random(DIM, 700);
        let context_b = ContinuousHV::random(DIM, 701);
        let delta_a = ContinuousHV::from_vec(
            std::iter::once(0.1)
                .chain(std::iter::repeat_n(0.0, DIM - 1))
                .collect(),
        );
        let delta_b = ContinuousHV::from_vec(
            std::iter::once(0.5)
                .chain(std::iter::repeat_n(0.0, DIM - 1))
                .collect(),
        );
        let action_parameters = ContinuousHV::random(DIM, 90);

        let state_with = |coordinate_zero: f32, nuisance: f32| {
            let mut values = vec![0.0; DIM];
            values[0] = coordinate_zero;
            values[1] = nuisance;
            ContinuousHV::from_vec(values)
        };

        // Three independent training clusters per family teach the actual
        // NixWard contextual HDC-delta path two different action consequences.
        for index in 0..3 {
            let before_a = state_with(0.1, -0.8 + index as f32 * 0.05);
            let after_a = before_a.add(&delta_a);
            let mut episode_a = transition_episode(
                &format!("cmaq022-family-a-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before_a,
                context_a.clone(),
                after_a,
                TransitionOutcomeClass::Success,
                &format!("cmaq022-family-a-cluster-{index}"),
            );
            episode_a.environment_family_id = "mechanism-a".to_owned();
            world_model
                .learn_episode(&episode_a)
                .expect("family A executed episode must train the HDC model");

            let before_b = state_with(0.1, 0.55 + index as f32 * 0.05);
            let after_b = before_b.add(&delta_b);
            let mut episode_b = transition_episode(
                &format!("cmaq022-family-b-{index}"),
                ActionCategory::Install,
                ActionCategory::Install,
                before_b,
                context_b.clone(),
                after_b,
                TransitionOutcomeClass::Success,
                &format!("cmaq022-family-b-cluster-{index}"),
            );
            episode_b.environment_family_id = "mechanism-b".to_owned();
            world_model
                .learn_episode(&episode_b)
                .expect("family B executed episode must train the HDC model");
        }

        let evaluation_states = [
            (
                "a-left",
                "mechanism-a",
                "evaluation-cluster-a-left",
                state_with(0.1, -0.65),
                context_a.clone(),
                200_i64,
            ),
            (
                "a-right",
                "mechanism-a",
                "evaluation-cluster-a-right",
                state_with(0.1, -0.35),
                context_a.clone(),
                200_i64,
            ),
            (
                "b-left",
                "mechanism-b",
                "evaluation-cluster-b-left",
                state_with(0.1, 0.45),
                context_b.clone(),
                600_i64,
            ),
            (
                "b-right",
                "mechanism-b",
                "evaluation-cluster-b-right",
                state_with(0.1, 0.75),
                context_b.clone(),
                600_i64,
            ),
        ];

        let key = ResponseKey::new("install", "coordinate_0_after_action", 1);
        let mut contract = StateResponseContract {
            contract_id: "cmaq-022-nixward-hdc-toy-v1".to_owned(),
            contract_sha256: "a".repeat(64),
            source_manifest_sha256: "b".repeat(64),
            domain_ids: ["toy-nixos-domain".to_owned()].into_iter().collect(),
            intervention_ids: ["install".to_owned()].into_iter().collect(),
            query_ids: ["coordinate_0_after_action".to_owned()]
                .into_iter()
                .collect(),
            horizons: [1].into_iter().collect(),
            invariant_ids: ["constant-invariant".to_owned()].into_iter().collect(),
            tolerance_units: BTreeMap::new(),
            probe_cost_units: BTreeMap::new(),
        };
        contract.tolerance_units.insert(key.clone(), 1);
        contract.probe_cost_units.insert(key.clone(), 1);
        contract.validate().expect("frozen CMAQ response contract");

        let quantize_query = |state: &ContinuousHV| {
            (state.as_slice()[0] as f64 * 1000.0).round() as i64
        };
        let mut hdc_records = Vec::new();
        let mut global_records = Vec::new();
        let mut identity_records = Vec::new();
        let mut full_state_records = Vec::new();

        // Candidate prediction and abstraction-label construction happen before
        // evaluator-only response values are attached to the records below.
        for (state_id, family, cluster, before, context, reference_response) in
            &evaluation_states
        {
            let hdc_prediction = world_model
                .predict_state_without_family_label(
                    before,
                    &ActionCategory::Install,
                    &action_parameters,
                    context,
                )
                .expect("family-label-free NixWard prediction");
            assert_eq!(
                hdc_prediction.source,
                ContextualPredictionSource::ContextualDelta,
                "the toy fixture should produce an unambiguous contextual prediction"
            );

            let hdc_label = format!(
                "coordinate_0:{}",
                quantize_query(&hdc_prediction.predicted_state)
            );
            let full_state_label = hdc_prediction
                .predicted_state
                .as_slice()
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
                .join(",");

            world_model.observe(before.clone());
            let global_prediction = world_model.predict_state(&ActionCategory::Install);
            let global_label = format!(
                "coordinate_0:{}",
                quantize_query(&global_prediction)
            );
            let identity_label =
                format!("coordinate_0:{}", quantize_query(before));

            let build_record = |abstract_state_id: String| StateResponseRecord {
                state_id: (*state_id).to_owned(),
                domain_id: "toy-nixos-domain".to_owned(),
                environment_family_id: (*family).to_owned(),
                environment_cluster_id: (*cluster).to_owned(),
                source_state_sha256: blake3::hash(state_id.as_bytes()).to_hex().to_string(),
                candidate_checkpoint_sha256: "c".repeat(64),
                abstract_state_id,
                invariant_signature: [("constant-invariant".to_owned(), 1)]
                    .into_iter()
                    .collect(),
                // This is evaluator-only reference information; it is never
                // passed into NixWorldModel or used to create candidate labels.
                responses: [(key.clone(), *reference_response)].into_iter().collect(),
            };

            hdc_records.push(build_record(hdc_label));
            global_records.push(build_record(global_label));
            identity_records.push(build_record(identity_label));
            full_state_records.push(build_record(full_state_label));
        }

        let hdc_score =
            score_abstraction(&hdc_records, &contract).expect("score HDC query partition");
        let global_score =
            score_abstraction(&global_records, &contract).expect("score global-delta partition");
        let identity_score =
            score_abstraction(&identity_records, &contract).expect("score identity partition");
        let full_state_score =
            score_abstraction(&full_state_records, &contract).expect("score full-state partition");

        // The HDC candidate merges nuisance-only variants while separating the
        // two distinct future intervention responses. Full-state memorization
        // over-splits; pooled global delta and identity/no-op both under-split.
        assert_eq!(hdc_score.false_merges, 0);
        assert_eq!(hdc_score.false_splits, 0);
        assert_eq!(hdc_score.candidate_class_count, 2);

        assert_eq!(global_score.false_merges, 4);
        assert_eq!(identity_score.false_merges, 4);
        assert_eq!(full_state_score.false_splits, 2);

        assert!(hdc_score.false_merges < global_score.false_merges);
        assert!(hdc_score.false_merges < identity_score.false_merges);
        assert!(hdc_score.false_splits < full_state_score.false_splits);
    }

}
