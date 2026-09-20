// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Memory — High-Φ System Event Consolidation
//!
//! Stores significant system events (builds, failures, rollbacks) with their
//! HDC context. Only high-Φ moments are stored (following the consciousness-
//! weighted consolidation pattern). Enables learning from past experience:
//! "last time this pattern led to failure."
//!
//! Human-readable action text is presentation data. Planner-conditioned learning
//! uses exact `ActionCategory` identity stored alongside the public episode so
//! Rust `Debug` spelling and substring coincidence cannot become semantics.

#[cfg(feature = "native")]
use crate::action::executor::NixOSCommand;
use super::world_model::ActionCategory;
use std::ops::Deref;
use symthaea_core::hdc::ContinuousHV;

/// Outcome of a system episode.
#[derive(Debug, Clone, PartialEq)]
pub enum EpisodeOutcome {
    /// Action succeeded.
    Success,
    /// Action failed with a reason.
    Failure(String),
    /// Action partially succeeded.
    PartialSuccess(String),
    /// Action was rolled back.
    RolledBack(String),
}

/// A single episodic memory — a state transition with context.
#[derive(Debug, Clone)]
pub struct SystemEpisode {
    /// System state before the action (HDC encoded).
    pub state_before: ContinuousHV,
    /// Human-readable action/event description.
    ///
    /// This field is presentation/legacy data only. It is deliberately not used
    /// as planner-action identity by `NixEpisodicMemory`.
    pub action: String,
    /// System state after the action (HDC encoded).
    pub state_after: ContinuousHV,
    /// What happened.
    pub outcome: EpisodeOutcome,
    /// Consciousness level when this was encoded.
    pub phi_at_encoding: f64,
    /// How surprising the outcome was (prediction error).
    pub prediction_error: f64,
    /// Positive (success) or negative (failure) valence.
    pub emotional_valence: f64,
    /// When this happened (unix timestamp).
    pub timestamp: i64,
}

/// Internal storage wrapper separating presentation text from exact planner semantics.
///
/// Keeping this wrapper private preserves the public `SystemEpisode` shape for the
/// actor/hippocampus bridge while allowing action-conditioned learning to use exact
/// typed identity. Episodes recorded through the legacy/public `record()` path have
/// no planner identity and therefore cannot influence planner-action-specific valence.
#[derive(Debug, Clone)]
struct StoredEpisode {
    episode: SystemEpisode,
    planner_action: Option<ActionCategory>,
}

impl Deref for StoredEpisode {
    type Target = SystemEpisode;

    fn deref(&self) -> &Self::Target {
        &self.episode
    }
}

/// In-memory episodic memory store.
///
/// Stores episodes above a Φ threshold. Higher prediction error =
/// higher priority for storage (surprising events are more memorable).
pub struct NixEpisodicMemory {
    /// Stored episodes plus private semantic provenance.
    episodes: Vec<StoredEpisode>,
    /// Minimum Φ for storage.
    phi_threshold: f64,
    /// Maximum number of episodes to retain in memory.
    max_episodes: usize,
}

impl NixEpisodicMemory {
    /// Create a new episodic memory with default settings.
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
            phi_threshold: 0.3,
            max_episodes: 1000,
        }
    }

    /// Create with custom Φ threshold.
    pub fn with_phi_threshold(phi_threshold: f64) -> Self {
        Self {
            phi_threshold,
            ..Self::new()
        }
    }

    fn record_with_planner_action(
        &mut self,
        episode: SystemEpisode,
        planner_action: Option<ActionCategory>,
    ) -> bool {
        if episode.phi_at_encoding < self.phi_threshold {
            return false;
        }

        self.episodes.push(StoredEpisode {
            episode,
            planner_action,
        });

        // If over capacity, evict lowest-importance episodes
        if self.episodes.len() > self.max_episodes {
            self.consolidate();
        }

        true
    }

    /// Record an episode without asserting typed planner-action provenance.
    ///
    /// This remains the compatibility path for observational events, working-memory
    /// graduation, actor/hippocampus callers, and historical display-only episodes.
    /// Such episodes participate in state-similarity memory but never in exact
    /// planner-action-conditioned valence.
    pub fn record(&mut self, episode: SystemEpisode) -> bool {
        self.record_with_planner_action(episode, None)
    }

    /// Record an already-built episode with exact planner-action provenance.
    ///
    /// The `SystemEpisode::action` string remains display-only; the supplied
    /// `ActionCategory` is the semantic key used for exact action-conditioned
    /// retrieval and valence prediction.
    pub fn record_planner_episode(
        &mut self,
        episode: SystemEpisode,
        planner_action: ActionCategory,
    ) -> bool {
        self.record_with_planner_action(episode, Some(planner_action))
    }

    /// Record an abstract planner action outcome without fabricating an executable
    /// `NixOSCommand` merely to satisfy the memory API.
    pub fn record_planner_transition(
        &mut self,
        state_before: ContinuousHV,
        planner_action: ActionCategory,
        state_after: ContinuousHV,
        outcome: EpisodeOutcome,
        phi: f64,
        prediction_error: f64,
    ) -> bool {
        let valence = outcome_valence(&outcome);
        let display = planner_action.to_string();
        let episode = SystemEpisode {
            state_before,
            action: display,
            state_after,
            outcome,
            phi_at_encoding: phi,
            prediction_error,
            emotional_valence: valence,
            timestamp: episode_timestamp(),
        };

        self.record_with_planner_action(episode, Some(planner_action))
    }

    /// Record from an actual command (convenience/legacy method).
    ///
    /// A concrete command is not automatically equivalent to an abstract planner
    /// category, so this path intentionally records no planner-action identity.
    #[cfg(feature = "native")]
    pub fn record_transition(
        &mut self,
        state_before: ContinuousHV,
        action: &NixOSCommand,
        state_after: ContinuousHV,
        outcome: EpisodeOutcome,
        phi: f64,
        prediction_error: f64,
    ) -> bool {
        let episode = SystemEpisode {
            state_before,
            action: format!("{action:?}"),
            state_after,
            outcome: outcome.clone(),
            phi_at_encoding: phi,
            prediction_error,
            emotional_valence: outcome_valence(&outcome),
            timestamp: episode_timestamp(),
        };

        self.record(episode)
    }

    /// Retrieve episodes similar to a query state.
    ///
    /// Returns episodes whose before-state is similar to the query,
    /// sorted by similarity (most similar first).
    pub fn retrieve_similar(&self, query: &ContinuousHV, limit: usize) -> Vec<&SystemEpisode> {
        let mut scored: Vec<(f64, &SystemEpisode)> = self
            .episodes
            .iter()
            .map(|stored| {
                let ep = &stored.episode;
                let sim = ep.state_before.similarity(query) as f64;
                (sim, ep)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().take(limit).map(|(_, ep)| ep).collect()
    }

    /// Retrieve episodes involving a presentation-text pattern.
    ///
    /// This is retained for compatibility/search UI only. It is not semantic
    /// planner-action matching and must not be used for action-conditioned policy.
    pub fn retrieve_by_action(&self, action_pattern: &str) -> Vec<&SystemEpisode> {
        self.episodes
            .iter()
            .filter(|stored| stored.episode.action.contains(action_pattern))
            .map(|stored| &stored.episode)
            .collect()
    }

    /// Retrieve episodes bound to exactly one planner action category.
    pub fn retrieve_by_planner_action(
        &self,
        planner_action: &ActionCategory,
    ) -> Vec<&SystemEpisode> {
        self.episodes
            .iter()
            .filter(|stored| stored.planner_action.as_ref() == Some(planner_action))
            .map(|stored| &stored.episode)
            .collect()
    }

    /// Get all failure episodes (for learning what to avoid).
    pub fn failures(&self) -> Vec<&SystemEpisode> {
        self.episodes
            .iter()
            .filter(|stored| matches!(stored.episode.outcome, EpisodeOutcome::Failure(_)))
            .map(|stored| &stored.episode)
            .collect()
    }

    /// Compute the average outcome valence for episodes similar to a state.
    ///
    /// Positive = past similar states led to success.
    /// Negative = past similar states led to failure.
    /// Returns 0.0 when no prior experience exists or when similarity
    /// values are degenerate (NaN/Inf).
    pub fn predict_valence(&self, state: &ContinuousHV) -> f64 {
        let similar = self.retrieve_similar(state, 5);
        weighted_valence(state, &similar)
    }

    /// Compute average valence for episodes with the exact planner action category.
    ///
    /// This is the semantic action-conditioned learning path. Display/debug strings
    /// are never consulted.
    pub fn predict_valence_for_planner_action(
        &self,
        state: &ContinuousHV,
        planner_action: &ActionCategory,
    ) -> f64 {
        let mut scored: Vec<(f64, &SystemEpisode)> = self
            .episodes
            .iter()
            .filter(|stored| stored.planner_action.as_ref() == Some(planner_action))
            .map(|stored| {
                let ep = &stored.episode;
                (ep.state_before.similarity(state) as f64, ep)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let similar: Vec<&SystemEpisode> = scored
            .into_iter()
            .take(5)
            .map(|(_, ep)| ep)
            .collect();
        weighted_valence(state, &similar)
    }

    /// Legacy display-text conditioned valence.
    ///
    /// Retained temporarily for source compatibility. New cognitive/policy code must
    /// use `predict_valence_for_planner_action`; this method performs exact display
    /// equality rather than the former bidirectional substring heuristic.
    #[deprecated(
        note = "display strings are not semantic action identity; use predict_valence_for_planner_action"
    )]
    pub fn predict_valence_for_action(&self, state: &ContinuousHV, action_text: &str) -> f64 {
        let similar: Vec<&SystemEpisode> = self
            .retrieve_similar(state, 10)
            .into_iter()
            .filter(|ep| ep.action == action_text)
            .take(5)
            .collect();
        weighted_valence(state, &similar)
    }

    /// Consolidate memory — keep high-importance episodes, evict low ones.
    fn consolidate(&mut self) {
        // Sort by importance: prediction_error * phi (surprising, conscious moments)
        self.episodes.sort_by(|a, b| {
            let imp_a = a.episode.prediction_error * a.episode.phi_at_encoding;
            let imp_b = b.episode.prediction_error * b.episode.phi_at_encoding;
            imp_b.total_cmp(&imp_a)
        });

        // Keep only max_episodes
        self.episodes.truncate(self.max_episodes);
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Total number of failure episodes.
    pub fn failure_count(&self) -> usize {
        self.episodes
            .iter()
            .filter(|stored| matches!(stored.episode.outcome, EpisodeOutcome::Failure(_)))
            .count()
    }
}

fn outcome_valence(outcome: &EpisodeOutcome) -> f64 {
    match outcome {
        EpisodeOutcome::Success => 1.0,
        EpisodeOutcome::PartialSuccess(_) => 0.3,
        EpisodeOutcome::Failure(_) => -1.0,
        EpisodeOutcome::RolledBack(_) => -0.5,
    }
}

fn weighted_valence(state: &ContinuousHV, episodes: &[&SystemEpisode]) -> f64 {
    if episodes.is_empty() {
        return 0.0;
    }

    let total_weight: f64 = episodes
        .iter()
        .map(|ep| {
            let sim = ep.state_before.similarity(state).max(0.0) as f64;
            if sim.is_finite() { sim } else { 0.0 }
        })
        .sum();

    if !total_weight.is_finite() || total_weight < 1e-6 {
        return 0.0;
    }

    let weighted: f64 = episodes
        .iter()
        .map(|ep| {
            let sim = ep.state_before.similarity(state).max(0.0) as f64;
            if sim.is_finite() {
                sim * ep.emotional_valence
            } else {
                0.0
            }
        })
        .sum();

    let result = weighted / total_weight;
    if result.is_finite() { result } else { 0.0 }
}

fn episode_timestamp() -> i64 {
    #[cfg(feature = "native")]
    {
        chrono::Utc::now().timestamp()
    }
    #[cfg(not(feature = "native"))]
    {
        0
    }
}

impl Default for NixEpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(1024, seed)
    }

    fn make_episode(seed: u64, outcome: EpisodeOutcome, phi: f64) -> SystemEpisode {
        let emotional_valence = match &outcome {
            EpisodeOutcome::Success => 1.0,
            EpisodeOutcome::Failure(_) => -1.0,
            _ => 0.0,
        };
        SystemEpisode {
            state_before: make_hv(seed),
            action: format!("action_{}", seed),
            state_after: make_hv(seed + 1000),
            outcome,
            phi_at_encoding: phi,
            prediction_error: 0.5,
            emotional_valence,
            timestamp: 0,
        }
    }

    #[test]
    fn test_phi_gating() {
        let mut mem = NixEpisodicMemory::with_phi_threshold(0.5);

        let low_phi = make_episode(1, EpisodeOutcome::Success, 0.2);
        assert!(!mem.record(low_phi));
        assert_eq!(mem.len(), 0);

        let high_phi = make_episode(2, EpisodeOutcome::Success, 0.8);
        assert!(mem.record(high_phi));
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_retrieve_similar() {
        let mut mem = NixEpisodicMemory::new();

        let ep1 = make_episode(1, EpisodeOutcome::Success, 0.5);
        let state1 = ep1.state_before.clone();
        mem.record(ep1);

        let ep2 = make_episode(100, EpisodeOutcome::Failure("err".into()), 0.5);
        mem.record(ep2);

        let results = mem.retrieve_similar(&state1, 1);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].outcome, EpisodeOutcome::Success));
    }

    #[test]
    fn test_failures() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(
            2,
            EpisodeOutcome::Failure("build failed".into()),
            0.5,
        ));
        mem.record(make_episode(
            3,
            EpisodeOutcome::Failure("hash mismatch".into()),
            0.5,
        ));

        assert_eq!(mem.failure_count(), 2);
        assert_eq!(mem.failures().len(), 2);
    }

    #[test]
    fn test_predict_valence() {
        let mut mem = NixEpisodicMemory::new();

        for i in 0..5 {
            mem.record(make_episode(1 + i, EpisodeOutcome::Success, 0.5));
        }

        for i in 0..5 {
            mem.record(make_episode(
                100 + i,
                EpisodeOutcome::Failure("err".into()),
                0.5,
            ));
        }

        let valence = mem.predict_valence(&make_hv(1));
        assert!(valence.is_finite());
        assert!(mem.len() == 10);
    }

    #[test]
    fn test_retrieve_by_action_matches() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(2, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(10, EpisodeOutcome::Success, 0.5));

        let results = mem.retrieve_by_action("action_2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, "action_2");
    }

    #[test]
    fn test_retrieve_by_action_partial_match_is_display_only() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(2, EpisodeOutcome::Success, 0.5));

        let results = mem.retrieve_by_action("action_");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_retrieve_by_action_no_match() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));

        let results = mem.retrieve_by_action("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_by_action_empty_memory() {
        let mem = NixEpisodicMemory::new();
        let results = mem.retrieve_by_action("action_1");
        assert!(results.is_empty());
    }

    #[test]
    fn planner_action_identity_is_exact_and_independent_of_display() {
        let mut mem = NixEpisodicMemory::new();
        let state = make_hv(50);
        let episode = SystemEpisode {
            state_before: state.clone(),
            action: "this text intentionally does not say rebuild".into(),
            state_after: make_hv(51),
            outcome: EpisodeOutcome::Failure("boom".into()),
            phi_at_encoding: 0.8,
            prediction_error: 0.9,
            emotional_valence: -1.0,
            timestamp: 0,
        };
        assert!(mem.record_planner_episode(episode, ActionCategory::Rebuild));

        assert_eq!(mem.retrieve_by_planner_action(&ActionCategory::Rebuild).len(), 1);
        assert!(mem.retrieve_by_planner_action(&ActionCategory::Enable).is_empty());
        assert!(
            mem.predict_valence_for_planner_action(&state, &ActionCategory::Rebuild) < 0.0
        );
        assert_eq!(
            mem.predict_valence_for_planner_action(&state, &ActionCategory::Enable),
            0.0
        );
    }

    #[test]
    fn custom_display_cannot_collide_with_standard_planner_action() {
        let mut mem = NixEpisodicMemory::new();
        let state = make_hv(60);
        assert!(mem.record_planner_transition(
            state.clone(),
            ActionCategory::Custom("Enable".into()),
            make_hv(61),
            EpisodeOutcome::Failure("custom failed".into()),
            0.8,
            0.9,
        ));

        assert!(
            mem.predict_valence_for_planner_action(
                &state,
                &ActionCategory::Custom("Enable".into())
            ) < 0.0
        );
        assert_eq!(
            mem.predict_valence_for_planner_action(&state, &ActionCategory::Enable),
            0.0
        );
    }

    #[test]
    fn untyped_display_episode_cannot_influence_planner_specific_valence() {
        let mut mem = NixEpisodicMemory::new();
        let state = make_hv(70);
        let mut episode = make_episode(70, EpisodeOutcome::Failure("failed".into()), 0.8);
        episode.state_before = state.clone();
        episode.action = "Enable".into();
        episode.emotional_valence = -1.0;
        assert!(mem.record(episode));

        assert_eq!(
            mem.predict_valence_for_planner_action(&state, &ActionCategory::Enable),
            0.0,
            "presentation text alone must never acquire typed planner semantics"
        );
    }

    #[test]
    fn planner_transition_respects_phi_gate() {
        let mut mem = NixEpisodicMemory::with_phi_threshold(0.5);
        assert!(!mem.record_planner_transition(
            make_hv(80),
            ActionCategory::Update,
            make_hv(81),
            EpisodeOutcome::Success,
            0.2,
            0.5,
        ));
        assert!(mem.retrieve_by_planner_action(&ActionCategory::Update).is_empty());
    }

    #[test]
    fn test_consolidation_keeps_important() {
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: 3,
        };

        for i in 0..5 {
            let mut ep = make_episode(i, EpisodeOutcome::Success, 0.5);
            ep.prediction_error = (i + 1) as f64 * 0.2;
            mem.record(ep);
        }

        assert_eq!(mem.len(), 3);
        for ep in &mem.episodes {
            assert!(
                ep.prediction_error >= 0.6,
                "Low-importance episodes should have been evicted, got pe={:.1}",
                ep.prediction_error
            );
        }
    }

    #[test]
    fn test_episode_outcome_variants() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(
            2,
            EpisodeOutcome::PartialSuccess("mostly worked".into()),
            0.5,
        ));
        mem.record(make_episode(
            3,
            EpisodeOutcome::RolledBack("reverted".into()),
            0.5,
        ));
        mem.record(make_episode(
            4,
            EpisodeOutcome::Failure("crashed".into()),
            0.5,
        ));

        assert_eq!(mem.len(), 4);
        assert_eq!(mem.failure_count(), 1);
    }

    #[test]
    fn test_predict_valence_empty_memory() {
        let mem = NixEpisodicMemory::new();
        let val = mem.predict_valence(&make_hv(42));
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_valence_returns_finite() {
        let mut mem = NixEpisodicMemory::new();
        for i in 0..10 {
            let outcome = if i % 2 == 0 {
                EpisodeOutcome::Success
            } else {
                EpisodeOutcome::Failure("err".into())
            };
            mem.record(make_episode(i, outcome, 0.5));
        }
        let val = mem.predict_valence(&make_hv(5));
        assert!(
            val.is_finite(),
            "predict_valence must always return finite, got {val}"
        );
        assert!((-1.0..=1.0).contains(&val), "valence out of range: {val}");
    }

    #[test]
    fn test_predict_valence_zero_vector() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        let zero = ContinuousHV::zero(1024);
        let val = mem.predict_valence(&zero);
        assert!(
            val.is_finite(),
            "predict_valence on zero vector should be finite"
        );
    }

    #[test]
    fn test_phi_zero_never_writes() {
        let mut mem = NixEpisodicMemory::with_phi_threshold(0.3);

        for i in 0u64..50 {
            let ep = make_episode(i, EpisodeOutcome::Success, 0.0);
            let stored = mem.record(ep);
            assert!(!stored, "Φ=0 must never be stored (step {i})");
        }

        assert_eq!(
            mem.len(),
            0,
            "memory must be empty after 50 Φ=0 submissions"
        );
    }

    #[test]
    fn test_phi_exactly_at_threshold_is_stored() {
        let mut mem = NixEpisodicMemory::with_phi_threshold(0.5);

        let ep = make_episode(1, EpisodeOutcome::Success, 0.5);
        assert!(
            mem.record(ep),
            "episode at exactly the threshold must be stored"
        );
        assert_eq!(mem.len(), 1);

        let ep2 = make_episode(2, EpisodeOutcome::Success, 0.4999);
        assert!(
            !mem.record(ep2),
            "episode just below threshold must not be stored"
        );
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_capacity_eviction_respects_max_episodes() {
        let max = 5usize;
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: max,
        };

        for i in 0u64..(max as u64 + 3) {
            let mut ep = make_episode(i, EpisodeOutcome::Success, 0.5);
            ep.prediction_error = (i + 1) as f64 * 0.1;
            mem.record(ep);
        }

        assert_eq!(
            mem.len(),
            max,
            "memory must not exceed max_episodes after overflow"
        );

        for ep in &mem.episodes {
            let importance = ep.prediction_error * ep.phi_at_encoding;
            assert!(
                importance.is_finite() && importance >= 0.0,
                "evicted episode importance must be finite and non-negative: {importance}"
            );
        }
    }

    #[test]
    fn test_predict_valence_all_failures_gives_negative_finite() {
        let mut mem = NixEpisodicMemory::new();
        let query_state = make_hv(1);

        for i in 0u64..10 {
            let mut ep = make_episode(1 + i, EpisodeOutcome::Failure("build error".into()), 0.5);
            ep.state_before = query_state.clone();
            mem.record(ep);
        }

        let val = mem.predict_valence(&query_state);
        assert!(val.is_finite(), "all-failure valence must be finite: {val}");
        assert!(val < 0.0, "all-failure valence must be negative: {val}");
        assert!(val >= -1.0, "valence must not go below -1.0: {val}");
    }

    #[test]
    fn test_predict_valence_always_in_unit_interval() {
        let mut mem = NixEpisodicMemory::new();

        for i in 0u64..20 {
            let outcome = match i % 4 {
                0 => EpisodeOutcome::Success,
                1 => EpisodeOutcome::Failure("err".into()),
                2 => EpisodeOutcome::PartialSuccess("partial".into()),
                _ => EpisodeOutcome::RolledBack("reverted".into()),
            };
            mem.record(make_episode(i, outcome, 0.5));
        }

        for seed in [1u64, 5, 10, 15, 20, 99, 999] {
            let val = mem.predict_valence(&make_hv(seed));
            assert!(
                val.is_finite(),
                "predict_valence must be finite for seed {seed}: {val}"
            );
            assert!(
                (-1.0..=1.0).contains(&val),
                "predict_valence out of [-1,1] for seed {seed}: {val}"
            );
        }
    }

    #[test]
    fn test_consolidation_idempotent() {
        let max = 3usize;
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: max,
        };

        for i in 0u64..max as u64 {
            let mut ep = make_episode(i, EpisodeOutcome::Success, 0.5);
            ep.prediction_error = (i + 1) as f64 * 0.2;
            mem.record(ep);
        }

        let len_before = mem.len();
        mem.consolidate();
        assert_eq!(
            mem.len(),
            len_before,
            "consolidation on non-overflowing memory must not remove episodes"
        );

        for ep in &mem.episodes {
            assert!(
                (ep.prediction_error * ep.phi_at_encoding).is_finite(),
                "importance must be finite after duplicate consolidation"
            );
        }
    }
}
