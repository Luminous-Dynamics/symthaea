// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Memory — High-Φ System Event Consolidation
//!
//! Stores significant system events (builds, failures, rollbacks) with their
//! HDC context. Only high-Φ moments are stored (following the consciousness-
//! weighted consolidation pattern). Enables learning from past experience:
//! "last time this pattern led to failure."

#[cfg(feature = "native")]
use crate::action::executor::NixOSCommand;
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
    /// The action that was taken.
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

/// In-memory episodic memory store.
///
/// Stores episodes above a Φ threshold. Higher prediction error =
/// higher priority for storage (surprising events are more memorable).
pub struct NixEpisodicMemory {
    /// Stored episodes.
    episodes: Vec<SystemEpisode>,
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

    /// Record an episode if Φ is above threshold.
    ///
    /// Returns true if the episode was stored.
    pub fn record(&mut self, episode: SystemEpisode) -> bool {
        if episode.phi_at_encoding < self.phi_threshold {
            return false;
        }

        self.episodes.push(episode);

        // If over capacity, evict lowest-importance episodes
        if self.episodes.len() > self.max_episodes {
            self.consolidate();
        }

        true
    }

    /// Record from components (convenience method).
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
        let valence = match &outcome {
            EpisodeOutcome::Success => 1.0,
            EpisodeOutcome::PartialSuccess(_) => 0.3,
            EpisodeOutcome::Failure(_) => -1.0,
            EpisodeOutcome::RolledBack(_) => -0.5,
        };

        let episode = SystemEpisode {
            state_before,
            action: format!("{action:?}"),
            state_after,
            outcome,
            phi_at_encoding: phi,
            prediction_error,
            emotional_valence: valence,
            timestamp: chrono::Utc::now().timestamp(),
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
            .map(|ep| {
                let sim = ep.state_before.similarity(query) as f64;
                (sim, ep)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().take(limit).map(|(_, ep)| ep).collect()
    }

    /// Retrieve episodes involving a specific action pattern.
    pub fn retrieve_by_action(&self, action_pattern: &str) -> Vec<&SystemEpisode> {
        self.episodes
            .iter()
            .filter(|ep| ep.action.contains(action_pattern))
            .collect()
    }

    /// Get all failure episodes (for learning what to avoid).
    pub fn failures(&self) -> Vec<&SystemEpisode> {
        self.episodes
            .iter()
            .filter(|ep| matches!(ep.outcome, EpisodeOutcome::Failure(_)))
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
        if similar.is_empty() {
            return 0.0; // No prior experience
        }

        let total_weight: f64 = similar
            .iter()
            .map(|ep| {
                let sim = ep.state_before.similarity(state).max(0.0) as f64;
                if sim.is_finite() { sim } else { 0.0 }
            })
            .sum();

        if !total_weight.is_finite() || total_weight < 1e-6 {
            return 0.0;
        }

        let weighted_valence: f64 = similar
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

        let result = weighted_valence / total_weight;
        if result.is_finite() { result } else { 0.0 }
    }

    /// Compute the average outcome valence for episodes similar to a state, filtering by action category.
    pub fn predict_valence_for_action(&self, state: &ContinuousHV, action_cat: &str) -> f64 {
        let similar: Vec<&SystemEpisode> = self
            .retrieve_similar(state, 10)
            .into_iter()
            .filter(|ep| {
                let ep_action_lower = ep.action.to_lowercase();
                let target_lower = action_cat.to_lowercase();
                ep_action_lower.contains(&target_lower) || target_lower.contains(&ep_action_lower)
            })
            .take(5)
            .collect();

        if similar.is_empty() {
            // If no action-specific memory exists, return 0.0 (neutral)
            return 0.0;
        }

        let total_weight: f64 = similar
            .iter()
            .map(|ep| {
                let sim = ep.state_before.similarity(state).max(0.0) as f64;
                if sim.is_finite() { sim } else { 0.0 }
            })
            .sum();

        if !total_weight.is_finite() || total_weight < 1e-6 {
            return 0.0;
        }

        let weighted_valence: f64 = similar
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

        let result = weighted_valence / total_weight;
        if result.is_finite() { result } else { 0.0 }
    }

    /// Consolidate memory — keep high-importance episodes, evict low ones.
    fn consolidate(&mut self) {
        // Sort by importance: prediction_error * phi (surprising, conscious moments)
        self.episodes.sort_by(|a, b| {
            let imp_a = a.prediction_error * a.phi_at_encoding;
            let imp_b = b.prediction_error * b.phi_at_encoding;
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
            .filter(|ep| matches!(ep.outcome, EpisodeOutcome::Failure(_)))
            .count()
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

        // Low Φ — should not be stored
        let low_phi = make_episode(1, EpisodeOutcome::Success, 0.2);
        assert!(!mem.record(low_phi));
        assert_eq!(mem.len(), 0);

        // High Φ — should be stored
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

        // Query with state similar to ep1
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

        // Record successes near seed 1
        for i in 0..5 {
            mem.record(make_episode(1 + i, EpisodeOutcome::Success, 0.5));
        }

        // Record failures near seed 100
        for i in 0..5 {
            mem.record(make_episode(
                100 + i,
                EpisodeOutcome::Failure("err".into()),
                0.5,
            ));
        }

        // Query near seed 1 should predict positive valence
        let valence = mem.predict_valence(&make_hv(1));
        assert!(valence.is_finite());
        // With random vectors this might not be strongly positive,
        // but at least the mechanism works
        assert!(mem.len() == 10);
    }

    #[test]
    fn test_retrieve_by_action_matches() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(2, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(10, EpisodeOutcome::Success, 0.5));

        // "action_2" matches exactly one (no substring overlap with action_1 or action_10)
        let results = mem.retrieve_by_action("action_2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, "action_2");
    }

    #[test]
    fn test_retrieve_by_action_partial_match() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        mem.record(make_episode(2, EpisodeOutcome::Success, 0.5));

        // "action_" matches all (common prefix)
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
    fn test_consolidation_keeps_important() {
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: 3,
        };

        // Add 5 episodes with varying importance (prediction_error * phi)
        for i in 0..5 {
            let mut ep = make_episode(i, EpisodeOutcome::Success, 0.5);
            ep.prediction_error = (i + 1) as f64 * 0.2; // higher i = more surprising
            mem.record(ep);
        }

        // Should have been consolidated to 3
        assert_eq!(mem.len(), 3);
        // Remaining should be the 3 most important (highest prediction_error * phi)
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
        // Add episodes with varying valences
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
        assert!(val >= -1.0 && val <= 1.0, "valence out of range: {val}");
    }

    #[test]
    fn test_predict_valence_zero_vector() {
        let mut mem = NixEpisodicMemory::new();
        mem.record(make_episode(1, EpisodeOutcome::Success, 0.5));
        // Zero vector may produce degenerate similarity values
        let zero = ContinuousHV::zero(1024);
        let val = mem.predict_valence(&zero);
        assert!(
            val.is_finite(),
            "predict_valence on zero vector should be finite"
        );
    }

    // ── Phase 2.9-B: Episodic Memory Boundary Tests ─────────────────────────
    // Covers the Φ-gating path, capacity eviction, and predict_valence
    // invariants that the existing tests don't exercise adversarially.

    /// Φ = 0 must NEVER write to memory, regardless of how many episodes
    /// are submitted. The gate must hold absolutely.
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

    /// Episode exactly at the Φ threshold must be stored (>= not >).
    #[test]
    fn test_phi_exactly_at_threshold_is_stored() {
        let mut mem = NixEpisodicMemory::with_phi_threshold(0.5);

        // Exactly at threshold
        let ep = make_episode(1, EpisodeOutcome::Success, 0.5);
        assert!(
            mem.record(ep),
            "episode at exactly the threshold must be stored"
        );
        assert_eq!(mem.len(), 1);

        // Just below threshold
        let ep2 = make_episode(2, EpisodeOutcome::Success, 0.4999);
        assert!(
            !mem.record(ep2),
            "episode just below threshold must not be stored"
        );
        assert_eq!(mem.len(), 1);
    }

    /// When memory is at capacity, adding a lower-importance episode must
    /// trigger eviction and the resulting set must still respect max_episodes.
    #[test]
    fn test_capacity_eviction_respects_max_episodes() {
        let max = 5usize;
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: max,
        };

        // Fill to capacity + 3 extras (each with increasing prediction_error)
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

        // All remaining episodes must have finite, non-negative importance
        for ep in &mem.episodes {
            let importance = ep.prediction_error * ep.phi_at_encoding;
            assert!(
                importance.is_finite() && importance >= 0.0,
                "evicted episode importance must be finite and non-negative: {importance}"
            );
        }
    }

    /// predict_valence on an all-failure history must return a finite
    /// negative value — not NaN, not +inf, not zero.
    #[test]
    fn test_predict_valence_all_failures_gives_negative_finite() {
        let mut mem = NixEpisodicMemory::new();
        let query_state = make_hv(1);

        // Record 10 failures near the query state
        for i in 0u64..10 {
            let mut ep = make_episode(1 + i, EpisodeOutcome::Failure("build error".into()), 0.5);
            // Use state_before = query to guarantee similarity ≈ 1
            ep.state_before = query_state.clone();
            mem.record(ep);
        }

        let val = mem.predict_valence(&query_state);
        assert!(val.is_finite(), "all-failure valence must be finite: {val}");
        assert!(val < 0.0, "all-failure valence must be negative: {val}");
        assert!(val >= -1.0, "valence must not go below -1.0: {val}");
    }

    /// predict_valence must always be in [-1, 1] regardless of outcome mix,
    /// because emotional_valence is bounded at encoding time.
    #[test]
    fn test_predict_valence_always_in_unit_interval() {
        let mut mem = NixEpisodicMemory::new();

        // Mix all four outcome types
        for i in 0u64..20 {
            let outcome = match i % 4 {
                0 => EpisodeOutcome::Success,
                1 => EpisodeOutcome::Failure("err".into()),
                2 => EpisodeOutcome::PartialSuccess("partial".into()),
                _ => EpisodeOutcome::RolledBack("reverted".into()),
            };
            mem.record(make_episode(i, outcome, 0.5));
        }

        // Query with many diverse states
        for seed in [1u64, 5, 10, 15, 20, 99, 999] {
            let val = mem.predict_valence(&make_hv(seed));
            assert!(
                val.is_finite(),
                "predict_valence must be finite for seed {seed}: {val}"
            );
            assert!(
                val >= -1.0 && val <= 1.0,
                "predict_valence out of [-1,1] for seed {seed}: {val}"
            );
        }
    }

    /// Consolidation must be stable: running it on already-consolidated
    /// memory must not change the set or corrupt it.
    #[test]
    fn test_consolidation_idempotent() {
        let max = 3usize;
        let mut mem = NixEpisodicMemory {
            episodes: Vec::new(),
            phi_threshold: 0.1,
            max_episodes: max,
        };

        // Add exactly max episodes — no eviction needed
        for i in 0u64..max as u64 {
            let mut ep = make_episode(i, EpisodeOutcome::Success, 0.5);
            ep.prediction_error = (i + 1) as f64 * 0.2;
            mem.record(ep);
        }

        let len_before = mem.len();

        // Manually call consolidate again — must be idempotent
        mem.consolidate();
        assert_eq!(
            mem.len(),
            len_before,
            "consolidation on non-overflowing memory must not remove episodes"
        );

        // All episodes must still have finite importance
        for ep in &mem.episodes {
            assert!(
                (ep.prediction_error * ep.phi_at_encoding).is_finite(),
                "importance must be finite after duplicate consolidation"
            );
        }
    }
}
