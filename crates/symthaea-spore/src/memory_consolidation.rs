// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory Consolidation — dream-driven learning for the Spore WASM kernel.
//!
//! Port of the desktop CantorDreamManager / memory consolidation subsystem
//! into a lightweight, WASM-compatible module. Records episodic memories
//! with Phi-weighted salience, consolidates them via dream-like replay,
//! and applies post-consolidation learning boosts.
//!
//! Fully WASM-compatible: no std::time, no filesystem, no tokio.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum number of episodes retained.
const DEFAULT_MAX_EPISODES: usize = 200;

/// Minimum salience to record an episode.
const SALIENCE_THRESHOLD: f32 = 0.05;

/// Number of top-salience episodes to replay per consolidation.
const CONSOLIDATION_BATCH_SIZE: usize = 10;

/// Learning rate boost multiplier after consolidation.
const CONSOLIDATION_LR_BOOST: f32 = 1.5;

/// Number of cycles the learning boost persists after consolidation.
const BOOST_DURATION_CYCLES: u32 = 20;

/// Salience formula weight for Phi.
const PHI_WEIGHT: f32 = 0.6;

/// Salience formula weight for prediction error.
const PE_WEIGHT: f32 = 0.4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An episodic memory entry with salience scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEntry {
    /// Human-readable description of the episode.
    pub description: String,
    /// Consciousness level (Phi) at recording time.
    pub phi_at_recording: f32,
    /// Prediction error at recording time.
    pub prediction_error: f32,
    /// Cycle at which this episode was recorded.
    pub cycle: u64,
    /// Salience score: weighted combination of Phi and prediction error.
    pub salience: f32,
    /// Whether this episode has been consolidated (dream-replayed).
    pub consolidated: bool,
    /// Number of times this episode has been replayed.
    pub replay_count: u32,
}

/// Result of a consolidation (dream replay) session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Number of episodes replayed.
    pub episodes_replayed: usize,
    /// Average salience of replayed episodes.
    pub avg_salience: f32,
    /// Number of episodes newly consolidated.
    pub newly_consolidated: usize,
    /// Whether learning boost was activated.
    pub boost_activated: bool,
    /// Total consolidation sessions so far.
    pub total_consolidations: u32,
}

/// Statistics for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub total_episodes: usize,
    pub unconsolidated_count: usize,
    pub consolidated_count: usize,
    pub consolidation_sessions: u32,
    pub learning_boost_active: bool,
    pub learning_boost_remaining: u32,
    pub avg_salience: f32,
}

// ---------------------------------------------------------------------------
// MemoryConsolidator
// ---------------------------------------------------------------------------

/// Memory consolidation via dream-like replay.
///
/// Records high-salience episodes during cognitive cycles, then consolidates
/// them via replay during "dream" phases. Consolidated episodes receive a
/// learning rate boost that persists for a configurable number of cycles.
pub struct MemoryConsolidator {
    episodes: Vec<EpisodicEntry>,
    max_episodes: usize,
    consolidation_count: u32,
    learning_boost_active: bool,
    learning_boost_remaining: u32,
}

impl MemoryConsolidator {
    /// Create a new MemoryConsolidator with the given capacity.
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_episodes.min(DEFAULT_MAX_EPISODES)),
            max_episodes: max_episodes.max(1),
            consolidation_count: 0,
            learning_boost_active: false,
            learning_boost_remaining: 0,
        }
    }

    /// Create with default capacity.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_EPISODES)
    }

    /// Record an episodic memory if its salience exceeds the threshold.
    ///
    /// Salience = PHI_WEIGHT * phi + PE_WEIGHT * prediction_error.
    /// High-consciousness, high-surprise moments are prioritized.
    pub fn record(&mut self, description: &str, phi: f32, prediction_error: f32, cycle: u64) {
        let salience = PHI_WEIGHT * phi + PE_WEIGHT * prediction_error;

        if salience < SALIENCE_THRESHOLD {
            return;
        }

        // Evict lowest-salience episode if at capacity
        if self.episodes.len() >= self.max_episodes {
            self.evict_lowest();
        }

        self.episodes.push(EpisodicEntry {
            description: description.to_string(),
            phi_at_recording: phi,
            prediction_error,
            cycle,
            salience,
            consolidated: false,
            replay_count: 0,
        });
    }

    /// "Dream replay": review top-salience unconsolidated episodes,
    /// mark them consolidated, and activate the learning boost.
    pub fn consolidate(&mut self) -> ConsolidationResult {
        self.consolidation_count += 1;

        // Collect indices of unconsolidated episodes, sorted by salience (descending)
        let mut candidates: Vec<usize> = self
            .episodes
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.consolidated)
            .map(|(i, _)| i)
            .collect();

        candidates.sort_by(|&a, &b| {
            self.episodes[b]
                .salience
                .partial_cmp(&self.episodes[a].salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let batch = candidates
            .iter()
            .take(CONSOLIDATION_BATCH_SIZE)
            .copied()
            .collect::<Vec<_>>();

        let mut total_salience = 0.0f32;
        let mut newly_consolidated = 0;

        for &idx in &batch {
            let episode = &mut self.episodes[idx];
            episode.replay_count += 1;
            total_salience += episode.salience;
            if !episode.consolidated {
                episode.consolidated = true;
                newly_consolidated += 1;
            }
        }

        let episodes_replayed = batch.len();
        let avg_salience = if episodes_replayed > 0 {
            total_salience / episodes_replayed as f32
        } else {
            0.0
        };

        // Activate learning boost if we replayed anything
        let boost_activated = episodes_replayed > 0;
        if boost_activated {
            self.learning_boost_active = true;
            self.learning_boost_remaining = BOOST_DURATION_CYCLES;
        }

        ConsolidationResult {
            episodes_replayed,
            avg_salience,
            newly_consolidated,
            boost_activated,
            total_consolidations: self.consolidation_count,
        }
    }

    /// Get all unconsolidated episodes.
    pub fn get_unconsolidated(&self) -> Vec<&EpisodicEntry> {
        self.episodes.iter().filter(|e| !e.consolidated).collect()
    }

    /// Whether the post-consolidation learning boost is active.
    pub fn learning_boost_active(&self) -> bool {
        self.learning_boost_active
    }

    /// Apply learning boost to a base learning rate.
    /// Returns the boosted rate, and decrements the remaining boost cycles.
    pub fn apply_learning_boost(&mut self, base_lr: f32) -> f32 {
        if !self.learning_boost_active {
            return base_lr;
        }

        if self.learning_boost_remaining > 0 {
            self.learning_boost_remaining -= 1;
        }

        if self.learning_boost_remaining == 0 {
            self.learning_boost_active = false;
        }

        base_lr * CONSOLIDATION_LR_BOOST
    }

    /// Remove low-salience, already-consolidated entries to free space.
    pub fn prune(&mut self) {
        // Keep unconsolidated episodes and high-salience consolidated ones
        let median_salience = if self.episodes.is_empty() {
            0.0
        } else {
            let mut saliences: Vec<f32> = self.episodes.iter().map(|e| e.salience).collect();
            saliences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            saliences[saliences.len() / 2]
        };

        self.episodes
            .retain(|e| !e.consolidated || e.salience >= median_salience);
    }

    /// Total number of episodes stored.
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Total consolidation sessions completed.
    pub fn consolidation_count(&self) -> u32 {
        self.consolidation_count
    }

    /// Export statistics for WASM/JSON serialization.
    pub fn stats(&self) -> ConsolidationStats {
        let unconsolidated_count = self.episodes.iter().filter(|e| !e.consolidated).count();
        let consolidated_count = self.episodes.iter().filter(|e| e.consolidated).count();
        let avg_salience = if self.episodes.is_empty() {
            0.0
        } else {
            self.episodes.iter().map(|e| e.salience).sum::<f32>() / self.episodes.len() as f32
        };

        ConsolidationStats {
            total_episodes: self.episodes.len(),
            unconsolidated_count,
            consolidated_count,
            consolidation_sessions: self.consolidation_count,
            learning_boost_active: self.learning_boost_active,
            learning_boost_remaining: self.learning_boost_remaining,
            avg_salience,
        }
    }

    // ── Internal ──────────────────────────────────────────────────────

    /// Evict the lowest-salience episode to make room.
    fn evict_lowest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        let min_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.salience
                    .partial_cmp(&b.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.episodes.remove(min_idx);
    }
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_consolidate() {
        let mut consolidator = MemoryConsolidator::new(50);

        // Record some episodes with varying salience
        consolidator.record("high surprise event", 0.8, 0.9, 1);
        consolidator.record("normal event", 0.3, 0.2, 2);
        consolidator.record("very important", 0.9, 0.8, 3);

        assert_eq!(consolidator.episode_count(), 3);
        assert_eq!(consolidator.get_unconsolidated().len(), 3);

        // Consolidate (dream replay)
        let result = consolidator.consolidate();
        assert_eq!(result.episodes_replayed, 3);
        assert_eq!(result.newly_consolidated, 3);
        assert!(result.boost_activated);
        assert!(result.avg_salience > 0.0);

        // After consolidation, all should be marked
        assert_eq!(consolidator.get_unconsolidated().len(), 0);
    }

    #[test]
    fn test_learning_boost() {
        let mut consolidator = MemoryConsolidator::new(50);
        consolidator.record("event", 0.5, 0.5, 1);
        consolidator.consolidate();

        assert!(consolidator.learning_boost_active());

        let base_lr = 0.01;
        let boosted = consolidator.apply_learning_boost(base_lr);
        assert!(boosted > base_lr, "boosted LR should exceed base");
        assert!((boosted - base_lr * CONSOLIDATION_LR_BOOST).abs() < 1e-6);

        // Exhaust the boost
        for _ in 0..BOOST_DURATION_CYCLES {
            consolidator.apply_learning_boost(base_lr);
        }
        assert!(!consolidator.learning_boost_active());
    }

    #[test]
    fn test_salience_threshold() {
        let mut consolidator = MemoryConsolidator::new(50);

        // Very low salience should be rejected
        consolidator.record("boring", 0.01, 0.01, 1);
        assert_eq!(consolidator.episode_count(), 0);

        // Above threshold should be accepted
        consolidator.record("interesting", 0.5, 0.3, 2);
        assert_eq!(consolidator.episode_count(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut consolidator = MemoryConsolidator::new(3);

        consolidator.record("a", 0.3, 0.3, 1); // salience: 0.30
        consolidator.record("b", 0.5, 0.5, 2); // salience: 0.50
        consolidator.record("c", 0.9, 0.9, 3); // salience: 0.90

        // Fourth should evict lowest
        consolidator.record("d", 0.7, 0.7, 4); // salience: 0.70
        assert_eq!(consolidator.episode_count(), 3);

        // 'a' (lowest salience) should have been evicted
        assert!(!consolidator.episodes.iter().any(|e| e.description == "a"));
    }

    #[test]
    fn test_prune() {
        let mut consolidator = MemoryConsolidator::new(50);
        consolidator.record("low", 0.2, 0.1, 1);
        consolidator.record("high", 0.9, 0.9, 2);

        // Consolidate both
        consolidator.consolidate();

        // Prune removes low-salience consolidated entries
        consolidator.prune();
        // At least the high-salience one should remain
        assert!(consolidator.episode_count() >= 1);
        assert!(
            consolidator
                .episodes
                .iter()
                .any(|e| e.description == "high")
        );
    }
}
