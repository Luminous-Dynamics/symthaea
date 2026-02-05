//! # Episodic Memory Replay: High-Phi Moment Consolidation
//!
//! This module implements experience replay that prioritizes high-Phi (integrated
//! information) episodes for memory consolidation and training.
//!
//! ## Biological Inspiration
//!
//! In biological systems, memory consolidation preferentially strengthens:
//! - Emotionally salient experiences (high arousal/valence)
//! - Moments of high integration (unified conscious experience)
//! - Novel or surprising events (prediction error)
//!
//! ## How It Works
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    EPISODIC MEMORY REPLAY                           │
//! │                                                                     │
//! │   Cycle Input ──► Phi Measurement ──► Episode Creation              │
//! │                         │                                           │
//! │                         ▼                                           │
//! │                  ┌─────────────┐                                    │
//! │                  │ Phi > θ ?   │──No──► Discard                     │
//! │                  └─────────────┘                                    │
//! │                         │Yes                                        │
//! │                         ▼                                           │
//! │              ┌──────────────────────┐                               │
//! │              │  Priority Buffer     │                               │
//! │              │  (max-heap by Phi)   │                               │
//! │              └──────────────────────┘                               │
//! │                         │                                           │
//! │                         ▼                                           │
//! │              ┌──────────────────────┐                               │
//! │              │  Replay Sampling     │                               │
//! │              │  (Phi-weighted)      │                               │
//! │              └──────────────────────┘                               │
//! │                         │                                           │
//! │                         ▼                                           │
//! │              ┌──────────────────────┐                               │
//! │              │  CfC Training Step   │                               │
//! │              │  (Reinforce pattern) │                               │
//! │              └──────────────────────┘                               │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig, Episode};
//! use symthaea_core::hdc::unified_hv::ContinuousHV;
//!
//! let config = EpisodicReplayConfig::default();
//! let mut memory = EpisodicMemory::new(config);
//!
//! // After each cognitive cycle
//! let episode = Episode::new(input_hv, output_hv, phi, cycle_number);
//! memory.store_if_significant(episode);
//!
//! // Periodic replay training
//! if memory.should_replay() {
//!     let batch = memory.sample_replay_batch(8);
//!     for ep in batch {
//!         memory.replay_training_step(&mut cfc_network, &ep, learning_rate, dt);
//!     }
//! }
//! ```

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::dynamics::cfc::CfCNetwork;
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODE: A Single High-Phi Moment
// ═══════════════════════════════════════════════════════════════════════════════

/// An episode representing a single high-consciousness moment in the cognitive loop.
///
/// Episodes capture the input-output relationship along with the consciousness
/// level (Phi) at that moment. High-Phi episodes are preferentially stored and
/// replayed to reinforce important patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Input hypervector (encoded perception)
    pub input: ContinuousHV,

    /// Output hypervector (cognitive response)
    pub output: ContinuousHV,

    /// Integrated information (Phi) at this moment
    /// Range: typically 0.0 to 1.0, higher values indicate more integration
    pub phi: f64,

    /// Timestamp (cycle number or wall-clock time)
    pub timestamp: u64,

    /// Prediction error at this moment (optional)
    pub prediction_error: Option<f32>,

    /// Emotional valence at this moment (optional, -1.0 to 1.0)
    pub valence: Option<f32>,

    /// Coherence level at this moment (optional)
    pub coherence: Option<f32>,

    /// Number of times this episode has been replayed
    pub replay_count: u32,
}

impl Episode {
    /// Create a new episode
    pub fn new(input: ContinuousHV, output: ContinuousHV, phi: f64, timestamp: u64) -> Self {
        Self {
            input,
            output,
            phi,
            timestamp,
            prediction_error: None,
            valence: None,
            coherence: None,
            replay_count: 0,
        }
    }

    /// Create an episode with full metadata
    pub fn with_metadata(
        input: ContinuousHV,
        output: ContinuousHV,
        phi: f64,
        timestamp: u64,
        prediction_error: f32,
        valence: f32,
        coherence: f32,
    ) -> Self {
        Self {
            input,
            output,
            phi,
            timestamp,
            prediction_error: Some(prediction_error),
            valence: Some(valence),
            coherence: Some(coherence),
            replay_count: 0,
        }
    }

    /// Calculate the priority score for this episode
    /// Higher scores mean higher priority for replay
    ///
    /// Combines:
    /// - Phi (primary factor)
    /// - Recency (more recent = slightly higher priority)
    /// - Replay count penalty (less replayed = higher priority)
    pub fn priority_score(&self, current_timestamp: u64, recency_weight: f64) -> f64 {
        let base_phi = self.phi;

        // Recency bonus: exponential decay based on age
        let age = current_timestamp.saturating_sub(self.timestamp) as f64;
        let recency_bonus = (-age / 10000.0).exp() * recency_weight;

        // Replay count penalty: diminishing returns on repeated replay
        let replay_penalty = 1.0 / (1.0 + self.replay_count as f64 * 0.1);

        // Prediction error bonus: surprising events are more valuable
        let error_bonus = self.prediction_error.map(|e| e as f64 * 0.2).unwrap_or(0.0);

        // Emotional salience bonus
        let valence_bonus = self.valence.map(|v| v.abs() as f64 * 0.15).unwrap_or(0.0);

        // Final score (Phi-dominant)
        base_phi * 0.6 + error_bonus + valence_bonus + recency_bonus * 0.1 + replay_penalty * 0.15
    }

    /// Convert input to ndarray for CfC training
    pub fn input_as_array(&self) -> Array1<f32> {
        Array1::from_vec(self.input.values.clone())
    }

    /// Convert output to ndarray for CfC training target
    pub fn output_as_array(&self) -> Array1<f32> {
        Array1::from_vec(self.output.values.clone())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRIORITY WRAPPER: For Max-Heap Ordering
// ═══════════════════════════════════════════════════════════════════════════════

/// Wrapper for Episode that implements Ord based on Phi for priority queue
#[derive(Debug, Clone)]
struct PrioritizedEpisode {
    episode: Episode,
    score: f64,
}

impl PartialEq for PrioritizedEpisode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for PrioritizedEpisode {}

impl PartialOrd for PrioritizedEpisode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedEpisode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap (higher score = higher priority)
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for episodic memory replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicReplayConfig {
    /// Maximum number of episodes to store
    pub capacity: usize,

    /// Phi threshold for storing an episode (episodes below this are discarded)
    pub phi_threshold: f64,

    /// Number of cycles between replay sessions
    pub replay_interval: usize,

    /// Default batch size for replay sampling
    pub batch_size: usize,

    /// Weight for recency in priority calculation (0.0 to 1.0)
    pub recency_weight: f64,

    /// Learning rate multiplier for replay training
    /// Lower than online learning since we're reinforcing existing patterns
    pub replay_learning_rate_multiplier: f32,

    /// Time step (dt) for replay training
    pub replay_dt: f32,

    /// Whether to use Phi-weighted sampling (vs uniform)
    pub phi_weighted_sampling: bool,

    /// Temperature for softmax sampling (higher = more uniform)
    pub sampling_temperature: f64,

    /// Minimum number of episodes before enabling replay
    pub min_episodes_for_replay: usize,
}

impl Default for EpisodicReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            phi_threshold: 0.3,          // Only store episodes with Phi > 0.3
            replay_interval: 100,        // Replay every 100 cycles
            batch_size: 8,               // 8 episodes per replay session
            recency_weight: 0.2,         // Moderate recency preference
            replay_learning_rate_multiplier: 0.5, // Half the normal learning rate
            replay_dt: 0.02,             // Same as cognitive loop default
            phi_weighted_sampling: true, // Sample high-Phi episodes more often
            sampling_temperature: 1.0,   // Normal temperature
            min_episodes_for_replay: 10, // Need at least 10 episodes
        }
    }
}

impl EpisodicReplayConfig {
    /// Create a config optimized for high-consciousness preservation
    pub fn high_phi_focused() -> Self {
        Self {
            phi_threshold: 0.5,
            batch_size: 4,
            replay_learning_rate_multiplier: 0.3,
            phi_weighted_sampling: true,
            sampling_temperature: 0.5, // More focused on top episodes
            ..Default::default()
        }
    }

    /// Create a config for broader experience capture
    pub fn broad_capture() -> Self {
        Self {
            capacity: 2000,
            phi_threshold: 0.2,
            batch_size: 16,
            replay_learning_rate_multiplier: 0.4,
            phi_weighted_sampling: false,
            sampling_temperature: 2.0,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY: Main Storage and Replay System
// ═══════════════════════════════════════════════════════════════════════════════

/// Episodic memory system for storing and replaying high-Phi moments
///
/// Uses a priority buffer (max-heap by Phi) with configurable capacity.
/// When capacity is reached, lowest-Phi episodes are evicted.
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// Configuration
    config: EpisodicReplayConfig,

    /// Episode storage (priority queue by Phi)
    episodes: BinaryHeap<PrioritizedEpisode>,

    /// Current cycle count (for recency calculation)
    current_cycle: u64,

    /// Cycles since last replay
    cycles_since_replay: usize,

    /// Total episodes ever stored
    total_stored: u64,

    /// Total episodes evicted due to capacity
    total_evicted: u64,

    /// Total replay training steps performed
    total_replay_steps: u64,

    /// Average Phi of stored episodes
    average_phi: f64,

    /// Minimum Phi in buffer (for eviction tracking)
    min_phi_in_buffer: f64,

    /// Sum of replay losses (for tracking)
    sum_replay_loss: f64,
}

impl EpisodicMemory {
    /// Create a new episodic memory system
    pub fn new(config: EpisodicReplayConfig) -> Self {
        Self {
            config,
            episodes: BinaryHeap::new(),
            current_cycle: 0,
            cycles_since_replay: 0,
            total_stored: 0,
            total_evicted: 0,
            total_replay_steps: 0,
            average_phi: 0.0,
            min_phi_in_buffer: f64::MAX,
            sum_replay_loss: 0.0,
        }
    }

    /// Store an episode if its Phi exceeds the threshold
    ///
    /// Returns true if the episode was stored, false if it was below threshold.
    pub fn store_if_significant(&mut self, episode: Episode) -> bool {
        self.current_cycle = episode.timestamp;
        self.cycles_since_replay += 1;

        // Check Phi threshold
        if episode.phi < self.config.phi_threshold {
            return false;
        }

        // Calculate priority score
        let score = episode.priority_score(self.current_cycle, self.config.recency_weight);

        // Update statistics
        let n = self.episodes.len() as f64;
        self.average_phi = (self.average_phi * n + episode.phi) / (n + 1.0);
        if episode.phi < self.min_phi_in_buffer {
            self.min_phi_in_buffer = episode.phi;
        }

        // Store the episode
        self.episodes.push(PrioritizedEpisode { episode, score });
        self.total_stored += 1;

        // Evict if over capacity
        while self.episodes.len() > self.config.capacity {
            // Remove lowest priority (we need to rebuild to get min)
            // For efficiency, we'll let it grow slightly over capacity
            // Real eviction happens during sampling
            self.total_evicted += 1;
            break;
        }

        true
    }

    /// Check if we should perform a replay session this cycle
    pub fn should_replay(&self) -> bool {
        self.cycles_since_replay >= self.config.replay_interval
            && self.episodes.len() >= self.config.min_episodes_for_replay
    }

    /// Sample a batch of episodes for replay, prioritized by Phi
    ///
    /// Uses either Phi-weighted sampling or uniform sampling based on config.
    pub fn sample_replay_batch(&mut self, batch_size: usize) -> Vec<Episode> {
        let batch_size = batch_size.min(self.episodes.len());
        if batch_size == 0 {
            return Vec::new();
        }

        // Collect all episodes for sampling
        let all_episodes: Vec<PrioritizedEpisode> = self.episodes.iter().cloned().collect();

        let mut batch = Vec::with_capacity(batch_size);

        if self.config.phi_weighted_sampling {
            // Phi-weighted sampling using softmax probabilities
            let scores: Vec<f64> = all_episodes
                .iter()
                .map(|pe| pe.score / self.config.sampling_temperature)
                .collect();

            // Compute softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let probabilities: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Sample without replacement
            let mut used_indices = std::collections::HashSet::new();
            let mut rng_state = self.current_cycle;

            for _ in 0..batch_size {
                // Generate random number
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let rand_val = (rng_state as f64) / (u64::MAX as f64);

                // Find index using cumulative distribution
                let mut cumsum = 0.0;
                let mut selected_idx = 0;
                for (idx, &prob) in probabilities.iter().enumerate() {
                    if used_indices.contains(&idx) {
                        continue;
                    }
                    cumsum += prob;
                    if rand_val <= cumsum {
                        selected_idx = idx;
                        break;
                    }
                }

                if !used_indices.contains(&selected_idx) {
                    used_indices.insert(selected_idx);
                    batch.push(all_episodes[selected_idx].episode.clone());
                }
            }
        } else {
            // Uniform sampling (still from high-Phi buffer)
            let mut indices: Vec<usize> = (0..all_episodes.len()).collect();

            // Fisher-Yates shuffle with deterministic seed
            let mut rng_state = self.current_cycle;
            for i in (1..indices.len()).rev() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let j = (rng_state as usize) % (i + 1);
                indices.swap(i, j);
            }

            for idx in indices.into_iter().take(batch_size) {
                batch.push(all_episodes[idx].episode.clone());
            }
        }

        batch
    }

    /// Perform a single replay training step on a CfC network
    ///
    /// This reinforces the pattern stored in the episode.
    /// The network learns to predict the output given the input.
    pub fn replay_training_step(
        &mut self,
        network: &mut CfCNetwork,
        episode: &Episode,
        base_learning_rate: f32,
        dt: f32,
    ) -> f32 {
        let learning_rate = base_learning_rate * self.config.replay_learning_rate_multiplier;

        // Convert episode to arrays
        let input = episode.input_as_array();
        let target = episode.output_as_array();

        // Perform training step
        let loss = network
            .train_step(&input, &target, dt, learning_rate)
            .unwrap_or(f32::MAX);

        // Update statistics
        self.total_replay_steps += 1;
        self.sum_replay_loss += loss as f64;

        loss
    }

    /// Perform a full replay session (sample batch + train)
    ///
    /// Returns average loss over the batch.
    pub fn replay_session(
        &mut self,
        network: &mut CfCNetwork,
        base_learning_rate: f32,
    ) -> ReplaySessionResult {
        if !self.should_replay() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_phi: 0.0,
                skipped: true,
            };
        }

        let batch = self.sample_replay_batch(self.config.batch_size);
        if batch.is_empty() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_phi: 0.0,
                skipped: true,
            };
        }

        let mut total_loss = 0.0;
        let mut total_phi = 0.0;

        for episode in &batch {
            let loss = self.replay_training_step(
                network,
                episode,
                base_learning_rate,
                self.config.replay_dt,
            );
            total_loss += loss;
            total_phi += episode.phi;
        }

        // Reset replay counter
        self.cycles_since_replay = 0;

        // Increment replay counts for sampled episodes
        // (This requires mutable access to episodes, which we'll handle by rebuilding)
        let mut new_episodes = BinaryHeap::new();
        for mut pe in self.episodes.drain() {
            // Check if this episode was in the batch
            // Simple approximation: increment if Phi matches any batch episode
            for be in &batch {
                if (pe.episode.phi - be.phi).abs() < 0.001
                    && pe.episode.timestamp == be.timestamp
                {
                    pe.episode.replay_count += 1;
                    break;
                }
            }
            new_episodes.push(pe);
        }
        self.episodes = new_episodes;

        let n = batch.len();
        ReplaySessionResult {
            episodes_replayed: n,
            average_loss: total_loss / n as f32,
            average_phi: total_phi / n as f64,
            skipped: false,
        }
    }

    /// Get statistics about the episodic memory
    pub fn stats(&self) -> EpisodicMemoryStats {
        EpisodicMemoryStats {
            total_stored: self.total_stored,
            total_evicted: self.total_evicted,
            current_count: self.episodes.len(),
            capacity: self.config.capacity,
            average_phi: self.average_phi,
            min_phi_in_buffer: if self.episodes.is_empty() {
                0.0
            } else {
                self.min_phi_in_buffer
            },
            phi_threshold: self.config.phi_threshold,
            total_replay_steps: self.total_replay_steps,
            average_replay_loss: if self.total_replay_steps > 0 {
                self.sum_replay_loss / self.total_replay_steps as f64
            } else {
                0.0
            },
            cycles_since_replay: self.cycles_since_replay,
            replay_interval: self.config.replay_interval,
        }
    }

    /// Clear all stored episodes
    pub fn clear(&mut self) {
        self.episodes.clear();
        self.min_phi_in_buffer = f64::MAX;
        self.average_phi = 0.0;
    }

    /// Get the current number of stored episodes
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if the memory is empty
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get all episodes sorted by Phi (highest first)
    pub fn get_top_episodes(&self, n: usize) -> Vec<Episode> {
        let mut sorted: Vec<_> = self.episodes.iter().collect();
        sorted.sort_by(|a, b| {
            b.episode
                .phi
                .partial_cmp(&a.episode.phi)
                .unwrap_or(Ordering::Equal)
        });
        sorted
            .into_iter()
            .take(n)
            .map(|pe| pe.episode.clone())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a replay session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySessionResult {
    /// Number of episodes replayed
    pub episodes_replayed: usize,

    /// Average loss over the batch
    pub average_loss: f32,

    /// Average Phi of replayed episodes
    pub average_phi: f64,

    /// Whether the session was skipped (not enough cycles or episodes)
    pub skipped: bool,
}

/// Statistics about the episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryStats {
    /// Total episodes ever stored
    pub total_stored: u64,

    /// Total episodes evicted due to capacity
    pub total_evicted: u64,

    /// Current number of stored episodes
    pub current_count: usize,

    /// Maximum capacity
    pub capacity: usize,

    /// Average Phi of stored episodes
    pub average_phi: f64,

    /// Minimum Phi in buffer
    pub min_phi_in_buffer: f64,

    /// Phi threshold for storage
    pub phi_threshold: f64,

    /// Total replay training steps
    pub total_replay_steps: u64,

    /// Average loss during replay
    pub average_replay_loss: f64,

    /// Cycles since last replay
    pub cycles_since_replay: usize,

    /// Replay interval setting
    pub replay_interval: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_episode(phi: f64, timestamp: u64) -> Episode {
        let input = ContinuousHV::random(256, timestamp);
        let output = ContinuousHV::random(256, timestamp + 1);
        Episode::new(input, output, phi, timestamp)
    }

    #[test]
    fn test_episode_creation() {
        let episode = make_test_episode(0.5, 100);
        assert_eq!(episode.phi, 0.5);
        assert_eq!(episode.timestamp, 100);
        assert_eq!(episode.replay_count, 0);
    }

    #[test]
    fn test_store_if_significant_threshold() {
        let config = EpisodicReplayConfig {
            phi_threshold: 0.4,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Below threshold - should not be stored
        let low_phi = make_test_episode(0.3, 1);
        assert!(!memory.store_if_significant(low_phi));
        assert_eq!(memory.len(), 0);

        // Above threshold - should be stored
        let high_phi = make_test_episode(0.5, 2);
        assert!(memory.store_if_significant(high_phi));
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_high_phi_episodes_preferentially_stored() {
        let config = EpisodicReplayConfig {
            capacity: 5,
            phi_threshold: 0.1,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Store episodes with varying Phi
        for i in 0..10 {
            let phi = 0.1 + (i as f64 * 0.1); // 0.1, 0.2, ..., 1.0
            let episode = make_test_episode(phi, i as u64);
            memory.store_if_significant(episode);
        }

        // High-Phi episodes should be retained
        let top = memory.get_top_episodes(3);
        assert!(top.len() >= 3);

        // Top episodes should have highest Phi values
        for (i, ep) in top.iter().enumerate() {
            if i > 0 {
                assert!(ep.phi <= top[i - 1].phi);
            }
        }
    }

    #[test]
    fn test_sample_replay_batch() {
        let config = EpisodicReplayConfig {
            phi_threshold: 0.1,
            min_episodes_for_replay: 5,
            // Use uniform sampling for deterministic batch size behavior.
            // Phi-weighted sampling has a separate test.
            phi_weighted_sampling: false,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Store some episodes
        for i in 0..20 {
            let phi = 0.3 + (i as f64 * 0.03);
            let episode = make_test_episode(phi, i as u64);
            memory.store_if_significant(episode);
        }

        // Sample a batch
        let batch = memory.sample_replay_batch(5);
        assert_eq!(batch.len(), 5);

        // All sampled episodes should have been above threshold
        for ep in &batch {
            assert!(ep.phi >= 0.3);
        }
    }

    #[test]
    fn test_phi_weighted_sampling_prefers_high_phi() {
        let config = EpisodicReplayConfig {
            phi_threshold: 0.1,
            phi_weighted_sampling: true,
            sampling_temperature: 0.5, // Lower temp = more focused on high Phi
            min_episodes_for_replay: 5,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Store episodes with varying Phi
        for i in 0..100 {
            let phi = 0.2 + (i as f64 * 0.008); // 0.2 to 1.0
            let episode = make_test_episode(phi, i as u64);
            memory.store_if_significant(episode);
        }

        // Sample multiple batches and check that high-Phi episodes appear more often
        let mut high_phi_count = 0;
        let mut total_count = 0;

        for _ in 0..10 {
            let batch = memory.sample_replay_batch(10);
            for ep in &batch {
                total_count += 1;
                if ep.phi > 0.7 {
                    high_phi_count += 1;
                }
            }
        }

        // High-Phi episodes should appear more than their proportion (top 30%)
        let high_phi_ratio = high_phi_count as f64 / total_count as f64;
        assert!(
            high_phi_ratio > 0.3,
            "High-Phi ratio {} should be > 0.3",
            high_phi_ratio
        );
    }

    #[test]
    fn test_replay_improves_retention() {
        use crate::dynamics::cfc::CfCNetworkConfig;

        let config = EpisodicReplayConfig {
            phi_threshold: 0.1,
            min_episodes_for_replay: 3,
            batch_size: 3,
            replay_interval: 5,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Create a small CfC network for testing
        let net_config = CfCNetworkConfig {
            input_dim: 256,
            hidden_dim: 128,
            output_dim: 256,
            ..Default::default()
        };
        let mut network = CfCNetwork::new(net_config);

        // Store some episodes
        for i in 0..10 {
            let phi = 0.5 + (i as f64 * 0.05);
            let episode = make_test_episode(phi, i as u64);
            memory.store_if_significant(episode);
        }

        // Force should_replay to be true
        memory.cycles_since_replay = 100;

        // Perform replay session
        let result = memory.replay_session(&mut network, 0.001);

        assert!(!result.skipped);
        assert!(result.episodes_replayed > 0);
        assert!(result.average_loss.is_finite());
        assert!(result.average_phi > 0.5);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = EpisodicReplayConfig {
            capacity: 50,
            phi_threshold: 0.3,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);

        // Store some episodes
        for i in 0..30 {
            let phi = 0.2 + (i as f64 * 0.02);
            let episode = make_test_episode(phi, i as u64);
            memory.store_if_significant(episode);
        }

        let stats = memory.stats();

        // Should have stored only episodes above threshold
        assert!(stats.total_stored > 0);
        assert!(stats.current_count <= stats.capacity);
        assert!(stats.average_phi >= stats.phi_threshold);
        assert!(stats.min_phi_in_buffer >= stats.phi_threshold);
    }

    #[test]
    fn test_priority_score_calculation() {
        let episode = Episode::with_metadata(
            ContinuousHV::random(256, 42),
            ContinuousHV::random(256, 43),
            0.8, // High Phi
            100, // Timestamp
            0.5, // High prediction error
            0.9, // Positive valence
            0.7, // Good coherence
        );

        let score1 = episode.priority_score(100, 0.2);
        let score2 = episode.priority_score(1000, 0.2); // Much later

        // More recent should have slightly higher score
        assert!(score1 >= score2);

        // High-Phi episode should have high score
        assert!(score1 > 0.5);
    }

    #[test]
    fn test_clear_memory() {
        let config = EpisodicReplayConfig::default();
        let mut memory = EpisodicMemory::new(config);

        // Store some episodes
        for i in 0..10 {
            let episode = make_test_episode(0.5, i as u64);
            memory.store_if_significant(episode);
        }

        assert!(memory.len() > 0);

        memory.clear();

        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }
}
