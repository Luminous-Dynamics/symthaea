// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use uuid::Uuid;

use crate::TrainableNetwork;
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODE: A Single High-Phi Moment
// ═══════════════════════════════════════════════════════════════════════════════

/// Storage-occurrence identity assigned by the canonical episodic store.
///
/// This is deliberately different from semantic/content identity: two byte-identical
/// experiences stored twice receive different instance IDs. The ID follows one stored
/// occurrence through replay, reconsolidation, quarantine, restoration, cloning, and
/// serialization. Fresh ordinary insertion always overwrites any caller-provided value.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct EpisodeInstanceId(Uuid);

impl EpisodeInstanceId {
    fn fresh() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying UUID for durable/audit representations.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for EpisodeInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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

    /// Psi — Consciousness estimate at this moment
    /// Range: typically 0.0 to 1.0, higher values indicate more integration
    pub psi: f64,

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

    /// Consolidation strength (increases with each retrieval/reconsolidation).
    /// Starts at 1.0 for new episodes; legacy deserialized episodes default to 0.0.
    #[serde(default)]
    pub consolidation_strength: f64,

    /// Number of times this episode has been actively retrieved (distinct from replay)
    #[serde(default)]
    pub retrieval_count: u32,

    /// Dopamine level at encoding — tags reward salience for replay prioritization.
    /// High-DA episodes are preferentially consolidated with stronger LR.
    /// Science: Lisman & Grace (2005) — hippocampal-VTA loop, DA tags memory consolidation.
    #[serde(default)]
    pub dopamine_at_encoding: Option<f32>,

    /// Neuromodulator bath state at encoding for state-dependent memory retrieval.
    /// Science: Godden & Baddeley (1975) — state-dependent memory;
    /// Eich (1980) — mood-dependent retrieval.
    #[serde(default)]
    pub bath_state_at_encoding: Option<[f32; 9]>,

    /// Semantic embedding from neural encoder (for embedding-based retrieval).
    #[serde(default)]
    pub semantic_embedding: Option<Vec<f32>>,

    /// Canonical storage-occurrence identity.
    ///
    /// `None` means the episode has not yet been accepted by an `EpisodicMemory`, or it was
    /// deserialized from legacy data. Normal insertion assigns a fresh ID only after the Psi
    /// threshold passes. This field is lifecycle metadata and must not be interpreted as part of
    /// the episode's semantic/content identity.
    #[serde(default)]
    pub instance_id: Option<EpisodeInstanceId>,
}

impl Episode {
    /// Create a new episode
    pub fn new(input: ContinuousHV, output: ContinuousHV, psi: f64, timestamp: u64) -> Self {
        Self {
            input,
            output,
            psi,
            timestamp,
            prediction_error: None,
            valence: None,
            coherence: None,
            replay_count: 0,
            consolidation_strength: 1.0,
            retrieval_count: 0,
            dopamine_at_encoding: None,
            bath_state_at_encoding: None,
            semantic_embedding: None,
            instance_id: None,
        }
    }

    /// Create an episode with full metadata
    pub fn with_metadata(
        input: ContinuousHV,
        output: ContinuousHV,
        psi: f64,
        timestamp: u64,
        prediction_error: f32,
        valence: f32,
        coherence: f32,
    ) -> Self {
        Self {
            input,
            output,
            psi,
            timestamp,
            prediction_error: Some(prediction_error),
            valence: Some(valence),
            coherence: Some(coherence),
            replay_count: 0,
            consolidation_strength: 1.0,
            retrieval_count: 0,
            dopamine_at_encoding: None,
            bath_state_at_encoding: None,
            semantic_embedding: None,
            instance_id: None,
        }
    }

    /// Set semantic embedding for embedding-based retrieval.
    pub fn with_semantic_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.semantic_embedding = Some(embedding);
        self
    }

    /// Set dopamine level at encoding for DA-tagged replay prioritization.
    pub fn with_dopamine(mut self, da: f32) -> Self {
        self.dopamine_at_encoding = Some(da.clamp(0.0, 1.0));
        self
    }

    /// Set neuromodulator bath state at encoding for state-dependent retrieval.
    /// Science: Godden & Baddeley (1975) — state-dependent memory.
    pub fn with_bath_state(mut self, state: [f32; 9]) -> Self {
        self.bath_state_at_encoding = Some(state);
        self
    }

    /// Calculate the priority score for this episode
    /// Higher scores mean higher priority for replay
    pub fn priority_score(&self, current_timestamp: u64, recency_weight: f64) -> f64 {
        self.survival_value(current_timestamp, recency_weight)
    }

    /// Calculate the survival value of this episode.
    ///
    /// This value determines if the memory is retained or "forgotten" (pruned).
    /// Factors: Phi, Consolidation, Recency, and Prediction Error.
    pub fn survival_value(&self, current_timestamp: u64, recency_weight: f64) -> f64 {
        let base_phi = self.psi;

        // Recency bonus: exponential decay based on age
        let age = current_timestamp.saturating_sub(self.timestamp) as f64;
        let recency_bonus = (-age / 10000.0).exp() * recency_weight;

        // Replay count penalty: diminishing returns on repeated replay (prevents over-fixation)
        let replay_penalty = 1.0 / (1.0 + self.replay_count as f64 * 0.1);

        // Prediction error bonus: surprising events are more valuable
        let error_bonus = self.prediction_error.map(|e| e as f64 * 0.2).unwrap_or(0.0);

        // Emotional salience bonus
        let valence_bonus = self.valence.map(|v| v.abs() as f64 * 0.15).unwrap_or(0.0);

        // Consolidation bonus: retrieved/reconsolidated memories are more valuable
        let consolidation_bonus = self.consolidation_strength * 0.1;

        // DA salience bonus: high-DA episodes are more valuable for consolidation
        // Science: Lisman & Grace (2005) — DA tags for consolidation priority
        let da_bonus = self
            .dopamine_at_encoding
            .map(|d| d as f64 * 0.25)
            .unwrap_or(0.0);

        // Final survival value
        base_phi * 0.5
            + error_bonus
            + valence_bonus
            + recency_bonus
            + consolidation_bonus
            + da_bonus
            + replay_penalty * 0.05
    }

    /// Reconsolidate this episode (called upon retrieval).
    ///
    /// Biological reconsolidation: each time a memory is retrieved, it becomes
    /// labile and is re-stored with updated strength. This models that process
    /// by boosting consolidation_strength logarithmically and incrementing
    /// retrieval_count.
    pub fn reconsolidate(&mut self, current_phi: f64) {
        self.retrieval_count += 1;
        // Logarithmic boost: diminishing returns on repeated retrieval
        let boost = (self.retrieval_count as f64).ln_1p() * 0.1;
        // Phi-weighted: reconsolidation is stronger during high-Phi states
        let phi_weight = 0.5 + current_phi.clamp(0.0, 1.0) * 0.5;
        self.consolidation_strength += boost * phi_weight;
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
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Reversible inactive storage for one exact episodic occurrence.
///
/// Quarantine removes an episode from active sampling/replay/retrieval without destroying it.
/// Governance, consent, justification, and durable audit evidence are intentionally handled by
/// higher assurance layers; this type is the canonical memory mechanism only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantinedEpisode {
    pub instance_id: EpisodeInstanceId,
    pub episode: Episode,
    pub score_at_quarantine: f64,
    pub quarantined_at_cycle: u64,
}

/// Mechanism-level quarantine error. This is not an authorization decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodicQuarantineError {
    InstanceNotFound(EpisodeInstanceId),
    AlreadyQuarantined(EpisodeInstanceId),
    ActiveInstanceCollision(EpisodeInstanceId),
}

impl fmt::Display for EpisodicQuarantineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstanceNotFound(id) => write!(f, "episodic instance not found: {id}"),
            Self::AlreadyQuarantined(id) => write!(f, "episodic instance already quarantined: {id}"),
            Self::ActiveInstanceCollision(id) => {
                write!(f, "episodic instance is already active: {id}")
            }
        }
    }
}

impl std::error::Error for EpisodicQuarantineError {}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for episodic memory replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicReplayConfig {
    /// Maximum number of episodes to store
    pub capacity: usize,

    /// Psi threshold for storing an episode (episodes below this are discarded)
    pub psi_threshold: f64,

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

    /// Whether to use Psi-weighted sampling (vs uniform)
    pub psi_weighted_sampling: bool,

    /// Temperature for softmax sampling (higher = more uniform)
    pub sampling_temperature: f64,

    /// Minimum number of episodes before enabling replay
    pub min_episodes_for_replay: usize,
}

impl Default for EpisodicReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            psi_threshold: 0.3,
            replay_interval: 100,
            batch_size: 8,
            recency_weight: 0.2,
            replay_learning_rate_multiplier: 0.5,
            replay_dt: 0.02,
            psi_weighted_sampling: true,
            sampling_temperature: 1.0,
            min_episodes_for_replay: 10,
        }
    }
}

impl EpisodicReplayConfig {
    /// Create a config optimized for high-consciousness preservation
    pub fn high_phi_focused() -> Self {
        Self {
            psi_threshold: 0.5,
            batch_size: 4,
            replay_learning_rate_multiplier: 0.3,
            psi_weighted_sampling: true,
            sampling_temperature: 0.5,
            ..Default::default()
        }
    }

    /// Create a config for broader experience capture
    pub fn broad_capture() -> Self {
        Self {
            capacity: 2000,
            psi_threshold: 0.2,
            batch_size: 16,
            replay_learning_rate_multiplier: 0.4,
            psi_weighted_sampling: false,
            sampling_temperature: 2.0,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISODIC MEMORY: Main Storage and Replay System
// ═══════════════════════════════════════════════════════════════════════════════

/// Cosine similarity between two 9-dimensional bath state vectors.
pub fn bath_cosine_similarity(a: &[f32; 9], b: &[f32; 9]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-10 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Episodic memory system for storing and replaying high-Phi moments
///
/// Uses a priority buffer (max-heap by Phi) with configurable capacity.
/// Quarantined episodes remain owned by this store but are excluded from every active read,
/// replay, and retrieval path until explicitly restored.
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    pub(crate) config: EpisodicReplayConfig,
    episodes: BinaryHeap<PrioritizedEpisode>,
    quarantined: HashMap<EpisodeInstanceId, QuarantinedEpisode>,
    current_cycle: u64,
    cycles_since_replay: usize,
    total_stored: u64,
    total_evicted: u64,
    total_replay_steps: u64,
    average_psi: f64,
    min_psi_in_buffer: f64,
    sum_replay_loss: f64,
    demand_replay_triggered: bool,
    demand_replay_count: u64,
}

impl EpisodicMemory {
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.config.batch_size = batch_size;
    }

    pub fn new(config: EpisodicReplayConfig) -> Self {
        Self {
            config,
            episodes: BinaryHeap::new(),
            quarantined: HashMap::new(),
            current_cycle: 0,
            cycles_since_replay: 0,
            total_stored: 0,
            total_evicted: 0,
            total_replay_steps: 0,
            average_psi: 0.0,
            min_psi_in_buffer: f64::MAX,
            sum_replay_loss: 0.0,
            demand_replay_triggered: false,
            demand_replay_count: 0,
        }
    }

    /// Store an episode if its Psi exceeds the threshold.
    ///
    /// Compatibility wrapper over [`Self::store_if_significant_with_id`].
    pub fn store_if_significant(&mut self, episode: Episode) -> bool {
        self.store_if_significant_with_id(episode).is_some()
    }

    /// Store an episode and return the exact occurrence identity assigned by this store.
    ///
    /// A new successful insertion always receives a fresh UUID, even if the caller supplied an
    /// `instance_id` by cloning or deserializing an older episode. This prevents duplicate content
    /// or caller-controlled IDs from collapsing two stored occurrences into one identity.
    pub fn store_if_significant_with_id(
        &mut self,
        mut episode: Episode,
    ) -> Option<EpisodeInstanceId> {
        self.current_cycle = episode.timestamp;
        self.cycles_since_replay += 1;

        if episode.psi < self.config.psi_threshold {
            return None;
        }

        let instance_id = EpisodeInstanceId::fresh();
        episode.instance_id = Some(instance_id);
        let score = episode.priority_score(self.current_cycle, self.config.recency_weight);

        let n = self.episodes.len() as f64;
        self.average_psi = (self.average_psi * n + episode.psi) / (n + 1.0);
        if episode.psi < self.min_psi_in_buffer {
            self.min_psi_in_buffer = episode.psi;
        }

        self.episodes.push(PrioritizedEpisode { episode, score });
        self.total_stored += 1;

        if self.episodes.len() > self.config.capacity {
            self.total_evicted += 1;
        }

        Some(instance_id)
    }

    /// Check if we should perform a replay session this cycle.
    pub fn should_replay(&self) -> bool {
        let periodic = self.cycles_since_replay >= self.config.replay_interval;
        let triggered = self.demand_replay_triggered;
        (periodic || triggered) && self.episodes.len() >= self.config.min_episodes_for_replay
    }

    pub fn trigger_demand_replay(&mut self) {
        self.demand_replay_triggered = true;
        self.demand_replay_count += 1;
    }

    pub fn adapt_replay_interval(&mut self, error_variance: f32) {
        let base = 100.0f32;
        let factor = if error_variance > 0.1 {
            0.5
        } else if error_variance > 0.05 {
            0.75
        } else if error_variance < 0.01 {
            2.0
        } else {
            1.0
        };
        self.config.replay_interval = (base * factor).clamp(25.0, 200.0) as usize;
    }

    /// Sample a batch of active episodes for replay, prioritized by Psi.
    pub fn sample_replay_batch(&mut self, batch_size: usize) -> Vec<Episode> {
        let batch_size = batch_size.min(self.episodes.len());
        if batch_size == 0 {
            return Vec::new();
        }

        let all_episodes: Vec<PrioritizedEpisode> = self.episodes.iter().cloned().collect();
        let mut batch = Vec::with_capacity(batch_size);

        if self.config.psi_weighted_sampling {
            let temp = self.config.sampling_temperature.max(1e-10);
            let scores: Vec<f64> = all_episodes.iter().map(|pe| pe.score / temp).collect();
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let probabilities: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();
            let mut used_indices = HashSet::new();
            let mut rng_state = self.current_cycle;

            for _ in 0..batch_size {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let rand_val = (rng_state as f64) / (u64::MAX as f64);
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
            let mut indices: Vec<usize> = (0..all_episodes.len()).collect();
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

    /// Sample a batch of active episodes conditioned on current neuromodulator bath state.
    pub fn sample_replay_batch_conditioned(
        &mut self,
        batch_size: usize,
        current_bath: Option<[f32; 9]>,
    ) -> Vec<Episode> {
        let current_bath = match current_bath {
            Some(b) => b,
            None => return self.sample_replay_batch(batch_size),
        };

        let batch_size = batch_size.min(self.episodes.len());
        if batch_size == 0 {
            return Vec::new();
        }
        let all_episodes: Vec<PrioritizedEpisode> = self.episodes.iter().cloned().collect();
        let scores: Vec<f64> = all_episodes
            .iter()
            .map(|pe| {
                let base = pe.score / self.config.sampling_temperature.max(1e-10);
                let similarity_bonus = pe
                    .episode
                    .bath_state_at_encoding
                    .as_ref()
                    .map(|enc| bath_cosine_similarity(enc, &current_bath) as f64 * 0.15)
                    .unwrap_or(0.0);
                base + similarity_bonus
            })
            .collect();
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let probabilities: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();
        let mut batch = Vec::with_capacity(batch_size);
        let mut used_indices = HashSet::new();
        let mut rng_state = self.current_cycle;

        for _ in 0..batch_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let rand_val = (rng_state as f64) / (u64::MAX as f64);
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

        batch
    }

    pub fn replay_training_step(
        &mut self,
        network: &mut impl TrainableNetwork,
        episode: &Episode,
        base_learning_rate: f32,
        dt: f32,
    ) -> f32 {
        let da_replay_scale = episode
            .dopamine_at_encoding
            .map(|d| 0.7 + d * 0.6)
            .unwrap_or(1.0);
        let learning_rate =
            base_learning_rate * self.config.replay_learning_rate_multiplier * da_replay_scale;
        let input = episode.input_as_array();
        let target = episode.output_as_array();
        let loss = network
            .train_step(&input, &target, dt, learning_rate)
            .unwrap_or(f32::MAX);
        self.total_replay_steps += 1;
        self.sum_replay_loss += loss as f64;
        loss
    }

    pub fn replay_session(
        &mut self,
        network: &mut impl TrainableNetwork,
        base_learning_rate: f32,
    ) -> ReplaySessionResult {
        if !self.should_replay() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_psi: 0.0,
                skipped: true,
            };
        }

        let batch = self.sample_replay_batch(self.config.batch_size);
        if batch.is_empty() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_psi: 0.0,
                skipped: true,
            };
        }

        let mut total_loss = 0.0;
        let mut total_psi = 0.0;
        for episode in &batch {
            let loss = self.replay_training_step(
                network,
                episode,
                base_learning_rate,
                self.config.replay_dt,
            );
            total_loss += loss;
            total_psi += episode.psi;
        }

        self.cycles_since_replay = 0;
        self.demand_replay_triggered = false;
        let current_psi = total_psi / batch.len().max(1) as f64;
        let sampled_ids: HashSet<EpisodeInstanceId> =
            batch.iter().filter_map(|episode| episode.instance_id).collect();
        let legacy_batch: Vec<&Episode> = batch
            .iter()
            .filter(|episode| episode.instance_id.is_none())
            .collect();

        let mut new_episodes = BinaryHeap::new();
        for mut pe in self.episodes.drain() {
            let exact_match = pe
                .episode
                .instance_id
                .is_some_and(|id| sampled_ids.contains(&id));
            let legacy_match = pe.episode.instance_id.is_none()
                && legacy_batch.iter().any(|be| {
                    (pe.episode.psi - be.psi).abs() < 0.001
                        && pe.episode.timestamp == be.timestamp
                });
            if exact_match || legacy_match {
                pe.episode.replay_count += 1;
                pe.episode.reconsolidate(current_psi);
            }
            new_episodes.push(pe);
        }
        self.episodes = new_episodes;

        let n = batch.len();
        ReplaySessionResult {
            episodes_replayed: n,
            average_loss: total_loss / n as f32,
            average_psi: total_psi / n as f64,
            skipped: false,
        }
    }

    pub fn replay_session_conditioned(
        &mut self,
        network: &mut impl TrainableNetwork,
        base_learning_rate: f32,
        current_bath: Option<[f32; 9]>,
    ) -> ReplaySessionResult {
        if !self.should_replay() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_psi: 0.0,
                skipped: true,
            };
        }

        let batch = self.sample_replay_batch_conditioned(self.config.batch_size, current_bath);
        if batch.is_empty() {
            return ReplaySessionResult {
                episodes_replayed: 0,
                average_loss: 0.0,
                average_psi: 0.0,
                skipped: true,
            };
        }

        let mut total_loss = 0.0;
        let mut total_psi = 0.0;
        for episode in &batch {
            let loss = self.replay_training_step(
                network,
                episode,
                base_learning_rate,
                self.config.replay_dt,
            );
            total_loss += loss;
            total_psi += episode.psi;
        }

        self.cycles_since_replay = 0;
        self.demand_replay_triggered = false;
        let current_psi = total_psi / batch.len().max(1) as f64;
        let sampled_ids: HashSet<EpisodeInstanceId> =
            batch.iter().filter_map(|episode| episode.instance_id).collect();
        let legacy_batch: Vec<&Episode> = batch
            .iter()
            .filter(|episode| episode.instance_id.is_none())
            .collect();

        let mut new_episodes = BinaryHeap::new();
        for mut pe in self.episodes.drain() {
            let exact_match = pe
                .episode
                .instance_id
                .is_some_and(|id| sampled_ids.contains(&id));
            let legacy_match = pe.episode.instance_id.is_none()
                && legacy_batch.iter().any(|be| {
                    (pe.episode.psi - be.psi).abs() < 0.001
                        && pe.episode.timestamp == be.timestamp
                });
            if exact_match || legacy_match {
                pe.episode.replay_count += 1;
                pe.episode.reconsolidate(current_psi);
            }
            new_episodes.push(pe);
        }
        self.episodes = new_episodes;

        let n = batch.len();
        ReplaySessionResult {
            episodes_replayed: n,
            average_loss: total_loss / n as f32,
            average_psi: total_psi / n as f64,
            skipped: false,
        }
    }

    pub fn stats(&self) -> EpisodicMemoryStats {
        EpisodicMemoryStats {
            total_stored: self.total_stored,
            total_evicted: self.total_evicted,
            current_count: self.episodes.len(),
            capacity: self.config.capacity,
            average_psi: self.average_psi,
            min_psi_in_buffer: if self.episodes.is_empty() {
                0.0
            } else {
                self.min_psi_in_buffer
            },
            psi_threshold: self.config.psi_threshold,
            total_replay_steps: self.total_replay_steps,
            average_replay_loss: if self.total_replay_steps > 0 {
                self.sum_replay_loss / self.total_replay_steps as f64
            } else {
                0.0
            },
            cycles_since_replay: self.cycles_since_replay,
            replay_interval: self.config.replay_interval,
            demand_replay_count: self.demand_replay_count,
        }
    }

    /// Clear all *active* stored episodes.
    ///
    /// Quarantined episodes are deliberately preserved. A generic active-memory reset must not
    /// become an implicit destruction path for content that was isolated specifically to remain
    /// reversible. Destruction of quarantined content requires a separate future governed path.
    pub fn clear(&mut self) {
        self.episodes.clear();
        self.min_psi_in_buffer = f64::MAX;
        self.average_psi = 0.0;
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Number of reversibly quarantined episode instances.
    pub fn quarantined_len(&self) -> usize {
        self.quarantined.len()
    }

    /// Whether both active and quarantined storage are empty.
    pub fn is_fully_empty(&self) -> bool {
        self.episodes.is_empty() && self.quarantined.is_empty()
    }

    /// Snapshot all quarantined instances in deterministic ID order.
    pub fn quarantined_instances(&self) -> Vec<QuarantinedEpisode> {
        let mut values: Vec<_> = self.quarantined.values().cloned().collect();
        values.sort_by_key(|entry| entry.instance_id);
        values
    }

    /// Read one quarantined instance without activating it.
    pub fn quarantined_instance(&self, id: EpisodeInstanceId) -> Option<QuarantinedEpisode> {
        self.quarantined.get(&id).cloned()
    }

    /// Reversibly remove one exact occurrence from all active memory behavior.
    ///
    /// This is a mechanism-only primitive. Production exogenous callers must still cross the
    /// welfare/authority assurance boundary. Quarantine does not increment `total_evicted` and
    /// does not rewrite the episode or its lifecycle counters.
    pub fn quarantine_instance(
        &mut self,
        id: EpisodeInstanceId,
    ) -> Result<QuarantinedEpisode, EpisodicQuarantineError> {
        if self.quarantined.contains_key(&id) {
            return Err(EpisodicQuarantineError::AlreadyQuarantined(id));
        }

        let mut retained = BinaryHeap::new();
        let mut selected = None;
        while let Some(pe) = self.episodes.pop() {
            if pe.episode.instance_id == Some(id) && selected.is_none() {
                selected = Some(pe);
            } else {
                retained.push(pe);
            }
        }
        self.episodes = retained;

        let selected = selected.ok_or(EpisodicQuarantineError::InstanceNotFound(id))?;
        let quarantined = QuarantinedEpisode {
            instance_id: id,
            episode: selected.episode,
            score_at_quarantine: selected.score,
            quarantined_at_cycle: self.current_cycle,
        };
        self.quarantined.insert(id, quarantined.clone());
        self.recompute_active_psi_stats();
        Ok(quarantined)
    }

    /// Restore one quarantined occurrence with the same instance ID and episode state.
    ///
    /// Priority is recomputed at the current cycle because recency is dynamic; the identity and
    /// episode lifecycle state are not rewritten and restoration is not counted as a new store.
    pub fn restore_quarantined_instance(
        &mut self,
        id: EpisodeInstanceId,
    ) -> Result<QuarantinedEpisode, EpisodicQuarantineError> {
        if self
            .episodes
            .iter()
            .any(|pe| pe.episode.instance_id == Some(id))
        {
            return Err(EpisodicQuarantineError::ActiveInstanceCollision(id));
        }
        let quarantined = self
            .quarantined
            .remove(&id)
            .ok_or(EpisodicQuarantineError::InstanceNotFound(id))?;
        let episode = quarantined.episode.clone();
        let score = episode.priority_score(self.current_cycle, self.config.recency_weight);
        self.episodes.push(PrioritizedEpisode { episode, score });
        self.recompute_active_psi_stats();
        Ok(quarantined)
    }

    /// Active episodes sorted by Psi, paired with their exact occurrence identity.
    pub fn get_top_episode_instances(
        &self,
        n: usize,
    ) -> Vec<(EpisodeInstanceId, Episode)> {
        let mut sorted: Vec<_> = self.episodes.iter().collect();
        sorted.sort_by(|a, b| {
            b.episode
                .psi
                .partial_cmp(&a.episode.psi)
                .unwrap_or(Ordering::Equal)
        });
        sorted
            .into_iter()
            .take(n)
            .filter_map(|pe| pe.episode.instance_id.map(|id| (id, pe.episode.clone())))
            .collect()
    }

    fn recompute_active_psi_stats(&mut self) {
        if self.episodes.is_empty() {
            self.average_psi = 0.0;
            self.min_psi_in_buffer = f64::MAX;
            return;
        }
        let mut total = 0.0;
        let mut minimum = f64::MAX;
        for pe in &self.episodes {
            total += pe.episode.psi;
            minimum = minimum.min(pe.episode.psi);
        }
        self.average_psi = total / self.episodes.len() as f64;
        self.min_psi_in_buffer = minimum;
    }

    pub fn boost_causal_consolidation(&mut self, cycle_numbers: &[u64], boost: f64) {
        if cycle_numbers.is_empty() {
            return;
        }
        let mut all: Vec<PrioritizedEpisode> = self.episodes.drain().collect();
        for pe in &mut all {
            if cycle_numbers.contains(&pe.episode.timestamp) {
                pe.episode.consolidation_strength =
                    (pe.episode.consolidation_strength + boost).min(5.0);
                pe.score = pe
                    .episode
                    .priority_score(self.current_cycle, self.config.recency_weight);
            }
        }
        self.episodes.extend(all);
    }

    pub fn boost_recent_consolidation(&mut self, boost: f64) {
        if boost <= 0.0 || self.episodes.is_empty() {
            return;
        }
        let mut all: Vec<PrioritizedEpisode> = self.episodes.drain().collect();
        all.sort_by_key(|a| std::cmp::Reverse(a.episode.timestamp));
        for pe in all.iter_mut().take(3) {
            pe.episode.consolidation_strength =
                (pe.episode.consolidation_strength + boost).min(5.0);
            pe.score = pe
                .episode
                .priority_score(self.current_cycle, self.config.recency_weight);
        }
        self.episodes.extend(all);
    }

    pub fn get_top_episodes(&self, n: usize) -> Vec<Episode> {
        let mut sorted: Vec<_> = self.episodes.iter().collect();
        sorted.sort_by(|a, b| {
            b.episode
                .psi
                .partial_cmp(&a.episode.psi)
                .unwrap_or(Ordering::Equal)
        });
        sorted
            .into_iter()
            .take(n)
            .map(|pe| pe.episode.clone())
            .collect()
    }

    pub fn retrieve_by_embedding_similarity(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Vec<(Episode, f32)> {
        if query.is_empty() {
            return Vec::new();
        }
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < 1e-12 {
            return Vec::new();
        }
        let mut scored: Vec<(Episode, f32)> = self
            .episodes
            .iter()
            .filter_map(|pe| {
                let emb = pe.episode.semantic_embedding.as_ref()?;
                if emb.len() != query.len() {
                    return None;
                }
                let dot: f32 = emb.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let emb_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if emb_norm < 1e-12 {
                    return None;
                }
                let sim = dot / (emb_norm * query_norm);
                Some((pe.episode.clone(), sim))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn retrieve_by_input_similarity(&self, query: &[f32], top_k: usize) -> Vec<(Episode, f32)> {
        if query.is_empty() {
            return Vec::new();
        }
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < 1e-12 {
            return Vec::new();
        }
        let mut scored: Vec<(Episode, f32)> = self
            .episodes
            .iter()
            .filter_map(|pe| {
                let input = &pe.episode.input.values;
                if input.len() != query.len() {
                    return None;
                }
                let dot: f32 = input.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
                if input_norm < 1e-12 {
                    return None;
                }
                Some((pe.episode.clone(), dot / (input_norm * query_norm)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn prune(&mut self, threshold: f64) -> usize {
        let initial_len = self.episodes.len();
        let current_ts = self.current_cycle;
        let recency_w = self.config.recency_weight;
        let episodes: Vec<PrioritizedEpisode> = self.episodes.drain().collect();
        let filtered: Vec<PrioritizedEpisode> = episodes
            .into_iter()
            .filter(|pe| pe.episode.survival_value(current_ts, recency_w) >= threshold)
            .collect();
        let pruned_count = initial_len - filtered.len();
        self.total_evicted += pruned_count as u64;
        self.episodes.extend(filtered);
        if pruned_count > 0 {
            self.recompute_active_psi_stats();
            tracing::info!(
                target: "symthaea::memory::episodic",
                pruned = pruned_count,
                remaining = self.episodes.len(),
                threshold = threshold,
                "Causal pruning complete (The Art of Forgetting)"
            );
        }
        pruned_count
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySessionResult {
    pub episodes_replayed: usize,
    pub average_loss: f32,
    pub average_psi: f64,
    pub skipped: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryStats {
    pub total_stored: u64,
    pub total_evicted: u64,
    pub current_count: usize,
    pub capacity: usize,
    pub average_psi: f64,
    pub min_psi_in_buffer: f64,
    pub psi_threshold: f64,
    pub total_replay_steps: u64,
    pub average_replay_loss: f64,
    pub cycles_since_replay: usize,
    pub replay_interval: usize,
    pub demand_replay_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_episode(psi: f64, timestamp: u64) -> Episode {
        let input = ContinuousHV::random(256, timestamp);
        let output = ContinuousHV::random(256, timestamp + 1);
        Episode::new(input, output, psi, timestamp)
    }

    #[test]
    fn test_episode_creation() {
        let episode = make_test_episode(0.5, 100);
        assert_eq!(episode.psi, 0.5);
        assert_eq!(episode.timestamp, 100);
        assert_eq!(episode.replay_count, 0);
        assert!(episode.instance_id.is_none());
    }

    #[test]
    fn test_store_if_significant_threshold() {
        let config = EpisodicReplayConfig {
            psi_threshold: 0.4,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        let low_phi = make_test_episode(0.3, 1);
        assert!(!memory.store_if_significant(low_phi));
        assert_eq!(memory.len(), 0);
        let high_phi = make_test_episode(0.5, 2);
        assert!(memory.store_if_significant(high_phi));
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_identical_content_gets_distinct_instance_ids() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let episode = make_test_episode(0.8, 10);
        let first = memory
            .store_if_significant_with_id(episode.clone())
            .expect("first insert");
        let second = memory
            .store_if_significant_with_id(episode)
            .expect("second insert");
        assert_ne!(first, second);
        let instances = memory.get_top_episode_instances(10);
        assert_eq!(instances.len(), 2);
        assert!(instances.iter().any(|(id, _)| *id == first));
        assert!(instances.iter().any(|(id, _)| *id == second));
    }

    #[test]
    fn test_quarantine_is_instance_exact_and_reversible() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let episode = make_test_episode(0.8, 10);
        let first = memory
            .store_if_significant_with_id(episode.clone())
            .unwrap();
        let second = memory.store_if_significant_with_id(episode).unwrap();

        let quarantined = memory.quarantine_instance(first).unwrap();
        assert_eq!(quarantined.instance_id, first);
        assert_eq!(memory.len(), 1);
        assert_eq!(memory.quarantined_len(), 1);
        assert_eq!(memory.get_top_episode_instances(10)[0].0, second);
        assert!(memory.quarantined_instance(first).is_some());

        let restored = memory.restore_quarantined_instance(first).unwrap();
        assert_eq!(restored.instance_id, first);
        assert_eq!(memory.len(), 2);
        assert_eq!(memory.quarantined_len(), 0);
        let active_ids: HashSet<_> = memory
            .get_top_episode_instances(10)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(active_ids, HashSet::from([first, second]));
    }

    #[test]
    fn test_active_clear_preserves_quarantine() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory
            .store_if_significant_with_id(make_test_episode(0.8, 10))
            .unwrap();
        memory.quarantine_instance(id).unwrap();
        memory.store_if_significant(make_test_episode(0.9, 11));
        memory.clear();
        assert!(memory.is_empty());
        assert_eq!(memory.quarantined_len(), 1);
        assert!(!memory.is_fully_empty());
        assert_eq!(memory.quarantined_instance(id).unwrap().instance_id, id);
    }

    #[test]
    fn test_replay_updates_only_sampled_duplicate_instance() {
        let config = EpisodicReplayConfig {
            psi_threshold: 0.0,
            replay_interval: 1,
            batch_size: 1,
            min_episodes_for_replay: 1,
            psi_weighted_sampling: false,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        let duplicate = make_test_episode(0.8, 10);
        let first = memory
            .store_if_significant_with_id(duplicate.clone())
            .unwrap();
        let second = memory.store_if_significant_with_id(duplicate).unwrap();
        let predicted_sample = memory.sample_replay_batch(1);
        let sampled_id = predicted_sample[0].instance_id.unwrap();
        assert!(sampled_id == first || sampled_id == second);

        let mut net = MockTrainableNetwork::new();
        let result = memory.replay_session(&mut net, 0.01);
        assert!(!result.skipped);
        let instances = memory.get_top_episode_instances(10);
        let sampled = instances
            .iter()
            .find(|(id, _)| *id == sampled_id)
            .unwrap();
        let unsampled = instances
            .iter()
            .find(|(id, _)| *id != sampled_id)
            .unwrap();
        assert_eq!(sampled.1.replay_count, 1);
        assert_eq!(unsampled.1.replay_count, 0);
    }

    #[test]
    fn test_high_phi_episodes_preferentially_stored() {
        let config = EpisodicReplayConfig {
            capacity: 5,
            psi_threshold: 0.1,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        for i in 0..10 {
            let phi = 0.1 + (i as f64 * 0.1);
            memory.store_if_significant(make_test_episode(phi, i as u64));
        }
        let top = memory.get_top_episodes(3);
        assert!(top.len() >= 3);
        for (i, ep) in top.iter().enumerate() {
            if i > 0 {
                assert!(ep.psi <= top[i - 1].psi);
            }
        }
    }

    #[test]
    fn test_sample_replay_batch() {
        let config = EpisodicReplayConfig {
            psi_threshold: 0.1,
            min_episodes_for_replay: 5,
            psi_weighted_sampling: false,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        for i in 0..20 {
            let phi = 0.3 + (i as f64 * 0.03);
            memory.store_if_significant(make_test_episode(phi, i as u64));
        }
        let batch = memory.sample_replay_batch(5);
        assert_eq!(batch.len(), 5);
        for ep in &batch {
            assert!(ep.psi >= 0.3);
            assert!(ep.instance_id.is_some());
        }
    }

    #[test]
    fn test_phi_weighted_sampling_prefers_high_phi() {
        let config = EpisodicReplayConfig {
            psi_threshold: 0.1,
            psi_weighted_sampling: true,
            sampling_temperature: 0.5,
            min_episodes_for_replay: 5,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        for i in 0..100 {
            let phi = 0.2 + (i as f64 * 0.008);
            memory.store_if_significant(make_test_episode(phi, i as u64));
        }
        let mut high_phi_count = 0;
        let mut total_count = 0;
        for _ in 0..10 {
            let batch = memory.sample_replay_batch(10);
            for ep in &batch {
                total_count += 1;
                if ep.psi > 0.7 {
                    high_phi_count += 1;
                }
            }
        }
        let high_phi_ratio = high_phi_count as f64 / total_count as f64;
        assert!(high_phi_ratio > 0.3);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = EpisodicReplayConfig {
            capacity: 50,
            psi_threshold: 0.3,
            ..Default::default()
        };
        let mut memory = EpisodicMemory::new(config);
        for i in 0..30 {
            let phi = 0.2 + (i as f64 * 0.02);
            memory.store_if_significant(make_test_episode(phi, i as u64));
        }
        let stats = memory.stats();
        assert!(stats.total_stored > 0);
        assert!(stats.current_count <= stats.capacity);
        assert!(stats.average_psi >= stats.psi_threshold);
        assert!(stats.min_psi_in_buffer >= stats.psi_threshold);
    }

    #[test]
    fn test_priority_score_calculation() {
        let episode = Episode::with_metadata(
            ContinuousHV::random(256, 42),
            ContinuousHV::random(256, 43),
            0.8,
            100,
            0.5,
            0.9,
            0.7,
        );
        let score1 = episode.priority_score(100, 0.2);
        let score2 = episode.priority_score(1000, 0.2);
        assert!(score1 >= score2);
        assert!(score1 > 0.5);
    }

    #[test]
    fn test_clear_memory() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::default());
        for i in 0..10 {
            memory.store_if_significant(make_test_episode(0.5, i as u64));
        }
        assert!(!memory.is_empty());
        memory.clear();
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_da_tagged_episode_higher_priority() {
        let mut ep_high = make_test_episode(0.5, 100);
        ep_high.dopamine_at_encoding = Some(0.8);
        let mut ep_low = make_test_episode(0.5, 100);
        ep_low.dopamine_at_encoding = Some(0.2);
        let high_score = ep_high.priority_score(200, 0.5);
        let low_score = ep_low.priority_score(200, 0.5);
        assert!(high_score > low_score);
    }

    #[test]
    fn test_da_replay_lr_scaling() {
        let config = EpisodicReplayConfig::default();
        let base_lr = 0.01_f32;
        let da_high_scale = 0.7 + 0.8 * 0.6;
        let da_low_scale = 0.7 + 0.2 * 0.6;
        let lr_high = base_lr * config.replay_learning_rate_multiplier * da_high_scale;
        let lr_low = base_lr * config.replay_learning_rate_multiplier * da_low_scale;
        assert!(lr_high > lr_low);
    }

    #[test]
    fn test_da_tag_backwards_compat() {
        let ep = make_test_episode(0.5, 100);
        assert!(ep.dopamine_at_encoding.is_none());
        let score = ep.priority_score(200, 0.5);
        assert!(score.is_finite());
    }

    #[test]
    fn test_episode_with_bath_state() {
        let input = ContinuousHV::from_vec(vec![0.0; 128]);
        let output = ContinuousHV::from_vec(vec![0.0; 128]);
        let bath = [0.5, 0.4, 0.6, 0.3, 0.4, 0.3, 0.3, 0.2, 0.3];
        let ep = Episode::new(input, output, 0.8, 100).with_bath_state(bath);
        assert_eq!(ep.bath_state_at_encoding.unwrap(), bath);
    }

    #[test]
    fn test_episode_default_no_bath_state() {
        let input = ContinuousHV::from_vec(vec![0.0; 128]);
        let output = ContinuousHV::from_vec(vec![0.0; 128]);
        let ep = Episode::new(input, output, 0.8, 100);
        assert!(ep.bath_state_at_encoding.is_none());
    }

    #[test]
    fn test_bath_cosine_identical() {
        let a = [0.5, 0.4, 0.6, 0.3, 0.4, 0.3, 0.3, 0.2, 0.3];
        let sim = bath_cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bath_cosine_zero_magnitude() {
        let a = [0.0; 9];
        let b = [0.5; 9];
        assert_eq!(bath_cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_conditioned_replay_prefers_similar() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig {
            capacity: 100,
            psi_threshold: 0.0,
            replay_interval: 1,
            batch_size: 5,
            min_episodes_for_replay: 1,
            psi_weighted_sampling: true,
            ..EpisodicReplayConfig::default()
        });
        memory.current_cycle = 42;
        let similar_bath = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3];
        let different_bath = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.3];
        for i in 0..20 {
            let input = ContinuousHV::from_vec(vec![i as f32 / 20.0; 128]);
            let output = ContinuousHV::from_vec(vec![0.0; 128]);
            let bath = if i < 10 { similar_bath } else { different_bath };
            let ep = Episode::new(input, output, 0.5, i as u64).with_bath_state(bath);
            memory.store_if_significant(ep);
        }
        let batch = memory.sample_replay_batch_conditioned(5, Some(similar_bath));
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_conditioned_replay_graceful_without_state() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig {
            capacity: 100,
            psi_threshold: 0.0,
            replay_interval: 1,
            batch_size: 3,
            min_episodes_for_replay: 1,
            psi_weighted_sampling: true,
            ..EpisodicReplayConfig::default()
        });
        memory.current_cycle = 10;
        for i in 0..10 {
            let input = ContinuousHV::from_vec(vec![i as f32 / 10.0; 128]);
            let output = ContinuousHV::from_vec(vec![0.0; 128]);
            memory.store_if_significant(Episode::new(input, output, 0.5, i as u64));
        }
        let batch = memory.sample_replay_batch_conditioned(3, Some([0.5; 9]));
        assert!(!batch.is_empty());
    }

    struct MockTrainableNetwork {
        train_calls: usize,
        should_fail: bool,
        last_lr: f32,
    }

    impl MockTrainableNetwork {
        fn new() -> Self {
            Self {
                train_calls: 0,
                should_fail: false,
                last_lr: 0.0,
            }
        }

        fn failing() -> Self {
            Self {
                train_calls: 0,
                should_fail: true,
                last_lr: 0.0,
            }
        }
    }

    impl crate::TrainableNetwork for MockTrainableNetwork {
        fn train_step(
            &mut self,
            _input: &ndarray::Array1<f32>,
            _target: &ndarray::Array1<f32>,
            _dt: f32,
            learning_rate: f32,
        ) -> anyhow::Result<f32> {
            self.train_calls += 1;
            self.last_lr = learning_rate;
            if self.should_fail {
                anyhow::bail!("mock training failure")
            } else {
                Ok(0.01)
            }
        }
    }

    #[test]
    fn test_mock_trainable_basic() {
        let mut net = MockTrainableNetwork::new();
        let input = ndarray::Array1::zeros(128);
        let target = ndarray::Array1::ones(128);
        let loss = net.train_step(&input, &target, 0.02, 0.001).unwrap();
        assert!((loss - 0.01).abs() < 1e-6);
        assert_eq!(net.train_calls, 1);
    }

    #[test]
    fn test_mock_trainable_call_count() {
        let mut net = MockTrainableNetwork::new();
        let input = ndarray::Array1::zeros(64);
        let target = ndarray::Array1::ones(64);
        for _ in 0..10 {
            let _ = net.train_step(&input, &target, 0.02, 0.001);
        }
        assert_eq!(net.train_calls, 10);
    }

    #[test]
    fn test_mock_trainable_failure_mode() {
        let mut net = MockTrainableNetwork::failing();
        let input = ndarray::Array1::zeros(64);
        let target = ndarray::Array1::ones(64);
        let result = net.train_step(&input, &target, 0.02, 0.001);
        assert!(result.is_err());
        assert_eq!(net.train_calls, 1);
    }

    #[test]
    fn test_mock_trainable_zero_lr() {
        let mut net = MockTrainableNetwork::new();
        let input = ndarray::Array1::zeros(64);
        let target = ndarray::Array1::ones(64);
        let loss = net.train_step(&input, &target, 0.02, 0.0).unwrap();
        assert!((loss - 0.01).abs() < 1e-6);
        assert_eq!(net.last_lr, 0.0);
        assert_eq!(net.train_calls, 1);
    }

    fn replay_config_immediate() -> EpisodicReplayConfig {
        EpisodicReplayConfig {
            capacity: 100,
            psi_threshold: 0.0,
            replay_interval: 1,
            batch_size: 8,
            min_episodes_for_replay: 1,
            psi_weighted_sampling: false,
            ..EpisodicReplayConfig::default()
        }
    }

    #[test]
    fn test_replay_training_step_single() {
        let mut memory = EpisodicMemory::new(replay_config_immediate());
        let mut net = MockTrainableNetwork::new();
        let ep = make_test_episode(0.8, 1);
        memory.store_if_significant(ep.clone());
        let loss = memory.replay_training_step(&mut net, &ep, 0.01, 0.02);
        assert!(loss.is_finite());
        assert_eq!(net.train_calls, 1);
        assert!(net.last_lr > 0.0);
    }

    #[test]
    fn test_replay_training_step_phi_weighted_lr() {
        let mut memory = EpisodicMemory::new(replay_config_immediate());
        let mut net = MockTrainableNetwork::new();
        let mut ep_high_da = make_test_episode(0.8, 1);
        ep_high_da.dopamine_at_encoding = Some(0.9);
        let mut ep_low_da = make_test_episode(0.8, 2);
        ep_low_da.dopamine_at_encoding = Some(0.1);
        memory.replay_training_step(&mut net, &ep_high_da, 0.01, 0.02);
        let lr_high = net.last_lr;
        memory.replay_training_step(&mut net, &ep_low_da, 0.01, 0.02);
        let lr_low = net.last_lr;
        assert!(lr_high > lr_low);
    }

    #[test]
    fn test_replay_session_multiple() {
        let config = EpisodicReplayConfig {
            batch_size: 5,
            ..replay_config_immediate()
        };
        let mut memory = EpisodicMemory::new(config);
        let mut net = MockTrainableNetwork::new();
        for i in 0..10 {
            memory.store_if_significant(make_test_episode(0.5 + (i as f64 * 0.05), i as u64));
        }
        let result = memory.replay_session(&mut net, 0.01);
        assert!(!result.skipped);
        assert_eq!(result.episodes_replayed, 5);
        assert_eq!(net.train_calls, 5);
        assert!(result.average_loss.is_finite());
        assert!(result.average_psi > 0.0);
    }

    #[test]
    fn test_replay_empty_buffer() {
        let mut memory = EpisodicMemory::new(replay_config_immediate());
        let mut net = MockTrainableNetwork::new();
        memory.current_cycle = 200;
        memory.cycles_since_replay = 200;
        let result = memory.replay_session(&mut net, 0.01);
        assert!(result.skipped);
        assert_eq!(result.episodes_replayed, 0);
        assert_eq!(net.train_calls, 0);
    }

    #[test]
    fn test_replay_respects_capacity() {
        let config = EpisodicReplayConfig {
            capacity: 5,
            psi_threshold: 0.0,
            replay_interval: 1,
            batch_size: 3,
            min_episodes_for_replay: 1,
            psi_weighted_sampling: false,
            ..EpisodicReplayConfig::default()
        };
        let mut memory = EpisodicMemory::new(config);
        for i in 0..20 {
            memory.store_if_significant(make_test_episode(0.3 + (i as f64 * 0.03), i as u64));
        }
        let stats = memory.stats();
        assert_eq!(stats.total_stored, 20);
        assert!(stats.total_evicted > 0);
        let mut net = MockTrainableNetwork::new();
        let result = memory.replay_session(&mut net, 0.01);
        assert!(!result.skipped);
        assert!(result.episodes_replayed > 0);
    }
}
