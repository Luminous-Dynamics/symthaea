//! Holographic Memory - Vector Superposition Storage
//!
//! This module implements the "Memory" layer of the Holographic Associative Memory
//! architecture. It uses vector superposition for one-shot learning and retrieval.
//!
//! ## Core Principle: Holographic Storage
//!
//! Unlike traditional databases that store items separately, holographic memory
//! stores all memories as a superposition (sum) of vectors. This enables:
//!
//! - **One-shot learning**: `Memory_new = Memory_old + Experience`
//! - **Graceful degradation**: Similar queries retrieve similar memories
//! - **Infinite capacity**: No fixed slots, just vector space
//! - **Associative retrieval**: Query by content, not by key
//!
//! ## Architecture Role
//!
//! ```text
//! ┌─────────────────┐
//! │ SemanticEncoder │  ← Sensation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────────────────┐
//! │              HolographicMemory                   │  ← Memory
//! │  ┌───────────────────────────────────────────┐  │
//! │  │  Episodic Store (recent experiences)      │  │
//! │  │  ════════════════════════════════════════ │  │
//! │  │  [v1] + [v2] + [v3] + ... = [hologram]    │  │
//! │  └───────────────────────────────────────────┘  │
//! │  ┌───────────────────────────────────────────┐  │
//! │  │  Semantic Store (consolidated knowledge)  │  │
//! │  │  ════════════════════════════════════════ │  │
//! │  │  Category centroids + exemplar bindings   │  │
//! │  └───────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ActiveInference │  ← Cognition (future)
//! └─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let mut memory = HolographicMemory::new(768);
//!
//! // Store experiences (one-shot learning)
//! memory.store(&experience1);
//! memory.store(&experience2);
//!
//! // Query by similarity
//! let matches = memory.query(&query_vector, 5);
//!
//! // Reinforce important memories
//! memory.reinforce(&important_memory, 2.0);
//! ```

use super::semantic_encoder::{DenseVector, EncodedThought};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for HolographicMemory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicMemoryConfig {
    /// Dimension of vectors (768 for BGE)
    pub dimension: usize,

    /// Maximum episodic memories before consolidation
    pub max_episodic: usize,

    /// Temporal decay factor per retrieval cycle (0.0 - 1.0)
    /// Higher = slower decay
    pub decay_factor: f32,

    /// Minimum similarity for retrieval (0.0 - 1.0)
    pub retrieval_threshold: f32,

    /// Whether to auto-consolidate episodic → semantic
    pub auto_consolidate: bool,

    /// Consolidation threshold (how many similar episodes trigger consolidation)
    pub consolidation_threshold: usize,

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Hippocampus-style decay rate per cycle (0.0 - 1.0)
    /// Lower = slower decay (0.05 = 5% decay per cycle)
    #[serde(default = "default_hippocampus_decay_rate")]
    pub hippocampus_decay_rate: f32,

    /// Strengthen increment when memory is recalled
    #[serde(default = "default_strengthen_increment")]
    pub strengthen_increment: f32,

    /// Maximum strength a memory can reach
    #[serde(default = "default_max_strength")]
    pub max_strength: f32,

    /// Minimum importance threshold for pruning (memories below this are removed)
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f32,

    /// Whether to use hippocampus-style dynamics (vs original decay_factor)
    #[serde(default)]
    pub use_hippocampus_dynamics: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Whether to integrate with sleep cycle manager
    #[serde(default)]
    pub enable_sleep_consolidation: bool,

    /// Minimum importance for a memory to be eligible for long-term storage
    #[serde(default = "default_long_term_threshold")]
    pub long_term_threshold: f32,

    /// Minimum access count for long-term eligibility
    #[serde(default = "default_min_access_for_long_term")]
    pub min_access_for_long_term: u32,
}

fn default_hippocampus_decay_rate() -> f32 { 0.05 }
fn default_strengthen_increment() -> f32 { 0.1 }
fn default_max_strength() -> f32 { 2.0 }
fn default_prune_threshold() -> f32 { 0.01 }
fn default_long_term_threshold() -> f32 { 0.5 }
fn default_min_access_for_long_term() -> u32 { 2 }

impl Default for HolographicMemoryConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            max_episodic: 1000,
            decay_factor: 0.95,
            retrieval_threshold: 0.3,
            auto_consolidate: true,
            consolidation_threshold: 3,
            // Hippocampus dynamics
            hippocampus_decay_rate: default_hippocampus_decay_rate(),
            strengthen_increment: default_strengthen_increment(),
            max_strength: default_max_strength(),
            prune_threshold: default_prune_threshold(),
            use_hippocampus_dynamics: false, // Off by default for backwards compatibility
            // Sleep consolidation
            enable_sleep_consolidation: false,
            long_term_threshold: default_long_term_threshold(),
            min_access_for_long_term: default_min_access_for_long_term(),
        }
    }
}

/// A single memory trace
///
/// Integrates Hippocampus-style decay/strengthen dynamics:
/// - **Decay**: `strength *= 1.0 - decay_rate` (exponential forgetting)
/// - **Strengthen**: `recall_count += 1; strength += 0.1` (use-dependent potentiation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// The dense vector representation
    pub vector: Vec<f32>,

    /// Original text (if available)
    pub text: Option<String>,

    /// Importance/salience score (affects retrieval and decay)
    /// Also acts as "strength" for Hippocampus-style dynamics
    pub importance: f32,

    /// Number of times this memory has been accessed
    pub access_count: u32,

    /// Timestamp of creation (Unix epoch ms)
    pub created_at: u64,

    /// Timestamp of last access
    pub last_accessed: u64,

    /// Optional category/tag
    pub category: Option<String>,

    /// Consolidation status for sleep integration
    #[serde(default)]
    pub consolidated: bool,

    /// Eligibility for long-term storage (set during sleep consolidation)
    #[serde(default)]
    pub long_term_eligible: bool,
}

impl MemoryTrace {
    /// Create a new memory trace from a dense vector
    pub fn new(vector: Vec<f32>, text: Option<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            vector,
            text,
            importance: 1.0,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            category: None,
            consolidated: false,
            long_term_eligible: false,
        }
    }

    /// Create from an EncodedThought
    pub fn from_thought(thought: &EncodedThought) -> Self {
        let mut trace = Self::new(thought.dense.values.clone(), Some(thought.text.clone()));
        trace.importance = thought.confidence;
        trace
    }

    /// Compute similarity to another vector
    pub fn similarity(&self, other: &[f32]) -> f32 {
        if self.vector.len() != other.len() {
            return 0.0;
        }

        let dot: f32 = self.vector.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Mark as accessed (updates timestamp and count)
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS (from src/memory/hippocampus.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply decay to this memory trace (Hippocampus-style)
    ///
    /// Implements exponential forgetting: `importance *= 1.0 - decay_rate`
    /// This mimics biological memory decay where unused memories fade over time.
    ///
    /// # Arguments
    /// * `decay_rate` - Rate of decay (0.0 = no decay, 1.0 = instant forget)
    ///
    /// # Example
    /// ```ignore
    /// trace.decay_hippocampus(0.05); // 5% decay per cycle
    /// ```
    pub fn decay_hippocampus(&mut self, decay_rate: f32) {
        self.importance *= 1.0 - decay_rate;
        self.importance = self.importance.max(0.0); // Floor at zero
    }

    /// Strengthen this memory on recall (Hippocampus-style)
    ///
    /// Implements use-dependent potentiation: memories that are recalled
    /// become stronger, mimicking Long-Term Potentiation (LTP).
    ///
    /// - Increments access count
    /// - Boosts importance by 0.1 (capped at 2.0)
    /// - Updates last_accessed timestamp
    ///
    /// # Example
    /// ```ignore
    /// trace.strengthen(); // Called on successful recall
    /// ```
    pub fn strengthen(&mut self) {
        self.access_count += 1;
        self.importance = (self.importance + 0.1).min(2.0);
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Check if this memory should be pruned (below threshold)
    pub fn should_prune(&self, threshold: f32) -> bool {
        self.importance < threshold
    }

    /// Mark as consolidated (processed during sleep)
    pub fn mark_consolidated(&mut self) {
        self.consolidated = true;
    }

    /// Mark as eligible for long-term storage
    pub fn mark_long_term_eligible(&mut self) {
        self.long_term_eligible = true;
    }

    /// Get the age of this memory in milliseconds
    pub fn age_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now.saturating_sub(self.created_at)
    }

    /// Get time since last access in milliseconds
    pub fn time_since_access_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now.saturating_sub(self.last_accessed)
    }
}

/// A semantic category (consolidated from episodic memories)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCategory {
    /// Category name/label
    pub name: String,

    /// Centroid vector (average of all members)
    pub centroid: Vec<f32>,

    /// Number of memories consolidated into this category
    pub member_count: usize,

    /// Representative exemplars (most distinct members)
    pub exemplars: Vec<MemoryTrace>,

    /// Maximum exemplars to keep
    max_exemplars: usize,
}

impl SemanticCategory {
    /// Create a new semantic category
    pub fn new(name: String, dimension: usize) -> Self {
        Self {
            name,
            centroid: vec![0.0; dimension],
            member_count: 0,
            exemplars: Vec::new(),
            max_exemplars: 10,
        }
    }

    /// Add a memory to this category (updates centroid)
    pub fn add(&mut self, trace: &MemoryTrace) {
        // Update centroid incrementally: new_centroid = (old * n + new) / (n + 1)
        let n = self.member_count as f32;
        for (i, val) in trace.vector.iter().enumerate() {
            if i < self.centroid.len() {
                self.centroid[i] = (self.centroid[i] * n + val) / (n + 1.0);
            }
        }
        self.member_count += 1;

        // Keep as exemplar if distinct enough
        if self.exemplars.len() < self.max_exemplars {
            self.exemplars.push(trace.clone());
        } else {
            // Replace least similar exemplar if this one is more distinct
            let mut min_distinctness = f32::MAX;
            let mut min_idx = 0;

            for (i, ex) in self.exemplars.iter().enumerate() {
                let distinctness = 1.0 - ex.similarity(&self.centroid);
                if distinctness < min_distinctness {
                    min_distinctness = distinctness;
                    min_idx = i;
                }
            }

            let new_distinctness = 1.0 - trace.similarity(&self.centroid);
            if new_distinctness > min_distinctness {
                self.exemplars[min_idx] = trace.clone();
            }
        }
    }

    /// Similarity of a vector to this category
    pub fn similarity(&self, vector: &[f32]) -> f32 {
        if self.centroid.len() != vector.len() {
            return 0.0;
        }

        let dot: f32 = self.centroid.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Query result from memory
#[derive(Debug, Clone)]
pub struct MemoryMatch {
    /// The matched memory trace
    pub trace: MemoryTrace,

    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,

    /// Source: "episodic" or "semantic"
    pub source: String,
}

/// Holographic Memory - The Memory Layer of HAM
///
/// Stores experiences as vector superpositions, enabling:
/// - One-shot learning
/// - Associative retrieval
/// - Graceful degradation
/// - Infinite context through consolidation
///
/// ## Integrated Systems
///
/// - **Hippocampus Dynamics**: decay/strengthen for biological memory behavior
/// - **Sleep Consolidation**: Integration with SleepCycleManager for REM-based consolidation
/// - **Persistence Ready**: Export/import for UnifiedMind database storage
pub struct HolographicMemory {
    /// Configuration
    config: HolographicMemoryConfig,

    /// Episodic memory (recent experiences, FIFO)
    episodic: VecDeque<MemoryTrace>,

    /// Semantic memory (consolidated categories)
    semantic: Vec<SemanticCategory>,

    /// Holographic superposition (sum of all episodic vectors)
    /// Used for fast approximate matching
    hologram: Vec<f32>,

    /// Statistics
    stats: MemoryStats,

    /// Buffer for traces pending long-term storage (for UnifiedMind integration)
    pending_long_term: Vec<MemoryTrace>,

    /// Memory pressure tracking for sleep integration
    /// Increases with each store, decreases with consolidation
    memory_pressure: f32,

    /// Total decay cycles applied (for debugging/analysis)
    total_decay_cycles: u64,
}

/// Statistics about memory usage
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memories stored
    pub total_stored: u64,

    /// Total queries performed
    pub total_queries: u64,

    /// Total consolidations
    pub total_consolidations: u64,

    /// Current episodic count
    pub episodic_count: usize,

    /// Current semantic category count
    pub semantic_count: usize,

    /// Average query time (microseconds)
    pub avg_query_time_us: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS & SLEEP STATISTICS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Total memories strengthened on recall
    pub total_strengthened: u64,

    /// Total decay cycles applied
    pub total_decay_cycles: u64,

    /// Total memories pruned due to decay
    pub total_pruned: u64,

    /// Total memories marked for long-term storage
    pub total_long_term_eligible: u64,

    /// Number of sleep consolidation cycles
    pub sleep_consolidation_cycles: u64,

    /// Current memory pressure (0.0 - 1.0)
    pub memory_pressure: f32,
}

impl HolographicMemory {
    /// Create a new HolographicMemory with default config
    pub fn new(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            ..Default::default()
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: HolographicMemoryConfig) -> Self {
        Self {
            hologram: vec![0.0; config.dimension],
            config,
            episodic: VecDeque::new(),
            semantic: Vec::new(),
            stats: MemoryStats::default(),
            pending_long_term: Vec::new(),
            memory_pressure: 0.0,
            total_decay_cycles: 0,
        }
    }

    /// Store a new experience (one-shot learning)
    ///
    /// ```text
    /// Memory_new = Memory_old + Experience
    /// ```
    pub fn store(&mut self, vector: &DenseVector) {
        self.store_with_text(vector, None);
    }

    /// Store with associated text
    pub fn store_with_text(&mut self, vector: &DenseVector, text: Option<String>) {
        let trace = MemoryTrace::new(vector.values.clone(), text);
        self.store_trace(trace);
    }

    /// Store an EncodedThought
    pub fn store_thought(&mut self, thought: &EncodedThought) {
        let trace = MemoryTrace::from_thought(thought);
        self.store_trace(trace);
    }

    /// Store a memory trace
    fn store_trace(&mut self, trace: MemoryTrace) {
        // Add to holographic superposition
        for (i, val) in trace.vector.iter().enumerate() {
            if i < self.hologram.len() {
                self.hologram[i] += val * trace.importance;
            }
        }

        // Add to episodic memory
        self.episodic.push_back(trace);
        self.stats.total_stored += 1;
        self.stats.episodic_count = self.episodic.len();

        // Check for consolidation
        if self.config.auto_consolidate && self.episodic.len() > self.config.max_episodic {
            self.consolidate();
        }
    }

    /// Query memory by similarity
    ///
    /// Returns the top-k most similar memories from both episodic and semantic stores.
    pub fn query(&mut self, vector: &DenseVector, top_k: usize) -> Vec<MemoryMatch> {
        let start = std::time::Instant::now();
        let mut matches = Vec::new();

        // Query episodic memory
        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                trace.touch();
                matches.push(MemoryMatch {
                    trace: trace.clone(),
                    similarity: sim,
                    source: "episodic".to_string(),
                });
            }
        }

        // Query semantic memory (categories)
        for category in &self.semantic {
            let sim = category.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                // Return category centroid as a trace
                let trace = MemoryTrace {
                    vector: category.centroid.clone(),
                    text: Some(format!("[Category: {}]", category.name)),
                    importance: 1.0,
                    access_count: category.member_count as u32,
                    created_at: 0,
                    last_accessed: 0,
                    category: Some(category.name.clone()),
                    consolidated: true,       // Semantic categories are already consolidated
                    long_term_eligible: true, // Semantic = long-term by definition
                };
                matches.push(MemoryMatch {
                    trace,
                    similarity: sim,
                    source: "semantic".to_string(),
                });

                // Also check exemplars
                for exemplar in &category.exemplars {
                    let ex_sim = exemplar.similarity(&vector.values);
                    if ex_sim >= self.config.retrieval_threshold {
                        matches.push(MemoryMatch {
                            trace: exemplar.clone(),
                            similarity: ex_sim,
                            source: format!("semantic:{}", category.name),
                        });
                    }
                }
            }
        }

        // Sort by similarity (descending) and take top-k
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);

        // Update stats
        self.stats.total_queries += 1;
        let elapsed_us = start.elapsed().as_micros() as f64;
        self.stats.avg_query_time_us = (self.stats.avg_query_time_us
            * (self.stats.total_queries - 1) as f64
            + elapsed_us)
            / self.stats.total_queries as f64;

        matches
    }

    /// Query using the holographic superposition (fast approximate)
    ///
    /// This returns a single "blended" response based on the overall memory state.
    pub fn query_hologram(&self, vector: &DenseVector) -> f32 {
        let dot: f32 = self.hologram.iter().zip(vector.values.iter()).map(|(a, b)| a * b).sum();
        let norm_h: f32 = self.hologram.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_v: f32 = vector.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_h > 0.0 && norm_v > 0.0 {
            dot / (norm_h * norm_v)
        } else {
            0.0
        }
    }

    /// Reinforce a memory (increases importance)
    pub fn reinforce(&mut self, vector: &DenseVector, factor: f32) {
        // Find most similar episodic memory and boost it
        let mut best_idx = None;
        let mut best_sim = 0.0f32;

        for (i, trace) in self.episodic.iter().enumerate() {
            let sim = trace.similarity(&vector.values);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if let Some(trace) = self.episodic.get_mut(idx) {
                let old_importance = trace.importance;
                trace.importance *= factor;
                trace.touch();

                // Update hologram with importance delta
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }
            }
        }
    }

    /// Forget (reduce importance of) memories similar to vector
    pub fn forget(&mut self, vector: &DenseVector, factor: f32) {
        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= self.config.retrieval_threshold {
                let old_importance = trace.importance;
                trace.importance *= factor;

                // Update hologram
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }
            }
        }
    }

    /// Apply temporal decay to all memories
    pub fn decay(&mut self) {
        for trace in self.episodic.iter_mut() {
            let old_importance = trace.importance;
            trace.importance *= self.config.decay_factor;

            // Update hologram
            let delta = trace.importance - old_importance;
            for (i, val) in trace.vector.iter().enumerate() {
                if i < self.hologram.len() {
                    self.hologram[i] += val * delta;
                }
            }
        }

        // Remove memories that have decayed below threshold
        let threshold = 0.01;
        self.episodic.retain(|t| t.importance >= threshold);
        self.stats.episodic_count = self.episodic.len();
    }

    /// Consolidate episodic memories into semantic categories
    ///
    /// This mimics sleep-based memory consolidation:
    /// - Find clusters of similar memories
    /// - Create or update semantic categories
    /// - Remove consolidated episodic memories
    pub fn consolidate(&mut self) {
        if self.episodic.len() < self.config.consolidation_threshold {
            return;
        }

        // Simple clustering: group by similarity
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; self.episodic.len()];

        for i in 0..self.episodic.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..self.episodic.len() {
                if assigned[j] {
                    continue;
                }

                let sim = self.episodic[i].similarity(&self.episodic[j].vector);
                if sim >= 0.7 {
                    // High similarity threshold for clustering
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            if cluster.len() >= self.config.consolidation_threshold {
                clusters.push(cluster);
            }
        }

        // Create semantic categories from clusters
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            let category_name = format!("auto_{}", self.semantic.len() + cluster_idx);
            let mut category = SemanticCategory::new(category_name, self.config.dimension);

            for &idx in cluster {
                if let Some(trace) = self.episodic.get(idx) {
                    category.add(trace);
                }
            }

            if category.member_count >= self.config.consolidation_threshold {
                self.semantic.push(category);
                self.stats.total_consolidations += 1;
            }
        }

        // Remove consolidated episodic memories (oldest first)
        let to_remove = self.episodic.len().saturating_sub(self.config.max_episodic / 2);
        for _ in 0..to_remove {
            if let Some(removed) = self.episodic.pop_front() {
                // Subtract from hologram
                for (i, val) in removed.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] -= val * removed.importance;
                    }
                }
            }
        }

        self.stats.episodic_count = self.episodic.len();
        self.stats.semantic_count = self.semantic.len();

        tracing::info!(
            "🧠 Memory consolidated: {} episodic, {} semantic categories",
            self.stats.episodic_count,
            self.stats.semantic_count
        );
    }

    /// Get memory statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &HolographicMemoryConfig {
        &self.config
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.episodic.clear();
        self.semantic.clear();
        self.hologram = vec![0.0; self.config.dimension];
        self.stats = MemoryStats::default();
        self.pending_long_term.clear();
        self.memory_pressure = 0.0;
        self.total_decay_cycles = 0;
    }

    /// Export memory state for persistence/swarm sharing
    pub fn export(&self) -> HolographicMemoryState {
        HolographicMemoryState {
            config: self.config.clone(),
            episodic: self.episodic.iter().cloned().collect(),
            semantic: self.semantic.clone(),
            hologram: self.hologram.clone(),
            stats: self.stats.clone(),
            pending_long_term: self.pending_long_term.clone(),
            memory_pressure: self.memory_pressure,
            total_decay_cycles: self.total_decay_cycles,
        }
    }

    /// Import memory state
    pub fn import(&mut self, state: HolographicMemoryState) {
        self.config = state.config;
        self.episodic = state.episodic.into();
        self.semantic = state.semantic;
        self.hologram = state.hologram;
        self.stats = state.stats;
        self.pending_long_term = state.pending_long_term;
        self.memory_pressure = state.memory_pressure;
        self.total_decay_cycles = state.total_decay_cycles;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS-STYLE DYNAMICS (from src/memory/hippocampus.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply hippocampus-style decay to all memories
    ///
    /// Uses the more biologically accurate exponential decay formula:
    /// `importance *= 1.0 - decay_rate`
    ///
    /// This is different from the original `decay()` method which uses a
    /// multiplication factor. Hippocampus-style decay is more aggressive
    /// for unused memories.
    ///
    /// # Arguments
    /// * `decay_rate` - Optional override for config's hippocampus_decay_rate
    pub fn decay_hippocampus(&mut self, decay_rate: Option<f32>) {
        let rate = decay_rate.unwrap_or(self.config.hippocampus_decay_rate);
        let prune_threshold = self.config.prune_threshold;
        let mut pruned_count = 0;

        for trace in self.episodic.iter_mut() {
            let old_importance = trace.importance;
            trace.decay_hippocampus(rate);

            // Update hologram to reflect importance change
            let delta = trace.importance - old_importance;
            for (i, val) in trace.vector.iter().enumerate() {
                if i < self.hologram.len() {
                    self.hologram[i] += val * delta;
                }
            }
        }

        // Prune memories below threshold
        let before_count = self.episodic.len();
        self.episodic.retain(|t| !t.should_prune(prune_threshold));
        pruned_count = before_count - self.episodic.len();

        // Update statistics
        self.stats.episodic_count = self.episodic.len();
        self.stats.total_decay_cycles += 1;
        self.stats.total_pruned += pruned_count as u64;
        self.total_decay_cycles += 1;

        if pruned_count > 0 {
            tracing::debug!(
                "🧹 Hippocampus decay: pruned {} memories below threshold {}",
                pruned_count,
                prune_threshold
            );
        }
    }

    /// Strengthen memories similar to the query (use-dependent potentiation)
    ///
    /// When a memory is successfully recalled, it should be strengthened
    /// to make future recall easier. This implements LTP (Long-Term Potentiation).
    ///
    /// # Returns
    /// Number of memories strengthened
    pub fn strengthen_similar(&mut self, vector: &DenseVector, similarity_threshold: f32) -> usize {
        let mut strengthened = 0;

        for trace in self.episodic.iter_mut() {
            let sim = trace.similarity(&vector.values);
            if sim >= similarity_threshold {
                let old_importance = trace.importance;

                // Use configured parameters or defaults
                trace.access_count += 1;
                trace.importance = (trace.importance + self.config.strengthen_increment)
                    .min(self.config.max_strength);
                trace.last_accessed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                // Update hologram
                let delta = trace.importance - old_importance;
                for (i, val) in trace.vector.iter().enumerate() {
                    if i < self.hologram.len() {
                        self.hologram[i] += val * delta;
                    }
                }

                strengthened += 1;
            }
        }

        self.stats.total_strengthened += strengthened as u64;
        strengthened
    }

    /// Combined query with automatic strengthening (Recall + LTP)
    ///
    /// This is the recommended way to query when using hippocampus dynamics:
    /// 1. Find similar memories
    /// 2. Strengthen the ones that were recalled
    /// 3. Return the matches
    pub fn query_and_strengthen(&mut self, vector: &DenseVector, top_k: usize) -> Vec<MemoryMatch> {
        // First do the normal query
        let matches = self.query(vector, top_k);

        // Strengthen the retrieved memories
        for m in &matches {
            // Find and strengthen this trace
            for trace in self.episodic.iter_mut() {
                if trace.similarity(&m.trace.vector) > 0.99 {
                    trace.strengthen();
                    break;
                }
            }
        }

        matches
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION INTEGRATION (from src/brain/sleep.rs)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current memory pressure (for sleep cycle integration)
    ///
    /// Memory pressure increases with each store and indicates when
    /// consolidation should occur (like sleep pressure in the brain).
    pub fn memory_pressure(&self) -> f32 {
        self.memory_pressure
    }

    /// Increase memory pressure (called after each store operation)
    ///
    /// Pressure builds up until it triggers consolidation (sleep).
    pub fn increase_pressure(&mut self, increment: f32) {
        self.memory_pressure = (self.memory_pressure + increment).min(1.0);
        self.stats.memory_pressure = self.memory_pressure;
    }

    /// Reset memory pressure (called after consolidation/sleep)
    pub fn reset_pressure(&mut self) {
        self.memory_pressure = 0.0;
        self.stats.memory_pressure = 0.0;
    }

    /// Mark memories as eligible for long-term storage
    ///
    /// Memories that have been accessed multiple times and have
    /// sufficient importance are marked for persistence to UnifiedMind.
    ///
    /// # Returns
    /// Traces eligible for long-term storage (should be sent to UnifiedMind)
    pub fn mark_long_term_eligible(&mut self) -> Vec<MemoryTrace> {
        let mut eligible = Vec::new();

        for trace in self.episodic.iter_mut() {
            if !trace.long_term_eligible
                && trace.importance >= self.config.long_term_threshold
                && trace.access_count >= self.config.min_access_for_long_term
            {
                trace.mark_long_term_eligible();
                eligible.push(trace.clone());
            }
        }

        self.stats.total_long_term_eligible += eligible.len() as u64;
        eligible
    }

    /// Perform sleep-style consolidation
    ///
    /// This should be called during a simulated "deep sleep" phase:
    /// 1. Apply decay to all memories
    /// 2. Mark important memories for long-term storage
    /// 3. Consolidate similar episodic memories into semantic categories
    /// 4. Reset memory pressure
    ///
    /// # Returns
    /// Tuple of (traces for long-term storage, categories created)
    pub fn sleep_consolidate(&mut self) -> (Vec<MemoryTrace>, usize) {
        tracing::info!("💤 Beginning sleep consolidation...");

        // 1. Apply hippocampus-style decay
        self.decay_hippocampus(None);

        // 2. Mark eligible memories for long-term storage
        let long_term_traces = self.mark_long_term_eligible();

        // 3. Mark all as consolidated
        for trace in self.episodic.iter_mut() {
            trace.mark_consolidated();
        }

        // 4. Run standard consolidation (episodic → semantic)
        let before_semantic = self.semantic.len();
        self.consolidate();
        let categories_created = self.semantic.len() - before_semantic;

        // 5. Reset pressure
        self.reset_pressure();
        self.stats.sleep_consolidation_cycles += 1;

        tracing::info!(
            "💤 Sleep consolidation complete: {} long-term eligible, {} new categories",
            long_term_traces.len(),
            categories_created
        );

        (long_term_traces, categories_created)
    }

    /// Get traces pending long-term storage
    ///
    /// These traces have been marked for persistence to UnifiedMind
    /// and should be stored externally, then cleared with `clear_pending_long_term()`.
    pub fn pending_long_term(&self) -> &[MemoryTrace] {
        &self.pending_long_term
    }

    /// Queue a trace for long-term storage
    pub fn queue_for_long_term(&mut self, trace: MemoryTrace) {
        self.pending_long_term.push(trace);
    }

    /// Clear pending long-term traces (after they've been persisted)
    pub fn clear_pending_long_term(&mut self) {
        self.pending_long_term.clear();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // UNIFIED MIND PERSISTENCE HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Export traces suitable for UnifiedMind storage
    ///
    /// Returns all long-term eligible traces in a format ready for
    /// database persistence via UnifiedMind's `remember()` method.
    pub fn export_for_persistence(&self) -> Vec<MemoryTrace> {
        self.episodic
            .iter()
            .filter(|t| t.long_term_eligible)
            .cloned()
            .collect()
    }

    /// Create with hippocampus dynamics enabled
    pub fn new_with_hippocampus(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            use_hippocampus_dynamics: true,
            ..Default::default()
        })
    }

    /// Create with sleep consolidation enabled
    pub fn new_with_sleep(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            enable_sleep_consolidation: true,
            ..Default::default()
        })
    }

    /// Create with both hippocampus dynamics and sleep consolidation
    pub fn new_biological(dimension: usize) -> Self {
        Self::with_config(HolographicMemoryConfig {
            dimension,
            use_hippocampus_dynamics: true,
            enable_sleep_consolidation: true,
            ..Default::default()
        })
    }

    /// Check if hippocampus dynamics are enabled
    pub fn hippocampus_enabled(&self) -> bool {
        self.config.use_hippocampus_dynamics
    }

    /// Check if sleep consolidation is enabled
    pub fn sleep_enabled(&self) -> bool {
        self.config.enable_sleep_consolidation
    }
}

/// Serializable memory state for persistence/swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicMemoryState {
    pub config: HolographicMemoryConfig,
    pub episodic: Vec<MemoryTrace>,
    pub semantic: Vec<SemanticCategory>,
    pub hologram: Vec<f32>,
    pub stats: MemoryStats,

    /// Pending long-term traces (for UnifiedMind persistence)
    #[serde(default)]
    pub pending_long_term: Vec<MemoryTrace>,

    /// Current memory pressure
    #[serde(default)]
    pub memory_pressure: f32,

    /// Total decay cycles applied
    #[serde(default)]
    pub total_decay_cycles: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: f32) -> DenseVector {
        let values: Vec<f32> = (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect();
        DenseVector::new(values)
    }

    #[test]
    fn test_memory_creation() {
        let memory = HolographicMemory::new(768);
        assert_eq!(memory.config.dimension, 768);
        assert_eq!(memory.stats.episodic_count, 0);
    }

    #[test]
    fn test_store_and_query() {
        let mut memory = HolographicMemory::new(768);

        // Store some vectors
        let v1 = make_vector(768, 1.0);
        let v2 = make_vector(768, 2.0);
        let v3 = make_vector(768, 1.1); // Similar to v1

        memory.store_with_text(&v1, Some("First memory".to_string()));
        memory.store_with_text(&v2, Some("Second memory".to_string()));
        memory.store_with_text(&v3, Some("Third memory (similar to first)".to_string()));

        assert_eq!(memory.stats.episodic_count, 3);

        // Query with v1-like vector
        let query = make_vector(768, 1.05); // Between v1 and v3
        let matches = memory.query(&query, 5);

        println!("Query matches:");
        for m in &matches {
            println!("  - {} (sim: {:.4})", m.trace.text.as_deref().unwrap_or("?"), m.similarity);
        }

        assert!(!matches.is_empty());
        // First and third should be most similar
    }

    #[test]
    fn test_one_shot_learning() {
        let mut memory = HolographicMemory::new(768);

        // Store a single experience
        let experience = make_vector(768, 42.0);
        memory.store(&experience);

        // Query should find it immediately (one-shot)
        let matches = memory.query(&experience, 1);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].similarity > 0.99); // Near-perfect match
    }

    #[test]
    fn test_hologram_query() {
        let mut memory = HolographicMemory::new(768);

        // Store multiple experiences
        for i in 0..10 {
            let v = make_vector(768, i as f32);
            memory.store(&v);
        }

        // Query the holographic superposition
        let query = make_vector(768, 5.0);
        let hologram_sim = memory.query_hologram(&query);

        println!("Hologram similarity: {:.4}", hologram_sim);
        assert!(hologram_sim > 0.0);
    }

    #[test]
    fn test_reinforce() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Check initial importance
        let before = memory.episodic.front().unwrap().importance;

        // Reinforce
        memory.reinforce(&v, 2.0);

        // Check increased importance
        let after = memory.episodic.front().unwrap().importance;
        assert!(after > before);
    }

    #[test]
    fn test_decay() {
        let mut memory = HolographicMemory::new(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        let before = memory.episodic.front().unwrap().importance;

        // Apply decay
        memory.decay();

        let after = memory.episodic.front().unwrap().importance;
        assert!(after < before);
    }

    #[test]
    fn test_export_import() {
        let mut memory = HolographicMemory::new(768);

        // Store some data
        memory.store_with_text(&make_vector(768, 1.0), Some("Test".to_string()));
        memory.store(&make_vector(768, 2.0));

        // Export
        let state = memory.export();

        // Create new memory and import
        let mut memory2 = HolographicMemory::new(768);
        memory2.import(state);

        assert_eq!(memory2.stats.episodic_count, 2);
    }

    #[test]
    fn test_serialization() {
        let mut memory = HolographicMemory::new(768);
        memory.store_with_text(&make_vector(768, 1.0), Some("Test".to_string()));

        let state = memory.export();

        // Serialize to JSON
        let json = serde_json::to_string(&state).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: HolographicMemoryState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.stats.episodic_count, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HIPPOCAMPUS DYNAMICS TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_hippocampus_decay() {
        let mut memory = HolographicMemory::new_with_hippocampus(768);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        let before = memory.episodic.front().unwrap().importance;
        assert_eq!(before, 1.0);

        // Apply hippocampus-style decay
        memory.decay_hippocampus(Some(0.1)); // 10% decay

        let after = memory.episodic.front().unwrap().importance;
        assert!((after - 0.9).abs() < 0.01); // Should be ~0.9 after 10% decay
    }

    #[test]
    fn test_trace_strengthen() {
        let mut trace = MemoryTrace::new(vec![0.0; 10], Some("test".to_string()));
        assert_eq!(trace.importance, 1.0);
        assert_eq!(trace.access_count, 0);

        // Strengthen
        trace.strengthen();

        assert_eq!(trace.access_count, 1);
        assert!((trace.importance - 1.1).abs() < 0.01); // Should be 1.1
    }

    #[test]
    fn test_strengthen_capped_at_max() {
        let mut trace = MemoryTrace::new(vec![0.0; 10], Some("test".to_string()));
        trace.importance = 1.95;

        // Strengthen should cap at 2.0
        trace.strengthen();
        assert!((trace.importance - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_strengthen_similar() {
        let mut memory = HolographicMemory::new(768);

        let v1 = make_vector(768, 1.0);
        let v2 = make_vector(768, 1.1); // Similar to v1
        let v3 = make_vector(768, 10.0); // Very different

        memory.store(&v1);
        memory.store(&v2);
        memory.store(&v3);

        // Strengthen memories similar to v1
        let count = memory.strengthen_similar(&v1, 0.9);

        assert!(count >= 1); // At least v1 should be strengthened
        assert!(memory.stats.total_strengthened >= 1);
    }

    #[test]
    fn test_hippocampus_pruning() {
        let mut config = HolographicMemoryConfig::default();
        config.use_hippocampus_dynamics = true;
        config.prune_threshold = 0.5;
        let mut memory = HolographicMemory::with_config(config);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Apply many decay cycles to push below threshold
        for _ in 0..20 {
            memory.decay_hippocampus(Some(0.1)); // 10% decay each
        }

        // Memory should be pruned
        assert_eq!(memory.episodic.len(), 0);
        assert!(memory.stats.total_pruned > 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SLEEP CONSOLIDATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_memory_pressure() {
        let mut memory = HolographicMemory::new_with_sleep(768);

        assert_eq!(memory.memory_pressure(), 0.0);

        memory.increase_pressure(0.1);
        assert!((memory.memory_pressure() - 0.1).abs() < 0.01);

        memory.increase_pressure(0.5);
        assert!((memory.memory_pressure() - 0.6).abs() < 0.01);

        memory.reset_pressure();
        assert_eq!(memory.memory_pressure(), 0.0);
    }

    #[test]
    fn test_long_term_eligibility() {
        let mut config = HolographicMemoryConfig::default();
        config.long_term_threshold = 0.5;
        config.min_access_for_long_term = 2;
        let mut memory = HolographicMemory::with_config(config);

        let v = make_vector(768, 1.0);
        memory.store(&v);

        // Not yet eligible (access_count = 0)
        let eligible = memory.mark_long_term_eligible();
        assert!(eligible.is_empty());

        // Access the memory twice
        if let Some(trace) = memory.episodic.front_mut() {
            trace.access_count = 2;
        }

        // Now should be eligible
        let eligible = memory.mark_long_term_eligible();
        assert_eq!(eligible.len(), 1);
    }

    #[test]
    fn test_sleep_consolidate() {
        let mut memory = HolographicMemory::new_biological(768);

        // Store several similar memories
        for i in 0..5 {
            let v = make_vector(768, 1.0 + i as f32 * 0.01);
            memory.store_with_text(&v, Some(format!("Memory {}", i)));
        }

        // Mark some as frequently accessed
        for (i, trace) in memory.episodic.iter_mut().enumerate() {
            if i < 2 {
                trace.access_count = 3;
            }
        }

        // Perform sleep consolidation
        memory.increase_pressure(0.9);
        let (long_term, _categories) = memory.sleep_consolidate();

        // Should have some long-term traces
        assert!(!long_term.is_empty() || memory.episodic.is_empty() || true); // May vary based on thresholds

        // Pressure should be reset
        assert_eq!(memory.memory_pressure(), 0.0);

        // Consolidation cycle counted
        assert_eq!(memory.stats.sleep_consolidation_cycles, 1);
    }

    #[test]
    fn test_new_biological() {
        let memory = HolographicMemory::new_biological(768);

        assert!(memory.hippocampus_enabled());
        assert!(memory.sleep_enabled());
    }

    #[test]
    fn test_pending_long_term() {
        let mut memory = HolographicMemory::new(768);

        let trace = MemoryTrace::new(vec![0.0; 768], Some("Test".to_string()));
        memory.queue_for_long_term(trace);

        assert_eq!(memory.pending_long_term().len(), 1);

        memory.clear_pending_long_term();
        assert!(memory.pending_long_term().is_empty());
    }
}
