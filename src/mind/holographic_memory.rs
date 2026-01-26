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
}

impl Default for HolographicMemoryConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            max_episodic: 1000,
            decay_factor: 0.95,
            retrieval_threshold: 0.3,
            auto_consolidate: true,
            consolidation_threshold: 3,
        }
    }
}

/// A single memory trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// The dense vector representation
    pub vector: Vec<f32>,

    /// Original text (if available)
    pub text: Option<String>,

    /// Importance/salience score (affects retrieval and decay)
    pub importance: f32,

    /// Number of times this memory has been accessed
    pub access_count: u32,

    /// Timestamp of creation (Unix epoch ms)
    pub created_at: u64,

    /// Timestamp of last access
    pub last_accessed: u64,

    /// Optional category/tag
    pub category: Option<String>,
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
    }

    /// Export memory state for persistence/swarm sharing
    pub fn export(&self) -> HolographicMemoryState {
        HolographicMemoryState {
            config: self.config.clone(),
            episodic: self.episodic.iter().cloned().collect(),
            semantic: self.semantic.clone(),
            hologram: self.hologram.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Import memory state
    pub fn import(&mut self, state: HolographicMemoryState) {
        self.config = state.config;
        self.episodic = state.episodic.into();
        self.semantic = state.semantic;
        self.hologram = state.hologram;
        self.stats = state.stats;
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
}
