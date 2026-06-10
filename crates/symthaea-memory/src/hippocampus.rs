// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hippocampus: Episodic Memory System
//!
//! The hippocampus manages episodic memory formation, consolidation, and recall.
//! It implements a biologically-inspired memory system with:
//!
//! - **Encoding**: Converting experiences into memory traces
//! - **Consolidation**: Strengthening important memories during "sleep"
//! - **Recall**: Retrieving memories based on cues
//! - **Emotional Tagging**: Attaching valence to memories
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                        HIPPOCAMPUS                                  │
//! │                                                                     │
//! │   Input (Experience)                                                │
//! │         │                                                           │
//! │         ▼                                                           │
//! │   ┌──────────────┐                                                  │
//! │   │   Encoder    │ → Pattern separation into memory trace           │
//! │   └──────┬───────┘                                                  │
//! │          │                                                          │
//! │          ▼                                                          │
//! │   ┌──────────────┐      ┌──────────────────────────────────┐       │
//! │   │  Emotional   │ ───► │  Memory Store (DG → CA3 → CA1)   │       │
//! │   │   Tagging    │      │  - Sparse encoding               │       │
//! │   └──────────────┘      │  - Pattern completion            │       │
//! │                         │  - Temporal sequences            │       │
//! │                         └──────────────┬───────────────────┘       │
//! │                                        │                            │
//! │                                        ▼                            │
//! │   ┌────────────────────────────────────────────────────────────┐   │
//! │   │                    Consolidation                           │   │
//! │   │    (During rest: strengthen important, prune weak)         │   │
//! │   └────────────────────────────────────────────────────────────┘   │
//! └────────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Emotional valence attached to memories
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum EmotionalValence {
    /// Highly positive experience
    VeryPositive,
    /// Positive experience
    Positive,
    /// Neutral experience
    #[default]
    Neutral,
    /// Negative experience
    Negative,
    /// Highly negative experience
    VeryNegative,
}

impl EmotionalValence {
    /// Convert to numeric value (-1.0 to 1.0)
    pub fn to_f64(&self) -> f64 {
        match self {
            EmotionalValence::VeryPositive => 1.0,
            EmotionalValence::Positive => 0.5,
            EmotionalValence::Neutral => 0.0,
            EmotionalValence::Negative => -0.5,
            EmotionalValence::VeryNegative => -1.0,
        }
    }

    /// Create from numeric value
    pub fn from_f64(value: f64) -> Self {
        if value > 0.75 {
            EmotionalValence::VeryPositive
        } else if value > 0.25 {
            EmotionalValence::Positive
        } else if value > -0.25 {
            EmotionalValence::Neutral
        } else if value > -0.75 {
            EmotionalValence::Negative
        } else {
            EmotionalValence::VeryNegative
        }
    }
}

/// A memory trace in the hippocampus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// Unique identifier
    pub id: u64,

    /// Timestamp of encoding (Unix milliseconds)
    pub timestamp: u64,

    /// Semantic embedding of the memory
    pub embedding: Vec<f32>,

    /// Textual content
    pub content: String,

    /// Context tags
    pub tags: Vec<String>,

    /// Emotional valence
    pub valence: EmotionalValence,

    /// Consolidation strength (0.0 to 1.0)
    pub strength: f64,

    /// Access count (for LRU-like decay)
    pub access_count: u32,

    /// Last access timestamp
    pub last_accessed: u64,

    /// Associated consciousness level (Φ) at encoding
    pub phi_at_encoding: Option<f64>,
}

impl MemoryTrace {
    /// Create a new memory trace
    pub fn new(
        id: u64,
        content: impl Into<String>,
        embedding: Vec<f32>,
        valence: EmotionalValence,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            timestamp: now,
            embedding,
            content: content.into(),
            tags: Vec::new(),
            valence,
            strength: 1.0,
            access_count: 0,
            last_accessed: now,
            phi_at_encoding: None,
        }
    }

    /// Add tags to the memory
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set consciousness level at encoding
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.phi_at_encoding = Some(phi);
        self
    }

    /// Compute similarity with another memory trace
    pub fn similarity(&self, other: &MemoryTrace) -> f32 {
        cosine_similarity(&self.embedding, &other.embedding)
    }

    /// Compute similarity with query embedding
    pub fn query_similarity(&self, query: &[f32]) -> f32 {
        cosine_similarity(&self.embedding, query)
    }
}

/// Query for memory recall
#[derive(Debug, Clone)]
pub struct RecallQuery {
    /// Semantic embedding to match
    pub embedding: Option<Vec<f32>>,

    /// Text to search for
    pub text: Option<String>,

    /// Filter by tags
    pub tags: Vec<String>,

    /// Filter by minimum valence
    pub min_valence: Option<EmotionalValence>,

    /// Filter by time range (start, end)
    pub time_range: Option<(u64, u64)>,

    /// Maximum results
    pub top_k: usize,

    /// Minimum similarity threshold
    pub min_similarity: f32,
}

impl Default for RecallQuery {
    fn default() -> Self {
        Self {
            embedding: None,
            text: None,
            tags: Vec::new(),
            min_valence: None,
            time_range: None,
            top_k: 10,
            min_similarity: 0.0,
        }
    }
}

impl RecallQuery {
    /// Create a query from embedding
    pub fn from_embedding(embedding: Vec<f32>) -> Self {
        Self {
            embedding: Some(embedding),
            ..Default::default()
        }
    }

    /// Create a query from text
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Default::default()
        }
    }

    /// Set maximum results
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set minimum similarity
    pub fn with_min_similarity(mut self, sim: f32) -> Self {
        self.min_similarity = sim;
        self
    }

    /// Filter by tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Result of memory recall
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// Memory traces found
    pub memories: Vec<MemoryTrace>,

    /// Similarity scores
    pub scores: Vec<f32>,

    /// Query time in milliseconds
    pub query_time_ms: f32,

    /// Total memories searched
    pub total_searched: usize,
}

/// Hippocampus actor - manages episodic memory.
///
/// **Deprecated**: Use `MemoryCoordinator` + `EpisodicMemory` instead. This module
/// duplicates their functionality, requires an async runtime (the cognitive loop is
/// sync/rayon), and has zero call sites. Kept for type availability only.
#[deprecated(
    since = "1.10.0",
    note = "Use MemoryCoordinator + EpisodicMemory instead"
)]
#[allow(dead_code)] // RESERVED(future): hippocampal memory consolidation actor
pub struct HippocampusActor {
    /// Memory store
    memories: Arc<RwLock<Vec<MemoryTrace>>>,

    /// Next memory ID
    next_id: Arc<RwLock<u64>>,

    /// Maximum memories to store
    max_memories: usize,

    /// Consolidation threshold
    consolidation_threshold: f64,

    /// Decay rate for unused memories
    decay_rate: f64,

    /// Statistics
    stats: HippocampusStats,
}

/// Statistics for the hippocampus
#[derive(Debug, Clone, Default)]
pub struct HippocampusStats {
    /// Total memories encoded
    pub total_encoded: u64,

    /// Total memories recalled
    pub total_recalled: u64,

    /// Memories consolidated
    pub consolidated: u64,

    /// Memories pruned
    pub pruned: u64,

    /// Average recall time (ms)
    pub avg_recall_time_ms: f32,
}

#[allow(deprecated)]
impl HippocampusActor {
    /// Create a new hippocampus actor
    pub fn new(max_memories: usize) -> Self {
        Self {
            memories: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(RwLock::new(0)),
            max_memories,
            consolidation_threshold: 0.3,
            decay_rate: 0.01,
            stats: HippocampusStats::default(),
        }
    }

    /// Create with default settings
    pub fn default_settings() -> Self {
        Self::new(10000)
    }

    /// Encode a new memory
    pub async fn encode(
        &mut self,
        content: impl Into<String>,
        embedding: Vec<f32>,
        valence: EmotionalValence,
    ) -> Result<u64> {
        let content = content.into();

        let id = {
            let mut next_id = self.next_id.write().await;
            *next_id += 1;
            *next_id
        };

        let memory = MemoryTrace::new(id, content, embedding, valence);

        let pruned_count = {
            let mut memories = self.memories.write().await;
            memories.push(memory);

            // Prune if over capacity
            if memories.len() > self.max_memories {
                Self::prune_weakest_memories(&mut memories)
            } else {
                0
            }
        };
        self.stats.pruned += pruned_count as u64;

        self.stats.total_encoded += 1;

        Ok(id)
    }

    /// Encode memory synchronously (for non-async contexts)
    pub fn encode_sync(
        &mut self,
        content: impl Into<String>,
        embedding: Vec<f32>,
        valence: EmotionalValence,
    ) -> Result<u64> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| anyhow::anyhow!("No runtime: {e}"))?;

        rt.block_on(self.encode(content, embedding, valence))
    }

    /// Recall memories based on query
    pub async fn recall(&mut self, query: RecallQuery) -> Result<RecallResult> {
        let start = std::time::Instant::now();

        let memories = self.memories.read().await;
        let total_searched = memories.len();

        let mut results: Vec<(MemoryTrace, f32)> = Vec::new();

        for memory in memories.iter() {
            // Compute similarity
            let similarity = if let Some(ref emb) = query.embedding {
                memory.query_similarity(emb)
            } else if let Some(ref text) = query.text {
                // Simple text matching fallback
                if memory.content.to_lowercase().contains(&text.to_lowercase()) {
                    0.8
                } else {
                    0.0
                }
            } else {
                1.0 // No query = match all
            };

            // Apply filters
            if similarity < query.min_similarity {
                continue;
            }

            if !query.tags.is_empty() {
                let has_tag = query.tags.iter().any(|t| memory.tags.contains(t));
                if !has_tag {
                    continue;
                }
            }

            if let Some(min_val) = query.min_valence {
                if memory.valence.to_f64() < min_val.to_f64() {
                    continue;
                }
            }

            if let Some((start_time, end_time)) = query.time_range {
                if memory.timestamp < start_time || memory.timestamp > end_time {
                    continue;
                }
            }

            results.push((memory.clone(), similarity));
        }

        // Handle k=0 edge case: return empty result
        let k = query.top_k;
        if k == 0 {
            results.clear();
        } else if results.len() > k {
            // Optimize top-k selection: use select_nth_unstable for O(n) average
            // instead of full O(n log n) sort when results exceed top_k
            // Partition so that the k largest elements are in results[0..k]
            // select_nth_unstable_by places the k-th element at index k-1 and
            // partitions such that elements before are >= and after are <=
            results.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
        }
        // Sort only the top-k results (small constant size)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        let n = (self.stats.total_recalled + 1) as f32;
        self.stats.avg_recall_time_ms = (self.stats.avg_recall_time_ms * (n - 1.0) + elapsed) / n;
        self.stats.total_recalled += 1;

        let (memories, scores): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        Ok(RecallResult {
            memories,
            scores,
            query_time_ms: elapsed,
            total_searched,
        })
    }

    /// Recall synchronously
    pub fn recall_sync(&mut self, query: RecallQuery) -> Result<RecallResult> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| anyhow::anyhow!("No runtime: {e}"))?;

        rt.block_on(self.recall(query))
    }

    /// Consolidate memories (strengthen important, weaken unimportant)
    pub async fn consolidate(&mut self) -> Result<usize> {
        let mut memories = self.memories.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut consolidated_count = 0;

        for memory in memories.iter_mut() {
            // Decay based on time since last access
            let age_hours =
                now.saturating_sub(memory.last_accessed) as f64 / (1000.0 * 60.0 * 60.0);
            let decay = (age_hours * self.decay_rate).min(0.5);
            memory.strength = (memory.strength - decay).max(0.0);

            // Strengthen memories with high emotional valence
            let emotion_boost = memory.valence.to_f64().abs() * 0.1;
            memory.strength = (memory.strength + emotion_boost).min(1.0);

            // Strengthen frequently accessed memories
            let access_boost = (memory.access_count as f64 / 100.0).min(0.2);
            memory.strength = (memory.strength + access_boost).min(1.0);

            consolidated_count += 1;
        }

        self.stats.consolidated += consolidated_count as u64;

        // Prune very weak memories
        let before_len = memories.len();
        memories.retain(|m| m.strength > 0.05);
        let pruned = before_len - memories.len();
        self.stats.pruned += pruned as u64;

        Ok(consolidated_count)
    }

    /// Prune the weakest memories (static to avoid borrow issues)
    fn prune_weakest_memories(memories: &mut Vec<MemoryTrace>) -> usize {
        // Sort by strength ascending
        memories.sort_by(|a, b| {
            a.strength
                .partial_cmp(&b.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove the weakest 10%
        let to_remove = memories.len() / 10;
        memories.drain(0..to_remove);

        to_remove
    }

    /// Get memory count
    pub async fn memory_count(&self) -> usize {
        self.memories.read().await.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &HippocampusStats {
        &self.stats
    }
}

#[allow(deprecated)]
impl Default for HippocampusActor {
    fn default() -> Self {
        Self::default_settings()
    }
}

use symthaea_core::math::cosine_similarity_f32 as cosine_similarity;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hippocampus_encode() {
        let mut hippo = HippocampusActor::new(100);
        let embedding = vec![0.1f32; 128];

        let id = hippo
            .encode("Test memory", embedding, EmotionalValence::Positive)
            .await
            .unwrap();
        assert!(id > 0);
        assert_eq!(hippo.memory_count().await, 1);
    }

    #[tokio::test]
    async fn test_hippocampus_recall() {
        let mut hippo = HippocampusActor::new(100);

        let emb1 = vec![0.1f32; 128];
        let emb2 = vec![0.2f32; 128];

        hippo
            .encode("First memory", emb1.clone(), EmotionalValence::Positive)
            .await
            .unwrap();
        hippo
            .encode("Second memory", emb2.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        let query = RecallQuery::from_embedding(emb1).with_top_k(5);
        let result = hippo.recall(query).await.unwrap();

        assert!(!result.memories.is_empty());
        assert_eq!(result.memories[0].content, "First memory");
    }

    #[tokio::test]
    async fn test_hippocampus_consolidation() {
        let mut hippo = HippocampusActor::new(100);
        let embedding = vec![0.1f32; 128];

        hippo
            .encode("Test memory", embedding, EmotionalValence::VeryPositive)
            .await
            .unwrap();

        let consolidated = hippo.consolidate().await.unwrap();
        assert_eq!(consolidated, 1);
    }

    #[test]
    fn test_emotional_valence() {
        assert_eq!(EmotionalValence::VeryPositive.to_f64(), 1.0);
        assert_eq!(EmotionalValence::Neutral.to_f64(), 0.0);
        assert_eq!(EmotionalValence::VeryNegative.to_f64(), -1.0);

        assert_eq!(
            EmotionalValence::from_f64(0.9),
            EmotionalValence::VeryPositive
        );
        assert_eq!(EmotionalValence::from_f64(0.0), EmotionalValence::Neutral);
        assert_eq!(
            EmotionalValence::from_f64(-0.9),
            EmotionalValence::VeryNegative
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    /// Test that top-k selection returns correct results with optimized algorithm
    #[tokio::test]
    async fn test_recall_top_k_correctness() {
        let mut hippo = HippocampusActor::new(10000);

        // Create embeddings with known similarity values
        let query_emb = vec![1.0f32; 128];

        // Add memories with decreasing similarity to query
        for i in 0..100 {
            // Each embedding differs more from query as i increases
            let mut emb = vec![1.0f32; 128];
            for j in 0..i.min(128) {
                emb[j] = 0.0; // Make embeddings progressively different
            }
            hippo
                .encode(format!("Memory {}", i), emb, EmotionalValence::Neutral)
                .await
                .unwrap();
        }

        // Request top 5
        let query = RecallQuery::from_embedding(query_emb.clone()).with_top_k(5);
        let result = hippo.recall(query).await.unwrap();

        // Should get exactly 5 results
        assert_eq!(result.memories.len(), 5);

        // Results should be sorted by similarity (descending)
        for i in 1..result.scores.len() {
            assert!(
                result.scores[i - 1] >= result.scores[i],
                "Results should be sorted by similarity descending"
            );
        }

        // First result should be the most similar (Memory 0, which is identical to query)
        assert_eq!(result.memories[0].content, "Memory 0");
    }

    /// Test edge cases for top-k selection
    #[tokio::test]
    async fn test_recall_top_k_edge_cases() {
        let mut hippo = HippocampusActor::new(100);
        let embedding = vec![0.5f32; 128];

        // Test with fewer results than k
        hippo
            .encode("Only memory", embedding.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        let query = RecallQuery::from_embedding(embedding.clone()).with_top_k(10);
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 1);

        // Test with k=0
        let query_zero = RecallQuery::from_embedding(embedding.clone()).with_top_k(0);
        let result_zero = hippo.recall(query_zero).await.unwrap();
        assert_eq!(result_zero.memories.len(), 0);
    }

    // ====================================================================
    // Additional tests for memory encoding, retrieval, and edge cases
    // ====================================================================

    #[test]
    fn test_emotional_valence_from_f64_boundary_values() {
        // Test boundary values for from_f64
        // Boundaries: >0.75 VP, >0.25 P, >-0.25 N, >-0.75 Neg, else VN
        assert_eq!(
            EmotionalValence::from_f64(0.76),
            EmotionalValence::VeryPositive
        );
        assert_eq!(EmotionalValence::from_f64(0.75), EmotionalValence::Positive);
        assert_eq!(EmotionalValence::from_f64(0.26), EmotionalValence::Positive);
        assert_eq!(EmotionalValence::from_f64(0.25), EmotionalValence::Neutral);
        assert_eq!(EmotionalValence::from_f64(0.0), EmotionalValence::Neutral);
        // -0.25 is NOT > -0.25, so it falls into Negative
        assert_eq!(
            EmotionalValence::from_f64(-0.25),
            EmotionalValence::Negative
        );
        assert_eq!(
            EmotionalValence::from_f64(-0.50),
            EmotionalValence::Negative
        );
        assert_eq!(
            EmotionalValence::from_f64(-0.75),
            EmotionalValence::VeryNegative
        );
        assert_eq!(
            EmotionalValence::from_f64(-0.76),
            EmotionalValence::VeryNegative
        );
    }

    #[test]
    fn test_emotional_valence_default_is_neutral() {
        assert_eq!(EmotionalValence::default(), EmotionalValence::Neutral);
    }

    #[test]
    fn test_memory_trace_with_tags_and_phi() {
        let trace = MemoryTrace::new(1, "tagged memory", vec![1.0; 8], EmotionalValence::Positive)
            .with_tags(vec!["foo".to_string(), "bar".to_string()])
            .with_phi(0.85);

        assert_eq!(trace.tags, vec!["foo", "bar"]);
        assert_eq!(trace.phi_at_encoding, Some(0.85));
        assert_eq!(trace.strength, 1.0);
        assert_eq!(trace.access_count, 0);
    }

    #[test]
    fn test_memory_trace_similarity_identical() {
        let emb = vec![1.0, 2.0, 3.0];
        let t1 = MemoryTrace::new(1, "a", emb.clone(), EmotionalValence::Neutral);
        let t2 = MemoryTrace::new(2, "b", emb, EmotionalValence::Neutral);
        let sim = t1.similarity(&t2);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical embeddings should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_memory_trace_query_similarity_orthogonal() {
        let trace = MemoryTrace::new(1, "a", vec![1.0, 0.0, 0.0], EmotionalValence::Neutral);
        let query = vec![0.0, 1.0, 0.0];
        let sim = trace.query_similarity(&query);
        assert!(
            sim.abs() < 0.001,
            "Orthogonal vectors should have ~0 similarity, got {sim}"
        );
    }

    #[tokio::test]
    async fn test_encode_assigns_sequential_ids() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        let id1 = hippo
            .encode("first", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();
        let id2 = hippo
            .encode("second", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();
        let id3 = hippo
            .encode("third", emb, EmotionalValence::Neutral)
            .await
            .unwrap();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[tokio::test]
    async fn test_capacity_eviction_prunes_weakest() {
        let mut hippo = HippocampusActor::new(10);
        let emb = vec![0.5f32; 64];

        for i in 0..11 {
            hippo
                .encode(
                    format!("memory-{i}"),
                    emb.clone(),
                    EmotionalValence::Neutral,
                )
                .await
                .unwrap();
        }

        // After inserting 11 with capacity 10, prune removes 10% = 1
        assert_eq!(hippo.memory_count().await, 10);
        assert_eq!(hippo.stats().pruned, 1);
    }

    #[tokio::test]
    async fn test_recall_with_tag_filter() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        let id1 = hippo
            .encode("tagged memory", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();
        let _id2 = hippo
            .encode("untagged memory", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        {
            let mut memories = hippo.memories.write().await;
            memories.iter_mut().find(|m| m.id == id1).unwrap().tags = vec!["important".to_string()];
        }

        let query = RecallQuery {
            embedding: Some(emb.clone()),
            tags: vec!["important".to_string()],
            top_k: 10,
            min_similarity: 0.0,
            ..Default::default()
        };
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].id, id1);

        let query_miss = RecallQuery {
            tags: vec!["nonexistent".to_string()],
            top_k: 10,
            ..Default::default()
        };
        let result_miss = hippo.recall(query_miss).await.unwrap();
        assert_eq!(result_miss.memories.len(), 0);
    }

    #[tokio::test]
    async fn test_recall_with_min_valence_filter() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        hippo
            .encode("happy memory", emb.clone(), EmotionalValence::VeryPositive)
            .await
            .unwrap();
        hippo
            .encode("sad memory", emb.clone(), EmotionalValence::VeryNegative)
            .await
            .unwrap();
        hippo
            .encode("neutral memory", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        let query = RecallQuery {
            embedding: Some(emb),
            min_valence: Some(EmotionalValence::Positive),
            top_k: 10,
            min_similarity: 0.0,
            ..Default::default()
        };
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].content, "happy memory");
    }

    #[tokio::test]
    async fn test_recall_with_time_range_filter() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        for i in 0..3 {
            hippo
                .encode(
                    format!("memory-{i}"),
                    emb.clone(),
                    EmotionalValence::Neutral,
                )
                .await
                .unwrap();
        }

        {
            let mut memories = hippo.memories.write().await;
            memories[0].timestamp = 1000;
            memories[1].timestamp = 2000;
            memories[2].timestamp = 3000;
        }

        let query = RecallQuery {
            embedding: Some(emb),
            time_range: Some((1500, 2500)),
            top_k: 10,
            min_similarity: 0.0,
            ..Default::default()
        };
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].timestamp, 2000);
    }

    #[tokio::test]
    async fn test_recall_text_matching() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        hippo
            .encode(
                "The cat sat on the mat",
                emb.clone(),
                EmotionalValence::Neutral,
            )
            .await
            .unwrap();
        hippo
            .encode(
                "The dog ran in the park",
                emb.clone(),
                EmotionalValence::Neutral,
            )
            .await
            .unwrap();

        let query = RecallQuery::from_text("CAT");
        let result = hippo.recall(query).await.unwrap();
        assert!(
            !result.memories.is_empty(),
            "Text search for 'CAT' should match 'cat'"
        );
        assert!(result.memories[0].content.to_lowercase().contains("cat"));
        assert!(result.scores[0] > 0.5, "Score should be positive for match");
    }

    #[tokio::test]
    async fn test_recall_no_query_matches_all() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        for i in 0..5 {
            hippo
                .encode(
                    format!("memory-{i}"),
                    emb.clone(),
                    EmotionalValence::Neutral,
                )
                .await
                .unwrap();
        }

        let query = RecallQuery {
            top_k: 10,
            ..Default::default()
        };
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 5);
        for score in &result.scores {
            assert!(
                (*score - 1.0).abs() < 0.001,
                "No-query recall should give similarity 1.0"
            );
        }
    }

    #[tokio::test]
    async fn test_recall_min_similarity_threshold() {
        let mut hippo = HippocampusActor::new(100);

        let query_emb = vec![1.0f32; 64];
        let similar_emb = vec![1.0f32; 64];
        let dissimilar_emb = vec![-1.0f32; 64];

        hippo
            .encode("similar", similar_emb, EmotionalValence::Neutral)
            .await
            .unwrap();
        hippo
            .encode("dissimilar", dissimilar_emb, EmotionalValence::Neutral)
            .await
            .unwrap();

        let query = RecallQuery::from_embedding(query_emb).with_min_similarity(0.9);
        let result = hippo.recall(query).await.unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].content, "similar");
    }

    #[tokio::test]
    async fn test_consolidation_strengthens_emotional_memories() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        hippo
            .encode(
                "emotional event",
                emb.clone(),
                EmotionalValence::VeryPositive,
            )
            .await
            .unwrap();
        hippo
            .encode("bland event", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        {
            let mut memories = hippo.memories.write().await;
            for m in memories.iter_mut() {
                m.strength = 0.5;
                m.last_accessed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
            }
        }

        hippo.consolidate().await.unwrap();

        let memories = hippo.memories.read().await;
        let emotional = memories
            .iter()
            .find(|m| m.content == "emotional event")
            .unwrap();
        let neutral = memories
            .iter()
            .find(|m| m.content == "bland event")
            .unwrap();

        assert!(
            emotional.strength > neutral.strength,
            "Emotional memories should be strengthened more: emotional={}, neutral={}",
            emotional.strength,
            neutral.strength
        );
    }

    #[tokio::test]
    async fn test_consolidation_prunes_very_weak_memories() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        hippo
            .encode("strong memory", emb.clone(), EmotionalValence::VeryPositive)
            .await
            .unwrap();
        hippo
            .encode("weak memory", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();

        {
            let mut memories = hippo.memories.write().await;
            let weak = memories
                .iter_mut()
                .find(|m| m.content == "weak memory")
                .unwrap();
            weak.strength = 0.01;
            weak.last_accessed = 0;
        }

        hippo.consolidate().await.unwrap();

        let memories = hippo.memories.read().await;
        assert!(
            !memories.iter().any(|m| m.content == "weak memory"),
            "Very weak memory should be pruned during consolidation"
        );
        assert!(
            memories.iter().any(|m| m.content == "strong memory"),
            "Strong memory should survive consolidation"
        );
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let mut hippo = HippocampusActor::new(100);
        let emb = vec![0.5f32; 64];

        assert_eq!(hippo.stats().total_encoded, 0);
        assert_eq!(hippo.stats().total_recalled, 0);

        hippo
            .encode("m1", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();
        hippo
            .encode("m2", emb.clone(), EmotionalValence::Neutral)
            .await
            .unwrap();
        assert_eq!(hippo.stats().total_encoded, 2);

        let query = RecallQuery::from_embedding(emb.clone()).with_top_k(5);
        hippo.recall(query).await.unwrap();
        assert_eq!(hippo.stats().total_recalled, 1);

        let query2 = RecallQuery::from_embedding(emb).with_top_k(5);
        hippo.recall(query2).await.unwrap();
        assert_eq!(hippo.stats().total_recalled, 2);
        assert!(hippo.stats().avg_recall_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_recall_empty_hippocampus() {
        let mut hippo = HippocampusActor::new(100);

        let query = RecallQuery::from_embedding(vec![1.0; 64]).with_top_k(5);
        let result = hippo.recall(query).await.unwrap();
        assert!(result.memories.is_empty());
        assert_eq!(result.total_searched, 0);
    }

    #[test]
    fn test_recall_query_builder_methods() {
        let query = RecallQuery::from_embedding(vec![1.0; 8])
            .with_top_k(3)
            .with_min_similarity(0.5)
            .with_tags(vec!["tag1".to_string()]);

        assert_eq!(query.top_k, 3);
        assert!((query.min_similarity - 0.5).abs() < f32::EPSILON);
        assert_eq!(query.tags, vec!["tag1"]);
        assert!(query.embedding.is_some());
    }

    #[test]
    fn test_hippocampus_default() {
        let hippo = HippocampusActor::default();
        assert_eq!(hippo.max_memories, 10000);
    }
}
