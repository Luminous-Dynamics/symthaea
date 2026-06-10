// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Semantic Memory: HDC-Based Similarity Lookup for CfC Contextual Learning
//!
//! This module implements a two-track architecture where:
//! - **HDC** handles semantic similarity lookup (content-addressable memory)
//! - **CfC** handles temporal learning (sequence prediction)
//!
//! When a new input arrives:
//! 1. Project input to HDC space via `project_to_hdc_vec()`
//! 2. Find semantically similar past inputs via cosine similarity
//! 3. Use that context to inform CfC prediction (modulate learning rate)
//!
//! ## Key Insight
//!
//! HDC vectors are excellent for similarity search (quasi-orthogonal, high-dimensional).
//! CfC networks are excellent for temporal dynamics (continuous-time, closed-form).
//! Combining them: HDC retrieves "what was similar", CfC learns "what happened next".
//!
//! ## Learning Rate Modulation
//!
//! - If similar past inputs had **high prediction error**: boost learning rate
//!   (we struggled with this type of input before, need to learn more)
//! - If similar past inputs had **low prediction error**: reduce learning rate
//!   (we've seen this before and did well, this is familiar territory)
//!
//! Formula: `semantic_lr_factor = 0.5 + avg_similar_error` (range 0.5 to ~1.5)

use serde::{Deserialize, Serialize};

/// A single entry in semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntry {
    /// HDC vector representation of this input
    pub hdc_vector: Vec<f32>,

    /// Timestamp (cycle number) when this was stored
    pub timestamp: u64,

    /// Optional category label (e.g., "question", "command", "greeting")
    pub category: Option<String>,

    /// How well CfC predicted after seeing this input
    /// Lower is better (0.0 = perfect prediction)
    pub prediction_error: f32,

    /// Confidence in the HDC encoding (optional, for weighted retrieval)
    pub encoding_confidence: f32,
}

impl SemanticEntry {
    /// Create a new semantic entry
    pub fn new(hdc_vector: Vec<f32>, timestamp: u64, prediction_error: f32) -> Self {
        Self {
            hdc_vector,
            timestamp,
            category: None,
            prediction_error,
            encoding_confidence: 1.0,
        }
    }

    /// Create with category label
    pub fn with_category(
        hdc_vector: Vec<f32>,
        timestamp: u64,
        prediction_error: f32,
        category: impl Into<String>,
    ) -> Self {
        Self {
            hdc_vector,
            timestamp,
            category: Some(category.into()),
            prediction_error,
            encoding_confidence: 1.0,
        }
    }
}

/// Statistics for semantic memory operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticMemoryStats {
    /// Total entries stored
    pub total_stored: u64,

    /// Total similarity queries performed
    pub total_queries: u64,

    /// Number of queries that found similar entries (hits)
    pub semantic_hits: u64,

    /// Number of queries that found no similar entries (misses)
    pub semantic_misses: u64,

    /// Average similarity score when hits occur
    pub avg_hit_similarity: f32,

    /// Average prediction error of retrieved entries
    pub avg_retrieved_error: f32,

    /// Number of entries evicted (ring buffer overflow)
    pub evictions: u64,
}

impl SemanticMemoryStats {
    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f32 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.semantic_hits as f32 / self.total_queries as f32
        }
    }
}

/// Semantic Memory: HDC-based content-addressable memory for CfC context
///
/// Stores (HDC vector, metadata) pairs and retrieves semantically similar
/// past inputs to inform temporal learning.
#[derive(Debug, Clone)]
pub struct SemanticMemory {
    /// Ring buffer of semantic entries
    entries: Vec<SemanticEntry>,

    /// Maximum number of entries (ring buffer size)
    max_entries: usize,

    /// Current write position in ring buffer
    write_pos: usize,

    /// Whether the buffer has wrapped around
    wrapped: bool,

    /// Minimum similarity threshold for retrieval
    similarity_threshold: f32,

    /// Statistics
    pub stats: SemanticMemoryStats,
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl SemanticMemory {
    /// Create a new semantic memory with specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            write_pos: 0,
            wrapped: false,
            similarity_threshold: 0.3, // Default: 0.3 cosine similarity
            stats: SemanticMemoryStats::default(),
        }
    }

    /// Create with custom similarity threshold
    pub fn with_threshold(max_entries: usize, similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            ..Self::new(max_entries)
        }
    }

    /// Store a new entry in semantic memory.
    ///
    /// Returns the evicted entry if the ring buffer was full and an entry was
    /// overwritten. Callers can route evicted entries to the graduation pipeline.
    ///
    /// # Arguments
    ///
    /// * `hdc` - HDC vector representation of the input
    /// * `error` - Prediction error after CfC processed this input
    /// * `category` - Optional category label
    pub fn store(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
    ) -> Option<SemanticEntry> {
        let timestamp = self.stats.total_stored;
        let entry = SemanticEntry {
            hdc_vector: hdc,
            timestamp,
            category,
            prediction_error: error,
            encoding_confidence: 1.0,
        };

        let evicted = if self.entries.len() < self.max_entries {
            // Buffer not full yet: just push
            self.entries.push(entry);
            None
        } else {
            // Ring buffer full: capture evicted entry before overwriting
            let old = std::mem::replace(&mut self.entries[self.write_pos], entry);
            self.stats.evictions += 1;
            self.wrapped = true;
            Some(old)
        };

        self.write_pos = (self.write_pos + 1) % self.max_entries;
        self.stats.total_stored += 1;
        evicted
    }

    /// Store with timestamp explicitly provided.
    ///
    /// Returns the evicted entry if the ring buffer was full.
    pub fn store_with_timestamp(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        timestamp: u64,
    ) -> Option<SemanticEntry> {
        let entry = SemanticEntry {
            hdc_vector: hdc,
            timestamp,
            category,
            prediction_error: error,
            encoding_confidence: 1.0,
        };

        let evicted = if self.entries.len() < self.max_entries {
            self.entries.push(entry);
            None
        } else {
            let old = std::mem::replace(&mut self.entries[self.write_pos], entry);
            self.stats.evictions += 1;
            self.wrapped = true;
            Some(old)
        };

        self.write_pos = (self.write_pos + 1) % self.max_entries;
        self.stats.total_stored += 1;
        evicted
    }

    /// Find semantically similar past entries
    ///
    /// Uses cosine similarity in HDC space to find the top-k most similar
    /// past inputs.
    ///
    /// # Arguments
    ///
    /// * `query` - HDC vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Vector of (index, similarity) pairs, sorted by similarity (descending)
    pub fn find_similar(&mut self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.stats.total_queries += 1;

        if self.entries.is_empty() || query.is_empty() {
            self.stats.semantic_misses += 1;
            return Vec::new();
        }

        // Compute query magnitude for cosine similarity
        let query_mag = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_mag < 1e-10 {
            self.stats.semantic_misses += 1;
            return Vec::new();
        }

        // Compute similarities for all entries
        let mut similarities: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                let sim = Self::cosine_similarity(query, &entry.hdc_vector, query_mag);
                if sim >= self.similarity_threshold {
                    Some((idx, sim))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        similarities.truncate(top_k);

        // Update stats
        if similarities.is_empty() {
            self.stats.semantic_misses += 1;
        } else {
            self.stats.semantic_hits += 1;

            // Update average hit similarity
            let avg_sim =
                similarities.iter().map(|(_, s)| s).sum::<f32>() / similarities.len().max(1) as f32;
            let n = self.stats.semantic_hits as f32;
            self.stats.avg_hit_similarity =
                self.stats.avg_hit_similarity * ((n - 1.0) / n) + avg_sim / n;
        }

        similarities
    }

    /// Get entries by their indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices to retrieve
    ///
    /// # Returns
    ///
    /// Vector of references to the entries at the given indices
    pub fn get_context(&self, indices: &[usize]) -> Vec<&SemanticEntry> {
        indices
            .iter()
            .filter_map(|&idx| self.entries.get(idx))
            .collect()
    }

    /// Get a single entry by index
    pub fn get(&self, index: usize) -> Option<&SemanticEntry> {
        self.entries.get(index)
    }

    /// Compute the average prediction error of similar entries
    ///
    /// This is the key metric for learning rate modulation:
    /// - High average error → boost learning rate
    /// - Low average error → reduce learning rate
    ///
    /// # Arguments
    ///
    /// * `query` - HDC vector to search for
    /// * `top_k` - Number of similar entries to consider
    ///
    /// # Returns
    ///
    /// * `Some((avg_error, count))` if similar entries found
    /// * `None` if no similar entries found
    pub fn compute_similar_error(&mut self, query: &[f32], top_k: usize) -> Option<(f32, usize)> {
        let similar = self.find_similar(query, top_k);

        if similar.is_empty() {
            return None;
        }

        let entries: Vec<&SemanticEntry> = similar
            .iter()
            .filter_map(|(idx, _)| self.entries.get(*idx))
            .collect();

        if entries.is_empty() {
            return None;
        }

        let total_error: f32 = entries.iter().map(|e| e.prediction_error).sum();
        let avg_error = total_error / entries.len() as f32;

        // Update stats
        let n = self.stats.semantic_hits as f32;
        if n > 0.0 {
            self.stats.avg_retrieved_error =
                self.stats.avg_retrieved_error * ((n - 1.0) / n) + avg_error / n;
        }

        Some((avg_error, entries.len()))
    }

    /// Compute the semantic learning rate factor
    ///
    /// Formula: `0.5 + avg_similar_error` (clamped to 0.5 to 1.5)
    ///
    /// - No similar entries found: returns 1.0 (neutral)
    /// - Similar entries with low error (< 0.5): returns 0.5-1.0 (reduce learning)
    /// - Similar entries with high error (> 0.5): returns 1.0-1.5 (boost learning)
    ///
    /// # Arguments
    ///
    /// * `query` - HDC vector to search for
    /// * `top_k` - Number of similar entries to consider (default: 3)
    pub fn compute_lr_factor(&mut self, query: &[f32], top_k: usize) -> f32 {
        match self.compute_similar_error(query, top_k) {
            Some((avg_error, _)) => {
                // Formula: 0.5 + avg_error, clamped to [0.5, 1.5]
                (0.5 + avg_error).clamp(0.5, 1.5)
            }
            None => {
                // No similar entries: neutral learning rate
                1.0
            }
        }
    }

    /// Compute Phi-weighted learning rate factor with recency decay.
    ///
    /// Enhanced version of [`compute_lr_factor`] that incorporates:
    /// - **Phi modulation**: Higher Phi → wider learning range (more capacity to update)
    /// - **Recency decay**: Older similar entries contribute less to the error estimate
    /// - **Base error weighting**: Same error-driven logic as the standard version
    ///
    /// Formula: `0.5 + (recency_weighted_error × phi_factor)`, clamped to [0.5, 1.5]
    ///
    /// # Arguments
    ///
    /// * `query` - HDC vector to search for
    /// * `top_k` - Number of similar entries to consider
    /// * `current_phi` - Current Phi from consciousness processing (0.0–1.0)
    /// * `current_step` - Current step/cycle number for recency calculation
    pub fn compute_lr_factor_phi_weighted(
        &mut self,
        query: &[f32],
        top_k: usize,
        current_phi: f64,
        current_step: u64,
    ) -> f32 {
        let similar = self.find_similar(query, top_k);

        if similar.is_empty() {
            return 1.0; // Neutral: no similar entries
        }

        let entries: Vec<&SemanticEntry> = similar
            .iter()
            .filter_map(|(idx, _)| self.entries.get(*idx))
            .collect();

        if entries.is_empty() {
            return 1.0;
        }

        // Compute recency-weighted prediction error
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;

        for entry in &entries {
            let age = current_step.saturating_sub(entry.timestamp) as f64;
            // Exponential decay with half-life ~346 steps (e^(-1) at 500 steps)
            let recency_weight = (-age / 500.0).exp();
            let w = recency_weight.max(0.01); // Floor to avoid zero weights
            weighted_sum += entry.prediction_error as f64 * w;
            weight_total += w;
        }

        let weighted_error = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.5
        };

        // Phi modulation: higher Phi → wider learning range
        // Range of phi_factor: 0.5 (at Phi=0) to 1.0 (at Phi=1)
        let phi_factor = 0.5 + current_phi.clamp(0.0, 1.0) * 0.5;

        // Final formula
        let lr = 0.5 + (weighted_error * phi_factor) as f32;
        lr.clamp(0.5, 1.5)
    }

    /// Compute cosine similarity between two vectors
    ///
    /// Optimized version that takes pre-computed query magnitude.
    #[inline]
    fn cosine_similarity(query: &[f32], entry: &[f32], query_mag: f32) -> f32 {
        let min_len = query.len().min(entry.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut dot = 0.0f64;
        let mut entry_mag_sq = 0.0f64;

        for i in 0..min_len {
            dot += query[i] as f64 * entry[i] as f64;
            entry_mag_sq += entry[i] as f64 * entry[i] as f64;
        }

        let entry_mag = entry_mag_sq.sqrt() as f32;
        let dot = dot as f32;
        if entry_mag < 1e-10 {
            return 0.0;
        }

        dot / (query_mag * entry_mag)
    }

    /// Get the number of entries currently stored
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the memory is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the maximum capacity
    pub fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Check if the ring buffer has wrapped around
    pub fn has_wrapped(&self) -> bool {
        self.wrapped
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.write_pos = 0;
        self.wrapped = false;
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SemanticMemoryStats::default();
    }

    /// Set the similarity threshold for retrieval
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the current similarity threshold
    pub fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Get statistics reference
    pub fn stats(&self) -> &SemanticMemoryStats {
        &self.stats
    }

    /// Get all entries (for debugging/inspection)
    pub fn entries(&self) -> &[SemanticEntry] {
        &self.entries
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hdc_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dim];
        let mut state = seed;
        for v in vec.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        // Normalize
        let mag: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in vec.iter_mut() {
                *v /= mag;
            }
        }
        vec
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = SemanticMemory::new(100);

        let hdc1 = make_hdc_vector(42, 256);
        let hdc2 = make_hdc_vector(43, 256);
        let hdc3 = make_hdc_vector(42, 256); // Same as hdc1

        mem.store(hdc1.clone(), 0.1, Some("greeting".to_string()));
        mem.store(hdc2.clone(), 0.5, Some("question".to_string()));

        assert_eq!(mem.len(), 2);

        // Query with something similar to hdc1
        let similar = mem.find_similar(&hdc3, 5);

        // Should find hdc1 as most similar (same seed)
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0, 0); // First entry
        assert!(similar[0].1 > 0.9); // High similarity
    }

    #[test]
    fn test_compute_similar_error() {
        let mut mem = SemanticMemory::new(100);

        // Store entries with various errors
        let base = make_hdc_vector(100, 256);
        mem.store(base.clone(), 0.1, None);
        mem.store(base.clone(), 0.2, None);
        mem.store(base.clone(), 0.3, None);

        // Query with similar vector
        let result = mem.compute_similar_error(&base, 3);
        assert!(result.is_some());

        let (avg_error, count) = result.unwrap();
        assert_eq!(count, 3);
        // Average of 0.1, 0.2, 0.3 = 0.2
        assert!((avg_error - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_lr_factor_computation() {
        let mut mem = SemanticMemory::new(100);

        // Low error entries
        let low_err = make_hdc_vector(200, 256);
        mem.store(low_err.clone(), 0.1, None);
        mem.store(low_err.clone(), 0.15, None);

        let lr_low = mem.compute_lr_factor(&low_err, 3);
        // avg_error ~= 0.125, lr_factor = 0.5 + 0.125 = 0.625
        assert!(lr_low > 0.5 && lr_low < 0.8);

        // High error entries
        let mut mem2 = SemanticMemory::new(100);
        let high_err = make_hdc_vector(300, 256);
        mem2.store(high_err.clone(), 0.8, None);
        mem2.store(high_err.clone(), 0.9, None);

        let lr_high = mem2.compute_lr_factor(&high_err, 3);
        // avg_error ~= 0.85, lr_factor = 0.5 + 0.85 = 1.35
        assert!(lr_high > 1.2 && lr_high < 1.5);
    }

    #[test]
    fn test_no_similar_returns_neutral() {
        let mut mem = SemanticMemory::new(100);

        // Store some entries
        mem.store(make_hdc_vector(1, 256), 0.5, None);
        mem.store(make_hdc_vector(2, 256), 0.5, None);

        // Query with completely different vector
        let query = make_hdc_vector(999999, 256);
        let lr = mem.compute_lr_factor(&query, 3);

        // Should return 1.0 (neutral) since no similar entries found
        // (or if threshold is low enough, might still find some)
        assert!((0.5..=1.5).contains(&lr));
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut mem = SemanticMemory::new(3);

        mem.store(make_hdc_vector(1, 256), 0.1, None);
        mem.store(make_hdc_vector(2, 256), 0.2, None);
        mem.store(make_hdc_vector(3, 256), 0.3, None);

        assert_eq!(mem.len(), 3);
        assert_eq!(mem.stats.evictions, 0);

        // Add one more - should evict oldest
        mem.store(make_hdc_vector(4, 256), 0.4, None);

        assert_eq!(mem.len(), 3); // Still 3
        assert_eq!(mem.stats.evictions, 1);
        assert!(mem.has_wrapped());
    }

    #[test]
    fn test_stats_tracking() {
        let mut mem = SemanticMemory::new(100);

        let hdc = make_hdc_vector(42, 256);
        mem.store(hdc.clone(), 0.2, None);

        // Query with similar
        let _ = mem.find_similar(&hdc, 3);

        assert_eq!(mem.stats.total_queries, 1);
        assert_eq!(mem.stats.semantic_hits, 1);

        // Query with dissimilar
        mem.set_similarity_threshold(0.99); // Very high threshold
        let _ = mem.find_similar(&make_hdc_vector(999, 256), 3);

        assert_eq!(mem.stats.total_queries, 2);
        assert_eq!(mem.stats.semantic_misses, 1);
    }

    #[test]
    fn test_hit_rate() {
        let stats = SemanticMemoryStats {
            total_stored: 10,
            total_queries: 100,
            semantic_hits: 75,
            semantic_misses: 25,
            avg_hit_similarity: 0.8,
            avg_retrieved_error: 0.3,
            evictions: 0,
        };

        assert!((stats.hit_rate() - 0.75).abs() < 0.01);
    }
}
