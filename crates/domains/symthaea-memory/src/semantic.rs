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

use crate::episodic_replay::GraduationProvenance;
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

/// Version of the provenance-bearing semantic replay envelope.
pub const PROVENANCED_SEMANTIC_ENTRY_SCHEMA_VERSION: u16 = 1;

/// Semantic entry plus source-class / producer-claim lineage.
///
/// The envelope is serializable, but SemanticMemory remains an in-memory ring
/// buffer. The producer claim is not an independent verifier receipt or permit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenancedSemanticEntry {
    pub schema_version: u16,
    pub entry: SemanticEntry,
    pub graduation_provenance: Option<GraduationProvenance>,
}

/// Input rejected by the semantic-memory admission or query boundary.
///
/// The compatibility APIs continue to return their historical shapes; callers
/// that need to distinguish an invalid input from an empty result should use
/// the corresponding `try_*` method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticMemoryInputError {
    ZeroCapacity,
    EmptyVector,
    NonFiniteVector { index: usize },
    NonFiniteQuery { index: usize },
    ZeroNormVector,
    NonFinitePredictionError,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidSimilarityThreshold,
}

impl std::fmt::Display for SemanticMemoryInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "semantic memory has zero capacity"),
            Self::EmptyVector => write!(f, "HDC vector must not be empty"),
            Self::NonFiniteVector { index } => {
                write!(f, "stored HDC vector contains a non-finite value at index {index}")
            }
            Self::NonFiniteQuery { index } => {
                write!(f, "query HDC vector contains a non-finite value at index {index}")
            }
            Self::ZeroNormVector => write!(f, "HDC vector must have non-zero norm"),
            Self::NonFinitePredictionError => {
                write!(f, "prediction error must be finite")
            }
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "HDC vector dimension mismatch: expected {expected}, received {actual}"
            ),
            Self::InvalidSimilarityThreshold => {
                write!(f, "similarity threshold must be finite and within [0, 1]")
            }
        }
    }
}

impl std::error::Error for SemanticMemoryInputError {}

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

    /// Parallel lineage slots; indices always match `entries`.
    graduation_provenance: Vec<Option<GraduationProvenance>>,

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
            graduation_provenance: Vec::with_capacity(max_entries),
            max_entries,
            write_pos: 0,
            wrapped: false,
            similarity_threshold: 0.3, // Default: 0.3 cosine similarity
            stats: SemanticMemoryStats::default(),
        }
    }

    /// Create with custom similarity threshold.
    ///
    /// Invalid thresholds fall back to the default threshold (0.3). Prefer
    /// `try_with_threshold` when configuration errors must be surfaced.
    pub fn with_threshold(max_entries: usize, similarity_threshold: f32) -> Self {
        Self::try_with_threshold(max_entries, similarity_threshold)
            .unwrap_or_else(|_| Self::new(max_entries))
    }

    /// Create with a validated similarity threshold.
    pub fn try_with_threshold(
        max_entries: usize,
        similarity_threshold: f32,
    ) -> Result<Self, SemanticMemoryInputError> {
        if !similarity_threshold.is_finite() || !(0.0..=1.0).contains(&similarity_threshold) {
            return Err(SemanticMemoryInputError::InvalidSimilarityThreshold);
        }
        Ok(Self {
            similarity_threshold,
            ..Self::new(max_entries)
        })
    }

    fn validate_hdc_vector(
        vector: &[f32],
        expected_dimension: Option<usize>,
        query: bool,
    ) -> Result<f64, SemanticMemoryInputError> {
        if vector.is_empty() {
            return Err(SemanticMemoryInputError::EmptyVector);
        }
        for (index, value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(if query {
                    SemanticMemoryInputError::NonFiniteQuery { index }
                } else {
                    SemanticMemoryInputError::NonFiniteVector { index }
                });
            }
        }
        let norm_squared = vector
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>();
        if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
            return Err(SemanticMemoryInputError::ZeroNormVector);
        }
        if let Some(expected) = expected_dimension {
            if vector.len() != expected {
                return Err(SemanticMemoryInputError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                });
            }
        }
        Ok(norm_squared.sqrt())
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
        self.store_with_timestamp_and_provenance(hdc, error, category, timestamp, None)
            .map(|evicted| evicted.entry)
    }

    /// Store an entry together with source lineage received from episodic promotion.
    ///
    /// The verification bit remains a producer claim. It does not affect the
    /// semantic similarity score, learning-rate factor, source trust, or action authority.
    pub fn store_with_provenance(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        graduation_provenance: Option<GraduationProvenance>,
    ) -> Option<ProvenancedSemanticEntry> {
        let timestamp = self.stats.total_stored;
        self.store_with_timestamp_and_provenance(
            hdc,
            error,
            category,
            timestamp,
            graduation_provenance,
        )
    }

    /// Validated semantic store. Unlike the compatibility API, this method
    /// distinguishes rejected input from a successful insertion without eviction.
    pub fn try_store_with_provenance(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        graduation_provenance: Option<GraduationProvenance>,
    ) -> Result<Option<ProvenancedSemanticEntry>, SemanticMemoryInputError> {
        let timestamp = self.stats.total_stored;
        self.try_store_with_timestamp_and_provenance(
            hdc,
            error,
            category,
            timestamp,
            graduation_provenance,
        )
    }

    /// Store with timestamp explicitly provided and optional source lineage.
    ///
    /// Returns the evicted entry and its prior lineage when the ring buffer wraps.
    pub fn store_with_timestamp_and_provenance(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        timestamp: u64,
        graduation_provenance: Option<GraduationProvenance>,
    ) -> Option<ProvenancedSemanticEntry> {
        self.try_store_with_timestamp_and_provenance(
            hdc,
            error,
            category,
            timestamp,
            graduation_provenance,
        )
        .ok()
        .flatten()
    }

    /// Validated timestamped semantic store. Inputs are checked before the
    /// ring buffer, statistics, or lineage slots are mutated.
    pub fn try_store_with_timestamp_and_provenance(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        timestamp: u64,
        graduation_provenance: Option<GraduationProvenance>,
    ) -> Result<Option<ProvenancedSemanticEntry>, SemanticMemoryInputError> {
        if self.max_entries == 0 {
            return Err(SemanticMemoryInputError::ZeroCapacity);
        }
        if !error.is_finite() {
            return Err(SemanticMemoryInputError::NonFinitePredictionError);
        }
        let expected_dimension = self.entries.first().map(|entry| entry.hdc_vector.len());
        Self::validate_hdc_vector(&hdc, expected_dimension, false)?;

        let entry = SemanticEntry {
            hdc_vector: hdc,
            timestamp,
            category,
            prediction_error: error,
            encoding_confidence: 1.0,
        };

        let evicted = if self.entries.len() < self.max_entries {
            self.entries.push(entry);
            self.graduation_provenance.push(graduation_provenance);
            None
        } else {
            let old_entry = std::mem::replace(&mut self.entries[self.write_pos], entry);
            let old_provenance = std::mem::replace(
                &mut self.graduation_provenance[self.write_pos],
                graduation_provenance,
            );
            self.stats.evictions += 1;
            self.wrapped = true;
            Some(ProvenancedSemanticEntry {
                schema_version: PROVENANCED_SEMANTIC_ENTRY_SCHEMA_VERSION,
                entry: old_entry,
                graduation_provenance: old_provenance,
            })
        };

        self.write_pos = (self.write_pos + 1) % self.max_entries;
        self.stats.total_stored += 1;
        Ok(evicted)
    }

    /// Legacy timestamped store. Lineage is explicitly absent for this caller.
    pub fn store_with_timestamp(
        &mut self,
        hdc: Vec<f32>,
        error: f32,
        category: Option<String>,
        timestamp: u64,
    ) -> Option<SemanticEntry> {
        self.store_with_timestamp_and_provenance(hdc, error, category, timestamp, None)
            .map(|evicted| evicted.entry)
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
        self.try_find_similar(query, top_k).unwrap_or_default()
    }

    /// Validated similarity query. Vectors must be finite, non-zero and have
    /// exactly the same dimension as the stored HDC representation.
    pub fn try_find_similar(
        &mut self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, SemanticMemoryInputError> {
        self.stats.total_queries += 1;
        let expected_dimension = self.entries.first().map(|entry| entry.hdc_vector.len());
        let query_mag = match Self::validate_hdc_vector(query, expected_dimension, true) {
            Ok(magnitude) => magnitude,
            Err(error) => {
                self.stats.semantic_misses += 1;
                return Err(error);
            }
        };

        if self.entries.is_empty() {
            self.stats.semantic_misses += 1;
            return Ok(Vec::new());
        }

        let mut similarities: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                let sim = Self::cosine_similarity(query, &entry.hdc_vector, query_mag);
                if sim.is_finite() && sim >= self.similarity_threshold {
                    Some((idx, sim))
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        if similarities.is_empty() {
            self.stats.semantic_misses += 1;
        } else {
            self.stats.semantic_hits += 1;
            let avg_sim =
                similarities.iter().map(|(_, s)| s).sum::<f32>() / similarities.len().max(1) as f32;
            let n = self.stats.semantic_hits as f32;
            self.stats.avg_hit_similarity =
                self.stats.avg_hit_similarity * ((n - 1.0) / n) + avg_sim / n;
        }

        Ok(similarities)
    }

    /// Find semantically similar entries with lineage preserved.
    pub fn find_similar_with_provenance(
        &mut self,
        query: &[f32],
        top_k: usize,
    ) -> Vec<(ProvenancedSemanticEntry, f32)> {
        let matches = self.find_similar(query, top_k);
        matches
            .into_iter()
            .filter_map(|(index, similarity)| {
                self.get_with_provenance(index)
                    .map(|entry| (entry, similarity))
            })
            .collect()
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

    /// Get a semantic entry with its optional episodic source lineage.
    pub fn get_with_provenance(&self, index: usize) -> Option<ProvenancedSemanticEntry> {
        Some(ProvenancedSemanticEntry {
            schema_version: PROVENANCED_SEMANTIC_ENTRY_SCHEMA_VERSION,
            entry: self.entries.get(index)?.clone(),
            graduation_provenance: self.graduation_provenance.get(index)?.clone(),
        })
    }

    /// Get multiple indexed entries with their corresponding lineage records.
    pub fn get_context_with_provenance(
        &self,
        indices: &[usize],
    ) -> Vec<ProvenancedSemanticEntry> {
        indices
            .iter()
            .filter_map(|&index| self.get_with_provenance(index))
            .collect()
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
        // Invalid cognitive-state input must not inject NaN into the learning
        // rate. Neutral is safer than manufacturing a modulation direction.
        if !current_phi.is_finite() {
            return 1.0;
        }
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
    fn cosine_similarity(query: &[f32], entry: &[f32], query_mag: f64) -> f32 {
        // Dimension mismatch is a hard non-match; never compare only a prefix.
        if query.is_empty() || query.len() != entry.len() {
            return 0.0;
        }

        let mut dot = 0.0f64;
        let mut entry_mag_sq = 0.0f64;
        for i in 0..query.len() {
            dot += query[i] as f64 * entry[i] as f64;
            entry_mag_sq += (entry[i] as f64) * (entry[i] as f64);
        }

        let entry_mag = entry_mag_sq.sqrt();
        if !entry_mag.is_finite() || entry_mag <= f64::EPSILON {
            return 0.0;
        }

        let similarity = dot / (query_mag * entry_mag);
        if similarity.is_finite() {
            similarity.clamp(-1.0, 1.0) as f32
        } else {
            0.0
        }
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
        self.graduation_provenance.clear();
        self.write_pos = 0;
        self.wrapped = false;
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SemanticMemoryStats::default();
    }

    /// Set the similarity threshold for retrieval
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        // Preserve the previous valid threshold when a compatibility caller
        // supplies NaN, infinity or an out-of-domain configuration value.
        let _ = self.try_set_similarity_threshold(threshold);
    }

    /// Set a validated similarity threshold, returning an error without
    /// mutating configuration when the value is invalid.
    pub fn try_set_similarity_threshold(
        &mut self,
        threshold: f32,
    ) -> Result<(), SemanticMemoryInputError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SemanticMemoryInputError::InvalidSimilarityThreshold);
        }
        self.similarity_threshold = threshold;
        Ok(())
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
    fn provenance_tracks_semantic_ring_slots_and_evicted_entry() {
        let mut mem = SemanticMemory::new(2);
        let first = GraduationProvenance {
            source: crate::coordinator::MemorySource::ActionFeedback,
            verification_claimed: true,
        };
        let second = GraduationProvenance {
            source: crate::coordinator::MemorySource::WebResearch,
            verification_claimed: false,
        };
        let third = GraduationProvenance {
            source: crate::coordinator::MemorySource::UserInteraction,
            verification_claimed: false,
        };

        assert!(mem
            .store_with_provenance(vec![1.0, 0.0], 0.1, None, Some(first))
            .is_none());
        assert!(mem
            .store_with_provenance(vec![0.0, 1.0], 0.2, None, Some(second))
            .is_none());

        let evicted = mem
            .store_with_provenance(vec![1.0, 0.0], 0.3, None, Some(third))
            .expect("third write must evict the first semantic entry");
        assert_eq!(
            evicted.graduation_provenance.unwrap().source,
            crate::coordinator::MemorySource::ActionFeedback
        );

        let slot_zero = mem.get_with_provenance(0).unwrap();
        let slot_one = mem.get_with_provenance(1).unwrap();
        assert_eq!(
            slot_zero.graduation_provenance.unwrap().source,
            crate::coordinator::MemorySource::UserInteraction
        );
        assert_eq!(
            slot_one.graduation_provenance.unwrap().source,
            crate::coordinator::MemorySource::WebResearch
        );

        let matches = mem.find_similar_with_provenance(&[1.0, 0.0], 1);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].0.graduation_provenance.as_ref().unwrap().source,
            crate::coordinator::MemorySource::UserInteraction
        );
    }

    #[test]
    fn semantic_provenance_envelope_bincode_round_trip_preserves_claim() {
        let mut mem = SemanticMemory::new(2);
        let provenance = GraduationProvenance {
            source: crate::coordinator::MemorySource::ActionFeedback,
            verification_claimed: true,
        };
        assert!(mem
            .store_with_provenance(
                vec![0.25, 0.75],
                0.125,
                Some("Consolidated Wisdom".to_owned()),
                Some(provenance),
            )
            .is_none());

        let record = mem.get_with_provenance(0).unwrap();
        assert_eq!(
            record.schema_version,
            PROVENANCED_SEMANTIC_ENTRY_SCHEMA_VERSION
        );
        let encoded = bincode::serialize(&record).expect("serialize semantic envelope");
        let decoded: ProvenancedSemanticEntry =
            bincode::deserialize(&encoded).expect("deserialize semantic envelope");

        assert_eq!(decoded.schema_version, PROVENANCED_SEMANTIC_ENTRY_SCHEMA_VERSION);
        assert_eq!(decoded.entry.hdc_vector, vec![0.25, 0.75]);
        assert_eq!(decoded.entry.timestamp, 0);
        assert_eq!(decoded.entry.category.as_deref(), Some("Consolidated Wisdom"));
        let claim = decoded.graduation_provenance.unwrap();
        assert_eq!(
            claim.source,
            crate::coordinator::MemorySource::ActionFeedback
        );
        assert!(claim.verification_claimed);
    }

    #[test]
    fn legacy_semantic_store_marks_lineage_absent_and_clear_removes_parallel_slots() {
        let mut mem = SemanticMemory::new(2);
        mem.store(vec![1.0, 0.0], 0.1, None);
        assert!(mem.get_with_provenance(0).unwrap().graduation_provenance.is_none());

        mem.store_with_provenance(
            vec![0.0, 1.0],
            0.2,
            None,
            Some(GraduationProvenance {
                source: crate::coordinator::MemorySource::ActionFeedback,
                verification_claimed: false,
            }),
        );
        mem.clear();
        assert!(mem.is_empty());
        assert!(mem.graduation_provenance.is_empty());
        assert!(mem.store_with_provenance(
            vec![1.0, 0.0],
            0.3,
            None,
            Some(GraduationProvenance {
                source: crate::coordinator::MemorySource::WebResearch,
                verification_claimed: false,
            }),
        ).is_none());
        assert_eq!(
            mem.get_with_provenance(0).unwrap().graduation_provenance.unwrap().source,
            crate::coordinator::MemorySource::WebResearch
        );
    }

    #[test]
    fn zero_capacity_semantic_memory_rejects_store_without_dividing_by_zero() {
        let mut mem = SemanticMemory::new(0);
        assert!(mem
            .store_with_provenance(vec![1.0], 0.1, None, None)
            .is_none());
        assert!(mem.is_empty());
        assert_eq!(mem.stats.total_stored, 0);
    }


    #[test]
    fn semantic_store_rejects_malformed_vectors_and_errors_before_mutation() {
        let mut mem = SemanticMemory::new(4);

        assert!(matches!(
            mem.try_store_with_provenance(vec![], 0.1, None, None),
            Err(SemanticMemoryInputError::EmptyVector)
        ));
        assert!(matches!(
            mem.try_store_with_provenance(vec![0.0, 0.0], 0.1, None, None),
            Err(SemanticMemoryInputError::ZeroNormVector)
        ));
        assert!(matches!(
            mem.try_store_with_provenance(vec![1.0, f32::NAN], 0.1, None, None),
            Err(SemanticMemoryInputError::NonFiniteVector { index: 1 })
        ));
        assert!(matches!(
            mem.try_store_with_provenance(vec![1.0, 0.0], f32::INFINITY, None, None),
            Err(SemanticMemoryInputError::NonFinitePredictionError)
        ));
        assert!(mem.is_empty());
        assert_eq!(mem.stats.total_stored, 0);

        assert!(mem.store(vec![1.0, f32::NAN], 0.1, None).is_none());
        assert!(mem.is_empty(), "legacy API must also refuse malformed inputs");
    }

    #[test]
    fn semantic_store_requires_exact_vector_dimensions() {
        let mut mem = SemanticMemory::new(4);
        assert!(mem
            .try_store_with_provenance(vec![1.0, 0.0, 0.0], 0.1, None, None)
            .is_ok());
        assert!(matches!(
            mem.try_store_with_provenance(vec![1.0, 0.0], 0.2, None, None),
            Err(SemanticMemoryInputError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert_eq!(mem.len(), 1);
        assert_eq!(mem.stats.total_stored, 1);
    }

    #[test]
    fn semantic_query_rejects_nonfinite_zero_and_wrong_dimension_inputs() {
        let mut mem = SemanticMemory::new(4);
        mem.store(vec![1.0, 0.0, 0.0], 0.1, None);

        assert_eq!(
            mem.try_find_similar(&[1.0, f32::NAN, 0.0], 1),
            Err(SemanticMemoryInputError::NonFiniteQuery { index: 1 })
        );
        assert_eq!(
            mem.try_find_similar(&[0.0, 0.0, 0.0], 1),
            Err(SemanticMemoryInputError::ZeroNormVector)
        );
        assert_eq!(
            mem.try_find_similar(&[1.0, 0.0], 1),
            Err(SemanticMemoryInputError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert!(mem.find_similar(&[1.0, 0.0], 1).is_empty());
        assert_eq!(mem.stats.total_queries, 4);
        assert_eq!(mem.stats.semantic_misses, 4);
    }

    #[test]
    fn semantic_cosine_uses_f64_magnitudes_for_large_finite_vectors() {
        let mut mem = SemanticMemory::new(2);
        let large = vec![f32::MAX; 4];
        assert!(mem.try_store_with_provenance(large.clone(), 0.1, None, None).is_ok());
        let matches = mem.try_find_similar(&large, 1).unwrap();
        assert_eq!(matches.len(), 1);
        assert!((matches[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_similarity_threshold_does_not_poison_configuration() {
        let mut mem = SemanticMemory::with_threshold(4, 0.6);
        assert_eq!(
            mem.try_set_similarity_threshold(f32::NAN),
            Err(SemanticMemoryInputError::InvalidSimilarityThreshold)
        );
        mem.set_similarity_threshold(f32::INFINITY);
        assert_eq!(mem.similarity_threshold(), 0.6);
        assert!(matches!(
            SemanticMemory::try_with_threshold(4, f32::NAN),
            Err(SemanticMemoryInputError::InvalidSimilarityThreshold)
        ));
        assert_eq!(SemanticMemory::with_threshold(4, f32::NAN).similarity_threshold(), 0.3);
    }

    #[test]
    fn nonfinite_phi_uses_neutral_semantic_learning_rate() {
        let mut mem = SemanticMemory::new(2);
        let vector = make_hdc_vector(123, 8);
        mem.store(vector.clone(), 0.9, None);
        assert_eq!(mem.compute_lr_factor_phi_weighted(&vector, 1, f64::NAN, 10), 1.0);
        assert_eq!(mem.compute_lr_factor_phi_weighted(&vector, 1, f64::INFINITY, 10), 1.0);
    }

    #[test]
    fn zero_capacity_try_store_returns_explicit_error() {
        let mut mem = SemanticMemory::new(0);
        assert!(matches!(
            mem.try_store_with_provenance(vec![1.0], 0.1, None, None),
            Err(SemanticMemoryInputError::ZeroCapacity)
        ));
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
