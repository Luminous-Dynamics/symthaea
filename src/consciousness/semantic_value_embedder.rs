//! # Semantic Value Embedder: Value-Aligned Semantic Representations
//!
//! Creates semantic embeddings that incorporate value alignment,
//! allowing concepts to be represented with their ethical implications.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::RealHV;
use super::seven_harmonies::Harmony;

/// Configuration for the semantic value embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Dimension of embeddings
    pub dimension: usize,
    /// Value weighting strength
    pub value_weight: f32,
    /// Semantic weighting strength
    pub semantic_weight: f32,
    /// Cache size
    pub cache_size: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            value_weight: 0.3,
            semantic_weight: 0.7,
            cache_size: 10000,
        }
    }
}

/// A value-embedded concept
#[derive(Debug, Clone)]
pub struct ValueEmbeddedConcept {
    /// Concept identifier
    pub id: String,
    /// Original semantic embedding
    pub semantic_embedding: RealHV,
    /// Value embedding
    pub value_embedding: RealHV,
    /// Combined embedding
    pub combined_embedding: RealHV,
    /// Value alignment scores
    pub value_scores: HashMap<Harmony, f32>,
    /// Concept metadata
    pub metadata: HashMap<String, String>,
}

impl ValueEmbeddedConcept {
    /// Get the primary embedding (combined)
    pub fn embedding(&self) -> &RealHV {
        &self.combined_embedding
    }

    /// Get semantic similarity to another concept
    pub fn semantic_similarity(&self, other: &ValueEmbeddedConcept) -> f32 {
        self.semantic_embedding.similarity(&other.semantic_embedding)
    }

    /// Get value similarity to another concept
    pub fn value_similarity(&self, other: &ValueEmbeddedConcept) -> f32 {
        self.value_embedding.similarity(&other.value_embedding)
    }

    /// Get overall similarity (weighted)
    pub fn similarity(&self, other: &ValueEmbeddedConcept, value_weight: f32) -> f32 {
        let semantic = self.semantic_similarity(other);
        let value = self.value_similarity(other);
        semantic * (1.0 - value_weight) + value * value_weight
    }
}

/// The semantic value embedder
#[derive(Debug)]
pub struct SemanticValueEmbedder {
    /// Configuration
    config: EmbedderConfig,
    /// Harmony basis vectors
    harmony_bases: HashMap<Harmony, RealHV>,
    /// Concept cache
    cache: HashMap<String, ValueEmbeddedConcept>,
    /// Statistics
    stats: EmbedderStats,
}

/// Statistics for the embedder
#[derive(Debug, Clone, Default)]
pub struct EmbedderStats {
    /// Total embeddings created
    pub embeddings_created: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}

impl SemanticValueEmbedder {
    /// Create a new embedder
    pub fn new(config: EmbedderConfig) -> Self {
        let dim = config.dimension;

        // Initialize harmony basis vectors
        let mut harmony_bases = HashMap::new();
        for harmony in [
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::InfinitePlay,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
            Harmony::EvolutionaryProgression,
        ] {
            harmony_bases.insert(harmony, RealHV::random(dim, 42));
        }

        Self {
            config,
            harmony_bases,
            cache: HashMap::new(),
            stats: EmbedderStats::default(),
        }
    }

    /// Embed a concept with value alignment
    pub fn embed(&mut self, id: impl Into<String>, semantic: RealHV) -> ValueEmbeddedConcept {
        let id = id.into();

        // Check cache
        if let Some(cached) = self.cache.get(&id) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }
        self.stats.cache_misses += 1;

        // Calculate value alignment scores
        let mut value_scores = HashMap::new();
        let mut value_components = Vec::new();

        for (harmony, basis) in &self.harmony_bases {
            let score = semantic.similarity(basis);
            value_scores.insert(*harmony, score);
            value_components.push(basis.clone().scale(score));
        }

        // Create value embedding by combining harmony components
        let value_embedding = if value_components.is_empty() {
            RealHV::random(self.config.dimension, 42)
        } else {
            RealHV::bundle(&value_components)
        };

        // Create combined embedding
        let semantic_scaled = semantic.clone().scale(self.config.semantic_weight);
        let value_scaled = value_embedding.clone().scale(self.config.value_weight);
        let combined_embedding = RealHV::bundle(&[semantic_scaled, value_scaled]);

        let concept = ValueEmbeddedConcept {
            id: id.clone(),
            semantic_embedding: semantic,
            value_embedding,
            combined_embedding,
            value_scores,
            metadata: HashMap::new(),
        };

        // Cache if within size limit
        if self.cache.len() < self.config.cache_size {
            self.cache.insert(id, concept.clone());
        }

        self.stats.embeddings_created += 1;
        concept
    }

    /// Embed text (requires external embedding model - placeholder)
    pub fn embed_text(&mut self, id: impl Into<String>, text: &str) -> ValueEmbeddedConcept {
        // Generate pseudo-embedding from text (would use real model in production)
        let semantic = self.text_to_hv(text);
        self.embed(id, semantic)
    }

    /// Convert text to hypervector (simplified hash-based method)
    fn text_to_hv(&self, text: &str) -> RealHV {
        let mut values = vec![0.0f32; self.config.dimension];

        // Simple character-based embedding (production would use real model)
        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i) % self.config.dimension;
            values[idx] += 1.0;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        RealHV::from_slice(&values)
    }

    /// Find most similar concepts
    pub fn find_similar(&self, query: &ValueEmbeddedConcept, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<_> = self.cache.iter()
            .map(|(id, concept)| {
                let sim = query.similarity(concept, self.config.value_weight);
                (id.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Get a cached concept
    pub fn get(&self, id: &str) -> Option<&ValueEmbeddedConcept> {
        self.cache.get(id)
    }

    /// Update harmony basis based on feedback
    pub fn update_harmony_basis(&mut self, harmony: Harmony, adjustment: &RealHV, rate: f32) {
        if let Some(basis) = self.harmony_bases.get_mut(&harmony) {
            let scaled = adjustment.clone().scale(rate);
            *basis = RealHV::bundle(&[basis.clone(), scaled]);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &EmbedderStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f32 / total as f32
        }
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SemanticValueEmbedder {
    fn default() -> Self {
        Self::new(EmbedderConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = SemanticValueEmbedder::default();
        assert_eq!(embedder.stats.embeddings_created, 0);
    }

    #[test]
    fn test_embedding() {
        let mut embedder = SemanticValueEmbedder::default();
        let semantic = RealHV::random(512, 42);
        let concept = embedder.embed("test", semantic);

        assert_eq!(concept.id, "test");
        assert!(!concept.value_scores.is_empty());
    }

    #[test]
    fn test_text_embedding() {
        let mut embedder = SemanticValueEmbedder::default();
        let concept = embedder.embed_text("greeting", "hello world");

        assert_eq!(concept.id, "greeting");
    }

    #[test]
    fn test_similarity() {
        let mut embedder = SemanticValueEmbedder::default();

        let c1 = embedder.embed("c1", RealHV::random(512, 42));
        let c2 = embedder.embed("c2", RealHV::random(512, 42));

        let sim = c1.similarity(&c2, 0.3);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_caching() {
        let mut embedder = SemanticValueEmbedder::default();

        // First embed creates new
        let _ = embedder.embed("test", RealHV::random(512, 42));
        assert_eq!(embedder.stats.cache_misses, 1);

        // Second embed uses cache
        let _ = embedder.embed("test", RealHV::random(512, 42));
        assert_eq!(embedder.stats.cache_hits, 1);
    }
}
