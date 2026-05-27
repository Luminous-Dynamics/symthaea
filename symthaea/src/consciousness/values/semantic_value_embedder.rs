// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Semantic Value Embedder: Value-Aligned Semantic Representations
//!
//! Creates semantic embeddings that incorporate value alignment,
//! allowing concepts to be represented with their ethical implications.

use super::eight_harmonies::Harmony;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

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
    pub semantic_embedding: ContinuousHV,
    /// Value embedding
    pub value_embedding: ContinuousHV,
    /// Combined embedding
    pub combined_embedding: ContinuousHV,
    /// Value alignment scores
    pub value_scores: HashMap<Harmony, f32>,
    /// Concept metadata
    pub metadata: HashMap<String, String>,
}

impl ValueEmbeddedConcept {
    /// Get the primary embedding (combined)
    pub fn embedding(&self) -> &ContinuousHV {
        &self.combined_embedding
    }

    /// Get semantic similarity to another concept
    pub fn semantic_similarity(&self, other: &ValueEmbeddedConcept) -> f32 {
        self.semantic_embedding
            .similarity(&other.semantic_embedding)
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
    harmony_bases: HashMap<Harmony, ContinuousHV>,
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
    /// Create a new embedder with primitive-grounded harmony bases
    pub fn new(config: EmbedderConfig) -> Self {
        let dim = config.dimension;
        let system = PrimitiveSystem::global();

        // Ground harmony bases in primitive tiers
        // Each harmony maps to a primitive tier, creating semantic alignment
        let harmony_bases = Self::build_primitive_grounded_bases(dim, system);

        Self {
            config,
            harmony_bases,
            cache: HashMap::new(),
            stats: EmbedderStats::default(),
        }
    }

    /// Build harmony bases grounded in primitive tiers
    ///
    /// Maps each of the 8 Harmonies to a primitive tier, then creates
    /// a basis vector by bundling the tier's primitive encodings.
    fn build_primitive_grounded_bases(
        dim: usize,
        system: &PrimitiveSystem,
    ) -> HashMap<Harmony, ContinuousHV> {
        let mut bases = HashMap::new();

        // Mapping of Harmonies to Primitive Tiers:
        // - ResonantCoherence → Geometric (structural coherence)
        // - PanSentientFlourishing → Consciousness (phenomenal experience)
        // - IntegralWisdom → Compositional (higher-order integration)
        // - InfinitePlay → Mathematical (infinite generativity)
        // - UniversalInterconnectedness → Physical (causal connections)
        // - SacredReciprocity → Strategic (game theory, reciprocity)
        // - EvolutionaryProgression → Temporal (change over time)

        let tier_mapping: [(Harmony, PrimitiveTier); 7] = [
            (Harmony::ResonantCoherence, PrimitiveTier::Geometric),
            (
                Harmony::PanSentientFlourishing,
                PrimitiveTier::Consciousness,
            ),
            (Harmony::IntegralWisdom, PrimitiveTier::Compositional),
            (Harmony::InfinitePlay, PrimitiveTier::Mathematical),
            (
                Harmony::UniversalInterconnectedness,
                PrimitiveTier::Physical,
            ),
            (Harmony::SacredReciprocity, PrimitiveTier::Strategic),
            (Harmony::EvolutionaryProgression, PrimitiveTier::Temporal),
        ];

        for (harmony, tier) in tier_mapping {
            let primitives = system.get_tier(tier);

            if primitives.is_empty() {
                // Fallback to random if tier is empty (e.g., NSM)
                bases.insert(harmony, ContinuousHV::random(dim, harmony as u64 + 42));
            } else {
                // Bundle primitive encodings (BinaryHV → ContinuousHV conversion)
                // Convert binary BinaryHV bits to bipolar ContinuousHV values (-1.0/+1.0)
                let real_hvs: Vec<ContinuousHV> = primitives
                    .iter()
                    .take(16) // Limit to avoid over-bundling
                    .map(|p| Self::hv16_to_real(&p.encoding, dim))
                    .collect();

                if real_hvs.is_empty() {
                    bases.insert(harmony, ContinuousHV::random(dim, harmony as u64 + 42));
                } else {
                    bases.insert(harmony, ContinuousHV::bundle_owned(&real_hvs));
                }
            }
        }

        bases
    }

    /// Convert BinaryHV to ContinuousHV with specified dimension
    ///
    /// Maps binary bits to bipolar values: 0 → -1.0, 1 → +1.0
    /// Uses BinaryHV::to_bipolar() then resamples to target dimension
    fn hv16_to_real(hv: &symthaea_core::hdc::binary_hv::BinaryHV, dim: usize) -> ContinuousHV {
        let bipolar = hv.to_bipolar(); // Returns Vec<f32> with ±1.0 values

        if dim == bipolar.len() {
            ContinuousHV::from_values(bipolar)
        } else {
            // Resample to target dimension
            let mut values = Vec::with_capacity(dim);
            for i in 0..dim {
                let idx = i % bipolar.len();
                values.push(bipolar[idx]);
            }
            ContinuousHV::from_values(values)
        }
    }

    /// Embed a concept with value alignment
    pub fn embed(&mut self, id: impl Into<String>, semantic: ContinuousHV) -> ValueEmbeddedConcept {
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
            ContinuousHV::random(self.config.dimension, 42)
        } else {
            ContinuousHV::bundle_owned(&value_components)
        };

        // Create combined embedding
        let semantic_scaled = semantic.clone().scale(self.config.semantic_weight);
        let value_scaled = value_embedding.clone().scale(self.config.value_weight);
        let combined_embedding = ContinuousHV::bundle_owned(&[semantic_scaled, value_scaled]);

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
    fn text_to_hv(&self, text: &str) -> ContinuousHV {
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

        ContinuousHV::from_slice(&values)
    }

    /// Find most similar concepts
    pub fn find_similar(&self, query: &ValueEmbeddedConcept, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<_> = self
            .cache
            .iter()
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
    pub fn update_harmony_basis(&mut self, harmony: Harmony, adjustment: &ContinuousHV, rate: f32) {
        if let Some(basis) = self.harmony_bases.get_mut(&harmony) {
            let scaled = adjustment.clone().scale(rate);
            *basis = ContinuousHV::bundle_owned(&[basis.clone(), scaled]);
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
        let semantic = ContinuousHV::random(512, 42);
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

        let c1 = embedder.embed("c1", ContinuousHV::random(512, 42));
        let c2 = embedder.embed("c2", ContinuousHV::random(512, 42));

        let sim = c1.similarity(&c2, 0.3);
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_caching() {
        let mut embedder = SemanticValueEmbedder::default();

        // First embed creates new
        let _ = embedder.embed("test", ContinuousHV::random(512, 42));
        assert_eq!(embedder.stats.cache_misses, 1);

        // Second embed uses cache
        let _ = embedder.embed("test", ContinuousHV::random(512, 42));
        assert_eq!(embedder.stats.cache_hits, 1);
    }
}
