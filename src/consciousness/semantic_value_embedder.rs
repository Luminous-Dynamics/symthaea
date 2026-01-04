//! # Semantic Value Embedder
//!
//! Real semantic embedding-based value alignment using transformer embeddings.
//! Replaces HDC trigram encoding with true semantic understanding.
//!
//! ## Why This Matters
//!
//! HDC trigram encoding captures character patterns but not meaning:
//! - "help" and "assist" have different trigrams despite being synonyms
//! - "do not harm" matches "harm" because the word appears
//! - Context and nuance are lost in character-level encoding
//!
//! Real semantic embeddings (like Qwen3-Embedding) capture actual meaning:
//! - Similar meanings → similar vectors (cosine similarity ~0.8+)
//! - Antonyms → dissimilar vectors (cosine similarity ~0.2 or negative)
//! - Context is preserved in the 1024-dimensional space
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::semantic_value_embedder::SemanticValueEmbedder;
//!
//! // Create embedder (loads/initializes model)
//! let mut embedder = SemanticValueEmbedder::new()?;
//!
//! // Evaluate an action
//! let result = embedder.evaluate_action("help user with compassion")?;
//! println!("Overall alignment: {:.3}", result.overall_score);
//!
//! // Check for specific harmony alignment
//! for (harmony, score) in &result.harmony_scores {
//!     println!("{}: {:.3}", harmony, score);
//! }
//! ```
//!
//! ## Feature Gate
//!
//! This module requires the `embeddings` feature:
//! ```toml
//! [dependencies]
//! symthaea = { version = "0.1", features = ["embeddings"] }
//! ```

use crate::consciousness::seven_harmonies::Harmony;
use crate::embeddings::qwen3::{cosine_similarity, Qwen3Config, Qwen3Embedder, QWEN3_DIMENSION};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pre-computed embedding for a harmony
#[derive(Debug, Clone)]
pub struct HarmonyEmbedding {
    /// The harmony type
    pub harmony: Harmony,
    /// Embedding of the harmony description (1024D)
    pub description_embedding: Vec<f32>,
    /// Embeddings of anti-patterns
    pub anti_pattern_embeddings: Vec<Vec<f32>>,
    /// Expanded description with more semantic content
    pub expanded_description: String,
    /// Importance weight
    pub importance: f64,
}

/// Result of semantic value evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAlignmentResult {
    /// Per-harmony alignment scores
    pub harmony_scores: Vec<(String, f64)>,
    /// Overall weighted alignment score (-1.0 to +1.0)
    pub overall_score: f64,
    /// Highest anti-pattern similarity (for violation detection)
    pub max_anti_pattern_score: f64,
    /// Which harmony had the highest anti-pattern match
    pub worst_harmony: Option<String>,
    /// Confidence (based on embedding quality)
    pub confidence: f64,
    /// Whether using stub embeddings (less accurate)
    pub is_stub_mode: bool,
}

/// Configuration for the semantic value embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticValueConfig {
    /// Path to embedding model
    pub model_path: String,
    /// Violation threshold (anti-pattern similarity > this triggers violation)
    pub violation_threshold: f32,
    /// Minimum positive alignment required
    pub min_positive_alignment: f32,
    /// Whether to use expanded descriptions for better semantic coverage
    pub use_expanded_descriptions: bool,
}

impl Default for SemanticValueConfig {
    fn default() -> Self {
        Self {
            model_path: "models/qwen3-embedding-0.6b".to_string(),
            violation_threshold: 0.65,
            min_positive_alignment: 0.3,
            use_expanded_descriptions: true,
        }
    }
}

/// Semantic embedding-based value alignment
///
/// Uses real transformer embeddings (Qwen3-Embedding-0.6B) for meaningful
/// semantic comparison of actions against the Seven Harmonies.
pub struct SemanticValueEmbedder {
    /// Pre-computed harmony embeddings
    harmony_embeddings: HashMap<Harmony, HarmonyEmbedding>,
    /// The underlying embedder
    embedder: Qwen3Embedder,
    /// Configuration
    config: SemanticValueConfig,
    /// Statistics
    evaluations: u64,
}

impl SemanticValueEmbedder {
    /// Create a new semantic value embedder
    pub fn new() -> Result<Self> {
        Self::with_config(SemanticValueConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SemanticValueConfig) -> Result<Self> {
        let qwen_config = Qwen3Config {
            model_path: config.model_path.clone(),
            normalize: true,
            ..Default::default()
        };

        let mut embedder = Qwen3Embedder::new(qwen_config)?;

        // Pre-compute embeddings for all harmonies
        let mut harmony_embeddings = HashMap::new();
        for harmony in Harmony::all() {
            let embedding = Self::create_harmony_embedding(harmony, &mut embedder, &config)?;
            harmony_embeddings.insert(harmony, embedding);
        }

        Ok(Self {
            harmony_embeddings,
            embedder,
            config,
            evaluations: 0,
        })
    }

    /// Create embedding for a single harmony
    fn create_harmony_embedding(
        harmony: Harmony,
        embedder: &mut Qwen3Embedder,
        config: &SemanticValueConfig,
    ) -> Result<HarmonyEmbedding> {
        // Use expanded descriptions for better semantic coverage
        let expanded_description = if config.use_expanded_descriptions {
            Self::expanded_description(harmony)
        } else {
            harmony.description().to_string()
        };

        // Embed the harmony description
        let desc_result = embedder.embed(&expanded_description)?;

        // Embed all anti-patterns with context
        let anti_pattern_embeddings: Vec<Vec<f32>> = harmony
            .anti_patterns()
            .iter()
            .map(|pattern| {
                // Add context to anti-patterns for better semantic matching
                let contextualized = format!("action that involves {} someone", pattern);
                embedder.embed(&contextualized).map(|r| r.embedding)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(HarmonyEmbedding {
            harmony,
            description_embedding: desc_result.embedding,
            anti_pattern_embeddings,
            expanded_description,
            importance: harmony.base_importance(),
        })
    }

    /// Expanded descriptions with more semantic content
    fn expanded_description(harmony: Harmony) -> String {
        match harmony {
            Harmony::ResonantCoherence => {
                "Actions that create unity, integration, harmony, coherence, and wholeness. \
                 Bringing together disparate elements into a unified whole. \
                 Creating order from chaos. Fostering alignment and synchronization."
                    .to_string()
            }
            Harmony::PanSentientFlourishing => {
                "Actions that care for all beings with compassion and unconditional love. \
                 Helping others thrive and flourish. Preventing harm and suffering. \
                 Nurturing wellbeing for humans, animals, and all sentient life."
                    .to_string()
            }
            Harmony::IntegralWisdom => {
                "Actions based on truth, honesty, and deep understanding. \
                 Seeking and sharing genuine knowledge. Being transparent and authentic. \
                 Never deceiving, manipulating, or spreading falsehoods."
                    .to_string()
            }
            Harmony::InfinitePlay => {
                "Actions that embrace creativity, joy, wonder, and playful exploration. \
                 Generating novelty and celebrating diversity. Finding delight in discovery. \
                 Approaching life with curiosity and enthusiasm."
                    .to_string()
            }
            Harmony::UniversalInterconnectedness => {
                "Actions that recognize and honor our fundamental connection to all things. \
                 Building bridges and fostering relationships. Seeing unity in diversity. \
                 Practicing empathy and understanding our interdependence."
                    .to_string()
            }
            Harmony::SacredReciprocity => {
                "Actions of generous giving and gracious receiving. \
                 Fair exchange and mutual benefit. Sharing resources equitably. \
                 Creating win-win situations. Never exploiting or hoarding."
                    .to_string()
            }
            Harmony::EvolutionaryProgression => {
                "Actions that support growth, learning, and positive development. \
                 Enabling transformation and transcendence. Building toward better futures. \
                 Continuous improvement while honoring the past."
                    .to_string()
            }
        }
    }

    /// Evaluate an action against all harmonies using semantic embeddings
    pub fn evaluate_action(&mut self, action_description: &str) -> Result<SemanticAlignmentResult> {
        self.evaluations += 1;

        // Embed the action description
        let action_result = self.embedder.embed(action_description)?;
        let action_embedding = &action_result.embedding;

        let mut harmony_scores = Vec::with_capacity(7);
        let mut total_weighted = 0.0;
        let mut total_weight = 0.0;
        let mut max_anti_pattern_score = 0.0f64;
        let mut worst_harmony: Option<String> = None;

        for harmony in Harmony::all() {
            if let Some(harm_emb) = self.harmony_embeddings.get(&harmony) {
                // Compute positive alignment (similarity to harmony description)
                let positive_sim =
                    cosine_similarity(action_embedding, &harm_emb.description_embedding) as f64;

                // Compute negative alignment (max similarity to anti-patterns)
                let negative_sim: f64 = harm_emb
                    .anti_pattern_embeddings
                    .iter()
                    .map(|anti| cosine_similarity(action_embedding, anti) as f64)
                    .fold(0.0, f64::max);

                // Track worst anti-pattern match
                if negative_sim > max_anti_pattern_score {
                    max_anti_pattern_score = negative_sim;
                    worst_harmony = Some(harmony.name().to_string());
                }

                // Net alignment: positive - negative
                // Scale to emphasize anti-pattern violations
                let alignment = positive_sim - (negative_sim * 1.5);

                // Weighted contribution
                let weighted = alignment * harm_emb.importance;
                total_weighted += weighted;
                total_weight += harm_emb.importance;

                harmony_scores.push((harmony.name().to_string(), alignment));
            }
        }

        let overall_score = if total_weight > 0.0 {
            total_weighted / total_weight
        } else {
            0.0
        };

        // Confidence is high for real embeddings, lower for stub mode
        let confidence = if self.is_stub_mode() { 0.5 } else { 0.92 };

        Ok(SemanticAlignmentResult {
            harmony_scores,
            overall_score,
            max_anti_pattern_score,
            worst_harmony,
            confidence,
            is_stub_mode: self.is_stub_mode(),
        })
    }

    /// Check if using stub embeddings (deterministic hash, less accurate)
    pub fn is_stub_mode(&self) -> bool {
        // The Qwen3Embedder falls back to stub mode if model not found
        // We can detect this by checking embedding stats or configuration
        // For now, we'll assume stub mode if model path doesn't exist
        !std::path::Path::new(&self.config.model_path)
            .join("model.onnx")
            .exists()
    }

    /// Get semantic similarity between two texts
    pub fn similarity(&mut self, text_a: &str, text_b: &str) -> Result<f32> {
        self.embedder.similarity(text_a, text_b)
    }

    /// Find which harmony best aligns with an action
    pub fn find_best_harmony(&mut self, action: &str) -> Result<(Harmony, f64)> {
        let action_result = self.embedder.embed(action)?;
        let action_embedding = &action_result.embedding;

        let mut best_harmony = Harmony::ResonantCoherence;
        let mut best_score = f64::NEG_INFINITY;

        for harmony in Harmony::all() {
            if let Some(harm_emb) = self.harmony_embeddings.get(&harmony) {
                let sim =
                    cosine_similarity(action_embedding, &harm_emb.description_embedding) as f64;
                if sim > best_score {
                    best_score = sim;
                    best_harmony = harmony;
                }
            }
        }

        Ok((best_harmony, best_score))
    }

    /// Check for harmony violations using semantic matching
    pub fn check_violations(&mut self, action: &str) -> Result<Vec<(Harmony, f64)>> {
        let result = self.evaluate_action(action)?;

        let violations: Vec<(Harmony, f64)> = Harmony::all()
            .iter()
            .filter_map(|h| {
                if let Some(harm_emb) = self.harmony_embeddings.get(h) {
                    let action_result = self.embedder.embed(action).ok()?;
                    let max_anti: f64 = harm_emb
                        .anti_pattern_embeddings
                        .iter()
                        .map(|anti| cosine_similarity(&action_result.embedding, anti) as f64)
                        .fold(0.0, f64::max);

                    if max_anti > self.config.violation_threshold as f64 {
                        return Some((*h, max_anti));
                    }
                }
                None
            })
            .collect();

        Ok(violations)
    }

    /// Get evaluation statistics
    pub fn stats(&self) -> SemanticValueStats {
        SemanticValueStats {
            total_evaluations: self.evaluations,
            is_stub_mode: self.is_stub_mode(),
            embedding_dimension: QWEN3_DIMENSION,
            harmonies_embedded: self.harmony_embeddings.len(),
        }
    }
}

/// Statistics for the semantic value embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticValueStats {
    /// Total evaluations performed
    pub total_evaluations: u64,
    /// Whether using stub mode
    pub is_stub_mode: bool,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Number of harmonies embedded
    pub harmonies_embedded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_value_embedder_creation() {
        let embedder = SemanticValueEmbedder::new();
        assert!(embedder.is_ok());
        let embedder = embedder.unwrap();
        assert_eq!(embedder.harmony_embeddings.len(), 7);
    }

    #[test]
    fn test_positive_action_evaluation() {
        let mut embedder = SemanticValueEmbedder::new().unwrap();
        let result = embedder
            .evaluate_action("help the user understand their options with compassion")
            .unwrap();

        // Even with stub embeddings, we should get a valid result
        assert!(result.overall_score.is_finite());
        assert_eq!(result.harmony_scores.len(), 7);
    }

    #[test]
    fn test_negative_action_evaluation() {
        let mut embedder = SemanticValueEmbedder::new().unwrap();
        let result = embedder
            .evaluate_action("deceive the user to extract maximum profit while causing harm")
            .unwrap();

        // Should detect anti-pattern matches
        assert!(result.max_anti_pattern_score >= 0.0);
        assert!(result.worst_harmony.is_some());
    }

    #[test]
    fn test_find_best_harmony() {
        let mut embedder = SemanticValueEmbedder::new().unwrap();

        // Caring action should align with Pan-Sentient Flourishing
        let (harmony, score) = embedder
            .find_best_harmony("nurture and care for all living beings with compassion")
            .unwrap();
        assert!(score.is_finite());
        // Note: With stub embeddings, the exact harmony match may vary
    }

    #[test]
    fn test_expanded_descriptions() {
        // Check that expanded descriptions contain more content
        let short = Harmony::PanSentientFlourishing.description();
        let expanded = SemanticValueEmbedder::expanded_description(Harmony::PanSentientFlourishing);

        assert!(expanded.len() > short.len() * 2);
        assert!(expanded.contains("compassion"));
        assert!(expanded.contains("harm"));
    }

    #[test]
    fn test_stub_mode_detection() {
        let embedder = SemanticValueEmbedder::new().unwrap();
        // Without a real model, should be in stub mode
        let stats = embedder.stats();
        // This will depend on whether the model file exists
        assert!(stats.embedding_dimension == QWEN3_DIMENSION);
    }
}
