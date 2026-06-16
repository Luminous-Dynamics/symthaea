// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Semantic Vectors
//!
//! HDC vectors augmented with uncertainty and confidence metadata.
//! Enables principled handling of semantic ambiguity in consciousness processing.
//!
//! ## Motivation
//!
//! Raw HDC vectors don't capture "how sure are we about this encoding?"
//! An EpistemicSemanticVector wraps a PackedBipolar with:
//!
//! - **Confidence**: Overall certainty in the encoding (0.0-1.0)
//! - **Stability**: How stable is this vector to paraphrase? (optional)
//! - **Uncertainty Sources**: What contributes to uncertainty?
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::epistemic_vector::EpistemicSemanticVector;
//!
//! // Get epistemic vector from Neural Bridge v2
//! let esv = bridge.encode_epistemic("quantum superposition")?;
//!
//! // Check confidence before high-stakes decisions
//! if esv.confidence < 0.7 {
//!     println!("Low confidence ({:.2}): {}", esv.confidence, esv.explain_uncertainty());
//! }
//!
//! // Use in weighted combinations
//! let combined = weighted_bundle(&[
//!     (&esv1, esv1.confidence),
//!     (&esv2, esv2.confidence),
//! ]);
//! ```

use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

/// Sources of uncertainty in semantic encoding
#[derive(Debug, Clone)]
pub enum UncertaintySource {
    /// Projection was close to decision boundary
    ProjectionMargin {
        avg_margin: f32,
        low_margin_ratio: f32,
    },

    /// Embedding norm deviated from expected
    EmbeddingNorm { norm: f32, expected: f32 },

    /// Text was very short (< N tokens)
    ShortText {
        token_count: usize,
        min_recommended: usize,
    },

    /// Text was very long (truncated)
    Truncated {
        original_tokens: usize,
        max_tokens: usize,
    },

    /// Out-of-distribution detected (unusual embedding)
    OutOfDistribution { score: f32, threshold: f32 },

    /// Polysemy detected (multiple meanings)
    Polysemy {
        num_senses: usize,
        dominant_sense_weight: f32,
    },

    /// Model confidence was low
    ModelConfidence { score: f32 },

    /// Paraphrase instability detected
    ParaphraseInstability { variation_score: f32 },

    /// Custom uncertainty source
    Custom { name: String, value: f32 },
}

impl UncertaintySource {
    /// Get a human-readable description of this uncertainty source
    pub fn describe(&self) -> String {
        match self {
            Self::ProjectionMargin {
                avg_margin,
                low_margin_ratio,
            } => {
                format!(
                    "Projection margin: {:.1}% of dimensions near decision boundary (avg margin: {:.3})",
                    low_margin_ratio * 100.0,
                    avg_margin
                )
            }
            Self::EmbeddingNorm { norm, expected } => {
                format!("Unusual embedding norm: {norm:.3} (expected ~{expected:.1})")
            }
            Self::ShortText {
                token_count,
                min_recommended,
            } => {
                format!("Short text: {token_count} tokens (recommend ≥{min_recommended})")
            }
            Self::Truncated {
                original_tokens,
                max_tokens,
            } => {
                format!("Text truncated: {original_tokens} → {max_tokens} tokens")
            }
            Self::OutOfDistribution { score, threshold } => {
                format!("Out-of-distribution: score {score:.3} > threshold {threshold:.3}")
            }
            Self::Polysemy {
                num_senses,
                dominant_sense_weight,
            } => {
                format!(
                    "Polysemous: {} senses, dominant={:.1}%",
                    num_senses,
                    dominant_sense_weight * 100.0
                )
            }
            Self::ModelConfidence { score } => {
                format!("Low model confidence: {:.1}%", score * 100.0)
            }
            Self::ParaphraseInstability { variation_score } => {
                format!(
                    "Paraphrase instability: {:.1}% variation",
                    variation_score * 100.0
                )
            }
            Self::Custom { name, value } => {
                format!("{name}: {value:.3}")
            }
        }
    }

    /// Get the uncertainty contribution (0.0-1.0)
    pub fn contribution(&self) -> f32 {
        match self {
            Self::ProjectionMargin {
                low_margin_ratio, ..
            } => *low_margin_ratio,
            Self::EmbeddingNorm { norm, expected } => ((norm - expected).abs() / expected).min(1.0),
            Self::ShortText {
                token_count,
                min_recommended,
            } => 1.0 - (*token_count as f32 / *min_recommended as f32).min(1.0),
            Self::Truncated {
                original_tokens,
                max_tokens,
            } => 1.0 - (*max_tokens as f32 / *original_tokens as f32),
            Self::OutOfDistribution { score, .. } => *score,
            Self::Polysemy {
                dominant_sense_weight,
                ..
            } => 1.0 - dominant_sense_weight,
            Self::ModelConfidence { score } => 1.0 - score,
            Self::ParaphraseInstability { variation_score } => *variation_score,
            Self::Custom { value, .. } => *value,
        }
    }
}

/// HDC vector with epistemic metadata
#[derive(Debug, Clone)]
pub struct EpistemicSemanticVector {
    /// The underlying HDC vector
    pub vector: PackedBipolar,

    /// Overall confidence in this encoding (0.0-1.0)
    pub confidence: f32,

    /// Stability score (how invariant to paraphrase)
    pub stability: Option<f32>,

    /// Sources contributing to uncertainty
    pub uncertainty_sources: Vec<UncertaintySource>,

    /// Original source text (for debugging)
    pub source_text: Option<String>,

    /// Encoding time in microseconds
    pub encode_time_us: u32,
}

impl EpistemicSemanticVector {
    /// Create from a packed bipolar vector with default confidence
    pub fn from_packed(vector: PackedBipolar) -> Self {
        Self {
            vector,
            confidence: 1.0,
            stability: None,
            uncertainty_sources: Vec::new(),
            source_text: None,
            encode_time_us: 0,
        }
    }

    /// Create with explicit confidence
    pub fn with_confidence(vector: PackedBipolar, confidence: f32) -> Self {
        Self {
            vector,
            confidence: confidence.clamp(0.0, 1.0),
            stability: None,
            uncertainty_sources: Vec::new(),
            source_text: None,
            encode_time_us: 0,
        }
    }

    /// Check if this is a high-confidence encoding
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Check if this is a low-confidence encoding
    pub fn is_low_confidence(&self) -> bool {
        self.confidence < 0.5
    }

    /// Get total uncertainty (1.0 - confidence)
    pub fn uncertainty(&self) -> f32 {
        1.0 - self.confidence
    }

    /// Human-readable explanation of uncertainty sources
    pub fn explain_uncertainty(&self) -> String {
        if self.uncertainty_sources.is_empty() {
            return "No specific uncertainty sources identified".to_string();
        }

        let descriptions: Vec<String> = self
            .uncertainty_sources
            .iter()
            .map(|s| format!("• {}", s.describe()))
            .collect();

        descriptions.join("\n")
    }

    /// Compute weighted similarity that accounts for confidence
    ///
    /// sim_weighted = sim_raw * sqrt(conf_a * conf_b)
    pub fn weighted_similarity(&self, other: &Self) -> f32 {
        let raw_sim = self.vector.xor_similarity(&other.vector);
        let confidence_factor = (self.confidence * other.confidence).sqrt();
        raw_sim * confidence_factor
    }

    /// Add an uncertainty source
    pub fn add_uncertainty(&mut self, source: UncertaintySource) {
        self.uncertainty_sources.push(source);

        // Recompute confidence as product of (1 - contributions)
        let mut conf = 1.0f32;
        for src in &self.uncertainty_sources {
            conf *= 1.0 - src.contribution().min(0.5); // Cap individual contribution
        }
        self.confidence = conf.max(0.0);
    }

    /// Get the most significant uncertainty source
    pub fn primary_uncertainty(&self) -> Option<&UncertaintySource> {
        self.uncertainty_sources
            .iter()
            .max_by(|a, b| a.contribution().total_cmp(&b.contribution()))
    }

    /// Create a builder for more complex construction
    pub fn builder() -> EpistemicSemanticVectorBuilder {
        EpistemicSemanticVectorBuilder::default()
    }
}

/// Builder for EpistemicSemanticVector
#[derive(Default)]
pub struct EpistemicSemanticVectorBuilder {
    vector: Option<PackedBipolar>,
    confidence: f32,
    stability: Option<f32>,
    uncertainty_sources: Vec<UncertaintySource>,
    source_text: Option<String>,
    encode_time_us: u32,
}

impl EpistemicSemanticVectorBuilder {
    pub fn new() -> Self {
        Self {
            confidence: 1.0,
            ..Default::default()
        }
    }

    pub fn vector(mut self, v: PackedBipolar) -> Self {
        self.vector = Some(v);
        self
    }

    pub fn confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    pub fn stability(mut self, s: f32) -> Self {
        self.stability = Some(s);
        self
    }

    pub fn add_uncertainty(mut self, source: UncertaintySource) -> Self {
        self.uncertainty_sources.push(source);
        self
    }

    pub fn source_text(mut self, text: &str) -> Self {
        self.source_text = Some(text.to_string());
        self
    }

    pub fn encode_time_us(mut self, us: u32) -> Self {
        self.encode_time_us = us;
        self
    }

    pub fn build(self) -> Option<EpistemicSemanticVector> {
        let vector = self.vector?;
        Some(EpistemicSemanticVector {
            vector,
            confidence: self.confidence,
            stability: self.stability,
            uncertainty_sources: self.uncertainty_sources,
            source_text: self.source_text,
            encode_time_us: self.encode_time_us,
        })
    }
}

/// Weighted bundle of epistemic vectors
///
/// Combines multiple vectors weighted by their confidence
pub fn weighted_bundle(vectors: &[(&EpistemicSemanticVector, f32)]) -> EpistemicSemanticVector {
    if vectors.is_empty() {
        return EpistemicSemanticVector::from_packed(PackedBipolar::default());
    }

    // Compute weighted sum in continuous space
    let mut sum = vec![0.0f32; HDC_DIMENSION];
    let mut total_weight = 0.0f32;
    let mut total_confidence = 0.0f32;

    for (esv, extra_weight) in vectors {
        let weight = esv.confidence * extra_weight;
        total_weight += weight;
        total_confidence += esv.confidence;

        let bipolar = esv.vector.to_bipolar();
        for (i, &b) in bipolar.iter().enumerate() {
            sum[i] += (b as f32) * weight;
        }
    }

    // Normalize and binarize
    if total_weight > 0.0 {
        for v in &mut sum {
            *v /= total_weight;
        }
    }

    let packed = PackedBipolar::from_continuous(&sum);
    let avg_confidence = total_confidence / vectors.len() as f32;

    EpistemicSemanticVector::with_confidence(packed, avg_confidence)
}

/// Consensus bundle that requires agreement
///
/// Only sets a bit if vectors agree above threshold
pub fn consensus_bundle(
    vectors: &[&EpistemicSemanticVector],
    agreement_threshold: f32,
) -> EpistemicSemanticVector {
    if vectors.is_empty() {
        return EpistemicSemanticVector::from_packed(PackedBipolar::default());
    }

    let _n = vectors.len() as f32; // Unused but documents intent
    let mut votes_positive = vec![0.0f32; HDC_DIMENSION];
    let mut total_confidence = 0.0f32;

    for esv in vectors {
        let bipolar = esv.vector.to_bipolar();
        for (i, &b) in bipolar.iter().enumerate() {
            if b == 1 {
                votes_positive[i] += esv.confidence;
            }
        }
        total_confidence += esv.confidence;
    }

    // Convert to consensus
    let mut result = vec![0i8; HDC_DIMENSION];
    let threshold = agreement_threshold * total_confidence;

    for i in 0..HDC_DIMENSION {
        if votes_positive[i] >= threshold {
            result[i] = 1;
        } else if (total_confidence - votes_positive[i]) >= threshold {
            result[i] = -1;
        } else {
            // No consensus - random tiebreak based on majority
            result[i] = if votes_positive[i] > total_confidence / 2.0 {
                1
            } else {
                -1
            };
        }
    }

    let packed = PackedBipolar::from_bipolar(&result);

    // Confidence is based on how many bits reached consensus
    let consensus_bits = votes_positive
        .iter()
        .filter(|&&v| v >= threshold || (total_confidence - v) >= threshold)
        .count();
    let consensus_ratio = consensus_bits as f32 / HDC_DIMENSION as f32;

    EpistemicSemanticVector::with_confidence(packed, consensus_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_contribution() {
        let source = UncertaintySource::ProjectionMargin {
            avg_margin: 0.5,
            low_margin_ratio: 0.2,
        };
        assert!((source.contribution() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_from_packed() {
        let packed = PackedBipolar::default();
        let esv = EpistemicSemanticVector::from_packed(packed);
        assert_eq!(esv.confidence, 1.0);
        assert!(esv.uncertainty_sources.is_empty());
    }

    #[test]
    fn test_add_uncertainty() {
        let packed = PackedBipolar::default();
        let mut esv = EpistemicSemanticVector::from_packed(packed);

        assert_eq!(esv.confidence, 1.0);

        esv.add_uncertainty(UncertaintySource::ProjectionMargin {
            avg_margin: 0.1,
            low_margin_ratio: 0.3,
        });

        assert!(esv.confidence < 1.0);
        assert_eq!(esv.uncertainty_sources.len(), 1);
    }

    #[test]
    fn test_is_high_low_confidence() {
        let packed = PackedBipolar::default();

        let high = EpistemicSemanticVector::with_confidence(packed.clone(), 0.9);
        assert!(high.is_high_confidence());
        assert!(!high.is_low_confidence());

        let low = EpistemicSemanticVector::with_confidence(packed.clone(), 0.3);
        assert!(!low.is_high_confidence());
        assert!(low.is_low_confidence());
    }

    #[test]
    fn test_builder() {
        let packed = PackedBipolar::default();
        let esv = EpistemicSemanticVector::builder()
            .vector(packed)
            .confidence(0.8)
            .source_text("test")
            .encode_time_us(1000)
            .build()
            .unwrap();

        assert_eq!(esv.confidence, 0.8);
        assert_eq!(esv.source_text, Some("test".to_string()));
        assert_eq!(esv.encode_time_us, 1000);
    }

    #[test]
    fn test_weighted_similarity() {
        // Create a non-empty vector for meaningful similarity
        let continuous: Vec<f32> = (0..HDC_DIMENSION)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let packed = PackedBipolar::from_continuous(&continuous);

        let high_conf = EpistemicSemanticVector::with_confidence(packed.clone(), 1.0);
        let low_conf = EpistemicSemanticVector::with_confidence(packed.clone(), 0.25);

        // Same vector, but weighted by confidence
        let sim1 = high_conf.weighted_similarity(&high_conf);
        let sim2 = high_conf.weighted_similarity(&low_conf);

        // sim2 should be lower due to confidence weighting
        // sim1 = 1.0 * sqrt(1.0 * 1.0) = 1.0
        // sim2 = 1.0 * sqrt(1.0 * 0.25) = 0.5
        assert!(sim1 > sim2, "Expected sim1 ({}) > sim2 ({})", sim1, sim2);
    }
}
