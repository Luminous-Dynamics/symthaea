// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Style embedding: a continuous vector representing artistic identity.
//!
//! Computed from gallery history — the aesthetic scores and harmony activations
//! of past artworks form a statistical signature. This vector persists across
//! sessions, enabling the system to develop a recognizable artistic voice.
//!
//! The style embedding can condition future generation: if the style vector
//! says "minimalist with high surprise," new artwork should trend that way.
//!
//! # Dimensions (16D)
//!
//! 8 harmony activation means + 6 aesthetic score means + score_trend + novelty_preference

use crate::GalleryIndex;
use serde::{Deserialize, Serialize};

/// Dimensionality of the style embedding.
pub const STYLE_DIM: usize = 16;

/// A continuous style embedding derived from gallery history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleEmbedding {
    /// 16-dimensional style vector.
    pub vector: [f32; STYLE_DIM],
    /// Number of artworks this embedding was computed from.
    pub sample_count: usize,
    /// Confidence (0-1): higher with more samples.
    pub confidence: f32,
}

impl StyleEmbedding {
    /// Neutral style (no history — all dimensions at 0.5).
    pub fn neutral() -> Self {
        Self {
            vector: [0.5; STYLE_DIM],
            sample_count: 0,
            confidence: 0.0,
        }
    }

    /// Cosine similarity between two style embeddings.
    pub fn similarity(&self, other: &StyleEmbedding) -> f32 {
        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f32 = self.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mag_b: f32 = other.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if mag_a < 0.001 || mag_b < 0.001 {
            return 0.0;
        }
        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    /// Blend two style embeddings with a weight (0 = self, 1 = other).
    pub fn blend(&self, other: &StyleEmbedding, weight: f32) -> StyleEmbedding {
        let w = weight.clamp(0.0, 1.0);
        let mut vector = [0.0f32; STYLE_DIM];
        for i in 0..STYLE_DIM {
            vector[i] = self.vector[i] * (1.0 - w) + other.vector[i] * w;
        }
        StyleEmbedding {
            vector,
            sample_count: self.sample_count + other.sample_count,
            confidence: self.confidence * (1.0 - w) + other.confidence * w,
        }
    }
}

impl Default for StyleEmbedding {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Compute a style embedding from the gallery's history.
///
/// Uses the most recent `window` entries (or all if fewer).
/// The embedding captures:
/// - Dimensions 0-7: mean harmony activations across artworks
/// - Dimensions 8-13: mean aesthetic score dimensions (order, complexity, surprise, harmony, birkhoff, composite)
/// - Dimension 14: score trend (improving vs declining)
/// - Dimension 15: novelty preference (how much the system values surprise)
pub fn compute_style(index: &GalleryIndex, window: usize) -> StyleEmbedding {
    if index.entries.is_empty() {
        return StyleEmbedding::neutral();
    }

    let entries = if index.entries.len() > window {
        &index.entries[index.entries.len() - window..]
    } else {
        &index.entries
    };

    let n = entries.len() as f32;
    let mut vector = [0.0f32; STYLE_DIM];

    // Dimensions 0-7: mean harmony activations
    for entry in entries {
        for i in 0..8 {
            vector[i] += entry.harmony_activations[i] / n;
        }
    }

    // Dimensions 8-13: mean aesthetic score dimensions
    for entry in entries {
        let s = &entry.aesthetic_score;
        vector[8] += s.order / n;
        vector[9] += s.complexity / n;
        vector[10] += s.surprise / n;
        vector[11] += s.harmony / n;
        vector[12] += s.birkhoff / n;
        vector[13] += s.composite / n;
    }

    // Dimension 14: score trend (linear regression slope, normalized)
    if entries.len() >= 2 {
        let scores: Vec<f32> = entries
            .iter()
            .map(|e| e.aesthetic_score.composite)
            .collect();
        let slope = linear_slope(&scores);
        vector[14] = (slope * 10.0 + 0.5).clamp(0.0, 1.0); // normalize: 0.5 = flat
    } else {
        vector[14] = 0.5;
    }

    // Dimension 15: novelty preference (mean surprise relative to mean order)
    let mean_surprise = vector[10];
    let mean_order = vector[8];
    vector[15] = if mean_order + mean_surprise > 0.0 {
        mean_surprise / (mean_order + mean_surprise)
    } else {
        0.5
    };

    let confidence = (n / 50.0).clamp(0.0, 1.0); // full confidence at 50+ samples

    StyleEmbedding {
        vector,
        sample_count: entries.len(),
        confidence,
    }
}

/// Apply a style embedding to condition a CognitiveSnapshot for generation.
///
/// Nudges the snapshot's harmony activations and neuromodulators toward
/// the style's learned preferences. The `strength` parameter (0-1) controls
/// how much influence the style has vs the current state.
pub fn apply_style(
    harmony_activations: &mut [f32; 8],
    valence: &mut f32,
    arousal: &mut f32,
    style: &StyleEmbedding,
    strength: f32,
) {
    let s = strength.clamp(0.0, 1.0) * style.confidence;

    // Blend harmony activations toward style
    for i in 0..8 {
        harmony_activations[i] = harmony_activations[i] * (1.0 - s) + style.vector[i] * s;
    }

    // Novelty preference → arousal (high novelty pref → more exploratory)
    let novelty_pref = style.vector[15];
    *arousal = (*arousal * (1.0 - s * 0.3) + novelty_pref * s * 0.3).clamp(0.0, 1.0);

    // Score trend → valence (improving → positive, declining → negative)
    let trend = style.vector[14] - 0.5; // -0.5 to +0.5
    *valence = (*valence + trend * s * 0.2).clamp(-1.0, 1.0);
}

/// Simple linear regression slope.
fn linear_slope(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f32 = values.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f32;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean) * (x - x_mean);
    }
    if den.abs() < 0.0001 { 0.0 } else { num / den }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GalleryEntry;
    use symthaea_aesthetic::AestheticScore;

    fn mock_entry(cycle: u64, harmony_idx: usize, score: f32) -> GalleryEntry {
        let mut harmony = [0.3f32; 8];
        harmony[harmony_idx] = 0.9;
        GalleryEntry {
            id: uuid::Uuid::new_v4(),
            created_at_cycle: cycle,
            created_at: chrono::Utc::now(),
            modality: crate::ArtModality::Visual {
                filename: "test.svg".into(),
            },
            aesthetic_score: AestheticScore {
                order: score * 0.8,
                complexity: 0.5,
                surprise: score * 0.3,
                harmony: score * 0.7,
                birkhoff: score * 0.6,
                composite: score,
            },
            harmony_activations: harmony,
            tags: vec![],
            protected: false,
        }
    }

    #[test]
    fn neutral_style_is_half() {
        let s = StyleEmbedding::neutral();
        for &v in &s.vector {
            assert_eq!(v, 0.5);
        }
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn empty_gallery_neutral() {
        let index = GalleryIndex {
            entries: vec![],
            max_entries: 100,
        };
        let s = compute_style(&index, 50);
        assert_eq!(s.sample_count, 0);
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn style_from_gallery() {
        let index = GalleryIndex {
            entries: vec![
                mock_entry(1, 3, 0.6), // InfinitePlay dominant
                mock_entry(2, 3, 0.7),
                mock_entry(3, 3, 0.8),
            ],
            max_entries: 100,
        };
        let s = compute_style(&index, 50);
        assert_eq!(s.sample_count, 3);
        assert!(s.confidence > 0.0);
        // InfinitePlay (index 3) should be highest harmony dimension
        assert!(s.vector[3] > s.vector[0], "harmony 3 should dominate");
    }

    #[test]
    fn improving_trend_positive() {
        let index = GalleryIndex {
            entries: vec![
                mock_entry(1, 0, 0.3),
                mock_entry(2, 0, 0.5),
                mock_entry(3, 0, 0.7),
                mock_entry(4, 0, 0.9),
            ],
            max_entries: 100,
        };
        let s = compute_style(&index, 50);
        // Dimension 14 = score trend, should be > 0.5 (improving)
        assert!(s.vector[14] > 0.5, "improving trend: {}", s.vector[14]);
    }

    #[test]
    fn declining_trend_negative() {
        let index = GalleryIndex {
            entries: vec![
                mock_entry(1, 0, 0.9),
                mock_entry(2, 0, 0.7),
                mock_entry(3, 0, 0.5),
                mock_entry(4, 0, 0.3),
            ],
            max_entries: 100,
        };
        let s = compute_style(&index, 50);
        assert!(s.vector[14] < 0.5, "declining trend: {}", s.vector[14]);
    }

    #[test]
    fn similarity_identical() {
        let s = StyleEmbedding {
            vector: [
                0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.5, 0.4, 0.3, 0.6, 0.5, 0.7, 0.5, 0.4,
            ],
            sample_count: 10,
            confidence: 0.8,
        };
        let sim = s.similarity(&s);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "self-similarity should be ~1.0: {sim}"
        );
    }

    #[test]
    fn blend_weights() {
        let a = StyleEmbedding {
            vector: [0.0; STYLE_DIM],
            sample_count: 5,
            confidence: 0.5,
        };
        let b = StyleEmbedding {
            vector: [1.0; STYLE_DIM],
            sample_count: 5,
            confidence: 1.0,
        };
        let mid = a.blend(&b, 0.5);
        for &v in &mid.vector {
            assert!((v - 0.5).abs() < 0.01, "midpoint blend should be 0.5: {v}");
        }
    }

    #[test]
    fn apply_style_modifies_harmonies() {
        let s = StyleEmbedding {
            vector: [
                0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.8,
            ],
            sample_count: 50,
            confidence: 1.0,
        };
        let mut harmonies = [0.5f32; 8];
        let mut valence = 0.0f32;
        let mut arousal = 0.5f32;
        apply_style(&mut harmonies, &mut valence, &mut arousal, &s, 0.5);

        // Odd harmonies should have moved toward 0.9, even toward 0.1
        assert!(
            harmonies[0] > 0.5,
            "harmony 0 should increase: {}",
            harmonies[0]
        );
        assert!(
            harmonies[1] < 0.5,
            "harmony 1 should decrease: {}",
            harmonies[1]
        );
    }

    #[test]
    fn window_limits_entries() {
        let index = GalleryIndex {
            entries: (0..100)
                .map(|i| mock_entry(i, (i % 8) as usize, 0.5))
                .collect(),
            max_entries: 200,
        };
        let s = compute_style(&index, 10);
        assert_eq!(s.sample_count, 10);
    }

    #[test]
    fn linear_slope_positive() {
        let slope = linear_slope(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(slope > 0.9, "slope should be ~1.0: {slope}");
    }

    #[test]
    fn linear_slope_flat() {
        let slope = linear_slope(&[3.0, 3.0, 3.0, 3.0]);
        assert!(
            slope.abs() < 0.01,
            "flat data should have ~0 slope: {slope}"
        );
    }
}
