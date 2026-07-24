// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generalized Birkhoff aesthetic measure: M = O / C.
//!
//! Birkhoff (1933) proposed that aesthetic pleasure is proportional to order
//! divided by complexity. This module provides a modality-independent
//! implementation that works with abstract feature vectors.
//!
//! The canvas-specific Birkhoff metric in `symthaea-canvas` can be expressed
//! as a specialization of this generalized measure.

use crate::AestheticScore;

/// Abstract features for Birkhoff evaluation.
///
/// Each creative modality extracts these from its artifact type.
#[derive(Debug, Clone)]
pub struct BirkhoffFeatures {
    /// Symmetry measure (0.0 = asymmetric, 1.0 = perfectly symmetric).
    /// For visual: coefficient of variation of spatial distribution.
    /// For music: rhythmic regularity and melodic palindromes.
    /// For poetry: meter regularity and rhyme scheme adherence.
    pub symmetry: f32,

    /// Harmony balance (0.0 = none, 1.0 = all harmonies active).
    /// Proportion of the Eight Harmonies that are active (> 0.3).
    pub harmony_balance: f32,

    /// Consciousness coupling (0.0 = disconnected, 1.0 = fully conscious).
    /// From the Psi level at the time of creation.
    pub consciousness_coupling: f32,

    /// Structural complexity (0.0 = trivial, 1.0 = maximally complex).
    /// Logarithmic element count, scaled to [0, 1].
    pub structural_complexity: f32,

    /// Topological complexity from Betti numbers.
    pub topological_complexity: f32,

    /// Diversity of distinct elements (color variety, interval variety, word variety).
    pub diversity: f32,

    /// Eight Harmony activations at time of creation.
    pub harmony_activations: [f32; 8],
}

impl BirkhoffFeatures {
    /// Compute order: weighted combination of symmetry, harmony balance,
    /// and consciousness coupling.
    ///
    /// Weights: 0.4 × symmetry + 0.4 × harmony_balance + 0.2 × consciousness_coupling
    /// (matching the original canvas implementation).
    pub fn order(&self) -> f32 {
        0.4 * finite_unit(self.symmetry)
            + 0.4 * finite_unit(self.harmony_balance)
            + 0.2 * finite_unit(self.consciousness_coupling)
    }

    /// Compute complexity: weighted combination of structural, topological,
    /// and diversity components.
    ///
    /// Returns a minimum of 0.01 to avoid division by zero.
    pub fn complexity(&self) -> f32 {
        (0.5 * finite_unit(self.structural_complexity)
            + 0.3 * finite_unit(self.topological_complexity)
            + 0.2 * finite_unit(self.diversity))
        .clamp(0.01, 1.0)
    }

    /// Compute the unbounded classical Birkhoff ratio M = O / C.
    ///
    /// This is exposed for diagnostics and comparison with historical uses of
    /// the measure. Consumers should generally prefer [`Self::birkhoff`],
    /// which calibrates the ratio without saturating half of the input space.
    pub fn birkhoff_raw_ratio(&self) -> f32 {
        self.order().max(0.0) / self.complexity()
    }

    /// Compute a calibrated Birkhoff measure in [0, 1].
    ///
    /// A direct `clamp(order / complexity, 0, 1)` collapses every ratio above
    /// one to the same perfect score. We instead map the log-ratio through a
    /// logistic curve. A ratio of one maps to 0.5, larger ratios approach one,
    /// and smaller ratios approach zero while preserving rank information.
    pub fn birkhoff(&self) -> f32 {
        let order = self.order().max(0.0);
        if order <= f32::EPSILON {
            return 0.0;
        }

        let log_ratio = (order / self.complexity()).ln();
        1.0 / (1.0 + (-1.5 * log_ratio).exp())
    }

    /// Compute a full `AestheticScore` from these features.
    ///
    /// The `surprise` field is left at 0.0 — it must be filled by the
    /// `AestheticTracker` which compares against the EMA.
    pub fn to_score(&self) -> AestheticScore {
        let order = self.order();
        let complexity = self.complexity();
        let birkhoff = self.birkhoff();

        // Harmony: average activation of the Eight Harmonies
        let harmony_mean: f32 = self
            .harmony_activations
            .iter()
            .map(|&activation| finite_unit(activation))
            .sum::<f32>()
            / 8.0;

        let mut score = AestheticScore {
            order,
            complexity,
            surprise: 0.0, // set externally
            harmony: harmony_mean,
            birkhoff,
            composite: 0.0,
        };
        score.compute_composite();
        score
    }
}

/// Extract Birkhoff features from harmony activations and basic metrics.
///
/// This is the common extraction path for all modalities. Each modality
/// provides its own `structural_complexity`, `topological_complexity`,
/// and `diversity` values.
pub fn extract_common_features(
    harmony_activations: &[f32; 8],
    consciousness_level: f32,
    structural_complexity: f32,
    topological_complexity: f32,
    diversity: f32,
) -> BirkhoffFeatures {
    // Symmetry: how evenly distributed harmony activations are
    let sanitized: [f32; 8] = std::array::from_fn(|i| finite_unit(harmony_activations[i]));
    let mean: f32 = sanitized.iter().sum::<f32>() / 8.0;
    let symmetry = if mean < 0.01 {
        0.0
    } else {
        let variance: f32 = sanitized.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / 8.0;
        let cv = variance.sqrt() / mean;
        (1.0 - cv).clamp(0.0, 1.0)
    };

    // Harmony balance: fraction of harmonies above 0.3 threshold
    let active = sanitized.iter().filter(|&&r| r > 0.3).count();
    let harmony_balance = (active as f32) / 8.0;

    BirkhoffFeatures {
        symmetry,
        harmony_balance,
        consciousness_coupling: finite_unit(consciousness_level),
        structural_complexity: finite_unit(structural_complexity),
        topological_complexity: finite_unit(topological_complexity),
        diversity: finite_unit(diversity),
        harmony_activations: sanitized,
    }
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(symmetry: f32, structural: f32) -> BirkhoffFeatures {
        BirkhoffFeatures {
            symmetry,
            harmony_balance: 0.5,
            consciousness_coupling: 0.7,
            structural_complexity: structural,
            topological_complexity: 0.3,
            diversity: 0.5,
            harmony_activations: [0.5; 8],
        }
    }

    #[test]
    fn birkhoff_in_range() {
        let features = make_features(0.8, 0.5);
        let b = features.birkhoff();
        assert!(b >= 0.0 && b <= 1.0, "birkhoff = {b}");
    }

    #[test]
    fn more_symmetric_higher_birkhoff() {
        let sym = make_features(0.9, 0.5);
        let asym = make_features(0.2, 0.5);
        assert!(
            sym.birkhoff() > asym.birkhoff(),
            "symmetric {} should > asymmetric {}",
            sym.birkhoff(),
            asym.birkhoff()
        );
    }

    #[test]
    fn high_complexity_lowers_birkhoff() {
        let simple = make_features(0.7, 0.2);
        let complex = make_features(0.7, 0.9);
        assert!(
            simple.birkhoff() > complex.birkhoff(),
            "simple {} should > complex {}",
            simple.birkhoff(),
            complex.birkhoff()
        );
    }

    #[test]
    fn to_score_produces_valid_composite() {
        let features = make_features(0.6, 0.5);
        let score = features.to_score();
        assert!(score.composite >= 0.0 && score.composite <= 1.0);
        assert!(score.birkhoff >= 0.0 && score.birkhoff <= 1.0);
        assert!(score.harmony >= 0.0 && score.harmony <= 1.0);
    }

    #[test]
    fn extract_common_symmetry() {
        // Perfectly uniform harmonies → high symmetry
        let uniform = extract_common_features(&[0.8; 8], 0.7, 0.5, 0.3, 0.5);
        // One spike → low symmetry
        let spiked = extract_common_features(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.7,
            0.5,
            0.3,
            0.5,
        );
        assert!(
            uniform.symmetry > spiked.symmetry,
            "uniform {} > spiked {}",
            uniform.symmetry,
            spiked.symmetry
        );
    }

    #[test]
    fn extract_common_harmony_balance() {
        let all_active = extract_common_features(&[0.8; 8], 0.5, 0.5, 0.3, 0.5);
        let none_active = extract_common_features(&[0.1; 8], 0.5, 0.5, 0.3, 0.5);
        assert_eq!(all_active.harmony_balance, 1.0);
        assert_eq!(none_active.harmony_balance, 0.0);
    }

    #[test]
    fn zero_mean_symmetry() {
        let features = extract_common_features(&[0.0; 8], 0.5, 0.5, 0.3, 0.5);
        assert_eq!(features.symmetry, 0.0);
    }

    #[test]
    fn calibrated_birkhoff_does_not_hard_saturate() {
        let features = BirkhoffFeatures {
            symmetry: 1.0,
            harmony_balance: 1.0,
            consciousness_coupling: 1.0,
            structural_complexity: 0.1,
            topological_complexity: 0.1,
            diversity: 0.1,
            harmony_activations: [1.0; 8],
        };
        assert!(features.birkhoff_raw_ratio() > 1.0);
        assert!(features.birkhoff() < 1.0);
        assert!(features.birkhoff() > 0.5);
    }

    #[test]
    fn complexity_never_zero() {
        let features = BirkhoffFeatures {
            symmetry: 0.5,
            harmony_balance: 0.5,
            consciousness_coupling: 0.5,
            structural_complexity: 0.0,
            topological_complexity: 0.0,
            diversity: 0.0,
            harmony_activations: [0.5; 8],
        };
        assert!(features.complexity() >= 0.01);
    }

    #[test]
    fn non_finite_features_do_not_poison_scores() {
        let features = BirkhoffFeatures {
            symmetry: f32::NAN,
            harmony_balance: f32::INFINITY,
            consciousness_coupling: 0.5,
            structural_complexity: f32::NAN,
            topological_complexity: 0.5,
            diversity: 0.5,
            harmony_activations: [f32::NAN; 8],
        };
        let score = features.to_score();
        assert!(score.composite.is_finite());
        assert!(score.birkhoff.is_finite());
    }
}
