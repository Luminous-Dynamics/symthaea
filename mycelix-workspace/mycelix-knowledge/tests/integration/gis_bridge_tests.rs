// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration Tests for GIS v4.0 Bridge
//!
//! Tests the Graceful Ignorance System integration including:
//! - GIS classification (5-type ignorance taxonomy)
//! - Rashomon multi-perspective analysis
//! - Dark Spot DHT publishing and resolution
//! - Harmonic alignment assessment

use knowledge_bridge_integrity::*;

// =============================================================================
// GIS CLASSIFICATION TESTS
// =============================================================================

#[cfg(test)]
mod gis_classification_tests {
    use super::*;

    #[test]
    fn test_ignorance_type_symbols() {
        assert_eq!(IgnoranceType::Known.symbol(), "κ");
        assert_eq!(IgnoranceType::KnownUnknown.symbol(), "ι₁");
        assert_eq!(IgnoranceType::UnknownUnknown.symbol(), "ι₂");
        assert_eq!(IgnoranceType::StructuralUnknown.symbol(), "ι₃");
        assert_eq!(IgnoranceType::Impossible.symbol(), "ι∞");
    }

    #[test]
    fn test_ignorance_type_resolvability() {
        assert_eq!(IgnoranceType::Known.resolvability(), 1.0);
        assert_eq!(IgnoranceType::KnownUnknown.resolvability(), 0.7);
        assert_eq!(IgnoranceType::UnknownUnknown.resolvability(), 0.3);
        assert_eq!(IgnoranceType::StructuralUnknown.resolvability(), 0.1);
        assert_eq!(IgnoranceType::Impossible.resolvability(), 0.0);
    }

    #[test]
    fn test_uncertainty_3d_creation() {
        let uncertainty = Uncertainty3D::new(0.5, 0.3, 0.2);
        assert_eq!(uncertainty.epistemic, 0.5);
        assert_eq!(uncertainty.aleatoric, 0.3);
        assert_eq!(uncertainty.structural, 0.2);
    }

    #[test]
    fn test_uncertainty_3d_clamping() {
        let uncertainty = Uncertainty3D::new(1.5, -0.5, 2.0);
        assert_eq!(uncertainty.epistemic, 1.0); // clamped from 1.5
        assert_eq!(uncertainty.aleatoric, 0.0); // clamped from -0.5
        assert_eq!(uncertainty.structural, 1.0); // clamped from 2.0
    }

    #[test]
    fn test_uncertainty_3d_total() {
        let uncertainty = Uncertainty3D::new(0.4, 0.2, 0.2);
        // total = 0.5 * 0.4 + 0.25 * 0.2 + 0.25 * 0.2 = 0.2 + 0.05 + 0.05 = 0.3
        let total = uncertainty.total();
        assert!((total - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_uncertainty_3d_confidence() {
        let uncertainty = Uncertainty3D::new(0.4, 0.2, 0.2);
        let confidence = uncertainty.confidence();
        // confidence = 1.0 - total = 1.0 - 0.3 = 0.7
        assert!((confidence - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_query_domain_display() {
        assert_eq!(format!("{}", QueryDomain::Mathematics), "Mathematics");
        assert_eq!(format!("{}", QueryDomain::Environmental), "Environmental");
    }
}

// =============================================================================
// HARMONY TESTS
// =============================================================================

#[cfg(test)]
mod harmony_tests {
    use super::*;

    #[test]
    fn test_harmony_all() {
        let harmonies = Harmony::all();
        assert_eq!(harmonies.len(), 7);
        assert!(harmonies.contains(&Harmony::ResonantCoherence));
        assert!(harmonies.contains(&Harmony::PanSentientFlourishing));
        assert!(harmonies.contains(&Harmony::IntegralWisdom));
        assert!(harmonies.contains(&Harmony::InfinitePlay));
        assert!(harmonies.contains(&Harmony::UniversalInterconnectedness));
        assert!(harmonies.contains(&Harmony::SacredReciprocity));
        assert!(harmonies.contains(&Harmony::EvolutionaryProgression));
    }

    #[test]
    fn test_harmony_epistemic_modes() {
        assert_eq!(Harmony::ResonantCoherence.epistemic_mode(), "Integration-Knowing");
        assert_eq!(Harmony::PanSentientFlourishing.epistemic_mode(), "Care-Knowing");
        assert_eq!(Harmony::IntegralWisdom.epistemic_mode(), "Truth-Knowing");
        assert_eq!(Harmony::InfinitePlay.epistemic_mode(), "Play-Knowing");
        assert_eq!(Harmony::UniversalInterconnectedness.epistemic_mode(), "Web-Knowing");
        assert_eq!(Harmony::SacredReciprocity.epistemic_mode(), "Exchange-Knowing");
        assert_eq!(Harmony::EvolutionaryProgression.epistemic_mode(), "Growth-Knowing");
    }

    #[test]
    fn test_harmony_primary_questions() {
        let questions = Harmony::PanSentientFlourishing.primary_questions();
        assert!(!questions.is_empty());
        assert!(questions.iter().any(|q| q.contains("affected")));
    }

    #[test]
    fn test_harmony_display() {
        assert_eq!(format!("{}", Harmony::PanSentientFlourishing), "Pan-Sentient Flourishing");
        assert_eq!(format!("{}", Harmony::IntegralWisdom), "Integral Wisdom");
    }
}

// =============================================================================
// HARMONIC SCORES TESTS
// =============================================================================

#[cfg(test)]
mod harmonic_scores_tests {
    use super::*;

    #[test]
    fn test_harmonic_scores_new() {
        let scores = HarmonicScores::new();
        // All scores should default to 0.5
        assert_eq!(scores.resonant_coherence, 0.5);
        assert_eq!(scores.pan_sentient_flourishing, 0.5);
        assert_eq!(scores.integral_wisdom, 0.5);
    }

    #[test]
    fn test_harmonic_scores_get_set() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.9);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 0.9);
    }

    #[test]
    fn test_harmonic_scores_clamping() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 1.5);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 1.0); // clamped

        scores.set(Harmony::IntegralWisdom, -0.5);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 0.0); // clamped
    }

    #[test]
    fn test_harmonic_scores_weighted_average() {
        let mut scores = HarmonicScores::new();
        // Set all to 1.0
        for harmony in Harmony::all() {
            scores.set(harmony, 1.0);
        }
        let avg = scores.weighted_average();
        // With all 1.0, weighted average should be 1.0
        assert!((avg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_harmonic_scores_primary_harmony() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.9);
        scores.set(Harmony::PanSentientFlourishing, 0.7);
        assert_eq!(scores.primary_harmony(), Harmony::IntegralWisdom);
    }

    #[test]
    fn test_harmonic_scores_misalignments() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.2); // below 0.3 threshold
        scores.set(Harmony::InfinitePlay, 0.1);    // below 0.3 threshold

        let misaligned = scores.misalignments(0.3);
        assert!(misaligned.contains(&Harmony::IntegralWisdom));
        assert!(misaligned.contains(&Harmony::InfinitePlay));
    }
}

// =============================================================================
// HARMONIC PERSPECTIVE TESTS
// =============================================================================

#[cfg(test)]
mod perspective_tests {
    use super::*;

    #[test]
    fn test_harmonic_perspective_new() {
        let perspective = HarmonicPerspective::new(
            Harmony::IntegralWisdom,
            "Truth-seeking perspective on the matter"
        );
        assert_eq!(perspective.harmony, Harmony::IntegralWisdom);
        assert!(perspective.content.contains("Truth-seeking"));
        assert_eq!(perspective.confidence, 0.5);
        assert_eq!(perspective.relevance, 0.5);
    }
}

// =============================================================================
// DARK SPOT STATUS TESTS
// =============================================================================

#[cfg(test)]
mod dark_spot_tests {
    use super::*;

    #[test]
    fn test_dark_spot_status_values() {
        assert_ne!(DarkSpotStatus::Active, DarkSpotStatus::Resolved);
        assert_ne!(DarkSpotStatus::ResolutionInProgress, DarkSpotStatus::Expired);
        assert_ne!(DarkSpotStatus::Impossible, DarkSpotStatus::Active);
    }
}

// =============================================================================
// WORKFLOW INTEGRATION TESTS (Mock)
// =============================================================================

#[cfg(test)]
mod workflow_tests {
    use super::*;

    /// Test the full GIS classification → Dark Spot workflow
    #[test]
    fn test_gis_to_dark_spot_workflow() {
        // 1. Classify a query
        let ignorance_type = IgnoranceType::KnownUnknown;
        let uncertainty = Uncertainty3D::new(0.5, 0.3, 0.2);

        // 2. Calculate EIG
        let potential_value = match ignorance_type {
            IgnoranceType::Known => 0.9,
            IgnoranceType::KnownUnknown => 0.7,
            IgnoranceType::UnknownUnknown => 0.4,
            IgnoranceType::StructuralUnknown => 0.2,
            IgnoranceType::Impossible => 0.05,
        };
        let likelihood = 1.0 - uncertainty.epistemic;
        let eig = potential_value * likelihood;

        // 3. Check if EIG meets Dark Spot threshold (0.3)
        assert!(eig >= 0.3, "EIG {} should meet Dark Spot threshold", eig);

        // 4. Dark Spot would be published
        let status = DarkSpotStatus::Active;
        assert_eq!(status, DarkSpotStatus::Active);
    }

    /// Test Rashomon multi-perspective synthesis
    #[test]
    fn test_rashomon_synthesis_workflow() {
        // 1. Generate perspectives from multiple harmonies
        let mut perspectives = Vec::new();

        for harmony in &[Harmony::PanSentientFlourishing, Harmony::IntegralWisdom, Harmony::SacredReciprocity] {
            let mut perspective = HarmonicPerspective::new(
                *harmony,
                &format!("{} perspective on the situation", harmony)
            );
            perspective.relevance = 0.8;
            perspective.confidence = 0.7;
            perspectives.push(perspective);
        }

        // 2. Calculate synthesis confidence
        let avg_confidence: f64 = perspectives.iter().map(|p| p.confidence).sum::<f64>()
            / perspectives.len() as f64;
        let avg_relevance: f64 = perspectives.iter().map(|p| p.relevance).sum::<f64>()
            / perspectives.len() as f64;
        let synthesis_confidence = (avg_confidence * 0.6 + avg_relevance * 0.4).min(1.0);

        assert!(synthesis_confidence > 0.7, "Synthesis confidence should be high");
    }

    /// Test harmonic alignment assessment
    #[test]
    fn test_harmonic_alignment_workflow() {
        // 1. Create scores for a proposal
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::PanSentientFlourishing, 0.9); // High care alignment
        scores.set(Harmony::IntegralWisdom, 0.8);         // Good truth alignment
        scores.set(Harmony::SacredReciprocity, 0.7);      // Good reciprocity
        scores.set(Harmony::ResonantCoherence, 0.6);      // Moderate coherence

        // 2. Calculate overall alignment
        let overall = scores.weighted_average();
        assert!(overall > 0.6, "Overall alignment should be good");

        // 3. Check primary harmony
        let primary = scores.primary_harmony();
        assert_eq!(primary, Harmony::PanSentientFlourishing);

        // 4. Check for misalignments
        let misalignments = scores.misalignments(0.3);
        assert!(misalignments.is_empty(), "Should have no misalignments");
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_uncertainty() {
        let uncertainty = Uncertainty3D::new(0.0, 0.0, 0.0);
        assert_eq!(uncertainty.total(), 0.0);
        assert_eq!(uncertainty.confidence(), 1.0);
    }

    #[test]
    fn test_max_uncertainty() {
        let uncertainty = Uncertainty3D::new(1.0, 1.0, 1.0);
        // total = 0.5 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 = 1.0
        assert_eq!(uncertainty.total(), 1.0);
        assert_eq!(uncertainty.confidence(), 0.0);
    }

    #[test]
    fn test_all_harmonies_zero() {
        let mut scores = HarmonicScores::new();
        for harmony in Harmony::all() {
            scores.set(harmony, 0.0);
        }
        assert_eq!(scores.weighted_average(), 0.0);
    }

    #[test]
    fn test_misalignments_with_zero_threshold() {
        let scores = HarmonicScores::new(); // All 0.5
        let misalignments = scores.misalignments(0.0);
        assert!(misalignments.is_empty(), "No harmony should be below 0.0");
    }

    #[test]
    fn test_misalignments_with_high_threshold() {
        let scores = HarmonicScores::new(); // All 0.5
        let misalignments = scores.misalignments(0.6);
        // All should be misaligned since all are 0.5 < 0.6
        assert_eq!(misalignments.len(), 7);
    }
}
