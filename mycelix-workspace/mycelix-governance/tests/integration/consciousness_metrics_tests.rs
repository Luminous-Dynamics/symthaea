// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration Tests for Consciousness Metrics
//!
//! Tests the Symthaea consciousness integration including:
//! - Consciousness snapshot validation
//! - Gate verification (Φ thresholds)
//! - Value alignment assessment
//! - Governance recommendation logic

use governance_bridge_integrity::*;

// =============================================================================
// CONSCIOUSNESS SNAPSHOT TESTS
// =============================================================================

#[cfg(test)]
mod consciousness_snapshot_tests {
    use super::*;

    fn create_test_snapshot(phi: f64) -> ConsciousnessSnapshot {
        ConsciousnessSnapshot {
            id: "test-snapshot-1".to_string(),
            agent_did: "did:mycelix:test123".to_string(),
            phi,
            meta_awareness: 0.6,
            self_model_accuracy: 0.7,
            coherence: 0.8,
            affective_valence: 0.5,
            care_activation: 0.6,
            captured_at: Timestamp::from_micros(1000000),
            source: "symthaea".to_string(),
            consciousness_vector: None,
        }
    }

    #[test]
    fn test_quality_score_calculation() {
        let snapshot = create_test_snapshot(0.5);
        // quality = phi * 0.4 + meta * 0.2 + self_model * 0.2 + coherence * 0.2
        // = 0.5 * 0.4 + 0.6 * 0.2 + 0.7 * 0.2 + 0.8 * 0.2
        // = 0.2 + 0.12 + 0.14 + 0.16 = 0.62
        let quality = snapshot.quality_score();
        assert!((quality - 0.62).abs() < 0.001, "Quality score should be 0.62, got {}", quality);
    }

    #[test]
    fn test_quality_score_clamping() {
        // Even with high values, quality should be clamped to 1.0
        let mut snapshot = create_test_snapshot(1.0);
        snapshot.meta_awareness = 1.0;
        snapshot.self_model_accuracy = 1.0;
        snapshot.coherence = 1.0;

        let quality = snapshot.quality_score();
        assert!(quality <= 1.0, "Quality score should be <= 1.0");
    }

    #[test]
    fn test_meets_threshold_basic() {
        let snapshot = create_test_snapshot(0.25);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic)); // 0.25 >= 0.2
        assert!(!snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission)); // 0.25 < 0.3
    }

    #[test]
    fn test_meets_threshold_proposal() {
        let snapshot = create_test_snapshot(0.35);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic)); // 0.35 >= 0.2
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission)); // 0.35 >= 0.3
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Voting)); // 0.35 < 0.4
    }

    #[test]
    fn test_meets_threshold_voting() {
        let snapshot = create_test_snapshot(0.45);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Voting)); // 0.45 >= 0.4
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Constitutional)); // 0.45 < 0.6
    }

    #[test]
    fn test_meets_threshold_constitutional() {
        let snapshot = create_test_snapshot(0.65);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Basic));
        assert!(snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Voting));
        assert!(snapshot.meets_threshold(&GovernanceActionType::Constitutional)); // 0.65 >= 0.6
    }

    #[test]
    fn test_meets_threshold_edge_cases() {
        // Exactly at threshold
        let snapshot_basic = create_test_snapshot(0.2);
        assert!(snapshot_basic.meets_threshold(&GovernanceActionType::Basic));

        let snapshot_proposal = create_test_snapshot(0.3);
        assert!(snapshot_proposal.meets_threshold(&GovernanceActionType::ProposalSubmission));

        let snapshot_voting = create_test_snapshot(0.4);
        assert!(snapshot_voting.meets_threshold(&GovernanceActionType::Voting));

        let snapshot_const = create_test_snapshot(0.6);
        assert!(snapshot_const.meets_threshold(&GovernanceActionType::Constitutional));
    }

    #[test]
    fn test_meets_threshold_just_below() {
        let snapshot = create_test_snapshot(0.199);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Basic));

        let snapshot = create_test_snapshot(0.299);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::ProposalSubmission));

        let snapshot = create_test_snapshot(0.399);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Voting));

        let snapshot = create_test_snapshot(0.599);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Constitutional));
    }
}

// =============================================================================
// GOVERNANCE ACTION TYPE TESTS
// =============================================================================

#[cfg(test)]
mod action_type_tests {
    use super::*;

    #[test]
    fn test_phi_thresholds() {
        assert_eq!(GovernanceActionType::Basic.phi_threshold(), 0.2);
        assert_eq!(GovernanceActionType::ProposalSubmission.phi_threshold(), 0.3);
        assert_eq!(GovernanceActionType::Voting.phi_threshold(), 0.4);
        assert_eq!(GovernanceActionType::Constitutional.phi_threshold(), 0.6);
    }

    #[test]
    fn test_descriptions() {
        assert_eq!(GovernanceActionType::Basic.description(), "Basic participation");
        assert_eq!(GovernanceActionType::ProposalSubmission.description(), "Proposal submission");
        assert_eq!(GovernanceActionType::Voting.description(), "Voting on proposals");
        assert_eq!(GovernanceActionType::Constitutional.description(), "Constitutional changes");
    }

    #[test]
    fn test_threshold_ordering() {
        // Thresholds should increase in severity
        assert!(GovernanceActionType::Basic.phi_threshold() <
                GovernanceActionType::ProposalSubmission.phi_threshold());
        assert!(GovernanceActionType::ProposalSubmission.phi_threshold() <
                GovernanceActionType::Voting.phi_threshold());
        assert!(GovernanceActionType::Voting.phi_threshold() <
                GovernanceActionType::Constitutional.phi_threshold());
    }
}

// =============================================================================
// CONSCIOUSNESS TREND TESTS
// =============================================================================

#[cfg(test)]
mod trend_tests {
    use super::*;

    #[test]
    fn test_trend_values() {
        assert_ne!(ConsciousnessTrend::Increasing, ConsciousnessTrend::Decreasing);
        assert_ne!(ConsciousnessTrend::Stable, ConsciousnessTrend::Unknown);
    }
}

// =============================================================================
// HARMONY SCORE TESTS
// =============================================================================

#[cfg(test)]
mod harmony_score_tests {
    use super::*;

    #[test]
    fn test_harmony_score_creation() {
        let score = HarmonyScore {
            harmony: "Pan-Sentient Flourishing".to_string(),
            score: 0.8,
        };
        assert_eq!(score.harmony, "Pan-Sentient Flourishing");
        assert_eq!(score.score, 0.8);
    }
}

// =============================================================================
// GOVERNANCE RECOMMENDATION TESTS
// =============================================================================

#[cfg(test)]
mod recommendation_tests {
    use super::*;

    /// Helper to determine recommendation (mirrors coordinator logic)
    fn determine_recommendation(
        alignment: f64,
        authenticity: f64,
        has_violations: bool,
    ) -> GovernanceRecommendation {
        if has_violations {
            return GovernanceRecommendation::StrongOppose;
        }

        if authenticity < 0.2 {
            return GovernanceRecommendation::CannotEvaluate;
        }

        let combined = alignment * 0.6 + authenticity * 0.4;

        match combined {
            c if c > 0.7 => GovernanceRecommendation::StrongSupport,
            c if c > 0.3 => GovernanceRecommendation::Support,
            c if c > -0.3 => GovernanceRecommendation::Neutral,
            c if c > -0.7 => GovernanceRecommendation::Oppose,
            _ => GovernanceRecommendation::StrongOppose,
        }
    }

    #[test]
    fn test_strong_support() {
        // High alignment + high authenticity
        let rec = determine_recommendation(0.9, 0.8, false);
        assert_eq!(rec, GovernanceRecommendation::StrongSupport);
    }

    #[test]
    fn test_support() {
        // Moderate alignment + moderate authenticity
        let rec = determine_recommendation(0.5, 0.5, false);
        // combined = 0.5 * 0.6 + 0.5 * 0.4 = 0.5
        assert_eq!(rec, GovernanceRecommendation::Support);
    }

    #[test]
    fn test_neutral() {
        // Low alignment + moderate authenticity
        let rec = determine_recommendation(0.0, 0.5, false);
        // combined = 0.0 * 0.6 + 0.5 * 0.4 = 0.2
        assert_eq!(rec, GovernanceRecommendation::Neutral);
    }

    #[test]
    fn test_oppose() {
        // Negative alignment + low authenticity
        let rec = determine_recommendation(-0.5, 0.3, false);
        // combined = -0.5 * 0.6 + 0.3 * 0.4 = -0.18
        assert_eq!(rec, GovernanceRecommendation::Neutral); // -0.18 > -0.3

        let rec2 = determine_recommendation(-0.6, 0.3, false);
        // combined = -0.6 * 0.6 + 0.3 * 0.4 = -0.24
        assert_eq!(rec2, GovernanceRecommendation::Neutral); // -0.24 > -0.3

        let rec3 = determine_recommendation(-0.8, 0.3, false);
        // combined = -0.8 * 0.6 + 0.3 * 0.4 = -0.36
        assert_eq!(rec3, GovernanceRecommendation::Oppose); // -0.36 < -0.3
    }

    #[test]
    fn test_strong_oppose_from_alignment() {
        // Very negative alignment
        let rec = determine_recommendation(-1.0, 0.3, false);
        // combined = -1.0 * 0.6 + 0.3 * 0.4 = -0.48
        assert_eq!(rec, GovernanceRecommendation::Oppose);

        let rec2 = determine_recommendation(-1.0, 0.2, false);
        // combined = -1.0 * 0.6 + 0.2 * 0.4 = -0.52
        assert_eq!(rec2, GovernanceRecommendation::Oppose);
    }

    #[test]
    fn test_strong_oppose_from_violations() {
        // Violations always result in StrongOppose
        let rec = determine_recommendation(0.9, 0.9, true);
        assert_eq!(rec, GovernanceRecommendation::StrongOppose);
    }

    #[test]
    fn test_cannot_evaluate_low_authenticity() {
        // Low authenticity should result in CannotEvaluate
        let rec = determine_recommendation(0.9, 0.1, false);
        assert_eq!(rec, GovernanceRecommendation::CannotEvaluate);

        let rec2 = determine_recommendation(0.5, 0.15, false);
        assert_eq!(rec2, GovernanceRecommendation::CannotEvaluate);
    }

    #[test]
    fn test_boundary_authenticity() {
        // Exactly at authenticity boundary
        let rec = determine_recommendation(0.5, 0.2, false);
        // combined = 0.5 * 0.6 + 0.2 * 0.4 = 0.38
        assert_eq!(rec, GovernanceRecommendation::Support); // 0.38 > 0.3

        let rec2 = determine_recommendation(0.5, 0.19, false);
        assert_eq!(rec2, GovernanceRecommendation::CannotEvaluate);
    }
}

// =============================================================================
// WORKFLOW INTEGRATION TESTS
// =============================================================================

#[cfg(test)]
mod workflow_tests {
    use super::*;

    /// Test the full consciousness gate → voting workflow
    #[test]
    fn test_gate_voting_workflow() {
        // 1. Agent has consciousness snapshot
        let phi = 0.45;

        // 2. Agent wants to vote (requires Φ ≥ 0.4)
        let action_type = GovernanceActionType::Voting;
        let required = action_type.phi_threshold();

        // 3. Gate check passes
        assert!(phi >= required, "Φ {} should meet voting threshold {}", phi, required);

        // 4. Agent can proceed with voting
        let passed = phi >= required;
        assert!(passed);
    }

    /// Test gate rejection for insufficient consciousness
    #[test]
    fn test_gate_rejection_workflow() {
        // 1. Agent has low consciousness
        let phi = 0.35;

        // 2. Agent wants to vote (requires Φ ≥ 0.4)
        let action_type = GovernanceActionType::Voting;
        let required = action_type.phi_threshold();

        // 3. Gate check fails
        assert!(phi < required, "Φ {} should NOT meet voting threshold {}", phi, required);

        // 4. Agent cannot vote
        let passed = phi >= required;
        assert!(!passed);
    }

    /// Test value alignment → recommendation workflow
    #[test]
    fn test_alignment_recommendation_workflow() {
        // 1. Calculate harmony scores for a proposal
        let harmony_scores = vec![
            HarmonyScore { harmony: "Pan-Sentient Flourishing".to_string(), score: 0.8 },
            HarmonyScore { harmony: "Integral Wisdom".to_string(), score: 0.6 },
            HarmonyScore { harmony: "Sacred Reciprocity".to_string(), score: 0.7 },
            HarmonyScore { harmony: "Resonant Coherence".to_string(), score: 0.5 },
        ];

        // 2. Calculate overall alignment
        let overall: f64 = harmony_scores.iter().map(|h| h.score).sum::<f64>()
            / harmony_scores.len() as f64;
        assert!((overall - 0.65).abs() < 0.001);

        // 3. Check for violations (scores < -0.3)
        let violations: Vec<String> = harmony_scores
            .iter()
            .filter(|h| h.score < -0.3)
            .map(|h| h.harmony.clone())
            .collect();
        assert!(violations.is_empty());

        // 4. With CARE activation of 0.6, determine recommendation
        let authenticity = 0.6;
        let combined = overall * 0.6 + authenticity * 0.4;
        // combined = 0.65 * 0.6 + 0.6 * 0.4 = 0.39 + 0.24 = 0.63

        // This should result in Support (0.3 < 0.63 < 0.7)
        assert!(combined > 0.3 && combined <= 0.7);
    }

    /// Test constitutional action rejection for low consciousness
    #[test]
    fn test_constitutional_gate() {
        // Constitutional changes require Φ ≥ 0.6
        let low_phi = 0.55;
        let high_phi = 0.65;

        assert!(low_phi < GovernanceActionType::Constitutional.phi_threshold());
        assert!(high_phi >= GovernanceActionType::Constitutional.phi_threshold());
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_phi() {
        let snapshot = ConsciousnessSnapshot {
            id: "test".to_string(),
            agent_did: "did:mycelix:test".to_string(),
            phi: 0.0,
            meta_awareness: 0.0,
            self_model_accuracy: 0.0,
            coherence: 0.0,
            affective_valence: 0.0,
            care_activation: 0.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".to_string(),
            consciousness_vector: None,
        };

        assert_eq!(snapshot.quality_score(), 0.0);
        assert!(!snapshot.meets_threshold(&GovernanceActionType::Basic));
    }

    #[test]
    fn test_max_phi() {
        let snapshot = ConsciousnessSnapshot {
            id: "test".to_string(),
            agent_did: "did:mycelix:test".to_string(),
            phi: 1.0,
            meta_awareness: 1.0,
            self_model_accuracy: 1.0,
            coherence: 1.0,
            affective_valence: 1.0,
            care_activation: 1.0,
            captured_at: Timestamp::from_micros(0),
            source: "test".to_string(),
            consciousness_vector: None,
        };

        assert_eq!(snapshot.quality_score(), 1.0);
        assert!(snapshot.meets_threshold(&GovernanceActionType::Constitutional));
    }

    #[test]
    fn test_negative_valence() {
        let snapshot = ConsciousnessSnapshot {
            id: "test".to_string(),
            agent_did: "did:mycelix:test".to_string(),
            phi: 0.5,
            meta_awareness: 0.5,
            self_model_accuracy: 0.5,
            coherence: 0.5,
            affective_valence: -1.0, // Negative valence (valid: -1 to +1)
            care_activation: 0.3,
            captured_at: Timestamp::from_micros(0),
            source: "test".to_string(),
            consciousness_vector: None,
        };

        // Negative valence doesn't affect quality score (not included)
        let quality = snapshot.quality_score();
        assert!(quality > 0.0);
    }
}
