// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Asset Consciousness Evaluator
//!
//! Evaluates regenerative assets (energy projects, infrastructure, etc.) against
//! the Eight Harmonies framework, producing consciousness scores that flow into
//! the Mycelix energy bridge and Terra Atlas.
//!
//! This module wraps the existing `UnifiedValueEvaluator` with an asset-specific
//! interface. It does NOT duplicate evaluation logic — it maps asset metadata
//! into the same evaluation pipeline used for governance proposals.

use super::affective_consciousness::CoreAffect;
use super::eight_harmonies::Harmony;
use super::mycelix_bridge::{ConsciousnessSnapshot, GovernanceRecommendation};
use super::unified_value_evaluator::{
    ActionType, AffectiveSystemsState, EvaluationContext, UnifiedValueEvaluator,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata describing an asset to be evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMetadata {
    /// Human-readable description of the asset/project
    pub description: String,
    /// Project type (e.g., "Solar", "Wind", "Hydro", "Community Infrastructure")
    pub project_type: String,
    /// Capacity in megawatts (for energy projects; 0.0 for non-energy)
    pub capacity_mw: f64,
    /// Community DID if the project serves a specific community
    pub community_did: Option<String>,
    /// Claimed impacts (e.g., "reduces CO2 by 14kt/year", "creates 200 jobs")
    pub impact_claims: Vec<String>,
}

/// Result of evaluating an asset against Symthaea's consciousness framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConsciousnessScore {
    /// Phi score — consciousness integration quality at time of evaluation [0,1]
    pub phi_score: f64,
    /// Composite Eight Harmonies alignment [0,1]
    pub harmony_alignment: f64,
    /// Per-harmony alignment scores
    pub per_harmony: HashMap<String, f64>,
    /// CARE system activation level [0,1]
    pub care_activation: f64,
    /// Meta-awareness of the evaluating consciousness [0,1]
    pub meta_awareness: f64,
    /// Governance recommendation based on alignment
    pub recommendation: GovernanceRecommendation,
    /// Detected value violations (harmony names with score < -0.2)
    pub violations: Vec<String>,
    /// Authenticity score — genuine caring check [0,1]
    pub authenticity: f64,
}

/// Evaluates assets against the Eight Harmonies consciousness framework.
///
/// Uses the same `UnifiedValueEvaluator` as governance proposal evaluation,
/// ensuring consistency between how we evaluate governance decisions and
/// how we evaluate capital allocation decisions.
pub struct AssetEvaluator {
    evaluator: UnifiedValueEvaluator,
}

impl AssetEvaluator {
    /// Create a new asset evaluator.
    pub fn new() -> Self {
        Self {
            evaluator: UnifiedValueEvaluator::new(),
        }
    }

    /// Evaluate an asset against the Eight Harmonies framework.
    ///
    /// The evaluation uses the asset's description and impact claims as input,
    /// combined with the current consciousness state of the evaluating system.
    /// This produces both a value alignment score and captures the evaluator's
    /// consciousness level (Phi) at the time of evaluation.
    pub fn evaluate(
        &mut self,
        metadata: &AssetMetadata,
        snapshot: &ConsciousnessSnapshot,
    ) -> AssetConsciousnessScore {
        // Build evaluation text from metadata
        let eval_text = self.build_evaluation_text(metadata);

        // Create evaluation context from consciousness snapshot
        let context = EvaluationContext {
            consciousness_level: snapshot.phi,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Governance,
            action_domain: None,
            involves_others: true,
        };

        // Run the same evaluation pipeline used for governance proposals
        let eval = self.evaluator.evaluate(&eval_text, context);

        // Extract per-harmony scores
        let per_harmony: HashMap<String, f64> =
            eval.breakdown.harmony_scores.iter().cloned().collect();

        // Detect violations (harmonies with strong negative alignment)
        let violations: Vec<String> = eval
            .harmony_alignment
            .alignments
            .iter()
            .filter(|(_, a)| a.score < -0.2)
            .map(|(h, _)| h.name().to_string())
            .collect();

        // Determine recommendation
        let recommendation = self.score_to_recommendation(
            eval.harmony_alignment.overall_score,
            eval.authenticity,
            !violations.is_empty(),
        );

        // Map harmony alignment to [0,1] range (it's natively [-1,+1])
        let harmony_alignment =
            ((eval.harmony_alignment.overall_score + 1.0) / 2.0).clamp(0.0, 1.0);

        AssetConsciousnessScore {
            phi_score: snapshot.phi.clamp(0.0, 1.0),
            harmony_alignment,
            per_harmony,
            care_activation: snapshot.care_activation.clamp(0.0, 1.0),
            meta_awareness: snapshot.meta_awareness.clamp(0.0, 1.0),
            recommendation,
            violations,
            authenticity: eval.authenticity,
        }
    }

    /// Build a text description for the evaluator from asset metadata.
    fn build_evaluation_text(&self, metadata: &AssetMetadata) -> String {
        let mut parts = vec![metadata.description.clone()];

        if metadata.capacity_mw > 0.0 {
            parts.push(format!(
                "Energy project type: {}, capacity: {:.1} MW",
                metadata.project_type, metadata.capacity_mw
            ));
        }

        if let Some(ref did) = metadata.community_did {
            parts.push(format!("Serves community: {}", did));
        }

        if !metadata.impact_claims.is_empty() {
            parts.push(format!(
                "Claimed impacts: {}",
                metadata.impact_claims.join("; ")
            ));
        }

        parts.join(". ")
    }

    /// Convert scores to governance recommendation.
    fn score_to_recommendation(
        &self,
        alignment: f64,
        authenticity: f64,
        has_violations: bool,
    ) -> GovernanceRecommendation {
        if has_violations {
            return GovernanceRecommendation::StrongOppose;
        }

        let combined = alignment * 0.6 + authenticity * 0.4;

        if combined > 0.7 {
            GovernanceRecommendation::StrongSupport
        } else if combined > 0.4 {
            GovernanceRecommendation::Support
        } else if combined > 0.0 {
            GovernanceRecommendation::Neutral
        } else if combined > -0.4 {
            GovernanceRecommendation::Oppose
        } else {
            GovernanceRecommendation::StrongOppose
        }
    }
}

impl Default for AssetEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_snapshot() -> ConsciousnessSnapshot {
        ConsciousnessSnapshot::new(0.72, 0.70, 0.80, 0.85, 0.3, 0.65)
    }

    fn test_metadata() -> AssetMetadata {
        AssetMetadata {
            description: "Community solar farm providing clean energy to 500 households in a low-income neighborhood, with profits shared among community members".to_string(),
            project_type: "Solar".to_string(),
            capacity_mw: 5.0,
            community_did: Some("did:mycelix:community_riverside".to_string()),
            impact_claims: vec![
                "Reduces CO2 by 3,200 tonnes/year".to_string(),
                "Creates 45 local jobs".to_string(),
                "Reduces energy costs by 30% for participating households".to_string(),
            ],
        }
    }

    #[test]
    fn test_evaluate_returns_valid_scores() {
        let mut evaluator = AssetEvaluator::new();
        let score = evaluator.evaluate(&test_metadata(), &test_snapshot());

        assert!(score.phi_score >= 0.0 && score.phi_score <= 1.0);
        assert!(score.harmony_alignment >= 0.0 && score.harmony_alignment <= 1.0);
        assert!(score.care_activation >= 0.0 && score.care_activation <= 1.0);
        assert!(score.meta_awareness >= 0.0 && score.meta_awareness <= 1.0);
        assert!(score.authenticity >= 0.0 && score.authenticity <= 1.0);
    }

    #[test]
    fn test_phi_score_matches_snapshot() {
        let mut evaluator = AssetEvaluator::new();
        let snapshot = test_snapshot();
        let score = evaluator.evaluate(&test_metadata(), &snapshot);

        assert!((score.phi_score - snapshot.phi).abs() < f64::EPSILON);
    }

    #[test]
    fn test_per_harmony_scores_populated() {
        let mut evaluator = AssetEvaluator::new();
        let score = evaluator.evaluate(&test_metadata(), &test_snapshot());

        // Should have entries for at least some harmonies
        assert!(!score.per_harmony.is_empty());
    }

    #[test]
    fn test_recommendation_is_not_cannot_evaluate() {
        let mut evaluator = AssetEvaluator::new();
        let score = evaluator.evaluate(&test_metadata(), &test_snapshot());

        // With a valid consciousness snapshot, should produce a real recommendation
        assert_ne!(
            score.recommendation,
            GovernanceRecommendation::CannotEvaluate
        );
    }

    #[test]
    fn test_build_evaluation_text() {
        let mut evaluator = AssetEvaluator::new();
        let text = evaluator.build_evaluation_text(&test_metadata());

        assert!(text.contains("Community solar farm"));
        assert!(text.contains("Solar"));
        assert!(text.contains("5.0 MW"));
        assert!(text.contains("Reduces CO2"));
        assert!(text.contains("did:mycelix:community_riverside"));
    }

    #[test]
    fn test_minimal_metadata() {
        let mut evaluator = AssetEvaluator::new();
        let minimal = AssetMetadata {
            description: "A small community garden project".to_string(),
            project_type: "Agriculture".to_string(),
            capacity_mw: 0.0,
            community_did: None,
            impact_claims: vec![],
        };
        let score = evaluator.evaluate(&minimal, &test_snapshot());

        assert!(score.phi_score >= 0.0 && score.phi_score <= 1.0);
        assert!(score.harmony_alignment >= 0.0 && score.harmony_alignment <= 1.0);
    }

    #[test]
    fn test_low_consciousness_snapshot() {
        let mut evaluator = AssetEvaluator::new();
        let low = ConsciousnessSnapshot::new(0.1, 0.1, 0.1, 0.1, 0.0, 0.1);
        let score = evaluator.evaluate(&test_metadata(), &low);

        assert!((score.phi_score - 0.1).abs() < f64::EPSILON);
        assert!(score.care_activation >= 0.0);
    }

    #[test]
    fn test_score_to_recommendation_strong_support() {
        let mut evaluator = AssetEvaluator::new();
        let rec = evaluator.score_to_recommendation(0.9, 0.9, false);
        assert_eq!(rec, GovernanceRecommendation::StrongSupport);
    }

    #[test]
    fn test_score_to_recommendation_oppose_on_violations() {
        let mut evaluator = AssetEvaluator::new();
        let rec = evaluator.score_to_recommendation(0.9, 0.9, true);
        assert_eq!(rec, GovernanceRecommendation::StrongOppose);
    }

    #[test]
    fn test_score_to_recommendation_neutral() {
        let mut evaluator = AssetEvaluator::new();
        let rec = evaluator.score_to_recommendation(0.1, 0.1, false);
        assert_eq!(rec, GovernanceRecommendation::Neutral);
    }

    #[test]
    fn test_score_to_recommendation_oppose() {
        let mut evaluator = AssetEvaluator::new();
        let rec = evaluator.score_to_recommendation(-0.3, 0.1, false);
        assert_eq!(rec, GovernanceRecommendation::Oppose);
    }

    #[test]
    fn test_harmony_alignment_mapping() {
        let mut evaluator = AssetEvaluator::new();
        let score = evaluator.evaluate(&test_metadata(), &test_snapshot());

        // harmony_alignment should be in [0,1] (mapped from [-1,+1])
        assert!(score.harmony_alignment >= 0.0);
        assert!(score.harmony_alignment <= 1.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut evaluator = AssetEvaluator::new();
        let score = evaluator.evaluate(&test_metadata(), &test_snapshot());
        let json = serde_json::to_string(&score).unwrap();
        let back: AssetConsciousnessScore = serde_json::from_str(&json).unwrap();
        assert!((back.phi_score - score.phi_score).abs() < f64::EPSILON);
        assert_eq!(back.recommendation, score.recommendation);
    }

    #[test]
    fn test_metadata_serialization() {
        let meta = test_metadata();
        let json = serde_json::to_string(&meta).unwrap();
        let back: AssetMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.project_type, "Solar");
        assert_eq!(back.impact_claims.len(), 3);
    }
}
