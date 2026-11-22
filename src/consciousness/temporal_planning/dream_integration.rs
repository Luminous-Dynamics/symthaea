// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Integration
//!
//! Converts DreamInsights from the dream feedback bridge into
//! ActionPriors for MCTS. This enables "retroactive self-improvement":
//! learning from pasts that never happened.

use super::types::PlannedAction;

#[cfg(feature = "magi_loop")]
use crate::consciousness::recursive_improvement::dream_feedback::DreamFeedbackBridge;
#[cfg(feature = "magi_loop")]
use crate::consciousness::recursive_improvement::{
    DreamConfidenceGate, EpistemicWorldRecord, WorldEvidenceKind,
};

/// Convert a dream feedback bridge's priors into MCTS action priors.
///
/// For each available action, checks if the dream bridge has a
/// prior for a matching context. If so, boosts that action's prior.
#[cfg(feature = "magi_loop")]
pub fn apply_dream_priors(
    actions: &mut [PlannedAction],
    bridge: &DreamFeedbackBridge,
    context_hash: u64,
) {
    if let Some(prior) = bridge.get_prior(context_hash) {
        for action in actions.iter_mut() {
            // Compute similarity between action embedding and dream prior direction
            let similarity = cosine_similarity(&action.embedding, &prior.preferred_direction);
            if similarity > 0.5 {
                // Boost prior proportional to similarity and dream strength
                action.prior = (action.prior + similarity as f64 * prior.strength).min(1.0);
            }
        }
    }
}

/// Apply dream confidence adjustments to a plan confidence.
#[cfg(feature = "magi_loop")]
pub fn adjust_plan_confidence(
    confidence: f64,
    bridge: &DreamFeedbackBridge,
    context_hash: u64,
) -> f64 {
    let (adjusted, _was_dream_informed) = bridge.adjust_confidence(confidence, context_hash);
    adjusted.clamp(0.0, 1.0)
}

/// Apply dream confidence adjustment with typed empirical authority.
///
/// The existing `adjust_plan_confidence` path remains non-promoting. This
/// opt-in API permits a positive dream-prior adjustment only when the gate has
/// a capability matching the exact context, claim identity and scope identity.
#[cfg(feature = "magi_loop")]
pub fn adjust_plan_confidence_with_gate(
    confidence: f64,
    bridge: &DreamFeedbackBridge,
    gate: &DreamConfidenceGate,
    context_hash: u64,
    claim_identity: &str,
    scope_identity: &str,
    prior_source_family_identity: &str,
) -> f64 {
    gate.adjust_confidence(
        bridge,
        confidence,
        context_hash,
        claim_identity,
        scope_identity,
        prior_source_family_identity,
    )
    .0
    .clamp(0.0, 1.0)
}

/// Cosine similarity between two vectors (delegates to shared implementation).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    symthaea_core::math::cosine_similarity_f32(a, b).clamp(-1.0, 1.0)
}

/// Create uniform priors for actions (when no dream data available).
pub fn uniform_priors(actions: &mut [PlannedAction]) {
    let prior = 1.0 / actions.len().max(1) as f64;
    for action in actions.iter_mut() {
        action.prior = prior;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "magi_loop")]
    use crate::consciousness::recursive_improvement::dream_feedback::DreamInsight;

    #[cfg(feature = "magi_loop")]
    #[test]
    fn ordinary_plan_confidence_path_remains_non_promoting() {
        let mut bridge = DreamFeedbackBridge::new();
        let insight = DreamInsight::new(12345, vec![0.0, 0.0], vec![1.0, 0.0], 0.2)
            .with_claim_scope_and_source("claim:alpha", "scope:site-a", "family:dream");
        assert!(bridge.process_insight(insight));

        assert!(adjust_plan_confidence(0.6, &bridge, 12345) <= 0.6);
    }

    #[cfg(feature = "magi_loop")]
    #[test]
    fn plan_confidence_promotion_requires_exact_claim_and_scope_binding() {
        let mut bridge = DreamFeedbackBridge::new();
        let insight = DreamInsight::new(12345, vec![0.0, 0.0], vec![1.0, 0.0], 0.2)
            .with_claim_scope_and_source("claim:alpha", "scope:site-a", "family:dream");
        assert!(bridge.process_insight(insight));

        let generated = EpistemicWorldRecord::new(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
        )
        .unwrap()
        .with_binding("claim:alpha", "scope:site-a", "family:dream")
        .unwrap();
        let observed = EpistemicWorldRecord::new(
            WorldEvidenceKind::Recorded,
            "event:observed",
        )
        .unwrap()
        .with_binding("claim:alpha", "scope:site-a", "family:sensor")
        .unwrap();
        let capability = generated.validated_by(&observed).unwrap();

        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_generated_context(12345, capability));

        let promoted = adjust_plan_confidence_with_gate(
            0.6,
            &bridge,
            &gate,
            12345,
            "claim:alpha",
            "scope:site-a",
            "family:dream",
        );
        assert!(promoted > 0.6);

        let wrong_claim = adjust_plan_confidence_with_gate(
            0.6,
            &bridge,
            &gate,
            12345,
            "claim:other",
            "scope:site-a",
            "family:dream",
        );
        let wrong_scope = adjust_plan_confidence_with_gate(
            0.6,
            &bridge,
            &gate,
            12345,
            "claim:alpha",
            "scope:site-b",
            "family:dream",
        );
        assert_eq!(wrong_claim, 0.6);
        assert_eq!(wrong_scope, 0.6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_priors() {
        let mut actions = vec![
            PlannedAction {
                id: "a".into(),
                description: "test".into(),
                embedding: vec![],
                prior: 0.0,
                is_epistemic: false,
            },
            PlannedAction {
                id: "b".into(),
                description: "test".into(),
                embedding: vec![],
                prior: 0.0,
                is_epistemic: false,
            },
        ];
        uniform_priors(&mut actions);
        assert!((actions[0].prior - 0.5).abs() < 1e-10);
        assert!((actions[1].prior - 0.5).abs() < 1e-10);
    }

    #[cfg(feature = "magi_loop")]
    #[test]
    fn test_dream_priors_applied() {
        let mut bridge = DreamFeedbackBridge::new();
        let insight = DreamInsight::new(12345, vec![0.0, 0.0], vec![1.0, 0.0], 0.2)
            .with_claim_scope_and_source("claim:alpha", "scope:site-a", "family:dream");
        bridge.process_insight(insight);

        let mut actions = vec![
            PlannedAction {
                id: "match".into(),
                description: "matches dream".into(),
                embedding: vec![0.9, 0.1],
                prior: 0.3,
                is_epistemic: false,
            },
            PlannedAction {
                id: "nomatch".into(),
                description: "doesn't match".into(),
                embedding: vec![-1.0, 0.0],
                prior: 0.3,
                is_epistemic: false,
            },
        ];

        apply_dream_priors(&mut actions, &bridge, 12345);
        // The matching action should have higher prior
        assert!(actions[0].prior > actions[1].prior);
    }
}
