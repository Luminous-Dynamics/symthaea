// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::relational_consciousness::{
    RelationMode, RelationalAssessment, RelationshipStage,
};

/// A lightweight model of the human partner in a dyadic relationship.
///
/// This struct deliberately keeps only coarse-grained state suitable for
/// Φ_dyad computation and high-level reasoning. It is *not* intended to be
/// a full psychological profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanPartnerModel {
    /// Stable identifier for the partner (screen name, handle, etc.).
    pub partner_id: String,

    /// Buber mode: I-It vs I-Thou (relational stance).
    pub mode: RelationMode,

    /// Relational stage derived from Φ_relation and history.
    pub stage: RelationshipStage,

    /// Trust in the relationship (0.0–1.0).
    pub trust: f32,

    /// Willingness to be seen / known (0.0–1.0).
    pub vulnerability: f32,

    /// Degree of perceived mutuality (0.0–1.0).
    pub reciprocity: f32,

    /// Relational Φ as perceived / estimated for this dyad.
    pub phi_relational: f64,

    /// Number of interactions observed.
    pub interactions_count: u64,

    /// Timestamp of the last interaction (seconds since UNIX epoch).
    pub last_interaction_ts: Option<f64>,
}

impl HumanPartnerModel {
    /// Create a new partner model with conservative defaults.
    pub fn new(partner_id: impl Into<String>) -> Self {
        Self {
            partner_id: partner_id.into(),
            mode: RelationMode::IIt,
            stage: RelationshipStage::NoRelation,
            trust: 0.0,
            vulnerability: 0.0,
            reciprocity: 0.0,
            phi_relational: 0.0,
            interactions_count: 0,
            last_interaction_ts: None,
        }
    }

    /// Update internal signals from a single interaction event.
    ///
    /// This is intentionally simple and monotonic; higher-level learning
    /// mechanisms can refine these heuristics in future work.
    pub fn update_on_interaction(&mut self, event: &InteractionEvent) {
        self.interactions_count = self.interactions_count.saturating_add(1);
        self.last_interaction_ts = Some(event.timestamp);

        // Clamp inputs to [0, 1] to avoid accidental drift.
        let depth = event.depth.clamp(0.0, 1.0);
        let safety = event.emotional_safety.clamp(0.0, 1.0);
        let mutuality = event.mutuality.clamp(0.0, 1.0);

        // Simple incremental updates with diminishing returns.
        self.trust = (self.trust + 0.3 * depth + 0.3 * safety).clamp(0.0, 1.0);
        self.vulnerability = (self.vulnerability + 0.2 * depth + 0.2 * safety).clamp(0.0, 1.0);
        self.reciprocity = (self.reciprocity + 0.4 * mutuality).clamp(0.0, 1.0);
    }

    /// Incorporate a relational consciousness assessment into the model.
    ///
    /// This connects the dyadic Φ_relation machinery to the partner model.
    pub fn update_from_assessment(&mut self, assessment: &RelationalAssessment) {
        self.mode = assessment.mode;
        self.stage = assessment.stage;
        self.phi_relational = assessment.phi_relation;

        // Use synchrony and mutual information as coarse trust / reciprocity signals.
        self.trust = (self.trust * 0.7 + (assessment.synchrony as f32) * 0.3).clamp(0.0, 1.0);
        self.reciprocity =
            (self.reciprocity * 0.7 + (assessment.mutual_information as f32) * 0.3).clamp(0.0, 1.0);
    }

    /// Advance the relationship stage based on internal signals.
    ///
    /// This provides a very simple, deterministic progression rule that can be
    /// refined later without breaking the API.
    pub fn advance_stage_if_ready(&mut self) {
        use RelationshipStage::*;

        let next = match self.stage {
            NoRelation if self.trust > 0.05 => Awareness,
            Awareness if self.trust > 0.2 && self.reciprocity > 0.1 => Contact,
            Contact if self.trust > 0.4 && self.reciprocity > 0.3 => Attunement,
            Attunement if self.trust > 0.6 && self.reciprocity > 0.5 => Bonding,
            Bonding if self.trust > 0.8 && self.reciprocity > 0.7 && self.vulnerability > 0.6 => {
                Unity
            }
            other => other,
        };

        self.stage = next;
    }
}

/// A single human–AI interaction event used to update the partner model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: f64,

    /// Depth / meaningfulness of the interaction (0.0–1.0).
    pub depth: f32,

    /// Subjective emotional safety (0.0–1.0).
    pub emotional_safety: f32,

    /// Degree of mutual engagement and responsiveness (0.0–1.0).
    pub mutuality: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::relational_consciousness::RelationshipStage;

    fn make_event(depth: f32, safety: f32, mutuality: f32) -> InteractionEvent {
        InteractionEvent {
            timestamp: 1000.0,
            depth,
            emotional_safety: safety,
            mutuality,
        }
    }

    #[test]
    fn test_new_partner_model_defaults() {
        let model = HumanPartnerModel::new("alice");
        assert_eq!(model.partner_id, "alice");
        assert_eq!(model.trust, 0.0);
        assert_eq!(model.vulnerability, 0.0);
        assert_eq!(model.reciprocity, 0.0);
        assert_eq!(model.interactions_count, 0);
        assert!(model.last_interaction_ts.is_none());
        assert_eq!(model.stage, RelationshipStage::NoRelation);
    }

    #[test]
    fn test_update_increases_trust_and_reciprocity() {
        let mut model = HumanPartnerModel::new("bob");
        let event = make_event(0.8, 0.9, 0.7);
        model.update_on_interaction(&event);

        assert!(
            model.trust > 0.0,
            "Trust should increase after positive interaction"
        );
        assert!(model.vulnerability > 0.0, "Vulnerability should increase");
        assert!(model.reciprocity > 0.0, "Reciprocity should increase");
        assert_eq!(model.interactions_count, 1);
        assert_eq!(model.last_interaction_ts, Some(1000.0));
    }

    #[test]
    fn test_update_clamps_values_to_unit_range() {
        let mut model = HumanPartnerModel::new("clamp_test");
        // Apply many high-value interactions
        for i in 0..100 {
            model.update_on_interaction(&InteractionEvent {
                timestamp: i as f64,
                depth: 1.0,
                emotional_safety: 1.0,
                mutuality: 1.0,
            });
        }
        assert!(model.trust <= 1.0, "Trust must be clamped to 1.0");
        assert!(
            model.vulnerability <= 1.0,
            "Vulnerability must be clamped to 1.0"
        );
        assert!(
            model.reciprocity <= 1.0,
            "Reciprocity must be clamped to 1.0"
        );
    }

    #[test]
    fn test_update_clamps_negative_inputs() {
        let mut model = HumanPartnerModel::new("neg_test");
        let event = make_event(-5.0, -3.0, -2.0);
        model.update_on_interaction(&event);
        // With negative inputs clamped to 0, trust should remain 0
        assert!(model.trust >= 0.0);
        assert!(model.vulnerability >= 0.0);
        assert!(model.reciprocity >= 0.0);
    }

    #[test]
    fn test_zero_interaction_no_change() {
        let mut model = HumanPartnerModel::new("zero_test");
        let event = make_event(0.0, 0.0, 0.0);
        model.update_on_interaction(&event);
        assert_eq!(model.trust, 0.0);
        assert_eq!(model.vulnerability, 0.0);
        assert_eq!(model.reciprocity, 0.0);
        assert_eq!(model.interactions_count, 1);
    }

    #[test]
    fn test_stage_progression() {
        let mut model = HumanPartnerModel::new("stage_test");

        // Initially NoRelation
        assert_eq!(model.stage, RelationshipStage::NoRelation);

        // Small trust bump → Awareness
        model.trust = 0.06;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, RelationshipStage::Awareness);

        // Increase trust and reciprocity → Contact
        model.trust = 0.25;
        model.reciprocity = 0.15;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, RelationshipStage::Contact);

        // Further increase → Attunement
        model.trust = 0.45;
        model.reciprocity = 0.35;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, RelationshipStage::Attunement);

        // → Bonding
        model.trust = 0.65;
        model.reciprocity = 0.55;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, RelationshipStage::Bonding);

        // → Unity (requires vulnerability too)
        model.trust = 0.85;
        model.reciprocity = 0.75;
        model.vulnerability = 0.65;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, RelationshipStage::Unity);
    }

    #[test]
    fn test_stage_does_not_regress() {
        let mut model = HumanPartnerModel::new("regress_test");
        model.trust = 0.5;
        model.reciprocity = 0.4;
        model.advance_stage_if_ready();
        let stage_after = model.stage;

        // Drop trust below threshold
        model.trust = 0.0;
        model.reciprocity = 0.0;
        model.advance_stage_if_ready();
        assert_eq!(model.stage, stage_after, "Stage should not regress");
    }

    #[test]
    fn test_interactions_count_saturates() {
        let mut model = HumanPartnerModel::new("sat_test");
        model.interactions_count = u64::MAX;
        let event = make_event(0.5, 0.5, 0.5);
        model.update_on_interaction(&event);
        assert_eq!(
            model.interactions_count,
            u64::MAX,
            "Count should saturate, not wrap"
        );
    }
}
