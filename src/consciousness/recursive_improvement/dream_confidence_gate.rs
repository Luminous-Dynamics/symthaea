// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empirical authority boundary for dream-derived confidence promotion.
//!
//! Generated/counterfactual evidence may propose actions and caution. It may not
//! increase epistemic confidence merely because a dream prior exists. Promotion
//! requires either direct empirical evidence or a runtime capability proving that
//! generated evidence was validated by a separate empirical record.

use super::dream_feedback::DreamFeedbackBridge;
use super::epistemic_world::{EpistemicWorldRecord, ValidatedEpistemicWorldRecord};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum DreamConfidenceAuthority {
    DirectEmpirical(EpistemicWorldRecord),
    ValidatedGenerated(ValidatedEpistemicWorldRecord),
}

impl DreamConfidenceAuthority {
    fn source_record(&self) -> &EpistemicWorldRecord {
        match self {
            Self::DirectEmpirical(record) => record,
            Self::ValidatedGenerated(validated) => validated.record(),
        }
    }

    fn validation_provenance_digest(&self) -> &str {
        match self {
            Self::DirectEmpirical(record) => record.provenance_digest(),
            Self::ValidatedGenerated(validated) => validated.validation().provenance_digest(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DreamConfidenceGate {
    validations: HashMap<u64, DreamConfidenceAuthority>,
}

impl DreamConfidenceGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Authorize dream-confidence promotion from evidence that is already
    /// Recorded or ReplayDerived.
    pub fn validate_context(
        &mut self,
        context_hash: u64,
        evidence: EpistemicWorldRecord,
    ) -> bool {
        if !evidence.may_promote_confidence() {
            return false;
        }
        self.validations
            .insert(context_hash, DreamConfidenceAuthority::DirectEmpirical(evidence));
        true
    }

    /// Authorize a generated/predicted source only after it has been bound to a
    /// separate empirical record through `EpistemicWorldRecord::validated_by`.
    pub fn validate_generated_context(
        &mut self,
        context_hash: u64,
        evidence: ValidatedEpistemicWorldRecord,
    ) -> bool {
        if !evidence.may_promote_confidence() {
            return false;
        }
        self.validations.insert(
            context_hash,
            DreamConfidenceAuthority::ValidatedGenerated(evidence),
        );
        true
    }

    pub fn is_validated(&self, context_hash: u64) -> bool {
        self.validations.contains_key(&context_hash)
    }

    /// Source record whose claim is being permitted to influence confidence.
    pub fn validation_source(&self, context_hash: u64) -> Option<&EpistemicWorldRecord> {
        self.validations
            .get(&context_hash)
            .map(DreamConfidenceAuthority::source_record)
    }

    /// Provenance digest of the empirical evidence that actually carries authority.
    /// For direct empirical evidence this is the source record itself; for generated
    /// evidence this is the separate validation record, never the dream/model digest.
    pub fn validation_provenance_digest(&self, context_hash: u64) -> Option<&str> {
        self.validations
            .get(&context_hash)
            .map(DreamConfidenceAuthority::validation_provenance_digest)
    }

    /// Apply dream feedback while enforcing the epistemic-promotion rule.
    ///
    /// - without authority, the bridge's public non-authorizing path is used;
    /// - with authority, the historical positive proposal may be applied;
    /// - cautionary decreases remain available in both cases.
    pub fn adjust_confidence(
        &self,
        bridge: &DreamFeedbackBridge,
        base_confidence: f64,
        context_hash: u64,
    ) -> (f64, bool) {
        if self.is_validated(context_hash) {
            bridge.proposed_confidence_adjustment(base_confidence, context_hash)
        } else {
            bridge.adjust_confidence(base_confidence, context_hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::dream_feedback::DreamInsight;
    use crate::consciousness::recursive_improvement::epistemic_world::WorldEvidenceKind;

    fn bridge_with_prior() -> DreamFeedbackBridge {
        let mut bridge = DreamFeedbackBridge::new();
        bridge.process_insight(DreamInsight::new(7, vec![0.1], vec![0.9], 0.5));
        bridge
    }

    fn evidence(kind: WorldEvidenceKind, digest: &str) -> EpistemicWorldRecord {
        EpistemicWorldRecord::new(kind, digest).expect("test evidence should be valid")
    }

    #[test]
    fn unvalidated_dream_prior_cannot_raise_confidence() {
        let bridge = bridge_with_prior();
        let gate = DreamConfidenceGate::new();
        let (adjusted, informed) = gate.adjust_confidence(&bridge, 0.6, 7);
        assert!(informed);
        assert_eq!(adjusted, 0.6);
    }

    #[test]
    fn risk_decrease_survives_without_validation() {
        let mut bridge = bridge_with_prior();
        bridge.mark_risky_context(7, "counterfactual failures".into(), 1.0);

        let gate = DreamConfidenceGate::new();
        let (adjusted, _) = gate.adjust_confidence(&bridge, 0.8, 7);
        assert!(adjusted < 0.8);
    }

    #[test]
    fn recorded_evidence_allows_promotion() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            evidence(WorldEvidenceKind::Recorded, "blake3:recorded-observation")
        ));
        assert_eq!(
            gate.validation_source(7)
                .expect("validation should exist")
                .kind(),
            WorldEvidenceKind::Recorded
        );
        assert_eq!(
            gate.validation_provenance_digest(7),
            Some("blake3:recorded-observation")
        );
        assert!(gate.adjust_confidence(&bridge, 0.6, 7).0 > 0.6);
    }

    #[test]
    fn replay_derived_evidence_allows_promotion() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            evidence(WorldEvidenceKind::ReplayDerived, "blake3:exact-replay")
        ));
        assert!(gate.adjust_confidence(&bridge, 0.6, 7).0 > 0.6);
    }

    #[test]
    fn unvalidated_counterfactual_cannot_authorize_promotion() {
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(
            7,
            evidence(WorldEvidenceKind::Counterfactual, "blake3:dream-source")
        ));
        assert!(!gate.is_validated(7));
    }

    #[test]
    fn legacy_boolean_cannot_authorize_promotion() {
        let mut counterfactual =
            evidence(WorldEvidenceKind::Counterfactual, "blake3:dream-source");
        counterfactual.empirically_validated = true;
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(7, counterfactual));
        assert!(!gate.is_validated(7));
    }

    #[test]
    fn separately_validated_counterfactual_can_authorize_promotion() {
        let bridge = bridge_with_prior();
        let empirical = evidence(
            WorldEvidenceKind::Recorded,
            "blake3:independent-observation",
        );
        let counterfactual =
            evidence(WorldEvidenceKind::Counterfactual, "blake3:dream-source");
        let validated = counterfactual
            .validated_by(&empirical)
            .expect("recorded evidence should create validation authority");

        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_generated_context(7, validated));
        assert_eq!(
            gate.validation_source(7)
                .expect("source should be retained")
                .kind(),
            WorldEvidenceKind::Counterfactual
        );
        assert_eq!(
            gate.validation_provenance_digest(7),
            Some("blake3:independent-observation")
        );
        assert!(gate.adjust_confidence(&bridge, 0.6, 7).0 > 0.6);
    }

    #[test]
    fn model_prediction_cannot_self_authorize() {
        let prediction = evidence(
            WorldEvidenceKind::ModelPredicted,
            "blake3:model-prediction",
        );
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(7, prediction));
        assert!(!gate.is_validated(7));
    }
}
