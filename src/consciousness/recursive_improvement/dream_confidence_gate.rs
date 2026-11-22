// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empirical authority boundary for dream-derived confidence promotion.
//!
//! Positive dream-prior adjustments are available only when a runtime-only
//! evidence capability matches the exact context, canonical claim identity and
//! scope. Ordinary bridge callers remain non-promoting.

use super::dream_feedback::DreamFeedbackBridge;
use super::epistemic_world::{EpistemicWorldRecord, ValidatedEpistemicWorldRecord};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ContextBinding {
    context_hash: u64,
    claim_identity: String,
    scope_identity: String,
    prior_source_family_identity: String,
}

impl ContextBinding {
    fn from_record(
        context_hash: u64,
        record: &EpistemicWorldRecord,
        prior_source_family_identity: &str,
    ) -> Option<Self> {
        if record.claim_identity.trim().is_empty()
            || record.scope_identity.trim().is_empty()
            || prior_source_family_identity.trim().is_empty()
        {
            return None;
        }
        Some(Self {
            context_hash,
            claim_identity: record.claim_identity.clone(),
            scope_identity: record.scope_identity.clone(),
            prior_source_family_identity: prior_source_family_identity.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum DreamConfidenceAuthority {
    DirectEmpirical(EpistemicWorldRecord),
    ValidatedGenerated(ValidatedEpistemicWorldRecord),
}

impl DreamConfidenceAuthority {
    fn source_record(&self) -> &EpistemicWorldRecord {
        match self {
            Self::DirectEmpirical(record) => record,
            Self::ValidatedGenerated(record) => record.record(),
        }
    }

    fn validation_provenance_digest(&self) -> &str {
        match self {
            Self::DirectEmpirical(record) => &record.provenance_digest,
            Self::ValidatedGenerated(record) => record.validation().provenance_digest(),
        }
    }
}

#[derive(Debug, Default)]
pub struct DreamConfidenceGate {
    validations: HashMap<ContextBinding, DreamConfidenceAuthority>,
}

impl DreamConfidenceGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate_context(
        &mut self,
        context_hash: u64,
        prior_source_family_identity: &str,
        evidence: EpistemicWorldRecord,
    ) -> bool {
        if !evidence.may_promote_confidence()
            || evidence.source_family_identity == prior_source_family_identity
        {
            return false;
        }
        let Some(binding) = ContextBinding::from_record(
            context_hash,
            &evidence,
            prior_source_family_identity,
        ) else {
            return false;
        };
        self.validations.insert(
            binding,
            DreamConfidenceAuthority::DirectEmpirical(evidence),
        );
        true
    }

    pub fn validate_generated_context(
        &mut self,
        context_hash: u64,
        evidence: ValidatedEpistemicWorldRecord,
    ) -> bool {
        if !evidence.may_promote_confidence() {
            return false;
        }
        let Some(binding) = ContextBinding::from_record(
            context_hash,
            evidence.record(),
            &evidence.record().source_family_identity,
        ) else {
            return false;
        };
        self.validations.insert(
            binding,
            DreamConfidenceAuthority::ValidatedGenerated(evidence),
        );
        true
    }

    pub fn is_validated(
        &self,
        context_hash: u64,
        claim_identity: &str,
        scope_identity: &str,
        prior_source_family_identity: &str,
    ) -> bool {
        self.validations.contains_key(&ContextBinding {
            context_hash,
            claim_identity: claim_identity.to_string(),
            scope_identity: scope_identity.to_string(),
            prior_source_family_identity: prior_source_family_identity.to_string(),
        })
    }

    pub fn validation_source(
        &self,
        context_hash: u64,
        claim_identity: &str,
        scope_identity: &str,
        prior_source_family_identity: &str,
    ) -> Option<&EpistemicWorldRecord> {
        let binding = ContextBinding {
            context_hash,
            claim_identity: claim_identity.to_string(),
            scope_identity: scope_identity.to_string(),
            prior_source_family_identity: prior_source_family_identity.to_string(),
        };
        self.validations
            .get(&binding)
            .map(DreamConfidenceAuthority::source_record)
    }

    pub fn validation_provenance_digest(
        &self,
        context_hash: u64,
        claim_identity: &str,
        scope_identity: &str,
        prior_source_family_identity: &str,
    ) -> Option<&str> {
        let binding = ContextBinding {
            context_hash,
            claim_identity: claim_identity.to_string(),
            scope_identity: scope_identity.to_string(),
            prior_source_family_identity: prior_source_family_identity.to_string(),
        };
        self.validations
            .get(&binding)
            .map(DreamConfidenceAuthority::validation_provenance_digest)
    }

    /// Applies positive dream-prior adjustment only when the typed authority
    /// exactly matches context + claim + scope. Otherwise the bridge's safe,
    /// non-increasing public path is used.
    pub fn adjust_confidence(
        &self,
        bridge: &DreamFeedbackBridge,
        base_confidence: f64,
        context_hash: u64,
        claim_identity: &str,
        scope_identity: &str,
        prior_source_family_identity: &str,
    ) -> (f64, bool) {
        if self.is_validated(
            context_hash,
            claim_identity,
            scope_identity,
            prior_source_family_identity,
        ) {
            bridge.proposed_confidence_adjustment(
                base_confidence,
                context_hash,
                claim_identity,
                scope_identity,
                prior_source_family_identity,
            )
        } else {
            bridge.adjust_confidence(base_confidence, context_hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        EpistemicWorldRecord, WorldEvidenceKind,
        dream_feedback::DreamInsight,
    };

    const CLAIM: &str = "claim:alpha";
    const SCOPE: &str = "scope:site-a";

    fn record(
        kind: WorldEvidenceKind,
        event: &str,
        family: &str,
        claim: &str,
        scope: &str,
    ) -> EpistemicWorldRecord {
        EpistemicWorldRecord::new(kind, event)
            .unwrap()
            .with_binding(claim, scope, family)
            .unwrap()
    }

    fn bridge_with_prior() -> DreamFeedbackBridge {
        let mut bridge = DreamFeedbackBridge::new();
        let insight = DreamInsight::new(7, vec![0.0, 0.0], vec![1.0, 0.0], 0.2)
            .with_claim_scope_and_source(CLAIM, SCOPE, "family:dream");
        assert!(bridge.process_insight(insight));
        bridge
    }

    #[test]
    fn direct_recorded_evidence_authorizes_matching_context_claim_and_scope() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            "family:dream",
            record(
                WorldEvidenceKind::Recorded,
                "event:recorded",
                "family:sensor",
                CLAIM,
                SCOPE,
            ),
        ));

        let (promoted, informed) = gate.adjust_confidence(
            &bridge,
            0.6,
            7,
            CLAIM,
            SCOPE,
            "family:dream",
        );
        assert!(informed);
        assert!(promoted > 0.6);
        assert_eq!(
            gate.validation_provenance_digest(7, CLAIM, SCOPE, "family:dream"),
            Some("event:recorded")
        );
    }

    #[test]
    fn replay_derived_evidence_can_authorize_matching_context() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            "family:dream",
            record(
                WorldEvidenceKind::ReplayDerived,
                "event:replay",
                "family:replay",
                CLAIM,
                SCOPE,
            ),
        ));
        assert!(gate.adjust_confidence(&bridge, 0.6, 7, CLAIM, SCOPE, "family:dream").0 > 0.6);
    }

    #[test]
    fn generated_record_needs_independent_empirical_support_for_same_claim_and_scope() {
        let bridge = bridge_with_prior();
        let generated = record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "family:dream",
            CLAIM,
            SCOPE,
        );
        let independent = record(
            WorldEvidenceKind::Recorded,
            "event:observed",
            "family:sensor",
            CLAIM,
            SCOPE,
        );
        let validated = generated.validated_by(&independent).unwrap();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_generated_context(7, validated));
        assert!(gate.adjust_confidence(&bridge, 0.6, 7, CLAIM, SCOPE, "family:dream").0 > 0.6);
    }

    #[test]
    fn different_claim_scope_or_context_does_not_authorize() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            "family:dream",
            record(
                WorldEvidenceKind::Recorded,
                "event:recorded",
                "family:sensor",
                CLAIM,
                SCOPE,
            ),
        ));

        assert_eq!(
            gate.adjust_confidence(
                &bridge,
                0.6,
                7,
                "claim:other",
                SCOPE,
                "family:dream",
            )
            .0,
            0.6
        );
        assert_eq!(
            gate.adjust_confidence(
                &bridge,
                0.6,
                7,
                CLAIM,
                "scope:other",
                "family:dream",
            )
            .0,
            0.6
        );
        assert_eq!(gate.adjust_confidence(&bridge, 0.6, 8, CLAIM, SCOPE, "family:dream").0, 0.6);
    }

    #[test]
    fn legacy_boolean_and_unvalidated_generated_record_cannot_authorize() {
        let bridge = bridge_with_prior();
        let mut generated = record(
            WorldEvidenceKind::Counterfactual,
            "event:dream",
            "family:dream",
            CLAIM,
            SCOPE,
        );
        generated.empirically_validated = true;

        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(7, "family:dream", generated));
        assert_eq!(gate.adjust_confidence(&bridge, 0.6, 7, CLAIM, SCOPE, "family:dream").0, 0.6);
    }

    #[test]
    fn same_source_family_direct_evidence_cannot_authorize() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(
            7,
            "family:dream",
            record(
                WorldEvidenceKind::Recorded,
                "event:recorded-dream-family",
                "family:dream",
                CLAIM,
                SCOPE,
            ),
        ));
        assert_eq!(
            gate.adjust_confidence(
                &bridge,
                0.6,
                7,
                CLAIM,
                SCOPE,
                "family:dream",
            )
            .0,
            0.6
        );
    }

    #[test]
    fn same_source_family_direct_evidence_is_not_independent() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(
            7,
            "family:dream",
            record(
                WorldEvidenceKind::Recorded,
                "event:recorded-same-family",
                "family:dream",
                CLAIM,
                SCOPE,
            ),
        ));
        let result = gate.adjust_confidence(
            &bridge,
            0.6,
            7,
            CLAIM,
            SCOPE,
            "family:dream",
        );
        assert_eq!(result.0, 0.6);
    }

    #[test]
    fn validator_for_different_prior_source_family_does_not_authorize() {
        let bridge = bridge_with_prior();
        let mut gate = DreamConfidenceGate::new();
        assert!(gate.validate_context(
            7,
            "family:other-dream",
            record(
                WorldEvidenceKind::Recorded,
                "event:recorded-other",
                "family:sensor",
                CLAIM,
                SCOPE,
            ),
        ));
        let result = gate.adjust_confidence(
            &bridge,
            0.6,
            7,
            CLAIM,
            SCOPE,
            "family:dream",
        );
        assert_eq!(result.0, 0.6);
    }

    #[test]
    fn model_prediction_cannot_self_authorize() {
        let mut gate = DreamConfidenceGate::new();
        let predicted = record(
            WorldEvidenceKind::ModelPredicted,
            "event:prediction",
            "family:model",
            CLAIM,
            SCOPE,
        );
        assert!(!gate.validate_context(7, "family:dream", predicted));
        assert!(!gate.is_validated(7, CLAIM, SCOPE, "family:dream"));
    }

    #[test]
    fn serialized_boolean_cannot_be_the_only_binding() {
        let mut record = EpistemicWorldRecord::new(
            WorldEvidenceKind::Recorded,
            "event:recorded",
        )
        .unwrap();
        record.empirically_validated = true;
        let mut gate = DreamConfidenceGate::new();
        assert!(!gate.validate_context(7, "family:dream", record));
    }
}
