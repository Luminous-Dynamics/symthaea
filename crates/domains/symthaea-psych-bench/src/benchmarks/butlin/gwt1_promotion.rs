// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative evidence-tier mapping for the direct GWT-1 qualification.
//!
//! This module deliberately cannot produce causal or functional support.
//! The direct specialist-independence experiment observes dissociable,
//! independently operable, concurrently executable specialist computations,
//! but it does not ablate a specialist and it does not demonstrate an
//! independent downstream competency consequence.
//!
//! The historical `specialization_fraction` aggregate is intentionally not an
//! input here. It remains dependent diagnostic evidence and may disagree with
//! the direct lineage without being averaged into it.

use serde::{Deserialize, Serialize};

use super::gwt1_evidence_envelope::Gwt1EvidenceEnvelopeResolutionV1;
use super::gwt1_qualification::Gwt1QualificationOutcomeV1;
use super::report::{EvidenceOutcome, SupportTier};

pub const GWT1_DIRECT_PROMOTION_POLICY_V1: &str = "butlin-gwt1-direct-promotion-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1DirectPromotionDecisionV1 {
    pub policy: String,
    pub qualification_outcome: Gwt1QualificationOutcomeV1,
    pub evidence_outcome: EvidenceOutcome,
}

/// Map an integrity-checked direct GWT-1 qualification into the existing
/// Butlin evidence ladder.
///
/// `Qualified` is capped at `Observed`. No value accepted by this function can
/// produce `CausallySupported` or `FunctionallySupported`.
pub fn promote_direct_gwt1_v1(
    resolution: &Gwt1EvidenceEnvelopeResolutionV1,
) -> Gwt1DirectPromotionDecisionV1 {
    let evidence_outcome = match resolution.outcome {
        Gwt1QualificationOutcomeV1::Qualified => EvidenceOutcome::Supported(SupportTier::Observed),
        Gwt1QualificationOutcomeV1::NotDemonstrated => EvidenceOutcome::NotDemonstrated,
        Gwt1QualificationOutcomeV1::Contradicted => EvidenceOutcome::Contradicted,
        Gwt1QualificationOutcomeV1::Inconclusive => EvidenceOutcome::Inconclusive,
    };

    Gwt1DirectPromotionDecisionV1 {
        policy: GWT1_DIRECT_PROMOTION_POLICY_V1.to_string(),
        qualification_outcome: resolution.outcome,
        evidence_outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::gwt1_evidence_envelope::Gwt1EvidenceEnvelopeResolutionV1;
    use crate::benchmarks::butlin::gwt1_qualification::{
        Gwt1QualificationResolutionV1, Gwt1QualificationOutcomeV1,
    };

    fn resolution(outcome: Gwt1QualificationOutcomeV1) -> Gwt1EvidenceEnvelopeResolutionV1 {
        Gwt1EvidenceEnvelopeResolutionV1 {
            outcome,
            artifact_failures: Vec::new(),
            receipt_resolution: Gwt1QualificationResolutionV1 {
                outcome,
                failures: Vec::new(),
            },
        }
    }

    #[test]
    fn qualified_is_capped_at_observed() {
        let decision = promote_direct_gwt1_v1(&resolution(Gwt1QualificationOutcomeV1::Qualified));
        assert_eq!(
            decision.evidence_outcome,
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
        assert_ne!(
            decision.evidence_outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
        assert_ne!(
            decision.evidence_outcome,
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        );
    }

    #[test]
    fn not_demonstrated_maps_without_softening() {
        assert_eq!(
            promote_direct_gwt1_v1(&resolution(Gwt1QualificationOutcomeV1::NotDemonstrated))
                .evidence_outcome,
            EvidenceOutcome::NotDemonstrated
        );
    }

    #[test]
    fn contradicted_maps_without_softening() {
        assert_eq!(
            promote_direct_gwt1_v1(&resolution(Gwt1QualificationOutcomeV1::Contradicted))
                .evidence_outcome,
            EvidenceOutcome::Contradicted
        );
    }

    #[test]
    fn inconclusive_maps_without_softening() {
        assert_eq!(
            promote_direct_gwt1_v1(&resolution(Gwt1QualificationOutcomeV1::Inconclusive))
                .evidence_outcome,
            EvidenceOutcome::Inconclusive
        );
    }

    #[test]
    fn policy_identity_is_frozen() {
        let decision = promote_direct_gwt1_v1(&resolution(Gwt1QualificationOutcomeV1::Qualified));
        assert_eq!(decision.policy, GWT1_DIRECT_PROMOTION_POLICY_V1);
    }
}
