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
//!
//! Promotion also re-checks the internal consistency of the resolution object.
//! A caller cannot manufacture `Observed` by constructing an outer
//! `Qualified` outcome around artifact failures, an inconsistent inner receipt,
//! or a supposedly-qualified receipt that still carries qualification failures.

use serde::{Deserialize, Serialize};

use super::gwt1_evidence_envelope::Gwt1EvidenceEnvelopeResolutionV1;
use super::gwt1_qualification::Gwt1QualificationOutcomeV1;
use super::report::{EvidenceOutcome, SupportTier};

pub const GWT1_DIRECT_PROMOTION_POLICY_V1: &str = "butlin-gwt1-direct-promotion-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1PromotionInputStatusV1 {
    Consistent,
    ArtifactIntegrityFailure,
    OutcomeMismatch,
    QualifiedReceiptHasFailures,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1DirectPromotionDecisionV1 {
    pub policy: String,
    pub reported_qualification_outcome: Gwt1QualificationOutcomeV1,
    pub qualification_outcome: Gwt1QualificationOutcomeV1,
    pub input_status: Gwt1PromotionInputStatusV1,
    pub evidence_outcome: EvidenceOutcome,
}

fn validate_promotion_input(
    resolution: &Gwt1EvidenceEnvelopeResolutionV1,
) -> Gwt1PromotionInputStatusV1 {
    if !resolution.artifact_failures.is_empty() {
        return Gwt1PromotionInputStatusV1::ArtifactIntegrityFailure;
    }
    if resolution.outcome != resolution.receipt_resolution.outcome {
        return Gwt1PromotionInputStatusV1::OutcomeMismatch;
    }
    if resolution.outcome == Gwt1QualificationOutcomeV1::Qualified
        && !resolution.receipt_resolution.failures.is_empty()
    {
        return Gwt1PromotionInputStatusV1::QualifiedReceiptHasFailures;
    }
    Gwt1PromotionInputStatusV1::Consistent
}

pub fn promote_direct_gwt1_v1(
    resolution: &Gwt1EvidenceEnvelopeResolutionV1,
) -> Gwt1DirectPromotionDecisionV1 {
    let input_status = validate_promotion_input(resolution);
    let qualification_outcome = if input_status == Gwt1PromotionInputStatusV1::Consistent {
        resolution.outcome
    } else {
        Gwt1QualificationOutcomeV1::Inconclusive
    };

    let evidence_outcome = match qualification_outcome {
        Gwt1QualificationOutcomeV1::Qualified => {
            EvidenceOutcome::Supported(SupportTier::Observed)
        }
        Gwt1QualificationOutcomeV1::NotDemonstrated => EvidenceOutcome::NotDemonstrated,
        Gwt1QualificationOutcomeV1::Contradicted => EvidenceOutcome::Contradicted,
        Gwt1QualificationOutcomeV1::Inconclusive => EvidenceOutcome::Inconclusive,
    };

    Gwt1DirectPromotionDecisionV1 {
        policy: GWT1_DIRECT_PROMOTION_POLICY_V1.to_string(),
        reported_qualification_outcome: resolution.outcome,
        qualification_outcome,
        input_status,
        evidence_outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::gwt1_evidence_envelope::{
        Gwt1ArtifactIntegrityFailureV1, Gwt1EvidenceEnvelopeResolutionV1,
    };
    use crate::benchmarks::butlin::gwt1_qualification::{
        Gwt1QualificationFailureV1, Gwt1QualificationOutcomeV1,
        Gwt1QualificationResolutionV1,
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
        assert_eq!(decision.input_status, Gwt1PromotionInputStatusV1::Consistent);
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
    fn negative_outcomes_map_without_softening() {
        for (outcome, expected) in [
            (Gwt1QualificationOutcomeV1::NotDemonstrated, EvidenceOutcome::NotDemonstrated),
            (Gwt1QualificationOutcomeV1::Contradicted, EvidenceOutcome::Contradicted),
            (Gwt1QualificationOutcomeV1::Inconclusive, EvidenceOutcome::Inconclusive),
        ] {
            assert_eq!(promote_direct_gwt1_v1(&resolution(outcome)).evidence_outcome, expected);
        }
    }

    #[test]
    fn artifact_failure_cannot_be_wrapped_in_qualified() {
        let mut malformed = resolution(Gwt1QualificationOutcomeV1::Qualified);
        malformed.artifact_failures.push(
            Gwt1ArtifactIntegrityFailureV1::RawObservationLengthMismatch {
                declared: 10,
                observed: 11,
            },
        );
        let decision = promote_direct_gwt1_v1(&malformed);
        assert_eq!(decision.input_status, Gwt1PromotionInputStatusV1::ArtifactIntegrityFailure);
        assert_eq!(decision.evidence_outcome, EvidenceOutcome::Inconclusive);
    }

    #[test]
    fn outer_inner_outcome_mismatch_fails_closed() {
        let mut malformed = resolution(Gwt1QualificationOutcomeV1::Qualified);
        malformed.receipt_resolution.outcome = Gwt1QualificationOutcomeV1::NotDemonstrated;
        assert_eq!(
            promote_direct_gwt1_v1(&malformed).input_status,
            Gwt1PromotionInputStatusV1::OutcomeMismatch
        );
    }

    #[test]
    fn qualified_receipt_with_failures_fails_closed() {
        let mut malformed = resolution(Gwt1QualificationOutcomeV1::Qualified);
        malformed.receipt_resolution.failures.push(
            Gwt1QualificationFailureV1::MissingExecutionIdentity {
                field: "toolchain".to_string(),
            },
        );
        assert_eq!(
            promote_direct_gwt1_v1(&malformed).input_status,
            Gwt1PromotionInputStatusV1::QualifiedReceiptHasFailures
        );
    }
}
