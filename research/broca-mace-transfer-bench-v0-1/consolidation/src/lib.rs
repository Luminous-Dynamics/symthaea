// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proposal-only consumer for receipt-qualified MACE consolidation summaries.
//!
//! This crate may propose a reusable procedure abstraction. It does not write
//! durable memory, erase data, grant authority, or replace Mycelix lifecycle
//! and privacy governance. The thresholds are prototype policy constants.

use broca_mace_ingress::ActionConsolidationEvidenceSummaryV1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldReason {
    NoTargetScopeEvidence,
    ScopeMismatch,
    PreRegistrationMissing,
    InsufficientQualifiedEvents,
    InsufficientIndependentFamilies,
    InsufficientIndependentRoots,
    MeasurementBoundMissing,
    MeasurementBoundTooWide,
    InvalidPolicyConfiguration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuarantineReason {
    UntrustedSourcePresent,
    UnknownTrustStatePresent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcedurePromotionProposalV1 {
    pub action: broca_mace_policy::Action,
    pub target_scope: broca_mace_policy::Scope,
    pub as_known: broca_mace_ingress::AsKnownQuery,
    pub evidence_event_ids: Vec<u32>,
    pub qualified_success_event_count: usize,
    pub independent_source_family_count: usize,
    pub independent_source_root_count: usize,
    pub maximum_measurement_error_bound: f64,
    pub minimum_qualified_success_events: usize,
    pub minimum_independent_source_families: usize,
    pub minimum_independent_source_roots: usize,
    pub policy_revision: &'static str,
}

/// A proposal is not a consolidation transaction or authority grant.
#[derive(Clone, Debug, PartialEq)]
pub enum ConsolidationProposalV1 {
    ProposeProcedure(ProcedurePromotionProposalV1),
    RetainProvisional(HoldReason),
    Quarantine(QuarantineReason),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsolidationPolicyV1 {
    pub minimum_qualified_success_events: usize,
    pub minimum_independent_source_families: usize,
    pub minimum_independent_source_roots: usize,
    pub maximum_measurement_error_bound: f64,
}

impl Default for ConsolidationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_qualified_success_events: 3,
            minimum_independent_source_families: 3,
            minimum_independent_source_roots: 3,
            maximum_measurement_error_bound: 0.05,
        }
    }
}

impl ConsolidationPolicyV1 {
    pub fn decide_procedure_promotion(
        &self,
        evidence: &ActionConsolidationEvidenceSummaryV1,
    ) -> ConsolidationProposalV1 {
        if self.minimum_qualified_success_events == 0
            || self.minimum_independent_source_families == 0
            || self.minimum_independent_source_roots == 0
            || !self.maximum_measurement_error_bound.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_measurement_error_bound)
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration,
            );
        }
        if evidence.target_scope_event_count == 0 {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::NoTargetScopeEvidence,
            );
        }
        if !evidence.scope_matches {
            return ConsolidationProposalV1::RetainProvisional(HoldReason::ScopeMismatch);
        }
        if evidence.untrusted_event_count > 0 {
            return ConsolidationProposalV1::Quarantine(
                QuarantineReason::UntrustedSourcePresent,
            );
        }
        if evidence.unknown_trust_event_count > 0 {
            return ConsolidationProposalV1::Quarantine(
                QuarantineReason::UnknownTrustStatePresent,
            );
        }
        if !evidence.all_trusted_predictions_pre_registered {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::PreRegistrationMissing,
            );
        }
        if evidence.qualified_success_event_count
            < self.minimum_qualified_success_events
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientQualifiedEvents,
            );
        }
        if evidence.qualified_success_source_family_count
            < self.minimum_independent_source_families
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentFamilies,
            );
        }
        if evidence.qualified_success_source_root_count
            < self.minimum_independent_source_roots
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentRoots,
            );
        }
        let Some(maximum_measurement_error_bound) =
            evidence.max_qualified_success_measurement_error_bound
        else {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::MeasurementBoundMissing,
            );
        };
        if !maximum_measurement_error_bound.is_finite()
            || !(0.0..=self.maximum_measurement_error_bound)
                .contains(&maximum_measurement_error_bound)
        {
            return ConsolidationProposalV1::RetainProvisional(
                HoldReason::MeasurementBoundTooWide,
            );
        }

        ConsolidationProposalV1::ProposeProcedure(ProcedurePromotionProposalV1 {
            action: evidence.action,
            target_scope: evidence.target_scope,
            as_known: evidence.as_known,
            evidence_event_ids: evidence.qualified_success_event_ids.clone(),
            qualified_success_event_count: evidence.qualified_success_event_count,
            independent_source_family_count:
                evidence.qualified_success_source_family_count,
            independent_source_root_count:
                evidence.qualified_success_source_root_count,
            maximum_measurement_error_bound,
            minimum_qualified_success_events: self.minimum_qualified_success_events,
            minimum_independent_source_families: self.minimum_independent_source_families,
            minimum_independent_source_roots: self.minimum_independent_source_roots,
            policy_revision: "mace-consolidation-proposal-v1",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_ingress::{
        Action, AsKnownQuery, ConsolidationEvidenceSummaryV1, SourceTrustStatus,
    };
    use broca_mace_policy::{Objective, Regime, Scope};

    fn summary() -> ActionConsolidationEvidenceSummaryV1 {
        ActionConsolidationEvidenceSummaryV1 {
            as_known: AsKnownQuery {
                valid_at_ms: 180,
                known_at_ms: 250,
            },
            target_scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
            action: Action::A,
            action_episode_count: 3,
            target_scope_event_count: 3,
            trusted_event_count: 3,
            trusted_source_family_count: 3,
            trusted_source_root_count: 3,
            qualified_success_event_count: 3,
            qualified_success_source_family_count: 3,
            qualified_success_source_root_count: 3,
            untrusted_event_count: 0,
            unknown_trust_event_count: 0,
            qualified_success_event_ids: vec![100, 101, 102],
            max_qualified_success_measurement_error_bound: Some(0.02),
            scope_matches: true,
            all_trusted_predictions_pre_registered: true,
        }
    }

    #[test]
    fn three_independent_roots_produce_a_source_linked_proposal_only() {
        let decision = ConsolidationPolicyV1::default()
            .decide_procedure_promotion(&summary());
        let ConsolidationProposalV1::ProposeProcedure(proposal) = decision else {
            panic!("three independently supported successes should propose promotion");
        };
        assert_eq!(proposal.action, Action::A);
        assert_eq!(proposal.evidence_event_ids, vec![100, 101, 102]);
        assert_eq!(proposal.independent_source_root_count, 3);
        assert_eq!(proposal.independent_source_family_count, 3);
        assert_eq!(proposal.minimum_qualified_success_events, 3);
        assert_eq!(proposal.minimum_independent_source_families, 3);
        assert_eq!(proposal.minimum_independent_source_roots, 3);
        assert_eq!(proposal.maximum_measurement_error_bound, 0.05);
        assert_eq!(proposal.policy_revision, "mace-consolidation-proposal-v1");
    }

    #[test]
    fn three_family_labels_from_one_root_do_not_promote() {
        let mut evidence = summary();
        evidence.qualified_success_source_root_count = 1;
        let decision = ConsolidationPolicyV1::default()
            .decide_procedure_promotion(&evidence);
        assert_eq!(
            decision,
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InsufficientIndependentRoots
            )
        );
    }

    #[test]
    fn untrusted_or_unknown_sources_are_quarantined_before_promotion() {
        let mut untrusted = summary();
        untrusted.untrusted_event_count = 1;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&untrusted),
            ConsolidationProposalV1::Quarantine(
                QuarantineReason::UntrustedSourcePresent
            )
        );

        let mut unknown = summary();
        unknown.unknown_trust_event_count = 1;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&unknown),
            ConsolidationProposalV1::Quarantine(
                QuarantineReason::UnknownTrustStatePresent
            )
        );
    }

    #[test]
    fn scope_and_preregistration_are_required() {
        let mut wrong_scope = summary();
        wrong_scope.scope_matches = false;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&wrong_scope),
            ConsolidationProposalV1::RetainProvisional(HoldReason::ScopeMismatch)
        );

        let mut unregistered = summary();
        unregistered.all_trusted_predictions_pre_registered = false;
        assert_eq!(
            ConsolidationPolicyV1::default().decide_procedure_promotion(&unregistered),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::PreRegistrationMissing
            )
        );
    }

    #[test]
    fn error_bounds_must_be_present_finite_and_within_policy() {
        for bound in [None, Some(f64::NAN), Some(0.051)] {
            let mut evidence = summary();
            evidence.max_qualified_success_measurement_error_bound = bound;
            let expected = if bound.is_none() {
                HoldReason::MeasurementBoundMissing
            } else {
                HoldReason::MeasurementBoundTooWide
            };
            assert_eq!(
                ConsolidationPolicyV1::default().decide_procedure_promotion(&evidence),
                ConsolidationProposalV1::RetainProvisional(expected)
            );
        }
    }

    #[test]
    fn invalid_policy_configuration_cannot_lower_the_promotion_gate() {
        let mut policy = ConsolidationPolicyV1::default();
        policy.minimum_independent_source_roots = 0;
        assert_eq!(
            policy.decide_procedure_promotion(&summary()),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration
            )
        );

        let mut invalid_bound = ConsolidationPolicyV1::default();
        invalid_bound.maximum_measurement_error_bound = f64::NAN;
        assert_eq!(
            invalid_bound.decide_procedure_promotion(&summary()),
            ConsolidationProposalV1::RetainProvisional(
                HoldReason::InvalidPolicyConfiguration
            )
        );
    }

    #[test]
    fn global_summary_type_remains_decision_neutral() {
        let _type_check: Option<ConsolidationEvidenceSummaryV1> = None;
        let _trust_class: SourceTrustStatus = SourceTrustStatus::TrustedForScope;
    }
}
