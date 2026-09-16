// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable preregistration for claim-discriminating inquiry proposals.
//!
//! The decision rule is fixed before evidence arrives. A valid preregistration
//! must define outcomes that would support the claim, contradict the claim, and
//! remain inconclusive. This prevents a later result from being forced into a
//! preferred interpretation after the fact.
//!
//! This module remains non-executing. A preregistered intervention protocol is
//! still only a proposal and carries no authority to perform an intervention.

use super::claim_evidence::ClaimId;
use super::ignorance_frontier::KnowledgeGap;
use super::inquiry_contract::{
    InquiryAuthority, InquiryContract, InquiryContractId, InquiryRequest, ReviewRequirement,
};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/// Interpretation fixed before observing a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecisionInterpretation {
    SupportsClaim,
    ContradictsClaim,
    Inconclusive,
}

/// One preregistered decision region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreregisteredDecisionRule {
    /// Stable human-readable label within this preregistration.
    pub label: String,
    /// Frozen criterion describing which observations fall in this region.
    pub criterion: String,
    pub interpretation: DecisionInterpretation,
}

impl PreregisteredDecisionRule {
    pub fn new(
        label: impl Into<String>,
        criterion: impl Into<String>,
        interpretation: DecisionInterpretation,
    ) -> Self {
        Self {
            label: label.into(),
            criterion: criterion.into(),
            interpretation,
        }
    }
}

/// Frozen protocol metadata for one claim-discriminating inquiry contract.
///
/// Fields are private so downstream code cannot silently mutate the decision
/// rule after evidence arrives. A changed protocol must be represented by a new
/// preregistration value rather than editing this one in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InquiryPreregistration {
    contract_id: InquiryContractId,
    claim_id: ClaimId,
    source_gap: KnowledgeGap,
    protocol: String,
    primary_measure: String,
    decision_rules: Vec<PreregisteredDecisionRule>,
    stopping_rule: String,
    exclusions: Vec<String>,
    confounders: Vec<String>,
    registered_at_cycle: u64,
    authority: InquiryAuthority,
    review_requirement: ReviewRequirement,
    external_side_effects_authorized: bool,
}

impl InquiryPreregistration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract: &InquiryContract,
        protocol: impl Into<String>,
        primary_measure: impl Into<String>,
        decision_rules: Vec<PreregisteredDecisionRule>,
        stopping_rule: impl Into<String>,
        exclusions: Vec<String>,
        confounders: Vec<String>,
        registered_at_cycle: u64,
    ) -> Result<Self, PreregistrationError> {
        if contract.authority != InquiryAuthority::ProposalOnly
            || contract.review_requirement != ReviewRequirement::RequiredBeforeExternalAction
            || contract.external_side_effects_authorized
        {
            return Err(PreregistrationError::AuthorityBoundaryViolated);
        }

        if !matches!(
            contract.request,
            InquiryRequest::SeekDiscriminatingEvidence
                | InquiryRequest::ResolveContradiction
                | InquiryRequest::DraftInterventionProtocol
        ) {
            return Err(PreregistrationError::UnsupportedInquiryRequest);
        }

        let protocol = protocol.into();
        if protocol.trim().is_empty() {
            return Err(PreregistrationError::EmptyProtocol);
        }
        let primary_measure = primary_measure.into();
        if primary_measure.trim().is_empty() {
            return Err(PreregistrationError::EmptyPrimaryMeasure);
        }
        let stopping_rule = stopping_rule.into();
        if stopping_rule.trim().is_empty() {
            return Err(PreregistrationError::EmptyStoppingRule);
        }

        validate_decision_rules(&decision_rules)?;

        Ok(Self {
            contract_id: contract.id,
            claim_id: contract.claim_id,
            source_gap: contract.source_gap.clone(),
            protocol,
            primary_measure,
            decision_rules,
            stopping_rule,
            exclusions,
            confounders,
            registered_at_cycle,
            authority: InquiryAuthority::ProposalOnly,
            review_requirement: ReviewRequirement::RequiredBeforeExternalAction,
            external_side_effects_authorized: false,
        })
    }

    pub fn contract_id(&self) -> InquiryContractId {
        self.contract_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn source_gap(&self) -> &KnowledgeGap {
        &self.source_gap
    }

    pub fn protocol(&self) -> &str {
        &self.protocol
    }

    pub fn primary_measure(&self) -> &str {
        &self.primary_measure
    }

    pub fn decision_rules(&self) -> &[PreregisteredDecisionRule] {
        &self.decision_rules
    }

    pub fn stopping_rule(&self) -> &str {
        &self.stopping_rule
    }

    pub fn exclusions(&self) -> &[String] {
        &self.exclusions
    }

    pub fn confounders(&self) -> &[String] {
        &self.confounders
    }

    pub fn registered_at_cycle(&self) -> u64 {
        self.registered_at_cycle
    }

    pub fn authority(&self) -> InquiryAuthority {
        self.authority
    }

    pub fn review_requirement(&self) -> ReviewRequirement {
        self.review_requirement
    }

    pub fn external_side_effects_authorized(&self) -> bool {
        self.external_side_effects_authorized
    }
}

fn validate_decision_rules(
    rules: &[PreregisteredDecisionRule],
) -> Result<(), PreregistrationError> {
    let mut labels = HashSet::new();
    let mut has_support = false;
    let mut has_contradiction = false;
    let mut has_inconclusive = false;

    for rule in rules {
        let label = rule.label.trim();
        if label.is_empty() {
            return Err(PreregistrationError::EmptyDecisionRuleLabel);
        }
        if rule.criterion.trim().is_empty() {
            return Err(PreregistrationError::EmptyDecisionCriterion {
                label: rule.label.clone(),
            });
        }
        let normalized_label = label.to_lowercase();
        if !labels.insert(normalized_label) {
            return Err(PreregistrationError::DuplicateDecisionRuleLabel(
                rule.label.clone(),
            ));
        }

        match rule.interpretation {
            DecisionInterpretation::SupportsClaim => has_support = true,
            DecisionInterpretation::ContradictsClaim => has_contradiction = true,
            DecisionInterpretation::Inconclusive => has_inconclusive = true,
        }
    }

    for (present, interpretation) in [
        (has_support, DecisionInterpretation::SupportsClaim),
        (has_contradiction, DecisionInterpretation::ContradictsClaim),
        (has_inconclusive, DecisionInterpretation::Inconclusive),
    ] {
        if !present {
            return Err(PreregistrationError::MissingDecisionRegion(interpretation));
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreregistrationError {
    AuthorityBoundaryViolated,
    UnsupportedInquiryRequest,
    EmptyProtocol,
    EmptyPrimaryMeasure,
    EmptyStoppingRule,
    EmptyDecisionRuleLabel,
    EmptyDecisionCriterion { label: String },
    DuplicateDecisionRuleLabel(String),
    MissingDecisionRegion(DecisionInterpretation),
}

impl fmt::Display for PreregistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorityBoundaryViolated => {
                write!(f, "inquiry contract does not satisfy the proposal-only authority boundary")
            }
            Self::UnsupportedInquiryRequest => write!(
                f,
                "only claim-discriminating inquiry requests can be preregistered here"
            ),
            Self::EmptyProtocol => write!(f, "preregistered protocol cannot be empty"),
            Self::EmptyPrimaryMeasure => write!(f, "primary measure cannot be empty"),
            Self::EmptyStoppingRule => write!(f, "stopping rule cannot be empty"),
            Self::EmptyDecisionRuleLabel => write!(f, "decision-rule label cannot be empty"),
            Self::EmptyDecisionCriterion { label } => {
                write!(f, "decision rule '{label}' has an empty criterion")
            }
            Self::DuplicateDecisionRuleLabel(label) => {
                write!(f, "duplicate decision-rule label '{label}'")
            }
            Self::MissingDecisionRegion(interpretation) => {
                write!(f, "missing preregistered {interpretation:?} decision region")
            }
        }
    }
}

impl Error for PreregistrationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, EpistemicLedger, IgnoranceFrontier, InquiryContractBuilder,
    };

    fn causal_contract() -> InquiryContract {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        plan.contracts
            .into_iter()
            .find(|contract| {
                contract.request == InquiryRequest::DraftInterventionProtocol
            })
            .unwrap()
    }

    fn three_way_rules() -> Vec<PreregisteredDecisionRule> {
        vec![
            PreregisteredDecisionRule::new(
                "supports",
                "primary measure is above the preregistered upper boundary",
                DecisionInterpretation::SupportsClaim,
            ),
            PreregisteredDecisionRule::new(
                "contradicts",
                "primary measure is below the preregistered lower boundary",
                DecisionInterpretation::ContradictsClaim,
            ),
            PreregisteredDecisionRule::new(
                "inconclusive",
                "primary measure lies between the two boundaries or data quality fails",
                DecisionInterpretation::Inconclusive,
            ),
        ]
    }

    #[test]
    fn valid_three_way_preregistration_remains_non_executing() {
        let contract = causal_contract();
        let prereg = InquiryPreregistration::new(
            &contract,
            "randomized bounded comparison",
            "difference in outcome rate",
            three_way_rules(),
            "stop after the preregistered sample budget",
            vec!["invalid instrumentation".into()],
            vec!["baseline drift".into()],
            10,
        )
        .unwrap();

        assert_eq!(prereg.contract_id(), contract.id);
        assert_eq!(prereg.claim_id(), contract.claim_id);
        assert_eq!(prereg.decision_rules().len(), 3);
        assert_eq!(prereg.authority(), InquiryAuthority::ProposalOnly);
        assert_eq!(
            prereg.review_requirement(),
            ReviewRequirement::RequiredBeforeExternalAction
        );
        assert!(!prereg.external_side_effects_authorized());
    }

    #[test]
    fn inconclusive_region_is_mandatory() {
        let contract = causal_contract();
        let rules = vec![
            PreregisteredDecisionRule::new(
                "supports",
                "measure > upper",
                DecisionInterpretation::SupportsClaim,
            ),
            PreregisteredDecisionRule::new(
                "contradicts",
                "measure < lower",
                DecisionInterpretation::ContradictsClaim,
            ),
        ];

        assert_eq!(
            InquiryPreregistration::new(
                &contract,
                "protocol",
                "measure",
                rules,
                "stop rule",
                vec![],
                vec![],
                1,
            )
            .unwrap_err(),
            PreregistrationError::MissingDecisionRegion(
                DecisionInterpretation::Inconclusive
            )
        );
    }

    #[test]
    fn one_sided_confirmation_rule_is_rejected() {
        let contract = causal_contract();
        let rules = vec![
            PreregisteredDecisionRule::new(
                "supports",
                "measure > threshold",
                DecisionInterpretation::SupportsClaim,
            ),
            PreregisteredDecisionRule::new(
                "unclear",
                "otherwise",
                DecisionInterpretation::Inconclusive,
            ),
        ];

        assert_eq!(
            InquiryPreregistration::new(
                &contract,
                "protocol",
                "measure",
                rules,
                "stop rule",
                vec![],
                vec![],
                1,
            )
            .unwrap_err(),
            PreregistrationError::MissingDecisionRegion(
                DecisionInterpretation::ContradictsClaim
            )
        );
    }

    #[test]
    fn uncertainty_assessment_contract_is_out_of_scope_for_claim_discrimination() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        let uncertainty_contract = plan
            .contracts
            .iter()
            .find(|contract| {
                matches!(contract.request, InquiryRequest::AssessUncertainty { .. })
            })
            .unwrap();

        assert_eq!(
            InquiryPreregistration::new(
                uncertainty_contract,
                "protocol",
                "measure",
                three_way_rules(),
                "stop rule",
                vec![],
                vec![],
                1,
            )
            .unwrap_err(),
            PreregistrationError::UnsupportedInquiryRequest
        );
    }

    #[test]
    fn duplicate_rule_labels_fail_closed_case_insensitively() {
        let contract = causal_contract();
        let mut rules = three_way_rules();
        rules.push(PreregisteredDecisionRule::new(
            "SUPPORTS",
            "another criterion",
            DecisionInterpretation::SupportsClaim,
        ));

        assert!(matches!(
            InquiryPreregistration::new(
                &contract,
                "protocol",
                "measure",
                rules,
                "stop rule",
                vec![],
                vec![],
                1,
            ),
            Err(PreregistrationError::DuplicateDecisionRuleLabel(_))
        ));
    }
}
