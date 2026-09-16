// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Result receipts bound to frozen inquiry preregistrations.
//!
//! A receipt records which preregistered decision region was applied to an
//! observed result, together with evidence kind and provenance. It is deliberately
//! descriptive: creating a receipt does not insert evidence into the epistemic
//! ledger, update confidence, mutate a causal model, or authorize external action.

use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceKind, ProvenanceId};
use super::inquiry_contract::{InquiryContract, InquiryContractId};
use super::inquiry_preregistration::{
    DecisionInterpretation, InquiryPreregistration, PreregisteredDecisionRule,
};
use std::error::Error;
use std::fmt;

/// Immutable receipt tying one observed result to a decision rule that existed
/// before that result was observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InquiryResultReceipt {
    contract_id: InquiryContractId,
    claim_id: ClaimId,
    evidence_kind: EvidenceKind,
    provenance_id: ProvenanceId,
    observed_at_cycle: u64,
    preregistered_at_cycle: u64,
    result_summary: String,
    decision_rule_label: String,
    decision_criterion: String,
    interpretation: DecisionInterpretation,
    /// Always false in this module. A later, separately qualified ingestion
    /// policy must decide whether/how a receipt becomes ledger evidence.
    ledger_update_authorized: bool,
    /// Always false. Recording an outcome never grants authority for another
    /// external action or experiment.
    external_action_authorized: bool,
}

impl InquiryResultReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ledger: &EpistemicLedger,
        contract: &InquiryContract,
        preregistration: &InquiryPreregistration,
        evidence_kind: EvidenceKind,
        provenance_id: ProvenanceId,
        observed_at_cycle: u64,
        result_summary: impl Into<String>,
        decision_rule_label: &str,
    ) -> Result<Self, InquiryResultError> {
        if contract.id != preregistration.contract_id()
            || contract.claim_id != preregistration.claim_id()
        {
            return Err(InquiryResultError::ContractPreregistrationMismatch);
        }

        if ledger.provenance(provenance_id).is_none() {
            return Err(InquiryResultError::UnknownProvenance(provenance_id));
        }

        if observed_at_cycle <= preregistration.registered_at_cycle() {
            return Err(InquiryResultError::ResultNotAfterPreregistration {
                preregistered_at_cycle: preregistration.registered_at_cycle(),
                observed_at_cycle,
            });
        }

        if !contract.evidence_target.accepted_kinds.is_empty()
            && !contract.evidence_target.accepted_kinds.contains(&evidence_kind)
        {
            return Err(InquiryResultError::EvidenceKindOutsideContract(evidence_kind));
        }

        let result_summary = result_summary.into();
        if result_summary.trim().is_empty() {
            return Err(InquiryResultError::EmptyResultSummary);
        }

        let decision_rule = find_decision_rule(preregistration, decision_rule_label)
            .ok_or_else(|| InquiryResultError::UnknownDecisionRule(decision_rule_label.into()))?;

        Ok(Self {
            contract_id: contract.id,
            claim_id: contract.claim_id,
            evidence_kind,
            provenance_id,
            observed_at_cycle,
            preregistered_at_cycle: preregistration.registered_at_cycle(),
            result_summary,
            decision_rule_label: decision_rule.label.clone(),
            decision_criterion: decision_rule.criterion.clone(),
            interpretation: decision_rule.interpretation,
            ledger_update_authorized: false,
            external_action_authorized: false,
        })
    }

    pub fn contract_id(&self) -> InquiryContractId {
        self.contract_id
    }

    pub fn claim_id(&self) -> ClaimId {
        self.claim_id
    }

    pub fn evidence_kind(&self) -> EvidenceKind {
        self.evidence_kind
    }

    pub fn provenance_id(&self) -> ProvenanceId {
        self.provenance_id
    }

    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }

    pub fn preregistered_at_cycle(&self) -> u64 {
        self.preregistered_at_cycle
    }

    pub fn result_summary(&self) -> &str {
        &self.result_summary
    }

    pub fn decision_rule_label(&self) -> &str {
        &self.decision_rule_label
    }

    pub fn decision_criterion(&self) -> &str {
        &self.decision_criterion
    }

    pub fn interpretation(&self) -> DecisionInterpretation {
        self.interpretation
    }

    pub fn ledger_update_authorized(&self) -> bool {
        self.ledger_update_authorized
    }

    pub fn external_action_authorized(&self) -> bool {
        self.external_action_authorized
    }
}

fn find_decision_rule<'a>(
    preregistration: &'a InquiryPreregistration,
    label: &str,
) -> Option<&'a PreregisteredDecisionRule> {
    let requested = label.trim();
    preregistration
        .decision_rules()
        .iter()
        .find(|rule| rule.label.trim().eq_ignore_ascii_case(requested))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InquiryResultError {
    ContractPreregistrationMismatch,
    UnknownProvenance(ProvenanceId),
    ResultNotAfterPreregistration {
        preregistered_at_cycle: u64,
        observed_at_cycle: u64,
    },
    EvidenceKindOutsideContract(EvidenceKind),
    EmptyResultSummary,
    UnknownDecisionRule(String),
}

impl fmt::Display for InquiryResultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContractPreregistrationMismatch => {
                write!(f, "inquiry contract does not match the preregistration")
            }
            Self::UnknownProvenance(id) => write!(f, "unknown provenance id {}", id.0),
            Self::ResultNotAfterPreregistration {
                preregistered_at_cycle,
                observed_at_cycle,
            } => write!(
                f,
                "result cycle {observed_at_cycle} must be after preregistration cycle {preregistered_at_cycle}"
            ),
            Self::EvidenceKindOutsideContract(kind) => {
                write!(f, "evidence kind {kind:?} is outside the inquiry contract")
            }
            Self::EmptyResultSummary => write!(f, "result summary cannot be empty"),
            Self::UnknownDecisionRule(label) => {
                write!(f, "unknown preregistered decision-rule label '{label}'")
            }
        }
    }
}

impl Error for InquiryResultError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, IgnoranceFrontier, InquiryContractBuilder, InquiryRequest,
        PreregisteredDecisionRule,
    };

    fn rules() -> Vec<PreregisteredDecisionRule> {
        vec![
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
            PreregisteredDecisionRule::new(
                "inconclusive",
                "otherwise",
                DecisionInterpretation::Inconclusive,
            ),
        ]
    }

    fn discriminating_fixture() -> (
        EpistemicLedger,
        InquiryContract,
        InquiryPreregistration,
        ProvenanceId,
    ) {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("measurement-source", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        let contract = plan
            .contracts
            .into_iter()
            .find(|contract| contract.request == InquiryRequest::SeekDiscriminatingEvidence)
            .unwrap();
        let preregistration = InquiryPreregistration::new(
            &contract,
            "bounded comparison",
            "prediction error delta",
            rules(),
            "stop after fixed sample budget",
            vec![],
            vec![],
            5,
        )
        .unwrap();
        (ledger, contract, preregistration, provenance)
    }

    #[test]
    fn receipt_binds_result_to_frozen_rule_without_mutating_ledger() {
        let (ledger, contract, preregistration, provenance) = discriminating_fixture();
        let claims_before = ledger.claim_count();
        let evidence_before = ledger.evidence_count();
        let provenance_before = ledger.provenance_count();

        let receipt = InquiryResultReceipt::new(
            &ledger,
            &contract,
            &preregistration,
            EvidenceKind::Measurement,
            provenance,
            6,
            "prediction error delta exceeded the upper boundary",
            "supports",
        )
        .unwrap();

        assert_eq!(receipt.interpretation(), DecisionInterpretation::SupportsClaim);
        assert_eq!(receipt.decision_criterion(), "measure > upper");
        assert!(!receipt.ledger_update_authorized());
        assert!(!receipt.external_action_authorized());
        assert_eq!(ledger.claim_count(), claims_before);
        assert_eq!(ledger.evidence_count(), evidence_before);
        assert_eq!(ledger.provenance_count(), provenance_before);
    }

    #[test]
    fn result_must_strictly_follow_preregistration() {
        let (ledger, contract, preregistration, provenance) = discriminating_fixture();
        assert_eq!(
            InquiryResultReceipt::new(
                &ledger,
                &contract,
                &preregistration,
                EvidenceKind::Measurement,
                provenance,
                5,
                "result",
                "supports",
            )
            .unwrap_err(),
            InquiryResultError::ResultNotAfterPreregistration {
                preregistered_at_cycle: 5,
                observed_at_cycle: 5,
            }
        );
    }

    #[test]
    fn unknown_decision_rule_fails_closed() {
        let (ledger, contract, preregistration, provenance) = discriminating_fixture();
        assert_eq!(
            InquiryResultReceipt::new(
                &ledger,
                &contract,
                &preregistration,
                EvidenceKind::Measurement,
                provenance,
                6,
                "result",
                "invented-after-seeing-result",
            )
            .unwrap_err(),
            InquiryResultError::UnknownDecisionRule(
                "invented-after-seeing-result".into()
            )
        );
    }

    #[test]
    fn inconclusive_outcome_remains_explicit() {
        let (ledger, contract, preregistration, provenance) = discriminating_fixture();
        let receipt = InquiryResultReceipt::new(
            &ledger,
            &contract,
            &preregistration,
            EvidenceKind::Observation,
            provenance,
            6,
            "measure remained inside the preregistered middle region",
            "INCONCLUSIVE",
        )
        .unwrap();
        assert_eq!(receipt.interpretation(), DecisionInterpretation::Inconclusive);
    }

    #[test]
    fn evidence_kind_must_be_allowed_by_contract() {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("source", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        let contract = plan
            .contracts
            .into_iter()
            .find(|contract| contract.request == InquiryRequest::DraftInterventionProtocol)
            .unwrap();
        let preregistration = InquiryPreregistration::new(
            &contract,
            "bounded intervention design",
            "outcome delta",
            rules(),
            "fixed budget",
            vec![],
            vec![],
            5,
        )
        .unwrap();

        assert_eq!(
            InquiryResultReceipt::new(
                &ledger,
                &contract,
                &preregistration,
                EvidenceKind::Report,
                provenance,
                6,
                "reported result",
                "supports",
            )
            .unwrap_err(),
            InquiryResultError::EvidenceKindOutsideContract(EvidenceKind::Report)
        );
    }

    #[test]
    fn unknown_provenance_fails_closed() {
        let (ledger, contract, preregistration, _) = discriminating_fixture();
        assert_eq!(
            InquiryResultReceipt::new(
                &ledger,
                &contract,
                &preregistration,
                EvidenceKind::Measurement,
                ProvenanceId(999),
                6,
                "result",
                "supports",
            )
            .unwrap_err(),
            InquiryResultError::UnknownProvenance(ProvenanceId(999))
        );
    }
}
