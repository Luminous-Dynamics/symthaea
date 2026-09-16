// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, reviewable inquiry contracts derived from explicit knowledge gaps.
//!
//! An inquiry contract describes what evidence would reduce a known epistemic gap.
//! It is deliberately non-executing: contracts cannot browse, call tools, run an
//! experiment, mutate knowledge, change confidence, or influence action selection.
//! The only authority represented here is `ProposalOnly`.

use super::claim_evidence::{ClaimId, EpistemicLedger, EvidenceKind};
use super::epistemic_vector::UncertaintyDimension;
use super::ignorance_frontier::{IgnoranceFrontierReport, KnowledgeGap};
use std::error::Error;
use std::fmt;

/// Stable within one generated inquiry plan. This is not a persistent global ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InquiryContractId(pub u64);

/// What kind of epistemic work is being requested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InquiryRequest {
    /// Seek evidence that could support, contradict, or contextualize a claim.
    SeekDiscriminatingEvidence,
    /// Seek evidence with a declared provenance lineage not descending from the
    /// currently concentrated supporting lineage(s). This does not assert that
    /// the new source is statistically or institutionally independent.
    SeekAdditionalProvenanceLineage,
    /// Compare conflicting evidence using an explicit protocol designed to
    /// discriminate between the competing observations/claims.
    ResolveContradiction,
    /// Draft an intervention/replication protocol for later human/authority review.
    /// This request does not authorize execution of the intervention.
    DraftInterventionProtocol,
    /// Produce or refresh a multidimensional uncertainty assessment.
    AssessUncertainty {
        dimension: Option<UncertaintyDimension>,
    },
    /// Reassess uncertainty because the previous assessment did not identify an
    /// evidence basis or because newer evidence has arrived.
    RefreshEvidenceBoundAssessment,
}

/// What evidence classes are relevant to satisfying a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceTarget {
    pub accepted_kinds: Vec<EvidenceKind>,
    /// `true` means evidence that only repeats the same declared ultimate
    /// provenance root is insufficient for this contract.
    pub require_new_declared_provenance_root: bool,
    /// `true` means the evidence should be capable of distinguishing competing
    /// possibilities rather than merely adding another agreeing report.
    pub discriminating: bool,
}

/// Authority boundary attached to every generated contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InquiryAuthority {
    /// May be displayed, discussed, revised, or approved by an external authority,
    /// but this module grants no permission to execute it.
    ProposalOnly,
}

/// Human/authority review requirements carried with the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewRequirement {
    /// Review is required before any external side effect or experiment could be
    /// delegated to a separate execution system.
    RequiredBeforeExternalAction,
}

/// One auditable proposal for reducing a specific gap on a specific claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InquiryContract {
    pub id: InquiryContractId,
    pub claim_id: ClaimId,
    /// Exact gap from which the contract was derived; retained for auditability.
    pub source_gap: KnowledgeGap,
    pub request: InquiryRequest,
    pub evidence_target: EvidenceTarget,
    /// Human-readable objective copied from the claim and gap class. It is a
    /// proposal description, not a command to an external system.
    pub objective: String,
    pub authority: InquiryAuthority,
    pub review_requirement: ReviewRequirement,
    /// Hard-coded false in all constructors in this module. The field is explicit
    /// so downstream code can test the authority boundary rather than infer it.
    pub external_side_effects_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InquiryPlan {
    pub contracts: Vec<InquiryContract>,
}

impl InquiryPlan {
    pub fn len(&self) -> usize {
        self.contracts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }

    /// Every contract emitted by this module is proposal-only and non-executing.
    pub fn is_non_executing(&self) -> bool {
        self.contracts.iter().all(|contract| {
            contract.authority == InquiryAuthority::ProposalOnly
                && contract.review_requirement
                    == ReviewRequirement::RequiredBeforeExternalAction
                && !contract.external_side_effects_authorized
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InquiryContractError {
    UnknownClaim(ClaimId),
}

impl fmt::Display for InquiryContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownClaim(id) => write!(f, "unknown claim id {}", id.0),
        }
    }
}

impl Error for InquiryContractError {}

/// Pure mapper from a diagnostic ignorance frontier to non-executing inquiry
/// proposals. It does not modify either input.
pub struct InquiryContractBuilder;

impl InquiryContractBuilder {
    pub fn from_frontier(
        ledger: &EpistemicLedger,
        frontier: &IgnoranceFrontierReport,
    ) -> Result<InquiryPlan, InquiryContractError> {
        let mut contracts = Vec::new();
        let mut next_id = 1u64;

        for profile in &frontier.profiles {
            let claim = ledger
                .claim(profile.claim_id)
                .ok_or(InquiryContractError::UnknownClaim(profile.claim_id))?;

            for gap in &profile.gaps {
                let (request, evidence_target, objective) = contract_for_gap(
                    &claim.statement,
                    gap,
                );
                contracts.push(InquiryContract {
                    id: InquiryContractId(next_id),
                    claim_id: profile.claim_id,
                    source_gap: gap.clone(),
                    request,
                    evidence_target,
                    objective,
                    authority: InquiryAuthority::ProposalOnly,
                    review_requirement: ReviewRequirement::RequiredBeforeExternalAction,
                    external_side_effects_authorized: false,
                });
                next_id += 1;
            }
        }

        Ok(InquiryPlan { contracts })
    }
}

fn contract_for_gap(
    claim_statement: &str,
    gap: &KnowledgeGap,
) -> (InquiryRequest, EvidenceTarget, String) {
    match gap {
        KnowledgeGap::NoEvidence | KnowledgeGap::NoSupportingEvidence => (
            InquiryRequest::SeekDiscriminatingEvidence,
            EvidenceTarget {
                accepted_kinds: vec![
                    EvidenceKind::Observation,
                    EvidenceKind::Measurement,
                    EvidenceKind::ToolResult,
                    EvidenceKind::Deduction,
                    EvidenceKind::Report,
                ],
                require_new_declared_provenance_root: false,
                discriminating: true,
            },
            format!(
                "Seek evidence capable of supporting, contradicting, or contextualizing the claim: {claim_statement}"
            ),
        ),
        KnowledgeGap::SharedSupportingProvenanceAncestry { .. } => (
            InquiryRequest::SeekAdditionalProvenanceLineage,
            EvidenceTarget {
                accepted_kinds: vec![
                    EvidenceKind::Observation,
                    EvidenceKind::Measurement,
                    EvidenceKind::ToolResult,
                    EvidenceKind::Report,
                    EvidenceKind::Replication,
                ],
                require_new_declared_provenance_root: true,
                discriminating: true,
            },
            format!(
                "Seek relevant evidence for '{claim_statement}' from a declared provenance lineage not descending from the currently concentrated supporting roots"
            ),
        ),
        KnowledgeGap::ContradictoryEvidence { .. } => (
            InquiryRequest::ResolveContradiction,
            EvidenceTarget {
                accepted_kinds: vec![
                    EvidenceKind::Measurement,
                    EvidenceKind::Observation,
                    EvidenceKind::ToolResult,
                    EvidenceKind::Replication,
                    EvidenceKind::Intervention,
                ],
                require_new_declared_provenance_root: false,
                discriminating: true,
            },
            format!(
                "Design a comparison that can discriminate among the conflicting evidence attached to: {claim_statement}"
            ),
        ),
        KnowledgeGap::NoInterventionalSupport => (
            InquiryRequest::DraftInterventionProtocol,
            EvidenceTarget {
                accepted_kinds: vec![EvidenceKind::Intervention, EvidenceKind::Replication],
                require_new_declared_provenance_root: false,
                discriminating: true,
            },
            format!(
                "Draft a reversible, bounded intervention or replication protocol that could test the causal claim: {claim_statement}"
            ),
        ),
        KnowledgeGap::NoUncertaintyAssessment => (
            InquiryRequest::AssessUncertainty { dimension: None },
            EvidenceTarget {
                accepted_kinds: vec![],
                require_new_declared_provenance_root: false,
                discriminating: false,
            },
            format!(
                "Assess epistemic, aleatoric, ontological, and distribution-shift uncertainty for: {claim_statement}"
            ),
        ),
        KnowledgeGap::UnassessedUncertaintyDimension(dimension) => (
            InquiryRequest::AssessUncertainty {
                dimension: Some(*dimension),
            },
            EvidenceTarget {
                accepted_kinds: vec![],
                require_new_declared_provenance_root: false,
                discriminating: false,
            },
            format!(
                "Assess {dimension:?} uncertainty for the claim: {claim_statement}"
            ),
        ),
        KnowledgeGap::AssessmentHasNoEvidenceBasis
        | KnowledgeGap::AssessmentPredatesEvidence { .. } => (
            InquiryRequest::RefreshEvidenceBoundAssessment,
            EvidenceTarget {
                accepted_kinds: vec![],
                require_new_declared_provenance_root: false,
                discriminating: false,
            },
            format!(
                "Refresh the uncertainty assessment for '{claim_statement}' using an explicit, current evidence basis"
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimKind, EvidenceKind, EvidencePolarity, IgnoranceFrontier,
    };

    #[test]
    fn every_generated_contract_is_proposal_only_and_non_executing() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();

        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        assert!(!plan.is_empty());
        assert!(plan.is_non_executing());
        assert!(plan
            .contracts
            .iter()
            .all(|contract| !contract.external_side_effects_authorized));
    }

    #[test]
    fn causal_gap_proposes_protocol_draft_not_intervention_execution() {
        let mut ledger = EpistemicLedger::new();
        let source = ledger
            .add_provenance("paper", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                source,
                2,
                None,
                None,
            )
            .unwrap();
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();

        let contract = plan
            .contracts
            .iter()
            .find(|contract| {
                matches!(contract.source_gap, KnowledgeGap::NoInterventionalSupport)
            })
            .unwrap();
        assert_eq!(contract.request, InquiryRequest::DraftInterventionProtocol);
        assert!(!contract.external_side_effects_authorized);
    }

    #[test]
    fn contradiction_requests_discriminating_evidence_not_confirmation() {
        let mut ledger = EpistemicLedger::new();
        let a = ledger
            .add_provenance("a", None, None, 1, vec![])
            .unwrap();
        let b = ledger
            .add_provenance("b", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                a,
                2,
                None,
                None,
            )
            .unwrap();
        ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Contradicts,
                b,
                3,
                None,
                None,
            )
            .unwrap();
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();

        let contract = plan
            .contracts
            .iter()
            .find(|contract| {
                matches!(contract.source_gap, KnowledgeGap::ContradictoryEvidence { .. })
            })
            .unwrap();
        assert_eq!(contract.request, InquiryRequest::ResolveContradiction);
        assert!(contract.evidence_target.discriminating);
    }

    #[test]
    fn shared_ancestry_requests_new_declared_lineage_without_claiming_independence() {
        let mut ledger = EpistemicLedger::new();
        let original = ledger
            .add_provenance("original", None, None, 1, vec![])
            .unwrap();
        let copy_a = ledger
            .add_provenance("copy-a", None, None, 2, vec![original])
            .unwrap();
        let copy_b = ledger
            .add_provenance("copy-b", None, None, 2, vec![original])
            .unwrap();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        for source in [copy_a, copy_b] {
            ledger
                .add_evidence(
                    claim,
                    EvidenceKind::Report,
                    EvidencePolarity::Supports,
                    source,
                    3,
                    None,
                    None,
                )
                .unwrap();
        }
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();
        let plan = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();

        let contract = plan
            .contracts
            .iter()
            .find(|contract| {
                matches!(
                    contract.source_gap,
                    KnowledgeGap::SharedSupportingProvenanceAncestry { .. }
                )
            })
            .unwrap();
        assert_eq!(contract.request, InquiryRequest::SeekAdditionalProvenanceLineage);
        assert!(contract.evidence_target.require_new_declared_provenance_root);
    }

    #[test]
    fn building_a_plan_does_not_mutate_the_ledger() {
        let mut ledger = EpistemicLedger::new();
        let claim = ledger.add_claim("X exists", ClaimKind::Descriptive, None, None, 1);
        let claim_count = ledger.claim_count();
        let evidence_count = ledger.evidence_count();
        let provenance_count = ledger.provenance_count();
        let frontier = IgnoranceFrontier::inspect(&ledger, &[claim], &[]).unwrap();

        let _ = InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap();
        assert_eq!(ledger.claim_count(), claim_count);
        assert_eq!(ledger.evidence_count(), evidence_count);
        assert_eq!(ledger.provenance_count(), provenance_count);
    }

    #[test]
    fn fabricated_frontier_with_unknown_claim_fails_closed() {
        use crate::knowledge::ClaimIgnoranceProfile;

        let ledger = EpistemicLedger::new();
        let frontier = IgnoranceFrontierReport {
            profiles: vec![ClaimIgnoranceProfile {
                claim_id: ClaimId(999),
                evidence_count: 0,
                supporting_evidence_count: 0,
                supporting_distinct_provenance_root_count: 0,
                contradicting_evidence_count: 0,
                interventional_support_count: 0,
                assessed_uncertainty_dimension_count: 0,
                gaps: vec![KnowledgeGap::NoEvidence],
            }],
            total_gap_count: 1,
        };

        assert_eq!(
            InquiryContractBuilder::from_frontier(&ledger, &frontier).unwrap_err(),
            InquiryContractError::UnknownClaim(ClaimId(999))
        );
    }
}
