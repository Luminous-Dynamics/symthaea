// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Host-neutral four-way artistic counterfactual evidence and choice receipts.
//!
//! A choice is deliberately not a scalar optimization result. Each candidate
//! retains separate observed consequence dimensions, uncertainty, and evidence
//! references. The abstention baseline is first-class and every choice remains
//! bound to the exact revision/frame where the alternatives were observed.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::{ProposalId, RevisionId, StudioFrame};

pub const FOUR_WAY_CHOICE_SCHEMA_V1: &str = "symthaea.art-world.four-way-choice.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsequenceDimensionEvidence {
    pub dimension: String,
    pub baseline_value: Option<f64>,
    pub candidate_value: Option<f64>,
    pub observed_delta: Option<f64>,
    pub uncertainty: Option<f64>,
    pub evidence_refs: Vec<String>,
}

impl ConsequenceDimensionEvidence {
    pub fn validate(&self) -> Result<(), FourWayChoiceError> {
        if self.dimension.trim().is_empty() {
            return Err(FourWayChoiceError::EmptyDimension);
        }
        for value in [
            self.baseline_value,
            self.candidate_value,
            self.observed_delta,
            self.uncertainty,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() {
                return Err(FourWayChoiceError::NonFiniteEvidence(
                    self.dimension.clone(),
                ));
            }
        }
        if self.uncertainty.is_some_and(|value| value < 0.0) {
            return Err(FourWayChoiceError::NegativeUncertainty(
                self.dimension.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualCandidateEvidence {
    pub candidate_id: String,
    /// `None` marks the do-nothing / abstention baseline.
    pub proposal_id: Option<ProposalId>,
    pub dimensions: Vec<ConsequenceDimensionEvidence>,
    pub evidence_refs: Vec<String>,
}

impl CounterfactualCandidateEvidence {
    pub fn is_abstention(&self) -> bool {
        self.proposal_id.is_none()
    }

    pub fn validate(&self) -> Result<(), FourWayChoiceError> {
        if self.candidate_id.trim().is_empty() {
            return Err(FourWayChoiceError::EmptyCandidateId);
        }
        let mut dimensions = BTreeSet::new();
        for dimension in &self.dimensions {
            dimension.validate()?;
            if !dimensions.insert(dimension.dimension.as_str()) {
                return Err(FourWayChoiceError::DuplicateDimension(
                    dimension.dimension.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Exactly one abstention baseline plus exactly three proposal alternatives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FourWayCounterfactualSet {
    pub schema: String,
    pub base_revision: RevisionId,
    pub frame: StudioFrame,
    pub candidates: Vec<CounterfactualCandidateEvidence>,
}

impl FourWayCounterfactualSet {
    pub fn validate(&self) -> Result<(), FourWayChoiceError> {
        if self.schema != FOUR_WAY_CHOICE_SCHEMA_V1 {
            return Err(FourWayChoiceError::SchemaMismatch(self.schema.clone()));
        }
        if self.candidates.len() != 4 {
            return Err(FourWayChoiceError::RequiresExactlyFourCandidates);
        }

        let mut candidate_ids = BTreeSet::new();
        let mut proposal_ids = BTreeSet::new();
        let mut abstentions = 0usize;
        for candidate in &self.candidates {
            candidate.validate()?;
            if !candidate_ids.insert(candidate.candidate_id.as_str()) {
                return Err(FourWayChoiceError::DuplicateCandidateId(
                    candidate.candidate_id.clone(),
                ));
            }
            match &candidate.proposal_id {
                Some(proposal_id) => {
                    if !proposal_ids.insert(proposal_id.0.as_str()) {
                        return Err(FourWayChoiceError::DuplicateProposalId(
                            proposal_id.0.clone(),
                        ));
                    }
                }
                None => abstentions += 1,
            }
        }

        if abstentions != 1 || proposal_ids.len() != 3 {
            return Err(FourWayChoiceError::RequiresOneAbstentionThreeProposals);
        }
        Ok(())
    }

    pub fn candidate(&self, candidate_id: &str) -> Option<&CounterfactualCandidateEvidence> {
        self.candidates
            .iter()
            .find(|candidate| candidate.candidate_id == candidate_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtisticChoice {
    /// Select one rendered proposal. This is still not commit authority.
    SelectProposal {
        candidate_id: String,
        proposal_id: ProposalId,
    },
    /// Preserve the committed revision.
    Abstain { candidate_id: String },
    /// None of the current alternatives is adequate; create new proposals.
    Revise { candidate_ids: Vec<String> },
    /// Evidence is insufficient or contradictory.
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtisticChoiceReceipt {
    pub schema: String,
    pub base_revision: RevisionId,
    pub frame: StudioFrame,
    pub choice: ArtisticChoice,
    pub rationale: Option<String>,
    pub evidence_refs: Vec<String>,
}

impl ArtisticChoiceReceipt {
    pub fn validate_against(
        &self,
        alternatives: &FourWayCounterfactualSet,
    ) -> Result<(), FourWayChoiceError> {
        alternatives.validate()?;
        if self.schema != FOUR_WAY_CHOICE_SCHEMA_V1 {
            return Err(FourWayChoiceError::SchemaMismatch(self.schema.clone()));
        }
        if self.base_revision != alternatives.base_revision || self.frame != alternatives.frame {
            return Err(FourWayChoiceError::ChoiceObservationMisalignment);
        }

        match &self.choice {
            ArtisticChoice::SelectProposal {
                candidate_id,
                proposal_id,
            } => {
                let candidate = alternatives
                    .candidate(candidate_id)
                    .ok_or_else(|| FourWayChoiceError::UnknownCandidate(candidate_id.clone()))?;
                if candidate.proposal_id.as_ref() != Some(proposal_id) {
                    return Err(FourWayChoiceError::ChoiceProposalMismatch);
                }
            }
            ArtisticChoice::Abstain { candidate_id } => {
                let candidate = alternatives
                    .candidate(candidate_id)
                    .ok_or_else(|| FourWayChoiceError::UnknownCandidate(candidate_id.clone()))?;
                if !candidate.is_abstention() {
                    return Err(FourWayChoiceError::AbstentionCandidateIsProposal);
                }
            }
            ArtisticChoice::Revise { candidate_ids } => {
                if candidate_ids.is_empty() {
                    return Err(FourWayChoiceError::EmptyRevisionSet);
                }
                let mut seen = BTreeSet::new();
                for id in candidate_ids {
                    if !seen.insert(id.as_str()) {
                        return Err(FourWayChoiceError::DuplicateRevisionCandidate(id.clone()));
                    }
                    if alternatives.candidate(id).is_none() {
                        return Err(FourWayChoiceError::UnknownCandidate(id.clone()));
                    }
                }
            }
            ArtisticChoice::Inconclusive => {}
        }
        Ok(())
    }

    /// Deliberately returns only the selected proposal identity. A host must
    /// separately obtain normal art-world commit authority before mutation.
    pub fn selected_proposal(&self) -> Option<&ProposalId> {
        match &self.choice {
            ArtisticChoice::SelectProposal { proposal_id, .. } => Some(proposal_id),
            _ => None,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FourWayChoiceError {
    #[error("unsupported four-way choice schema: {0}")]
    SchemaMismatch(String),
    #[error("four-way comparison requires exactly four candidates")]
    RequiresExactlyFourCandidates,
    #[error("four-way comparison requires one abstention baseline and three proposals")]
    RequiresOneAbstentionThreeProposals,
    #[error("candidate id may not be empty")]
    EmptyCandidateId,
    #[error("duplicate candidate id: {0}")]
    DuplicateCandidateId(String),
    #[error("duplicate proposal id: {0}")]
    DuplicateProposalId(String),
    #[error("evidence dimension may not be empty")]
    EmptyDimension,
    #[error("evidence dimension contains non-finite data: {0}")]
    NonFiniteEvidence(String),
    #[error("evidence uncertainty may not be negative: {0}")]
    NegativeUncertainty(String),
    #[error("candidate contains duplicate evidence dimension: {0}")]
    DuplicateDimension(String),
    #[error("choice is not aligned to the observed revision/frame")]
    ChoiceObservationMisalignment,
    #[error("unknown candidate: {0}")]
    UnknownCandidate(String),
    #[error("selected candidate/proposal identities disagree")]
    ChoiceProposalMismatch,
    #[error("abstention choice points at a proposal candidate")]
    AbstentionCandidateIsProposal,
    #[error("revise choice requires at least one candidate")]
    EmptyRevisionSet,
    #[error("revise choice repeats candidate: {0}")]
    DuplicateRevisionCandidate(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, proposal: Option<&str>) -> CounterfactualCandidateEvidence {
        CounterfactualCandidateEvidence {
            candidate_id: id.into(),
            proposal_id: proposal.map(ProposalId::from),
            dimensions: vec![ConsequenceDimensionEvidence {
                dimension: "mean_luminance".into(),
                baseline_value: Some(0.4),
                candidate_value: Some(0.5),
                observed_delta: Some(0.1),
                uncertainty: None,
                evidence_refs: vec![format!("capture:{id}")],
            }],
            evidence_refs: vec![],
        }
    }

    fn alternatives() -> FourWayCounterfactualSet {
        FourWayCounterfactualSet {
            schema: FOUR_WAY_CHOICE_SCHEMA_V1.into(),
            base_revision: RevisionId::from("r1"),
            frame: StudioFrame(12),
            candidates: vec![
                candidate("baseline", None),
                candidate("a", Some("p1")),
                candidate("b", Some("p2")),
                candidate("c", Some("p3")),
            ],
        }
    }

    #[test]
    fn requires_one_baseline_and_three_unique_proposals() {
        assert!(alternatives().validate().is_ok());
        let mut bad = alternatives();
        bad.candidates[3].proposal_id = None;
        assert_eq!(
            bad.validate(),
            Err(FourWayChoiceError::RequiresOneAbstentionThreeProposals)
        );
    }

    #[test]
    fn selected_proposal_must_match_candidate_identity() {
        let receipt = ArtisticChoiceReceipt {
            schema: FOUR_WAY_CHOICE_SCHEMA_V1.into(),
            base_revision: RevisionId::from("r1"),
            frame: StudioFrame(12),
            choice: ArtisticChoice::SelectProposal {
                candidate_id: "a".into(),
                proposal_id: ProposalId::from("p2"),
            },
            rationale: None,
            evidence_refs: vec![],
        };
        assert_eq!(
            receipt.validate_against(&alternatives()),
            Err(FourWayChoiceError::ChoiceProposalMismatch)
        );
    }

    #[test]
    fn choice_does_not_create_commit_authority() {
        let receipt = ArtisticChoiceReceipt {
            schema: FOUR_WAY_CHOICE_SCHEMA_V1.into(),
            base_revision: RevisionId::from("r1"),
            frame: StudioFrame(12),
            choice: ArtisticChoice::SelectProposal {
                candidate_id: "a".into(),
                proposal_id: ProposalId::from("p1"),
            },
            rationale: Some("preserves silhouette while opening negative space".into()),
            evidence_refs: vec!["study:vart-ghost-001".into()],
        };
        receipt.validate_against(&alternatives()).unwrap();
        assert_eq!(receipt.selected_proposal().unwrap().0, "p1");
    }
}
