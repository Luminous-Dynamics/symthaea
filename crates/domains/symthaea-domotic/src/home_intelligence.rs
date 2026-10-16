// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pure proposal types and a deterministic structural validator for HOME-INTEL-001.
//!
//! Threat boundary: this module does not open sockets, hold device credentials, issue
//! Home Assistant service calls, issue Matter commands, call Holochain, or dispatch
//! physical effects. A structurally valid proposal is not policy authorization.
//! Provider-specific entity capability and household authority must be checked again
//! by Mycelix at any later effect-admission boundary.
//!
//! Raw provider attributes, free-form sensor payloads, audio and video intentionally
//! have no field in these types. The module is additive to the older domotic bridge;
//! it does not route the bridge's UDP telemetry into a proposal or actuation path.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HOME_PROPOSAL_SCHEMA_V1: u16 = 1;

const MAX_ID_BYTES: usize = 128;
const MAX_SUMMARY_BYTES: usize = 512;
const MAX_DETAIL_BYTES: usize = 256;
const MAX_EVIDENCE: usize = 32;
const MAX_DERIVATION_PARENTS: usize = 8;
const MAX_CANDIDATES: usize = 8;
const MAX_EFFECTS_PER_CANDIDATE: usize = 8;
const MAX_TEXT_ITEMS: usize = 16;
const MAX_TRADEOFFS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicStatus {
    Observed,
    Derived,
    Unavailable,
    Conflicting,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Stable, opaque ID for this evidence item within one proposal.
    pub evidence_id: String,
    /// Stable ID of an admitted source profile, not a URL or credential.
    pub source_ref: String,
    /// Opaque entity reference from an admitted provider profile.
    pub entity_ref: String,
    pub observed_at_ms: u64,
    /// The observation is no longer usable as current evidence at/after this time.
    pub fresh_until_ms: u64,
    /// Changes on reconnect/session reset; old epochs must not be treated as live.
    pub session_epoch: u64,
    pub status: EpistemicStatus,
    /// Derived facts list all parent evidence IDs before the derived item in the vector.
    pub derived_from: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalMode {
    Explain,
    Clarify,
    Recommend,
    Preview,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyLevel {
    Low,
    Moderate,
    High,
    Uncalibrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskClass {
    Low,
    Moderate,
    High,
    SafetyCritical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalRequirement {
    /// Informational only: no physical effect may be inferred from this value.
    None,
    ResidentConfirmation,
    MultiResidentConfirmation,
    SpecialistReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationTopic {
    ProviderUnavailable,
    EnergyAnomaly,
    RoutineConflict,
    MaintenanceReminder,
    Recommendation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposedEffect {
    /// Semantic proposal only; this is not a provider service-call payload.
    SetLightState {
        entity_ref: String,
        enabled: bool,
    },
    /// Ask the resident to consider a comfort target; never dispatch a thermostat write.
    SuggestComfortTarget {
        entity_ref: String,
        target_millicelsius: i16,
    },
    RequestMaintenanceReview {
        entity_ref: String,
    },
    NotifyResident {
        topic: NotificationTopic,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePlan {
    pub summary: String,
    pub effects: Vec<ProposedEffect>,
    pub tradeoffs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomeProposal {
    pub schema_version: u16,
    pub proposal_id: String,
    pub request_ref: String,
    pub generated_at_ms: u64,
    pub expires_at_ms: u64,
    pub mode: ProposalMode,
    pub summary: String,
    pub evidence: Vec<EvidenceRef>,
    pub assumptions: Vec<String>,
    pub unknowns: Vec<String>,
    pub uncertainty: UncertaintyLevel,
    pub risk_class: RiskClass,
    /// This states the approval required *if* a separate authorized workflow later
    /// considers an effect. It does not itself constitute approval or authority.
    pub approval_requirement: ApprovalRequirement,
    pub candidates: Vec<CandidatePlan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalValidationError {
    WrongSchemaVersion,
    InvalidIdentifier,
    InvalidText,
    TooManyItems,
    InvalidTimeWindow,
    FutureProposal,
    ExpiredProposal,
    MissingCurrentEpoch,
    SessionEpochMismatch,
    FutureEvidence,
    StaleEvidence,
    UnusableEvidence,
    InvalidDerivation,
    DuplicateEvidence,
    MissingEvidence,
    ModeCandidateMismatch,
    MissingClarificationTarget,
    InvalidCandidate,
    CandidateTooRisky,
    UncertaintyTooHighForCandidate,
    ApprovalRequired,
    ComfortTargetOutOfRange,
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_ID_BYTES
        && value.trim() == value
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_bytes
        && !value.chars().any(char::is_control)
}

fn validate_detail_list(values: &[String]) -> Result<(), ProposalValidationError> {
    if values.len() > MAX_TEXT_ITEMS {
        return Err(ProposalValidationError::TooManyItems);
    }
    if values.iter().any(|value| !valid_text(value, MAX_DETAIL_BYTES)) {
        return Err(ProposalValidationError::InvalidText);
    }
    Ok(())
}

/// Validate the proposal's bounded shape and its referenced evidence against a caller
/// supplied current session-epoch map. This is not a household-policy or actuation
/// authorization check.
pub fn validate_home_proposal(
    proposal: &HomeProposal,
    now_ms: u64,
    current_session_epochs: &BTreeMap<String, u64>,
) -> Result<(), ProposalValidationError> {
    if proposal.schema_version != HOME_PROPOSAL_SCHEMA_V1 {
        return Err(ProposalValidationError::WrongSchemaVersion);
    }
    if !valid_identifier(&proposal.proposal_id) || !valid_identifier(&proposal.request_ref) {
        return Err(ProposalValidationError::InvalidIdentifier);
    }
    if proposal.generated_at_ms > now_ms {
        return Err(ProposalValidationError::FutureProposal);
    }
    if proposal.expires_at_ms <= proposal.generated_at_ms {
        return Err(ProposalValidationError::InvalidTimeWindow);
    }
    if proposal.expires_at_ms <= now_ms {
        return Err(ProposalValidationError::ExpiredProposal);
    }
    if !valid_text(&proposal.summary, MAX_SUMMARY_BYTES) {
        return Err(ProposalValidationError::InvalidText);
    }
    validate_detail_list(&proposal.assumptions)?;
    validate_detail_list(&proposal.unknowns)?;

    if proposal.evidence.len() > MAX_EVIDENCE {
        return Err(ProposalValidationError::TooManyItems);
    }

    let mut evidence_ids = BTreeSet::new();
    for evidence in &proposal.evidence {
        if !valid_identifier(&evidence.evidence_id)
            || !valid_identifier(&evidence.source_ref)
            || !valid_identifier(&evidence.entity_ref)
        {
            return Err(ProposalValidationError::InvalidIdentifier);
        }
        if !evidence_ids.insert(evidence.evidence_id.as_str()) {
            return Err(ProposalValidationError::DuplicateEvidence);
        }
        if evidence.observed_at_ms > now_ms {
            return Err(ProposalValidationError::FutureEvidence);
        }
        if evidence.fresh_until_ms < evidence.observed_at_ms {
            return Err(ProposalValidationError::InvalidTimeWindow);
        }
        if evidence.fresh_until_ms <= now_ms {
            return Err(ProposalValidationError::StaleEvidence);
        }
        let current_epoch = current_session_epochs
            .get(&evidence.source_ref)
            .ok_or(ProposalValidationError::MissingCurrentEpoch)?;
        if *current_epoch != evidence.session_epoch {
            return Err(ProposalValidationError::SessionEpochMismatch);
        }

        match evidence.status {
            EpistemicStatus::Observed => {
                if !evidence.derived_from.is_empty() {
                    return Err(ProposalValidationError::InvalidDerivation);
                }
            }
            EpistemicStatus::Derived => {
                if evidence.derived_from.is_empty()
                    || evidence.derived_from.len() > MAX_DERIVATION_PARENTS
                {
                    return Err(ProposalValidationError::InvalidDerivation);
                }
                // Parent IDs must occur earlier in the evidence vector. This defines a
                // simple canonical topological order and prevents cycles/self-reference.
                let current_index = proposal
                    .evidence
                    .iter()
                    .position(|item| item.evidence_id == evidence.evidence_id)
                    .ok_or(ProposalValidationError::InvalidDerivation)?;
                for parent_id in &evidence.derived_from {
                    let parent_index = proposal
                        .evidence
                        .iter()
                        .position(|item| item.evidence_id == *parent_id)
                        .ok_or(ProposalValidationError::InvalidDerivation)?;
                    if parent_index >= current_index {
                        return Err(ProposalValidationError::InvalidDerivation);
                    }
                }
            }
            EpistemicStatus::Unavailable
            | EpistemicStatus::Conflicting
            | EpistemicStatus::Unknown => {
                return Err(ProposalValidationError::UnusableEvidence);
            }
        }
    }

    match proposal.mode {
        ProposalMode::Explain => {
            if proposal.candidates.is_empty() && proposal.evidence.is_empty() {
                return Err(ProposalValidationError::MissingEvidence);
            }
            if !proposal.candidates.is_empty() {
                return Err(ProposalValidationError::ModeCandidateMismatch);
            }
        }
        ProposalMode::Clarify => {
            if !proposal.candidates.is_empty() {
                return Err(ProposalValidationError::ModeCandidateMismatch);
            }
            if proposal.unknowns.is_empty() {
                return Err(ProposalValidationError::MissingClarificationTarget);
            }
        }
        ProposalMode::Abstain => {
            if !proposal.candidates.is_empty() {
                return Err(ProposalValidationError::ModeCandidateMismatch);
            }
        }
        ProposalMode::Recommend | ProposalMode::Preview => {
            if proposal.candidates.is_empty() {
                return Err(ProposalValidationError::ModeCandidateMismatch);
            }
            if proposal.evidence.is_empty() {
                return Err(ProposalValidationError::MissingEvidence);
            }
        }
    }

    if proposal.candidates.len() > MAX_CANDIDATES {
        return Err(ProposalValidationError::TooManyItems);
    }
    if !proposal.candidates.is_empty() {
        if proposal.risk_class >= RiskClass::High {
            return Err(ProposalValidationError::CandidateTooRisky);
        }
        if matches!(
            proposal.uncertainty,
            UncertaintyLevel::High | UncertaintyLevel::Uncalibrated
        ) {
            return Err(ProposalValidationError::UncertaintyTooHighForCandidate);
        }
        // No proposal can self-authorize. A future authorized workflow must re-check
        // current policy and authority; this field is not itself a consent receipt.
        if proposal.approval_requirement == ApprovalRequirement::None {
            return Err(ProposalValidationError::ApprovalRequired);
        }
    }

    for candidate in &proposal.candidates {
        if !valid_text(&candidate.summary, MAX_SUMMARY_BYTES)
            || candidate.effects.is_empty()
            || candidate.effects.len() > MAX_EFFECTS_PER_CANDIDATE
            || candidate.tradeoffs.len() > MAX_TRADEOFFS
            || candidate
                .tradeoffs
                .iter()
                .any(|value| !valid_text(value, MAX_DETAIL_BYTES))
        {
            return Err(ProposalValidationError::InvalidCandidate);
        }
        for effect in &candidate.effects {
            match effect {
                ProposedEffect::SetLightState { entity_ref, .. }
                | ProposedEffect::RequestMaintenanceReview { entity_ref }
                | ProposedEffect::SuggestComfortTarget { entity_ref, .. } => {
                    if !valid_identifier(entity_ref) {
                        return Err(ProposalValidationError::InvalidIdentifier);
                    }
                }
                ProposedEffect::NotifyResident { .. } => {}
            }
            if let ProposedEffect::SuggestComfortTarget {
                target_millicelsius,
                ..
            } = effect
            {
                if !(16_000..=30_000).contains(target_millicelsius) {
                    return Err(ProposalValidationError::ComfortTargetOutOfRange);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(status: EpistemicStatus, epoch: u64) -> EvidenceRef {
        EvidenceRef {
            evidence_id: "evidence-1".into(),
            source_ref: "ha-source-1".into(),
            entity_ref: "sensor.living_room_temperature".into(),
            observed_at_ms: 900,
            fresh_until_ms: 2_000,
            session_epoch: epoch,
            status,
            derived_from: Vec::new(),
        }
    }

    fn valid_proposal() -> HomeProposal {
        HomeProposal {
            schema_version: HOME_PROPOSAL_SCHEMA_V1,
            proposal_id: "proposal-1".into(),
            request_ref: "request-1".into(),
            generated_at_ms: 950,
            expires_at_ms: 1_500,
            mode: ProposalMode::Recommend,
            summary: "Consider turning on the living-room lamp.".into(),
            evidence: vec![evidence(EpistemicStatus::Observed, 7)],
            assumptions: vec!["The resident wants the room illuminated.".into()],
            unknowns: Vec::new(),
            uncertainty: UncertaintyLevel::Low,
            risk_class: RiskClass::Low,
            approval_requirement: ApprovalRequirement::ResidentConfirmation,
            candidates: vec![CandidatePlan {
                summary: "Turn on the named lamp.".into(),
                effects: vec![ProposedEffect::SetLightState {
                    entity_ref: "light.living_room".into(),
                    enabled: true,
                }],
                tradeoffs: vec!["Uses a small amount of electricity.".into()],
            }],
        }
    }

    fn epochs() -> BTreeMap<String, u64> {
        BTreeMap::from([("ha-source-1".to_string(), 7)])
    }

    #[test]
    fn accepts_current_bounded_recommendation_without_granting_authority() {
        let proposal = valid_proposal();
        assert_eq!(validate_home_proposal(&proposal, 1_000, &epochs()), Ok(()));
        // There is intentionally no dispatcher/credential in HomeProposal or this module.
    }

    #[test]
    fn rejects_stale_evidence() {
        let mut proposal = valid_proposal();
        proposal.evidence[0].fresh_until_ms = 999;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::StaleEvidence)
        );
    }

    #[test]
    fn rejects_prior_session_epoch_after_reconnect() {
        let proposal = valid_proposal();
        let current = BTreeMap::from([("ha-source-1".to_string(), 8)]);
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &current),
            Err(ProposalValidationError::SessionEpochMismatch)
        );
    }

    #[test]
    fn rejects_unknown_observation_as_evidence() {
        let mut proposal = valid_proposal();
        proposal.evidence[0].status = EpistemicStatus::Unknown;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::UnusableEvidence)
        );
    }

    #[test]
    fn rejects_missing_evidence_for_recommendation() {
        let mut proposal = valid_proposal();
        proposal.evidence.clear();
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::MissingEvidence)
        );
    }

    #[test]
    fn clarify_requires_a_specific_unknown_and_no_candidate_effects() {
        let mut proposal = valid_proposal();
        proposal.mode = ProposalMode::Clarify;
        proposal.candidates.clear();
        proposal.unknowns = vec!["Which room do you mean?".into()];
        assert_eq!(validate_home_proposal(&proposal, 1_000, &epochs()), Ok(()));

        proposal.candidates.push(CandidatePlan {
            summary: "Unexpected effect".into(),
            effects: vec![ProposedEffect::NotifyResident {
                topic: NotificationTopic::Recommendation,
            }],
            tradeoffs: Vec::new(),
        });
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::ModeCandidateMismatch)
        );
    }

    #[test]
    fn rejects_high_risk_candidate_even_with_specialist_review() {
        let mut proposal = valid_proposal();
        proposal.risk_class = RiskClass::High;
        proposal.approval_requirement = ApprovalRequirement::SpecialistReview;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::CandidateTooRisky)
        );
    }

    #[test]
    fn rejects_high_or_uncalibrated_uncertainty_for_action_candidates() {
        let mut proposal = valid_proposal();
        proposal.uncertainty = UncertaintyLevel::High;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::UncertaintyTooHighForCandidate)
        );

        proposal.uncertainty = UncertaintyLevel::Uncalibrated;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::UncertaintyTooHighForCandidate)
        );
    }

    #[test]
    fn rejects_expired_proposal() {
        let proposal = valid_proposal();
        assert_eq!(
            validate_home_proposal(&proposal, 1_500, &epochs()),
            Err(ProposalValidationError::ExpiredProposal)
        );
    }

    #[test]
    fn rejects_future_dated_evidence() {
        let mut proposal = valid_proposal();
        proposal.evidence[0].observed_at_ms = 1_001;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::FutureEvidence)
        );
    }

    #[test]
    fn derived_evidence_requires_preceding_parent_reference() {
        let mut proposal = valid_proposal();
        let mut derived = evidence(EpistemicStatus::Derived, 7);
        derived.evidence_id = "derived-2".into();
        derived.derived_from = vec!["missing-parent".into()];
        proposal.evidence.push(derived);
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::InvalidDerivation)
        );
    }

    #[test]
    fn rejects_invalid_or_unbounded_summary() {
        let mut proposal = valid_proposal();
        proposal.summary = "  ".into();
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::InvalidText)
        );

        proposal.summary = "x".repeat(MAX_SUMMARY_BYTES + 1);
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::InvalidText)
        );
    }

    #[test]
    fn rejects_candidate_without_an_approval_requirement() {
        let mut proposal = valid_proposal();
        proposal.approval_requirement = ApprovalRequirement::None;
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::ApprovalRequired)
        );
    }

    #[test]
    fn rejects_comfort_target_outside_initial_bounds() {
        let mut proposal = valid_proposal();
        proposal.candidates[0].effects = vec![ProposedEffect::SuggestComfortTarget {
            entity_ref: "climate.living_room".into(),
            target_millicelsius: 35_000,
        }];
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::ComfortTargetOutOfRange)
        );
    }

    #[test]
    fn rejects_derived_fact_with_non_topological_parent_order() {
        let mut proposal = valid_proposal();
        let mut derived = evidence(EpistemicStatus::Derived, 7);
        derived.evidence_id = "derived-2".into();
        derived.derived_from = vec!["evidence-1".into()];
        proposal.evidence.push(derived);
        assert_eq!(validate_home_proposal(&proposal, 1_000, &epochs()), Ok(()));

        proposal.evidence.swap(0, 1);
        assert_eq!(
            validate_home_proposal(&proposal, 1_000, &epochs()),
            Err(ProposalValidationError::InvalidDerivation)
        );
    }
}
