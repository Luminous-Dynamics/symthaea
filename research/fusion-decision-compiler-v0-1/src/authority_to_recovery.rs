//! Authority-to-recovery contract for fusion plant supervisory control.
//!
//! This module does not execute actuator commands. It verifies that a
//! controller proposal or operating-state transition is authorized, fresh,
//! inside a declared envelope, protected by an independent safety veto, and
//! bound to a recoverable state-machine path.

use std::collections::{BTreeMap, BTreeSet};

pub const AUTHORITY_TO_RECOVERY_VERSION: &str = "fusion-authority-to-recovery/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ControlTimingClass {
    HardProtection,
    FastRegulation,
    SupervisoryCoordination,
    OptimizationAdvisory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ControlEvidenceClass {
    Synthetic,
    Simulated,
    HardwareInLoop,
    Experimental,
    IndependentlyVerified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyDecision {
    Permit,
    Veto,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardResult {
    Satisfied,
    Unsatisfied,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionAcknowledgement {
    Pending,
    Succeeded,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequalificationResult {
    NotRequired,
    Pending,
    Passed,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlDecisionDisposition {
    Allowed,
    Rejected,
    HoldSafe,
    RecoveryRequired,
    NotComparable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlDiagnostic {
    EmptyCycleIdentity,
    EmptyClockDomain,
    ClockUncertaintyExceeded,
    DeadlineExceeded,
    MissingTimingEvidence,
    InvalidTimingProfile,
    MissingSafetyAuthority,
    SafetyAuthorityIdentityMissing,
    SafetyAuthorityEvidenceMissing,
    SafetyVetoActive,
    SafetyDecisionUnknown,
    DuplicateProposalId,
    EmptyProposalIdentity,
    InvalidObservationDigest,
    InvalidModelStateDigest,
    ProposalCycleMismatch,
    ProposalClockDomainMismatch,
    ProposalExpired,
    ProposalFromFuture,
    StaleObservation,
    ControllerNotAuthorized,
    ControllerRevisionMismatch,
    ChannelNotAuthorized,
    OperatingModeNotAuthorized,
    ChannelIdentityMismatch,
    UnitMismatch,
    NonFiniteCommand,
    CommandOutsideAbsoluteEnvelope,
    CommandExceedsSlewRate,
    MissingAuthorityTieBreak,
    InvalidAuthorityTieBreak,
    AuthorityManifestConflict,
    MissingFallbackState,
    FallbackStateNotSafe,
    UnknownTransition,
    TransitionSourceMismatch,
    TransitionTargetMismatch,
    InvalidTransitionTimeout,
    TransitionClockDomainMismatch,
    TransitionFromFuture,
    TransitionTimedOut,
    MissingGuard,
    DuplicateGuard,
    GuardUnknown,
    GuardUnsatisfied,
    GuardEvidenceMissing,
    GuardEvidenceDigestInvalid,
    UnexpectedGuard,
    UnknownState,
    TransitionAcknowledgementPending,
    TransitionAcknowledgementFailed,
    RequalificationRequired,
    RequalificationPending,
    RequalificationFailed,
    MissingTransitionEvidence,
    TripRecoveryBypass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlTimingEvidence {
    pub cycle_id: String,
    pub clock_domain: String,
    pub timing_class: ControlTimingClass,
    pub now_ns: u64,
    pub maximum_clock_uncertainty_ns: u64,
    pub observed_clock_uncertainty_ns: u64,
    pub maximum_end_to_end_latency_ns: u64,
    pub observed_end_to_end_latency_ns: u64,
    pub timing_evidence_ref: String,
    pub evidence_class: ControlEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerAuthority {
    pub controller_id: String,
    pub controller_revision: String,
    /// Lower numerical rank is higher authority within the declared manifest.
    pub authority_rank: u32,
    pub authorized_channels: Vec<String>,
    pub allowed_state_ids: Vec<String>,
    pub maximum_observation_age_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActuatorChannelProfile {
    pub channel_id: String,
    pub unit: String,
    pub authority_manifest_ref: String,
    pub authority_manifest_sha256: String,
    pub minimum_value: f64,
    pub maximum_value: f64,
    pub maximum_slew_per_second: f64,
    pub previous_value: f64,
    pub elapsed_since_previous_command_ns: u64,
    /// A total order for same-rank controllers; empty means no tie-break.
    pub tie_break_controller_order: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControllerProposal {
    pub proposal_id: String,
    pub controller_id: String,
    pub controller_revision: String,
    pub channel_id: String,
    pub cycle_id: String,
    pub clock_domain: String,
    pub operating_state_id: String,
    pub command_value: f64,
    pub unit: String,
    pub created_at_ns: u64,
    pub valid_until_ns: u64,
    pub observation_timestamp_ns: u64,
    pub observation_sha256: String,
    pub model_state_sha256: String,
    pub proposal_evidence_ref: String,
    pub evidence_class: ControlEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetyAuthorityReceipt {
    pub authority_id: String,
    pub independent_of_controller_ids: Vec<String>,
    pub channel_id: String,
    pub cycle_id: String,
    pub clock_domain: String,
    pub decision: SafetyDecision,
    pub evidence_ref: String,
    pub evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProposalRejection {
    pub proposal_id: String,
    pub diagnostics: Vec<ControlDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProposalEvidenceBinding {
    pub proposal_id: String,
    pub controller_id: String,
    pub controller_revision: String,
    pub observation_timestamp_ns: u64,
    pub valid_until_ns: u64,
    pub observation_sha256: String,
    pub model_state_sha256: String,
    pub proposal_evidence_ref: String,
    pub evidence_class: ControlEvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArbitrationReport {
    pub contract_version: &'static str,
    pub cycle_id: String,
    pub channel_id: String,
    pub unit: String,
    pub authority_manifest_ref: String,
    pub authority_manifest_sha256: String,
    pub timing_evidence: ControlTimingEvidence,
    pub disposition: ControlDecisionDisposition,
    pub winner_proposal_id: Option<String>,
    pub winner_controller_id: Option<String>,
    pub selected_command_value: Option<f64>,
    pub rejected_proposals: Vec<ProposalRejection>,
    pub considered_proposal_ids: Vec<String>,
    pub proposal_evidence_bindings: Vec<ProposalEvidenceBinding>,
    pub safety_authority_id: Option<String>,
    pub safety_evidence_ref: Option<String>,
    pub safety_evidence_sha256: Option<String>,
    pub timing_evidence_ref: String,
    pub diagnostics: Vec<ControlDiagnostic>,
    pub evidence_classes: Vec<ControlEvidenceClass>,
    pub claim_ceiling: &'static str,
}

/// Resolve one actuator channel for one cycle. This function never sends a
/// command. It produces an auditable winner or an explicit safe disposition.
pub fn arbitrate_actuator_channel(
    current_state_id: &str,
    channel: &ActuatorChannelProfile,
    authorities: &[ControllerAuthority],
    proposals: &[ControllerProposal],
    safety: Option<&SafetyAuthorityReceipt>,
    timing: &ControlTimingEvidence,
) -> ArbitrationReport {
    let mut diagnostics = validate_timing(timing);
    let mut rejections = Vec::new();
    let mut seen_proposal_ids = BTreeSet::new();
    let mut valid: Vec<(&ControllerProposal, &ControllerAuthority)> = Vec::new();

    if channel.channel_id.trim().is_empty() || channel.unit.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::ChannelIdentityMismatch);
    }
    if channel.authority_manifest_ref.trim().is_empty()
        || !is_sha256(&channel.authority_manifest_sha256)
    {
        diagnostics.push(ControlDiagnostic::AuthorityManifestConflict);
    }
    if !channel.minimum_value.is_finite()
        || !channel.maximum_value.is_finite()
        || channel.minimum_value > channel.maximum_value
        || !channel.maximum_slew_per_second.is_finite()
        || channel.maximum_slew_per_second < 0.0
        || !channel.previous_value.is_finite()
    {
        diagnostics.push(ControlDiagnostic::InvalidTimingProfile);
    }

    let Some(safety) = safety else {
        diagnostics.push(ControlDiagnostic::MissingSafetyAuthority);
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: None,
                safety_evidence_ref: None,
                safety_evidence_sha256: None,
                diagnostics,
            },
            proposals,
        );
    };

    if safety.authority_id.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::SafetyAuthorityIdentityMissing);
    }
    if safety.evidence_ref.trim().is_empty() || !is_sha256(&safety.evidence_sha256) {
        diagnostics.push(ControlDiagnostic::SafetyAuthorityEvidenceMissing);
    }
    if safety.channel_id != channel.channel_id
        || safety.cycle_id != timing.cycle_id
        || safety.clock_domain != timing.clock_domain
    {
        diagnostics.push(ControlDiagnostic::MissingSafetyAuthority);
    }
    if proposals.iter().any(|proposal| {
        !safety
            .independent_of_controller_ids
            .contains(&proposal.controller_id)
    }) {
        diagnostics.push(ControlDiagnostic::AuthorityManifestConflict);
    }

    if safety.decision == SafetyDecision::Veto {
        diagnostics.push(ControlDiagnostic::SafetyVetoActive);
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: proposals
                    .iter()
                    .map(|proposal| ProposalRejection {
                        proposal_id: proposal.proposal_id.clone(),
                        diagnostics: vec![ControlDiagnostic::SafetyVetoActive],
                    })
                    .collect(),
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    }
    if safety.decision == SafetyDecision::Unknown {
        diagnostics.push(ControlDiagnostic::SafetyDecisionUnknown);
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    }

    for proposal in proposals {
        let mut proposal_diagnostics = Vec::new();
        if proposal.proposal_id.trim().is_empty()
            || proposal.controller_id.trim().is_empty()
            || proposal.controller_revision.trim().is_empty()
            || proposal.proposal_evidence_ref.trim().is_empty()
        {
            proposal_diagnostics.push(ControlDiagnostic::EmptyProposalIdentity);
        }
        if !seen_proposal_ids.insert(proposal.proposal_id.as_str()) {
            proposal_diagnostics.push(ControlDiagnostic::DuplicateProposalId);
            diagnostics.push(ControlDiagnostic::DuplicateProposalId);
        }
        if !is_sha256(&proposal.observation_sha256) {
            proposal_diagnostics.push(ControlDiagnostic::InvalidObservationDigest);
        }
        if !is_sha256(&proposal.model_state_sha256) {
            proposal_diagnostics.push(ControlDiagnostic::InvalidModelStateDigest);
        }
        if proposal.cycle_id != timing.cycle_id {
            proposal_diagnostics.push(ControlDiagnostic::ProposalCycleMismatch);
        }
        if proposal.clock_domain != timing.clock_domain {
            proposal_diagnostics.push(ControlDiagnostic::ProposalClockDomainMismatch);
        }
        if proposal.created_at_ns > timing.now_ns
            || proposal.observation_timestamp_ns > timing.now_ns
        {
            proposal_diagnostics.push(ControlDiagnostic::ProposalFromFuture);
        }
        if proposal.valid_until_ns < timing.now_ns {
            proposal_diagnostics.push(ControlDiagnostic::ProposalExpired);
        }
        let age = timing
            .now_ns
            .saturating_sub(proposal.observation_timestamp_ns);
        if let Some(authority) = authorities
            .iter()
            .find(|authority| authority.controller_id == proposal.controller_id)
        {
            if authority.controller_revision != proposal.controller_revision {
                proposal_diagnostics.push(ControlDiagnostic::ControllerRevisionMismatch);
            }
            if !authority
                .authorized_channels
                .iter()
                .any(|candidate| candidate == &proposal.channel_id)
            {
                proposal_diagnostics.push(ControlDiagnostic::ChannelNotAuthorized);
            }
            if !authority
                .allowed_state_ids
                .iter()
                .any(|state| state == current_state_id)
            {
                proposal_diagnostics.push(ControlDiagnostic::OperatingModeNotAuthorized);
            }
            if age > authority.maximum_observation_age_ns {
                proposal_diagnostics.push(ControlDiagnostic::StaleObservation);
            }
        } else {
            proposal_diagnostics.push(ControlDiagnostic::ControllerNotAuthorized);
        }
        if proposal.channel_id != channel.channel_id {
            proposal_diagnostics.push(ControlDiagnostic::ChannelIdentityMismatch);
        }
        if proposal.operating_state_id != current_state_id {
            proposal_diagnostics.push(ControlDiagnostic::OperatingModeNotAuthorized);
        }
        if proposal.unit != channel.unit {
            proposal_diagnostics.push(ControlDiagnostic::UnitMismatch);
        }
        if !proposal.command_value.is_finite() {
            proposal_diagnostics.push(ControlDiagnostic::NonFiniteCommand);
        } else if proposal.command_value < channel.minimum_value
            || proposal.command_value > channel.maximum_value
        {
            proposal_diagnostics.push(ControlDiagnostic::CommandOutsideAbsoluteEnvelope);
        } else if !slew_is_valid(channel, proposal.command_value) {
            proposal_diagnostics.push(ControlDiagnostic::CommandExceedsSlewRate);
        }

        if proposal_diagnostics.is_empty() {
            if let Some(authority) = authorities
                .iter()
                .find(|authority| authority.controller_id == proposal.controller_id)
            {
                valid.push((proposal, authority));
            }
        } else {
            rejections.push(ProposalRejection {
                proposal_id: proposal.proposal_id.clone(),
                diagnostics: proposal_diagnostics,
            });
        }
    }

    rejections.sort_by(|a, b| a.proposal_id.cmp(&b.proposal_id));
    if !diagnostics.is_empty() {
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    }
    if valid.is_empty() {
        diagnostics.push(ControlDiagnostic::ControllerNotAuthorized);
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    }

    valid.sort_by(|(left_proposal, left_authority), (right_proposal, right_authority)| {
        left_authority
            .authority_rank
            .cmp(&right_authority.authority_rank)
            .then_with(|| left_proposal.controller_id.cmp(&right_proposal.controller_id))
            .then_with(|| left_proposal.proposal_id.cmp(&right_proposal.proposal_id))
    });

    let highest_rank = valid[0].1.authority_rank;
    let tied: Vec<_> = valid
        .iter()
        .filter(|(_, authority)| authority.authority_rank == highest_rank)
        .collect();

    let winner = if tied.len() == 1 {
        Some(tied[0].0)
    } else {
        let declared_positions: BTreeMap<&str, usize> = channel
            .tie_break_controller_order
            .iter()
            .enumerate()
            .map(|(index, controller_id)| (controller_id.as_str(), index))
            .collect();
        let has_duplicate_order = declared_positions.len() != channel.tie_break_controller_order.len();
        let tied_ids: Vec<&str> = tied
            .iter()
            .map(|(proposal, _)| proposal.controller_id.as_str())
            .collect();
        let has_all_tied = tied_ids
            .iter()
            .all(|controller_id| declared_positions.contains_key(controller_id));
        if channel.tie_break_controller_order.is_empty() || has_duplicate_order || !has_all_tied {
            diagnostics.push(ControlDiagnostic::MissingAuthorityTieBreak);
            None
        } else {
            tied.iter()
                .min_by_key(|(proposal, _)| declared_positions[proposal.controller_id.as_str()])
                .map(|(proposal, _)| *proposal)
        }
    };

    let Some(winner) = winner else {
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::HoldSafe,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    };

    // Defend in depth: revalidate the winner after the arbitration selection.
    if winner.command_value < channel.minimum_value
        || winner.command_value > channel.maximum_value
        || !slew_is_valid(channel, winner.command_value)
    {
        diagnostics.push(ControlDiagnostic::CommandOutsideAbsoluteEnvelope);
        return arbitration_report(
            timing,
            channel,
            ArbitrationOutcome {
                disposition: ControlDecisionDisposition::Rejected,
                winner_proposal_id: None,
                winner_controller_id: None,
                selected_command_value: None,
                rejected_proposals: rejections,
                safety_authority_id: Some(safety.authority_id.clone()),
                safety_evidence_ref: Some(safety.evidence_ref.clone()),
                safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
                diagnostics,
            },
            proposals,
        );
    }

    arbitration_report(
        timing,
        channel,
        ArbitrationOutcome {
            disposition: ControlDecisionDisposition::Allowed,
            winner_proposal_id: Some(winner.proposal_id.clone()),
            winner_controller_id: Some(winner.controller_id.clone()),
            selected_command_value: Some(winner.command_value),
            rejected_proposals: rejections,
            safety_authority_id: Some(safety.authority_id.clone()),
            safety_evidence_ref: Some(safety.evidence_ref.clone()),
            safety_evidence_sha256: Some(safety.evidence_sha256.clone()),
            diagnostics,
        },
        proposals,
    )
}

fn slew_is_valid(channel: &ActuatorChannelProfile, proposed: f64) -> bool {
    let elapsed_seconds = channel.elapsed_since_previous_command_ns as f64 / 1_000_000_000.0;
    let allowed_delta = channel.maximum_slew_per_second * elapsed_seconds;
    (proposed - channel.previous_value).abs() <= allowed_delta
}

fn validate_timing(timing: &ControlTimingEvidence) -> Vec<ControlDiagnostic> {
    let mut diagnostics = Vec::new();
    if timing.cycle_id.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::EmptyCycleIdentity);
    }
    if timing.clock_domain.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::EmptyClockDomain);
    }
    if timing.timing_evidence_ref.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::MissingTimingEvidence);
    }
    if timing.maximum_end_to_end_latency_ns == 0 {
        diagnostics.push(ControlDiagnostic::InvalidTimingProfile);
    }
    if timing.observed_clock_uncertainty_ns > timing.maximum_clock_uncertainty_ns {
        diagnostics.push(ControlDiagnostic::ClockUncertaintyExceeded);
    }
    if timing.observed_end_to_end_latency_ns > timing.maximum_end_to_end_latency_ns {
        diagnostics.push(ControlDiagnostic::DeadlineExceeded);
    }
    diagnostics
}

struct ArbitrationOutcome {
    disposition: ControlDecisionDisposition,
    winner_proposal_id: Option<String>,
    winner_controller_id: Option<String>,
    selected_command_value: Option<f64>,
    rejected_proposals: Vec<ProposalRejection>,
    safety_authority_id: Option<String>,
    safety_evidence_ref: Option<String>,
    safety_evidence_sha256: Option<String>,
    diagnostics: Vec<ControlDiagnostic>,
}

fn arbitration_report(
    timing: &ControlTimingEvidence,
    channel: &ActuatorChannelProfile,
    outcome: ArbitrationOutcome,
    proposals: &[ControllerProposal],
) -> ArbitrationReport {
    let mut considered_proposal_ids: Vec<String> = proposals
        .iter()
        .map(|proposal| proposal.proposal_id.clone())
        .collect();
    considered_proposal_ids.sort();
    let mut evidence_classes: Vec<ControlEvidenceClass> = proposals
        .iter()
        .map(|proposal| proposal.evidence_class)
        .collect();
    evidence_classes.sort_by_key(|class| *class as u8);
    evidence_classes.dedup();
    evidence_classes.push(timing.evidence_class);
    ArbitrationReport {
        contract_version: AUTHORITY_TO_RECOVERY_VERSION,
        cycle_id: timing.cycle_id.clone(),
        channel_id: channel.channel_id.clone(),
        unit: channel.unit.clone(),
        authority_manifest_ref: channel.authority_manifest_ref.clone(),
        authority_manifest_sha256: channel.authority_manifest_sha256.clone(),
        timing_evidence: timing.clone(),
        disposition: outcome.disposition,
        winner_proposal_id: outcome.winner_proposal_id,
        winner_controller_id: outcome.winner_controller_id,
        selected_command_value: outcome.selected_command_value,
        rejected_proposals: outcome.rejected_proposals,
        considered_proposal_ids,
        proposal_evidence_bindings: proposals
            .iter()
            .map(|proposal| ProposalEvidenceBinding {
                proposal_id: proposal.proposal_id.clone(),
                controller_id: proposal.controller_id.clone(),
                controller_revision: proposal.controller_revision.clone(),
                observation_timestamp_ns: proposal.observation_timestamp_ns,
                valid_until_ns: proposal.valid_until_ns,
                observation_sha256: proposal.observation_sha256.clone(),
                model_state_sha256: proposal.model_state_sha256.clone(),
                proposal_evidence_ref: proposal.proposal_evidence_ref.clone(),
                evidence_class: proposal.evidence_class,
            })
            .collect(),
        safety_authority_id: outcome.safety_authority_id,
        safety_evidence_ref: outcome.safety_evidence_ref,
        safety_evidence_sha256: outcome.safety_evidence_sha256,
        timing_evidence_ref: timing.timing_evidence_ref.clone(),
        diagnostics: outcome.diagnostics,
        evidence_classes,
        claim_ceiling: "proposal arbitration contract only; no actuator execution or safety certification",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatingState {
    pub state_id: String,
    pub safe_hold_state: bool,
    pub operational_state: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionRule {
    pub transition_id: String,
    pub from_state_id: String,
    pub to_state_id: String,
    pub required_guard_ids: Vec<String>,
    pub timeout_ns: u64,
    pub fallback_state_id: String,
    pub requires_requalification: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatingStateManifest {
    pub manifest_id: String,
    pub manifest_sha256: String,
    pub clock_domain: String,
    pub states: Vec<OperatingState>,
    pub transitions: Vec<TransitionRule>,
    pub protection_trip_state_ids: Vec<String>,
    pub require_requalification_for_operational_entry: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardEvidence {
    pub guard_id: String,
    pub result: GuardResult,
    pub evidence_ref: String,
    pub evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionAttempt {
    pub transition_id: String,
    pub from_state_id: String,
    pub to_state_id: String,
    pub initiated_at_ns: u64,
    pub evaluated_at_ns: u64,
    pub clock_domain: String,
    pub guards: Vec<GuardEvidence>,
    pub acknowledgement: TransitionAcknowledgement,
    pub requalification: RequalificationResult,
    pub transition_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateTransitionReport {
    pub contract_version: &'static str,
    pub manifest_id: String,
    pub manifest_sha256: String,
    pub clock_domain: String,
    pub transition_id: String,
    pub from_state_id: String,
    pub to_state_id: String,
    pub disposition: ControlDecisionDisposition,
    pub fallback_state_id: Option<String>,
    pub diagnostics: Vec<ControlDiagnostic>,
    pub guard_evidence_refs: Vec<String>,
    pub guard_evidence: Vec<GuardEvidence>,
    pub transition_evidence_ref: String,
    pub claim_ceiling: &'static str,
}

pub fn evaluate_state_transition(
    manifest: &OperatingStateManifest,
    attempt: &TransitionAttempt,
) -> StateTransitionReport {
    let mut diagnostics = Vec::new();
    if manifest.manifest_id.trim().is_empty() || !is_sha256(&manifest.manifest_sha256) {
        diagnostics.push(ControlDiagnostic::AuthorityManifestConflict);
    }
    if manifest.clock_domain.trim().is_empty()
        || attempt.clock_domain != manifest.clock_domain
    {
        diagnostics.push(ControlDiagnostic::TransitionClockDomainMismatch);
    }
    if attempt.transition_evidence_ref.trim().is_empty() {
        diagnostics.push(ControlDiagnostic::MissingTransitionEvidence);
    }
    if attempt.evaluated_at_ns < attempt.initiated_at_ns {
        diagnostics.push(ControlDiagnostic::TransitionFromFuture);
    }

    let Some(rule) = manifest
        .transitions
        .iter()
        .find(|rule| rule.transition_id == attempt.transition_id)
    else {
        diagnostics.push(ControlDiagnostic::UnknownTransition);
        return transition_report(manifest, attempt, None, ControlDecisionDisposition::NotComparable, diagnostics);
    };

    if attempt.from_state_id != rule.from_state_id {
        diagnostics.push(ControlDiagnostic::TransitionSourceMismatch);
    }
    if attempt.to_state_id != rule.to_state_id {
        diagnostics.push(ControlDiagnostic::TransitionTargetMismatch);
    }
    let state_ids: BTreeSet<&str> = manifest.states.iter().map(|state| state.state_id.as_str()).collect();
    let safe_states: BTreeSet<&str> = manifest
        .states
        .iter()
        .filter(|state| state.safe_hold_state)
        .map(|state| state.state_id.as_str())
        .collect();
    if !state_ids.contains(rule.fallback_state_id.as_str()) {
        diagnostics.push(ControlDiagnostic::MissingFallbackState);
    } else if !safe_states.contains(rule.fallback_state_id.as_str()) {
        diagnostics.push(ControlDiagnostic::FallbackStateNotSafe);
    }
    if rule.timeout_ns == 0 {
        diagnostics.push(ControlDiagnostic::InvalidTransitionTimeout);
    }
    if attempt.evaluated_at_ns.saturating_sub(attempt.initiated_at_ns) > rule.timeout_ns {
        diagnostics.push(ControlDiagnostic::TransitionTimedOut);
    }

    if !state_ids.contains(rule.from_state_id.as_str())
        || !state_ids.contains(rule.to_state_id.as_str())
    {
        diagnostics.push(ControlDiagnostic::UnknownState);
    }

    let mut expected_guards = rule.required_guard_ids.clone();
    expected_guards.sort();
    let original_expected_guard_count = expected_guards.len();
    expected_guards.dedup();
    if expected_guards.len() != original_expected_guard_count {
        diagnostics.push(ControlDiagnostic::AuthorityManifestConflict);
    }
    let mut supplied_guards: BTreeMap<&str, &GuardEvidence> = BTreeMap::new();
    for guard in &attempt.guards {
        if supplied_guards.insert(guard.guard_id.as_str(), guard).is_some() {
            diagnostics.push(ControlDiagnostic::DuplicateGuard);
        }
        if guard.evidence_ref.trim().is_empty() {
            diagnostics.push(ControlDiagnostic::GuardEvidenceMissing);
        }
        if !is_sha256(&guard.evidence_sha256) {
            diagnostics.push(ControlDiagnostic::GuardEvidenceDigestInvalid);
        }
    }

    if supplied_guards
        .keys()
        .any(|guard_id| !expected_guards.iter().any(|expected| expected.as_str() == *guard_id))
    {
        diagnostics.push(ControlDiagnostic::UnexpectedGuard);
    }

    for guard_id in &expected_guards {
        match supplied_guards.get(guard_id.as_str()) {
            None => diagnostics.push(ControlDiagnostic::MissingGuard),
            Some(guard) => match guard.result {
                GuardResult::Satisfied => {}
                GuardResult::Unsatisfied => diagnostics.push(ControlDiagnostic::GuardUnsatisfied),
                GuardResult::Unknown => diagnostics.push(ControlDiagnostic::GuardUnknown),
            },
        }
    }

    let target_is_operational = manifest
        .states
        .iter()
        .find(|state| state.state_id == rule.to_state_id)
        .is_some_and(|state| state.operational_state);
    let from_protection_trip = manifest
        .protection_trip_state_ids
        .iter()
        .any(|state_id| state_id == &rule.from_state_id);
    let requires_requalification = rule.requires_requalification
        || (manifest.require_requalification_for_operational_entry && target_is_operational)
        || from_protection_trip && target_is_operational;
    if from_protection_trip && target_is_operational && !rule.requires_requalification {
        diagnostics.push(ControlDiagnostic::TripRecoveryBypass);
    }
    if requires_requalification {
        match attempt.requalification {
            RequalificationResult::Passed => {}
            RequalificationResult::Pending | RequalificationResult::NotRequired => {
                diagnostics.push(ControlDiagnostic::RequalificationPending);
            }
            RequalificationResult::Failed => {
                diagnostics.push(ControlDiagnostic::RequalificationFailed);
            }
        }
    }

    if attempt.acknowledgement == TransitionAcknowledgement::Failed
        || diagnostics.contains(&ControlDiagnostic::TransitionTimedOut)
    {
        diagnostics.push(ControlDiagnostic::TransitionAcknowledgementFailed);
        return transition_report(
            manifest,
            attempt,
            Some(rule),
            ControlDecisionDisposition::RecoveryRequired,
            diagnostics,
        );
    }
    if attempt.acknowledgement == TransitionAcknowledgement::Pending {
        diagnostics.push(ControlDiagnostic::TransitionAcknowledgementPending);
    }

    if !diagnostics.is_empty() {
        let hold = diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic,
                ControlDiagnostic::GuardUnknown
                    | ControlDiagnostic::MissingGuard
                    | ControlDiagnostic::RequalificationPending
                    | ControlDiagnostic::TransitionAcknowledgementPending
                    | ControlDiagnostic::TransitionClockDomainMismatch
                    | ControlDiagnostic::TransitionFromFuture
                    | ControlDiagnostic::MissingTransitionEvidence
                    | ControlDiagnostic::GuardEvidenceMissing
                    | ControlDiagnostic::GuardEvidenceDigestInvalid
            )
        });
        let disposition = if hold {
            ControlDecisionDisposition::HoldSafe
        } else {
            ControlDecisionDisposition::Rejected
        };
        return transition_report(manifest, attempt, Some(rule), disposition, diagnostics);
    }

    if attempt.acknowledgement != TransitionAcknowledgement::Succeeded {
        diagnostics.push(ControlDiagnostic::TransitionAcknowledgementPending);
        return transition_report(
            manifest,
            attempt,
            Some(rule),
            ControlDecisionDisposition::HoldSafe,
            diagnostics,
        );
    }

    transition_report(
        manifest,
        attempt,
        Some(rule),
        ControlDecisionDisposition::Allowed,
        diagnostics,
    )
}

fn transition_report(
    manifest: &OperatingStateManifest,
    attempt: &TransitionAttempt,
    rule: Option<&TransitionRule>,
    disposition: ControlDecisionDisposition,
    diagnostics: Vec<ControlDiagnostic>,
) -> StateTransitionReport {
    StateTransitionReport {
        contract_version: AUTHORITY_TO_RECOVERY_VERSION,
        manifest_id: manifest.manifest_id.clone(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        clock_domain: manifest.clock_domain.clone(),
        transition_id: attempt.transition_id.clone(),
        from_state_id: attempt.from_state_id.clone(),
        to_state_id: attempt.to_state_id.clone(),
        disposition,
        fallback_state_id: rule.map(|value| value.fallback_state_id.clone()),
        diagnostics,
        guard_evidence_refs: attempt
            .guards
            .iter()
            .map(|guard| guard.evidence_ref.clone())
            .collect(),
        guard_evidence: attempt.guards.clone(),
        transition_evidence_ref: attempt.transition_evidence_ref.clone(),
        claim_ceiling: "declared state-transition and recovery contract only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn timing() -> ControlTimingEvidence {
        ControlTimingEvidence {
            cycle_id: "cycle-42".to_string(),
            clock_domain: "monotonic-ptp-domain-1".to_string(),
            timing_class: ControlTimingClass::SupervisoryCoordination,
            now_ns: 1_000_000_000,
            maximum_clock_uncertainty_ns: 1_000,
            observed_clock_uncertainty_ns: 100,
            maximum_end_to_end_latency_ns: 20_000_000,
            observed_end_to_end_latency_ns: 8_000_000,
            timing_evidence_ref: "timing-receipt-cycle-42".to_string(),
            evidence_class: ControlEvidenceClass::Synthetic,
        }
    }

    fn channel() -> ActuatorChannelProfile {
        ActuatorChannelProfile {
            channel_id: "heating-power".to_string(),
            unit: "MW".to_string(),
            authority_manifest_ref: "synthetic-authority-manifest-v1".to_string(),
            authority_manifest_sha256: sha('9'),
            minimum_value: 0.0,
            maximum_value: 100.0,
            maximum_slew_per_second: 50.0,
            previous_value: 20.0,
            elapsed_since_previous_command_ns: 1_000_000_000,
            tie_break_controller_order: vec![
                "deterministic-mpc".to_string(),
                "ml-supervisor".to_string(),
            ],
        }
    }

    fn authorities() -> Vec<ControllerAuthority> {
        vec![
            ControllerAuthority {
                controller_id: "deterministic-mpc".to_string(),
                controller_revision: "mpc-r4".to_string(),
                authority_rank: 1,
                authorized_channels: vec!["heating-power".to_string()],
                allowed_state_ids: vec!["steady-operation".to_string()],
                maximum_observation_age_ns: 100_000_000,
            },
            ControllerAuthority {
                controller_id: "ml-supervisor".to_string(),
                controller_revision: "ml-r8".to_string(),
                authority_rank: 1,
                authorized_channels: vec!["heating-power".to_string()],
                allowed_state_ids: vec!["steady-operation".to_string()],
                maximum_observation_age_ns: 50_000_000,
            },
        ]
    }

    fn proposal(id: &str, controller: &str, revision: &str, value: f64) -> ControllerProposal {
        ControllerProposal {
            proposal_id: id.to_string(),
            controller_id: controller.to_string(),
            controller_revision: revision.to_string(),
            channel_id: "heating-power".to_string(),
            cycle_id: "cycle-42".to_string(),
            clock_domain: "monotonic-ptp-domain-1".to_string(),
            operating_state_id: "steady-operation".to_string(),
            command_value: value,
            unit: "MW".to_string(),
            created_at_ns: 995_000_000,
            valid_until_ns: 1_050_000_000,
            observation_timestamp_ns: 990_000_000,
            observation_sha256: sha('a'),
            model_state_sha256: sha('b'),
            proposal_evidence_ref: format!("proposal-evidence-{id}"),
            evidence_class: ControlEvidenceClass::Synthetic,
        }
    }

    fn safety(decision: SafetyDecision) -> SafetyAuthorityReceipt {
        SafetyAuthorityReceipt {
            authority_id: "independent-protection-plc".to_string(),
            independent_of_controller_ids: vec![
                "deterministic-mpc".to_string(),
                "ml-supervisor".to_string(),
            ],
            channel_id: "heating-power".to_string(),
            cycle_id: "cycle-42".to_string(),
            clock_domain: "monotonic-ptp-domain-1".to_string(),
            decision,
            evidence_ref: "safety-plc-receipt".to_string(),
            evidence_sha256: sha('c'),
        }
    }

    fn arbitrate(
        proposals: &[ControllerProposal],
        safety: SafetyDecision,
        time: &ControlTimingEvidence,
        tie_break: &[String],
    ) -> ArbitrationReport {
        let mut channel = channel();
        channel.tie_break_controller_order = tie_break.to_vec();
        arbitrate_actuator_channel(
            "steady-operation",
            &channel,
            &authorities(),
            proposals,
            Some(&self::safety(safety)),
            time,
        )
    }

    #[test]
    fn arbitration_receipt_preserves_authority_timing_and_proposal_evidence() {
        let proposals = vec![proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)];
        let report = arbitrate(
            &proposals,
            SafetyDecision::Permit,
            &timing(),
            &channel().tie_break_controller_order,
        );
        assert_eq!(report.proposal_evidence_bindings.len(), 1);
        assert_eq!(
            report.proposal_evidence_bindings[0].observation_sha256,
            proposals[0].observation_sha256
        );
        assert_eq!(report.safety_evidence_ref.as_deref(), Some("safety-plc-receipt"));
        assert_eq!(report.timing_evidence.cycle_id, "cycle-42");
        assert!(!report.authority_manifest_sha256.is_empty());
    }

    #[test]
    fn independent_safety_veto_overrides_high_performance_proposal() {
        let proposals = vec![proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)];
        let report = arbitrate(&proposals, SafetyDecision::Veto, &timing(), &channel().tie_break_controller_order);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert_eq!(report.winner_proposal_id, None);
        assert!(report.diagnostics.contains(&ControlDiagnostic::SafetyVetoActive));
    }

    #[test]
    fn unknown_safety_decision_holds_safe() {
        let proposals = vec![proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)];
        let report = arbitrate(&proposals, SafetyDecision::Unknown, &timing(), &channel().tie_break_controller_order);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::SafetyDecisionUnknown));
    }

    #[test]
    fn missing_independent_safety_authority_holds_safe() {
        let report = arbitrate_actuator_channel(
            "steady-operation",
            &channel(),
            &authorities(),
            &[proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)],
            None,
            &timing(),
        );
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
    }

    #[test]
    fn deterministic_tie_break_selects_manifest_order_not_input_order() {
        let first = proposal("p1", "ml-supervisor", "ml-r8", 45.0);
        let second = proposal("p2", "deterministic-mpc", "mpc-r4", 40.0);
        let order = channel().tie_break_controller_order;
        let report_a = arbitrate(&[first.clone(), second.clone()], SafetyDecision::Permit, &timing(), &order);
        let report_b = arbitrate(&[second, first], SafetyDecision::Permit, &timing(), &order);
        assert_eq!(report_a, report_b);
        assert_eq!(report_a.winner_controller_id.as_deref(), Some("deterministic-mpc"));
    }

    #[test]
    fn equal_authority_without_declared_tie_break_holds_safe() {
        let proposals = vec![
            proposal("p1", "ml-supervisor", "ml-r8", 45.0),
            proposal("p2", "deterministic-mpc", "mpc-r4", 40.0),
        ];
        let report = arbitrate(&proposals, SafetyDecision::Permit, &timing(), &[]);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::MissingAuthorityTieBreak));
    }

    #[test]
    fn safety_authority_must_explicitly_cover_every_controller() {
        let mut safety_receipt = safety(SafetyDecision::Permit);
        safety_receipt.independent_of_controller_ids = vec!["deterministic-mpc".to_string()];
        let report = arbitrate_actuator_channel(
            "steady-operation",
            &channel(),
            &authorities(),
            &[proposal("p1", "ml-supervisor", "ml-r8", 40.0)],
            Some(&safety_receipt),
            &timing(),
        );
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::AuthorityManifestConflict));
    }

    #[test]
    fn duplicate_proposal_ids_block_the_whole_arbitration_cycle() {
        let proposals = vec![
            proposal("duplicate", "deterministic-mpc", "mpc-r4", 40.0),
            proposal("duplicate", "ml-supervisor", "ml-r8", 45.0),
        ];
        let report = arbitrate(
            &proposals,
            SafetyDecision::Permit,
            &timing(),
            &channel().tie_break_controller_order,
        );
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::DuplicateProposalId));
    }

    #[test]
    fn out_of_manifest_guard_is_not_ignored() {
        let manifest = transition_manifest();
        let mut attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        attempt.guards.push(GuardEvidence {
            guard_id: "unlisted-guard".to_string(),
            result: GuardResult::Satisfied,
            evidence_ref: "extra-guard".to_string(),
            evidence_sha256: sha('f'),
        });
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Rejected);
        assert!(report.diagnostics.contains(&ControlDiagnostic::UnexpectedGuard));
    }

    #[test]
    fn command_outside_absolute_envelope_is_rejected() {
        let proposals = vec![proposal("p1", "deterministic-mpc", "mpc-r4", 120.0)];
        let report = arbitrate(&proposals, SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.rejected_proposals[0].diagnostics.contains(
            &ControlDiagnostic::CommandOutsideAbsoluteEnvelope
        ));
    }

    #[test]
    fn command_exceeding_slew_rate_is_rejected() {
        let proposals = vec![proposal("p1", "deterministic-mpc", "mpc-r4", 90.0)];
        let report = arbitrate(&proposals, SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.rejected_proposals[0].diagnostics.contains(
            &ControlDiagnostic::CommandExceedsSlewRate
        ));
    }

    #[test]
    fn stale_observation_is_rejected() {
        let mut stale = proposal("p1", "deterministic-mpc", "mpc-r4", 40.0);
        stale.observation_timestamp_ns = 0;
        let report = arbitrate(&[stale], SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert!(report.rejected_proposals[0].diagnostics.contains(&ControlDiagnostic::StaleObservation));
    }

    #[test]
    fn expired_proposal_is_rejected() {
        let mut expired = proposal("p1", "deterministic-mpc", "mpc-r4", 40.0);
        expired.valid_until_ns = 999_999_999;
        let report = arbitrate(&[expired], SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert!(report.rejected_proposals[0].diagnostics.contains(&ControlDiagnostic::ProposalExpired));
    }

    #[test]
    fn wrong_clock_domain_holds_safe() {
        let mut bad = proposal("p1", "deterministic-mpc", "mpc-r4", 40.0);
        bad.clock_domain = "unsynchronized-host-clock".to_string();
        let report = arbitrate(&[bad], SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
    }

    #[test]
    fn hard_deadline_miss_holds_safe() {
        let mut delayed = timing();
        delayed.timing_class = ControlTimingClass::HardProtection;
        delayed.observed_end_to_end_latency_ns = 20_000_001;
        let report = arbitrate(
            &[proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)],
            SafetyDecision::Permit,
            &delayed,
            &channel().tie_break_controller_order,
        );
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::DeadlineExceeded));
    }

    #[test]
    fn clock_uncertainty_above_bound_holds_safe() {
        let mut delayed = timing();
        delayed.observed_clock_uncertainty_ns = 2_000;
        let report = arbitrate(
            &[proposal("p1", "deterministic-mpc", "mpc-r4", 40.0)],
            SafetyDecision::Permit,
            &delayed,
            &channel().tie_break_controller_order,
        );
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
    }

    #[test]
    fn unauthorized_controller_channel_and_mode_are_rejected() {
        let mut bad = proposal("p1", "unregistered-ai", "unknown", 40.0);
        bad.channel_id = "magnet-current".to_string();
        let report = arbitrate(&[bad], SafetyDecision::Permit, &timing(), &channel().tie_break_controller_order);
        assert!(!report.rejected_proposals.is_empty());
    }

    fn transition_manifest() -> OperatingStateManifest {
        OperatingStateManifest {
            manifest_id: "fusion-operating-state-v1".to_string(),
            manifest_sha256: sha('d'),
            clock_domain: "monotonic-ptp-domain-1".to_string(),
            states: vec![
                OperatingState {
                    state_id: "steady-operation".to_string(),
                    safe_hold_state: false,
                    operational_state: true,
                },
                OperatingState {
                    state_id: "recovery-hold".to_string(),
                    safe_hold_state: true,
                    operational_state: false,
                },
                OperatingState {
                    state_id: "protection-trip".to_string(),
                    safe_hold_state: true,
                    operational_state: false,
                },
                OperatingState {
                    state_id: "requalification".to_string(),
                    safe_hold_state: true,
                    operational_state: false,
                },
            ],
            transitions: vec![
                TransitionRule {
                    transition_id: "enter-recovery-hold".to_string(),
                    from_state_id: "steady-operation".to_string(),
                    to_state_id: "recovery-hold".to_string(),
                    required_guard_ids: vec!["plant-safe".to_string()],
                    timeout_ns: 100_000_000,
                    fallback_state_id: "protection-trip".to_string(),
                    requires_requalification: false,
                },
                TransitionRule {
                    transition_id: "requalify-to-operation".to_string(),
                    from_state_id: "requalification".to_string(),
                    to_state_id: "steady-operation".to_string(),
                    required_guard_ids: vec!["all-systems-ready".to_string()],
                    timeout_ns: 200_000_000,
                    fallback_state_id: "recovery-hold".to_string(),
                    requires_requalification: true,
                },
                TransitionRule {
                    transition_id: "bypass-trip".to_string(),
                    from_state_id: "protection-trip".to_string(),
                    to_state_id: "steady-operation".to_string(),
                    required_guard_ids: vec!["all-systems-ready".to_string()],
                    timeout_ns: 200_000_000,
                    fallback_state_id: "recovery-hold".to_string(),
                    requires_requalification: false,
                },
            ],
            protection_trip_state_ids: vec!["protection-trip".to_string()],
            require_requalification_for_operational_entry: true,
        }
    }

    fn transition_attempt(
        transition_id: &str,
        from: &str,
        to: &str,
        ack: TransitionAcknowledgement,
        requalification: RequalificationResult,
    ) -> TransitionAttempt {
        TransitionAttempt {
            transition_id: transition_id.to_string(),
            from_state_id: from.to_string(),
            to_state_id: to.to_string(),
            initiated_at_ns: 1_000,
            evaluated_at_ns: 2_000,
            clock_domain: "monotonic-ptp-domain-1".to_string(),
            guards: vec![GuardEvidence {
                guard_id: if transition_id == "enter-recovery-hold" {
                    "plant-safe".to_string()
                } else {
                    "all-systems-ready".to_string()
                },
                result: GuardResult::Satisfied,
                evidence_ref: "guard-evidence".to_string(),
                evidence_sha256: sha('e'),
            }],
            acknowledgement: ack,
            requalification,
            transition_evidence_ref: "transition-evidence".to_string(),
        }
    }

    #[test]
    fn declared_transition_with_all_guards_and_ack_is_allowed() {
        let manifest = transition_manifest();
        let attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Allowed);
    }

    #[test]
    fn unknown_guard_holds_safe() {
        let manifest = transition_manifest();
        let mut attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        attempt.guards[0].result = GuardResult::Unknown;
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
    }

    #[test]
    fn false_guard_rejects_transition() {
        let manifest = transition_manifest();
        let mut attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        attempt.guards[0].result = GuardResult::Unsatisfied;
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Rejected);
    }

    #[test]
    fn missing_guard_holds_safe() {
        let manifest = transition_manifest();
        let mut attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        attempt.guards.clear();
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
    }

    #[test]
    fn trip_state_cannot_bypass_requalification() {
        let manifest = transition_manifest();
        let attempt = transition_attempt(
            "bypass-trip",
            "protection-trip",
            "steady-operation",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::HoldSafe);
        assert!(report.diagnostics.contains(&ControlDiagnostic::TripRecoveryBypass));
    }

    #[test]
    fn successful_requalification_allows_declared_return_to_operation() {
        let manifest = transition_manifest();
        let attempt = transition_attempt(
            "requalify-to-operation",
            "requalification",
            "steady-operation",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::Passed,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Allowed);
    }

    #[test]
    fn failed_requalification_rejects_return_to_operation() {
        let manifest = transition_manifest();
        let attempt = transition_attempt(
            "requalify-to-operation",
            "requalification",
            "steady-operation",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::Failed,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Rejected);
    }

    #[test]
    fn transition_timeout_requires_recovery() {
        let manifest = transition_manifest();
        let mut attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Pending,
            RequalificationResult::NotRequired,
        );
        attempt.evaluated_at_ns = 200_000_000;
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::RecoveryRequired);
    }

    #[test]
    fn failed_transition_acknowledgement_requires_recovery() {
        let manifest = transition_manifest();
        let attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Failed,
            RequalificationResult::NotRequired,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::RecoveryRequired);
    }

    #[test]
    fn invalid_fallback_state_is_not_comparable() {
        let mut manifest = transition_manifest();
        manifest.transitions[0].fallback_state_id = "missing-state".to_string();
        let attempt = transition_attempt(
            "enter-recovery-hold",
            "steady-operation",
            "recovery-hold",
            TransitionAcknowledgement::Succeeded,
            RequalificationResult::NotRequired,
        );
        let report = evaluate_state_transition(&manifest, &attempt);
        assert_eq!(report.disposition, ControlDecisionDisposition::Rejected);
        assert!(report.diagnostics.contains(&ControlDiagnostic::MissingFallbackState));
    }
}
