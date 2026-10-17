// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Report-only effort-policy validation for assistive-mobility research.
//!
//! This module describes permitted *simulation requests*. It does not generate
//! actuator commands, set physical force limits, authenticate profile eligibility,
//! estimate human physiology, or establish clinical or hardware safety. All numeric
//! assistance/resistance values are abstract normalized policy units in [0, 1];
//! they are not torque, force, dose, or a hardware command.

use serde::{Deserialize, Serialize};

/// Current schema version for effort-policy requests and decisions.
pub const EFFORT_POLICY_SCHEMA_VERSION: u16 = 1;

/// User-selected goal for an assistive-mobility session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortModeV1 {
    /// Everyday mobility, access, comfort, and energy conservation.
    MobilityConserve,
    /// Bounded user participation with assistance when needed.
    AssistAsNeeded,
    /// Explicitly selected, profile-bounded resistance or training.
    TrainingResistive,
    /// Lower demand and offer additional support or rest.
    RecoveryHighSupport,
}

/// Declared state of evidence used when considering an effort increase.
///
/// These are caller-provided labels; this module does not authenticate sensors,
/// determine calibration, or infer a clinical state from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortEvidenceStatusV1 {
    /// The caller declares that the relevant evidence is present and current.
    Fresh,
    /// No relevant measurement is available.
    Unavailable,
    /// No voluntary motor signal is available for this task/profile.
    NoVoluntarySignal,
    /// The evidence is older than the caller's freshness policy permits.
    Stale,
    /// Relevant sources disagree beyond the caller's declared tolerance.
    Conflicted,
    /// The evidence failed the caller's validity checks.
    Invalid,
}

/// A normalized policy quantity checked by the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortFieldV1 {
    /// Abstract assistance request.
    RequestedAssistance,
    /// Abstract assistance ceiling.
    AssistanceLimit,
    /// Abstract resistance request.
    RequestedResistance,
    /// Abstract resistance ceiling.
    ResistanceLimit,
}

/// Stable, machine-readable findings from effort-policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortFindingV1 {
    /// Session identifier is empty.
    MissingSessionId,
    /// Selected profile identifier is empty.
    MissingSelectedProfileId,
    /// Active session identifier is empty.
    MissingActiveSessionId,
    /// Request session and active configuration session do not match.
    SessionMismatch,
    /// Active profile identifier is empty.
    MissingActiveProfileId,
    /// Selected profile and active configuration do not match.
    ProfileMismatch,
    /// Request uses a schema version this evaluator does not understand.
    UnsupportedSchemaVersion {
        /// Schema version declared by the request.
        received: u16,
    },
    /// A numeric value is NaN or infinite.
    NonFiniteValue {
        /// Quantity that contained a non-finite value.
        field: EffortFieldV1,
    },
    /// A normalized numeric value is outside [0, 1].
    OutsideNormalizedRange {
        /// Quantity outside the normalized range.
        field: EffortFieldV1,
    },
    /// Requested assistance or resistance exceeds its declared ceiling.
    RequestedAboveLimit {
        /// Quantity above the declared ceiling.
        field: EffortFieldV1,
    },
    /// Nonzero resistance was requested outside the training mode.
    ResistanceRequiresTrainingMode,
    /// Training mode lacks explicit opt-in.
    TrainingOptInRequired,
    /// Training mode lacks a caller-declared eligible profile.
    TrainingProfileNotEligible,
    /// An increase was requested without fresh effort evidence.
    ResistanceIncreaseRequiresFreshEvidence,
    /// A stop request conflicts with a nonzero resistive request.
    StopConflictsWithResistance,
    /// A recovery request conflicts with ongoing resistance.
    RecoveryConflictsWithResistance,
    /// A request for more assistance conflicts with ongoing resistance.
    MoreAssistanceConflictsWithResistance,
    /// The user explicitly requested stop; no request is admitted.
    UserRequestedStop,
    /// The user explicitly requested recovery/high support.
    UserRequestedRecovery,
    /// The user explicitly requested more assistance.
    UserRequestedMoreAssistance,
}

/// Report-only disposition of a policy request.
///
/// AdmissibleForSimulation is deliberately not hardware authorization, a safety
/// certificate, a medical decision, or an actuator command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortDecisionStatusV1 {
    /// All declared policy checks passed for simulation/reporting only.
    AdmissibleForSimulation,
    /// At least one policy check failed.
    Rejected,
    /// Stop was requested; stop takes precedence over any requested motion.
    StopRequired,
    /// Recovery/high-support behavior was requested.
    RecoveryRequired,
    /// More assistance was requested.
    SupportRequired,
}

/// Versioned request evaluated without touching any device or controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffortRequestV1 {
    /// Input schema version; currently must equal EFFORT_POLICY_SCHEMA_VERSION.
    pub schema_version: u16,
    /// Session requested by the caller.
    pub session_id: String,
    /// Session identity bound to the currently active configuration.
    pub active_session_id: String,
    /// Profile selected by the user or session coordinator.
    pub selected_profile_id: String,
    /// Profile actually bound to the active session configuration.
    pub active_profile_id: String,
    /// User-selected session mode.
    pub mode: EffortModeV1,
    /// Whether the user explicitly opted into this training session.
    pub explicit_training_opt_in: bool,
    /// Whether an upstream policy declares this profile eligible for this mode.
    ///
    /// This boolean is an untrusted declaration, not proof of clinical suitability.
    pub profile_training_eligible: bool,
    /// Declared state of evidence for any requested training progression.
    pub effort_evidence: EffortEvidenceStatusV1,
    /// Whether the user has requested an immediate stop.
    pub user_requested_stop: bool,
    /// Whether the user requested recovery or high-support behavior.
    pub user_requested_recovery: bool,
    /// Whether the user requested more assistance.
    pub user_requested_more_assistance: bool,
    /// Whether this request would increase resistance over the preexisting target.
    pub resistance_increase_requested: bool,
    /// Abstract normalized assistance request, in [0, 1].
    pub requested_assistance: f32,
    /// Abstract normalized assistance ceiling, in [0, 1].
    pub assistance_limit: f32,
    /// Abstract normalized resistance request, in [0, 1].
    pub requested_resistance: f32,
    /// Abstract normalized resistance ceiling, in [0, 1].
    pub resistance_limit: f32,
}

/// Deterministic report from effort-policy evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffortDecisionV1 {
    /// Output schema version.
    pub schema_version: u16,
    /// Source session identity, retained even when other fields are invalid.
    pub session_id: String,
    /// Report-only disposition.
    pub status: EffortDecisionStatusV1,
    /// Findings in a stable, documented validation order.
    pub findings: Vec<EffortFindingV1>,
}

/// Evaluate a request for simulation/reporting only.
///
/// Findings are appended in a fixed order so equivalent inputs produce equivalent
/// reports. Stop, recovery, and more-assistance requests take precedence in the
/// returned status. No returned status authorizes physical actuation.
pub fn evaluate_effort_request(request: &EffortRequestV1) -> EffortDecisionV1 {
    let mut findings = Vec::new();

    if request.session_id.trim().is_empty() {
        findings.push(EffortFindingV1::MissingSessionId);
    }
    if request.active_session_id.trim().is_empty() {
        findings.push(EffortFindingV1::MissingActiveSessionId);
    }
    if request.session_id != request.active_session_id {
        findings.push(EffortFindingV1::SessionMismatch);
    }
    if request.selected_profile_id.trim().is_empty() {
        findings.push(EffortFindingV1::MissingSelectedProfileId);
    }
    if request.active_profile_id.trim().is_empty() {
        findings.push(EffortFindingV1::MissingActiveProfileId);
    }
    if request.selected_profile_id != request.active_profile_id {
        findings.push(EffortFindingV1::ProfileMismatch);
    }
    if request.schema_version != EFFORT_POLICY_SCHEMA_VERSION {
        findings.push(EffortFindingV1::UnsupportedSchemaVersion {
            received: request.schema_version,
        });
    }

    let values = [
        (
            EffortFieldV1::RequestedAssistance,
            request.requested_assistance,
        ),
        (EffortFieldV1::AssistanceLimit, request.assistance_limit),
        (
            EffortFieldV1::RequestedResistance,
            request.requested_resistance,
        ),
        (EffortFieldV1::ResistanceLimit, request.resistance_limit),
    ];
    for (field, value) in values {
        if !value.is_finite() {
            findings.push(EffortFindingV1::NonFiniteValue { field });
        } else if !(0.0..=1.0).contains(&value) {
            findings.push(EffortFindingV1::OutsideNormalizedRange { field });
        }
    }

    if request.requested_assistance.is_finite()
        && request.assistance_limit.is_finite()
        && request.requested_assistance > request.assistance_limit
    {
        findings.push(EffortFindingV1::RequestedAboveLimit {
            field: EffortFieldV1::RequestedAssistance,
        });
    }
    if request.requested_resistance.is_finite()
        && request.resistance_limit.is_finite()
        && request.requested_resistance > request.resistance_limit
    {
        findings.push(EffortFindingV1::RequestedAboveLimit {
            field: EffortFieldV1::RequestedResistance,
        });
    }

    if request.mode != EffortModeV1::TrainingResistive
        && (request.requested_resistance != 0.0 || request.resistance_increase_requested)
    {
        findings.push(EffortFindingV1::ResistanceRequiresTrainingMode);
    }
    if request.mode == EffortModeV1::TrainingResistive {
        if !request.explicit_training_opt_in {
            findings.push(EffortFindingV1::TrainingOptInRequired);
        }
        if !request.profile_training_eligible {
            findings.push(EffortFindingV1::TrainingProfileNotEligible);
        }
        if request.resistance_increase_requested
            && request.effort_evidence != EffortEvidenceStatusV1::Fresh
        {
            findings.push(EffortFindingV1::ResistanceIncreaseRequiresFreshEvidence);
        }
    }

    if request.user_requested_stop
        && (request.requested_resistance > 0.0 || request.resistance_increase_requested)
    {
        findings.push(EffortFindingV1::StopConflictsWithResistance);
    }
    let recovery_requested =
        request.user_requested_recovery || request.mode == EffortModeV1::RecoveryHighSupport;
    if recovery_requested
        && (request.requested_resistance > 0.0 || request.resistance_increase_requested)
    {
        findings.push(EffortFindingV1::RecoveryConflictsWithResistance);
    }
    if request.user_requested_more_assistance
        && (request.requested_resistance > 0.0 || request.resistance_increase_requested)
    {
        findings.push(EffortFindingV1::MoreAssistanceConflictsWithResistance);
    }

    if request.user_requested_stop {
        findings.push(EffortFindingV1::UserRequestedStop);
    }
    if recovery_requested {
        findings.push(EffortFindingV1::UserRequestedRecovery);
    }
    if request.user_requested_more_assistance {
        findings.push(EffortFindingV1::UserRequestedMoreAssistance);
    }

    let status = if request.user_requested_stop {
        EffortDecisionStatusV1::StopRequired
    } else if recovery_requested {
        EffortDecisionStatusV1::RecoveryRequired
    } else if request.user_requested_more_assistance {
        EffortDecisionStatusV1::SupportRequired
    } else if findings.is_empty() {
        EffortDecisionStatusV1::AdmissibleForSimulation
    } else {
        EffortDecisionStatusV1::Rejected
    };

    EffortDecisionV1 {
        schema_version: EFFORT_POLICY_SCHEMA_VERSION,
        session_id: request.session_id.clone(),
        status,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_training_request() -> EffortRequestV1 {
        EffortRequestV1 {
            schema_version: EFFORT_POLICY_SCHEMA_VERSION,
            session_id: "session-001".to_owned(),
            active_session_id: "session-001".to_owned(),
            selected_profile_id: "profile-training-a".to_owned(),
            active_profile_id: "profile-training-a".to_owned(),
            mode: EffortModeV1::TrainingResistive,
            explicit_training_opt_in: true,
            profile_training_eligible: true,
            effort_evidence: EffortEvidenceStatusV1::Fresh,
            user_requested_stop: false,
            user_requested_recovery: false,
            user_requested_more_assistance: false,
            resistance_increase_requested: false,
            requested_assistance: 0.4,
            assistance_limit: 0.8,
            requested_resistance: 0.3,
            resistance_limit: 0.5,
        }
    }

    fn has_finding(decision: &EffortDecisionV1, expected: EffortFindingV1) -> bool {
        decision.findings.contains(&expected)
    }

    #[test]
    fn valid_opted_in_training_is_admissible_for_simulation_only() {
        let decision = evaluate_effort_request(&valid_training_request());
        assert_eq!(
            decision.status,
            EffortDecisionStatusV1::AdmissibleForSimulation
        );
        assert!(decision.findings.is_empty());
    }

    #[test]
    fn mobility_mode_rejects_nonzero_resistance() {
        let mut request = valid_training_request();
        request.mode = EffortModeV1::MobilityConserve;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::ResistanceRequiresTrainingMode
        ));
    }

    #[test]
    fn resistance_progression_is_forbidden_outside_training_mode() {
        let mut request = valid_training_request();
        request.mode = EffortModeV1::MobilityConserve;
        request.requested_resistance = 0.0;
        request.resistance_increase_requested = true;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::ResistanceRequiresTrainingMode
        ));
    }

    #[test]
    fn training_requires_opt_in_and_eligible_profile() {
        let mut request = valid_training_request();
        request.explicit_training_opt_in = false;
        request.profile_training_eligible = false;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert_eq!(
            decision.findings,
            vec![
                EffortFindingV1::TrainingOptInRequired,
                EffortFindingV1::TrainingProfileNotEligible
            ]
        );
    }

    #[test]
    fn stop_request_preempts_resistive_request() {
        let mut request = valid_training_request();
        request.user_requested_stop = true;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::StopRequired);
        assert!(has_finding(
            &decision,
            EffortFindingV1::UserRequestedStop
        ));
        assert!(has_finding(
            &decision,
            EffortFindingV1::StopConflictsWithResistance
        ));
        assert!(!matches!(
            decision.status,
            EffortDecisionStatusV1::AdmissibleForSimulation
        ));
    }

    #[test]
    fn recovery_request_preempts_resistive_request() {
        let mut request = valid_training_request();
        request.user_requested_recovery = true;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::RecoveryRequired);
        assert!(has_finding(
            &decision,
            EffortFindingV1::RecoveryConflictsWithResistance
        ));
    }

    #[test]
    fn recovery_mode_has_recovery_disposition_without_extra_flag() {
        let mut request = valid_training_request();
        request.mode = EffortModeV1::RecoveryHighSupport;
        request.requested_resistance = 0.0;
        request.resistance_increase_requested = false;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::RecoveryRequired);
        assert!(has_finding(
            &decision,
            EffortFindingV1::UserRequestedRecovery
        ));
        assert!(!has_finding(
            &decision,
            EffortFindingV1::ResistanceRequiresTrainingMode
        ));
    }

    #[test]
    fn more_assistance_request_preempts_resistive_request() {
        let mut request = valid_training_request();
        request.user_requested_more_assistance = true;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::SupportRequired);
        assert!(has_finding(
            &decision,
            EffortFindingV1::MoreAssistanceConflictsWithResistance
        ));
    }

    #[test]
    fn stale_session_configuration_fails_closed() {
        let mut request = valid_training_request();
        request.active_session_id = "previous-session".to_owned();
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(&decision, EffortFindingV1::SessionMismatch));
    }

    #[test]
    fn empty_identity_fields_fail_closed() {
        let mut request = valid_training_request();
        request.session_id.clear();
        request.active_session_id.clear();
        request.selected_profile_id.clear();
        request.active_profile_id.clear();
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(&decision, EffortFindingV1::MissingSessionId));
        assert!(has_finding(
            &decision,
            EffortFindingV1::MissingActiveSessionId
        ));
        assert!(has_finding(
            &decision,
            EffortFindingV1::MissingSelectedProfileId
        ));
        assert!(has_finding(
            &decision,
            EffortFindingV1::MissingActiveProfileId
        ));
    }

    #[test]
    fn profile_mismatch_fails_closed() {
        let mut request = valid_training_request();
        request.active_profile_id = "different-profile".to_owned();
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(&decision, EffortFindingV1::ProfileMismatch));
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut request = valid_training_request();
        request.requested_resistance = f32::NAN;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::NonFiniteValue {
                field: EffortFieldV1::RequestedResistance
            }
        ));
    }

    #[test]
    fn normalized_values_and_profile_limits_are_enforced() {
        let mut request = valid_training_request();
        request.requested_assistance = 1.2;
        request.requested_resistance = 0.7;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::OutsideNormalizedRange {
                field: EffortFieldV1::RequestedAssistance
            }
        ));
        assert!(has_finding(
            &decision,
            EffortFindingV1::RequestedAboveLimit {
                field: EffortFieldV1::RequestedResistance
            }
        ));
    }

    #[test]
    fn stale_effort_evidence_blocks_resistance_progression() {
        let mut request = valid_training_request();
        request.effort_evidence = EffortEvidenceStatusV1::Stale;
        request.resistance_increase_requested = true;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::ResistanceIncreaseRequiresFreshEvidence
        ));
    }

    #[test]
    fn no_voluntary_signal_is_not_itself_a_failure() {
        // A previously selected in-envelope profile is not labelled as
        // non-cooperation merely because voluntary effort cannot be observed.
        // Progression remains blocked until appropriate fresh evidence exists.
        let mut request = valid_training_request();
        request.effort_evidence = EffortEvidenceStatusV1::NoVoluntarySignal;
        let decision = evaluate_effort_request(&request);
        assert_eq!(
            decision.status,
            EffortDecisionStatusV1::AdmissibleForSimulation
        );
        assert!(decision.findings.is_empty());

        request.resistance_increase_requested = true;
        let progressed = evaluate_effort_request(&request);
        assert_eq!(progressed.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &progressed,
            EffortFindingV1::ResistanceIncreaseRequiresFreshEvidence
        ));
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let mut request = valid_training_request();
        request.schema_version = EFFORT_POLICY_SCHEMA_VERSION + 1;
        let decision = evaluate_effort_request(&request);
        assert_eq!(decision.status, EffortDecisionStatusV1::Rejected);
        assert!(has_finding(
            &decision,
            EffortFindingV1::UnsupportedSchemaVersion {
                received: EFFORT_POLICY_SCHEMA_VERSION + 1
            }
        ));
    }

    #[test]
    fn findings_and_serialized_report_are_deterministic() {
        let mut request = valid_training_request();
        request.explicit_training_opt_in = false;
        request.resistance_increase_requested = true;
        request.effort_evidence = EffortEvidenceStatusV1::Conflicted;

        let first = evaluate_effort_request(&request);
        for _ in 0..1_000 {
            let next = evaluate_effort_request(&request);
            assert_eq!(first, next);
            assert_eq!(
                serde_json::to_vec(&first).expect("decision serializes"),
                serde_json::to_vec(&next).expect("decision serializes")
            );
        }
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        let json = serde_json::to_string(&valid_training_request()).expect("request serializes");
        let with_unknown = json.trim_end_matches('}').to_owned() + ",\"surprise\":true}";
        assert!(serde_json::from_str::<EffortRequestV1>(&with_unknown).is_err());
    }
}
