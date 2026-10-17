// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Report-only lifecycle and wearability gate for exosuit research.
//!
//! This module classifies declared workflow evidence for synthetic/replay use.
//! It cannot confirm physical latch engagement, skin condition, safe pressure,
//! or hardware disengagement. It emits no actuator commands and is not a safety
//! controller. Evidence values are caller-provided assertions unless independently
//! measured and traced by a separate validated system.

use serde::{Deserialize, Serialize};

/// Current schema version for wearability inputs and decisions.
pub const WEARABILITY_SCHEMA_VERSION: u16 = 1;

fn default_schema_version() -> u16 {
    WEARABILITY_SCHEMA_VERSION
}

/// Lifecycle stage being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityStageV1 {
    /// Retrieve device and evaluate task energy.
    Prepare,
    /// Put the wearable robot on.
    Don,
    /// Verify fastening, profile and fit before use.
    FitCheck,
    /// Select a mode and start a session.
    Start,
    /// Perform a declared task.
    Perform,
    /// Remove or disengage the wearable robot.
    Doff,
    /// Clean, inspect, repair or service the device.
    Maintain,
}

/// Status of evidence about a required fastening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FasteningEvidenceV1 {
    /// Caller declares that all required fastenings were verified.
    Verified,
    /// A required fastening is incomplete.
    Incomplete,
    /// Evidence is not available or not current.
    Unknown,
}

/// Status of selected-profile compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileMatchV1 {
    /// Declared wearer/device profile match was verified by the chosen procedure.
    Verified,
    /// Profile mismatch was identified.
    Mismatch,
    /// Compatibility is unknown.
    Unknown,
}

/// Freshness/validity status for fit evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitEvidenceV1 {
    /// Fit evidence is current under the declared profile policy.
    Current,
    /// Fit evidence aged or fit may have shifted.
    Stale,
    /// Sources disagree.
    Conflicted,
    /// No relevant evidence is available.
    Unavailable,
    /// Evidence failed declared validity checks.
    Invalid,
}

/// Status of a required or optional measurement channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalEvidenceV1 {
    /// A measurement is available; this does not prove calibration or accuracy.
    Available,
    /// Sources disagree or the channel failed a consistency check.
    Conflicted,
    /// The measurement is missing or unavailable.
    Unavailable,
}

/// Whether a comfort report has been collected for this scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfortReportV1 {
    /// A wearer comfort report was collected.
    Collected,
    /// No wearer comfort report was collected.
    NotCollected,
}

/// Declared support requirement for the donning workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowSupportV1 {
    /// This workflow requires a caregiver.
    CaregiverRequired,
    /// This workflow can be completed with or without a caregiver.
    CaregiverOptional,
    /// This declared profile claims wearer-only completion.
    WearerOnly,
}

/// Presence of a pending motion request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestPresenceV1 {
    /// A motion request is present.
    Present,
    /// No motion request is pending.
    Absent,
}

/// Declared session mode at the lifecycle boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityModeV1 {
    /// Everyday mobility and conservation.
    MobilityConserve,
    /// User participation with assistance as needed.
    AssistAsNeeded,
    /// Explicitly selected, bounded training mode.
    TrainingResistive,
    /// Recovery/high-support mode.
    RecoveryHighSupport,
    /// No current mode has been selected.
    Unset,
}

/// Status of configuration lineage across a restart or service event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigurationLineageV1 {
    /// Current configuration was loaded across the declared boundary.
    Verified,
    /// A restart boundary invalidated assumptions from the prior session.
    RestartBoundary,
    /// Configuration lineage cannot be verified.
    Unknown,
}

/// State of a predicted energy reserve for the requested task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyReserveV1 {
    /// The declared prediction is below the profile-specific minimum.
    BelowProfileMinimum,
    /// The declared prediction meets its profile-specific minimum.
    MeetsProfileMinimum,
    /// The prediction or its profile-specific minimum is unavailable.
    Unknown,
}

/// Confidence declaration for an energy estimate. This is not calibration proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateConfidenceV1 {
    /// Caller declares low confidence.
    DeclaredLow,
    /// Caller declares the estimate meets its documented confidence gate.
    Adequate,
    /// Confidence is not assessed.
    Unknown,
}

/// State of post-service calibration lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationLineageV1 {
    /// Calibration was verified preserved or re-established by a declared procedure.
    VerifiedPreservedOrReestablished,
    /// Calibration is missing.
    Missing,
    /// The service event invalidated the former calibration.
    Invalidated,
}

/// Result of declared non-safety-critical service diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticResultV1 {
    /// Declared checks completed successfully; does not prove hardware safety.
    PassDeclaredChecks,
    /// A declared check failed.
    Failed,
    /// Checks have not been run.
    NotRun,
}

/// Versioned input assertions used by the deterministic wearability evaluator.
///
/// All fields are optional because different lifecycle stages require different
/// evidence. Missing evidence is distinct from a measured zero and is never
/// silently converted into a pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct WearabilityInputV1 {
    /// Input schema version; defaults to version 1 for fixture compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u16,
    /// Required-fastening evidence for fit/start checks.
    pub fastening_evidence: Option<FasteningEvidenceV1>,
    /// Wearer/device profile compatibility evidence.
    pub profile_match: Option<ProfileMatchV1>,
    /// Current state of the fit evidence.
    pub fit_evidence: Option<FitEvidenceV1>,
    /// Consistency of relevant sensors.
    pub sensor_consistency: Option<SignalEvidenceV1>,
    /// Pressure-channel evidence, when included by the profile.
    pub pressure_sensor: Option<SignalEvidenceV1>,
    /// Shear-channel evidence, when included by the profile.
    pub shear_sensor: Option<SignalEvidenceV1>,
    /// Whether a wearer comfort report was collected.
    pub user_comfort_report: Option<ComfortReportV1>,
    /// Declared support needs for donning.
    pub workflow_support_requirement: Option<WorkflowSupportV1>,
    /// Whether the required caregiver is available.
    pub caregiver_available: Option<bool>,
    /// Explicit user request to remove/disengage.
    pub user_requested_doff: Option<bool>,
    /// Whether a motion request is currently pending.
    pub active_motion_request: Option<RequestPresenceV1>,
    /// Mode carried from the previous session, if any.
    pub prior_session_mode: Option<WearabilityModeV1>,
    /// Mode selected for the current session.
    pub current_session_mode: Option<WearabilityModeV1>,
    /// Configuration lineage across restart or transition.
    pub configuration_lineage: Option<ConfigurationLineageV1>,
    /// Whether task energy can be estimated from declared data.
    pub task_energy_estimate: Option<SignalEvidenceV1>,
    /// Declared prediction for reserve after the proposed task.
    pub predicted_reserve_after_task: Option<EnergyReserveV1>,
    /// Confidence declaration for the estimate.
    pub estimate_confidence: Option<EstimateConfidenceV1>,
    /// Whether a module was changed during service.
    pub module_changed: Option<bool>,
    /// Calibration lineage after service.
    pub calibration_lineage_after_service: Option<CalibrationLineageV1>,
    /// Result of declared diagnostics after service.
    pub diagnostic_result: Option<DiagnosticResultV1>,
    /// Whether the donning workflow exceeded its profile-specific time budget.
    pub workflow_timeout: Option<bool>,
    /// Whether the required critical workflow step is complete.
    pub critical_step_completed: Option<bool>,
    /// Interface-migration sensor status, when available.
    pub interface_migration_sensor: Option<SignalEvidenceV1>,
    /// Measured migration value, in caller-defined units; no universal threshold is implied.
    pub measured_migration_value: Option<f32>,
}

/// Report-only disposition of a lifecycle evidence check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityDispositionV1 {
    /// The declared evidence is sufficient to proceed in synthetic simulation only.
    ReadyForSimulation,
    /// Start cannot be admitted because required profile/fastening evidence is missing or invalid.
    StartNotAdmitted,
    /// A fresh fit check is required.
    FitRecheckRequired,
    /// Required interface evidence is unknown, conflicting or missing.
    InterfaceStateUnknown,
    /// A caregiver-dependent workflow is blocked, with a support path required.
    WorkflowBlockedWithSupportPath,
    /// User-requested doff/stop takes precedence over pending new motion.
    DoffRequestPreemptsNewMotion,
    /// Current session requires explicit mode selection.
    ExplicitModeSelectionRequired,
    /// Task plan requires review or fallback due to energy uncertainty/shortfall.
    TaskPlanRequiresReviewOrFallback,
    /// Device cannot be treated as returned to service.
    ReturnToServiceNotAdmitted,
    /// Workflow remains incomplete; offer accessible help/options without penalty.
    RemainUnreadyAndOfferAccessibleOptions,
    /// A required interface-migration metric is not available.
    MigrationMetricMissing,
    /// Input is not supported or lacks required identity/schema metadata.
    InvalidInput,
}

/// Deterministic decision receipt. This output is not a command or safety claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WearabilityDecisionV1 {
    /// Output schema version.
    pub schema_version: u16,
    /// Profile identifier for this evaluated scenario.
    pub profile_id: String,
    /// Lifecycle stage being evaluated.
    pub stage: WearabilityStageV1,
    /// Report-only disposition.
    pub disposition: WearabilityDispositionV1,
}

/// Evaluate one lifecycle scenario using stable, explicit priority rules.
///
/// User-requested doff has highest priority. Subsequent rules check the evidence
/// relevant to the selected lifecycle stage. Missing/unknown evidence never becomes
/// a pass. This evaluator does not independently observe or control a physical suit.
pub fn evaluate_wearability_request(
    profile_id: &str,
    stage: WearabilityStageV1,
    input: &WearabilityInputV1,
) -> WearabilityDecisionV1 {
    let disposition = if profile_id.trim().is_empty()
        || profile_id.trim() != profile_id
        || input.schema_version != WEARABILITY_SCHEMA_VERSION
    {
        WearabilityDispositionV1::InvalidInput
    } else if input.user_requested_doff == Some(true) {
        WearabilityDispositionV1::DoffRequestPreemptsNewMotion
    } else {
        match stage {
            WearabilityStageV1::FitCheck => evaluate_fit_check(input),
            WearabilityStageV1::Start => evaluate_start(input),
            WearabilityStageV1::Don => evaluate_don(input),
            WearabilityStageV1::Prepare => evaluate_prepare(input),
            WearabilityStageV1::Maintain => evaluate_maintain(input),
            WearabilityStageV1::Perform => evaluate_perform(input),
            WearabilityStageV1::Doff => WearabilityDispositionV1::DoffRequestPreemptsNewMotion,
        }
    };

    WearabilityDecisionV1 {
        schema_version: WEARABILITY_SCHEMA_VERSION,
        profile_id: profile_id.to_owned(),
        stage,
        disposition,
    }
}

fn evaluate_fit_check(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.profile_match == Some(ProfileMatchV1::Mismatch)
        || input.fastening_evidence == Some(FasteningEvidenceV1::Incomplete)
    {
        return WearabilityDispositionV1::StartNotAdmitted;
    }
    if input.fit_evidence == Some(FitEvidenceV1::Stale) {
        return WearabilityDispositionV1::FitRecheckRequired;
    }
    if fit_or_interface_unknown(input) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    if input.profile_match != Some(ProfileMatchV1::Verified)
        || input.fastening_evidence != Some(FasteningEvidenceV1::Verified)
        || input.fit_evidence != Some(FitEvidenceV1::Current)
    {
        return WearabilityDispositionV1::StartNotAdmitted;
    }
    WearabilityDispositionV1::ReadyForSimulation
}

fn evaluate_start(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.configuration_lineage == Some(ConfigurationLineageV1::RestartBoundary)
        || input.configuration_lineage == Some(ConfigurationLineageV1::Unknown)
        || input.current_session_mode == Some(WearabilityModeV1::Unset)
        || input.current_session_mode.is_none()
    {
        return WearabilityDispositionV1::ExplicitModeSelectionRequired;
    }
    if input.profile_match != Some(ProfileMatchV1::Verified)
        || input.fastening_evidence != Some(FasteningEvidenceV1::Verified)
    {
        return WearabilityDispositionV1::StartNotAdmitted;
    }
    if input.fit_evidence == Some(FitEvidenceV1::Stale) {
        return WearabilityDispositionV1::FitRecheckRequired;
    }
    if fit_or_interface_unknown(input) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    WearabilityDispositionV1::ReadyForSimulation
}

fn evaluate_don(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.workflow_timeout == Some(true)
        && input.critical_step_completed != Some(true)
    {
        return WearabilityDispositionV1::RemainUnreadyAndOfferAccessibleOptions;
    }
    if input.workflow_support_requirement == Some(WorkflowSupportV1::CaregiverRequired)
        && input.caregiver_available != Some(true)
    {
        return WearabilityDispositionV1::WorkflowBlockedWithSupportPath;
    }
    if input.critical_step_completed == Some(true) {
        WearabilityDispositionV1::ReadyForSimulation
    } else {
        WearabilityDispositionV1::RemainUnreadyAndOfferAccessibleOptions
    }
}

fn evaluate_prepare(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.predicted_reserve_after_task == Some(EnergyReserveV1::BelowProfileMinimum)
        || input.predicted_reserve_after_task == Some(EnergyReserveV1::Unknown)
        || input.task_energy_estimate != Some(SignalEvidenceV1::Available)
        || input.estimate_confidence != Some(EstimateConfidenceV1::Adequate)
    {
        return WearabilityDispositionV1::TaskPlanRequiresReviewOrFallback;
    }
    WearabilityDispositionV1::ReadyForSimulation
}

fn evaluate_maintain(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    let module_changed = input.module_changed == Some(true);
    if module_changed
        && (input.calibration_lineage_after_service
            != Some(CalibrationLineageV1::VerifiedPreservedOrReestablished)
            || input.diagnostic_result != Some(DiagnosticResultV1::PassDeclaredChecks))
    {
        return WearabilityDispositionV1::ReturnToServiceNotAdmitted;
    }
    if input.module_changed == Some(false) {
        return WearabilityDispositionV1::ReadyForSimulation;
    }
    if module_changed {
        return WearabilityDispositionV1::ReadyForSimulation;
    }
    WearabilityDispositionV1::InvalidInput
}

fn evaluate_perform(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.fit_evidence == Some(FitEvidenceV1::Stale) {
        return WearabilityDispositionV1::FitRecheckRequired;
    }
    if fit_or_interface_unknown(input) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    if input.interface_migration_sensor == Some(SignalEvidenceV1::Conflicted) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    if input.interface_migration_sensor == Some(SignalEvidenceV1::Unavailable)
        || input.measured_migration_value.is_none()
    {
        return WearabilityDispositionV1::MigrationMetricMissing;
    }
    if input.measured_migration_value.is_some_and(|value| !value.is_finite()) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    WearabilityDispositionV1::ReadyForSimulation
}

fn fit_or_interface_unknown(input: &WearabilityInputV1) -> bool {
    matches!(
        input.fit_evidence,
        Some(FitEvidenceV1::Conflicted | FitEvidenceV1::Unavailable | FitEvidenceV1::Invalid)
    ) || matches!(
        input.sensor_consistency,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    ) || matches!(
        input.pressure_sensor,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    ) || matches!(
        input.shear_sensor,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_fit() -> WearabilityInputV1 {
        WearabilityInputV1 {
            fastening_evidence: Some(FasteningEvidenceV1::Verified),
            profile_match: Some(ProfileMatchV1::Verified),
            fit_evidence: Some(FitEvidenceV1::Current),
            sensor_consistency: Some(SignalEvidenceV1::Available),
            ..Default::default()
        }
    }

    #[test]
    fn missing_fastening_never_passes_fit_check() {
        let mut input = valid_fit();
        input.fastening_evidence = Some(FasteningEvidenceV1::Incomplete);
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input).disposition,
            WearabilityDispositionV1::StartNotAdmitted
        );
    }

    #[test]
    fn stale_fit_requires_recheck() {
        let mut input = valid_fit();
        input.fit_evidence = Some(FitEvidenceV1::Stale);
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input).disposition,
            WearabilityDispositionV1::FitRecheckRequired
        );
    }

    #[test]
    fn user_doff_preempts_pending_motion() {
        let input = WearabilityInputV1 {
            user_requested_doff: Some(true),
            active_motion_request: Some(RequestPresenceV1::Present),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::DoffRequestPreemptsNewMotion
        );
    }

    #[test]
    fn restart_boundary_requires_explicit_mode_selection() {
        let input = WearabilityInputV1 {
            prior_session_mode: Some(WearabilityModeV1::TrainingResistive),
            current_session_mode: Some(WearabilityModeV1::Unset),
            configuration_lineage: Some(ConfigurationLineageV1::RestartBoundary),
            ..valid_fit()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Start, &input).disposition,
            WearabilityDispositionV1::ExplicitModeSelectionRequired
        );
    }

    #[test]
    fn insufficient_or_uncertain_energy_requires_review() {
        let input = WearabilityInputV1 {
            task_energy_estimate: Some(SignalEvidenceV1::Available),
            predicted_reserve_after_task: Some(EnergyReserveV1::BelowProfileMinimum),
            estimate_confidence: Some(EstimateConfidenceV1::Adequate),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Prepare, &input).disposition,
            WearabilityDispositionV1::TaskPlanRequiresReviewOrFallback
        );
    }

    #[test]
    fn service_cannot_return_to_service_without_calibration_and_diagnostics() {
        let input = WearabilityInputV1 {
            module_changed: Some(true),
            calibration_lineage_after_service: Some(CalibrationLineageV1::Missing),
            diagnostic_result: Some(DiagnosticResultV1::PassDeclaredChecks),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Maintain, &input).disposition,
            WearabilityDispositionV1::ReturnToServiceNotAdmitted
        );
    }

    #[test]
    fn missing_measurement_is_not_measured_zero() {
        let input = WearabilityInputV1 {
            interface_migration_sensor: Some(SignalEvidenceV1::Unavailable),
            measured_migration_value: None,
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::MigrationMetricMissing
        );
    }

    #[test]
    fn invalid_input_identity_fails_closed() {
        let input = valid_fit();
        assert_eq!(
            evaluate_wearability_request(" ", WearabilityStageV1::FitCheck, &input).disposition,
            WearabilityDispositionV1::InvalidInput
        );
    }
}
