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
use std::collections::HashSet;

/// Current schema version for wearability input requests.
pub const WEARABILITY_SCHEMA_VERSION: u16 = 1;
/// Current schema version for decision receipts, independently versioned from inputs.
pub const WEARABILITY_DECISION_SCHEMA_VERSION: u16 = 3;

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
    /// The relevant sensor sources agree under the caller's declared consistency check.
    Consistent,
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

/// Declared compatibility of a task with a selected exosuit profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCompatibilityV1 {
    /// This task was declared within the profile's scope.
    Supported,
    /// This task is explicitly outside the profile's scope.
    Unsupported,
    /// Compatibility has not been established.
    Unknown,
}

/// Clearance state for the current physical environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovementClearanceV1 {
    /// Declared movement envelope fits this environment.
    Compatible,
    /// Space or obstacle geometry conflicts with the declared movement envelope.
    Restricted,
    /// Environmental clearance is unknown.
    Unknown,
}

/// Wearer-reported thermal/comfort state; not a medical diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalComfortV1 {
    /// No thermal discomfort was reported for this observation.
    WithinPreference,
    /// The wearer reported thermal discomfort.
    DiscomfortReported,
    /// No wearer comfort report was collected.
    NotCollected,
    /// State has not been assessed.
    Unknown,
}

/// State of a declared task-specific wear-duration or duty-cycle constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DutyCycleV1 {
    /// The declared profile limit has not been reached.
    WithinProfileLimit,
    /// The declared profile limit has been reached.
    LimitReached,
    /// Duration or limit is unknown.
    Unknown,
}

/// Compatibility with a declared hygiene/workflow requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HygieneCompatibilityV1 {
    /// The declared task and device cleaning requirements are compatible.
    Compatible,
    /// The device's declared cleaning/workflow requirements conflict with the task.
    Incompatible,
    /// Compatibility has not been established.
    Unknown,
}

/// Input field whose declared source is recorded in an evidence-provenance receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityEvidenceFieldV1 {
    /// Required-fastening state.
    FasteningEvidence,
    /// Device/wearer profile match.
    ProfileMatch,
    /// Fit evidence status.
    FitEvidence,
    /// Relevant sensor consistency state.
    SensorConsistency,
    /// Pressure measurement channel status.
    PressureSensor,
    /// Shear measurement channel status.
    ShearSensor,
    /// User comfort report state.
    UserComfortReport,
    /// Task/profile suitability.
    TaskCompatibility,
    /// Movement-space clearance.
    MovementClearance,
    /// Wearer-reported thermal comfort.
    ThermalComfort,
    /// Declared profile duty-cycle state.
    DutyCycle,
    /// Task hygiene compatibility.
    HygieneCompatibility,
    /// Required support profile for donning.
    WorkflowSupportRequirement,
    /// Caregiver availability.
    CaregiverAvailable,
    /// User-requested disengagement.
    UserRequestedDoff,
    /// Pending motion-request state.
    ActiveMotionRequest,
    /// Prior session mode.
    PriorSessionMode,
    /// Current session mode.
    CurrentSessionMode,
    /// Configuration lineage.
    ConfigurationLineage,
    /// Task-energy estimate availability.
    TaskEnergyEstimate,
    /// Predicted reserve after task.
    PredictedReserveAfterTask,
    /// Declared energy-estimate confidence.
    EstimateConfidence,
    /// Whether a module changed during service.
    ModuleChanged,
    /// Calibration lineage after service.
    CalibrationLineageAfterService,
    /// Service diagnostic result.
    DiagnosticResult,
    /// Donning workflow timeout.
    WorkflowTimeout,
    /// Completion of the required critical step.
    CriticalStepCompleted,
    /// Interface migration sensor state.
    InterfaceMigrationSensor,
    /// Interface migration measurement value.
    MeasuredMigrationValue,
}

/// Declared origin class for evidence metadata. The evaluator cannot authenticate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityEvidenceOriginV1 {
    /// Caller-provided assertion or synthetic fixture value.
    CallerAssertion,
    /// Sensor observation, which needs a timestamp/timebase and calibration reference.
    SensorObservation,
    /// Independently acquired measurement, with a timestamp/timebase.
    IndependentMeasurement,
    /// Operator-entered annotation.
    OperatorAnnotation,
}

/// Provenance metadata associated with one input field.
///
/// Identifiers and timestamps improve traceability but do not authenticate a
/// sensor, establish calibration validity, or prove that a physical event occurred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WearabilityEvidenceProvenanceV1 {
    /// Field in the input whose value this record describes.
    pub field: WearabilityEvidenceFieldV1,
    /// Declared origin class.
    pub origin: WearabilityEvidenceOriginV1,
    /// Non-empty identifier for the source record or declared assertion.
    pub evidence_id: Option<String>,
    /// Timestamp ticks in the declared timebase, if provided.
    pub observed_at_ticks: Option<u64>,
    /// Clock/timebase identifier, required for timed sensor/independent observations.
    pub timebase_id: Option<String>,
    /// Calibration/configuration reference, required for sensor observations.
    pub calibration_reference: Option<String>,
}

/// Completeness of declared provenance for the fields present in an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityProvenanceStatusV1 {
    /// Every present input field has one structurally valid provenance record.
    CompleteForPresentInputs,
    /// One or more present input fields have no provenance record.
    MissingForPresentInputs,
    /// Provenance is duplicated or structurally invalid.
    Invalid,
}

/// Machine-readable reason that provenance is incomplete or malformed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityProvenanceFindingV1 {
    /// A present input field has no corresponding provenance record.
    MissingProvenance { field: WearabilityEvidenceFieldV1 },
    /// More than one provenance record describes the same field in schema v1.
    DuplicateProvenance { field: WearabilityEvidenceFieldV1 },
    /// Evidence ID is missing, empty, or has surrounding whitespace.
    InvalidEvidenceId { field: WearabilityEvidenceFieldV1 },
    /// Sensor/independent evidence has no observation timestamp.
    ObservationTimestampMissing { field: WearabilityEvidenceFieldV1 },
    /// Sensor/independent evidence has no clock/timebase identifier.
    TimebaseMissing { field: WearabilityEvidenceFieldV1 },
    /// Sensor evidence has no declared calibration/configuration reference.
    SensorCalibrationReferenceMissing { field: WearabilityEvidenceFieldV1 },
}

/// Versioned input assertions used by the deterministic wearability evaluator.
///
/// All fields are optional because different lifecycle stages require different
/// evidence. Missing evidence is distinct from a measured zero and is never
/// silently converted into a pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WearabilityInputV1 {
    /// Input schema version; defaults to version 1 for fixture compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u16,
    /// Declared source metadata for populated input fields. Empty means provenance missing.
    pub evidence_provenance: Vec<WearabilityEvidenceProvenanceV1>,
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
    /// Declared task suitability for the selected profile.
    pub task_compatibility: Option<TaskCompatibilityV1>,
    /// Movement-space suitability for the current environment.
    pub movement_clearance: Option<MovementClearanceV1>,
    /// Wearer-reported thermal comfort; not a safety sensor.
    pub thermal_comfort: Option<ThermalComfortV1>,
    /// Whether the profile-specific duty-cycle limit has been reached.
    pub duty_cycle: Option<DutyCycleV1>,
    /// Compatibility with the current task's hygiene requirements.
    pub hygiene_compatibility: Option<HygieneCompatibilityV1>,
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

/// Creates an empty input with the current schema version.
impl Default for WearabilityInputV1 {
    fn default() -> Self {
        Self {
            schema_version: WEARABILITY_SCHEMA_VERSION,
            evidence_provenance: Vec::new(),
            fastening_evidence: None,
            profile_match: None,
            fit_evidence: None,
            sensor_consistency: None,
            pressure_sensor: None,
            shear_sensor: None,
            user_comfort_report: None,
            task_compatibility: None,
            movement_clearance: None,
            thermal_comfort: None,
            duty_cycle: None,
            hygiene_compatibility: None,
            workflow_support_requirement: None,
            caregiver_available: None,
            user_requested_doff: None,
            active_motion_request: None,
            prior_session_mode: None,
            current_session_mode: None,
            configuration_lineage: None,
            task_energy_estimate: None,
            predicted_reserve_after_task: None,
            estimate_confidence: None,
            module_changed: None,
            calibration_lineage_after_service: None,
            diagnostic_result: None,
            workflow_timeout: None,
            critical_step_completed: None,
            interface_migration_sensor: None,
            measured_migration_value: None,
        }
    }
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
    /// Declared service checks passed; a responsible reviewer may consider return to service.
    /// This disposition is not physical safety certification.
    EligibleForReviewedReturnToService,
    /// Workflow remains incomplete; offer accessible help/options without penalty.
    RemainUnreadyAndOfferAccessibleOptions,
    /// A required interface-migration metric is not available.
    MigrationMetricMissing,
    /// The current task/environment is outside the selected profile's declared scope.
    TaskNotAdmittedForProfile,
    /// Task compatibility, comfort or duty-cycle evidence requires review before proceeding.
    TaskReviewOrAlternateProfileRequired,
    /// Input is not supported or lacks required identity/schema metadata.
    InvalidInput,
}

/// Stable machine-readable reasons retained in a wearability decision receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearabilityFindingV1 {
    /// Profile identifier is empty or contains surrounding whitespace.
    InvalidProfileId,
    /// Input schema version is unsupported.
    UnsupportedSchemaVersion,
    /// User explicitly requested disengagement/doff.
    UserRequestedDoff,
    /// Workflow requires a doff stage; that stage outranks ordinary motion planning.
    DoffStageHasPriority,
    /// A required fastening is explicitly incomplete.
    RequiredFasteningIncomplete,
    /// Required fastening evidence is missing or unknown.
    RequiredFasteningEvidenceMissing,
    /// Wearer/device profile was identified as incompatible.
    ProfileMismatch,
    /// Profile compatibility evidence is missing or unknown.
    ProfileCompatibilityUnknown,
    /// Fit evidence is stale.
    FitEvidenceStale,
    /// Fit evidence is missing, invalid, conflicted, or unavailable.
    FitEvidenceNotCurrentOrUncertain,
    /// Relevant sensors are unavailable or contradictory.
    SensorEvidenceUncertain,
    /// Required caregiver is unavailable for this workflow.
    CaregiverUnavailable,
    /// Donning timed out before the critical workflow step completed.
    DonningTimedOutBeforeCriticalStep,
    /// A critical donning step is incomplete.
    CriticalDonningStepIncomplete,
    /// Restart/configuration lineage requires explicit mode selection.
    ExplicitModeSelectionRequired,
    /// Energy reserve is below the profile-specific minimum.
    EnergyReserveBelowProfileMinimum,
    /// Energy estimate, reserve evidence, or confidence is unavailable/insufficient.
    EnergyEstimateUncertain,
    /// A module changed but calibration lineage is missing or invalid.
    CalibrationLineageNotVerified,
    /// Post-service diagnostic checks were not passed.
    PostServiceDiagnosticsNotPassed,
    /// Interface migration sensor reports conflict.
    MigrationSensorConflicted,
    /// Interface migration evidence is unavailable.
    MigrationMeasurementUnavailable,
    /// Interface migration value is non-finite.
    MigrationMeasurementInvalid,
    /// Task is outside the selected profile's scope.
    TaskOutsideProfile,
    /// Environment clearance conflicts with the profile.
    MovementClearanceRestricted,
    /// Task/environment compatibility is unknown.
    TaskCompatibilityUnknown,
    /// Wearer reported thermal discomfort.
    ThermalDiscomfortReported,
    /// Thermal comfort has not been assessed.
    ThermalComfortUnknown,
    /// Declared profile duty-cycle limit has been reached.
    DutyCycleLimitReached,
    /// Duty-cycle state or limit is unknown.
    DutyCycleUnknown,
    /// Task hygiene requirements conflict with the device/workflow.
    HygieneWorkflowIncompatible,
    /// Hygiene compatibility is unknown.
    HygieneCompatibilityUnknown,
    /// A simulated workflow is admitted under the declared fit/fastening evidence.
    FitCheckAdmittedForSimulation,
    /// Lifecycle task is admitted under current declared evidence.
    TaskAdmittedForSimulation,
    /// Declared module service is eligible for a separate review.
    ServiceEligibleForReviewOnly,
    /// The evaluator could not establish the evidence needed to decide.
    RequiredEvidenceUnknown,
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
    /// Ordered machine-readable policy reasons, using a stable evaluator-defined order.
    pub findings: Vec<WearabilityFindingV1>,
    /// Declared provenance copied into the decision receipt for audit/replay.
    pub evidence_provenance: Vec<WearabilityEvidenceProvenanceV1>,
    /// Provenance completeness for the populated input fields; not proof of truth.
    pub provenance_status: WearabilityProvenanceStatusV1,
    /// Stable, ordered reasons for missing or malformed provenance.
    pub provenance_findings: Vec<WearabilityProvenanceFindingV1>,
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
    let disposition = if input.user_requested_doff == Some(true) {
        // Explicit disengagement intent remains visible even if other metadata is invalid.
        WearabilityDispositionV1::DoffRequestPreemptsNewMotion
    } else if profile_id.trim().is_empty()
        || profile_id.trim() != profile_id
        || input.schema_version != WEARABILITY_SCHEMA_VERSION
    {
        WearabilityDispositionV1::InvalidInput
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

    let findings = collect_wearability_findings(profile_id, stage, input, disposition);
    let (provenance_status, provenance_findings) = validate_wearability_provenance(input);

    WearabilityDecisionV1 {
        schema_version: WEARABILITY_DECISION_SCHEMA_VERSION,
        profile_id: profile_id.to_owned(),
        stage,
        disposition,
        findings,
        evidence_provenance: input.evidence_provenance.clone(),
        provenance_status,
        provenance_findings,
    }
}

fn validate_wearability_provenance(
    input: &WearabilityInputV1,
) -> (WearabilityProvenanceStatusV1, Vec<WearabilityProvenanceFindingV1>) {
    use WearabilityEvidenceFieldV1 as Field;
    use WearabilityEvidenceOriginV1 as Origin;
    use WearabilityProvenanceFindingV1 as Finding;
    use WearabilityProvenanceStatusV1 as Status;

    let mut present_fields = Vec::new();
    macro_rules! present {
        ($field:ident, $variant:ident) => {
            if input.$field.is_some() {
                present_fields.push(Field::$variant);
            }
        };
    }
    present!(fastening_evidence, FasteningEvidence);
    present!(profile_match, ProfileMatch);
    present!(fit_evidence, FitEvidence);
    present!(sensor_consistency, SensorConsistency);
    present!(pressure_sensor, PressureSensor);
    present!(shear_sensor, ShearSensor);
    present!(user_comfort_report, UserComfortReport);
    present!(task_compatibility, TaskCompatibility);
    present!(movement_clearance, MovementClearance);
    present!(thermal_comfort, ThermalComfort);
    present!(duty_cycle, DutyCycle);
    present!(hygiene_compatibility, HygieneCompatibility);
    present!(workflow_support_requirement, WorkflowSupportRequirement);
    present!(caregiver_available, CaregiverAvailable);
    present!(user_requested_doff, UserRequestedDoff);
    present!(active_motion_request, ActiveMotionRequest);
    present!(prior_session_mode, PriorSessionMode);
    present!(current_session_mode, CurrentSessionMode);
    present!(configuration_lineage, ConfigurationLineage);
    present!(task_energy_estimate, TaskEnergyEstimate);
    present!(predicted_reserve_after_task, PredictedReserveAfterTask);
    present!(estimate_confidence, EstimateConfidence);
    present!(module_changed, ModuleChanged);
    present!(calibration_lineage_after_service, CalibrationLineageAfterService);
    present!(diagnostic_result, DiagnosticResult);
    present!(workflow_timeout, WorkflowTimeout);
    present!(critical_step_completed, CriticalStepCompleted);
    present!(interface_migration_sensor, InterfaceMigrationSensor);
    present!(measured_migration_value, MeasuredMigrationValue);

    let mut findings = Vec::new();
    for field in present_fields.iter().copied() {
        if !input.evidence_provenance.iter().any(|entry| entry.field == field) {
            findings.push(Finding::MissingProvenance { field });
        }
    }

    let mut seen = HashSet::new();
    for entry in &input.evidence_provenance {
        if !seen.insert(entry.field) {
            findings.push(Finding::DuplicateProvenance { field: entry.field });
        }
        if entry.evidence_id.as_deref().is_none_or(|id| id.trim().is_empty() || id.trim() != id) {
            findings.push(Finding::InvalidEvidenceId { field: entry.field });
        }
        if matches!(entry.origin, Origin::SensorObservation | Origin::IndependentMeasurement)
            && entry.observed_at_ticks.is_none()
        {
            findings.push(Finding::ObservationTimestampMissing { field: entry.field });
        }
        if matches!(entry.origin, Origin::SensorObservation | Origin::IndependentMeasurement)
            && entry.timebase_id.as_deref().is_none_or(|id| id.trim().is_empty() || id.trim() != id)
        {
            findings.push(Finding::TimebaseMissing { field: entry.field });
        }
        if entry.origin == Origin::SensorObservation
            && entry.calibration_reference.as_deref().is_none_or(|id| id.trim().is_empty() || id.trim() != id)
        {
            findings.push(Finding::SensorCalibrationReferenceMissing { field: entry.field });
        }
    }

    let invalid = findings.iter().any(|finding| !matches!(finding, Finding::MissingProvenance { .. }));
    let status = if invalid {
        Status::Invalid
    } else if !findings.is_empty() {
        Status::MissingForPresentInputs
    } else {
        Status::CompleteForPresentInputs
    };
    (status, findings)
}

fn collect_wearability_findings(
    profile_id: &str,
    stage: WearabilityStageV1,
    input: &WearabilityInputV1,
    disposition: WearabilityDispositionV1,
) -> Vec<WearabilityFindingV1> {
    use WearabilityFindingV1 as F;
    let mut findings = Vec::new();

    // Finding order is deliberately fixed; do not sort by debug/string output.
    if profile_id.trim().is_empty() || profile_id.trim() != profile_id {
        findings.push(F::InvalidProfileId);
    }
    if input.schema_version != WEARABILITY_SCHEMA_VERSION {
        findings.push(F::UnsupportedSchemaVersion);
    }
    if input.user_requested_doff == Some(true) {
        findings.push(F::UserRequestedDoff);
    }
    if stage == WearabilityStageV1::Doff {
        findings.push(F::DoffStageHasPriority);
    }
    if input.fastening_evidence == Some(FasteningEvidenceV1::Incomplete) {
        findings.push(F::RequiredFasteningIncomplete);
    } else if input.fastening_evidence.is_none()
        && matches!(stage, WearabilityStageV1::FitCheck | WearabilityStageV1::Start)
    {
        findings.push(F::RequiredFasteningEvidenceMissing);
    }
    match input.profile_match {
        Some(ProfileMatchV1::Mismatch) => findings.push(F::ProfileMismatch),
        Some(ProfileMatchV1::Unknown) | None
            if matches!(stage, WearabilityStageV1::FitCheck | WearabilityStageV1::Start) =>
        {
            findings.push(F::ProfileCompatibilityUnknown);
        }
        _ => {}
    }
    match input.fit_evidence {
        Some(FitEvidenceV1::Stale) => findings.push(F::FitEvidenceStale),
        Some(FitEvidenceV1::Conflicted | FitEvidenceV1::Unavailable | FitEvidenceV1::Invalid)
        | None
            if matches!(stage, WearabilityStageV1::FitCheck | WearabilityStageV1::Start | WearabilityStageV1::Perform) =>
        {
            findings.push(F::FitEvidenceNotCurrentOrUncertain);
        }
        _ => {}
    }
    if matches!(
        input.sensor_consistency,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    ) || matches!(
        input.pressure_sensor,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    ) || matches!(
        input.shear_sensor,
        Some(SignalEvidenceV1::Conflicted | SignalEvidenceV1::Unavailable)
    ) {
        findings.push(F::SensorEvidenceUncertain);
    }
    if input.workflow_support_requirement == Some(WorkflowSupportV1::CaregiverRequired)
        && input.caregiver_available != Some(true)
    {
        findings.push(F::CaregiverUnavailable);
    }
    if input.workflow_timeout == Some(true) && input.critical_step_completed != Some(true) {
        findings.push(F::DonningTimedOutBeforeCriticalStep);
    }
    if stage == WearabilityStageV1::Don && input.critical_step_completed != Some(true) {
        findings.push(F::CriticalDonningStepIncomplete);
    }
    if input.configuration_lineage == Some(ConfigurationLineageV1::RestartBoundary)
        || input.configuration_lineage == Some(ConfigurationLineageV1::Unknown)
        || input.current_session_mode == Some(WearabilityModeV1::Unset)
        || (stage == WearabilityStageV1::Start && input.current_session_mode.is_none())
    {
        findings.push(F::ExplicitModeSelectionRequired);
    }
    if input.predicted_reserve_after_task == Some(EnergyReserveV1::BelowProfileMinimum) {
        findings.push(F::EnergyReserveBelowProfileMinimum);
    }
    if matches!(
        input.predicted_reserve_after_task,
        Some(EnergyReserveV1::Unknown) | None
    ) || input.task_energy_estimate != Some(SignalEvidenceV1::Available)
        || input.estimate_confidence != Some(EstimateConfidenceV1::Adequate)
    {
        if stage == WearabilityStageV1::Prepare {
            findings.push(F::EnergyEstimateUncertain);
        }
    }
    if input.module_changed == Some(true)
        && input.calibration_lineage_after_service
            != Some(CalibrationLineageV1::VerifiedPreservedOrReestablished)
    {
        findings.push(F::CalibrationLineageNotVerified);
    }
    if input.module_changed == Some(true)
        && input.diagnostic_result != Some(DiagnosticResultV1::PassDeclaredChecks)
    {
        findings.push(F::PostServiceDiagnosticsNotPassed);
    }
    if input.interface_migration_sensor == Some(SignalEvidenceV1::Conflicted) {
        findings.push(F::MigrationSensorConflicted);
    } else if input.interface_migration_sensor == Some(SignalEvidenceV1::Unavailable)
        || (input.interface_migration_sensor.is_some() && input.measured_migration_value.is_none())
    {
        findings.push(F::MigrationMeasurementUnavailable);
    }
    if input.measured_migration_value.is_some_and(|value| !value.is_finite()) {
        findings.push(F::MigrationMeasurementInvalid);
    }
    if input.task_compatibility == Some(TaskCompatibilityV1::Unsupported) {
        findings.push(F::TaskOutsideProfile);
    } else if input.task_compatibility == Some(TaskCompatibilityV1::Unknown) {
        findings.push(F::TaskCompatibilityUnknown);
    }
    if input.movement_clearance == Some(MovementClearanceV1::Restricted) {
        findings.push(F::MovementClearanceRestricted);
    }
    if input.thermal_comfort == Some(ThermalComfortV1::DiscomfortReported) {
        findings.push(F::ThermalDiscomfortReported);
    } else if input.thermal_comfort == Some(ThermalComfortV1::Unknown) {
        findings.push(F::ThermalComfortUnknown);
    }
    if input.duty_cycle == Some(DutyCycleV1::LimitReached) {
        findings.push(F::DutyCycleLimitReached);
    } else if input.duty_cycle == Some(DutyCycleV1::Unknown) {
        findings.push(F::DutyCycleUnknown);
    }
    if input.hygiene_compatibility == Some(HygieneCompatibilityV1::Incompatible) {
        findings.push(F::HygieneWorkflowIncompatible);
    } else if input.hygiene_compatibility == Some(HygieneCompatibilityV1::Unknown) {
        findings.push(F::HygieneCompatibilityUnknown);
    }

    if findings.is_empty() {
        match disposition {
            WearabilityDispositionV1::ReadyForSimulation => {
                if stage == WearabilityStageV1::FitCheck {
                    findings.push(F::FitCheckAdmittedForSimulation);
                } else {
                    findings.push(F::TaskAdmittedForSimulation);
                }
            }
            WearabilityDispositionV1::EligibleForReviewedReturnToService => {
                findings.push(F::ServiceEligibleForReviewOnly);
            }
            WearabilityDispositionV1::DoffRequestPreemptsNewMotion => {
                findings.push(F::DoffStageHasPriority);
            }
            _ => findings.push(F::RequiredEvidenceUnknown),
        }
    }

    findings
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
    if input.profile_match != Some(ProfileMatchV1::Verified)
        || input.fastening_evidence != Some(FasteningEvidenceV1::Verified)
    {
        return WearabilityDispositionV1::StartNotAdmitted;
    }
    if input.fit_evidence != Some(FitEvidenceV1::Current) || fit_or_interface_unknown(input) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
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
    if input.fit_evidence != Some(FitEvidenceV1::Current) || fit_or_interface_unknown(input) {
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
        return WearabilityDispositionV1::EligibleForReviewedReturnToService;
    }
    WearabilityDispositionV1::InvalidInput
}

fn evaluate_perform(input: &WearabilityInputV1) -> WearabilityDispositionV1 {
    if input.fit_evidence == Some(FitEvidenceV1::Stale) {
        return WearabilityDispositionV1::FitRecheckRequired;
    }
    if input.fit_evidence != Some(FitEvidenceV1::Current) || fit_or_interface_unknown(input) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    if let Some(task_disposition) = evaluate_task_context(input) {
        return task_disposition;
    }
    if input.interface_migration_sensor == Some(SignalEvidenceV1::Conflicted) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    if input.interface_migration_sensor == Some(SignalEvidenceV1::Unavailable)
        || (input.interface_migration_sensor.is_some() && input.measured_migration_value.is_none())
    {
        return WearabilityDispositionV1::MigrationMetricMissing;
    }
    if input.measured_migration_value.is_some_and(|value| !value.is_finite()) {
        return WearabilityDispositionV1::InterfaceStateUnknown;
    }
    WearabilityDispositionV1::ReadyForSimulation
}

fn evaluate_task_context(input: &WearabilityInputV1) -> Option<WearabilityDispositionV1> {
    if input.task_compatibility == Some(TaskCompatibilityV1::Unsupported)
        || input.movement_clearance == Some(MovementClearanceV1::Restricted)
        || input.hygiene_compatibility == Some(HygieneCompatibilityV1::Incompatible)
    {
        return Some(WearabilityDispositionV1::TaskNotAdmittedForProfile);
    }

    if input.task_compatibility == Some(TaskCompatibilityV1::Unknown)
        || input.movement_clearance == Some(MovementClearanceV1::Unknown)
        || input.hygiene_compatibility == Some(HygieneCompatibilityV1::Unknown)
        || input.thermal_comfort == Some(ThermalComfortV1::DiscomfortReported)
        || input.thermal_comfort == Some(ThermalComfortV1::Unknown)
        || input.duty_cycle == Some(DutyCycleV1::LimitReached)
        || input.duty_cycle == Some(DutyCycleV1::Unknown)
    {
        return Some(WearabilityDispositionV1::TaskReviewOrAlternateProfileRequired);
    }

    None
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
    fn unsupported_task_or_restricted_clearance_is_not_admitted() {
        let input = WearabilityInputV1 {
            fit_evidence: Some(FitEvidenceV1::Current),
            task_compatibility: Some(TaskCompatibilityV1::Unsupported),
            movement_clearance: Some(MovementClearanceV1::Compatible),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::TaskNotAdmittedForProfile
        );

        let restricted = WearabilityInputV1 {
            task_compatibility: Some(TaskCompatibilityV1::Supported),
            movement_clearance: Some(MovementClearanceV1::Restricted),
            ..input
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &restricted).disposition,
            WearabilityDispositionV1::TaskNotAdmittedForProfile
        );
    }

    #[test]
    fn reported_thermal_discomfort_requires_review_not_a_medical_diagnosis() {
        let input = WearabilityInputV1 {
            fit_evidence: Some(FitEvidenceV1::Current),
            task_compatibility: Some(TaskCompatibilityV1::Supported),
            movement_clearance: Some(MovementClearanceV1::Compatible),
            thermal_comfort: Some(ThermalComfortV1::DiscomfortReported),
            duty_cycle: Some(DutyCycleV1::WithinProfileLimit),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::TaskReviewOrAlternateProfileRequired
        );
    }

    #[test]
    fn duty_cycle_limit_requires_review() {
        let input = WearabilityInputV1 {
            fit_evidence: Some(FitEvidenceV1::Current),
            task_compatibility: Some(TaskCompatibilityV1::Supported),
            movement_clearance: Some(MovementClearanceV1::Compatible),
            duty_cycle: Some(DutyCycleV1::LimitReached),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::TaskReviewOrAlternateProfileRequired
        );
    }

    #[test]
    fn reviewed_service_requires_calibration_and_diagnostics_to_be_declared() {
        let input = WearabilityInputV1 {
            module_changed: Some(true),
            calibration_lineage_after_service: Some(
                CalibrationLineageV1::VerifiedPreservedOrReestablished,
            ),
            diagnostic_result: Some(DiagnosticResultV1::PassDeclaredChecks),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Maintain, &input).disposition,
            WearabilityDispositionV1::EligibleForReviewedReturnToService
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
    fn perform_requires_current_fit_evidence() {
        let input = WearabilityInputV1 {
            sensor_consistency: Some(SignalEvidenceV1::Consistent),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::InterfaceStateUnknown
        );
    }

    #[test]
    fn optional_migration_sensor_does_not_block_an_uninstrumented_profile() {
        let input = WearabilityInputV1 {
            fit_evidence: Some(FitEvidenceV1::Current),
            sensor_consistency: Some(SignalEvidenceV1::Consistent),
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input).disposition,
            WearabilityDispositionV1::ReadyForSimulation
        );
    }

    #[test]
    fn unsupported_input_schema_fails_closed() {
        let input = WearabilityInputV1 {
            schema_version: WEARABILITY_SCHEMA_VERSION + 1,
            ..Default::default()
        };
        assert_eq!(
            evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input).disposition,
            WearabilityDispositionV1::InvalidInput
        );
    }

    #[test]
    fn provenance_is_complete_only_when_each_present_field_has_a_record() {
        let mut input = valid_fit();
        input.evidence_provenance = vec![
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::FasteningEvidence,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fixture-fastening".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::ProfileMatch,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fixture-profile".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::FitEvidence,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fixture-fit".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::SensorConsistency,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fixture-sensor-consistency".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
        ];

        let result = evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input);
        assert_eq!(
            result.provenance_status,
            WearabilityProvenanceStatusV1::CompleteForPresentInputs
        );
        assert!(result.provenance_findings.is_empty());
        assert_eq!(result.evidence_provenance, input.evidence_provenance);
    }

    #[test]
    fn missing_provenance_is_distinct_from_policy_disposition() {
        let input = valid_fit();
        let result = evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input);
        assert_eq!(result.disposition, WearabilityDispositionV1::ReadyForSimulation);
        assert_eq!(
            result.provenance_status,
            WearabilityProvenanceStatusV1::MissingForPresentInputs
        );
        assert_eq!(result.provenance_findings.len(), 4);
        assert!(result.findings.contains(&WearabilityFindingV1::FitCheckAdmittedForSimulation));
    }

    #[test]
    fn sensor_provenance_requires_timestamp_timebase_and_calibration_reference() {
        let mut input = valid_fit();
        input.pressure_sensor = Some(SignalEvidenceV1::Available);
        input.evidence_provenance = vec![
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::FasteningEvidence,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fastening".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::ProfileMatch,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("profile".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::FitEvidence,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("fit".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::SensorConsistency,
                origin: WearabilityEvidenceOriginV1::CallerAssertion,
                evidence_id: Some("sensor-consistency".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
            WearabilityEvidenceProvenanceV1 {
                field: WearabilityEvidenceFieldV1::PressureSensor,
                origin: WearabilityEvidenceOriginV1::SensorObservation,
                evidence_id: Some("pressure-observation".to_owned()),
                observed_at_ticks: None,
                timebase_id: None,
                calibration_reference: None,
            },
        ];

        let result = evaluate_wearability_request("profile-a", WearabilityStageV1::FitCheck, &input);
        assert_eq!(result.provenance_status, WearabilityProvenanceStatusV1::Invalid);
        assert!(result.provenance_findings.contains(
            &WearabilityProvenanceFindingV1::ObservationTimestampMissing {
                field: WearabilityEvidenceFieldV1::PressureSensor
            }
        ));
        assert!(result.provenance_findings.contains(
            &WearabilityProvenanceFindingV1::TimebaseMissing {
                field: WearabilityEvidenceFieldV1::PressureSensor
            }
        ));
        assert!(result.provenance_findings.contains(
            &WearabilityProvenanceFindingV1::SensorCalibrationReferenceMissing {
                field: WearabilityEvidenceFieldV1::PressureSensor
            }
        ));
    }

    #[test]
    fn explicit_doff_remains_highest_priority_with_invalid_other_metadata() {
        let input = WearabilityInputV1 {
            schema_version: WEARABILITY_SCHEMA_VERSION + 99,
            user_requested_doff: Some(true),
            ..Default::default()
        };
        let decision = evaluate_wearability_request(
            "profile-a",
            WearabilityStageV1::Perform,
            &input,
        );
        assert_eq!(
            decision.disposition,
            WearabilityDispositionV1::DoffRequestPreemptsNewMotion
        );
        assert!(decision.findings.contains(&WearabilityFindingV1::UserRequestedDoff));
        assert!(decision.findings.contains(&WearabilityFindingV1::UnsupportedSchemaVersion));
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
