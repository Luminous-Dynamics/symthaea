// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint observation-process ledger for semiconductor VM.
//!
//! This is a benchmark/evidence ledger, not another risk controller. It keeps
//! physical metrology acquisition, K1 qualification, controller feedback
//! tracking, and independent audit sampling as four distinct mechanisms.
//! It composes the existing partial-identification estimator but never
//! multiplies their propensities into an unproved universal IPW weight.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::selection_risk_bounds::{
    IndependentAuditLossObservationV1, SelectionAuditDesignV1, SelectionRiskInputV1,
    SelectionRiskReportV1, SELECTION_RISK_INPUT_SCHEMA_V1,
    evaluate_selection_adjusted_population_risk_v1,
};

pub const OBSERVATION_PROCESS_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process.v1";
pub const OBSERVATION_PROCESS_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationRiskControllerV1 {
    None,
    FixedDelayAci,
    CapacityAwareOcrcBounded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalAcquisitionStateV1 {
    NotSelected,
    SelectedPending,
    Measured,
    MeasurementUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalSelectionDesignKindV1 {
    DeterministicOrAdaptive,
    RandomizedKnownInclusionProbability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum K1QualificationStateV1 {
    NotApplicable,
    Pending,
    Eligible,
    Unknown,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackTrackingStateV1 {
    NotApplicable,
    UnlimitedPending,
    UnlimitedReceived,
    UnlimitedForfeited,
    ProxyNotTracked,
    AdmittedPending,
    CapacityDenied,
    Evicted,
    ProxyExpired,
    TrackedReceived,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerAssumptionStateV1 {
    NotApplicable,
    CompleteUnderDeclaredAssumptions,
    DeclaredButNotVerified,
    InvalidatedByMissingOrUnqualifiedOutcome,
    DelayContractExceeded,
    TrackingCapacityExceeded,
    TraceIncomplete,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalAcquisitionRecordV1 {
    pub state: PhysicalAcquisitionStateV1,
    pub selection_design_kind: PhysicalSelectionDesignKindV1,
    pub selection_design_sha256: Option<String>,
    /// Recorded for analysis only. It is not automatically used as an IPW
    /// multiplier by the population-risk estimator.
    pub inclusion_probability: Option<f64>,
    pub acquisition_cost_units: Option<f64>,
    pub selected_at_round: Option<u64>,
    pub measured_at_unix_seconds: Option<i64>,
    pub operational_measurement_method_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct K1QualificationRecordV1 {
    pub state: K1QualificationStateV1,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub label_payload_sha256: Option<String>,
    pub measurement_event_id: Option<String>,
    pub label_available_at_unix_seconds: Option<i64>,
    pub report_sha256: Option<String>,
    pub report_available_at_unix_seconds: Option<i64>,
    /// Loss against the immutable prediction/interval. It is recorded for
    /// offline population evaluation even when the controller did not retain
    /// the feedback slot.
    pub qualified_loss: Option<f64>,
    pub qualified_loss_evidence_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedbackTrackingRecordV1 {
    pub state: FeedbackTrackingStateV1,
    /// CA-OCRC proxy delay. -1 means the feedback was not selected for tracking.
    pub proxy_delay_rounds: Option<i64>,
    /// Actual physical feedback delay, including rounds not retained by the
    /// controller, when the outcome is later known to the offline evaluator.
    pub actual_feedback_delay_rounds: Option<u64>,
    /// When the controller actually received a tracked physical/K1 outcome.
    pub controller_feedback_received_at_unix_seconds: Option<i64>,
    /// CA-OCRC survival probability for the relevant observed delay.
    pub proxy_survival_probability: Option<f64>,
    pub tracking_slot_id: Option<u64>,
    /// Inclusive round at which a tracked slot ceased to consume capacity.
    pub tracking_end_round: Option<u64>,
    pub action_receipt_sha256: Option<String>,
    pub controller_update_applied: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndependentAuditObservationV1 {
    pub round_id: u64,
    pub loss: f64,
    pub inclusion_probability: f64,
    pub audit_measurement_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessRoundV1 {
    pub round_id: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub prediction_receipt_sha256: String,
    pub interval_receipt_sha256: String,
    pub model_artifact_sha256: String,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub acquisition: PhysicalAcquisitionRecordV1,
    pub qualification: K1QualificationRecordV1,
    pub tracking: FeedbackTrackingRecordV1,
    pub independent_audit: Option<IndependentAuditObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessInputV1 {
    pub schema: String,
    pub run_id: String,
    pub population_manifest_sha256: String,
    /// Digest of the canonical, round-sorted observation rows.
    pub source_ledger_sha256: String,
    pub controller: ObservationRiskControllerV1,
    pub controller_policy_sha256: Option<String>,
    pub delay_contract_sha256: Option<String>,
    pub delay_contract_declared_and_enforced: bool,
    pub fixed_delay_rounds: Option<u64>,
    pub maximum_feedback_delay_rounds: Option<u64>,
    pub tracking_capacity: Option<u64>,
    /// The final round represented by the sealed trace.
    pub closeout_round: u64,
    pub independent_audit_design: Option<SelectionAuditDesignV1>,
    pub rounds: Vec<ObservationProcessRoundV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessFunnelV1 {
    pub population_rounds: usize,
    pub not_selected: usize,
    pub selected_pending: usize,
    pub measurement_unavailable: usize,
    pub measured: usize,
    pub qualification_pending: usize,
    pub k1_eligible: usize,
    pub k1_unknown: usize,
    pub k1_blocked: usize,
    pub feedback_not_applicable: usize,
    pub feedback_pending: usize,
    pub feedback_received: usize,
    pub feedback_forfeited_or_lost: usize,
    pub capacity_denied: usize,
    pub proxy_not_tracked: usize,
    pub controller_updates_applied: usize,
    pub controller_updates_withheld: usize,
    pub independent_audit_observations: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessReportV1 {
    pub schema: String,
    pub run_id: String,
    pub controller: ObservationRiskControllerV1,
    pub controller_assumption_state: ControllerAssumptionStateV1,
    pub controller_assumption_reasons: Vec<String>,
    pub maximum_observed_tracking_occupancy: u64,
    pub funnel: ObservationProcessFunnelV1,
    pub population_risk: SelectionRiskReportV1,
    pub input_sha256: String,
    pub report_sha256: String,
    pub equipment_control_authority: bool,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ObservationProcessErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ObservationProcessErrorV1> {
    if value.trim().is_empty() {
        return Err(ObservationProcessErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, ObservationProcessErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ObservationProcessErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn canonical_rounds(
    rounds: &[ObservationProcessRoundV1],
) -> Vec<ObservationProcessRoundV1> {
    let mut canonical = rounds.to_vec();
    canonical.sort_by_key(|row| row.round_id);
    canonical
}

fn report_digest(
    report: &ObservationProcessReportV1,
) -> Result<String, ObservationProcessErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_optional_digest(
    value: Option<&str>,
    field: &str,
) -> Result<(), ObservationProcessErrorV1> {
    if value.is_some_and(|digest| !valid_sha256(digest)) {
        return Err(ObservationProcessErrorV1(format!(
            "{field}: malformed lowercase SHA-256"
        )));
    }
    Ok(())
}

fn validate_acquisition(
    row: &ObservationProcessRoundV1,
) -> Result<(), ObservationProcessErrorV1> {
    let acquisition = &row.acquisition;
    validate_optional_digest(
        acquisition.selection_design_sha256.as_deref(),
        "acquisition.selection_design_sha256",
    )?;
    match acquisition.selection_design_kind {
        PhysicalSelectionDesignKindV1::RandomizedKnownInclusionProbability => {
            if acquisition.selection_design_sha256.is_none()
                || !acquisition
                    .inclusion_probability
                    .is_some_and(|p| p.is_finite() && p > 0.0 && p <= 1.0)
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} randomized acquisition requires a design digest and positive known inclusion probability",
                    row.round_id
                )));
            }
        }
        PhysicalSelectionDesignKindV1::DeterministicOrAdaptive => {
            if acquisition.inclusion_probability.is_some() {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} must not attach a statistical inclusion probability to deterministic/adaptive selection",
                    row.round_id
                )));
            }
        }
    }
    if acquisition
        .acquisition_cost_units
        .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} has invalid physical acquisition cost",
            row.round_id
        )));
    }
    if acquisition.selected_at_round.is_some_and(|selected| selected > row.round_id) {
        return Err(ObservationProcessErrorV1(format!(
            "round {} selection timestamp is from a future round",
            row.round_id
        )));
    }
    if acquisition
        .measured_at_unix_seconds
        .is_some_and(|measured| measured < row.event_time_unix_seconds)
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} measurement time precedes the physical event",
            row.round_id
        )));
    }

    match acquisition.state {
        PhysicalAcquisitionStateV1::NotSelected => {
            if acquisition.operational_measurement_method_sha256.is_some()
                || acquisition.selected_at_round.is_some()
                || acquisition.measured_at_unix_seconds.is_some()
                || acquisition.acquisition_cost_units.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} has acquisition metadata despite NOT_SELECTED",
                    row.round_id
                )));
            }
        }
        PhysicalAcquisitionStateV1::SelectedPending => {
            if acquisition.operational_measurement_method_sha256.is_some()
                || acquisition.selected_at_round.is_none()
                || acquisition.measured_at_unix_seconds.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} SELECTED_PENDING requires selection time and no measurement time",
                    row.round_id
                )));
            }
        }
        PhysicalAcquisitionStateV1::Measured => {
            if acquisition.operational_measurement_method_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || acquisition.selected_at_round.is_none()
                || acquisition.measured_at_unix_seconds.is_none()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} MEASURED requires selection and measurement times",
                    row.round_id
                )));
            }
        }
        PhysicalAcquisitionStateV1::MeasurementUnavailable => {
            if acquisition.operational_measurement_method_sha256.is_some()
                || acquisition.selected_at_round.is_none()
                || acquisition.measured_at_unix_seconds.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} MEASUREMENT_UNAVAILABLE requires selection time and no measurement time",
                    row.round_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_qualification(
    row: &ObservationProcessRoundV1,
) -> Result<(), ObservationProcessErrorV1> {
    let qualification = &row.qualification;
    validate_optional_digest(qualification.label_payload_sha256.as_deref(), "qualification.label_payload_sha256")?;
    validate_optional_digest(qualification.report_sha256.as_deref(), "qualification.report_sha256")?;
    validate_optional_digest(
        qualification.qualified_loss_evidence_sha256.as_deref(),
        "qualification.qualified_loss_evidence_sha256",
    )?;
    if qualification.label_revision.is_some_and(|revision| revision == 0) {
        return Err(ObservationProcessErrorV1(format!(
            "round {} has a zero label revision",
            row.round_id
        )));
    }

    match row.acquisition.state {
        PhysicalAcquisitionStateV1::Measured => {
            if qualification.state == K1QualificationStateV1::NotApplicable {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} is measured but has no qualification state",
                    row.round_id
                )));
            }
        }
        _ if qualification.state != K1QualificationStateV1::NotApplicable => {
            return Err(ObservationProcessErrorV1(format!(
                "round {} has a qualification state without a completed measurement",
                row.round_id
            )));
        }
        _ => {}
    }

    match qualification.state {
        K1QualificationStateV1::NotApplicable => {
            if qualification.label_id.is_some()
                || qualification.label_revision.is_some()
                || qualification.label_payload_sha256.is_some()
                || qualification.measurement_event_id.is_some()
                || qualification.label_available_at_unix_seconds.is_some()
                || qualification.report_sha256.is_some()
                || qualification.report_available_at_unix_seconds.is_some()
                || qualification.qualified_loss.is_some()
                || qualification.qualified_loss_evidence_sha256.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} NOT_APPLICABLE qualification contains label evidence",
                    row.round_id
                )));
            }
        }
        K1QualificationStateV1::Pending => {
            require_text(
                qualification.label_id.as_deref().unwrap_or_default(),
                "qualification.label_id",
            )?;
            if qualification.label_revision.is_none()
                || qualification.label_payload_sha256.is_none()
                || qualification.label_available_at_unix_seconds.is_none()
                || qualification.report_sha256.is_some()
                || qualification.report_available_at_unix_seconds.is_some()
                || qualification.qualified_loss.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} pending qualification has incomplete or premature report/loss fields",
                    row.round_id
                )));
            }
        }
        K1QualificationStateV1::Eligible => {
            require_text(
                qualification.label_id.as_deref().unwrap_or_default(),
                "qualification.label_id",
            )?;
            if qualification.label_revision.is_none()
                || qualification.label_payload_sha256.is_none()
                || qualification.measurement_event_id.as_deref()
                    != Some(row.measurement_event_id.as_str())
                || qualification.label_available_at_unix_seconds.is_none()
                || qualification.report_sha256.is_none()
                || qualification.report_available_at_unix_seconds.is_none()
                || !qualification
                    .qualified_loss
                    .is_some_and(|loss| loss.is_finite() && (0.0..=1.0).contains(&loss))
                || qualification.qualified_loss_evidence_sha256.is_none()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} eligible K1 qualification requires matching event, revision, payload/report receipts and a bounded loss",
                    row.round_id
                )));
            }
        }
        K1QualificationStateV1::Unknown | K1QualificationStateV1::Blocked => {
            require_text(
                qualification.label_id.as_deref().unwrap_or_default(),
                "qualification.label_id",
            )?;
            if qualification.label_revision.is_none()
                || qualification.label_payload_sha256.is_none()
                || qualification.measurement_event_id.as_deref()
                    != Some(row.measurement_event_id.as_str())
                || qualification.label_available_at_unix_seconds.is_none()
                || qualification.report_sha256.is_none()
                || qualification.report_available_at_unix_seconds.is_none()
                || qualification.qualified_loss.is_some()
                || qualification.qualified_loss_evidence_sha256.is_some()
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} UNKNOWN/BLOCKED qualification must carry its K1 report but no qualified loss",
                    row.round_id
                )));
            }
        }
    }

    if let Some(label_available) = qualification.label_available_at_unix_seconds {
        if row
            .acquisition
            .measured_at_unix_seconds
            .is_some_and(|measured| label_available < measured)
            || label_available < row.event_time_unix_seconds
        {
            return Err(ObservationProcessErrorV1(format!(
                "round {} label availability predates the measurement or physical event",
                row.round_id
            )));
        }
    }
    if qualification.report_available_at_unix_seconds.is_some_and(|report_available| {
        qualification
            .label_available_at_unix_seconds
            .is_some_and(|label_available| report_available < label_available)
    }) {
        return Err(ObservationProcessErrorV1(format!(
            "round {} K1 report predates physical label availability",
            row.round_id
        )));
    }
    Ok(())
}

fn validate_tracking(
    input: &ObservationProcessInputV1,
    row: &ObservationProcessRoundV1,
) -> Result<(), ObservationProcessErrorV1> {
    let tracking = &row.tracking;
    validate_optional_digest(tracking.action_receipt_sha256.as_deref(), "tracking.action_receipt_sha256")?;
    if tracking
        .proxy_survival_probability
        .is_some_and(|probability| {
            !probability.is_finite() || probability <= 0.0 || probability > 1.0
        })
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} has invalid proxy survival probability",
            row.round_id
        )));
    }

    let requires_controller_receipt = matches!(
        tracking.state,
        FeedbackTrackingStateV1::UnlimitedReceived | FeedbackTrackingStateV1::TrackedReceived
    );
    if requires_controller_receipt != tracking.controller_feedback_received_at_unix_seconds.is_some()
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} controller receipt timestamp does not match its feedback tracking state",
            row.round_id
        )));
    }
    if let Some(received_at) = tracking.controller_feedback_received_at_unix_seconds {
        if received_at < row.prediction_time_unix_seconds
            || row
                .qualification
                .label_available_at_unix_seconds
                .is_some_and(|available| received_at < available)
            || row
                .qualification
                .report_available_at_unix_seconds
                .is_some_and(|available| received_at < available)
        {
            return Err(ObservationProcessErrorV1(format!(
                "round {} controller received feedback before the prediction, physical label or K1 report was available",
                row.round_id
            )));
        }
    }

    if tracking.controller_update_applied
        && (row.qualification.state != K1QualificationStateV1::Eligible
            || tracking.action_receipt_sha256.is_none()
            || !matches!(
                tracking.state,
                FeedbackTrackingStateV1::UnlimitedReceived
                    | FeedbackTrackingStateV1::TrackedReceived
            ))
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} controller update requires K1-eligible evidence and an action receipt",
            row.round_id
        )));
    }

    if input.controller == ObservationRiskControllerV1::CapacityAwareOcrcBounded {
        let physically_measured = row.acquisition.state == PhysicalAcquisitionStateV1::Measured;
        if physically_measured
            == (tracking.state == FeedbackTrackingStateV1::NotApplicable)
        {
            return Err(ObservationProcessErrorV1(format!(
                "round {} CA-OCRC tracking is NOT_APPLICABLE exactly when no physical measurement exists",
                row.round_id
            )));
        }
    }

    match input.controller {
        ObservationRiskControllerV1::None => {
            if tracking.state != FeedbackTrackingStateV1::NotApplicable
                || tracking.proxy_delay_rounds.is_some()
                || tracking.actual_feedback_delay_rounds.is_some()
                || tracking.proxy_survival_probability.is_some()
                || tracking.tracking_slot_id.is_some()
                || tracking.tracking_end_round.is_some()
                || tracking.action_receipt_sha256.is_some()
                || tracking.controller_update_applied
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} has feedback tracking metadata in a no-controller run",
                    row.round_id
                )));
            }
        }
        ObservationRiskControllerV1::FixedDelayAci => {
            let delay = input.fixed_delay_rounds.expect("policy validated");
            match tracking.state {
                FeedbackTrackingStateV1::UnlimitedPending => {
                    if tracking.proxy_delay_rounds.is_some()
                        || tracking.tracking_slot_id.is_some()
                        || tracking.tracking_end_round.is_some()
                        || tracking.actual_feedback_delay_rounds.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} unlimited pending feedback has inconsistent tracking fields",
                            row.round_id
                        )));
                    }
                    if row.round.saturating_add(delay) <= input.closeout_round {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} feedback should be due by the declared closeout",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::UnlimitedReceived => {
                    if tracking.actual_feedback_delay_rounds.is_none()
                        || tracking.tracking_slot_id.is_some()
                        || tracking.tracking_end_round.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} ACI feedback must arrive after exactly the fixed delay",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::UnlimitedForfeited => {
                    if tracking.action_receipt_sha256.is_none()
                        || tracking.tracking_slot_id.is_some()
                        || tracking.tracking_end_round.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} ACI forfeiture cannot invent an actual outcome delay",
                            row.round_id
                        )));
                    }
                }
                _ => {
                    return Err(ObservationProcessErrorV1(format!(
                        "round {} uses a CA-OCRC tracking state in a fixed-delay ACI run",
                        row.round_id
                    )));
                }
            }
        }
        ObservationRiskControllerV1::CapacityAwareOcrcBounded => {
            let maximum_delay = input.maximum_feedback_delay_rounds.expect("policy validated");
            match tracking.state {
                FeedbackTrackingStateV1::NotApplicable => {
                    // No physical measurement exists for this population round;
                    // there is no feedback outcome to place in the controller tracker.
                }
                FeedbackTrackingStateV1::ProxyNotTracked => {
                    if tracking.proxy_delay_rounds != Some(-1)
                        || tracking.action_receipt_sha256.is_none()
                        || tracking.tracking_slot_id.is_some()
                        || tracking.tracking_end_round.is_some()
                        || tracking.proxy_survival_probability.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} proxy-not-tracked state must use proxy delay -1 and no slot",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::CapacityDenied => {
                    if !tracking.proxy_delay_rounds.is_some_and(|delay| delay >= 0 && delay as u64 <= maximum_delay)
                        || tracking.action_receipt_sha256.is_none()
                        || tracking.tracking_slot_id.is_some()
                        || tracking.tracking_end_round.is_some()
                        || tracking.proxy_survival_probability.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} capacity-denied feedback requires a nonnegative proxy delay and no slot",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::AdmittedPending => {
                    let proxy = tracking.proxy_delay_rounds;
                    if !proxy.is_some_and(|delay| delay >= 0 && delay as u64 <= maximum_delay)
                        || tracking.tracking_slot_id.is_none()
                        || tracking.tracking_end_round.is_some()
                        || tracking.proxy_survival_probability.is_some()
                        || tracking.actual_feedback_delay_rounds.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} admitted-pending tracking fields are inconsistent",
                            row.round_id
                        )));
                    }
                    let deadline = row.round.saturating_add(proxy.unwrap_or(0) as u64);
                    if deadline <= input.closeout_round {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} proxy deadline elapsed before closeout but feedback remains pending",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::Evicted => {
                    let proxy = tracking.proxy_delay_rounds;
                    let ended = tracking.tracking_end_round;
                    if tracking.action_receipt_sha256.is_none()
                        || !proxy.is_some_and(|delay| delay >= 0 && delay as u64 <= maximum_delay)
                        || tracking.tracking_slot_id.is_none()
                        || !ended.is_some_and(|end| end >= row.round && end < row.round + proxy.unwrap_or(0) as u64)
                        || tracking.actual_feedback_delay_rounds.is_some_and(|actual| {
                            ended.is_some_and(|end| actual <= end.saturating_sub(row.round))
                        })
                        || tracking.proxy_survival_probability.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} eviction must end tracking strictly before proxy expiry",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::ProxyExpired => {
                    let proxy = tracking.proxy_delay_rounds;
                    if tracking.action_receipt_sha256.is_none()
                        || !proxy.is_some_and(|delay| delay >= 0 && delay as u64 <= maximum_delay)
                        || tracking.tracking_slot_id.is_none()
                        || tracking.tracking_end_round
                            != Some(row.round.saturating_add(proxy.unwrap_or(0) as u64))
                        || tracking.actual_feedback_delay_rounds.is_some_and(|actual| {
                            actual <= proxy.unwrap_or(-1) as u64
                        })
                        || tracking.proxy_survival_probability.is_some()
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} proxy expiry must end tracking at its exact deadline",
                            row.round_id
                        )));
                    }
                }
                FeedbackTrackingStateV1::TrackedReceived => {
                    let proxy = tracking.proxy_delay_rounds;
                    let actual = tracking.actual_feedback_delay_rounds;
                    if !proxy.is_some_and(|delay| delay >= 0 && delay as u64 <= maximum_delay)
                        || !actual.is_some_and(|delay| delay <= maximum_delay)
                        || actual.is_some_and(|delay| delay > proxy.unwrap_or(-1) as u64)
                        || tracking.tracking_slot_id.is_none()
                        || tracking.tracking_end_round
                            != actual.map(|delay| row.round.saturating_add(delay))
                    {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} tracked feedback must arrive by the proxy horizon and release its slot at actual arrival",
                            row.round_id
                        )));
                    }
                    let actual = actual.expect("validated actual delay");
                    if !tracking.proxy_survival_probability.is_some_and(|probability| {
                        probability.is_finite() && probability > 0.0 && probability <= 1.0
                    }) {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} tracked feedback requires a positive inverse-survival denominator",
                            row.round_id
                        )));
                    }
                    if row.round.saturating_add(actual) > input.closeout_round {
                        return Err(ObservationProcessErrorV1(format!(
                            "round {} reports feedback after trace closeout",
                            row.round_id
                        )));
                    }
                }
                _ => {
                    return Err(ObservationProcessErrorV1(format!(
                        "round {} uses a fixed-delay ACI tracking state in a CA-OCRC run",
                        row.round_id
                    )));
                }
            }
        }
    }

    if matches!(
        tracking.state,
        FeedbackTrackingStateV1::UnlimitedReceived | FeedbackTrackingStateV1::TrackedReceived
    ) && tracking.action_receipt_sha256.is_none()
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} received feedback requires an accepted-or-withheld action receipt",
            row.round_id
        )));
    }
    if tracking.controller_update_applied
        != (matches!(
            tracking.state,
            FeedbackTrackingStateV1::UnlimitedReceived | FeedbackTrackingStateV1::TrackedReceived
        ) && row.qualification.state == K1QualificationStateV1::Eligible)
    {
        return Err(ObservationProcessErrorV1(format!(
            "round {} K1 eligibility and controller feedback disposition disagree about whether an update was applied",
            row.round_id
        )));
    }

    if !tracking.controller_update_applied
        && row.qualification.state != K1QualificationStateV1::Eligible
    {
        // This is expected for UNKNOWN/BLOCKED/PENDING observations, but
        // explicitly forbids turning an invalid label into a numerical update.
        if tracking.state == FeedbackTrackingStateV1::TrackedReceived
            && tracking.action_receipt_sha256.is_none()
        {
            return Err(ObservationProcessErrorV1(format!(
                "round {} received feedback must have an action/withheld receipt",
                row.round_id
            )));
        }
    }
    Ok(())
}

fn maximum_tracking_occupancy(
    rows: &[ObservationProcessRoundV1],
) -> u64 {
    let mut occupancy_by_round: BTreeMap<u64, i64> = BTreeMap::new();
    for row in rows {
        let tracking = &row.tracking;
        let tracked = matches!(
            tracking.state,
            FeedbackTrackingStateV1::AdmittedPending
                | FeedbackTrackingStateV1::Evicted
                | FeedbackTrackingStateV1::ProxyExpired
                | FeedbackTrackingStateV1::TrackedReceived
        );
        if !tracked {
            continue;
        }
        let start = row.round;
        let end = tracking.tracking_end_round.unwrap_or(u64::MAX);
        *occupancy_by_round.entry(start).or_default() += 1;
        if end < u64::MAX {
            *occupancy_by_round.entry(end.saturating_add(1)).or_default() -= 1;
        }
    }
    let mut current = 0_i64;
    let mut maximum = 0_i64;
    for delta in occupancy_by_round.values() {
        current += *delta;
        maximum = maximum.max(current);
    }
    maximum.max(0) as u64
}

fn assumption_state(
    input: &ObservationProcessInputV1,
    rows: &[ObservationProcessRoundV1],
    maximum_occupancy: u64,
) -> (ControllerAssumptionStateV1, Vec<String>) {
    if input.controller == ObservationRiskControllerV1::None {
        return (
            ControllerAssumptionStateV1::NotApplicable,
            vec!["No online conformal controller was evaluated in this trace.".into()],
        );
    }
    let mut reasons = Vec::new();
    if !input.delay_contract_declared_and_enforced || input.delay_contract_sha256.is_none() {
        reasons.push("The physical feedback-delay contract is not independently declared and enforced.".into());
    }

    match input.controller {
        ObservationRiskControllerV1::None => {}
        ObservationRiskControllerV1::FixedDelayAci => {
            let fixed_delay = input.fixed_delay_rounds.unwrap_or(0);
            if rows.iter().any(|row| {
                row.tracking.state == FeedbackTrackingStateV1::UnlimitedReceived
                    && row.tracking.actual_feedback_delay_rounds != Some(fixed_delay)
            }) {
                reasons.push("At least one ACI outcome arrived outside the declared fixed-delay schedule.".into());
                return (ControllerAssumptionStateV1::DelayContractExceeded, reasons);
            }
            if rows.iter().any(|row| {
                row.tracking.state == FeedbackTrackingStateV1::UnlimitedPending
                    || row.acquisition.state == PhysicalAcquisitionStateV1::SelectedPending
                    || row.qualification.state == K1QualificationStateV1::Pending
            }) {
                reasons.push("The closeout contains unresolved physical acquisition, K1 qualification or fixed-delay feedback.".into());
                return (ControllerAssumptionStateV1::TraceIncomplete, reasons);
            }
            for row in rows {
                if row.acquisition.state != PhysicalAcquisitionStateV1::Measured {
                    reasons.push(format!("round {} has no completed physical measurement", row.round_id));
                }
                if row.qualification.state != K1QualificationStateV1::Eligible {
                    reasons.push(format!("round {} has no K1-qualified outcome", row.round_id));
                }
                if row.tracking.state != FeedbackTrackingStateV1::UnlimitedReceived
                    || row.tracking.actual_feedback_delay_rounds != Some(fixed_delay)
                {
                    reasons.push(format!("round {} does not have complete fixed-delay feedback", row.round_id));
                }
            }
        }
        ObservationRiskControllerV1::CapacityAwareOcrcBounded => {
            let maximum_delay = input.maximum_feedback_delay_rounds.unwrap_or(0);
            if rows.iter().any(|row| {
                row.tracking.actual_feedback_delay_rounds
                    .is_some_and(|delay| delay > maximum_delay)
            }) {
                reasons.push("At least one physical feedback result exceeded the declared maximum delay.".into());
                return (ControllerAssumptionStateV1::DelayContractExceeded, reasons);
            }
            if maximum_occupancy > input.tracking_capacity.unwrap_or(0) {
                reasons.push(format!(
                    "Observed peak tracking occupancy {maximum_occupancy} exceeded declared capacity {}.",
                    input.tracking_capacity.unwrap_or(0)
                ));
                return (ControllerAssumptionStateV1::TrackingCapacityExceeded, reasons);
            }
            if rows.iter().any(|row| {
                row.acquisition.state == PhysicalAcquisitionStateV1::SelectedPending
                    || row.qualification.state == K1QualificationStateV1::Pending
                    || row.tracking.state == FeedbackTrackingStateV1::AdmittedPending
            }) {
                reasons.push("The closeout contains unresolved physical acquisition, K1 qualification or proxy-tracked feedback.".into());
                return (ControllerAssumptionStateV1::TraceIncomplete, reasons);
            }
            for row in rows {
                if row.acquisition.state != PhysicalAcquisitionStateV1::Measured {
                    reasons.push(format!("round {} has no completed physical measurement", row.round_id));
                }
                if row.qualification.state != K1QualificationStateV1::Eligible {
                    reasons.push(format!("round {} has no K1-qualified physical outcome", row.round_id));
                }
                if !row
                    .tracking
                    .actual_feedback_delay_rounds
                    .is_some_and(|delay| delay <= maximum_delay)
                {
                    reasons.push(format!("round {} has no observed delay within the bounded-delay contract", row.round_id));
                }
            }
            if maximum_occupancy > input.tracking_capacity.unwrap_or(0) {
                reasons.push("Observed concurrent tracking occupancy exceeded declared capacity.".into());
                return (
                    ControllerAssumptionStateV1::TrackingCapacityExceeded,
                    reasons,
                );
            }
        }
    }

    if !input.delay_contract_declared_and_enforced || input.delay_contract_sha256.is_none() {
        return (
            ControllerAssumptionStateV1::DeclaredButNotVerified,
            reasons,
        );
    }
    if !reasons.is_empty() {
        return (
            ControllerAssumptionStateV1::InvalidatedByMissingOrUnqualifiedOutcome,
            reasons,
        );
    }
    (
        ControllerAssumptionStateV1::CompleteUnderDeclaredAssumptions,
        vec!["All row-level observations are complete and match the declared controller/delay contract; the declaration itself is not independent physical proof.".into()],
    )
}

fn validate_policy(input: &ObservationProcessInputV1) -> Result<(), ObservationProcessErrorV1> {
    if input.schema != OBSERVATION_PROCESS_SCHEMA_V1 {
        return Err(ObservationProcessErrorV1(
            "observation-process schema is unsupported".into(),
        ));
    }
    require_text(&input.run_id, "input.run_id")?;
    if !valid_sha256(&input.population_manifest_sha256)
        || !valid_sha256(&input.source_ledger_sha256)
    {
        return Err(ObservationProcessErrorV1(
            "population or source ledger digest is malformed".into(),
        ));
    }
    validate_optional_digest(
        input.controller_policy_sha256.as_deref(),
        "controller_policy_sha256",
    )?;
    validate_optional_digest(input.delay_contract_sha256.as_deref(), "delay_contract_sha256")?;

    match input.controller {
        ObservationRiskControllerV1::None => {
            if input.controller_policy_sha256.is_some()
                || input.fixed_delay_rounds.is_some()
                || input.maximum_feedback_delay_rounds.is_some()
                || input.tracking_capacity.is_some()
                || input.delay_contract_sha256.is_some()
                || input.delay_contract_declared_and_enforced
            {
                return Err(ObservationProcessErrorV1(
                    "no-controller policy must not carry controller/delay contract fields".into(),
                ));
            }
        }
        ObservationRiskControllerV1::FixedDelayAci => {
            if input.controller_policy_sha256.is_none()
                || input.fixed_delay_rounds.is_none_or(|delay| delay == 0)
                || input.maximum_feedback_delay_rounds.is_some()
                || input.tracking_capacity.is_some()
            {
                return Err(ObservationProcessErrorV1(
                    "fixed-delay ACI requires a policy digest and positive fixed delay only".into(),
                ));
            }
        }
        ObservationRiskControllerV1::CapacityAwareOcrcBounded => {
            if input.controller_policy_sha256.is_none()
                || input.fixed_delay_rounds.is_some()
                || input.maximum_feedback_delay_rounds.is_none()
                || input.tracking_capacity.is_none()
                || input.maximum_feedback_delay_rounds == Some(0)
                || input.tracking_capacity == Some(0)
            {
                return Err(ObservationProcessErrorV1(
                    "bounded CA-OCRC requires a policy digest, positive maximum delay and positive tracking capacity".into(),
                ));
            }
        }
    }

    if let Some(design) = &input.independent_audit_design {
        if design.schema != super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1 {
            return Err(ObservationProcessErrorV1(
                "independent audit design schema is unsupported".into(),
            ));
        }
        validate_optional_digest(Some(&design.design_sha256), "audit.design_sha256")?;
        validate_optional_digest(
            Some(&design.randomization_provenance_sha256),
            "audit.randomization_provenance_sha256",
        )?;
        validate_optional_digest(
            Some(&design.reference_method_sha256),
            "audit.reference_method_sha256",
        )?;
        if !design.independent_of_operational_qualification {
            return Err(ObservationProcessErrorV1(
                "independent audit design must be declared independent of operational qualification".into(),
            ));
        }
    }
    Ok(())
}

/// Validates and seals the four-stage observation process, then computes
/// selection-adjusted population-risk bounds from the qualified and
/// independent-audit observations. Acquisition propensity and CA-OCRC proxy
/// survival are deliberately retained as separate fields; they are never
/// multiplied into an unproved joint weight here.
pub fn evaluate_observation_process_v1(
    input: &ObservationProcessInputV1,
) -> Result<ObservationProcessReportV1, ObservationProcessErrorV1> {
    validate_policy(input)?;
    if input.rounds.is_empty() {
        return Err(ObservationProcessErrorV1(
            "observation process must contain at least one population round".into(),
        ));
    }
    let rows = canonical_rounds(&input.rounds);
    if rows.iter().any(|row| row.round_id > input.closeout_round) {
        return Err(ObservationProcessErrorV1(
            "closeout round precedes one or more population prediction rounds".into(),
        ));
    }
    let mut round_ids = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();
    let mut measurement_event_ids = BTreeSet::new();
    let mut qualified_round_ids = Vec::new();
    let mut unknown_round_ids = Vec::new();
    let mut blocked_round_ids = Vec::new();
    let mut unmeasured_round_ids = Vec::new();
    let mut qualified_losses = Vec::new();
    let mut audit_losses = Vec::new();

    let mut funnel = ObservationProcessFunnelV1 {
        population_rounds: rows.len(),
        not_selected: 0,
        selected_pending: 0,
        measurement_unavailable: 0,
        measured: 0,
        qualification_pending: 0,
        k1_eligible: 0,
        k1_unknown: 0,
        k1_blocked: 0,
        feedback_not_applicable: 0,
        feedback_pending: 0,
        feedback_received: 0,
        feedback_forfeited_or_lost: 0,
        capacity_denied: 0,
        proxy_not_tracked: 0,
        controller_updates_applied: 0,
        controller_updates_withheld: 0,
        independent_audit_observations: 0,
    };

    for row in &rows {
        require_text(&row.sample_id, "round.sample_id")?;
        require_text(&row.measurement_event_id, "round.measurement_event_id")?;
        require_text(&row.clock_domain_id, "round.clock_domain_id")?;
        for (digest, field) in [
            (&row.prediction_receipt_sha256, "prediction receipt"),
            (&row.interval_receipt_sha256, "interval receipt"),
            (&row.model_artifact_sha256, "model artifact"),
        ] {
            if !valid_sha256(digest) {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} has malformed {field} digest",
                    row.round_id
                )));
            }
        }
        if row.event_time_unix_seconds > row.prediction_time_unix_seconds {
            return Err(ObservationProcessErrorV1(format!(
                "round {} prediction precedes its event time",
                row.round_id
            )));
        }
        if !round_ids.insert(row.round_id)
            || !sample_ids.insert(row.sample_id.as_str())
            || !measurement_event_ids.insert(row.measurement_event_id.as_str())
        {
            return Err(ObservationProcessErrorV1(
                "duplicate round, sample or measurement-event identity".into(),
            ));
        }

        validate_acquisition(row)?;
        validate_qualification(row)?;
        validate_tracking(input, row)?;

        match row.acquisition.state {
            PhysicalAcquisitionStateV1::NotSelected => {
                funnel.not_selected += 1;
                unmeasured_round_ids.push(row.round_id);
            }
            PhysicalAcquisitionStateV1::SelectedPending => {
                funnel.selected_pending += 1;
                unmeasured_round_ids.push(row.round_id);
            }
            PhysicalAcquisitionStateV1::MeasurementUnavailable => {
                funnel.measurement_unavailable += 1;
                unmeasured_round_ids.push(row.round_id);
            }
            PhysicalAcquisitionStateV1::Measured => funnel.measured += 1,
        }
        match row.qualification.state {
            K1QualificationStateV1::NotApplicable => {
                if row.acquisition.state == PhysicalAcquisitionStateV1::Measured {
                    return Err(ObservationProcessErrorV1(format!(
                        "round {} measured row was not classified for K1 qualification",
                        row.round_id
                    )));
                }
            }
            K1QualificationStateV1::Pending => {
                funnel.qualification_pending += 1;
                unknown_round_ids.push(row.round_id);
            }
            K1QualificationStateV1::Eligible => {
                funnel.k1_eligible += 1;
                qualified_round_ids.push(row.round_id);
                qualified_losses.push(super::selection_risk_bounds::QualifiedLossObservationV1 {
                    round_id: row.round_id,
                    loss: row.qualification.qualified_loss.expect("validated eligible loss"),
                    evidence_receipt_sha256: row
                        .qualification
                        .qualified_loss_evidence_sha256
                        .clone()
                        .expect("validated eligible loss receipt"),
                });
            }
            K1QualificationStateV1::Unknown => {
                funnel.k1_unknown += 1;
                unknown_round_ids.push(row.round_id);
            }
            K1QualificationStateV1::Blocked => {
                funnel.k1_blocked += 1;
                blocked_round_ids.push(row.round_id);
            }
        }
        match row.tracking.state {
            FeedbackTrackingStateV1::NotApplicable => funnel.feedback_not_applicable += 1,
            FeedbackTrackingStateV1::UnlimitedPending | FeedbackTrackingStateV1::AdmittedPending => {
                funnel.feedback_pending += 1;
            }
            FeedbackTrackingStateV1::UnlimitedReceived | FeedbackTrackingStateV1::TrackedReceived => {
                funnel.feedback_received += 1;
            }
            FeedbackTrackingStateV1::UnlimitedForfeited
            | FeedbackTrackingStateV1::Evicted
            | FeedbackTrackingStateV1::ProxyExpired => {
                funnel.feedback_forfeited_or_lost += 1;
            }
            FeedbackTrackingStateV1::CapacityDenied => funnel.capacity_denied += 1,
            FeedbackTrackingStateV1::ProxyNotTracked => funnel.proxy_not_tracked += 1,
        }
        if row.tracking.controller_update_applied {
            funnel.controller_updates_applied += 1;
        } else if !matches!(
            row.tracking.state,
            FeedbackTrackingStateV1::NotApplicable
                | FeedbackTrackingStateV1::UnlimitedPending
                | FeedbackTrackingStateV1::AdmittedPending
        ) {
            funnel.controller_updates_withheld += 1;
        }

        if let Some(audit) = &row.independent_audit {
            let design = input.independent_audit_design.as_ref().ok_or_else(|| {
                ObservationProcessErrorV1(
                    "audit observation exists without a declared independent audit design".into(),
                )
            })?;
            let operational_method = row
                .acquisition
                .operational_measurement_method_sha256
                .as_deref();
            if audit.round_id != row.round_id
                || row.qualification.state == K1QualificationStateV1::Eligible
                || !valid_sha256(&audit.audit_measurement_receipt_sha256)
                || operational_method == Some(design.reference_method_sha256.as_str())
            {
                return Err(ObservationProcessErrorV1(format!(
                    "round {} audit record is misidentified, overlaps a qualified round, or has a malformed receipt",
                    row.round_id
                )));
            }
            audit_losses.push(IndependentAuditLossObservationV1 {
                round_id: audit.round_id,
                loss: audit.loss,
                inclusion_probability: audit.inclusion_probability,
                audit_measurement_receipt_sha256: audit.audit_measurement_receipt_sha256.clone(),
            });
            if design.design_sha256.is_empty() {
                return Err(ObservationProcessErrorV1(
                    "independent audit design digest is empty".into(),
                ));
            }
            funnel.independent_audit_observations += 1;
        }
    }

    let maximum_occupancy = maximum_tracking_occupancy(&rows);
    let (controller_assumption_state, controller_assumption_reasons) =
        assumption_state(input, &rows, maximum_occupancy);

    let risk_input = SelectionRiskInputV1 {
        schema: SELECTION_RISK_INPUT_SCHEMA_V1.into(),
        input_id: input.run_id.clone(),
        population_manifest_sha256: input.population_manifest_sha256.clone(),
        source_ledger_sha256: input.source_ledger_sha256.clone(),
        population_round_ids: rows.iter().map(|row| row.round_id).collect(),
        qualified_round_ids,
        unknown_round_ids,
        blocked_round_ids,
        unmeasured_round_ids,
        qualified_losses,
        audit_design: input.independent_audit_design.clone(),
        independent_audit_losses: audit_losses,
    };
    let expected_source_digest = json_sha256(&rows)?;
    if expected_source_digest != input.source_ledger_sha256 {
        return Err(ObservationProcessErrorV1(
            "source ledger digest does not match canonical observation rows".into(),
        ));
    }
    let population_risk = evaluate_selection_adjusted_population_risk_v1(&risk_input)
        .map_err(|error| ObservationProcessErrorV1(format!("population-risk evaluation failed: {error}")))?;
    let mut canonical_input = input.clone();
    canonical_input.rounds = rows.clone();
    let mut report = ObservationProcessReportV1 {
        schema: OBSERVATION_PROCESS_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        controller: input.controller,
        controller_assumption_state,
        controller_assumption_reasons,
        maximum_observed_tracking_occupancy: maximum_occupancy,
        funnel,
        population_risk,
        input_sha256: json_sha256(&canonical_input)?,
        report_sha256: String::new(),
        equipment_control_authority: false,
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn row(
        round_id: u64,
        acquisition: PhysicalAcquisitionStateV1,
        qualification: K1QualificationStateV1,
        tracking: FeedbackTrackingStateV1,
    ) -> ObservationProcessRoundV1 {
        let measured = acquisition == PhysicalAcquisitionStateV1::Measured;
        let eligible = qualification == K1QualificationStateV1::Eligible;
        let has_label = measured && qualification != K1QualificationStateV1::NotApplicable;
        ObservationProcessRoundV1 {
            round_id,
            sample_id: format!("sample-{round_id}"),
            measurement_event_id: format!("event-{round_id}"),
            prediction_receipt_sha256: digest('a'),
            interval_receipt_sha256: digest('b'),
            model_artifact_sha256: digest('c'),
            event_time_unix_seconds: 100 + round_id as i64 * 10,
            prediction_time_unix_seconds: 101 + round_id as i64 * 10,
            clock_domain_id: "fab-clock".into(),
            acquisition: PhysicalAcquisitionRecordV1 {
                state: acquisition,
                selection_design_kind: PhysicalSelectionDesignKindV1::DeterministicOrAdaptive,
                selection_design_sha256: None,
                inclusion_probability: None,
                acquisition_cost_units: measured.then_some(1.0),
                selected_at_round: matches!(
                    acquisition,
                    PhysicalAcquisitionStateV1::SelectedPending
                        | PhysicalAcquisitionStateV1::Measured
                        | PhysicalAcquisitionStateV1::MeasurementUnavailable
                )
                .then_some(round_id),
                measured_at_unix_seconds: measured.then_some(105 + round_id as i64 * 10),
                operational_measurement_method_sha256: measured.then(|| digest('0')),
            },
            qualification: K1QualificationRecordV1 {
                state: qualification,
                label_id: has_label.then(|| format!("label-{round_id}")),
                label_revision: has_label.then_some(1),
                label_payload_sha256: has_label.then(|| digest('d')),
                measurement_event_id: has_label.then(|| format!("event-{round_id}")),
                label_available_at_unix_seconds: has_label.then_some(106 + round_id as i64 * 10),
                report_sha256: matches!(
                    qualification,
                    K1QualificationStateV1::Eligible
                        | K1QualificationStateV1::Unknown
                        | K1QualificationStateV1::Blocked
                )
                .then(|| digest('e')),
                report_available_at_unix_seconds: matches!(
                    qualification,
                    K1QualificationStateV1::Eligible
                        | K1QualificationStateV1::Unknown
                        | K1QualificationStateV1::Blocked
                )
                .then_some(107 + round_id as i64 * 10),
                qualified_loss: eligible.then_some(if round_id % 2 == 0 { 0.0 } else { 1.0 }),
                qualified_loss_evidence_sha256: eligible.then(|| digest('f')),
            },
            tracking: FeedbackTrackingRecordV1 {
                state: tracking,
                proxy_delay_rounds: None,
                actual_feedback_delay_rounds: None,
                controller_feedback_received_at_unix_seconds: None,
                proxy_survival_probability: None,
                tracking_slot_id: None,
                tracking_end_round: None,
                action_receipt_sha256: matches!(
                tracking,
                FeedbackTrackingStateV1::UnlimitedReceived
                    | FeedbackTrackingStateV1::UnlimitedForfeited
                    | FeedbackTrackingStateV1::ProxyNotTracked
                    | FeedbackTrackingStateV1::CapacityDenied
                    | FeedbackTrackingStateV1::Evicted
                    | FeedbackTrackingStateV1::ProxyExpired
                    | FeedbackTrackingStateV1::TrackedReceived
            )
            .then(|| digest('1')),
                controller_feedback_received_at_unix_seconds: matches!(
                    tracking,
                    FeedbackTrackingStateV1::UnlimitedReceived | FeedbackTrackingStateV1::TrackedReceived
                )
                .then_some(108 + round_id as i64 * 10),
                controller_update_applied: tracking == FeedbackTrackingStateV1::UnlimitedReceived
                    && eligible,
            },
            independent_audit: None,
        }
    }

    fn input(
        controller: ObservationRiskControllerV1,
        mut rounds: Vec<ObservationProcessRoundV1>,
    ) -> ObservationProcessInputV1 {
        rounds.sort_by_key(|row| row.round_id);
        let source_ledger_sha256 = json_sha256(&rounds).unwrap();
        let (policy, delay, max_delay, capacity, declared, delay_sha) = match controller {
            ObservationRiskControllerV1::None => (None, None, None, None, false, None),
            ObservationRiskControllerV1::FixedDelayAci => {
                (Some(digest('2')), Some(1), None, None, true, Some(digest('3')))
            }
            ObservationRiskControllerV1::CapacityAwareOcrcBounded => (
                Some(digest('4')),
                None,
                Some(2),
                Some(2),
                true,
                Some(digest('5')),
            ),
        };
        ObservationProcessInputV1 {
            schema: OBSERVATION_PROCESS_SCHEMA_V1.into(),
            run_id: "observation-run-001".into(),
            population_manifest_sha256: digest('6'),
            source_ledger_sha256,
            controller,
            controller_policy_sha256: policy,
            delay_contract_sha256: delay_sha,
            delay_contract_declared_and_enforced: declared,
            fixed_delay_rounds: delay,
            maximum_feedback_delay_rounds: max_delay,
            tracking_capacity: capacity,
            closeout_round: 10,
            independent_audit_design: None,
            rounds,
        }
    }

    #[test]
    fn population_selection_and_k1_qualification_are_distinct_funnel_stages() {
        let data = input(
            ObservationRiskControllerV1::None,
            vec![
                row(
                    1,
                    PhysicalAcquisitionStateV1::NotSelected,
                    K1QualificationStateV1::NotApplicable,
                    FeedbackTrackingStateV1::NotApplicable,
                ),
                row(
                    2,
                    PhysicalAcquisitionStateV1::Measured,
                    K1QualificationStateV1::Eligible,
                    FeedbackTrackingStateV1::NotApplicable,
                ),
                row(
                    3,
                    PhysicalAcquisitionStateV1::Measured,
                    K1QualificationStateV1::Unknown,
                    FeedbackTrackingStateV1::NotApplicable,
                ),
                row(
                    4,
                    PhysicalAcquisitionStateV1::Measured,
                    K1QualificationStateV1::Blocked,
                    FeedbackTrackingStateV1::NotApplicable,
                ),
            ],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.not_selected, 1);
        assert_eq!(report.funnel.measured, 3);
        assert_eq!(report.funnel.k1_eligible, 1);
        assert_eq!(report.funnel.k1_unknown, 1);
        assert_eq!(report.funnel.k1_blocked, 1);
        assert_eq!(report.population_risk.partial_identification_lower, 0.0);
        assert_eq!(report.population_risk.partial_identification_upper, 0.75);
    }

    #[test]
    fn unmeasured_population_round_has_no_ca_ocrc_feedback_outcome() {
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![row(
                1,
                PhysicalAcquisitionStateV1::NotSelected,
                K1QualificationStateV1::NotApplicable,
                FeedbackTrackingStateV1::NotApplicable,
            )],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.not_selected, 1);
        assert_eq!(report.funnel.feedback_not_applicable, 1);
        assert_eq!(report.funnel.proxy_not_tracked, 0);
        assert_eq!(report.funnel.capacity_denied, 0);
        assert_eq!(report.controller_assumption_state, ControllerAssumptionStateV1::InvalidatedByMissingOrUnqualifiedOutcome);
        assert_eq!(report.population_risk.missing_count, 1);
    }

    #[test]
    fn capacity_denial_is_not_physical_nonselection_or_k1_failure() {
        let mut denied = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::CapacityDenied,
        );
        denied.tracking.proxy_delay_rounds = Some(2);
        denied.tracking.action_receipt_sha256 = Some(digest('1'));
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![denied],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.measured, 1);
        assert_eq!(report.funnel.k1_eligible, 1);
        assert_eq!(report.funnel.capacity_denied, 1);
        assert_eq!(report.funnel.not_selected, 0);
        assert_eq!(report.population_risk.population_risk_exactly_identified, true);
    }

    #[test]
    fn capacity_denied_but_physically_qualified_label_still_counts_as_observed_population_evidence() {
        let mut denied = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::CapacityDenied,
        );
        denied.tracking.proxy_delay_rounds = Some(2);
        denied.tracking.actual_feedback_delay_rounds = Some(1);
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![denied],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.measured, 1);
        assert_eq!(report.funnel.k1_eligible, 1);
        assert_eq!(report.funnel.capacity_denied, 1);
        assert_eq!(report.population_risk.missing_count, 0);
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::CompleteUnderDeclaredAssumptions
        );
    }

    #[test]
    fn proxy_not_tracked_remains_distinct_from_physical_nonselection() {
        let mut not_tracked = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::ProxyNotTracked,
        );
        not_tracked.tracking.proxy_delay_rounds = Some(-1);
        not_tracked.tracking.actual_feedback_delay_rounds = Some(2);
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![not_tracked],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.proxy_not_tracked, 1);
        assert_eq!(report.funnel.not_selected, 0);
        assert_eq!(report.funnel.measured, 1);
        assert_eq!(report.funnel.k1_eligible, 1);
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::CompleteUnderDeclaredAssumptions
        );
    }

    #[test]
    fn tracked_feedback_releases_capacity_after_actual_arrival_and_applies_k1_update() {
        let mut received = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::TrackedReceived,
        );
        received.tracking.proxy_delay_rounds = Some(2);
        received.tracking.actual_feedback_delay_rounds = Some(1);
        received.tracking.proxy_survival_probability = Some(0.5);
        received.tracking.tracking_slot_id = Some(101);
        received.tracking.tracking_end_round = Some(2);
        received.tracking.controller_feedback_received_at_unix_seconds = Some(119);
        received.tracking.controller_update_applied = true;
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![received],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.maximum_observed_tracking_occupancy, 1);
        assert_eq!(report.funnel.feedback_received, 1);
        assert_eq!(report.funnel.controller_updates_applied, 1);
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::CompleteUnderDeclaredAssumptions
        );
    }

    #[test]
    fn peak_tracking_occupancy_exceeding_k_is_reported_as_assumption_failure() {
        let mut first = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::AdmittedPending,
        );
        first.tracking.proxy_delay_rounds = Some(2);
        first.tracking.tracking_slot_id = Some(101);
        let mut second = row(
            2,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::AdmittedPending,
        );
        second.tracking.proxy_delay_rounds = Some(1);
        second.tracking.tracking_slot_id = Some(102);
        let mut data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![first, second],
        );
        data.tracking_capacity = Some(1);
        data.closeout_round = 2;
        data.source_ledger_sha256 = json_sha256(&canonical_rounds(&data.rounds)).unwrap();
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.maximum_observed_tracking_occupancy, 2);
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::TrackingCapacityExceeded
        );
    }

    #[test]
    fn exceeding_declared_maximum_physical_delay_is_reported_explicitly() {
        let mut denied = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::CapacityDenied,
        );
        denied.tracking.proxy_delay_rounds = Some(2);
        denied.tracking.actual_feedback_delay_rounds = Some(3);
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![denied],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::DelayContractExceeded
        );
    }

    #[test]
    fn fixed_delay_mismatch_is_reported_as_delay_contract_exceeded() {
        let mut received = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::UnlimitedReceived,
        );
        received.tracking.actual_feedback_delay_rounds = Some(2);
        let data = input(ObservationRiskControllerV1::FixedDelayAci, vec![received]);
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::DelayContractExceeded
        );
    }

    #[test]
    fn fixed_delay_aci_and_capacity_ocrc_cannot_be_mixed_in_one_run() {
        let mut data = input(
            ObservationRiskControllerV1::FixedDelayAci,
            vec![row(
                1,
                PhysicalAcquisitionStateV1::Measured,
                K1QualificationStateV1::Eligible,
                FeedbackTrackingStateV1::UnlimitedReceived,
            )],
        );
        data.rounds[0].tracking.state = FeedbackTrackingStateV1::TrackedReceived;
        assert!(evaluate_observation_process_v1(&data).is_err());
    }

    #[test]
    fn aci_missing_or_unqualified_feedback_invalidates_full_population_assumption() {
        let data = input(
            ObservationRiskControllerV1::FixedDelayAci,
            vec![
                row(
                    0,
                    PhysicalAcquisitionStateV1::Measured,
                    K1QualificationStateV1::Eligible,
                    FeedbackTrackingStateV1::UnlimitedReceived,
                ),
                row(
                    1,
                    PhysicalAcquisitionStateV1::Measured,
                    K1QualificationStateV1::Blocked,
                    FeedbackTrackingStateV1::UnlimitedForfeited,
                ),
            ],
        );
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::InvalidatedByMissingOrUnqualifiedOutcome
        );
    }

    #[test]
    fn randomized_acquisition_propensity_is_logged_but_not_multiplied_with_tracking_weight() {
        let mut measured = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::NotApplicable,
        );
        measured.acquisition.selection_design_kind =
            PhysicalSelectionDesignKindV1::RandomizedKnownInclusionProbability;
        measured.acquisition.selection_design_sha256 = Some(digest('7'));
        measured.acquisition.inclusion_probability = Some(0.25);
        let data = input(ObservationRiskControllerV1::None, vec![measured]);
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.population_risk.population_point_estimate, Some(1.0));
        assert_eq!(report.population_risk.point_estimate_method.as_deref(), Some("complete_qualified_population"));
    }

    #[test]
    fn k1_unknown_or_blocked_loss_cannot_be_filled_with_zero() {
        let mut invalid = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Blocked,
            FeedbackTrackingStateV1::NotApplicable,
        );
        invalid.qualification.qualified_loss = Some(0.0);
        invalid.qualification.qualified_loss_evidence_sha256 = Some(digest('8'));
        let data = input(ObservationRiskControllerV1::None, vec![invalid]);
        assert!(evaluate_observation_process_v1(&data).is_err());
    }

    #[test]
    fn audit_observations_are_separate_from_controller_tracking() {
        let mut data = input(
            ObservationRiskControllerV1::None,
            vec![row(
                1,
                PhysicalAcquisitionStateV1::MeasurementUnavailable,
                K1QualificationStateV1::NotApplicable,
                FeedbackTrackingStateV1::NotApplicable,
            )],
        );
        data.independent_audit_design = Some(SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('9'),
            randomization_provenance_sha256: digest('a'),
            reference_method_sha256: digest('b'),
            minimum_inclusion_probability: 1.0,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        });
        data.independent_audit_design.as_mut().unwrap().reference_method_sha256 = digest('d');
        data.rounds[0].independent_audit = Some(IndependentAuditObservationV1 {
            round_id: 1,
            loss: 1.0,
            inclusion_probability: 1.0,
            audit_measurement_receipt_sha256: digest('c'),
        });
        data.source_ledger_sha256 = json_sha256(&canonical_rounds(&data.rounds)).unwrap();
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(report.funnel.independent_audit_observations, 1);
        assert!(report.population_risk.independent_audit_estimate.is_some());
    }

    #[test]
    fn independent_audit_reference_method_must_differ_from_operational_method() {
        let mut data = input(
            ObservationRiskControllerV1::None,
            vec![row(
                1,
                PhysicalAcquisitionStateV1::Measured,
                K1QualificationStateV1::Unknown,
                FeedbackTrackingStateV1::NotApplicable,
            )],
        );
        data.independent_audit_design = Some(SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('9'),
            randomization_provenance_sha256: digest('a'),
            reference_method_sha256: digest('0'),
            minimum_inclusion_probability: 1.0,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        });
        data.rounds[0].independent_audit = Some(IndependentAuditObservationV1 {
            round_id: 1,
            loss: 1.0,
            inclusion_probability: 1.0,
            audit_measurement_receipt_sha256: digest('c'),
        });
        data.source_ledger_sha256 = json_sha256(&canonical_rounds(&data.rounds)).unwrap();
        assert!(evaluate_observation_process_v1(&data).is_err());
    }

    #[test]
    fn label_availability_cannot_predate_physical_measurement() {
        let mut measured = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::NotApplicable,
        );
        measured.qualification.label_available_at_unix_seconds = Some(114);
        let data = input(ObservationRiskControllerV1::None, vec![measured]);
        assert!(evaluate_observation_process_v1(&data).is_err());
    }

    #[test]
    fn controller_feedback_cannot_predate_the_k1_report() {
        let mut received = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::TrackedReceived,
        );
        received.tracking.proxy_delay_rounds = Some(2);
        received.tracking.actual_feedback_delay_rounds = Some(1);
        received.tracking.proxy_survival_probability = Some(0.5);
        received.tracking.tracking_slot_id = Some(101);
        received.tracking.tracking_end_round = Some(2);
        received.tracking.controller_feedback_received_at_unix_seconds =
            Some(received.qualification.report_available_at_unix_seconds.unwrap() - 1);
        received.tracking.controller_update_applied = true;
        let data = input(
            ObservationRiskControllerV1::CapacityAwareOcrcBounded,
            vec![received],
        );
        assert!(evaluate_observation_process_v1(&data).is_err());
    }

    #[test]
    fn closeout_must_include_every_population_prediction_round() {
        let data = input(
            ObservationRiskControllerV1::None,
            vec![row(
                1,
                PhysicalAcquisitionStateV1::NotSelected,
                K1QualificationStateV1::NotApplicable,
                FeedbackTrackingStateV1::NotApplicable,
            )],
        );
        let mut truncated = data;
        truncated.closeout_round = 0;
        assert!(evaluate_observation_process_v1(&truncated).is_err());
    }

    #[test]
    fn unresolved_due_feedback_is_reported_as_trace_incomplete() {
        let mut pending = row(
            1,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            FeedbackTrackingStateV1::UnlimitedPending,
        );
        pending.tracking.action_receipt_sha256 = None;
        pending.tracking.controller_update_applied = false;
        let mut data = input(ObservationRiskControllerV1::FixedDelayAci, vec![pending]);
        data.closeout_round = 1;
        let report = evaluate_observation_process_v1(&data).unwrap();
        assert_eq!(
            report.controller_assumption_state,
            ControllerAssumptionStateV1::TraceIncomplete
        );
    }

    #[test]
    fn input_order_does_not_change_sealed_ledger_or_report() {
        let rows = vec![
            row(
                1,
                PhysicalAcquisitionStateV1::NotSelected,
                K1QualificationStateV1::NotApplicable,
                FeedbackTrackingStateV1::NotApplicable,
            ),
            row(
                2,
                PhysicalAcquisitionStateV1::Measured,
                K1QualificationStateV1::Eligible,
                FeedbackTrackingStateV1::NotApplicable,
            ),
        ];
        let first = input(ObservationRiskControllerV1::None, rows.clone());
        let mut second = input(ObservationRiskControllerV1::None, rows);
        second.rounds.reverse();
        let a = evaluate_observation_process_v1(&first).unwrap();
        let b = evaluate_observation_process_v1(&second).unwrap();
        assert_eq!(a.report_sha256, b.report_sha256);
    }
}
