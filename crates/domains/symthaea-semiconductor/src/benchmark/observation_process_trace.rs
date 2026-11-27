// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed observation-process trace across metrology selection, K1 qualification,
//! online feedback tracking, and independent population audit.
//!
//! This is an evidence-path validator, not a combined estimator or controller.
//! It deliberately keeps operational measurement, K1 eligibility, ACI/CA-OCRC
//! tracking, and independent audit as distinct mechanisms.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};
use super::selection_risk_bounds::{
    SelectionAuditDesignV1, SelectionRiskReportV1,
};

pub const OBSERVATION_PROCESS_TRACE_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-trace.v1";
pub const OBSERVATION_PROCESS_ROW_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-row.v1";
pub const OBSERVATION_PROCESS_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationProcessStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationalAcquisitionStateV1 {
    Unselected,
    SelectedPending,
    Measured,
    MeasurementUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationalQualificationStateV1 {
    NotApplicable,
    Eligible,
    Unknown,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnlineControllerKindV1 {
    None,
    TauDaci,
    CaOcrc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerTrackingStateV1 {
    NotSubmitted,
    Pending,
    RetainedPending,
    ConsumedQualified,
    ReceivedUnqualified,
    Expired,
    Evicted,
    NotAdmitted,
    ImmediateTrivialResolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentAuditStateV1 {
    NotConfigured,
    NotInAuditFrame,
    NotSelected,
    SelectedPending,
    Measured,
    AuditUnavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationalAcquisitionEvidenceV1 {
    pub state: OperationalAcquisitionStateV1,
    /// Optional for deterministic/adaptive selectors whose exact per-unit
    /// inclusion probability is not known. Missing is never imputed as one.
    pub inclusion_probability: Option<f64>,
    pub selection_design_sha256: Option<String>,
    pub measurement_system_sha256: Option<String>,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub label_payload_sha256: Option<String>,
    pub measurement_value: Option<f64>,
    pub standard_uncertainty_upper_bound: Option<f64>,
    pub qualification_state: OperationalQualificationStateV1,
    pub k1_report: Option<LabelConsumptionEligibilityReportV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerTrackingEvidenceV1 {
    pub state: ControllerTrackingStateV1,
    pub prediction_receipt_sha256: Option<String>,
    pub feedback_receipt_sha256: Option<String>,
    pub degenerate_resolution_receipt_sha256: Option<String>,
    pub feedback_update_applied: bool,
    pub actual_delay_steps: Option<u64>,
    pub proxy_delay_steps: Option<i64>,
    pub proxy_survival_probability: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndependentAuditEvidenceV1 {
    pub state: IndependentAuditStateV1,
    /// Bernoulli audit inclusion probability for nonqualified rounds.
    pub inclusion_probability: Option<f64>,
    pub selection_receipt_sha256: Option<String>,
    pub measurement_receipt_sha256: Option<String>,
    pub loss: Option<f64>,
    /// Audit results are archived on a distinct path. They are not silently
    /// inserted into the operational ACI/CA-OCRC feedback stream.
    pub independently_archived: bool,
    pub archive_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessRowV1 {
    pub schema: String,
    pub round_id: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_time_unix_seconds: i64,
    pub round_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub prediction_receipt_sha256: String,
    pub operational: OperationalAcquisitionEvidenceV1,
    pub controller: ControllerTrackingEvidenceV1,
    pub audit: IndependentAuditEvidenceV1,
    pub row_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessControllerDeclarationV1 {
    pub kind: OnlineControllerKindV1,
    pub run_id: String,
    pub policy_sha256: Option<String>,
    pub delay_contract_sha256: Option<String>,
    pub maximum_delay_steps: Option<u64>,
    pub tracking_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessTraceInputV1 {
    pub schema: String,
    pub trace_id: String,
    pub clock_domain_id: String,
    pub population_manifest_sha256: String,
    pub source_ledger_sha256: String,
    pub inventory_complete: bool,
    pub population_round_ids: Vec<u64>,
    pub controller: ObservationProcessControllerDeclarationV1,
    pub audit_design: Option<SelectionAuditDesignV1>,
    pub selection_risk_report: Option<SelectionRiskReportV1>,
    pub rows: Vec<ObservationProcessRowV1>,
    pub trace_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessTraceReportV1 {
    pub schema: String,
    pub trace_id: String,
    pub trace_sha256: String,
    pub status: ObservationProcessStatusV1,
    pub reason_codes: Vec<String>,
    pub population_count: usize,
    pub visible_row_count: usize,
    pub missing_round_ids: Vec<u64>,
    pub duplicate_round_count: usize,
    pub operational_unselected_count: usize,
    pub operational_selected_pending_count: usize,
    pub operational_measured_count: usize,
    pub operational_measurement_unavailable_count: usize,
    pub operational_inclusion_probability_known_count: usize,
    pub k1_eligible_count: usize,
    pub k1_unknown_count: usize,
    pub k1_blocked_count: usize,
    pub controller_pending_count: usize,
    pub controller_retained_pending_count: usize,
    pub controller_consumed_qualified_count: usize,
    pub controller_received_unqualified_count: usize,
    pub controller_expired_count: usize,
    pub controller_evicted_count: usize,
    pub controller_not_admitted_count: usize,
    pub controller_immediate_trivial_count: usize,
    pub audit_frame_count: usize,
    pub audit_not_in_frame_count: usize,
    pub audit_not_selected_count: usize,
    pub audit_selected_pending_count: usize,
    pub audit_selected_count: usize,
    pub audit_measured_count: usize,
    pub audit_unavailable_count: usize,
    pub selection_risk_report_sha256: Option<String>,
    pub joint_inclusion_probability_computed: bool,
    pub combined_population_point_estimate_emitted: bool,
    pub combined_coverage_theorem_emitted: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ObservationProcessTraceErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), ObservationProcessTraceErrorV1> {
    if value.trim().is_empty() {
        return Err(ObservationProcessTraceErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, ObservationProcessTraceErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ObservationProcessTraceErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn label_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, ObservationProcessTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn selection_risk_report_digest(
    report: &SelectionRiskReportV1,
) -> Result<String, ObservationProcessTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn row_digest(row: &ObservationProcessRowV1) -> Result<String, ObservationProcessTraceErrorV1> {
    let mut canonical = row.clone();
    canonical.row_sha256.clear();
    json_sha256(&canonical)
}

pub fn seal_observation_process_row_v1(
    row: &ObservationProcessRowV1,
) -> Result<String, ObservationProcessTraceErrorV1> {
    row_digest(row)
}

pub fn seal_observation_process_trace_v1(
    trace: &ObservationProcessTraceInputV1,
) -> Result<String, ObservationProcessTraceErrorV1> {
    let mut canonical = trace.clone();
    canonical.trace_sha256.clear();
    canonical.population_round_ids.sort_unstable();
    canonical.rows.sort_by_key(|row| row.round_id);
    json_sha256(&canonical)
}

fn validate_controller_declaration(
    declaration: &ObservationProcessControllerDeclarationV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    match declaration.kind {
        OnlineControllerKindV1::None => {
            if declaration.policy_sha256.is_some()
                || declaration.delay_contract_sha256.is_some()
                || declaration.maximum_delay_steps.is_some()
                || declaration.tracking_capacity.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(
                    "NONE controller declaration must not carry policy/delay/capacity".into(),
                ));
            }
        }
        OnlineControllerKindV1::TauDaci => {
            require_text(&declaration.run_id, "controller.run_id")?;
            if !declaration.policy_sha256.as_deref().is_some_and(valid_sha256)
                || !declaration.delay_contract_sha256.as_deref().is_some_and(valid_sha256)
                || declaration.maximum_delay_steps.is_none_or(|delay| delay == 0)
                || declaration.tracking_capacity.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(
                    "tau-DACI declaration requires policy, fixed-delay contract and no capacity".into(),
                ));
            }
        }
        OnlineControllerKindV1::CaOcrc => {
            require_text(&declaration.run_id, "controller.run_id")?;
            if !declaration.policy_sha256.as_deref().is_some_and(valid_sha256)
                || !declaration.delay_contract_sha256.as_deref().is_some_and(valid_sha256)
                || declaration.maximum_delay_steps.is_none_or(|delay| delay == 0)
                || declaration.tracking_capacity.is_none_or(|capacity| capacity == 0)
            {
                return Err(ObservationProcessTraceErrorV1(
                    "CA-OCRC declaration requires policy, bounded-delay contract and positive capacity".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_operational(
    row: &ObservationProcessRowV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let op = &row.operational;
    if let Some(probability) = op.inclusion_probability
        && (!probability.is_finite() || probability <= 0.0 || probability > 1.0)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} operational inclusion probability is invalid",
            row.round_id
        )));
    }
    if let Some(digest) = op.selection_design_sha256.as_deref()
        && !valid_sha256(digest)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} operational selection design digest is malformed",
            row.round_id
        )));
    }

    match op.state {
        OperationalAcquisitionStateV1::Unselected
        | OperationalAcquisitionStateV1::SelectedPending
        | OperationalAcquisitionStateV1::MeasurementUnavailable => {
            if op.measurement_system_sha256.is_some()
                || op.label_id.is_some()
                || op.label_revision.is_some()
                || op.label_payload_sha256.is_some()
                || op.measurement_value.is_some()
                || op.standard_uncertainty_upper_bound.is_some()
                || op.k1_report.is_some()
                || op.qualification_state != OperationalQualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} has label/K1 evidence without a completed operational measurement",
                    row.round_id
                )));
            }
        }
        OperationalAcquisitionStateV1::Measured => {
            require_text(
                op.measurement_system_sha256.as_deref().unwrap_or(""),
                "operational.measurement_system_sha256",
            )?;
            if !valid_sha256(op.measurement_system_sha256.as_deref().unwrap_or(""))
                || op.label_id.as_deref().is_none_or(str::is_empty)
                || op.label_revision.is_none_or(|revision| revision == 0)
                || !op.label_payload_sha256.as_deref().is_some_and(valid_sha256)
                || op.measurement_value.is_none_or(|value| !value.is_finite())
                || op
                    .standard_uncertainty_upper_bound
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} measured label identity/value/uncertainty is incomplete",
                    row.round_id
                )));
            }
            let report = op.k1_report.as_ref().ok_or_else(|| {
                ObservationProcessTraceErrorV1(format!(
                    "round {} measured result is missing its K1 qualification report",
                    row.round_id
                ))
            })?;
            if label_report_digest(report)? != report.report_sha256
                || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
                || report.label_id != op.label_id.as_deref().unwrap_or("")
                || Some(report.label_revision) != op.label_revision
                || report.measurement_event_id != row.measurement_event_id
                || Some(report.label_sha256.as_str()) != op.label_payload_sha256.as_deref()
                || !report.qualified_corrected_value.is_finite()
                || report
                    .qualified_standard_uncertainty_upper_bound
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                || report.qualified_standard_uncertainty_upper_bound
                    != op.standard_uncertainty_upper_bound
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} K1 report does not bind the exact operational label",
                    row.round_id
                )));
            }
            let expected = match report.eligibility {
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                    if report.downstream_consumption_allowed =>
                {
                    OperationalQualificationStateV1::Eligible
                }
                LabelConsumptionEligibilityV1::Unknown => OperationalQualificationStateV1::Unknown,
                LabelConsumptionEligibilityV1::Blocked => OperationalQualificationStateV1::Blocked,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
                    return Err(ObservationProcessTraceErrorV1(format!(
                        "round {} K1 report is eligible but downstream consumption is not allowed",
                        row.round_id
                    )));
                }
            };
            if op.qualification_state != expected {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} operational K1 state conflicts with its receipt",
                    row.round_id
                )));
            }
            if expected == OperationalQualificationStateV1::Eligible
                && (op.measurement_value != Some(report.qualified_corrected_value)
                    || op.standard_uncertainty_upper_bound.is_none())
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} eligible K1 label must exactly match the qualified value and uncertainty bound",
                    row.round_id
                )));
            }
            if report.training_snapshot_mutated || report.equipment_control_authority {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} K1 report exceeds evidence-only authority",
                    row.round_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_controller(
    row: &ObservationProcessRowV1,
    declaration: &ObservationProcessControllerDeclarationV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let controller = &row.controller;
    if controller
        .proxy_survival_probability
        .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} proxy survival probability is invalid",
            row.round_id
        )));
    }
    if let Some(digest) = controller.prediction_receipt_sha256.as_deref()
        && !valid_sha256(digest)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} controller prediction digest is malformed",
            row.round_id
        )));
    }
    if let Some(digest) = controller.feedback_receipt_sha256.as_deref()
        && !valid_sha256(digest)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} controller feedback digest is malformed",
            row.round_id
        )));
    }
    if let Some(digest) = controller.degenerate_resolution_receipt_sha256.as_deref()
        && !valid_sha256(digest)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} degenerate resolution digest is malformed",
            row.round_id
        )));
    }

    if declaration.kind == OnlineControllerKindV1::None {
        if controller.state != ControllerTrackingStateV1::NotSubmitted
            || controller.prediction_receipt_sha256.is_some()
            || controller.feedback_receipt_sha256.is_some()
            || controller.degenerate_resolution_receipt_sha256.is_some()
            || controller.feedback_update_applied
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} carries controller state in a controller-free trace",
                row.round_id
            )));
        }
        return Ok(());
    }

    if controller.prediction_receipt_sha256.is_none() {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} online controller path is missing its prediction receipt",
            row.round_id
        )));
    }

    let op_eligible =
        row.operational.qualification_state == OperationalQualificationStateV1::Eligible;
    let op_unqualified = matches!(
        row.operational.qualification_state,
        OperationalQualificationStateV1::Unknown | OperationalQualificationStateV1::Blocked
    );

    match controller.state {
        ControllerTrackingStateV1::NotSubmitted => {
            if controller.feedback_receipt_sha256.is_some()
                || controller.degenerate_resolution_receipt_sha256.is_some()
                || controller.feedback_update_applied
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} NOT_SUBMITTED cannot have feedback or mutation",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::Pending => {
            if controller.feedback_receipt_sha256.is_some()
                || controller.degenerate_resolution_receipt_sha256.is_some()
                || controller.feedback_update_applied
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} PENDING cannot have a feedback receipt or mutation",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::RetainedPending => {
            if declaration.kind != OnlineControllerKindV1::CaOcrc
                || !matches!(
                    row.operational.state,
                    OperationalAcquisitionStateV1::SelectedPending
                        | OperationalAcquisitionStateV1::Measured
                )
                || controller.proxy_delay_steps.is_none_or(|delay| delay < 0)
                || controller.feedback_receipt_sha256.is_some()
                || controller.feedback_update_applied
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} RETAINED_PENDING is valid only for an admitted CA-OCRC entry",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ConsumedQualified => {
            if !op_eligible
                || controller.feedback_receipt_sha256.is_none()
                || !controller.feedback_update_applied
                || !matches!(
                    row.operational.state,
                    OperationalAcquisitionStateV1::Measured
                )
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} consumed feedback requires an eligible K1 label and update receipt",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ReceivedUnqualified => {
            if !op_unqualified
                || controller.feedback_receipt_sha256.is_none()
                || controller.feedback_update_applied
                || row.operational.state != OperationalAcquisitionStateV1::Measured
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} unqualified feedback must be UNKNOWN/BLOCKED and cannot update",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::Expired | ControllerTrackingStateV1::Evicted => {
            if controller.feedback_update_applied {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} expired/evicted feedback cannot update an online controller",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::NotAdmitted => {
            if declaration.kind != OnlineControllerKindV1::CaOcrc
                || controller.feedback_update_applied
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} NOT_ADMITTED is only valid for CA-OCRC and cannot update",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ImmediateTrivialResolved => {
            if controller.degenerate_resolution_receipt_sha256.is_none()
                || !controller.feedback_update_applied
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} immediate trivial resolution requires its own receipt",
                    row.round_id
                )));
            }
        }
    }

    if controller.state == ControllerTrackingStateV1::NotSubmitted {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} declares an online controller but no prediction/feedback path was submitted",
            row.round_id
        )));
    }

    match declaration.kind {
        OnlineControllerKindV1::TauDaci => {
            if matches!(
                controller.state,
                ControllerTrackingStateV1::ConsumedQualified
                    | ControllerTrackingStateV1::ReceivedUnqualified
            ) && controller.actual_delay_steps != declaration.maximum_delay_steps
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} tau-DACI feedback delay differs from the fixed policy delay",
                    row.round_id
                )));
            }
            if controller.proxy_delay_steps.is_some()
                || controller.proxy_survival_probability.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} tau-DACI trace cannot carry CA-OCRC proxy fields",
                    row.round_id
                )));
            }
        }
        OnlineControllerKindV1::CaOcrc => {
            if controller.state != ControllerTrackingStateV1::ImmediateTrivialResolved
                && controller.proxy_delay_steps.is_none()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} CA-OCRC path is missing its proxy delay",
                    row.round_id
                )));
            }
            if matches!(
                controller.state,
                ControllerTrackingStateV1::ConsumedQualified
                    | ControllerTrackingStateV1::ReceivedUnqualified
            ) {
                let actual = controller.actual_delay_steps.ok_or_else(|| {
                    ObservationProcessTraceErrorV1(format!(
                        "round {} consumed CA-OCRC feedback is missing actual delay",
                        row.round_id
                    ))
                })?;
                let proxy = controller.proxy_delay_steps.unwrap_or(-1);
                if actual > declaration.maximum_delay_steps.unwrap_or(0)
                    || proxy < 0
                    || proxy < actual as i64
                    || controller
                        .proxy_survival_probability
                        .is_none_or(|probability| probability <= 0.0)
                {
                    return Err(ObservationProcessTraceErrorV1(format!(
                        "round {} CA-OCRC consumed feedback is outside proxy survival bounds",
                        row.round_id
                    )));
                }
            }
        }
        OnlineControllerKindV1::None => {}
    }

    Ok(())
}

fn validate_audit(
    row: &ObservationProcessRowV1,
    design: Option<&SelectionAuditDesignV1>,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let audit = &row.audit;
    if design.is_none() {
        if audit.state != IndependentAuditStateV1::NotConfigured
            || audit.inclusion_probability.is_some()
            || audit.selection_receipt_sha256.is_some()
            || audit.measurement_receipt_sha256.is_some()
            || audit.loss.is_some()
            || audit.independently_archived
            || audit.archive_receipt_sha256.is_some()
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} carries audit data without a declared audit design",
                row.round_id
            )));
        }
        return Ok(());
    }

    let design = design.expect("design checked as present");
    let nonqualified = row.operational.qualification_state
        != OperationalQualificationStateV1::Eligible;
    if !nonqualified {
        if audit.state != IndependentAuditStateV1::NotInAuditFrame
            || audit.inclusion_probability.is_some()
            || audit.selection_receipt_sha256.is_some()
            || audit.measurement_receipt_sha256.is_some()
            || audit.loss.is_some()
            || audit.independently_archived
            || audit.archive_receipt_sha256.is_some()
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} qualified operational sample is outside the nonqualified audit frame",
                row.round_id
            )));
        }
        return Ok(());
    }

    let probability = audit.inclusion_probability.ok_or_else(|| {
        ObservationProcessTraceErrorV1(format!(
            "round {} nonqualified audit-frame unit is missing its declared inclusion probability",
            row.round_id
        ))
    })?;
    if !probability.is_finite()
        || probability < design.minimum_inclusion_probability
        || probability > 1.0
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {} audit inclusion probability violates positivity",
            row.round_id
        )));
    }

    match audit.state {
        IndependentAuditStateV1::NotConfigured | IndependentAuditStateV1::NotInAuditFrame => {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} nonqualified audit-frame unit has an invalid audit state",
                row.round_id
            )));
        }
        IndependentAuditStateV1::NotSelected => {
            if audit.selection_receipt_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || audit.measurement_receipt_sha256.is_some()
                || audit.loss.is_some()
                || audit.independently_archived
                || audit.archive_receipt_sha256.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} not-selected audit unit requires a randomization decision receipt and no measurement/archive",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::SelectedPending => {
            if audit.selection_receipt_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || audit.measurement_receipt_sha256.is_some()
                || audit.loss.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} selected-pending audit unit has inconsistent receipt fields",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::Measured => {
            if audit.selection_receipt_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || audit.measurement_receipt_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || audit.loss.is_none_or(|loss| !loss.is_finite() || !(0.0..=1.0).contains(&loss))
                || row
                    .operational
                    .measurement_system_sha256
                    .as_deref()
                    .is_some_and(|value| value == design.reference_method_sha256)
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} measured audit unit requires selection, measurement and bounded-loss evidence",
                    row.round_id
                )));
            }
            if audit.independently_archived
                != audit.archive_receipt_sha256.as_deref().is_some_and(valid_sha256)
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} audit archive flag and receipt disagree",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::AuditUnavailable => {
            if audit.selection_receipt_sha256.as_deref().is_none_or(|value| !valid_sha256(value))
                || audit.measurement_receipt_sha256.is_some()
                || audit.loss.is_some()
                || audit.independently_archived
                || audit.archive_receipt_sha256.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {} unavailable audit unit has inconsistent receipt fields",
                    row.round_id
                )));
            }
        }
    }
    Ok(())
}

fn report_digest(
    report: &ObservationProcessTraceReportV1,
) -> Result<String, ObservationProcessTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

pub fn evaluate_observation_process_trace_v1(
    input: &ObservationProcessTraceInputV1,
) -> Result<ObservationProcessTraceReportV1, ObservationProcessTraceErrorV1> {
    if input.schema != OBSERVATION_PROCESS_TRACE_SCHEMA_V1 {
        return Err(ObservationProcessTraceErrorV1(
            "observation-process trace schema is unsupported".into(),
        ));
    }
    require_text(&input.trace_id, "trace.trace_id")?;
    require_text(&input.clock_domain_id, "trace.clock_domain_id")?;
    if !valid_sha256(&input.population_manifest_sha256)
        || !valid_sha256(&input.source_ledger_sha256)
        || !valid_sha256(&input.trace_sha256)
        || seal_observation_process_trace_v1(input)? != input.trace_sha256
    {
        return Err(ObservationProcessTraceErrorV1(
            "trace manifest/source digest is missing or inconsistent".into(),
        ));
    }
    validate_controller_declaration(&input.controller)?;

    if let Some(design) = &input.audit_design {
        if design.schema != super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1
            || !valid_sha256(&design.design_sha256)
            || !valid_sha256(&design.randomization_provenance_sha256)
            || !valid_sha256(&design.reference_method_sha256)
            || !design.independent_of_operational_qualification
            || !design.minimum_inclusion_probability.is_finite()
            || design.minimum_inclusion_probability <= 0.0
            || design.minimum_inclusion_probability > 1.0
            || !design.confidence_delta.is_finite()
            || !(0.0..1.0).contains(&design.confidence_delta)
        {
            return Err(ObservationProcessTraceErrorV1(
                "independent audit design is incomplete or not independent of operational qualification".into(),
            ));
        }
    }

    let mut population_ids = BTreeSet::new();
    for round_id in &input.population_round_ids {
        if *round_id == 0 || !population_ids.insert(*round_id) {
            return Err(ObservationProcessTraceErrorV1(
                "population round IDs must be unique and nonzero".into(),
            ));
        }
    }
    if population_ids.is_empty() {
        return Err(ObservationProcessTraceErrorV1(
            "population manifest must not be empty".into(),
        ));
    }

    let mut seen_rounds = BTreeSet::new();
    let mut counts = [0_usize; 16];
    for row in &input.rows {
        if row.schema != OBSERVATION_PROCESS_ROW_SCHEMA_V1 {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} row schema is unsupported",
                row.round_id
            )));
        }
        require_text(&row.sample_id, "row.sample_id")?;
        require_text(&row.measurement_event_id, "row.measurement_event_id")?;
        if row.round_id == 0
            || !population_ids.contains(&row.round_id)
            || !seen_rounds.insert(row.round_id)
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} is outside population or duplicated",
                row.round_id
            )));
        }
        if row.clock_domain_id != input.clock_domain_id
            || row.event_time_unix_seconds > row.prediction_time_unix_seconds
            || row.prediction_time_unix_seconds > row.round_time_unix_seconds
            || !valid_sha256(&row.prediction_receipt_sha256)
            || row_digest(row)? != row.row_sha256
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} prediction/clock/row receipt is invalid",
                row.round_id
            )));
        }
        validate_operational(row)?;
        validate_controller(row, &input.controller)?;
        validate_audit(row, input.audit_design.as_ref())?;

        match row.operational.state {
            OperationalAcquisitionStateV1::Unselected => counts[0] += 1,
            OperationalAcquisitionStateV1::SelectedPending => counts[1] += 1,
            OperationalAcquisitionStateV1::Measured => counts[2] += 1,
            OperationalAcquisitionStateV1::MeasurementUnavailable => counts[3] += 1,
        }
        match row.operational.qualification_state {
            OperationalQualificationStateV1::NotApplicable => {}
            OperationalQualificationStateV1::Eligible => counts[4] += 1,
            OperationalQualificationStateV1::Unknown => counts[5] += 1,
            OperationalQualificationStateV1::Blocked => counts[6] += 1,
        }
        match row.controller.state {
            ControllerTrackingStateV1::NotSubmitted => {}
            ControllerTrackingStateV1::Pending => counts[7] += 1,
            ControllerTrackingStateV1::RetainedPending => counts[8] += 1,
            ControllerTrackingStateV1::ConsumedQualified => counts[9] += 1,
            ControllerTrackingStateV1::ReceivedUnqualified => counts[10] += 1,
            ControllerTrackingStateV1::Expired => counts[11] += 1,
            ControllerTrackingStateV1::Evicted => counts[12] += 1,
            ControllerTrackingStateV1::NotAdmitted => counts[13] += 1,
            ControllerTrackingStateV1::ImmediateTrivialResolved => counts[14] += 1,
        }
        match row.audit.state {
            IndependentAuditStateV1::NotConfigured | IndependentAuditStateV1::NotInAuditFrame => {}
            IndependentAuditStateV1::NotSelected => counts[15] += 1,
            IndependentAuditStateV1::SelectedPending
            | IndependentAuditStateV1::Measured
            | IndependentAuditStateV1::AuditUnavailable => {}
        }
    }

    let mut missing_round_ids: Vec<u64> = population_ids
        .difference(&seen_rounds)
        .copied()
        .collect();
    missing_round_ids.sort_unstable();
    if input.inventory_complete && !missing_round_ids.is_empty() {
        return Err(ObservationProcessTraceErrorV1(
            "inventory marked complete but population rounds are missing".into(),
        ));
    }
    if !input.inventory_complete && missing_round_ids.is_empty() {
        return Err(ObservationProcessTraceErrorV1(
            "inventory marked incomplete but no missing round is identified".into(),
        ));
    }

    let mut selection_risk_report_sha256 = None;
    if let Some(risk_report) = &input.selection_risk_report {
        if !valid_sha256(&risk_report.input_sha256)
            || !valid_sha256(&risk_report.report_sha256)
            || selection_risk_report_digest(risk_report)? != risk_report.report_sha256
            || risk_report.population_count != population_ids.len()
        {
            return Err(ObservationProcessTraceErrorV1(
                "selection-risk report digest or population count does not match the trace".into(),
            ));
        }
        let qualified = input.rows.iter().filter(|row| {
            row.operational.qualification_state == OperationalQualificationStateV1::Eligible
        }).count();
        let unknown = input.rows.iter().filter(|row| {
            row.operational.qualification_state == OperationalQualificationStateV1::Unknown
        }).count();
        let blocked = input.rows.iter().filter(|row| {
            row.operational.qualification_state == OperationalQualificationStateV1::Blocked
        }).count();
        let unmeasured = input.rows.len() - qualified - unknown - blocked;
        if input.inventory_complete
            && (risk_report.qualified_count != qualified
                || risk_report.unknown_count != unknown
                || risk_report.blocked_count != blocked
                || risk_report.unmeasured_count != unmeasured)
        {
            return Err(ObservationProcessTraceErrorV1(
                "selection-risk partition counts disagree with operational observation trace".into(),
            ));
        }
        let measured_audit_count = input.rows.iter().filter(|row| {
            row.audit.state == IndependentAuditStateV1::Measured
        }).count();
        if let Some(audit_estimate) = &risk_report.independent_audit_estimate {
            if input.audit_design.is_none()
                || audit_estimate.audit_sample_count != measured_audit_count
            {
                return Err(ObservationProcessTraceErrorV1(
                    "selection-risk audit sample count/design disagrees with observation trace".into(),
                ));
            }
        } else if measured_audit_count > 0 && input.inventory_complete {
            return Err(ObservationProcessTraceErrorV1(
                "trace contains measured independent audits but the selection-risk report has no audit estimate".into(),
            ));
        }
        selection_risk_report_sha256 = Some(risk_report.report_sha256.clone());
    }

    let status = if input.inventory_complete {
        ObservationProcessStatusV1::Complete
    } else {
        ObservationProcessStatusV1::Unknown
    };
    let mut report = ObservationProcessTraceReportV1 {
        schema: OBSERVATION_PROCESS_REPORT_SCHEMA_V1.into(),
        trace_id: input.trace_id.clone(),
        trace_sha256: input.trace_sha256.clone(),
        status,
        reason_codes: if input.inventory_complete {
            Vec::new()
        } else {
            vec!["population_inventory_incomplete".into()]
        },
        population_count: population_ids.len(),
        visible_row_count: input.rows.len(),
        missing_round_ids,
        duplicate_round_count: 0,
        operational_unselected_count: counts[0],
        operational_selected_pending_count: counts[1],
        operational_measured_count: counts[2],
        operational_measurement_unavailable_count: counts[3],
        k1_eligible_count: counts[4],
        k1_unknown_count: counts[5],
        k1_blocked_count: counts[6],
        controller_pending_count: counts[7],
        controller_retained_pending_count: counts[8],
        controller_consumed_qualified_count: counts[9],
        controller_received_unqualified_count: counts[10],
        controller_expired_count: counts[11],
        controller_evicted_count: counts[12],
        controller_not_admitted_count: counts[13],
        controller_immediate_trivial_count: counts[14],
        operational_inclusion_probability_known_count: input.rows.iter().filter(|row| {
            row.operational.inclusion_probability.is_some()
        }).count(),
        audit_frame_count: input.rows.iter().filter(|row| {
            row.operational.qualification_state != OperationalQualificationStateV1::Eligible
                && input.audit_design.is_some()
        }).count(),
        audit_not_in_frame_count: input.rows.iter().filter(|row| {
            row.audit.state == IndependentAuditStateV1::NotInAuditFrame
        }).count(),
        audit_not_selected_count: counts[15],
        audit_selected_pending_count: input.rows.iter().filter(|row| {
            row.audit.state == IndependentAuditStateV1::SelectedPending
        }).count(),
        audit_selected_count: input.rows.iter().filter(|row| {
            matches!(
                row.audit.state,
                IndependentAuditStateV1::SelectedPending
                    | IndependentAuditStateV1::Measured
                    | IndependentAuditStateV1::AuditUnavailable
            )
        }).count(),
        audit_measured_count: input.rows.iter().filter(|row| {
            row.audit.state == IndependentAuditStateV1::Measured
        }).count(),
        audit_unavailable_count: input.rows.iter().filter(|row| {
            row.audit.state == IndependentAuditStateV1::AuditUnavailable
        }).count(),
        selection_risk_report_sha256,
        joint_inclusion_probability_computed: false,
        combined_population_point_estimate_emitted: false,
        combined_coverage_theorem_emitted: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
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

    fn base_row(round_id: u64) -> ObservationProcessRowV1 {
        let mut row = ObservationProcessRowV1 {
            schema: OBSERVATION_PROCESS_ROW_SCHEMA_V1.into(),
            round_id,
            sample_id: format!("sample-{round_id}"),
            measurement_event_id: format!("event-{round_id}"),
            event_time_unix_seconds: 100 + round_id as i64,
            prediction_time_unix_seconds: 102 + round_id as i64,
            round_time_unix_seconds: 103 + round_id as i64,
            clock_domain_id: "fab-clock".into(),
            prediction_receipt_sha256: digest('a'),
            operational: OperationalAcquisitionEvidenceV1 {
                state: OperationalAcquisitionStateV1::Unselected,
                inclusion_probability: None,
                selection_design_sha256: None,
                measurement_system_sha256: None,
                label_id: None,
                label_revision: None,
                label_payload_sha256: None,
                measurement_value: None,
                standard_uncertainty_upper_bound: None,
                qualification_state: OperationalQualificationStateV1::NotApplicable,
                k1_report: None,
            },
            controller: ControllerTrackingEvidenceV1 {
                state: ControllerTrackingStateV1::NotSubmitted,
                prediction_receipt_sha256: None,
                feedback_receipt_sha256: None,
                degenerate_resolution_receipt_sha256: None,
                feedback_update_applied: false,
                actual_delay_steps: None,
                proxy_delay_steps: None,
                proxy_survival_probability: None,
            },
            audit: IndependentAuditEvidenceV1 {
                state: IndependentAuditStateV1::NotConfigured,
                inclusion_probability: None,
                selection_receipt_sha256: None,
                measurement_receipt_sha256: None,
                loss: None,
                independently_archived: false,
                archive_receipt_sha256: None,
            },
            row_sha256: String::new(),
        };
        row.row_sha256 = row_digest(&row).unwrap();
        row
    }

    fn measured_row(
        round_id: u64,
        eligibility: LabelConsumptionEligibilityV1,
    ) -> ObservationProcessRowV1 {
        let mut row = base_row(round_id);
        let label_id = format!("label-{round_id}");
        let payload = digest('4');
        let corrected_value = if eligibility == LabelConsumptionEligibilityV1::Blocked {
            11.0
        } else {
            10.0
        };
        let uncertainty = if eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy {
            Some(0.1)
        } else {
            None
        };
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.clone(),
            label_revision: 1,
            measurement_event_id: row.measurement_event_id.clone(),
            calibration_certificate_id: "cal-cert-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility,
            downstream_consumption_allowed:
                eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('5'),
            qualified_corrected_value: corrected_value,
            qualified_standard_uncertainty_upper_bound: uncertainty,
            qualification_report_sha256: digest('6'),
            requalification_report_sha256: digest('7'),
            requalification_frontier_sha256: digest('8'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = label_report_digest(&report).unwrap();
        row.operational = OperationalAcquisitionEvidenceV1 {
            state: OperationalAcquisitionStateV1::Measured,
            inclusion_probability: Some(0.1),
            selection_design_sha256: Some(digest('9')),
            measurement_system_sha256: Some(digest('a')),
            label_id: Some(label_id),
            label_revision: Some(1),
            label_payload_sha256: Some(payload),
            measurement_value: Some(10.0),
            standard_uncertainty_upper_bound: uncertainty,
            qualification_state: match eligibility {
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
                    OperationalQualificationStateV1::Eligible
                }
                LabelConsumptionEligibilityV1::Unknown => OperationalQualificationStateV1::Unknown,
                LabelConsumptionEligibilityV1::Blocked => OperationalQualificationStateV1::Blocked,
            },
            k1_report: Some(report),
        };
        row.row_sha256 = row_digest(&row).unwrap();
        row
    }

    fn input(rows: Vec<ObservationProcessRowV1>, inventory_complete: bool) -> ObservationProcessTraceInputV1 {
        let mut value = ObservationProcessTraceInputV1 {
            schema: OBSERVATION_PROCESS_TRACE_SCHEMA_V1.into(),
            trace_id: "obs-path-001".into(),
            clock_domain_id: "fab-clock".into(),
            population_manifest_sha256: digest('b'),
            source_ledger_sha256: digest('c'),
            inventory_complete,
            population_round_ids: vec![1, 2],
            controller: ObservationProcessControllerDeclarationV1 {
                kind: OnlineControllerKindV1::None,
                run_id: String::new(),
                policy_sha256: None,
                delay_contract_sha256: None,
                maximum_delay_steps: None,
                tracking_capacity: None,
            },
            audit_design: None,
            selection_risk_report: None,
            rows,
            trace_sha256: String::new(),
        };
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        value
    }

    #[test]
    fn complete_trace_counts_acquisition_and_no_controller_separately() {
        let rows = vec![base_row(1), base_row(2)];
        let report = evaluate_observation_process_trace_v1(&input(rows, true)).unwrap();
        assert_eq!(report.status, ObservationProcessStatusV1::Complete);
        assert_eq!(report.operational_unselected_count, 2);
        assert_eq!(report.operational_measured_count, 0);
        assert!(!report.joint_inclusion_probability_computed);
        assert!(!report.combined_population_point_estimate_emitted);
        assert!(!report.combined_coverage_theorem_emitted);
    }

    #[test]
    fn incomplete_inventory_returns_unknown_with_explicit_missing_rounds() {
        let report =
            evaluate_observation_process_trace_v1(&input(vec![base_row(1)], false)).unwrap();
        assert_eq!(report.status, ObservationProcessStatusV1::Unknown);
        assert_eq!(report.missing_round_ids, vec![2]);
    }

    #[test]
    fn unselected_round_cannot_carry_operational_label_or_k1_report() {
        let mut row = base_row(1);
        row.operational.label_id = Some("label-1".into());
        row.row_sha256 = row_digest(&row).unwrap();
        let error = evaluate_observation_process_trace_v1(&input(vec![row, base_row(2)], true));
        assert!(error.is_err());
    }

    #[test]
    fn controller_cannot_consume_unknown_or_blocked_k1_label() {
        let mut row = measured_row(1, LabelConsumptionEligibilityV1::Unknown);
        row.controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::ConsumedQualified,
            prediction_receipt_sha256: Some(digest('d')),
            feedback_receipt_sha256: Some(digest('e')),
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: true,
            actual_delay_steps: Some(2),
            proxy_delay_steps: None,
            proxy_survival_probability: None,
        };
        row.row_sha256 = row_digest(&row).unwrap();
        let mut second = base_row(2);
        second.controller.state = ControllerTrackingStateV1::Pending;
        second.controller.prediction_receipt_sha256 = Some(digest('f'));
        second.row_sha256 = row_digest(&second).unwrap();

        let mut value = input(vec![row, second], true);
        value.controller = ObservationProcessControllerDeclarationV1 {
            kind: OnlineControllerKindV1::TauDaci,
            run_id: "aci-run-unknown".into(),
            policy_sha256: Some(digest('1')),
            delay_contract_sha256: Some(digest('2')),
            maximum_delay_steps: Some(2),
            tracking_capacity: None,
        };
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        assert!(evaluate_observation_process_trace_v1(&value).is_err());
    }
    #[test]
    fn eligible_operational_label_can_update_tau_daci_only_with_matching_report() {
        let mut row = measured_row(
            1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        );
        row.controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::ConsumedQualified,
            prediction_receipt_sha256: Some(digest('b')),
            feedback_receipt_sha256: Some(digest('c')),
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: true,
            actual_delay_steps: Some(2),
            proxy_delay_steps: None,
            proxy_survival_probability: None,
        };
        row.row_sha256 = row_digest(&row).unwrap();
        let mut second = base_row(2);
        second.controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::Pending,
            prediction_receipt_sha256: Some(digest('d')),
            feedback_receipt_sha256: None,
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: false,
            actual_delay_steps: None,
            proxy_delay_steps: None,
            proxy_survival_probability: None,
        };
        second.row_sha256 = row_digest(&second).unwrap();

        let mut value = input(vec![row, second], true);
        value.controller = ObservationProcessControllerDeclarationV1 {
            kind: OnlineControllerKindV1::TauDaci,
            run_id: "tau-run-eligible".into(),
            policy_sha256: Some(digest('e')),
            delay_contract_sha256: Some(digest('f')),
            maximum_delay_steps: Some(2),
            tracking_capacity: None,
        };
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        let report = evaluate_observation_process_trace_v1(&value).unwrap();
        assert_eq!(report.k1_eligible_count, 1);
        assert_eq!(report.controller_consumed_qualified_count, 1);
        assert_eq!(report.operational_inclusion_probability_known_count, 1);
    }

    #[test]
    fn blocked_measured_label_can_be_recorded_but_never_consumed() {
        let mut row = measured_row(1, LabelConsumptionEligibilityV1::Blocked);
        row.controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::ReceivedUnqualified,
            prediction_receipt_sha256: Some(digest('b')),
            feedback_receipt_sha256: Some(digest('c')),
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: false,
            actual_delay_steps: Some(2),
            proxy_delay_steps: None,
            proxy_survival_probability: None,
        };
        row.row_sha256 = row_digest(&row).unwrap();
        let mut second = base_row(2);
        second.controller.state = ControllerTrackingStateV1::Pending;
        second.controller.prediction_receipt_sha256 = Some(digest('d'));
        second.row_sha256 = row_digest(&second).unwrap();
        let mut value = input(vec![row, second], true);
        value.controller = ObservationProcessControllerDeclarationV1 {
            kind: OnlineControllerKindV1::TauDaci,
            run_id: "tau-run-blocked".into(),
            policy_sha256: Some(digest('e')),
            delay_contract_sha256: Some(digest('f')),
            maximum_delay_steps: Some(2),
            tracking_capacity: None,
        };
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        let report = evaluate_observation_process_trace_v1(&value).unwrap();
        assert_eq!(report.k1_blocked_count, 1);
        assert_eq!(report.controller_received_unqualified_count, 1);
    }

    #[test]
    fn independent_audit_not_selected_still_requires_randomization_receipt() {
        let mut value = input(vec![base_row(1), base_row(2)], true);
        value.audit_design = Some(SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('a'),
            randomization_provenance_sha256: digest('b'),
            reference_method_sha256: digest('c'),
            minimum_inclusion_probability: 0.10,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        });
        value.rows[0].audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::NotSelected,
            inclusion_probability: Some(0.2),
            selection_receipt_sha256: Some(digest('d')),
            measurement_receipt_sha256: None,
            loss: None,
            independently_archived: false,
            archive_receipt_sha256: None,
        };
        value.rows[1].audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::NotSelected,
            inclusion_probability: Some(0.2),
            selection_receipt_sha256: Some(digest('g')),
            measurement_receipt_sha256: None,
            loss: None,
            independently_archived: false,
            archive_receipt_sha256: None,
        };
        for row in &mut value.rows {
            row.row_sha256 = row_digest(row).unwrap();
        }
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        assert!(evaluate_observation_process_trace_v1(&value).is_ok());
        value.rows[0].audit.selection_receipt_sha256 = None;
        value.rows[0].row_sha256 = row_digest(&value.rows[0]).unwrap();
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        assert!(evaluate_observation_process_trace_v1(&value).is_err());
    }

    #[test]
    fn controller_tracking_capacity_and_proxy_fields_are_not_valid_for_tau_daci() {
        let mut value = input(vec![base_row(1), base_row(2)], true);
        value.controller = ObservationProcessControllerDeclarationV1 {
            kind: OnlineControllerKindV1::TauDaci,
            run_id: "aci-run-2".into(),
            policy_sha256: Some(digest('a')),
            delay_contract_sha256: Some(digest('b')),
            maximum_delay_steps: Some(2),
            tracking_capacity: None,
        };
        value.rows[0].controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::Pending,
            prediction_receipt_sha256: Some(digest('c')),
            feedback_receipt_sha256: None,
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: false,
            actual_delay_steps: None,
            proxy_delay_steps: Some(1),
            proxy_survival_probability: Some(0.5),
        };
        value.rows[1].controller.state = ControllerTrackingStateV1::Pending;
        value.rows[1].controller.prediction_receipt_sha256 = Some(digest('d'));
        for row in &mut value.rows {
            row.row_sha256 = row_digest(row).unwrap();
        }
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        assert!(evaluate_observation_process_trace_v1(&value).is_err());
    }

    #[test]
    fn ca_ocrc_admitted_path_requires_capacity_and_proxy_delay() {
        let mut selected = base_row(1);
        selected.operational.state = OperationalAcquisitionStateV1::SelectedPending;
        selected.operational.inclusion_probability = Some(0.1);
        selected.operational.selection_design_sha256 = Some(digest('9'));
        selected.row_sha256 = row_digest(&selected).unwrap();
        let mut value = input(vec![selected, base_row(2)], true);
        value.controller = ObservationProcessControllerDeclarationV1 {
            kind: OnlineControllerKindV1::CaOcrc,
            run_id: "ca-run-1".into(),
            policy_sha256: Some(digest('a')),
            delay_contract_sha256: Some(digest('b')),
            maximum_delay_steps: Some(4),
            tracking_capacity: Some(2),
        };
        value.rows[0].controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::RetainedPending,
            prediction_receipt_sha256: Some(digest('c')),
            feedback_receipt_sha256: None,
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: false,
            actual_delay_steps: None,
            proxy_delay_steps: Some(3),
            proxy_survival_probability: Some(0.2),
        };
        value.rows[1].controller = ControllerTrackingEvidenceV1 {
            state: ControllerTrackingStateV1::NotAdmitted,
            prediction_receipt_sha256: Some(digest('d')),
            feedback_receipt_sha256: None,
            degenerate_resolution_receipt_sha256: None,
            feedback_update_applied: false,
            actual_delay_steps: None,
            proxy_delay_steps: Some(-1),
            proxy_survival_probability: None,
        };
        for row in &mut value.rows {
            row.row_sha256 = row_digest(row).unwrap();
        }
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        let report = evaluate_observation_process_trace_v1(&value).unwrap();
        assert_eq!(report.controller_retained_pending_count, 1);
    }

    #[test]
    fn audit_sample_must_be_independent_and_outside_operational_qualified_frame() {
        let mut value = input(vec![base_row(1), base_row(2)], true);
        value.audit_design = Some(SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('a'),
            randomization_provenance_sha256: digest('b'),
            reference_method_sha256: digest('c'),
            minimum_inclusion_probability: 0.10,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        });
        value.rows[0].audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::Measured,
            inclusion_probability: Some(0.2),
            selection_receipt_sha256: Some(digest('d')),
            measurement_receipt_sha256: Some(digest('e')),
            loss: Some(0.4),
            independently_archived: true,
            archive_receipt_sha256: Some(digest('f')),
        };
        value.rows[1].audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::NotSelected,
            inclusion_probability: Some(0.2),
            selection_receipt_sha256: Some(digest('g')),
            measurement_receipt_sha256: None,
            loss: None,
            independently_archived: false,
            archive_receipt_sha256: None,
        };
        for row in &mut value.rows {
            row.row_sha256 = row_digest(row).unwrap();
        }
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        let report = evaluate_observation_process_trace_v1(&value).unwrap();
        assert_eq!(report.audit_measured_count, 1);

        let mut not_independent = value;
        not_independent.audit_design.as_mut().unwrap().independent_of_operational_qualification = false;
        not_independent.trace_sha256 = seal_observation_process_trace_v1(&not_independent).unwrap();
        assert!(evaluate_observation_process_trace_v1(&not_independent).is_err());
    }

    #[test]
    fn order_invariant_trace_digest_and_counts() {
        let mut value = input(vec![base_row(1), base_row(2)], true);
        let first = evaluate_observation_process_trace_v1(&value).unwrap();
        value.rows.reverse();
        value.trace_sha256 = seal_observation_process_trace_v1(&value).unwrap();
        let second = evaluate_observation_process_trace_v1(&value).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
    }
}
