// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint observation-process trace for selective semiconductor metrology.
//!
//! This module keeps physical acquisition, K1 evidence qualification, online
//! controller feedback tracking, and independent population audit as separate
//! mechanisms. It composes receipts by digest and emits no combined IPW
//! estimator or combined coverage theorem.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::selection_risk_bounds::{
    IndependentAuditLossObservationV1, PopulationOutcomeStateV1, QualifiedLossObservationV1,
    SelectionRiskInputV1, SelectionRiskReportV1, SelectionAuditDesignV1,
    SELECTION_RISK_INPUT_SCHEMA_V1, evaluate_selection_adjusted_population_risk_v1,
};

pub const OBSERVATION_TRACE_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-trace.v1";
pub const OBSERVATION_TRACE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.observation-process-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationControllerKindV1 {
    None,
    TauDaci,
    CaOcrc,
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
pub enum K1QualificationStateV1 {
    NotApplicable,
    Eligible,
    Unknown,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerTrackingStateV1 {
    NotSubmitted,
    Pending,
    Retained,
    ReceivedQualified,
    ReceivedUnqualified,
    Expired,
    Evicted,
    ImmediateTrivial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentAuditStateV1 {
    NotSelected,
    SelectedPending,
    AuditMeasured,
    AuditUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditRetentionStateV1 {
    Neither,
    OnlineAuditStore,
    OfflineArchive,
    Both,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationalAcquisitionEvidenceV1 {
    pub state: OperationalAcquisitionStateV1,
    pub design_sha256: Option<String>,
    /// Present only when the declared operational design identifies it.
    pub inclusion_probability: Option<f64>,
    pub measured_at_unix_seconds: Option<i64>,
    pub measurement_system_sha256: Option<String>,
    pub physical_measurement_cost: Option<f64>,
    pub physical_measurement_latency_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct K1QualificationEvidenceV1 {
    pub state: K1QualificationStateV1,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub label_payload_sha256: Option<String>,
    pub qualification_report_sha256: Option<String>,
    pub measurement_event_id: Option<String>,
    pub corrected_value: Option<f64>,
    pub standard_uncertainty_upper_bound: Option<f64>,
    pub label_available_at_unix_seconds: Option<i64>,
    pub report_available_at_unix_seconds: Option<i64>,
    pub calibration_certificate_sha256: Option<String>,
    pub requalification_report_sha256: Option<String>,
    /// Only the qualified stream receives an operational loss contribution.
    pub qualified_loss: Option<f64>,
    pub no_promotion_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControllerFeedbackEvidenceV1 {
    pub tracking_state: ControllerTrackingStateV1,
    pub policy_sha256: Option<String>,
    pub delay_contract_sha256: Option<String>,
    pub tracking_capacity: Option<usize>,
    pub tracking_occupancy_after: Option<usize>,
    pub feedback_delay_contract_steps: Option<u64>,
    pub actual_feedback_delay_steps: Option<u64>,
    pub feedback_round: Option<u64>,
    /// CA-OCRC proxy draw and sampled horizon; absent for tau-DACI.
    pub proxy_uniform_draw: Option<f64>,
    pub proxy_delay_steps: Option<i64>,
    pub proxy_survival_probability: Option<f64>,
    pub prediction_receipt_sha256: Option<String>,
    pub feedback_receipt_sha256: Option<String>,
    pub update_receipt_sha256: Option<String>,
    pub controller_state_changed: bool,
    pub theorem_assumptions_intact: bool,
    /// Known loss for a trivial empty/full prediction set; never a physical label.
    pub immediate_trivial_loss: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndependentAuditEvidenceV1 {
    pub state: IndependentAuditStateV1,
    pub retention: AuditRetentionStateV1,
    pub inclusion_probability: Option<f64>,
    pub audit_selected_at_unix_seconds: Option<i64>,
    pub audit_measured_at_unix_seconds: Option<i64>,
    pub audit_measurement_cost: Option<f64>,
    pub audit_measurement_receipt_sha256: Option<String>,
    pub audit_loss: Option<f64>,
    pub unavailable_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationRoundTraceV1 {
    pub round_id: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub event_time_unix_seconds: i64,
    pub prediction_at_unix_seconds: i64,
    pub round_time_unix_seconds: i64,
    pub clock_domain_id: String,
    pub prediction_receipt_sha256: String,
    pub model_artifact_sha256: String,
    pub interval_method_sha256: String,
    pub operational: OperationalAcquisitionEvidenceV1,
    pub k1: K1QualificationEvidenceV1,
    pub controller: ControllerFeedbackEvidenceV1,
    pub independent_audit: IndependentAuditEvidenceV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessTraceInputV1 {
    pub schema: String,
    pub trace_id: String,
    pub population_manifest_sha256: String,
    pub source_ledger_sha256: String,
    pub clock_domain_id: String,
    pub controller_kind: ObservationControllerKindV1,
    /// Required for tau-DACI or CA-OCRC; absent when no controller is evaluated.
    pub controller_policy_sha256: Option<String>,
    pub controller_delay_contract_sha256: Option<String>,
    pub controller_delay_steps: Option<u64>,
    pub controller_tracking_capacity: Option<usize>,
    pub operational_measurement_budget: usize,
    pub independent_audit_budget: usize,
    pub audit_design: Option<SelectionAuditDesignV1>,
    pub rounds: Vec<ObservationRoundTraceV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessTraceReportV1 {
    pub schema: String,
    pub trace_id: String,
    pub input_sha256: String,
    pub report_sha256: String,
    pub population_count: usize,
    pub operational_unselected_count: usize,
    pub operational_selected_pending_count: usize,
    pub operational_measured_count: usize,
    pub operational_unavailable_count: usize,
    pub k1_eligible_count: usize,
    pub k1_unknown_count: usize,
    pub k1_blocked_count: usize,
    pub controller_not_submitted_count: usize,
    pub controller_pending_count: usize,
    pub controller_retained_count: usize,
    pub controller_received_qualified_count: usize,
    pub controller_received_unqualified_count: usize,
    pub controller_expired_count: usize,
    pub controller_evicted_count: usize,
    pub controller_immediate_trivial_count: usize,
    pub controller_state_update_count: usize,
    pub controller_assumption_break_count: usize,
    pub audit_not_selected_count: usize,
    pub audit_selected_pending_count: usize,
    pub audit_measured_count: usize,
    pub audit_unavailable_count: usize,
    pub operational_measurement_cost_total: f64,
    pub independent_audit_cost_total: f64,
    pub maximum_controller_occupancy: usize,
    pub controller_capacity_violation_count: usize,
    pub selection_risk_report: SelectionRiskReportV1,
    /// Deliberately false: the separate mechanisms are not collapsed into one IPW estimator.
    pub combined_ipw_estimate_emitted: bool,
    /// Deliberately false: selected-stream coverage and population risk are distinct estimands.
    pub combined_coverage_theorem_emitted: bool,
    pub assumptions_and_limits: Vec<String>,
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

fn valid_probability(value: f64, field: &str) -> Result<(), ObservationProcessTraceErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ObservationProcessTraceErrorV1(format!(
            "{field}: expected a finite probability in [0,1]"
        )));
    }
    Ok(())
}

fn valid_nonnegative(value: f64, field: &str) -> Result<(), ObservationProcessTraceErrorV1> {
    if !value.is_finite() || value < 0.0 {
        return Err(ObservationProcessTraceErrorV1(format!(
            "{field}: expected a finite non-negative value"
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

fn validate_optional_digest(
    digest: &Option<String>,
    field: &str,
) -> Result<(), ObservationProcessTraceErrorV1> {
    if let Some(value) = digest
        && !valid_sha256(value)
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "{field}: malformed lowercase SHA-256"
        )));
    }
    Ok(())
}

fn validate_operational(
    row: &ObservationRoundTraceV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let evidence = &row.operational;
    validate_optional_digest(&evidence.design_sha256, "operational.design_sha256")?;
    validate_optional_digest(
        &evidence.measurement_system_sha256,
        "operational.measurement_system_sha256",
    )?;
    if let Some(probability) = evidence.inclusion_probability {
        valid_probability(probability, "operational.inclusion_probability")?;
    }
    if let Some(cost) = evidence.physical_measurement_cost {
        valid_nonnegative(cost, "operational.physical_measurement_cost")?;
    }
    if let Some(latency) = evidence.physical_measurement_latency_seconds {
        valid_nonnegative(latency, "operational.physical_measurement_latency_seconds")?;
    }
    match evidence.state {
        OperationalAcquisitionStateV1::Unselected => {
            if evidence.measured_at_unix_seconds.is_some()
                || evidence.measurement_system_sha256.is_some()
                || evidence.physical_measurement_cost.is_some()
                || evidence.physical_measurement_latency_seconds.is_some()
                || row.k1.state != K1QualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: UNSELECTED cannot carry a physical measurement or K1 status",
                    row.round_id
                )));
            }
        }
        OperationalAcquisitionStateV1::SelectedPending => {
            if evidence.design_sha256.is_none()
                || evidence.measured_at_unix_seconds.is_some()
                || evidence.measurement_system_sha256.is_some()
                || row.k1.state != K1QualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: SELECTED_PENDING requires a design but no measurement/K1 result",
                    row.round_id
                )));
            }
        }
        OperationalAcquisitionStateV1::Measured => {
            if evidence.design_sha256.is_none()
                || evidence.measured_at_unix_seconds.is_none()
                || evidence.measurement_system_sha256.is_none()
                || evidence.physical_measurement_cost.is_none()
                || row.k1.state == K1QualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: MEASURED requires measurement identity/cost and a K1 disposition",
                    row.round_id
                )));
            }
            if evidence
                .measured_at_unix_seconds
                .is_some_and(|time| time < row.event_time_unix_seconds)
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: measurement predates its process event",
                    row.round_id
                )));
            }
        }
        OperationalAcquisitionStateV1::MeasurementUnavailable => {
            if evidence.design_sha256.is_none()
                || evidence.measured_at_unix_seconds.is_some()
                || evidence.measurement_system_sha256.is_some()
                || row.k1.state != K1QualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: MEASUREMENT_UNAVAILABLE cannot carry a physical label/K1 result",
                    row.round_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_k1(
    row: &ObservationRoundTraceV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let k1 = &row.k1;
    validate_optional_digest(&k1.label_payload_sha256, "k1.label_payload_sha256")?;
    validate_optional_digest(
        &k1.qualification_report_sha256,
        "k1.qualification_report_sha256",
    )?;
    validate_optional_digest(
        &k1.calibration_certificate_sha256,
        "k1.calibration_certificate_sha256",
    )?;
    validate_optional_digest(
        &k1.requalification_report_sha256,
        "k1.requalification_report_sha256",
    )?;

    let any_label_identity = k1.label_id.is_some()
        || k1.label_revision.is_some()
        || k1.label_payload_sha256.is_some()
        || k1.qualification_report_sha256.is_some();
    match k1.state {
        K1QualificationStateV1::NotApplicable => {
            if any_label_identity
                || k1.corrected_value.is_some()
                || k1.standard_uncertainty_upper_bound.is_some()
                || k1.qualified_loss.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: NOT_APPLICABLE K1 cannot contain operational label data",
                    row.round_id
                )));
            }
        }
        K1QualificationStateV1::Eligible
        | K1QualificationStateV1::Unknown
        | K1QualificationStateV1::Blocked => {
            if row.operational.state != OperationalAcquisitionStateV1::Measured
                || k1.label_id.as_deref().is_none_or(str::is_empty)
                || k1.label_revision.is_none_or(|revision| revision == 0)
                || k1.label_payload_sha256.is_none()
                || k1.qualification_report_sha256.is_none()
                || k1.measurement_event_id.as_deref() != Some(row.measurement_event_id.as_str())
                || k1.label_available_at_unix_seconds.is_none()
                || k1.report_available_at_unix_seconds.is_none()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: K1 disposition requires a measured, identity-bound label/report",
                    row.round_id
                )));
            }
            let measured_at = row.operational.measured_at_unix_seconds.unwrap();
            let label_available = k1.label_available_at_unix_seconds.unwrap();
            let report_available = k1.report_available_at_unix_seconds.unwrap();
            if label_available < measured_at || report_available < label_available {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: label/K1 report availability chronology is invalid",
                    row.round_id
                )));
            }
            if let Some(value) = k1.corrected_value
                && !value.is_finite()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: corrected label value is non-finite",
                    row.round_id
                )));
            }
            if let Some(uncertainty) = k1.standard_uncertainty_upper_bound {
                valid_nonnegative(uncertainty, "k1.standard_uncertainty_upper_bound")?;
            }
            match k1.state {
                K1QualificationStateV1::Eligible => {
                    if k1.corrected_value.is_none()
                        || k1.standard_uncertainty_upper_bound.is_none()
                        || k1.calibration_certificate_sha256.is_none()
                        || k1.requalification_report_sha256.is_none()
                        || k1.qualified_loss.is_none()
                    {
                        return Err(ObservationProcessTraceErrorV1(format!(
                            "round {}: ELIGIBLE requires corrected value, uncertainty, calibration/requalification and loss",
                            row.round_id
                        )));
                    }
                    valid_nonnegative(
                        k1.standard_uncertainty_upper_bound.unwrap(),
                        "eligible uncertainty",
                    )?;
                    let loss = k1.qualified_loss.unwrap();
                    if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
                        return Err(ObservationProcessTraceErrorV1(format!(
                            "round {}: qualified loss must be in [0,1]",
                            row.round_id
                        )));
                    }
                }
                K1QualificationStateV1::Unknown | K1QualificationStateV1::Blocked => {
                    if k1.qualified_loss.is_some() {
                        return Err(ObservationProcessTraceErrorV1(format!(
                            "round {}: UNKNOWN/BLOCKED cannot contribute an operational loss",
                            row.round_id
                        )));
                    }
                }
                K1QualificationStateV1::NotApplicable => unreachable!(),
            }
        }
    }
    Ok(())
}

fn validate_controller(
    input: &ObservationProcessTraceInputV1,
    row: &ObservationRoundTraceV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let controller = &row.controller;
    validate_optional_digest(&controller.policy_sha256, "controller.policy_sha256")?;
    validate_optional_digest(
        &controller.delay_contract_sha256,
        "controller.delay_contract_sha256",
    )?;
    validate_optional_digest(
        &controller.prediction_receipt_sha256,
        "controller.prediction_receipt_sha256",
    )?;
    validate_optional_digest(
        &controller.feedback_receipt_sha256,
        "controller.feedback_receipt_sha256",
    )?;
    validate_optional_digest(
        &controller.update_receipt_sha256,
        "controller.update_receipt_sha256",
    )?;

    if let Some(draw) = controller.proxy_uniform_draw {
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: CA-OCRC proxy draw must be in [0,1)",
                row.round_id
            )));
        }
    }
    if let Some(probability) = controller.proxy_survival_probability {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: proxy survival probability must be in [0,1]",
                row.round_id
            )));
        }
    }
    if let Some(loss) = controller.immediate_trivial_loss {
        if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: immediate trivial loss must be in [0,1]",
                row.round_id
            )));
        }
    }

    let has_receipts = controller.prediction_receipt_sha256.is_some()
        || controller.feedback_receipt_sha256.is_some()
        || controller.update_receipt_sha256.is_some();
    match input.controller_kind {
        ObservationControllerKindV1::None => {
            if controller.tracking_state != ControllerTrackingStateV1::NotSubmitted
                || has_receipts
                || controller.controller_state_changed
                || controller.policy_sha256.is_some()
                || controller.delay_contract_sha256.is_some()
                || controller.tracking_capacity.is_some()
                || controller.tracking_occupancy_after.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: controller evidence supplied for controller_kind NONE",
                    row.round_id
                )));
            }
        }
        ObservationControllerKindV1::TauDaci => {
            if controller.policy_sha256.as_deref() != input.controller_policy_sha256.as_deref()
                || controller.delay_contract_sha256.is_some()
                || controller.tracking_capacity.is_some()
                || controller.proxy_uniform_draw.is_some()
                || controller.proxy_delay_steps.is_some()
                || controller.proxy_survival_probability.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: τ-DACI metadata must bind its policy and omit CA-OCRC tracking fields",
                    row.round_id
                )));
            }
            if !matches!(
                controller.tracking_state,
                ControllerTrackingStateV1::NotSubmitted
                    | ControllerTrackingStateV1::Pending
                    | ControllerTrackingStateV1::ReceivedQualified
                    | ControllerTrackingStateV1::ReceivedUnqualified
                    | ControllerTrackingStateV1::ImmediateTrivial
            ) {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: τ-DACI cannot use CA-OCRC expiry/eviction states",
                    row.round_id
                )));
            }
        }
        ObservationControllerKindV1::CaOcrc => {
            if controller.policy_sha256.as_deref() != input.controller_policy_sha256.as_deref()
                || controller.delay_contract_sha256.as_deref()
                    != input.controller_delay_contract_sha256.as_deref()
                || controller.tracking_capacity != input.controller_tracking_capacity
                || controller.tracking_capacity.is_none()
                || controller.tracking_occupancy_after.is_none()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: CA-OCRC must bind policy, delay contract, capacity and occupancy",
                    row.round_id
                )));
            }
            if let (Some(capacity), Some(occupancy)) = (
                controller.tracking_capacity,
                controller.tracking_occupancy_after,
            ) && occupancy > capacity
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: CA-OCRC occupancy exceeds declared capacity",
                    row.round_id
                )));
            }
            if matches!(
                controller.tracking_state,
                ControllerTrackingStateV1::Pending
                    | ControllerTrackingStateV1::Retained
                    | ControllerTrackingStateV1::ReceivedQualified
                    | ControllerTrackingStateV1::ReceivedUnqualified
                    | ControllerTrackingStateV1::Expired
                    | ControllerTrackingStateV1::Evicted
            ) {
                if controller.proxy_delay_steps.is_none()
                    || controller.proxy_uniform_draw.is_none()
                    || controller.proxy_survival_probability.is_none()
                {
                    return Err(ObservationProcessTraceErrorV1(format!(
                        "round {}: tracked CA-OCRC round requires draw, proxy delay and survival probability",
                        row.round_id
                    )));
                }
            }
        }
    }

    match controller.tracking_state {
        ControllerTrackingStateV1::NotSubmitted => {
            if has_receipts || controller.controller_state_changed {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: NOT_SUBMITTED cannot carry controller receipts or mutate state",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::Pending | ControllerTrackingStateV1::Retained => {
            if controller.prediction_receipt_sha256.is_none()
                || controller.feedback_receipt_sha256.is_some()
                || controller.update_receipt_sha256.is_some()
                || controller.controller_state_changed
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: PENDING/RETAINED must have a prediction receipt only",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ReceivedQualified => {
            if row.k1.state != K1QualificationStateV1::Eligible
                || controller.prediction_receipt_sha256.is_none()
                || controller.feedback_receipt_sha256.is_none()
                || controller.update_receipt_sha256.is_none()
                || !controller.controller_state_changed
                || controller.feedback_round.is_none()
                || controller.actual_feedback_delay_steps.is_none()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: RECEIVED_QUALIFIED requires eligible K1 and a state-changing controller receipt chain",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ReceivedUnqualified => {
            if !matches!(
                row.k1.state,
                K1QualificationStateV1::Unknown | K1QualificationStateV1::Blocked
            ) || controller.prediction_receipt_sha256.is_none()
                || controller.feedback_receipt_sha256.is_none()
                || controller.update_receipt_sha256.is_some()
                || controller.controller_state_changed
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: RECEIVED_UNQUALIFIED must not carry a controller state update",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::Expired | ControllerTrackingStateV1::Evicted => {
            if controller.update_receipt_sha256.is_some() || controller.controller_state_changed {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: EXPIRED/EVICTED feedback cannot mutate online controller state",
                    row.round_id
                )));
            }
        }
        ControllerTrackingStateV1::ImmediateTrivial => {
            if controller.immediate_trivial_loss.is_none()
                || controller.update_receipt_sha256.is_none()
                || !controller.controller_state_changed
                || row.k1.state != K1QualificationStateV1::NotApplicable
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: IMMEDIATE_TRIVIAL requires a known non-physical loss and controller update receipt",
                    row.round_id
                )));
            }
        }
    }

    if controller.controller_state_changed
        && !matches!(
            controller.tracking_state,
            ControllerTrackingStateV1::ReceivedQualified
                | ControllerTrackingStateV1::ImmediateTrivial
        )
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {}: only qualified physical feedback or known trivial loss may update controller state",
            row.round_id
        )));
    }

    if input.controller_kind == ObservationControllerKindV1::TauDaci
        && matches!(
            controller.tracking_state,
            ControllerTrackingStateV1::ReceivedQualified
                | ControllerTrackingStateV1::ReceivedUnqualified
        )
    {
        let delay = input.controller_delay_steps.ok_or_else(|| {
            ObservationProcessTraceErrorV1("τ-DACI trace is missing its fixed delay".into())
        })?;
        if controller.actual_feedback_delay_steps != Some(delay)
            || controller.feedback_round != Some(row.round_id.saturating_add(delay))
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: τ-DACI feedback does not match its fixed-delay schedule",
                row.round_id
            )));
        }
    }

    if input.controller_kind == ObservationControllerKindV1::CaOcrc
        && matches!(
            controller.tracking_state,
            ControllerTrackingStateV1::ReceivedQualified
                | ControllerTrackingStateV1::ReceivedUnqualified
        )
    {
        let actual_delay = controller.actual_feedback_delay_steps.ok_or_else(|| {
            ObservationProcessTraceErrorV1("CA-OCRC feedback is missing actual delay".into())
        })?;
        if actual_delay > input.controller_delay_steps.unwrap_or(0) {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: actual CA-OCRC delay exceeds the declared d_max",
                row.round_id
            )));
        }
        let proxy = controller.proxy_delay_steps.unwrap_or(-1);
        if proxy < 0 || actual_delay > proxy as u64 {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: CA-OCRC feedback arrived outside its retained proxy window",
                row.round_id
            )));
        }
    }

    Ok(())
}

fn validate_audit(
    input: &ObservationProcessTraceInputV1,
    row: &ObservationRoundTraceV1,
) -> Result<(), ObservationProcessTraceErrorV1> {
    let audit = &row.independent_audit;
    if let Some(probability) = audit.inclusion_probability {
        valid_probability(probability, "audit.inclusion_probability")?;
        if probability == 0.0 {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {}: selected audit item cannot have zero inclusion probability",
                row.round_id
            )));
        }
    }
    validate_optional_digest(
        &audit.audit_measurement_receipt_sha256,
        "audit.audit_measurement_receipt_sha256",
    )?;
    if let Some(cost) = audit.audit_measurement_cost {
        valid_nonnegative(cost, "audit.audit_measurement_cost")?;
    }
    if let Some(loss) = audit.audit_loss
        && (!loss.is_finite() || !(0.0..=1.0).contains(&loss))
    {
        return Err(ObservationProcessTraceErrorV1(format!(
            "round {}: independent audit loss must be in [0,1]",
            row.round_id
        )));
    }

    match audit.state {
        IndependentAuditStateV1::NotSelected => {
            if audit.inclusion_probability.is_some()
                || audit.audit_selected_at_unix_seconds.is_some()
                || audit.audit_measured_at_unix_seconds.is_some()
                || audit.audit_measurement_cost.is_some()
                || audit.audit_measurement_receipt_sha256.is_some()
                || audit.audit_loss.is_some()
                || audit.retention != AuditRetentionStateV1::Neither
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: NOT_SELECTED audit cannot carry an audit sample or retained loss",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::SelectedPending => {
            if input.audit_design.is_none()
                || audit.inclusion_probability.is_none()
                || audit.audit_selected_at_unix_seconds.is_none()
                || audit.audit_measured_at_unix_seconds.is_some()
                || audit.audit_measurement_receipt_sha256.is_some()
                || audit.audit_loss.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: SELECTED_PENDING audit lacks design/probability or has a measurement",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::AuditMeasured => {
            if input.audit_design.is_none()
                || audit.inclusion_probability.is_none()
                || audit.audit_selected_at_unix_seconds.is_none()
                || audit.audit_measured_at_unix_seconds.is_none()
                || audit.audit_measurement_cost.is_none()
                || audit.audit_measurement_receipt_sha256.is_none()
                || audit.audit_loss.is_none()
                || !matches!(
                    audit.retention,
                    AuditRetentionStateV1::OfflineArchive | AuditRetentionStateV1::Both
                )
                || row.k1.state == K1QualificationStateV1::Eligible
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: AUDIT_MEASURED must be an offline-retained independent audit of nonqualified population",
                    row.round_id
                )));
            }
            if audit.audit_measured_at_unix_seconds.unwrap()
                < audit.audit_selected_at_unix_seconds.unwrap()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: audit measurement predates audit selection",
                    row.round_id
                )));
            }
        }
        IndependentAuditStateV1::AuditUnavailable => {
            if input.audit_design.is_none()
                || audit.inclusion_probability.is_none()
                || audit.audit_selected_at_unix_seconds.is_none()
                || audit.audit_measured_at_unix_seconds.is_some()
                || audit.audit_measurement_receipt_sha256.is_some()
                || audit.audit_loss.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(format!(
                    "round {}: AUDIT_UNAVAILABLE requires a selected audit but no result",
                    row.round_id
                )));
            }
        }
    }
    Ok(())
}

/// Validates and seals a four-path observation-process trace.
///
/// The function calls the independent selection-risk module with only:
/// - qualified operational losses;
/// - UNKNOWN/BLOCKED/unmeasured partition;
/// - offline-retained results from the independent audit design.
///
/// Controller proxy-survival weights are deliberately not multiplied by
/// operational inclusion probabilities or audit probabilities.
pub fn evaluate_observation_process_trace_v1(
    input: &ObservationProcessTraceInputV1,
) -> Result<ObservationProcessTraceReportV1, ObservationProcessTraceErrorV1> {
    if input.schema != OBSERVATION_TRACE_SCHEMA_V1 {
        return Err(ObservationProcessTraceErrorV1(
            "observation trace schema is unsupported".into(),
        ));
    }
    require_text(&input.trace_id, "trace_id")?;
    require_text(&input.clock_domain_id, "clock_domain_id")?;
    if !valid_sha256(&input.population_manifest_sha256)
        || !valid_sha256(&input.source_ledger_sha256)
    {
        return Err(ObservationProcessTraceErrorV1(
            "population/source ledger digest is malformed".into(),
        ));
    }
    match input.controller_kind {
        ObservationControllerKindV1::None => {
            if input.controller_policy_sha256.is_some()
                || input.controller_delay_contract_sha256.is_some()
                || input.controller_delay_steps.is_some()
                || input.controller_tracking_capacity.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(
                    "controller metadata must be absent when controller_kind is NONE".into(),
                ));
            }
        }
        ObservationControllerKindV1::TauDaci => {
            if input.controller_policy_sha256.as_deref().is_none_or(|v| !valid_sha256(v))
                || input.controller_delay_steps.is_none_or(|delay| delay == 0)
                || input.controller_delay_contract_sha256.is_some()
                || input.controller_tracking_capacity.is_some()
            {
                return Err(ObservationProcessTraceErrorV1(
                    "τ-DACI requires a policy digest and positive fixed delay only".into(),
                ));
            }
        }
        ObservationControllerKindV1::CaOcrc => {
            if input.controller_policy_sha256.as_deref().is_none_or(|v| !valid_sha256(v))
                || input
                    .controller_delay_contract_sha256
                    .as_deref()
                    .is_none_or(|v| !valid_sha256(v))
                || input.controller_delay_steps.is_none()
                || input.controller_tracking_capacity.is_none_or(|capacity| capacity == 0)
            {
                return Err(ObservationProcessTraceErrorV1(
                    "CA-OCRC requires policy/delay-contract digests, d_max and positive capacity".into(),
                ));
            }
        }
    }
    if input.rounds.is_empty() {
        return Err(ObservationProcessTraceErrorV1(
            "observation trace must contain at least one population round".into(),
        ));
    }
    let mut canonical = input.clone();
    canonical.rounds.sort_by_key(|row| row.round_id);
    let mut round_ids = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();
    let mut prediction_receipts = BTreeSet::new();
    for row in &canonical.rounds {
        if !round_ids.insert(row.round_id)
            || !sample_ids.insert(row.sample_id.as_str())
            || !prediction_receipts.insert(row.prediction_receipt_sha256.as_str())
        {
            return Err(ObservationProcessTraceErrorV1(
                "duplicate round/sample/prediction receipt identity".into(),
            ));
        }
        require_text(&row.sample_id, "round.sample_id")?;
        require_text(&row.measurement_event_id, "round.measurement_event_id")?;
        require_text(&row.clock_domain_id, "round.clock_domain_id")?;
        if row.clock_domain_id != input.clock_domain_id
            || row.event_time_unix_seconds > row.prediction_at_unix_seconds
            || row.prediction_at_unix_seconds > row.round_time_unix_seconds
            || !valid_sha256(&row.prediction_receipt_sha256)
            || !valid_sha256(&row.model_artifact_sha256)
            || !valid_sha256(&row.interval_method_sha256)
        {
            return Err(ObservationProcessTraceErrorV1(format!(
                "round {} has invalid identity, digest, clock or event chronology",
                row.round_id
            )));
        }
        validate_operational(row)?;
        validate_k1(row)?;
        validate_controller(&canonical, row)?;
        validate_audit(&canonical, row)?;
    }

    let population_round_ids: Vec<u64> =
        canonical.rounds.iter().map(|row| row.round_id).collect();
    let qualified_round_ids: Vec<u64> = canonical
        .rounds
        .iter()
        .filter(|row| row.k1.state == K1QualificationStateV1::Eligible)
        .map(|row| row.round_id)
        .collect();
    let unknown_round_ids: Vec<u64> = canonical
        .rounds
        .iter()
        .filter(|row| row.k1.state == K1QualificationStateV1::Unknown)
        .map(|row| row.round_id)
        .collect();
    let blocked_round_ids: Vec<u64> = canonical
        .rounds
        .iter()
        .filter(|row| row.k1.state == K1QualificationStateV1::Blocked)
        .map(|row| row.round_id)
        .collect();
    let unmeasured_round_ids: Vec<u64> = canonical
        .rounds
        .iter()
        .filter(|row| {
            row.k1.state == K1QualificationStateV1::NotApplicable
                && row.operational.state != OperationalAcquisitionStateV1::Measured
        })
        .map(|row| row.round_id)
        .collect();
    let qualified_losses: Vec<QualifiedLossObservationV1> = canonical
        .rounds
        .iter()
        .filter(|row| row.k1.state == K1QualificationStateV1::Eligible)
        .map(|row| QualifiedLossObservationV1 {
            round_id: row.round_id,
            loss: row.k1.qualified_loss.expect("eligible loss validated"),
            evidence_receipt_sha256: row
                .k1
                .qualification_report_sha256
                .clone()
                .expect("eligible report validated"),
        })
        .collect();
    let independent_audit_losses: Vec<IndependentAuditLossObservationV1> = canonical
        .rounds
        .iter()
        .filter(|row| row.independent_audit.state == IndependentAuditStateV1::AuditMeasured)
        .map(|row| IndependentAuditLossObservationV1 {
            round_id: row.round_id,
            loss: row.independent_audit.audit_loss.expect("audit loss validated"),
            inclusion_probability: row
                .independent_audit
                .inclusion_probability
                .expect("audit probability validated"),
            audit_measurement_receipt_sha256: row
                .independent_audit
                .audit_measurement_receipt_sha256
                .clone()
                .expect("audit receipt validated"),
        })
        .collect();

    let selection_input = SelectionRiskInputV1 {
        schema: SELECTION_RISK_INPUT_SCHEMA_V1.into(),
        input_id: format!("{}:selection-risk", input.trace_id),
        population_manifest_sha256: input.population_manifest_sha256.clone(),
        source_ledger_sha256: input.source_ledger_sha256.clone(),
        population_round_ids,
        qualified_round_ids,
        unknown_round_ids,
        blocked_round_ids,
        unmeasured_round_ids,
        qualified_losses,
        audit_design: input.audit_design.clone(),
        independent_audit_losses,
    };
    let selection_risk_report = evaluate_selection_adjusted_population_risk_v1(&selection_input)
        .map_err(|error| ObservationProcessTraceErrorV1(format!(
            "independent population-risk estimator rejected the trace: {error}"
        )))?;

    let count = |predicate: &dyn Fn(&ObservationRoundTraceV1) -> bool| {
        canonical.rounds.iter().filter(|row| predicate(row)).count()
    };
    let mut operational_measurement_cost_total = 0.0;
    let mut independent_audit_cost_total = 0.0;
    let mut maximum_controller_occupancy = 0_usize;
    let mut controller_capacity_violation_count = 0_usize;
    for row in &canonical.rounds {
        operational_measurement_cost_total +=
            row.operational.physical_measurement_cost.unwrap_or(0.0);
        independent_audit_cost_total +=
            row.independent_audit.audit_measurement_cost.unwrap_or(0.0);
        if let (Some(occupancy), Some(capacity)) = (
            row.controller.tracking_occupancy_after,
            row.controller.tracking_capacity,
        ) {
            maximum_controller_occupancy = maximum_controller_occupancy.max(occupancy);
            if occupancy > capacity {
                controller_capacity_violation_count += 1;
            }
        }
    }

    let mut report = ObservationProcessTraceReportV1 {
        schema: OBSERVATION_TRACE_REPORT_SCHEMA_V1.into(),
        trace_id: input.trace_id.clone(),
        input_sha256: json_sha256(&canonical)?,
        report_sha256: String::new(),
        population_count: canonical.rounds.len(),
        operational_unselected_count: count(&|row| row.operational.state == OperationalAcquisitionStateV1::Unselected),
        operational_selected_pending_count: count(&|row| row.operational.state == OperationalAcquisitionStateV1::SelectedPending),
        operational_measured_count: count(&|row| row.operational.state == OperationalAcquisitionStateV1::Measured),
        operational_unavailable_count: count(&|row| row.operational.state == OperationalAcquisitionStateV1::MeasurementUnavailable),
        k1_eligible_count: count(&|row| row.k1.state == K1QualificationStateV1::Eligible),
        k1_unknown_count: count(&|row| row.k1.state == K1QualificationStateV1::Unknown),
        k1_blocked_count: count(&|row| row.k1.state == K1QualificationStateV1::Blocked),
        controller_not_submitted_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::NotSubmitted),
        controller_pending_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::Pending),
        controller_retained_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::Retained),
        controller_received_qualified_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::ReceivedQualified),
        controller_received_unqualified_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::ReceivedUnqualified),
        controller_expired_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::Expired),
        controller_evicted_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::Evicted),
        controller_immediate_trivial_count: count(&|row| row.controller.tracking_state == ControllerTrackingStateV1::ImmediateTrivial),
        controller_state_update_count: count(&|row| row.controller.controller_state_changed),
        controller_assumption_break_count: count(&|row| {
            input.controller_kind != ObservationControllerKindV1::None
                && !row.controller.theorem_assumptions_intact
        }),
        audit_not_selected_count: count(&|row| row.independent_audit.state == IndependentAuditStateV1::NotSelected),
        audit_selected_pending_count: count(&|row| row.independent_audit.state == IndependentAuditStateV1::SelectedPending),
        audit_measured_count: count(&|row| row.independent_audit.state == IndependentAuditStateV1::AuditMeasured),
        audit_unavailable_count: count(&|row| row.independent_audit.state == IndependentAuditStateV1::AuditUnavailable),
        operational_measurement_cost_total,
        independent_audit_cost_total,
        maximum_controller_occupancy,
        controller_capacity_violation_count,
        selection_risk_report,
        combined_ipw_estimate_emitted: false,
        combined_coverage_theorem_emitted: false,
        assumptions_and_limits: vec![
            "Operational physical-metrology selection, K1 qualification, online controller tracking and independent audit are separate observation mechanisms.".into(),
            "CA-OCRC proxy-survival weighting corrects only the declared controller tracking process; it does not correct operational selection or K1 qualification.".into(),
            "The independent audit estimate requires the declared independent Bernoulli audit design and positivity; invalid operational evidence is never repaired by inverse-probability weighting.".into(),
            "τ-DACI and CA-OCRC are alternative controller runs; their alpha/threshold recursions and theorems are not combined.".into(),
            "Selected-stream coverage and population risk are distinct estimands; no combined IPW estimator or joint coverage theorem is emitted.".into(),
        ],
    };
    report.report_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn row(round_id: u64) -> ObservationRoundTraceV1 {
        ObservationRoundTraceV1 {
            round_id,
            sample_id: format!("sample-{round_id}"),
            measurement_event_id: format!("event-{round_id}"),
            event_time_unix_seconds: 1000 + round_id as i64,
            prediction_at_unix_seconds: 1001 + round_id as i64,
            round_time_unix_seconds: 1001 + round_id as i64,
            clock_domain_id: "fab-clock".into(),
            prediction_receipt_sha256: digest('a'),
            model_artifact_sha256: digest('b'),
            interval_method_sha256: digest('c'),
            operational: OperationalAcquisitionEvidenceV1 {
                state: OperationalAcquisitionStateV1::Unselected,
                design_sha256: Some(digest('d')),
                inclusion_probability: Some(0.25),
                measured_at_unix_seconds: None,
                measurement_system_sha256: None,
                physical_measurement_cost: None,
                physical_measurement_latency_seconds: None,
            },
            k1: K1QualificationEvidenceV1 {
                state: K1QualificationStateV1::NotApplicable,
                label_id: None,
                label_revision: None,
                label_payload_sha256: None,
                qualification_report_sha256: None,
                measurement_event_id: None,
                corrected_value: None,
                standard_uncertainty_upper_bound: None,
                label_available_at_unix_seconds: None,
                report_available_at_unix_seconds: None,
                calibration_certificate_sha256: None,
                requalification_report_sha256: None,
                qualified_loss: None,
                no_promotion_reason: None,
            },
            controller: ControllerFeedbackEvidenceV1 {
                tracking_state: ControllerTrackingStateV1::NotSubmitted,
                policy_sha256: None,
                delay_contract_sha256: None,
                tracking_capacity: None,
                tracking_occupancy_after: None,
                feedback_delay_contract_steps: None,
                actual_feedback_delay_steps: None,
                feedback_round: None,
                proxy_uniform_draw: None,
                proxy_delay_steps: None,
                proxy_survival_probability: None,
                prediction_receipt_sha256: None,
                feedback_receipt_sha256: None,
                update_receipt_sha256: None,
                controller_state_changed: false,
                theorem_assumptions_intact: false,
                immediate_trivial_loss: None,
            },
            independent_audit: IndependentAuditEvidenceV1 {
                state: IndependentAuditStateV1::NotSelected,
                retention: AuditRetentionStateV1::Neither,
                inclusion_probability: None,
                audit_selected_at_unix_seconds: None,
                audit_measured_at_unix_seconds: None,
                audit_measurement_cost: None,
                audit_measurement_receipt_sha256: None,
                audit_loss: None,
                unavailable_reason: None,
            },
        }
    }

    fn input(rounds: Vec<ObservationRoundTraceV1>) -> ObservationProcessTraceInputV1 {
        ObservationProcessTraceInputV1 {
            schema: OBSERVATION_TRACE_SCHEMA_V1.into(),
            trace_id: "observation-trace-001".into(),
            population_manifest_sha256: digest('1'),
            source_ledger_sha256: digest('2'),
            clock_domain_id: "fab-clock".into(),
            controller_kind: ObservationControllerKindV1::None,
            controller_policy_sha256: None,
            controller_delay_contract_sha256: None,
            controller_delay_steps: None,
            controller_tracking_capacity: None,
            operational_measurement_budget: 10,
            independent_audit_budget: 10,
            audit_design: None,
            rounds,
        }
    }

    #[test]
    fn unmeasured_population_generates_worst_case_bounds_not_point_estimate() {
        let report = evaluate_observation_process_trace_v1(&input(vec![row(1), row(2)])).unwrap();
        assert_eq!(report.population_count, 2);
        assert_eq!(report.operational_unselected_count, 2);
        assert_eq!(report.selection_risk_report.partial_identification_lower, 0.0);
        assert_eq!(report.selection_risk_report.partial_identification_upper, 1.0);
        assert_eq!(report.selection_risk_report.population_point_estimate, None);
        assert!(!report.combined_ipw_estimate_emitted);
        assert!(!report.combined_coverage_theorem_emitted);
    }

    #[test]
    fn one_trace_has_disjoint_qualified_unknown_blocked_unmeasured_partition() {
        let mut qualified = row(1);
        qualified.operational.state = OperationalAcquisitionStateV1::Measured;
        qualified.operational.measured_at_unix_seconds = Some(1010);
        qualified.operational.measurement_system_sha256 = Some(digest('3'));
        qualified.operational.physical_measurement_cost = Some(2.0);
        qualified.k1 = K1QualificationEvidenceV1 {
            state: K1QualificationStateV1::Eligible,
            label_id: Some("label-1".into()),
            label_revision: Some(1),
            label_payload_sha256: Some(digest('4')),
            qualification_report_sha256: Some(digest('5')),
            measurement_event_id: Some("event-1".into()),
            corrected_value: Some(1.25),
            standard_uncertainty_upper_bound: Some(0.05),
            label_available_at_unix_seconds: Some(1011),
            report_available_at_unix_seconds: Some(1012),
            calibration_certificate_sha256: Some(digest('6')),
            requalification_report_sha256: Some(digest('7')),
            qualified_loss: Some(0.1),
            no_promotion_reason: None,
        };
        let mut unknown = qualified.clone();
        unknown.round_id = 2;
        unknown.sample_id = "sample-2".into();
        unknown.measurement_event_id = "event-2".into();
        unknown.prediction_receipt_sha256 = digest('8');
        unknown.operational.measured_at_unix_seconds = Some(1011);
        unknown.k1.label_id = Some("label-2".into());
        unknown.k1.measurement_event_id = Some("event-2".into());
        unknown.k1.state = K1QualificationStateV1::Unknown;
        unknown.k1.qualified_loss = None;
        unknown.k1.corrected_value = None;
        unknown.k1.standard_uncertainty_upper_bound = None;
        unknown.k1.calibration_certificate_sha256 = None;
        unknown.k1.requalification_report_sha256 = None;
        let mut blocked = unknown.clone();
        blocked.round_id = 3;
        blocked.sample_id = "sample-3".into();
        blocked.measurement_event_id = "event-3".into();
        blocked.prediction_receipt_sha256 = digest('9');
        blocked.k1.label_id = Some("label-3".into());
        blocked.k1.measurement_event_id = Some("event-3".into());
        blocked.k1.state = K1QualificationStateV1::Blocked;
        let unmeasured = row(4);

        let report = evaluate_observation_process_trace_v1(&input(vec![
            qualified, unknown, blocked, unmeasured,
        ]))
        .unwrap();
        assert_eq!(report.k1_eligible_count, 1);
        assert_eq!(report.k1_unknown_count, 1);
        assert_eq!(report.k1_blocked_count, 1);
        assert_eq!(report.selection_risk_report.missing_count, 3);
        assert_eq!(report.selection_risk_report.qualified_subset_risk, Some(0.1));
        assert_eq!(report.selection_risk_report.partial_identification_upper, 0.775);
    }

    #[test]
    fn invalid_label_on_unselected_round_is_rejected() {
        let mut unselected = row(1);
        unselected.k1.state = K1QualificationStateV1::Blocked;
        assert!(evaluate_observation_process_trace_v1(&input(vec![unselected])).is_err());
    }

    #[test]
    fn audit_result_is_independent_offline_population_observation() {
        let mut unselected = row(1);
        unselected.independent_audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::AuditMeasured,
            retention: AuditRetentionStateV1::OfflineArchive,
            inclusion_probability: Some(0.5),
            audit_selected_at_unix_seconds: Some(1010),
            audit_measured_at_unix_seconds: Some(1012),
            audit_measurement_cost: Some(8.0),
            audit_measurement_receipt_sha256: Some(digest('a')),
            audit_loss: Some(1.0),
            unavailable_reason: None,
        };
        let mut design = SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('b'),
            randomization_provenance_sha256: digest('c'),
            reference_method_sha256: digest('d'),
            minimum_inclusion_probability: 0.5,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        };
        design.minimum_inclusion_probability = 0.5;
        let mut data = input(vec![unselected]);
        data.audit_design = Some(design);
        let report = evaluate_observation_process_trace_v1(&data).unwrap();
        assert_eq!(report.audit_measured_count, 1);
        assert!(report.selection_risk_report.independent_audit_estimate.is_some());
        assert_eq!(report.selection_risk_report.population_point_estimate, Some(1.0));
    }

    #[test]
    fn audit_cannot_sample_an_operationally_qualified_round() {
        let mut measured = row(1);
        measured.operational.state = OperationalAcquisitionStateV1::Measured;
        measured.operational.design_sha256 = Some(digest('d'));
        measured.operational.measured_at_unix_seconds = Some(1010);
        measured.operational.measurement_system_sha256 = Some(digest('e'));
        measured.operational.physical_measurement_cost = Some(1.0);
        measured.k1 = K1QualificationEvidenceV1 {
            state: K1QualificationStateV1::Eligible,
            label_id: Some("label-1".into()),
            label_revision: Some(1),
            label_payload_sha256: Some(digest('1')),
            qualification_report_sha256: Some(digest('2')),
            measurement_event_id: Some("event-1".into()),
            corrected_value: Some(1.0),
            standard_uncertainty_upper_bound: Some(0.1),
            label_available_at_unix_seconds: Some(1011),
            report_available_at_unix_seconds: Some(1012),
            calibration_certificate_sha256: Some(digest('3')),
            requalification_report_sha256: Some(digest('4')),
            qualified_loss: Some(0.2),
            no_promotion_reason: None,
        };
        measured.independent_audit = IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::AuditMeasured,
            retention: AuditRetentionStateV1::OfflineArchive,
            inclusion_probability: Some(0.5),
            audit_selected_at_unix_seconds: Some(1010),
            audit_measured_at_unix_seconds: Some(1012),
            audit_measurement_cost: Some(5.0),
            audit_measurement_receipt_sha256: Some(digest('5')),
            audit_loss: Some(0.2),
            unavailable_reason: None,
        };
        let mut data = input(vec![measured]);
        data.audit_design = Some(SelectionAuditDesignV1 {
            schema: super::super::selection_risk_bounds::SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('6'),
            randomization_provenance_sha256: digest('7'),
            reference_method_sha256: digest('8'),
            minimum_inclusion_probability: 0.5,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        });
        assert!(evaluate_observation_process_trace_v1(&data).is_err());
    }

    #[test]
    fn qualified_feedback_requires_qualified_k1_and_update_receipt() {
        let mut trace = row(1);
        trace.controller.tracking_state = ControllerTrackingStateV1::ReceivedQualified;
        trace.controller.controller_state_changed = true;
        trace.controller.prediction_receipt_sha256 = Some(digest('a'));
        trace.controller.feedback_receipt_sha256 = Some(digest('b'));
        trace.controller.update_receipt_sha256 = Some(digest('c'));
        trace.controller.policy_sha256 = Some(digest('d'));
        let mut data = input(vec![trace]);
        data.controller_kind = ObservationControllerKindV1::TauDaci;
        data.controller_policy_sha256 = Some(digest('d'));
        data.controller_delay_steps = Some(2);
        assert!(evaluate_observation_process_trace_v1(&data).is_err());
    }

    #[test]
    fn taudaci_feedback_is_bound_to_fixed_delay_and_cannot_be_mixed_with_ca_fields() {
        let mut trace = row(1);
        trace.operational.state = OperationalAcquisitionStateV1::Measured;
        trace.operational.design_sha256 = Some(digest('1'));
        trace.operational.measured_at_unix_seconds = Some(1010);
        trace.operational.measurement_system_sha256 = Some(digest('2'));
        trace.operational.physical_measurement_cost = Some(1.0);
        trace.k1 = K1QualificationEvidenceV1 {
            state: K1QualificationStateV1::Eligible,
            label_id: Some("label-1".into()),
            label_revision: Some(1),
            label_payload_sha256: Some(digest('3')),
            qualification_report_sha256: Some(digest('4')),
            measurement_event_id: Some("event-1".into()),
            corrected_value: Some(1.0),
            standard_uncertainty_upper_bound: Some(0.1),
            label_available_at_unix_seconds: Some(1011),
            report_available_at_unix_seconds: Some(1012),
            calibration_certificate_sha256: Some(digest('5')),
            requalification_report_sha256: Some(digest('6')),
            qualified_loss: Some(0.2),
            no_promotion_reason: None,
        };
        trace.controller.tracking_state = ControllerTrackingStateV1::ReceivedQualified;
        trace.controller.policy_sha256 = Some(digest('7'));
        trace.controller.prediction_receipt_sha256 = Some(digest('8'));
        trace.controller.feedback_receipt_sha256 = Some(digest('9'));
        trace.controller.update_receipt_sha256 = Some(digest('a'));
        trace.controller.controller_state_changed = true;
        trace.controller.actual_feedback_delay_steps = Some(2);
        trace.controller.feedback_round = Some(3);
        trace.controller.theorem_assumptions_intact = true;
        let mut data = input(vec![trace]);
        data.controller_kind = ObservationControllerKindV1::TauDaci;
        data.controller_policy_sha256 = Some(digest('7'));
        data.controller_delay_steps = Some(2);
        let report = evaluate_observation_process_trace_v1(&data).unwrap();
        assert_eq!(report.controller_received_qualified_count, 1);
        assert_eq!(report.controller_state_update_count, 1);
    }

    #[test]
    fn caocrc_capacity_is_distinct_from_physical_measurement_budget() {
        let mut trace = row(1);
        trace.controller.tracking_state = ControllerTrackingStateV1::Pending;
        trace.controller.policy_sha256 = Some(digest('a'));
        trace.controller.delay_contract_sha256 = Some(digest('b'));
        trace.controller.tracking_capacity = Some(2);
        trace.controller.tracking_occupancy_after = Some(3);
        trace.controller.proxy_uniform_draw = Some(0.5);
        trace.controller.proxy_delay_steps = Some(2);
        trace.controller.proxy_survival_probability = Some(0.5);
        trace.controller.prediction_receipt_sha256 = Some(digest('c'));
        let mut data = input(vec![trace]);
        data.controller_kind = ObservationControllerKindV1::CaOcrc;
        data.controller_policy_sha256 = Some(digest('a'));
        data.controller_delay_contract_sha256 = Some(digest('b'));
        data.controller_delay_steps = Some(2);
        data.controller_tracking_capacity = Some(2);
        data.operational_measurement_budget = 1;
        assert!(evaluate_observation_process_trace_v1(&data).is_err());
    }

    #[test]
    fn permutation_preserves_trace_and_report_digest() {
        let first = input(vec![row(1), row(2), row(3)]);
        let first_report = evaluate_observation_process_trace_v1(&first).unwrap();
        let mut second = input(vec![row(3), row(1), row(2)]);
        second.trace_id = first.trace_id.clone();
        let second_report = evaluate_observation_process_trace_v1(&second).unwrap();
        assert_eq!(first_report.input_sha256, second_report.input_sha256);
        assert_eq!(first_report.report_sha256, second_report.report_sha256);
    }
}
