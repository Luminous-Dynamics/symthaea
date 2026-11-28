// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared paired replay trace for metrology selection, K1 qualification,
//! controller feedback tracking, and independent population audit.
//!
//! This module does not merge or modify tau-ACI, CA-OCRC, or selection-risk
//! algorithms. It validates that their independent receipts refer to one
//! common population and acquisition stream, then computes the selection
//! adjusted risk report for each arm.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::ca_ocrc_bounded::{
    CA_OCRC_ROUND_INPUT_SCHEMA_V1, CaOcrcFeedbackDispositionV1, CaOcrcLabelFeedbackV1,
    CaOcrcPolicyV1, CaOcrcRoundInputV1, CaOcrcRoundReportV1,
};
use super::delayed_aci::{
    AciDegenerateResolutionReceiptV1, AciGuaranteeStateV1, AciIntervalKindV1,
    AciMissingFeedbackRecordV1, AciUpdateDispositionV1, DelayedAciLabelDeliveryV1,
    DelayedAciPolicyV1, DelayedAciPredictionReceiptV1, DelayedAciUpdateReceiptV1,
};
use super::selection_risk_bounds::{
    PopulationOutcomeStateV1, SelectionRiskInputV1, SelectionRiskReportV1,
    evaluate_selection_adjusted_population_risk_v1,
};

pub const JOINT_RISK_TRACE_SCHEMA_V1: &str =
    "luminous.semi-bench.joint-risk-trace.v1";
pub const JOINT_RISK_TRACE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.joint-risk-trace-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointBenchmarkModeV1 {
    PairedSharedAcquisition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointControllerMethodV1 {
    FixedDelayTauAci,
    BoundedDelayCaOcrc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointTraceStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointPopulationRowV1 {
    pub round_id: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub context_key: String,
    pub physical_measurement_selected: bool,
    pub physical_metrology_cost_units: f64,
    pub outcome_state: PopulationOutcomeStateV1,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub label_payload_sha256: Option<String>,
    pub k1_report_sha256: Option<String>,
    pub label_value: Option<f64>,
    pub label_available_at_unix_seconds: Option<i64>,
    pub eligibility_report_available_at_unix_seconds: Option<i64>,
    pub label_delivered_at_unix_seconds: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointTauAciArmEvidenceV1 {
    pub policy: DelayedAciPolicyV1,
    pub interval_method_sha256: String,
    pub prediction_stream_closed: bool,
    pub predictions: Vec<DelayedAciPredictionReceiptV1>,
    pub deliveries: Vec<DelayedAciLabelDeliveryV1>,
    pub updates: Vec<DelayedAciUpdateReceiptV1>,
    pub missing_feedback: Vec<AciMissingFeedbackRecordV1>,
    pub degenerate_resolutions: Vec<AciDegenerateResolutionReceiptV1>,
    pub guarantee_state: AciGuaranteeStateV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointCaOcrcArmEvidenceV1 {
    pub policy: CaOcrcPolicyV1,
    pub round_inputs: Vec<CaOcrcRoundInputV1>,
    pub round_reports: Vec<CaOcrcRoundReportV1>,
    pub feedback_horizon_closed: bool,
    pub unresolved_tracking_rounds: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "evidence", rename_all = "snake_case")]
pub enum JointControllerEvidenceV1 {
    FixedDelayTauAci(JointTauAciArmEvidenceV1),
    BoundedDelayCaOcrc(JointCaOcrcArmEvidenceV1),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointBenchmarkArmV1 {
    pub arm_id: String,
    pub method: JointControllerMethodV1,
    pub global_model_artifact_sha256: String,
    pub interval_method_sha256: String,
    pub controller_policy_sha256: String,
    pub acquisition_policy_sha256: String,
    pub physical_metrology_budget_units: f64,
    pub selection_risk_input: SelectionRiskInputV1,
    pub controller_evidence: JointControllerEvidenceV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointRiskTraceInputV1 {
    pub schema: String,
    pub trace_id: String,
    pub mode: JointBenchmarkModeV1,
    pub population_manifest_sha256: String,
    pub source_ledger_sha256: String,
    pub population: Vec<JointPopulationRowV1>,
    pub arms: Vec<JointBenchmarkArmV1>,
    pub trace_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointArmRiskSummaryV1 {
    pub arm_id: String,
    pub method: JointControllerMethodV1,
    pub qualified_subset_risk: Option<f64>,
    pub partial_identification_lower: f64,
    pub partial_identification_upper: f64,
    pub identification_interval_width: f64,
    pub independent_audit_population_estimate: Option<f64>,
    pub independent_audit_standard_error: Option<f64>,
    pub independent_audit_hoeffding_lower: Option<f64>,
    pub independent_audit_hoeffding_upper: Option<f64>,
    pub selection_risk_report_sha256: String,
    pub observed_miscoverage_count: Option<usize>,
    pub observed_miscoverage_count_denominator: Option<usize>,
    pub aci_held_unknown_count: Option<usize>,
    pub aci_blocked_count: Option<usize>,
    pub aci_rejected_count: Option<usize>,
    pub aci_missing_feedback_count: Option<usize>,
    pub aci_degenerate_resolution_count: Option<usize>,
    pub ca_ocrc_weighted_feedback_sum: Option<f64>,
    pub ca_ocrc_updated_feedback_count: Option<usize>,
    pub ca_ocrc_held_unknown_count: Option<usize>,
    pub ca_ocrc_blocked_count: Option<usize>,
    pub ca_ocrc_not_tracked_count: Option<usize>,
    pub ca_ocrc_rejected_count: Option<usize>,
    pub ca_ocrc_duplicate_count: Option<usize>,
    pub ca_ocrc_max_tracking_occupancy: Option<usize>,
    pub ca_ocrc_tracking_capacity: Option<usize>,
    pub ca_ocrc_theorem_assumptions_intact: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointRiskTraceReportV1 {
    pub schema: String,
    pub trace_id: String,
    pub trace_sha256: String,
    pub status: JointTraceStatusV1,
    pub reason_codes: Vec<String>,
    pub population_count: usize,
    pub physical_measurement_count: usize,
    pub physical_metrology_cost_units: f64,
    pub qualified_count: usize,
    pub unknown_count: usize,
    pub blocked_count: usize,
    pub unmeasured_count: usize,
    pub independent_audit_count: usize,
    pub shared_acquisition_policy_sha256: String,
    pub arm_summaries: Vec<JointArmRiskSummaryV1>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct JointRiskTraceErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), JointRiskTraceErrorV1> {
    if value.trim().is_empty() {
        return Err(JointRiskTraceErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, JointRiskTraceErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        JointRiskTraceErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn canonicalize_selection_input(input: &mut SelectionRiskInputV1) {
    input.population_round_ids.sort_unstable();
    input.qualified_round_ids.sort_unstable();
    input.unknown_round_ids.sort_unstable();
    input.blocked_round_ids.sort_unstable();
    input.unmeasured_round_ids.sort_unstable();
    input.qualified_losses.sort_by_key(|entry| entry.round_id);
    input
        .independent_audit_losses
        .sort_by_key(|entry| entry.round_id);
}

fn canonicalize_trace(input: &JointRiskTraceInputV1) -> JointRiskTraceInputV1 {
    let mut value = input.clone();
    value.trace_sha256.clear();
    value.population.sort_by_key(|row| row.round_id);
    value.arms.sort_by(|left, right| left.arm_id.cmp(&right.arm_id));
    for arm in &mut value.arms {
        canonicalize_selection_input(&mut arm.selection_risk_input);
        match &mut arm.controller_evidence {
            JointControllerEvidenceV1::FixedDelayTauAci(evidence) => {
                evidence.predictions.sort_by_key(|entry| entry.round);
                evidence.deliveries.sort_by(|left, right| {
                    (left.delivered_round, left.sample_id.as_str(), left.label_id.as_str(), left.label_revision)
                        .cmp(&(right.delivered_round, right.sample_id.as_str(), right.label_id.as_str(), right.label_revision))
                });
                evidence.missing_feedback.sort_by_key(|entry| entry.issued_round);
                evidence.updates.sort_by(|left, right| {
                    (left.sample_id.as_str(), left.label_id.as_str(), left.label_revision)
                        .cmp(&(right.sample_id.as_str(), right.label_id.as_str(), right.label_revision))
                });
                evidence
                    .degenerate_resolutions
                    .sort_by_key(|entry| entry.issued_round);
            }
            JointControllerEvidenceV1::BoundedDelayCaOcrc(evidence) => {
                evidence.round_inputs.sort_by_key(|entry| entry.prediction.round);
                for round_input in &mut evidence.round_inputs {
                    round_input.feedbacks.sort_by(|left, right| {
                        (left.sample_id.as_str(), left.label_id.as_str(), left.label_revision)
                            .cmp(&(right.sample_id.as_str(), right.label_id.as_str(), right.label_revision))
                    });
                }
                evidence.round_reports.sort_by_key(|entry| entry.round);
                for report in &mut evidence.round_reports {
                    report.feedback_results.sort_by(|left, right| {
                        (left.originating_round, left.label_id.as_str(), left.label_revision)
                            .cmp(&(right.originating_round, right.label_id.as_str(), right.label_revision))
                    });
                    report.evicted_rounds.sort_unstable();
                }
            }
        }
    }
    value
}

pub fn seal_joint_risk_trace_v1(
    input: &JointRiskTraceInputV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let canonical = canonicalize_trace(input);
    json_sha256(&canonical)
}

fn validate_population(
    input: &JointRiskTraceInputV1,
) -> Result<BTreeMap<u64, &JointPopulationRowV1>, JointRiskTraceErrorV1> {
    if input.population.is_empty() {
        return Err(JointRiskTraceErrorV1(
            "joint trace population must be non-empty".into(),
        ));
    }
    let mut rows = BTreeMap::new();
    let mut sample_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    for row in &input.population {
        if row.round_id == 0 {
            return Err(JointRiskTraceErrorV1(
                "population round IDs must be nonzero".into(),
            ));
        }
        require_text(&row.sample_id, "population.sample_id")?;
        require_text(&row.measurement_event_id, "population.measurement_event_id")?;
        require_text(&row.context_key, "population.context_key")?;
        if !row.physical_metrology_cost_units.is_finite()
            || row.physical_metrology_cost_units < 0.0
            || (row.physical_measurement_selected && row.physical_metrology_cost_units <= 0.0)
            || (!row.physical_measurement_selected && row.physical_metrology_cost_units != 0.0)
        {
            return Err(JointRiskTraceErrorV1(format!(
                "round {} has invalid physical-metrology cost/selection",
                row.round_id
            )));
        }
        if !sample_ids.insert(row.sample_id.as_str())
            || !event_ids.insert(row.measurement_event_id.as_str())
            || rows.insert(row.round_id, row).is_some()
        {
            return Err(JointRiskTraceErrorV1(
                "population contains duplicate round, sample, or measurement-event identity".into(),
            ));
        }
        match row.outcome_state {
            PopulationOutcomeStateV1::Unmeasured => {
                if row.physical_measurement_selected
                    || row.label_id.is_some()
                    || row.label_revision.is_some()
                    || row.label_payload_sha256.is_some()
                    || row.k1_report_sha256.is_some()
                    || row.label_value.is_some()
                    || row.label_available_at_unix_seconds.is_some()
                    || row.eligibility_report_available_at_unix_seconds.is_some()
                    || row.label_delivered_at_unix_seconds.is_some()
                {
                    return Err(JointRiskTraceErrorV1(format!(
                        "unmeasured round {} contains physical label evidence",
                        row.round_id
                    )));
                }
            }
            PopulationOutcomeStateV1::Qualified
            | PopulationOutcomeStateV1::Unknown
            | PopulationOutcomeStateV1::Blocked => {
                if !row.physical_measurement_selected {
                    return Err(JointRiskTraceErrorV1(format!(
                        "round {} has a label outcome without physical measurement selection",
                        row.round_id
                    )));
                }
                require_text(
                    row.label_id.as_deref().unwrap_or_default(),
                    "population.label_id",
                )?;
                if row.label_revision.unwrap_or(0) == 0
                    || !row
                        .label_payload_sha256
                        .as_deref()
                        .is_some_and(valid_sha256)
                    || !row.k1_report_sha256.as_deref().is_some_and(valid_sha256)
                    || row.label_available_at_unix_seconds.is_none()
                    || row.eligibility_report_available_at_unix_seconds.is_none()
                    || row.label_delivered_at_unix_seconds.is_none()
                {
                    return Err(JointRiskTraceErrorV1(format!(
                        "measured round {} lacks a revision or valid label/K1 digest",
                        row.round_id
                    )));
                }
                let label_available = row.label_available_at_unix_seconds.unwrap();
                let report_available = row.eligibility_report_available_at_unix_seconds.unwrap();
                let label_delivered = row.label_delivered_at_unix_seconds.unwrap();
                if label_available > report_available || report_available > label_delivered {
                    return Err(JointRiskTraceErrorV1(format!(
                        "measured round {} has invalid label/report/delivery chronology",
                        row.round_id
                    )));
                }
                if row.outcome_state == PopulationOutcomeStateV1::Qualified {
                    let value = row.label_value.ok_or_else(|| {
                        JointRiskTraceErrorV1(format!(
                            "qualified round {} has no numeric label value",
                            row.round_id
                        ))
                    })?;
                    if !value.is_finite() {
                        return Err(JointRiskTraceErrorV1(format!(
                            "qualified round {} has a non-finite label value",
                            row.round_id
                        )));
                    }
                    if row.label_available_at_unix_seconds.is_none()
                        || row.label_delivered_at_unix_seconds.is_none()
                    {
                        return Err(JointRiskTraceErrorV1(format!(
                            "qualified round {} lacks label availability/delivery time",
                            row.round_id
                        )));
                    }
                }
            }
        }
    }
    Ok(rows)
}

fn expected_partitions(
    population: &BTreeMap<u64, &JointPopulationRowV1>,
) -> (
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
) {
    let all: Vec<u64> = population.keys().copied().collect();
    let mut qualified = Vec::new();
    let mut unknown = Vec::new();
    let mut blocked = Vec::new();
    let mut unmeasured = Vec::new();
    for (round, row) in population {
        match row.outcome_state {
            PopulationOutcomeStateV1::Qualified => qualified.push(*round),
            PopulationOutcomeStateV1::Unknown => unknown.push(*round),
            PopulationOutcomeStateV1::Blocked => blocked.push(*round),
            PopulationOutcomeStateV1::Unmeasured => unmeasured.push(*round),
        }
    }
    (all, qualified, unknown, blocked, unmeasured)
}

fn validate_selection_risk_input(
    arm: &JointBenchmarkArmV1,
    input: &JointRiskTraceInputV1,
    population: &BTreeMap<u64, &JointPopulationRowV1>,
) -> Result<SelectionRiskReportV1, JointRiskTraceErrorV1> {
    let selection = &arm.selection_risk_input;
    if selection.population_manifest_sha256 != input.population_manifest_sha256
        || selection.source_ledger_sha256 != input.source_ledger_sha256
    {
        return Err(JointRiskTraceErrorV1(format!(
            "arm {} selection-risk input references another population/source ledger",
            arm.arm_id
        )));
    }
    let (all, qualified, unknown, blocked, unmeasured) = expected_partitions(population);
    let mut canonical = selection.clone();
    canonicalize_selection_input(&mut canonical);
    if canonical.population_round_ids != all
        || canonical.qualified_round_ids != qualified
        || canonical.unknown_round_ids != unknown
        || canonical.blocked_round_ids != blocked
        || canonical.unmeasured_round_ids != unmeasured
    {
        return Err(JointRiskTraceErrorV1(format!(
            "arm {} selection-risk partitions do not match shared physical/K1 ledger",
            arm.arm_id
        )));
    }
    let loss_rounds: BTreeSet<u64> = canonical
        .qualified_losses
        .iter()
        .map(|loss| loss.round_id)
        .collect();
    if loss_rounds.len() != canonical.qualified_losses.len()
        || loss_rounds != qualified.iter().copied().collect()
    {
        return Err(JointRiskTraceErrorV1(format!(
            "arm {} must supply exactly one qualified loss per K1-qualified round",
            arm.arm_id
        )));
    }
    let Some(audit) = canonical.audit_design.as_ref() else {
        return Err(JointRiskTraceErrorV1(format!(
            "arm {} has no independent audit design",
            arm.arm_id
        )));
    };
    if !audit.independent_of_operational_qualification
        || !valid_sha256(&audit.design_sha256)
        || !valid_sha256(&audit.randomization_provenance_sha256)
        || !valid_sha256(&audit.reference_method_sha256)
        || !audit.minimum_inclusion_probability.is_finite()
        || audit.minimum_inclusion_probability <= 0.0
        || audit.minimum_inclusion_probability > 1.0
        || !audit.confidence_delta.is_finite()
        || audit.confidence_delta <= 0.0
        || audit.confidence_delta >= 1.0
    {
        return Err(JointRiskTraceErrorV1(format!(
            "arm {} audit design is not independent of operational qualification",
            arm.arm_id
        )));
    }
    for audit_loss in &canonical.independent_audit_losses {
        if !population.contains_key(&audit_loss.round_id)
            || !audit_loss.inclusion_probability.is_finite()
            || audit_loss.inclusion_probability < audit.minimum_inclusion_probability
            || audit_loss.inclusion_probability > 1.0
            || !valid_sha256(&audit_loss.audit_measurement_receipt_sha256)
        {
            return Err(JointRiskTraceErrorV1(format!(
                "arm {} has invalid independent audit row {}",
                arm.arm_id, audit_loss.round_id
            )));
        }
    }
    evaluate_selection_adjusted_population_risk_v1(&canonical)
        .map_err(|error| JointRiskTraceErrorV1(error.to_string()))
}

fn eligibility_report_digest(
    report: &super::label_consumption_guard::LabelConsumptionEligibilityReportV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn expected_label_eligibility(
    state: PopulationOutcomeStateV1,
) -> Option<super::label_consumption_guard::LabelConsumptionEligibilityV1> {
    match state {
        PopulationOutcomeStateV1::Qualified => {
            Some(super::label_consumption_guard::LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy)
        }
        PopulationOutcomeStateV1::Unknown => {
            Some(super::label_consumption_guard::LabelConsumptionEligibilityV1::Unknown)
        }
        PopulationOutcomeStateV1::Blocked => {
            Some(super::label_consumption_guard::LabelConsumptionEligibilityV1::Blocked)
        }
        PopulationOutcomeStateV1::Unmeasured => None,
    }
}

fn validate_aci_delivery(
    delivery: &DelayedAciLabelDeliveryV1,
    prediction: &DelayedAciPredictionReceiptV1,
    row: &JointPopulationRowV1,
) -> Result<(), JointRiskTraceErrorV1> {
    let report = &delivery.eligibility_report;
    let expected_eligibility = expected_label_eligibility(row.outcome_state);
    if expected_eligibility.is_none()
        || delivery.sample_id != row.sample_id
        || delivery.measurement_event_id != row.measurement_event_id
        || delivery.label_id != row.label_id.as_deref().unwrap_or_default()
        || Some(delivery.label_revision) != row.label_revision
        || Some(delivery.label_payload_sha256.as_str()) != row.label_payload_sha256.as_deref()
        || Some(report.report_sha256.as_str()) != row.k1_report_sha256.as_deref()
        || row.label_value.is_some_and(|value| delivery.label_value != value)
        || Some(delivery.label_available_at_unix_seconds) != row.label_available_at_unix_seconds
        || Some(delivery.eligibility_report_available_at_unix_seconds)
            != row.eligibility_report_available_at_unix_seconds
        || Some(delivery.delivered_at_unix_seconds) != row.label_delivered_at_unix_seconds
        || report.label_id != delivery.label_id
        || report.label_revision != delivery.label_revision
        || report.measurement_event_id != delivery.measurement_event_id
        || report.label_sha256 != delivery.label_payload_sha256
        || report.qualified_corrected_value != delivery.label_value
        || report.frontier != super::label_consumption_guard::LabelEvidenceFrontierV1::CurrentAtK1
        || Some(report.eligibility) != expected_eligibility
        || report.downstream_consumption_allowed
            != (row.outcome_state == PopulationOutcomeStateV1::Qualified)
        || eligibility_report_digest(report)? != report.report_sha256
        || delivery.delivered_round != prediction.expected_delivery_round
        || delivery.clock_domain_id != prediction.clock_domain_id
        || delivery.idempotency_key != format!("{}:{}", delivery.label_id, delivery.label_revision)
        || delivery.delivered_at_unix_seconds < prediction.prediction_time_unix_seconds
        || delivery.delivered_at_unix_seconds < delivery.label_available_at_unix_seconds
        || delivery.delivered_at_unix_seconds < delivery.eligibility_report_available_at_unix_seconds
        || delivery.eligibility_report_available_at_unix_seconds < delivery.label_available_at_unix_seconds
        || delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds
    {
        return Err(JointRiskTraceErrorV1(format!(
            "ACI delivery for round {} does not match the shared K1 label/event/availability ledger",
            row.round_id
        )));
    }
    Ok(())
}

fn validate_ca_feedback_input(
    feedback: &CaOcrcLabelFeedbackV1,
    current_round: u64,
    row: &JointPopulationRowV1,
) -> Result<(), JointRiskTraceErrorV1> {
    let report = &feedback.eligibility_report;
    if feedback.feedback_round != current_round
        || feedback.sample_id != row.sample_id
        || feedback.measurement_event_id != row.measurement_event_id
        || feedback.label_id != row.label_id.as_deref().unwrap_or_default()
        || Some(feedback.label_revision) != row.label_revision
        || Some(feedback.label_payload_sha256.as_str()) != row.label_payload_sha256.as_deref()
        || Some(report.report_sha256.as_str()) != row.k1_report_sha256.as_deref()
        || row.label_value.is_some_and(|value| feedback.label_value != value)
        || Some(feedback.label_available_at_unix_seconds) != row.label_available_at_unix_seconds
        || Some(feedback.eligibility_report_available_at_unix_seconds)
            != row.eligibility_report_available_at_unix_seconds
        || Some(feedback.delivered_at_unix_seconds) != row.label_delivered_at_unix_seconds
        || feedback.delivered_at_unix_seconds < feedback.label_available_at_unix_seconds
        || feedback.eligibility_report_available_at_unix_seconds
            < feedback.label_available_at_unix_seconds
        || feedback.delivered_at_unix_seconds
            < feedback.eligibility_report_available_at_unix_seconds
        || report.label_id != feedback.label_id
        || report.label_revision != feedback.label_revision
        || report.measurement_event_id != feedback.measurement_event_id
        || report.label_sha256 != feedback.label_payload_sha256
        || report.qualified_corrected_value != feedback.label_value
        || eligibility_report_digest(report)? != report.report_sha256
    {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC feedback input for round {current_round} does not match shared label ledger"
        )));
    }
    let Some(expected_eligibility) = expected_label_eligibility(row.outcome_state) else {
        return Err(JointRiskTraceErrorV1(
            "CA-OCRC feedback input exists for an unmeasured row".into(),
        ));
    };
    if report.eligibility != expected_eligibility
        || report.frontier
            != super::label_consumption_guard::LabelEvidenceFrontierV1::CurrentAtK1
        || report.downstream_consumption_allowed
            != (row.outcome_state == PopulationOutcomeStateV1::Qualified)
    {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC feedback K1 state disagrees with shared row {}",
            row.round_id
        )));
    }
    Ok(())
}

fn aci_prediction_digest(
    receipt: &DelayedAciPredictionReceiptV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn aci_update_digest(
    receipt: &DelayedAciUpdateReceiptV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn aci_degenerate_digest(
    receipt: &AciDegenerateResolutionReceiptV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn ca_prediction_digest(
    receipt: &super::ca_ocrc_bounded::CaOcrcPredictionReceiptV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn ca_feedback_digest(
    receipt: &super::ca_ocrc_bounded::CaOcrcFeedbackResultV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = receipt.clone();
    canonical.feedback_receipt_sha256.clear();
    json_sha256(&canonical)
}

fn ca_round_report_digest(
    report: &CaOcrcRoundReportV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_aci_arm(
    arm: &JointBenchmarkArmV1,
    evidence: &JointTauAciArmEvidenceV1,
    population: &BTreeMap<u64, &JointPopulationRowV1>,
) -> Result<Option<(usize, usize, usize, usize, usize)>, JointRiskTraceErrorV1> {
    if !evidence.prediction_stream_closed {
        return Err(JointRiskTraceErrorV1(format!(
            "tau-ACI arm {} prediction stream is not explicitly closed",
            arm.arm_id
        )));
    }
    if evidence.predictions.len() != population.len() {
        return Err(JointRiskTraceErrorV1(format!(
            "tau-ACI arm {} must issue exactly one prediction per population row",
            arm.arm_id
        )));
    }
    if json_sha256(&evidence.policy)? != arm.controller_policy_sha256
        || evidence.interval_method_sha256 != arm.interval_method_sha256
        || evidence.policy.delay_rounds == 0
    {
        return Err(JointRiskTraceErrorV1(format!(
            "tau-ACI arm {} policy digest/delay does not match the declared arm",
            arm.arm_id
        )));
    }
    let mut predictions = BTreeMap::new();
    for prediction in &evidence.predictions {
        let Some(row) = population.get(&prediction.round) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} prediction round {} is outside population",
                arm.arm_id, prediction.round
            )));
        };
        if prediction.delay_rounds == 0
            || prediction.sample_id != row.sample_id
            || prediction.measurement_event_id != row.measurement_event_id
            || prediction.model_artifact_sha256 != arm.global_model_artifact_sha256
            || prediction.policy_sha256 != arm.controller_policy_sha256
            || prediction.delay_rounds != evidence.policy.delay_rounds
            || prediction.target_miscoverage != evidence.policy.target_miscoverage
            || prediction.learning_rate != evidence.policy.learning_rate
            || prediction.phase != prediction.round % prediction.delay_rounds
            || prediction.expected_delivery_round
                != prediction.round.checked_add(prediction.delay_rounds).unwrap_or(u64::MAX)
            || aci_prediction_digest(prediction)? != prediction.receipt_sha256
            || predictions.insert(prediction.sample_id.as_str(), prediction).is_some()
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} prediction receipt does not bind the shared row/policy",
                arm.arm_id
            )));
        }
    }

    let mut deliveries_by_sample = BTreeMap::new();
    for delivery in &evidence.deliveries {
        let Some(prediction) = predictions.get(delivery.sample_id.as_str()) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} delivery has no prediction",
                arm.arm_id
            )));
        };
        let Some(row) = population.values().find(|row| row.sample_id == delivery.sample_id) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} delivery references unknown sample",
                arm.arm_id
            )));
        };
        validate_aci_delivery(delivery, prediction, row)?;
        if deliveries_by_sample.insert(delivery.sample_id.as_str(), delivery).is_some() {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} has duplicate delivered label for sample {}",
                arm.arm_id, delivery.sample_id
            )));
        }
    }
    let mut missing_by_sample = BTreeMap::new();
    for missing in &evidence.missing_feedback {
        let Some(prediction) = predictions.get(missing.sample_id.as_str()) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} missing-feedback record has no prediction",
                arm.arm_id
            )));
        };
        if missing.measurement_event_id != prediction.measurement_event_id
            || missing.issued_round != prediction.round
            || missing.expected_delivery_round != prediction.expected_delivery_round
            || missing_by_sample.insert(missing.sample_id.as_str(), missing).is_some()
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} missing-feedback record does not match its due round",
                arm.arm_id
            )));
        }
    }

    let mut update_receipts = BTreeSet::new();
    let mut update_samples = BTreeSet::new();
    let mut observed_miscoverage = 0_usize;
    let mut observed_count = 0_usize;
    let mut held_unknown_count = 0_usize;
    let mut blocked_count = 0_usize;
    let mut rejected_count = 0_usize;
    for update in &evidence.updates {
        if !update_receipts.insert(update.receipt_sha256.as_str())
            || !update_samples.insert(update.sample_id.as_str())
            || aci_update_digest(update)? != update.receipt_sha256
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} repeats or tampers with an update receipt",
                arm.arm_id
            )));
        }
        let Some(row) = population.values().find(|row| row.sample_id == update.sample_id) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} update references unknown sample {}",
                arm.arm_id, update.sample_id
            )));
        };
        let prediction = predictions.get(update.sample_id.as_str()).expect("prediction exists");
        if update.measurement_event_id != row.measurement_event_id
            || update.prediction_receipt_sha256.as_deref() != Some(prediction.receipt_sha256.as_str())
            || update.label_id.is_empty()
            || update.label_revision == 0
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} update receipt identity differs from its population row",
                arm.arm_id
            )));
        }
        let mutates = update.disposition == AciUpdateDispositionV1::Updated;
        if update.disposition == AciUpdateDispositionV1::PendingPredictionMissing
            || update.phase_state_mutated != mutates
            || update.phase != Some(prediction.phase)
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} update mutation/phase flag conflicts with receipt",
                arm.arm_id
            )));
        }
        let delivery = deliveries_by_sample.get(update.sample_id.as_str());
        if row.outcome_state != PopulationOutcomeStateV1::Unmeasured {
            let Some(delivery) = delivery else {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} update has no matching label delivery input",
                    arm.arm_id
                )));
            };
            if delivery.label_id != update.label_id
                || delivery.label_revision != update.label_revision
                || delivery.label_payload_sha256 != update.label_payload_sha256
                || update.eligibility_report_sha256.as_deref()
                    != Some(delivery.eligibility_report.report_sha256.as_str())
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} update does not bind delivered label ID/revision/payload/report",
                    arm.arm_id
                )));
            }
        }
        match update.disposition {
            AciUpdateDispositionV1::HeldUnknown => held_unknown_count += 1,
            AciUpdateDispositionV1::Blocked => blocked_count += 1,
            AciUpdateDispositionV1::Rejected | AciUpdateDispositionV1::PendingPredictionMissing => {
                rejected_count += 1;
            }
            AciUpdateDispositionV1::Updated | AciUpdateDispositionV1::Duplicate => {}
        }
        match row.outcome_state {
            PopulationOutcomeStateV1::Qualified if !matches!(
                update.disposition,
                AciUpdateDispositionV1::Updated | AciUpdateDispositionV1::Duplicate
            ) => {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} does not accept its qualified feedback for {}",
                    arm.arm_id, row.sample_id
                )));
            }
            PopulationOutcomeStateV1::Unknown
                if update.disposition != AciUpdateDispositionV1::HeldUnknown =>
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} UNKNOWN row {} lacks a held-unknown receipt",
                    arm.arm_id, row.sample_id
                )));
            }
            PopulationOutcomeStateV1::Blocked
                if update.disposition != AciUpdateDispositionV1::Blocked =>
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} BLOCKED row {} lacks a blocked receipt",
                    arm.arm_id, row.sample_id
                )));
            }
            PopulationOutcomeStateV1::Unmeasured => {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} emits a label update for unmeasured row {}",
                    arm.arm_id, row.sample_id
                )));
            }
            _ => {}
        }
        if mutates {
            if row.outcome_state != PopulationOutcomeStateV1::Qualified {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} updated from non-qualified round {}",
                    arm.arm_id, row.round_id
                )));
            }
            let delivery = deliveries_by_sample.get(update.sample_id.as_str()).ok_or_else(|| {
                JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} update has no matching delivered K1 input",
                    arm.arm_id
                ))
            })?;
            if delivery.label_id != update.label_id
                || delivery.label_revision != update.label_revision
                || delivery.label_payload_sha256 != update.label_payload_sha256
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} update receipt does not match delivered label revision/payload",
                    arm.arm_id
                )));
            }
            if let Some(miss) = update.miscoverage_observed {
                let expected_miss = match prediction.interval.kind {
                    AciIntervalKindV1::FullRealLine => false,
                    AciIntervalKindV1::Empty => true,
                    AciIntervalKindV1::Bounded => {
                        let lower = prediction.interval.lower.unwrap_or(f64::NAN);
                        let upper = prediction.interval.upper.unwrap_or(f64::NAN);
                        delivery.label_value < lower || delivery.label_value > upper
                    }
                };
                let expected_alpha_after = prediction.alpha_at_issue
                    + prediction.learning_rate
                        * (prediction.target_miscoverage - f64::from(expected_miss));
                if miss != expected_miss
                    || update.alpha_before != Some(prediction.alpha_at_issue)
                    || update.alpha_after.is_none_or(|value| {
                        !value.is_finite() || (value - expected_alpha_after).abs() > 1.0e-12
                    })
                    || update.update_sequence.is_none()
                    || !valid_sha256(&update.calibration_state_before_sha256)
                    || !valid_sha256(&update.calibration_state_after_sha256)
                {
                    return Err(JointRiskTraceErrorV1(format!(
                        "tau-ACI arm {} alpha/miss/state receipt differs from its frozen prediction and recursion",
                        arm.arm_id
                    )));
                }
                observed_count += 1;
                observed_miscoverage += usize::from(miss);
            } else {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} applied update without a frozen-interval miss bit",
                    arm.arm_id
                )));
            }
        } else if row.outcome_state == PopulationOutcomeStateV1::Qualified
            && update.disposition == AciUpdateDispositionV1::HeldUnknown
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} holds an already-qualified shared label as UNKNOWN",
                arm.arm_id
            )));
        }
    }

    let mut degenerate_samples = BTreeSet::new();
    for resolution in &evidence.degenerate_resolutions {
        if !degenerate_samples.insert(resolution.sample_id.as_str()) {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} repeats a degenerate resolution",
                arm.arm_id
            )));
        }
        let Some(prediction) = predictions.get(resolution.sample_id.as_str()) else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} degenerate resolution has no prediction",
                arm.arm_id
            )));
        };
        let expected_degenerate_miss = match prediction.interval.kind {
            AciIntervalKindV1::FullRealLine => false,
            AciIntervalKindV1::Empty => true,
            AciIntervalKindV1::Bounded => false,
        };
        let expected_degenerate_alpha_after = prediction.alpha_at_issue
            + prediction.learning_rate
                * (prediction.target_miscoverage - f64::from(expected_degenerate_miss));
        if aci_degenerate_digest(resolution)? != resolution.receipt_sha256
            || resolution.measurement_event_id != prediction.measurement_event_id
            || resolution.issued_round != prediction.round
            || resolution.resolution_round != prediction.expected_delivery_round
            || resolution.interval_kind != prediction.interval.kind
            || (prediction.interval.kind == AciIntervalKindV1::Bounded)
            || resolution.miscoverage_observed != expected_degenerate_miss
            || resolution.alpha_before != prediction.alpha_at_issue
            || (resolution.alpha_after - expected_degenerate_alpha_after).abs() > 1.0e-12
            || !valid_sha256(&resolution.calibration_state_before_sha256)
            || !valid_sha256(&resolution.calibration_state_after_sha256)
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} degenerate resolution is inconsistent with frozen interval",
                arm.arm_id
            )));
        }
    }

    if !evidence.degenerate_resolutions.is_empty() && !evidence.updates.is_empty() {
        // The two evidence classes may coexist for different predictions; only
        // duplicate resolution of the same sample is prohibited below.
        let mut resolved = BTreeSet::new();
        for receipt in &evidence.degenerate_resolutions {
            if !resolved.insert(receipt.sample_id.as_str()) {
                return Err(JointRiskTraceErrorV1(format!(
                    "tau-ACI arm {} repeats a degenerate resolution",
                    arm.arm_id
                )));
            }
        }
    }

    for prediction in predictions.values() {
        let row = population
            .values()
            .find(|row| row.sample_id == prediction.sample_id)
            .expect("prediction is mapped to population");
        let has_update = evidence
            .updates
            .iter()
            .any(|update| update.sample_id == prediction.sample_id);
        let has_delivery = deliveries_by_sample.contains_key(prediction.sample_id.as_str());
        let has_missing = missing_by_sample.contains_key(prediction.sample_id.as_str());
        let has_degenerate = degenerate_samples.contains(prediction.sample_id.as_str());
        let terminal_paths = usize::from(has_delivery)
            + usize::from(has_missing)
            + usize::from(has_degenerate);
        if terminal_paths != 1 || has_delivery != has_update {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} must have exactly one delivery+update, missing, or degenerate path for sample {}",
                arm.arm_id, prediction.sample_id
            )));
        }
        let due_outcome_present = terminal_paths == 1;
        if !due_outcome_present {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} omits due feedback/forfeit/degenerate record for sample {}",
                arm.arm_id, prediction.sample_id
            )));
        }
        if has_missing && has_update
            && evidence.updates.iter().any(|update| {
                update.sample_id == prediction.sample_id
                    && update.disposition == AciUpdateDispositionV1::Updated
            })
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} both forfeits and updates the same sample",
                arm.arm_id
            )));
        }
        if row.outcome_state == PopulationOutcomeStateV1::Unmeasured && has_delivery {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} consumes delivery for unmeasured sample {}",
                arm.arm_id, prediction.sample_id
            )));
        }
        if has_degenerate
            && (prediction.interval.kind == AciIntervalKindV1::Bounded || has_delivery)
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} records degenerate resolution for bounded interval {}",
                arm.arm_id, prediction.sample_id
            )));
        }
    }
    let should_be_uncertified = !evidence.missing_feedback.is_empty()
        || evidence
            .updates
            .iter()
            .any(|update| update.disposition != AciUpdateDispositionV1::Updated)
        || predictions.values().any(|prediction| {
            let row = population
                .values()
                .find(|row| row.sample_id == prediction.sample_id)
                .unwrap();
            row.outcome_state != PopulationOutcomeStateV1::Qualified
                && prediction.interval.kind == AciIntervalKindV1::Bounded
        });
    for resolution in &evidence.degenerate_resolutions {
        observed_count += 1;
        observed_miscoverage += usize::from(resolution.miscoverage_observed);
    }
    if should_be_uncertified
        && evidence.guarantee_state == AciGuaranteeStateV1::UnderDeclaredFixedDelayAssumptions
    {
        return Err(JointRiskTraceErrorV1(format!(
            "tau-ACI arm {} claims fixed-delay guarantee despite invalid/missing physical feedback",
            arm.arm_id
        )));
    }
    if predictions.values().any(|prediction| prediction.policy_sha256 != arm.controller_policy_sha256) {
        return Err(JointRiskTraceErrorV1(format!(
            "tau-ACI arm {} prediction policy digest mismatch",
            arm.arm_id
        )));
    }
    Ok(Some((
        observed_miscoverage,
        observed_count,
        held_unknown_count,
        blocked_count,
        rejected_count,
    )))
}

fn validate_ca_arm(
    arm: &JointBenchmarkArmV1,
    evidence: &JointCaOcrcArmEvidenceV1,
    population: &BTreeMap<u64, &JointPopulationRowV1>,
) -> Result<
    (
        Option<f64>,
        usize,
        usize,
        bool,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    JointRiskTraceErrorV1,
> {
    if !evidence.feedback_horizon_closed {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} feedback horizon is not explicitly closed",
            arm.arm_id
        )));
    }
    if evidence.round_reports.len() != population.len()
        || evidence.round_inputs.len() != population.len()
    {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} must supply one input and report per population row",
            arm.arm_id
        )));
    }
    let inputs_by_round: BTreeMap<u64, &CaOcrcRoundInputV1> = evidence
        .round_inputs
        .iter()
        .map(|entry| (entry.prediction.round, entry))
        .collect();
    if inputs_by_round.len() != evidence.round_inputs.len() {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} repeats a round input",
            arm.arm_id
        )));
    }
    let unresolved: BTreeSet<u64> = evidence
        .unresolved_tracking_rounds
        .iter()
        .copied()
        .collect();
    if unresolved.len() != evidence.unresolved_tracking_rounds.len()
        || unresolved.iter().any(|round| !population.contains_key(round))
    {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} unresolved tracking rounds are duplicated or outside population",
            arm.arm_id
        )));
    }
    if json_sha256(&evidence.policy)? != arm.controller_policy_sha256
        || evidence.policy.tracking_capacity == 0
        || !evidence.policy.bounded_actual_delay_assumption_declared
        || !valid_sha256(&evidence.policy.actual_delay_contract_sha256)
    {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} policy digest/capacity/delay contract does not match the declared arm",
            arm.arm_id
        )));
    }
    let mut reports = BTreeMap::new();
    let mut seen_feedback_results = BTreeSet::new();
    let mut updated_feedback_count = 0_usize;
    let mut held_unknown_feedback_count = 0_usize;
    let mut blocked_feedback_count = 0_usize;
    let mut not_tracked_feedback_count = 0_usize;
    let mut rejected_feedback_count = 0_usize;
    let mut duplicate_feedback_count = 0_usize;
    let mut weighted_feedback_sum = 0.0;
    let mut max_occupancy = 0_usize;
    let mut capacity: Option<usize> = None;
    let mut theorem_intact = true;

    for report in &evidence.round_reports {
        let Some(row) = population.get(&report.round) else {
            return Err(JointRiskTraceErrorV1(format!(
                "CA-OCRC arm {} report round {} is outside population",
                arm.arm_id, report.round
            )));
        };
        let prediction = &report.prediction_receipt;
        let round_input = inputs_by_round.get(&report.round).ok_or_else(|| {
            JointRiskTraceErrorV1(format!(
                "CA-OCRC arm {} report has no corresponding original round input",
                arm.arm_id
            ))
        })?;
        let mut canonical_round_input = (*round_input).clone();
        canonical_round_input.feedbacks.sort_by(|left, right| {
            (left.sample_id.as_str(), left.label_id.as_str(), left.label_revision).cmp(&(
                right.sample_id.as_str(),
                right.label_id.as_str(),
                right.label_revision,
            ))
        });
        let input_digest = json_sha256(&canonical_round_input)?;
        if round_input.schema != CA_OCRC_ROUND_INPUT_SCHEMA_V1
            || round_input.prediction.round != report.round
            || round_input.prediction.sample_id != row.sample_id
            || round_input.prediction.measurement_event_id != row.measurement_event_id
            || round_input.prediction.physical_measurement_selected != row.physical_measurement_selected
            || round_input.prediction.global_model_artifact_sha256 != arm.global_model_artifact_sha256
            || report.input_sha256 != input_digest
            || ca_prediction_digest(prediction)? != prediction.receipt_sha256
            || ca_round_report_digest(report)? != report.report_sha256
            || prediction.round != report.round
            || prediction.sample_id != row.sample_id
            || prediction.measurement_event_id != row.measurement_event_id
            || prediction.physical_measurement_selected != row.physical_measurement_selected
            || prediction.global_model_artifact_sha256 != arm.global_model_artifact_sha256
            || prediction.interval_method_sha256 != arm.interval_method_sha256
            || prediction.policy_sha256 != arm.controller_policy_sha256
            || report.tracking_capacity != evidence.policy.tracking_capacity
            || report.tracking_occupancy_after > report.tracking_capacity
            || !report.weighted_feedback_sum.is_finite()
            || !report.threshold_before.is_finite()
            || !report.threshold_after.is_finite()
            || !report.tracking_load_ell_f.is_finite()
            || !report.proxy_survival_at_max_delay.is_finite()
            || report.proxy_survival_at_max_delay < 0.0
            || report.proxy_survival_at_max_delay > 1.0
            || reports.insert(report.round, report).is_some()
        {
            return Err(JointRiskTraceErrorV1(format!(
                "CA-OCRC arm {} round report does not bind shared acquisition/identity/capacity",
                arm.arm_id
            )));
        }
        if let Some(existing) = capacity {
            if existing != report.tracking_capacity {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} changes tracking capacity mid-run",
                    arm.arm_id
                )));
            }
        } else {
            capacity = Some(report.tracking_capacity);
        }
        for feedback in &round_input.feedbacks {
            let Some(outcome_row) = population
                .values()
                .find(|candidate| candidate.sample_id == feedback.sample_id)
            else {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} input references unknown sample {}",
                    arm.arm_id, feedback.sample_id
                )));
            };
            validate_ca_feedback_input(feedback, report.round, outcome_row)?;
            if feedback.clock_domain_id != round_input.prediction.clock_domain_id
                || feedback.delivered_at_unix_seconds
                    < round_input.prediction.prediction_at_unix_seconds
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} feedback clock/delivery chronology conflicts with round {}",
                    arm.arm_id, report.round
                )));
            }
            let result = report.feedback_results.iter().find(|result| {
                result.sample_id == feedback.sample_id
                    && result.label_id == feedback.label_id
                    && result.label_revision == feedback.label_revision
            });
            let Some(result) = result else {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} round {} has an input feedback without a result receipt",
                    arm.arm_id, report.round
                )));
            };
            let result_key = (
                report.round,
                result.sample_id.as_str(),
                result.label_id.as_str(),
                result.label_revision,
            );
            let tracked_result = result.disposition != CaOcrcFeedbackDispositionV1::NotTracked;
            if !seen_feedback_results.insert(result_key)
                || (tracked_result
                    && (result.originating_round != Some(outcome_row.round_id)
                        || result.actual_delay_steps
                            != Some(report.round.saturating_sub(outcome_row.round_id))))
                || (!tracked_result
                    && (result.originating_round.is_some()
                        || result.actual_delay_steps.is_some()
                        || result.weighted_update_contribution.is_some()))
                || ca_feedback_digest(result)? != result.feedback_receipt_sha256
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} has a duplicate, incomplete or tampered feedback result receipt",
                    arm.arm_id
                )));
            }
            match result.disposition {
                CaOcrcFeedbackDispositionV1::Updated => updated_feedback_count += 1,
                CaOcrcFeedbackDispositionV1::HeldUnknown => held_unknown_feedback_count += 1,
                CaOcrcFeedbackDispositionV1::Blocked => blocked_feedback_count += 1,
                CaOcrcFeedbackDispositionV1::NotTracked => not_tracked_feedback_count += 1,
                CaOcrcFeedbackDispositionV1::Rejected => rejected_feedback_count += 1,
                CaOcrcFeedbackDispositionV1::Duplicate => duplicate_feedback_count += 1,
            }
        }
        weighted_feedback_sum += report.weighted_feedback_sum;
        max_occupancy = max_occupancy.max(report.tracking_occupancy_after);
        theorem_intact &= report.theorem_assumptions_intact;
        if !report.theorem_assumptions_intact && report.paper_risk_gap_bound.is_some() {
            return Err(JointRiskTraceErrorV1(format!(
                "CA-OCRC arm {} emits a paper risk bound after assumptions were invalidated",
                arm.arm_id
            )));
        }
        if !row.physical_measurement_selected && report.theorem_assumptions_intact {
            return Err(JointRiskTraceErrorV1(format!(
                "CA-OCRC arm {} preserves complete-feedback theorem after an unmeasured row",
                arm.arm_id
            )));
        }
        for feedback in &report.feedback_results {
            let Some(outcome_row) = population
                .values()
                .find(|candidate| candidate.sample_id == feedback.sample_id)
            else {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} feedback references unknown sample {}",
                    arm.arm_id, feedback.sample_id
                )));
            };
            if outcome_row.label_id.as_deref() != Some(feedback.label_id.as_str())
                || outcome_row.label_revision != Some(feedback.label_revision)
                || (feedback.disposition == CaOcrcFeedbackDispositionV1::Updated
                    && outcome_row.outcome_state != PopulationOutcomeStateV1::Qualified)
                || (outcome_row.outcome_state == PopulationOutcomeStateV1::Unknown
                    && !matches!(
                        feedback.disposition,
                        CaOcrcFeedbackDispositionV1::HeldUnknown
                            | CaOcrcFeedbackDispositionV1::NotTracked
                    ))
                || (outcome_row.outcome_state == PopulationOutcomeStateV1::Blocked
                    && !matches!(
                        feedback.disposition,
                        CaOcrcFeedbackDispositionV1::Blocked
                            | CaOcrcFeedbackDispositionV1::NotTracked
                    ))
                || (outcome_row.outcome_state == PopulationOutcomeStateV1::Unmeasured)
            {
                return Err(JointRiskTraceErrorV1(format!(
                    "CA-OCRC arm {} feedback result disposition/identity conflicts with shared label state for {}",
                    arm.arm_id, feedback.label_id
                )));
            }
        }
    }
    let input_feedback_keys: BTreeSet<(u64, &str, &str, u64)> = evidence
        .round_inputs
        .iter()
        .flat_map(|input| {
            input.feedbacks.iter().map(move |feedback| {
                (
                    feedback.feedback_round,
                    feedback.sample_id.as_str(),
                    feedback.label_id.as_str(),
                    feedback.label_revision,
                )
            })
        })
        .collect();
    if seen_feedback_results != input_feedback_keys {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} has a feedback result without exactly one original input",
            arm.arm_id
        )));
    }
    let incomplete_physical_feedback = population.values().any(|row| {
        !row.physical_measurement_selected
            || row.outcome_state != PopulationOutcomeStateV1::Qualified
    });
    if incomplete_physical_feedback && theorem_intact {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} claims theorem assumptions intact despite unselected/unqualified physical feedback",
            arm.arm_id
        )));
    }
    if !unresolved.is_empty() && theorem_intact {
        return Err(JointRiskTraceErrorV1(format!(
            "CA-OCRC arm {} claims theorem assumptions intact with unresolved feedback at horizon close",
            arm.arm_id
        )));
    }
    Ok((
        Some(weighted_feedback_sum),
        max_occupancy,
        capacity.unwrap_or(0),
        theorem_intact,
        updated_feedback_count,
        held_unknown_feedback_count,
        blocked_feedback_count,
        not_tracked_feedback_count,
        rejected_feedback_count,
        duplicate_feedback_count,
    ))
}

fn validate_shared_audit(
    arms: &[JointBenchmarkArmV1],
) -> Result<usize, JointRiskTraceErrorV1> {
    let first = arms.first().expect("joint trace requires at least one arm");
    let first_input = &first.selection_risk_input;
    let Some(first_design) = first_input.audit_design.as_ref() else {
        return Err(JointRiskTraceErrorV1(
            "first arm has no independent audit design".into(),
        ));
    };
    let mut first_audit: Vec<(u64, u64, u64, String)> = first_input
        .independent_audit_losses
        .iter()
        .map(|row| {
            (
                row.round_id,
                row.inclusion_probability.to_bits(),
                row.loss.to_bits(),
                row.audit_measurement_receipt_sha256.clone(),
            )
        })
        .collect();
    first_audit.sort_by_key(|row| row.0);
    for arm in arms.iter().skip(1) {
        let input = &arm.selection_risk_input;
        let Some(design) = input.audit_design.as_ref() else {
            return Err(JointRiskTraceErrorV1(format!(
                "arm {} has no audit design",
                arm.arm_id
            )));
        };
        if design != first_design {
            return Err(JointRiskTraceErrorV1(format!(
                "arm {} uses a different independent audit design",
                arm.arm_id
            )));
        }
        let mut audit: Vec<(u64, u64, u64, String)> = input
            .independent_audit_losses
            .iter()
            .map(|row| {
                (
                    row.round_id,
                    row.inclusion_probability.to_bits(),
                    row.loss.to_bits(),
                    row.audit_measurement_receipt_sha256.clone(),
                )
            })
            .collect();
        audit.sort_by_key(|row| row.0);
        if audit != first_audit {
            return Err(JointRiskTraceErrorV1(format!(
                "arm {} audit draw or reference-receipt set differs from paired arms",
                arm.arm_id
            )));
        }
    }
    if first_audit.is_empty() {
        return Err(JointRiskTraceErrorV1(
            "paired benchmark requires at least one independent audit observation".into(),
        ));
    }
    Ok(first_audit.len())
}

/// Validates one Mode-A paired trace and emits separate controller and
/// population-risk summaries. It does not recompute controller updates.
pub fn evaluate_joint_risk_trace_v1(
    input: &JointRiskTraceInputV1,
) -> Result<JointRiskTraceReportV1, JointRiskTraceErrorV1> {
    if input.schema != JOINT_RISK_TRACE_SCHEMA_V1 {
        return Err(JointRiskTraceErrorV1(
            "joint trace schema is unsupported".into(),
        ));
    }
    require_text(&input.trace_id, "trace.trace_id")?;
    if input.mode != JointBenchmarkModeV1::PairedSharedAcquisition {
        return Err(JointRiskTraceErrorV1(
            "only paired shared-acquisition mode is implemented in v1".into(),
        ));
    }
    if !valid_sha256(&input.population_manifest_sha256)
        || !valid_sha256(&input.source_ledger_sha256)
        || !valid_sha256(&input.trace_sha256)
        || seal_joint_risk_trace_v1(input)? != input.trace_sha256
    {
        return Err(JointRiskTraceErrorV1(
            "joint trace or source manifest digest is malformed or inconsistent".into(),
        ));
    }
    let population = validate_population(input)?;
    if input.arms.len() < 2 {
        return Err(JointRiskTraceErrorV1(
            "paired benchmark requires at least two controller arms".into(),
        ));
    }
    let mut arm_ids = BTreeSet::new();
    let mut shared_acquisition_policy: Option<&str> = None;
    let mut shared_budget: Option<f64> = None;
    let mut arm_summaries = Vec::new();

    for arm in &input.arms {
        require_text(&arm.arm_id, "arm.arm_id")?;
        if !arm_ids.insert(arm.arm_id.as_str()) {
            return Err(JointRiskTraceErrorV1("duplicate arm ID".into()));
        }
        if !valid_sha256(&arm.global_model_artifact_sha256)
            || !valid_sha256(&arm.interval_method_sha256)
            || !valid_sha256(&arm.controller_policy_sha256)
            || !valid_sha256(&arm.acquisition_policy_sha256)
            || !arm.physical_metrology_budget_units.is_finite()
            || arm.physical_metrology_budget_units < 0.0
        {
            return Err(JointRiskTraceErrorV1(format!(
                "arm {} has malformed digest or budget",
                arm.arm_id
            )));
        }
        if let Some(policy) = shared_acquisition_policy {
            if policy != arm.acquisition_policy_sha256 {
                return Err(JointRiskTraceErrorV1(
                    "paired shared-acquisition arms use different acquisition policy digests".into(),
                ));
            }
        } else {
            shared_acquisition_policy = Some(&arm.acquisition_policy_sha256);
        }
        if let Some(budget) = shared_budget {
            if (budget - arm.physical_metrology_budget_units).abs() > f64::EPSILON {
                return Err(JointRiskTraceErrorV1(
                    "paired shared-acquisition arms declare different metrology budgets".into(),
                ));
            }
        } else {
            shared_budget = Some(arm.physical_metrology_budget_units);
        }

        let selection_report = validate_selection_risk_input(arm, input, &population)?;
        let summary = match (&arm.controller_evidence, arm.method) {
            (JointControllerEvidenceV1::FixedDelayTauAci(evidence), JointControllerMethodV1::FixedDelayTauAci) => {
                let (
                    observed_miscoverage,
                    observed_count,
                    held_unknown,
                    blocked,
                    rejected,
                ) = validate_aci_arm(arm, evidence, &population)?
                    .expect("ACI validator returns observed counts");
                JointArmRiskSummaryV1 {
                    arm_id: arm.arm_id.clone(),
                    method: arm.method,
                    qualified_subset_risk: selection_report.qualified_subset_risk,
                    partial_identification_lower: selection_report.partial_identification_lower,
                    partial_identification_upper: selection_report.partial_identification_upper,
                    identification_interval_width: selection_report.identification_interval_width,
                    independent_audit_population_estimate: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.population_risk_estimate),
                    independent_audit_standard_error: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.standard_error),
                    independent_audit_hoeffding_lower: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.hoeffding_lower),
                    independent_audit_hoeffding_upper: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.hoeffding_upper),
                    selection_risk_report_sha256: selection_report.report_sha256,
                    observed_miscoverage_count: Some(observed_miscoverage),
                    observed_miscoverage_count_denominator: Some(observed_count),
                    aci_held_unknown_count: Some(held_unknown),
                    aci_blocked_count: Some(blocked),
                    aci_rejected_count: Some(rejected),
                    aci_missing_feedback_count: Some(evidence.missing_feedback.len()),
                    aci_degenerate_resolution_count: Some(evidence.degenerate_resolutions.len()),
                    ca_ocrc_weighted_feedback_sum: None,
                    ca_ocrc_updated_feedback_count: None,
                    ca_ocrc_held_unknown_count: None,
                    ca_ocrc_blocked_count: None,
                    ca_ocrc_not_tracked_count: None,
                    ca_ocrc_rejected_count: None,
                    ca_ocrc_duplicate_count: None,
                    ca_ocrc_max_tracking_occupancy: None,
                    ca_ocrc_tracking_capacity: None,
                    ca_ocrc_theorem_assumptions_intact: None,
                }
            }
            (JointControllerEvidenceV1::BoundedDelayCaOcrc(evidence), JointControllerMethodV1::BoundedDelayCaOcrc) => {
                let (
                    weighted_sum,
                    occupancy,
                    capacity,
                    theorem_intact,
                    updated,
                    held_unknown,
                    blocked,
                    not_tracked,
                    rejected,
                    duplicate,
                ) = validate_ca_arm(arm, evidence, &population)?;
                JointArmRiskSummaryV1 {
                    arm_id: arm.arm_id.clone(),
                    method: arm.method,
                    qualified_subset_risk: selection_report.qualified_subset_risk,
                    partial_identification_lower: selection_report.partial_identification_lower,
                    partial_identification_upper: selection_report.partial_identification_upper,
                    identification_interval_width: selection_report.identification_interval_width,
                    independent_audit_population_estimate: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.population_risk_estimate),
                    independent_audit_standard_error: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.standard_error),
                    independent_audit_hoeffding_lower: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.hoeffding_lower),
                    independent_audit_hoeffding_upper: selection_report
                        .independent_audit_estimate
                        .as_ref()
                        .map(|value| value.hoeffding_upper),
                    selection_risk_report_sha256: selection_report.report_sha256,
                    observed_miscoverage_count: None,
                    observed_miscoverage_count_denominator: None,
                    aci_held_unknown_count: None,
                    aci_blocked_count: None,
                    aci_rejected_count: None,
                    aci_missing_feedback_count: None,
                    aci_degenerate_resolution_count: None,
                    ca_ocrc_weighted_feedback_sum: weighted_sum,
                    ca_ocrc_updated_feedback_count: Some(updated),
                    ca_ocrc_held_unknown_count: Some(held_unknown),
                    ca_ocrc_blocked_count: Some(blocked),
                    ca_ocrc_not_tracked_count: Some(not_tracked),
                    ca_ocrc_rejected_count: Some(rejected),
                    ca_ocrc_duplicate_count: Some(duplicate),
                    ca_ocrc_max_tracking_occupancy: Some(occupancy),
                    ca_ocrc_tracking_capacity: Some(capacity),
                    ca_ocrc_theorem_assumptions_intact: Some(theorem_intact),
                }
            }
            _ => {
                return Err(JointRiskTraceErrorV1(format!(
                    "arm {} method kind does not match its typed controller evidence",
                    arm.arm_id
                )));
            }
        };
        arm_summaries.push(summary);
    }

    let audit_count = validate_shared_audit(&input.arms)?;
    let selected_count = input
        .population
        .iter()
        .filter(|row| row.physical_measurement_selected)
        .count();
    let physical_cost: f64 = input
        .population
        .iter()
        .map(|row| row.physical_metrology_cost_units)
        .sum();
    if physical_cost > shared_budget.unwrap_or(0.0) + f64::EPSILON {
        return Err(JointRiskTraceErrorV1(
            "shared physical-metrology selections exceed the declared budget".into(),
        ));
    }
    let qualified_count = input.population.iter().filter(|row| row.outcome_state == PopulationOutcomeStateV1::Qualified).count();
    let unknown_count = input.population.iter().filter(|row| row.outcome_state == PopulationOutcomeStateV1::Unknown).count();
    let blocked_count = input.population.iter().filter(|row| row.outcome_state == PopulationOutcomeStateV1::Blocked).count();
    let unmeasured_count = input.population.iter().filter(|row| row.outcome_state == PopulationOutcomeStateV1::Unmeasured).count();

    let mut report = JointRiskTraceReportV1 {
        schema: JOINT_RISK_TRACE_REPORT_SCHEMA_V1.into(),
        trace_id: input.trace_id.clone(),
        trace_sha256: input.trace_sha256.clone(),
        status: JointTraceStatusV1::Complete,
        reason_codes: Vec::new(),
        population_count: input.population.len(),
        physical_measurement_count: selected_count,
        physical_metrology_cost_units: physical_cost,
        qualified_count,
        unknown_count,
        blocked_count,
        unmeasured_count,
        independent_audit_count: audit_count,
        shared_acquisition_policy_sha256: shared_acquisition_policy.unwrap_or_default().into(),
        arm_summaries,
        report_sha256: String::new(),
    };
    report.report_sha256 = joint_report_digest(&report)?;
    Ok(report)
}

fn joint_report_digest(
    report: &JointRiskTraceReportV1,
) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}
