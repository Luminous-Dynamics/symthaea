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
    CA_OCRC_ROUND_INPUT_SCHEMA_V1, CA_OCRC_PREDICTION_SCHEMA_V1, CaOcrcFeedbackDispositionV1,
    CaOcrcLabelFeedbackV1, CaOcrcPolicyV1, CaOcrcPredictionInputV1, CaOcrcPredictionSetV1,
    CaOcrcRoundInputV1, CaOcrcRoundReportV1, CaOcrcSetKindV1, close_ca_ocrc_feedback_horizon_v1,
    create_ca_ocrc_engine_v1, step_ca_ocrc_round_v1,
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

pub const JOINT_RISK_TRACE_SCHEMA_V1: &str = "luminous.semi-bench.joint-risk-trace.v1";
pub const JOINT_RISK_TRACE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.joint-risk-trace-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointBenchmarkModeV1 {
    PairedSharedAcquisition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointComparisonScopeV1 {
    /// Isolates the controller by requiring the same global model and interval method.
    ControllerAblation,
    /// Compares full end-to-end pipelines; model/interval differences are allowed and must be disclosed.
    EndToEndPipeline,
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
    pub event_time_unix_seconds: i64,
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

pub const JOINT_AUDIT_DRAW_SCHEMA_V1: &str =
    "luminous.semi-bench.joint-audit-draw.v1";
pub const JOINT_AUDIT_DRAW_ALGORITHM_V1: &str =
    "sha256-counter-bernoulli-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointAuditInclusionProbabilityV1 {
    pub round_id: u64,
    pub inclusion_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointAuditRandomizationReceiptV1 {
    pub schema: String,
    pub algorithm_id: String,
    pub population_manifest_sha256: String,
    /// A revealed 32-byte seed. The commitment allows the draw to be replayed;
    /// it does not prove that the seed was generated independently or signed.
    pub seed_hex: String,
    pub seed_commitment_sha256: String,
    pub randomization_source_receipt_sha256: String,
    /// Must predate the first population event time.
    pub randomness_committed_at_unix_seconds: i64,
    /// One Bernoulli inclusion probability for every row in the frozen population.
    pub inclusion_probabilities: Vec<JointAuditInclusionProbabilityV1>,
    pub selected_round_ids: Vec<u64>,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointRiskTraceInputV1 {
    pub schema: String,
    pub trace_id: String,
    pub mode: JointBenchmarkModeV1,
    pub comparison_scope: JointComparisonScopeV1,
    pub population_manifest_sha256: String,
    pub source_ledger_sha256: String,
    pub population: Vec<JointPopulationRowV1>,
    pub audit_randomization: JointAuditRandomizationReceiptV1,
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
    pub independent_audit_qualified_frame_count: Option<usize>,
    pub independent_audit_missing_frame_count: Option<usize>,
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
    pub comparison_scope: JointComparisonScopeV1,
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
    pub audit_randomization_receipt_sha256: String,
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
    value
        .arms
        .sort_by(|left, right| left.arm_id.cmp(&right.arm_id));
    for arm in &mut value.arms {
        canonicalize_selection_input(&mut arm.selection_risk_input);
        match &mut arm.controller_evidence {
            JointControllerEvidenceV1::FixedDelayTauAci(evidence) => {
                evidence.predictions.sort_by_key(|entry| entry.round);
                evidence.deliveries.sort_by(|left, right| {
                    (
                        left.delivered_round,
                        left.sample_id.as_str(),
                        left.label_id.as_str(),
                        left.label_revision,
                    )
                        .cmp(&(
                            right.delivered_round,
                            right.sample_id.as_str(),
                            right.label_id.as_str(),
                            right.label_revision,
                        ))
                });
                evidence
                    .missing_feedback
                    .sort_by_key(|entry| entry.issued_round);
                evidence.updates.sort_by(|left, right| {
                    (
                        left.sample_id.as_str(),
                        left.label_id.as_str(),
                        left.label_revision,
                    )
                        .cmp(&(
                            right.sample_id.as_str(),
                            right.label_id.as_str(),
                            right.label_revision,
                        ))
                });
                evidence
                    .degenerate_resolutions
                    .sort_by_key(|entry| entry.issued_round);
            }
            JointControllerEvidenceV1::BoundedDelayCaOcrc(evidence) => {
                evidence
                    .round_inputs
                    .sort_by_key(|entry| entry.prediction.round);
                for round_input in &mut evidence.round_inputs {
                    round_input.feedbacks.sort_by(|left, right| {
                        (
                            left.sample_id.as_str(),
                            left.label_id.as_str(),
                            left.label_revision,
                        )
                            .cmp(&(
                                right.sample_id.as_str(),
                                right.label_id.as_str(),
                                right.label_revision,
                            ))
                    });
                }
                evidence.round_reports.sort_by_key(|entry| entry.round);
                for report in &mut evidence.round_reports {
                    report.feedback_results.sort_by(|left, right| {
                        (
                            left.originating_round,
                            left.label_id.as_str(),
                            left.label_revision,
                        )
                            .cmp(&(
                                right.originating_round,
                                right.label_id.as_str(),
                                right.label_revision,
                            ))
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
        if row.event_time_unix_seconds <= 0 {
            return Err(JointRiskTraceErrorV1(format!(
                "round {} has an invalid population event timestamp",
                row.round_id
            )));
        }
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
    let mut previous_event_time = None;
    for row in rows.values() {
        if previous_event_time.is_some_and(|previous| row.event_time_unix_seconds <= previous) {
            return Err(JointRiskTraceErrorV1(
                "population event times must increase strictly with round order".into(),
            ));
        }
        previous_event_time = Some(row.event_time_unix_seconds);
    }
    Ok(rows)
}

fn expected_partitions(
    population: &BTreeMap<u64, &JointPopulationRowV1>,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
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
        || row
            .label_value
            .is_some_and(|value| delivery.label_value != value)
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
        || delivery.delivered_at_unix_seconds
            < delivery.eligibility_report_available_at_unix_seconds
        || delivery.eligibility_report_available_at_unix_seconds
            < delivery.label_available_at_unix_seconds
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
        || row
            .label_value
            .is_some_and(|value| feedback.label_value != value)
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
        || report.frontier != super::label_consumption_guard::LabelEvidenceFrontierV1::CurrentAtK1
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

fn aci_update_digest(receipt: &DelayedAciUpdateReceiptV1) -> Result<String, JointRiskTraceErrorV1> {
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

fn ca_round_report_digest(report: &CaOcrcRoundReportV1) -> Result<String, JointRiskTraceErrorV1> {
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
            || prediction.event_time_unix_seconds != row.event_time_unix_seconds
            || prediction.model_artifact_sha256 != arm.global_model_artifact_sha256
            || prediction.policy_sha256 != arm.controller_policy_sha256
            || prediction.delay_rounds != evidence.policy.delay_rounds
            || prediction.target_miscoverage != evidence.policy.target_miscoverage
            || prediction.learning_rate != evidence.policy.learning_rate
            || prediction.phase != prediction.round % prediction.delay_rounds
            || prediction.expected_delivery_round
                != prediction
                    .round
                    .checked_add(prediction.delay_rounds)
                    .unwrap_or(u64::MAX)
            || aci_prediction_digest(prediction)? != prediction.receipt_sha256
            || predictions
                .insert(prediction.sample_id.as_str(), prediction)
                .is_some()
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
        let Some(row) = population
            .values()
            .find(|row| row.sample_id == delivery.sample_id)
        else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} delivery references unknown sample",
                arm.arm_id
            )));
        };
        validate_aci_delivery(delivery, prediction, row)?;
        if let Some(delivery_round_prediction) = evidence
            .predictions
            .iter()
            .find(|candidate| candidate.round == delivery.delivered_round)
            && delivery.delivered_at_unix_seconds
                < delivery_round_prediction.prediction_time_unix_seconds
        {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} receives sample {} feedback before the delivery-round prediction",
                arm.arm_id, delivery.sample_id
            )));
        }
        if deliveries_by_sample
            .insert(delivery.sample_id.as_str(), delivery)
            .is_some()
        {
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
            || missing_by_sample
                .insert(missing.sample_id.as_str(), missing)
                .is_some()
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
        let Some(row) = population
            .values()
            .find(|row| row.sample_id == update.sample_id)
        else {
            return Err(JointRiskTraceErrorV1(format!(
                "tau-ACI arm {} update references unknown sample {}",
                arm.arm_id, update.sample_id
            )));
        };
        let prediction = predictions
            .get(update.sample_id.as_str())
            .expect("prediction exists");
        if update.measurement_event_id != row.measurement_event_id
            || update.prediction_receipt_sha256.as_deref()
                != Some(prediction.receipt_sha256.as_str())
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
            PopulationOutcomeStateV1::Qualified
                if !matches!(
                    update.disposition,
                    AciUpdateDispositionV1::Updated | AciUpdateDispositionV1::Duplicate
                ) =>
            {
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
            let delivery = deliveries_by_sample
                .get(update.sample_id.as_str())
                .ok_or_else(|| {
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
        let terminal_paths =
            usize::from(has_delivery) + usize::from(has_missing) + usize::from(has_degenerate);
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
        if has_missing
            && has_update
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
    if predictions
        .values()
        .any(|prediction| prediction.policy_sha256 != arm.controller_policy_sha256)
    {
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
        || unresolved
            .iter()
            .any(|round| !population.contains_key(round))
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
            (
                left.sample_id.as_str(),
                left.label_id.as_str(),
                left.label_revision,
            )
                .cmp(&(
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
            || round_input.prediction.physical_measurement_selected
                != row.physical_measurement_selected
            || round_input.prediction.global_model_artifact_sha256
                != arm.global_model_artifact_sha256
            || report.input_sha256 != input_digest
            || ca_prediction_digest(prediction)? != prediction.receipt_sha256
            || ca_round_report_digest(report)? != report.report_sha256
            || prediction.round != report.round
            || prediction.sample_id != row.sample_id
            || prediction.measurement_event_id != row.measurement_event_id
            || prediction.event_time_unix_seconds != row.event_time_unix_seconds
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

fn validate_shared_audit(arms: &[JointBenchmarkArmV1]) -> Result<usize, JointRiskTraceErrorV1> {
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
    let mut shared_global_model: Option<&str> = None;
    let mut shared_interval_method: Option<&str> = None;
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
                    "paired shared-acquisition arms use different acquisition policy digests"
                        .into(),
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

        if input.comparison_scope == JointComparisonScopeV1::ControllerAblation {
            if shared_global_model.is_some_and(|digest| digest != arm.global_model_artifact_sha256)
                || shared_interval_method
                    .is_some_and(|digest| digest != arm.interval_method_sha256)
            {
                return Err(JointRiskTraceErrorV1(
                    "controller-ablation arms must share the same global-model and interval-method artifacts; use end-to-end scope for pipeline comparisons".into(),
                ));
            }
            shared_global_model.get_or_insert(&arm.global_model_artifact_sha256);
            shared_interval_method.get_or_insert(&arm.interval_method_sha256);
        }

        let selection_report = validate_selection_risk_input(arm, input, &population)?;
        let summary = match (&arm.controller_evidence, arm.method) {
            (
                JointControllerEvidenceV1::FixedDelayTauAci(evidence),
                JointControllerMethodV1::FixedDelayTauAci,
            ) => {
                let (observed_miscoverage, observed_count, held_unknown, blocked, rejected) =
                    validate_aci_arm(arm, evidence, &population)?
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
            (
                JointControllerEvidenceV1::BoundedDelayCaOcrc(evidence),
                JointControllerMethodV1::BoundedDelayCaOcrc,
            ) => {
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
    let qualified_count = input
        .population
        .iter()
        .filter(|row| row.outcome_state == PopulationOutcomeStateV1::Qualified)
        .count();
    let unknown_count = input
        .population
        .iter()
        .filter(|row| row.outcome_state == PopulationOutcomeStateV1::Unknown)
        .count();
    let blocked_count = input
        .population
        .iter()
        .filter(|row| row.outcome_state == PopulationOutcomeStateV1::Blocked)
        .count();
    let unmeasured_count = input
        .population
        .iter()
        .filter(|row| row.outcome_state == PopulationOutcomeStateV1::Unmeasured)
        .count();

    let mut report = JointRiskTraceReportV1 {
        schema: JOINT_RISK_TRACE_REPORT_SCHEMA_V1.into(),
        trace_id: input.trace_id.clone(),
        trace_sha256: input.trace_sha256.clone(),
        comparison_scope: input.comparison_scope,
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

fn joint_report_digest(report: &JointRiskTraceReportV1) -> Result<String, JointRiskTraceErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

#[cfg(test)]
mod tests {
    use super::super::delayed_aci::{
        DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_POLICY_SCHEMA_V1,
        DELAYED_ACI_PREDICTION_SCHEMA_V1, DelayedAciPredictionInputV1, begin_delayed_aci_round_v1,
        close_delayed_aci_stream_v1, create_delayed_aci_engine_v1, deliver_delayed_aci_label_v1,
        issue_delayed_aci_prediction_v1, resolve_degenerate_aci_interval_v1,
    };
    use super::super::label_consumption_guard::{
        LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
    };
    use super::super::selection_risk_bounds::{
        IndependentAuditLossObservationV1, QualifiedLossObservationV1,
        SELECTION_AUDIT_DESIGN_SCHEMA_V1, SELECTION_RISK_INPUT_SCHEMA_V1, SelectionAuditDesignV1,
    };
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn eligibility_report(
        label_id: &str,
        revision: u64,
        event_id: &str,
        payload: &str,
        value: f64,
        eligibility: LabelConsumptionEligibilityV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "joint-cal-cert-001".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility,
            downstream_consumption_allowed: eligibility
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('b'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.05),
            qualification_report_sha256: digest('c'),
            requalification_report_sha256: digest('d'),
            requalification_frontier_sha256: digest('e'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        report
    }

    fn selection_input(
        population_manifest_sha256: &str,
        source_ledger_sha256: &str,
        evidence_sha256: &str,
        audit_loss: f64,
    ) -> SelectionRiskInputV1 {
        SelectionRiskInputV1 {
            schema: SELECTION_RISK_INPUT_SCHEMA_V1.into(),
            input_id: "joint-risk-selection-input-v1".into(),
            population_manifest_sha256: population_manifest_sha256.into(),
            source_ledger_sha256: source_ledger_sha256.into(),
            population_round_ids: vec![1, 2],
            qualified_round_ids: vec![1],
            unknown_round_ids: Vec::new(),
            blocked_round_ids: Vec::new(),
            unmeasured_round_ids: vec![2],
            qualified_losses: vec![QualifiedLossObservationV1 {
                round_id: 1,
                loss: 1.0,
                evidence_receipt_sha256: evidence_sha256.into(),
            }],
            audit_design: Some(SelectionAuditDesignV1 {
                schema: SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
                design_sha256: digest('f'),
                randomization_provenance_sha256: digest('1'),
                reference_method_sha256: digest('2'),
                minimum_inclusion_probability: 1.0,
                confidence_delta: 0.05,
                independent_of_operational_qualification: true,
            }),
            independent_audit_losses: vec![IndependentAuditLossObservationV1 {
                round_id: 2,
                loss: audit_loss,
                inclusion_probability: 1.0,
                audit_measurement_receipt_sha256: digest('3'),
            }],
        }
    }

    fn aci_evidence(engine_id: &str, policy: DelayedAciPolicyV1) -> JointTauAciArmEvidenceV1 {
        let mut engine =
            create_delayed_aci_engine_v1(engine_id, "fab-clock", policy.clone()).unwrap();

        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let first_prediction = issue_delayed_aci_prediction_v1(
            &mut engine,
            &DelayedAciPredictionInputV1 {
                schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
                sample_id: "wafer-001".into(),
                measurement_event_id: "meas-001".into(),
                round: 1,
                event_time_unix_seconds: 1_900_000_000,
                prediction_time_unix_seconds: 1_900_000_001,
                clock_domain_id: "fab-clock".into(),
                model_artifact_sha256: digest('4'),
                point_prediction: 10.0,
                interval: AciPredictionIntervalV1::bounded(9.0, 11.0),
            },
        )
        .unwrap();

        begin_delayed_aci_round_v1(&mut engine, 2).unwrap();
        let payload = digest('5');
        let first_delivery = DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
            sample_id: "wafer-001".into(),
            label_id: "label-001".into(),
            label_revision: 1,
            measurement_event_id: "meas-001".into(),
            label_payload_sha256: payload.clone(),
            label_value: 12.0,
            label_available_at_unix_seconds: 1_900_000_010,
            eligibility_report_available_at_unix_seconds: 1_900_000_015,
            delivered_at_unix_seconds: 1_900_000_040,
            delivered_round: 2,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: "label-001:1".into(),
            eligibility_report: eligibility_report(
                "label-001",
                1,
                "meas-001",
                &payload,
                12.0,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            ),
        };
        let second_prediction = issue_delayed_aci_prediction_v1(
            &mut engine,
            &DelayedAciPredictionInputV1 {
                schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
                sample_id: "wafer-002".into(),
                measurement_event_id: "meas-002".into(),
                round: 2,
                event_time_unix_seconds: 1_900_000_030,
                prediction_time_unix_seconds: 1_900_000_031,
                clock_domain_id: "fab-clock".into(),
                model_artifact_sha256: digest('4'),
                point_prediction: 10.0,
                interval: AciPredictionIntervalV1::full_real_line(),
            },
        )
        .unwrap();
        // The round-2 prediction is frozen before the round-1 label arrives.
        let first_update = deliver_delayed_aci_label_v1(&mut engine, &first_delivery).unwrap();

        begin_delayed_aci_round_v1(&mut engine, 3).unwrap();
        let degenerate = resolve_degenerate_aci_interval_v1(&mut engine, "wafer-002").unwrap();
        let missing = close_delayed_aci_stream_v1(&mut engine).unwrap();
        assert!(missing.is_empty());

        JointTauAciArmEvidenceV1 {
            policy: policy.clone(),
            interval_method_sha256: digest('6'),
            prediction_stream_closed: true,
            predictions: vec![first_prediction, second_prediction],
            deliveries: vec![first_delivery],
            updates: vec![first_update],
            missing_feedback: missing,
            degenerate_resolutions: vec![degenerate],
            guarantee_state: engine.guarantee_state,
        }
    }

    fn ca_ocrc_evidence(policy: CaOcrcPolicyV1) -> JointCaOcrcArmEvidenceV1 {
        let mut engine = create_ca_ocrc_engine_v1(
            "joint-ca-ocrc-engine",
            "fab-clock",
            &digest('4'),
            &digest('6'),
            0.5,
            policy.clone(),
        )
        .unwrap();
        let first_input = CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            prediction: CaOcrcPredictionInputV1 {
                schema: CA_OCRC_PREDICTION_SCHEMA_V1.into(),
                round: 1,
                sample_id: "wafer-001".into(),
                measurement_event_id: "meas-001".into(),
                event_time_unix_seconds: 1_900_000_000,
                prediction_at_unix_seconds: 1_900_000_001,
                round_time_unix_seconds: 1_900_000_005,
                clock_domain_id: "fab-clock".into(),
                global_model_artifact_sha256: digest('4'),
                interval_method_sha256: digest('6'),
                threshold_used: 0.5,
                prediction_set: CaOcrcPredictionSetV1 {
                    kind: CaOcrcSetKindV1::Interval,
                    lower: Some(9.0),
                    upper: Some(11.0),
                    measurand_id: "cd".into(),
                    unit: "nm".into(),
                },
                physical_measurement_selected: true,
                proxy_uniform_draw: Some(0.75),
            },
            feedbacks: Vec::new(),
        };
        let first_report = step_ca_ocrc_round_v1(&mut engine, &first_input).unwrap();

        let payload = digest('5');
        let second_input = CaOcrcRoundInputV1 {
            schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
            prediction: CaOcrcPredictionInputV1 {
                schema: CA_OCRC_PREDICTION_SCHEMA_V1.into(),
                round: 2,
                sample_id: "wafer-002".into(),
                measurement_event_id: "meas-002".into(),
                event_time_unix_seconds: 1_900_000_030,
                prediction_at_unix_seconds: 1_900_000_031,
                round_time_unix_seconds: 1_900_000_045,
                clock_domain_id: "fab-clock".into(),
                global_model_artifact_sha256: digest('4'),
                interval_method_sha256: digest('6'),
                threshold_used: engine.state.threshold,
                prediction_set: CaOcrcPredictionSetV1 {
                    kind: CaOcrcSetKindV1::Interval,
                    lower: Some(9.0),
                    upper: Some(11.0),
                    measurand_id: "cd".into(),
                    unit: "nm".into(),
                },
                physical_measurement_selected: false,
                proxy_uniform_draw: Some(0.75),
            },
            feedbacks: vec![CaOcrcLabelFeedbackV1 {
                schema: super::super::ca_ocrc_bounded::CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
                feedback_round: 2,
                sample_id: "wafer-001".into(),
                label_id: "label-001".into(),
                label_revision: 1,
                measurement_event_id: "meas-001".into(),
                measurand_id: "cd".into(),
                unit: "nm".into(),
                label_payload_sha256: payload.clone(),
                label_value: 12.0,
                label_available_at_unix_seconds: 1_900_000_010,
                eligibility_report_available_at_unix_seconds: 1_900_000_015,
                delivered_at_unix_seconds: 1_900_000_040,
                clock_domain_id: "fab-clock".into(),
                eligibility_report: eligibility_report(
                    "label-001",
                    1,
                    "meas-001",
                    &payload,
                    12.0,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                ),
            }],
        };
        let second_report = step_ca_ocrc_round_v1(&mut engine, &second_input).unwrap();
        let unresolved_tracking_rounds = close_ca_ocrc_feedback_horizon_v1(&mut engine).unwrap();
        JointCaOcrcArmEvidenceV1 {
            policy,
            round_inputs: vec![first_input, second_input],
            round_reports: vec![first_report, second_report],
            feedback_horizon_closed: true,
            unresolved_tracking_rounds,
        }
    }

    fn paired_fixture() -> JointRiskTraceInputV1 {
        let policy = DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "joint-fixture-tau-aci-v1".into(),
            target_miscoverage: 0.10,
            learning_rate: 0.20,
            delay_rounds: 1,
            initial_phase_alpha: 0.10,
        };
        let evidence = aci_evidence("joint-aci-engine", policy.clone());

        let population_manifest = digest('7');
        let source_ledger = digest('8');
        let selection = selection_input(
            &population_manifest,
            &source_ledger,
            &evidence.updates[0].receipt_sha256,
            0.0,
        );
        let arm = |arm_id: &str| JointBenchmarkArmV1 {
            arm_id: arm_id.into(),
            method: JointControllerMethodV1::FixedDelayTauAci,
            global_model_artifact_sha256: digest('4'),
            interval_method_sha256: digest('6'),
            controller_policy_sha256: json_sha256(&policy).unwrap(),
            acquisition_policy_sha256: digest('9'),
            physical_metrology_budget_units: 1.0,
            selection_risk_input: selection.clone(),
            controller_evidence: JointControllerEvidenceV1::FixedDelayTauAci(evidence.clone()),
        };

        let mut input = JointRiskTraceInputV1 {
            schema: JOINT_RISK_TRACE_SCHEMA_V1.into(),
            trace_id: "paired-joint-fixture-v1".into(),
            mode: JointBenchmarkModeV1::PairedSharedAcquisition,
            comparison_scope: JointComparisonScopeV1::ControllerAblation,
            population_manifest_sha256: population_manifest,
            source_ledger_sha256: source_ledger,
            population: vec![
                JointPopulationRowV1 {
                    round_id: 1,
                    sample_id: "wafer-001".into(),
                    measurement_event_id: "meas-001".into(),
                    event_time_unix_seconds: 1_900_000_000,
                    context_key: "eqp-01/ch-01".into(),
                    physical_measurement_selected: true,
                    physical_metrology_cost_units: 1.0,
                    outcome_state: PopulationOutcomeStateV1::Qualified,
                    label_id: Some("label-001".into()),
                    label_revision: Some(1),
                    label_payload_sha256: Some(digest('5')),
                    k1_report_sha256: Some(
                        evidence.deliveries[0]
                            .eligibility_report
                            .report_sha256
                            .clone(),
                    ),
                    label_value: Some(12.0),
                    label_available_at_unix_seconds: Some(1_900_000_010),
                    eligibility_report_available_at_unix_seconds: Some(1_900_000_015),
                    label_delivered_at_unix_seconds: Some(1_900_000_040),
                },
                JointPopulationRowV1 {
                    round_id: 2,
                    sample_id: "wafer-002".into(),
                    measurement_event_id: "meas-002".into(),
                    event_time_unix_seconds: 1_900_000_030,
                    context_key: "eqp-01/ch-01".into(),
                    physical_measurement_selected: false,
                    physical_metrology_cost_units: 0.0,
                    outcome_state: PopulationOutcomeStateV1::Unmeasured,
                    label_id: None,
                    label_revision: None,
                    label_payload_sha256: None,
                    k1_report_sha256: None,
                    label_value: None,
                    label_available_at_unix_seconds: None,
                    eligibility_report_available_at_unix_seconds: None,
                    label_delivered_at_unix_seconds: None,
                },
            ],
            arms: vec![arm("tau-aci-a"), arm("tau-aci-b")],
            trace_sha256: String::new(),
        };
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        input
    }

    #[test]
    fn paired_shared_acquisition_trace_replays_aci_and_population_audit() {
        let input = paired_fixture();
        let report = evaluate_joint_risk_trace_v1(&input).unwrap();
        assert_eq!(report.status, JointTraceStatusV1::Complete);
        assert_eq!(report.population_count, 2);
        assert_eq!(report.physical_measurement_count, 1);
        assert_eq!(report.qualified_count, 1);
        assert_eq!(report.unmeasured_count, 1);
        assert_eq!(report.independent_audit_count, 1);
        assert_eq!(report.arm_summaries.len(), 2);
        for summary in &report.arm_summaries {
            assert_eq!(summary.observed_miscoverage_count, Some(1));
            assert_eq!(summary.observed_miscoverage_count_denominator, Some(2));
            assert_eq!(summary.aci_degenerate_resolution_count, Some(1));
            assert_eq!(summary.aci_missing_feedback_count, Some(0));
            assert!(summary.independent_audit_population_estimate.is_some());
            assert!(summary.partial_identification_upper >= summary.partial_identification_lower);
        }
    }

    fn paired_mixed_controller_fixture() -> JointRiskTraceInputV1 {
        let mut input = paired_fixture();
        let ca_policy = CaOcrcPolicyV1 {
            schema: super::super::ca_ocrc_bounded::CA_OCRC_POLICY_SCHEMA_V1.into(),
            policy_id: "joint-fixture-ca-ocrc-v1".into(),
            revision: 1,
            target_risk_alpha: 0.10,
            learning_rate_eta: 0.05,
            bounded_score_upper: 1.0,
            maximum_actual_delay_steps: 1,
            tracking_capacity: 2,
            proxy_delay_pmf: vec![0.0, 0.5, 0.5],
            actual_delay_contract_sha256: digest('f'),
            bounded_actual_delay_assumption_declared: true,
        };
        let ca_evidence = ca_ocrc_evidence(ca_policy.clone());
        input.arms[1].method = JointControllerMethodV1::BoundedDelayCaOcrc;
        input.arms[1].controller_policy_sha256 = json_sha256(&ca_policy).unwrap();
        input.arms[1].controller_evidence =
            JointControllerEvidenceV1::BoundedDelayCaOcrc(ca_evidence);
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        input
    }

    #[test]
    fn mixed_tau_aci_and_ca_ocrc_arms_validate_on_one_shared_population_and_audit() {
        let input = paired_mixed_controller_fixture();
        let report = evaluate_joint_risk_trace_v1(&input).unwrap();
        assert_eq!(report.status, JointTraceStatusV1::Complete);
        assert_eq!(report.comparison_scope, JointComparisonScopeV1::ControllerAblation);
        assert_eq!(report.arm_summaries.len(), 2);
        let aci = report
            .arm_summaries
            .iter()
            .find(|summary| summary.method == JointControllerMethodV1::FixedDelayTauAci)
            .unwrap();
        let ca = report
            .arm_summaries
            .iter()
            .find(|summary| summary.method == JointControllerMethodV1::BoundedDelayCaOcrc)
            .unwrap();
        assert_eq!(aci.observed_miscoverage_count_denominator, Some(2));
        assert_eq!(ca.ca_ocrc_tracking_capacity, Some(2));
        assert_eq!(ca.ca_ocrc_updated_feedback_count, Some(1));
        assert_eq!(ca.ca_ocrc_theorem_assumptions_intact, Some(false));
        assert_eq!(report.independent_audit_count, 1);
    }

    #[test]
    fn controller_ablation_rejects_different_model_or_interval_artifacts() {
        let mut model_mismatch = paired_fixture();
        model_mismatch.arms[1].interval_method_sha256 = digest('a');
        if let JointControllerEvidenceV1::FixedDelayTauAci(evidence) =
            &mut model_mismatch.arms[1].controller_evidence
        {
            evidence.interval_method_sha256 = digest('a');
        }
        model_mismatch.trace_sha256 = seal_joint_risk_trace_v1(&model_mismatch).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&model_mismatch).is_err());

        let mut global_model_mismatch = paired_fixture();
        global_model_mismatch.arms[1].global_model_artifact_sha256 = digest('b');
        global_model_mismatch.trace_sha256 =
            seal_joint_risk_trace_v1(&global_model_mismatch).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&global_model_mismatch).is_err());
    }

    #[test]
    fn end_to_end_scope_allows_different_interval_methods_and_reports_scope() {
        let mut input = paired_fixture();
        input.comparison_scope = JointComparisonScopeV1::EndToEndPipeline;
        input.arms[1].interval_method_sha256 = digest('a');
        if let JointControllerEvidenceV1::FixedDelayTauAci(evidence) =
            &mut input.arms[1].controller_evidence
        {
            evidence.interval_method_sha256 = digest('a');
        }
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        let report = evaluate_joint_risk_trace_v1(&input).unwrap();
        assert_eq!(
            report.comparison_scope,
            JointComparisonScopeV1::EndToEndPipeline
        );
    }

    #[test]
    fn paired_trace_rejects_feedback_delivered_before_same_round_prediction() {
        let mut input = paired_fixture();
        input.population[0].label_delivered_at_unix_seconds = Some(1_900_000_030);
        for arm in &mut input.arms {
            if let JointControllerEvidenceV1::FixedDelayTauAci(evidence) =
                &mut arm.controller_evidence
            {
                evidence.deliveries[0].delivered_at_unix_seconds = 1_900_000_030;
            }
        }
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&input).is_err());
    }

    #[test]
    fn paired_trace_rejects_numeric_or_k1_row_mismatch() {
        let mut input = paired_fixture();
        input.population[0].label_value = Some(11.5);
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&input).is_err());
    }

    #[test]
    fn paired_trace_rejects_audit_loss_difference_between_arms() {
        let mut input = paired_fixture();
        input.arms[1].selection_risk_input.independent_audit_losses[0].loss = 1.0;
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&input).is_err());
    }

    #[test]
    fn paired_trace_rejects_selection_budget_overrun() {
        let mut input = paired_fixture();
        input.arms[0].physical_metrology_budget_units = 0.5;
        input.arms[1].physical_metrology_budget_units = 0.5;
        input.trace_sha256 = seal_joint_risk_trace_v1(&input).unwrap();
        assert!(evaluate_joint_risk_trace_v1(&input).is_err());
    }

    #[test]
    fn trace_digest_is_invariant_to_population_and_arm_order() {
        let input = paired_fixture();
        let first = seal_joint_risk_trace_v1(&input).unwrap();
        let mut permuted = input;
        permuted.population.reverse();
        permuted.arms.reverse();
        assert_eq!(first, seal_joint_risk_trace_v1(&permuted).unwrap());
    }

    #[test]
    fn unmeasured_row_cannot_carry_label_or_report_time() {
        let mut input = paired_fixture();
        input.population[1].label_delivered_at_unix_seconds = Some(1_900_000_099);
        assert!(validate_population(&input).is_err());
    }

    #[test]
    fn ca_feedback_input_binds_unknown_and_blocked_states_without_promoting_them() {
        let unknown = JointPopulationRowV1 {
            round_id: 3,
            sample_id: "wafer-003".into(),
            measurement_event_id: "meas-003".into(),
            event_time_unix_seconds: 1_900_000_100,
            context_key: "eqp-02/ch-01".into(),
            physical_measurement_selected: true,
            physical_metrology_cost_units: 1.0,
            outcome_state: PopulationOutcomeStateV1::Unknown,
            label_id: Some("label-003".into()),
            label_revision: Some(2),
            label_payload_sha256: Some(digest('a')),
            k1_report_sha256: None,
            label_value: None,
            label_available_at_unix_seconds: Some(1_900_000_100),
            eligibility_report_available_at_unix_seconds: Some(1_900_000_105),
            label_delivered_at_unix_seconds: Some(1_900_000_110),
        };
        let mut report = eligibility_report(
            "label-003",
            2,
            "meas-003",
            &digest('a'),
            13.0,
            LabelConsumptionEligibilityV1::Unknown,
        );
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        let feedback = CaOcrcLabelFeedbackV1 {
            schema: super::super::ca_ocrc_bounded::CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
            feedback_round: 4,
            sample_id: "wafer-003".into(),
            label_id: "label-003".into(),
            label_revision: 2,
            measurement_event_id: "meas-003".into(),
            measurand_id: "cd".into(),
            unit: "nm".into(),
            label_payload_sha256: digest('a'),
            label_value: 13.0,
            label_available_at_unix_seconds: 1_900_000_100,
            eligibility_report_available_at_unix_seconds: 1_900_000_105,
            delivered_at_unix_seconds: 1_900_000_110,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: report,
        };
        assert!(validate_ca_feedback_input(&feedback, 4, &unknown).is_ok());

        let mut blocked = unknown;
        blocked.outcome_state = PopulationOutcomeStateV1::Blocked;
        let mut blocked_feedback = feedback;
        blocked_feedback.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Blocked;
        blocked_feedback
            .eligibility_report
            .downstream_consumption_allowed = false;
        blocked_feedback.eligibility_report.report_sha256 =
            eligibility_report_digest(&blocked_feedback.eligibility_report).unwrap();
        blocked.k1_report_sha256 = Some(blocked_feedback.eligibility_report.report_sha256.clone());
        assert!(validate_ca_feedback_input(&blocked_feedback, 4, &blocked).is_ok());
    }
}
