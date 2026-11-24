// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuous-target population error estimation over the existing stratified
//! metrology audit design. Deterministic exploit rows are a census; residual
//! rows are sampled by stratified simple random sampling without replacement.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};
use super::metrology_audit::{
    MetrologyAuditSelectionReportV1, MetrologySelectionMethodV1, SELECTION_SCHEMA_V1,
};
use super::metrology_semantics::validate_binary_measurand_semantics_v1;
use super::process_event_graph::{
    MetrologyLabelStatusV1, PredictionInformationCutoffV1, ProcessEventGraphV1,
    build_information_cutoff_snapshot_v1,
};

pub const CONTINUOUS_AUDIT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.continuous-metrology-audit-input.v1";
pub const CONTINUOUS_AUDIT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.continuous-metrology-audit-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuousAuditOutcomeStateV1 {
    ReturnedQualified,
    ReturnedUnqualified,
    SelectedButNotReturned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuousAuditStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousMetrologyPredictionV1 {
    pub candidate_id: String,
    pub subject_id: String,
    pub stratum_id: String,
    pub measurement_event_id: String,
    pub target_measurand_id: String,
    pub unit: String,
    pub vm_prediction_value: f64,
    /// Frozen prediction of absolute error, used only for model-assisted residual correction.
    pub predicted_absolute_error: f64,
    pub prediction_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousMetrologyOutcomeV1 {
    pub candidate_id: String,
    pub state: ContinuousAuditOutcomeStateV1,
    pub label_id: Option<String>,
    pub eligibility_report: Option<LabelConsumptionEligibilityReportV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousMetrologyAuditInputV1 {
    pub schema: String,
    pub input_id: String,
    pub frame_id: String,
    pub frame_sha256: String,
    pub population_frame_complete: bool,
    pub target_measurand_id: String,
    pub unit: String,
    pub selection: MetrologyAuditSelectionReportV1,
    /// Exactly one immutable prediction for every candidate in the selection frame.
    pub predictions: Vec<ContinuousMetrologyPredictionV1>,
    /// Exactly one result state for every selected candidate. Unselected candidates have no outcome row.
    pub outcomes: Vec<ContinuousMetrologyOutcomeV1>,
    pub graph: ProcessEventGraphV1,
    pub cutoff: PredictionInformationCutoffV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousMetrologyStratumEstimateV1 {
    pub stratum_id: String,
    pub exploit_population: usize,
    pub residual_population: usize,
    pub audit_sample_count: usize,
    pub inclusion_probability: Option<f64>,
    pub exploit_absolute_error_total: f64,
    pub residual_absolute_error_total_ht: f64,
    pub residual_absolute_error_total_model_assisted: f64,
    pub residual_ht_variance: Option<f64>,
    pub residual_model_assisted_variance: Option<f64>,
    pub estimated_absolute_error_total_ht: f64,
    pub estimated_absolute_error_total_model_assisted: f64,
    pub variance_qualified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousMetrologyAuditReportV1 {
    pub schema: String,
    pub input_id: String,
    pub input_sha256: String,
    pub frame_id: String,
    pub frame_sha256: String,
    pub selection_sha256: String,
    pub graph_sha256: String,
    pub cutoff_id: String,
    pub target_measurand_id: String,
    pub unit: String,
    pub status: ContinuousAuditStatusV1,
    pub reason_codes: Vec<String>,
    pub population_count: usize,
    pub selected_count: usize,
    pub qualified_outcome_count: usize,
    pub unqualified_count: usize,
    pub nonresponse_count: usize,
    pub model_only_mean_absolute_error: Option<f64>,
    pub estimated_mean_absolute_error_ht: Option<f64>,
    pub standard_error_mean_absolute_error_ht: Option<f64>,
    pub estimated_mean_absolute_error_model_assisted: Option<f64>,
    pub standard_error_mean_absolute_error_model_assisted: Option<f64>,
    pub effective_sample_size: Option<f64>,
    pub maximum_inverse_probability_weight: Option<f64>,
    pub measurement_fraction: f64,
    pub strata: Vec<ContinuousMetrologyStratumEstimateV1>,
    pub claim_limits: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ContinuousMetrologyAuditErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), ContinuousMetrologyAuditErrorV1> {
    if value.trim().is_empty() {
        return Err(ContinuousMetrologyAuditErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ContinuousMetrologyAuditErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ContinuousMetrologyAuditErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

/// Computes the canonical digest for the complete frozen prediction frame.
pub fn compute_continuous_audit_frame_sha256_v1(
    frame_id: &str,
    target_measurand_id: &str,
    unit: &str,
    predictions: &[ContinuousMetrologyPredictionV1],
) -> Result<String, ContinuousMetrologyAuditErrorV1> {
    let mut ordered = predictions.to_vec();
    ordered.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    json_sha256(&(frame_id, target_measurand_id, unit, ordered.as_slice()))
}

fn selection_digest(
    selection: &MetrologyAuditSelectionReportV1,
) -> Result<String, ContinuousMetrologyAuditErrorV1> {
    let mut canonical = selection.clone();
    canonical.selection_sha256.clear();
    json_sha256(&canonical)
}

fn label_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, ContinuousMetrologyAuditErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn output_digest(
    report: &ContinuousMetrologyAuditReportV1,
) -> Result<String, ContinuousMetrologyAuditErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Some(
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64,
    )
}

fn unknown_report(
    input: &ContinuousMetrologyAuditInputV1,
    input_sha256: &str,
    selected_count: usize,
    qualified_count: usize,
    unqualified_count: usize,
    nonresponse_count: usize,
    reasons: Vec<String>,
) -> Result<ContinuousMetrologyAuditReportV1, ContinuousMetrologyAuditErrorV1> {
    let mut report = ContinuousMetrologyAuditReportV1 {
        schema: CONTINUOUS_AUDIT_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        input_sha256: input_sha256.into(),
        frame_id: input.frame_id.clone(),
        frame_sha256: input.frame_sha256.clone(),
        selection_sha256: input.selection.selection_sha256.clone(),
        graph_sha256: json_sha256(&input.graph)?,
        cutoff_id: input.cutoff.cutoff_id.clone(),
        target_measurand_id: input.target_measurand_id.clone(),
        unit: input.unit.clone(),
        status: ContinuousAuditStatusV1::Unknown,
        reason_codes: reasons,
        population_count: input.selection.population_count,
        selected_count,
        qualified_outcome_count: qualified_count,
        unqualified_count,
        nonresponse_count,
        model_only_mean_absolute_error: None,
        estimated_mean_absolute_error_ht: None,
        standard_error_mean_absolute_error_ht: None,
        estimated_mean_absolute_error_model_assisted: None,
        standard_error_mean_absolute_error_model_assisted: None,
        effective_sample_size: None,
        maximum_inverse_probability_weight: None,
        measurement_fraction: if input.selection.population_count > 0 {
            selected_count as f64 / input.selection.population_count as f64
        } else {
            0.0
        },
        strata: Vec::new(),
        claim_limits: vec![
            "Population error is not estimable when the candidate frame is incomplete or selected outcomes are missing/unqualified".into(),
            "The estimator assumes the logged deterministic-exploit plus stratified SRSWOR audit design".into(),
            "The report does not authenticate source issuers or establish physical traceability".into(),
        ],
        report_sha256: String::new(),
    };
    report.report_sha256 = output_digest(&report)?;
    Ok(report)
}

/// Estimates population mean absolute VM error using the existing exploit +
/// stratified-SRSWOR audit design. Missing or unqualified selected outcomes
/// return UNKNOWN; they never shrink the population denominator.
pub fn estimate_continuous_metrology_audit_v1(
    input: &ContinuousMetrologyAuditInputV1,
) -> Result<ContinuousMetrologyAuditReportV1, ContinuousMetrologyAuditErrorV1> {
    if input.schema != CONTINUOUS_AUDIT_INPUT_SCHEMA_V1 {
        return Err(ContinuousMetrologyAuditErrorV1(
            "continuous audit input schema is unsupported".into(),
        ));
    }
    for (field, value) in [
        ("input_id", input.input_id.as_str()),
        ("frame_id", input.frame_id.as_str()),
        ("target_measurand_id", input.target_measurand_id.as_str()),
        ("unit", input.unit.as_str()),
    ] {
        require_text(value, field)?;
    }
    if !valid_sha256(&input.frame_sha256) {
        return Err(ContinuousMetrologyAuditErrorV1(
            "population frame digest is malformed".into(),
        ));
    }
    let mut canonical = input.clone();
    canonical
        .predictions
        .sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    canonical
        .outcomes
        .sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    if compute_continuous_audit_frame_sha256_v1(
        &input.frame_id,
        &input.target_measurand_id,
        &input.unit,
        &canonical.predictions,
    )? != input.frame_sha256 {
        return Err(ContinuousMetrologyAuditErrorV1(
            "population frame digest does not match the frozen candidate prediction frame".into(),
        ));
    }
    let input_sha256 = json_sha256(&canonical)?;
    validate_binary_measurand_semantics_v1(&input.selection.binary_semantics)
        .map_err(|error| ContinuousMetrologyAuditErrorV1(format!(
            "selection policy semantics rejected: {error}"
        )))?;

    if input.selection.schema != SELECTION_SCHEMA_V1
        || !valid_sha256(&input.selection.selection_sha256)
        || selection_digest(&input.selection)? != input.selection.selection_sha256
    {
        return Err(ContinuousMetrologyAuditErrorV1(
            "selection schema/digest is invalid".into(),
        ));
    }
    if !input.population_frame_complete {
        return unknown_report(
            input,
            &input_sha256,
            0,
            0,
            0,
            0,
            vec!["population_frame_incomplete".into()],
        );
    }
    if input.selection.population_count == 0
        || input.selection.rows.len() != input.selection.population_count
        || input.predictions.len() != input.selection.population_count
    {
        return Err(ContinuousMetrologyAuditErrorV1(
            "selection, prediction and population frame counts do not reconcile".into(),
        ));
    }

    let mut prediction_by_id = BTreeMap::new();
    for prediction in &input.predictions {
        require_text(&prediction.candidate_id, "prediction.candidate_id")?;
        require_text(&prediction.subject_id, "prediction.subject_id")?;
        require_text(&prediction.stratum_id, "prediction.stratum_id")?;
        require_text(&prediction.measurement_event_id, "prediction.measurement_event_id")?;
        require_text(&prediction.target_measurand_id, "prediction.target_measurand_id")?;
        require_text(&prediction.unit, "prediction.unit")?;
        if prediction.target_measurand_id != input.target_measurand_id
            || prediction.unit != input.unit
        {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "candidate {} has a different measurand or unit",
                prediction.candidate_id
            )));
        }
        if !prediction.vm_prediction_value.is_finite()
            || !prediction.predicted_absolute_error.is_finite()
            || prediction.predicted_absolute_error < 0.0
            || !valid_sha256(&prediction.prediction_receipt_sha256)
        {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "candidate {} has invalid numeric prediction or receipt digest",
                prediction.candidate_id
            )));
        }
        if prediction_by_id
            .insert(prediction.candidate_id.as_str(), prediction)
            .is_some()
        {
            return Err(ContinuousMetrologyAuditErrorV1(
                "duplicate candidate prediction".into(),
            ));
        }
    }

    let selection_by_id: BTreeMap<&str, _> = input
        .selection
        .rows
        .iter()
        .map(|row| (row.candidate_id.as_str(), row))
        .collect();
    if selection_by_id.len() != input.selection.population_count {
        return Err(ContinuousMetrologyAuditErrorV1(
            "selection contains duplicate candidate IDs".into(),
        ));
    }
    for row in &input.selection.rows {
        let Some(prediction) = prediction_by_id.get(row.candidate_id.as_str()) else {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "candidate {} is missing its frozen VM prediction",
                row.candidate_id
            )));
        };
        if prediction.subject_id != row.subject_id || prediction.stratum_id != row.stratum_id
        {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "candidate {} prediction metadata differs from selection frame",
                row.candidate_id
            )));
        }
        let method_consistent = match &row.method {
            MetrologySelectionMethodV1::DeterministicExploit => {
                row.selected && (row.inclusion_probability - 1.0).abs() <= 1.0e-12
            }
            MetrologySelectionMethodV1::RandomizedAudit => row.selected,
            MetrologySelectionMethodV1::NotSelected => !row.selected,
        };
        if !row.inclusion_probability.is_finite()
            || row.inclusion_probability <= 0.0
            || row.inclusion_probability > 1.0
            || !method_consistent
        {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "candidate {} has invalid selection disposition or inclusion probability",
                row.candidate_id
            )));
        }
    }

    let mut design_by_stratum: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for row in &input.selection.rows {
        if row.method != MetrologySelectionMethodV1::DeterministicExploit {
            let counts = design_by_stratum.entry(row.stratum_id.as_str()).or_default();
            counts.0 += 1;
            if row.method == MetrologySelectionMethodV1::RandomizedAudit {
                counts.1 += 1;
            }
        }
    }
    for (stratum_id, (residual_count, audit_count)) in design_by_stratum {
        if audit_count == 0 || audit_count > residual_count {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "stratum {stratum_id} lacks positive randomized audit support"
            )));
        }
        let expected_probability = audit_count as f64 / residual_count as f64;
        for row in input.selection.rows.iter().filter(|row| {
            row.stratum_id == stratum_id
                && row.method != MetrologySelectionMethodV1::DeterministicExploit
        }) {
            if (row.inclusion_probability - expected_probability).abs() > 1.0e-12 {
                return Err(ContinuousMetrologyAuditErrorV1(format!(
                    "stratum {stratum_id} row propensity disagrees with the logged SRSWOR design"
                )));
            }
        }
    }

    let selected_rows: Vec<_> = input.selection.rows.iter().filter(|row| row.selected).collect();
    let selected_count = selected_rows.len();
    if input.outcomes.len() != selected_count {
        return Err(ContinuousMetrologyAuditErrorV1(
            "every selected candidate must have exactly one explicit returned/unqualified/nonresponse state".into(),
        ));
    }
    let mut outcome_by_id = BTreeMap::new();
    for outcome in &input.outcomes {
        if outcome_by_id.insert(outcome.candidate_id.as_str(), outcome).is_some() {
            return Err(ContinuousMetrologyAuditErrorV1(
                "duplicate selected outcome state".into(),
            ));
        }
        if !selection_by_id
            .get(outcome.candidate_id.as_str())
            .is_some_and(|row| row.selected)
        {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "outcome {} does not correspond to a selected candidate",
                outcome.candidate_id
            )));
        }
    }

    let snapshot = build_information_cutoff_snapshot_v1(&input.graph, &input.cutoff)
        .map_err(|error| ContinuousMetrologyAuditErrorV1(format!("maturity graph rejected: {error}")))?;
    let matured: BTreeSet<&str> = snapshot
        .matured_validated_label_ids
        .iter()
        .map(String::as_str)
        .collect();

    let mut qualified_values: BTreeMap<&str, f64> = BTreeMap::new();
    let mut nonresponse_count = 0usize;
    let mut unqualified_count = 0usize;
    let mut reason_codes = BTreeSet::new();

    for row in &selected_rows {
        let outcome = outcome_by_id[row.candidate_id.as_str()];
        match outcome.state {
            ContinuousAuditOutcomeStateV1::SelectedButNotReturned => {
                if outcome.label_id.is_some() || outcome.eligibility_report.is_some() {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} nonresponse state contains a label receipt",
                        row.candidate_id
                    )));
                }
                nonresponse_count += 1;
                reason_codes.insert("selected_outcome_not_returned".to_string());
            }
            ContinuousAuditOutcomeStateV1::ReturnedUnqualified => {
                let report = outcome.eligibility_report.as_ref().ok_or_else(|| {
                    ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} returned-unqualified state lacks an eligibility receipt",
                        row.candidate_id
                    ))
                })?;
                if outcome.label_id.as_deref() != Some(report.label_id.as_str())
                    || label_report_digest(report)? != report.report_sha256
                    || report.label_revision == 0
                    || !valid_sha256(&report.label_sha256)
                {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} returned-unqualified receipt identity/digest mismatch",
                        row.candidate_id
                    )));
                }
                if report.eligibility == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                    && report.frontier == LabelEvidenceFrontierV1::CurrentAtK1
                    && report.downstream_consumption_allowed
                {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} marked unqualified despite eligible current K1 receipt",
                        row.candidate_id
                    )));
                }
                unqualified_count += 1;
                reason_codes.insert("selected_outcome_not_k1_qualified".to_string());
            }
            ContinuousAuditOutcomeStateV1::ReturnedQualified => {
                let report = outcome.eligibility_report.as_ref().ok_or_else(|| {
                    ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} qualified state lacks an eligibility receipt",
                        row.candidate_id
                    ))
                })?;
                let prediction = prediction_by_id[row.candidate_id.as_str()];
                if outcome.label_id.as_deref() != Some(report.label_id.as_str())
                    || label_report_digest(report)? != report.report_sha256
                    || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
                    || report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                    || !report.downstream_consumption_allowed
                    || report.label_revision == 0
                    || !valid_sha256(&report.label_sha256)
                    || !valid_sha256(&report.qualification_report_sha256)
                    || !valid_sha256(&report.requalification_report_sha256)
                    || !valid_sha256(&report.requalification_frontier_sha256)
                    || !valid_sha256(&report.measurement_sha256)
                    || report.calibration_certificate_id.trim().is_empty()
                    || report.training_snapshot_mutated
                    || report.equipment_control_authority
                    || report.measurement_event_id != prediction.measurement_event_id
                {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} K1 numeric label receipt failed identity, eligibility or digest checks",
                        row.candidate_id
                    )));
                }
                if !matured.contains(report.label_id.as_str()) {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} label is not matured at the declared cutoff",
                        row.candidate_id
                    )));
                }
                let graph_labels: Vec<_> = input.graph.metrology_labels
                    .iter()
                    .filter(|label| label.label_id == report.label_id)
                    .collect();
                if graph_labels.len() != 1 {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} requires exactly one graph label",
                        row.candidate_id
                    )));
                }
                let graph_label = graph_labels[0];
                if graph_label.status != MetrologyLabelStatusV1::Validated
                    || json_sha256(graph_label)? != report.label_sha256
                    || graph_label.subject_id != prediction.subject_id
                    || graph_label.measurand_id != input.target_measurand_id
                    || graph_label.measurement_event_id != prediction.measurement_event_id
                    || graph_label.revision != report.label_revision
                    || graph_label.unit != input.unit
                    || graph_label.value != Some(report.qualified_corrected_value)
                {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} graph label does not match the qualified numeric report",
                        row.candidate_id
                    )));
                }
                if !report.qualified_corrected_value.is_finite()
                    || report.qualified_standard_uncertainty_upper_bound
                        .is_none_or(|value| !value.is_finite() || value < 0.0)
                {
                    return Err(ContinuousMetrologyAuditErrorV1(format!(
                        "candidate {} qualified value or uncertainty bound is invalid",
                        row.candidate_id
                    )));
                }
                qualified_values.insert(row.candidate_id.as_str(), report.qualified_corrected_value);
            }
        }
    }

    if nonresponse_count > 0 || unqualified_count > 0 {
        return unknown_report(
            input,
            &input_sha256,
            selected_count,
            qualified_values.len(),
            unqualified_count,
            nonresponse_count,
            reason_codes.into_iter().collect(),
        );
    }

    let mut strata_rows: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for row in &input.selection.rows {
        strata_rows.entry(row.stratum_id.clone()).or_default().push(row);
    }

    let mut strata = Vec::new();
    let mut total_ht = 0.0;
    let mut total_model_assisted = 0.0;
    let mut total_ht_variance = 0.0;
    let mut total_model_assisted_variance = 0.0;
    let mut variance_qualified = true;
    let mut model_only_total = 0.0;
    let mut sum_weights = 0.0;
    let mut sum_squared_weights = 0.0;
    let mut maximum_weight: f64 = 0.0;

    for row in &input.selection.rows {
        model_only_total += prediction_by_id[row.candidate_id.as_str()].predicted_absolute_error;
    }

    for (stratum_id, rows) in strata_rows {
        let exploit: Vec<_> = rows.iter().filter(|row| row.method == MetrologySelectionMethodV1::DeterministicExploit).collect();
        let residual: Vec<_> = rows.iter().filter(|row| row.method != MetrologySelectionMethodV1::DeterministicExploit).collect();
        let audit: Vec<_> = residual.iter().filter(|row| row.method == MetrologySelectionMethodV1::RandomizedAudit).collect();
        let residual_count = residual.len();
        let audit_count = audit.len();
        if residual_count > 0 && (audit_count == 0 || audit_count > residual_count) {
            return Err(ContinuousMetrologyAuditErrorV1(format!(
                "stratum {stratum_id}: audit positivity/support failure"
            )));
        }
        let inclusion_probability = (residual_count > 0).then_some(audit_count as f64 / residual_count as f64);
        for row in &residual {
            if (row.inclusion_probability - inclusion_probability.unwrap_or(1.0)).abs() > 1.0e-12 {
                return Err(ContinuousMetrologyAuditErrorV1(format!(
                    "stratum {stratum_id}: row inclusion probability disagrees with SRSWOR design"
                )));
            }
        }

        let mut exploit_total = 0.0;
        for row in &exploit {
            let loss = (qualified_values[row.candidate_id.as_str()]
                - prediction_by_id[row.candidate_id.as_str()].vm_prediction_value).abs();
            exploit_total += loss;
            sum_weights += 1.0;
            sum_squared_weights += 1.0;
            maximum_weight = maximum_weight.max(1.0);
        }

        let sampled_losses: Vec<f64> = audit
            .iter()
            .map(|row| {
                (qualified_values[row.candidate_id.as_str()]
                    - prediction_by_id[row.candidate_id.as_str()].vm_prediction_value).abs()
            })
            .collect();
        let sampled_residuals: Vec<f64> = audit
            .iter()
            .map(|row| {
                let prediction = prediction_by_id[row.candidate_id.as_str()];
                let loss = (qualified_values[row.candidate_id.as_str()]
                    - prediction.vm_prediction_value).abs();
                loss - prediction.predicted_absolute_error
            })
            .collect();

        let ht_residual_total = if residual_count == 0 {
            0.0
        } else {
            residual_count as f64 * sampled_losses.iter().sum::<f64>() / audit_count as f64
        };
        let model_total_residual: f64 = residual
            .iter()
            .map(|row| prediction_by_id[row.candidate_id.as_str()].predicted_absolute_error)
            .sum();
        let model_assisted_residual_total = if residual_count == 0 {
            0.0
        } else {
            model_total_residual
                + residual_count as f64 * sampled_residuals.iter().sum::<f64>() / audit_count as f64
        };
        let ht_variance = if residual_count == 0 || audit_count == residual_count {
            Some(0.0)
        } else {
            sample_variance(&sampled_losses).map(|variance| {
                (residual_count as f64).powi(2)
                    * (1.0 - audit_count as f64 / residual_count as f64)
                    * variance
                    / audit_count as f64
            })
        };
        let model_assisted_variance = if residual_count == 0 || audit_count == residual_count {
            Some(0.0)
        } else {
            sample_variance(&sampled_residuals).map(|variance| {
                (residual_count as f64).powi(2)
                    * (1.0 - audit_count as f64 / residual_count as f64)
                    * variance
                    / audit_count as f64
            })
        };
        let stratum_variance_qualified = ht_variance.is_some() && model_assisted_variance.is_some();
        if let Some(variance) = ht_variance { total_ht_variance += variance; } else { variance_qualified = false; }
        if let Some(variance) = model_assisted_variance { total_model_assisted_variance += variance; } else { variance_qualified = false; }

        total_ht += exploit_total + ht_residual_total;
        total_model_assisted += exploit_total + model_assisted_residual_total;
        for row in &exploit {
            let _ = row;
        }
        if residual_count > 0 {
            let weight = residual_count as f64 / audit_count as f64;
            for _ in &audit {
                sum_weights += weight;
                sum_squared_weights += weight * weight;
                maximum_weight = maximum_weight.max(weight);
            }
        }
        strata.push(ContinuousMetrologyStratumEstimateV1 {
            stratum_id,
            exploit_population: exploit.len(),
            residual_population: residual_count,
            audit_sample_count: audit_count,
            inclusion_probability,
            exploit_absolute_error_total: exploit_total,
            residual_absolute_error_total_ht: ht_residual_total,
            residual_absolute_error_total_model_assisted: model_assisted_residual_total,
            residual_ht_variance: ht_variance,
            residual_model_assisted_variance: model_assisted_variance,
            estimated_absolute_error_total_ht: exploit_total + ht_residual_total,
            estimated_absolute_error_total_model_assisted: exploit_total + model_assisted_residual_total,
            variance_qualified: stratum_variance_qualified,
        });
    }

    let population_count = input.selection.population_count as f64;
    let mut report = ContinuousMetrologyAuditReportV1 {
        schema: CONTINUOUS_AUDIT_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        input_sha256,
        frame_id: input.frame_id.clone(),
        frame_sha256: input.frame_sha256.clone(),
        selection_sha256: input.selection.selection_sha256.clone(),
        graph_sha256: snapshot.graph_sha256,
        cutoff_id: input.cutoff.cutoff_id.clone(),
        target_measurand_id: input.target_measurand_id.clone(),
        unit: input.unit.clone(),
        status: ContinuousAuditStatusV1::Complete,
        reason_codes: Vec::new(),
        population_count: input.selection.population_count,
        selected_count,
        qualified_outcome_count: qualified_values.len(),
        unqualified_count,
        nonresponse_count,
        model_only_mean_absolute_error: Some(model_only_total / population_count),
        estimated_mean_absolute_error_ht: Some(total_ht / population_count),
        standard_error_mean_absolute_error_ht: variance_qualified.then(|| total_ht_variance.sqrt() / population_count),
        estimated_mean_absolute_error_model_assisted: Some(total_model_assisted / population_count),
        standard_error_mean_absolute_error_model_assisted: variance_qualified.then(|| total_model_assisted_variance.sqrt() / population_count),
        effective_sample_size: (sum_squared_weights > 0.0).then_some(sum_weights * sum_weights / sum_squared_weights),
        maximum_inverse_probability_weight: (maximum_weight > 0.0).then_some(maximum_weight),
        measurement_fraction: selected_count as f64 / population_count,
        strata,
        claim_limits: vec![
            "Continuous HT and model-assisted totals use the logged stratified SRSWOR residual audit design".into(),
            "Model-assisted residual correction is design-unbiased only under the declared randomized audit design and complete frame".into(),
            "Selected but missing or unqualified outcomes return UNKNOWN; no nonresponse imputation is performed".into(),
            "The result is a population estimate in one declared measurand/unit, not selected-set MAE".into(),
            "This report does not authenticate issuers or authorize process control".into(),
        ],
        report_sha256: String::new(),
    };
    report.report_sha256 = output_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrology_audit::{
        MetrologyAuditSelectionRowV1, MetrologySelectionMethodV1,
    };
    use crate::benchmark::metrology_semantics::{
        BinaryMeasurandSemanticsV1, PositiveConditionV1, SEMANTICS_SCHEMA_V1,
        seal_binary_measurand_semantics_v1,
    };
    use crate::benchmark::process_event_graph::{
        ClockBasisV1, MetrologyLabelV1, ProcessEventKindV1, ProcessEventV1,
        GRAPH_SCHEMA_V1,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn semantics() -> BinaryMeasurandSemanticsV1 {
        let mut value = BinaryMeasurandSemanticsV1 {
            schema: SEMANTICS_SCHEMA_V1.into(),
            semantics_id: "risk-defect-v1".into(),
            revision: 1,
            target_measurand_id: "risk-defect".into(),
            unit: "binary".into(),
            positive_class_name: "defect".into(),
            negative_class_name: "pass".into(),
            positive_condition: PositiveConditionV1::GreaterThanOrEqual,
            threshold: 0.5,
            authority_ref: "quality-plan:risk-defect-v1".into(),
            semantics_sha256: String::new(),
        };
        seal_binary_measurand_semantics_v1(&mut value).unwrap();
        value
    }

    fn selection(full_census: bool) -> MetrologyAuditSelectionReportV1 {
        let semantics = semantics();
        let mut rows = Vec::new();
        for (index, stratum) in (0..8).map(|i| (i, if i < 4 { "tool-a" } else { "tool-b" })) {
            let (method, selected, probability) = if full_census {
                (MetrologySelectionMethodV1::DeterministicExploit, true, 1.0)
            } else if index == 0 || index == 4 {
                (MetrologySelectionMethodV1::DeterministicExploit, true, 1.0)
            } else if index == 1 || index == 2 || index == 5 || index == 6 {
                (MetrologySelectionMethodV1::RandomizedAudit, true, 2.0 / 3.0)
            } else {
                (MetrologySelectionMethodV1::NotSelected, false, 2.0 / 3.0)
            };
            rows.push(MetrologyAuditSelectionRowV1 {
                candidate_id: format!("candidate-{index:02}"),
                subject_id: format!("wafer-{index:02}"),
                stratum_id: stratum.into(),
                target_measurand_id: "risk-defect".into(),
                class_semantics_ref: semantics.semantics_id.clone(),
                class_semantics_sha256: semantics.semantics_sha256.clone(),
                predicted_defect_probability: 0.2,
                priority_score: 0.4,
                measurement_cost: 1,
                inclusion_probability: probability,
                selected,
                method,
            });
        }

        let mut report = MetrologyAuditSelectionReportV1 {
            schema: SELECTION_SCHEMA_V1.into(),
            policy_id: "selection-policy-v1".into(),
            model_sha256: digest('a'),
            prediction_packet_sha256: digest('b'),
            policy_config_sha256: digest('c'),
            binary_semantics: semantics,
            seed: 77,
            total_cost_budget: 8,
            exploit_cost_budget: 2,
            audit_cost_budget: if full_census { 0 } else { 4 },
            exploit_cost_used: if full_census { 8 } else { 2 },
            audit_cost_used: if full_census { 0 } else { 4 },
            audit_worst_case_cost: if full_census { 0 } else { 4 },
            total_selected_cost: if full_census { 8 } else { 6 },
            population_count: 8,
            exploit_count: if full_census { 8 } else { 2 },
            randomized_audit_count: if full_census { 0 } else { 4 },
            residual_population_by_stratum: if full_census {
                BTreeMap::new()
            } else {
                BTreeMap::from([("tool-a".into(), 3), ("tool-b".into(), 3)])
            },
            audit_quota_by_stratum: if full_census {
                BTreeMap::new()
            } else {
                BTreeMap::from([("tool-a".into(), 2), ("tool-b".into(), 2)])
            },
            rows,
            selection_sha256: String::new(),
        };
        report.selection_sha256 = selection_digest(&report).unwrap();
        report
    }

    fn predictions() -> Vec<ContinuousMetrologyPredictionV1> {
        (0..8)
            .map(|index| ContinuousMetrologyPredictionV1 {
                candidate_id: format!("candidate-{index:02}"),
                subject_id: format!("wafer-{index:02}"),
                stratum_id: if index < 4 { "tool-a" } else { "tool-b" }.into(),
                measurement_event_id: format!("event-{index:02}"),
                target_measurand_id: "film-thickness".into(),
                unit: "nm".into(),
                vm_prediction_value: 10.0,
                predicted_absolute_error: 1.0 + 0.25 * index as f64,
                prediction_receipt_sha256: digest(char::from(b'd' + (index % 3) as u8)),
            })
            .collect()
    }

    fn outcome_value(index: usize) -> f64 {
        match index {
            0 => 12.0,
            1 => 11.0,
            2 => 14.0,
            4 => 13.0,
            5 => 9.0,
            6 => 12.0,
            _ => 10.0,
        }
    }

    fn label(index: usize, value: f64) -> MetrologyLabelV1 {
        MetrologyLabelV1 {
            label_id: format!("label-{index:02}"),
            measurement_event_id: format!("event-{index:02}"),
            subject_id: format!("wafer-{index:02}"),
            measurand_id: "film-thickness".into(),
            value: Some(value),
            unit: "nm".into(),
            standard_uncertainty: Some(0.10),
            censoring_lower_bound: None,
            censoring_upper_bound: None,
            censoring_basis: None,
            status: MetrologyLabelStatusV1::Validated,
            measurement_time: "2026-11-22T08:00:00".into(),
            measurement_clock_domain_id: "fab-local".into(),
            measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
            available_at: "2026-11-22T09:00:00".into(),
            availability_clock_domain_id: "fab-local".into(),
            availability_clock_basis: ClockBasisV1::SourceLocalNaive,
            instrument_id: Some("ellipsometer-1".into()),
            calibration_ref: Some("cal-certificate-1".into()),
            revision: 1,
            supersedes_label_id: None,
            source_artifact_sha256: digest('e'),
            authority_ref: "lims:validated".into(),
        }
    }

    fn label_report(label: &MetrologyLabelV1) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: crate::benchmark::label_consumption_guard::LABEL_CONSUMPTION_REPORT_SCHEMA_V1.into(),
            label_id: label.label_id.clone(),
            label_revision: label.revision,
            measurement_event_id: label.measurement_event_id.clone(),
            calibration_certificate_id: "cal-certificate-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: json_sha256(label).unwrap(),
            measurement_sha256: digest('f'),
            qualified_corrected_value: label.value.unwrap(),
            qualified_standard_uncertainty_upper_bound: Some(0.10),
            qualification_report_sha256: digest('a'),
            requalification_report_sha256: digest('b'),
            requalification_frontier_sha256: digest('c'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = label_report_digest(&report).unwrap();
        report
    }

    fn graph_and_cutoff(full_census: bool) -> (ProcessEventGraphV1, PredictionInformationCutoffV1) {
        let selected: Vec<usize> = if full_census {
            (0..8).collect()
        } else {
            vec![0, 1, 2, 4, 5, 6]
        };
        let mut events = Vec::new();
        let mut labels = Vec::new();
        for (sequence, index) in selected.into_iter().enumerate() {
            let event_id = format!("event-{index:02}");
            events.push(ProcessEventV1 {
                event_id: event_id.clone(),
                event_kind: ProcessEventKindV1::Metrology,
                subject_id: Some(format!("wafer-{index:02}")),
                wafer_id: Some(format!("wafer-{index:02}")),
                lot_id: Some("lot-continuous-audit".into()),
                carrier_id: None,
                tool_id: Some(if index < 4 { "tool-a" } else { "tool-b" }.into()),
                chamber_id: Some("chamber-1".into()),
                module_id: None,
                process_step_id: Some("film-deposition".into()),
                recipe_id: Some("recipe-1".into()),
                recipe_revision: Some("r1".into()),
                event_time: "2026-11-22T08:00:00".into(),
                event_clock_domain_id: "fab-local".into(),
                event_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:00:01".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                source_sequence: Some(sequence as u64 + 1),
                source_artifact_sha256: digest('a'),
                transform_sha256: digest('b'),
                authority_ref: "metrology-system".into(),
            });
            labels.push(label(index, outcome_value(index)));
        }
        (
            ProcessEventGraphV1 {
                schema: GRAPH_SCHEMA_V1.into(),
                graph_id: "continuous-audit-graph-v1".into(),
                source_artifact_sha256: digest('c'),
                transform_sha256: digest('d'),
                events,
                edges: Vec::new(),
                metrology_labels: labels,
            },
            PredictionInformationCutoffV1 {
                cutoff_id: "continuous-audit-k1".into(),
                cutoff_time: "2026-11-22T10:00:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
        )
    }

    fn input(full_census: bool) -> ContinuousMetrologyAuditInputV1 {
        let selection = selection(full_census);
        let predictions = predictions();
        let frame_id = "film-thickness-frame-2026w47".to_string();
        let frame_sha256 = compute_continuous_audit_frame_sha256_v1(
            &frame_id,
            "film-thickness",
            "nm",
            &predictions,
        )
        .unwrap();
        let selected_indices: Vec<usize> = selection
            .rows
            .iter()
            .filter(|row| row.selected)
            .map(|row| {
                row.candidate_id
                    .strip_prefix("candidate-")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
            })
            .collect();
        let outcomes = selected_indices
            .iter()
            .map(|index| {
                let label = label(*index, outcome_value(*index));
                ContinuousMetrologyOutcomeV1 {
                    candidate_id: format!("candidate-{index:02}"),
                    state: ContinuousAuditOutcomeStateV1::ReturnedQualified,
                    label_id: Some(label.label_id.clone()),
                    eligibility_report: Some(label_report(&label)),
                }
            })
            .collect();
        let (graph, cutoff) = graph_and_cutoff(full_census);
        ContinuousMetrologyAuditInputV1 {
            schema: CONTINUOUS_AUDIT_INPUT_SCHEMA_V1.into(),
            input_id: "continuous-audit-input-v1".into(),
            frame_id,
            frame_sha256,
            population_frame_complete: true,
            target_measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            selection,
            predictions,
            outcomes,
            graph,
            cutoff,
        }
    }

    #[test]
    fn stratified_ht_and_model_assisted_estimators_match_known_answers() {
        let report = estimate_continuous_metrology_audit_v1(&input(false)).unwrap();
        assert_eq!(report.status, ContinuousAuditStatusV1::Complete);
        assert_eq!(report.population_count, 8);
        assert_eq!(report.selected_count, 6);
        assert_eq!(report.qualified_outcome_count, 6);
        assert!((report.estimated_mean_absolute_error_ht.unwrap() - 2.125).abs() < 1.0e-12);
        assert!(
            (report.estimated_mean_absolute_error_model_assisted.unwrap() - 2.21875).abs()
                < 1.0e-12
        );
        assert!((report.model_only_mean_absolute_error.unwrap() - 1.875).abs() < 1.0e-12);
        assert!(report.standard_error_mean_absolute_error_ht.unwrap() > 0.0);
        assert!(report.standard_error_mean_absolute_error_model_assisted.unwrap() > 0.0);
        assert_eq!(report.strata.len(), 2);
        assert!(valid_sha256(&report.report_sha256));
    }

    #[test]
    fn full_census_reproduces_exact_population_mae_with_zero_variance() {
        let report = estimate_continuous_metrology_audit_v1(&input(true)).unwrap();
        let exact = (0..8)
            .map(|index| (outcome_value(index) - 10.0).abs())
            .sum::<f64>()
            / 8.0;
        assert_eq!(report.status, ContinuousAuditStatusV1::Complete);
        assert!((report.estimated_mean_absolute_error_ht.unwrap() - exact).abs() < 1.0e-12);
        assert!((report.estimated_mean_absolute_error_model_assisted.unwrap() - exact).abs() < 1.0e-12);
        assert_eq!(report.standard_error_mean_absolute_error_ht, Some(0.0));
        assert_eq!(report.standard_error_mean_absolute_error_model_assisted, Some(0.0));
    }

    #[test]
    fn selected_but_unreturned_result_returns_unknown_without_denominator_shrinkage() {
        let mut input = input(false);
        let candidate = input
            .selection
            .rows
            .iter()
            .find(|row| row.method == MetrologySelectionMethodV1::RandomizedAudit)
            .unwrap()
            .candidate_id
            .clone();
        let outcome = input
            .outcomes
            .iter_mut()
            .find(|outcome| outcome.candidate_id == candidate)
            .unwrap();
        outcome.state = ContinuousAuditOutcomeStateV1::SelectedButNotReturned;
        outcome.label_id = None;
        outcome.eligibility_report = None;

        let report = estimate_continuous_metrology_audit_v1(&input).unwrap();
        assert_eq!(report.status, ContinuousAuditStatusV1::Unknown);
        assert_eq!(report.population_count, 8);
        assert_eq!(report.selected_count, 6);
        assert_eq!(report.nonresponse_count, 1);
        assert_eq!(report.estimated_mean_absolute_error_ht, None);
        assert!(report.reason_codes.contains(&"selected_outcome_not_returned".into()));
    }

    #[test]
    fn blocked_k1_label_is_nonresponse_not_an_observed_continuous_outcome() {
        let mut input = input(false);
        let outcome = input.outcomes.first_mut().unwrap();
        outcome.state = ContinuousAuditOutcomeStateV1::ReturnedUnqualified;
        let report = outcome.eligibility_report.as_mut().unwrap();
        report.eligibility = LabelConsumptionEligibilityV1::Blocked;
        report.downstream_consumption_allowed = false;
        report.report_sha256 = label_report_digest(report).unwrap();

        let estimate = estimate_continuous_metrology_audit_v1(&input).unwrap();
        assert_eq!(estimate.status, ContinuousAuditStatusV1::Unknown);
        assert_eq!(estimate.unqualified_count, 1);
        assert_eq!(estimate.estimated_mean_absolute_error_model_assisted, None);
    }

    #[test]
    fn missing_frame_or_invalid_audit_propensity_never_produces_population_estimate() {
        let mut incomplete = input(false);
        incomplete.population_frame_complete = false;
        let report = estimate_continuous_metrology_audit_v1(&incomplete).unwrap();
        assert_eq!(report.status, ContinuousAuditStatusV1::Unknown);
        assert!(report.reason_codes.contains(&"population_frame_incomplete".into()));

        let mut invalid = input(false);
        let audit = invalid
            .selection
            .rows
            .iter_mut()
            .find(|row| row.method == MetrologySelectionMethodV1::RandomizedAudit)
            .unwrap();
        audit.inclusion_probability = 0.0;
        invalid.selection.selection_sha256 = selection_digest(&invalid.selection).unwrap();
        assert!(estimate_continuous_metrology_audit_v1(&invalid).is_err());
    }

    #[test]
    fn unit_mismatch_and_tampered_label_payload_reject() {
        let mut wrong_unit = input(false);
        wrong_unit.predictions[0].unit = "angstrom".into();
        wrong_unit.frame_sha256 = compute_continuous_audit_frame_sha256_v1(
            &wrong_unit.frame_id,
            &wrong_unit.target_measurand_id,
            &wrong_unit.unit,
            &wrong_unit.predictions,
        )
        .unwrap();
        assert!(estimate_continuous_metrology_audit_v1(&wrong_unit).is_err());

        let mut tampered = input(false);
        let outcome = tampered.outcomes.first_mut().unwrap();
        outcome.eligibility_report.as_mut().unwrap().label_sha256 = digest('0');
        let report = outcome.eligibility_report.as_mut().unwrap();
        report.report_sha256 = label_report_digest(report).unwrap();
        assert!(estimate_continuous_metrology_audit_v1(&tampered).is_err());
    }

    #[test]
    fn fixed_frame_and_receipt_replay_is_deterministic() {
        let input = input(false);
        let first = estimate_continuous_metrology_audit_v1(&input).unwrap();
        let second = estimate_continuous_metrology_audit_v1(&input).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
        assert_eq!(first.estimated_mean_absolute_error_ht, second.estimated_mean_absolute_error_ht);
        assert_eq!(
            first.estimated_mean_absolute_error_model_assisted,
            second.estimated_mean_absolute_error_model_assisted
        );
    }
}
