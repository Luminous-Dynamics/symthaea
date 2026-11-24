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
        graph_sha256: input.graph.source_artifact_sha256.clone(),
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
    let frame_material = (
        input.frame_id.as_str(),
        input.target_measurand_id.as_str(),
        input.unit.as_str(),
        &canonical.predictions,
    );
    if json_sha256(&frame_material)? != input.frame_sha256 {
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
