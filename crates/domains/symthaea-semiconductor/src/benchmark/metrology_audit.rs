// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adaptive metrology selection with a stratified randomized audit lane.
//!
//! Deterministic exploit candidates are measured with inclusion probability 1.
//! Every remaining candidate is assigned a positive, known inclusion
//! probability under simple random sampling without replacement in its
//! source-defined stratum. Population estimates use the actual design.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::benchmark::metrology_semantics::{
    BinaryMeasurandSemanticsV1, classify_validated_metrology_label_v1,
    validate_binary_measurand_semantics_v1,
};
use crate::benchmark::process_event_graph::{
    PredictionInformationCutoffV1, ProcessEventGraphV1, build_information_cutoff_snapshot_v1,
};

pub const POLICY_SCHEMA_V1: &str = "luminous.semi-bench.metrology-audit-policy.v1";
pub const SELECTION_SCHEMA_V1: &str = "luminous.semi-bench.metrology-audit-selection.v1";
pub const ESTIMATE_SCHEMA_V1: &str = "luminous.semi-bench.metrology-audit-estimate.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditCandidateV1 {
    pub candidate_id: String,
    pub subject_id: String,
    /// Source-defined context such as tool/chamber/recipe/product/time block.
    pub stratum_id: String,
    pub target_measurand_id: String,
    pub predicted_defect_probability: f64,
    pub epistemic_uncertainty: f64,
    pub expected_value_of_information: f64,
    pub measurement_cost: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub model_sha256: String,
    pub prediction_packet_sha256: String,
    pub policy_config_sha256: String,
    pub binary_semantics: BinaryMeasurandSemanticsV1,
    pub total_cost_budget: u64,
    pub exploit_cost_budget: u64,
    pub audit_cost_budget: u64,
    pub risk_weight: f64,
    pub uncertainty_weight: f64,
    pub evsi_weight: f64,
    /// Fixed audit sample counts by source-defined residual stratum.
    pub audit_quota_by_stratum: BTreeMap<String, usize>,
    pub seed: u64,
    pub candidates: Vec<MetrologyAuditCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologySelectionMethodV1 {
    DeterministicExploit,
    RandomizedAudit,
    NotSelected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditSelectionRowV1 {
    pub candidate_id: String,
    pub subject_id: String,
    pub stratum_id: String,
    pub target_measurand_id: String,
    pub class_semantics_ref: String,
    pub class_semantics_sha256: String,
    pub predicted_defect_probability: f64,
    pub priority_score: f64,
    pub measurement_cost: u64,
    /// First-order inclusion probability under the logged selection design.
    pub inclusion_probability: f64,
    pub selected: bool,
    pub method: MetrologySelectionMethodV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditSelectionReportV1 {
    pub schema: String,
    pub policy_id: String,
    pub model_sha256: String,
    pub prediction_packet_sha256: String,
    pub policy_config_sha256: String,
    pub binary_semantics: BinaryMeasurandSemanticsV1,
    pub seed: u64,
    pub total_cost_budget: u64,
    pub exploit_cost_budget: u64,
    pub audit_cost_budget: u64,
    pub exploit_cost_used: u64,
    pub audit_cost_used: u64,
    pub audit_worst_case_cost: u64,
    pub total_selected_cost: u64,
    pub population_count: usize,
    pub exploit_count: usize,
    pub randomized_audit_count: usize,
    pub residual_population_by_stratum: BTreeMap<String, usize>,
    pub audit_quota_by_stratum: BTreeMap<String, usize>,
    pub rows: Vec<MetrologyAuditSelectionRowV1>,
    /// Hash of the full report with this field empty.
    pub selection_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditOutcomeV1 {
    pub candidate_id: String,
    pub label_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditStratumEstimateV1 {
    pub stratum_id: String,
    pub exploit_population: usize,
    pub residual_population: usize,
    pub audit_sample_count: usize,
    pub inclusion_probability: Option<f64>,
    pub estimated_positive_total: f64,
    pub estimated_brier_total: f64,
    pub positive_total_variance: Option<f64>,
    pub brier_total_variance: Option<f64>,
    pub variance_qualified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyAuditPopulationEstimateV1 {
    pub schema: String,
    pub selection_sha256: String,
    pub graph_sha256: String,
    pub maturity_cutoff_id: String,
    pub population_count: usize,
    pub measured_count: usize,
    pub measurement_fraction: f64,
    pub measurement_reduction_fraction: f64,
    pub estimated_positive_total_ht: f64,
    pub estimated_defect_prevalence_ht: f64,
    pub standard_error_defect_prevalence_ht: Option<f64>,
    pub estimated_brier_total_ht: f64,
    pub estimated_mean_brier_ht: f64,
    pub standard_error_mean_brier_ht: Option<f64>,
    pub estimated_defect_prevalence_hajek: f64,
    pub estimated_mean_brier_hajek: f64,
    pub effective_sample_size: f64,
    pub audit_outcome_count: usize,
    pub exploit_outcome_count: usize,
    pub strata: Vec<MetrologyAuditStratumEstimateV1>,
    pub claim_limits: Vec<String>,
    pub estimate_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct MetrologyAuditError(pub String);

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: usize) -> usize {
        let bound = upper_exclusive as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

fn shuffle<T>(values: &mut [T], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        values.swap(index, rng.below(index + 1));
    }
}

fn require_text(value: &str, field: &str) -> Result<(), MetrologyAuditError> {
    if value.trim().is_empty() {
        Err(MetrologyAuditError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<(), MetrologyAuditError> {
    if valid_sha256(value) {
        Ok(())
    } else {
        Err(MetrologyAuditError(format!(
            "{field}: expected lowercase SHA-256"
        )))
    }
}

fn finite_probability(value: f64, field: &str) -> Result<(), MetrologyAuditError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MetrologyAuditError(format!(
            "{field}: expected finite value in [0, 1]"
        )))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn report_digest(report: &MetrologyAuditSelectionReportV1) -> Result<String, MetrologyAuditError> {
    let mut material = report.clone();
    material.selection_sha256.clear();
    let bytes = serde_json::to_vec(&material)
        .map_err(|error| MetrologyAuditError(format!("selection serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn estimate_digest(
    report: &MetrologyAuditPopulationEstimateV1,
) -> Result<String, MetrologyAuditError> {
    let mut material = report.clone();
    material.estimate_sha256.clear();
    let bytes = serde_json::to_vec(&material)
        .map_err(|error| MetrologyAuditError(format!("estimate serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn validate_policy(policy: &MetrologyAuditPolicyV1) -> Result<(), MetrologyAuditError> {
    if policy.schema != POLICY_SCHEMA_V1 {
        return Err(MetrologyAuditError("policy.schema: unsupported".into()));
    }
    require_text(&policy.policy_id, "policy_id")?;
    require_sha256(&policy.model_sha256, "model_sha256")?;
    require_sha256(&policy.prediction_packet_sha256, "prediction_packet_sha256")?;
    require_sha256(&policy.policy_config_sha256, "policy_config_sha256")?;
    validate_binary_measurand_semantics_v1(&policy.binary_semantics).map_err(|error| {
        MetrologyAuditError(format!("binary metrology semantics rejected: {error}"))
    })?;
    if policy.total_cost_budget == 0
        || policy.exploit_cost_budget > policy.total_cost_budget
        || policy.audit_cost_budget > policy.total_cost_budget
        || policy
            .exploit_cost_budget
            .checked_add(policy.audit_cost_budget)
            .is_none_or(|sum| sum > policy.total_cost_budget)
    {
        return Err(MetrologyAuditError(
            "cost budgets: exploit + audit must fit within positive total budget".into(),
        ));
    }
    let weights = [
        policy.risk_weight,
        policy.uncertainty_weight,
        policy.evsi_weight,
    ];
    if weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1.0e-9
    {
        return Err(MetrologyAuditError(
            "priority weights must be finite, non-negative and sum to one".into(),
        ));
    }
    if policy.candidates.is_empty() {
        return Err(MetrologyAuditError("candidates: expected non-empty array".into()));
    }
    let mut candidate_ids = BTreeSet::new();
    for (index, candidate) in policy.candidates.iter().enumerate() {
        for (field, value) in [
            ("candidate_id", candidate.candidate_id.as_str()),
            ("subject_id", candidate.subject_id.as_str()),
            ("stratum_id", candidate.stratum_id.as_str()),
            ("target_measurand_id", candidate.target_measurand_id.as_str()),
        ] {
            require_text(value, &format!("candidates[{index}].{field}"))?;
        }
        if candidate.target_measurand_id != policy.binary_semantics.target_measurand_id {
            return Err(MetrologyAuditError(format!(
                "candidates[{index}].target_measurand_id differs from frozen binary semantics"
            )));
        }
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(MetrologyAuditError(format!(
                "candidates[{index}].candidate_id: duplicate identity"
            )));
        }
        finite_probability(
            candidate.predicted_defect_probability,
            &format!("candidates[{index}].predicted_defect_probability"),
        )?;
        finite_probability(
            candidate.epistemic_uncertainty,
            &format!("candidates[{index}].epistemic_uncertainty"),
        )?;
        finite_probability(
            candidate.expected_value_of_information,
            &format!("candidates[{index}].expected_value_of_information"),
        )?;
        if candidate.measurement_cost == 0 {
            return Err(MetrologyAuditError(format!(
                "candidates[{index}].measurement_cost: must be positive"
            )));
        }
    }
    Ok(())
}

fn stratum_seed(seed: u64, stratum_id: &str) -> u64 {
    let mut digest = Sha256::new();
    digest.update(seed.to_le_bytes());
    digest.update(stratum_id.as_bytes());
    let bytes = digest.finalize();
    u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 prefix has eight bytes"))
}

/// Selects deterministic high-utility candidates first, then performs a
/// seeded simple random sample without replacement in every residual stratum.
/// The audit quota is fixed before any outcome is observed.
pub fn select_metrology_audit_v1(
    policy: &MetrologyAuditPolicyV1,
) -> Result<MetrologyAuditSelectionReportV1, MetrologyAuditError> {
    validate_policy(policy)?;
    let mut candidates = policy.candidates.clone();
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let priority = |candidate: &MetrologyAuditCandidateV1| {
        policy.risk_weight * candidate.predicted_defect_probability
            + policy.uncertainty_weight * candidate.epistemic_uncertainty
            + policy.evsi_weight * candidate.expected_value_of_information
    };

    let mut exploit_order: Vec<usize> = (0..candidates.len()).collect();
    exploit_order.sort_by(|left, right| {
        let left_candidate = &candidates[*left];
        let right_candidate = &candidates[*right];
        let left_ratio = priority(left_candidate) / left_candidate.measurement_cost as f64;
        let right_ratio = priority(right_candidate) / right_candidate.measurement_cost as f64;
        right_ratio
            .total_cmp(&left_ratio)
            .then_with(|| left_candidate.candidate_id.cmp(&right_candidate.candidate_id))
    });

    let mut selected_method = vec![MetrologySelectionMethodV1::NotSelected; candidates.len()];
    let mut exploit_cost_used = 0u64;
    for index in exploit_order {
        let cost = candidates[index].measurement_cost;
        if priority(&candidates[index]) <= 0.0 {
            continue;
        }
        if let Some(next) = exploit_cost_used.checked_add(cost)
            && next <= policy.exploit_cost_budget
        {
            selected_method[index] = MetrologySelectionMethodV1::DeterministicExploit;
            exploit_cost_used = next;
        }
    }

    let mut residual_by_stratum: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if matches!(
            &selected_method[index],
            MetrologySelectionMethodV1::NotSelected
        ) {
            residual_by_stratum
                .entry(candidate.stratum_id.clone())
                .or_default()
                .push(index);
        }
    }
    let residual_population_by_stratum: BTreeMap<String, usize> = residual_by_stratum
        .iter()
        .map(|(stratum, indices)| (stratum.clone(), indices.len()))
        .collect();
    let residual_strata: BTreeSet<&str> = residual_by_stratum.keys().map(String::as_str).collect();
    let quota_strata: BTreeSet<&str> = policy
        .audit_quota_by_stratum
        .keys()
        .map(String::as_str)
        .collect();
    if residual_strata != quota_strata {
        return Err(MetrologyAuditError(
            "audit_quota_by_stratum must exactly cover the non-empty residual strata".into(),
        ));
    }

    let mut audit_worst_case_cost = 0u64;
    let mut audit_cost_used = 0u64;
    let mut randomized_audit_count = 0usize;
    let mut audit_quota_by_stratum = BTreeMap::new();
    for (stratum, indices) in &residual_by_stratum {
        let quota = policy.audit_quota_by_stratum[stratum];
        if quota == 0 || quota > indices.len() {
            return Err(MetrologyAuditError(format!(
                "stratum {stratum}: audit quota must be in [1, residual population]"
            )));
        }
        let maximum_cost = indices
            .iter()
            .map(|index| candidates[*index].measurement_cost)
            .max()
            .unwrap_or(0);
        let stratum_worst_case = maximum_cost
            .checked_mul(quota as u64)
            .ok_or_else(|| MetrologyAuditError(format!("stratum {stratum}: audit cost overflow")))?;
        audit_worst_case_cost = audit_worst_case_cost
            .checked_add(stratum_worst_case)
            .ok_or_else(|| MetrologyAuditError("audit cost overflow".into()))?;
        audit_quota_by_stratum.insert(stratum.clone(), quota);
        randomized_audit_count += quota;
    }
    if audit_worst_case_cost > policy.audit_cost_budget {
        return Err(MetrologyAuditError(
            "worst-case stratified audit cost exceeds audit_cost_budget".into(),
        ));
    }

    for (stratum, indices) in &residual_by_stratum {
        let quota = policy.audit_quota_by_stratum[stratum];
        let mut draw = indices.clone();
        let mut rng = SplitMix64::new(stratum_seed(policy.seed, stratum));
        shuffle(&mut draw, &mut rng);
        for index in draw.into_iter().take(quota) {
            selected_method[index] = MetrologySelectionMethodV1::RandomizedAudit;
            audit_cost_used = audit_cost_used
                .checked_add(candidates[index].measurement_cost)
                .ok_or_else(|| MetrologyAuditError("realized audit cost overflow".into()))?;
        }
    }

    let mut rows = Vec::with_capacity(candidates.len());
    let mut total_selected_cost = 0u64;
    let mut exploit_count = 0usize;
    for (index, candidate) in candidates.iter().enumerate() {
        let method = selected_method[index].clone();
        let selected = !matches!(&method, MetrologySelectionMethodV1::NotSelected);
        let inclusion_probability = match &method {
            MetrologySelectionMethodV1::DeterministicExploit => {
                exploit_count += 1;
                total_selected_cost = total_selected_cost
                    .checked_add(candidate.measurement_cost)
                    .ok_or_else(|| MetrologyAuditError("total selected cost overflow".into()))?;
                1.0
            }
            MetrologySelectionMethodV1::RandomizedAudit => {
                total_selected_cost = total_selected_cost
                    .checked_add(candidate.measurement_cost)
                    .ok_or_else(|| MetrologyAuditError("total selected cost overflow".into()))?;
                let population = residual_population_by_stratum[&candidate.stratum_id];
                let quota = audit_quota_by_stratum[&candidate.stratum_id];
                quota as f64 / population as f64
            }
            MetrologySelectionMethodV1::NotSelected => {
                let population = residual_population_by_stratum[&candidate.stratum_id];
                let quota = audit_quota_by_stratum[&candidate.stratum_id];
                quota as f64 / population as f64
            }
        };
        if !(0.0..=1.0).contains(&inclusion_probability) || inclusion_probability == 0.0 {
            return Err(MetrologyAuditError(format!(
                "candidate {}: invalid inclusion probability",
                candidate.candidate_id
            )));
        }
        rows.push(MetrologyAuditSelectionRowV1 {
            candidate_id: candidate.candidate_id.clone(),
            subject_id: candidate.subject_id.clone(),
            stratum_id: candidate.stratum_id.clone(),
            target_measurand_id: candidate.target_measurand_id.clone(),
            class_semantics_ref: policy.binary_semantics.semantics_id.clone(),
            class_semantics_sha256: policy.binary_semantics.semantics_sha256.clone(),
            predicted_defect_probability: candidate.predicted_defect_probability,
            priority_score: priority(candidate),
            measurement_cost: candidate.measurement_cost,
            inclusion_probability,
            selected,
            method,
        });
    }
    if exploit_cost_used > policy.exploit_cost_budget
        || audit_cost_used > policy.audit_cost_budget
        || total_selected_cost > policy.total_cost_budget
    {
        return Err(MetrologyAuditError(
            "realized selection cost exceeds a declared budget".into(),
        ));
    }
    let mut report = MetrologyAuditSelectionReportV1 {
        schema: SELECTION_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        model_sha256: policy.model_sha256.clone(),
        prediction_packet_sha256: policy.prediction_packet_sha256.clone(),
        policy_config_sha256: policy.policy_config_sha256.clone(),
        binary_semantics: policy.binary_semantics.clone(),
        seed: policy.seed,
        total_cost_budget: policy.total_cost_budget,
        exploit_cost_budget: policy.exploit_cost_budget,
        audit_cost_budget: policy.audit_cost_budget,
        exploit_cost_used,
        audit_cost_used,
        audit_worst_case_cost,
        total_selected_cost,
        population_count: candidates.len(),
        exploit_count,
        randomized_audit_count,
        residual_population_by_stratum,
        audit_quota_by_stratum,
        rows,
        selection_sha256: String::new(),
    };
    report.selection_sha256 = report_digest(&report)?;
    Ok(report)
}

fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squared = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    Some(sum_squared / (values.len() - 1) as f64)
}

/// Estimates population defect prevalence and model Brier loss. All selected
/// outcomes must be matured/validated in a freshly rebuilt graph snapshot.
/// The deterministic exploit component is treated as a census; the residual
/// component is estimated using stratified SRSWOR design weights.
pub fn estimate_metrology_population_v1(
    selection: &MetrologyAuditSelectionReportV1,
    outcomes: &[MetrologyAuditOutcomeV1],
    graph: &ProcessEventGraphV1,
    cutoff: &PredictionInformationCutoffV1,
) -> Result<MetrologyAuditPopulationEstimateV1, MetrologyAuditError> {
    if selection.schema != SELECTION_SCHEMA_V1 || !valid_sha256(&selection.selection_sha256) {
        return Err(MetrologyAuditError("selection schema/digest invalid".into()));
    }
    if report_digest(selection)? != selection.selection_sha256 {
        return Err(MetrologyAuditError("selection digest does not match payload".into()));
    }
    if outcomes.len() != selection.rows.iter().filter(|row| row.selected).count() {
        return Err(MetrologyAuditError(
            "every selected candidate must have exactly one matured outcome".into(),
        ));
    }
    let snapshot = build_information_cutoff_snapshot_v1(graph, cutoff)
        .map_err(|error| MetrologyAuditError(format!("maturity graph rejected: {error}")))?;
    let matured: BTreeSet<&str> = snapshot
        .matured_validated_label_ids
        .iter()
        .map(String::as_str)
        .collect();
    let graph_labels: BTreeMap<&str, _> = graph
        .metrology_labels
        .iter()
        .map(|label| (label.label_id.as_str(), label))
        .collect();
    let selection_by_id: BTreeMap<&str, &MetrologyAuditSelectionRowV1> = selection
        .rows
        .iter()
        .map(|row| (row.candidate_id.as_str(), row))
        .collect();
    let mut outcome_by_candidate = BTreeMap::new();
    let mut used_label_ids = BTreeSet::new();
    let mut used_candidate_ids = BTreeSet::new();
    for (index, outcome) in outcomes.iter().enumerate() {
        require_text(&outcome.candidate_id, &format!("outcomes[{index}].candidate_id"))?;
        require_text(&outcome.label_id, &format!("outcomes[{index}].label_id"))?;
        require_text(
            &outcome.class_semantics_ref,
            &format!("outcomes[{index}].class_semantics_ref"),
        )?;
        require_sha256(
            &outcome.class_semantics_sha256,
            &format!("outcomes[{index}].class_semantics_sha256"),
        )?;
        let selected = selection_by_id
            .get(outcome.candidate_id.as_str())
            .ok_or_else(|| MetrologyAuditError(format!(
                "outcomes[{index}].candidate_id does not exist in selection"
            )))?;
        if !selected.selected {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}] refers to an unselected candidate"
            )));
        }
        if !used_candidate_ids.insert(outcome.candidate_id.as_str()) {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}].candidate_id: duplicate outcome"
            )));
        }
        if !matured.contains(outcome.label_id.as_str()) {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}].label_id is not matured and validated"
            )));
        }
        if !used_label_ids.insert(outcome.label_id.as_str()) {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}].label_id: duplicate label"
            )));
        }
        if outcome.class_semantics_ref != selected.class_semantics_ref
            || outcome.class_semantics_sha256 != selected.class_semantics_sha256
        {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}] class-semantics mapping mismatch"
            )));
        }
        let label = graph_labels
            .get(outcome.label_id.as_str())
            .ok_or_else(|| MetrologyAuditError(format!(
                "outcomes[{index}].label_id is absent from graph"
            )))?;
        if !matches!(&label.status, MetrologyLabelStatusV1::Validated)
            || label.subject_id != selected.subject_id
            || label.measurand_id != selected.target_measurand_id
        {
            return Err(MetrologyAuditError(format!(
                "outcomes[{index}] graph label subject/measurand/status mismatch"
            )));
        }
        outcome_by_candidate.insert(outcome.candidate_id.as_str(), outcome);
    }

    let mut stratum_map: BTreeMap<String, Vec<&MetrologyAuditSelectionRowV1>> = BTreeMap::new();
    for row in &selection.rows {
        stratum_map
            .entry(row.stratum_id.clone())
            .or_default()
            .push(row);
    }
    let mut strata = Vec::new();
    let mut total_positive = 0.0;
    let mut total_brier = 0.0;
    let mut total_positive_variance = 0.0;
    let mut total_brier_variance = 0.0;
    let mut variance_qualified = true;
    let mut weighted_positive = 0.0;
    let mut weighted_brier = 0.0;
    let mut sum_weights = 0.0;
    let mut sum_squared_weights = 0.0;
    let mut exploit_outcome_count = 0usize;
    let mut audit_outcome_count = 0usize;

    for (stratum_id, rows) in stratum_map {
        let exploit: Vec<_> = rows
            .iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::DeterministicExploit))
            .collect();
        let residual: Vec<_> = rows
            .iter()
            .filter(|row| !matches!(&row.method, MetrologySelectionMethodV1::DeterministicExploit))
            .collect();
        let audit: Vec<_> = residual
            .iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit))
            .collect();
        let residual_count = residual.len();
        let audit_count = audit.len();
        if residual_count > 0 && (audit_count == 0 || audit_count > residual_count) {
            return Err(MetrologyAuditError(format!(
                "stratum {stratum_id}: randomized audit positivity/support failure"
            )));
        }
        let inclusion_probability = if residual_count > 0 {
            Some(audit_count as f64 / residual_count as f64)
        } else {
            None
        };

        let mut exploit_positive = 0.0;
        let mut exploit_brier = 0.0;
        let mut audit_positive = Vec::with_capacity(audit_count);
        let mut audit_brier = Vec::with_capacity(audit_count);

        for row in exploit {
            let outcome = outcome_by_candidate[row.candidate_id.as_str()];
            let positive = if outcome.positive { 1.0 } else { 0.0 };
            let brier = (row.predicted_defect_probability - positive).powi(2);
            exploit_positive += positive;
            exploit_brier += brier;
            let weight = 1.0;
            weighted_positive += positive * weight;
            weighted_brier += brier * weight;
            sum_weights += weight;
            sum_squared_weights += weight * weight;
            exploit_outcome_count += 1;
        }
        for row in &audit {
            let outcome = outcome_by_candidate[row.candidate_id.as_str()];
            let positive = if outcome.positive { 1.0 } else { 0.0 };
            let brier = (row.predicted_defect_probability - positive).powi(2);
            audit_positive.push(positive);
            audit_brier.push(brier);
            let weight = 1.0 / row.inclusion_probability;
            weighted_positive += positive * weight;
            weighted_brier += brier * weight;
            sum_weights += weight;
            sum_squared_weights += weight * weight;
            audit_outcome_count += 1;
        }

        let residual_positive_total = if residual_count == 0 {
            0.0
        } else {
            residual_count as f64 * (audit_positive.iter().sum::<f64>() / audit_count as f64)
        };
        let residual_brier_total = if residual_count == 0 {
            0.0
        } else {
            residual_count as f64 * (audit_brier.iter().sum::<f64>() / audit_count as f64)
        };
        let positive_variance = if residual_count == 0 || audit_count == residual_count {
            Some(0.0)
        } else {
            sample_variance(&audit_positive).map(|sample_var| {
                (residual_count as f64).powi(2)
                    * (1.0 - audit_count as f64 / residual_count as f64)
                    * sample_var
                    / audit_count as f64
            })
        };
        let brier_variance = if residual_count == 0 || audit_count == residual_count {
            Some(0.0)
        } else {
            sample_variance(&audit_brier).map(|sample_var| {
                (residual_count as f64).powi(2)
                    * (1.0 - audit_count as f64 / residual_count as f64)
                    * sample_var
                    / audit_count as f64
            })
        };
        let stratum_variance_qualified =
            positive_variance.is_some() && brier_variance.is_some();
        if let Some(variance) = positive_variance {
            total_positive_variance += variance;
        } else {
            variance_qualified = false;
        }
        if let Some(variance) = brier_variance {
            total_brier_variance += variance;
        } else {
            variance_qualified = false;
        }
        let estimated_positive_total = exploit_positive + residual_positive_total;
        let estimated_brier_total = exploit_brier + residual_brier_total;
        total_positive += estimated_positive_total;
        total_brier += estimated_brier_total;
        strata.push(MetrologyAuditStratumEstimateV1 {
            stratum_id,
            exploit_population: rows.len() - residual_count,
            residual_population: residual_count,
            audit_sample_count: audit_count,
            inclusion_probability,
            estimated_positive_total,
            estimated_brier_total,
            positive_total_variance: positive_variance,
            brier_total_variance: brier_variance,
            variance_qualified: stratum_variance_qualified,
        });
    }
    let population_count = selection.population_count;
    let measured_count = selection.rows.iter().filter(|row| row.selected).count();
    if population_count == 0 || measured_count != outcomes.len() {
        return Err(MetrologyAuditError(
            "selection/outcome population support does not reconcile".into(),
        ));
    }
    let estimated_defect_prevalence_ht = total_positive / population_count as f64;
    let estimated_mean_brier_ht = total_brier / population_count as f64;
    let estimated_defect_prevalence_hajek = weighted_positive / sum_weights;
    let estimated_mean_brier_hajek = weighted_brier / sum_weights;
    let mut estimate = MetrologyAuditPopulationEstimateV1 {
        schema: ESTIMATE_SCHEMA_V1.into(),
        selection_sha256: selection.selection_sha256.clone(),
        graph_sha256: snapshot.graph_sha256,
        maturity_cutoff_id: cutoff.cutoff_id.clone(),
        population_count,
        measured_count,
        measurement_fraction: measured_count as f64 / population_count as f64,
        measurement_reduction_fraction: 1.0 - measured_count as f64 / population_count as f64,
        estimated_positive_total_ht: total_positive,
        estimated_defect_prevalence_ht,
        standard_error_defect_prevalence_ht: variance_qualified
            .then(|| total_positive_variance.sqrt() / population_count as f64),
        estimated_brier_total_ht: total_brier,
        estimated_mean_brier_ht,
        standard_error_mean_brier_ht: variance_qualified
            .then(|| total_brier_variance.sqrt() / population_count as f64),
        estimated_defect_prevalence_hajek,
        estimated_mean_brier_hajek,
        effective_sample_size: if sum_squared_weights > 0.0 {
            sum_weights.powi(2) / sum_squared_weights
        } else {
            0.0
        },
        audit_outcome_count,
        exploit_outcome_count,
        strata,
        claim_limits: vec![
            "Horvitz-Thompson totals are design-based under the logged stratified SRSWOR audit design".into(),
            "Deterministic exploit outcomes are treated as a census conditional on observed policy inputs".into(),
            "Class semantics must be independently qualified and is only digest-bound by this report".into(),
            "Variance estimates require at least two randomized audit outcomes in every non-census residual stratum".into(),
            "The estimate does not authorize process control or establish causal effects of measurement policy".into(),
        ],
        estimate_sha256: String::new(),
    };
    estimate.estimate_sha256 = estimate_digest(&estimate)?;
    Ok(estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::process_event_graph::{
        ClockBasisV1, ProcessEventKindV1, ProcessEventV1, MetrologyLabelV1,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn candidate(index: usize, stratum: &str) -> MetrologyAuditCandidateV1 {
        MetrologyAuditCandidateV1 {
            candidate_id: format!("candidate-{index:02}"),
            subject_id: format!("wafer-{index:02}"),
            stratum_id: stratum.into(),
            target_measurand_id: "defect-rate".into(),
            class_semantics_ref: "defect-threshold-v1".into(),
            class_semantics_sha256: SHA_C.into(),
            predicted_defect_probability: (index % 5) as f64 / 5.0,
            epistemic_uncertainty: 0.2 + (index % 3) as f64 * 0.1,
            expected_value_of_information: 0.3 + (index % 4) as f64 * 0.1,
            measurement_cost: 1,
        }
    }

    fn policy() -> MetrologyAuditPolicyV1 {
        let candidates = (0..12)
            .map(|index| candidate(index, if index < 6 { "tool-a" } else { "tool-b" }))
            .collect();
        MetrologyAuditPolicyV1 {
            schema: POLICY_SCHEMA_V1.into(),
            policy_id: "audit-policy-v1".into(),
            model_sha256: SHA_A.into(),
            prediction_packet_sha256: SHA_B.into(),
            policy_config_sha256: SHA_C.into(),
            total_cost_budget: 8,
            exploit_cost_budget: 2,
            audit_cost_budget: 6,
            risk_weight: 0.4,
            uncertainty_weight: 0.3,
            evsi_weight: 0.3,
            audit_quota_by_stratum: BTreeMap::from([
                ("tool-a".into(), 3),
                ("tool-b".into(), 3),
            ]),
            seed: 77,
            candidates,
        }
    }

    #[test]
    fn randomized_audit_gives_positive_known_propensity_to_every_residual_candidate() {
        let report = select_metrology_audit_v1(&policy()).unwrap();
        for row in &report.rows {
            assert!(row.inclusion_probability > 0.0 && row.inclusion_probability <= 1.0);
            if matches!(&row.method, MetrologySelectionMethodV1::DeterministicExploit) {
                assert_eq!(row.inclusion_probability, 1.0);
            }
        }
        assert_eq!(report.randomized_audit_count, 6);
        assert_eq!(report.total_selected_cost, 8);
        assert_eq!(report.selection_sha256, report_digest(&report).unwrap());
    }

    #[test]
    fn input_order_does_not_change_selection_or_propensities() {
        let mut first = policy();
        let expected = select_metrology_audit_v1(&first).unwrap();
        first.candidates.reverse();
        let actual = select_metrology_audit_v1(&first).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn new_seed_changes_randomized_audit_draw_but_not_exploit() {
        let first = select_metrology_audit_v1(&policy()).unwrap();
        let mut changed = policy();
        changed.seed += 1;
        let second = select_metrology_audit_v1(&changed).unwrap();
        let exploit_a: Vec<_> = first.rows.iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::DeterministicExploit))
            .map(|row| row.candidate_id.clone()).collect();
        let exploit_b: Vec<_> = second.rows.iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::DeterministicExploit))
            .map(|row| row.candidate_id.clone()).collect();
        assert_eq!(exploit_a, exploit_b);
        let audit_a: Vec<_> = first.rows.iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit))
            .map(|row| row.candidate_id.clone()).collect();
        let audit_b: Vec<_> = second.rows.iter()
            .filter(|row| matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit))
            .map(|row| row.candidate_id.clone()).collect();
        assert_ne!(audit_a, audit_b);
    }

    #[test]
    fn audit_budget_overrun_and_zero_probability_residual_strata_reject() {
        let mut too_expensive = policy();
        too_expensive.audit_cost_budget = 2;
        assert!(select_metrology_audit_v1(&too_expensive).is_err());

        let mut no_audit = policy();
        no_audit.audit_quota_by_stratum.insert("tool-a".into(), 0);
        assert!(select_metrology_audit_v1(&no_audit).is_err());
    }

    #[test]
    fn graph_maturity_and_ht_estimator_reconcile_census_audit_design() {
        let mut input = policy();
        input.audit_quota_by_stratum = BTreeMap::from([
            ("tool-a".into(), 4),
            ("tool-b".into(), 4),
        ]);
        input.exploit_cost_budget = 0;
        input.audit_cost_budget = 8;
        input.total_cost_budget = 8;
        let selection = select_metrology_audit_v1(&input).unwrap();
        // Eight audit outcomes are selected across two strata. The selection
        // report itself determines exactly which candidate subjects need labels.
        let selected: Vec<_> = selection.rows.iter().filter(|row| row.selected).collect();
        let mut events = Vec::new();
        let mut labels = Vec::new();
        let mut outcomes = Vec::new();
        for (index, row) in selected.iter().enumerate() {
            let event_id = format!("met-event-{index}");
            let label_id = format!("met-label-{index}");
            events.push(ProcessEventV1 {
                event_id: event_id.clone(),
                event_kind: ProcessEventKindV1::Metrology,
                subject_id: Some(row.subject_id.clone()),
                wafer_id: Some(row.subject_id.clone()),
                lot_id: Some("lot-audit".into()),
                carrier_id: None,
                tool_id: Some(row.stratum_id.clone()),
                chamber_id: None,
                module_id: None,
                process_step_id: None,
                recipe_id: None,
                recipe_revision: None,
                event_time: "2026-11-22T08:00:00".into(),
                event_clock_domain_id: "fab-local".into(),
                event_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:00:01".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                source_sequence: Some(index as u64 + 1),
                source_artifact_sha256: SHA_A.into(),
                transform_sha256: SHA_B.into(),
                authority_ref: "metrology-source".into(),
            });
            labels.push(MetrologyLabelV1 {
                label_id: label_id.clone(),
                measurement_event_id: event_id,
                subject_id: row.subject_id.clone(),
                measurand_id: row.target_measurand_id.clone(),
                value: Some(if index % 3 == 0 { 1.0 } else { 0.0 }),
                unit: "binary".into(),
                standard_uncertainty: Some(0.0),
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
                instrument_id: Some("metrology-1".into()),
                calibration_ref: Some("cal-1".into()),
                revision: 1,
                supersedes_label_id: None,
                source_artifact_sha256: SHA_C.into(),
                authority_ref: "lims-validated".into(),
            });
            outcomes.push(MetrologyAuditOutcomeV1 {
                candidate_id: row.candidate_id.clone(),
                label_id,
                positive: index % 3 == 0,
                class_semantics_ref: row.class_semantics_ref.clone(),
                class_semantics_sha256: row.class_semantics_sha256.clone(),
            });
        }
        let graph = ProcessEventGraphV1 {
            schema: crate::benchmark::process_event_graph::GRAPH_SCHEMA_V1.into(),
            graph_id: "audit-test-graph".into(),
            source_artifact_sha256: SHA_A.into(),
            transform_sha256: SHA_B.into(),
            events,
            edges: Vec::new(),
            metrology_labels: labels,
        };
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "audit-cutoff".into(),
            cutoff_time: "2026-11-22T10:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let estimate = estimate_metrology_population_v1(&selection, &outcomes, &graph, &cutoff)
            .unwrap();
        assert_eq!(estimate.population_count, 12);
        assert_eq!(estimate.measured_count, 8);
        assert!((0.0..=1.0).contains(&estimate.measurement_reduction_fraction));
        assert_eq!(estimate.audit_outcome_count, 8);
        assert!(valid_sha256(&estimate.estimate_sha256));
    }

    #[test]
    fn immature_or_wrong_subject_outcome_rejects() {
        let mut input = policy();
        input.exploit_cost_budget = 0;
        input.audit_cost_budget = 6;
        input.total_cost_budget = 6;
        let selection = select_metrology_audit_v1(&input).unwrap();
        // A selected outcome cannot be evaluated without a matching matured
        // graph label. An empty graph must fail closed.
        let outcome = MetrologyAuditOutcomeV1 {
            candidate_id: selection.rows.iter().find(|row| row.selected).unwrap().candidate_id.clone(),
            label_id: "not-in-graph".into(),
            positive: false,
            class_semantics_ref: "defect-threshold-v1".into(),
            class_semantics_sha256: SHA_C.into(),
        };
        let graph = ProcessEventGraphV1 {
            schema: crate::benchmark::process_event_graph::GRAPH_SCHEMA_V1.into(),
            graph_id: "empty-audit-graph".into(),
            source_artifact_sha256: SHA_A.into(),
            transform_sha256: SHA_B.into(),
            events: vec![ProcessEventV1 {
                event_id: "event".into(),
                event_kind: ProcessEventKindV1::Metrology,
                subject_id: Some("unknown".into()),
                wafer_id: None,
                lot_id: None,
                carrier_id: None,
                tool_id: None,
                chamber_id: None,
                module_id: None,
                process_step_id: None,
                recipe_id: None,
                recipe_revision: None,
                event_time: "2026-11-22T08:00:00".into(),
                event_clock_domain_id: "fab-local".into(),
                event_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:00:01".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                source_sequence: Some(1),
                source_artifact_sha256: SHA_A.into(),
                transform_sha256: SHA_B.into(),
                authority_ref: "source".into(),
            }],
            edges: Vec::new(),
            metrology_labels: Vec::new(),
        };
        let cutoff = PredictionInformationCutoffV1 {
            cutoff_id: "audit-cutoff".into(),
            cutoff_time: "2026-11-22T10:00:00".into(),
            clock_domain_id: "fab-local".into(),
            clock_basis: ClockBasisV1::SourceLocalNaive,
        };
        let outcomes: Vec<_> = selection
            .rows
            .iter()
            .filter(|row| row.selected)
            .map(|row| MetrologyAuditOutcomeV1 {
                candidate_id: row.candidate_id.clone(),
                label_id: format!("unmatured-{}", row.candidate_id),
                positive: false,
                class_semantics_ref: row.class_semantics_ref.clone(),
                class_semantics_sha256: row.class_semantics_sha256.clone(),
            })
            .collect();
        assert!(estimate_metrology_population_v1(&selection, &outcomes, &graph, &cutoff).is_err());

    }
}
