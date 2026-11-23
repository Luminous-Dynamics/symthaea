// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Known-ground-truth synthetic fab episodes for adaptive-metrology policy replay.
//!
//! Policy APIs receive only the as-observed view. Latent process state, defect
//! outcomes and label-maturation times remain private to this evaluator module.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::benchmark::metrology_audit::{
    MetrologyAuditSelectionReportV1, MetrologySelectionMethodV1,
};
use crate::benchmark::metrology_semantics::validate_binary_measurand_semantics_v1;

pub const REPLAY_SCENARIO_SCHEMA_V1: &str = "luminous.semi-bench.synthetic-fab-replay-scenario.v1";
pub const REPLAY_POLICY_VIEW_SCHEMA_V1: &str = "luminous.semi-bench.synthetic-fab-policy-view.v1";
pub const REPLAY_EVALUATION_SCHEMA_V1: &str =
    "luminous.semi-bench.synthetic-fab-policy-evaluation.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticFabRegimeV1 {
    Stable,
    GradualProcessDrift,
    AbruptProcessShift,
    CorrelatedSensorOutage,
    CalibrationDrift,
    PrevalenceShift,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFabScenarioConfigV1 {
    pub schema: String,
    pub scenario_id: String,
    pub seed: u64,
    pub wafer_count: usize,
    pub tool_count: usize,
    pub chamber_count: usize,
    pub regime: SyntheticFabRegimeV1,
    pub sensor_dropout_probability: f64,
    pub measurement_noise_stddev: f64,
    pub defect_threshold: f64,
    /// Label availability is retained only in the hidden outcome oracle.
    pub label_delay_steps: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticSensorStateV1 {
    Observed,
    CommunicationLoss,
    NotInstalled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSensorObservationV1 {
    pub state: SyntheticSensorStateV1,
    pub value: Option<f64>,
    /// Synthetic units cannot be mistaken for qualified physical measurands.
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFabObservedWaferV1 {
    pub candidate_id: String,
    pub subject_id: String,
    pub lot_id: String,
    pub tool_id: String,
    pub chamber_id: String,
    pub stratum_id: String,
    pub process_step: u64,
    pub sensor_a: SyntheticSensorObservationV1,
    pub sensor_b: SyntheticSensorObservationV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFabPolicyViewV1 {
    pub schema: String,
    pub population_count: usize,
    pub observations: Vec<SyntheticFabObservedWaferV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFabPolicyEvaluationV1 {
    pub schema: String,
    pub scenario_id: String,
    pub scenario_config_sha256: String,
    pub observation_packet_sha256: String,
    pub selection_sha256: String,
    pub hidden_truth_sha256: String,
    pub population_count: usize,
    pub true_positive_count: usize,
    pub selected_count: usize,
    pub selected_positive_count: usize,
    pub escaped_defect_count: usize,
    pub true_defect_prevalence: f64,
    pub mean_latent_process_state: f64,
    pub mean_defect_score: f64,
    pub selected_sample_prevalence: f64,
    pub horvitz_thompson_prevalence: f64,
    pub horvitz_thompson_absolute_error: f64,
    pub standard_error_ht_prevalence: Option<f64>,
    pub horvitz_thompson_mean_brier: f64,
    pub standard_error_ht_mean_brier: Option<f64>,
    pub design_variance_qualified: bool,
    pub true_population_mean_brier: f64,
    pub measured_fraction: f64,
    pub effective_sample_size: f64,
    pub matured_label_count_at_horizon: usize,
    pub claim_limits: Vec<String>,
    pub evaluation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SyntheticFabReplayError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SyntheticFabHiddenTruthV1 {
    candidate_id: String,
    latent_process_state: f64,
    defect_score: f64,
    defect_positive: bool,
    label_available_step: u64,
}

#[derive(Clone, Debug)]
pub struct SyntheticFabEpisodeV1 {
    policy_view: SyntheticFabPolicyViewV1,
    scenario_id: String,
    scenario_config_sha256: String,
    observation_packet_sha256: String,
    hidden_truth: Vec<SyntheticFabHiddenTruthV1>,
}

impl SyntheticFabEpisodeV1 {
    /// Returns the only representation available to a policy.
    pub fn policy_view(&self) -> &SyntheticFabPolicyViewV1 {
        &self.policy_view
    }
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Irwin-Hall approximation with zero mean and unit variance.
    /// Avoids platform-dependent trigonometric transforms in fixtures.
    fn centered_noise(&mut self) -> f64 {
        (0..12).map(|_| self.unit_interval()).sum::<f64>() - 6.0
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize>(value: &T) -> Result<String, SyntheticFabReplayError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SyntheticFabReplayError(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn selection_digest(
    selection: &MetrologyAuditSelectionReportV1,
) -> Result<String, SyntheticFabReplayError> {
    let mut material = selection.clone();
    material.selection_sha256.clear();
    json_sha256(&material)
}

#[cfg(test)]
fn seal_selection_for_test(selection: &mut MetrologyAuditSelectionReportV1) {
    selection.selection_sha256.clear();
    selection.selection_sha256 = selection_digest(selection).expect("selection should hash");
}

fn validate_selection_design(
    selection: &MetrologyAuditSelectionReportV1,
) -> Result<(), SyntheticFabReplayError> {
    let mut residual_population_by_stratum = BTreeMap::<String, usize>::new();
    let mut randomized_selected_by_stratum = BTreeMap::<String, usize>::new();
    let mut maximum_cost_by_stratum = BTreeMap::<String, u64>::new();
    let mut exploit_count = 0_usize;
    let mut randomized_audit_count = 0_usize;
    let mut exploit_cost_used = 0_u64;
    let mut audit_cost_used = 0_u64;
    let mut total_selected_cost = 0_u64;

    for (index, row) in selection.rows.iter().enumerate() {
        if row.measurement_cost == 0 {
            return Err(SyntheticFabReplayError(format!(
                "selection.rows[{index}].measurement_cost must be positive"
            )));
        }
        match &row.method {
            MetrologySelectionMethodV1::DeterministicExploit => {
                if !row.selected || row.inclusion_probability != 1.0 {
                    return Err(SyntheticFabReplayError(format!(
                        "selection.rows[{index}]: exploit must be selected with pi=1"
                    )));
                }
                exploit_count += 1;
                exploit_cost_used = exploit_cost_used
                    .checked_add(row.measurement_cost)
                    .ok_or_else(|| SyntheticFabReplayError("exploit cost overflow".into()))?;
                total_selected_cost = total_selected_cost
                    .checked_add(row.measurement_cost)
                    .ok_or_else(|| SyntheticFabReplayError("selected cost overflow".into()))?;
            }
            MetrologySelectionMethodV1::RandomizedAudit => {
                if !row.selected {
                    return Err(SyntheticFabReplayError(format!(
                        "selection.rows[{index}]: randomized audit must be selected"
                    )));
                }
                randomized_audit_count += 1;
                *residual_population_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default() += 1;
                *randomized_selected_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default() += 1;
                let maximum = maximum_cost_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default();
                *maximum = (*maximum).max(row.measurement_cost);
                audit_cost_used = audit_cost_used
                    .checked_add(row.measurement_cost)
                    .ok_or_else(|| SyntheticFabReplayError("audit cost overflow".into()))?;
                total_selected_cost = total_selected_cost
                    .checked_add(row.measurement_cost)
                    .ok_or_else(|| SyntheticFabReplayError("selected cost overflow".into()))?;
            }
            MetrologySelectionMethodV1::NotSelected => {
                if row.selected {
                    return Err(SyntheticFabReplayError(format!(
                        "selection.rows[{index}]: not_selected conflicts with selected=true"
                    )));
                }
                *residual_population_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default() += 1;
                let maximum = maximum_cost_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default();
                *maximum = (*maximum).max(row.measurement_cost);
            }
        }
    }

    if exploit_count != selection.exploit_count
        || randomized_audit_count != selection.randomized_audit_count
        || residual_population_by_stratum != selection.residual_population_by_stratum
    {
        return Err(SyntheticFabReplayError(
            "selection report counts do not reconcile with row-level methods".into(),
        ));
    }
    if residual_population_by_stratum.keys().collect::<BTreeSet<_>>()
        != selection.audit_quota_by_stratum.keys().collect::<BTreeSet<_>>()
    {
        return Err(SyntheticFabReplayError(
            "audit quotas do not exactly cover residual strata".into(),
        ));
    }

    let mut quota_total = 0_usize;
    let mut audit_worst_case_cost = 0_u64;
    for (stratum, residual_population) in &residual_population_by_stratum {
        let quota = selection.audit_quota_by_stratum[stratum];
        if quota == 0 || quota > *residual_population {
            return Err(SyntheticFabReplayError(format!(
                "stratum {stratum}: audit quota outside [1, residual population]"
            )));
        }
        let sampled = randomized_selected_by_stratum.get(stratum).copied().unwrap_or(0);
        if sampled != quota {
            return Err(SyntheticFabReplayError(format!(
                "stratum {stratum}: selected audit count differs from declared quota"
            )));
        }
        let expected_probability = quota as f64 / *residual_population as f64;
        for row in selection.rows.iter().filter(|row| {
            row.stratum_id == *stratum
                && !matches!(
                    &row.method,
                    MetrologySelectionMethodV1::DeterministicExploit
                )
        }) {
            if (row.inclusion_probability - expected_probability).abs() > 1.0e-12 {
                return Err(SyntheticFabReplayError(format!(
                    "stratum {stratum}: row inclusion probability disagrees with quota/population"
                )));
            }
        }
        quota_total += quota;
        let maximum_cost = maximum_cost_by_stratum.get(stratum).copied().unwrap_or(0);
        audit_worst_case_cost = audit_worst_case_cost
            .checked_add(maximum_cost.checked_mul(quota as u64).ok_or_else(|| {
                SyntheticFabReplayError(format!("stratum {stratum}: worst-case cost overflow"))
            })?)
            .ok_or_else(|| SyntheticFabReplayError("worst-case audit cost overflow".into()))?;
    }

    if quota_total != selection.randomized_audit_count
        || exploit_cost_used != selection.exploit_cost_used
        || audit_cost_used != selection.audit_cost_used
        || total_selected_cost != selection.total_selected_cost
        || audit_worst_case_cost != selection.audit_worst_case_cost
    {
        return Err(SyntheticFabReplayError(
            "selection report costs/counts do not reconcile with its rows".into(),
        ));
    }
    if selection.total_cost_budget == 0
        || exploit_cost_used > selection.exploit_cost_budget
        || audit_cost_used > selection.audit_cost_budget
        || audit_worst_case_cost > selection.audit_cost_budget
        || total_selected_cost > selection.total_cost_budget
    {
        return Err(SyntheticFabReplayError(
            "selection rows exceed a declared cost budget".into(),
        ));
    }
    Ok(())
}

fn finite_population_sample_variance(values: &[f64]) -> Option<f64> {
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

fn validate_config(config: &SyntheticFabScenarioConfigV1) -> Result<(), SyntheticFabReplayError> {
    if config.schema != REPLAY_SCENARIO_SCHEMA_V1 {
        return Err(SyntheticFabReplayError(
            "scenario.schema: unsupported".into(),
        ));
    }
    if config.scenario_id.trim().is_empty() {
        return Err(SyntheticFabReplayError(
            "scenario_id: expected non-empty string".into(),
        ));
    }
    if !(1..=100_000).contains(&config.wafer_count) {
        return Err(SyntheticFabReplayError(
            "wafer_count: expected 1..=100000".into(),
        ));
    }
    if config.tool_count == 0 || config.chamber_count == 0 {
        return Err(SyntheticFabReplayError(
            "tool_count and chamber_count must be positive".into(),
        ));
    }
    if !config.sensor_dropout_probability.is_finite()
        || !(0.0..=1.0).contains(&config.sensor_dropout_probability)
    {
        return Err(SyntheticFabReplayError(
            "sensor_dropout_probability: expected finite value in [0, 1]".into(),
        ));
    }
    if !config.measurement_noise_stddev.is_finite() || config.measurement_noise_stddev < 0.0 {
        return Err(SyntheticFabReplayError(
            "measurement_noise_stddev: expected finite non-negative value".into(),
        ));
    }
    if !config.defect_threshold.is_finite() {
        return Err(SyntheticFabReplayError(
            "defect_threshold: expected finite value".into(),
        ));
    }
    Ok(())
}

fn process_shift(regime: SyntheticFabRegimeV1, fraction: f64) -> f64 {
    match regime {
        SyntheticFabRegimeV1::Stable
        | SyntheticFabRegimeV1::CorrelatedSensorOutage
        | SyntheticFabRegimeV1::CalibrationDrift => 0.0,
        SyntheticFabRegimeV1::GradualProcessDrift => 0.60 * fraction,
        SyntheticFabRegimeV1::AbruptProcessShift if fraction >= 0.5 => 0.65,
        SyntheticFabRegimeV1::PrevalenceShift if fraction >= 0.5 => 0.45,
        SyntheticFabRegimeV1::AbruptProcessShift | SyntheticFabRegimeV1::PrevalenceShift => 0.0,
    }
}

fn sensor_observation(value: f64, state: SyntheticSensorStateV1) -> SyntheticSensorObservationV1 {
    SyntheticSensorObservationV1 {
        state,
        value: (state == SyntheticSensorStateV1::Observed).then_some(value),
        unit: "synthetic_process_unit".into(),
    }
}

/// Generates a deterministic finite population. The policy receives only
/// policy_view(); the hidden truth is available only to the evaluator.
pub fn generate_synthetic_fab_episode_v1(
    config: &SyntheticFabScenarioConfigV1,
) -> Result<SyntheticFabEpisodeV1, SyntheticFabReplayError> {
    validate_config(config)?;
    let scenario_config_sha256 = json_sha256(config)?;
    let mut rng = SplitMix64::new(config.seed);
    let mut observations = Vec::with_capacity(config.wafer_count);
    let mut hidden_truth = Vec::with_capacity(config.wafer_count);

    for index in 0..config.wafer_count {
        let fraction = if config.wafer_count <= 1 {
            0.0
        } else {
            index as f64 / (config.wafer_count - 1) as f64
        };
        let tool_index = index % config.tool_count;
        let chamber_index = (index / config.tool_count) % config.chamber_count;
        let tool_id = format!("tool-{tool_index:02}");
        let chamber_id = format!("chamber-{chamber_index:02}");
        let stratum_id = format!("{tool_id}|{chamber_id}");
        let candidate_id = format!("wafer-{index:06}");
        let subject_id = candidate_id.clone();
        let lot_id = format!("lot-{:04}", index / 25);

        let baseline_state = 0.24 + config.measurement_noise_stddev * rng.centered_noise();
        let latent_process_state = baseline_state + process_shift(config.regime, fraction);
        let defect_score = latent_process_state + 0.14 * rng.centered_noise();
        let defect_positive = defect_score >= config.defect_threshold;

        let calibration_offset = if config.regime == SyntheticFabRegimeV1::CalibrationDrift {
            0.45 * fraction
        } else {
            0.0
        };
        let sensor_a_raw = latent_process_state
            + config.measurement_noise_stddev * rng.centered_noise()
            + calibration_offset;
        let sensor_b_raw = 0.72 * latent_process_state
            + config.measurement_noise_stddev * rng.centered_noise()
            - 0.5 * calibration_offset;

        let sensor_a_state = if rng.unit_interval() < config.sensor_dropout_probability {
            SyntheticSensorStateV1::CommunicationLoss
        } else {
            SyntheticSensorStateV1::Observed
        };
        let sensor_b_not_installed = tool_index + 1 == config.tool_count;
        let sensor_b_correlated_dropout = !sensor_b_not_installed
            && config.regime == SyntheticFabRegimeV1::CorrelatedSensorOutage
            && (index / 8) % 5 == 2;
        let sensor_b_random_dropout = !sensor_b_not_installed
            && !sensor_b_correlated_dropout
            && rng.unit_interval() < config.sensor_dropout_probability;
        let sensor_b_state = if sensor_b_not_installed {
            SyntheticSensorStateV1::NotInstalled
        } else if sensor_b_correlated_dropout || sensor_b_random_dropout {
            SyntheticSensorStateV1::CommunicationLoss
        } else {
            SyntheticSensorStateV1::Observed
        };

        observations.push(SyntheticFabObservedWaferV1 {
            candidate_id: candidate_id.clone(),
            subject_id,
            lot_id,
            tool_id,
            chamber_id,
            stratum_id,
            process_step: index as u64,
            sensor_a: sensor_observation(sensor_a_raw, sensor_a_state),
            sensor_b: sensor_observation(sensor_b_raw, sensor_b_state),
        });
        hidden_truth.push(SyntheticFabHiddenTruthV1 {
            candidate_id,
            latent_process_state,
            defect_score,
            defect_positive,
            label_available_step: (index as u64).saturating_add(config.label_delay_steps),
        });
    }

    let observation_packet_sha256 = json_sha256(&observations)?;
    let policy_view = SyntheticFabPolicyViewV1 {
        schema: REPLAY_POLICY_VIEW_SCHEMA_V1.into(),
        population_count: observations.len(),
        observations,
    };

    Ok(SyntheticFabEpisodeV1 {
        policy_view,
        scenario_id: config.scenario_id.clone(),
        scenario_config_sha256,
        observation_packet_sha256,
        hidden_truth,
    })
}

/// Scores a completed selection against hidden finite-population truth.
/// Every unit must have a positive, logged first-order inclusion probability.
pub fn evaluate_synthetic_metrology_policy_v1(
    episode: &SyntheticFabEpisodeV1,
    selection: &MetrologyAuditSelectionReportV1,
) -> Result<SyntheticFabPolicyEvaluationV1, SyntheticFabReplayError> {
    if selection.schema != crate::benchmark::metrology_audit::SELECTION_SCHEMA_V1
        || selection.population_count != episode.policy_view.population_count
        || selection.rows.len() != episode.policy_view.population_count
    {
        return Err(SyntheticFabReplayError(
            "selection receipt schema/population does not match scenario".into(),
        ));
    }
    if selection_digest(selection)? != selection.selection_sha256 {
        return Err(SyntheticFabReplayError(
            "selection receipt digest does not match payload".into(),
        ));
    }
    validate_binary_measurand_semantics_v1(&selection.binary_semantics).map_err(|error| {
        SyntheticFabReplayError(format!("selection class semantics rejected: {error}"))
    })?;
    validate_selection_design(selection)?;

    let mut selection_by_id = BTreeMap::new();
    for (index, row) in selection.rows.iter().enumerate() {
        if row.candidate_id.trim().is_empty()
            || !row.inclusion_probability.is_finite()
            || row.inclusion_probability <= 0.0
            || row.inclusion_probability > 1.0
            || !row.predicted_defect_probability.is_finite()
            || !(0.0..=1.0).contains(&row.predicted_defect_probability)
        {
            return Err(SyntheticFabReplayError(format!(
                "selection.rows[{index}]: invalid identity, probability or score"
            )));
        }
        match &row.method {
            MetrologySelectionMethodV1::DeterministicExploit
                if !row.selected || row.inclusion_probability != 1.0 =>
            {
                return Err(SyntheticFabReplayError(format!(
                    "selection.rows[{index}]: deterministic exploit requires selected=true and pi=1"
                )));
            }
            MetrologySelectionMethodV1::RandomizedAudit if !row.selected => {
                return Err(SyntheticFabReplayError(format!(
                    "selection.rows[{index}]: randomized audit method requires selected=true"
                )));
            }
            MetrologySelectionMethodV1::NotSelected if row.selected => {
                return Err(SyntheticFabReplayError(format!(
                    "selection.rows[{index}]: method conflicts with selected=true"
                )));
            }
            _ => {}
        }
        if selection_by_id
            .insert(row.candidate_id.as_str(), row)
            .is_some()
        {
            return Err(SyntheticFabReplayError(format!(
                "selection.rows[{index}]: duplicate candidate identity"
            )));
        }
    }

    let truth_by_id: BTreeMap<_, _> = episode
        .hidden_truth
        .iter()
        .map(|truth| (truth.candidate_id.as_str(), truth))
        .collect();
    let mut selected_count = 0_usize;
    let mut selected_positive_count = 0_usize;
    let mut true_positive_count = 0_usize;
    let mut latent_process_state_total = 0.0;
    let mut defect_score_total = 0.0;
    let mut ht_positive_total = 0.0;
    let mut ht_brier_total = 0.0;
    let mut true_population_brier_total = 0.0;
    let mut weight_sum = 0.0;
    let mut squared_weight_sum = 0.0;
    let mut randomized_sample_by_stratum = BTreeMap::<String, Vec<(f64, f64)>>::new();
    let mut matured_label_count = 0_usize;
    let final_horizon = episode.policy_view.population_count.saturating_sub(1) as u64;

    for observed in &episode.policy_view.observations {
        let row = selection_by_id
            .get(observed.candidate_id.as_str())
            .ok_or_else(|| {
                SyntheticFabReplayError(format!(
                    "selection receipt omits candidate {}",
                    observed.candidate_id
                ))
            })?;
        if row.subject_id != observed.subject_id
            || row.stratum_id != observed.stratum_id
            || row.target_measurand_id != selection.binary_semantics.target_measurand_id
            || row.class_semantics_ref != selection.binary_semantics.semantics_id
            || row.class_semantics_sha256 != selection.binary_semantics.semantics_sha256
        {
            return Err(SyntheticFabReplayError(format!(
                "selection receipt identity/stratum/semantics mismatch for {}",
                observed.candidate_id
            )));
        }
        let truth = truth_by_id
            .get(observed.candidate_id.as_str())
            .ok_or_else(|| SyntheticFabReplayError("hidden truth identity mismatch".into()))?;
        if truth.label_available_step <= final_horizon {
            matured_label_count += 1;
        }
        latent_process_state_total += truth.latent_process_state;
        defect_score_total += truth.defect_score;
        if truth.defect_positive {
            true_positive_count += 1;
        }
        let target = if truth.defect_positive { 1.0 } else { 0.0 };
        let squared_error = (row.predicted_defect_probability - target).powi(2);
        true_population_brier_total += squared_error;
        if row.selected {
            selected_count += 1;
            if truth.defect_positive {
                selected_positive_count += 1;
                ht_positive_total += 1.0 / row.inclusion_probability;
            }
            ht_brier_total += squared_error / row.inclusion_probability;
            if matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit) {
                randomized_sample_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default()
                    .push((target, squared_error));
            }
            let weight = 1.0 / row.inclusion_probability;
            weight_sum += weight;
            squared_weight_sum += weight * weight;
        }
    }
    if selection_by_id.len() != episode.policy_view.observations.len()
        || selection_by_id
            .keys()
            .any(|candidate_id| !truth_by_id.contains_key(candidate_id))
    {
        return Err(SyntheticFabReplayError(
            "selection receipt contains candidates outside the scenario".into(),
        ));
    }
    if selected_count == 0 {
        return Err(SyntheticFabReplayError(
            "cannot evaluate a policy with zero selected measurements".into(),
        ));
    }

    let population_count = episode.policy_view.population_count;
    let escaped_defect_count = true_positive_count.saturating_sub(selected_positive_count);
    let true_defect_prevalence = true_positive_count as f64 / population_count as f64;
    let selected_sample_prevalence = selected_positive_count as f64 / selected_count as f64;
    let horvitz_thompson_prevalence = ht_positive_total / population_count as f64;
    let horvitz_thompson_mean_brier = ht_brier_total / population_count as f64;
    let true_population_mean_brier = true_population_brier_total / population_count as f64;
    let measured_fraction = selected_count as f64 / population_count as f64;
    let effective_sample_size = if squared_weight_sum > 0.0 {
        weight_sum * weight_sum / squared_weight_sum
    } else {
        0.0
    };
    let mut positive_total_variance = 0.0;
    let mut brier_total_variance = 0.0;
    let mut design_variance_qualified = true;
    for (stratum, residual_population) in &selection.residual_population_by_stratum {
        let sample_count = selection.audit_quota_by_stratum[stratum];
        if sample_count == *residual_population {
            continue;
        }
        if sample_count < 2 {
            design_variance_qualified = false;
            continue;
        }
        let Some(sample) = randomized_sample_by_stratum.get(stratum) else {
            design_variance_qualified = false;
            continue;
        };
        let positive_values: Vec<f64> = sample.iter().map(|(positive, _)| *positive).collect();
        let brier_values: Vec<f64> = sample.iter().map(|(_, brier)| *brier).collect();
        let Some(positive_variance) = finite_population_sample_variance(&positive_values) else {
            design_variance_qualified = false;
            continue;
        };
        let Some(brier_variance) = finite_population_sample_variance(&brier_values) else {
            design_variance_qualified = false;
            continue;
        };
        let finite_population_correction =
            1.0 - sample_count as f64 / *residual_population as f64;
        positive_total_variance += (*residual_population as f64).powi(2)
            * finite_population_correction
            * positive_variance
            / sample_count as f64;
        brier_total_variance += (*residual_population as f64).powi(2)
            * finite_population_correction
            * brier_variance
            / sample_count as f64;
    }
    let standard_error_ht_prevalence = design_variance_qualified.then(|| {
        positive_total_variance.sqrt() / population_count as f64
    });
    let standard_error_ht_mean_brier = design_variance_qualified.then(|| {
        brier_total_variance.sqrt() / population_count as f64
    });
    let hidden_truth_sha256 = json_sha256(&episode.hidden_truth)?;

    let mut report = SyntheticFabPolicyEvaluationV1 {
        schema: REPLAY_EVALUATION_SCHEMA_V1.into(),
        scenario_id: episode.scenario_id.clone(),
        scenario_config_sha256: episode.scenario_config_sha256.clone(),
        observation_packet_sha256: episode.observation_packet_sha256.clone(),
        selection_sha256: selection.selection_sha256.clone(),
        hidden_truth_sha256,
        population_count,
        true_positive_count,
        selected_count,
        selected_positive_count,
        escaped_defect_count,
        true_defect_prevalence,
        mean_latent_process_state: latent_process_state_total / population_count as f64,
        mean_defect_score: defect_score_total / population_count as f64,
        selected_sample_prevalence,
        horvitz_thompson_prevalence,
        horvitz_thompson_absolute_error: (horvitz_thompson_prevalence - true_defect_prevalence).abs(),
        standard_error_ht_prevalence,
        horvitz_thompson_mean_brier,
        standard_error_ht_mean_brier,
        design_variance_qualified,
        true_population_mean_brier,
        measured_fraction,
        effective_sample_size,
        matured_label_count_at_horizon: matured_label_count,
        claim_limits: vec![
            "This report evaluates synthetic hidden truth, not physical fab outcomes".into(),
            "A single randomized draw can differ from population truth; paired multi-seed reports are required".into(),
            "Horvitz-Thompson estimates are design-unbiased only when logged inclusion probabilities match the executed sampling design".into(),
            "Synthetic sensor units are not calibrated physical measurands".into(),
            "No physical sampling, wafer disposition, recipe or APC authority is created".into(),
        ],
        evaluation_sha256: String::new(),
    };
    report.evaluation_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrology_audit::MetrologyAuditSelectionRowV1;

    fn config(regime: SyntheticFabRegimeV1, seed: u64) -> SyntheticFabScenarioConfigV1 {
        SyntheticFabScenarioConfigV1 {
            schema: REPLAY_SCENARIO_SCHEMA_V1.into(),
            scenario_id: "synthetic-cmp-drift-01".into(),
            seed,
            wafer_count: 128,
            tool_count: 2,
            chamber_count: 2,
            regime,
            sensor_dropout_probability: 0.12,
            measurement_noise_stddev: 0.08,
            defect_threshold: 0.62,
            label_delay_steps: 5,
        }
    }

    fn fake_selection(episode: &SyntheticFabEpisodeV1) -> MetrologyAuditSelectionReportV1 {
        let mut semantics = crate::benchmark::metrology_semantics::BinaryMeasurandSemanticsV1 {
            schema: crate::benchmark::metrology_semantics::SEMANTICS_SCHEMA_V1.into(),
            semantics_id: "synthetic-defect-rule".into(),
            revision: 1,
            target_measurand_id: "synthetic-defect-score".into(),
            unit: "synthetic_unit".into(),
            positive_class_name: "defect".into(),
            negative_class_name: "pass".into(),
            positive_condition:
                crate::benchmark::metrology_semantics::PositiveConditionV1::GreaterThanOrEqual,
            threshold: 0.62,
            authority_ref: "synthetic-benchmark-only".into(),
            semantics_sha256: String::new(),
        };
        crate::benchmark::metrology_semantics::seal_binary_measurand_semantics_v1(&mut semantics)
            .expect("synthetic semantics should seal");
        let mut position_by_stratum = BTreeMap::<String, usize>::new();
        let rows = episode
            .policy_view
            .observations
            .iter()
            .map(|observed| {
                let position = position_by_stratum
                    .entry(observed.stratum_id.clone())
                    .or_default();
                let selected = *position % 4 == 0;
                *position += 1;
                MetrologyAuditSelectionRowV1 {
                    candidate_id: observed.candidate_id.clone(),
                    subject_id: observed.subject_id.clone(),
                    stratum_id: observed.stratum_id.clone(),
                    target_measurand_id: semantics.target_measurand_id.clone(),
                    class_semantics_ref: semantics.semantics_id.clone(),
                    class_semantics_sha256: semantics.semantics_sha256.clone(),
                    predicted_defect_probability: 0.1,
                    priority_score: 0.1,
                    measurement_cost: 1,
                    inclusion_probability: 0.25,
                    selected,
                    method: if selected {
                        MetrologySelectionMethodV1::RandomizedAudit
                    } else {
                        MetrologySelectionMethodV1::NotSelected
                    },
                }
            })
            .collect::<Vec<_>>();
        let mut residual_population_by_stratum = BTreeMap::<String, usize>::new();
        for observed in &episode.policy_view.observations {
            *residual_population_by_stratum
                .entry(observed.stratum_id.clone())
                .or_default() += 1;
        }
        let audit_quota_by_stratum = residual_population_by_stratum
            .iter()
            .map(|(stratum, population)| (stratum.clone(), population / 4))
            .collect();
        let mut selection = MetrologyAuditSelectionReportV1 {
            schema: crate::benchmark::metrology_audit::SELECTION_SCHEMA_V1.into(),
            policy_id: "synthetic-test-policy".into(),
            model_sha256: "a".repeat(64),
            prediction_packet_sha256: "b".repeat(64),
            policy_config_sha256: "c".repeat(64),
            binary_semantics: semantics,
            seed: 7,
            total_cost_budget: 32,
            exploit_cost_budget: 0,
            audit_cost_budget: 32,
            exploit_cost_used: 0,
            audit_cost_used: 32,
            audit_worst_case_cost: 32,
            total_selected_cost: 32,
            population_count: episode.policy_view.population_count,
            exploit_count: 0,
            randomized_audit_count: 32,
            residual_population_by_stratum,
            audit_quota_by_stratum,
            rows,
            selection_sha256: String::new(),
        };
        seal_selection_for_test(&mut selection);
        selection
    }

    #[test]
    fn fixed_seed_reproduces_observation_packet_and_truth_digests() {
        let first = generate_synthetic_fab_episode_v1(&config(
            SyntheticFabRegimeV1::GradualProcessDrift,
            42,
        ))
        .unwrap();
        let second = generate_synthetic_fab_episode_v1(&config(
            SyntheticFabRegimeV1::GradualProcessDrift,
            42,
        ))
        .unwrap();
        assert_eq!(
            json_sha256(&first.policy_view.observations).unwrap(),
            json_sha256(&second.policy_view.observations).unwrap()
        );
        assert_eq!(
            json_sha256(&first.hidden_truth).unwrap(),
            json_sha256(&second.hidden_truth).unwrap()
        );
    }

    #[test]
    fn changed_seed_changes_the_observation_packet_digest() {
        let first =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 42)).unwrap();
        let second =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 43)).unwrap();
        assert_ne!(
            json_sha256(&first.policy_view.observations).unwrap(),
            json_sha256(&second.policy_view.observations).unwrap()
        );
    }

    #[test]
    fn policy_view_contains_no_hidden_outcome_or_label_maturity_fields() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 5)).unwrap();
        let serialized = serde_json::to_string(episode.policy_view()).unwrap();
        assert!(!serialized.contains("defect_positive"));
        assert!(!serialized.contains("defect_score"));
        assert!(!serialized.contains("label_available_step"));
        assert!(!serialized.contains("latent_process_state"));
        assert!(!serialized.contains("scenario_id"));
        assert!(!serialized.contains("scenario_config_sha256"));
        assert!(!serialized.contains("observation_packet_sha256"));
    }

    #[test]
    fn hidden_truth_changes_cannot_mutate_the_policy_view() {
        let mut episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 9)).unwrap();
        let before = json_sha256(&episode.policy_view.observations).unwrap();
        for truth in &mut episode.hidden_truth {
            truth.defect_positive = !truth.defect_positive;
        }
        assert_eq!(
            before,
            json_sha256(&episode.policy_view.observations).unwrap()
        );
    }

    #[test]
    fn evaluation_compares_selected_sample_and_ht_to_finite_population_truth() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 13)).unwrap();
        let selection = fake_selection(&episode);
        let report = evaluate_synthetic_metrology_policy_v1(&episode, &selection).unwrap();
        assert_eq!(report.population_count, 128);
        assert_eq!(report.selected_count, 32);
        assert_eq!(report.measured_fraction, 0.25);
        assert!(report.horvitz_thompson_prevalence.is_finite());
        assert!(report.horvitz_thompson_absolute_error.is_finite());
        assert!(report.design_variance_qualified);
        assert!(report.standard_error_ht_prevalence.is_some());
        assert!(report.standard_error_ht_mean_brier.is_some());
        assert_eq!(report.matured_label_count_at_horizon, 123);
    }

    #[test]
    fn zero_probability_residuals_reject_population_inference() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 15)).unwrap();
        let mut selection = fake_selection(&episode);
        selection.rows[0].inclusion_probability = 0.0;
        seal_selection_for_test(&mut selection);
        assert!(evaluate_synthetic_metrology_policy_v1(&episode, &selection).is_err());
    }

    #[test]
    fn structural_sensor_absence_is_not_encoded_as_communication_loss() {
        let episode = generate_synthetic_fab_episode_v1(&config(
            SyntheticFabRegimeV1::CorrelatedSensorOutage,
            19,
        ))
        .unwrap();
        assert!(episode.policy_view.observations.iter().any(|row| {
            row.sensor_b.state == SyntheticSensorStateV1::NotInstalled
                && row.sensor_b.value.is_none()
        }));
        assert!(episode.policy_view.observations.iter().any(|row| {
            row.sensor_b.state == SyntheticSensorStateV1::CommunicationLoss
                && row.sensor_b.value.is_none()
        }));
    }

    #[test]
    fn evaluator_rejects_duplicate_or_extra_candidate_rows() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 23)).unwrap();
        let mut selection = fake_selection(&episode);
        selection.rows[1].candidate_id = selection.rows[0].candidate_id.clone();
        seal_selection_for_test(&mut selection);
        assert!(evaluate_synthetic_metrology_policy_v1(&episode, &selection).is_err());
    }
}
