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
    MetrologyAuditCandidateV1, MetrologyAuditPolicyV1, MetrologyAuditSelectionReportV1,
    MetrologySelectionMethodV1, POLICY_SCHEMA_V1, select_metrology_audit_v1,
};
use crate::benchmark::metrology_semantics::{
    BinaryMeasurandSemanticsV1, PositiveConditionV1, SEMANTICS_SCHEMA_V1,
    seal_binary_measurand_semantics_v1, validate_binary_measurand_semantics_v1,
};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticFabMissingnessMechanismV1 {
    Mcar,
    Mar,
    Mnar,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyCharacteristicsV1 {
    /// Probability that a true defect is reported as positive.
    pub sensitivity: f64,
    /// Probability that a true non-defect is reported as negative.
    pub specificity: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticBinaryDecisionV1 {
    Accept,
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticBinaryDecisionLossV1 {
    pub false_negative_cost: f64,
    pub false_positive_cost: f64,
    /// Cost of acquiring one physical measurement, in the same loss units.
    pub measurement_cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticBinaryEvsiReportV1 {
    pub prior_defect_probability: f64,
    pub probability_assay_positive: f64,
    pub probability_assay_negative: f64,
    pub posterior_defect_given_positive: Option<f64>,
    pub posterior_defect_given_negative: Option<f64>,
    pub decision_without_measurement: SyntheticBinaryDecisionV1,
    pub decision_after_positive: Option<SyntheticBinaryDecisionV1>,
    pub decision_after_negative: Option<SyntheticBinaryDecisionV1>,
    pub expected_loss_without_measurement: f64,
    pub expected_loss_after_measurement: f64,
    /// Gross decision-theoretic EVSI before acquisition cost.
    pub expected_value_of_sample_information: f64,
    pub measurement_cost: f64,
    pub net_value_after_measurement_cost: f64,
    /// Gross EVSI scaled by max(false-negative, false-positive loss), in [0, 1].
    pub normalized_evsi_score: f64,
    /// Net-positive hurdle score; zero means the measurement does not justify
    /// its declared acquisition cost under this decision model.
    pub normalized_net_value_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFabScenarioConfigV1 {
    pub schema: String,
    pub scenario_id: String,
    pub seed: u64,
    /// Independent deterministic random stream for synthetic assay errors.
    pub measurement_seed: u64,
    pub wafer_count: usize,
    pub tool_count: usize,
    pub chamber_count: usize,
    pub regime: SyntheticFabRegimeV1,
    pub missingness_mechanism: SyntheticFabMissingnessMechanismV1,
    pub sensor_dropout_probability: f64,
    pub measurement_noise_stddev: f64,
    pub defect_threshold: f64,
    pub metrology_characteristics: SyntheticMetrologyCharacteristicsV1,
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
    /// Defects not selected for physical metrology.
    pub escaped_defect_count: usize,
    /// True selected defects detected by the simulated metrology assay.
    pub metrology_detected_true_positive_count: usize,
    pub metrology_false_positive_count: usize,
    pub metrology_false_negative_count: usize,
    pub metrology_true_negative_count: usize,
    pub metrology_observed_positive_count: usize,
    pub metrology_empirical_sensitivity: Option<f64>,
    pub metrology_empirical_specificity: Option<f64>,
    pub metrology_positive_predictive_value: Option<f64>,
    pub metrology_negative_predictive_value: Option<f64>,
    /// Total defects not detected: unselected defects plus assay false negatives.
    pub total_undetected_defect_count: usize,
    pub total_undetected_defect_fraction: Option<f64>,
    pub detection_capture_fraction: Option<f64>,
    pub defect_capture_fraction: Option<f64>,
    pub escaped_defect_fraction: Option<f64>,
    pub true_defect_prevalence: f64,
    pub metrology_sensitivity: f64,
    pub metrology_specificity: f64,
    /// Measurement-error seed and realized selected-assay outcome digest.
    pub assay_measurement_seed: u64,
    pub assay_outcomes_sha256: String,
    pub assay_observed_ht_positive_rate: f64,
    pub assay_adjusted_ht_prevalence: Option<f64>,
    pub assay_adjusted_ht_absolute_error: Option<f64>,
    pub assay_adjusted_ht_standard_error: Option<f64>,
    pub assay_estimator_qualified: bool,
    pub mean_latent_process_state: f64,
    pub mean_defect_score: f64,
    pub selected_sample_prevalence: f64,
    pub horvitz_thompson_prevalence: f64,
    pub horvitz_thompson_absolute_error: f64,
    pub standard_error_ht_prevalence: Option<f64>,
    pub hajek_prevalence: f64,
    pub hajek_absolute_error: f64,
    pub standard_error_hajek_prevalence: Option<f64>,
    pub horvitz_thompson_mean_brier: f64,
    pub standard_error_ht_mean_brier: Option<f64>,
    pub design_variance_qualified: bool,
    pub true_population_mean_brier: f64,
    pub measured_fraction: f64,
    pub measurement_reduction_fraction: f64,
    pub total_selected_cost: u64,
    pub cost_per_defect_selected: Option<f64>,
    pub cost_per_defect_captured: Option<f64>,
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
    defect_threshold: f64,
    metrology_characteristics: SyntheticMetrologyCharacteristicsV1,
    measurement_seed: u64,
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

    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Irwin-Hall approximation with zero mean and unit variance.
    /// Avoids platform-dependent trigonometric transforms in fixtures.
    fn centered_noise(&mut self) -> f64 {
        (0..12).map(|_| self.unit_interval()).sum::<f64>() - 6.0
    }
}

fn stratum_seed(seed: u64, stratum_id: &str) -> u64 {
    let mut digest = Sha256::new();
    digest.update(seed.to_le_bytes());
    digest.update(stratum_id.as_bytes());
    let bytes = digest.finalize();
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("SHA-256 prefix has eight bytes"),
    )
}

fn shuffle_indices(indices: &mut [usize], rng: &mut SplitMix64) {
    for index in (1..indices.len()).rev() {
        indices.swap(index, rng.below(index + 1));
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, SyntheticFabReplayError> {
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

pub const SYNTHETIC_HEURISTIC_MODEL_V1: &str = "semi.synthetic.sensor-rule-heuristic.v1";
pub const REPLAY_TOURNAMENT_SCHEMA_V1: &str =
    "luminous.semi-bench.synthetic-metrology-policy-tournament.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPolicyTournamentRunV1 {
    pub scenario_id: String,
    pub scenario_seed: u64,
    pub policy_id: String,
    pub selection_sha256: String,
    pub evaluation: SyntheticFabPolicyEvaluationV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPolicyTournamentSummaryV1 {
    pub policy_id: String,
    pub run_count: usize,
    pub measurement_reduction_mean: f64,
    pub measurement_reduction_sample_stddev: f64,
    pub detection_capture_support: usize,
    pub detection_capture_mean: Option<f64>,
    pub detection_capture_sample_stddev: Option<f64>,
    pub undetected_defect_fraction_support: usize,
    pub undetected_defect_fraction_mean: Option<f64>,
    pub undetected_defect_fraction_sample_stddev: Option<f64>,
    pub assay_adjusted_prevalence_error_support: usize,
    pub assay_adjusted_prevalence_absolute_error_mean: Option<f64>,
    pub assay_adjusted_prevalence_absolute_error_sample_stddev: Option<f64>,
    pub cost_per_detected_defect_support: usize,
    pub cost_per_detected_defect_mean: Option<f64>,
    pub cost_per_detected_defect_sample_stddev: Option<f64>,
    pub evaluation_sha256s: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPairedPolicyComparisonV1 {
    pub reference_policy_id: String,
    pub comparison_policy_id: String,
    pub paired_scenario_count: usize,
    pub detection_capture_support: usize,
    /// Comparison policy minus reference policy; positive means more defects detected.
    pub detection_capture_mean_delta: Option<f64>,
    pub detection_capture_sample_stddev_delta: Option<f64>,
    pub undetected_defect_fraction_support: usize,
    /// Comparison policy minus reference policy; negative is better.
    pub undetected_defect_fraction_mean_delta: Option<f64>,
    pub undetected_defect_fraction_sample_stddev_delta: Option<f64>,
    pub assay_adjusted_error_support: usize,
    /// Comparison policy minus reference policy; negative is better.
    pub assay_adjusted_prevalence_absolute_error_mean_delta: Option<f64>,
    pub assay_adjusted_prevalence_absolute_error_sample_stddev_delta: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPolicyTournamentV1 {
    pub schema: String,
    pub tournament_id: String,
    pub reference_policy_id: String,
    pub scenario_manifest_sha256: String,
    pub policy_manifest_sha256: String,
    pub total_cost_budget: u64,
    pub paired_scenario_count: usize,
    pub policy_count: usize,
    pub runs: Vec<SyntheticMetrologyPolicyTournamentRunV1>,
    pub summaries: Vec<SyntheticMetrologyPolicyTournamentSummaryV1>,
    pub paired_comparisons: Vec<SyntheticMetrologyPairedPolicyComparisonV1>,
    pub tournament_sha256: String,
}

/// Sample mean and sample standard deviation. A one-observation standard
/// deviation is reported as zero and must be interpreted with its support count.
fn mean_sample_stddev(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return Some((mean, 0.0));
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some((mean, variance.sqrt()))
}

/// Executes every policy template on every scenario seed. Policies are paired
/// by receiving the same generated episode; the per-policy selection seed is
/// deterministically derived from the template seed and scenario seed.
/// All policies must use the same total cost budget.
pub fn run_synthetic_metrology_policy_tournament_v1(
    tournament_id: &str,
    reference_policy_id: &str,
    scenarios: &[SyntheticFabScenarioConfigV1],
    policy_templates: &[SyntheticMetrologyPolicyConfigV1],
) -> Result<SyntheticMetrologyPolicyTournamentV1, SyntheticFabReplayError> {
    if tournament_id.trim().is_empty()
        || reference_policy_id.trim().is_empty()
        || scenarios.is_empty()
        || policy_templates.is_empty()
    {
        return Err(SyntheticFabReplayError(
            "tournament requires an ID, reference policy, scenarios and policy templates".into(),
        ));
    }
    let common_cost_budget = policy_templates[0].total_cost_budget;
    if common_cost_budget == 0
        || policy_templates
            .iter()
            .any(|policy| policy.total_cost_budget != common_cost_budget)
    {
        return Err(SyntheticFabReplayError(
            "all tournament policies must share one positive total cost budget".into(),
        ));
    }
    let shared_policy_seed = policy_templates[0].seed;
    if policy_templates
        .iter()
        .any(|policy| policy.seed != shared_policy_seed)
    {
        return Err(SyntheticFabReplayError(
            "paired tournament policies must share one selection seed".into(),
        ));
    }

    let mut scenario_keys = BTreeSet::new();
    let mut policy_ids = BTreeSet::new();
    let first = &scenarios[0];
    for (index, scenario) in scenarios.iter().enumerate() {
        validate_config(scenario)?;
        if !scenario_keys.insert((scenario.scenario_id.as_str(), scenario.seed)) {
            return Err(SyntheticFabReplayError(format!(
                "scenarios[{index}]: duplicate scenario ID and seed"
            )));
        }
        if scenario.wafer_count != first.wafer_count
            || scenario.tool_count != first.tool_count
            || scenario.chamber_count != first.chamber_count
        {
            return Err(SyntheticFabReplayError(
                "paired scenarios must share wafer, tool and chamber populations".into(),
            ));
        }
    }
    for (index, policy) in policy_templates.iter().enumerate() {
        if policy.policy_id.trim().is_empty() || !policy_ids.insert(policy.policy_id.as_str()) {
            return Err(SyntheticFabReplayError(format!(
                "policy_templates[{index}]: empty or duplicate policy ID"
            )));
        }
        if policy.total_cost_budget != common_cost_budget {
            return Err(SyntheticFabReplayError(
                "tournament cost budgets must be matched".into(),
            ));
        }
    }

    if !policy_ids.contains(reference_policy_id) {
        return Err(SyntheticFabReplayError(
            "reference_policy_id must name one of the tournament policy templates".into(),
        ));
    }

    let scenario_manifest_sha256 = json_sha256(scenarios)?;
    let policy_manifest_sha256 = json_sha256(policy_templates)?;
    let mut runs = Vec::with_capacity(scenarios.len() * policy_templates.len());

    for scenario in scenarios {
        let episode = generate_synthetic_fab_episode_v1(scenario)?;
        let semantics = synthetic_defect_semantics_v1(scenario.defect_threshold)?;
        for template in policy_templates {
            let mut policy = template.clone();
            policy.seed = stratum_seed(template.seed ^ scenario.seed, &scenario.scenario_id);
            let run = run_synthetic_metrology_policy_v1(
                &episode,
                episode.policy_view(),
                &semantics,
                &policy,
            )?;
            runs.push(SyntheticMetrologyPolicyTournamentRunV1 {
                scenario_id: scenario.scenario_id.clone(),
                scenario_seed: scenario.seed,
                policy_id: template.policy_id.clone(),
                selection_sha256: run.selection.selection_sha256,
                evaluation: run.evaluation,
            });
        }
    }

    let mut summaries = Vec::with_capacity(policy_templates.len());
    for template in policy_templates {
        let policy_runs: Vec<_> = runs
            .iter()
            .filter(|run| run.policy_id == template.policy_id)
            .collect();
        let measurement_reductions: Vec<f64> = policy_runs
            .iter()
            .map(|run| run.evaluation.measurement_reduction_fraction)
            .collect();
        let detection_capture: Vec<f64> = policy_runs
            .iter()
            .filter_map(|run| run.evaluation.detection_capture_fraction)
            .collect();
        let undetected_fraction: Vec<f64> = policy_runs
            .iter()
            .filter_map(|run| run.evaluation.total_undetected_defect_fraction)
            .collect();
        let assay_adjusted_error: Vec<f64> = policy_runs
            .iter()
            .filter_map(|run| run.evaluation.assay_adjusted_ht_absolute_error)
            .collect();
        let cost_per_detected: Vec<f64> = policy_runs
            .iter()
            .filter_map(|run| run.evaluation.cost_per_defect_captured)
            .collect();

        let (measurement_reduction_mean, measurement_reduction_sample_stddev) =
            mean_sample_stddev(&measurement_reductions).ok_or_else(|| {
                SyntheticFabReplayError("policy tournament has no measurement results".into())
            })?;
        let detection_capture_stats = mean_sample_stddev(&detection_capture);
        let undetected_fraction_stats = mean_sample_stddev(&undetected_fraction);
        let assay_error_stats = mean_sample_stddev(&assay_adjusted_error);
        let cost_per_detected_stats = mean_sample_stddev(&cost_per_detected);

        summaries.push(SyntheticMetrologyPolicyTournamentSummaryV1 {
            policy_id: template.policy_id.clone(),
            run_count: policy_runs.len(),
            measurement_reduction_mean,
            measurement_reduction_sample_stddev,
            detection_capture_support: detection_capture.len(),
            detection_capture_mean: detection_capture_stats.map(|stats| stats.0),
            detection_capture_sample_stddev: detection_capture_stats.map(|stats| stats.1),
            undetected_defect_fraction_support: undetected_fraction.len(),
            undetected_defect_fraction_mean: undetected_fraction_stats.map(|stats| stats.0),
            undetected_defect_fraction_sample_stddev: undetected_fraction_stats
                .map(|stats| stats.1),
            assay_adjusted_prevalence_error_support: assay_adjusted_error.len(),
            assay_adjusted_prevalence_absolute_error_mean: assay_error_stats.map(|stats| stats.0),
            assay_adjusted_prevalence_absolute_error_sample_stddev: assay_error_stats
                .map(|stats| stats.1),
            cost_per_detected_defect_support: cost_per_detected.len(),
            cost_per_detected_defect_mean: cost_per_detected_stats.map(|stats| stats.0),
            cost_per_detected_defect_sample_stddev: cost_per_detected_stats.map(|stats| stats.1),
            evaluation_sha256s: policy_runs
                .iter()
                .map(|run| run.evaluation.evaluation_sha256.clone())
                .collect(),
        });
    }

    let reference_runs: BTreeMap<_, _> = runs
        .iter()
        .filter(|run| run.policy_id == reference_policy_id)
        .map(|run| ((run.scenario_id.clone(), run.scenario_seed), run))
        .collect();
    let mut paired_comparisons = Vec::new();
    for template in policy_templates {
        if template.policy_id == reference_policy_id {
            continue;
        }
        let comparison_runs: BTreeMap<_, _> = runs
            .iter()
            .filter(|run| run.policy_id == template.policy_id)
            .map(|run| ((run.scenario_id.clone(), run.scenario_seed), run))
            .collect();
        let mut detection_capture_deltas = Vec::new();
        let mut undetected_fraction_deltas = Vec::new();
        let mut assay_error_deltas = Vec::new();

        for (scenario_key, reference_run) in &reference_runs {
            let Some(comparison_run) = comparison_runs.get(scenario_key) else {
                return Err(SyntheticFabReplayError(format!(
                    "policy {} is missing paired scenario {:?}",
                    template.policy_id, scenario_key
                )));
            };
            if let (Some(comparison), Some(reference)) = (
                comparison_run.evaluation.detection_capture_fraction,
                reference_run.evaluation.detection_capture_fraction,
            ) {
                detection_capture_deltas.push(comparison - reference);
            }
            if let (Some(comparison), Some(reference)) = (
                comparison_run.evaluation.total_undetected_defect_fraction,
                reference_run.evaluation.total_undetected_defect_fraction,
            ) {
                undetected_fraction_deltas.push(comparison - reference);
            }
            if let (Some(comparison), Some(reference)) = (
                comparison_run.evaluation.assay_adjusted_ht_absolute_error,
                reference_run.evaluation.assay_adjusted_ht_absolute_error,
            ) {
                assay_error_deltas.push(comparison - reference);
            }
        }

        let detection_stats = mean_sample_stddev(&detection_capture_deltas);
        let undetected_stats = mean_sample_stddev(&undetected_fraction_deltas);
        let assay_error_stats = mean_sample_stddev(&assay_error_deltas);
        paired_comparisons.push(SyntheticMetrologyPairedPolicyComparisonV1 {
            reference_policy_id: reference_policy_id.into(),
            comparison_policy_id: template.policy_id.clone(),
            paired_scenario_count: reference_runs.len(),
            detection_capture_support: detection_capture_deltas.len(),
            detection_capture_mean_delta: detection_stats.map(|stats| stats.0),
            detection_capture_sample_stddev_delta: detection_stats.map(|stats| stats.1),
            undetected_defect_fraction_support: undetected_fraction_deltas.len(),
            undetected_defect_fraction_mean_delta: undetected_stats.map(|stats| stats.0),
            undetected_defect_fraction_sample_stddev_delta: undetected_stats.map(|stats| stats.1),
            assay_adjusted_error_support: assay_error_deltas.len(),
            assay_adjusted_prevalence_absolute_error_mean_delta: assay_error_stats
                .map(|stats| stats.0),
            assay_adjusted_prevalence_absolute_error_sample_stddev_delta: assay_error_stats
                .map(|stats| stats.1),
        });
    }

    let mut tournament = SyntheticMetrologyPolicyTournamentV1 {
        schema: REPLAY_TOURNAMENT_SCHEMA_V1.into(),
        tournament_id: tournament_id.into(),
        reference_policy_id: reference_policy_id.into(),
        scenario_manifest_sha256,
        policy_manifest_sha256,
        total_cost_budget: common_cost_budget,
        paired_scenario_count: scenarios.len(),
        policy_count: policy_templates.len(),
        runs,
        summaries,
        paired_comparisons,
        tournament_sha256: String::new(),
    };
    tournament.tournament_sha256 = json_sha256(&tournament)?;
    Ok(tournament)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPolicyConfigV1 {
    pub policy_id: String,
    pub seed: u64,
    pub total_cost_budget: u64,
    pub exploit_cost_budget: u64,
    pub audit_cost_budget: u64,
    pub risk_weight: f64,
    pub uncertainty_weight: f64,
    pub evsi_weight: f64,
    pub audit_quota_by_stratum: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMetrologyPolicyRunV1 {
    pub selection: MetrologyAuditSelectionReportV1,
    pub evaluation: SyntheticFabPolicyEvaluationV1,
}

/// Creates the frozen synthetic target semantics. The threshold must match
/// the scenario configuration and is checked again by the hidden-truth oracle.
pub fn synthetic_defect_semantics_v1(
    threshold: f64,
) -> Result<BinaryMeasurandSemanticsV1, SyntheticFabReplayError> {
    if !threshold.is_finite() {
        return Err(SyntheticFabReplayError(
            "synthetic defect threshold must be finite".into(),
        ));
    }
    let mut semantics = BinaryMeasurandSemanticsV1 {
        schema: SEMANTICS_SCHEMA_V1.into(),
        semantics_id: "synthetic-defect-rule".into(),
        revision: 1,
        target_measurand_id: "synthetic-defect-score".into(),
        unit: "synthetic_unit".into(),
        positive_class_name: "defect".into(),
        negative_class_name: "pass".into(),
        positive_condition: PositiveConditionV1::GreaterThanOrEqual,
        threshold,
        authority_ref: "synthetic-benchmark-only".into(),
        semantics_sha256: String::new(),
    };
    seal_binary_measurand_semantics_v1(&mut semantics).map_err(|error| {
        SyntheticFabReplayError(format!("synthetic semantics invalid: {error}"))
    })?;
    Ok(semantics)
}

fn build_synthetic_candidates(
    policy_view: &SyntheticFabPolicyViewV1,
    semantics: &BinaryMeasurandSemanticsV1,
) -> Result<Vec<MetrologyAuditCandidateV1>, SyntheticFabReplayError> {
    if policy_view.schema != REPLAY_POLICY_VIEW_SCHEMA_V1
        || policy_view.population_count != policy_view.observations.len()
        || policy_view.population_count == 0
    {
        return Err(SyntheticFabReplayError(
            "policy view schema or population count is invalid".into(),
        ));
    }
    validate_binary_measurand_semantics_v1(semantics)
        .map_err(|error| SyntheticFabReplayError(format!("target semantics invalid: {error}")))?;

    policy_view
        .observations
        .iter()
        .map(|observed| {
            for (sensor_name, sensor) in [
                ("sensor_a", &observed.sensor_a),
                ("sensor_b", &observed.sensor_b),
            ] {
                let observed_state = sensor.state == SyntheticSensorStateV1::Observed;
                if observed_state != sensor.value.is_some()
                    || sensor.unit != "synthetic_process_unit"
                    || sensor.value.is_some_and(|value| !value.is_finite())
                {
                    return Err(SyntheticFabReplayError(format!(
                        "{}.{sensor_name}: invalid state/value/unit contract",
                        observed.candidate_id
                    )));
                }
            }
            if observed.candidate_id.trim().is_empty()
                || observed.subject_id.trim().is_empty()
                || observed.stratum_id.trim().is_empty()
            {
                return Err(SyntheticFabReplayError(
                    "observation contains an empty candidate, subject or stratum ID".into(),
                ));
            }
            let sensor_a = observed.sensor_a.value;
            let sensor_b_normalized = observed.sensor_b.value.map(|value| value / 0.72);
            let readings: Vec<f64> = [sensor_a, sensor_b_normalized]
                .into_iter()
                .flatten()
                .collect();
            if readings.iter().any(|value| !value.is_finite()) {
                return Err(SyntheticFabReplayError(format!(
                    "{}: sensor packet contains a non-finite value",
                    observed.candidate_id
                )));
            }
            let estimate = if readings.is_empty() {
                0.24
            } else {
                readings.iter().sum::<f64>() / readings.len() as f64
            };
            let logit = (estimate - semantics.threshold) * 12.0;
            let probability = (1.0 / (1.0 + (-logit).exp())).clamp(0.0, 1.0);
            let disagreement = match (sensor_a, sensor_b_normalized) {
                (Some(left), Some(right)) => (left - right).abs(),
                _ => 0.0,
            };
            let missing_count = [sensor_a.is_none(), sensor_b_normalized.is_none()]
                .into_iter()
                .filter(|missing| *missing)
                .count();
            let uncertainty = match readings.len() {
                0 => 1.0,
                1 => 0.65,
                _ => (0.08 + disagreement * 0.75).clamp(0.0, 1.0),
            };
            let missingness_penalty = missing_count as f64 * 0.08;
            let evsi_proxy = (4.0 * probability * (1.0 - probability) * uncertainty
                + missingness_penalty)
                .clamp(0.0, 1.0);
            Ok(MetrologyAuditCandidateV1 {
                candidate_id: observed.candidate_id.clone(),
                subject_id: observed.subject_id.clone(),
                stratum_id: observed.stratum_id.clone(),
                target_measurand_id: semantics.target_measurand_id.clone(),
                predicted_defect_probability: probability,
                epistemic_uncertainty: uncertainty,
                expected_value_of_information: evsi_proxy,
                measurement_cost: 1,
            })
        })
        .collect()
}

/// Builds an explicitly heuristic challenger policy using only observed data.
/// The uncertainty and EVSI fields are ranking proxies, not calibrated VM
/// uncertainty or a decision-theoretic EVSI estimate.
pub fn build_synthetic_metrology_policy_v1(
    policy_view: &SyntheticFabPolicyViewV1,
    semantics: &BinaryMeasurandSemanticsV1,
    config: &SyntheticMetrologyPolicyConfigV1,
) -> Result<MetrologyAuditPolicyV1, SyntheticFabReplayError> {
    if config.policy_id.trim().is_empty() {
        return Err(SyntheticFabReplayError(
            "policy_id must be non-empty".into(),
        ));
    }
    let candidates = build_synthetic_candidates(policy_view, semantics)?;
    let model_sha256 = sha256_hex(SYNTHETIC_HEURISTIC_MODEL_V1.as_bytes());
    let prediction_packet_sha256 = json_sha256(&candidates)?;
    let observation_packet_sha256 = json_sha256(&policy_view.observations)?;
    let policy_config_sha256 = json_sha256(&(config, observation_packet_sha256))?;
    Ok(MetrologyAuditPolicyV1 {
        schema: POLICY_SCHEMA_V1.into(),
        policy_id: config.policy_id.clone(),
        model_sha256,
        prediction_packet_sha256,
        policy_config_sha256,
        binary_semantics: semantics.clone(),
        total_cost_budget: config.total_cost_budget,
        exploit_cost_budget: config.exploit_cost_budget,
        audit_cost_budget: config.audit_cost_budget,
        risk_weight: config.risk_weight,
        uncertainty_weight: config.uncertainty_weight,
        evsi_weight: config.evsi_weight,
        audit_quota_by_stratum: config.audit_quota_by_stratum.clone(),
        seed: config.seed,
        candidates,
    })
}

/// Builds an observation-only policy whose EVSI score is the exact one-step
/// expected reduction in binary decision loss under the declared assay model.
/// Candidates with non-positive net value after measurement cost receive
/// zero EVSI selection priority; positive candidates retain normalized gross
/// EVSI scores. The supplied assay profile and loss model are bound
/// into the policy and prediction digests; the policy still receives no hidden
/// outcome or realized assay result.
pub fn build_synthetic_decision_evsi_policy_v1(
    policy_view: &SyntheticFabPolicyViewV1,
    semantics: &BinaryMeasurandSemanticsV1,
    metrology_characteristics: SyntheticMetrologyCharacteristicsV1,
    decision_loss: SyntheticBinaryDecisionLossV1,
    config: &SyntheticMetrologyPolicyConfigV1,
) -> Result<MetrologyAuditPolicyV1, SyntheticFabReplayError> {
    if config.evsi_weight <= 0.0 {
        return Err(SyntheticFabReplayError(
            "decision-theoretic EVSI policy requires a positive EVSI weight".into(),
        ));
    }
    let mut policy = build_synthetic_metrology_policy_v1(policy_view, semantics, config)?;
    for candidate in &mut policy.candidates {
        let evsi = calculate_synthetic_binary_evsi_v1(
            candidate.predicted_defect_probability,
            metrology_characteristics,
            decision_loss,
        )?;
        candidate.expected_value_of_information = if evsi.net_value_after_measurement_cost > 0.0 {
            evsi.normalized_evsi_score
        } else {
            0.0
        };
    }
    let model_material = (
        SYNTHETIC_HEURISTIC_MODEL_V1,
        "synthetic-binary-decision-evsi.v1",
        metrology_characteristics,
        decision_loss,
    );
    policy.model_sha256 = json_sha256(&model_material)?;
    policy.prediction_packet_sha256 = json_sha256(&policy.candidates)?;
    let observation_packet_sha256 = json_sha256(&policy_view.observations)?;
    policy.policy_config_sha256 = json_sha256(&(
        config,
        observation_packet_sha256,
        metrology_characteristics,
        decision_loss,
        "synthetic-binary-decision-evsi.v1",
    ))?;
    Ok(policy)
}

/// Runs the decision-theoretic EVSI challenger. The assay profile and loss
/// model are declared policy inputs; only the final evaluator receives the
/// episode's hidden outcomes.
pub fn run_synthetic_decision_evsi_policy_v1(
    episode: &SyntheticFabEpisodeV1,
    policy_view: &SyntheticFabPolicyViewV1,
    semantics: &BinaryMeasurandSemanticsV1,
    metrology_characteristics: SyntheticMetrologyCharacteristicsV1,
    decision_loss: SyntheticBinaryDecisionLossV1,
    config: &SyntheticMetrologyPolicyConfigV1,
) -> Result<SyntheticMetrologyPolicyRunV1, SyntheticFabReplayError> {
    if json_sha256(&policy_view.observations)? != episode.observation_packet_sha256 {
        return Err(SyntheticFabReplayError(
            "policy view does not belong to the hidden-truth episode".into(),
        ));
    }
    let policy = build_synthetic_decision_evsi_policy_v1(
        policy_view,
        semantics,
        metrology_characteristics,
        decision_loss,
        config,
    )?;
    let selection = select_metrology_audit_v1(&policy)
        .map_err(|error| SyntheticFabReplayError(format!("policy selection rejected: {error}")))?;
    let evaluation = evaluate_synthetic_metrology_policy_v1(episode, &selection)?;
    Ok(SyntheticMetrologyPolicyRunV1 {
        selection,
        evaluation,
    })
}

/// Runs the existing audited selection mechanism and hidden-truth evaluator.
/// A policy cannot access the episode or hidden truth during candidate building.
pub fn run_synthetic_metrology_policy_v1(
    episode: &SyntheticFabEpisodeV1,
    policy_view: &SyntheticFabPolicyViewV1,
    semantics: &BinaryMeasurandSemanticsV1,
    config: &SyntheticMetrologyPolicyConfigV1,
) -> Result<SyntheticMetrologyPolicyRunV1, SyntheticFabReplayError> {
    if json_sha256(&policy_view.observations)? != episode.observation_packet_sha256 {
        return Err(SyntheticFabReplayError(
            "policy view does not belong to the hidden-truth episode".into(),
        ));
    }
    let policy = build_synthetic_metrology_policy_v1(policy_view, semantics, config)?;
    let selection = select_metrology_audit_v1(&policy)
        .map_err(|error| SyntheticFabReplayError(format!("policy selection rejected: {error}")))?;
    let evaluation = evaluate_synthetic_metrology_policy_v1(episode, &selection)?;
    Ok(SyntheticMetrologyPolicyRunV1 {
        selection,
        evaluation,
    })
}

fn validate_selection_design(
    selection: &MetrologyAuditSelectionReportV1,
) -> Result<(), SyntheticFabReplayError> {
    if selection.rows.is_empty() {
        return Err(SyntheticFabReplayError(
            "selection receipt must contain candidate rows".into(),
        ));
    }

    let mut previous_candidate_id: Option<&str> = None;
    for (index, row) in selection.rows.iter().enumerate() {
        if row.candidate_id.trim().is_empty()
            || row.subject_id.trim().is_empty()
            || row.stratum_id.trim().is_empty()
            || row.measurement_cost == 0
            || !row.priority_score.is_finite()
            || row.priority_score < 0.0
        {
            return Err(SyntheticFabReplayError(format!(
                "selection.rows[{index}]: invalid identity, cost or priority"
            )));
        }
        if previous_candidate_id.is_some_and(|previous| previous >= row.candidate_id.as_str()) {
            return Err(SyntheticFabReplayError(
                "selection rows must be unique and sorted by candidate_id".into(),
            ));
        }
        previous_candidate_id = Some(row.candidate_id.as_str());
    }

    // Reconstruct the deterministic exploit stage from the recorded priority
    // scores and costs. This verifies which rows were certain selections.
    let mut exploit_order: Vec<usize> = (0..selection.rows.len()).collect();
    exploit_order.sort_by(|left, right| {
        let left_row = &selection.rows[*left];
        let right_row = &selection.rows[*right];
        let left_ratio = left_row.priority_score / left_row.measurement_cost as f64;
        let right_ratio = right_row.priority_score / right_row.measurement_cost as f64;
        right_ratio
            .total_cmp(&left_ratio)
            .then_with(|| left_row.candidate_id.cmp(&right_row.candidate_id))
    });

    let mut expected_exploit = vec![false; selection.rows.len()];
    let mut expected_exploit_cost = 0_u64;
    for index in exploit_order {
        let row = &selection.rows[index];
        if row.priority_score <= 0.0 {
            continue;
        }
        if let Some(next_cost) = expected_exploit_cost.checked_add(row.measurement_cost)
            && next_cost <= selection.exploit_cost_budget
        {
            expected_exploit[index] = true;
            expected_exploit_cost = next_cost;
        }
    }

    let mut residual_by_stratum = BTreeMap::<String, Vec<usize>>::new();
    let mut residual_population_by_stratum = BTreeMap::<String, usize>::new();
    let mut randomized_selected_by_stratum = BTreeMap::<String, usize>::new();
    let mut maximum_cost_by_stratum = BTreeMap::<String, u64>::new();
    let mut exploit_count = 0_usize;
    let mut randomized_audit_count = 0_usize;
    let mut exploit_cost_used = 0_u64;
    let mut audit_cost_used = 0_u64;
    let mut total_selected_cost = 0_u64;

    for (index, row) in selection.rows.iter().enumerate() {
        let is_exploit = expected_exploit[index];
        if is_exploit {
            if !matches!(
                &row.method,
                MetrologySelectionMethodV1::DeterministicExploit
            ) || !row.selected
                || row.inclusion_probability != 1.0
            {
                return Err(SyntheticFabReplayError(format!(
                    "selection.rows[{index}]: deterministic exploit receipt does not replay"
                )));
            }
            exploit_count += 1;
            exploit_cost_used = exploit_cost_used
                .checked_add(row.measurement_cost)
                .ok_or_else(|| SyntheticFabReplayError("exploit cost overflow".into()))?;
            total_selected_cost = total_selected_cost
                .checked_add(row.measurement_cost)
                .ok_or_else(|| SyntheticFabReplayError("selected cost overflow".into()))?;
            continue;
        }

        residual_by_stratum
            .entry(row.stratum_id.clone())
            .or_default()
            .push(index);
        *residual_population_by_stratum
            .entry(row.stratum_id.clone())
            .or_default() += 1;
        let maximum = maximum_cost_by_stratum
            .entry(row.stratum_id.clone())
            .or_default();
        *maximum = (*maximum).max(row.measurement_cost);

        match &row.method {
            MetrologySelectionMethodV1::RandomizedAudit => {
                if !row.selected {
                    return Err(SyntheticFabReplayError(format!(
                        "selection.rows[{index}]: randomized audit must be selected"
                    )));
                }
                randomized_audit_count += 1;
                *randomized_selected_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default() += 1;
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
            }
            MetrologySelectionMethodV1::DeterministicExploit => {
                return Err(SyntheticFabReplayError(format!(
                    "selection.rows[{index}]: unexpected exploit outside reconstructed exploit set"
                )));
            }
        }
    }

    if exploit_count != selection.exploit_count
        || randomized_audit_count != selection.randomized_audit_count
        || residual_population_by_stratum != selection.residual_population_by_stratum
        || exploit_cost_used != selection.exploit_cost_used
    {
        return Err(SyntheticFabReplayError(
            "selection report counts/costs do not reconcile with row-level methods".into(),
        ));
    }

    if residual_population_by_stratum
        .keys()
        .collect::<BTreeSet<_>>()
        != selection
            .audit_quota_by_stratum
            .keys()
            .collect::<BTreeSet<_>>()
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
        let sampled = randomized_selected_by_stratum
            .get(stratum)
            .copied()
            .unwrap_or(0);
        if sampled != quota {
            return Err(SyntheticFabReplayError(format!(
                "stratum {stratum}: selected audit count differs from declared quota"
            )));
        }

        let expected_probability = quota as f64 / *residual_population as f64;
        for index in &residual_by_stratum[stratum] {
            let row = &selection.rows[*index];
            if (row.inclusion_probability - expected_probability).abs() > 1.0e-12 {
                return Err(SyntheticFabReplayError(format!(
                    "stratum {stratum}: row inclusion probability disagrees with quota/population"
                )));
            }
        }

        // Independently replay the exact seeded SRSWOR draw used by the
        // selector. A periodic or hand-picked sample cannot claim audit status.
        let mut expected_draw = residual_by_stratum[stratum].clone();
        let mut rng = SplitMix64::new(stratum_seed(selection.seed, stratum));
        shuffle_indices(&mut expected_draw, &mut rng);
        let expected_audit: BTreeSet<usize> = expected_draw.into_iter().take(quota).collect();
        for index in &residual_by_stratum[stratum] {
            let should_be_selected = expected_audit.contains(index);
            let row = &selection.rows[*index];
            if should_be_selected
                != matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit)
            {
                return Err(SyntheticFabReplayError(format!(
                    "stratum {stratum}: randomized audit draw does not replay from declared seed"
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
    if !config.metrology_characteristics.sensitivity.is_finite()
        || !(0.0..=1.0).contains(&config.metrology_characteristics.sensitivity)
        || !config.metrology_characteristics.specificity.is_finite()
        || !(0.0..=1.0).contains(&config.metrology_characteristics.specificity)
    {
        return Err(SyntheticFabReplayError(
            "metrology sensitivity/specificity must be finite values in [0, 1]".into(),
        ));
    }
    Ok(())
}

fn optimal_binary_disposition(
    defect_probability: f64,
    loss: SyntheticBinaryDecisionLossV1,
) -> (SyntheticBinaryDecisionV1, f64) {
    let accept_loss = defect_probability * loss.false_negative_cost;
    let hold_loss = (1.0 - defect_probability) * loss.false_positive_cost;
    if accept_loss <= hold_loss {
        (SyntheticBinaryDecisionV1::Accept, accept_loss)
    } else {
        (SyntheticBinaryDecisionV1::Hold, hold_loss)
    }
}

/// Computes exact one-step model-implied EVSI for a binary physical assay.
///
/// The result is conditional on the supplied defect probability, fixed assay
/// sensitivity/specificity and declared binary decision losses. It is not an
/// empirical or calibrated fab EVSI unless those inputs are independently
/// qualified. Gross EVSI is reported separately from measurement cost.
pub fn calculate_synthetic_binary_evsi_v1(
    prior_defect_probability: f64,
    characteristics: SyntheticMetrologyCharacteristicsV1,
    loss: SyntheticBinaryDecisionLossV1,
) -> Result<SyntheticBinaryEvsiReportV1, SyntheticFabReplayError> {
    if !prior_defect_probability.is_finite()
        || !(0.0..=1.0).contains(&prior_defect_probability)
        || !characteristics.sensitivity.is_finite()
        || !(0.0..=1.0).contains(&characteristics.sensitivity)
        || !characteristics.specificity.is_finite()
        || !(0.0..=1.0).contains(&characteristics.specificity)
    {
        return Err(SyntheticFabReplayError(
            "EVSI probabilities must be finite and lie in [0, 1]".into(),
        ));
    }
    if !loss.false_negative_cost.is_finite()
        || loss.false_negative_cost < 0.0
        || !loss.false_positive_cost.is_finite()
        || loss.false_positive_cost < 0.0
        || !loss.measurement_cost.is_finite()
        || loss.measurement_cost < 0.0
        || loss.false_negative_cost.max(loss.false_positive_cost) <= 0.0
    {
        return Err(SyntheticFabReplayError(
            "EVSI decision losses must be finite and non-negative, with at least one positive loss"
                .into(),
        ));
    }

    let prior = prior_defect_probability;
    let probability_positive =
        prior * characteristics.sensitivity + (1.0 - prior) * (1.0 - characteristics.specificity);
    let probability_negative = (1.0 - probability_positive).clamp(0.0, 1.0);

    let posterior_positive = (probability_positive > 0.0)
        .then(|| (prior * characteristics.sensitivity / probability_positive).clamp(0.0, 1.0));
    let posterior_negative = (probability_negative > 0.0).then(|| {
        (prior * (1.0 - characteristics.sensitivity) / probability_negative).clamp(0.0, 1.0)
    });

    let (decision_without_measurement, expected_loss_without_measurement) =
        optimal_binary_disposition(prior, loss);

    let (decision_after_positive, positive_conditional_loss) =
        posterior_positive.map_or((None, 0.0), |posterior| {
            let (decision, conditional_loss) = optimal_binary_disposition(posterior, loss);
            (Some(decision), conditional_loss)
        });
    let (decision_after_negative, negative_conditional_loss) =
        posterior_negative.map_or((None, 0.0), |posterior| {
            let (decision, conditional_loss) = optimal_binary_disposition(posterior, loss);
            (Some(decision), conditional_loss)
        });

    let expected_loss_after_measurement = probability_positive * positive_conditional_loss
        + probability_negative * negative_conditional_loss;
    let gross_evsi = (expected_loss_without_measurement - expected_loss_after_measurement).max(0.0);
    let net_value_after_measurement_cost = gross_evsi - loss.measurement_cost;
    let maximum_decision_loss = loss.false_negative_cost.max(loss.false_positive_cost);
    let normalized_evsi_score = (gross_evsi / maximum_decision_loss).clamp(0.0, 1.0);
    let normalized_net_value_score = if net_value_after_measurement_cost > 0.0 {
        (net_value_after_measurement_cost / maximum_decision_loss).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(SyntheticBinaryEvsiReportV1 {
        prior_defect_probability: prior,
        probability_assay_positive: probability_positive,
        probability_assay_negative: probability_negative,
        posterior_defect_given_positive: posterior_positive,
        posterior_defect_given_negative: posterior_negative,
        decision_without_measurement,
        decision_after_positive,
        decision_after_negative,
        expected_loss_without_measurement,
        expected_loss_after_measurement,
        expected_value_of_sample_information: gross_evsi,
        measurement_cost: loss.measurement_cost,
        net_value_after_measurement_cost,
        normalized_evsi_score,
        normalized_net_value_score,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticBinaryEvsiAssumptionProfileV1 {
    pub weight: f64,
    pub metrology_characteristics: SyntheticMetrologyCharacteristicsV1,
    pub decision_loss: SyntheticBinaryDecisionLossV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticBinaryEvsiAssumptionScenarioResultV1 {
    pub scenario_id: String,
    pub weight: f64,
    pub report: SyntheticBinaryEvsiReportV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticBinaryEvsiSensitivityReportV1 {
    pub schema: String,
    pub assumption_set_sha256: String,
    pub scenario_count: usize,
    pub gross_evsi_min: f64,
    pub gross_evsi_weighted_mean: f64,
    pub gross_evsi_max: f64,
    pub net_value_min: f64,
    pub net_value_weighted_mean: f64,
    pub net_value_weighted_p10: f64,
    pub net_value_weighted_median: f64,
    pub net_value_weighted_p90: f64,
    pub weighted_net_positive_fraction: f64,
    /// True only when net value is non-negative in every declared scenario.
    pub robust_measurement_hurdle_pass: bool,
    /// Min of each scenario's normalized net-value score.
    pub conservative_normalized_net_value_score: f64,
    pub scenarios: Vec<SyntheticBinaryEvsiAssumptionScenarioResultV1>,
}

fn weighted_quantile(values: &[(f64, f64)], quantile: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut cumulative_weight = 0.0;
    for (value, weight) in &ordered {
        cumulative_weight += *weight;
        if cumulative_weight + 1.0e-12 >= quantile {
            return *value;
        }
    }
    ordered.last().map(|(value, _)| *value).unwrap_or(0.0)
}

/// Evaluates binary decision EVSI across a finite, explicitly declared set of
/// plausible model/assay/loss assumptions. Scenario weights are user-declared
/// sensitivity weights, not empirically calibrated posterior probabilities.
pub fn evaluate_synthetic_binary_evsi_sensitivity_v1(
    prior_defect_probability: f64,
    assumption_set: &std::collections::BTreeMap<String, SyntheticBinaryEvsiAssumptionProfileV1>,
) -> Result<SyntheticBinaryEvsiSensitivityReportV1, SyntheticFabReplayError> {
    if assumption_set.is_empty() {
        return Err(SyntheticFabReplayError(
            "EVSI assumption set must be non-empty".into(),
        ));
    }
    let total_weight = assumption_set
        .values()
        .map(|profile| profile.weight)
        .sum::<f64>();
    if !total_weight.is_finite()
        || (total_weight - 1.0).abs() > 1.0e-9
        || assumption_set.values().any(|profile| {
            !profile.weight.is_finite() || profile.weight <= 0.0
        })
    {
        return Err(SyntheticFabReplayError(
            "EVSI assumption weights must be finite, positive and sum to one".into(),
        ));
    }

    let mut scenarios = Vec::with_capacity(assumption_set.len());
    let mut gross_values = Vec::with_capacity(assumption_set.len());
    let mut net_values = Vec::with_capacity(assumption_set.len());
    let mut normalized_net_values = Vec::with_capacity(assumption_set.len());
    let mut weighted_gross_mean = 0.0;
    let mut weighted_net_mean = 0.0;
    let mut weighted_net_positive_fraction = 0.0;
    let mut robust_measurement_hurdle_pass = true;

    for (scenario_id, profile) in assumption_set {
        let report = calculate_synthetic_binary_evsi_v1(
            prior_defect_probability,
            profile.metrology_characteristics,
            profile.decision_loss,
        )?;
        weighted_gross_mean += profile.weight * report.expected_value_of_sample_information;
        weighted_net_mean += profile.weight * report.net_value_after_measurement_cost;
        if report.net_value_after_measurement_cost > 0.0 {
            weighted_net_positive_fraction += profile.weight;
        } else {
            robust_measurement_hurdle_pass = false;
        }
        gross_values.push(report.expected_value_of_sample_information);
        net_values.push(report.net_value_after_measurement_cost);
        normalized_net_values.push(report.normalized_net_value_score);
        scenarios.push(SyntheticBinaryEvsiAssumptionScenarioResultV1 {
            scenario_id: scenario_id.clone(),
            weight: profile.weight,
            report,
        });
    }

    let weighted_net_values: Vec<(f64, f64)> = scenarios
        .iter()
        .map(|scenario| {
            (
                scenario.report.net_value_after_measurement_cost,
                scenario.weight,
            )
        })
        .collect();
    let assumption_set_sha256 = json_sha256(assumption_set)?;
    let gross_evsi_min = gross_values.iter().copied().fold(f64::INFINITY, f64::min);
    let gross_evsi_max = gross_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let net_value_min = net_values.iter().copied().fold(f64::INFINITY, f64::min);
    let conservative_normalized_net_value_score = normalized_net_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    Ok(SyntheticBinaryEvsiSensitivityReportV1 {
        schema: "luminous.semi-bench.synthetic-binary-evsi-sensitivity.v1".into(),
        assumption_set_sha256,
        scenario_count: scenarios.len(),
        gross_evsi_min,
        gross_evsi_weighted_mean: weighted_gross_mean,
        gross_evsi_max,
        net_value_min,
        net_value_weighted_mean: weighted_net_mean,
        net_value_weighted_p10: weighted_quantile(&weighted_net_values, 0.10),
        net_value_weighted_median: weighted_quantile(&weighted_net_values, 0.50),
        net_value_weighted_p90: weighted_quantile(&weighted_net_values, 0.90),
        weighted_net_positive_fraction,
        robust_measurement_hurdle_pass,
        conservative_normalized_net_value_score,
        scenarios,
    })
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

fn conditional_dropout_probability(
    mechanism: SyntheticFabMissingnessMechanismV1,
    base_probability: f64,
    observed_tool_fraction: f64,
    latent_defect_positive: bool,
) -> f64 {
    match mechanism {
        SyntheticFabMissingnessMechanismV1::Mcar => base_probability,
        SyntheticFabMissingnessMechanismV1::Mar => {
            (base_probability + 0.45 * observed_tool_fraction).min(1.0)
        }
        SyntheticFabMissingnessMechanismV1::Mnar => {
            (base_probability + if latent_defect_positive { 0.55 } else { 0.0 }).min(1.0)
        }
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

        let observed_tool_fraction = if config.tool_count <= 1 {
            0.0
        } else {
            tool_index as f64 / (config.tool_count - 1) as f64
        };
        let dropout_probability = conditional_dropout_probability(
            config.missingness_mechanism,
            config.sensor_dropout_probability,
            observed_tool_fraction,
            defect_positive,
        );
        let sensor_a_state = if rng.unit_interval() < dropout_probability {
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
            && rng.unit_interval() < dropout_probability;
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
        defect_threshold: config.defect_threshold,
        metrology_characteristics: config.metrology_characteristics,
        measurement_seed: config.measurement_seed,
        hidden_truth,
    })
}

/// Scores a completed selection against hidden finite-population truth.
/// Every unit must have a positive, logged first-order inclusion probability.
fn simulate_metrology_result_v1(
    episode: &SyntheticFabEpisodeV1,
    candidate_id: &str,
    true_defect: bool,
) -> bool {
    let mut rng = SplitMix64::new(stratum_seed(episode.measurement_seed, candidate_id));
    let draw = rng.unit_interval();
    if true_defect {
        draw < episode.metrology_characteristics.sensitivity
    } else {
        draw < 1.0 - episode.metrology_characteristics.specificity
    }
}

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
    if selection.policy_id.trim().is_empty()
        || !is_lowercase_sha256(&selection.model_sha256)
        || !is_lowercase_sha256(&selection.prediction_packet_sha256)
        || !is_lowercase_sha256(&selection.policy_config_sha256)
    {
        return Err(SyntheticFabReplayError(
            "selection receipt requires policy identity and valid model/prediction/config digests"
                .into(),
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
    if selection.binary_semantics.target_measurand_id != "synthetic-defect-score"
        || selection.binary_semantics.unit != "synthetic_unit"
        || selection.binary_semantics.threshold != episode.defect_threshold
        || !matches!(
            &selection.binary_semantics.positive_condition,
            PositiveConditionV1::GreaterThanOrEqual
        )
    {
        return Err(SyntheticFabReplayError(
            "selection semantics do not match the hidden scenario's frozen defect rule".into(),
        ));
    }
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
    let mut metrology_detected_true_positive_count = 0_usize;
    let mut metrology_false_positive_count = 0_usize;
    let mut metrology_false_negative_count = 0_usize;
    let mut assay_outcomes = Vec::<(String, bool)>::new();
    let mut latent_process_state_total = 0.0;
    let mut defect_score_total = 0.0;
    let mut ht_positive_total = 0.0;
    let mut assay_ht_positive_total = 0.0;
    let mut ht_brier_total = 0.0;
    let mut true_population_brier_total = 0.0;
    let mut weight_sum = 0.0;
    let mut squared_weight_sum = 0.0;
    let mut randomized_sample_by_stratum = BTreeMap::<String, Vec<(f64, f64)>>::new();
    let mut randomized_assay_by_stratum = BTreeMap::<String, Vec<f64>>::new();
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
            let assay_positive = simulate_metrology_result_v1(
                episode,
                &observed.candidate_id,
                truth.defect_positive,
            );
            assay_outcomes.push((observed.candidate_id.clone(), assay_positive));
            if assay_positive {
                assay_ht_positive_total += 1.0 / row.inclusion_probability;
            }
            if truth.defect_positive {
                selected_positive_count += 1;
                ht_positive_total += 1.0 / row.inclusion_probability;
                if assay_positive {
                    metrology_detected_true_positive_count += 1;
                } else {
                    metrology_false_negative_count += 1;
                }
            } else if assay_positive {
                metrology_false_positive_count += 1;
            }
            ht_brier_total += squared_error / row.inclusion_probability;
            if matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit) {
                randomized_sample_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default()
                    .push((target, squared_error));
                randomized_assay_by_stratum
                    .entry(row.stratum_id.clone())
                    .or_default()
                    .push(if assay_positive { 1.0 } else { 0.0 });
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
    let total_undetected_defect_count =
        escaped_defect_count.saturating_add(metrology_false_negative_count);
    let metrology_true_negative_count =
        (selected_count - selected_positive_count).saturating_sub(metrology_false_positive_count);
    let metrology_observed_positive_count =
        metrology_detected_true_positive_count + metrology_false_positive_count;
    let metrology_observed_negative_count = selected_count - metrology_observed_positive_count;
    let metrology_empirical_sensitivity = (selected_positive_count > 0)
        .then(|| metrology_detected_true_positive_count as f64 / selected_positive_count as f64);
    let metrology_empirical_specificity = (selected_count > selected_positive_count).then(|| {
        metrology_true_negative_count as f64 / (selected_count - selected_positive_count) as f64
    });
    let metrology_positive_predictive_value = (metrology_observed_positive_count > 0).then(|| {
        metrology_detected_true_positive_count as f64 / metrology_observed_positive_count as f64
    });
    let metrology_negative_predictive_value = (metrology_observed_negative_count > 0)
        .then(|| metrology_true_negative_count as f64 / metrology_observed_negative_count as f64);
    let total_undetected_defect_fraction = (true_positive_count > 0)
        .then(|| total_undetected_defect_count as f64 / true_positive_count as f64);
    let detection_capture_fraction = (true_positive_count > 0)
        .then(|| metrology_detected_true_positive_count as f64 / true_positive_count as f64);
    let defect_capture_fraction = (true_positive_count > 0)
        .then(|| selected_positive_count as f64 / true_positive_count as f64);
    let escaped_defect_fraction =
        (true_positive_count > 0).then(|| escaped_defect_count as f64 / true_positive_count as f64);
    let cost_per_defect_selected = (selected_positive_count > 0)
        .then(|| selection.total_selected_cost as f64 / selected_positive_count as f64);
    let cost_per_defect_captured = (metrology_detected_true_positive_count > 0).then(|| {
        selection.total_selected_cost as f64 / metrology_detected_true_positive_count as f64
    });
    let true_defect_prevalence = true_positive_count as f64 / population_count as f64;
    let selected_sample_prevalence = selected_positive_count as f64 / selected_count as f64;
    let horvitz_thompson_prevalence = ht_positive_total / population_count as f64;
    let hajek_prevalence = ht_positive_total / weight_sum;
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
        let finite_population_correction = 1.0 - sample_count as f64 / *residual_population as f64;
        positive_total_variance += (*residual_population as f64).powi(2)
            * finite_population_correction
            * positive_variance
            / sample_count as f64;
        brier_total_variance +=
            (*residual_population as f64).powi(2) * finite_population_correction * brier_variance
                / sample_count as f64;
    }
    let standard_error_ht_prevalence =
        design_variance_qualified.then(|| positive_total_variance.sqrt() / population_count as f64);
    let mut hajek_total_variance = 0.0;
    let mut hajek_variance_qualified = true;
    for (stratum, residual_population) in &selection.residual_population_by_stratum {
        let sample_count = selection.audit_quota_by_stratum[stratum];
        if sample_count == *residual_population {
            continue;
        }
        if sample_count < 2 {
            hajek_variance_qualified = false;
            continue;
        }
        let Some(sample) = randomized_sample_by_stratum.get(stratum) else {
            hajek_variance_qualified = false;
            continue;
        };
        let residual_values: Vec<f64> = sample
            .iter()
            .map(|(positive, _)| *positive - hajek_prevalence)
            .collect();
        let Some(residual_variance) = finite_population_sample_variance(&residual_values) else {
            hajek_variance_qualified = false;
            continue;
        };
        let finite_population_correction = 1.0 - sample_count as f64 / *residual_population as f64;
        hajek_total_variance += (*residual_population as f64).powi(2)
            * finite_population_correction
            * residual_variance
            / sample_count as f64;
    }
    let standard_error_hajek_prevalence =
        hajek_variance_qualified.then(|| hajek_total_variance.sqrt() / weight_sum);
    let standard_error_ht_mean_brier =
        design_variance_qualified.then(|| brier_total_variance.sqrt() / population_count as f64);

    let assay_observed_ht_positive_rate = assay_ht_positive_total / population_count as f64;
    let assay_outcomes_sha256 = json_sha256(&assay_outcomes)?;
    let assay_informativeness = episode.metrology_characteristics.sensitivity
        + episode.metrology_characteristics.specificity
        - 1.0;
    let assay_adjusted_ht_prevalence = (assay_informativeness.abs() > 1.0e-12).then(|| {
        (assay_observed_ht_positive_rate - (1.0 - episode.metrology_characteristics.specificity))
            / assay_informativeness
    });

    let mut assay_positive_total_variance = 0.0;
    let mut assay_design_variance_qualified = true;
    for (stratum, residual_population) in &selection.residual_population_by_stratum {
        let sample_count = selection.audit_quota_by_stratum[stratum];
        if sample_count == *residual_population {
            continue;
        }
        if sample_count < 2 {
            assay_design_variance_qualified = false;
            continue;
        }
        let Some(sample) = randomized_assay_by_stratum.get(stratum) else {
            assay_design_variance_qualified = false;
            continue;
        };
        let Some(sample_variance) = finite_population_sample_variance(sample) else {
            assay_design_variance_qualified = false;
            continue;
        };
        let finite_population_correction = 1.0 - sample_count as f64 / *residual_population as f64;
        assay_positive_total_variance +=
            (*residual_population as f64).powi(2) * finite_population_correction * sample_variance
                / sample_count as f64;
    }
    let assay_estimator_qualified =
        assay_design_variance_qualified && assay_informativeness.abs() > 1.0e-12;
    let assay_adjusted_ht_standard_error = assay_estimator_qualified.then(|| {
        assay_positive_total_variance.sqrt()
            / (population_count as f64 * assay_informativeness.abs())
    });
    let assay_adjusted_ht_absolute_error =
        assay_adjusted_ht_prevalence.map(|estimate| (estimate - true_defect_prevalence).abs());
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
        metrology_detected_true_positive_count,
        metrology_false_positive_count,
        metrology_false_negative_count,
        metrology_true_negative_count,
        metrology_observed_positive_count,
        metrology_empirical_sensitivity,
        metrology_empirical_specificity,
        metrology_positive_predictive_value,
        metrology_negative_predictive_value,
        total_undetected_defect_count,
        total_undetected_defect_fraction,
        detection_capture_fraction,
        defect_capture_fraction,
        escaped_defect_fraction,
        true_defect_prevalence,
        metrology_sensitivity: episode.metrology_characteristics.sensitivity,
        metrology_specificity: episode.metrology_characteristics.specificity,
        assay_measurement_seed: episode.measurement_seed,
        assay_outcomes_sha256,
        assay_observed_ht_positive_rate,
        assay_adjusted_ht_prevalence,
        assay_adjusted_ht_absolute_error,
        assay_adjusted_ht_standard_error,
        assay_estimator_qualified,
        mean_latent_process_state: latent_process_state_total / population_count as f64,
        mean_defect_score: defect_score_total / population_count as f64,
        selected_sample_prevalence,
        horvitz_thompson_prevalence,
        horvitz_thompson_absolute_error: (horvitz_thompson_prevalence - true_defect_prevalence).abs(),
        standard_error_ht_prevalence,
        hajek_prevalence,
        hajek_absolute_error: (hajek_prevalence - true_defect_prevalence).abs(),
        standard_error_hajek_prevalence,
        horvitz_thompson_mean_brier,
        standard_error_ht_mean_brier,
        design_variance_qualified,
        true_population_mean_brier,
        measured_fraction,
        measurement_reduction_fraction: 1.0 - measured_fraction,
        total_selected_cost: selection.total_selected_cost,
        cost_per_defect_selected,
        cost_per_defect_captured,
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
    fn config(regime: SyntheticFabRegimeV1, seed: u64) -> SyntheticFabScenarioConfigV1 {
        SyntheticFabScenarioConfigV1 {
            schema: REPLAY_SCENARIO_SCHEMA_V1.into(),
            scenario_id: "synthetic-cmp-drift-01".into(),
            seed,
            measurement_seed: seed ^ 0x4d45_5452_4f4c,
            wafer_count: 128,
            tool_count: 2,
            chamber_count: 2,
            regime,
            missingness_mechanism: SyntheticFabMissingnessMechanismV1::Mcar,
            sensor_dropout_probability: 0.12,
            measurement_noise_stddev: 0.08,
            defect_threshold: 0.62,
            metrology_characteristics: SyntheticMetrologyCharacteristicsV1 {
                sensitivity: 0.95,
                specificity: 0.99,
            },
            label_delay_steps: 5,
        }
    }

    fn policy_quotas(
        policy_view: &SyntheticFabPolicyViewV1,
        quota_per_stratum: usize,
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::<String, usize>::new();
        for observed in &policy_view.observations {
            *counts.entry(observed.stratum_id.clone()).or_default() += 1;
        }
        counts
            .into_keys()
            .map(|stratum| (stratum, quota_per_stratum))
            .collect()
    }

    #[derive(Clone, Copy)]
    struct PolicyWeightsV1 {
        risk: f64,
        uncertainty: f64,
        evsi: f64,
    }

    fn policy_config(
        policy_view: &SyntheticFabPolicyViewV1,
        policy_id: &str,
        exploit_budget: u64,
        audit_budget: u64,
        quota_per_stratum: usize,
        seed: u64,
    ) -> SyntheticMetrologyPolicyConfigV1 {
        policy_config_with_weights(
            policy_view,
            policy_id,
            exploit_budget,
            audit_budget,
            quota_per_stratum,
            seed,
            PolicyWeightsV1 {
                risk: 1.0,
                uncertainty: 0.0,
                evsi: 0.0,
            },
        )
    }

    fn policy_config_with_weights(
        policy_view: &SyntheticFabPolicyViewV1,
        policy_id: &str,
        exploit_budget: u64,
        audit_budget: u64,
        quota_per_stratum: usize,
        seed: u64,
        weights: PolicyWeightsV1,
    ) -> SyntheticMetrologyPolicyConfigV1 {
        SyntheticMetrologyPolicyConfigV1 {
            policy_id: policy_id.into(),
            seed,
            total_cost_budget: exploit_budget + audit_budget,
            exploit_cost_budget: exploit_budget,
            audit_cost_budget: audit_budget,
            risk_weight: weights.risk,
            uncertainty_weight: weights.uncertainty,
            evsi_weight: weights.evsi,
            audit_quota_by_stratum: policy_quotas(policy_view, quota_per_stratum),
        }
    }

    fn fake_selection(episode: &SyntheticFabEpisodeV1) -> MetrologyAuditSelectionReportV1 {
        let semantics = synthetic_defect_semantics_v1(episode.defect_threshold)
            .expect("synthetic target semantics should seal");
        let config = policy_config(episode.policy_view(), "synthetic-test-policy", 0, 32, 8, 7);
        let policy =
            build_synthetic_metrology_policy_v1(episode.policy_view(), &semantics, &config)
                .expect("synthetic policy should build");
        select_metrology_audit_v1(&policy).expect("seeded audit selection should validate")
    }

    #[test]
    fn binary_evsi_matches_known_bayes_decision_calculation() {
        let report = calculate_synthetic_binary_evsi_v1(
            0.10,
            SyntheticMetrologyCharacteristicsV1 {
                sensitivity: 0.95,
                specificity: 0.99,
            },
            SyntheticBinaryDecisionLossV1 {
                false_negative_cost: 100.0,
                false_positive_cost: 5.0,
                measurement_cost: 2.0,
            },
        )
        .unwrap();

        assert!((report.probability_assay_positive - 0.104).abs() < 1.0e-12);
        assert!((report.probability_assay_negative - 0.896).abs() < 1.0e-12);
        assert!((report.expected_loss_without_measurement - 4.5).abs() < 1.0e-12);
        assert!((report.expected_loss_after_measurement - 0.545).abs() < 1.0e-12);
        assert!((report.expected_value_of_sample_information - 3.955).abs() < 1.0e-12);
        assert!((report.net_value_after_measurement_cost - 1.955).abs() < 1.0e-12);
        assert_eq!(
            report.decision_without_measurement,
            SyntheticBinaryDecisionV1::Hold
        );
        assert_eq!(
            report.decision_after_positive,
            Some(SyntheticBinaryDecisionV1::Hold)
        );
        assert_eq!(
            report.decision_after_negative,
            Some(SyntheticBinaryDecisionV1::Accept)
        );
    }

    #[test]
    fn uninformative_assay_has_zero_decision_theoretic_evsi() {
        let report = calculate_synthetic_binary_evsi_v1(
            0.10,
            SyntheticMetrologyCharacteristicsV1 {
                sensitivity: 0.70,
                specificity: 0.30,
            },
            SyntheticBinaryDecisionLossV1 {
                false_negative_cost: 100.0,
                false_positive_cost: 5.0,
                measurement_cost: 0.0,
            },
        )
        .unwrap();
        assert!(report.expected_value_of_sample_information.abs() < 1.0e-12);
        assert!(report.normalized_evsi_score.abs() < 1.0e-12);
        assert!(report.normalized_net_value_score.abs() < 1.0e-12);
        assert!(
            (report.expected_loss_after_measurement - report.expected_loss_without_measurement)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn measurement_cost_hurdle_distinguishes_gross_evsi_from_net_value() {
        let report = calculate_synthetic_binary_evsi_v1(
            0.10,
            SyntheticMetrologyCharacteristicsV1 {
                sensitivity: 0.95,
                specificity: 0.99,
            },
            SyntheticBinaryDecisionLossV1 {
                false_negative_cost: 100.0,
                false_positive_cost: 5.0,
                measurement_cost: 10.0,
            },
        )
        .unwrap();
        assert!(report.expected_value_of_sample_information > 0.0);
        assert!(report.net_value_after_measurement_cost < 0.0);
        assert_eq!(report.normalized_net_value_score, 0.0);
    }

    #[test]
    fn decision_evsi_policy_uses_loss_and_assay_profile_bound_scores() {
        let scenario = config(SyntheticFabRegimeV1::GradualProcessDrift, 83);
        let episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let semantics = synthetic_defect_semantics_v1(scenario.defect_threshold).unwrap();
        let view = episode.policy_view();
        let loss = SyntheticBinaryDecisionLossV1 {
            false_negative_cost: 100.0,
            false_positive_cost: 5.0,
            measurement_cost: 2.0,
        };
        let policy_config = policy_config_with_weights(
            view,
            "decision-evsi",
            16,
            16,
            4,
            109,
            PolicyWeightsV1 {
                risk: 0.0,
                uncertainty: 0.0,
                evsi: 1.0,
            },
        );
        let policy = build_synthetic_decision_evsi_policy_v1(
            view,
            &semantics,
            scenario.metrology_characteristics,
            loss,
            &policy_config,
        )
        .unwrap();
        assert!(
            policy
                .candidates
                .iter()
                .any(|candidate| { candidate.expected_value_of_information > 0.0 })
        );
        assert!(
            policy.candidates.iter().all(|candidate| {
                (0.0..=1.0).contains(&candidate.expected_value_of_information)
            })
        );

        let run = run_synthetic_decision_evsi_policy_v1(
            &episode,
            view,
            &semantics,
            scenario.metrology_characteristics,
            loss,
            &policy_config,
        )
        .unwrap();
        assert_eq!(run.selection.total_selected_cost, 32);
        assert_eq!(run.selection.exploit_count, 16);
        assert_eq!(run.selection.randomized_audit_count, 16);
    }

    #[test]
    fn binary_evsi_rejects_invalid_probabilities_and_losses() {
        assert!(
            calculate_synthetic_binary_evsi_v1(
                1.1,
                SyntheticMetrologyCharacteristicsV1 {
                    sensitivity: 0.95,
                    specificity: 0.99,
                },
                SyntheticBinaryDecisionLossV1 {
                    false_negative_cost: 100.0,
                    false_positive_cost: 5.0,
                    measurement_cost: 2.0,
                }
            )
            .is_err()
        );
        assert!(
            calculate_synthetic_binary_evsi_v1(
                0.1,
                SyntheticMetrologyCharacteristicsV1 {
                    sensitivity: 0.95,
                    specificity: 0.99,
                },
                SyntheticBinaryDecisionLossV1 {
                    false_negative_cost: -1.0,
                    false_positive_cost: 5.0,
                    measurement_cost: 2.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn mcar_mar_and_mnar_apply_distinct_dropout_mechanisms() {
        let mcar = conditional_dropout_probability(
            SyntheticFabMissingnessMechanismV1::Mcar,
            0.12,
            1.0,
            true,
        );
        let mar = conditional_dropout_probability(
            SyntheticFabMissingnessMechanismV1::Mar,
            0.12,
            1.0,
            false,
        );
        let mnar = conditional_dropout_probability(
            SyntheticFabMissingnessMechanismV1::Mnar,
            0.12,
            0.0,
            true,
        );
        assert!((mcar - 0.12).abs() < 1.0e-12);
        assert!((mar - 0.57).abs() < 1.0e-12);
        assert!((mnar - 0.67).abs() < 1.0e-12);
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
        assert!(!serialized.contains("metrology_characteristics"));
        assert!(!serialized.contains("sensitivity"));
        assert!(!serialized.contains("specificity"));
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
        assert_eq!(report.measurement_reduction_fraction, 0.75);
        assert_eq!(report.total_selected_cost, 32);
        assert!(report.horvitz_thompson_prevalence.is_finite());
        assert!(report.horvitz_thompson_absolute_error.is_finite());
        assert!(report.design_variance_qualified);
        assert!(report.standard_error_ht_prevalence.is_some());
        assert!(report.standard_error_ht_mean_brier.is_some());
        assert_eq!(report.matured_label_count_at_horizon, 123);
    }

    #[test]
    fn imperfect_metrology_separates_sampling_escapes_from_assay_errors() {
        let mut scenario = config(SyntheticFabRegimeV1::Stable, 61);
        scenario.metrology_characteristics = SyntheticMetrologyCharacteristicsV1 {
            sensitivity: 0.0,
            specificity: 0.0,
        };
        let episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let selection = fake_selection(&episode);
        let report = evaluate_synthetic_metrology_policy_v1(&episode, &selection).unwrap();

        assert_eq!(report.metrology_detected_true_positive_count, 0);
        assert_eq!(
            report.metrology_false_negative_count,
            report.selected_positive_count
        );
        assert_eq!(
            report.metrology_false_positive_count,
            report.selected_count - report.selected_positive_count
        );
        assert_eq!(
            report.total_undetected_defect_count,
            report.true_positive_count
        );
        assert_eq!(report.metrology_sensitivity, 0.0);
        assert_eq!(report.metrology_specificity, 0.0);
        assert_eq!(report.metrology_true_negative_count, 0);
        assert_eq!(
            report.metrology_observed_positive_count,
            report.selected_count
        );
        assert!(report.metrology_empirical_sensitivity.is_none());
        assert_eq!(report.metrology_empirical_specificity, Some(0.0));
        assert_eq!(report.metrology_positive_predictive_value, Some(0.0));
        assert!(report.metrology_negative_predictive_value.is_none());
        assert!(report.detection_capture_fraction.is_none());
        assert!(report.assay_estimator_qualified);
        assert!(report.assay_adjusted_ht_prevalence.is_some());
    }

    #[test]
    fn simulated_metrology_results_and_evaluation_digests_are_reproducible() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 65)).unwrap();
        let selection = fake_selection(&episode);
        let first = evaluate_synthetic_metrology_policy_v1(&episode, &selection).unwrap();
        let second = evaluate_synthetic_metrology_policy_v1(&episode, &selection).unwrap();

        assert_eq!(
            first.metrology_detected_true_positive_count,
            second.metrology_detected_true_positive_count
        );
        assert_eq!(
            first.metrology_false_positive_count,
            second.metrology_false_positive_count
        );
        assert_eq!(
            first.metrology_false_negative_count,
            second.metrology_false_negative_count
        );
        assert_eq!(
            first.assay_adjusted_ht_prevalence,
            second.assay_adjusted_ht_prevalence
        );
        assert_eq!(first.evaluation_sha256, second.evaluation_sha256);
    }

    #[test]
    fn uninformative_assay_does_not_claim_adjusted_population_prevalence() {
        let mut scenario = config(SyntheticFabRegimeV1::Stable, 63);
        scenario.metrology_characteristics = SyntheticMetrologyCharacteristicsV1 {
            sensitivity: 1.0,
            specificity: 0.0,
        };
        let episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let selection = fake_selection(&episode);
        let report = evaluate_synthetic_metrology_policy_v1(&episode, &selection).unwrap();

        assert_eq!(report.assay_observed_ht_positive_rate, 1.0);
        assert!(report.assay_adjusted_ht_prevalence.is_none());
        assert!(report.assay_adjusted_ht_standard_error.is_none());
        assert!(!report.assay_estimator_qualified);
    }

    #[test]
    fn randomized_audit_estimators_recover_finite_population_over_paired_seed_replays() {
        let mut scenario = config(SyntheticFabRegimeV1::GradualProcessDrift, 101);
        scenario.metrology_characteristics = SyntheticMetrologyCharacteristicsV1 {
            sensitivity: 0.92,
            specificity: 0.97,
        };
        let base_episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let semantics = synthetic_defect_semantics_v1(scenario.defect_threshold).unwrap();
        let base_observation_digest = json_sha256(&base_episode.policy_view.observations).unwrap();
        let truth =
            evaluate_synthetic_metrology_policy_v1(&base_episode, &fake_selection(&base_episode))
                .unwrap()
                .true_defect_prevalence;

        let mut ht_estimates = Vec::new();
        let mut hajek_estimates = Vec::new();
        let mut assay_adjusted_estimates = Vec::new();
        let mut ht_covered = 0_usize;
        let mut hajek_covered = 0_usize;
        let replay_count = 256_usize;

        for seed in 0..replay_count {
            let mut replicate_scenario = scenario.clone();
            replicate_scenario.measurement_seed = seed as u64;
            let replicate_episode = generate_synthetic_fab_episode_v1(&replicate_scenario).unwrap();
            assert_eq!(
                json_sha256(&replicate_episode.policy_view.observations).unwrap(),
                base_observation_digest
            );
            let view = replicate_episode.policy_view();
            let policy = policy_config(view, "stratified-random", 0, 32, 8, seed as u64);
            let run =
                run_synthetic_metrology_policy_v1(&replicate_episode, view, &semantics, &policy)
                    .unwrap();
            let report = run.evaluation;
            assert!(report.design_variance_qualified);
            assert!(report.assay_estimator_qualified);
            assert_eq!(report.assay_measurement_seed, seed as u64);
            assert!(is_lowercase_sha256(&report.assay_outcomes_sha256));

            let ht = report.horvitz_thompson_prevalence;
            let ht_se = report.standard_error_ht_prevalence.unwrap();
            if (ht - truth).abs() <= 1.96 * ht_se {
                ht_covered += 1;
            }
            ht_estimates.push(ht);

            let hajek = report.hajek_prevalence;
            let hajek_se = report.standard_error_hajek_prevalence.unwrap();
            if (hajek - truth).abs() <= 1.96 * hajek_se {
                hajek_covered += 1;
            }
            hajek_estimates.push(hajek);
            assay_adjusted_estimates.push(report.assay_adjusted_ht_prevalence.unwrap());
        }

        let mean_ht = ht_estimates.iter().sum::<f64>() / replay_count as f64;
        let mean_hajek = hajek_estimates.iter().sum::<f64>() / replay_count as f64;
        let mean_assay_adjusted =
            assay_adjusted_estimates.iter().sum::<f64>() / replay_count as f64;
        let ht_coverage = ht_covered as f64 / replay_count as f64;
        let hajek_coverage = hajek_covered as f64 / replay_count as f64;

        assert!((mean_ht - truth).abs() < 0.02);
        assert!((mean_hajek - truth).abs() < 0.02);
        assert!((mean_assay_adjusted - truth).abs() < 0.03);
        assert!((0.88..=0.99).contains(&ht_coverage));
        assert!((0.88..=0.99).contains(&hajek_coverage));
    }

    #[test]
    fn policy_runner_compares_random_audit_and_risk_exploit_at_equal_budget() {
        let episode = generate_synthetic_fab_episode_v1(&config(
            SyntheticFabRegimeV1::GradualProcessDrift,
            31,
        ))
        .unwrap();
        let semantics = synthetic_defect_semantics_v1(0.62).unwrap();
        let view = episode.policy_view();

        let random_policy = policy_config(view, "stratified-random", 0, 32, 8, 41);
        let random_run =
            run_synthetic_metrology_policy_v1(&episode, view, &semantics, &random_policy).unwrap();

        let risk_policy = policy_config(view, "risk-plus-audit", 16, 16, 4, 41);
        let risk_run =
            run_synthetic_metrology_policy_v1(&episode, view, &semantics, &risk_policy).unwrap();

        assert_eq!(random_run.selection.total_selected_cost, 32);
        assert_eq!(risk_run.selection.total_selected_cost, 32);
        assert_eq!(random_run.evaluation.selected_count, 32);
        assert_eq!(risk_run.evaluation.selected_count, 32);
        assert_eq!(random_run.selection.exploit_count, 0);
        assert_eq!(random_run.selection.randomized_audit_count, 32);
        assert_eq!(risk_run.selection.exploit_count, 16);
        assert_eq!(risk_run.selection.randomized_audit_count, 16);
        assert!(random_run.evaluation.design_variance_qualified);
        assert!(risk_run.evaluation.design_variance_qualified);
        assert!(risk_run.evaluation.defect_capture_fraction.is_some());
        assert!(risk_run.evaluation.escaped_defect_fraction.is_some());
        assert!(risk_run.evaluation.cost_per_defect_captured.is_some());
        assert_eq!(risk_run.evaluation.measurement_reduction_fraction, 0.75);
    }

    #[test]
    fn risk_uncertainty_and_evsi_proxy_policies_run_at_matched_budgets() {
        let episode = generate_synthetic_fab_episode_v1(&config(
            SyntheticFabRegimeV1::GradualProcessDrift,
            37,
        ))
        .unwrap();
        let view = episode.policy_view();
        let semantics = synthetic_defect_semantics_v1(0.62).unwrap();
        let policy_specs = [
            (
                "risk-only",
                PolicyWeightsV1 {
                    risk: 1.0,
                    uncertainty: 0.0,
                    evsi: 0.0,
                },
            ),
            (
                "uncertainty-only",
                PolicyWeightsV1 {
                    risk: 0.0,
                    uncertainty: 1.0,
                    evsi: 0.0,
                },
            ),
            (
                "evsi-proxy-only",
                PolicyWeightsV1 {
                    risk: 0.0,
                    uncertainty: 0.0,
                    evsi: 1.0,
                },
            ),
        ];

        for (policy_id, weights) in policy_specs {
            let policy = policy_config_with_weights(view, policy_id, 16, 16, 4, 53, weights);
            let run =
                run_synthetic_metrology_policy_v1(&episode, view, &semantics, &policy).unwrap();
            assert_eq!(run.selection.total_selected_cost, 32);
            assert_eq!(run.selection.exploit_count, 16);
            assert_eq!(run.selection.randomized_audit_count, 16);
            assert!(run.evaluation.design_variance_qualified);
        }
    }

    #[test]
    fn paired_tournament_replays_all_policies_on_the_same_scenario_seeds() {
        let scenarios = vec![
            config(SyntheticFabRegimeV1::Stable, 71),
            config(SyntheticFabRegimeV1::GradualProcessDrift, 72),
            config(SyntheticFabRegimeV1::AbruptProcessShift, 73),
        ];
        let first_episode = generate_synthetic_fab_episode_v1(&scenarios[0]).unwrap();
        let view = first_episode.policy_view();
        let policies = vec![
            policy_config(view, "random-audit", 0, 32, 8, 101),
            policy_config(view, "risk-exploit-audit", 16, 16, 4, 101),
            policy_config_with_weights(
                view,
                "uncertainty-exploit-audit",
                16,
                16,
                4,
                101,
                PolicyWeightsV1 {
                    risk: 0.0,
                    uncertainty: 1.0,
                    evsi: 0.0,
                },
            ),
        ];

        let first = run_synthetic_metrology_policy_tournament_v1(
            "paired-three-seed-test",
            "random-audit",
            &scenarios,
            &policies,
        )
        .unwrap();
        let second = run_synthetic_metrology_policy_tournament_v1(
            "paired-three-seed-test",
            "random-audit",
            &scenarios,
            &policies,
        )
        .unwrap();

        assert_eq!(first.paired_scenario_count, 3);
        assert_eq!(first.policy_count, 3);
        assert_eq!(first.runs.len(), 9);
        assert_eq!(first.summaries.len(), 3);
        assert_eq!(first.reference_policy_id, "random-audit");
        assert_eq!(first.paired_comparisons.len(), 2);
        assert!(
            first
                .paired_comparisons
                .iter()
                .all(|comparison| comparison.paired_scenario_count == 3)
        );
        assert!(
            first
                .paired_comparisons
                .iter()
                .all(|comparison| comparison.detection_capture_support >= 2)
        );
        assert!(
            first
                .paired_comparisons
                .iter()
                .all(|comparison| comparison.undetected_defect_fraction_support >= 2)
        );
        assert_eq!(first.tournament_sha256, second.tournament_sha256);
        for summary in &first.summaries {
            assert_eq!(summary.run_count, 3);
            assert_eq!(summary.measurement_reduction_mean, 0.75);
            assert_eq!(summary.measurement_reduction_sample_stddev, 0.0);
        }
    }

    #[test]
    fn paired_tournament_rejects_unmatched_selection_seeds() {
        let scenario = config(SyntheticFabRegimeV1::Stable, 77);
        let episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let view = episode.policy_view();
        let first = policy_config(view, "seed-a", 0, 32, 8, 101);
        let second = policy_config(view, "seed-b", 0, 32, 8, 102);
        assert!(
            run_synthetic_metrology_policy_tournament_v1(
                "unmatched-seed-test",
                "seed-a",
                &[scenario],
                &[first, second]
            )
            .is_err()
        );
    }

    #[test]
    fn paired_tournament_rejects_unmatched_measurement_budgets() {
        let scenario = config(SyntheticFabRegimeV1::Stable, 75);
        let episode = generate_synthetic_fab_episode_v1(&scenario).unwrap();
        let view = episode.policy_view();
        let first = policy_config(view, "budget-a", 0, 32, 8, 101);
        let second = policy_config(view, "budget-b", 0, 16, 4, 101);
        assert!(
            run_synthetic_metrology_policy_tournament_v1(
                "unmatched-budget-test",
                "budget-a",
                &[scenario],
                &[first, second]
            )
            .is_err()
        );
    }

    #[test]
    fn policy_runner_rejects_an_observation_packet_from_another_episode() {
        let first =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 51)).unwrap();
        let second =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 52)).unwrap();
        let semantics = synthetic_defect_semantics_v1(0.62).unwrap();
        let policy = policy_config(first.policy_view(), "mismatched-packet", 0, 32, 8, 9);
        assert!(
            run_synthetic_metrology_policy_v1(&first, second.policy_view(), &semantics, &policy)
                .is_err()
        );
    }

    #[test]
    fn evaluator_rejects_label_semantics_that_disagree_with_hidden_threshold() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 29)).unwrap();
        let mut selection = fake_selection(&episode);
        selection.binary_semantics.threshold = 0.50;
        crate::benchmark::metrology_semantics::seal_binary_measurand_semantics_v1(
            &mut selection.binary_semantics,
        )
        .unwrap();
        for row in &mut selection.rows {
            row.class_semantics_sha256 = selection.binary_semantics.semantics_sha256.clone();
        }
        seal_selection_for_test(&mut selection);
        assert!(evaluate_synthetic_metrology_policy_v1(&episode, &selection).is_err());
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
    fn evaluator_rejects_a_periodic_draw_mislabeled_as_randomized_audit() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 25)).unwrap();
        let mut selection = fake_selection(&episode);
        let stratum = selection.rows[0].stratum_id.clone();
        let selected_index = selection
            .rows
            .iter()
            .position(|row| {
                row.stratum_id == stratum
                    && matches!(&row.method, MetrologySelectionMethodV1::RandomizedAudit)
            })
            .expect("stratum has randomized audit selection");
        let unselected_index = selection
            .rows
            .iter()
            .position(|row| {
                row.stratum_id == stratum
                    && matches!(&row.method, MetrologySelectionMethodV1::NotSelected)
            })
            .expect("stratum has residual unselected candidate");

        selection.rows[selected_index].selected = false;
        selection.rows[selected_index].method = MetrologySelectionMethodV1::NotSelected;
        selection.rows[unselected_index].selected = true;
        selection.rows[unselected_index].method = MetrologySelectionMethodV1::RandomizedAudit;
        seal_selection_for_test(&mut selection);
        assert!(evaluate_synthetic_metrology_policy_v1(&episode, &selection).is_err());
    }

    #[test]
    fn evaluator_rejects_a_resealed_but_false_inclusion_probability() {
        let episode =
            generate_synthetic_fab_episode_v1(&config(SyntheticFabRegimeV1::Stable, 27)).unwrap();
        let mut selection = fake_selection(&episode);
        selection.rows[0].inclusion_probability = 0.5;
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
