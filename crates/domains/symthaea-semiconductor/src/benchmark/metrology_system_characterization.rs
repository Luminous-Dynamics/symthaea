// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic crossed gauge study and ANOVA variance-component evaluation.
//!
//! The initial profile is a balanced part × appraiser × replicate design for
//! one instrument configuration. It deliberately does not claim full ISO/AIAG
//! MSA conformity or substitute for traceable physical reference artifacts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SYNTHETIC_MSA_SCENARIO_SCHEMA_V1: &str = "luminous.semi-bench.synthetic-msa-scenario.v1";
pub const SYNTHETIC_MSA_POLICY_VIEW_SCHEMA_V1: &str =
    "luminous.semi-bench.synthetic-msa-policy-view.v1";
pub const SYNTHETIC_MSA_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.synthetic-msa-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMsaScenarioConfigV1 {
    pub schema: String,
    pub scenario_id: String,
    pub seed: u64,
    pub part_count: usize,
    pub appraiser_count: usize,
    pub replicates_per_cell: usize,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub reference_value_start: f64,
    pub reference_value_step: f64,
    pub calibration_bias: f64,
    pub appraiser_effect_stddev: f64,
    pub part_appraiser_interaction_stddev: f64,
    pub repeatability_stddev: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMsaObservationV1 {
    pub observation_id: String,
    pub part_id: String,
    pub appraiser_id: String,
    pub instrument_id: String,
    pub replicate_index: usize,
    pub measurand_id: String,
    pub unit: String,
    pub reference_value: f64,
    pub measured_value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMsaPolicyViewV1 {
    pub schema: String,
    pub scenario_id: String,
    pub scenario_config_sha256: String,
    pub observation_packet_sha256: String,
    pub part_count: usize,
    pub appraiser_count: usize,
    pub replicates_per_cell: usize,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub observations: Vec<SyntheticMsaObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticMsaEvaluationReportV1 {
    pub schema: String,
    pub scenario_id: String,
    pub scenario_config_sha256: String,
    pub observation_packet_sha256: String,
    pub report_sha256: String,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub part_count: usize,
    pub appraiser_count: usize,
    pub replicates_per_cell: usize,
    pub observation_count: usize,
    pub grand_mean: f64,
    pub mean_bias: f64,
    pub bias_sample_stddev: Option<f64>,
    pub ms_part: f64,
    pub ms_appraiser: f64,
    pub ms_part_appraiser: f64,
    pub ms_repeatability: f64,
    pub estimated_part_variance_raw: f64,
    pub estimated_part_variance: f64,
    pub estimated_appraiser_variance_raw: f64,
    pub estimated_appraiser_variance: f64,
    pub estimated_part_appraiser_interaction_variance_raw: f64,
    pub estimated_part_appraiser_interaction_variance: f64,
    pub estimated_repeatability_variance: f64,
    pub numerical_variance_floor: f64,
    pub estimated_gauge_rr_variance: f64,
    pub true_part_variance: f64,
    pub known_appraiser_variance: f64,
    pub known_part_appraiser_interaction_variance: f64,
    pub known_repeatability_variance: f64,
    pub known_gauge_rr_variance: f64,
    pub gauge_rr_variance_absolute_error: f64,
    pub percent_gauge_rr_contribution: Option<f64>,
    /// Unrounded 1.41 × part standard deviation / gauge R&R standard deviation.
    pub ndc_unrounded: Option<f64>,
    pub negative_raw_component_count: usize,
    pub claim_limits: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SyntheticMsaError(pub String);

#[derive(Clone, Debug)]
struct SyntheticMsaKnownModelV1 {
    known_appraiser_variance: f64,
    known_part_appraiser_interaction_variance: f64,
    known_repeatability_variance: f64,
}

#[derive(Clone, Debug)]
pub struct SyntheticMsaEpisodeV1 {
    config: SyntheticMsaScenarioConfigV1,
    policy_view: SyntheticMsaPolicyViewV1,
    known_model: SyntheticMsaKnownModelV1,
}

impl SyntheticMsaEpisodeV1 {
    /// Returns the measurement records and reference values only.
    pub fn policy_view(&self) -> &SyntheticMsaPolicyViewV1 {
        &self.policy_view
    }
}

#[derive(Clone, Debug)]
struct MsaSplitMix64 {
    state: u64,
}

impl MsaSplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Irwin-Hall normal approximation with zero mean and unit variance.
    /// It provides deterministic, portable synthetic noise for known-answer tests.
    fn centered_noise(&mut self) -> f64 {
        (0..12).map(|_| self.unit_interval()).sum::<f64>() - 6.0
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, SyntheticMsaError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| SyntheticMsaError(format!("serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
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

fn validate_config(config: &SyntheticMsaScenarioConfigV1) -> Result<(), SyntheticMsaError> {
    if config.schema != SYNTHETIC_MSA_SCENARIO_SCHEMA_V1 {
        return Err(SyntheticMsaError("scenario.schema: unsupported".into()));
    }
    if config.scenario_id.trim().is_empty()
        || config.instrument_id.trim().is_empty()
        || config.measurand_id.trim().is_empty()
        || config.unit.trim().is_empty()
    {
        return Err(SyntheticMsaError(
            "scenario identity, instrument, measurand and unit must be non-empty".into(),
        ));
    }
    if !(2..=500).contains(&config.part_count)
        || !(2..=100).contains(&config.appraiser_count)
        || !(2..=100).contains(&config.replicates_per_cell)
    {
        return Err(SyntheticMsaError(
            "MSA design requires 2..=500 parts, 2..=100 appraisers and 2..=100 replicates".into(),
        ));
    }
    for (name, value) in [
        ("reference_value_start", config.reference_value_start),
        ("reference_value_step", config.reference_value_step),
        ("calibration_bias", config.calibration_bias),
        ("appraiser_effect_stddev", config.appraiser_effect_stddev),
        (
            "part_appraiser_interaction_stddev",
            config.part_appraiser_interaction_stddev,
        ),
        ("repeatability_stddev", config.repeatability_stddev),
    ] {
        if !value.is_finite() {
            return Err(SyntheticMsaError(format!(
                "scenario.{name}: expected finite value"
            )));
        }
    }
    if config.reference_value_step <= 0.0
        || config.appraiser_effect_stddev < 0.0
        || config.part_appraiser_interaction_stddev < 0.0
        || config.repeatability_stddev < 0.0
    {
        return Err(SyntheticMsaError(
            "reference step must be positive and standard deviations non-negative".into(),
        ));
    }
    config
        .part_count
        .checked_mul(config.appraiser_count)
        .and_then(|value| value.checked_mul(config.replicates_per_cell))
        .filter(|value| *value <= 1_000_000)
        .ok_or_else(|| SyntheticMsaError("MSA design size exceeds 1,000,000 records".into()))?;
    Ok(())
}

/// Generates a balanced crossed part × appraiser × replicate study using one
/// instrument configuration and explicit reference artifact values.
pub fn generate_synthetic_msa_episode_v1(
    config: &SyntheticMsaScenarioConfigV1,
) -> Result<SyntheticMsaEpisodeV1, SyntheticMsaError> {
    validate_config(config)?;
    let scenario_config_sha256 = json_sha256(config)?;
    let mut rng = MsaSplitMix64::new(config.seed);
    let appraiser_effects: Vec<f64> = (0..config.appraiser_count)
        .map(|_| config.appraiser_effect_stddev * rng.centered_noise())
        .collect();
    let mut observations =
        Vec::with_capacity(config.part_count * config.appraiser_count * config.replicates_per_cell);

    for part_index in 0..config.part_count {
        let part_id = format!("artifact-{part_index:04}");
        let reference_value =
            config.reference_value_start + part_index as f64 * config.reference_value_step;
        for appraiser_index in 0..config.appraiser_count {
            let appraiser_id = format!("appraiser-{appraiser_index:03}");
            let interaction = config.part_appraiser_interaction_stddev * rng.centered_noise();
            for replicate_index in 0..config.replicates_per_cell {
                let repeatability_error = config.repeatability_stddev * rng.centered_noise();
                let measured_value = reference_value
                    + config.calibration_bias
                    + appraiser_effects[appraiser_index]
                    + interaction
                    + repeatability_error;
                let observation_id = format!(
                    "measurement-{part_index:04}-{appraiser_index:03}-{replicate_index:03}"
                );
                observations.push(SyntheticMsaObservationV1 {
                    observation_id,
                    part_id: part_id.clone(),
                    appraiser_id: appraiser_id.clone(),
                    instrument_id: config.instrument_id.clone(),
                    replicate_index,
                    measurand_id: config.measurand_id.clone(),
                    unit: config.unit.clone(),
                    reference_value,
                    measured_value,
                });
            }
        }
    }

    let observation_packet_sha256 = json_sha256(&observations)?;
    let policy_view = SyntheticMsaPolicyViewV1 {
        schema: SYNTHETIC_MSA_POLICY_VIEW_SCHEMA_V1.into(),
        scenario_id: config.scenario_id.clone(),
        scenario_config_sha256,
        observation_packet_sha256,
        part_count: config.part_count,
        appraiser_count: config.appraiser_count,
        replicates_per_cell: config.replicates_per_cell,
        instrument_id: config.instrument_id.clone(),
        measurand_id: config.measurand_id.clone(),
        unit: config.unit.clone(),
        observations,
    };

    Ok(SyntheticMsaEpisodeV1 {
        config: config.clone(),
        policy_view,
        known_model: SyntheticMsaKnownModelV1 {
            known_appraiser_variance: config.appraiser_effect_stddev.powi(2),
            known_part_appraiser_interaction_variance: config
                .part_appraiser_interaction_stddev
                .powi(2),
            known_repeatability_variance: config.repeatability_stddev.powi(2),
        },
    })
}

/// Evaluates a complete balanced crossed study using two-way random-effects
/// ANOVA method-of-moments estimates. Negative raw variance components are
/// retained in the report and separately truncated at zero for the constrained
/// variance-component summary.
pub fn evaluate_synthetic_msa_episode_v1(
    episode: &SyntheticMsaEpisodeV1,
) -> Result<SyntheticMsaEvaluationReportV1, SyntheticMsaError> {
    let view = &episode.policy_view;
    validate_config(&episode.config)?;
    if json_sha256(&episode.config)? != view.scenario_config_sha256 {
        return Err(SyntheticMsaError(
            "scenario config digest does not match evaluator inputs".into(),
        ));
    }
    if view.schema != SYNTHETIC_MSA_POLICY_VIEW_SCHEMA_V1
        || view.scenario_id != episode.config.scenario_id
        || view.part_count != episode.config.part_count
        || view.appraiser_count != episode.config.appraiser_count
        || view.replicates_per_cell != episode.config.replicates_per_cell
        || view.instrument_id != episode.config.instrument_id
        || view.measurand_id != episode.config.measurand_id
        || view.unit != episode.config.unit
    {
        return Err(SyntheticMsaError(
            "policy view identity/schema differs from frozen scenario configuration".into(),
        ));
    }
    if view.part_count < 2 || view.appraiser_count < 2 || view.replicates_per_cell < 2 {
        return Err(SyntheticMsaError(
            "MSA design degrees of freedom are insufficient".into(),
        ));
    }
    if json_sha256(&view.observations)? != view.observation_packet_sha256 {
        return Err(SyntheticMsaError(
            "observation packet digest does not match payload".into(),
        ));
    }
    let expected_count = view.part_count * view.appraiser_count * view.replicates_per_cell;
    if view.observations.len() != expected_count {
        return Err(SyntheticMsaError(
            "balanced MSA study is missing or has extra observations".into(),
        ));
    }

    let part_ids: Vec<String> = (0..view.part_count)
        .map(|index| format!("artifact-{index:04}"))
        .collect();
    let appraiser_ids: Vec<String> = (0..view.appraiser_count)
        .map(|index| format!("appraiser-{index:03}"))
        .collect();
    let part_index_by_id: BTreeMap<&str, usize> = part_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    let appraiser_index_by_id: BTreeMap<&str, usize> = appraiser_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();

    let mut identities = BTreeSet::new();
    let mut observation_ids = BTreeSet::new();
    let mut cell_values = vec![vec![Vec::<f64>::new(); view.appraiser_count]; view.part_count];
    let mut part_references = vec![None::<f64>; view.part_count];
    let mut bias_values = Vec::with_capacity(expected_count);

    for (index, observation) in view.observations.iter().enumerate() {
        let part_index = *part_index_by_id
            .get(observation.part_id.as_str())
            .ok_or_else(|| {
                SyntheticMsaError(format!(
                    "observations[{index}].part_id is outside the frozen design"
                ))
            })?;
        let appraiser_index = *appraiser_index_by_id
            .get(observation.appraiser_id.as_str())
            .ok_or_else(|| {
                SyntheticMsaError(format!(
                    "observations[{index}].appraiser_id is outside the frozen design"
                ))
            })?;
        if observation.instrument_id != episode.config.instrument_id
            || observation.measurand_id != episode.config.measurand_id
            || observation.unit != episode.config.unit
            || !observation.reference_value.is_finite()
            || !observation.measured_value.is_finite()
        {
            return Err(SyntheticMsaError(format!(
                "observations[{index}]: invalid identity, unit or numeric value"
            )));
        }
        if observation.replicate_index >= view.replicates_per_cell {
            return Err(SyntheticMsaError(format!(
                "observations[{index}].replicate_index is outside the frozen design"
            )));
        }
        if !observation_ids.insert(observation.observation_id.as_str())
            || !identities.insert((part_index, appraiser_index, observation.replicate_index))
        {
            return Err(SyntheticMsaError(format!(
                "observations[{index}]: duplicate observation identity"
            )));
        }
        if part_references[part_index]
            .is_some_and(|reference| (reference - observation.reference_value).abs() > 1.0e-12)
        {
            return Err(SyntheticMsaError(format!(
                "observations[{index}]: reference value changes within part identity"
            )));
        }
        let expected_reference = episode.config.reference_value_start
            + part_index as f64 * episode.config.reference_value_step;
        if (expected_reference - observation.reference_value).abs() > 1.0e-12 {
            return Err(SyntheticMsaError(format!(
                "observations[{index}]: reference value differs from frozen artifact value"
            )));
        }
        part_references[part_index] = Some(observation.reference_value);
        cell_values[part_index][appraiser_index].push(observation.measured_value);
        bias_values.push(observation.measured_value - observation.reference_value);
    }

    for (part_index, part_cells) in cell_values.iter().enumerate() {
        for (appraiser_index, values) in part_cells.iter().enumerate() {
            if values.len() != view.replicates_per_cell {
                return Err(SyntheticMsaError(format!(
                    "cell artifact-{part_index:04}/appraiser-{appraiser_index:03} is incomplete"
                )));
            }
        }
    }

    let part_count = view.part_count;
    let appraiser_count = view.appraiser_count;
    let replicate_count = view.replicates_per_cell;
    let total_count = expected_count as f64;

    let mut part_means = vec![0.0; part_count];
    let mut appraiser_means = vec![0.0; appraiser_count];
    let mut cell_means = vec![vec![0.0; appraiser_count]; part_count];
    let grand_mean = view
        .observations
        .iter()
        .map(|observation| observation.measured_value)
        .sum::<f64>()
        / total_count;

    for part_index in 0..part_count {
        for appraiser_index in 0..appraiser_count {
            let values = &cell_values[part_index][appraiser_index];
            let mean = values.iter().sum::<f64>() / replicate_count as f64;
            cell_means[part_index][appraiser_index] = mean;
            part_means[part_index] += values.iter().sum::<f64>();
            appraiser_means[appraiser_index] += values.iter().sum::<f64>();
        }
    }
    for mean in &mut part_means {
        *mean /= (appraiser_count * replicate_count) as f64;
    }
    for mean in &mut appraiser_means {
        *mean /= (part_count * replicate_count) as f64;
    }

    let mut ss_repeatability = 0.0;
    let mut ss_interaction = 0.0;
    let mut ss_part = 0.0;
    let mut ss_appraiser = 0.0;

    for part_index in 0..part_count {
        for appraiser_index in 0..appraiser_count {
            for value in &cell_values[part_index][appraiser_index] {
                ss_repeatability += (value - cell_means[part_index][appraiser_index]).powi(2);
            }
            let interaction_residual = cell_means[part_index][appraiser_index]
                - part_means[part_index]
                - appraiser_means[appraiser_index]
                + grand_mean;
            ss_interaction += interaction_residual.powi(2);
        }
        ss_part += (part_means[part_index] - grand_mean).powi(2);
    }
    for mean in &appraiser_means {
        ss_appraiser += (mean - grand_mean).powi(2);
    }

    let df_part = (part_count - 1) as f64;
    let df_appraiser = (appraiser_count - 1) as f64;
    let df_interaction = ((part_count - 1) * (appraiser_count - 1)) as f64;
    let df_repeatability = (part_count * appraiser_count * (replicate_count - 1)) as f64;
    let ms_repeatability = ss_repeatability / df_repeatability;
    let ms_interaction = replicate_count as f64 * ss_interaction / df_interaction;
    let ms_part = (appraiser_count * replicate_count) as f64 * ss_part / df_part;
    let ms_appraiser = (part_count * replicate_count) as f64 * ss_appraiser / df_appraiser;

    let estimated_interaction_raw = (ms_interaction - ms_repeatability) / replicate_count as f64;
    let estimated_appraiser_raw =
        (ms_appraiser - ms_interaction) / (part_count * replicate_count) as f64;
    let estimated_part_raw =
        (ms_part - ms_interaction) / (appraiser_count * replicate_count) as f64;

    let numerical_variance_floor = f64::EPSILON * grand_mean.abs().max(1.0).powi(2) * 16.0;
    let constrained_component = |raw: f64| {
        let non_negative = raw.max(0.0);
        if non_negative <= numerical_variance_floor {
            0.0
        } else {
            non_negative
        }
    };
    let estimated_interaction = constrained_component(estimated_interaction_raw);
    let estimated_appraiser = constrained_component(estimated_appraiser_raw);
    let estimated_part = constrained_component(estimated_part_raw);
    let estimated_repeatability = constrained_component(ms_repeatability);
    let estimated_gauge_rr = estimated_appraiser + estimated_interaction + estimated_repeatability;
    let reference_values: Vec<f64> = part_references
        .into_iter()
        .map(|value| value.expect("balanced design validates every part reference"))
        .collect();
    let true_part_variance = sample_variance(&reference_values).unwrap_or(0.0);
    let estimated_total_variance = estimated_part + estimated_gauge_rr;
    let percent_gauge_rr_contribution =
        (estimated_total_variance > 0.0).then(|| estimated_gauge_rr / estimated_total_variance);
    let ndc_unrounded =
        (estimated_gauge_rr > 0.0).then(|| 1.41 * (estimated_part / estimated_gauge_rr).sqrt());
    let mean_bias = bias_values.iter().sum::<f64>() / bias_values.len() as f64;
    let bias_sample_stddev = sample_variance(&bias_values).map(f64::sqrt);
    let known_appraiser_variance = episode.known_model.known_appraiser_variance;
    let known_interaction_variance = episode
        .known_model
        .known_part_appraiser_interaction_variance;
    let known_repeatability_variance = episode.known_model.known_repeatability_variance;
    let known_gauge_rr_variance =
        known_appraiser_variance + known_interaction_variance + known_repeatability_variance;
    let negative_raw_component_count = [
        estimated_part_raw,
        estimated_appraiser_raw,
        estimated_interaction_raw,
    ]
    .into_iter()
    .filter(|value| *value < 0.0)
    .count();

    let mut report = SyntheticMsaEvaluationReportV1 {
        schema: SYNTHETIC_MSA_REPORT_SCHEMA_V1.into(),
        scenario_id: view.scenario_id.clone(),
        scenario_config_sha256: view.scenario_config_sha256.clone(),
        observation_packet_sha256: view.observation_packet_sha256.clone(),
        report_sha256: String::new(),
        instrument_id: view.instrument_id.clone(),
        measurand_id: view.measurand_id.clone(),
        unit: view.unit.clone(),
        part_count,
        appraiser_count,
        replicates_per_cell: replicate_count,
        observation_count: expected_count,
        grand_mean,
        mean_bias,
        bias_sample_stddev,
        ms_part,
        ms_appraiser,
        ms_part_appraiser: ms_interaction,
        ms_repeatability,
        estimated_part_variance_raw: estimated_part_raw,
        estimated_part_variance: estimated_part,
        estimated_appraiser_variance_raw: estimated_appraiser_raw,
        estimated_appraiser_variance: estimated_appraiser,
        estimated_part_appraiser_interaction_variance_raw: estimated_interaction_raw,
        estimated_part_appraiser_interaction_variance: estimated_interaction,
        estimated_repeatability_variance: estimated_repeatability,
        numerical_variance_floor,
        estimated_gauge_rr_variance: estimated_gauge_rr,
        true_part_variance,
        known_appraiser_variance,
        known_part_appraiser_interaction_variance: known_interaction_variance,
        known_repeatability_variance,
        known_gauge_rr_variance,
        gauge_rr_variance_absolute_error: (estimated_gauge_rr - known_gauge_rr_variance).abs(),
        percent_gauge_rr_contribution,
        ndc_unrounded,
        negative_raw_component_count,
        claim_limits: vec![
            "Synthetic crossed part-by-appraiser study only; not a physical gauge R&R".into(),
            "One instrument configuration is represented; instrument/site variance is not estimated".into(),
            "Known variance parameters are simulation inputs, not traceable reference evidence".into(),
            "Irwin-Hall noise is a deterministic normal approximation, not a validated instrument-error distribution".into(),
            "ANOVA estimates assume the balanced crossed random-effects model declared by this schema".into(),
            "No ISO 5725 or AIAG MSA conformance claim is established".into(),
        ],
    };
    report.report_sha256.clear();
    report.report_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> SyntheticMsaScenarioConfigV1 {
        SyntheticMsaScenarioConfigV1 {
            schema: SYNTHETIC_MSA_SCENARIO_SCHEMA_V1.into(),
            scenario_id: "synthetic-crossed-gauge-study-01".into(),
            seed,
            part_count: 40,
            appraiser_count: 12,
            replicates_per_cell: 6,
            instrument_id: "instrument-config-01".into(),
            measurand_id: "synthetic-dimension".into(),
            unit: "synthetic_mm".into(),
            reference_value_start: 10.0,
            reference_value_step: 0.1,
            calibration_bias: 0.15,
            appraiser_effect_stddev: 0.5,
            part_appraiser_interaction_stddev: 0.3,
            repeatability_stddev: 0.2,
        }
    }

    #[test]
    fn fixed_seed_reproduces_observations_and_msa_report() {
        let first = generate_synthetic_msa_episode_v1(&config(42)).unwrap();
        let second = generate_synthetic_msa_episode_v1(&config(42)).unwrap();
        let first_report = evaluate_synthetic_msa_episode_v1(&first).unwrap();
        let second_report = evaluate_synthetic_msa_episode_v1(&second).unwrap();
        assert_eq!(
            first.policy_view.observation_packet_sha256,
            second.policy_view.observation_packet_sha256
        );
        assert_eq!(first_report.report_sha256, second_report.report_sha256);
    }

    #[test]
    fn crossed_anova_recovers_planted_variance_components_within_tolerance() {
        let report = evaluate_synthetic_msa_episode_v1(
            &generate_synthetic_msa_episode_v1(&config(77)).unwrap(),
        )
        .unwrap();
        assert!(report.estimated_repeatability_variance > 0.0);
        assert!(report.estimated_appraiser_variance > 0.0);
        assert!(report.estimated_part_appraiser_interaction_variance > 0.0);
        assert!(
            (report.estimated_repeatability_variance - report.known_repeatability_variance).abs()
                < 0.03
        );
        assert!((report.estimated_gauge_rr_variance - report.known_gauge_rr_variance).abs() < 0.20);
        assert!((report.mean_bias - 0.15).abs() < 0.30);
    }

    #[test]
    fn zero_gauge_error_yields_zero_rr_and_undefined_ndc() {
        let mut scenario = config(11);
        scenario.calibration_bias = 0.0;
        scenario.appraiser_effect_stddev = 0.0;
        scenario.part_appraiser_interaction_stddev = 0.0;
        scenario.repeatability_stddev = 0.0;
        let report = evaluate_synthetic_msa_episode_v1(
            &generate_synthetic_msa_episode_v1(&scenario).unwrap(),
        )
        .unwrap();
        assert_eq!(report.estimated_gauge_rr_variance, 0.0);
        assert_eq!(report.estimated_repeatability_variance, 0.0);
        assert!(report.ndc_unrounded.is_none());
        assert_eq!(report.percent_gauge_rr_contribution, Some(0.0));
        assert_eq!(report.mean_bias, 0.0);
    }

    #[test]
    fn evaluator_rejects_a_missing_measurement_cell_even_after_resealing_packet() {
        let mut episode = generate_synthetic_msa_episode_v1(&config(19)).unwrap();
        episode.policy_view.observations.pop();
        episode.policy_view.observation_packet_sha256 =
            json_sha256(&episode.policy_view.observations).unwrap();
        assert!(evaluate_synthetic_msa_episode_v1(&episode).is_err());
    }

    #[test]
    fn evaluator_rejects_duplicate_measurement_identity() {
        let mut episode = generate_synthetic_msa_episode_v1(&config(23)).unwrap();
        episode.policy_view.observations[1].observation_id =
            episode.policy_view.observations[0].observation_id.clone();
        episode.policy_view.observation_packet_sha256 =
            json_sha256(&episode.policy_view.observations).unwrap();
        assert!(evaluate_synthetic_msa_episode_v1(&episode).is_err());
    }

    #[test]
    fn non_finite_and_negative_uncertainty_parameters_are_rejected() {
        let mut scenario = config(29);
        scenario.repeatability_stddev = f64::NAN;
        assert!(generate_synthetic_msa_episode_v1(&scenario).is_err());
        scenario.repeatability_stddev = -0.1;
        assert!(generate_synthetic_msa_episode_v1(&scenario).is_err());
    }
}
