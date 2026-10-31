//! DVD reproduction and two-state synthetic-difficulty analysis.
//!
//! Faithful paper score: average the K least probable response-token log
//! likelihoods per generation, then compute population variance across N
//! generations (denominator N).
//!
//! Extension: fit one- and two-Gaussian models and report BIC delta, component
//! separation, support and posterior entropy. These diagnostics are uncalibrated
//! scores, not contamination probabilities.
//!
//! A model adapter must provide actual per-token conditional log probabilities.
//! Do not replace missing log probabilities with text-only heuristics.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DVD_INPUT_SCHEMA: &str = "symthaea.dvd_generation_input.v1";
pub const DVD_REPORT_SCHEMA: &str = "symthaea.dvd_item_report.v1";
pub const DVD_PAPER_GENERATIONS: u32 = 50;
pub const DVD_PAPER_TAIL_TOKENS: u32 = 20;
pub const DVD_GMM_MAX_ITERATIONS: usize = 250;
pub const DVD_GMM_MIN_COMPONENT_WEIGHT: f64 = 0.05;
const GMM_VARIANCE_FLOOR: f64 = 1.0e-12;
const GMM_CONVERGENCE_TOLERANCE: f64 = 1.0e-10;
const GAUSSIAN_LOG_2PI: f64 = 1.837_877_066_409_345_3;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DvdGenerationV1 {
    pub generation_id: String,
    pub generation_seed_commitment_sha256: String,
    pub generated_token_count: u64,
    /// Per-token conditional log probabilities for the generated response.
    pub response_token_log_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DvdGenerationInputV1 {
    pub schema: String,
    pub item_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub harness_revision_sha256: String,
    pub prompt_sha256: String,
    pub decoding_config_sha256: String,
    pub environment_sha256: String,
    pub temperature_milli: u16,
    pub requested_generation_count: u32,
    pub tail_token_count: u32,
    pub generation_seed_commitment_sha256: String,
    pub generations: Vec<DvdGenerationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianComponentV1 {
    pub weight: f64,
    pub mean: f64,
    pub variance: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianMixtureDiagnosticV1 {
    pub converged: bool,
    pub iterations: u64,
    pub log_likelihood: f64,
    pub bic: f64,
    pub components: Vec<GaussianComponentV1>,
    pub standardized_mean_separation: f64,
    pub mean_posterior_entropy_nats: f64,
    pub minimum_component_weight_met: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DvdItemReportV1 {
    pub schema: String,
    pub item_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub harness_revision_sha256: String,
    pub prompt_sha256: String,
    pub decoding_config_sha256: String,
    pub environment_sha256: String,
    pub temperature_milli: u16,
    pub generation_count: u64,
    pub tail_token_count: u64,
    pub sampled_output_token_count: u64,
    pub generation_seed_commitment_sha256: String,
    pub input_sha256: String,
    pub mean_synthetic_difficulty: f64,
    /// Exact paper DVD statistic: population variance, denominator N.
    pub dvd_population_variance: f64,
    pub dvd_sample_variance: f64,
    pub median_absolute_deviation: f64,
    pub winsorized_variance_10_90: f64,
    pub one_component_log_likelihood: f64,
    pub one_component_bic: f64,
    pub two_component: GaussianMixtureDiagnosticV1,
    /// Positive values prefer the two-component model: BIC_1 - BIC_2.
    pub delta_bic_two_vs_one: f64,
    pub score_semantics: String,
    pub calibration_status: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone)]
struct GaussianFit {
    log_likelihood: f64,
    bic: f64,
    converged: bool,
    iterations: usize,
    components: Vec<GaussianComponentV1>,
    mean_posterior_entropy_nats: f64,
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot calculate mean of empty values".to_owned());
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() {
        return Err("mean sum overflowed or became non-finite".to_owned());
    }
    let result = total / values.len() as f64;
    if !result.is_finite() {
        return Err("mean is non-finite".to_owned());
    }
    Ok(result)
}

fn population_variance(values: &[f64], center: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot calculate variance of empty values".to_owned());
    }
    let sum = values.iter().try_fold(0.0_f64, |total, value| {
        let difference = *value - center;
        let next = total + difference * difference;
        if next.is_finite() {
            Ok(next)
        } else {
            Err("variance accumulation overflowed".to_owned())
        }
    })?;
    let result = sum / values.len() as f64;
    if !result.is_finite() {
        return Err("population variance is non-finite".to_owned());
    }
    Ok(result)
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile_sorted(&sorted, 0.5)
}

fn winsorized_variance(values: &[f64]) -> Result<f64, String> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let lower = quantile_sorted(&sorted, 0.10);
    let upper = quantile_sorted(&sorted, 0.90);
    let winsorized: Vec<f64> = values
        .iter()
        .map(|value| value.clamp(lower, upper))
        .collect();
    let center = mean(&winsorized)?;
    population_variance(&winsorized, center)
}

fn synthetic_difficulty(log_probabilities: &[f64], tail_token_count: usize) -> Result<f64, String> {
    if log_probabilities.len() < tail_token_count {
        return Err(format!(
            "response has {} token log-probabilities; at least {tail_token_count} required",
            log_probabilities.len()
        ));
    }
    if log_probabilities
        .iter()
        .any(|value| !value.is_finite() || *value > 1.0e-12 || *value < -1_000.0)
    {
        return Err("token log-probabilities must be finite and in [-1000, 0]".to_owned());
    }
    let mut ordered = log_probabilities.to_vec();
    ordered.sort_by(f64::total_cmp);
    mean(&ordered[..tail_token_count])
}

fn gaussian_log_density(value: f64, center: f64, variance: f64) -> f64 {
    -0.5 * (GAUSSIAN_LOG_2PI + variance.ln() + (value - center).powi(2) / variance)
}

fn log_sum_exp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn one_component_fit(values: &[f64]) -> Result<GaussianFit, String> {
    let center = mean(values)?;
    let variance = population_variance(values, center)?.max(GMM_VARIANCE_FLOOR);
    let log_likelihood: f64 = values
        .iter()
        .map(|value| gaussian_log_density(*value, center, variance))
        .sum();
    if !log_likelihood.is_finite() {
        return Err("one-component Gaussian log-likelihood is non-finite".to_owned());
    }
    // One Gaussian has two estimated parameters: mean and variance.
    let bic = -2.0 * log_likelihood + 2.0 * (values.len() as f64).ln();
    Ok(GaussianFit {
        log_likelihood,
        bic,
        converged: true,
        iterations: 1,
        components: vec![GaussianComponentV1 {
            weight: 1.0,
            mean: center,
            variance,
        }],
        mean_posterior_entropy_nats: 0.0,
    })
}

fn fit_two_component_from_initialization(
    values: &[f64],
    mut first_mean: f64,
    mut second_mean: f64,
    variance_floor: f64,
) -> Result<GaussianFit, String> {
    let n = values.len();
    let initial_variance = population_variance(values, mean(values)?)?
        .mul_add(0.5, 0.0)
        .max(variance_floor);
    let mut first_variance = initial_variance;
    let mut second_variance = initial_variance;
    let mut first_weight = 0.5_f64;
    let mut responsibilities = vec![0.5_f64; n];
    let mut previous_log_likelihood = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0_usize;

    for iteration in 0..DVD_GMM_MAX_ITERATIONS {
        iterations = iteration + 1;
        let mut log_likelihood = 0.0_f64;
        for (index, value) in values.iter().copied().enumerate() {
            let first_log =
                first_weight.ln() + gaussian_log_density(value, first_mean, first_variance);
            let second_log = (1.0 - first_weight).ln()
                + gaussian_log_density(value, second_mean, second_variance);
            let normalizer = log_sum_exp(first_log, second_log);
            if !normalizer.is_finite() {
                return Err("two-component EM produced non-finite log normalizer".to_owned());
            }
            responsibilities[index] = (first_log - normalizer).exp();
            log_likelihood += normalizer;
        }
        if !log_likelihood.is_finite() {
            return Err("two-component EM produced non-finite likelihood".to_owned());
        }

        let first_mass: f64 = responsibilities.iter().sum();
        let second_mass = n as f64 - first_mass;
        if first_mass <= 0.0 || second_mass <= 0.0 {
            return Err("two-component EM collapsed a component".to_owned());
        }
        let new_first_weight = (first_mass / n as f64)
            .clamp(DVD_GMM_MIN_COMPONENT_WEIGHT, 1.0 - DVD_GMM_MIN_COMPONENT_WEIGHT);
        let new_first_mean = values
            .iter()
            .zip(&responsibilities)
            .map(|(value, responsibility)| value * responsibility)
            .sum::<f64>()
            / first_mass;
        let new_second_mean = values
            .iter()
            .zip(&responsibilities)
            .map(|(value, responsibility)| value * (1.0 - responsibility))
            .sum::<f64>()
            / second_mass;
        let new_first_variance = values
            .iter()
            .zip(&responsibilities)
            .map(|(value, responsibility)| responsibility * (value - new_first_mean).powi(2))
            .sum::<f64>()
            / first_mass;
        let new_second_variance = values
            .iter()
            .zip(&responsibilities)
            .map(|(value, responsibility)| (1.0 - responsibility) * (value - new_second_mean).powi(2))
            .sum::<f64>()
            / second_mass;

        first_mean = new_first_mean;
        second_mean = new_second_mean;
        first_variance = new_first_variance.max(variance_floor);
        second_variance = new_second_variance.max(variance_floor);
        first_weight = new_first_weight;

        let improvement = (log_likelihood - previous_log_likelihood).abs();
        if previous_log_likelihood.is_finite()
            && improvement <= GMM_CONVERGENCE_TOLERANCE * (1.0 + log_likelihood.abs())
        {
            converged = true;
            break;
        }
        previous_log_likelihood = log_likelihood;
    }

    let mut final_log_likelihood = 0.0_f64;
    let mut entropy_total = 0.0_f64;
    for value in values {
        let first_log =
            first_weight.ln() + gaussian_log_density(*value, first_mean, first_variance);
        let second_log = (1.0 - first_weight).ln()
            + gaussian_log_density(*value, second_mean, second_variance);
        let normalizer = log_sum_exp(first_log, second_log);
        final_log_likelihood += normalizer;
        let posterior = (first_log - normalizer).exp().clamp(0.0, 1.0);
        let complement = 1.0 - posterior;
        if posterior > 0.0 {
            entropy_total -= posterior * posterior.ln();
        }
        if complement > 0.0 {
            entropy_total -= complement * complement.ln();
        }
    }
    if !final_log_likelihood.is_finite() || !entropy_total.is_finite() {
        return Err("two-component final diagnostics are non-finite".to_owned());
    }

    // Two Gaussians have five free parameters: two means, two variances and one weight.
    let bic = -2.0 * final_log_likelihood + 5.0 * (n as f64).ln();
    let mut components = vec![
        GaussianComponentV1 {
            weight: first_weight,
            mean: first_mean,
            variance: first_variance,
        },
        GaussianComponentV1 {
            weight: 1.0 - first_weight,
            mean: second_mean,
            variance: second_variance,
        },
    ];
    components.sort_by(|left, right| left.mean.total_cmp(&right.mean));
    Ok(GaussianFit {
        log_likelihood: final_log_likelihood,
        bic,
        converged,
        iterations,
        components,
        mean_posterior_entropy_nats: entropy_total / n as f64,
    })
}

fn two_component_fit(values: &[f64]) -> Result<GaussianFit, String> {
    if values.len() < 10 {
        return Err("two-component diagnostic requires at least 10 generations".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let global_mean = mean(values)?;
    let global_variance = population_variance(values, global_mean)?;
    let variance_floor = (global_variance * 1.0e-8).max(GMM_VARIANCE_FLOOR);
    if global_variance <= f64::EPSILON {
        let component = GaussianComponentV1 {
            weight: 0.5,
            mean: global_mean,
            variance: GMM_VARIANCE_FLOOR,
        };
        let log_likelihood = values
            .iter()
            .map(|value| gaussian_log_density(*value, global_mean, GMM_VARIANCE_FLOOR))
            .sum::<f64>();
        let bic = -2.0 * log_likelihood + 5.0 * (values.len() as f64).ln();
        return Ok(GaussianFit {
            log_likelihood,
            bic,
            converged: true,
            iterations: 0,
            components: vec![component.clone(), component],
            mean_posterior_entropy_nats: std::f64::consts::LN_2,
        });
    }
    let initializations = [(0.10, 0.90), (0.20, 0.80), (0.25, 0.75), (0.35, 0.65)];
    let mut best: Option<GaussianFit> = None;
    for (left_quantile, right_quantile) in initializations {
        let left = quantile_sorted(&sorted, left_quantile);
        let right = quantile_sorted(&sorted, right_quantile);
        if (right - left).abs() <= f64::EPSILON {
            continue;
        }
        let candidate = fit_two_component_from_initialization(
            values,
            left,
            right,
            variance_floor,
        )?;
        let replace = best
            .as_ref()
            .map_or(true, |current| candidate.log_likelihood > current.log_likelihood);
        if replace {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| "two-component mixture could not be initialized".to_owned())
}

fn standardized_separation(components: &[GaussianComponentV1]) -> Result<f64, String> {
    if components.len() != 2 {
        return Err("two-component separation requires exactly two components".to_owned());
    }
    let pooled_variance =
        ((components[0].variance + components[1].variance) / 2.0).max(GMM_VARIANCE_FLOOR);
    let separation = (components[0].mean - components[1].mean).abs() / pooled_variance.sqrt();
    if !separation.is_finite() {
        return Err("standardized state separation is non-finite".to_owned());
    }
    Ok(separation)
}

/// Reproduce DVD's population-variance statistic and add one-vs-two Gaussian
/// state diagnostics. Returned scores are uncalibrated risk-like statistics.
pub fn score_dvd_item(input: &DvdGenerationInputV1) -> Result<DvdItemReportV1, String> {
    if input.schema != DVD_INPUT_SCHEMA {
        return Err(format!("input schema must be {DVD_INPUT_SCHEMA}"));
    }
    for (value, label) in [
        (&input.harness_revision_sha256, "harness_revision_sha256"),
        (&input.prompt_sha256, "prompt_sha256"),
        (&input.decoding_config_sha256, "decoding_config_sha256"),
        (&input.environment_sha256, "environment_sha256"),
        (&input.generation_seed_commitment_sha256, "generation_seed_commitment_sha256"),
    ] {
        validate_digest(value, label)?;
    }
    if input.item_id.trim().is_empty()
        || input.item_id.len() > 256
        || input.benchmark_id.trim().is_empty()
        || input.benchmark_version.trim().is_empty()
        || input.model_provider.trim().is_empty()
        || input.model_name.trim().is_empty()
        || input.model_version.trim().is_empty()
        || input.model_access_mode.trim().is_empty()
    {
        return Err("item, benchmark and model identity fields must be nonempty".to_owned());
    }
    if !(10..=1_000).contains(&input.requested_generation_count) {
        return Err("requested_generation_count must be in [10, 1000]".to_owned());
    }
    if !(1..=1_000).contains(&input.tail_token_count) {
        return Err("tail_token_count must be in [1, 1000]".to_owned());
    }
    if input.temperature_milli > 5_000 {
        return Err("temperature_milli must be in [0, 5000]".to_owned());
    }
    if usize::try_from(input.requested_generation_count).ok() != Some(input.generations.len()) {
        return Err("generation records must exactly match requested_generation_count".to_owned());
    }

    let mut generation_ids = BTreeSet::new();
    let mut synthetic_difficulties = Vec::with_capacity(input.generations.len());
    let mut total_generated_tokens = 0_u64;
    for generation in &input.generations {
        if generation.generation_id.trim().is_empty()
            || generation.generation_id.len() > 256
            || !generation_ids.insert(generation.generation_id.as_str())
        {
            return Err("generation IDs must be unique, nonempty and <=256 bytes".to_owned());
        }
        validate_digest(
            &generation.generation_seed_commitment_sha256,
            "generation.generation_seed_commitment_sha256",
        )?;
        if generation.generation_seed_commitment_sha256
            != input.generation_seed_commitment_sha256
        {
            return Err(format!(
                "generation {} seed commitment does not match input manifest",
                generation.generation_id
            ));
        }
        if generation.generated_token_count < u64::from(input.tail_token_count) {
            return Err(format!(
                "generation {} token count is below tail_token_count",
                generation.generation_id
            ));
        }
        total_generated_tokens = total_generated_tokens
            .checked_add(generation.generated_token_count)
            .ok_or_else(|| "sampled output token count overflows u64".to_owned())?;
        synthetic_difficulties.push(synthetic_difficulty(
            &generation.response_token_log_probabilities,
            input.tail_token_count as usize,
        )?);
    }

    let generation_count = synthetic_difficulties.len() as f64;
    let mean_difficulty = mean(&synthetic_difficulties)?;
    let dvd_variance = population_variance(&synthetic_difficulties, mean_difficulty)?;
    let sample_variance = dvd_variance * generation_count / (generation_count - 1.0);
    let center = median(&synthetic_difficulties);
    let absolute_deviations: Vec<f64> = synthetic_difficulties
        .iter()
        .map(|value| (value - center).abs())
        .collect();
    let mad = median(&absolute_deviations);
    let winsorized = winsorized_variance(&synthetic_difficulties)?;
    let one_component = one_component_fit(&synthetic_difficulties)?;
    let two_component = two_component_fit(&synthetic_difficulties)?;
    let separation = standardized_separation(&two_component.components)?;
    let minimum_component_weight_met = two_component
        .components
        .iter()
        .all(|component| component.weight >= DVD_GMM_MIN_COMPONENT_WEIGHT);
    let delta_bic = one_component.bic - two_component.bic;

    let input_bytes = serde_json::to_vec(input)
        .map_err(|error| format!("cannot serialize DVD input: {error}"))?;

    Ok(DvdItemReportV1 {
        schema: DVD_REPORT_SCHEMA.to_owned(),
        item_id: input.item_id.clone(),
        benchmark_id: input.benchmark_id.clone(),
        benchmark_version: input.benchmark_version.clone(),
        model_provider: input.model_provider.clone(),
        model_name: input.model_name.clone(),
        model_version: input.model_version.clone(),
        model_access_mode: input.model_access_mode.clone(),
        harness_revision_sha256: input.harness_revision_sha256.clone(),
        prompt_sha256: input.prompt_sha256.clone(),
        decoding_config_sha256: input.decoding_config_sha256.clone(),
        environment_sha256: input.environment_sha256.clone(),
        temperature_milli: input.temperature_milli,
        generation_count: u64::try_from(input.generations.len())
            .map_err(|_| "generation count overflows u64")?,
        tail_token_count: u64::from(input.tail_token_count),
        sampled_output_token_count: total_generated_tokens,
        generation_seed_commitment_sha256: input.generation_seed_commitment_sha256.clone(),
        input_sha256: hash_domain(b"dvd-input.v1", &input_bytes),
        mean_synthetic_difficulty: mean_difficulty,
        dvd_population_variance: dvd_variance,
        dvd_sample_variance: sample_variance,
        median_absolute_deviation: mad,
        winsorized_variance_10_90: winsorized,
        one_component_log_likelihood: one_component.log_likelihood,
        one_component_bic: one_component.bic,
        two_component: GaussianMixtureDiagnosticV1 {
            converged: two_component.converged,
            iterations: u64::try_from(two_component.iterations)
                .map_err(|_| "EM iteration count overflows")?,
            log_likelihood: two_component.log_likelihood,
            bic: two_component.bic,
            components: two_component.components,
            standardized_mean_separation: separation,
            mean_posterior_entropy_nats: two_component.mean_posterior_entropy_nats,
            minimum_component_weight_met,
        },
        delta_bic_two_vs_one: delta_bic,
        score_semantics: "higher_dvd_population_variance_or_positive_delta_bic_is_more_two_state_like; both_are_uncalibrated_risk_scores".to_owned(),
        calibration_status: "not_calibrated_threshold_not_selected".to_owned(),
        claim_boundary: "model_log_probability_diagnostics_only_not_proof_of_training_exposure_or_training_data_absence".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        hash_domain(b"dvd-test", label.as_bytes())
    }

    fn input_from_difficulties(values: &[f64]) -> DvdGenerationInputV1 {
        let tail_token_count = 4_u32;
        let generations = values
            .iter()
            .enumerate()
            .map(|(index, difficulty)| DvdGenerationV1 {
                generation_id: format!("generation-{index:03}"),
                generation_seed_commitment_sha256: digest("seed"),
                generated_token_count: 12,
                response_token_log_probabilities: vec![
                    *difficulty - 0.3,
                    *difficulty - 0.2,
                    *difficulty - 0.1,
                    *difficulty,
                    -0.01,
                    -0.02,
                ],
            })
            .collect::<Vec<_>>();
        DvdGenerationInputV1 {
            schema: DVD_INPUT_SCHEMA.to_owned(),
            item_id: "item-0001".to_owned(),
            benchmark_id: "dvd-fixture".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_provider: "local".to_owned(),
            model_name: "fixture-model".to_owned(),
            model_version: "1".to_owned(),
            model_access_mode: "logprob".to_owned(),
            harness_revision_sha256: digest("harness"),
            prompt_sha256: digest("prompt"),
            decoding_config_sha256: digest("decode"),
            environment_sha256: digest("environment"),
            temperature_milli: 800,
            requested_generation_count: u32::try_from(values.len()).unwrap(),
            tail_token_count,
            generation_seed_commitment_sha256: digest("seed"),
            generations,
        }
    }

    #[test]
    fn paper_dvd_is_population_variance_over_n_generations() {
        let difficulties: Vec<f64> = (0..50).map(|index| index as f64 / 100.0 - 2.0).collect();
        let input = input_from_difficulties(&difficulties);
        let report = score_dvd_item(&input).unwrap();
        let center = difficulties.iter().sum::<f64>() / 50.0;
        let expected = difficulties
            .iter()
            .map(|value| (value - center).powi(2))
            .sum::<f64>()
            / 50.0;
        assert!((report.dvd_population_variance - expected).abs() < 1.0e-10);
        assert_eq!(report.generation_count, 50);
        assert_eq!(report.tail_token_count, 4);
    }

    #[test]
    fn separated_two_state_fixture_is_preferred_by_bic() {
        let values: Vec<f64> = (0..50)
            .map(|index| if index < 25 { -8.0 + index as f64 * 0.01 } else { -1.0 + (index - 25) as f64 * 0.01 })
            .collect();
        let report = score_dvd_item(&input_from_difficulties(&values)).unwrap();
        assert!(report.delta_bic_two_vs_one > 0.0);
        assert!(report.two_component.standardized_mean_separation > 5.0);
        assert!(report.two_component.minimum_component_weight_met);
    }

    #[test]
    fn constant_difficulty_has_zero_dvd_variance_and_finite_mixture_diagnostics() {
        let report = score_dvd_item(&input_from_difficulties(&vec![-2.0; 50])).unwrap();
        assert_eq!(report.dvd_population_variance, 0.0);
        assert!(report.delta_bic_two_vs_one.is_finite());
        assert!(report.two_component.mean_posterior_entropy_nats.is_finite());
    }

    #[test]
    fn one_extreme_outlier_is_visible_in_raw_and_robust_dispersion() {
        let mut values = vec![-2.0; 49];
        values.push(-20.0);
        let report = score_dvd_item(&input_from_difficulties(&values)).unwrap();
        assert!(report.dvd_population_variance > 0.0);
        assert!(report.winsorized_variance_10_90 < report.dvd_population_variance);
        assert!(report.median_absolute_deviation < 1.0e-12);
    }

    #[test]
    fn generation_order_does_not_change_distributional_scores() {
        let values: Vec<f64> = (0..50)
            .map(|index| if index % 2 == 0 { -7.0 } else { -1.0 })
            .collect();
        let mut input = input_from_difficulties(&values);
        let first = score_dvd_item(&input).unwrap();
        input.generations.reverse();
        let second = score_dvd_item(&input).unwrap();
        assert!((first.dvd_population_variance - second.dvd_population_variance).abs() < 1.0e-12);
        assert!((first.delta_bic_two_vs_one - second.delta_bic_two_vs_one).abs() < 1.0e-8);
    }

    #[test]
    fn too_few_generations_or_wrong_generation_count_fails_closed() {
        let input = input_from_difficulties(&[-2.0; 9]);
        assert!(score_dvd_item(&input).is_err());
        let mut input = input_from_difficulties(&[-2.0; 10]);
        input.requested_generation_count = 11;
        assert!(score_dvd_item(&input).unwrap_err().contains("exactly match"));
    }

    #[test]
    fn response_with_too_few_token_logprobs_is_rejected() {
        let mut input = input_from_difficulties(&[-2.0; 10]);
        input.generations[0].response_token_log_probabilities = vec![-0.1, -0.2];
        assert!(score_dvd_item(&input).unwrap_err().contains("at least 4 required"));
    }

    #[test]
    fn positive_or_nonfinite_log_probabilities_fail_closed() {
        let mut input = input_from_difficulties(&[-2.0; 10]);
        input.generations[0].response_token_log_probabilities[0] = 0.1;
        assert!(score_dvd_item(&input).is_err());
        input.generations[0].response_token_log_probabilities[0] = f64::NAN;
        assert!(score_dvd_item(&input).is_err());
    }

    #[test]
    fn duplicate_generation_ids_and_seed_commitment_mismatch_fail_closed() {
        let mut input = input_from_difficulties(&[-2.0; 10]);
        input.generations[1].generation_id = input.generations[0].generation_id.clone();
        assert!(score_dvd_item(&input).unwrap_err().contains("generation IDs"));
        let mut input = input_from_difficulties(&[-2.0; 10]);
        input.generations[0].generation_seed_commitment_sha256 = digest("other-seed");
        assert!(score_dvd_item(&input).unwrap_err().contains("does not match"));
    }

    #[test]
    fn report_binds_model_prompt_decoding_and_generation_input() {
        let input = input_from_difficulties(&[-3.0; 50]);
        let first = score_dvd_item(&input).unwrap();
        let mut changed = input;
        changed.temperature_milli += 100;
        let second = score_dvd_item(&changed).unwrap();
        assert_ne!(first.input_sha256, second.input_sha256);
    }
}
