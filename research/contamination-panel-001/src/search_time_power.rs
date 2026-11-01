//! Monte Carlo design qualification for randomized search-time access trials.
//!
//! The simulator models the paired three-condition block estimator used by
//! search_time_causal.rs. Run blocks, not benchmark items, are independent.
//! Within-block scores include shared block noise, correlated residual noise,
//! randomized run-order effects, heterogeneous access effects, bounded-score
//! clipping and whole-block MCAR failure.
//!
//! Results qualify the planned estimator only under the declared data-generating
//! assumptions. They do not establish that those assumptions describe a real
//! agent deployment.

use crate::{
    hash_domain,
    search_time_causal::{
        SearchTrialConditionV1, SearchTimeAccessTrialStudyV1,
    },
};
use serde::{Deserialize, Serialize};

pub const SEARCH_POWER_CONFIG_SCHEMA: &str =
    "symthaea.search_time_power_config.v1";
pub const SEARCH_POWER_REPORT_SCHEMA: &str =
    "symthaea.search_time_power_report.v1";
pub const SEARCH_POWER_MAX_CANDIDATES: usize = 20;
pub const SEARCH_POWER_MAX_CANDIDATE_BLOCKS: u32 = 10_000;
pub const SEARCH_POWER_MAX_SIMULATIONS: u32 = 100_000;
pub const SEARCH_POWER_MAX_BLOCK_TRIALS: u64 = 5_000_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchPowerPrimaryContrastV1 {
    AllowlistedMinusNetworkDisabled,
    UnrestrictedMinusNetworkDisabled,
    UnrestrictedMinusAllowlisted,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchPowerMultiplicityV1 {
    SinglePrimary,
    BonferroniThreeContrasts,
}

impl SearchPowerPrimaryContrastV1 {
    fn treated_condition(self) -> SearchTrialConditionV1 {
        match self {
            Self::AllowlistedMinusNetworkDisabled => SearchTrialConditionV1::Allowlisted,
            Self::UnrestrictedMinusNetworkDisabled
            | Self::UnrestrictedMinusAllowlisted => SearchTrialConditionV1::Unrestricted,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimePowerConfigV1 {
    pub schema: String,
    pub design_id: String,
    pub primary_contrast: SearchPowerPrimaryContrastV1,
    pub multiplicity_method: SearchPowerMultiplicityV1,
    pub candidate_block_counts: Vec<u32>,
    pub monte_carlo_repetitions: u32,
    pub baseline_score_permille: f64,
    pub score_floor_permille: u16,
    pub score_ceiling_permille: u16,
    pub smallest_effect_of_interest_permille: f64,
    pub shared_block_noise_sd_permille: f64,
    pub arm_residual_noise_sd_permille: f64,
    pub within_block_correlation_permille: u16,
    pub effect_heterogeneity_sd_permille: f64,
    pub run_order_effect_permille: f64,
    /// Score shift applied to each later run after an earlier network-enabled arm.
    pub carryover_after_network_access_permille: f64,
    /// Whole-block MCAR loss. Any loss invalidates the entire Monte Carlo replicate.
    pub missing_block_rate_permille: u16,
    /// Independent budget-truncation loss applied at the block level.
    pub budget_truncation_rate_permille: u16,
    /// Extra failure probability when any simulated arm falls below the threshold.
    pub outcome_dependent_failure_rate_permille: u16,
    pub failure_score_threshold_permille: u16,
    pub alpha_permille: u16,
    pub maximum_type_i_upper_permille: u16,
    pub minimum_power_lower_permille: u16,
    pub minimum_coverage_lower_permille: u16,
    pub maximum_coverage_upper_permille: u16,
    pub maximum_inconclusive_rate_permille: u16,
    pub maximum_clipped_score_fraction_permille: u16,
    /// Evaluator-held root seed. Keep the config/seed bundle protected.
    pub simulation_seed_hex: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPowerProportionIntervalV1 {
    pub successes: u64,
    pub trials: u64,
    pub estimate_permille: Option<f64>,
    pub lower_95_permille: Option<f64>,
    pub upper_95_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPowerSimulationOutcomeV1 {
    pub planned_repetitions: u32,
    pub estimable_repetitions: u32,
    pub inconclusive_repetitions: u32,
    /// Conservative rejection/power rate with all planned Monte Carlo repetitions
    /// in the denominator; inconclusive replicates count as no rejection.
    pub rejection_rate: SearchPowerProportionIntervalV1,
    /// Conditional rejection rate among fully estimable repetitions.
    pub estimable_rejection_rate: SearchPowerProportionIntervalV1,
    /// Coverage among fully estimable repetitions only.
    pub coverage_rate: SearchPowerProportionIntervalV1,
    pub completion_rate: SearchPowerProportionIntervalV1,
    pub inconclusive_rate: SearchPowerProportionIntervalV1,
    pub mean_estimated_effect_permille: Option<f64>,
    pub mean_bias_permille: Option<f64>,
    pub mean_standard_error_permille: Option<f64>,
    pub mean_completed_blocks: Option<f64>,
    pub clipped_score_fraction_permille: Option<f64>,
    pub total_planned_blocks: u64,
    pub total_missing_blocks: u64,
    pub total_budget_truncated_blocks: u64,
    pub total_outcome_dependent_failures: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPowerCandidateResultV1 {
    pub candidate_block_count: u32,
    pub null_type_i_error: SearchPowerSimulationOutcomeV1,
    pub positive_effect_power: SearchPowerSimulationOutcomeV1,
    pub negative_effect_power: SearchPowerSimulationOutcomeV1,
    pub qualification_passes: bool,
    pub qualification_failures: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimePowerReportV1 {
    pub schema: String,
    pub design_id: String,
    pub primary_contrast: SearchPowerPrimaryContrastV1,
    pub multiplicity_method: SearchPowerMultiplicityV1,
    pub effective_test_alpha_permille: u16,
    pub score_floor_permille: u16,
    pub score_ceiling_permille: u16,
    pub carryover_after_network_access_permille: f64,
    pub configuration_sha256: String,
    pub simulation_seed_commitment_sha256: String,
    pub monte_carlo_repetitions: u32,
    pub smallest_effect_of_interest_permille: f64,
    pub alpha_permille: u16,
    pub minimum_power_lower_permille: u16,
    pub minimum_coverage_lower_permille: u16,
    pub maximum_coverage_upper_permille: u16,
    pub maximum_inconclusive_rate_permille: u16,
    pub maximum_clipped_score_fraction_permille: u16,
    pub maximum_type_i_upper_permille: u16,
    pub candidate_results: Vec<SearchPowerCandidateResultV1>,
    pub minimum_qualified_block_count: Option<u32>,
    pub design_qualified: bool,
    pub report_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: &[u8], candidate: u32, repetition: u32, scenario: u8) -> Self {
        let mut material = Vec::with_capacity(seed.len() + 9);
        material.extend_from_slice(seed);
        material.extend_from_slice(&candidate.to_be_bytes());
        material.extend_from_slice(&repetition.to_be_bytes());
        material.push(scenario);
        let digest = hash_domain(b"search-time-power-prng-seed.v1", &material);
        let state = u64::from_str_radix(&digest[..16], 16).unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self {
            state: if state == 0 { 0x9E37_79B9_7F4A_7C15 } else { state },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open();
        let u2 = self.uniform_open();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn shuffle_three(&mut self, values: &mut [usize; 3]) {
        for index in (1..values.len()).rev() {
            let selected = (self.next_u64() % (index as u64 + 1)) as usize;
            values.swap(index, selected);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimulationSummary {
    planned_repetitions: u32,
    planned_blocks: u64,
    estimable_repetitions: u32,
    inconclusive_repetitions: u32,
    rejections: u64,
    covered: u64,
    estimate_sum: f64,
    bias_sum: f64,
    standard_error_sum: f64,
    completed_blocks_sum: f64,
    clipped_scores: u64,
    total_scores: u64,
    total_missing_blocks: u64,
    total_budget_truncated_blocks: u64,
    total_outcome_dependent_failures: u64,
}

impl SimulationSummary {
    fn new(planned_repetitions: u32, candidate_blocks: u32) -> Self {
        Self {
            planned_repetitions,
            planned_blocks: u64::from(planned_repetitions) * u64::from(candidate_blocks),
            estimable_repetitions: 0,
            inconclusive_repetitions: 0,
            rejections: 0,
            covered: 0,
            estimate_sum: 0.0,
            bias_sum: 0.0,
            standard_error_sum: 0.0,
            completed_blocks_sum: 0.0,
            clipped_scores: 0,
            total_scores: 0,
            total_missing_blocks: 0,
            total_budget_truncated_blocks: 0,
            total_outcome_dependent_failures: 0,
        }
    }

    fn finish(self) -> SearchPowerSimulationOutcomeV1 {
        let planned = u64::from(self.planned_repetitions);
        let estimable = u64::from(self.estimable_repetitions);
        SearchPowerSimulationOutcomeV1 {
            planned_repetitions: self.planned_repetitions,
            estimable_repetitions: self.estimable_repetitions,
            inconclusive_repetitions: self.inconclusive_repetitions,
            rejection_rate: wilson_interval(self.rejections, planned),
            estimable_rejection_rate: wilson_interval(self.rejections, estimable),
            coverage_rate: wilson_interval(self.covered, estimable),
            completion_rate: wilson_interval(estimable, planned),
            inconclusive_rate: wilson_interval(
                u64::from(self.inconclusive_repetitions),
                planned,
            ),
            mean_estimated_effect_permille: (estimable > 0)
                .then_some(self.estimate_sum / estimable as f64),
            mean_bias_permille: (estimable > 0)
                .then_some(self.bias_sum / estimable as f64),
            mean_standard_error_permille: (estimable > 0)
                .then_some(self.standard_error_sum / estimable as f64),
            mean_completed_blocks: (planned > 0)
                .then_some(self.completed_blocks_sum / planned as f64),
            clipped_score_fraction_permille: (self.total_scores > 0)
                .then_some(self.clipped_scores as f64 * 1_000.0 / self.total_scores as f64),
            total_planned_blocks: self.planned_blocks,
            total_missing_blocks: self.total_missing_blocks,
            total_budget_truncated_blocks: self.total_budget_truncated_blocks,
            total_outcome_dependent_failures: self.total_outcome_dependent_failures,
        }
    }
}

fn validate_config(config: &SearchTimePowerConfigV1) -> Result<Vec<u8>, String> {
    if config.schema != SEARCH_POWER_CONFIG_SCHEMA {
        return Err(format!("power config schema must be {SEARCH_POWER_CONFIG_SCHEMA}"));
    }
    if config.design_id.trim().is_empty() || config.design_id.len() > 128 {
        return Err("power design_id must be nonempty and <=128 bytes".to_owned());
    }
    if config.candidate_block_counts.is_empty()
        || config.candidate_block_counts.len() > SEARCH_POWER_MAX_CANDIDATES
        || config.candidate_block_counts.iter().any(|count| *count < 2 || *count > SEARCH_POWER_MAX_CANDIDATE_BLOCKS)
    {
        return Err("candidate block counts must contain 1..=20 values in [2, 10000]".to_owned());
    }
    if config.candidate_block_counts.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("candidate block counts must be strictly increasing".to_owned());
    }
    if !(1_000..=SEARCH_POWER_MAX_SIMULATIONS).contains(&config.monte_carlo_repetitions) {
        return Err(format!("monte_carlo_repetitions must be in [1000, {SEARCH_POWER_MAX_SIMULATIONS}]"));
    }
    let total_block_trials = config.candidate_block_counts.iter()
        .try_fold(0_u64, |sum, count| sum.checked_add(u64::from(*count)))
        .and_then(|sum| sum.checked_mul(u64::from(config.monte_carlo_repetitions)))
        .and_then(|sum| sum.checked_mul(3))
        .ok_or_else(|| "total simulation block trials overflow u64".to_owned())?;
    if total_block_trials > SEARCH_POWER_MAX_BLOCK_TRIALS {
        return Err(format!(
            "candidate grid exceeds {SEARCH_POWER_MAX_BLOCK_TRIALS} simulation block-trials"
        ));
    }
    if !config.baseline_score_permille.is_finite()
        || !(0.0..=1_000.0).contains(&config.baseline_score_permille)
        || !config.smallest_effect_of_interest_permille.is_finite()
        || !(0.1..=1_000.0).contains(&config.smallest_effect_of_interest_permille)
    {
        return Err("baseline and smallest effect of interest must be finite and in bounds".to_owned());
    }
    for (value, label) in [
        (config.shared_block_noise_sd_permille, "shared_block_noise_sd_permille"),
        (config.arm_residual_noise_sd_permille, "arm_residual_noise_sd_permille"),
        (config.effect_heterogeneity_sd_permille, "effect_heterogeneity_sd_permille"),
        (config.run_order_effect_permille, "run_order_effect_permille"),
    ] {
        if !value.is_finite() || !(0.0..=1_000.0).contains(&value) {
            return Err(format!("{label} must be finite and in [0, 1000]"));
        }
    }
    if config.within_block_correlation_permille > 999
        || config.missing_block_rate_permille > 999
        || config.alpha_permille != 50
        || config.maximum_type_i_upper_permille < config.alpha_permille
        || config.maximum_type_i_upper_permille > 500
        || config.minimum_power_lower_permille < 500
        || config.minimum_power_lower_permille > 990
        || config.minimum_coverage_lower_permille > 990
        || config.maximum_coverage_upper_permille < 950
        || config.maximum_coverage_upper_permille > 1_000
        || config.minimum_coverage_lower_permille >= config.maximum_coverage_upper_permille
        || config.maximum_inconclusive_rate_permille > 500
        || config.maximum_clipped_score_fraction_permille > 500
    {
        return Err("V1 requires a two-sided 5% test and bounded power/coverage/error thresholds".to_owned());
    }
    let seed = decode_seed_hex(&config.simulation_seed_hex)?;
    Ok(seed)
}

fn decode_seed_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() < 64 || value.len() % 2 != 0 {
        return Err("simulation seed must contain at least 32 bytes of lowercase hex".to_owned());
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        let high = digit(pair[0]).ok_or_else(|| "simulation seed must be lowercase hex".to_owned())?;
        let low = digit(pair[1]).ok_or_else(|| "simulation seed must be lowercase hex".to_owned())?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn normalized_proportion(successes: u64, trials: u64) -> Option<f64> {
    (trials > 0).then_some(successes as f64 / trials as f64)
}

fn wilson_interval(successes: u64, trials: u64) -> SearchPowerProportionIntervalV1 {
    let Some(p) = normalized_proportion(successes, trials) else {
        return SearchPowerProportionIntervalV1 {
            successes,
            trials,
            estimate_permille: None,
            lower_95_permille: None,
            upper_95_permille: None,
        };
    };
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let n = trials as f64;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    SearchPowerProportionIntervalV1 {
        successes,
        trials,
        estimate_permille: Some(p * 1_000.0),
        lower_95_permille: Some((center - half).max(0.0) * 1_000.0),
        upper_95_permille: Some((center + half).min(1.0) * 1_000.0),
    }
}

fn t_critical_975(df: usize) -> f64 {
    const TABLE: &[(usize, f64)] = &[
        (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
        (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
        (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145), (15, 2.131),
        (16, 2.120), (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086),
        (21, 2.080), (22, 2.074), (23, 2.069), (24, 2.064), (25, 2.060),
        (26, 2.056), (27, 2.052), (28, 2.048), (29, 2.045), (30, 2.042),
        (40, 2.021), (60, 2.000), (120, 1.980), (usize::MAX, 1.980),
    ];
    TABLE.iter()
        .find(|(upper_df, _)| df <= *upper_df)
        .map(|(_, value)| *value)
        .unwrap_or(1.98)
}

fn sample_mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot estimate an empty paired-block sample".to_owned());
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("sample mean is non-finite".to_owned())
    }
}

fn sample_sd(values: &[f64], center: f64) -> Result<f64, String> {
    if values.len() < 2 {
        return Err("at least two completed blocks are required".to_owned());
    }
    let ss = values.iter().map(|value| (value - center).powi(2)).sum::<f64>();
    let sd = (ss / (values.len() - 1) as f64).sqrt();
    if sd.is_finite() {
        Ok(sd)
    } else {
        Err("sample standard deviation is non-finite".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct SimulatedBlock {
    difference_permille: f64,
    clipped_scores: u64,
}

fn simulate_block(
    config: &SearchTimePowerConfigV1,
    effect_permille: f64,
    rng: &mut DeterministicRng,
) -> SimulatedBlock {
    // 0 = network-disabled, 1 = allowlisted, 2 = unrestricted.
    let treated_index = match config.primary_contrast {
        SearchPowerPrimaryContrastV1::AllowlistedMinusNetworkDisabled => 1,
        SearchPowerPrimaryContrastV1::UnrestrictedMinusNetworkDisabled => 2,
    };
    let mut order = [0_usize, 1, 2];
    rng.shuffle_three(&mut order);
    let mut positions = [0_usize; 3];
    for (position, condition) in order.iter().copied().enumerate() {
        positions[condition] = position;
    }

    let shared_block_noise =
        config.shared_block_noise_sd_permille * rng.normal();
    let shared_arm_noise = rng.normal();
    let rho = f64::from(config.within_block_correlation_permille) / 1_000.0;
    let shared_weight = rho.sqrt();
    let independent_weight = (1.0 - rho).sqrt();

    let mut scores = [0.0_f64; 3];
    let mut clipped_scores = 0_u64;
    for condition in 0..3 {
        let residual_noise = config.arm_residual_noise_sd_permille
            * (shared_weight * shared_arm_noise + independent_weight * rng.normal());
        let heterogeneity = if condition == treated_index {
            config.effect_heterogeneity_sd_permille * rng.normal()
        } else {
            0.0
        };
        let treatment_effect = if condition == treated_index {
            effect_permille
        } else {
            0.0
        };
        let position_offset = positions[condition] as f64 - 1.0;
        let raw_score = config.baseline_score_permille
            + shared_block_noise
            + residual_noise
            + heterogeneity
            + treatment_effect
            + config.run_order_effect_permille * position_offset;
        if !(0.0..=1_000.0).contains(&raw_score) {
            clipped_scores += 1;
        }
        scores[condition] = raw_score.clamp(0.0, 1_000.0);
    }
    SimulatedBlock {
        difference_permille: scores[treated_index] - scores[0],
        clipped_scores,
    }
}

fn simulate_repetitions(
    config: &SearchTimePowerConfigV1,
    seed: &[u8],
    candidate_blocks: u32,
    true_effect_permille: f64,
    scenario_id: u8,
) -> Result<SimulationSummary, String> {
    let mut summary = SimulationSummary::new(config.monte_carlo_repetitions);
    for repetition in 0..config.monte_carlo_repetitions {
        let mut rng = DeterministicRng::new(seed, candidate_blocks, repetition, scenario_id);
        let mut differences = Vec::with_capacity(candidate_blocks as usize);
        for _ in 0..candidate_blocks {
            let missing_draw = rng.uniform_open() * 1_000.0;
            if missing_draw < f64::from(config.missing_block_rate_permille) {
                continue;
            }
            let block = simulate_block(config, true_effect_permille, &mut rng);
            differences.push(block.difference_permille);
            summary.clipped_scores = summary.clipped_scores.saturating_add(block.clipped_scores);
            summary.total_scores = summary.total_scores.saturating_add(3);
        }
        let completed_blocks = differences.len();
        if completed_blocks < 2 {
            summary.inconclusive_repetitions =
                summary.inconclusive_repetitions.saturating_add(1);
            continue;
        }
        let estimate = sample_mean(&differences)?;
        let sd = sample_sd(&differences, estimate)?;
        let standard_error = sd / (completed_blocks as f64).sqrt();
        if !standard_error.is_finite() {
            summary.inconclusive_repetitions =
                summary.inconclusive_repetitions.saturating_add(1);
            continue;
        }
        let critical = t_critical_975(completed_blocks - 1);
        let margin = critical * standard_error;
        let lower = estimate - margin;
        let upper = estimate + margin;
        let rejects_null = lower > 0.0 || upper < 0.0;
        let covers_truth = lower <= true_effect_permille && true_effect_permille <= upper;
        summary.estimable_repetitions = summary.estimable_repetitions.saturating_add(1);
        if rejects_null {
            summary.rejections = summary.rejections.saturating_add(1);
        }
        if covers_truth {
            summary.covered = summary.covered.saturating_add(1);
        }
        summary.estimate_sum += estimate;
        summary.bias_sum += estimate - true_effect_permille;
        summary.standard_error_sum += standard_error;
        summary.completed_blocks_sum += completed_blocks as f64;
    }
    Ok(summary)
}

fn qualification_failures(
    config: &SearchTimePowerConfigV1,
    null: &SimulationSummary,
    positive: &SimulationSummary,
    negative: &SimulationSummary,
) -> Vec<String> {
    let null_outcome = null.finish();
    let positive_outcome = positive.finish();
    let negative_outcome = negative.finish();
    let mut failures = Vec::new();
    match null_outcome.rejection_rate.upper_95_permille {
        Some(value) if value <= f64::from(config.maximum_type_i_upper_permille) => {}
        Some(value) => failures.push(format!(
            "null type-I Wilson upper bound {:.1}‰ exceeds maximum {}‰",
            value, config.maximum_type_i_upper_permille
        )),
        None => failures.push("null type-I error is not estimable".to_owned()),
    }
    for (name, result) in [("positive", &positive_outcome), ("negative", &negative_outcome)] {
        match result.rejection_rate.lower_95_permille {
            Some(value) if value >= f64::from(config.minimum_power_lower_permille) => {}
            Some(value) => failures.push(format!(
                "{name} power Wilson lower bound {:.1}‰ is below target {}‰",
                value, config.minimum_power_lower_permille
            )),
            None => failures.push(format!("{name} power is not estimable")),
        }
        match (
            result.coverage_rate.lower_95_permille,
            result.coverage_rate.upper_95_permille,
        ) {
            (Some(lower), Some(upper))
                if lower >= f64::from(config.minimum_coverage_lower_permille)
                    && upper <= f64::from(config.maximum_coverage_upper_permille) => {}
            (Some(lower), Some(upper)) => failures.push(format!(
                "{name} coverage Wilson interval [{lower:.1}, {upper:.1}]‰ is outside [{}, {}]‰",
                config.minimum_coverage_lower_permille,
                config.maximum_coverage_upper_permille
            )),
            _ => failures.push(format!("{name} interval coverage is not estimable")),
        }
        let inconclusive_rate = if result.planned_repetitions == 0 {
            1_000.0
        } else {
            result.inconclusive_repetitions as f64 * 1_000.0
                / result.planned_repetitions as f64
        };
        if inconclusive_rate > f64::from(config.maximum_inconclusive_rate_permille) {
            failures.push(format!(
                "{name} inconclusive simulation rate {:.1}‰ exceeds maximum {}‰",
                inconclusive_rate, config.maximum_inconclusive_rate_permille
            ));
        }
        if let Some(clipped) = result.clipped_score_fraction_permille {
            if clipped > f64::from(config.maximum_clipped_score_fraction_permille) {
                failures.push(format!(
                    "{name} clipped-score fraction {:.1}‰ exceeds maximum {}‰",
                    clipped, config.maximum_clipped_score_fraction_permille
                ));
            }
        } else {
            failures.push(format!("{name} has no completed score observations"));
        }
    }
    failures
}

/// Run null, positive-effect and negative-effect Monte Carlo scenarios for
/// each candidate block count. The primary contrast is selected before
/// simulation and receives a two-sided t-test at the configured alpha.
pub fn configuration_digest(config: &SearchTimePowerConfigV1) -> Result<String, String> {
    let mut normalized = config.clone();
    normalized.simulation_seed_hex.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize normalized power config: {error}"))?;
    Ok(hash_domain(b"search-time-power-config.v1", &bytes))
}

fn qualify_search_time_trial_design(
    config: &SearchTimePowerConfigV1,
) -> Result<SearchTimePowerReportV1, String> {
    let seed = validate_config(config)?;
    let configuration_sha256 = configuration_digest(config)?;
    let simulation_seed_commitment_sha256 =
        hash_domain(b"search-time-power-seed.v1", &seed);

    let mut candidate_results = Vec::with_capacity(config.candidate_block_counts.len());
    let mut minimum_qualified_block_count = None;
    for candidate in &config.candidate_block_counts {
        let null = simulate_repetitions(config, &seed, *candidate, 0.0, 0)?;
        let positive = simulate_repetitions(
            config,
            &seed,
            *candidate,
            config.smallest_effect_of_interest_permille,
            1,
        )?;
        let negative = simulate_repetitions(
            config,
            &seed,
            *candidate,
            -config.smallest_effect_of_interest_permille,
            2,
        )?;
        let failures = qualification_failures(config, &null, &positive, &negative);
        let passes = failures.is_empty();
        if passes && minimum_qualified_block_count.is_none() {
            minimum_qualified_block_count = Some(*candidate);
        }
        candidate_results.push(SearchPowerCandidateResultV1 {
            candidate_block_count: *candidate,
            null_type_i_error: null.finish(),
            positive_effect_power: positive.finish(),
            negative_effect_power: negative.finish(),
            qualification_passes: passes,
            qualification_failures: failures,
        });
    }
    let design_qualified = minimum_qualified_block_count.is_some();
    let mut report = SearchTimePowerReportV1 {
        schema: SEARCH_POWER_REPORT_SCHEMA.to_owned(),
        design_id: config.design_id.clone(),
        primary_contrast: config.primary_contrast,
        configuration_sha256,
        simulation_seed_commitment_sha256,
        monte_carlo_repetitions: config.monte_carlo_repetitions,
        smallest_effect_of_interest_permille: config.smallest_effect_of_interest_permille,
        alpha_permille: config.alpha_permille,
        minimum_power_lower_permille: config.minimum_power_lower_permille,
        minimum_coverage_lower_permille: config.minimum_coverage_lower_permille,
        maximum_coverage_upper_permille: config.maximum_coverage_upper_permille,
        maximum_inconclusive_rate_permille: config.maximum_inconclusive_rate_permille,
        maximum_clipped_score_fraction_permille: config.maximum_clipped_score_fraction_permille,
        maximum_type_i_upper_permille: config.maximum_type_i_upper_permille,
        candidate_results,
        minimum_qualified_block_count,
        design_qualified,
        report_sha256: String::new(),
        claim_boundary: "simulated_design_qualification_under_declared_data_generating_assumptions_only;not_evidence_of_real_world_power_or_confirmatory_access_effect".to_owned(),
    };
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("cannot serialize search-time power report: {error}"))?;
    report.report_sha256 = hash_domain(b"search-time-power-report.v1", &report_bytes);
    Ok(report)
}

/// Verify exact report integrity and recompute whether any candidate passes.
pub fn verify_search_time_power_report(
    config: &SearchTimePowerConfigV1,
    report: &SearchTimePowerReportV1,
) -> Result<(), String> {
    // Re-run the exact bounded simulation from the evaluator config/seed.
    // This is stronger than trusting a rehashed self-reported result table.
    let expected = qualify_search_time_trial_design(config)?;
    let expected_bytes = serde_json::to_vec(&expected)
        .map_err(|error| format!("cannot serialize regenerated power report: {error}"))?;
    let supplied_bytes = serde_json::to_vec(report)
        .map_err(|error| format!("cannot serialize supplied power report: {error}"))?;
    if expected_bytes != supplied_bytes {
        return Err("power report does not match deterministic regeneration from the supplied config and seed".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SearchTimePowerConfigV1 {
        SearchTimePowerConfigV1 {
            schema: SEARCH_POWER_CONFIG_SCHEMA.to_owned(),
            design_id: "search-time-block-power-pilot".to_owned(),
            primary_contrast: SearchPowerPrimaryContrastV1::AllowlistedMinusNetworkDisabled,
            candidate_block_counts: vec![20, 40, 60],
            monte_carlo_repetitions: 2_000,
            baseline_score_permille: 700.0,
            smallest_effect_of_interest_permille: 30.0,
            shared_block_noise_sd_permille: 20.0,
            arm_residual_noise_sd_permille: 45.0,
            within_block_correlation_permille: 500,
            effect_heterogeneity_sd_permille: 10.0,
            run_order_effect_permille: 5.0,
            missing_block_rate_permille: 20,
            alpha_permille: 50,
            maximum_type_i_upper_permille: 60,
            minimum_power_lower_permille: 800,
            minimum_coverage_lower_permille: 930,
            maximum_coverage_upper_permille: 970,
            maximum_inconclusive_rate_permille: 1,
            maximum_clipped_score_fraction_permille: 50,
            simulation_seed_hex: "ab".repeat(32),
        }
    }

    #[test]
    fn deterministic_power_reports_repeat_exactly() {
        let config = config();
        let first = qualify_search_time_trial_design(&config).unwrap();
        let second = qualify_search_time_trial_design(&config).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        verify_search_time_power_report(&config, &first).unwrap();
    }

    #[test]
    fn sample_size_grid_is_sorted_and_bounded() {
        let mut config = config();
        config.candidate_block_counts = vec![40, 20];
        assert!(qualify_search_time_trial_design(&config)
            .unwrap_err().contains("strictly increasing"));
        let mut config = config();
        config.candidate_block_counts = vec![10_001];
        assert!(qualify_search_time_trial_design(&config).is_err());
    }

    #[test]
    fn impossible_thresholds_and_short_seeds_fail_closed() {
        let mut config = config();
        config.within_block_correlation_permille = 1_000;
        assert!(qualify_search_time_trial_design(&config).is_err());
        let mut config = config();
        config.simulation_seed_hex = "aabb".to_owned();
        assert!(qualify_search_time_trial_design(&config).is_err());
    }

    #[test]
    fn wilson_interval_is_bounded_and_handles_no_trials() {
        let interval = wilson_interval(950, 1_000);
        assert!(interval.lower_95_permille.unwrap() < 950.0);
        assert!(interval.upper_95_permille.unwrap() > 950.0);
        let empty = wilson_interval(0, 0);
        assert_eq!(empty.estimate_permille, None);
        assert_eq!(empty.lower_95_permille, None);
        assert_eq!(empty.upper_95_permille, None);
    }

    #[test]
    fn bounded_scores_report_floor_and_ceiling_clipping() {
        let mut config = config();
        config.baseline_score_permille = 999.0;
        config.shared_block_noise_sd_permille = 100.0;
        config.arm_residual_noise_sd_permille = 100.0;
        config.smallest_effect_of_interest_permille = 100.0;
        let seed = decode_seed_hex(&config.simulation_seed_hex).unwrap();
        let mut rng = DeterministicRng::new(&seed, 20, 0, 1);
        let block = simulate_block(&config, 100.0, &mut rng);
        assert!(block.clipped_scores > 0);
    }

    #[test]
    fn report_tampering_is_detected_even_if_summary_fields_change() {
        let report = qualify_search_time_trial_design(&config()).unwrap();
        let mut changed = report.clone();
        changed.candidate_results[0].qualification_passes =
            !changed.candidate_results[0].qualification_passes;
        assert!(verify_search_time_power_report(&config(), &changed)
            .unwrap_err().contains("deterministic regeneration"));
        let mut rehashed = changed;
        rehashed.report_sha256.clear();
        let bytes = serde_json::to_vec(&rehashed).unwrap();
        rehashed.report_sha256 = hash_domain(b"search-time-power-report.v1", &bytes);
        assert!(verify_search_time_power_report(&config(), &rehashed)
            .unwrap_err().contains("deterministic regeneration"));
    }
}
