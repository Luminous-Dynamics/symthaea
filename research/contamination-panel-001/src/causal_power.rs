//! Monte Carlo power and coverage planning for replicated model-pair causal studies.
//!
//! The independent unit is the trained treatment/placebo model pair. Each pair
//! contributes one cluster-weighted global effect summary. Families and source
//! clusters contribute correlated nested measurements to that summary; they are
//! never counted as independent training replications.
//!
//! V1 assumes Gaussian random effects, fixed balanced task-family allocations,
//! MCAR model-pair failures and a homogeneous direct/spillover effect. The planner
//! is a design simulation under those assumptions, not proof that actual training
//! variation follows the model.

use crate::{
    causal_analysis::t_critical_975,
    hash_domain,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CAUSAL_POWER_CONFIG_SCHEMA: &str =
    "symthaea.contamination_causal_power_config.v1";
pub const CAUSAL_POWER_REPORT_SCHEMA: &str =
    "symthaea.contamination_causal_power_report.v1";
pub const CAUSAL_POWER_MIN_SIMULATIONS: u32 = 1_000;
pub const CAUSAL_POWER_MAX_SIMULATIONS: u32 = 100_000;
pub const CAUSAL_POWER_MAX_OPERATIONS: u128 = 100_000_000;
pub const CAUSAL_POWER_COVERAGE_TARGET_PERMILLE: u16 = 950;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalPowerPlanningConfigV1 {
    pub schema: String,
    pub study_id: String,
    pub primary_metric_id: String,
    /// Candidate independent training-pair replicate counts, in ascending order.
    pub candidate_model_pair_counts: Vec<u32>,
    pub clusters_per_model_pair: u32,
    pub task_family_count: u32,
    pub exposure_fraction_permille: u16,
    /// Incremental paired outcome effect on exposed source-family clusters.
    pub direct_exposure_effect_permille: f64,
    /// Global model-wide paired advantage on unexposed clusters.
    pub spillover_effect_permille: f64,
    /// SD of the global paired advantage between independently trained model pairs.
    pub between_model_pair_sd_permille: f64,
    /// Marginal SD of family/cluster outcome variation nested within a model pair.
    pub within_cluster_sd_permille: f64,
    /// Share of within-pair nested outcome variance attributable to family shocks.
    pub within_family_icc_permille: u16,
    /// Probability that an entire independently trained pair is unusable (MCAR V1).
    pub pair_failure_probability_permille: u16,
    pub simulation_replicates: u32,
    pub alpha_permille: u16,
    pub target_power_permille: u16,
    /// Tolerance in permille around nominal alpha and 95% interval coverage.
    pub calibration_tolerance_permille: u16,
    pub maximum_inconclusive_fraction_permille: u16,
    /// Evaluator-held seed for reproducible Monte Carlo design simulation.
    pub simulation_seed_hex: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalPowerCandidateEstimateV1 {
    pub model_pair_count: u32,
    pub null_evaluable_simulations: u32,
    pub null_inconclusive_simulations: u32,
    pub null_inconclusive_fraction_permille: f64,
    pub type_i_error_permille: Option<f64>,
    pub type_i_error_monte_carlo_se_permille: Option<f64>,
    pub alternative_evaluable_simulations: u32,
    pub alternative_inconclusive_simulations: u32,
    pub alternative_inconclusive_fraction_permille: f64,
    pub estimated_power_permille: Option<f64>,
    pub power_monte_carlo_se_permille: Option<f64>,
    pub interval_coverage_permille: Option<f64>,
    pub interval_coverage_monte_carlo_se_permille: Option<f64>,
    pub mean_surviving_model_pairs: f64,
    pub estimated_primary_effect_sd_permille: Option<f64>,
    pub estimated_primary_effect_se_permille: Option<f64>,
    pub meets_preregistered_targets: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalPowerPlanningReportV1 {
    pub schema: String,
    pub study_id: String,
    pub primary_metric_id: String,
    pub config_sha256: String,
    pub simulation_seed_commitment_sha256: String,
    pub simulation_replicates: u32,
    pub nominal_alpha_permille: u16,
    pub target_power_permille: u16,
    pub target_interval_coverage_permille: u16,
    pub calibration_tolerance_permille: u16,
    pub requested_exposure_fraction_permille: u16,
    pub actual_balanced_exposure_fraction_permille: f64,
    pub expected_direct_exposure_effect_permille: f64,
    pub expected_global_spillover_effect_permille: f64,
    pub expected_primary_global_effect_permille: f64,
    pub between_model_pair_sd_permille: f64,
    pub within_cluster_sd_permille: f64,
    pub within_family_icc_permille: u16,
    pub pair_failure_probability_permille: u16,
    pub candidate_estimates: Vec<CausalPowerCandidateEstimateV1>,
    pub smallest_qualifying_model_pair_count: Option<u32>,
    pub assumption_summary: String,
    pub design_interpretation: String,
    pub claim_boundary: String,
    pub report_sha256: String,
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.causal-power-splitmix-seed.v1\0");
        hasher.update(seed);
        let digest = hasher.finalize();
        let state = u64::from_be_bytes(digest[0..8].try_into().unwrap_or([0; 8]));
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open01(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform_open01();
        let second = self.uniform_open01();
        (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
    }
}

fn decode_seed(value: &str) -> Result<Vec<u8>, String> {
    if value.len() < 64 || value.len() % 2 != 0 {
        return Err("simulation_seed_hex must encode at least 32 bytes".to_owned());
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

fn canonical_config_digest(config: &CausalPowerPlanningConfigV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(config)
        .map_err(|error| format!("cannot serialize causal power config: {error}"))?;
    Ok(hash_domain(b"causal-power-planning-config.v1", &bytes))
}

fn validate_config(config: &CausalPowerPlanningConfigV1) -> Result<Vec<u8>, String> {
    if config.schema != CAUSAL_POWER_CONFIG_SCHEMA {
        return Err(format!("power config schema must be {CAUSAL_POWER_CONFIG_SCHEMA}"));
    }
    if config.study_id.trim().is_empty()
        || config.study_id.len() > 128
        || config.primary_metric_id.trim().is_empty()
    {
        return Err("study_id and primary_metric_id must be nonempty".to_owned());
    }
    if config.candidate_model_pair_counts.is_empty()
        || config.candidate_model_pair_counts.len() > 32
        || config.candidate_model_pair_counts.iter().any(|count| !(2..=1_000).contains(count))
        || config.candidate_model_pair_counts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("candidate model-pair counts must be 1..=32 strictly increasing values in [2,1000]".to_owned());
    }
    if config.task_family_count == 0
        || config.clusters_per_model_pair < config.task_family_count.saturating_mul(4)
        || config.clusters_per_model_pair % config.task_family_count != 0
    {
        return Err("clusters_per_model_pair must divide evenly into at least four clusters per task family".to_owned());
    }
    if !(1..=999).contains(&config.exposure_fraction_permille) {
        return Err("exposure_fraction_permille must be an interior dose in [1,999]".to_owned());
    }
    for (value, label) in [
        (config.direct_exposure_effect_permille, "direct_exposure_effect_permille"),
        (config.spillover_effect_permille, "spillover_effect_permille"),
    ] {
        if !value.is_finite() || value.abs() > 1_000.0 {
            return Err(format!("{label} must be finite and in [-1000,1000]"));
        }
    }
    for (value, label) in [
        (config.between_model_pair_sd_permille, "between_model_pair_sd_permille"),
        (config.within_cluster_sd_permille, "within_cluster_sd_permille"),
    ] {
        if !value.is_finite() || !(0.0..=1_000.0).contains(&value) {
            return Err(format!("{label} must be finite and in [0,1000]"));
        }
    }
    if config.within_family_icc_permille > 900
        || config.pair_failure_probability_permille > 900
    {
        return Err("ICC and pair failure probability must be in [0,900] permille".to_owned());
    }
    if !(CAUSAL_POWER_MIN_SIMULATIONS..=CAUSAL_POWER_MAX_SIMULATIONS)
        .contains(&config.simulation_replicates)
    {
        return Err(format!(
            "simulation_replicates must be in [{CAUSAL_POWER_MIN_SIMULATIONS},{CAUSAL_POWER_MAX_SIMULATIONS}]"
        ));
    }
    if !(10..=200).contains(&config.alpha_permille)
        || !(500..=990).contains(&config.target_power_permille)
        || !(5..=100).contains(&config.calibration_tolerance_permille)
        || config.maximum_inconclusive_fraction_permille > 500
    {
        return Err("alpha, target power, calibration tolerance or inconclusive threshold is invalid".to_owned());
    }
    let max_pairs = *config.candidate_model_pair_counts.last().unwrap_or(&0) as u128;
    let operations = u128::from(config.simulation_replicates)
        .checked_mul(max_pairs)
        .and_then(|value| value.checked_mul(u128::from(config.clusters_per_model_pair)))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "simulation operation estimate overflowed".to_owned())?;
    if operations > CAUSAL_POWER_MAX_OPERATIONS {
        return Err(format!(
            "simulation exceeds operation budget: {operations} > {CAUSAL_POWER_MAX_OPERATIONS}"
        ));
    }
    decode_seed(&config.simulation_seed_hex)
}

fn sample_mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot average an empty model-pair sample".to_owned());
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("model-pair sample mean is non-finite".to_owned())
    }
}

fn sample_sd(values: &[f64], center: f64) -> Result<f64, String> {
    if values.len() < 2 {
        return Err("at least two surviving model-pair replicates are required".to_owned());
    }
    let variance = values.iter().map(|value| (value - center).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    let sd = variance.sqrt();
    if sd.is_finite() {
        Ok(sd)
    } else {
        Err("model-pair sample SD is non-finite".to_owned())
    }
}

fn simulated_pair_effect(
    config: &CausalPowerPlanningConfigV1,
    rng: &mut SplitMix64,
    direct_effect: f64,
    spillover_effect: f64,
) -> f64 {
    let pair_shift = config.between_model_pair_sd_permille * rng.normal();
    let family_count = config.task_family_count as usize;
    let clusters_per_family =
        (config.clusters_per_model_pair / config.task_family_count) as usize;
    let exposure_count = ((clusters_per_family as u64
        * u64::from(config.exposure_fraction_permille)
        + 500) / 1_000)
        .clamp(1, (clusters_per_family - 1) as u64) as usize;
    let icc = f64::from(config.within_family_icc_permille) / 1_000.0;
    let family_sd = config.within_cluster_sd_permille * icc.sqrt();
    let cluster_sd = config.within_cluster_sd_permille * (1.0 - icc).sqrt();
    let mut total = 0.0;
    for _family in 0..family_count {
        let family_shift = family_sd * rng.normal();
        for cluster_index in 0..clusters_per_family {
            let exposed = cluster_index < exposure_count;
            let exposure_effect = if exposed { direct_effect } else { 0.0 };
            let cluster_noise = cluster_sd * rng.normal();
            total += pair_shift + family_shift + cluster_noise
                + spillover_effect + exposure_effect;
        }
    }
    total / f64::from(config.clusters_per_model_pair)
}

#[derive(Debug, Clone, Copy, Default)]
struct ScenarioCounts {
    evaluable: u32,
    inconclusive: u32,
    rejected: u32,
    covered: u32,
    surviving_pair_sum: u64,
    effect_sd_sum: f64,
    effect_se_sum: f64,
}

fn run_scenario(
    config: &CausalPowerPlanningConfigV1,
    rng: &mut SplitMix64,
    pair_count: u32,
    direct_effect: f64,
    spillover_effect: f64,
    true_primary_effect: f64,
    check_coverage: bool,
) -> Result<ScenarioCounts, String> {
    let mut counts = ScenarioCounts::default();
    let failure_probability =
        f64::from(config.pair_failure_probability_permille) / 1_000.0;
    for _simulation in 0..config.simulation_replicates {
        let mut pair_effects = Vec::with_capacity(pair_count as usize);
        for _pair in 0..*pair_count {
            if rng.uniform_open01() < failure_probability {
                continue;
            }
            let actual_exposure_fraction = f64::from(config.exposure_fraction_permille) / 1_000.0;
            pair_effects.push(simulated_pair_effect(
                config,
                rng,
                direct_effect,
                spillover_effect,
            ));
        }
        if pair_effects.len() < 2 {
            counts.inconclusive += 1;
            continue;
        }
        let center = sample_mean(&pair_effects)?;
        let sd = sample_sd(&pair_effects, center)?;
        let se = sd / (pair_effects.len() as f64).sqrt();
        let critical = t_critical_975(pair_effects.len() - 1);
        let margin = critical * se;
        let rejected = if se <= f64::EPSILON {
            center.abs() > f64::EPSILON
        } else {
            center.abs() / se > critical
        };
        counts.evaluable += 1;
        if rejected {
            counts.rejected += 1;
        }
        counts.surviving_pair_sum += pair_effects.len() as u64;
        counts.effect_sd_sum += sd;
        counts.effect_se_sum += se;
        if check_coverage && center - margin <= true_primary_effect
            && true_primary_effect <= center + margin
        {
            counts.covered += 1;
        }
    }
    Ok(counts)
}

fn proportion_and_se(successes: u32, trials: u32) -> (Option<f64>, Option<f64>) {
    if trials == 0 {
        return (None, None);
    }
    let proportion = successes as f64 / trials as f64;
    let standard_error = (proportion * (1.0 - proportion) / trials as f64).sqrt();
    (
        Some(proportion * 1_000.0),
        Some(standard_error * 1_000.0),
    )
}

/// Run deterministic Monte Carlo null/alternative simulations. The design
/// effect is spillover + actual_exposure_fraction * direct_exposure_effect.
pub fn plan_causal_power(
    config: &CausalPowerPlanningConfigV1,
) -> Result<CausalPowerPlanningReportV1, String> {
    let seed = validate_config(config)?;
    let config_bytes = serde_json::to_vec(config)
        .map_err(|error| format!("cannot serialize power config: {error}"))?;
    let config_sha256 = hash_domain(b"causal-power-planning-config.v1", &config_bytes);
    let simulation_seed_commitment_sha256 =
        hash_domain(b"causal-power-simulation-seed.v1", &seed);
    let clusters_per_family = config.clusters_per_model_pair / config.task_family_count;
    let exposed_per_family = ((u64::from(clusters_per_family)
        * u64::from(config.exposure_fraction_permille)
        + 500) / 1_000)
        .clamp(1, u64::from(clusters_per_family - 1));
    let actual_exposure_fraction = exposed_per_family as f64
        * f64::from(config.task_family_count)
        / f64::from(config.clusters_per_model_pair);
    let expected_primary_effect = config.spillover_effect_permille
        + actual_exposure_fraction * config.direct_exposure_effect_permille;
    let calibration_tolerance = f64::from(config.calibration_tolerance_permille);
    let alpha = f64::from(config.alpha_permille);
    let target_power = f64::from(config.target_power_permille);
    let max_inconclusive = f64::from(config.maximum_inconclusive_fraction_permille);
    let mut rng = SplitMix64::new(&seed);
    let mut candidate_estimates = Vec::with_capacity(config.candidate_model_pair_counts.len());
    let mut smallest_qualifying = None;

    for pair_count in &config.candidate_model_pair_counts {
        let null = run_scenario(
            config,
            &mut rng,
            *pair_count,
            0.0,
            0.0,
            0.0,
            false,
        )?;
        let alternative = run_scenario(
            config,
            &mut rng,
            *pair_count,
            config.direct_exposure_effect_permille,
            config.spillover_effect_permille,
            expected_primary_effect,
            true,
        )?;
        let (type_i_error, type_i_se) =
            proportion_and_se(null.rejected, null.evaluable);
        let (power, power_se) =
            proportion_and_se(alternative.rejected, alternative.evaluable);
        let (coverage, coverage_se) =
            proportion_and_se(alternative.covered, alternative.evaluable);
        let null_inconclusive_fraction =
            null.inconclusive as f64 * 1_000.0 / f64::from(config.simulation_replicates);
        let alternative_inconclusive_fraction =
            alternative.inconclusive as f64 * 1_000.0 / f64::from(config.simulation_replicates);
        let mean_surviving_pairs =
            alternative.surviving_pair_sum as f64 / f64::from(config.simulation_replicates);
        let estimated_sd = if alternative.evaluable == 0 {
            None
        } else {
            Some(alternative.effect_sd_sum / f64::from(alternative.evaluable))
        };
        let estimated_se = if alternative.evaluable == 0 {
            None
        } else {
            Some(alternative.effect_se_sum / f64::from(alternative.evaluable))
        };
        let type_i_pass = type_i_error.is_some_and(|value| value <= alpha + calibration_tolerance);
        let power_pass = power.is_some_and(|value| value >= target_power);
        let coverage_pass = coverage.is_some_and(|value| {
            (value - f64::from(CAUSAL_POWER_COVERAGE_TARGET_PERMILLE)).abs()
                <= calibration_tolerance
        });
        let inconclusive_pass =
            null_inconclusive_fraction <= max_inconclusive
                && alternative_inconclusive_fraction <= max_inconclusive;
        let meets = type_i_pass && power_pass && coverage_pass && inconclusive_pass;
        if meets && smallest_qualifying.is_none() {
            smallest_qualifying = Some(*pair_count);
        }
        candidate_estimates.push(CausalPowerCandidateEstimateV1 {
            model_pair_count: *pair_count,
            null_evaluable_simulations: null.evaluable,
            null_inconclusive_simulations: null.inconclusive,
            null_inconclusive_fraction_permille: null_inconclusive_fraction,
            type_i_error_permille: type_i_error,
            type_i_error_monte_carlo_se_permille: type_i_se,
            alternative_evaluable_simulations: alternative.evaluable,
            alternative_inconclusive_simulations: alternative.inconclusive,
            alternative_inconclusive_fraction_permille: alternative_inconclusive_fraction,
            estimated_power_permille: power,
            power_monte_carlo_se_permille: power_se,
            interval_coverage_permille: coverage,
            interval_coverage_monte_carlo_se_permille: coverage_se,
            mean_surviving_model_pairs,
            estimated_primary_effect_sd_permille: estimated_sd,
            estimated_primary_effect_se_permille: estimated_se,
            meets_preregistered_targets: meets,
        });
    }

    let mut report = CausalPowerPlanningReportV1 {
        schema: CAUSAL_POWER_REPORT_SCHEMA.to_owned(),
        study_id: config.study_id.clone(),
        primary_metric_id: config.primary_metric_id.clone(),
        config_sha256,
        simulation_seed_commitment_sha256,
        simulation_replicates: config.simulation_replicates,
        nominal_alpha_permille: config.alpha_permille,
        target_power_permille: config.target_power_permille,
        target_interval_coverage_permille: CAUSAL_POWER_COVERAGE_TARGET_PERMILLE,
        calibration_tolerance_permille: config.calibration_tolerance_permille,
        requested_exposure_fraction_permille: config.exposure_fraction_permille,
        actual_balanced_exposure_fraction_permille: actual_exposure_fraction * 1_000.0,
        expected_direct_exposure_effect_permille: config.direct_exposure_effect_permille,
        expected_global_spillover_effect_permille: config.spillover_effect_permille,
        expected_primary_global_effect_permille: expected_primary_effect,
        between_model_pair_sd_permille: config.between_model_pair_sd_permille,
        within_cluster_sd_permille: config.within_cluster_sd_permille,
        within_family_icc_permille: config.within_family_icc_permille,
        pair_failure_probability_permille: config.pair_failure_probability_permille,
        candidate_estimates,
        smallest_qualifying_model_pair_count: smallest_qualifying,
        assumption_summary: "independent model-pair random effects; Gaussian between-pair and family/cluster noise; fixed balanced exposure allocation within task families; MCAR pair failures; homogeneous direct and spillover effects; primary estimand is the all-cluster mean paired advantage".to_owned(),
        design_interpretation: "type-I error and power are estimated across independent model-pair studies; nested source clusters contribute to each pair's cluster-weighted mean but are not independent training replications".to_owned(),
        claim_boundary: "assumption_driven_monte_carlo_design_simulation_not_empirical_training_variance_or_a_guarantee_of_actual_study_power".to_owned(),
        report_sha256: String::new(),
    };
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("cannot serialize power report: {error}"))?;
    report.report_sha256 = hash_domain(b"causal-power-planning-report.v1", &report_bytes);
    Ok(report)
}

pub fn verify_causal_power_report(report: &CausalPowerPlanningReportV1) -> Result<(), String> {
    if report.schema != CAUSAL_POWER_REPORT_SCHEMA {
        return Err(format!("power report schema must be {CAUSAL_POWER_REPORT_SCHEMA}"));
    }
    if !report
        .report_sha256
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || report.report_sha256.len() != 64
    {
        return Err("report_sha256 must be a lowercase SHA-256 digest".to_owned());
    }
    let mut normalized = report.clone();
    normalized.report_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize normalized power report: {error}"))?;
    if hash_domain(b"causal-power-planning-report.v1", &bytes) != report.report_sha256 {
        return Err("power report digest does not match contents".to_owned());
    }
    if report.candidate_estimates.windows(2)
        .any(|pair| pair[0].model_pair_count >= pair[1].model_pair_count)
    {
        return Err("power report candidate counts must be strictly increasing".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> CausalPowerPlanningConfigV1 {
        CausalPowerPlanningConfigV1 {
            schema: CAUSAL_POWER_CONFIG_SCHEMA.to_owned(),
            study_id: "causal-power-fixture".to_owned(),
            primary_metric_id: "exact-correctness".to_owned(),
            candidate_model_pair_counts: vec![6, 12, 20],
            clusters_per_model_pair: 32,
            task_family_count: 4,
            exposure_fraction_permille: 500,
            direct_exposure_effect_permille: 250.0,
            spillover_effect_permille: 100.0,
            between_model_pair_sd_permille: 25.0,
            within_cluster_sd_permille: 100.0,
            within_family_icc_permille: 200,
            pair_failure_probability_permille: 50,
            simulation_replicates: 2_000,
            alpha_permille: 50,
            target_power_permille: 800,
            calibration_tolerance_permille: 30,
            maximum_inconclusive_fraction_permille: 50,
            simulation_seed_hex: "a5".repeat(32),
        }
    }

    #[test]
    fn deterministic_power_plan_and_report_digest_verify() {
        let first = plan_causal_power(&config()).unwrap();
        let second = plan_causal_power(&config()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(verify_causal_power_report(&first).is_ok());
        assert_eq!(first.expected_primary_global_effect_permille, 225.0);
        assert_eq!(first.actual_balanced_exposure_fraction_permille, 500.0);
    }

    #[test]
    fn larger_effects_do_not_reduce_simulated_power_by_large_margin() {
        let mut small = config();
        small.direct_exposure_effect_permille = 25.0;
        small.spillover_effect_permille = 10.0;
        let mut large = config();
        large.direct_exposure_effect_permille = 250.0;
        large.spillover_effect_permille = 100.0;
        let small_report = plan_causal_power(&small).unwrap();
        let large_report = plan_causal_power(&large).unwrap();
        let small_power = small_report.candidate_estimates.last().unwrap()
            .estimated_power_permille.unwrap();
        let large_power = large_report.candidate_estimates.last().unwrap()
            .estimated_power_permille.unwrap();
        assert!(large_power > small_power + 100.0);
    }

    #[test]
    fn higher_between_pair_variability_does_not_improve_power() {
        let mut low = config();
        low.between_model_pair_sd_permille = 10.0;
        let mut high = config();
        high.between_model_pair_sd_permille = 200.0;
        let low_report = plan_causal_power(&low).unwrap();
        let high_report = plan_causal_power(&high).unwrap();
        let low_power = low_report.candidate_estimates.last().unwrap()
            .estimated_power_permille.unwrap();
        let high_power = high_report.candidate_estimates.last().unwrap()
            .estimated_power_permille.unwrap();
        assert!(high_power <= low_power + 10.0);
    }

    #[test]
    fn higher_pair_failure_probability_increases_inconclusive_trials() {
        let mut low = config();
        low.pair_failure_probability_permille = 0;
        let mut high = config();
        high.pair_failure_probability_permille = 500;
        let low_report = plan_causal_power(&low).unwrap();
        let high_report = plan_causal_power(&high).unwrap();
        assert!(
            high_report.candidate_estimates[0].alternative_inconclusive_fraction_permille
                >= low_report.candidate_estimates[0].alternative_inconclusive_fraction_permille
        );
    }

    #[test]
    fn null_simulation_reports_type_i_error_and_coverage_uncertainty() {
        let report = plan_causal_power(&config()).unwrap();
        let first = &report.candidate_estimates[0];
        assert!(first.type_i_error_permille.is_some());
        assert!(first.type_i_error_monte_carlo_se_permille.is_some());
        assert!(first.interval_coverage_permille.is_some());
        assert!(first.interval_coverage_monte_carlo_se_permille.is_some());
    }

    #[test]
    fn invalid_cluster_layout_and_simulation_budget_fail_closed() {
        let mut invalid = config();
        invalid.clusters_per_model_pair = 31;
        assert!(plan_causal_power(&invalid).unwrap_err().contains("divide evenly"));
        let mut invalid = config();
        invalid.simulation_replicates = 999;
        assert!(plan_causal_power(&invalid).unwrap_err().contains("simulation_replicates"));
        let mut invalid = config();
        invalid.simulation_replicates = 100_000;
        invalid.candidate_model_pair_counts = vec![1_000];
        invalid.clusters_per_model_pair = 1_000;
        invalid.task_family_count = 10;
        assert!(plan_causal_power(&invalid).unwrap_err().contains("operation budget"));
    }

    #[test]
    fn report_digest_mutations_are_rejected() {
        let mut report = plan_causal_power(&config()).unwrap();
        report.candidate_estimates[0].estimated_power_permille = Some(999.0);
        assert!(verify_causal_power_report(&report).unwrap_err().contains("digest does not match"));
    }

    #[test]
    fn simulation_seed_and_config_changes_change_plan_commitments() {
        let first = plan_causal_power(&config()).unwrap();
        let mut changed = config();
        changed.simulation_seed_hex = "5a".repeat(32);
        let second = plan_causal_power(&changed).unwrap();
        assert_ne!(first.simulation_seed_commitment_sha256, second.simulation_seed_commitment_sha256);
        assert_ne!(first.config_sha256, second.config_sha256);
    }

    #[test]
    fn config_and_report_schema_field_sets_match_rust_structs() {
        let config_schema: Value = serde_json::from_str(include_str!(
            "../schema/causal-power-config.schema.json"
        )).unwrap();
        let report_schema: Value = serde_json::from_str(include_str!(
            "../schema/causal-power-report.schema.json"
        )).unwrap();
        let config_value = serde_json::to_value(config()).unwrap();
        let report_value = serde_json::to_value(plan_causal_power(&config()).unwrap()).unwrap();
        for (schema, value, expected) in [
            (&config_schema, config_value, CAUSAL_POWER_CONFIG_SCHEMA),
            (&report_schema, report_value, CAUSAL_POWER_REPORT_SCHEMA),
        ] {
            assert_eq!(schema["properties"]["schema"]["const"].as_str(), Some(expected));
            let required: std::collections::BTreeSet<String> = schema["required"].as_array().unwrap()
                .iter().map(|item| item.as_str().unwrap().to_owned()).collect();
            let properties: std::collections::BTreeSet<String> = schema["properties"].as_object().unwrap()
                .keys().cloned().collect();
            let actual: std::collections::BTreeSet<String> = value.as_object().unwrap()
                .keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
    }
}
