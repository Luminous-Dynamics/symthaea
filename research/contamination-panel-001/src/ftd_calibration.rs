//! Finite-sample Monte Carlo qualification for FTD-style clean-set selection.
//!
//! This simulator stresses finite reference banks, item clustering, detector
//! dependence, score-tail direction, and reference-distribution mismatch.
//! It reports empirical diagnostics with Monte Carlo uncertainty; it does not
//! prove or disprove a theorem from a finite number of repetitions.

use crate::{
    ftd::{ftd_weighted_cauchy_bh, FeaturePValuesV1, FtdWeightPolicyV1, PairPValueV1, ReferenceTailV1},
    hash_domain, hmac_sha256,
};
use serde::{Deserialize, Serialize};

pub const CALIBRATION_REPORT_SCHEMA: &str =
    "symthaea.contamination_panel_ftd_calibration_report.v1";
const MAX_MONTE_CARLO_WORK: u128 = 100_000_000;
const HOEFFDING_LOG_95: f64 = 3.688_879_454_113_936_3;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDesignV1 {
    SharedPerFeatureBank,
    IndependentPerHypothesis,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FtdCalibrationConfigV1 {
    pub repetitions: u32,
    pub hypothesis_count: u32,
    pub reference_count_per_hypothesis: u32,
    pub feature_count: u8,
    pub clean_fraction_permille: u16,
    /// Positive pushes the reference distribution lower; negative pushes it higher.
    pub reference_shift_permille: i16,
    pub reference_design: ReferenceDesignV1,
    pub item_cluster_size: u16,
    pub item_cluster_correlation_permille: u16,
    pub feature_correlation_permille: u16,
    pub tail_direction: ReferenceTailV1,
    pub weight_policy: FtdWeightPolicyV1,
    pub softmax_temperature_milli: Option<u16>,
    pub alpha_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedEstimateV1 {
    pub estimate: f64,
    pub lower_95_hoeffding: f64,
    pub upper_95_hoeffding: f64,
    pub monte_carlo_standard_error: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FtdCalibrationReportV1 {
    pub schema: String,
    pub config_sha256: String,
    pub seed_commitment_sha256: String,
    pub config: FtdCalibrationConfigV1,
    pub repetitions_completed: u32,
    pub null_hypothesis_count_total: u64,
    pub clean_hypothesis_count_total: u64,
    pub mean_false_discovery_proportion: BoundedEstimateV1,
    pub run_level_fdr_exceedance_rate: BoundedEstimateV1,
    pub mean_clean_sample_power: Option<BoundedEstimateV1>,
    pub mean_selected_fraction: BoundedEstimateV1,
    pub mean_selected_clean_count: f64,
    pub mean_selected_contaminated_count: f64,
    pub mean_bh_retained_count: f64,
    pub fdr_target: f64,
    pub run_fdr_exceedance_upper_95_hoeffding: f64,
    pub calibration_design: String,
    pub monte_carlo_interpretation: String,
    pub nominal_fdr_guarantee: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_seed(seed: &[u8], repetition: u64) -> Self {
        let digest = hmac_sha256(seed, &repetition.to_be_bytes());
        let state = u64::from_le_bytes(digest[..8].try_into().expect("fixed digest slice"));
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn choose_permille(&mut self, probability: u16) -> bool {
        probability >= 1_000
            || (probability > 0 && self.next_u64() % 1_000 < u64::from(probability))
    }
}

fn validate_config(config: &FtdCalibrationConfigV1, seed: &[u8]) -> Result<(), String> {
    if seed.len() < 32 {
        return Err("calibration seed must contain at least 32 bytes".to_owned());
    }
    if !(10..=10_000).contains(&config.repetitions) {
        return Err("repetitions must be in [10, 10000]".to_owned());
    }
    if !(10..=10_000).contains(&config.hypothesis_count) {
        return Err("hypothesis_count must be in [10, 10000]".to_owned());
    }
    if !(10..=10_000).contains(&config.reference_count_per_hypothesis) {
        return Err("reference_count_per_hypothesis must be in [10, 10000]".to_owned());
    }
    if !(1..=8).contains(&config.feature_count) {
        return Err("feature_count must be in [1, 8]".to_owned());
    }
    if config.clean_fraction_permille > 1_000 {
        return Err("clean_fraction_permille must be in [0, 1000]".to_owned());
    }
    if !(-1_000..=1_000).contains(&config.reference_shift_permille) {
        return Err("reference_shift_permille must be in [-1000, 1000]".to_owned());
    }
    if !(1..=1_000).contains(&config.item_cluster_size) {
        return Err("item_cluster_size must be in [1, 1000]".to_owned());
    }
    if config.item_cluster_correlation_permille > 1_000
        || config.feature_correlation_permille > 1_000
    {
        return Err("correlation permille fields must be in [0, 1000]".to_owned());
    }
    if !(1..=200).contains(&config.alpha_permille) {
        return Err("alpha_permille must be in [1, 200]".to_owned());
    }
    match (config.weight_policy, config.softmax_temperature_milli) {
        (FtdWeightPolicyV1::PaperNormalizedRejectionCounts, None) => {}
        (FtdWeightPolicyV1::PaperNormalizedRejectionCounts, Some(_)) => {
            return Err("paper weight policy must not have a softmax temperature".to_owned());
        }
        (FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened, Some(value))
            if (1..=10_000).contains(&value) => {}
        (FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened, _) => {
            return Err("official weight policy requires temperature in [1, 10000]".to_owned());
        }
    }
    let work = u128::from(config.repetitions)
        .checked_mul(u128::from(config.hypothesis_count))
        .and_then(|value| value.checked_mul(u128::from(config.reference_count_per_hypothesis)))
        .and_then(|value| value.checked_mul(u128::from(config.feature_count)))
        .ok_or_else(|| "calibration work estimate overflows".to_owned())?;
    if work > MAX_MONTE_CARLO_WORK {
        return Err(format!(
            "calibration exceeds safety work budget ({work} > {MAX_MONTE_CARLO_WORK})"
        ));
    }
    Ok(())
}

fn reference_score(rng: &mut SplitMix64, shift_permille: i16) -> f64 {
    let uniform = rng.uniform_open();
    let magnitude = f64::from(shift_permille.unsigned_abs()) / 1_000.0;
    let exponent = if shift_permille >= 0 {
        1.0 + magnitude
    } else {
        1.0 / (1.0 + magnitude)
    };
    uniform.powf(exponent)
}

fn build_item_scores(
    config: &FtdCalibrationConfigV1,
    rng: &mut SplitMix64,
) -> (Vec<bool>, Vec<Vec<f64>>) {
    let n = config.hypothesis_count as usize;
    let k = config.feature_count as usize;
    let clean_count =
        (u64::from(config.hypothesis_count) * u64::from(config.clean_fraction_permille) / 1_000)
            as usize;
    let mut is_clean = vec![false; n];
    for value in is_clean.iter_mut().take(clean_count) {
        *value = true;
    }
    for index in (1..n).rev() {
        let other = (rng.next_u64() % (index as u64 + 1)) as usize;
        is_clean.swap(index, other);
    }

    let mut scores = vec![vec![0.0; n]; k];
    let cluster_size = usize::from(config.item_cluster_size).max(1);
    for cluster_start in (0..n).step_by(cluster_size) {
        let cluster_end = (cluster_start + cluster_size).min(n);
        let share_cluster = rng.choose_permille(config.item_cluster_correlation_permille);
        let shared_cluster_feature_values: Vec<f64> =
            (0..k).map(|_| rng.uniform_open()).collect();

        for item_index in cluster_start..cluster_end {
            let share_features = rng.choose_permille(config.feature_correlation_permille);
            let shared_item_feature_value = rng.uniform_open();

            for feature_index in 0..k {
                let raw_score = if share_cluster {
                    shared_cluster_feature_values[feature_index]
                } else if share_features {
                    shared_item_feature_value
                } else {
                    rng.uniform_open()
                };
                // Mixture dependence preserves each detector/item marginal as
                // Uniform(0,1), unlike a weighted average of uniforms.
                scores[feature_index][item_index] = if is_clean[item_index] {
                    raw_score.sqrt()
                } else {
                    raw_score
                };
            }
        }
    }
    (is_clean, scores)
}

fn randomized_empirical_p(
    reference_sorted: &[f64],
    test_score: f64,
    direction: ReferenceTailV1,
    tie_uniform: f64,
) -> f64 {
    let count = match direction {
        ReferenceTailV1::Greater => {
            let first_ge = reference_sorted.partition_point(|value| *value < test_score);
            reference_sorted.len() - first_ge
        }
        ReferenceTailV1::Less => {
            reference_sorted.partition_point(|value| *value <= test_score)
        }
    };
    (count as f64 + tie_uniform) / (reference_sorted.len() as f64 + 1.0)
}

fn make_feature_p_values(
    config: &FtdCalibrationConfigV1,
    rng: &mut SplitMix64,
    feature_index: usize,
    test_scores: &[f64],
) -> Result<FeaturePValuesV1, String> {
    let n = test_scores.len();
    let m = config.reference_count_per_hypothesis as usize;
    let mut p_values = Vec::with_capacity(n);
    let mut reference_commitment_material = Vec::<Vec<f64>>::new();

    match config.reference_design {
        ReferenceDesignV1::SharedPerFeatureBank => {
            let mut reference: Vec<f64> = (0..m)
                .map(|_| reference_score(rng, config.reference_shift_permille))
                .collect();
            reference.sort_by(f64::total_cmp);
            reference_commitment_material.push(reference.clone());
            for (index, score) in test_scores.iter().copied().enumerate() {
                p_values.push(PairPValueV1 {
                    pair_id: format!("item-{index:08}"),
                    p_value: randomized_empirical_p(
                        &reference,
                        score,
                        config.tail_direction,
                        rng.uniform_open(),
                    ),
                });
            }
        }
        ReferenceDesignV1::IndependentPerHypothesis => {
            for (index, score) in test_scores.iter().copied().enumerate() {
                let mut reference: Vec<f64> = (0..m)
                    .map(|_| reference_score(rng, config.reference_shift_permille))
                    .collect();
                reference.sort_by(f64::total_cmp);
                reference_commitment_material.push(reference.clone());
                p_values.push(PairPValueV1 {
                    pair_id: format!("item-{index:08}"),
                    p_value: randomized_empirical_p(
                        &reference,
                        score,
                        config.tail_direction,
                        rng.uniform_open(),
                    ),
                });
            }
        }
    }

    let encoded = serde_json::to_vec(&reference_commitment_material)
        .map_err(|error| format!("cannot serialize reference banks: {error}"))?;
    Ok(FeaturePValuesV1 {
        feature_id: format!("feature-{feature_index:02}"),
        direction: config.tail_direction,
        reference_count: u64::from(config.reference_count_per_hypothesis),
        reference_scores_sha256: hash_domain(b"ftd-calibration-reference.v1", &encoded),
        p_values,
    })
}

fn bounded_estimate(values: &[f64]) -> Result<BoundedEstimateV1, String> {
    if values.is_empty() {
        return Err("cannot estimate an empty Monte Carlo outcome vector".to_owned());
    }
    if values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err("Monte Carlo outcomes must be finite and in [0, 1]".to_owned());
    }
    let count = values.len() as f64;
    let estimate = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|value| (value - estimate).powi(2)).sum::<f64>() / count;
    let standard_error = (variance / count).sqrt();
    let radius = (HOEFFDING_LOG_95 / (2.0 * count)).sqrt();
    Ok(BoundedEstimateV1 {
        estimate,
        lower_95_hoeffding: (estimate - radius).max(0.0),
        upper_95_hoeffding: (estimate + radius).min(1.0),
        monte_carlo_standard_error: standard_error,
    })
}

/// Run a seeded Monte Carlo experiment over FTD-style empirical references and
/// the paper or official-code fusion/selection policy.
pub fn run_ftd_calibration(
    config: &FtdCalibrationConfigV1,
    seed: &[u8],
) -> Result<FtdCalibrationReportV1, String> {
    validate_config(config, seed)?;
    let config_bytes = serde_json::to_vec(config)
        .map_err(|error| format!("cannot serialize calibration config: {error}"))?;

    let mut fdp_values = Vec::with_capacity(config.repetitions as usize);
    let mut power_values = Vec::with_capacity(config.repetitions as usize);
    let mut selected_fraction_values = Vec::with_capacity(config.repetitions as usize);
    let mut run_exceedances = Vec::with_capacity(config.repetitions as usize);
    let mut selected_clean_total = 0_u64;
    let mut selected_contaminated_total = 0_u64;
    let mut total_nulls = 0_u64;
    let mut total_clean = 0_u64;
    let mut selected_total = 0_u64;
    let alpha = f64::from(config.alpha_permille) / 1_000.0;

    for repetition in 0..u64::from(config.repetitions) {
        let mut rng = SplitMix64::from_seed(seed, repetition);
        let (is_clean, scores) = build_item_scores(config, &mut rng);
        let clean_count = is_clean.iter().filter(|value| **value).count() as u64;
        let contaminated_count = u64::try_from(is_clean.len())
            .map_err(|_| "item count overflows u64")?
            .checked_sub(clean_count)
            .ok_or_else(|| "contaminated count underflows".to_owned())?;
        total_clean = total_clean.checked_add(clean_count)
            .ok_or_else(|| "total clean count overflows u64".to_owned())?;
        total_nulls = total_nulls.checked_add(contaminated_count)
            .ok_or_else(|| "total contaminated count overflows u64".to_owned())?;

        let mut features = Vec::with_capacity(config.feature_count as usize);
        for (feature_index, feature_scores) in scores.iter().enumerate() {
            features.push(make_feature_p_values(config, &mut rng, feature_index, feature_scores)?);
        }
        let fused = ftd_weighted_cauchy_bh(
            &features,
            alpha,
            config.weight_policy,
            config.softmax_temperature_milli,
        )?;

        let mut selected = 0_u64;
        let mut selected_clean = 0_u64;
        let mut selected_contaminated = 0_u64;
        for row in &fused.combined_p_values {
            if !row.retained_as_clean {
                continue;
            }
            let index = row.pair_id.strip_prefix("item-")
                .ok_or_else(|| "fused report contains invalid item ID".to_owned())?
                .parse::<usize>()
                .map_err(|_| "fused report item ID is not numeric".to_owned())?;
            let truth = *is_clean.get(index)
                .ok_or_else(|| "fused report references unknown item".to_owned())?;
            selected = selected.checked_add(1)
                .ok_or_else(|| "selected-item count overflows u64")?;
            if truth {
                selected_clean = selected_clean.checked_add(1)
                    .ok_or_else(|| "selected-clean count overflows u64")?;
            } else {
                selected_contaminated = selected_contaminated.checked_add(1)
                    .ok_or_else(|| "selected-contaminated count overflows u64")?;
            }
        }

        selected_clean_total = selected_clean_total.checked_add(selected_clean)
            .ok_or_else(|| "selected-clean total overflows u64")?;
        selected_contaminated_total = selected_contaminated_total.checked_add(selected_contaminated)
            .ok_or_else(|| "selected-contaminated total overflows u64")?;
        selected_total = selected_total.checked_add(selected)
            .ok_or_else(|| "selected total overflows u64")?;

        let fdp = if selected == 0 { 0.0 } else { selected_contaminated as f64 / selected as f64 };
        fdp_values.push(fdp);
        if clean_count > 0 {
            power_values.push(selected_clean as f64 / clean_count as f64);
        }
        selected_fraction_values.push(selected as f64 / is_clean.len() as f64);
        run_exceedances.push(if fdp > alpha { 1.0 } else { 0.0 });
    }

    let fdr_estimate = bounded_estimate(&fdp_values)?;
    let exceedance_estimate = bounded_estimate(&run_exceedances)?;
    let selected_fraction = bounded_estimate(&selected_fraction_values)?;
    let power_estimate = if power_values.is_empty() {
        None
    } else {
        Some(bounded_estimate(&power_values)?)
    };
    let repetitions = f64::from(config.repetitions);
    let fdr_target = alpha;

    Ok(FtdCalibrationReportV1 {
        schema: CALIBRATION_REPORT_SCHEMA.to_owned(),
        config_sha256: hash_domain(b"ftd-calibration-config.v1", &config_bytes),
        seed_commitment_sha256: hash_domain(b"ftd-calibration-seed.v1", seed),
        config: config.clone(),
        repetitions_completed: config.repetitions,
        null_hypothesis_count_total: total_nulls,
        clean_hypothesis_count_total: total_clean,
        mean_false_discovery_proportion: fdr_estimate,
        run_level_fdr_exceedance_rate: exceedance_estimate,
        mean_clean_sample_power: power_estimate,
        mean_selected_fraction: selected_fraction,
        mean_selected_clean_count: selected_clean_total as f64 / repetitions,
        mean_selected_contaminated_count: selected_contaminated_total as f64 / repetitions,
        mean_bh_retained_count: selected_total as f64 / repetitions,
        fdr_target,
        run_fdr_exceedance_upper_95_hoeffding: exceedance_estimate.upper_95_hoeffding,
        calibration_design: match config.reference_design {
            ReferenceDesignV1::SharedPerFeatureBank =>
                "one_finite_reference_bank_per_feature_reused_across_all_test_items".to_owned(),
            ReferenceDesignV1::IndependentPerHypothesis =>
                "independent_finite_reference_bank_per_feature_and_test_item".to_owned(),
        },
        monte_carlo_interpretation: "bounded_outcome_mean_estimates_with_95_percent_hoeffding_intervals_over_independent_seeded_repetitions".to_owned(),
        nominal_fdr_guarantee: "not_established_by_monte_carlo; empirical_stress_diagnostic_only".to_owned(),
        claim_boundary: "synthetic_score_population_only; does_not_establish_real_model_training_exposure_or_FDR_theorem".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FtdCalibrationConfigV1 {
        FtdCalibrationConfigV1 {
            repetitions: 40,
            hypothesis_count: 64,
            reference_count_per_hypothesis: 64,
            feature_count: 3,
            clean_fraction_permille: 300,
            reference_shift_permille: 0,
            reference_design: ReferenceDesignV1::SharedPerFeatureBank,
            item_cluster_size: 8,
            item_cluster_correlation_permille: 0,
            feature_correlation_permille: 0,
            tail_direction: ReferenceTailV1::Greater,
            weight_policy: FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
            softmax_temperature_milli: None,
            alpha_permille: 50,
        }
    }

    #[test]
    fn calibration_is_deterministic_for_fixed_seed_and_config() {
        let cfg = config();
        let first = run_ftd_calibration(&cfg, &[0x51; 32]).unwrap();
        let second = run_ftd_calibration(&cfg, &[0x51; 32]).unwrap();
        assert_eq!(serde_json::to_vec(&first).unwrap(), serde_json::to_vec(&second).unwrap());
        assert_eq!(first.repetitions_completed, 40);
        assert_eq!(first.nominal_fdr_guarantee, "not_established_by_monte_carlo; empirical_stress_diagnostic_only");
    }

    #[test]
    fn report_contains_bounded_fdr_power_and_retention_estimates() {
        let report = run_ftd_calibration(&config(), &[0x52; 32]).unwrap();
        assert!((0.0..=1.0).contains(&report.mean_false_discovery_proportion.estimate));
        assert!(report.mean_false_discovery_proportion.lower_95_hoeffding
            <= report.mean_false_discovery_proportion.estimate);
        assert!(report.mean_false_discovery_proportion.upper_95_hoeffding
            >= report.mean_false_discovery_proportion.estimate);
        assert!(report.mean_clean_sample_power.is_some());
        assert!((0.0..=1.0).contains(&report.mean_selected_fraction.estimate));
    }

    #[test]
    fn all_null_calibration_reports_power_as_not_estimable() {
        let mut cfg = config();
        cfg.clean_fraction_permille = 0;
        let report = run_ftd_calibration(&cfg, &[0x53; 32]).unwrap();
        assert_eq!(report.clean_hypothesis_count_total, 0);
        assert!(report.mean_clean_sample_power.is_none());
    }

    #[test]
    fn shifted_reference_and_block_dependence_are_explicit() {
        let mut cfg = config();
        cfg.reference_shift_permille = 750;
        cfg.item_cluster_correlation_permille = 1_000;
        cfg.feature_correlation_permille = 1_000;
        let report = run_ftd_calibration(&cfg, &[0x54; 32]).unwrap();
        assert_eq!(report.config.reference_shift_permille, 750);
        assert_eq!(report.config.item_cluster_correlation_permille, 1_000);
        assert_eq!(report.config.feature_correlation_permille, 1_000);
    }

    #[test]
    fn independent_per_hypothesis_reference_is_distinct_from_shared_bank() {
        let mut cfg = config();
        cfg.reference_design = ReferenceDesignV1::IndependentPerHypothesis;
        let report = run_ftd_calibration(&cfg, &[0x55; 32]).unwrap();
        assert_eq!(report.calibration_design,
            "independent_finite_reference_bank_per_feature_and_test_item");
    }

    #[test]
    fn invalid_and_excessive_calibration_budgets_fail_closed() {
        let mut cfg = config();
        cfg.feature_count = 0;
        assert!(run_ftd_calibration(&cfg, &[0x56; 32]).is_err());
        cfg = config();
        cfg.repetitions = 10_000;
        cfg.hypothesis_count = 10_000;
        cfg.reference_count_per_hypothesis = 10_000;
        assert!(run_ftd_calibration(&cfg, &[0x56; 32]).is_err());
    }

    #[test]
    fn high_score_clean_alternative_has_greater_power_under_upper_tail() {
        let mut upper = config();
        upper.repetitions = 100;
        upper.hypothesis_count = 128;
        upper.reference_count_per_hypothesis = 128;
        upper.clean_fraction_permille = 500;
        upper.reference_design = ReferenceDesignV1::IndependentPerHypothesis;
        upper.tail_direction = ReferenceTailV1::Greater;
        let upper_report = run_ftd_calibration(&upper, &[0x58; 32]).unwrap();

        let mut lower = upper.clone();
        lower.tail_direction = ReferenceTailV1::Less;
        let lower_report = run_ftd_calibration(&lower, &[0x58; 32]).unwrap();

        assert!(
            upper_report.mean_clean_sample_power.unwrap().estimate
                > lower_report.mean_clean_sample_power.unwrap().estimate
        );
    }

    #[test]
    fn upper_and_lower_tails_are_separately_configurable() {
        let mut upper = config();
        upper.reference_design = ReferenceDesignV1::IndependentPerHypothesis;
        upper.item_cluster_correlation_permille = 0;
        upper.feature_correlation_permille = 0;
        upper.tail_direction = ReferenceTailV1::Greater;
        let upper_report = run_ftd_calibration(&upper, &[0x57; 32]).unwrap();
        let mut lower = upper.clone();
        lower.tail_direction = ReferenceTailV1::Less;
        let lower_report = run_ftd_calibration(&lower, &[0x57; 32]).unwrap();
        assert_eq!(upper_report.config.tail_direction, ReferenceTailV1::Greater);
        assert_eq!(lower_report.config.tail_direction, ReferenceTailV1::Less);
    }
}
