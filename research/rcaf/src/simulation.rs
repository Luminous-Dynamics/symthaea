//! Synthetic null and injection-recovery calibration for RCAF.
//!
//! The generator samples valid correlation matrices from a Gaussian factor model.
//! It is an implementation-calibration harness, not a biological simulator.

use super::{
    exact_sign_test, weighted_jaccard_correlation_matrices, CorrelationWjMode, MetricError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityEstimator {
    AbsoluteWeightedJaccard,
    SignedShiftWeightedJaccard,
    NormalizedMeanAbsoluteEdgeError,
}

impl SimilarityEstimator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AbsoluteWeightedJaccard => "absolute_weighted_jaccard",
            Self::SignedShiftWeightedJaccard => "signed_shift_weighted_jaccard",
            Self::NormalizedMeanAbsoluteEdgeError => "normalized_mean_absolute_edge_error",
        }
    }
}

pub fn correlation_similarity(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    estimator: SimilarityEstimator,
) -> Result<f64, MetricError> {
    match estimator {
        SimilarityEstimator::AbsoluteWeightedJaccard => {
            weighted_jaccard_correlation_matrices(left, right, CorrelationWjMode::Absolute, 1)
        }
        SimilarityEstimator::SignedShiftWeightedJaccard => {
            weighted_jaccard_correlation_matrices(left, right, CorrelationWjMode::SignedShift, 1)
        }
        SimilarityEstimator::NormalizedMeanAbsoluteEdgeError => {
            let left_size = super::validate_correlation_matrix(left)?;
            let right_size = super::validate_correlation_matrix(right)?;
            if left_size != right_size {
                return Err(MetricError::MatrixSizeMismatch {
                    left: left_size,
                    right: right_size,
                });
            }
            let mut absolute_error = 0.0;
            let mut edge_count = 0usize;
            for (row_index, row) in left.iter().enumerate() {
                for (&left_value, &right_value) in
                    row.iter().zip(right[row_index].iter()).skip(row_index + 1)
                {
                    absolute_error += (left_value - right_value).abs();
                    edge_count += 1;
                }
            }
            if edge_count == 0 {
                return Err(MetricError::TooFewRegions);
            }
            Ok((1.0 - absolute_error / edge_count as f64 / 2.0).clamp(0.0, 1.0))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorTournamentReport {
    pub schema_version: &'static str,
    pub estimators_compared: usize,
    pub results: Vec<InjectionRecoveryReport>,
}

impl EstimatorTournamentReport {
    pub fn to_json_pretty(&self) -> String {
        let rendered_reports = self
            .results
            .iter()
            .map(|report| {
                report
                    .to_json_pretty()
                    .lines()
                    .map(|line| format!("  {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"estimators_compared\": {},\n",
                "  \"results\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\"synthetic_matched_inputs_only\", \"not_a_biological_estimator_ranking\", \"not_clinically_validated\"]\n",
                "}}\n"
            ),
            self.schema_version,
            self.estimators_compared,
            rendered_reports
        )
    }
}

pub fn run_estimator_tournament(
    config: &InjectionRecoveryConfig,
) -> Result<EstimatorTournamentReport, MetricError> {
    let estimators = [
        SimilarityEstimator::AbsoluteWeightedJaccard,
        SimilarityEstimator::SignedShiftWeightedJaccard,
        SimilarityEstimator::NormalizedMeanAbsoluteEdgeError,
    ];
    let mut results = Vec::with_capacity(estimators.len());
    for estimator in estimators {
        let mut estimator_config = config.clone();
        estimator_config.estimator = estimator;
        results.push(run_injection_recovery(&estimator_config)?);
    }
    Ok(EstimatorTournamentReport {
        schema_version: "rcaf.estimator-tournament/0.2.0",
        estimators_compared: results.len(),
        results,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorDoseResponseCurve {
    pub estimator: &'static str,
    pub sweep: InjectionRecoverySweepReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorDoseResponseTournamentReport {
    pub schema_version: &'static str,
    pub estimators_compared: usize,
    pub effect_sizes: Vec<f64>,
    pub curves: Vec<EstimatorDoseResponseCurve>,
}

impl EstimatorDoseResponseTournamentReport {
    pub fn to_json_pretty(&self) -> String {
        let rendered_curves = self
            .curves
            .iter()
            .map(|curve| {
                let sweep_json = curve.sweep.to_json_pretty();
                let indented_sweep = sweep_json
                    .lines()
                    .map(|line| format!("      {line}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    concat!(
                        "    {{\n",
                        "      \"estimator\": \"{}\",\n",
                        "      \"sweep\": {}\n",
                        "    }}"
                    ),
                    curve.estimator, indented_sweep
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let rendered_effects = self
            .effect_sizes
            .iter()
            .map(|value| format!("{value:.12}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"estimators_compared\": {},\n",
                "  \"effect_sizes\": [{}],\n",
                "  \"curves\": [\n{} \n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"matched_synthetic_inputs_only\",\n",
                "    \"not_a_biological_estimator_ranking\",\n",
                "    \"no_universal_metric_winner_is_implied\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version, self.estimators_compared, rendered_effects, rendered_curves
        )
    }
}

/// Compare each estimator across the same predeclared injection strengths.
///
/// Each estimator receives the same seed, participant loading draws, temporal innovations,
/// repeat segments and injection grid. This produces an estimator-by-dose surface rather than
/// a single strong-effect comparison that may saturate all methods.
pub fn run_estimator_dose_response_tournament(
    config: &InjectionRecoveryConfig,
    effect_sizes: &[f64],
) -> Result<EstimatorDoseResponseTournamentReport, MetricError> {
    let estimators = [
        SimilarityEstimator::AbsoluteWeightedJaccard,
        SimilarityEstimator::SignedShiftWeightedJaccard,
        SimilarityEstimator::NormalizedMeanAbsoluteEdgeError,
    ];
    let mut curves = Vec::with_capacity(estimators.len());
    for estimator in estimators {
        let mut estimator_config = config.clone();
        estimator_config.estimator = estimator;
        let sweep = run_injection_recovery_sweep(&estimator_config, effect_sizes)?;
        curves.push(EstimatorDoseResponseCurve {
            estimator: estimator.as_str(),
            sweep,
        });
    }
    Ok(EstimatorDoseResponseTournamentReport {
        schema_version: "rcaf.estimator-dose-response-tournament/0.2.0",
        estimators_compared: curves.len(),
        effect_sizes: effect_sizes.to_vec(),
        curves,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionRecoveryConfig {
    pub participants: usize,
    pub regions: usize,
    pub estimator: SimilarityEstimator,
    pub samples_per_segment: usize,
    pub monte_carlo_replicates: usize,
    pub module_start: usize,
    pub module_size: usize,
    pub global_loading: f64,
    pub module_loading: f64,
    pub loading_jitter: f64,
    /// Strength of a shared observation-only nuisance factor injected into one repeat.
    pub common_mode_nuisance_loading: f64,
    /// Stationary AR(1) coefficient applied to factors and residuals.
    pub temporal_autocorrelation: f64,
    pub injection_loading_delta: f64,
    pub alpha: f64,
    pub seed: u64,
}

impl Default for InjectionRecoveryConfig {
    fn default() -> Self {
        Self {
            participants: 24,
            regions: 12,
            estimator: SimilarityEstimator::AbsoluteWeightedJaccard,
            samples_per_segment: 160,
            monte_carlo_replicates: 100,
            module_start: 4,
            module_size: 5,
            global_loading: 0.20,
            module_loading: 0.18,
            loading_jitter: 0.06,
            common_mode_nuisance_loading: 0.20,
            temporal_autocorrelation: 0.30,
            injection_loading_delta: 0.40,
            alpha: 0.05,
            seed: 0xC37B_2026_1121,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinomialRate {
    pub successes: usize,
    pub trials: usize,
    pub estimate: f64,
    pub ci95_lower: f64,
    pub ci95_upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionRecoveryReport {
    pub schema_version: &'static str,
    pub seed: u64,
    pub participants_per_replicate: usize,
    pub regions: usize,
    pub samples_per_segment: usize,
    pub monte_carlo_replicates: usize,
    pub module_start: usize,
    pub module_size: usize,
    pub global_loading: f64,
    pub module_loading: f64,
    pub loading_jitter: f64,
    pub common_mode_nuisance_loading: f64,
    pub estimator: &'static str,
    pub injection_loading_delta: f64,
    pub temporal_autocorrelation: f64,
    pub alpha: f64,
    /// Type-I error estimate under exchangeable no-change matrices.
    pub null_false_positive_rate: BinomialRate,
    /// Rejection rate when a shared observation nuisance changes but latent loadings do not.
    pub nuisance_only_rejection_rate: BinomialRate,
    /// Two-sided difference detection, regardless of observed direction (legacy metric).
    pub injection_detection_power: BinomialRate,
    /// Significant recovery in the preregistered positive paired-gap direction.
    pub injection_directional_detection_power: BinomialRate,
    /// Significant two-sided rejection in the direction opposite to the positive injection.
    pub injection_wrong_direction_rejection_rate: BinomialRate,
    /// Median across Monte Carlo replicates of participant-level median paired gaps.
    pub median_null_paired_gap: f64,
    pub median_nuisance_only_paired_gap: f64,
    pub median_injected_paired_gap: f64,
    /// Median across replicates of the proportion of participants with positive gaps.
    pub median_null_positive_gap_fraction: f64,
    pub median_injected_positive_gap_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionRecoverySweepReport {
    pub schema_version: &'static str,
    pub effect_size_points: usize,
    pub endpoint_median_gap_difference: f64,
    pub monotonic_adjacent_pair_fraction: f64,
    pub reports: Vec<InjectionRecoveryReport>,
}

impl InjectionRecoverySweepReport {
    /// Render the complete dose-response curve as stable JSON.
    pub fn to_json_pretty(&self) -> String {
        let rendered_reports = self
            .reports
            .iter()
            .map(|report| {
                report
                    .to_json_pretty()
                    .lines()
                    .map(|line| format!("  {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"effect_size_points\": {},\n",
                "  \"endpoint_median_gap_difference\": {:.12},\n",
                "  \"monotonic_adjacent_pair_fraction\": {:.12},\n",
                "  \"reports\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\"synthetic_factor_model_only\", \"not_a_biological_dose_response\", \"not_clinically_validated\"]\n",
                "}}\n"
            ),
            self.schema_version,
            self.effect_size_points,
            self.endpoint_median_gap_difference,
            self.monotonic_adjacent_pair_fraction,
            rendered_reports,
        )
    }
}

/// Evaluate an increasing grid of factor-loading perturbations with common random numbers.
///
/// The same seed and same draw order are used at every effect size. Thus the baseline
/// participant loadings and sampled segments are held constant while only the declared
/// injected loading delta changes. The first point may be zero and is the null-dose control.
pub fn run_injection_recovery_sweep(
    config: &InjectionRecoveryConfig,
    effect_sizes: &[f64],
) -> Result<InjectionRecoverySweepReport, MetricError> {
    if effect_sizes.len() < 2 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "at least two effect sizes are required for a sweep",
        });
    }
    let mut previous = -1.0;
    for effect in effect_sizes {
        if !effect.is_finite() || *effect < 0.0 || *effect <= previous {
            return Err(MetricError::InvalidSimulationConfig {
                reason: "effect sizes must be finite, non-negative and strictly increasing",
            });
        }
        previous = *effect;
    }

    let mut reports = Vec::with_capacity(effect_sizes.len());
    for effect in effect_sizes {
        let mut point_config = config.clone();
        point_config.injection_loading_delta = *effect;
        reports.push(run_injection_recovery(&point_config)?);
    }

    let first_gap = reports
        .first()
        .expect("validated effect-size grid is non-empty")
        .median_injected_paired_gap;
    let last_gap = reports
        .last()
        .expect("validated effect-size grid is non-empty")
        .median_injected_paired_gap;
    let adjacent_increases = reports
        .windows(2)
        .filter(|pair| pair[1].median_injected_paired_gap > pair[0].median_injected_paired_gap)
        .count();

    Ok(InjectionRecoverySweepReport {
        schema_version: "rcaf.injection-recovery-sweep/0.2.0",
        effect_size_points: reports.len(),
        endpoint_median_gap_difference: last_gap - first_gap,
        monotonic_adjacent_pair_fraction: adjacent_increases as f64 / (reports.len() - 1) as f64,
        reports,
    })
}

/// Classify a significant two-sided injection result against its known positive direction.
///
/// A significant result is only a correct recovery when positive paired gaps outnumber
/// negative paired gaps. Opposite-direction significance is tracked separately and is never
/// counted as successful directional recovery.
fn classify_expected_positive_injection(
    positive_count: usize,
    negative_count: usize,
    p_two_sided_exact: f64,
    alpha: f64,
) -> (bool, bool) {
    let significant = p_two_sided_exact < alpha;
    (
        significant && positive_count > negative_count,
        significant && negative_count > positive_count,
    )
}

impl InjectionRecoveryReport {
    /// Render a stable JSON representation without a serialization dependency.
    pub fn to_json_pretty(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"seed\": {},\n",
                "  \"participants_per_replicate\": {},\n",
                "  \"regions\": {},\n",
                "  \"samples_per_segment\": {},\n",
                "  \"monte_carlo_replicates\": {},\n",
                "  \"module_start\": {},\n",
                "  \"module_size\": {},\n",
                "  \"global_loading\": {:.12},\n",
                "  \"module_loading\": {:.12},\n",
                "  \"loading_jitter\": {:.12},\n",
                "  \"common_mode_nuisance_loading\": {:.12},\n",
                "  \"estimator\": \"{}\",\n",
                "  \"injection_loading_delta\": {:.12},\n",
                "  \"temporal_autocorrelation\": {:.12},\n",
                "  \"alpha\": {:.12},\n",
                "  \"null_false_positive_rate\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"nuisance_only_rejection_rate\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"injection_detection_power\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"injection_directional_detection_power\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"injection_wrong_direction_rejection_rate\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"median_null_paired_gap\": {:.12},\n",
                "  \"median_nuisance_only_paired_gap\": {:.12},\n",
                "  \"median_injected_paired_gap\": {:.12},\n",
                "  \"median_null_positive_gap_fraction\": {:.12},\n",
                "  \"median_injected_positive_gap_fraction\": {:.12},\n",
                "  \"claim_limits\": [\n",
                "    \"synthetic_factor_model_only\",\n",
                "    \"not_a_biological_effect_estimate\",\n",
                "    \"does_not_validate_a_neuroimaging_preprocessing_pipeline\",\n",
                "    \"does_not_establish_consciousness_or_clinical_validity\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.seed,
            self.participants_per_replicate,
            self.regions,
            self.samples_per_segment,
            self.monte_carlo_replicates,
            self.module_start,
            self.module_size,
            self.global_loading,
            self.module_loading,
            self.loading_jitter,
            self.common_mode_nuisance_loading,
            self.estimator,
            self.injection_loading_delta,
            self.temporal_autocorrelation,
            self.alpha,
            self.null_false_positive_rate.successes,
            self.null_false_positive_rate.trials,
            self.null_false_positive_rate.estimate,
            self.null_false_positive_rate.ci95_lower,
            self.null_false_positive_rate.ci95_upper,
            self.nuisance_only_rejection_rate.successes,
            self.nuisance_only_rejection_rate.trials,
            self.nuisance_only_rejection_rate.estimate,
            self.nuisance_only_rejection_rate.ci95_lower,
            self.nuisance_only_rejection_rate.ci95_upper,
            self.injection_detection_power.successes,
            self.injection_detection_power.trials,
            self.injection_detection_power.estimate,
            self.injection_detection_power.ci95_lower,
            self.injection_detection_power.ci95_upper,
            self.injection_directional_detection_power.successes,
            self.injection_directional_detection_power.trials,
            self.injection_directional_detection_power.estimate,
            self.injection_directional_detection_power.ci95_lower,
            self.injection_directional_detection_power.ci95_upper,
            self.injection_wrong_direction_rejection_rate.successes,
            self.injection_wrong_direction_rejection_rate.trials,
            self.injection_wrong_direction_rejection_rate.estimate,
            self.injection_wrong_direction_rejection_rate.ci95_lower,
            self.injection_wrong_direction_rejection_rate.ci95_upper,
            self.median_null_paired_gap,
            self.median_nuisance_only_paired_gap,
            self.median_injected_paired_gap,
            self.median_null_positive_gap_fraction,
            self.median_injected_positive_gap_fraction,
        )
    }
}

/// Run paired null and known-change injection-recovery experiments.
///
/// For each independent Monte-Carlo replicate, every participant contributes:
/// - two independent samples from the same factor model (within-state repeatability);
/// - one independent same-state sample (null comparison);
/// - one independent sample after a declared module-loading perturbation.
///
/// The no-change comparison exchanges two identically distributed same-state segments.
/// Its false-positive rate therefore checks the metric plus participant-level test under
/// a known null. The injected comparison measures sensitivity to this specific perturbation.
pub fn run_injection_recovery(
    config: &InjectionRecoveryConfig,
) -> Result<InjectionRecoveryReport, MetricError> {
    validate_config(config)?;
    let mut rng = GaussianRng::new(config.seed);
    let mut null_rejections = 0usize;
    let mut nuisance_rejections = 0usize;
    let mut injection_rejections = 0usize;
    let mut directional_injection_rejections = 0usize;
    let mut wrong_direction_rejections = 0usize;
    let mut null_median_gaps = Vec::with_capacity(config.monte_carlo_replicates);
    let mut nuisance_median_gaps = Vec::with_capacity(config.monte_carlo_replicates);
    let mut injected_median_gaps = Vec::with_capacity(config.monte_carlo_replicates);
    let mut null_positive_fractions = Vec::with_capacity(config.monte_carlo_replicates);
    let mut injected_positive_fractions = Vec::with_capacity(config.monte_carlo_replicates);

    for _ in 0..config.monte_carlo_replicates {
        let mut null_gaps = Vec::with_capacity(config.participants);
        let mut nuisance_gaps = Vec::with_capacity(config.participants);
        let mut injected_gaps = Vec::with_capacity(config.participants);

        for _ in 0..config.participants {
            let mut global_loadings = Vec::with_capacity(config.regions);
            let mut module_loadings = Vec::with_capacity(config.regions);
            let mut injected_loadings = Vec::with_capacity(config.regions);

            for region in 0..config.regions {
                let global = config.global_loading + rng.symmetric(config.loading_jitter);
                let base_module = if (config.module_start..config.module_start + config.module_size)
                    .contains(&region)
                {
                    config.module_loading + rng.symmetric(config.loading_jitter)
                } else {
                    0.0
                };
                global_loadings.push(global);
                module_loadings.push(base_module);
                injected_loadings.push(if base_module != 0.0 {
                    base_module + config.injection_loading_delta
                } else {
                    0.0
                });
            }

            let repeat_a = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &module_loadings,
                config.temporal_autocorrelation,
                0.0,
            );
            let repeat_b = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &module_loadings,
                config.temporal_autocorrelation,
                0.0,
            );
            let null_c = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &module_loadings,
                config.temporal_autocorrelation,
                0.0,
            );
            let injected_c = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &injected_loadings,
                config.temporal_autocorrelation,
                0.0,
            );
            // The latent global/module loadings stay fixed. Only an independent shared
            // observation nuisance is added to this repeat. At zero strength, reuse the
            // exact same null sample so the control is deterministic and identical.
            let nuisance_c = if config.common_mode_nuisance_loading == 0.0 {
                null_c.clone()
            } else {
                sample_correlation_matrix(
                    &mut rng,
                    config.samples_per_segment,
                    &global_loadings,
                    &module_loadings,
                    config.temporal_autocorrelation,
                    config.common_mode_nuisance_loading,
                )
            };

            let within_similarity = correlation_similarity(&repeat_a, &repeat_b, config.estimator)?;
            let null_between_similarity =
                correlation_similarity(&repeat_a, &null_c, config.estimator)?;
            let injected_between_similarity =
                correlation_similarity(&repeat_a, &injected_c, config.estimator)?;
            let nuisance_between_similarity =
                correlation_similarity(&repeat_a, &nuisance_c, config.estimator)?;

            null_gaps.push(within_similarity - null_between_similarity);
            nuisance_gaps.push(within_similarity - nuisance_between_similarity);
            injected_gaps.push(within_similarity - injected_between_similarity);
        }

        let null_test = exact_sign_test(&null_gaps)?;
        let nuisance_test = exact_sign_test(&nuisance_gaps)?;
        let injected_test = exact_sign_test(&injected_gaps)?;
        if null_test.p_two_sided_exact < config.alpha {
            null_rejections += 1;
        }
        if nuisance_test.p_two_sided_exact < config.alpha {
            nuisance_rejections += 1;
        }
        if injected_test.p_two_sided_exact < config.alpha {
            injection_rejections += 1;
        }
        let (directionally_detected, wrong_direction_detected) =
            classify_expected_positive_injection(
                injected_test.positive_count,
                injected_test.negative_count,
                injected_test.p_two_sided_exact,
                config.alpha,
            );
        if directionally_detected {
            directional_injection_rejections += 1;
        }
        if wrong_direction_detected {
            wrong_direction_rejections += 1;
        }
        null_positive_fractions.push(null_test.positive_count as f64 / config.participants as f64);
        injected_positive_fractions
            .push(injected_test.positive_count as f64 / config.participants as f64);
        null_median_gaps.push(median(&null_gaps));
        nuisance_median_gaps.push(median(&nuisance_gaps));
        injected_median_gaps.push(median(&injected_gaps));
    }

    Ok(InjectionRecoveryReport {
        schema_version: "rcaf.injection-recovery/0.3.0",
        seed: config.seed,
        participants_per_replicate: config.participants,
        regions: config.regions,
        samples_per_segment: config.samples_per_segment,
        monte_carlo_replicates: config.monte_carlo_replicates,
        module_start: config.module_start,
        module_size: config.module_size,
        global_loading: config.global_loading,
        module_loading: config.module_loading,
        loading_jitter: config.loading_jitter,
        common_mode_nuisance_loading: config.common_mode_nuisance_loading,
        estimator: config.estimator.as_str(),
        injection_loading_delta: config.injection_loading_delta,
        temporal_autocorrelation: config.temporal_autocorrelation,
        alpha: config.alpha,
        null_false_positive_rate: wilson_rate(null_rejections, config.monte_carlo_replicates),
        nuisance_only_rejection_rate: wilson_rate(
            nuisance_rejections,
            config.monte_carlo_replicates,
        ),
        injection_detection_power: wilson_rate(injection_rejections, config.monte_carlo_replicates),
        injection_directional_detection_power: wilson_rate(
            directional_injection_rejections,
            config.monte_carlo_replicates,
        ),
        injection_wrong_direction_rejection_rate: wilson_rate(
            wrong_direction_rejections,
            config.monte_carlo_replicates,
        ),
        median_null_paired_gap: median(&null_median_gaps),
        median_nuisance_only_paired_gap: median(&nuisance_median_gaps),
        median_injected_paired_gap: median(&injected_median_gaps),
        median_null_positive_gap_fraction: median(&null_positive_fractions),
        median_injected_positive_gap_fraction: median(&injected_positive_fractions),
    })
}

fn validate_config(config: &InjectionRecoveryConfig) -> Result<(), MetricError> {
    if config.participants < 8 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "at least eight participants are required",
        });
    }
    if config.regions < 4 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "at least four regions are required",
        });
    }
    if config.samples_per_segment < 30 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "at least 30 time points per segment are required",
        });
    }
    if config.monte_carlo_replicates < 20 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "at least 20 Monte-Carlo replicates are required",
        });
    }
    let Some(module_end) = config.module_start.checked_add(config.module_size) else {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "injected module bounds overflow",
        });
    };
    if config.module_size < 2 || module_end > config.regions {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "injected module must contain at least two in-range regions",
        });
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha >= 0.5 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "alpha must be finite and in (0, 0.5)",
        });
    }
    if !config.injection_loading_delta.is_finite() || config.injection_loading_delta < 0.0 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "injection_loading_delta must be finite and non-negative",
        });
    }
    if !config.loading_jitter.is_finite() || config.loading_jitter < 0.0 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "loading_jitter must be finite and non-negative",
        });
    }
    if !config.common_mode_nuisance_loading.is_finite()
        || config.common_mode_nuisance_loading < 0.0
    {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "common_mode_nuisance_loading must be finite and non-negative",
        });
    }
    if !config.temporal_autocorrelation.is_finite() || config.temporal_autocorrelation.abs() >= 1.0
    {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "temporal_autocorrelation must be finite and strictly between -1 and 1",
        });
    }
    if !config.global_loading.is_finite()
        || !config.module_loading.is_finite()
        || config.global_loading < 0.0
        || config.module_loading < 0.0
    {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "factor loadings must be finite and non-negative",
        });
    }
    let max_global = config.global_loading + config.loading_jitter / 2.0;
    let max_module =
        config.module_loading + config.loading_jitter / 2.0 + config.injection_loading_delta;
    let nuisance_loading = config.common_mode_nuisance_loading;
    if max_global.powi(2) + max_module.powi(2) + nuisance_loading.powi(2) >= 1.0 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "factor loadings must leave positive residual variance",
        });
    }
    Ok(())
}

fn sample_correlation_matrix(
    rng: &mut GaussianRng,
    samples: usize,
    global_loadings: &[f64],
    module_loadings: &[f64],
    temporal_autocorrelation: f64,
    common_mode_nuisance_loading: f64,
) -> Vec<Vec<f64>> {
    let regions = global_loadings.len();
    let mut sums = vec![0.0; regions];
    let mut cross_products = vec![0.0; regions * regions];
    let innovation_scale = (1.0 - temporal_autocorrelation * temporal_autocorrelation).sqrt();
    let mut previous_global = rng.normal();
    let mut previous_module = rng.normal();
    let mut previous_nuisance = if common_mode_nuisance_loading > 0.0 {
        rng.normal()
    } else {
        0.0
    };
    let mut previous_residuals: Vec<f64> = (0..regions).map(|_| rng.normal()).collect();

    for _ in 0..samples {
        previous_global =
            temporal_autocorrelation * previous_global + innovation_scale * rng.normal();
        previous_module =
            temporal_autocorrelation * previous_module + innovation_scale * rng.normal();
        if common_mode_nuisance_loading > 0.0 {
            previous_nuisance = temporal_autocorrelation * previous_nuisance
                + innovation_scale * rng.normal();
        }
        let global_factor = previous_global;
        let module_factor = previous_module;
        let nuisance_factor = previous_nuisance;
        let mut observation = Vec::with_capacity(regions);
        for ((region, (&global, &module)), sum) in global_loadings
            .iter()
            .zip(module_loadings.iter())
            .enumerate()
            .zip(sums.iter_mut())
        {
            let residual = (1.0
                - global * global
                - module * module
                - common_mode_nuisance_loading * common_mode_nuisance_loading)
                .sqrt();
            previous_residuals[region] = temporal_autocorrelation * previous_residuals[region]
                + innovation_scale * rng.normal();
            let value = global * global_factor
                + module * module_factor
                + common_mode_nuisance_loading * nuisance_factor
                + residual * previous_residuals[region];
            observation.push(value);
            *sum += value;
        }
        for (i, &left) in observation.iter().enumerate() {
            for (j, &right) in observation.iter().enumerate() {
                cross_products[i * regions + j] += left * right;
            }
        }
    }

    let sample_count = samples as f64;
    let mut variances = vec![0.0; regions];
    for (region, (variance, &sum)) in variances.iter_mut().zip(sums.iter()).enumerate() {
        let centered = cross_products[region * regions + region] - sum * sum / sample_count;
        *variance = centered / (sample_count - 1.0);
    }

    let mut flat = vec![0.0; regions * regions];
    for i in 0..regions {
        flat[i * regions + i] = 1.0;
        for j in (i + 1)..regions {
            let covariance = (cross_products[i * regions + j] - sums[i] * sums[j] / sample_count)
                / (sample_count - 1.0);
            let correlation = (covariance / (variances[i] * variances[j]).sqrt()).clamp(-1.0, 1.0);
            flat[i * regions + j] = correlation;
            flat[j * regions + i] = correlation;
        }
    }

    flat.chunks(regions).map(|row| row.to_vec()).collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn wilson_rate(successes: usize, trials: usize) -> BinomialRate {
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let estimate = successes as f64 / n;
    let z_squared = Z * Z;
    let denominator = 1.0 + z_squared / n;
    let center = (estimate + z_squared / (2.0 * n)) / denominator;
    let half_width =
        Z * (estimate * (1.0 - estimate) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;
    BinomialRate {
        successes,
        trials,
        estimate,
        ci95_lower: (center - half_width).max(0.0),
        ci95_upper: (center + half_width).min(1.0),
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next() >> 11;
        (mantissa as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }
}

struct GaussianRng {
    uniform: XorShift64,
    spare: Option<f64>,
}

impl GaussianRng {
    fn new(seed: u64) -> Self {
        Self {
            uniform: XorShift64::new(seed),
            spare: None,
        }
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.uniform.uniform_open();
        let u2 = self.uniform.uniform_open();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn symmetric(&mut self, width: f64) -> f64 {
        (self.uniform.uniform_open() - 0.5) * width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> InjectionRecoveryConfig {
        InjectionRecoveryConfig {
            monte_carlo_replicates: 80,
            ..InjectionRecoveryConfig::default()
        }
    }

    #[test]
    fn null_false_positive_rate_is_controlled() {
        let result = run_injection_recovery(&test_config()).unwrap();
        assert!(
            result.null_false_positive_rate.estimate <= 0.15,
            "observed null rejection rate {}",
            result.null_false_positive_rate.estimate
        );
        assert!(
            result.median_null_paired_gap.abs() < 0.025,
            "median null gap {}",
            result.median_null_paired_gap
        );
    }

    #[test]
    fn known_network_injection_is_recovered_in_the_expected_direction() {
        let result = run_injection_recovery(&test_config()).unwrap();
        assert!(
            result.injection_detection_power.estimate >= 0.80,
            "observed two-sided difference detection {}",
            result.injection_detection_power.estimate
        );
        assert!(
            result.injection_directional_detection_power.estimate >= 0.80,
            "observed expected-direction recovery {}",
            result.injection_directional_detection_power.estimate
        );
        assert_eq!(
            result.injection_detection_power.successes,
            result.injection_directional_detection_power.successes
                + result.injection_wrong_direction_rejection_rate.successes,
            "two-sided detections must partition into expected- and wrong-direction results"
        );
        assert_eq!(
            result.injection_wrong_direction_rejection_rate.successes,
            0,
            "strong positive loading injection produced wrong-direction rejection"
        );
        let json = result.to_json_pretty();
        assert!(json.contains("\"injection_directional_detection_power\""));
        assert!(json.contains("\"injection_wrong_direction_rejection_rate\""));
        assert!(
            result.median_injected_paired_gap > result.median_null_paired_gap + 0.035,
            "null gap {} vs injected gap {}",
            result.median_null_paired_gap,
            result.median_injected_paired_gap
        );
    }

    #[test]
    fn opposite_direction_significance_is_never_counted_as_successful_recovery() {
        let alpha = 0.05;
        let n = 23usize;
        let positive_test = exact_sign_test(&vec![1.0; n]).unwrap();
        let negative_test = exact_sign_test(&vec![-1.0; n]).unwrap();
        let non_significant_test = exact_sign_test(&[1.0; 12][..]).unwrap();

        let (correct, wrong) = classify_expected_positive_injection(
            positive_test.positive_count,
            positive_test.negative_count,
            positive_test.p_two_sided_exact,
            alpha,
        );
        assert!(correct);
        assert!(!wrong);

        let (correct, wrong) = classify_expected_positive_injection(
            negative_test.positive_count,
            negative_test.negative_count,
            negative_test.p_two_sided_exact,
            alpha,
        );
        assert!(!correct, "opposite-direction rejection must not count as recovery");
        assert!(wrong, "opposite-direction rejection must be recorded separately");

        let (correct, wrong) = classify_expected_positive_injection(
            non_significant_test.positive_count,
            non_significant_test.negative_count,
            non_significant_test.p_two_sided_exact,
            alpha,
        );
        assert!(!correct, "non-significant positive imbalance is not recovery");
        assert!(!wrong);
    }

    #[test]
    fn zero_strength_common_mode_nuisance_reuses_the_exact_null_control() {
        let mut config = test_config();
        config.monte_carlo_replicates = 30;
        config.common_mode_nuisance_loading = 0.0;
        let report = run_injection_recovery(&config).unwrap();
        assert_eq!(
            report.nuisance_only_rejection_rate,
            report.null_false_positive_rate
        );
        assert_eq!(
            report.median_nuisance_only_paired_gap,
            report.median_null_paired_gap
        );
    }

    #[test]
    fn common_mode_nuisance_is_reported_separately_from_latent_injection() {
        let report = run_injection_recovery(&test_config()).unwrap();
        assert_eq!(report.common_mode_nuisance_loading, 0.20);
        assert_eq!(report.nuisance_only_rejection_rate.trials, report.monte_carlo_replicates);
        assert!(report.median_nuisance_only_paired_gap.is_finite());
        let json = report.to_json_pretty();
        assert!(json.contains("\"nuisance_only_rejection_rate\""));
        assert!(json.contains("\"common_mode_nuisance_loading\""));
    }

    #[test]
    fn seed_reproduces_report_exactly() {
        let mut config = test_config();
        config.monte_carlo_replicates = 20;
        let left = run_injection_recovery(&config).unwrap();
        let right = run_injection_recovery(&config).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn invalid_factor_model_fails_closed() {
        let mut config = test_config();
        config.injection_loading_delta = 1.2;
        assert!(matches!(
            run_injection_recovery(&config),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
    }

    #[test]
    fn invalid_temporal_autocorrelation_fails_closed() {
        let mut config = test_config();
        config.temporal_autocorrelation = 1.0;
        assert!(matches!(
            run_injection_recovery(&config),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
        config.temporal_autocorrelation = f64::NAN;
        assert!(matches!(
            run_injection_recovery(&config),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
    }

    #[test]
    fn null_calibration_survives_stronger_temporal_dependence() {
        let mut config = test_config();
        config.temporal_autocorrelation = 0.65;
        let result = run_injection_recovery(&config).unwrap();
        assert!(
            result.null_false_positive_rate.estimate <= 0.15,
            "AR(1) null rejection rate {}",
            result.null_false_positive_rate.estimate
        );
        assert!(
            result.median_null_paired_gap.abs() < 0.04,
            "AR(1) median null gap {}",
            result.median_null_paired_gap
        );
    }

    #[test]
    fn dose_response_sweep_uses_sorted_effect_sizes_and_increases_at_strong_endpoint() {
        let mut config = test_config();
        config.monte_carlo_replicates = 30;
        let sweep = run_injection_recovery_sweep(&config, &[0.0, 0.2, 0.4]).unwrap();
        assert_eq!(sweep.effect_size_points, 3);
        assert!(sweep.endpoint_median_gap_difference > 0.04);
        assert!(
            sweep.reports[2].median_injected_paired_gap
                > sweep.reports[0].median_injected_paired_gap
        );
        assert!(sweep.monotonic_adjacent_pair_fraction >= 0.5);
    }

    #[test]
    fn dose_response_rejects_unsorted_or_repeated_effect_sizes() {
        let config = test_config();
        assert!(matches!(
            run_injection_recovery_sweep(&config, &[0.2, 0.1]),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
        assert!(matches!(
            run_injection_recovery_sweep(&config, &[0.1, 0.1]),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
    }

    #[test]
    fn estimator_tournament_uses_same_synthetic_seed_and_reports_three_metrics() {
        let mut config = test_config();
        config.monte_carlo_replicates = 25;
        let tournament = run_estimator_tournament(&config).unwrap();
        assert_eq!(tournament.estimators_compared, 3);
        assert_eq!(tournament.results[0].seed, tournament.results[1].seed);
        assert_eq!(tournament.results[1].seed, tournament.results[2].seed);
        assert_eq!(
            tournament
                .results
                .iter()
                .map(|r| r.estimator)
                .collect::<Vec<_>>(),
            vec![
                "absolute_weighted_jaccard",
                "signed_shift_weighted_jaccard",
                "normalized_mean_absolute_edge_error",
            ]
        );
    }

    #[test]
    fn estimator_dose_response_tournament_uses_common_seed_and_grid() {
        let config = InjectionRecoveryConfig {
            monte_carlo_replicates: 20,
            ..test_config()
        };
        let effects = [0.0, 0.1, 0.2, 0.3];
        let tournament = run_estimator_dose_response_tournament(&config, &effects).unwrap();
        assert_eq!(tournament.estimators_compared, 3);
        assert_eq!(tournament.effect_sizes, effects);
        assert_eq!(
            tournament
                .curves
                .iter()
                .map(|curve| curve.estimator)
                .collect::<Vec<_>>(),
            vec![
                "absolute_weighted_jaccard",
                "signed_shift_weighted_jaccard",
                "normalized_mean_absolute_edge_error",
            ]
        );
        for curve in &tournament.curves {
            assert_eq!(
                curve
                    .sweep
                    .reports
                    .iter()
                    .map(|row| row.seed)
                    .collect::<Vec<_>>(),
                vec![config.seed; effects.len()]
            );
            assert_eq!(
                curve
                    .sweep
                    .reports
                    .iter()
                    .map(|row| row.injection_loading_delta)
                    .collect::<Vec<_>>(),
                effects
            );
        }
        let output = tournament.to_json_pretty();
        assert!(output.contains("rcaf.estimator-dose-response-tournament/0.2.0"));
        assert!(output.contains("signed_shift_weighted_jaccard"));
    }

    #[test]
    fn estimator_dose_response_tournament_rejects_bad_effect_grid() {
        let config = test_config();
        assert!(matches!(
            run_estimator_dose_response_tournament(&config, &[0.2, 0.1]),
            Err(MetricError::InvalidSimulationConfig { .. })
        ));
    }

    #[test]
    fn matched_estimators_represent_polarity_differences_differently() {
        let left = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let right = vec![vec![1.0, -0.5], vec![-0.5, 1.0]];
        let absolute =
            correlation_similarity(&left, &right, SimilarityEstimator::AbsoluteWeightedJaccard)
                .unwrap();
        let signed = correlation_similarity(
            &left,
            &right,
            SimilarityEstimator::SignedShiftWeightedJaccard,
        )
        .unwrap();
        let mae = correlation_similarity(
            &left,
            &right,
            SimilarityEstimator::NormalizedMeanAbsoluteEdgeError,
        )
        .unwrap();
        assert!((absolute - 1.0).abs() < 1e-12);
        assert!((signed - (1.0 / 3.0)).abs() < 1e-12);
        assert!((mae - 0.5).abs() < 1e-12);
    }
}
