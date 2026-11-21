//! Synthetic null and injection-recovery calibration for RCAF.
//!
//! The generator samples valid correlation matrices from a Gaussian factor model.
//! It is an implementation-calibration harness, not a biological simulator.

use super::{exact_sign_test, weighted_jaccard_correlation_matrices, CorrelationWjMode, MetricError};

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionRecoveryConfig {
    pub participants: usize,
    pub regions: usize,
    pub samples_per_segment: usize,
    pub monte_carlo_replicates: usize,
    pub module_start: usize,
    pub module_size: usize,
    pub global_loading: f64,
    pub module_loading: f64,
    pub loading_jitter: f64,
    pub injection_loading_delta: f64,
    pub alpha: f64,
    pub seed: u64,
}

impl Default for InjectionRecoveryConfig {
    fn default() -> Self {
        Self {
            participants: 24,
            regions: 12,
            samples_per_segment: 160,
            monte_carlo_replicates: 100,
            module_start: 4,
            module_size: 5,
            global_loading: 0.20,
            module_loading: 0.18,
            loading_jitter: 0.06,
            injection_loading_delta: 0.40,
            alpha: 0.05,
            seed: 0xC37B_2026_11_21,
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
    pub injection_loading_delta: f64,
    pub alpha: f64,
    /// Type-I error estimate under exchangeable no-change matrices.
    pub null_false_positive_rate: BinomialRate,
    /// Detection rate under the declared factor-loading injection.
    pub injection_detection_power: BinomialRate,
    /// Median across Monte Carlo replicates of participant-level median paired gaps.
    pub median_null_paired_gap: f64,
    pub median_injected_paired_gap: f64,
    /// Median across replicates of the proportion of participants with positive gaps.
    pub median_null_positive_gap_fraction: f64,
    pub median_injected_positive_gap_fraction: f64,
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
                "  \"injection_loading_delta\": {:.12},\n",
                "  \"alpha\": {:.12},\n",
                "  \"null_false_positive_rate\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"injection_detection_power\": {{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"ci95_lower\": {:.12}, \"ci95_upper\": {:.12}}},\n",
                "  \"median_null_paired_gap\": {:.12},\n",
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
            self.injection_loading_delta,
            self.alpha,
            self.null_false_positive_rate.successes,
            self.null_false_positive_rate.trials,
            self.null_false_positive_rate.estimate,
            self.null_false_positive_rate.ci95_lower,
            self.null_false_positive_rate.ci95_upper,
            self.injection_detection_power.successes,
            self.injection_detection_power.trials,
            self.injection_detection_power.estimate,
            self.injection_detection_power.ci95_lower,
            self.injection_detection_power.ci95_upper,
            self.median_null_paired_gap,
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
    let mut injection_rejections = 0usize;
    let mut null_median_gaps = Vec::with_capacity(config.monte_carlo_replicates);
    let mut injected_median_gaps = Vec::with_capacity(config.monte_carlo_replicates);
    let mut null_positive_fractions = Vec::with_capacity(config.monte_carlo_replicates);
    let mut injected_positive_fractions = Vec::with_capacity(config.monte_carlo_replicates);

    for _ in 0..config.monte_carlo_replicates {
        let mut null_gaps = Vec::with_capacity(config.participants);
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
            );
            let repeat_b = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &module_loadings,
            );
            let null_c = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &module_loadings,
            );
            let injected_c = sample_correlation_matrix(
                &mut rng,
                config.samples_per_segment,
                &global_loadings,
                &injected_loadings,
            );

            let within_similarity = weighted_jaccard_correlation_matrices(
                &repeat_a,
                &repeat_b,
                CorrelationWjMode::Absolute,
                1,
            )?;
            let null_between_similarity = weighted_jaccard_correlation_matrices(
                &repeat_a,
                &null_c,
                CorrelationWjMode::Absolute,
                1,
            )?;
            let injected_between_similarity = weighted_jaccard_correlation_matrices(
                &repeat_a,
                &injected_c,
                CorrelationWjMode::Absolute,
                1,
            )?;

            null_gaps.push(within_similarity - null_between_similarity);
            injected_gaps.push(within_similarity - injected_between_similarity);
        }

        let null_test = exact_sign_test(&null_gaps)?;
        let injected_test = exact_sign_test(&injected_gaps)?;
        if null_test.p_two_sided_exact < config.alpha {
            null_rejections += 1;
        }
        if injected_test.p_two_sided_exact < config.alpha {
            injection_rejections += 1;
        }
        null_positive_fractions.push(
            null_test.positive_count as f64 / config.participants as f64,
        );
        injected_positive_fractions.push(
            injected_test.positive_count as f64 / config.participants as f64,
        );
        null_median_gaps.push(median(&null_gaps));
        injected_median_gaps.push(median(&injected_gaps));
    }

    Ok(InjectionRecoveryReport {
        schema_version: "rcaf.injection-recovery/0.1.0",
        seed: config.seed,
        participants_per_replicate: config.participants,
        regions: config.regions,
        samples_per_segment: config.samples_per_segment,
        monte_carlo_replicates: config.monte_carlo_replicates,
        injection_loading_delta: config.injection_loading_delta,
        alpha: config.alpha,
        null_false_positive_rate: wilson_rate(
            null_rejections,
            config.monte_carlo_replicates,
        ),
        injection_detection_power: wilson_rate(
            injection_rejections,
            config.monte_carlo_replicates,
        ),
        median_null_paired_gap: median(&null_median_gaps),
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
    if config.module_size < 2
        || config.module_start + config.module_size > config.regions
    {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "injected module must contain at least two in-range regions",
        });
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha >= 0.5 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "alpha must be finite and in (0, 0.5)",
        });
    }
    if !config.injection_loading_delta.is_finite() || config.injection_loading_delta <= 0.0 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "injection_loading_delta must be finite and positive",
        });
    }
    if !config.loading_jitter.is_finite() || config.loading_jitter < 0.0 {
        return Err(MetricError::InvalidSimulationConfig {
            reason: "loading_jitter must be finite and non-negative",
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
    if max_global.powi(2) + max_module.powi(2) >= 1.0 {
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
) -> Vec<Vec<f64>> {
    let regions = global_loadings.len();
    let mut sums = vec![0.0; regions];
    let mut cross_products = vec![0.0; regions * regions];

    for _ in 0..samples {
        let global_factor = rng.normal();
        let module_factor = rng.normal();
        let mut observation = vec![0.0; regions];
        for region in 0..regions {
            let global = global_loadings[region];
            let module = module_loadings[region];
            let residual = (1.0 - global * global - module * module).sqrt();
            observation[region] = global * global_factor
                + module * module_factor
                + residual * rng.normal();
            sums[region] += observation[region];
        }
        for i in 0..regions {
            for j in 0..regions {
                cross_products[i * regions + j] += observation[i] * observation[j];
            }
        }
    }

    let sample_count = samples as f64;
    let mut variances = vec![0.0; regions];
    for region in 0..regions {
        let centered = cross_products[region * regions + region]
            - sums[region] * sums[region] / sample_count;
        variances[region] = centered / (sample_count - 1.0);
    }

    let mut flat = vec![0.0; regions * regions];
    for i in 0..regions {
        flat[i * regions + i] = 1.0;
        for j in (i + 1)..regions {
            let covariance = (cross_products[i * regions + j]
                - sums[i] * sums[j] / sample_count)
                / (sample_count - 1.0);
            let correlation = (covariance / (variances[i] * variances[j]).sqrt())
                .clamp(-1.0, 1.0);
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
    let half_width = Z
        * (estimate * (1.0 - estimate) / n + z_squared / (4.0 * n * n)).sqrt()
        / denominator;
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
    fn known_network_injection_is_recovered() {
        let result = run_injection_recovery(&test_config()).unwrap();
        assert!(
            result.injection_detection_power.estimate >= 0.80,
            "observed injection recovery {}",
            result.injection_detection_power.estimate
        );
        assert!(
            result.median_injected_paired_gap > result.median_null_paired_gap + 0.035,
            "null gap {} vs injected gap {}",
            result.median_null_paired_gap,
            result.median_injected_paired_gap
        );
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
}
