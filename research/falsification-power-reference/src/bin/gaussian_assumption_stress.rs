//! Assumption-stress campaign for the fixed independent-Gaussian terminal test.
//!
//! The IID Gaussian condition is the sole in-contract positive control.
//! AR(1), Student-t, heteroscedastic, variance-regime and contamination
//! conditions are explicitly out of contract for this test. Their empirical
//! size/power is measured as an assumption-failure surface, not graded as
//! successful calibration or used to extend the theorem.

use falsification_power_reference::gaussian_terminal_induction::GaussianTerminalInducedTest;
use std::error::Error;
use std::f64::consts::TAU;

const ROOT_SEEDS: [u64; 5] = [
    0x31c7_1b2a_9001_0001,
    0x31c7_1b2a_9001_0002,
    0x31c7_1b2a_9001_0003,
    0x31c7_1b2a_9001_0004,
    0x31c7_1b2a_9001_0005,
];
const PATHS_PER_SEED: u64 = 10_000;
const HORIZON: usize = 80;
const ALPHA: f64 = 0.05;
const DISCREPANCY: f64 = 0.22;
const SIGMA: f64 = 0.80;
const STUDENT_DF: f64 = 5.0;
const AR1_RHO: f64 = 0.65;
const CONTAMINATION_PROBABILITY: f64 = 0.02;
const CONTAMINATION_BASE_VARIANCE: f64 = 0.25;
const CONTAMINATION_OUTLIER_VARIANCE: f64 = 37.75;
const PATH_SEED_MULTIPLIER: u64 = 0xd6e8_feb8_6659_fd93;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoiseRegime {
    MatchedIidGaussian,
    StationaryAr1,
    StandardizedStudentT5,
    SmoothHeteroscedastic,
    AbruptVarianceShift,
    SparseContamination,
}

impl NoiseRegime {
    const ALL: [Self; 6] = [
        Self::MatchedIidGaussian,
        Self::StationaryAr1,
        Self::StandardizedStudentT5,
        Self::SmoothHeteroscedastic,
        Self::AbruptVarianceShift,
        Self::SparseContamination,
    ];

    fn id(self) -> &'static str {
        match self {
            Self::MatchedIidGaussian => "matched_iid_gaussian",
            Self::StationaryAr1 => "stationary_ar1_rho_0_65",
            Self::StandardizedStudentT5 => "standardized_student_t5",
            Self::SmoothHeteroscedastic => "smooth_heteroscedastic_ramp",
            Self::AbruptVarianceShift => "abrupt_variance_regime_shift",
            Self::SparseContamination => "sparse_two_percent_contamination",
        }
    }

    fn assumption_status(self) -> &'static str {
        match self {
            Self::MatchedIidGaussian => "IN_CONTRACT",
            _ => "OUT_OF_CONTRACT",
        }
    }

    fn expected_lag_one(self) -> f64 {
        match self {
            Self::StationaryAr1 => AR1_RHO,
            _ => 0.0,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::MatchedIidGaussian => "Independent Gaussian residuals with known constant sigma.",
            Self::StationaryAr1 => {
                "Stationary Gaussian AR(1) residuals with known generator rho but a test that assumes independence."
            }
            Self::StandardizedStudentT5 => {
                "Independent Student-t(5) residuals scaled to unit variance."
            }
            Self::SmoothHeteroscedastic => {
                "Independent Gaussian errors with a smooth variance ramp and mean variance sigma squared."
            }
            Self::AbruptVarianceShift => {
                "Independent Gaussian errors with a low-variance first regime and high-variance second regime; average variance is sigma squared."
            }
            Self::SparseContamination => {
                "Independent Gaussian scale mixture with two-percent gross-error component and unit marginal variance."
            }
        }
    }

    fn variance_factor(self, look: usize) -> f64 {
        match self {
            Self::SmoothHeteroscedastic => 0.5 + (look as f64 + 0.5) / HORIZON as f64,
            Self::AbruptVarianceShift => {
                if look < 32 {
                    0.25
                } else {
                    1.5
                }
            }
            _ => 1.0,
        }
    }
}

/// SplitMix64 / Box-Muller stream for deterministic simulation only. This is
/// neither cryptographic nor suitable for real experiment randomization.
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let u1 = self.uniform().max(0.5 / 9_007_199_254_740_992.0);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = TAU * u2;
        self.spare_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SufficientMoments {
    count: u64,
    sum: f64,
    sum_squared: f64,
    pair_count: u64,
    previous_sum: f64,
    current_sum: f64,
    previous_squared: f64,
    current_squared: f64,
    lag_product: f64,
}

impl SufficientMoments {
    fn add_path(&mut self, residuals: &[f64]) {
        self.count += residuals.len() as u64;
        self.sum += residuals.iter().sum::<f64>();
        self.sum_squared += residuals.iter().map(|value| value * value).sum::<f64>();
        for pair in residuals.windows(2) {
            self.pair_count += 1;
            self.previous_sum += pair[0];
            self.current_sum += pair[1];
            self.previous_squared += pair[0] * pair[0];
            self.current_squared += pair[1] * pair[1];
            self.lag_product += pair[0] * pair[1];
        }
    }

    fn mean(self) -> f64 {
        self.sum / self.count.max(1) as f64
    }

    fn variance(self) -> f64 {
        let n = self.count.max(1) as f64;
        (self.sum_squared - self.sum * self.sum / n) / n
    }

    fn lag_one_correlation(self) -> f64 {
        let n = self.pair_count.max(1) as f64;
        let covariance = self.lag_product - self.previous_sum * self.current_sum / n;
        let previous_variance = self.previous_squared - self.previous_sum * self.previous_sum / n;
        let current_variance = self.current_squared - self.current_sum * self.current_sum / n;
        let denominator = (previous_variance * current_variance).sqrt();
        if denominator <= 0.0 {
            0.0
        } else {
            covariance / denominator
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RejectionCounts {
    paths: u64,
    rejections: u64,
}

impl RejectionCounts {
    fn rate(self) -> f64 {
        self.rejections as f64 / self.paths.max(1) as f64
    }
}

#[derive(Debug, Clone)]
struct SeedResult {
    root_seed: u64,
    null: RejectionCounts,
    alternative: RejectionCounts,
    null_moments: SufficientMoments,
}

#[derive(Debug, Clone)]
struct RegimeResult {
    regime: NoiseRegime,
    seeds: Vec<SeedResult>,
}

fn wilson_95(successes: u64, trials: u64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_Z_95 * WILSON_Z_95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = WILSON_Z_95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn generate_residuals(regime: NoiseRegime, rng: &mut SplitMix64) -> Vec<f64> {
    let mut residuals = Vec::with_capacity(HORIZON);
    match regime {
        NoiseRegime::MatchedIidGaussian => {
            for _ in 0..HORIZON {
                residuals.push(SIGMA * rng.standard_normal());
            }
        }
        NoiseRegime::StationaryAr1 => {
            let innovation_scale = (1.0 - AR1_RHO * AR1_RHO).sqrt() * SIGMA;
            let mut previous = SIGMA * rng.standard_normal();
            residuals.push(previous);
            for _ in 1..HORIZON {
                previous = AR1_RHO * previous + innovation_scale * rng.standard_normal();
                residuals.push(previous);
            }
        }
        NoiseRegime::StandardizedStudentT5 => {
            let variance_correction = ((STUDENT_DF - 2.0) / STUDENT_DF).sqrt();
            for _ in 0..HORIZON {
                let normal = rng.standard_normal();
                let chi_squared = (0..STUDENT_DF as usize)
                    .map(|_| {
                        let draw = rng.standard_normal();
                        draw * draw
                    })
                    .sum::<f64>();
                let t_draw = normal / (chi_squared / STUDENT_DF).sqrt();
                residuals.push(SIGMA * variance_correction * t_draw);
            }
        }
        NoiseRegime::SmoothHeteroscedastic | NoiseRegime::AbruptVarianceShift => {
            for look in 0..HORIZON {
                let scale = SIGMA * regime.variance_factor(look).sqrt();
                residuals.push(scale * rng.standard_normal());
            }
        }
        NoiseRegime::SparseContamination => {
            let base_scale = SIGMA * CONTAMINATION_BASE_VARIANCE.sqrt();
            let outlier_scale = SIGMA * CONTAMINATION_OUTLIER_VARIANCE.sqrt();
            for _ in 0..HORIZON {
                let scale = if rng.uniform() < CONTAMINATION_PROBABILITY {
                    outlier_scale
                } else {
                    base_scale
                };
                residuals.push(scale * rng.standard_normal());
            }
        }
    }
    residuals
}

fn run_seed(
    regime: NoiseRegime,
    root_seed: u64,
    test: &GaussianTerminalInducedTest,
) -> Result<SeedResult, Box<dyn Error>> {
    let mut null = RejectionCounts::default();
    let mut alternative = RejectionCounts::default();
    let mut null_moments = SufficientMoments::default();
    let regime_salt = (NoiseRegime::ALL
        .iter()
        .position(|candidate| *candidate == regime)
        .unwrap_or(0) as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);

    for path in 0..PATHS_PER_SEED {
        // Null and alternative share the same residual realization. The
        // alternative adds only the frozen discrepancy after generation.
        let seed = root_seed ^ regime_salt ^ path.wrapping_mul(PATH_SEED_MULTIPLIER);
        let mut rng = SplitMix64::new(seed);
        let residuals = generate_residuals(regime, &mut rng);
        null_moments.add_path(&residuals);
        let alternative_observations = residuals
            .iter()
            .map(|residual| residual + DISCREPANCY)
            .collect::<Vec<_>>();

        let null_terminal = test.evaluate_prefix(&residuals)?;
        let alternative_terminal = test.evaluate_prefix(&alternative_observations)?;

        null.paths += 1;
        alternative.paths += 1;
        if null_terminal.conditional_rejection_probability >= 0.5 {
            null.rejections += 1;
        }
        if alternative_terminal.conditional_rejection_probability >= 0.5 {
            alternative.rejections += 1;
        }
    }

    Ok(SeedResult {
        root_seed,
        null,
        alternative,
        null_moments,
    })
}

fn run_regime(
    regime: NoiseRegime,
    test: &GaussianTerminalInducedTest,
) -> Result<RegimeResult, Box<dyn Error>> {
    let mut seeds = Vec::with_capacity(ROOT_SEEDS.len());
    for root_seed in ROOT_SEEDS {
        seeds.push(run_seed(regime, root_seed, test)?);
    }
    Ok(RegimeResult { regime, seeds })
}

fn aggregate_counts(results: &[SeedResult], alternative: bool) -> RejectionCounts {
    let mut pooled = RejectionCounts::default();
    for result in results {
        let counts = if alternative {
            result.alternative
        } else {
            result.null
        };
        pooled.paths += counts.paths;
        pooled.rejections += counts.rejections;
    }
    pooled
}

fn aggregate_moments(results: &[SeedResult]) -> SufficientMoments {
    let mut pooled = SufficientMoments::default();
    for result in results {
        let m = result.null_moments;
        pooled.count += m.count;
        pooled.sum += m.sum;
        pooled.sum_squared += m.sum_squared;
        pooled.pair_count += m.pair_count;
        pooled.previous_sum += m.previous_sum;
        pooled.current_sum += m.current_sum;
        pooled.previous_squared += m.previous_squared;
        pooled.current_squared += m.current_squared;
        pooled.lag_product += m.lag_product;
    }
    pooled
}

fn counts_json(counts: RejectionCounts) -> String {
    let (low, high) = wilson_95(counts.rejections, counts.paths);
    format!(
        concat!(
            "{{",
            "\"paths\":{},",
            "\"rejections\":{},",
            "\"rate\":{:.8},",
            "\"wilson95\":[{:.8},{:.8}]",
            "}}"
        ),
        counts.paths,
        counts.rejections,
        counts.rate(),
        low,
        high
    )
}

fn seed_json(result: &SeedResult) -> String {
    format!(
        concat!(
            "{{",
            "\"root_seed\":\"0x{:016x}\",",
            "\"null\":{},",
            "\"alternative\":{},",
            "\"null_mean\":{:.8},",
            "\"null_variance\":{:.8},",
            "\"null_lag_one_correlation\":{:.8}",
            "}}"
        ),
        result.root_seed,
        counts_json(result.null),
        counts_json(result.alternative),
        result.null_moments.mean(),
        result.null_moments.variance(),
        result.null_moments.lag_one_correlation()
    )
}

fn regime_json(result: &RegimeResult, analytic_power: f64) -> String {
    let null = aggregate_counts(&result.seeds, false);
    let alternative = aggregate_counts(&result.seeds, true);
    let moments = aggregate_moments(&result.seeds);
    let per_seed = result
        .seeds
        .iter()
        .map(seed_json)
        .collect::<Vec<_>>()
        .join(",");
    let null_rate = null.rate();
    format!(
        concat!(
            "{{",
            "\"regime\":\"{}\",",
            "\"assumption_status\":\"{}\",",
            "\"description\":\"{}\",",
            "\"expected_lag_one_correlation\":{:.8},",
            "\"null\":{},",
            "\"alternative\":{},",
            "\"analytic_iid_terminal_power\":{:.8},",
            "\"null_false_alarm_excess_over_alpha\":{:.8},",
            "\"pooled_null_mean\":{:.8},",
            "\"pooled_null_variance\":{:.8},",
            "\"pooled_null_lag_one_correlation\":{:.8},",
            "\"per_seed\":[{}]",
            "}}"
        ),
        result.regime.id(),
        result.regime.assumption_status(),
        result.regime.description(),
        result.regime.expected_lag_one(),
        counts_json(null),
        counts_json(alternative),
        analytic_power,
        null_rate - ALPHA,
        moments.mean(),
        moments.variance(),
        moments.lag_one_correlation(),
        per_seed
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let test =
        GaussianTerminalInducedTest::new(ALPHA, vec![DISCREPANCY; HORIZON], vec![SIGMA; HORIZON])?;
    let unique_seeds = ROOT_SEEDS
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique_seeds.len() != ROOT_SEEDS.len() {
        return Err("root seed list contains duplicates".into());
    }

    let mut regimes = Vec::with_capacity(NoiseRegime::ALL.len());
    for regime in NoiseRegime::ALL {
        regimes.push(run_regime(regime, &test)?);
    }
    let regime_rows = regimes
        .iter()
        .map(|result| regime_json(result, test.terminal_power()))
        .collect::<Vec<_>>()
        .join(",");
    let roots = ROOT_SEEDS
        .iter()
        .map(|seed| format!("\"0x{seed:016x}\""))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        concat!(
            "{{",
            "\"schema\":\"gaussian_assumption_stress_v0\",",
            "\"prng\":\"SplitMix64_BoxMuller_v1\",",
            "\"root_seed_count\":{},",
            "\"root_seeds\":[{}],",
            "\"paths_per_seed\":{},",
            "\"paths_per_regime_scenario\":{},",
            "\"horizon\":{},",
            "\"nominal_alpha\":{:.8},",
            "\"discrepancy\":{:.8},",
            "\"design_sigma\":{:.8},",
            "\"analytic_iid_terminal_power\":{:.8},",
            "\"claim\":\"fixed terminal iid-Gaussian assumption-stress map only; out-of-contract cells do not inherit its theorem\",",
            "\"regimes\":[{}]",
            "}}"
        ),
        ROOT_SEEDS.len(),
        roots,
        PATHS_PER_SEED,
        PATHS_PER_SEED * ROOT_SEEDS.len() as u64,
        HORIZON,
        ALPHA,
        DISCREPANCY,
        SIGMA,
        test.terminal_power(),
        regime_rows
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_matched_iid_gaussian_is_inside_this_test_contract() {
        assert_eq!(
            NoiseRegime::MatchedIidGaussian.assumption_status(),
            "IN_CONTRACT"
        );
        for regime in NoiseRegime::ALL.into_iter().skip(1) {
            assert_eq!(regime.assumption_status(), "OUT_OF_CONTRACT");
        }
    }

    #[test]
    fn heteroscedastic_variance_ramp_has_unit_average_variance_factor() {
        let average = (0..HORIZON)
            .map(|look| NoiseRegime::SmoothHeteroscedastic.variance_factor(look))
            .sum::<f64>()
            / HORIZON as f64;
        assert!((average - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn abrupt_variance_regimes_have_unit_average_variance_factor() {
        let average = (0..HORIZON)
            .map(|look| NoiseRegime::AbruptVarianceShift.variance_factor(look))
            .sum::<f64>()
            / HORIZON as f64;
        assert!((average - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn contamination_mixture_has_unit_marginal_variance_by_construction() {
        let p = CONTAMINATION_PROBABILITY;
        let variance = (1.0 - p) * CONTAMINATION_BASE_VARIANCE + p * CONTAMINATION_OUTLIER_VARIANCE;
        assert!((variance - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn student_t5_scale_is_unit_variance() {
        let df = STUDENT_DF;
        let correction_squared = (df - 2.0) / df;
        let t_variance = df / (df - 2.0);
        assert!((correction_squared * t_variance - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn ar1_stationary_innovation_scale_preserves_marginal_variance() {
        let innovation_variance_factor = 1.0 - AR1_RHO * AR1_RHO;
        let stationary_variance = AR1_RHO * AR1_RHO + innovation_variance_factor;
        assert!((stationary_variance - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn splitmix_stream_replays_exactly() {
        let mut first = SplitMix64::new(ROOT_SEEDS[0]);
        let mut second = SplitMix64::new(ROOT_SEEDS[0]);
        let mut other = SplitMix64::new(ROOT_SEEDS[1]);
        for _ in 0..32 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        assert_ne!(second.next_u64(), other.next_u64());
    }

    #[test]
    fn fixed_iid_reference_has_expected_analytic_power() {
        let test = GaussianTerminalInducedTest::new(
            ALPHA,
            vec![DISCREPANCY; HORIZON],
            vec![SIGMA; HORIZON],
        )
        .unwrap();
        assert!((test.total_information() - 6.05).abs() < 1.0e-12);
        assert!((test.terminal_power() - 0.792_412_72).abs() < 1.0e-7);
    }
}
