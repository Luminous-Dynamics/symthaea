//! Confirmatory 50,000-trial stress for identical-marginal dependence-regime shifts under Gaussian and standardized Student-t(5) marginals.
//!
//! A Gaussian AR(1) process with innovation scale sqrt(1-rho^2) has
//! N(0, 1) marginals for every |rho| < 1. Changing rho therefore preserves
//! every one-step marginal while changing the within-block distribution of
//! max-absolute-residual scores. This isolates dependence-regime shift from
//! ordinary mean/variance drift.
//!
//! Run:
//!   rustc --edition=2024 scripts/research/sdcb_student_t5_common_scale_stress.rs -o /tmp/sdcb-t5-regime
//!   /tmp/sdcb-t5-regime
//!   rustc --edition=2024 --test scripts/research/sdcb_student_t5_common_scale_stress.rs -o /tmp/sdcb-t5-regime-tests
//!   /tmp/sdcb-t5-regime-tests
//!
//! Diagnostic simulation only. It does not prove a universal coverage guarantee
//! or establish a production dependence detector.

const CALIBRATION_RHO_GRID: [f64; 2] = [0.05, 0.95];
const TEST_RHO_GRID: [f64; 2] = [0.05, 0.95];
const BLOCK_LENGTH_GRID: [usize; 3] = [1, 8, 32];
const SERIAL_EMBARGO_GRID: [usize; 2] = [0, 64];
const CALIBRATION_BLOCKS: usize = 39;
const ALPHA: f64 = 0.10;
const TRIALS: u64 = 50_000;
const TEST_TRIALS: u64 = 5_000;
const WILSON_Z: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform_open();
        radius * angle.cos()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarginalFamily {
    Gaussian,
    StandardizedStudentT5CommonScale,
}

impl MarginalFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::StandardizedStudentT5CommonScale => "student_t5_common_scale",
        }
    }
}

#[derive(Clone, Debug)]
struct Ar1Stream {
    rng: SplitMix64,
    state: f64,
    marginal_scale: f64,
}

impl Ar1Stream {
    fn new(seed: u64, marginal: MarginalFamily) -> Self {
        let mut initial_rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
        let marginal_scale = match marginal {
            MarginalFamily::Gaussian => 1.0,
            MarginalFamily::StandardizedStudentT5CommonScale => {
                // T_5 = Z / sqrt(V/5), standardized to unit variance by
                // multiplying sqrt(3/5): net scale sqrt(3/V). One V is shared
                // across this path, yielding a multivariate-t common scale.
                let mut scale_rng = SplitMix64::new(seed ^ 0x8CB9_2BA7_2F3D_8DD7);
                let chi_square = (0..5)
                    .map(|_| scale_rng.standard_normal().powi(2))
                    .sum::<f64>();
                (3.0 / chi_square).sqrt()
            }
        };
        Self {
            rng: SplitMix64::new(seed),
            state: initial_rng.standard_normal(),
            marginal_scale,
        }
    }

    fn next(&mut self, rho: f64) -> f64 {
        let innovation_scale = (1.0 - rho * rho).sqrt();
        self.state = rho * self.state + innovation_scale * self.rng.standard_normal();
        self.state * self.marginal_scale
    }

    fn block_maximum(&mut self, rho: f64, block_len: usize, gap: usize) -> f64 {
        let mut maximum = 0.0_f64;
        for _ in 0..block_len {
            maximum = maximum.max(self.next(rho).abs());
        }
        for _ in 0..gap {
            let _ = self.next(rho);
        }
        maximum
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SamplingDesign {
    IndependentBlocks,
    OneSerialPath,
}

impl SamplingDesign {
    fn label(self) -> &'static str {
        match self {
            Self::IndependentBlocks => "independent_blocks",
            Self::OneSerialPath => "one_serial_path",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RegimeReport {
    marginal: MarginalFamily,
    design: SamplingDesign,
    calibration_rho: f64,
    test_rho: f64,
    block_len: usize,
    embargo_gap: usize,
    calibration_blocks: usize,
    trials: u64,
    seed: u64,
    alarms: u64,
    empirical_alarm_rate: f64,
    alarm_rate_ci95_lower: f64,
    alarm_rate_ci95_upper: f64,
    empirical_coverage: f64,
}

fn wilson_interval(successes: u64, trials: u64) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = WILSON_Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    (
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    )
}

fn conformal_p_value(calibration_scores: &[f64], test_score: f64) -> f64 {
    let at_least_as_extreme = calibration_scores
        .iter()
        .filter(|score| **score >= test_score)
        .count();
    (1 + at_least_as_extreme) as f64 / (calibration_scores.len() + 1) as f64
}

fn independent_block_score(
    master: &mut SplitMix64,
    rho: f64,
    block_len: usize,
    marginal: MarginalFamily,
) -> f64 {
    let seed = master.next_u64();
    let mut stream = Ar1Stream::new(seed, marginal);
    stream.block_maximum(rho, block_len, 0)
}

fn run_regime_experiment(
    design: SamplingDesign,
    marginal: MarginalFamily,
    calibration_rho: f64,
    test_rho: f64,
    block_len: usize,
    embargo_gap: usize,
    trials: u64,
    seed: u64,
) -> RegimeReport {
    assert!(calibration_rho.abs() < 1.0);
    assert!(test_rho.abs() < 1.0);
    assert!(block_len > 0);
    assert!(trials > 0);

    let mut alarms = 0_u64;
    for trial in 0..trials {
        let trial_seed = seed ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut master = SplitMix64::new(trial_seed);
        let mut calibration = Vec::with_capacity(CALIBRATION_BLOCKS);
        let test_score;

        match design {
            SamplingDesign::IndependentBlocks => {
                for _ in 0..CALIBRATION_BLOCKS {
                    calibration.push(independent_block_score(
                        &mut master,
                        calibration_rho,
                        block_len,
                        marginal,
                    ));
                }
                test_score = independent_block_score(&mut master, test_rho, block_len, marginal);
            }
            SamplingDesign::OneSerialPath => {
                let mut stream = Ar1Stream::new(master.next_u64(), marginal);
                for _ in 0..CALIBRATION_BLOCKS {
                    calibration.push(stream.block_maximum(calibration_rho, block_len, embargo_gap));
                }
                // The regime changes only after the calibration prefix and
                // declared embargo. Every one-step marginal remains fixed within the selected family (Gaussian or standardized t5).
                test_score = stream.block_maximum(test_rho, block_len, 0);
            }
        }

        if conformal_p_value(&calibration, test_score) <= ALPHA {
            alarms = alarms.saturating_add(1);
        }
    }

    let rate = alarms as f64 / trials as f64;
    let (lower, upper) = wilson_interval(alarms, trials);
    RegimeReport {
        marginal,
        design,
        calibration_rho,
        test_rho,
        block_len,
        embargo_gap: if design == SamplingDesign::IndependentBlocks {
            0
        } else {
            embargo_gap
        },
        calibration_blocks: CALIBRATION_BLOCKS,
        trials,
        seed,
        alarms,
        empirical_alarm_rate: rate,
        alarm_rate_ci95_lower: lower,
        alarm_rate_ci95_upper: upper,
        empirical_coverage: 1.0 - rate,
    }
}

fn run_suite(trials: u64) -> Vec<RegimeReport> {
    let mut reports = Vec::new();
    for marginal in [
        MarginalFamily::Gaussian,
        MarginalFamily::StandardizedStudentT5CommonScale,
    ] {
        for (calibration_rho, test_rho) in [(0.05, 0.95), (0.95, 0.05)] {
            for block_len in [8_usize, 32_usize] {
                // Independent seed domain relative to the exploratory 2,500-trial grid.
                let base_seed = 0x5344_4343_5000_0000
                    ^ calibration_rho.to_bits().rotate_left(11)
                    ^ test_rho.to_bits().rotate_left(29)
                    ^ ((block_len as u64) << 32);
                reports.push(run_regime_experiment(
                    SamplingDesign::IndependentBlocks,
                    marginal,
                    calibration_rho,
                    test_rho,
                    block_len,
                    0,
                    trials,
                    base_seed,
                ));
            }
        }
    }
    reports
}

fn print_report(report: &RegimeReport) {
    println!(
        "{},{},{:.2},{:.2},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
        report.marginal.label(),
        report.design.label(),
        report.calibration_rho,
        report.test_rho,
        report.block_len,
        report.embargo_gap,
        report.calibration_blocks,
        report.trials,
        report.seed,
        report.alarms,
        ALPHA,
        report.empirical_alarm_rate,
        report.alarm_rate_ci95_lower,
        report.alarm_rate_ci95_upper,
        report.empirical_coverage,
    );
}

#[cfg(not(test))]
fn main() {
    println!(
        "marginal,design,calibration_rho,test_rho,block_len,embargo_gap,calibration_blocks,trials,seed,alarms,nominal_alpha,empirical_alarm_rate,wilson95_alarm_lower,wilson95_alarm_upper,empirical_coverage"
    );
    for report in run_suite(TRIALS) {
        print_report(&report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_independent_blocks_preserve_nominal_rank_under_rho_switch() {
        let report = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::Gaussian,
            0.05,
            0.95,
            1,
            0,
            20_000,
            0xB10C_C0DE_5101,
        );
        assert!((report.empirical_alarm_rate - ALPHA).abs() <= 0.02);
    }

    #[test]
    fn unchanged_marginals_can_produce_large_block_max_distribution_shift() {
        let forward = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::Gaussian,
            0.05,
            0.95,
            32,
            0,
            TEST_TRIALS,
            0xB10C_C0DE_5102,
        );
        let reverse = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::Gaussian,
            0.95,
            0.05,
            32,
            0,
            TEST_TRIALS,
            0xB10C_C0DE_5103,
        );
        assert!(forward.empirical_alarm_rate < 0.05);
        assert!(
            reverse.empirical_alarm_rate > 0.25,
            "reverse regime shift should produce a material alarm excess, observed {:.4}",
            reverse.empirical_alarm_rate
        );
    }

    #[test]
    fn standardized_student_t5_preserves_unit_variance_and_regime_shift() {
        let low_to_high = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::StandardizedStudentT5CommonScale,
            0.05,
            0.95,
            32,
            0,
            TEST_TRIALS,
            0xB10C_C0DE_5201,
        );
        let high_to_low = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::StandardizedStudentT5CommonScale,
            0.95,
            0.05,
            32,
            0,
            TEST_TRIALS,
            0xB10C_C0DE_5202,
        );
        assert!(low_to_high.empirical_alarm_rate < 0.08);
        assert!(high_to_low.empirical_alarm_rate > 0.18);
    }

    #[test]
    fn student_t5_marginal_is_unit_variance() {
        let mut sum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        let samples = 50_000_u64;
        for index in 0..samples {
            let mut stream = Ar1Stream::new(
                0xB10C_C0DE_5300 ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                MarginalFamily::StandardizedStudentT5CommonScale,
            );
            let value = stream.next(0.0);
            sum += value;
            sum_squares += value * value;
        }
        let mean = sum / samples as f64;
        let variance = sum_squares / samples as f64 - mean * mean;
        assert!(
            (variance - 1.0).abs() < 0.08,
            "observed variance {:.4}",
            variance
        );
    }

    #[test]
    fn student_t5_one_point_blocks_keep_nominal_rate_when_rho_changes() {
        let report = run_regime_experiment(
            SamplingDesign::IndependentBlocks,
            MarginalFamily::StandardizedStudentT5CommonScale,
            0.05,
            0.95,
            1,
            0,
            20_000,
            0xB10C_C0DE_5203,
        );
        assert!((report.empirical_alarm_rate - ALPHA).abs() <= 0.02);
    }

    #[test]
    fn regime_shift_report_is_exactly_replayable() {
        let first = run_regime_experiment(
            SamplingDesign::OneSerialPath,
            MarginalFamily::Gaussian,
            0.95,
            0.05,
            8,
            64,
            2_000,
            0xB10C_C0DE_5104,
        );
        let second = run_regime_experiment(
            SamplingDesign::OneSerialPath,
            MarginalFamily::Gaussian,
            0.95,
            0.05,
            8,
            64,
            2_000,
            0xB10C_C0DE_5104,
        );
        assert_eq!(first, second);
    }

    #[test]
    fn regime_grid_has_complete_finite_report_receipts() {
        let reports = run_suite(20);
        assert_eq!(reports.len(), 2 * 2 * 2);
        for report in reports {
            assert_eq!(report.calibration_blocks, CALIBRATION_BLOCKS);
            assert_eq!(report.trials, 20);
            assert!(report.empirical_alarm_rate.is_finite());
            assert!((0.0..=1.0).contains(&report.empirical_alarm_rate));
            assert!(report.alarm_rate_ci95_lower <= report.empirical_alarm_rate);
            assert!(report.alarm_rate_ci95_upper >= report.empirical_alarm_rate);
        }
    }
}
