//! Empirical stress test for finite-sample block-conformal validity under AR(1) dependence.
//!
//! Run:
//!   rustc --edition=2024 scripts/research/sdcb_block_conformal_dependence_stress.rs -o /tmp/sdcb-block-stress
//!   /tmp/sdcb-block-stress
//!   rustc --edition=2024 --test scripts/research/sdcb_block_conformal_dependence_stress.rs -o /tmp/sdcb-block-stress-tests
//!   /tmp/sdcb-block-stress-tests
//!
//! This experiment is diagnostic, not a theorem or a substitute for the
//! exchangeability assumptions of conformal inference. It compares independent
//! stationary AR(1) blocks with blocks sampled consecutively from one stationary
//! AR(1) path, using embargo gaps to expose residual dependence.

const AR1_RHO: f64 = 0.95;
const CALIBRATION_BLOCKS: usize = 39;
const ALPHA: f64 = 0.10;
const TRIALS: u64 = 5_000;
const TEST_TRIALS: u64 = 2_500;
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

#[derive(Clone, Debug)]
struct Ar1Stream {
    rng: SplitMix64,
    state: f64,
    innovation_scale: f64,
}

impl Ar1Stream {
    fn new(seed: u64, rho: f64) -> Self {
        let mut initial_rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
        Self {
            rng: SplitMix64::new(seed),
            // Stationary initialization: X_0 ~ N(0, 1).
            state: initial_rng.standard_normal(),
            innovation_scale: (1.0 - rho * rho).sqrt(),
        }
    }

    fn next(&mut self, rho: f64) -> f64 {
        self.state = rho * self.state + self.innovation_scale * self.rng.standard_normal();
        self.state
    }

    /// Maximum absolute residual in one non-overlapping block, followed by an
    /// embargo interval that is not included in the block score.
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
struct CoverageReport {
    design: SamplingDesign,
    rho: f64,
    block_len: usize,
    embargo_gap: usize,
    calibration_blocks: usize,
    trials: u64,
    seed: u64,
    alarms: u64,
    nominal_alpha: f64,
    empirical_alarm_rate: f64,
    alarm_rate_ci95_lower: f64,
    alarm_rate_ci95_upper: f64,
    empirical_coverage: f64,
    coverage_ci95_lower: f64,
    coverage_ci95_upper: f64,
}

fn wilson_interval(successes: u64, trials: u64) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = WILSON_Z
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    ((center - half_width).max(0.0), (center + half_width).min(1.0))
}

fn conformal_p_value(calibration_scores: &[f64], test_score: f64) -> f64 {
    let at_least_as_extreme = calibration_scores
        .iter()
        .filter(|score| **score >= test_score)
        .count();
    (1 + at_least_as_extreme) as f64 / (calibration_scores.len() + 1) as f64
}

fn independent_block_score(master: &mut SplitMix64, rho: f64, block_len: usize) -> f64 {
    let seed = master.next_u64();
    let mut stream = Ar1Stream::new(seed, rho);
    stream.block_maximum(rho, block_len, 0)
}

fn run_coverage_experiment(
    design: SamplingDesign,
    rho: f64,
    block_len: usize,
    embargo_gap: usize,
    trials: u64,
    seed: u64,
) -> CoverageReport {
    assert!(rho.abs() < 1.0);
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
                    calibration.push(independent_block_score(&mut master, rho, block_len));
                }
                test_score = independent_block_score(&mut master, rho, block_len);
            }
            SamplingDesign::OneSerialPath => {
                let mut stream = Ar1Stream::new(master.next_u64(), rho);
                for _ in 0..CALIBRATION_BLOCKS {
                    calibration.push(stream.block_maximum(rho, block_len, embargo_gap));
                }
                test_score = stream.block_maximum(rho, block_len, embargo_gap);
            }
        }

        if conformal_p_value(&calibration, test_score) <= ALPHA {
            alarms += 1;
        }
    }

    let rate = alarms as f64 / trials as f64;
    let (alarm_lower, alarm_upper) = wilson_interval(alarms, trials);
    CoverageReport {
        design,
        rho,
        block_len,
        embargo_gap,
        calibration_blocks: CALIBRATION_BLOCKS,
        trials,
        seed,
        alarms,
        nominal_alpha: ALPHA,
        empirical_alarm_rate: rate,
        alarm_rate_ci95_lower: alarm_lower,
        alarm_rate_ci95_upper: alarm_upper,
        empirical_coverage: 1.0 - rate,
        coverage_ci95_lower: 1.0 - alarm_upper,
        coverage_ci95_upper: 1.0 - alarm_lower,
    }
}

fn run_suite(trials: u64) -> Vec<CoverageReport> {
    let mut reports = Vec::new();
    for block_len in [1, 8, 32] {
        reports.push(run_coverage_experiment(
            SamplingDesign::IndependentBlocks,
            AR1_RHO,
            block_len,
            0,
            trials,
            0x5344_4342_0000_0000 ^ block_len as u64,
        ));
        for gap in [0, 16, 64] {
            reports.push(run_coverage_experiment(
                SamplingDesign::OneSerialPath,
                AR1_RHO,
                block_len,
                gap,
                trials,
                0x5344_4342_1000_0000
                    ^ ((block_len as u64) << 16)
                    ^ gap as u64,
            ));
        }
    }
    reports
}

fn print_report(report: &CoverageReport) {
    println!(
        "{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
        report.design.label(),
        report.rho,
        report.block_len,
        report.embargo_gap,
        report.calibration_blocks,
        report.trials,
        report.seed,
        report.alarms,
        report.nominal_alpha,
        report.empirical_alarm_rate,
        report.alarm_rate_ci95_lower,
        report.alarm_rate_ci95_upper,
        report.empirical_coverage
    );
}

#[cfg(not(test))]
fn main() {
    println!(
        "design,rho,block_len,embargo_gap,calibration_blocks,trials,seed,alarms,nominal_alpha,empirical_alarm_rate,wilson95_alarm_lower,wilson95_alarm_upper,empirical_coverage"
    );
    for report in run_suite(TRIALS) {
        print_report(&report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_stationary_blocks_respect_finite_sample_alarm_rate() {
        let report = run_coverage_experiment(
            SamplingDesign::IndependentBlocks,
            AR1_RHO,
            8,
            0,
            TEST_TRIALS,
            0xB10C_C0DE_0001,
        );
        assert!(report.empirical_alarm_rate <= ALPHA + 0.025);
        assert!(report.empirical_alarm_rate >= ALPHA - 0.025);
        assert!(report.coverage_ci95_lower <= report.empirical_coverage);
        assert!(report.coverage_ci95_upper >= report.empirical_coverage);
    }

    #[test]
    fn serial_path_experiment_is_exactly_replayable() {
        let first = run_coverage_experiment(
            SamplingDesign::OneSerialPath,
            AR1_RHO,
            8,
            16,
            1_000,
            0xB10C_C0DE_0002,
        );
        let second = run_coverage_experiment(
            SamplingDesign::OneSerialPath,
            AR1_RHO,
            8,
            16,
            1_000,
            0xB10C_C0DE_0002,
        );
        assert_eq!(first, second);
    }

    #[test]
    fn block_length_and_embargo_grid_returns_well_formed_coverage_receipts() {
        for report in run_suite(100) {
            assert_eq!(report.calibration_blocks, CALIBRATION_BLOCKS);
            assert_eq!(report.trials, 100);
            assert!(report.empirical_alarm_rate.is_finite());
            assert!((0.0..=1.0).contains(&report.empirical_alarm_rate));
            assert!(report.coverage_ci95_lower <= report.empirical_coverage);
            assert!(report.coverage_ci95_upper >= report.empirical_coverage);
        }
    }
}
