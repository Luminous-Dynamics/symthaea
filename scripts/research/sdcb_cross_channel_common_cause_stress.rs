//! Cross-channel common-cause regime shift with unchanged per-channel marginals.
//!
//! Two stationary AR(1) sensor channels have the same rho and unit marginal
//! variance. Only their contemporaneous innovation correlation gamma changes.
//! The block score is max(abs(channel_1), abs(channel_2)) over all ticks.
//!
//! Gaussian and standardized multivariate Student-t(5) common-scale paths are
//! evaluated. The t5 scale is shared across both channels and the whole path.
//!
//! Run:
//!   rustc --edition=2024 scripts/research/sdcb_cross_channel_common_cause_stress.rs -o /tmp/sdcb-cross-channel
//!   /tmp/sdcb-cross-channel
//!   rustc --edition=2024 --test scripts/research/sdcb_cross_channel_common_cause_stress.rs -o /tmp/sdcb-cross-channel-tests
//!   /tmp/sdcb-cross-channel-tests
//!
//! Synthetic diagnostic only. No universal coverage or physical sensor claim.

const CROSS_CHANNEL_RHO: f64 = 0.80;
const GAMMA_GRID: [f64; 2] = [0.05, 0.95];
const BLOCK_LENGTH_GRID: [usize; 3] = [1, 8, 32];
const CALIBRATION_BLOCKS: usize = 39;
const ALPHA: f64 = 0.10;
const TRIALS: u64 = 2_500;
const TEST_TRIALS: u64 = 10_000;
const WILSON_Z: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    fn uniform_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform_open();
        radius * angle.cos()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarginalFamily { Gaussian, StudentT5CommonScale }

impl MarginalFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::StudentT5CommonScale => "student_t5_common_scale",
        }
    }
}

#[derive(Clone, Debug)]
struct TwoChannelAr1 {
    rng: SplitMix64,
    x: f64,
    y: f64,
    common_scale: f64,
}

impl TwoChannelAr1 {
    fn new(seed: u64, marginal: MarginalFamily, gamma: f64) -> Self {
        assert!(gamma.abs() < 1.0);
        let mut initial = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
        let x0 = initial.standard_normal();
        let y0_independent = initial.standard_normal();
        let y0 = gamma * x0 + (1.0 - gamma * gamma).sqrt() * y0_independent;
        let common_scale = match marginal {
            MarginalFamily::Gaussian => 1.0,
            MarginalFamily::StudentT5CommonScale => {
                let mut scale_rng = SplitMix64::new(seed ^ 0x8CB9_2BA7_2F3D_8DD7);
                let chi_square = (0..5).map(|_| scale_rng.standard_normal().powi(2)).sum::<f64>();
                (3.0 / chi_square).sqrt()
            }
        };
        Self { rng: SplitMix64::new(seed), x: x0, y: y0, common_scale }
    }

    fn next(&mut self, rho: f64, gamma: f64) -> (f64, f64) {
        let innovation_x = self.rng.standard_normal();
        let innovation_y_independent = self.rng.standard_normal();
        let innovation_y = gamma * innovation_x
            + (1.0 - gamma * gamma).sqrt() * innovation_y_independent;
        let scale = (1.0 - rho * rho).sqrt();
        self.x = rho * self.x + scale * innovation_x;
        self.y = rho * self.y + scale * innovation_y;
        (self.x * self.common_scale, self.y * self.common_scale)
    }

    fn block_maximum(&mut self, rho: f64, gamma: f64, block_len: usize) -> f64 {
        let mut maximum = 0.0_f64;
        for _ in 0..block_len {
            let (x, y) = self.next(rho, gamma);
            maximum = maximum.max(x.abs()).max(y.abs());
        }
        maximum
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Report {
    marginal: MarginalFamily,
    calibration_gamma: f64,
    test_gamma: f64,
    block_len: usize,
    trials: u64,
    seed: u64,
    alarms: u64,
    alarm_rate: f64,
    ci_lower: f64,
    ci_upper: f64,
}

fn wilson(successes: u64, trials: u64) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = WILSON_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn independent_block_score(master: &mut SplitMix64, gamma: f64, block_len: usize, marginal: MarginalFamily) -> f64 {
    let mut path = TwoChannelAr1::new(master.next_u64(), marginal, gamma);
    path.block_maximum(CROSS_CHANNEL_RHO, gamma, block_len)
}

fn run_experiment(
    marginal: MarginalFamily,
    calibration_gamma: f64,
    test_gamma: f64,
    block_len: usize,
    trials: u64,
    seed: u64,
) -> Report {
    assert!(calibration_gamma.abs() < 1.0 && test_gamma.abs() < 1.0);
    assert!(block_len > 0 && trials > 0);
    let mut alarms = 0_u64;
    for trial in 0..trials {
        let mut master = SplitMix64::new(seed ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut calibration = Vec::with_capacity(CALIBRATION_BLOCKS);
        for _ in 0..CALIBRATION_BLOCKS {
            calibration.push(independent_block_score(&mut master, calibration_gamma, block_len, marginal));
        }
        let test_score = independent_block_score(&mut master, test_gamma, block_len, marginal);
        let exceedances = calibration.iter().filter(|score| **score >= test_score).count();
        let p_value = (1 + exceedances) as f64 / (CALIBRATION_BLOCKS + 1) as f64;
        if p_value <= ALPHA { alarms += 1; }
    }
    let (ci_lower, ci_upper) = wilson(alarms, trials);
    Report {
        marginal, calibration_gamma, test_gamma, block_len, trials, seed, alarms,
        alarm_rate: alarms as f64 / trials as f64, ci_lower, ci_upper,
    }
}

fn run_suite(trials: u64) -> Vec<Report> {
    let mut reports = Vec::new();
    for marginal in [MarginalFamily::Gaussian, MarginalFamily::StudentT5CommonScale] {
        for calibration_gamma in GAMMA_GRID {
            for test_gamma in GAMMA_GRID {
                for block_len in BLOCK_LENGTH_GRID {
                    let seed = 0x5344_4342_0000_0000
                        ^ calibration_gamma.to_bits().rotate_left(13)
                        ^ test_gamma.to_bits().rotate_left(31)
                        ^ ((block_len as u64) << 32);
                    reports.push(run_experiment(
                        marginal, calibration_gamma, test_gamma, block_len, trials, seed,
                    ));
                }
            }
        }
    }
    reports
}

#[cfg(not(test))]
fn main() {
    println!("marginal,calibration_gamma,test_gamma,rho,block_len,calibration_blocks,trials,seed,alarms,nominal_alpha,alarm_rate,wilson95_lower,wilson95_upper");
    for report in run_suite(TRIALS) {
        println!(
            "{},{:.2},{:.2},{:.2},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            report.marginal.label(), report.calibration_gamma, report.test_gamma,
            CROSS_CHANNEL_RHO, report.block_len, CALIBRATION_BLOCKS, report.trials,
            report.seed, report.alarms, ALPHA, report.alarm_rate, report.ci_lower, report.ci_upper
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_channel_marginals_are_unchanged_when_cross_channel_gamma_changes() {
        let samples = 50_000_u64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        for index in 0..samples {
            let seed = 0xB10C_C0DE_6100 ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut low = TwoChannelAr1::new(seed, MarginalFamily::Gaussian, 0.05);
            let mut high = TwoChannelAr1::new(seed, MarginalFamily::Gaussian, 0.95);
            let (lx, ly) = low.next(CROSS_CHANNEL_RHO, 0.05);
            let (hx, hy) = high.next(CROSS_CHANNEL_RHO, 0.95);
            sum_x += lx + hx;
            sum_y += ly + hy;
            sum_x2 += lx * lx + hx * hx;
            sum_y2 += ly * ly + hy * hy;
        }
        let count = (2 * samples) as f64;
        let variance_x = sum_x2 / count - (sum_x / count).powi(2);
        let variance_y = sum_y2 / count - (sum_y / count).powi(2);
        assert!((variance_x - 1.0).abs() < 0.04);
        assert!((variance_y - 1.0).abs() < 0.04);
    }

    #[test]
    fn cross_channel_shift_changes_max_score_in_both_directions() {
        let low_to_high = run_experiment(
            MarginalFamily::Gaussian, 0.05, 0.95, 8, TEST_TRIALS, 0xB10C_C0DE_6201,
        );
        let high_to_low = run_experiment(
            MarginalFamily::Gaussian, 0.95, 0.05, 8, TEST_TRIALS, 0xB10C_C0DE_6202,
        );
        assert!(low_to_high.alarm_rate < 0.06, "low→high {:.4}", low_to_high.alarm_rate);
        assert!(high_to_low.alarm_rate > 0.15, "high→low {:.4}", high_to_low.alarm_rate);
    }

    #[test]
    fn student_t5_common_scale_also_exposes_cross_channel_shift() {
        let low_to_high = run_experiment(
            MarginalFamily::StudentT5CommonScale, 0.05, 0.95, 8, TEST_TRIALS, 0xB10C_C0DE_6301,
        );
        let high_to_low = run_experiment(
            MarginalFamily::StudentT5CommonScale, 0.95, 0.05, 8, TEST_TRIALS, 0xB10C_C0DE_6302,
        );
        assert!(low_to_high.alarm_rate < 0.08, "t5 low→high {:.4}", low_to_high.alarm_rate);
        assert!(high_to_low.alarm_rate > 0.13, "t5 high→low {:.4}", high_to_low.alarm_rate);
    }

    #[test]
    fn unchanged_gamma_report_is_replayable() {
        let first = run_experiment(MarginalFamily::Gaussian, 0.95, 0.95, 8, 500, 0xB10C_C0DE_6401);
        let second = run_experiment(MarginalFamily::Gaussian, 0.95, 0.95, 8, 500, 0xB10C_C0DE_6401);
        assert_eq!(first, second);
    }

    #[test]
    fn grid_has_all_gamma_directions_and_finite_intervals() {
        let reports = run_suite(10);
        assert_eq!(reports.len(), 2 * GAMMA_GRID.len() * GAMMA_GRID.len() * BLOCK_LENGTH_GRID.len());
        for report in reports {
            assert_eq!(report.calibration_gamma.abs() < 1.0, true);
            assert_eq!(report.test_gamma.abs() < 1.0, true);
            assert_eq!(report.trials, 10);
            assert!(report.alarm_rate.is_finite());
            assert!((0.0..=1.0).contains(&report.alarm_rate));
            assert!(report.ci_lower <= report.alarm_rate && report.ci_upper >= report.alarm_rate);
        }
    }
}
