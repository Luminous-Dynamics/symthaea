//! Paired operating-characteristic comparison of Gaussian sequential tests.
//!
//! Every method receives the same observation path within each episode.
//! This is synthetic calibration evidence for one fixed simple-vs-simple
//! Gaussian design, not a theorem or a physical validation result.

use falsification_power_reference::gaussian_terminal_induction::GaussianTerminalInducedTest;
use falsification_power_reference::sequential::PredictableGaussianEProcess;
use std::error::Error;
use std::f64::consts::TAU;

const PATHS: u64 = 20_000;
const HORIZON: usize = 80;
const ALPHA: f64 = 0.05;
const DISCREPANCY: f64 = 0.22;
const SIGMA: f64 = 0.80;
const INDUCED_STOP_THRESHOLDS: [f64; 3] = [0.75, 0.90, 0.99];
const RANDOMIZER_SALTS: [u64; 3] = [
    0xa5a5_5a5a_1234_9876,
    0x6a09_e667_f3bc_c909,
    0xbb67_ae85_84ca_a73b,
];
const Z_CRITICAL_ONE_SIDED_05: f64 = 1.644_853_626_951_472_2;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;
const ROOT_SEED: u64 = 0x4f1b_bcdd_6a09_e667;

#[derive(Clone, Copy)]
enum Scenario {
    SimpleNull,
    DeclaredAlternative,
}

impl Scenario {
    fn id(self) -> &'static str {
        match self {
            Self::SimpleNull => "fixed_gaussian_simple_null",
            Self::DeclaredAlternative => "fixed_gaussian_declared_alternative",
        }
    }

    fn seed_offset(self) -> u64 {
        match self {
            Self::SimpleNull => 0x1111_2222_3333_4444,
            Self::DeclaredAlternative => 0x9999_aaaa_bbbb_cccc,
        }
    }

    fn mean(self) -> f64 {
        match self {
            Self::SimpleNull => 0.0,
            Self::DeclaredAlternative => DISCREPANCY,
        }
    }
}

/// Deterministic SplitMix64 / Box-Muller stream for simulator calibration
/// only. It is not cryptographic and must not be used for real randomization.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
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
        let u1 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        let u2 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        let open_u1 = u1.max(0.5 / 9_007_199_254_740_992.0);
        (-2.0 * open_u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

#[derive(Default)]
struct StopPolicyCounts {
    threshold: f64,
    randomized_rejections: u64,
    early_stops: u64,
    total_stop_looks: u64,
}

#[derive(Default)]
struct Counts {
    terminal_np_rejections: u64,
    induced_policies: Vec<StopPolicyCounts>,
    e_process_crossings: u64,
    naive_repeated_rejections: u64,
    e_process_first_crossing_looks: u64,
    e_process_crossers: u64,
    terminal_induced_mismatches: u64,
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
    let half = WILSON_Z_95 * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn run_scenario(scenario: Scenario) -> Result<Counts, Box<dyn Error>> {
    let design =
        GaussianTerminalInducedTest::new(ALPHA, vec![DISCREPANCY; HORIZON], vec![SIGMA; HORIZON])?;
    let mut counts = Counts {
        induced_policies: INDUCED_STOP_THRESHOLDS
            .iter()
            .copied()
            .map(|threshold| StopPolicyCounts {
                threshold,
                ..StopPolicyCounts::default()
            })
            .collect(),
        ..Counts::default()
    };

    for path in 0..PATHS {
        let seed = ROOT_SEED ^ scenario.seed_offset() ^ path.wrapping_mul(0xd6e8_feb8_6659_fd93);
        let mut observation_rng = SplitMix64::new(seed);
        let mut e_process = PredictableGaussianEProcess::new(SIGMA, ALPHA)?;
        let mut observations = Vec::with_capacity(HORIZON);
        let mut cumulative_standardized = 0.0;
        let mut induced_stops = vec![None; INDUCED_STOP_THRESHOLDS.len()];
        let mut e_process_first_crossing = None;
        let mut naive_rejected = false;

        for look in 1..=HORIZON {
            let observed = scenario.mean() + SIGMA * observation_rng.standard_normal();
            observations.push(observed);
            cumulative_standardized += observed / SIGMA;

            let observation_id = format!("{}-{path}-{look}", scenario.id());
            let decision_digest = format!("{}-decision-{path}-{look}", scenario.id());
            e_process.commit_prediction_with_sigma(
                &observation_id,
                &decision_digest,
                DISCREPANCY,
                SIGMA,
            )?;
            let e_update = e_process.observe(&observation_id, &decision_digest, observed)?;
            if e_update.crossed_alpha_boundary && e_process_first_crossing.is_none() {
                e_process_first_crossing = Some(look as u64);
            }

            // Deliberately invalid negative control: repeat an unadjusted
            // one-sided fixed-time 5% z test at every prefix.
            let naive_z = cumulative_standardized / (look as f64).sqrt();
            if naive_z >= Z_CRITICAL_ONE_SIDED_05 {
                naive_rejected = true;
            }

            let induced = design.evaluate_prefix(&observations)?;
            for (policy_index, threshold) in INDUCED_STOP_THRESHOLDS.iter().copied().enumerate() {
                if induced_stops[policy_index].is_none()
                    && (induced.conditional_rejection_probability >= threshold || look == HORIZON)
                {
                    induced_stops[policy_index] = Some(induced);
                }
            }
        }

        let terminal = design.evaluate_prefix(&observations)?;
        let terminal_np_reject = terminal.conditional_rejection_probability >= 0.5;
        if terminal.conditional_rejection_probability != if terminal_np_reject { 1.0 } else { 0.0 }
        {
            counts.terminal_induced_mismatches += 1;
        }
        if terminal_np_reject {
            counts.terminal_np_rejections += 1;
        }

        for (policy_index, policy) in counts.induced_policies.iter_mut().enumerate() {
            let stop = induced_stops[policy_index]
                .expect("the fixed horizon always produces a stop value");
            if stop.observations < HORIZON {
                policy.early_stops += 1;
            }
            policy.total_stop_looks += stop.observations as u64;
            let mut randomizer_rng =
                SplitMix64::new(seed ^ RANDOMIZER_SALTS[policy_index]);
            if stop.randomized_reject(randomizer_rng.uniform())? {
                policy.randomized_rejections += 1;
            }
        }

        if e_process_first_crossing.is_some() {
            counts.e_process_crossings += 1;
            counts.e_process_crossers += 1;
            counts.e_process_first_crossing_looks +=
                e_process_first_crossing.unwrap_or(HORIZON as u64);
        }
        if naive_rejected {
            counts.naive_repeated_rejections += 1;
        }
    }

    Ok(counts)
}

fn emit_counts(scenario: Scenario, counts: &Counts, expected_terminal_power: f64) {
    let (terminal_low, terminal_high) = wilson_95(counts.terminal_np_rejections, PATHS);
    let (e_low, e_high) = wilson_95(counts.e_process_crossings, PATHS);
    let (naive_low, naive_high) = wilson_95(counts.naive_repeated_rejections, PATHS);
    let mean_e_crossing_look = if counts.e_process_crossers == 0 {
        0.0
    } else {
        counts.e_process_first_crossing_looks as f64 / counts.e_process_crossers as f64
    };
    let policy_json = counts
        .induced_policies
        .iter()
        .map(|policy| {
            let (low, high) = wilson_95(policy.randomized_rejections, PATHS);
            format!(
                concat!(
                    "{{",
                    "\"stop_threshold\":{:.4},",
                    "\"randomized_rejection_rate\":{:.8},",
                    "\"randomized_rejection_wilson95\":[{:.8},{:.8}],",
                    "\"early_stop_rate\":{:.8},",
                    "\"mean_stop_look\":{:.4}",
                    "}}"
                ),
                policy.threshold,
                policy.randomized_rejections as f64 / PATHS as f64,
                low,
                high,
                policy.early_stops as f64 / PATHS as f64,
                policy.total_stop_looks as f64 / PATHS as f64
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    println!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",",
            "\"paths\":{},",
            "\"horizon\":{},",
            "\"terminal_np_rejection_rate\":{:.8},",
            "\"terminal_np_wilson95\":[{:.8},{:.8}],",
            "\"induced_stop_policies\":[{}],",
            "\"e_process_crossing_rate\":{:.8},",
            "\"e_process_wilson95\":[{:.8},{:.8}],",
            "\"e_process_mean_crossing_look\":{:.4},",
            "\"naive_repeated_fixed_time_rate\":{:.8},",
            "\"naive_repeated_wilson95\":[{:.8},{:.8}],",
            "\"analytic_terminal_np_power\":{:.8},",
            "\"terminal_induced_mismatches\":{}",
            "}}"
        ),
        scenario.id(),
        PATHS,
        HORIZON,
        counts.terminal_np_rejections as f64 / PATHS as f64,
        terminal_low,
        terminal_high,
        policy_json,
        counts.e_process_crossings as f64 / PATHS as f64,
        e_low,
        e_high,
        mean_e_crossing_look,
        counts.naive_repeated_rejections as f64 / PATHS as f64,
        naive_low,
        naive_high,
        expected_terminal_power,
        counts.terminal_induced_mismatches
    );
}
fn main() -> Result<(), Box<dyn Error>> {
    let design =
        GaussianTerminalInducedTest::new(ALPHA, vec![DISCREPANCY; HORIZON], vec![SIGMA; HORIZON])?;
    let null = run_scenario(Scenario::SimpleNull)?;
    let alternative = run_scenario(Scenario::DeclaredAlternative)?;

    println!(
        concat!(
            "{{",
            "\"schema\":\"gaussian_terminal_test_induction_calibration_v1\",",
            "\"root_seed\":\"0x{:016x}\",",
            "\"prng\":\"splitmix64_box_muller_split_streams_v1\",",
            "\"nominal_alpha\":{:.8},",
            "\"paths_per_scenario\":{},",
            "\"maximum_looks\":{},",
            "\"induced_stop_thresholds\":[{:.4},{:.4},{:.4}],"
            "\"design_information\":{:.8},",
            "\"analytic_terminal_np_power\":{:.8},",
            "\"claim\":\"paired synthetic calibration diagnostic for a frozen simple Gaussian design only\",",
            "\"scenarios\":["
        ),
        ROOT_SEED,
        ALPHA,
        PATHS,
        HORIZON,
        INDUCED_STOP_THRESHOLDS[0],
        INDUCED_STOP_THRESHOLDS[1],
        INDUCED_STOP_THRESHOLDS[2],
        design.total_information(),
        design.terminal_power()
    );
    emit_counts(Scenario::SimpleNull, &null, design.terminal_power());
    println!(",");
    emit_counts(
        Scenario::DeclaredAlternative,
        &alternative,
        design.terminal_power(),
    );
    println!("]}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_streams_replay_and_separate_uniform_draws() {
        let mut left = SplitMix64::new(ROOT_SEED);
        let mut replay = SplitMix64::new(ROOT_SEED);
        let mut separate = SplitMix64::new(ROOT_SEED ^ 0xa5a5_5a5a_1234_9876);
        for _ in 0..12 {
            assert_eq!(left.next_u64(), replay.next_u64());
        }
        assert_ne!(replay.next_u64(), separate.next_u64());
    }

    #[test]
    fn wilson_interval_contains_observed_rate() {
        let (lower, upper) = wilson_95(1_000, 20_000);
        let observed = 0.05;
        assert!(lower <= observed);
        assert!(observed <= upper);
    }
}
