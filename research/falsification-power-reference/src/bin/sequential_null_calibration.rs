//! Reproducible operating-characteristic diagnostic for the Gaussian
//! sequential falsification reference.
//!
//! This is a synthetic calibration artifact, not proof of a statistical
//! guarantee and not a real-world validation result. It contrasts the
//! time-uniform e-process with naive repeated fixed-time z tests.

use falsification_power_reference::sequential::PredictableGaussianEProcess;
use std::error::Error;
use std::f64::consts::TAU;

const PATHS: u64 = 20_000;
const HORIZON: u64 = 100;
const ALPHA: f64 = 0.05;
const ROOT_SEED: u64 = 0x5eed_cafe_d00d_f00d;
const Z_CRITICAL_ONE_SIDED_05: f64 = 1.644_853_626_951_472_2;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy)]
enum Scenario {
    FixedNull,
    AdaptiveHeteroskedasticNull,
    AdaptiveHeteroskedasticAlternative,
}

impl Scenario {
    fn id(self) -> &'static str {
        match self {
            Self::FixedNull => "fixed_known_variance_null",
            Self::AdaptiveHeteroskedasticNull => "predictable_adaptive_variance_null",
            Self::AdaptiveHeteroskedasticAlternative => {
                "predictable_adaptive_variance_material_alternative"
            }
        }
    }

    fn seed_offset(self) -> u64 {
        match self {
            Self::FixedNull => 0x1111_2222_3333_4444,
            Self::AdaptiveHeteroskedasticNull => 0x5555_6666_7777_8888,
            Self::AdaptiveHeteroskedasticAlternative => 0x9999_aaaa_bbbb_cccc,
        }
    }

    fn is_alternative(self) -> bool {
        matches!(self, Self::AdaptiveHeteroskedasticAlternative)
    }

    fn design(self, previous_observation: f64) -> (f64, f64) {
        match self {
            Self::FixedNull => (0.25, 1.0),
            Self::AdaptiveHeteroskedasticNull | Self::AdaptiveHeteroskedasticAlternative => {
                if previous_observation >= 0.0 {
                    (0.25, 0.7)
                } else {
                    (-0.25, 1.3)
                }
            }
        }
    }
}

/// Versioned deterministic SplitMix64 stream for simulator calibration only.
/// It is not cryptographic randomness and must not be used for interventions.
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

    fn next_open_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_open_unit();
        let u2 = self.next_open_unit();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

#[derive(Default)]
struct OperatingCounts {
    e_process_crossings: u64,
    naive_repeated_test_crossings: u64,
    detection_steps: u64,
    detections_with_step: u64,
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

fn run_scenario(scenario: Scenario) -> Result<OperatingCounts, Box<dyn Error>> {
    let mut counts = OperatingCounts::default();

    for path in 0..PATHS {
        let mut rng = SplitMix64::new(
            ROOT_SEED ^ scenario.seed_offset() ^ path.wrapping_mul(0xd6e8_feb8_6659_fd93),
        );
        let mut process = PredictableGaussianEProcess::new(1.0, ALPHA)?;
        let mut previous_observation = 0.0;
        let mut standardized_sum = 0.0;
        let mut e_process_first_crossing = None;
        let mut naive_first_crossing = false;

        for look in 1..=HORIZON {
            let (discrepancy, sigma) = scenario.design(previous_observation);
            let noise = rng.standard_normal();
            let observed = sigma * noise
                + if scenario.is_alternative() {
                    discrepancy
                } else {
                    0.0
                };

            // The alternative prediction and sigma are selected using only
            // the previous observation and are committed before this outcome.
            let observation_id = format!("{}-{path}-{look}", scenario.id());
            let decision_digest = format!("{}-decision-{path}-{look}", scenario.id());
            process.commit_prediction_with_sigma(
                &observation_id,
                &decision_digest,
                discrepancy,
                sigma,
            )?;
            let update = process.observe(&observation_id, &decision_digest, observed)?;

            if update.crossed_alpha_boundary && e_process_first_crossing.is_none() {
                e_process_first_crossing = Some(look);
            }

            // Deliberately invalid control: repeatedly inspect an unadjusted
            // one-sided fixed-time 5% z test at every prefix.
            standardized_sum += observed / sigma;
            let naive_z = standardized_sum / (look as f64).sqrt();
            if naive_z >= Z_CRITICAL_ONE_SIDED_05 {
                naive_first_crossing = true;
            }

            previous_observation = observed;
        }

        if let Some(detection_step) = e_process_first_crossing {
            counts.e_process_crossings += 1;
            counts.detections_with_step += 1;
            counts.detection_steps += detection_step;
        }
        if naive_first_crossing {
            counts.naive_repeated_test_crossings += 1;
        }
    }

    Ok(counts)
}

fn emit_scenario(scenario: Scenario, counts: &OperatingCounts) {
    let e_rate = counts.e_process_crossings as f64 / PATHS as f64;
    let naive_rate = counts.naive_repeated_test_crossings as f64 / PATHS as f64;
    let (e_lower, e_upper) = wilson_95(counts.e_process_crossings, PATHS);
    let (naive_lower, naive_upper) = wilson_95(counts.naive_repeated_test_crossings, PATHS);
    let mean_detection_step = if counts.detections_with_step == 0 {
        0.0
    } else {
        counts.detection_steps as f64 / counts.detections_with_step as f64
    };

    println!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",",
            "\"paths\":{},",
            "\"horizon\":{},",
            "\"e_process_crossings\":{},",
            "\"e_process_rate\":{:.8},",
            "\"e_process_wilson95\":[{:.8},{:.8}],",
            "\"naive_repeated_fixed_time_crossings\":{},",
            "\"naive_repeated_fixed_time_rate\":{:.8},",
            "\"naive_repeated_fixed_time_wilson95\":[{:.8},{:.8}],",
            "\"mean_e_process_detection_look\":{:.4},",
            "\"undetected_paths\":{}",
            "}}"
        ),
        scenario.id(),
        PATHS,
        HORIZON,
        counts.e_process_crossings,
        e_rate,
        e_lower,
        e_upper,
        counts.naive_repeated_test_crossings,
        naive_rate,
        naive_lower,
        naive_upper,
        mean_detection_step,
        PATHS - counts.e_process_crossings
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let fixed_null = run_scenario(Scenario::FixedNull)?;
    let adaptive_null = run_scenario(Scenario::AdaptiveHeteroskedasticNull)?;
    let adaptive_alternative = run_scenario(Scenario::AdaptiveHeteroskedasticAlternative)?;

    println!(
        concat!(
            "{{",
            "\"schema\":\"symthaea_sequential_null_calibration_v0\",",
            "\"root_seed\":\"0x{:016x}\",",
            "\"prng\":\"splitmix64_box_muller_v1\",",
            "\"nominal_alpha\":{:.8},",
            "\"paths_per_scenario\":{},",
            "\"maximum_looks\":{},",
            "\"claim\":\"synthetic Monte Carlo calibration diagnostic only; not a theorem or physical validation\",",
            "\"scenarios\":["
        ),
        ROOT_SEED, ALPHA, PATHS, HORIZON
    );
    emit_scenario(Scenario::FixedNull, &fixed_null);
    println!(",");
    emit_scenario(Scenario::AdaptiveHeteroskedasticNull, &adaptive_null);
    println!(",");
    emit_scenario(
        Scenario::AdaptiveHeteroskedasticAlternative,
        &adaptive_alternative,
    );
    println!("]}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_contains_empirical_rate_and_is_ordered() {
        let (lower, upper) = wilson_95(1_000, 20_000);
        let rate = 1_000.0 / 20_000.0;
        assert!(lower <= rate);
        assert!(rate <= upper);
        assert!(lower < upper);
    }

    #[test]
    fn deterministic_prng_replays_and_derived_paths_diverge() {
        let mut first = SplitMix64::new(ROOT_SEED);
        let mut replay = SplitMix64::new(ROOT_SEED);
        for _ in 0..8 {
            assert_eq!(first.next_u64(), replay.next_u64());
        }
        let first_seeded_value = SplitMix64::new(ROOT_SEED).next_u64();
        let changed_seeded_value = SplitMix64::new(ROOT_SEED ^ 1).next_u64();
        assert_ne!(first_seeded_value, changed_seeded_value);
    }
}
