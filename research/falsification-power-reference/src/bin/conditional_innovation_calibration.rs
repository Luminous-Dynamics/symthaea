//! Reproducible conditional-innovation calibration under an AR(1) null.
//!
//! The correct monitor commits the predictable AR(1) conditional mean before
//! each observation. A second monitor deliberately assumes a zero conditional
//! mean on the same correlated paths to expose misspecification sensitivity.
//! These are synthetic calibration diagnostics, not universal guarantees.

use falsification_power_reference::sequential::PredictableGaussianEProcess;
use std::error::Error;
use std::f64::consts::TAU;

const PATHS: u64 = 20_000;
const HORIZON: u64 = 100;
const ALPHA: f64 = 0.05;
const RHO: f64 = 0.65;
const SIGMA: f64 = 0.80;
const DISCREPANCY: f64 = 0.22;
const ROOT_SEED: u64 = 0x91e1_0da5_c79e_7b1d;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy)]
enum Scenario {
    ConditionalNull,
    ConditionalAlternative,
}

impl Scenario {
    fn id(self) -> &'static str {
        match self {
            Self::ConditionalNull => "ar1_predictable_conditional_null",
            Self::ConditionalAlternative => "ar1_predictable_conditional_alternative",
        }
    }

    fn offset(self) -> u64 {
        match self {
            Self::ConditionalNull => 0x1020_3040_5060_7080,
            Self::ConditionalAlternative => 0x90a0_b0c0_d0e0_f001,
        }
    }

    fn shift(self) -> f64 {
        match self {
            Self::ConditionalNull => 0.0,
            Self::ConditionalAlternative => DISCREPANCY,
        }
    }
}

/// Deterministic SplitMix64 / Box-Muller stream for synthetic calibration.
/// It is not cryptographic and must not be used for real randomization.
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
struct Counts {
    conditional_crossings: u64,
    zero_mean_crossings: u64,
    conditional_crossing_looks: u64,
    zero_mean_crossing_looks: u64,
    conditional_crossers: u64,
    zero_mean_crossers: u64,
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
    let mut counts = Counts::default();

    for path in 0..PATHS {
        let seed = ROOT_SEED ^ scenario.offset() ^ path.wrapping_mul(0xd6e8_feb8_6659_fd93);
        let mut rng = SplitMix64::new(seed);
        let mut conditional = PredictableGaussianEProcess::new(SIGMA, ALPHA)?;
        let mut zero_mean = PredictableGaussianEProcess::new(SIGMA, ALPHA)?;
        let mut previous_observation = 0.0;
        let mut conditional_first_crossing = None;
        let mut zero_mean_first_crossing = None;

        for look in 1..=HORIZON {
            // This mean is a deterministic function of the previous public
            // observation. It is committed before the current noise is drawn.
            let null_conditional_mean = RHO * previous_observation;
            let observed = null_conditional_mean + scenario.shift() + SIGMA * rng.standard_normal();

            if conditional_first_crossing.is_none() {
                let observation_id = format!("conditional-{path}-{look}");
                let decision_digest = format!("conditional-decision-{path}-{look}");
                conditional.commit_prediction_with_conditional_gaussian(
                    &observation_id,
                    &decision_digest,
                    DISCREPANCY,
                    null_conditional_mean,
                    SIGMA,
                )?;
                let update = conditional.observe(&observation_id, &decision_digest, observed)?;
                if update.crossed_alpha_boundary {
                    conditional_first_crossing = Some(look);
                }
            }

            if zero_mean_first_crossing.is_none() {
                let observation_id = format!("zero-mean-{path}-{look}");
                let decision_digest = format!("zero-mean-decision-{path}-{look}");
                zero_mean.commit_prediction_with_sigma(
                    &observation_id,
                    &decision_digest,
                    DISCREPANCY,
                    SIGMA,
                )?;
                let update = zero_mean.observe(&observation_id, &decision_digest, observed)?;
                if update.crossed_alpha_boundary {
                    zero_mean_first_crossing = Some(look);
                }
            }

            previous_observation = observed;
            if conditional_first_crossing.is_some() && zero_mean_first_crossing.is_some() {
                break;
            }
        }

        if let Some(look) = conditional_first_crossing {
            counts.conditional_crossings += 1;
            counts.conditional_crossers += 1;
            counts.conditional_crossing_looks += look;
        }
        if let Some(look) = zero_mean_first_crossing {
            counts.zero_mean_crossings += 1;
            counts.zero_mean_crossers += 1;
            counts.zero_mean_crossing_looks += look;
        }
    }

    Ok(counts)
}

fn emit_scenario(scenario: Scenario, counts: &Counts) {
    let (conditional_low, conditional_high) = wilson_95(counts.conditional_crossings, PATHS);
    let (zero_mean_low, zero_mean_high) = wilson_95(counts.zero_mean_crossings, PATHS);
    let conditional_mean_look = if counts.conditional_crossers == 0 {
        0.0
    } else {
        counts.conditional_crossing_looks as f64 / counts.conditional_crossers as f64
    };
    let zero_mean_mean_look = if counts.zero_mean_crossers == 0 {
        0.0
    } else {
        counts.zero_mean_crossing_looks as f64 / counts.zero_mean_crossers as f64
    };
    let conditional_mean_observations = (counts.conditional_crossing_looks
        + (PATHS - counts.conditional_crossers) * HORIZON)
        as f64
        / PATHS as f64;
    let zero_mean_mean_observations =
        (counts.zero_mean_crossing_looks + (PATHS - counts.zero_mean_crossers) * HORIZON) as f64
            / PATHS as f64;

    println!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",",
            "\"paths\":{},",
            "\"horizon\":{},",
            "\"conditional_innovation_crossings\":{},",
            "\"conditional_innovation_rate\":{:.8},",
            "\"conditional_innovation_wilson95\":[{:.8},{:.8}],",
            "\"conditional_innovation_mean_crossing_look\":{:.4},",
            "\"conditional_innovation_mean_observations_used\":{:.4},",
            "\"zero_mean_misspecified_crossings\":{},",
            "\"zero_mean_misspecified_rate\":{:.8},",
            "\"zero_mean_misspecified_wilson95\":[{:.8},{:.8}],",
            "\"zero_mean_misspecified_mean_crossing_look\":{:.4},",
            "\"zero_mean_misspecified_mean_observations_used\":{:.4}",
            "}}"
        ),
        scenario.id(),
        PATHS,
        HORIZON,
        counts.conditional_crossings,
        counts.conditional_crossings as f64 / PATHS as f64,
        conditional_low,
        conditional_high,
        conditional_mean_look,
        conditional_mean_observations,
        counts.zero_mean_crossings,
        counts.zero_mean_crossings as f64 / PATHS as f64,
        zero_mean_low,
        zero_mean_high,
        zero_mean_mean_look,
        zero_mean_mean_observations
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let null = run_scenario(Scenario::ConditionalNull)?;
    let alternative = run_scenario(Scenario::ConditionalAlternative)?;

    println!(
        concat!(
            "{{",
            "\"schema\":\"gaussian_conditional_innovation_calibration_v0\",",
            "\"root_seed\":\"0x{:016x}\",",
            "\"prng\":\"splitmix64_box_muller_v1\",",
            "\"nominal_alpha\":{:.8},",
            "\"paths_per_scenario\":{},",
            "\"maximum_looks\":{},",
            "\"ar1_rho\":{:.8},",
            "\"sigma\":{:.8},",
            "\"discrepancy\":{:.8},",
            "\"claim\":\"synthetic conditional-Gaussian AR(1) calibration diagnostic only\",",
            "\"scenarios\":["
        ),
        ROOT_SEED, ALPHA, PATHS, HORIZON, RHO, SIGMA, DISCREPANCY
    );
    emit_scenario(Scenario::ConditionalNull, &null);
    println!(",");
    emit_scenario(Scenario::ConditionalAlternative, &alternative);
    println!("]}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_rng_replays() {
        let mut left = SplitMix64::new(ROOT_SEED);
        let mut right = SplitMix64::new(ROOT_SEED);
        for _ in 0..16 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn wilson_interval_contains_observed_rate() {
        let (low, high) = wilson_95(1_000, 20_000);
        assert!(low <= 0.05);
        assert!(high >= 0.05);
    }
}
