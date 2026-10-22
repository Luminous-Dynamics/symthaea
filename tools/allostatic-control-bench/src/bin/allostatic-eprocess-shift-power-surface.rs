//! Empirical power surface for predeclared bounded-risk e-process betting rates.
//!
//! Usage: allostatic-eprocess-shift-power-surface <episodes.csv> <summary.csv>
//! Every comparator sees the same deterministic Bernoulli loss sequence per
//! episode. The null is conditional expected loss <= risk_bound; the sweep tests
//! power across persistent, gradual, transient, recurrent and unobservable shifts.

use allostatic_control_bench::bounded_risk_eprocess::{
    BoundedRiskEProcess, MixtureBoundedRiskEProcess,
};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZON: usize = 256;
const EPISODES_PER_SCENARIO: usize = 5_000;
const RISK_BOUND: f64 = 0.15;
const GLOBAL_DELTA: f64 = 0.05;
const PARALLEL_STREAMS: usize = 5;
const LOW_BET_FRACTIONS: [f64; 3] = [0.02, 0.10, 0.25];
const HIGH_BET_FRACTIONS: [f64; 3] = [0.25, 0.50, 0.75];
const FIXED_BET_FRACTIONS: [f64; 5] = [0.02, 0.10, 0.25, 0.50, 0.75];
const ALARM_SENTINEL: usize = HORIZON;

const RAW_HEADER: &str = "scenario,latent_shift,risk_violating,pure_null,episode,seed,horizon,mean_expected_loss,max_expected_loss,observed_mean_loss,observed_events,low_mix_first_alarm,low_mix_detected,low_mix_final_log_e,low_mix_peak_log_e,low_component_log_e_002,low_component_log_e_010,low_component_log_e_025,high_mix_first_alarm,high_mix_detected,high_mix_final_log_e,high_mix_peak_log_e,high_component_log_e_025,high_component_log_e_050,high_component_log_e_075,fixed_002_first_alarm,fixed_002_detected,fixed_002_final_log_e,fixed_010_first_alarm,fixed_010_detected,fixed_010_final_log_e,fixed_025_first_alarm,fixed_025_detected,fixed_025_final_log_e,fixed_050_first_alarm,fixed_050_detected,fixed_050_final_log_e,fixed_075_first_alarm,fixed_075_detected,fixed_075_final_log_e,log_alarm_threshold,risk_bound,global_delta,parallel_streams";
const SUMMARY_HEADER: &str = "scenario,latent_shift,risk_violating,pure_null,episodes,mean_expected_loss,mean_observed_loss,max_expected_loss,low_mix_faster_than_best_fixed,high_mix_faster_than_best_fixed,low_mix_alarms,low_mix_alarm_rate,low_mix_rate_l95,low_mix_rate_u95,low_mix_false_alarms,low_mix_risk_misses,low_mix_latent_misses,low_mix_mean_delay,low_mix_p95_delay,low_mix_median_delay,low_mix_mean_final_log_e,high_mix_alarms,high_mix_alarm_rate,high_mix_rate_l95,high_mix_rate_u95,high_mix_false_alarms,high_mix_risk_misses,high_mix_latent_misses,high_mix_mean_delay,high_mix_p95_delay,high_mix_median_delay,high_mix_mean_final_log_e,fixed_002_alarms,fixed_002_alarm_rate,fixed_002_rate_l95,fixed_002_rate_u95,fixed_002_false_alarms,fixed_002_risk_misses,fixed_002_latent_misses,fixed_002_mean_delay,fixed_002_p95_delay,fixed_002_median_delay,fixed_002_mean_final_log_e,fixed_010_alarms,fixed_010_alarm_rate,fixed_010_rate_l95,fixed_010_rate_u95,fixed_010_false_alarms,fixed_010_risk_misses,fixed_010_latent_misses,fixed_010_mean_delay,fixed_010_p95_delay,fixed_010_median_delay,fixed_010_mean_final_log_e,fixed_025_alarms,fixed_025_alarm_rate,fixed_025_rate_l95,fixed_025_rate_u95,fixed_025_false_alarms,fixed_025_risk_misses,fixed_025_latent_misses,fixed_025_mean_delay,fixed_025_p95_delay,fixed_025_median_delay,fixed_025_mean_final_log_e,fixed_050_alarms,fixed_050_alarm_rate,fixed_050_rate_l95,fixed_050_rate_u95,fixed_050_false_alarms,fixed_050_risk_misses,fixed_050_latent_misses,fixed_050_mean_delay,fixed_050_p95_delay,fixed_050_median_delay,fixed_050_mean_final_log_e,fixed_075_alarms,fixed_075_alarm_rate,fixed_075_rate_l95,fixed_075_rate_u95,fixed_075_false_alarms,fixed_075_risk_misses,fixed_075_latent_misses,fixed_075_mean_delay,fixed_075_p95_delay,fixed_075_median_delay,fixed_075_mean_final_log_e";

#[derive(Clone, Copy, Debug)]
enum Profile {
    Constant(f64),
    LinearRamp { start: f64, end: f64 },
    TransientBurst { base: f64, burst: f64, start: usize, end: usize },
    RecurrentBursts { base: f64, burst: f64, starts: [usize; 3], width: usize },
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    latent_shift: bool,
    risk_violating: bool,
    pure_null: bool,
    profile: Profile,
}

impl Scenario {
    fn expected_loss(self, step: usize) -> f64 {
        match self.profile {
            Profile::Constant(probability) => probability,
            Profile::LinearRamp { start, end } => {
                start + (end - start) * step as f64 / (HORIZON - 1) as f64
            }
            Profile::TransientBurst { base, burst, start, end } => {
                if (start..end).contains(&step) { burst } else { base }
            }
            Profile::RecurrentBursts { base, burst, starts, width } => {
                if starts.iter().any(|start| (*start..(*start + width)).contains(&step)) {
                    burst
                } else {
                    base
                }
            }
        }
    }

    fn mean_expected_loss(self) -> f64 {
        (0..HORIZON)
            .map(|step| self.expected_loss(step))
            .sum::<f64>()
            / HORIZON as f64
    }

    fn max_expected_loss(self) -> f64 {
        (0..HORIZON)
            .map(|step| self.expected_loss(step))
            .fold(0.0_f64, f64::max)
    }
}

const SCENARIOS: [Scenario; 13] = [
    Scenario {
        name: "null_low",
        latent_shift: false,
        risk_violating: false,
        pure_null: true,
        profile: Profile::Constant(0.10),
    },
    Scenario {
        name: "null_boundary",
        latent_shift: false,
        risk_violating: false,
        pure_null: true,
        profile: Profile::Constant(0.15),
    },
    Scenario {
        name: "persistent_0175",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::Constant(0.175),
    },
    Scenario {
        name: "persistent_020",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::Constant(0.20),
    },
    Scenario {
        name: "persistent_025",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::Constant(0.25),
    },
    Scenario {
        name: "persistent_035",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::Constant(0.35),
    },
    Scenario {
        name: "ramp_to_020",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::LinearRamp { start: 0.08, end: 0.20 },
    },
    Scenario {
        name: "ramp_to_030",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::LinearRamp { start: 0.08, end: 0.30 },
    },
    Scenario {
        name: "ramp_to_045",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::LinearRamp { start: 0.08, end: 0.45 },
    },
    Scenario {
        name: "burst_008",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::TransientBurst { base: 0.10, burst: 0.65, start: 64, end: 72 },
    },
    Scenario {
        name: "burst_024",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::TransientBurst { base: 0.10, burst: 0.65, start: 64, end: 88 },
    },
    Scenario {
        name: "recurrent_bursts",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        profile: Profile::RecurrentBursts {
            base: 0.10,
            burst: 0.65,
            starts: [32, 96, 160],
            width: 8,
        },
    },
    Scenario {
        name: "stealth_paired",
        latent_shift: true,
        risk_violating: false,
        pure_null: false,
        profile: Profile::Constant(0.10),
    },
];

#[derive(Clone, Copy, Debug)]
struct DetectorOutcome {
    first_alarm_step: usize,
    final_log_e: f64,
    peak_log_e: f64,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeOutcome {
    seed: u64,
    observed_events: usize,
    mean_observed_loss: f64,
    mean_expected_loss: f64,
    max_expected_loss: f64,
    outcomes: [DetectorOutcome; 7],
    low_component_logs: [f64; 3],
    high_component_logs: [f64; 3],
    low_faster_than_best_fixed: bool,
    high_faster_than_best_fixed: bool,
}

#[derive(Clone, Debug, Default)]
struct DetectorStats {
    alarms: usize,
    false_alarms: usize,
    risk_misses: usize,
    latent_misses: usize,
    detection_steps: Vec<usize>,
    final_log_sum: f64,
}

impl DetectorStats {
    fn observe(&mut self, scenario: Scenario, outcome: DetectorOutcome) {
        let detected = outcome.first_alarm_step < ALARM_SENTINEL;
        self.alarms += usize::from(detected);
        self.false_alarms += usize::from(!scenario.risk_violating && detected);
        self.risk_misses += usize::from(scenario.risk_violating && !detected);
        self.latent_misses += usize::from(scenario.latent_shift && !detected);
        if detected {
            self.detection_steps.push(outcome.first_alarm_step);
        }
        self.final_log_sum += outcome.final_log_e;
    }
}

#[derive(Clone, Debug)]
struct ScenarioSummary {
    episodes: usize,
    mean_expected_sum: f64,
    mean_observed_sum: f64,
    max_expected_loss: f64,
    low_faster_than_best_fixed: usize,
    high_faster_than_best_fixed: usize,
    detectors: [DetectorStats; 7],
}

impl Default for ScenarioSummary {
    fn default() -> Self {
        Self {
            episodes: 0,
            mean_expected_sum: 0.0,
            mean_observed_sum: 0.0,
            max_expected_loss: 0.0,
            low_faster_than_best_fixed: 0,
            high_faster_than_best_fixed: 0,
            detectors: std::array::from_fn(|_| DetectorStats::default()),
        }
    }
}

fn unit_interval(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn draw(seed: u64, step: usize) -> f64 {
    unit_interval(
        seed.wrapping_add((step as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)),
    )
}

fn create_mixture(fractions: [f64; 3]) -> Result<MixtureBoundedRiskEProcess, Box<dyn Error>> {
    MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        fractions,
        GLOBAL_DELTA,
        PARALLEL_STREAMS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid mixture monitor").into())
}

fn create_fixed(fraction: f64) -> Result<BoundedRiskEProcess, Box<dyn Error>> {
    BoundedRiskEProcess::new(
        RISK_BOUND,
        fraction / RISK_BOUND,
        GLOBAL_DELTA,
        PARALLEL_STREAMS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid fixed-bet monitor").into())
}

fn simulate_episode(
    scenario: Scenario,
    episode: usize,
    log_threshold: f64,
) -> Result<EpisodeOutcome, Box<dyn Error>> {
    let seed = 0xA110_57A7_0000_0000_u64.wrapping_add(episode as u64);
    let mut low_mixture = create_mixture(LOW_BET_FRACTIONS)?;
    let mut high_mixture = create_mixture(HIGH_BET_FRACTIONS)?;
    let mut fixed = [
        create_fixed(FIXED_BET_FRACTIONS[0])?,
        create_fixed(FIXED_BET_FRACTIONS[1])?,
        create_fixed(FIXED_BET_FRACTIONS[2])?,
        create_fixed(FIXED_BET_FRACTIONS[3])?,
        create_fixed(FIXED_BET_FRACTIONS[4])?,
    ];

    let mut first_alarm_steps = [ALARM_SENTINEL; 7];
    let mut peak_logs = [0.0_f64; 7];
    let mut observed_events = 0_usize;
    let mut expected_sum = 0.0_f64;
    let mut observed_sum = 0.0_f64;

    for step in 0..HORIZON {
        let expected_loss = scenario.expected_loss(step);
        expected_sum += expected_loss;
        let observed_loss = if draw(seed, step) < expected_loss {
            observed_events += 1;
            1.0
        } else {
            0.0
        };
        observed_sum += observed_loss;

        let alarms = [
            low_mixture.observe(observed_loss),
            high_mixture.observe(observed_loss),
            fixed[0].observe(observed_loss),
            fixed[1].observe(observed_loss),
            fixed[2].observe(observed_loss),
            fixed[3].observe(observed_loss),
            fixed[4].observe(observed_loss),
        ];
        for (index, alarm_result) in alarms.into_iter().enumerate() {
            let alarm = alarm_result.ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "monitor rejected a bounded loss")
            })?;
            if alarm && first_alarm_steps[index] == ALARM_SENTINEL {
                first_alarm_steps[index] = step;
            }
        }

        let logs = [
            low_mixture.log_e_value(),
            high_mixture.log_e_value(),
            fixed[0].log_e_value(),
            fixed[1].log_e_value(),
            fixed[2].log_e_value(),
            fixed[3].log_e_value(),
            fixed[4].log_e_value(),
        ];
        for (index, log_e) in logs.into_iter().enumerate() {
            peak_logs[index] = peak_logs[index].max(log_e);
        }
    }

    if (low_mixture.log_alarm_threshold() - log_threshold).abs() > 1e-10
        || (high_mixture.log_alarm_threshold() - log_threshold).abs() > 1e-10
        || fixed.iter().any(|monitor| {
            (monitor.log_alarm_threshold() - log_threshold).abs() > 1e-10
        })
    {
        return Err(IoError::new(ErrorKind::InvalidData, "detector thresholds diverged").into());
    }

    let best_fixed_step = *first_alarm_steps[2..]
        .iter()
        .min()
        .expect("five fixed comparators");
    Ok(EpisodeOutcome {
        seed,
        observed_events,
        mean_observed_loss: observed_sum / HORIZON as f64,
        mean_expected_loss: expected_sum / HORIZON as f64,
        max_expected_loss: scenario.max_expected_loss(),
        outcomes: std::array::from_fn(|index| DetectorOutcome {
            first_alarm_step: first_alarm_steps[index],
            final_log_e: [
                low_mixture.log_e_value(),
                high_mixture.log_e_value(),
                fixed[0].log_e_value(),
                fixed[1].log_e_value(),
                fixed[2].log_e_value(),
                fixed[3].log_e_value(),
                fixed[4].log_e_value(),
            ][index],
            peak_log_e: peak_logs[index],
        }),
        low_component_logs: low_mixture.component_log_e_values(),
        high_component_logs: high_mixture.component_log_e_values(),
        low_faster_than_best_fixed: first_alarm_steps[0] < best_fixed_step,
        high_faster_than_best_fixed: first_alarm_steps[1] < best_fixed_step,
    })
}

fn detection_interval(detected: usize, episodes: usize) -> (f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    let n = episodes as f64;
    let p = detected as f64 / n;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn percentile(values: &[usize], percentile: usize) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * percentile).div_ceil(100).max(1);
    Some(sorted[rank - 1])
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn optional_number<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.9}"))
}

fn record_summary(
    summary: &mut ScenarioSummary,
    scenario: Scenario,
    outcome: EpisodeOutcome,
) {
    summary.episodes += 1;
    summary.mean_expected_sum += outcome.mean_expected_loss;
    summary.mean_observed_sum += outcome.mean_observed_loss;
    summary.max_expected_loss = summary.max_expected_loss.max(outcome.max_expected_loss);
    summary.low_faster_than_best_fixed += usize::from(outcome.low_faster_than_best_fixed);
    summary.high_faster_than_best_fixed += usize::from(outcome.high_faster_than_best_fixed);
    for (index, detector) in outcome.outcomes.into_iter().enumerate() {
        summary.detectors[index].observe(scenario, detector);
    }
}

fn summary_fields(
    output: &mut String,
    scenario: Scenario,
    summary: &ScenarioSummary,
) {
    let mut fields = vec![
        scenario.name.to_string(),
        scenario.latent_shift.to_string(),
        scenario.risk_violating.to_string(),
        scenario.pure_null.to_string(),
        summary.episodes.to_string(),
        format!("{:.9}", summary.mean_expected_sum / summary.episodes as f64),
        format!("{:.9}", summary.mean_observed_sum / summary.episodes as f64),
        format!("{:.9}", summary.max_expected_loss),
        summary.low_faster_than_best_fixed.to_string(),
        summary.high_faster_than_best_fixed.to_string(),
    ];

    for detector in &summary.detectors {
        let rate = detector.alarms as f64 / summary.episodes as f64;
        let (lower, upper) = detection_interval(detector.alarms, summary.episodes);
        let delay_values: Vec<f64> = detector
            .detection_steps
            .iter()
            .map(|step| *step as f64)
            .collect();
        fields.extend([
            detector.alarms.to_string(),
            format!("{rate:.9}"),
            format!("{lower:.9}"),
            format!("{upper:.9}"),
            detector.false_alarms.to_string(),
            detector.risk_misses.to_string(),
            detector.latent_misses.to_string(),
            optional_float(mean(&delay_values)),
            optional_number(percentile(&detector.detection_steps, 95)),
            optional_number(percentile(&detector.detection_steps, 50)),
            format!("{:.9}", detector.final_log_sum / summary.episodes as f64),
        ]);
    }
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn raw_fields(scenario: Scenario, episode: usize, outcome: EpisodeOutcome, log_threshold: f64) -> String {
    let mut fields = vec![
        scenario.name.to_string(),
        scenario.latent_shift.to_string(),
        scenario.risk_violating.to_string(),
        scenario.pure_null.to_string(),
        episode.to_string(),
        outcome.seed.to_string(),
        HORIZON.to_string(),
        format!("{:.9}", outcome.mean_expected_loss),
        format!("{:.9}", outcome.max_expected_loss),
        format!("{:.9}", outcome.mean_observed_loss),
        outcome.observed_events.to_string(),
    ];
    let low = outcome.outcomes[0];
    let high = outcome.outcomes[1];
    let fixed002 = outcome.outcomes[2];
    let fixed010 = outcome.outcomes[3];
    let fixed025 = outcome.outcomes[4];
    let fixed050 = outcome.outcomes[5];
    let fixed075 = outcome.outcomes[6];
    fields.extend([
        low.first_alarm_step.to_string(),
        (low.first_alarm_step < ALARM_SENTINEL).to_string(),
        format!("{:.9}", low.final_log_e),
        format!("{:.9}", low.peak_log_e),
    ]);
    fields.extend(outcome.low_component_logs.map(|value| format!("{value:.9}")));
    fields.extend([
        high.first_alarm_step.to_string(),
        (high.first_alarm_step < ALARM_SENTINEL).to_string(),
        format!("{:.9}", high.final_log_e),
        format!("{:.9}", high.peak_log_e),
    ]);
    fields.extend(outcome.high_component_logs.map(|value| format!("{value:.9}")));
    for fixed in [fixed002, fixed010, fixed025, fixed050, fixed075] {
        fields.extend([
            fixed.first_alarm_step.to_string(),
            (fixed.first_alarm_step < ALARM_SENTINEL).to_string(),
            format!("{:.9}", fixed.final_log_e),
        ]);
    }
    fields.extend([
        format!("{log_threshold:.9}"),
        format!("{RISK_BOUND:.9}"),
        format!("{GLOBAL_DELTA:.9}"),
        PARALLEL_STREAMS.to_string(),
    ]);
    fields.join(",")
}

fn validate_design() -> Result<(), Box<dyn Error>> {
    if SCENARIOS.len() != 13
        || SCENARIOS.iter().filter(|scenario| scenario.pure_null).count() != 2
        || SCENARIOS.iter().filter(|scenario| scenario.name == "stealth_paired").count() != 1
        || SCENARIOS.iter().any(|scenario| {
            !(0.0..=1.0).contains(&scenario.mean_expected_loss())
                || !(0.0..=1.0).contains(&scenario.max_expected_loss())
                || scenario.max_expected_loss() < RISK_BOUND && scenario.risk_violating
        })
    {
        return Err(IoError::new(ErrorKind::InvalidInput, "shift-surface design is inconsistent").into());
    }
    for step in 0..HORIZON {
        if SCENARIOS[0].expected_loss(step) != SCENARIOS[12].expected_loss(step) {
            return Err(IoError::new(ErrorKind::InvalidInput, "stealth probability profile differs from null").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-eprocess-shift-power-surface <episodes.csv> <summary.csv>",
        )
        .into());
    }
    validate_design()?;
    let threshold = MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        LOW_BET_FRACTIONS,
        GLOBAL_DELTA,
        PARALLEL_STREAMS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid threshold monitor"))?
    .log_alarm_threshold();

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    let mut total_rows = 0_usize;
    for scenario in SCENARIOS {
        let mut summary = ScenarioSummary::default();
        for episode in 0..EPISODES_PER_SCENARIO {
            let outcome = simulate_episode(scenario, episode, threshold)?;
            record_summary(&mut summary, scenario, outcome);
            raw.push_str(&raw_fields(scenario, episode, outcome, threshold));
            raw.push('\n');
            total_rows += 1;
        }
        if summary.episodes != EPISODES_PER_SCENARIO {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!("scenario {} has incorrect episode coverage", scenario.name),
            )
            .into());
        }
        summary_fields(&mut summary_output, scenario, &summary);
    }

    let expected_rows = SCENARIOS.len() * EPISODES_PER_SCENARIO;
    if total_rows != expected_rows || total_rows != 65_000 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 65,000 shift-surface episodes, generated {total_rows}"),
        )
        .into());
    }
    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary_output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_grids_are_predeclared_and_strictly_inside_the_valid_range() {
        assert!(LOW_BET_FRACTIONS.iter().all(|x| *x > 0.0 && *x < 1.0));
        assert!(HIGH_BET_FRACTIONS.iter().all(|x| *x > 0.0 && *x < 1.0));
        assert_ne!(LOW_BET_FRACTIONS, HIGH_BET_FRACTIONS);
        assert_eq!(FIXED_BET_FRACTIONS, [0.02, 0.10, 0.25, 0.50, 0.75]);
    }

    #[test]
    fn nulls_respect_the_declared_conditional_risk_bound() {
        for scenario in SCENARIOS.iter().filter(|x| x.pure_null) {
            assert!(scenario.max_expected_loss() <= RISK_BOUND);
        }
    }

    #[test]
    fn stealth_scenario_is_exactly_the_null_low_probability_profile() {
        let null = SCENARIOS.iter().find(|x| x.name == "null_low").unwrap();
        let stealth = SCENARIOS.iter().find(|x| x.name == "stealth_paired").unwrap();
        for step in 0..HORIZON {
            assert_eq!(null.expected_loss(step), stealth.expected_loss(step));
        }
        assert_ne!(null.latent_shift, stealth.latent_shift);
        assert_eq!(null.risk_violating, stealth.risk_violating);
    }

    #[test]
    fn probability_profiles_stay_bounded_and_flag_detectable_shift() {
        for scenario in SCENARIOS {
            assert!(scenario.mean_expected_loss() >= 0.0);
            assert!(scenario.max_expected_loss() <= 1.0);
        }
        assert!(SCENARIOS.iter().filter(|x| x.name == "persistent_035").all(|x| {
            x.max_expected_loss() > RISK_BOUND && x.risk_violating
        }));
    }

    #[test]
    fn all_detectors_share_the_same_alarm_threshold() {
        let threshold = (PARALLEL_STREAMS as f64 / GLOBAL_DELTA).ln();
        let low = create_mixture(LOW_BET_FRACTIONS).unwrap();
        let high = create_mixture(HIGH_BET_FRACTIONS).unwrap();
        let fixed: Vec<_> = FIXED_BET_FRACTIONS
            .iter()
            .map(|fraction| create_fixed(*fraction).unwrap())
            .collect();
        assert!((low.log_alarm_threshold() - threshold).abs() < 1e-10);
        assert!((high.log_alarm_threshold() - threshold).abs() < 1e-10);
        assert!(fixed.iter().all(|x| (x.log_alarm_threshold() - threshold).abs() < 1e-10));
    }
}
