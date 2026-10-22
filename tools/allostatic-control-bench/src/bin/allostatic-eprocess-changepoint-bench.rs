//! Paired benchmark of a global bounded-risk e-process and an alpha-spent
//! multi-start monitor when risk increases late in a stream.
//!
//! Usage: allostatic-eprocess-changepoint-bench <episodes.csv> <summary.csv>
//! All methods see the same deterministic bounded-loss sequence for a given
//! episode number across scenario profiles. The alpha-spent detector uses a
//! fixed recent-start window; it never reuses the allocation of expired starts.

use allostatic_control_bench::alpha_spent_changepoint::{
    AlphaSpentChangePointMonitor, ChangePointAlarm,
};
use allostatic_control_bench::bounded_risk_eprocess::MixtureBoundedRiskEProcess;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const HORIZON: usize = 256;
const EPISODES_PER_SCENARIO: usize = 500;
const RISK_BOUND: f64 = 0.15;
const GLOBAL_DELTA: f64 = 0.05;
const SIMULTANEOUS_STREAMS: usize = 5;
const BET_FRACTIONS: [f64; 3] = [0.02, 0.10, 0.25];
const CANDIDATE_WINDOW: usize = 64;
const NO_ALARM_STEP: usize = HORIZON;

const RAW_HEADER: &str = "scenario,latent_shift,risk_violating,pure_null,episode,seed,horizon,change_start,baseline_probability,post_shift_probability,mean_expected_loss,mean_observed_loss,observed_events,global_first_alarm_step,global_detected,global_final_log_e,global_peak_log_e,changepoint_detected_at,changepoint_detected,estimated_change_start,changepoint_log_e,changepoint_log_threshold,changepoint_evidence_excess,global_detection_delay,changepoint_detection_delay,change_start_error,changepoint_within_8_steps,peak_active_candidates,global_log_threshold,total_delta,simultaneous_streams,candidate_window";
const SUMMARY_HEADER: &str = "scenario,latent_shift,risk_violating,pure_null,episodes,change_start,baseline_probability,post_shift_probability,mean_expected_loss,mean_observed_loss,global_alarms,global_alarm_rate,global_rate_l95,global_rate_u95,global_false_alarms,global_risk_misses,global_latent_misses,global_pre_change_alarms,global_mean_detection_step,global_p95_detection_step,global_median_detection_step,global_mean_detection_delay,global_mean_final_log_e,changepoint_alarms,changepoint_alarm_rate,changepoint_rate_l95,changepoint_rate_u95,changepoint_false_alarms,changepoint_risk_misses,changepoint_latent_misses,changepoint_pre_change_alarms,changepoint_mean_detection_step,changepoint_p95_detection_step,changepoint_median_detection_step,changepoint_mean_detection_delay,changepoint_mean_start_error,changepoint_within_8_steps,changepoint_mean_log_e,changepoint_mean_log_threshold,peak_active_candidates";

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    latent_shift: bool,
    risk_violating: bool,
    pure_null: bool,
    baseline_probability: f64,
    post_shift_probability: f64,
    change_start: Option<usize>,
}

impl Scenario {
    fn expected_loss(self, step: usize) -> f64 {
        if self.change_start.is_some_and(|start| step >= start) {
            self.post_shift_probability
        } else {
            self.baseline_probability
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

const SCENARIOS: [Scenario; 7] = [
    Scenario {
        name: "null_low",
        latent_shift: false,
        risk_violating: false,
        pure_null: true,
        baseline_probability: 0.10,
        post_shift_probability: 0.10,
        change_start: None,
    },
    Scenario {
        name: "null_boundary",
        latent_shift: false,
        risk_violating: false,
        pure_null: true,
        baseline_probability: 0.15,
        post_shift_probability: 0.15,
        change_start: None,
    },
    Scenario {
        name: "early_shift_050",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        baseline_probability: 0.02,
        post_shift_probability: 0.50,
        change_start: Some(16),
    },
    Scenario {
        name: "late_shift_050",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        baseline_probability: 0.02,
        post_shift_probability: 0.50,
        change_start: Some(192),
    },
    Scenario {
        name: "late_shift_065",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        baseline_probability: 0.02,
        post_shift_probability: 0.65,
        change_start: Some(192),
    },
    Scenario {
        name: "very_late_shift_065",
        latent_shift: true,
        risk_violating: true,
        pure_null: false,
        baseline_probability: 0.02,
        post_shift_probability: 0.65,
        change_start: Some(224),
    },
    Scenario {
        name: "stealth_paired",
        latent_shift: true,
        risk_violating: false,
        pure_null: false,
        baseline_probability: 0.10,
        post_shift_probability: 0.10,
        change_start: None,
    },
];

#[derive(Clone, Copy, Debug)]
struct EpisodeOutcome {
    seed: u64,
    observed_events: usize,
    mean_expected_loss: f64,
    mean_observed_loss: f64,
    global_first_alarm_step: usize,
    global_final_log_e: f64,
    global_peak_log_e: f64,
    changepoint_alarm: Option<ChangePointAlarm>,
    peak_active_candidates: usize,
}

#[derive(Clone, Debug, Default)]
struct DetectorStats {
    alarms: usize,
    false_alarms: usize,
    risk_misses: usize,
    latent_misses: usize,
    pre_change_alarms: usize,
    detection_steps: Vec<usize>,
    detection_delays: Vec<usize>,
    final_log_values: Vec<f64>,
    start_errors: Vec<usize>,
    within_8_steps: usize,
    log_e_at_detection: Vec<f64>,
    log_threshold_at_detection: Vec<f64>,
    peak_active_candidates: usize,
}

#[derive(Clone, Debug, Default)]
struct ScenarioSummary {
    episodes: usize,
    expected_loss_sum: f64,
    observed_loss_sum: f64,
    global: DetectorStats,
    changepoint: DetectorStats,
}

impl DetectorStats {
    fn observe_global(
        &mut self,
        scenario: Scenario,
        first_alarm_step: usize,
        final_log_e: f64,
    ) {
        let detected = first_alarm_step < NO_ALARM_STEP;
        let successful = scenario.change_start.is_some_and(|start| {
            detected && first_alarm_step >= start
        });
        let pre_change = scenario
            .change_start
            .is_some_and(|start| detected && first_alarm_step < start);
        let false_alarm = pre_change || (!scenario.risk_violating && detected);

        self.alarms += usize::from(detected);
        self.false_alarms += usize::from(false_alarm);
        self.risk_misses += usize::from(scenario.risk_violating && !successful);
        self.latent_misses += usize::from(
            scenario.latent_shift && (scenario.change_start.is_none() || !successful),
        );
        self.pre_change_alarms += usize::from(pre_change);
        if detected {
            self.detection_steps.push(first_alarm_step);
        }
        if successful {
            self.detection_delays.push(
                first_alarm_step - scenario.change_start.expect("successful start"),
            );
        }
        self.final_log_values.push(final_log_e);
    }

    fn observe_changepoint(
        &mut self,
        scenario: Scenario,
        alarm: Option<ChangePointAlarm>,
        peak_candidates: usize,
    ) {
        self.peak_active_candidates = self.peak_active_candidates.max(peak_candidates);
        let detected = alarm.is_some();
        let detected_at = alarm.map_or(NO_ALARM_STEP, |value| value.detected_at() as usize);
        let successful = scenario.change_start.is_some_and(|start| {
            detected && detected_at >= start
        });
        let pre_change = scenario
            .change_start
            .is_some_and(|start| detected && detected_at < start);
        let false_alarm = pre_change || (!scenario.risk_violating && detected);

        self.alarms += usize::from(detected);
        self.false_alarms += usize::from(false_alarm);
        self.risk_misses += usize::from(scenario.risk_violating && !successful);
        self.latent_misses += usize::from(
            scenario.latent_shift && (scenario.change_start.is_none() || !successful),
        );
        self.pre_change_alarms += usize::from(pre_change);
        if detected {
            self.detection_steps.push(detected_at);
            let alarm = alarm.expect("detected alarm");
            self.log_e_at_detection.push(alarm.log_e_value());
            self.log_threshold_at_detection.push(alarm.log_threshold());
        }
        if successful {
            let alarm = alarm.expect("successful alarm");
            let true_start = scenario.change_start.expect("successful start") as u64;
            let error = alarm.start_index().abs_diff(true_start) as usize;
            self.start_errors.push(error);
            self.within_8_steps += usize::from(error <= 8);
            self.detection_delays.push(detected_at - true_start as usize);
        }
    }
}

fn splitmix64(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn draw(seed: u64, step: usize) -> f64 {
    splitmix64(
        seed.wrapping_add((step as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)),
    )
}

fn make_global() -> Result<MixtureBoundedRiskEProcess, Box<dyn Error>> {
    MixtureBoundedRiskEProcess::new(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        SIMULTANEOUS_STREAMS,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid global e-process").into())
}

fn make_changepoint() -> Result<AlphaSpentChangePointMonitor, Box<dyn Error>> {
    AlphaSpentChangePointMonitor::new(
        RISK_BOUND,
        BET_FRACTIONS,
        GLOBAL_DELTA,
        SIMULTANEOUS_STREAMS,
        CANDIDATE_WINDOW,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid alpha-spent bank").into())
}

fn simulate_episode(
    scenario: Scenario,
    episode: usize,
    global_log_threshold: f64,
) -> Result<EpisodeOutcome, Box<dyn Error>> {
    // Same episode seed and step hash across scenarios make null/stealth rows
    // exactly paired, and make changes compare on common random numbers.
    let seed = 0xC4A_06E0_0000_0000_u64.wrapping_add(episode as u64);
    let mut global = make_global()?;
    let mut changepoint = make_changepoint()?;
    let mut global_first_alarm_step = NO_ALARM_STEP;
    let mut global_peak_log_e = 0.0_f64;
    let mut peak_active_candidates = 0_usize;
    let mut expected_sum = 0.0_f64;
    let mut observed_sum = 0.0_f64;
    let mut observed_events = 0_usize;
    let mut changepoint_alarm = None;

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

        if global.observe(observed_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "global e-process rejected bounded loss")
        })? && global_first_alarm_step == NO_ALARM_STEP
        {
            global_first_alarm_step = step;
        }
        global_peak_log_e = global_peak_log_e.max(global.log_e_value());

        let update = changepoint.observe(observed_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "change-point monitor rejected bounded loss")
        })?;
        peak_active_candidates = peak_active_candidates.max(update.active_candidates());
        if let Some(alarm) = update.new_alarm() {
            changepoint_alarm = Some(alarm);
        }
    }

    if (global.log_alarm_threshold() - global_log_threshold).abs() > 1e-10 {
        return Err(IoError::new(ErrorKind::InvalidData, "global threshold mismatch").into());
    }

    Ok(EpisodeOutcome {
        seed,
        observed_events,
        mean_expected_loss: expected_sum / HORIZON as f64,
        mean_observed_loss: observed_sum / HORIZON as f64,
        global_first_alarm_step,
        global_final_log_e: global.log_e_value(),
        global_peak_log_e,
        changepoint_alarm,
        peak_active_candidates,
    })
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z * Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_usize(values: &[usize]) -> Option<f64> {
    (!values.is_empty())
        .then(|| values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64)
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

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.9}"))
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn csv_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn summary_fields(output: &mut String, scenario: Scenario, summary: &ScenarioSummary) {
    let episodes = summary.episodes as f64;
    let shift_start = optional_usize(scenario.change_start);
    let mut fields = vec![
        scenario.name.to_string(),
        csv_bool(scenario.latent_shift).to_string(),
        csv_bool(scenario.risk_violating).to_string(),
        csv_bool(scenario.pure_null).to_string(),
        summary.episodes.to_string(),
        shift_start,
        format!("{:.9}", scenario.baseline_probability),
        format!("{:.9}", scenario.post_shift_probability),
        format!("{:.9}", summary.expected_loss_sum / episodes),
        format!("{:.9}", summary.observed_loss_sum / episodes),
    ];

    for (is_global, detector) in [
        (true, &summary.global),
        (false, &summary.changepoint),
    ] {
        let rate = detector.alarms as f64 / episodes;
        let (lower, upper) = wilson_interval(detector.alarms, summary.episodes);
        fields.extend([
            detector.alarms.to_string(),
            format!("{rate:.9}"),
            format!("{lower:.9}"),
            format!("{upper:.9}"),
            detector.false_alarms.to_string(),
            detector.risk_misses.to_string(),
            detector.latent_misses.to_string(),
            detector.pre_change_alarms.to_string(),
            optional_float(mean_usize(&detector.detection_steps)),
            optional_usize(percentile(&detector.detection_steps, 95)),
            optional_usize(percentile(&detector.detection_steps, 50)),
            optional_float(mean_usize(&detector.detection_delays)),
        ]);
        if is_global {
            fields.push(optional_float(mean_f64(&detector.final_log_values)));
        } else {
            fields.extend([
                optional_float(mean_usize(&detector.start_errors)),
                detector.within_8_steps.to_string(),
                optional_float(mean_f64(&detector.log_e_at_detection)),
                optional_float(mean_f64(&detector.log_threshold_at_detection)),
                detector.peak_active_candidates.to_string(),
            ]);
        }
    }

    output.push_str(&fields.join(","));
    output.push('\n');
}

fn raw_fields(
    scenario: Scenario,
    episode: usize,
    outcome: EpisodeOutcome,
    global_log_threshold: f64,
) -> String {
    let alarm = outcome.changepoint_alarm;
    let (cp_at, cp_start, cp_log_e, cp_threshold, cp_excess) = match alarm {
        Some(value) => (
            value.detected_at() as usize,
            optional_usize(Some(value.start_index() as usize)),
            optional_float(Some(value.log_e_value())),
            optional_float(Some(value.log_threshold())),
            optional_float(Some(value.evidence_excess())),
        ),
        None => (
            NO_ALARM_STEP,
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
            "NA".to_string(),
        ),
    };

    let global_delay = scenario
        .change_start
        .filter(|start| {
            outcome.global_first_alarm_step < HORIZON && outcome.global_first_alarm_step >= *start
        })
        .map(|start| outcome.global_first_alarm_step - start);
    let cp_delay = scenario.change_start
        .filter(|start| alarm.is_some_and(|value| value.detected_at() as usize >= *start))
        .map(|start| cp_at - start);
    let cp_start_error = scenario.change_start.and_then(|start| {
        alarm
            .filter(|value| value.detected_at() as usize >= start)
            .map(|value| (value.start_index() as usize).abs_diff(start))
    });
    let cp_within_8 = cp_start_error.map(|error| error <= 8);

    let fields = [
        scenario.name.to_string(),
        csv_bool(scenario.latent_shift).to_string(),
        csv_bool(scenario.risk_violating).to_string(),
        csv_bool(scenario.pure_null).to_string(),
        episode.to_string(),
        outcome.seed.to_string(),
        HORIZON.to_string(),
        optional_usize(scenario.change_start),
        format!("{:.9}", scenario.baseline_probability),
        format!("{:.9}", scenario.post_shift_probability),
        format!("{:.9}", outcome.mean_expected_loss),
        format!("{:.9}", outcome.mean_observed_loss),
        outcome.observed_events.to_string(),
        outcome.global_first_alarm_step.to_string(),
        csv_bool(outcome.global_first_alarm_step < HORIZON).to_string(),
        format!("{:.9}", outcome.global_final_log_e),
        format!("{:.9}", outcome.global_peak_log_e),
        cp_at.to_string(),
        csv_bool(alarm.is_some()).to_string(),
        cp_start,
        cp_log_e,
        cp_threshold,
        cp_excess,
        optional_usize(global_delay),
        optional_usize(cp_delay),
        optional_usize(cp_start_error),
        cp_within_8.map_or_else(|| "NA".to_string(), |value| csv_bool(value).to_string()),
        outcome.peak_active_candidates.to_string(),
        format!("{global_log_threshold:.9}"),
        format!("{GLOBAL_DELTA:.9}"),
        SIMULTANEOUS_STREAMS.to_string(),
        CANDIDATE_WINDOW.to_string(),
    ];
    fields.join(",")
}

fn validate_design() -> Result<(), Box<dyn Error>> {
    if SCENARIOS.len() != 7
        || SCENARIOS.iter().filter(|scenario| scenario.pure_null).count() != 2
        || SCENARIOS.iter().filter(|scenario| scenario.name == "stealth_paired").count() != 1
        || SCENARIOS.iter().any(|scenario| {
            !(0.0..=1.0).contains(&scenario.baseline_probability)
                || !(0.0..=1.0).contains(&scenario.post_shift_probability)
                || scenario.risk_violating
                    && scenario.max_expected_loss() <= RISK_BOUND
        })
    {
        return Err(
            IoError::new(ErrorKind::InvalidInput, "change-point benchmark design is inconsistent")
                .into(),
        );
    }
    for step in 0..HORIZON {
        if SCENARIOS[0].expected_loss(step) != SCENARIOS[6].expected_loss(step) {
            return Err(
                IoError::new(ErrorKind::InvalidInput, "stealth profile is not paired to null low")
                    .into(),
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-eprocess-changepoint-bench <episodes.csv> <summary.csv>",
        )
        .into());
    }
    validate_design()?;
    let global_log_threshold = make_global()?.log_alarm_threshold();
    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    let mut row_count = 0_usize;

    for scenario in SCENARIOS {
        let mut summary = ScenarioSummary::default();
        for episode in 0..EPISODES_PER_SCENARIO {
            let outcome = simulate_episode(scenario, episode, global_log_threshold)?;
            summary.episodes += 1;
            summary.expected_loss_sum += outcome.mean_expected_loss;
            summary.observed_loss_sum += outcome.mean_observed_loss;
            summary.global.observe_global(
                scenario,
                outcome.global_first_alarm_step,
                outcome.global_final_log_e,
            );
            summary.changepoint.observe_changepoint(
                scenario,
                outcome.changepoint_alarm,
                outcome.peak_active_candidates,
            );
            raw.push_str(&raw_fields(scenario, episode, outcome, global_log_threshold));
            raw.push('\n');
            row_count += 1;
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
    if row_count != expected_rows || row_count != 3500 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 3,500 episode rows, generated {row_count}"),
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
    fn design_pairs_stealth_and_covers_late_changes() {
        validate_design().expect("valid design");
        let stealth = SCENARIOS
            .iter()
            .find(|scenario| scenario.name == "stealth_paired")
            .unwrap();
        let null = SCENARIOS
            .iter()
            .find(|scenario| scenario.name == "null_low")
            .unwrap();
        assert!((0..HORIZON).all(|step| stealth.expected_loss(step) == null.expected_loss(step)));
        assert_eq!(
            SCENARIOS
                .iter()
                .filter(|scenario| scenario.change_start == Some(192))
                .count(),
            2
        );
    }

    #[test]
    fn late_quiet_then_high_loss_reaches_alpha_spent_monitor() {
        let mut monitor = make_changepoint().expect("valid detector");
        for _ in 0..192 {
            let update = monitor.observe(0.0).expect("valid quiet observation");
            assert!(update.new_alarm().is_none());
        }
        let mut found = None;
        for _ in 0..64 {
            if let Some(update) = monitor.observe(1.0) {
                if update.new_alarm().is_some() {
                    found = update.new_alarm();
                    break;
                }
            }
        }
        assert!(found.is_some());
        assert!(found.unwrap().start_index() >= 192);
    }

    #[test]
    fn global_e_process_and_change_point_monitor_share_the_null_budget_metadata() {
        let global = make_global().expect("valid global process");
        let mut changepoint = make_changepoint().expect("valid change-point process");
        assert!((global.log_alarm_threshold() - 100.0_f64.ln()).abs() < 1e-10);
        assert_eq!(changepoint.risk_bound(), RISK_BOUND);
        assert_eq!(changepoint.bet_fractions(), BET_FRACTIONS);
        assert_eq!(changepoint.total_delta(), GLOBAL_DELTA);
        assert_eq!(changepoint.simultaneous_streams(), SIMULTANEOUS_STREAMS);
        assert_eq!(changepoint.max_active_starts(), CANDIDATE_WINDOW);
    }
}
