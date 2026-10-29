use std::process;

const HORIZON: usize = 256;
const CHANGE_AT: usize = 128;
const SEED_START: u64 = 4_242;
const SEED_COUNT: usize = 128;
const NULL_MEAN_REWARD: f64 = 0.8;
const CUSUM_REFERENCE_DRIFT: f64 = 0.3;
const OBSERVATION_FLIP_PROBABILITY: f64 = 0.30;
const SENSOR_SEED_XOR: u64 = 0xD1B5_4A32_D192_ED03;
const DETECTOR_THRESHOLDS: [f64; 7] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
const REPORT_SCHEMA: &str = "cmaq-003-regime-shift-report-v0.2";

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

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Stationary,
    AbruptShift,
    ObservationDrift,
}

impl Lane {
    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary",
            Self::AbruptShift => "abrupt_shift",
            Self::ObservationDrift => "observation_only_drift",
        }
    }

    fn change_at(self) -> Option<usize> {
        match self {
            Self::AbruptShift => Some(CHANGE_AT),
            Self::Stationary | Self::ObservationDrift => None,
        }
    }

    fn observation_drift_at(self) -> Option<usize> {
        match self {
            Self::ObservationDrift => Some(CHANGE_AT),
            Self::Stationary | Self::AbruptShift => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Episode {
    alarm_at_step: Option<usize>,
    reward_total: u64,
    regret_total: u64,
}

fn run_episode(seed: u64, lane: Lane, threshold: Option<f64>) -> Episode {
    let mut rng = SplitMix64::new(seed);
    let mut sensor_rng = SplitMix64::new(seed ^ SENSOR_SEED_XOR);
    let mut statistic = 0.0_f64;
    let mut alarm_at_step = None;
    let mut reward_total = 0_u64;
    let mut regret_total = 0_u64;

    for step in 0..HORIZON {
        // One shared uniform draw is used for the chosen action and oracle action.
        // This common-random-number coupling makes paired regret less noisy.
        let noise = rng.unit_f64();
        let sensor_noise = sensor_rng.unit_f64();
        let changed = lane.change_at().is_some_and(|change_at| step >= change_at);
        let observation_drift = lane
            .observation_drift_at()
            .is_some_and(|drift_at| step >= drift_at);
        let switched = alarm_at_step.is_some_and(|alarm_at| step > alarm_at);
        let chosen_action = if switched { 1_usize } else { 0_usize };

        // Regime A favours action 0; regime B favours action 1. This is a small
        // decision-proxy environment, not a physical safety or causal-discovery claim.
        let chosen_probability = match (changed, chosen_action) {
            (false, 0) | (true, 1) => 0.8,
            (false, 1) | (true, 0) => 0.2,
            _ => unreachable!("the action space is binary"),
        };

        let actual_reward = if noise < chosen_probability { 1_u64 } else { 0_u64 };
        let observed_reward = if observation_drift && sensor_noise < OBSERVATION_FLIP_PROBABILITY {
            1_u64 - actual_reward
        } else {
            actual_reward
        };
        let oracle_reward = if noise < NULL_MEAN_REWARD { 1_u64 } else { 0_u64 };
        reward_total += actual_reward;
        regret_total += oracle_reward.saturating_sub(actual_reward);

        // The first implementation uses a one-sided CUSUM-like detector on
        // reward residuals while action 0 is being used. Low rewards raise the
        // statistic; a single alarm switches the policy to action 1.
        if let Some(limit) = threshold {
            if alarm_at_step.is_none() {
                let increment = NULL_MEAN_REWARD - observed_reward as f64 - CUSUM_REFERENCE_DRIFT;
                statistic = (statistic + increment).max(0.0);
                if statistic + 1e-12 >= limit {
                    alarm_at_step = Some(step);
                }
            }
        }
    }

    Episode {
        alarm_at_step,
        reward_total,
        regret_total,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Metrics {
    episodes: usize,
    false_alarm_episodes: usize,
    pre_shift_false_alarm_episodes: usize,
    false_causal_diagnosis_episodes: usize,
    detection_episodes: usize,
    detection_delays: Vec<usize>,
    mean_regret_per_episode: f64,
    mean_reward_rate: f64,
    false_alarm_rate_per_1000_steps: f64,
}

impl Metrics {
    fn summarize(lane: Lane, threshold: Option<f64>) -> Self {
        let mut false_alarm_episodes = 0usize;
        let mut pre_shift_false_alarm_episodes = 0usize;
        let mut false_causal_diagnosis_episodes = 0usize;
        let mut detection_delays = Vec::new();
        let mut reward_total = 0_u64;
        let mut regret_total = 0_u64;

        for offset in 0..SEED_COUNT {
            let seed = SEED_START + offset as u64;
            let episode = run_episode(seed, lane, threshold);
            reward_total += episode.reward_total;
            regret_total += episode.regret_total;

            match (lane, episode.alarm_at_step) {
                (Lane::Stationary, Some(_)) => false_alarm_episodes += 1,
                (Lane::AbruptShift, Some(step)) if step < CHANGE_AT => {
                    false_alarm_episodes += 1;
                    pre_shift_false_alarm_episodes += 1;
                }
                (Lane::AbruptShift, Some(step)) => {
                    detection_delays.push(step - CHANGE_AT + 1);
                }
                (Lane::ObservationDrift, Some(_)) => {
                    // A detector alarm is interpreted as a causal-regime reset even though
                    // only the observation channel changed in this lane.
                    false_causal_diagnosis_episodes += 1;
                }
                _ => {}
            }
        }

        detection_delays.sort_unstable();
        let steps = (SEED_COUNT * HORIZON) as f64;
        Self {
            episodes: SEED_COUNT,
            false_alarm_episodes,
            pre_shift_false_alarm_episodes,
            false_causal_diagnosis_episodes,
            detection_episodes: detection_delays.len(),
            detection_delays,
            mean_regret_per_episode: regret_total as f64 / SEED_COUNT as f64,
            mean_reward_rate: reward_total as f64 / steps,
            false_alarm_rate_per_1000_steps: false_alarm_episodes as f64 / steps * 1_000.0,
        }
    }

    fn detection_rate(&self) -> f64 {
        self.detection_episodes as f64 / self.episodes as f64
    }

    fn quantile_nearest_rank(&self, quantile: f64) -> Option<usize> {
        if self.detection_delays.is_empty() {
            return None;
        }
        let rank = (quantile * self.detection_delays.len() as f64).ceil() as usize;
        Some(self.detection_delays[rank.saturating_sub(1).min(self.detection_delays.len() - 1)])
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StrategyReport {
    name: String,
    detector_threshold: Option<f64>,
    stationary: Metrics,
    abrupt_shift: Metrics,
    observation_only_drift: Metrics,
}

fn strategy_name(threshold: Option<f64>) -> String {
    match threshold {
        None => "frozen".to_owned(),
        Some(value) => format!("cusum_h{value:.1}").replace('.', "_"),
    }
}

fn strategy_report(threshold: Option<f64>) -> StrategyReport {
    StrategyReport {
        name: strategy_name(threshold),
        detector_threshold: threshold,
        stationary: Metrics::summarize(Lane::Stationary, threshold),
        abrupt_shift: Metrics::summarize(Lane::AbruptShift, threshold),
        observation_only_drift: Metrics::summarize(Lane::ObservationDrift, threshold),
    }
}

fn emit_report(reports: &[StrategyReport]) {
    print!(
        "{{\"schema_version\":\"{}\",\"horizon\":{},\"change_at\":{},\"seed_start\":{},\"seed_count\":{},\"reference_mean_reward\":{:.3},\"cusum_reference_drift\":{:.3},\"strategies\":[",
        REPORT_SCHEMA, HORIZON, CHANGE_AT, SEED_START, SEED_COUNT,
        NULL_MEAN_REWARD, CUSUM_REFERENCE_DRIFT
    );
    for (index, report) in reports.iter().enumerate() {
        if index > 0 {
            print!(",");
        }
        print!(
            "{{\"strategy\":\"{}\",\"detector_threshold\":",
            report.name
        );
        match report.detector_threshold {
            Some(value) => print!("{value:.1}"),
            None => print!("null"),
        }
        print!(",\"stationary\":");
        emit_metrics_inline(&report.stationary, Lane::Stationary);
        print!(",\"abrupt_shift\":");
        emit_metrics_inline(&report.abrupt_shift, Lane::AbruptShift);
        print!(",\"observation_only_drift\":");
        emit_metrics_inline(&report.observation_only_drift, Lane::ObservationDrift);
        print!("}}");
    }
    println!("]}}");
}

fn emit_metrics_inline(metrics: &Metrics, lane: Lane) {
    print!(
        "{{\"lane\":\"{}\",\"episodes\":{},\"false_alarm_episodes\":{},\"false_alarm_rate_per_1000_steps\":{:.6},\"pre_shift_false_alarm_episodes\":{},\"\"false_causal_diagnosis_episodes\":{},detection_episodes\":{},\"detection_rate\":{:.6},\"p50_detection_delay_steps\":",
        lane.name(),
        metrics.episodes,
        metrics.false_alarm_episodes,
        metrics.false_alarm_rate_per_1000_steps,
        metrics.pre_shift_false_alarm_episodes,
        metrics.false_causal_diagnosis_episodes,
        metrics.detection_episodes,
        metrics.detection_rate()
    );
    match metrics.quantile_nearest_rank(0.50) {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
    print!(",\"p90_detection_delay_steps\":");
    match metrics.quantile_nearest_rank(0.90) {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
    print!(
        ",\"mean_regret_per_episode\":{:.6},\"mean_reward_rate\":{:.6}}}",
        metrics.mean_regret_per_episode, metrics.mean_reward_rate
    );
}

fn run() {
    let mut reports = vec![strategy_report(None)];
    reports.extend(
        DETECTOR_THRESHOLDS
            .iter()
            .copied()
            .map(|threshold| strategy_report(Some(threshold))),
    );
    emit_report(&reports);
}

fn main() {
    let result = std::panic::catch_unwind(run);
    if result.is_err() {
        eprintln!("CMAQ-003 smoke harness failed");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_stream_is_reproducible() {
        let mut first = SplitMix64::new(123);
        let mut second = SplitMix64::new(123);
        for _ in 0..512 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn frozen_policy_has_no_regret_in_stationary_regime_a() {
        let metrics = Metrics::summarize(Lane::Stationary, None);
        assert_eq!(metrics.false_alarm_episodes, 0);
        assert_eq!(metrics.mean_regret_per_episode, 0.0);
    }

    #[test]
    fn frozen_policy_accumulates_regret_after_hidden_abrupt_shift() {
        let stationary = Metrics::summarize(Lane::Stationary, None);
        let shifted = Metrics::summarize(Lane::AbruptShift, None);
        assert!(shifted.mean_regret_per_episode > stationary.mean_regret_per_episode);
        assert_eq!(shifted.detection_episodes, 0);
    }

    #[test]
    fn threshold_sweep_exposes_false_alarm_detection_tradeoff() {
        let low_stationary = Metrics::summarize(Lane::Stationary, Some(1.5));
        let high_stationary = Metrics::summarize(Lane::Stationary, Some(3.0));
        let low_shift = Metrics::summarize(Lane::AbruptShift, Some(1.5));
        let high_shift = Metrics::summarize(Lane::AbruptShift, Some(3.0));

        assert!(
            low_stationary.false_alarm_episodes > high_stationary.false_alarm_episodes,
            "lower thresholds should generate more stationary false alarms on this fixed tape"
        );
        assert!(
            low_shift.detection_episodes < high_shift.detection_episodes,
            "lower thresholds should produce more pre-shift false alarms on this fixed tape"
        );
        assert!(
            low_shift.quantile_nearest_rank(0.90).unwrap()
                < high_shift.quantile_nearest_rank(0.90).unwrap(),
            "among detected episodes, the more sensitive threshold should detect faster"
        );
    }

    #[test]
    fn observation_only_drift_is_not_counted_as_a_true_causal_detection() {
        let frozen = Metrics::summarize(Lane::ObservationDrift, None);
        let detected = Metrics::summarize(Lane::ObservationDrift, Some(3.0));
        assert_eq!(frozen.false_causal_diagnosis_episodes, 0);
        assert!(detected.false_causal_diagnosis_episodes > 0);
        assert_eq!(detected.detection_episodes, 0);
    }

    #[test]
    fn delay_quantiles_are_null_when_no_changes_are_detected() {
        let metrics = Metrics::summarize(Lane::AbruptShift, None);
        assert_eq!(metrics.quantile_nearest_rank(0.90), None);
    }

    #[test]
    fn episode_replay_is_deterministic() {
        let a = run_episode(4_242, Lane::AbruptShift, Some(2.5));
        let b = run_episode(4_242, Lane::AbruptShift, Some(2.5));
        assert_eq!(a, b);
    }
}
