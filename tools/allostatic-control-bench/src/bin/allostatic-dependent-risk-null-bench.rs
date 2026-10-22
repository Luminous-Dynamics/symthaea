//! Deterministic Monte Carlo falsification of bounded-risk e-processes under
//! dependent Markov loss streams.
//!
//! Usage: allostatic-dependent-risk-null-bench <raw.csv> <summary.csv>
//! Null regime probabilities are at or below the declared risk bound in every
//! latent state. Shifted streams violate that bound after a predeclared change
//! point. Each strategy receives the same per-stream delta in its own comparison.

use allostatic_control_bench::bounded_risk_eprocess::MixtureBoundedRiskEProcess;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const TRIALS: usize = 3_000;
const TOTAL_STEPS: usize = 400;
const EPISODE_LENGTH: usize = 20;
const EPISODES: usize = TOTAL_STEPS / EPISODE_LENGTH;
const SHIFT_STEP: usize = 133;
const RISK_BOUND: f64 = 0.15;
const GLOBAL_DELTA: f64 = 0.05;
const HIGH_BETS: [f64; 3] = [0.25, 0.50, 0.75];
const LOW_BETS: [f64; 3] = [0.03, 0.08, 0.15];
const WILSON_Z: f64 = 1.959_963_984_540_054;

const RAW_HEADER: &str = "condition,trial,seed,loss_events,high_regime_steps,regime_transitions,loss_trace_hex,regime_trace_hex,high_persistent_alarm,high_persistent_first_step,high_reset_any_alarm,high_reset_first_step,high_reset_alarm_episodes,low_persistent_alarm,low_persistent_first_step,low_reset_any_alarm,low_reset_first_step,low_reset_alarm_episodes";
const SUMMARY_HEADER: &str = "condition,trials,mean_loss_rate,mean_high_regime_fraction,mean_regime_transitions,high_persistent_alarms,high_persistent_alarm_rate,high_persistent_ci95_low,high_persistent_ci95_high,high_persistent_false_alarms,high_persistent_missed_shifts,high_persistent_mean_detection_step,high_reset_trial_alarms,high_reset_alarm_rate,high_reset_ci95_low,high_reset_ci95_high,high_reset_false_alarms,high_reset_missed_shifts,high_reset_mean_detection_step,high_reset_mean_alarm_episodes,low_persistent_alarms,low_persistent_alarm_rate,low_persistent_ci95_low,low_persistent_ci95_high,low_persistent_false_alarms,low_persistent_missed_shifts,low_persistent_mean_detection_step,low_reset_trial_alarms,low_reset_alarm_rate,low_reset_ci95_low,low_reset_ci95_high,low_reset_false_alarms,low_reset_missed_shifts,low_reset_mean_detection_step,low_reset_mean_alarm_episodes,global_delta,risk_bound,shift_step";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    ConditionalRiskNull,
    ShiftedRisk,
}

impl Condition {
    const ALL: [Self; 2] = [Self::ConditionalRiskNull, Self::ShiftedRisk];

    const fn name(self) -> &'static str {
        match self {
            Self::ConditionalRiskNull => "conditional_risk_null",
            Self::ShiftedRisk => "shifted_risk",
        }
    }

    const fn is_shifted(self) -> bool {
        matches!(self, Self::ShiftedRisk)
    }

}

#[derive(Clone, Copy, Debug, Default)]
struct StrategyOutcome {
    first_alarm_step: Option<usize>,
    alarm_episodes: usize,
}

impl StrategyOutcome {
    fn alarmed(self) -> bool {
        self.first_alarm_step.is_some()
    }
}

#[derive(Clone, Debug)]
struct TrialOutcome {
    condition: Condition,
    trial: usize,
    seed: u64,
    loss_events: usize,
    high_regime_steps: usize,
    regime_transitions: usize,
    losses: Vec<bool>,
    regimes: Vec<bool>,
    high_persistent: StrategyOutcome,
    high_reset: StrategyOutcome,
    low_persistent: StrategyOutcome,
    low_reset: StrategyOutcome,
}

#[derive(Clone, Debug, Default)]
struct StrategySummary {
    alarms: usize,
    false_alarms: usize,
    missed_shifts: usize,
    detected_steps: Vec<usize>,
    reset_alarm_episodes: usize,
}

impl StrategySummary {
    fn observe(&mut self, condition: Condition, outcome: StrategyOutcome) {
        self.alarms += usize::from(outcome.alarmed());
        self.false_alarms +=
            usize::from(!condition.is_shifted() && outcome.alarmed());
        self.missed_shifts +=
            usize::from(condition.is_shifted() && !outcome.alarmed());
        if let Some(step) = outcome.first_alarm_step {
            self.detected_steps.push(step);
        }
        self.reset_alarm_episodes += outcome.alarm_episodes;
    }

    fn alarm_rate(&self) -> f64 {
        self.alarms as f64 / TRIALS as f64
    }

    fn wilson_interval(&self) -> (f64, f64) {
        let n = TRIALS as f64;
        let proportion = self.alarm_rate();
        let z2 = WILSON_Z * WILSON_Z;
        let denominator = 1.0 + z2 / n;
        let center = (proportion + z2 / (2.0 * n)) / denominator;
        let half = WILSON_Z / denominator
            * (proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)).sqrt();
        ((center - half).max(0.0), (center + half).min(1.0))
    }

    fn mean_detection_step(&self) -> Option<f64> {
        (!self.detected_steps.is_empty()).then(|| {
            self.detected_steps.iter().map(|step| *step as f64).sum::<f64>()
                / self.detected_steps.len() as f64
        })
    }
}

#[derive(Clone, Debug, Default)]
struct ConditionSummary {
    trials: usize,
    loss_events: usize,
    high_regime_steps: usize,
    regime_transitions: usize,
    high_persistent: StrategySummary,
    high_reset: StrategySummary,
    low_persistent: StrategySummary,
    low_reset: StrategySummary,
}

impl ConditionSummary {
    fn observe(&mut self, trial: &TrialOutcome) {
        self.trials += 1;
        self.loss_events += trial.loss_events;
        self.high_regime_steps += trial.high_regime_steps;
        self.regime_transitions += trial.regime_transitions;
        self.high_persistent
            .observe(trial.condition, trial.high_persistent);
        self.high_reset.observe(trial.condition, trial.high_reset);
        self.low_persistent
            .observe(trial.condition, trial.low_persistent);
        self.low_reset.observe(trial.condition, trial.low_reset);
    }
}

fn next_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn make_monitor(
    bet_fractions: [f64; 3],
    episode_index: Option<u64>,
) -> Result<MixtureBoundedRiskEProcess, Box<dyn Error>> {
    let monitor = if let Some(index) = episode_index {
        MixtureBoundedRiskEProcess::new_with_alpha_spending(
            RISK_BOUND,
            bet_fractions,
            GLOBAL_DELTA,
            1,
            index,
        )
    } else {
        MixtureBoundedRiskEProcess::new(RISK_BOUND, bet_fractions, GLOBAL_DELTA, 1)
    };
    monitor.ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid monitor configuration").into())
}

fn to_hex_trace(bits: &[bool]) -> String {
    let mut output = String::with_capacity(bits.len() / 4);
    for chunk in bits.chunks_exact(4) {
        let mut nibble = 0_u8;
        for (bit_index, set) in chunk.iter().copied().enumerate() {
            if set {
                nibble |= 1 << (3 - bit_index);
            }
        }
        output.push(char::from_digit(u32::from(nibble), 16).expect("four-bit nibble"));
    }
    output
}

fn run_trial(condition: Condition, trial: usize) -> Result<TrialOutcome, Box<dyn Error>> {
    // Pair null and shifted controls on the same random innovations; only the
    // post-change conditional loss probability differs.
    let seed = 700_000 + trial as u64 * 7_919;
    let mut random_state = seed;
    let mut high_regime = false;
    let mut previous_regime = false;
    let mut loss_events = 0_usize;
    let mut high_regime_steps = 0_usize;
    let mut regime_transitions = 0_usize;
    let mut losses = Vec::with_capacity(TOTAL_STEPS);
    let mut regimes = Vec::with_capacity(TOTAL_STEPS);

    let mut high_persistent = make_monitor(HIGH_BETS, None)?;
    let mut low_persistent = make_monitor(LOW_BETS, None)?;
    let mut high_reset = make_monitor(HIGH_BETS, Some(1))?;
    let mut low_reset = make_monitor(LOW_BETS, Some(1))?;

    let mut high_persistent_result = StrategyOutcome::default();
    let mut low_persistent_result = StrategyOutcome::default();
    let mut high_reset_result = StrategyOutcome::default();
    let mut low_reset_result = StrategyOutcome::default();

    for step in 0..TOTAL_STEPS {
        let transition_roll = next_uniform(&mut random_state);
        previous_regime = high_regime;
        if !high_regime && transition_roll < 0.02 {
            high_regime = true;
        } else if high_regime && transition_roll < 0.05 {
            high_regime = false;
        }
        regime_transitions += usize::from(previous_regime != high_regime);
        high_regime_steps += usize::from(high_regime);
        regimes.push(high_regime);

        let mut loss_probability = if high_regime { 0.15 } else { 0.13 };
        if condition.is_shifted() && step >= SHIFT_STEP {
            loss_probability = if high_regime { 0.48 } else { 0.30 };
        }
        let loss = next_uniform(&mut random_state) < loss_probability;
        losses.push(loss);
        loss_events += usize::from(loss);
        let bounded_loss = if loss { 1.0 } else { 0.0 };

        let high_persistent_alarm = high_persistent.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "high-bet persistent monitor rejected loss")
        })?;
        let low_persistent_alarm = low_persistent.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "low-bet persistent monitor rejected loss")
        })?;
        if high_persistent_alarm && high_persistent_result.first_alarm_step.is_none() {
            high_persistent_result.first_alarm_step = Some(step);
        }
        if low_persistent_alarm && low_persistent_result.first_alarm_step.is_none() {
            low_persistent_result.first_alarm_step = Some(step);
        }

        if step % EPISODE_LENGTH == 0 && step > 0 {
            let next_episode_index = (step / EPISODE_LENGTH + 1) as u64;
            high_reset = make_monitor(HIGH_BETS, Some(next_episode_index))?;
            low_reset = make_monitor(LOW_BETS, Some(next_episode_index))?;
        }

        let high_reset_was_alarmed = high_reset.alarmed();
        let high_reset_alarm = high_reset.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "high-bet reset monitor rejected loss")
        })?;
        if high_reset_alarm && !high_reset_was_alarmed {
            high_reset_result.alarm_episodes += 1;
            if high_reset_result.first_alarm_step.is_none() {
                high_reset_result.first_alarm_step = Some(step);
            }
        }

        let low_reset_was_alarmed = low_reset.alarmed();
        let low_reset_alarm = low_reset.observe(bounded_loss).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "low-bet reset monitor rejected loss")
        })?;
        if low_reset_alarm && !low_reset_was_alarmed {
            low_reset_result.alarm_episodes += 1;
            if low_reset_result.first_alarm_step.is_none() {
                low_reset_result.first_alarm_step = Some(step);
            }
        }
    }

    Ok(TrialOutcome {
        condition,
        trial,
        seed,
        loss_events,
        high_regime_steps,
        regime_transitions,
        losses,
        regimes,
        high_persistent: high_persistent_result,
        high_reset: high_reset_result,
        low_persistent: low_persistent_result,
        low_reset: low_reset_result,
    })
}

fn csv_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn optional_step(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |step| step.to_string())
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.6}"))
}

fn summary_row(output: &mut String, condition: Condition, summary: &ConditionSummary) {
    let mean_loss_rate =
        summary.loss_events as f64 / (summary.trials * TOTAL_STEPS) as f64;
    let mean_high_regime_fraction =
        summary.high_regime_steps as f64 / (summary.trials * TOTAL_STEPS) as f64;
    let mean_regime_transitions =
        summary.regime_transitions as f64 / summary.trials as f64;
    let mut fields = vec![
        condition.name().to_string(),
        summary.trials.to_string(),
        format!("{mean_loss_rate:.9}"),
        format!("{mean_high_regime_fraction:.9}"),
        format!("{mean_regime_transitions:.6}"),
    ];

    for (index, strategy) in [
        &summary.high_persistent,
        &summary.high_reset,
        &summary.low_persistent,
        &summary.low_reset,
    ]
    .into_iter()
    .enumerate()
    {
        let (ci_low, ci_high) = strategy.wilson_interval();
        fields.push(strategy.alarms.to_string());
        fields.push(format!("{:.9}", strategy.alarm_rate()));
        fields.push(format!("{ci_low:.9}"));
        fields.push(format!("{ci_high:.9}"));
        fields.push(strategy.false_alarms.to_string());
        fields.push(strategy.missed_shifts.to_string());
        fields.push(optional_float(strategy.mean_detection_step()));
        if index == 1 || index == 3 {
            fields.push(format!(
                "{:.6}",
                strategy.reset_alarm_episodes as f64 / summary.trials as f64
            ));
        }
    }
    fields.push(format!("{GLOBAL_DELTA:.6}"));
    fields.push(format!("{RISK_BOUND:.6}"));
    fields.push(SHIFT_STEP.to_string());
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-dependent-risk-null-bench <raw.csv> <summary.csv>",
        )
        .into());
    }

    let mut raw_output = format!("{RAW_HEADER}\n");
    let mut summaries: Vec<(Condition, ConditionSummary)> = Condition::ALL
        .iter()
        .copied()
        .map(|condition| (condition, ConditionSummary::default()))
        .collect();

    for (condition, summary) in &mut summaries {
        for trial_index in 0..TRIALS {
            let outcome = run_trial(*condition, trial_index)?;
            summary.observe(&outcome);
            let fields = [
                condition.name().to_string(),
                outcome.trial.to_string(),
                outcome.seed.to_string(),
                outcome.loss_events.to_string(),
                outcome.high_regime_steps.to_string(),
                outcome.regime_transitions.to_string(),
                to_hex_trace(&outcome.losses),
                to_hex_trace(&outcome.regimes),
                csv_bool(outcome.high_persistent.alarmed()).to_string(),
                optional_step(outcome.high_persistent.first_alarm_step),
                csv_bool(outcome.high_reset.alarmed()).to_string(),
                optional_step(outcome.high_reset.first_alarm_step),
                outcome.high_reset.alarm_episodes.to_string(),
                csv_bool(outcome.low_persistent.alarmed()).to_string(),
                optional_step(outcome.low_persistent.first_alarm_step),
                csv_bool(outcome.low_reset.alarmed()).to_string(),
                optional_step(outcome.low_reset.first_alarm_step),
                outcome.low_reset.alarm_episodes.to_string(),
            ];
            raw_output.push_str(&fields.join(","));
            raw_output.push('\n');
        }
    }

    if summaries.len() != 2
        || summaries
            .iter()
            .any(|(_, summary)| summary.trials != TRIALS)
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "dependent-null benchmark did not cover both 3,000-trial conditions",
        )
        .into());
    }

    let null_summary = &summaries[0].1;
    let shift_summary = &summaries[1].1;
    let persistent_false_alarm_rate = null_summary.high_persistent.alarm_rate();
    let high_reset_false_alarm_rate = null_summary.high_reset.alarm_rate();
    let low_persistent_false_alarm_rate = null_summary.low_persistent.alarm_rate();
    let low_reset_false_alarm_rate = null_summary.low_reset.alarm_rate();
    if persistent_false_alarm_rate > GLOBAL_DELTA
        || high_reset_false_alarm_rate > GLOBAL_DELTA
        || low_persistent_false_alarm_rate > GLOBAL_DELTA
        || low_reset_false_alarm_rate > GLOBAL_DELTA
        || shift_summary.low_persistent.alarm_rate() < 0.90
        || shift_summary.high_persistent.alarm_rate() < 0.75
        || shift_summary.high_reset.alarm_rate() < 0.50
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "dependent-risk statistical gates failed: null rates high-persistent={persistent_false_alarm_rate:.4}, high-reset={high_reset_false_alarm_rate:.4}, low-persistent={low_persistent_false_alarm_rate:.4}, low-reset={low_reset_false_alarm_rate:.4}; shifted power high-persistent={:.4}, high-reset={:.4}, low-persistent={:.4}",
                shift_summary.high_persistent.alarm_rate(),
                shift_summary.high_reset.alarm_rate(),
                shift_summary.low_persistent.alarm_rate()
            ),
        )
        .into());
    }

    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    for (condition, summary) in &summaries {
        summary_row(&mut summary_output, *condition, summary);
    }
    fs::write(&args[1], raw_output)?;
    fs::write(&args[2], summary_output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_hex_uses_stable_most_significant_bit_first_order() {
        assert_eq!(
            to_hex_trace(&[true, false, false, false, false, false, false, true]),
            "81"
        );
        assert_eq!(to_hex_trace(&[true, false, false, false]), "8");
    }

    #[test]
    fn null_and_shifted_controls_share_regime_and_prechange_loss_traces() {
        let null = run_trial(Condition::ConditionalRiskNull, 19).expect("null trial");
        let shifted = run_trial(Condition::ShiftedRisk, 19).expect("shifted trial");
        assert_eq!(null.seed, shifted.seed);
        assert_eq!(null.regimes, shifted.regimes);
        assert_eq!(null.losses[..SHIFT_STEP], shifted.losses[..SHIFT_STEP]);
        assert_ne!(null.losses[SHIFT_STEP..], shifted.losses[SHIFT_STEP..]);
    }

    #[test]
    fn every_null_regime_obeys_the_conditional_loss_bound() {
        assert!(0.13 <= RISK_BOUND);
        assert!(0.15 <= RISK_BOUND);
        assert!(0.30 > RISK_BOUND);
        assert!(0.48 > RISK_BOUND);
    }
}
