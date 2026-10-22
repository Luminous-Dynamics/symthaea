//! Benchmark the deadline between a distribution shift and a safety hazard.
//!
//! Usage: allostatic-monitorability-frontier-bench <raw.csv> <summary.csv>
//! Synthetic only. Score evidence is available before the current action; one-step
//! residual evidence arrives after the transition and can affect the next action.
//! The model records this timing explicitly rather than treating all alarms as
//! equally useful or as proof of physical safety.

use allostatic_control_bench::drift_monitor::{assess_drift, calibrate_drift_threshold};
use allostatic_control_bench::innovation_monitor::{
    calibrate_residual_threshold, SequentialResidualMonitor,
};
use std::error::Error;
use std::fs;

const TARGET_FALSE_ALARM_RATE: f64 = 0.05;
const CALIBRATION_EPISODES: u64 = 159;
const EVALUATION_EPISODES: u64 = 300;
const EPISODE_STEPS: usize = 48;
const SHIFT_STEP: usize = 12;
const RESIDUAL_SLACK: f64 = 0.06;

const OPERATING_POINTS: [&str; 5] = [
    "critical_low_reserve_high_load",
    "reserve_limited",
    "load_limited",
    "balanced",
    "high_margin",
];
const SIGNAL_MODES: [&str; 3] = ["score_only", "residual_only", "sensor_blind"];
const SHIFT_MAGNITUDES: [f64; 4] = [0.02, 0.05, 0.10, 0.20];
const HAZARD_LEADS: [usize; 5] = [0, 1, 2, 4, 8];

const RAW_HEADER: &str = "operating_point,signal_mode,shift_magnitude,hazard_lead_steps,episode_seed,shift_step,first_unsafe_action_step,score_threshold,score_alarm_step,residual_threshold,residual_alarm_step,score_alarm_before_deadline,residual_alarm_before_deadline,score_late_alarm,residual_late_alarm";
const SUMMARY_HEADER: &str = "operating_point,signal_mode,shift_magnitude,hazard_lead_steps,episodes,score_pre_shift_alarms,residual_pre_shift_alarms,score_post_shift_alarms,residual_post_shift_alarms,score_no_post_shift_alarm,residual_no_post_shift_alarm,score_alarm_before_deadlines,residual_alarm_before_deadlines,score_late_alarms,residual_late_alarms,mean_score_post_shift_alarm_delay,mean_residual_post_shift_alarm_delay";

#[derive(Clone, Copy, Debug)]
struct Thresholds {
    score: allostatic_control_bench::drift_monitor::DriftThreshold,
    residual: allostatic_control_bench::innovation_monitor::ResidualThreshold,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeRecord {
    operating_point: usize,
    signal_mode: usize,
    shift_magnitude: f64,
    hazard_lead: usize,
    episode_seed: u64,
    first_unsafe_action_step: usize,
    score_threshold: f64,
    score_alarm_step: usize,
    residual_threshold: f64,
    residual_alarm_step: usize,
}

impl EpisodeRecord {
    fn score_alarm_before_deadline(self) -> bool {
        self.score_alarm_step > 0 && self.score_alarm_step <= self.first_unsafe_action_step
    }

    fn residual_alarm_before_deadline(self) -> bool {
        self.residual_alarm_step > 0
            && self.residual_alarm_step.saturating_add(1) <= self.first_unsafe_action_step
    }

    fn score_late_alarm(self) -> bool {
        self.score_alarm_step > self.first_unsafe_action_step
    }

    fn residual_late_alarm(self) -> bool {
        self.residual_alarm_step > 0
            && self.residual_alarm_step.saturating_add(1) > self.first_unsafe_action_step
    }
}

fn unit_interval(seed: u64, operating_point: usize, step: usize, lane: u64) -> f64 {
    let mut value = seed
        .wrapping_add((operating_point as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add((step as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(lane);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}

fn nominal_residual(seed: u64, operating_point: usize, step: usize) -> f64 {
    0.02 + unit_interval(seed, operating_point, step, 0x7265_7369_6475_616c) * 0.08
}

fn nominal_ood_score(seed: u64, operating_point: usize, step: usize) -> f64 {
    0.02 + unit_interval(seed, operating_point, step, 0x6f6f_645f_7363_6f72) * 0.04
}

fn calibrate(operating_point: usize) -> Result<Thresholds, Box<dyn Error>> {
    let mut score_maxima = Vec::with_capacity(CALIBRATION_EPISODES as usize);
    let mut residual_episodes = Vec::with_capacity(CALIBRATION_EPISODES as usize);

    for episode in 0..CALIBRATION_EPISODES {
        let seed = 100_000 + episode;
        let mut maximum_score: f64 = 0.0;
        let mut residuals = Vec::with_capacity(EPISODE_STEPS);
        for step in 0..EPISODE_STEPS {
            maximum_score = maximum_score.max(nominal_ood_score(seed, operating_point, step));
            residuals.push(nominal_residual(seed, operating_point, step));
        }
        score_maxima.push(maximum_score);
        residual_episodes.push(residuals);
    }

    let score = calibrate_drift_threshold(&score_maxima, TARGET_FALSE_ALARM_RATE)
        .ok_or("failed to calibrate episode-level score threshold")?;
    let residual = calibrate_residual_threshold(
        &residual_episodes,
        RESIDUAL_SLACK,
        TARGET_FALSE_ALARM_RATE,
    )
    .ok_or("failed to calibrate episode-level residual threshold")?;
    Ok(Thresholds { score, residual })
}

fn run_episode(
    operating_point: usize,
    signal_mode: usize,
    shift_magnitude: f64,
    hazard_lead: usize,
    episode: u64,
    thresholds: Thresholds,
) -> Result<EpisodeRecord, Box<dyn Error>> {
    let seed = 200_000 + episode;
    let mut score_alarm_step = 0;
    let mut residual_alarm_step = 0;
    let mut residual_monitor = SequentialResidualMonitor::new(thresholds.residual)
        .ok_or("invalid residual threshold")?;

    for step in 0..EPISODE_STEPS {
        let control_step = step + 1;
        let shift_active = control_step >= SHIFT_STEP;

        // A score computed from the current observation is available to the
        // controller before this step's action is selected.
        let score_delta = if shift_active && signal_mode == 0 {
            shift_magnitude
        } else {
            0.0
        };
        let score = nominal_ood_score(seed, operating_point, step) + score_delta;
        if assess_drift(score, thresholds.score)
            .ok_or("invalid score")?
            .drift_detected()
            && score_alarm_step == 0
        {
            score_alarm_step = control_step;
        }

        // Residual evidence arrives after the action/transition at this step;
        // an alarm can only affect the following action.
        let residual_delta = if shift_active && signal_mode == 1 {
            shift_magnitude
        } else {
            0.0
        };
        let residual = nominal_residual(seed, operating_point, step) + residual_delta;
        let assessment = residual_monitor.observe(residual).ok_or("invalid residual")?;
        if assessment.newly_alarmed() {
            residual_alarm_step = control_step;
        }
    }

    Ok(EpisodeRecord {
        operating_point,
        signal_mode,
        shift_magnitude,
        hazard_lead,
        episode_seed: seed,
        first_unsafe_action_step: SHIFT_STEP + hazard_lead,
        score_threshold: thresholds.score.score_threshold(),
        score_alarm_step,
        residual_threshold: thresholds.residual.statistic_threshold(),
        residual_alarm_step,
    })
}

fn write_raw(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::with_capacity(records.len() * 170);
    output.push_str(RAW_HEADER);
    output.push('\n');
    for record in records {
        output.push_str(&format!(
            "{},{},{:.2},{},{},{},{},{:.9},{},{:.9},{},{},{},{},{}\n",
            OPERATING_POINTS[record.operating_point],
            SIGNAL_MODES[record.signal_mode],
            record.shift_magnitude,
            record.hazard_lead,
            record.episode_seed,
            SHIFT_STEP,
            record.first_unsafe_action_step,
            record.score_threshold,
            record.score_alarm_step,
            record.residual_threshold,
            record.residual_alarm_step,
            record.score_alarm_before_deadline(),
            record.residual_alarm_before_deadline(),
            record.score_late_alarm(),
            record.residual_late_alarm(),
        ));
    }
    fs::write(path, output)?;
    Ok(())
}

fn mean_post_shift_alarm_delay(records: &[EpisodeRecord], score: bool) -> String {
    let mut total_delay = 0_usize;
    let mut detections = 0_usize;
    for record in records {
        let alarm = if score {
            record.score_alarm_step
        } else {
            record.residual_alarm_step
        };
        if alarm >= SHIFT_STEP && alarm > 0 {
            total_delay += alarm - SHIFT_STEP;
            detections += 1;
        }
    }
    if detections == 0 {
        "NA".to_owned()
    } else {
        format!("{:.6}", total_delay as f64 / detections as f64)
    }
}

fn write_summary(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    output.push_str(SUMMARY_HEADER);
    output.push('\n');

    for op in 0..OPERATING_POINTS.len() {
        for mode in 0..SIGNAL_MODES.len() {
            for magnitude in SHIFT_MAGNITUDES {
                for hazard_lead in HAZARD_LEADS {
                    let cell: Vec<EpisodeRecord> = records
                        .iter()
                        .copied()
                        .filter(|record| {
                            record.operating_point == op
                                && record.signal_mode == mode
                                && (record.shift_magnitude - magnitude).abs() < f64::EPSILON
                                && record.hazard_lead == hazard_lead
                        })
                        .collect();
                    let score_pre_shift = cell
                        .iter()
                        .filter(|record| {
                            record.score_alarm_step > 0
                                && record.score_alarm_step < SHIFT_STEP
                        })
                        .count();
                    let residual_pre_shift = cell
                        .iter()
                        .filter(|record| {
                            record.residual_alarm_step > 0
                                && record.residual_alarm_step < SHIFT_STEP
                        })
                        .count();
                    let score_detected = cell
                        .iter()
                        .filter(|record| record.score_alarm_step >= SHIFT_STEP)
                        .count();
                    let residual_detected = cell
                        .iter()
                        .filter(|record| record.residual_alarm_step >= SHIFT_STEP)
                        .count();
                    let score_pre_hazard = cell
                        .iter()
                        .filter(|record| record.score_alarm_before_deadline())
                        .count();
                    let residual_pre_hazard = cell
                        .iter()
                        .filter(|record| record.residual_alarm_before_deadline())
                        .count();
                    let score_late = cell.iter().filter(|record| record.score_late_alarm()).count();
                    let residual_late = cell
                        .iter()
                        .filter(|record| record.residual_late_alarm())
                        .count();
                    output.push_str(&format!(
                        "{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                        OPERATING_POINTS[op],
                        SIGNAL_MODES[mode],
                        magnitude,
                        hazard_lead,
                        cell.len(),
                        score_pre_shift,
                        residual_pre_shift,
                        score_detected,
                        residual_detected,
                        EVALUATION_EPISODES as usize - score_detected,
                        EVALUATION_EPISODES as usize - residual_detected,
                        score_pre_hazard,
                        residual_pre_hazard,
                        score_late,
                        residual_late,
                        mean_post_shift_alarm_delay(&cell, true),
                        mean_post_shift_alarm_delay(&cell, false),
                    ));
                }
            }
        }
    }
    fs::write(path, output)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-monitorability-frontier-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let mut records = Vec::with_capacity(
        OPERATING_POINTS.len()
            * SIGNAL_MODES.len()
            * SHIFT_MAGNITUDES.len()
            * HAZARD_LEADS.len()
            * EVALUATION_EPISODES as usize,
    );
    for op in 0..OPERATING_POINTS.len() {
        let thresholds = calibrate(op)?;
        for mode in 0..SIGNAL_MODES.len() {
            for magnitude in SHIFT_MAGNITUDES {
                for hazard_lead in HAZARD_LEADS {
                    for episode in 0..EVALUATION_EPISODES {
                        records.push(run_episode(
                            op,
                            mode,
                            magnitude,
                            hazard_lead,
                            episode,
                            thresholds,
                        )?);
                    }
                }
            }
        }
    }

    write_raw(&args[1], &records)?;
    write_summary(&args[2], &records)?;
    println!("Monitorability raw results: {}", args[1]);
    println!("Monitorability summary: {}", args[2]);
    println!("Episodes: {}", records.len());
    Ok(())
}
