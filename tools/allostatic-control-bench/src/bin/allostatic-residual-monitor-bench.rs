//! Compare episode-calibrated score alarms with sequential residual monitoring.
//!
//! Usage: allostatic-residual-monitor-bench <raw.csv> <summary.csv>
//! All transitions and residual streams here are deterministic synthetic controls.
//! This benchmark tests sequential alarm timing and negative-control accounting,
//! not physical safety or real-world OOD detector accuracy.

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

#[derive(Clone, Copy, Debug)]
struct Condition {
    name: &'static str,
    actual_shift: bool,
    residual_shift: f64,
    score_shift: f64,
    residual_observable: bool,
}

const CONDITIONS: [Condition; 5] = [
    Condition {
        name: "nominal",
        actual_shift: false,
        residual_shift: 0.0,
        score_shift: 0.0,
        residual_observable: true,
    },
    Condition {
        name: "residual_only_shift",
        actual_shift: true,
        residual_shift: 0.10,
        score_shift: 0.0,
        residual_observable: true,
    },
    Condition {
        name: "observable_strong_shift",
        actual_shift: true,
        residual_shift: 0.30,
        score_shift: 0.20,
        residual_observable: true,
    },
    Condition {
        name: "stealth_score_shift",
        actual_shift: true,
        residual_shift: 0.30,
        score_shift: 0.0,
        residual_observable: true,
    },
    Condition {
        name: "sensor_blind_shift",
        actual_shift: true,
        residual_shift: 0.0,
        score_shift: 0.0,
        residual_observable: false,
    },
];

const RAW_HEADER: &str = "operating_point,condition,episode_seed,first_shift_step,max_ood_score,score_threshold,score_alarm_step,residual_threshold,residual_alarm_step,score_only_alarm,residual_alarm,actual_shift,residual_evidence_shift";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,shifted_episodes,score_alarm_episodes,residual_alarm_episodes,score_detections,residual_detections,score_misses,residual_misses,mean_score_detection_delay,mean_residual_detection_delay";

#[derive(Clone, Copy, Debug)]
struct EpisodeRecord {
    operating_point: usize,
    condition: usize,
    seed: u64,
    first_shift_step: usize,
    max_ood_score: f64,
    score_threshold: f64,
    score_alarm_step: usize,
    residual_threshold: f64,
    residual_alarm_step: usize,
    actual_shift: bool,
    residual_evidence_shift: bool,
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

fn calibration_thresholds(
    operating_point: usize,
) -> Result<
    (
        allostatic_control_bench::drift_monitor::DriftThreshold,
        allostatic_control_bench::innovation_monitor::ResidualThreshold,
    ),
    Box<dyn Error>,
> {
    let mut episode_max_scores = Vec::with_capacity(CALIBRATION_EPISODES as usize);
    let mut residual_sequences = Vec::with_capacity(CALIBRATION_EPISODES as usize);

    for episode in 0..CALIBRATION_EPISODES {
        let seed = 100_000 + episode;
        let mut max_score: f64 = 0.0;
        let mut residuals = Vec::with_capacity(EPISODE_STEPS);
        for step in 0..EPISODE_STEPS {
            max_score = max_score.max(nominal_ood_score(seed, operating_point, step));
            residuals.push(nominal_residual(seed, operating_point, step));
        }
        episode_max_scores.push(max_score);
        residual_sequences.push(residuals);
    }

    let score_threshold =
        calibrate_drift_threshold(&episode_max_scores, TARGET_FALSE_ALARM_RATE)
            .ok_or("failed to calibrate episode-level OOD score")?;
    let residual_threshold = calibrate_residual_threshold(
        &residual_sequences,
        RESIDUAL_SLACK,
        TARGET_FALSE_ALARM_RATE,
    )
    .ok_or("failed to calibrate sequential residual threshold")?;

    Ok((score_threshold, residual_threshold))
}

fn run_episode(
    operating_point: usize,
    condition_index: usize,
    episode: u64,
    score_threshold: allostatic_control_bench::drift_monitor::DriftThreshold,
    residual_threshold: allostatic_control_bench::innovation_monitor::ResidualThreshold,
) -> Result<EpisodeRecord, Box<dyn Error>> {
    let condition = CONDITIONS[condition_index];
    let seed = 200_000 + episode;
    let first_shift_step = if condition.actual_shift { SHIFT_STEP } else { 0 };
    let mut max_ood_score: f64 = 0.0;
    let mut score_alarm_step = 0;
    let mut residual_alarm_step = 0;
    let mut residual_monitor = SequentialResidualMonitor::new(residual_threshold)
        .ok_or("invalid calibrated sequential residual monitor")?;

    for step in 0..EPISODE_STEPS {
        let shifted_now = condition.actual_shift && step + 1 >= SHIFT_STEP;
        let base_score = nominal_ood_score(seed, operating_point, step);
        let score = base_score
            + if shifted_now { condition.score_shift } else { 0.0 };
        max_ood_score = max_ood_score.max(score);
        let assessment = assess_drift(score, score_threshold).ok_or("invalid OOD score")?;
        if assessment.drift_detected() && score_alarm_step == 0 {
            score_alarm_step = step + 1;
        }

        let residual_shift =
            if shifted_now && condition.residual_observable {
                condition.residual_shift
            } else {
                0.0
            };
        let residual = nominal_residual(seed, operating_point, step) + residual_shift;
        let residual_assessment =
            residual_monitor.observe(residual).ok_or("invalid residual score")?;
        if residual_assessment.newly_alarmed() {
            residual_alarm_step = step + 1;
        }
    }

    Ok(EpisodeRecord {
        operating_point,
        condition: condition_index,
        seed,
        first_shift_step,
        max_ood_score,
        score_threshold: score_threshold.score_threshold(),
        score_alarm_step,
        residual_threshold: residual_threshold.statistic_threshold(),
        residual_alarm_step,
        actual_shift: condition.actual_shift,
        residual_evidence_shift: condition.residual_observable && condition.actual_shift,
    })
}

fn write_raw(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    output.push_str(RAW_HEADER);
    output.push('\n');
    for record in records {
        let op = OPERATING_POINTS[record.operating_point];
        let condition = CONDITIONS[record.condition];
        output.push_str(&format!(
            "{},{},{},{},{:.9},{:.9},{},{:.9},{},{},{},{},{}\n",
            op,
            condition.name,
            record.seed,
            record.first_shift_step,
            record.max_ood_score,
            record.score_threshold,
            record.score_alarm_step,
            record.residual_threshold,
            record.residual_alarm_step,
            record.score_alarm_step != 0,
            record.residual_alarm_step != 0,
            record.actual_shift,
            record.residual_evidence_shift,
        ));
    }
    fs::write(path, output)?;
    Ok(())
}

fn mean_detection_delay(records: &[EpisodeRecord], residual: bool) -> String {
    let mut total = 0_usize;
    let mut count = 0_usize;
    for record in records.iter().filter(|record| record.first_shift_step > 0) {
        let alarm_step = if residual {
            record.residual_alarm_step
        } else {
            record.score_alarm_step
        };
        if alarm_step >= record.first_shift_step && alarm_step > 0 {
            total += alarm_step - record.first_shift_step;
            count += 1;
        }
    }
    if count == 0 {
        "NA".to_owned()
    } else {
        format!("{:.6}", total as f64 / count as f64)
    }
}

fn write_summary(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    output.push_str(SUMMARY_HEADER);
    output.push('\n');

    for op in 0..OPERATING_POINTS.len() {
        for condition_index in 0..CONDITIONS.len() {
            let cell: Vec<EpisodeRecord> = records
                .iter()
                .copied()
                .filter(|record| {
                    record.operating_point == op && record.condition == condition_index
                })
                .collect();
            let shifted_episodes = cell.iter().filter(|record| record.actual_shift).count();
            let score_alarms = cell
                .iter()
                .filter(|record| record.score_alarm_step > 0)
                .count();
            let residual_alarms = cell
                .iter()
                .filter(|record| record.residual_alarm_step > 0)
                .count();
            let score_detections = cell
                .iter()
                .filter(|record| {
                    record.first_shift_step > 0
                        && record.score_alarm_step >= record.first_shift_step
                        && record.score_alarm_step > 0
                })
                .count();
            let residual_detections = cell
                .iter()
                .filter(|record| {
                    record.first_shift_step > 0
                        && record.residual_alarm_step >= record.first_shift_step
                        && record.residual_alarm_step > 0
                })
                .count();
            let score_misses = shifted_episodes - score_detections;
            let residual_misses = shifted_episodes - residual_detections;
            output.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                OPERATING_POINTS[op],
                CONDITIONS[condition_index].name,
                cell.len(),
                shifted_episodes,
                score_alarms,
                residual_alarms,
                score_detections,
                residual_detections,
                score_misses,
                residual_misses,
                mean_detection_delay(&cell, false),
                mean_detection_delay(&cell, true),
            ));
        }
    }
    fs::write(path, output)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-residual-monitor-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let mut records = Vec::with_capacity(
        OPERATING_POINTS.len() * CONDITIONS.len() * EVALUATION_EPISODES as usize,
    );
    for operating_point in 0..OPERATING_POINTS.len() {
        let (score_threshold, residual_threshold) = calibration_thresholds(operating_point)?;
        for condition in 0..CONDITIONS.len() {
            for episode in 0..EVALUATION_EPISODES {
                records.push(run_episode(
                    operating_point,
                    condition,
                    episode,
                    score_threshold,
                    residual_threshold,
                )?);
            }
        }
    }

    write_raw(&args[1], &records)?;
    write_summary(&args[2], &records)?;
    println!("Sequential residual raw results: {}", args[1]);
    println!("Sequential residual summary: {}", args[2]);
    println!(
        "Episodes: {}",
        OPERATING_POINTS.len() * CONDITIONS.len() * EVALUATION_EPISODES as usize
    );
    Ok(())
}
