//! Compare pointwise residual alarms with a calibrated sequential CUSUM monitor.
//!
//! Usage: allostatic-sequential-scope-bench <raw.csv> <summary.csv>
//! Deterministic synthetic controls test gradual drift accumulation, not real-world
//! safety or superiority over published control baselines.

use allostatic_control_bench::robust_viability::ViabilityLimits;
use allostatic_control_bench::scope_monitor::{
    assess_prediction_scope, calibrate_scope_threshold, score_prediction_error,
    PredictionErrorScales, PredictionErrorScore, ScopeCusumAccumulator, ScopeCusumMonitor,
    ScopeThreshold,
};
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const TARGET_FALSE_ALARM_RATE: f64 = 0.05;
const CALIBRATION_EPISODES: usize = 159;
const EVALUATION_EPISODES: usize = 300;
const HORIZON: usize = 8;
const CUSUM_ALLOWANCE: f64 = 0.010;

const RAW_HEADER: &str = "operating_point,condition,episode_seed,step,residual_score,pointwise_threshold,pointwise_step_alarm,pointwise_first_alarm_step,cusum_statistic,cusum_threshold,cusum_latched_alarm,cusum_first_alarm_step,outside_predictor_envelope,all_episode_scores_below_pointwise_threshold";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,pointwise_alarms,cusum_alarms,pointwise_false_alarms,cusum_false_alarms,pointwise_missed_shifts,cusum_missed_shifts,pointwise_mean_detection_step,cusum_mean_detection_step,pointwise_p95_detection_step,cusum_p95_detection_step,pointwise_median_detection_step,cusum_median_detection_step,cusum_earlier_episodes,pointwise_earlier_episodes,tied_detection_episodes,mean_episode_max_point_score,max_episode_point_score,mean_episode_max_cusum_statistic,max_episode_cusum_statistic,episodes_all_scores_below_point_threshold,cusum_caught_below_point_threshold";

#[derive(Clone, Copy, Debug)]
struct OperatingPoint {
    name: &'static str,
}

const OPERATING_POINTS: [OperatingPoint; 5] = [
    OperatingPoint { name: "critical_low_reserve_high_load" },
    OperatingPoint { name: "reserve_limited" },
    OperatingPoint { name: "load_limited" },
    OperatingPoint { name: "balanced" },
    OperatingPoint { name: "high_margin" },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Nominal,
    GradualShift,
    AbruptShift,
    StealthShift,
}

impl Condition {
    const ALL: [Self; 4] = [
        Self::Nominal,
        Self::GradualShift,
        Self::AbruptShift,
        Self::StealthShift,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::GradualShift => "gradual_shift",
            Self::AbruptShift => "abrupt_shift",
            Self::StealthShift => "stealth_shift",
        }
    }

    const fn is_shifted(self) -> bool {
        !matches!(self, Self::Nominal)
    }

    const fn score_condition(self) -> Self {
        match self {
            Self::StealthShift => Self::Nominal,
            other => other,
        }
    }

    const fn outside_predictor_envelope(self) -> bool {
        !matches!(self, Self::Nominal)
    }
}

#[derive(Clone, Copy, Debug)]
struct StepScore {
    magnitude: f64,
    prediction_error: PredictionErrorScore,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeOutcome {
    condition: Condition,
    first_pointwise_alarm_step: usize,
    first_cusum_alarm_step: usize,
    max_point_score: f64,
    max_cusum_statistic: f64,
    all_scores_below_pointwise_threshold: bool,
}

#[derive(Clone, Debug, Default)]
struct CellSummary {
    episodes: usize,
    pointwise_alarms: usize,
    cusum_alarms: usize,
    pointwise_false_alarms: usize,
    cusum_false_alarms: usize,
    pointwise_missed_shifts: usize,
    cusum_missed_shifts: usize,
    pointwise_detection_steps: Vec<usize>,
    cusum_detection_steps: Vec<usize>,
    cusum_earlier_episodes: usize,
    pointwise_earlier_episodes: usize,
    tied_detection_episodes: usize,
    max_point_scores: Vec<f64>,
    max_cusum_statistics: Vec<f64>,
    episodes_all_scores_below_point_threshold: usize,
    cusum_caught_below_point_threshold: usize,
}

impl CellSummary {
    fn observe(&mut self, outcome: EpisodeOutcome) {
        self.episodes += 1;
        let pointwise_detected = outcome.first_pointwise_alarm_step < HORIZON;
        let cusum_detected = outcome.first_cusum_alarm_step < HORIZON;
        self.pointwise_alarms += usize::from(pointwise_detected);
        self.cusum_alarms += usize::from(cusum_detected);
        self.pointwise_false_alarms +=
            usize::from(!outcome.condition.is_shifted() && pointwise_detected);
        self.cusum_false_alarms +=
            usize::from(!outcome.condition.is_shifted() && cusum_detected);
        self.pointwise_missed_shifts +=
            usize::from(outcome.condition.is_shifted() && !pointwise_detected);
        self.cusum_missed_shifts +=
            usize::from(outcome.condition.is_shifted() && !cusum_detected);

        if pointwise_detected {
            self.pointwise_detection_steps
                .push(outcome.first_pointwise_alarm_step);
        }
        if cusum_detected {
            self.cusum_detection_steps
                .push(outcome.first_cusum_alarm_step);
        }
        if pointwise_detected && cusum_detected {
            if outcome.first_cusum_alarm_step < outcome.first_pointwise_alarm_step {
                self.cusum_earlier_episodes += 1;
            } else if outcome.first_pointwise_alarm_step < outcome.first_cusum_alarm_step {
                self.pointwise_earlier_episodes += 1;
            } else {
                self.tied_detection_episodes += 1;
            }
        }
        self.max_point_scores.push(outcome.max_point_score);
        self.max_cusum_statistics.push(outcome.max_cusum_statistic);
        self.episodes_all_scores_below_point_threshold +=
            usize::from(outcome.all_scores_below_pointwise_threshold);
        self.cusum_caught_below_point_threshold += usize::from(
            outcome.all_scores_below_pointwise_threshold && cusum_detected,
        );
    }
}

fn unit_interval(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn residual_magnitude(seed: u64, point_index: usize, step: usize, condition: Condition) -> f64 {
    let key = seed
        .wrapping_add((point_index as u64).wrapping_mul(1_009))
        .wrapping_add((step as u64).wrapping_mul(97));
    let spike_roll = unit_interval(key.wrapping_add(1_234_567));
    let value = unit_interval(key.wrapping_add(7_654_321));
    match condition.score_condition() {
        Condition::Nominal => {
            if spike_roll < 0.005 {
                0.035 + value * 0.005
            } else {
                0.003 + value * 0.006
            }
        }
        Condition::GradualShift => 0.008 + step as f64 * 0.0032 + value * 0.002,
        Condition::AbruptShift if step < 3 => {
            if spike_roll < 0.005 {
                0.035 + value * 0.005
            } else {
                0.003 + value * 0.006
            }
        }
        Condition::AbruptShift => 0.060 + value * 0.010,
        Condition::StealthShift => unreachable!("mapped to nominal scores"),
    }
}

fn score_from_magnitude(
    magnitude: f64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<PredictionErrorScore, Box<dyn Error>> {
    score_prediction_error(
        0.5,
        0.5,
        0.5 - magnitude * 0.02,
        0.5 + magnitude * 0.02,
        limits,
        scales,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid generated residual").into())
}

fn score_episode(
    seed: u64,
    point_index: usize,
    condition: Condition,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<Vec<StepScore>, Box<dyn Error>> {
    (0..HORIZON)
        .map(|step| {
            let magnitude = residual_magnitude(seed, point_index, step, condition);
            Ok(StepScore {
                magnitude,
                prediction_error: score_from_magnitude(magnitude, limits, scales)?,
            })
        })
        .collect()
}

fn episode_max_scores(
    seed: u64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<(f64, f64), Box<dyn Error>> {
    let mut pointwise_max = 0.0_f64;
    let mut cusum_max = 0.0_f64;
    for point_index in 0..OPERATING_POINTS.len() {
        let trace = score_episode(seed, point_index, Condition::Nominal, limits, scales)?;
        let mut accumulator = ScopeCusumAccumulator::new(CUSUM_ALLOWANCE)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid CUSUM allowance"))?;
        for item in trace {
            pointwise_max = pointwise_max.max(item.prediction_error.score());
            cusum_max = cusum_max.max(accumulator.observe(item.prediction_error));
        }
    }
    Ok((pointwise_max, cusum_max))
}

fn pointwise_calibration_threshold(
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<ScopeThreshold, Box<dyn Error>> {
    let mut episode_maxima = Vec::with_capacity(CALIBRATION_EPISODES);
    for episode in 0..CALIBRATION_EPISODES {
        episode_maxima.push(episode_max_scores(10_000 + episode as u64, limits, scales)?.0);
    }
    calibrate_scope_threshold(&episode_maxima, TARGET_FALSE_ALARM_RATE)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "pointwise calibration failed").into())
}

fn cusum_calibration_threshold(
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<ScopeThreshold, Box<dyn Error>> {
    let mut episode_maxima = Vec::with_capacity(CALIBRATION_EPISODES);
    for episode in 0..CALIBRATION_EPISODES {
        episode_maxima.push(episode_max_scores(10_000 + episode as u64, limits, scales)?.1);
    }
    calibrate_scope_threshold(&episode_maxima, TARGET_FALSE_ALARM_RATE)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "CUSUM calibration failed").into())
}

fn detect_episode(
    seed: u64,
    point_index: usize,
    condition: Condition,
    pointwise_threshold: ScopeThreshold,
    cusum_threshold: ScopeThreshold,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<(EpisodeOutcome, Vec<(StepScore, bool, bool, f64)>), Box<dyn Error>> {
    let trace = score_episode(seed, point_index, condition, limits, scales)?;
    let mut cusum = ScopeCusumMonitor::new(CUSUM_ALLOWANCE, cusum_threshold)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "invalid CUSUM monitor"))?;
    let mut first_pointwise_alarm_step = HORIZON;
    let mut first_cusum_alarm_step = HORIZON;
    let mut max_point_score = 0.0_f64;
    let mut max_cusum_statistic = 0.0_f64;
    let mut all_scores_below_pointwise_threshold = true;
    let mut rows = Vec::with_capacity(HORIZON);

    for (step, item) in trace.into_iter().enumerate() {
        let pointwise_assessment =
            allostatic_control_bench::scope_monitor::assess_prediction_scope(
                0.5,
                0.5,
                0.5 - item.magnitude * 0.02,
                0.5 + item.magnitude * 0.02,
                limits,
                scales,
                pointwise_threshold,
            )
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "pointwise assessment failed"))?;
        let cusum_alarm = cusum.observe(item.prediction_error);
        let cusum_statistic = cusum.statistic();
        let pointwise_alarm = pointwise_assessment.alarm();
        if first_pointwise_alarm_step == HORIZON && pointwise_alarm {
            first_pointwise_alarm_step = step;
        }
        if first_cusum_alarm_step == HORIZON && cusum_alarm {
            first_cusum_alarm_step = step;
        }
        max_point_score = max_point_score.max(item.prediction_error.score());
        max_cusum_statistic = max_cusum_statistic.max(cusum_statistic);
        all_scores_below_pointwise_threshold &=
            item.prediction_error.score() <= pointwise_threshold.score_threshold();
        rows.push((item, pointwise_alarm, cusum_alarm, cusum_statistic));
    }

    Ok((
        EpisodeOutcome {
            condition,
            first_pointwise_alarm_step,
            first_cusum_alarm_step,
            max_point_score,
            max_cusum_statistic,
            all_scores_below_pointwise_threshold,
        },
        rows,
    ))
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
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

fn csv_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn csv_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| number.to_string())
}

fn csv_optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_string(), |number| format!("{number:.6}"))
}

fn summary_row(
    output: &mut String,
    point: OperatingPoint,
    condition: Condition,
    summary: &CellSummary,
) {
    let pointwise_steps: Vec<f64> = summary
        .pointwise_detection_steps
        .iter()
        .map(|value| *value as f64)
        .collect();
    let cusum_steps: Vec<f64> = summary
        .cusum_detection_steps
        .iter()
        .map(|value| *value as f64)
        .collect();
    let fields = [
        point.name.to_string(),
        condition.name().to_string(),
        summary.episodes.to_string(),
        summary.pointwise_alarms.to_string(),
        summary.cusum_alarms.to_string(),
        summary.pointwise_false_alarms.to_string(),
        summary.cusum_false_alarms.to_string(),
        summary.pointwise_missed_shifts.to_string(),
        summary.cusum_missed_shifts.to_string(),
        csv_optional_f64(mean(&pointwise_steps)),
        csv_optional_f64(mean(&cusum_steps)),
        csv_optional_usize(percentile(&summary.pointwise_detection_steps, 95)),
        csv_optional_usize(percentile(&summary.cusum_detection_steps, 95)),
        csv_optional_usize(percentile(&summary.pointwise_detection_steps, 50)),
        csv_optional_usize(percentile(&summary.cusum_detection_steps, 50)),
        summary.cusum_earlier_episodes.to_string(),
        summary.pointwise_earlier_episodes.to_string(),
        summary.tied_detection_episodes.to_string(),
        csv_optional_f64(mean(&summary.max_point_scores)),
        summary
            .max_point_scores
            .iter()
            .copied()
            .reduce(f64::max)
            .map_or_else(|| "NA".to_string(), |value| format!("{value:.6}")),
        csv_optional_f64(mean(&summary.max_cusum_statistics)),
        summary
            .max_cusum_statistics
            .iter()
            .copied()
            .reduce(f64::max)
            .map_or_else(|| "NA".to_string(), |value| format!("{value:.6}")),
        summary.episodes_all_scores_below_point_threshold.to_string(),
        summary.cusum_caught_below_point_threshold.to_string(),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-sequential-scope-bench <raw.csv> <summary.csv>",
        )
        .into());
    }

    let limits = ViabilityLimits::new(0.15, 0.88)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid safety limits"))?;
    let scales = PredictionErrorScales::new(0.02, 0.02, 0.02)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid residual scales"))?;
    let pointwise_threshold = pointwise_calibration_threshold(limits, scales)?;
    let cusum_threshold = cusum_calibration_threshold(limits, scales)?;

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary_output = format!("{SUMMARY_HEADER}\n");
    let mut raw_rows = 0_usize;
    for (point_index, point) in OPERATING_POINTS.iter().copied().enumerate() {
        for condition in Condition::ALL {
            let mut cell = CellSummary::default();
            for episode in 0..EVALUATION_EPISODES {
                let seed = 50_000 + episode as u64;
                let (outcome, trace_rows) = detect_episode(
                    seed,
                    point_index,
                    condition,
                    pointwise_threshold,
                    cusum_threshold,
                    limits,
                    scales,
                )?;
                cell.observe(outcome);
                for (step, (item, pointwise_alarm, cusum_alarm, cusum_statistic)) in
                    trace_rows.into_iter().enumerate()
                {
                    let fields = [
                        point.name.to_string(),
                        condition.name().to_string(),
                        seed.to_string(),
                        step.to_string(),
                        format!("{:.9}", item.prediction_error.score()),
                        format!("{:.9}", pointwise_threshold.score_threshold()),
                        csv_bool(pointwise_alarm).to_string(),
                        outcome.first_pointwise_alarm_step.to_string(),
                        format!("{cusum_statistic:.9}"),
                        format!("{:.9}", cusum_threshold.score_threshold()),
                        csv_bool(cusum_alarm).to_string(),
                        outcome.first_cusum_alarm_step.to_string(),
                        csv_bool(condition.outside_predictor_envelope()).to_string(),
                        csv_bool(outcome.all_scores_below_pointwise_threshold).to_string(),
                    ];
                    raw.push_str(&fields.join(","));
                    raw.push('\n');
                    raw_rows += 1;
                }
            }
            summary_row(&mut summary_output, point, condition, &cell);
        }
    }

    let expected_raw_rows =
        OPERATING_POINTS.len() * Condition::ALL.len() * EVALUATION_EPISODES * HORIZON;
    let expected_summary_rows = OPERATING_POINTS.len() * Condition::ALL.len();
    if raw_rows != expected_raw_rows || raw_rows != 48_000 || expected_summary_rows != 20 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 48,000 sequential-scope rows, generated {raw_rows}"),
        )
        .into());
    }
    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary_output)?;
    Ok(())
}
