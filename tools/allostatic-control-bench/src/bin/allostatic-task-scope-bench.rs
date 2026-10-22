//! Compare feature-space OOD detection with task-relevant model-scope detection.
//!
//! Usage: allostatic-task-scope-bench <raw.csv> <summary.csv>
//! This deterministic fixture measures detector contracts, not real-world safety
//! or superiority over published controls.

use allostatic_control_bench::drift_monitor::calibrate_drift_threshold;
use allostatic_control_bench::robust_viability::ViabilityLimits;
use allostatic_control_bench::scope_monitor::{
    assess_prediction_scope, calibrate_scope_threshold, score_prediction_error,
    PredictionErrorScales, PredictionErrorScore, ScopeThreshold,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const FALSE_ALARM_TARGET: f64 = 0.05;
const CALIBRATION_EPISODES: usize = 159;
const EVALUATION_EPISODES: usize = 300;
const HORIZON: usize = 8;

const RAW_HEADER: &str = "operating_point,condition,episode_seed,feature_score,feature_threshold,feature_alarm,scope_score,scope_threshold,scope_alarm,combined_alarm,outside_predictor_envelope,first_feature_alarm_step,first_scope_alarm_step,first_combined_alarm_step,first_false_safe_step,feature_flagged_by_false_safe_step,scope_flagged_by_false_safe_step,combined_flagged_by_false_safe_step,scope_false_safe_override";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,feature_alarms,scope_alarms,combined_alarms,feature_false_alarms,scope_false_alarms,combined_false_alarms,feature_missed_shifts,scope_missed_shifts,combined_missed_shifts,false_safe_episodes,feature_flagged_by_false_safe,scope_flagged_by_false_safe,combined_flagged_by_false_safe,feature_missed_false_safe,scope_missed_false_safe,combined_missed_false_safe,mean_scope_score,max_scope_score,mean_scope_alarm_step,mean_combined_alarm_step";

#[derive(Clone, Copy, Debug)]
struct OperatingPoint {
    name: &'static str,
    reserve: f64,
    load: f64,
}

const OPERATING_POINTS: [OperatingPoint; 5] = [
    OperatingPoint {
        name: "critical_low_reserve_high_load",
        reserve: 0.22,
        load: 0.80,
    },
    OperatingPoint {
        name: "reserve_limited",
        reserve: 0.20,
        load: 0.45,
    },
    OperatingPoint {
        name: "load_limited",
        reserve: 0.50,
        load: 0.78,
    },
    OperatingPoint {
        name: "balanced",
        reserve: 0.40,
        load: 0.55,
    },
    OperatingPoint {
        name: "high_margin",
        reserve: 0.65,
        load: 0.30,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Nominal,
    ModerateShift,
    StrongShift,
    StealthShift,
}

impl Condition {
    const ALL: [Self; 4] = [
        Self::Nominal,
        Self::ModerateShift,
        Self::StrongShift,
        Self::StealthShift,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::ModerateShift => "moderate_shift",
            Self::StrongShift => "strong_shift",
            Self::StealthShift => "stealth_shift",
        }
    }

    const fn is_shifted(self) -> bool {
        !matches!(self, Self::Nominal)
    }

    const fn dynamics_kind(self) -> Self {
        match self {
            Self::StealthShift => Self::StrongShift,
            other => other,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActionEffect {
    reserve: f64,
    load: f64,
    reserve_bound: f64,
    load_bound: f64,
}

fn effect(action: Action) -> ActionEffect {
    match action {
        Action::Push => ActionEffect {
            reserve: -0.0175,
            load: 0.050,
            reserve_bound: 0.0125,
            load_bound: 0.020,
        },
        Action::Maintain => ActionEffect {
            reserve: 0.025,
            load: 0.010,
            reserve_bound: 0.015,
            load_bound: 0.020,
        },
        Action::EmergencyRecover => ActionEffect {
            reserve: 0.0525,
            load: -0.120,
            reserve_bound: 0.0175,
            load_bound: 0.040,
        },
    }
}

const ACTION_PLAN: [Action; HORIZON] = [
    Action::Push,
    Action::Maintain,
    Action::Push,
    Action::Maintain,
    Action::EmergencyRecover,
    Action::Push,
    Action::Maintain,
    Action::EmergencyRecover,
];

#[derive(Clone, Copy, Debug)]
struct StepObservation {
    predicted_reserve: f64,
    predicted_load: f64,
    observed_reserve: f64,
    observed_load: f64,
    score: PredictionErrorScore,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeResult {
    feature_score: f64,
    scope_score: f64,
    feature_alarm: bool,
    scope_alarm: bool,
    combined_alarm: bool,
    outside_predictor_envelope: bool,
    first_feature_alarm_step: usize,
    first_scope_alarm_step: usize,
    first_combined_alarm_step: usize,
    first_false_safe_step: usize,
    feature_flagged_by_false_safe_step: bool,
    scope_flagged_by_false_safe_step: bool,
    combined_flagged_by_false_safe_step: bool,
    scope_false_safe_override: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    episodes: usize,
    feature_alarms: usize,
    scope_alarms: usize,
    combined_alarms: usize,
    feature_false_alarms: usize,
    scope_false_alarms: usize,
    combined_false_alarms: usize,
    feature_missed_shifts: usize,
    scope_missed_shifts: usize,
    combined_missed_shifts: usize,
    false_safe_episodes: usize,
    feature_flagged_by_false_safe: usize,
    scope_flagged_by_false_safe: usize,
    combined_flagged_by_false_safe: usize,
    feature_missed_false_safe: usize,
    scope_missed_false_safe: usize,
    combined_missed_false_safe: usize,
    scope_score_sum: f64,
    max_scope_score: f64,
    scope_alarm_step_sum: usize,
    combined_alarm_step_sum: usize,
}

impl Counters {
    fn observe(&mut self, condition: Condition, result: EpisodeResult) {
        let has_false_safe = result.first_false_safe_step < HORIZON;
        self.episodes += 1;
        self.feature_alarms += usize::from(result.feature_alarm);
        self.scope_alarms += usize::from(result.scope_alarm);
        self.combined_alarms += usize::from(result.combined_alarm);
        self.feature_false_alarms += usize::from(!condition.is_shifted() && result.feature_alarm);
        self.scope_false_alarms += usize::from(!condition.is_shifted() && result.scope_alarm);
        self.combined_false_alarms += usize::from(!condition.is_shifted() && result.combined_alarm);
        self.feature_missed_shifts += usize::from(condition.is_shifted() && !result.feature_alarm);
        self.scope_missed_shifts += usize::from(condition.is_shifted() && !result.scope_alarm);
        self.combined_missed_shifts +=
            usize::from(condition.is_shifted() && !result.combined_alarm);
        self.false_safe_episodes += usize::from(has_false_safe);
        self.feature_flagged_by_false_safe +=
            usize::from(has_false_safe && result.feature_flagged_by_false_safe_step);
        self.scope_flagged_by_false_safe +=
            usize::from(has_false_safe && result.scope_flagged_by_false_safe_step);
        self.combined_flagged_by_false_safe +=
            usize::from(has_false_safe && result.combined_flagged_by_false_safe_step);
        self.feature_missed_false_safe +=
            usize::from(has_false_safe && !result.feature_flagged_by_false_safe_step);
        self.scope_missed_false_safe +=
            usize::from(has_false_safe && !result.scope_flagged_by_false_safe_step);
        self.combined_missed_false_safe +=
            usize::from(has_false_safe && !result.combined_flagged_by_false_safe_step);
        self.scope_score_sum += result.scope_score;
        self.max_scope_score = self.max_scope_score.max(result.scope_score);
        self.scope_alarm_step_sum += result.first_scope_alarm_step;
        self.combined_alarm_step_sum += result.first_combined_alarm_step;
    }
}

fn unit_interval(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn clamp_state(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn feature_score(condition: Condition, seed: u64) -> f64 {
    let u = unit_interval(seed);
    match condition {
        Condition::Nominal | Condition::StealthShift => u * 0.25,
        Condition::ModerateShift => 0.15 + u * 0.40,
        Condition::StrongShift => 0.50 + u * 0.50,
    }
}

fn dynamics_offset(condition: Condition, action: Action) -> (f64, f64) {
    match (condition.dynamics_kind(), action) {
        (Condition::Nominal, _) => (0.0, 0.0),
        (Condition::ModerateShift, Action::Push) => (-0.025, 0.050),
        (Condition::ModerateShift, Action::Maintain) => (-0.020, 0.035),
        (Condition::ModerateShift, Action::EmergencyRecover) => (-0.030, 0.060),
        (Condition::StrongShift, Action::Push) => (-0.055, 0.100),
        (Condition::StrongShift, Action::Maintain) => (-0.025, 0.070),
        (Condition::StrongShift, Action::EmergencyRecover) => (-0.035, 0.080),
        (Condition::StealthShift, _) => unreachable!("mapped to strong shift"),
    }
}

fn noise(seed: u64, step: usize, dimension: u64) -> f64 {
    (unit_interval(seed.wrapping_add(step as u64 * 17 + dimension * 101)) - 0.5) * 0.004
}

fn predict_step(actual_reserve: f64, actual_load: f64, action: Action) -> (f64, f64) {
    let nominal = effect(action);
    (
        clamp_state(actual_reserve + nominal.reserve),
        clamp_state(actual_load + nominal.load),
    )
}

fn observe_step(
    actual_reserve: f64,
    actual_load: f64,
    action: Action,
    condition: Condition,
    seed: u64,
    step: usize,
) -> (f64, f64) {
    let nominal = effect(action);
    let (reserve_offset, load_offset) = dynamics_offset(condition, action);
    (
        clamp_state(actual_reserve + nominal.reserve + reserve_offset + noise(seed, step, 0)),
        clamp_state(actual_load + nominal.load + load_offset + noise(seed, step, 1)),
    )
}

fn outside_predictor_envelope(condition: Condition) -> bool {
    ACTION_PLAN.iter().copied().any(|action| {
        let nominal = effect(action);
        let (reserve_offset, load_offset) = dynamics_offset(condition, action);
        reserve_offset.abs() > nominal.reserve_bound || load_offset.abs() > nominal.load_bound
    })
}

fn trace_episode(
    point: OperatingPoint,
    condition: Condition,
    seed: u64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<Vec<StepObservation>, Box<dyn Error>> {
    let mut actual_reserve = point.reserve;
    let mut actual_load = point.load;
    let mut trace = Vec::with_capacity(HORIZON);

    for (step, action) in ACTION_PLAN.iter().copied().enumerate() {
        let (predicted_reserve, predicted_load) = predict_step(actual_reserve, actual_load, action);
        let (observed_reserve, observed_load) =
            observe_step(actual_reserve, actual_load, action, condition, seed, step);
        let score = score_prediction_error(
            predicted_reserve,
            predicted_load,
            observed_reserve,
            observed_load,
            limits,
            scales,
        )
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid prediction pair"))?;
        trace.push(StepObservation {
            predicted_reserve,
            predicted_load,
            observed_reserve,
            observed_load,
            score,
        });
        actual_reserve = observed_reserve;
        actual_load = observed_load;
    }
    Ok(trace)
}

fn episode_scope_score(
    point: OperatingPoint,
    condition: Condition,
    seed: u64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<f64, Box<dyn Error>> {
    let trace = trace_episode(point, condition, seed, limits, scales)?;
    Ok(trace
        .iter()
        .map(|step| step.score.score())
        .fold(0.0_f64, f64::max))
}

fn calibration_score(
    seed: u64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<f64, Box<dyn Error>> {
    let mut maximum: f64 = 0.0;
    for point in OPERATING_POINTS {
        maximum = maximum.max(episode_scope_score(
            point,
            Condition::Nominal,
            seed,
            limits,
            scales,
        )?);
    }
    Ok(maximum)
}

fn run_episode(
    point: OperatingPoint,
    condition: Condition,
    seed: u64,
    feature_threshold: f64,
    scope_threshold: ScopeThreshold,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Result<EpisodeResult, Box<dyn Error>> {
    let feature_score = feature_score(condition, seed.wrapping_add(point.reserve.to_bits()));
    let feature_alarm = feature_score > feature_threshold;
    let first_feature_alarm_step = if feature_alarm { 0 } else { HORIZON };
    let trace = trace_episode(point, condition, seed, limits, scales)?;
    let scope_score = trace
        .iter()
        .map(|step| step.score.score())
        .fold(0.0_f64, f64::max);

    let mut first_scope_alarm_step = HORIZON;
    let mut first_false_safe_step = HORIZON;
    let mut scope_false_safe_override = false;
    for (step, observation) in trace.iter().enumerate() {
        let assessment = assess_prediction_scope(
            observation.predicted_reserve,
            observation.predicted_load,
            observation.observed_reserve,
            observation.observed_load,
            limits,
            scales,
            scope_threshold,
        )
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidData,
                "scope monitor rejected a valid synthetic transition",
            )
        })?;
        if first_scope_alarm_step == HORIZON && assessment.alarm() {
            first_scope_alarm_step = step;
        }
        if first_false_safe_step == HORIZON && assessment.prediction_error().false_safe_prediction() {
            first_false_safe_step = step;
        }
        scope_false_safe_override |= assessment.false_safe_override();
    }

    let first_combined_alarm_step = first_feature_alarm_step.min(first_scope_alarm_step);
    let feature_flagged_by_false_safe_step =
        first_false_safe_step < HORIZON && first_feature_alarm_step <= first_false_safe_step;
    let scope_flagged_by_false_safe_step =
        first_false_safe_step < HORIZON && first_scope_alarm_step <= first_false_safe_step;
    let combined_flagged_by_false_safe_step =
        first_false_safe_step < HORIZON && first_combined_alarm_step <= first_false_safe_step;

    Ok(EpisodeResult {
        feature_score,
        scope_score,
        feature_alarm,
        scope_alarm: first_scope_alarm_step < HORIZON,
        combined_alarm: first_combined_alarm_step < HORIZON,
        outside_predictor_envelope: outside_predictor_envelope(condition),
        first_feature_alarm_step,
        first_scope_alarm_step,
        first_combined_alarm_step,
        first_false_safe_step,
        feature_flagged_by_false_safe_step,
        scope_flagged_by_false_safe_step,
        combined_flagged_by_false_safe_step,
        scope_false_safe_override,
    })
}

fn csv_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn summary_row(output: &mut String, point: OperatingPoint, condition: Condition, c: Counters) {
    let episodes = c.episodes as f64;
    let fields = [
        point.name.to_string(),
        condition.name().to_string(),
        c.episodes.to_string(),
        c.feature_alarms.to_string(),
        c.scope_alarms.to_string(),
        c.combined_alarms.to_string(),
        c.feature_false_alarms.to_string(),
        c.scope_false_alarms.to_string(),
        c.combined_false_alarms.to_string(),
        c.feature_missed_shifts.to_string(),
        c.scope_missed_shifts.to_string(),
        c.combined_missed_shifts.to_string(),
        c.false_safe_episodes.to_string(),
        c.feature_flagged_by_false_safe.to_string(),
        c.scope_flagged_by_false_safe.to_string(),
        c.combined_flagged_by_false_safe.to_string(),
        c.feature_missed_false_safe.to_string(),
        c.scope_missed_false_safe.to_string(),
        c.combined_missed_false_safe.to_string(),
        format!("{:.6}", c.scope_score_sum / episodes),
        format!("{:.6}", c.max_scope_score),
        format!("{:.6}", c.scope_alarm_step_sum as f64 / episodes),
        format!("{:.6}", c.combined_alarm_step_sum as f64 / episodes),
    ];
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-task-scope-bench <raw.csv> <summary.csv>",
        )
        .into());
    }
    let limits = ViabilityLimits::new(0.15, 0.88)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid safety limits"))?;
    let scales = PredictionErrorScales::new(0.02, 0.02, 0.02)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid residual scales"))?;

    let feature_calibration: Vec<f64> = (0..CALIBRATION_EPISODES)
        .map(|i| feature_score(Condition::Nominal, 10_000 + i as u64))
        .collect();
    let feature_threshold = calibrate_drift_threshold(&feature_calibration, FALSE_ALARM_TARGET)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "feature-score calibration failed"))?;
    let mut scope_calibration = Vec::with_capacity(CALIBRATION_EPISODES);
    for i in 0..CALIBRATION_EPISODES {
        scope_calibration.push(calibration_score(30_000 + i as u64, limits, scales)?);
    }
    let scope_threshold = calibrate_scope_threshold(&scope_calibration, FALSE_ALARM_TARGET)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "scope-score calibration failed"))?;

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let mut rows = 0_usize;
    for point in OPERATING_POINTS {
        for condition in Condition::ALL {
            let mut counters = Counters::default();
            for episode in 0..EVALUATION_EPISODES {
                let seed = 50_000 + episode as u64;
                let result = run_episode(
                    point,
                    condition,
                    seed,
                    feature_threshold.score_threshold(),
                    scope_threshold,
                    limits,
                    scales,
                )?;
                counters.observe(condition, result);
                let fields = [
                    point.name.to_string(),
                    condition.name().to_string(),
                    episode.to_string(),
                    format!("{:.9}", result.feature_score),
                    format!("{:.9}", feature_threshold.score_threshold()),
                    csv_bool(result.feature_alarm).to_string(),
                    format!("{:.9}", result.scope_score),
                    format!("{:.9}", scope_threshold.score_threshold()),
                    csv_bool(result.scope_alarm).to_string(),
                    csv_bool(result.combined_alarm).to_string(),
                    csv_bool(result.outside_predictor_envelope).to_string(),
                    result.first_feature_alarm_step.to_string(),
                    result.first_scope_alarm_step.to_string(),
                    result.first_combined_alarm_step.to_string(),
                    result.first_false_safe_step.to_string(),
                    csv_bool(result.feature_flagged_by_false_safe_step).to_string(),
                    csv_bool(result.scope_flagged_by_false_safe_step).to_string(),
                    csv_bool(result.combined_flagged_by_false_safe_step).to_string(),
                    csv_bool(result.scope_false_safe_override).to_string(),
                ];
                raw.push_str(&fields.join(","));
                raw.push('\n');
                rows += 1;
            }
            summary_row(&mut summary, point, condition, counters);
        }
    }
    let expected_rows = OPERATING_POINTS.len() * Condition::ALL.len() * EVALUATION_EPISODES;
    if rows != expected_rows || rows != 6_000 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 6,000 task-scope rows, generated {rows}"),
        )
        .into());
    }
    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    Ok(())
}
