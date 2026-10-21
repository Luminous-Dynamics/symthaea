//! Evaluate an episode-calibrated OOD monitor with a multi-step reachability fallback.
//!
//! Usage: allostatic-ood-fallback-bench <raw.csv> <summary.csv>
//! The score distributions are synthetic controls. They test mode switching and
//! accounting, not empirical false-alarm guarantees on a physical controller.

use allostatic_control_bench::drift_monitor::{
    calibrate_drift_threshold, select_control_action, ControlMode,
};
use allostatic_control_bench::multistep_viability::evaluate_horizon_action_set;
use allostatic_control_bench::robust_viability::{
    evaluate_action_set, ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope,
    ViabilityLimits,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const TARGET_FALSE_ALARM_RATE: f64 = 0.05;
const CALIBRATION_EPISODES: u64 = 159;
const EVALUATION_EPISODES: u64 = 300;
const FALLBACK_HORIZON: usize = 3;
const RAW_HEADER: &str = "operating_point,condition,episode_seed,ood_score,score_threshold,drift_detected,control_mode,proposed_action,predictive_viable_count,fallback_viable_count,predictive_action,fallback_action,selected_action,selected_action_certified,no_certified_action,predictive_proposal_preserved,fallback_proposal_preserved";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,alarms,false_alarms,missed_detections,false_alarm_rate,miss_rate,fallback_activations,fallback_no_action,predictive_proposal_preserved,fallback_proposal_preserved,mean_predictive_viable_actions,mean_fallback_viable_actions";

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

#[derive(Clone, Copy, Debug)]
struct Condition {
    name: &'static str,
    shifted: bool,
}

const CONDITIONS: [Condition; 3] = [
    Condition {
        name: "nominal",
        shifted: false,
    },
    Condition {
        name: "moderate_shift",
        shifted: true,
    },
    Condition {
        name: "strong_shift",
        shifted: true,
    },
];

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    episodes: usize,
    alarms: usize,
    false_alarms: usize,
    missed_detections: usize,
    fallback_activations: usize,
    fallback_no_action: usize,
    predictive_proposal_preserved: usize,
    fallback_proposal_preserved: usize,
    predictive_viable_actions: usize,
    fallback_viable_actions: usize,
}

impl Counters {
    fn observe(
        &mut self,
        condition: Condition,
        detected: bool,
        selected_action: Option<Action>,
        predictive_count: usize,
        fallback_count: usize,
        predictive_preserves_proposal: bool,
        fallback_preserves_proposal: bool,
    ) {
        self.episodes += 1;
        self.alarms += usize::from(detected);
        self.false_alarms += usize::from(!condition.shifted && detected);
        self.missed_detections += usize::from(condition.shifted && !detected);
        self.fallback_activations += usize::from(detected);
        self.fallback_no_action +=
            usize::from(detected && selected_action.is_none());
        self.predictive_proposal_preserved += usize::from(predictive_preserves_proposal);
        self.fallback_proposal_preserved += usize::from(fallback_preserves_proposal);
        self.predictive_viable_actions += predictive_count;
        self.fallback_viable_actions += fallback_count;
    }
}

fn unit_interval(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Synthetic scalar episode score. Nominal scores lie in [0,.25); moderate
/// shifts overlap that range; strong shifts lie in [.50,1). This is a designed
/// test distribution, not a trained or validated OOD detector.
fn score_for(condition: Condition, seed: u64) -> f64 {
    let unit = unit_interval(seed);
    match condition.name {
        "nominal" => unit * 0.25,
        "moderate_shift" => 0.15 + unit * 0.40,
        "strong_shift" => 0.50 + unit * 0.50,
        _ => f64::NAN,
    }
}

fn interval(lower: f64, upper: f64) -> Result<Interval, Box<dyn Error>> {
    Interval::new(lower, upper)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid interval").into())
}

fn shifted_effect(
    reserve: (f64, f64),
    load: (f64, f64),
    seed: u64,
    action_index: u64,
) -> Result<ActionEffectBounds, Box<dyn Error>> {
    let reserve_shift = (unit_interval(seed + action_index * 2) - 0.5) * 0.016;
    let load_shift = (unit_interval(seed + action_index * 2 + 1) - 0.5) * 0.020;
    ActionEffectBounds::new(
        interval(reserve.0 + reserve_shift, reserve.1 + reserve_shift)?,
        interval(load.0 + load_shift, load.1 + load_shift)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid action-effect envelope").into())
}

fn dynamics_for(seed: u64) -> Result<DynamicsBounds, Box<dyn Error>> {
    let push = shifted_effect((-0.030, -0.005), (0.030, 0.070), seed, 0)?;
    let maintain = shifted_effect((0.010, 0.040), (-0.010, 0.030), seed, 1)?;
    let emergency_recover = shifted_effect((0.025, 0.060), (-0.160, -0.080), seed, 2)?;
    Ok(DynamicsBounds::new(push, maintain, emergency_recover))
}

fn state_for(point: OperatingPoint) -> Result<StateEnvelope, Box<dyn Error>> {
    StateEnvelope::new(
        interval(point.reserve, point.reserve)?,
        interval(point.load, point.load)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid operating point").into())
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::Push) => "push",
        Some(Action::Maintain) => "maintain",
        Some(Action::EmergencyRecover) => "emergency_recover",
        None => "none",
    }
}

fn control_mode_name(mode: ControlMode) -> &'static str {
    match mode {
        ControlMode::Predictive => "predictive",
        ControlMode::ReachabilityFallback => "reachability_fallback",
    }
}

fn csv_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn rate(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "NA".to_string()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn summary_row(
    output: &mut String,
    operating_point: &str,
    condition: Condition,
    counters: Counters,
) {
    output.push_str(&format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6}\n",
        operating_point,
        condition.name,
        counters.episodes,
        counters.alarms,
        counters.false_alarms,
        counters.missed_detections,
        rate(counters.false_alarms, counters.episodes),
        rate(counters.missed_detections, counters.episodes),
        counters.fallback_activations,
        counters.fallback_no_action,
        counters.predictive_proposal_preserved,
        counters.fallback_proposal_preserved,
        counters.predictive_viable_actions as f64 / counters.episodes as f64,
        counters.fallback_viable_actions as f64 / counters.episodes as f64,
    ));
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: allostatic-ood-fallback-bench <raw.csv> <summary.csv>",
        )
        .into());
    }

    let calibration_scores: Vec<f64> = (0..CALIBRATION_EPISODES)
        .map(|offset| score_for(CONDITIONS[0], 10_000 + offset))
        .collect();
    let threshold = calibrate_drift_threshold(&calibration_scores, TARGET_FALSE_ALARM_RATE)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "drift calibration failed"))?;
    let limits = ViabilityLimits::new(0.15, 0.88)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid viability limits"))?;

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let mut row_count = 0_usize;

    for point in OPERATING_POINTS {
        let state = state_for(point)?;
        for condition in CONDITIONS {
            let mut counters = Counters::default();
            for episode in 0..EVALUATION_EPISODES {
                let seed = 20_000 + episode;
                let dynamics = dynamics_for(seed)?;
                let predictive = evaluate_action_set(state, dynamics, limits, true);
                let fallback =
                    evaluate_horizon_action_set(state, dynamics, limits, true, FALLBACK_HORIZON)
                        .ok_or_else(|| {
                            IoError::new(
                                ErrorKind::InvalidData,
                                "reachability fallback evaluator failed",
                            )
                        })?;
                let score = score_for(condition, seed + point.reserve.to_bits());
                let proposed = Action::Push;
                let decision = select_control_action(
                    score,
                    threshold,
                    proposed,
                    predictive,
                    fallback,
                )
                .ok_or_else(|| {
                    IoError::new(
                        ErrorKind::InvalidData,
                        "drift assessment rejected a finite synthetic score",
                    )
                })?;
                let detected = decision.assessment().drift_detected();
                let selected = decision.selected_action();
                let selected_certified = selected.is_some_and(|action| {
                    if detected {
                        fallback.contains(action)
                    } else {
                        predictive.contains(action)
                    }
                });
                if selected.is_some() && !selected_certified {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        "selected action is not certified by the active controller",
                    )
                    .into());
                }

                counters.observe(
                    condition,
                    detected,
                    selected,
                    predictive.count(),
                    fallback.count(),
                    predictive.contains(proposed),
                    fallback.contains(proposed),
                );

                raw.push_str(&format!(
                    "{},{},{},{:.9},{:.9},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    point.name,
                    condition.name,
                    episode,
                    score,
                    threshold.score_threshold(),
                    csv_bool(detected),
                    control_mode_name(decision.assessment().mode()),
                    action_name(Some(proposed)),
                    predictive.count(),
                    fallback.count(),
                    action_name(predictive.filter(proposed)),
                    action_name(fallback.filter(proposed)),
                    action_name(selected),
                    csv_bool(selected_certified),
                    csv_bool(decision.no_certified_action()),
                    csv_bool(predictive.contains(proposed)),
                    csv_bool(fallback.contains(proposed)),
                ));
                row_count += 1;
            }
            summary_row(&mut summary, point.name, condition, counters);
        }
    }

    let expected_rows =
        OPERATING_POINTS.len() * CONDITIONS.len() * EVALUATION_EPISODES as usize;
    if row_count != expected_rows || row_count != 4_500 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("expected 4,500 episode rows, generated {row_count}"),
        )
        .into());
    }

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    Ok(())
}
