//! Evaluate a calibrated OOD monitor and a separately widened reachability fallback.
//!
//! Usage: allostatic-ood-fallback-bench <raw.csv> <summary.csv>
//! The score distributions and true transition models are deterministic synthetic
//! controls. They test fail-closed switching and metric accounting, not real-world
//! detector calibration or guarantees on a physical controller.

use allostatic_control_bench::drift_monitor::{
    calibrate_drift_threshold, select_control_action, ControlMode,
};
use allostatic_control_bench::multistep_viability::{
    evaluate_horizon_action_set, HorizonViableActionSet,
};
use allostatic_control_bench::robust_viability::{
    evaluate_action_set, ActionEffectBounds, DynamicsBounds, Interval, RobustActionSet,
    StateEnvelope, ViabilityLimits,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};

const TARGET_FALSE_ALARM_RATE: f64 = 0.05;
const CALIBRATION_EPISODES: u64 = 159;
const EVALUATION_EPISODES: u64 = 300;
const FALLBACK_HORIZON: usize = 3;
const RAW_HEADER: &str = "operating_point,condition,episode_seed,ood_score,score_threshold,drift_detected,control_mode,proposed_action,predictive_viable_count,fallback_viable_count,predictive_action,fallback_action,selected_action,active_filter_certified,no_certified_action,predictive_proposal_preserved,fallback_proposal_preserved,actual_one_step_viable_count,actual_horizon_viable_count,selected_true_one_step_safe,selected_true_horizon_viable,false_one_step_safety_certificate,actual_inside_predictive_envelope,actual_inside_fallback_envelope,predictive_action_true_one_step_safe,predictive_action_true_horizon_viable,fallback_action_true_one_step_safe,fallback_action_true_horizon_viable";
const SUMMARY_HEADER: &str = "operating_point,condition,episodes,alarms,false_alarms,missed_detections,false_alarm_rate,miss_rate,fallback_activations,fallback_no_action,predictive_proposal_preserved,fallback_proposal_preserved,mean_predictive_viable_actions,mean_fallback_viable_actions,mean_actual_one_step_viable_actions,mean_actual_horizon_viable_actions,selected_actions,selected_true_one_step_safe,selected_true_horizon_viable,false_one_step_safety_certificates,false_one_step_certificate_rate,actual_inside_predictive_envelope,actual_inside_fallback_envelope,predictive_policy_selected_actions,predictive_policy_false_one_step_certificates,predictive_policy_false_certificate_rate,predictive_policy_horizon_viable_selections,fallback_policy_selected_actions,fallback_policy_false_one_step_certificates,fallback_policy_false_certificate_rate,fallback_policy_horizon_viable_selections";

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
#[derive(Clone, Copy, Debug)]
struct EpisodeObservation {
    condition: Condition,
    detected: bool,
    proposed_action: Action,
    predictive_action: Option<Action>,
    fallback_action: Option<Action>,
    selected_action: Option<Action>,
    predictive_viable_count: usize,
    fallback_viable_count: usize,
    actual_one_step: RobustActionSet,
    actual_horizon: HorizonViableActionSet,
    actual_inside_predictive_envelope: bool,
    actual_inside_fallback_envelope: bool,
}

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
    actual_one_step_viable_actions: usize,
    actual_horizon_viable_actions: usize,
    selected_actions: usize,
    selected_true_one_step_safe: usize,
    selected_true_horizon_viable: usize,
    false_one_step_safety_certificates: usize,
    actual_inside_predictive_envelope: usize,
    actual_inside_fallback_envelope: usize,
    predictive_policy_selected_actions: usize,
    predictive_policy_false_one_step_certificates: usize,
    predictive_policy_horizon_viable_selections: usize,
    fallback_policy_selected_actions: usize,
    fallback_policy_false_one_step_certificates: usize,
    fallback_policy_horizon_viable_selections: usize,
}

impl Counters {
    fn observe(&mut self, sample: EpisodeObservation) {
        let selected_true_one_step_safe = sample
            .selected_action
            .is_some_and(|action| sample.actual_one_step.contains(action));
        let selected_true_horizon_viable = sample
            .selected_action
            .is_some_and(|action| sample.actual_horizon.contains(action));
        let predictive_true_one_step_safe = sample
            .predictive_action
            .is_some_and(|action| sample.actual_one_step.contains(action));
        let predictive_true_horizon_viable = sample
            .predictive_action
            .is_some_and(|action| sample.actual_horizon.contains(action));
        let fallback_true_one_step_safe = sample
            .fallback_action
            .is_some_and(|action| sample.actual_one_step.contains(action));
        let fallback_true_horizon_viable = sample
            .fallback_action
            .is_some_and(|action| sample.actual_horizon.contains(action));

        self.episodes += 1;
        self.alarms += usize::from(sample.detected);
        self.false_alarms += usize::from(!sample.condition.shifted && sample.detected);
        self.missed_detections +=
            usize::from(sample.condition.shifted && !sample.detected);
        self.fallback_activations += usize::from(sample.detected);
        self.fallback_no_action +=
            usize::from(sample.detected && sample.selected_action.is_none());
        self.predictive_proposal_preserved +=
            usize::from(sample.predictive_action == Some(sample.proposed_action));
        self.fallback_proposal_preserved +=
            usize::from(sample.fallback_action == Some(sample.proposed_action));
        self.predictive_viable_actions += sample.predictive_viable_count;
        self.fallback_viable_actions += sample.fallback_viable_count;
        self.actual_one_step_viable_actions += sample.actual_one_step.count();
        self.actual_horizon_viable_actions += sample.actual_horizon.count();
        self.selected_actions += usize::from(sample.selected_action.is_some());
        self.selected_true_one_step_safe +=
            usize::from(sample.selected_action.is_some() && selected_true_one_step_safe);
        self.selected_true_horizon_viable +=
            usize::from(sample.selected_action.is_some() && selected_true_horizon_viable);
        self.false_one_step_safety_certificates += usize::from(
            sample.selected_action.is_some() && !selected_true_one_step_safe,
        );
        self.actual_inside_predictive_envelope +=
            usize::from(sample.actual_inside_predictive_envelope);
        self.actual_inside_fallback_envelope += usize::from(sample.actual_inside_fallback_envelope);
        self.predictive_policy_selected_actions += usize::from(sample.predictive_action.is_some());
        self.predictive_policy_false_one_step_certificates += usize::from(
            sample.predictive_action.is_some() && !predictive_true_one_step_safe,
        );
        self.predictive_policy_horizon_viable_selections += usize::from(
            sample.predictive_action.is_some() && predictive_true_horizon_viable,
        );
        self.fallback_policy_selected_actions += usize::from(sample.fallback_action.is_some());
        self.fallback_policy_false_one_step_certificates += usize::from(
            sample.fallback_action.is_some() && !fallback_true_one_step_safe,
        );
        self.fallback_policy_horizon_viable_selections += usize::from(
            sample.fallback_action.is_some() && fallback_true_horizon_viable,
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

/// Synthetic scalar episode score. Nominal scores lie in [0,.25); moderate
/// shifts overlap that range; strong shifts lie in [.50,1). This is a designed
/// control distribution, not a trained or validated OOD detector.
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

fn predictive_dynamics_for(seed: u64) -> Result<DynamicsBounds, Box<dyn Error>> {
    let push = shifted_effect((-0.030, -0.005), (0.030, 0.070), seed, 0)?;
    let maintain = shifted_effect((0.010, 0.040), (-0.010, 0.030), seed, 1)?;
    let emergency_recover = shifted_effect((0.025, 0.060), (-0.160, -0.080), seed, 2)?;
    Ok(DynamicsBounds::new(push, maintain, emergency_recover))
}

/// OOD fallback uses a wider, predeclared reachability envelope rather than
/// reusing the predictive model's narrower intervals.
fn fallback_dynamics() -> Result<DynamicsBounds, Box<dyn Error>> {
    let push = ActionEffectBounds::new(
        interval(-0.09, 0.01)?,
        interval(0.02, 0.18)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid fallback push bounds"))?;
    let maintain = ActionEffectBounds::new(
        interval(-0.03, 0.06)?,
        interval(-0.02, 0.10)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid fallback maintain bounds"))?;
    let emergency_recover = ActionEffectBounds::new(
        interval(-0.02, 0.08)?,
        interval(-0.20, -0.02)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid fallback recovery bounds"))?;
    Ok(DynamicsBounds::new(push, maintain, emergency_recover))
}

fn center(bounds: ActionEffectBounds) -> (f64, f64) {
    (
        (bounds.reserve_delta().lower() + bounds.reserve_delta().upper()) * 0.5,
        (bounds.load_delta().lower() + bounds.load_delta().upper()) * 0.5,
    )
}

fn point_effect(reserve: f64, load: f64) -> Result<ActionEffectBounds, Box<dyn Error>> {
    ActionEffectBounds::new(
        interval(reserve, reserve)?,
        interval(load, load)?,
    )
    .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid true point effect").into())
}

/// The synthetic actual plant starts at the predictive interval center. Shift
/// offsets move it out of the narrow predictive envelope, but inside the widened
/// OOD fallback envelope. The offsets are designed falsification controls.
fn actual_dynamics_for(
    seed: u64,
    condition: Condition,
) -> Result<DynamicsBounds, Box<dyn Error>> {
    let estimate = predictive_dynamics_for(seed)?;
    let offsets = match condition.name {
        "nominal" => [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "moderate_shift" => [
            (-0.025, 0.050),
            (-0.015, 0.035),
            (-0.030, 0.060),
        ],
        "strong_shift" => [
            (-0.055, 0.100),
            (-0.025, 0.070),
            (-0.035, 0.080),
        ],
        _ => return Err(IoError::new(ErrorKind::InvalidInput, "unknown shift condition").into()),
    };
    let (push_r, push_l) = center(estimate.effect_for_action(Action::Push));
    let (maintain_r, maintain_l) = center(estimate.effect_for_action(Action::Maintain));
    let (recover_r, recover_l) = center(estimate.effect_for_action(Action::EmergencyRecover));
    Ok(DynamicsBounds::new(
        point_effect(push_r + offsets[0].0, push_l + offsets[0].1)?,
        point_effect(maintain_r + offsets[1].0, maintain_l + offsets[1].1)?,
        point_effect(recover_r + offsets[2].0, recover_l + offsets[2].1)?,
    ))
}

fn dynamics_contained(actual: DynamicsBounds, envelope: DynamicsBounds) -> bool {
    [Action::Push, Action::Maintain, Action::EmergencyRecover]
        .into_iter()
        .all(|action| {
            let observed = actual.effect_for_action(action);
            let allowed = envelope.effect_for_action(action);
            observed.reserve_delta().lower() >= allowed.reserve_delta().lower()
                && observed.reserve_delta().upper() <= allowed.reserve_delta().upper()
                && observed.load_delta().lower() >= allowed.load_delta().lower()
                && observed.load_delta().upper() <= allowed.load_delta().upper()
        })
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

fn csv_row(output: &mut String, fields: Vec<String>) {
    output.push_str(&fields.join(","));
    output.push('\n');
}

fn summary_row(
    output: &mut String,
    operating_point: &str,
    condition: Condition,
    counters: Counters,
) {
    let fields = vec![
        operating_point.to_string(),
        condition.name.to_string(),
        counters.episodes.to_string(),
        counters.alarms.to_string(),
        counters.false_alarms.to_string(),
        counters.missed_detections.to_string(),
        rate(counters.false_alarms, counters.episodes),
        rate(counters.missed_detections, counters.episodes),
        counters.fallback_activations.to_string(),
        counters.fallback_no_action.to_string(),
        counters.predictive_proposal_preserved.to_string(),
        counters.fallback_proposal_preserved.to_string(),
        format!(
            "{:.6}",
            counters.predictive_viable_actions as f64 / counters.episodes as f64
        ),
        format!(
            "{:.6}",
            counters.fallback_viable_actions as f64 / counters.episodes as f64
        ),
        format!(
            "{:.6}",
            counters.actual_one_step_viable_actions as f64 / counters.episodes as f64
        ),
        format!(
            "{:.6}",
            counters.actual_horizon_viable_actions as f64 / counters.episodes as f64
        ),
        counters.selected_actions.to_string(),
        counters.selected_true_one_step_safe.to_string(),
        counters.selected_true_horizon_viable.to_string(),
        counters.false_one_step_safety_certificates.to_string(),
        rate(
            counters.false_one_step_safety_certificates,
            counters.selected_actions
        ),
        counters.actual_inside_predictive_envelope.to_string(),
        counters.actual_inside_fallback_envelope.to_string(),
        counters.predictive_policy_selected_actions.to_string(),
        counters.predictive_policy_false_one_step_certificates.to_string(),
        rate(
            counters.predictive_policy_false_one_step_certificates,
            counters.predictive_policy_selected_actions,
        ),
        counters.predictive_policy_horizon_viable_selections.to_string(),
        counters.fallback_policy_selected_actions.to_string(),
        counters.fallback_policy_false_one_step_certificates.to_string(),
        rate(
            counters.fallback_policy_false_one_step_certificates,
            counters.fallback_policy_selected_actions,
        ),
        counters.fallback_policy_horizon_viable_selections.to_string(),
    ];
    csv_row(output, fields);
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
    let fallback_dynamics = fallback_dynamics()?;

    let mut raw = format!("{RAW_HEADER}\n");
    let mut summary = format!("{SUMMARY_HEADER}\n");
    let mut row_count = 0_usize;

    for point in OPERATING_POINTS {
        let state = state_for(point)?;
        for condition in CONDITIONS {
            let mut counters = Counters::default();
            for episode in 0..EVALUATION_EPISODES {
                let seed = 20_000 + episode;
                let predictive_dynamics = predictive_dynamics_for(seed)?;
                let actual_dynamics = actual_dynamics_for(seed, condition)?;
                let actual_inside_predictive =
                    dynamics_contained(actual_dynamics, predictive_dynamics);
                let actual_inside_fallback = dynamics_contained(actual_dynamics, fallback_dynamics);
                if (condition.name == "nominal" && !actual_inside_predictive)
                    || (condition.name != "nominal" && actual_inside_predictive)
                    || !actual_inside_fallback
                {
                    return Err(IoError::new(
                        ErrorKind::InvalidData,
                        format!(
                            "synthetic model-containment label mismatch: condition={}",
                            condition.name
                        ),
                    )
                    .into());
                }
                let predictive = evaluate_action_set(
                    state,
                    predictive_dynamics,
                    limits,
                    true,
                );
                let fallback = evaluate_horizon_action_set(
                    state,
                    fallback_dynamics,
                    limits,
                    true,
                    FALLBACK_HORIZON,
                )
                .ok_or_else(|| {
                    IoError::new(
                        ErrorKind::InvalidData,
                        "reachability fallback evaluator failed",
                    )
                })?;
                let actual_one_step = evaluate_action_set(state, actual_dynamics, limits, true);
                let actual_horizon = evaluate_horizon_action_set(
                    state,
                    actual_dynamics,
                    limits,
                    true,
                    FALLBACK_HORIZON,
                )
                .ok_or_else(|| {
                    IoError::new(
                        ErrorKind::InvalidData,
                        "actual-system horizon evaluator failed",
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
                let selected_true_one_step_safe =
                    selected.is_some_and(|action| actual_one_step.contains(action));
                let selected_true_horizon_viable =
                    selected.is_some_and(|action| actual_horizon.contains(action));
                let false_safety_certificate =
                    selected.is_some() && !selected_true_one_step_safe;

                let predictive_action = predictive.filter(proposed);
                let fallback_action = fallback.filter(proposed);
                counters.observe(EpisodeObservation {
                    condition,
                    detected,
                    proposed_action: proposed,
                    predictive_action,
                    fallback_action,
                    selected_action: selected,
                    predictive_viable_count: predictive.count(),
                    fallback_viable_count: fallback.count(),
                    actual_one_step,
                    actual_horizon,
                    actual_inside_predictive_envelope: actual_inside_predictive,
                    actual_inside_fallback_envelope: actual_inside_fallback,
                });

                let fields = vec![
                    point.name.to_string(),
                    condition.name.to_string(),
                    episode.to_string(),
                    format!("{score:.9}"),
                    format!("{:.9}", threshold.score_threshold()),
                    csv_bool(detected).to_string(),
                    control_mode_name(decision.assessment().mode()).to_string(),
                    action_name(Some(proposed)).to_string(),
                    predictive.count().to_string(),
                    fallback.count().to_string(),
                    action_name(predictive_action).to_string(),
                    action_name(fallback_action).to_string(),
                    action_name(selected).to_string(),
                    csv_bool(selected_certified).to_string(),
                    csv_bool(decision.no_certified_action()).to_string(),
                    csv_bool(predictive.contains(proposed)).to_string(),
                    csv_bool(fallback.contains(proposed)).to_string(),
                    actual_one_step.count().to_string(),
                    actual_horizon.count().to_string(),
                    csv_bool(selected_true_one_step_safe).to_string(),
                    csv_bool(selected_true_horizon_viable).to_string(),
                    csv_bool(false_safety_certificate).to_string(),
                    csv_bool(actual_inside_predictive).to_string(),
                    csv_bool(actual_inside_fallback).to_string(),
                    csv_bool(
                        predictive_action
                            .is_some_and(|action| actual_one_step.contains(action)),
                    )
                    .to_string(),
                    csv_bool(
                        predictive_action
                            .is_some_and(|action| actual_horizon.contains(action)),
                    )
                    .to_string(),
                    csv_bool(
                        fallback_action
                            .is_some_and(|action| actual_one_step.contains(action)),
                    )
                    .to_string(),
                    csv_bool(
                        fallback_action
                            .is_some_and(|action| actual_horizon.contains(action)),
                    )
                    .to_string(),
                ];
                csv_row(&mut raw, fields);
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
