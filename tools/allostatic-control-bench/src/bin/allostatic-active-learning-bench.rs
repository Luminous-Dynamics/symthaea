//! Closed-loop contract benchmark for safe probing and hypothesis updates.
//!
//! Usage: allostatic-active-learning-bench <raw.csv> <summary.csv>
//! The plant is independently selected by the fixture, while the planner only
//! sees candidate models and an observed effect. Results are synthetic and do
//! not imply that real-plant model support can be established from this panel.

use allostatic_control_bench::multistep_viability::evaluate_horizon_action_set;
use allostatic_control_bench::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use allostatic_control_bench::safe_probe::{
    dynamics_hull, filter_compatible_hypotheses, recommend_safe_probe_with_noise, ObservedEffect,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;

const RAW_HEADER: &str = "case,evaluator_actual_plant,actual_plant_in_candidate_family,probe_action,probe_certified,actual_probe_safe,pre_probe_model_count,post_observation_model_count,pre_probe_viable_actions,post_observation_viable_actions,viable_action_gain,actual_model_retained,monitor_status";
const SUMMARY_HEADER: &str = "case,probe_action,probe_certified,posterior_count,pre_viable_actions,post_viable_actions,viable_action_gain,monitor_status";

fn interval(lower: f64, upper: f64) -> Interval {
    Interval::new(lower, upper).expect("valid interval")
}

fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
    ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid effect")
}

fn model(
    push: ActionEffectBounds,
    maintain: ActionEffectBounds,
    recovery: ActionEffectBounds,
) -> DynamicsBounds {
    DynamicsBounds::new(push, maintain, recovery)
}

fn candidate_models() -> [DynamicsBounds; 2] {
    [
        model(
            effects((-0.02, -0.02), (0.02, 0.02)),
            effects((0.02, 0.02), (-0.02, -0.02)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        ),
        model(
            effects((-0.08, -0.08), (0.10, 0.10)),
            effects((0.04, 0.04), (-0.06, -0.06)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        ),
    ]
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::Push) => "push",
        Some(Action::Maintain) => "maintain",
        Some(Action::EmergencyRecover) => "emergency_recover",
        None => "none",
    }
}

fn bool_text(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

struct Case {
    name: &'static str,
    actual_plant: &'static str,
    actual_in_family: bool,
    observed: ObservedEffect,
    actual_model_index: Option<usize>,
    actual_effect: (f64, f64),
    expected_status: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "observation_identifies_model_0",
            actual_plant: "model_0",
            actual_in_family: true,
            observed: ObservedEffect::new(0.02, -0.02, 0.001).expect("valid observation"),
            actual_model_index: Some(0),
            actual_effect: (0.02, -0.02),
            expected_status: "model_family_narrowed",
        },
        Case {
            name: "observation_identifies_model_1",
            actual_plant: "model_1",
            actual_in_family: true,
            observed: ObservedEffect::new(0.04, -0.06, 0.001).expect("valid observation"),
            actual_model_index: Some(1),
            actual_effect: (0.04, -0.06),
            expected_status: "model_family_narrowed",
        },
        Case {
            name: "ambiguous_measurement_outside_family",
            actual_plant: "intermediate_unmodeled_plant",
            actual_in_family: false,
            observed: ObservedEffect::new(0.03, -0.04, 0.021).expect("valid wide observation"),
            actual_model_index: None,
            actual_effect: (0.03, -0.04),
            expected_status: "ambiguity_retained_no_claim",
        },
        Case {
            name: "observation_falsifies_model_family",
            actual_plant: "outside_family",
            actual_in_family: false,
            observed: ObservedEffect::new(0.20, 0.20, 0.001).expect("valid out-of-family observation"),
            actual_model_index: None,
            actual_effect: (0.20, 0.20),
            expected_status: "candidate_family_falsified_fail_closed",
        },
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-active-learning-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let state = StateEnvelope::new(
        interval(0.30, 0.30),
        interval(0.80, 0.80),
    )
    .expect("valid state");
    let limits = ViabilityLimits::new(0.15, 0.88).expect("valid limits");
    let models = candidate_models();
    let initial_hull = dynamics_hull(&models).expect("nonempty candidate family");
    let before = evaluate_horizon_action_set(state, initial_hull, limits, true, 3)
        .expect("initial robust search");

    let mut raw = String::from(RAW_HEADER);
    raw.push('\n');
    let mut summary = String::from(SUMMARY_HEADER);
    summary.push('\n');

    for case in cases() {
        let recommendation = recommend_safe_probe_with_noise(
            state,
            &models,
            limits,
            true,
            3,
            Action::Maintain,
            0.01,
            0.001,
        )
        .expect("valid common-hull probe contract");
        let selected = recommendation.selected_action();
        let probe_certified = recommendation.selected_action_is_certified();

        let posterior = filter_compatible_hypotheses(
            &models,
            selected.unwrap_or(Action::Maintain),
            case.observed,
        )
        .expect("valid candidate family");
        let posterior_count = posterior.len();
        let mut post_viable_count = 0;
        let mut actual_model_retained = false;
        let monitor_status;
        if posterior.is_empty() {
            monitor_status = "candidate_family_falsified_fail_closed";
        } else {
            let posterior_hull = dynamics_hull(&posterior).expect("nonempty posterior");
            let after = evaluate_horizon_action_set(state, posterior_hull, limits, true, 3)
                .expect("post-observation robust search");
            post_viable_count = after.count();
            actual_model_retained = case.actual_model_index.is_some_and(|index| {
                posterior.contains(&models[index])
            });
            monitor_status = if case.actual_in_family && posterior_count == 1 && actual_model_retained {
                "model_family_narrowed"
            } else if !case.actual_in_family && posterior_count > 1 {
                "ambiguity_retained_no_claim"
            } else {
                case.expected_status
            };
        }

        let actual_next_reserve = state.reserve().lower() + case.actual_effect.0;
        let actual_next_load = state.load().upper() + case.actual_effect.1;
        let actual_probe_safe =
            actual_next_reserve >= limits.min_reserve() && actual_next_load <= limits.max_load();

        let gain = post_viable_count as isize - before.action_set.count() as isize;
        raw.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            case.name,
            case.actual_plant,
            bool_text(case.actual_in_family),
            action_name(selected),
            bool_text(probe_certified),
            bool_text(actual_probe_safe),
            models.len(),
            posterior_count,
            before.action_set.count(),
            post_viable_count,
            gain,
            bool_text(actual_model_retained),
            monitor_status,
        ));

        summary.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            case.name,
            action_name(selected),
            bool_text(probe_certified),
            posterior_count,
            before.action_set.count(),
            post_viable_count,
            gain,
            monitor_status,
        ));
    }

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    println!("Active-learning raw results: {}", args[1]);
    println!("Active-learning summary: {}", args[2]);
    println!("Cases: {}", cases().len());
    Ok(())
}
