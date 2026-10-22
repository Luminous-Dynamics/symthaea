//! Deterministic benchmark for safety-constrained active model discrimination.
//!
//! Usage: allostatic-safe-probe-bench <raw.csv> <summary.csv>
//! This fixture validates action-selection contracts only. It is not evidence
//! that the candidate model family contains a real plant or that probes are safe
//! outside the declared rectangular model hull.

use allostatic_control_bench::multistep_viability::evaluate_horizon_action_set;
use allostatic_control_bench::robust_viability::{
    evaluate_action_set, ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope,
    ViabilityLimits,
};
use allostatic_control_bench::safe_probe::{dynamics_hull, recommend_safe_probe};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;

const RAW_HEADER: &str = "case,state_reserve_lower,state_reserve_upper,state_load_lower,state_load_upper,horizon,shield_can_recover,hypothesis_count,minimum_discrimination,push_one_step_safe,push_horizon_safe,maintain_horizon_safe,recovery_horizon_safe,safe_action_count,push_discrimination,maintain_discrimination,recovery_discrimination,selected_action,selected_score,selected_certified,expected_contract";
const SUMMARY_HEADER: &str = "case,probe_recommended,selected_action,selected_score,safe_action_count,push_one_step_safe,push_horizon_safe,expected_contract";

struct Case {
    name: &'static str,
    state: StateEnvelope,
    models: [DynamicsBounds; 2],
    horizon: usize,
    shield_can_recover: bool,
    minimum_discrimination: f64,
    preferred_action: Action,
    expected_contract: &'static str,
}

fn interval(lower: f64, upper: f64) -> Interval {
    Interval::new(lower, upper).expect("valid fixture interval")
}

fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
    StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid fixture state")
}

fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
    ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid fixture effect")
}

fn model(
    push: ActionEffectBounds,
    maintain: ActionEffectBounds,
    recovery: ActionEffectBounds,
) -> DynamicsBounds {
    DynamicsBounds::new(push, maintain, recovery)
}

fn informative_models() -> [DynamicsBounds; 2] {
    [
        model(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        ),
        model(
            effects((-0.09, -0.08), (0.09, 0.10)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        ),
    ]
}

fn uninformative_models() -> [DynamicsBounds; 2] {
    let same = model(
        effects((-0.02, -0.01), (0.01, 0.02)),
        effects((0.02, 0.03), (-0.02, -0.01)),
        effects((0.08, 0.10), (-0.10, -0.08)),
    );
    [same, same]
}

fn horizon_dead_end_models() -> [DynamicsBounds; 2] {
    let first = model(
        effects((-0.005, -0.005), (0.01, 0.01)),
        effects((0.0, 0.0), (0.005, 0.005)),
        effects((-0.02, -0.02), (-0.30, -0.25)),
    );
    let second = model(
        effects((-0.006, -0.006), (0.01, 0.01)),
        effects((0.0, 0.0), (0.005, 0.005)),
        effects((-0.02, -0.02), (-0.30, -0.25)),
    );
    [first, second]
}

fn recovery_informative_models() -> [DynamicsBounds; 2] {
    [
        model(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.02, 0.03), (-0.02, -0.01)),
        ),
        model(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.15, 0.20), (-0.25, -0.20)),
        ),
    ]
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "informative_safe_action",
            state: state((0.50, 0.50), (0.50, 0.50)),
            models: informative_models(),
            horizon: 3,
            shield_can_recover: true,
            minimum_discrimination: 0.02,
            preferred_action: Action::Maintain,
            expected_contract: "probe_selected_push",
        },
        Case {
            name: "informative_action_unsafe",
            state: state((0.18, 0.18), (0.80, 0.80)),
            models: informative_models(),
            horizon: 3,
            shield_can_recover: true,
            minimum_discrimination: 0.02,
            preferred_action: Action::Push,
            expected_contract: "no_probe_unsafe_discriminator",
        },
        Case {
            name: "indistinguishable_models",
            state: state((0.50, 0.50), (0.50, 0.50)),
            models: uninformative_models(),
            horizon: 3,
            shield_can_recover: true,
            minimum_discrimination: 0.0,
            preferred_action: Action::Maintain,
            expected_contract: "no_probe_without_information",
        },
        Case {
            name: "one_step_safe_dead_end",
            state: state((0.16, 0.16), (0.87, 0.87)),
            models: horizon_dead_end_models(),
            horizon: 2,
            shield_can_recover: true,
            minimum_discrimination: 0.0,
            preferred_action: Action::Push,
            expected_contract: "no_probe_without_safe_continuation",
        },
        Case {
            name: "recovery_authority_required",
            state: state((0.50, 0.50), (0.50, 0.50)),
            models: recovery_informative_models(),
            horizon: 2,
            shield_can_recover: false,
            minimum_discrimination: 0.02,
            preferred_action: Action::EmergencyRecover,
            expected_contract: "no_probe_without_recovery_authority",
        },
        Case {
            name: "invalid_horizon_rejected",
            state: state((0.50, 0.50), (0.50, 0.50)),
            models: informative_models(),
            horizon: 99,
            shield_can_recover: true,
            minimum_discrimination: 0.02,
            preferred_action: Action::Push,
            expected_contract: "invalid_request_rejected",
        },
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-safe-probe-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let limits = ViabilityLimits::new(0.15, 0.88).expect("valid limits");
    let mut raw = String::from(RAW_HEADER);
    raw.push('\n');
    let mut summary = String::from(SUMMARY_HEADER);
    summary.push('\n');

    for case in cases() {
        let hull = dynamics_hull(&case.models).expect("well-formed candidate models");
        let one_step = evaluate_action_set(
            case.state,
            hull,
            limits,
            case.shield_can_recover,
        );
        let horizon = evaluate_horizon_action_set(
            case.state,
            hull,
            limits,
            case.shield_can_recover,
            case.horizon,
        );
        let recommendation = recommend_safe_probe(
            case.state,
            &case.models,
            limits,
            case.shield_can_recover,
            case.horizon,
            case.preferred_action,
            case.minimum_discrimination,
        );

        let (safe_count, push_horizon_safe, maintain_horizon_safe, recovery_horizon_safe,
            scores, selected, selected_score, selected_certified) =
            if let Some(result) = recommendation {
                let safe = result.safe_actions();
                (
                    safe.count(),
                    safe.push(),
                    safe.maintain(),
                    safe.emergency_recover(),
                    result.action_scores(),
                    result.selected_action(),
                    result.selected_score(),
                    result.selected_action_is_certified(),
                )
            } else {
                (0, false, false, false, [0.0; 3], None, 0.0, false)
            };

        let push_horizon_safe = horizon
            .map(|result| result.action_set.push())
            .unwrap_or(push_horizon_safe);
        let maintain_horizon_safe = horizon
            .map(|result| result.action_set.maintain())
            .unwrap_or(maintain_horizon_safe);
        let recovery_horizon_safe = horizon
            .map(|result| result.action_set.emergency_recover())
            .unwrap_or(recovery_horizon_safe);

        raw.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{},{:.6},{},{}\n",
            case.name,
            case.state.reserve().lower(),
            case.state.reserve().upper(),
            case.state.load().lower(),
            case.state.load().upper(),
            case.horizon,
            bool_text(case.shield_can_recover),
            case.models.len(),
            case.minimum_discrimination,
            bool_text(one_step.push()),
            bool_text(push_horizon_safe),
            bool_text(maintain_horizon_safe),
            bool_text(recovery_horizon_safe),
            safe_count,
            scores[0],
            scores[1],
            scores[2],
            action_name(selected),
            selected_score,
            bool_text(selected_certified),
            case.expected_contract,
        ));

        summary.push_str(&format!(
            "{},{},{},{:.6},{},{},{},{}\n",
            case.name,
            bool_text(selected.is_some()),
            action_name(selected),
            selected_score,
            safe_count,
            bool_text(one_step.push()),
            bool_text(push_horizon_safe),
            case.expected_contract,
        ));
    }

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    println!("Active-probe raw results: {}", args[1]);
    println!("Active-probe summary: {}", args[2]);
    println!("Cases: {}", cases().len());
    Ok(())
}
