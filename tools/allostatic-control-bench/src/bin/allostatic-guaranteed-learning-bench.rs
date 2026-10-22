//! Contract benchmark for guaranteed partial learning with finite hypotheses.
//!
//! Usage: allostatic-guaranteed-learning-bench <raw.csv> <summary.csv>
//! The fixture demonstrates guaranteed hypothesis-set reduction when complete
//! one-step identification is impossible. It is not A-SLS and makes no physical
//! or SOTA claim.

use allostatic_control_bench::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use allostatic_control_bench::safe_probe::{
    recommend_safe_probe_for_guaranteed_reduction, recommend_safe_probe_with_noise,
};
use allostatic_control_bench::Action;
use std::error::Error;
use std::fs;

const RAW_HEADER: &str = "case,candidate_hypotheses,horizon,measurement_radius,full_pairwise_action,full_pairwise_score,guaranteed_reduction_action,guaranteed_reduction_score,selected_action_certified,pairwise_distinguishable,worst_case_remaining_push,guaranteed_eliminations_push,worst_case_remaining_maintain,guaranteed_eliminations_maintain,worst_case_remaining_recovery,guaranteed_eliminations_recovery,expected_contract";
const SUMMARY_HEADER: &str = "case,full_pairwise_action,guaranteed_reduction_action,guaranteed_eliminations_push,expected_contract";

fn interval(lower: f64, upper: f64) -> Interval {
    Interval::new(lower, upper).expect("valid interval")
}

fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
    ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
        .expect("valid effect")
}

fn model(push: ActionEffectBounds) -> DynamicsBounds {
    let maintain = effects((0.02, 0.03), (-0.02, -0.01));
    let recovery = effects((0.08, 0.10), (-0.10, -0.08));
    DynamicsBounds::new(push, maintain, recovery)
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
        eprintln!("usage: allostatic-guaranteed-learning-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let hypotheses = [
        model(effects((-0.02, -0.01), (0.01, 0.02))),
        model(effects((-0.02, -0.01), (0.01, 0.02))),
        model(effects((-0.09, -0.08), (0.09, 0.10))),
    ];
    let state = StateEnvelope::new(interval(0.50, 0.50), interval(0.50, 0.50))
        .expect("valid state");
    let limits = ViabilityLimits::new(0.15, 0.88).expect("valid limits");
    let full_pairwise = recommend_safe_probe_with_noise(
        state,
        &hypotheses,
        limits,
        true,
        3,
        Action::Maintain,
        0.0,
        0.005,
    )
    .expect("valid pairwise selection request");
    let guaranteed_reduction = recommend_safe_probe_for_guaranteed_reduction(
        state,
        &hypotheses,
        limits,
        true,
        3,
        Action::Maintain,
        0.005,
    )
    .expect("valid guaranteed-reduction request");

    let remaining = guaranteed_reduction.worst_case_remaining_hypotheses();
    let eliminated = guaranteed_reduction.guaranteed_hypothesis_eliminations();
    let certified = guaranteed_reduction.selected_action_is_certified();
    let expected_contract = "partial_learning_without_full_identification";

    let mut raw = String::from(RAW_HEADER);
    raw.push('\n');
    raw.push_str(&format!(
        "three_candidate_partial_reduction,{},3,0.005,{},{:.9},{},{:.9},{},{},{},{},{},{},{},{},{}\n",
        guaranteed_reduction.candidate_hypotheses(),
        action_name(full_pairwise.selected_action()),
        full_pairwise.selected_score(),
        action_name(guaranteed_reduction.selected_action()),
        guaranteed_reduction.selected_score(),
        bool_text(certified),
        bool_text(guaranteed_reduction.selected_action_guarantees_pairwise_distinguishability()),
        remaining[0],
        eliminated[0],
        remaining[1],
        eliminated[1],
        remaining[2],
        eliminated[2],
        expected_contract,
    ));

    let mut summary = String::from(SUMMARY_HEADER);
    summary.push('\n');
    summary.push_str(&format!(
        "three_candidate_partial_reduction,{},{},{},{}\n",
        action_name(full_pairwise.selected_action()),
        action_name(guaranteed_reduction.selected_action()),
        eliminated[0],
        expected_contract,
    ));

    fs::write(&args[1], raw)?;
    fs::write(&args[2], summary)?;
    println!("Guaranteed-learning raw results: {}", args[1]);
    println!("Guaranteed-learning summary: {}", args[2]);
    println!("Cases: 1");
    Ok(())
}
