//! AERB lost-acknowledgement conformance example.
//!
//! This is a synthetic evaluator-side demonstration of the existing prior-free
//! robust decision-value API. It contains no probabilities and executes no probe.
//! The AERB candidate process must never receive the hidden world/loss tables.

use std::collections::BTreeMap;

use decision_relevant_predictability::robust::{
    evaluate_robust_probe, RobustDecisionProblem, RobustProbe, RobustProbeContext,
    RobustProbeOutcome,
};

fn loss_row(
    states: &[&str],
    values: &[f64],
) -> BTreeMap<String, f64> {
    states
        .iter()
        .zip(values.iter())
        .map(|(state, loss)| ((*state).to_owned(), *loss))
        .collect()
}

fn main() {
    let states = [
        "not_dispatched",
        "dispatched_no_effect",
        "applied_once_ack_lost",
        "partial_effect",
        "duplicate_effect",
        "stale_authority_effect",
        "unresolved_command_holder",
        "open_world_residual",
    ];

    let problem = RobustDecisionProblem {
        state_ids: states.iter().map(|state| (*state).to_owned()).collect(),
        decision_ids: vec![
            "hold_reconcile".to_owned(),
            "recommend_fresh_attempt".to_owned(),
            "abstain".to_owned(),
        ],
        decision_loss: BTreeMap::from([
            (
                "hold_reconcile".to_owned(),
                loss_row(&states, &[0.35, 0.35, 0.05, 0.10, 0.15, 0.25, 0.25, 0.30]),
            ),
            (
                "recommend_fresh_attempt".to_owned(),
                loss_row(&states, &[0.0, 0.0, 0.95, 0.95, 1.0, 1.0, 0.90, 1.0]),
            ),
            (
                "abstain".to_owned(),
                loss_row(&states, &[0.45; 8]),
            ),
        ]),
        maximum_worst_case_loss: 1.0,
    };

    // A journal read plus independently validated scope-closure receipt
    // separates known effect histories while retaining an unresolved outcome.
    // A journal read alone cannot eliminate open_world_residual.
    let journal_probe = RobustProbe {
        probe_id: "reconcile_effect_and_scope".to_owned(),
        acquisition_cost_units: 2.0,
        result_delay_steps: 5,
        outcomes: vec![
            RobustProbeOutcome {
                outcome_id: "complete_no_effect".to_owned(),
                possible_state_ids: vec![
                    "not_dispatched".to_owned(),
                    "dispatched_no_effect".to_owned(),
                ],
            },
            RobustProbeOutcome {
                outcome_id: "complete_applied_once".to_owned(),
                possible_state_ids: vec!["applied_once_ack_lost".to_owned()],
            },
            RobustProbeOutcome {
                outcome_id: "partial_duplicate_or_stale".to_owned(),
                possible_state_ids: vec![
                    "partial_effect".to_owned(),
                    "duplicate_effect".to_owned(),
                    "stale_authority_effect".to_owned(),
                ],
            },
            RobustProbeOutcome {
                outcome_id: "scope_incomplete_or_unmodelled".to_owned(),
                possible_state_ids: vec![
                    "unresolved_command_holder".to_owned(),
                    "open_world_residual".to_owned(),
                ],
            },
        ],
    };

    // This second probe returns distinct outcomes but each outcome still
    // mixes worlds requiring different dispositions. It illustrates that
    // observation cardinality/entropy is not the same as decision value.
    let weak_probe = RobustProbe {
        probe_id: "weakly_discriminating_camera_sample".to_owned(),
        acquisition_cost_units: 1.0,
        result_delay_steps: 1,
        outcomes: vec![
            RobustProbeOutcome {
                outcome_id: "visual_pattern_a".to_owned(),
                possible_state_ids: vec![
                    "not_dispatched".to_owned(),
                    "duplicate_effect".to_owned(),
                    "unresolved_command_holder".to_owned(),
                    "open_world_residual".to_owned(),
                ],
            },
            RobustProbeOutcome {
                outcome_id: "visual_pattern_b".to_owned(),
                possible_state_ids: vec![
                    "dispatched_no_effect".to_owned(),
                    "applied_once_ack_lost".to_owned(),
                    "partial_effect".to_owned(),
                    "stale_authority_effect".to_owned(),
                    "open_world_residual".to_owned(),
                ],
            },
        ],
    };

    let context = RobustProbeContext {
        permitted_probe_ids: vec![
            "reconcile_effect_and_scope".to_owned(),
            "weakly_discriminating_camera_sample".to_owned(),
        ],
        remaining_cost_budget_units: 2.0,
        current_step: 40,
        decision_deadline_step: 50,
    };

    let journal = evaluate_robust_probe(&problem, &journal_probe, &context)
        .expect("journal probe has complete outcome support");
    let weak = evaluate_robust_probe(&problem, &weak_probe, &context)
        .expect("weak probe has complete outcome support");

    assert_eq!(journal.current_minimax_loss, 0.35);
    assert_eq!(journal.worst_case_post_observation_minimax_loss, 0.30);
    assert!((journal.guaranteed_worst_case_loss_reduction - 0.05).abs() < 1e-12);
    assert_eq!(weak.current_minimax_loss, 0.35);
    assert_eq!(weak.worst_case_post_observation_minimax_loss, 0.35);
    assert_eq!(weak.guaranteed_worst_case_loss_reduction, 0.0);

    println!(
        "AERB synthetic robust probe comparison: journal + independent scope closure guaranteed loss reduction = {:.2}; weak visual probe = {:.2}; no prior probabilities used.",
        journal.guaranteed_worst_case_loss_reduction,
        weak.guaranteed_worst_case_loss_reduction
    );
}
