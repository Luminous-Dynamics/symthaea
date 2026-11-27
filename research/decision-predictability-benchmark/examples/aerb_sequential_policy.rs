//! AERB sequential counterfactual-policy conformance example.
//!
//! The evaluator-only oracle plans a bounded, non-actuating observation tree.
//! Its possible-state supports and policy tree are hidden evaluator data: they
//! must never be serialized into the candidate process or factual robot ledger.
//! A terminal fresh-attempt recommendation is still subject to the independent
//! RRVR execution-finality and authority gate; this example executes no action.

use std::collections::BTreeMap;

use decision_relevant_predictability::sequential::{
    SequentialAction, SequentialOutcome, SequentialPolicyNode, SequentialProtocolState,
    SequentialRobustProblem, enumerate_sequential_robust, evaluate_sequential_robust,
};

const STATES: [&str; 8] = [
    "not_dispatched",
    "dispatched_no_effect",
    "applied_once_ack_lost",
    "partial_effect",
    "duplicate_effect",
    "stale_authority_effect",
    "unresolved_command_holder",
    "open_world_residual",
];

fn map_values(keys: &[&str], values: &[f64]) -> BTreeMap<String, f64> {
    keys.iter()
        .zip(values.iter())
        .map(|(key, value)| ((*key).to_owned(), *value))
        .collect()
}

fn resource_costs(probe_units: u64, energy_units: u64, compute_units: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("probe_units".to_owned(), probe_units),
        ("sensor_energy_units".to_owned(), energy_units),
        ("compute_units".to_owned(), compute_units),
    ])
}

fn branch(
    outcome_id: &str,
    possible_state_ids: &[&str],
    next_protocol_state_id: &str,
) -> SequentialOutcome {
    SequentialOutcome {
        outcome_id: outcome_id.to_owned(),
        possible_state_ids: possible_state_ids
            .iter()
            .map(|state| (*state).to_owned())
            .collect(),
        next_protocol_state_id: next_protocol_state_id.to_owned(),
    }
}

fn problem() -> SequentialRobustProblem {
    let decision_loss = BTreeMap::from([
        (
            "hold_reconcile".to_owned(),
            map_values(&STATES, &[0.35, 0.35, 0.05, 0.10, 0.15, 0.25, 0.25, 0.30]),
        ),
        (
            "recommend_fresh_attempt".to_owned(),
            map_values(&STATES, &[0.0, 0.0, 0.95, 0.95, 1.0, 1.0, 0.90, 1.0]),
        ),
        (
            "abstain".to_owned(),
            map_values(&STATES, &[0.45; 8]),
        ),
    ]);

    // An ordinary endpoint-journal read narrows known histories but retains
    // open_world_residual in every outcome. This can make a second observation
    // useful only on some branches.
    let journal_read = SequentialAction {
        action_id: "read_endpoint_journal".to_owned(),
        resource_costs: resource_costs(1, 1, 1),
        result_delay_steps: 2,
        outcomes: vec![
            branch(
                "journal_no_effect",
                &[
                    "not_dispatched",
                    "dispatched_no_effect",
                    "open_world_residual",
                ],
                "after_journal_no_effect",
            ),
            branch(
                "journal_applied_once",
                &["applied_once_ack_lost", "open_world_residual"],
                "after_journal_applied_once",
            ),
            branch(
                "journal_partial_duplicate_or_stale",
                &[
                    "partial_effect",
                    "duplicate_effect",
                    "stale_authority_effect",
                    "open_world_residual",
                ],
                "after_journal_other_effect",
            ),
            branch(
                "journal_holder_unresolved",
                &["unresolved_command_holder", "open_world_residual"],
                "after_journal_unresolved",
            ),
        ],
    };

    // This is a combined effect reconciliation plus independently verified
    // scope-closure observation. Only the closed-scope outcomes can eliminate
    // the open-world residual; the final outcome explicitly preserves it.
    let scope_reconciliation = SequentialAction {
        action_id: "reconcile_effect_and_scope".to_owned(),
        resource_costs: resource_costs(2, 2, 1),
        result_delay_steps: 3,
        outcomes: vec![
            branch(
                "complete_no_effect",
                &["not_dispatched", "dispatched_no_effect"],
                "terminal",
            ),
            branch(
                "complete_applied_once",
                &["applied_once_ack_lost"],
                "terminal",
            ),
            branch(
                "partial_duplicate_or_stale",
                &[
                    "partial_effect",
                    "duplicate_effect",
                    "stale_authority_effect",
                ],
                "terminal",
            ),
            branch(
                "scope_incomplete_or_unmodelled",
                &["unresolved_command_holder", "open_world_residual"],
                "terminal",
            ),
        ],
    };

    SequentialRobustProblem {
        state_ids: STATES.iter().map(|state| (*state).to_owned()).collect(),
        decision_ids: vec![
            "hold_reconcile".to_owned(),
            "recommend_fresh_attempt".to_owned(),
            "abstain".to_owned(),
        ],
        decision_loss,
        maximum_worst_case_loss: 0.45,
        initial_protocol_state_id: "before_observation".to_owned(),
        protocol_states: vec![
            SequentialProtocolState {
                protocol_state_id: "before_observation".to_owned(),
                legal_action_ids: vec!["read_endpoint_journal".to_owned()],
            },
            SequentialProtocolState {
                protocol_state_id: "after_journal_no_effect".to_owned(),
                legal_action_ids: vec!["reconcile_effect_and_scope".to_owned()],
            },
            SequentialProtocolState {
                protocol_state_id: "after_journal_applied_once".to_owned(),
                legal_action_ids: vec!["reconcile_effect_and_scope".to_owned()],
            },
            SequentialProtocolState {
                protocol_state_id: "after_journal_other_effect".to_owned(),
                legal_action_ids: vec!["reconcile_effect_and_scope".to_owned()],
            },
            SequentialProtocolState {
                protocol_state_id: "after_journal_unresolved".to_owned(),
                legal_action_ids: vec!["reconcile_effect_and_scope".to_owned()],
            },
            SequentialProtocolState {
                protocol_state_id: "terminal".to_owned(),
                legal_action_ids: Vec::new(),
            },
        ],
        actions: vec![journal_read, scope_reconciliation],
        resource_budget: BTreeMap::from([
            ("probe_units".to_owned(), 3),
            ("sensor_energy_units".to_owned(), 3),
            ("compute_units".to_owned(), 2),
        ]),
        current_step: 40,
        decision_deadline_step: 50,
        maximum_actions: 2,
    }
}

fn main() {
    let problem = problem();
    let dynamic = evaluate_sequential_robust(&problem)
        .expect("AERB sequential dynamic programme must produce a policy");
    let exhaustive = enumerate_sequential_robust(&problem)
        .expect("AERB independent exhaustive enumerator must produce a policy");

    assert!((dynamic.current_minimax_loss - 0.35).abs() < 1e-12);
    assert!((dynamic.worst_case_terminal_loss - 0.30).abs() < 1e-12);
    assert!((exhaustive.worst_case_terminal_loss - 0.30).abs() < 1e-12);
    assert!(dynamic.policy_terminal_risk_gate_passed);
    assert_eq!(dynamic.worst_case_terminal_loss, exhaustive.worst_case_terminal_loss);

    let SequentialPolicyNode::Probe {
        action_id,
        branches,
    } = &dynamic.policy
    else {
        panic!("the only root action must be the endpoint-journal observation");
    };
    assert_eq!(action_id, "read_endpoint_journal");
    assert_eq!(branches.len(), 4);

    let no_effect = branches
        .iter()
        .find(|branch| branch.outcome_id == "journal_no_effect")
        .expect("journal no-effect branch");
    let SequentialPolicyNode::Probe {
        action_id: follow_up_id,
        branches: follow_up_branches,
    } = no_effect.continuation.as_ref()
    else {
        panic!("the no-effect ambiguity requires scope reconciliation");
    };
    assert_eq!(follow_up_id, "reconcile_effect_and_scope");
    assert_eq!(follow_up_branches.len(), 2);

    let complete_no_effect = follow_up_branches
        .iter()
        .find(|branch| branch.outcome_id == "complete_no_effect")
        .expect("closed-scope no-effect outcome");
    assert!(matches!(
        complete_no_effect.continuation.as_ref(),
        SequentialPolicyNode::Stop {
            decision_id,
            worst_case_terminal_loss
        } if decision_id == "recommend_fresh_attempt" && *worst_case_terminal_loss == 0.0
    ));

    let unresolved_scope = follow_up_branches
        .iter()
        .find(|branch| branch.outcome_id == "scope_incomplete_or_unmodelled")
        .expect("unresolved scope outcome");
    assert!(matches!(
        unresolved_scope.continuation.as_ref(),
        SequentialPolicyNode::Stop {
            decision_id,
            worst_case_terminal_loss
        } if decision_id == "hold_reconcile" && (*worst_case_terminal_loss - 0.30).abs() < 1e-12
    ));

    // Other first-stage outcomes do not trigger the second observation:
    // their worst-case terminal loss cannot be strictly improved by it.
    for first_stage in branches
        .iter()
        .filter(|branch| branch.outcome_id != "journal_no_effect")
    {
        assert!(matches!(
            first_stage.continuation.as_ref(),
            SequentialPolicyNode::Stop {
                decision_id,
                worst_case_terminal_loss
            } if decision_id == "hold_reconcile" && (*worst_case_terminal_loss - 0.30).abs() < 1e-12
        ));
    }

    // Five terminal paths: two after the conditional second probe and one
    // after each of the other three first-stage outcomes.
    assert_eq!(dynamic.terminal_resource_paths.len(), 5);

    println!(
        "AERB sequential conformance: root=read_endpoint_journal; follow-up=reconcile_effect_and_scope only after journal_no_effect; worst-case terminal loss={:.2}; terminal paths={}; DP/enumerator agree; no physical branch executed.",
        dynamic.worst_case_terminal_loss,
        dynamic.terminal_resource_paths.len()
    );
}
