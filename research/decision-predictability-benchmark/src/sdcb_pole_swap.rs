//! SDCB pole-swap integration fixture for the prior-free sequential policy oracle.
//!
//! The frozen two-world challenge is sourced from
//! docs/research/SDCB_POLE_SWAP_DECISION_SUFFICIENCY_V1.json (PR #14243).
//! This module maps its set-valued probe outcomes and terminal losses into the
//! existing #13567 oracle. It does not implement another planner, likelihood
//! engine, calibration model, evidence authority, or action executor.
//!
//! Important boundary: the policy oracle consumes *supports*, not outcome
//! probabilities. The one-bit nuisance entropy negative control is therefore
//! recorded in the accompanying manifest; this adapter checks that both
//! nuisance outcomes preserve the full state support.

use super::sequential::{
    enumerate_sequential_robust, evaluate_sequential_robust, SequentialAction,
    SequentialOutcome, SequentialPolicyNode, SequentialPolicyValue,
    SequentialProtocolState, SequentialRobustProblem,
};
use std::collections::{BTreeMap, BTreeSet};

const WORLD_A: &str = "world_a_physical_slow_sensor_fast";
const WORLD_B: &str = "world_b_physical_fast_sensor_slow";
const BOTH: [&str; 2] = [WORLD_A, WORLD_B];
const ROOT: &str = "root";
const TERMINAL: &str = "terminal";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionTask {
    /// Both worlds have the same uniquely optimal terminal action.
    DecisionInvariant,
    /// The optimal terminal action differs between the two worlds.
    DecisionSensitive,
}

fn resources(
    elapsed_slots: u64,
    reference_units: u64,
    actuator_energy_units: u64,
) -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("actuator_energy_units".to_owned(), actuator_energy_units),
        ("compute_units".to_owned(), 0),
        ("elapsed_slots".to_owned(), elapsed_slots),
        ("reference_units".to_owned(), reference_units),
    ])
}

fn outcome(
    outcome_id: &str,
    possible_state_ids: &[&str],
) -> SequentialOutcome {
    SequentialOutcome {
        outcome_id: outcome_id.to_owned(),
        possible_state_ids: possible_state_ids.iter().map(|id| (*id).to_owned()).collect(),
        next_protocol_state_id: TERMINAL.to_owned(),
    }
}

fn probe(
    action_id: &str,
    elapsed_slots: u64,
    reference_units: u64,
    actuator_energy_units: u64,
    result_delay_steps: u64,
    outcomes: Vec<SequentialOutcome>,
) -> SequentialAction {
    SequentialAction {
        action_id: action_id.to_owned(),
        resource_costs: resources(elapsed_slots, reference_units, actuator_energy_units),
        result_delay_steps,
        outcomes,
    }
}

/// Build the exact support/loss projection of the frozen #14243 task.
///
/// The default budget admits all three disambiguating probes. Equal-value
/// policies use the #13567 oracle's deterministic action-ID tie-break; this
/// is not a least-cost or Pareto-optimal acquisition claim.
pub fn build_problem(task: DecisionTask) -> SequentialRobustProblem {
    let decision_ids = match task {
        DecisionTask::DecisionInvariant => vec![
            "maintain_safe_configuration".to_owned(),
            "replace_sensor".to_owned(),
        ],
        DecisionTask::DecisionSensitive => vec![
            "select_fast_sensor_path".to_owned(),
            "select_slow_sensor_path".to_owned(),
        ],
    };

    let decision_loss = match task {
        DecisionTask::DecisionInvariant => BTreeMap::from([
            (
                "maintain_safe_configuration".to_owned(),
                BTreeMap::from([(WORLD_A.to_owned(), 0.0), (WORLD_B.to_owned(), 0.0)]),
            ),
            (
                "replace_sensor".to_owned(),
                BTreeMap::from([(WORLD_A.to_owned(), 1.0), (WORLD_B.to_owned(), 1.0)]),
            ),
        ]),
        DecisionTask::DecisionSensitive => BTreeMap::from([
            (
                "select_fast_sensor_path".to_owned(),
                BTreeMap::from([(WORLD_A.to_owned(), 0.0), (WORLD_B.to_owned(), 1.0)]),
            ),
            (
                "select_slow_sensor_path".to_owned(),
                BTreeMap::from([(WORLD_A.to_owned(), 1.0), (WORLD_B.to_owned(), 0.0)]),
            ),
        ]),
    };

    let actions = vec![
        probe(
            "passive_sensor_output_autocovariance",
            0, 0, 0, 0,
            vec![outcome("same_passive_law", &BOTH)],
        ),
        probe(
            "upstream_command_first_order_relaxation_cascade",
            1, 0, 1, 1,
            vec![outcome("same_cascade_response", &BOTH)],
        ),
        probe(
            "traceable_sensor_inlet_step",
            1, 1, 1, 1,
            vec![
                outcome("sensor_inlet_response_world_a", &[WORLD_A]),
                outcome("sensor_inlet_response_world_b", &[WORLD_B]),
            ],
        ),
        probe(
            "traceable_sensor_inlet_frequency_sweep",
            2, 1, 2, 2,
            vec![
                outcome("frequency_response_world_a", &[WORLD_A]),
                outcome("frequency_response_world_b", &[WORLD_B]),
            ],
        ),
        probe(
            "independent_reference_of_latent_process",
            1, 2, 0, 1,
            vec![
                outcome("latent_reference_world_a", &[WORLD_A]),
                outcome("latent_reference_world_b", &[WORLD_B]),
            ],
        ),
        probe(
            "high_entropy_world_independent_nuisance_control",
            1, 0, 0, 1,
            vec![
                outcome("nuisance_outcome_0", &BOTH),
                outcome("nuisance_outcome_1", &BOTH),
            ],
        ),
    ];

    SequentialRobustProblem {
        state_ids: BOTH.iter().map(|id| (*id).to_owned()).collect(),
        decision_ids,
        decision_loss,
        maximum_worst_case_loss: 0.0,
        initial_protocol_state_id: ROOT.to_owned(),
        protocol_states: vec![
            SequentialProtocolState {
                protocol_state_id: ROOT.to_owned(),
                legal_action_ids: actions.iter().map(|action| action.action_id.clone()).collect(),
            },
            SequentialProtocolState {
                protocol_state_id: TERMINAL.to_owned(),
                legal_action_ids: Vec::new(),
            },
        ],
        actions,
        resource_budget: resources(2, 2, 2),
        current_step: 0,
        decision_deadline_step: 2,
        maximum_actions: 1,
    }
}

fn retain_only(problem: &mut SequentialRobustProblem, action_ids: &[&str]) {
    let allowed: BTreeSet<&str> = action_ids.iter().copied().collect();
    problem.actions.retain(|action| allowed.contains(action.action_id.as_str()));
    problem
        .protocol_states
        .iter_mut()
        .find(|state| state.protocol_state_id == ROOT)
        .expect("root protocol state exists")
        .legal_action_ids
        .retain(|id| allowed.contains(id.as_str()));
}

fn assert_dp_matches_exhaustive(problem: &SequentialRobustProblem) -> SequentialPolicyValue {
    let dynamic = evaluate_sequential_robust(problem).expect("dynamic programme accepts fixture");
    let exhaustive =
        enumerate_sequential_robust(problem).expect("exhaustive enumerator accepts fixture");
    assert_eq!(dynamic.current_minimax_loss, exhaustive.current_minimax_loss);
    assert_eq!(dynamic.worst_case_terminal_loss, exhaustive.worst_case_terminal_loss);
    assert_eq!(dynamic.current_risk_gate_passed, exhaustive.current_risk_gate_passed);
    assert_eq!(
        dynamic.policy_terminal_risk_gate_passed,
        exhaustive.policy_terminal_risk_gate_passed
    );
    assert_eq!(dynamic.policy, exhaustive.policy);
    assert_eq!(dynamic.terminal_resource_paths, exhaustive.terminal_resource_paths);
    dynamic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_invariant_task_stops_without_unnecessary_acquisition() {
        let result = assert_dp_matches_exhaustive(&build_problem(DecisionTask::DecisionInvariant));
        assert_eq!(result.current_minimax_loss, 0.0);
        assert_eq!(result.worst_case_terminal_loss, 0.0);
        assert!(result.current_risk_gate_passed);
        assert!(result.policy_terminal_risk_gate_passed);
        assert!(matches!(
            result.policy,
            SequentialPolicyNode::Stop {
                ref decision_id,
                worst_case_terminal_loss: 0.0
            } if decision_id == "maintain_safe_configuration"
        ));
        assert!(result.terminal_resource_paths.iter().all(|path| path.steps.is_empty()));
    }

    #[test]
    fn decision_sensitive_task_uses_a_disambiguating_probe_and_resolves_actions() {
        let result = assert_dp_matches_exhaustive(&build_problem(DecisionTask::DecisionSensitive));
        assert_eq!(result.current_minimax_loss, 1.0);
        assert_eq!(result.worst_case_terminal_loss, 0.0);
        assert!(!result.current_risk_gate_passed);
        assert!(result.policy_terminal_risk_gate_passed);

        let SequentialPolicyNode::Probe { action_id, branches } = &result.policy else {
            panic!("sensitive task must acquire decision-relevant evidence");
        };
        // Three resolvers have the same terminal loss. The existing oracle
        // breaks this exact tie by action ID, not by resource cost.
        assert_eq!(action_id, "independent_reference_of_latent_process");
        assert_eq!(branches.len(), 2);
        for branch in branches {
            let expected_decision = if branch.possible_state_ids == [WORLD_A] {
                "select_fast_sensor_path"
            } else {
                assert_eq!(branch.possible_state_ids, [WORLD_B]);
                "select_slow_sensor_path"
            };
            assert!(matches!(
                branch.continuation.as_ref(),
                SequentialPolicyNode::Stop { decision_id, worst_case_terminal_loss: 0.0 }
                    if decision_id == expected_decision
            ));
        }
        assert_eq!(result.terminal_resource_paths.len(), 2);
        for path in &result.terminal_resource_paths {
            assert_eq!(path.terminal_worst_case_loss, 0.0);
            assert_eq!(path.resource_usage, resources(1, 2, 0));
        }
    }

    #[test]
    fn passive_upstream_and_one_bit_nuisance_controls_do_not_resolve_the_world() {
        for non_identifying in [
            "passive_sensor_output_autocovariance",
            "upstream_command_first_order_relaxation_cascade",
            "high_entropy_world_independent_nuisance_control",
        ] {
            let mut problem = build_problem(DecisionTask::DecisionSensitive);
            retain_only(&mut problem, &[non_identifying]);
            let result = assert_dp_matches_exhaustive(&problem);
            assert_eq!(result.current_minimax_loss, 1.0);
            assert_eq!(result.worst_case_terminal_loss, 1.0);
            assert!(matches!(result.policy, SequentialPolicyNode::Stop { .. }));
        }

        let nuisance = build_problem(DecisionTask::DecisionSensitive);
        let action = nuisance.actions.iter()
            .find(|action| action.action_id == "high_entropy_world_independent_nuisance_control")
            .expect("nuisance negative control exists");
        assert_eq!(action.outcomes.len(), 2);
        assert!(action.outcomes.iter().all(|outcome| outcome.possible_state_ids.len() == 2));
    }

    #[test]
    fn every_declared_disambiguator_resolves_the_sensitive_task() {
        for resolver in [
            "traceable_sensor_inlet_step",
            "traceable_sensor_inlet_frequency_sweep",
            "independent_reference_of_latent_process",
        ] {
            let mut problem = build_problem(DecisionTask::DecisionSensitive);
            retain_only(&mut problem, &[resolver]);
            let result = assert_dp_matches_exhaustive(&problem);
            assert_eq!(result.worst_case_terminal_loss, 0.0);
            let SequentialPolicyNode::Probe { action_id, branches } = result.policy else {
                panic!("declared disambiguator must be selected");
            };
            assert_eq!(action_id, resolver);
            assert_eq!(branches.len(), 2);
        }
    }

    #[test]
    fn componentwise_budget_and_deadline_select_the_inlet_step_without_scalarization() {
        let mut problem = build_problem(DecisionTask::DecisionSensitive);
        problem.resource_budget = resources(1, 1, 1);
        problem.decision_deadline_step = 1;
        let result = assert_dp_matches_exhaustive(&problem);
        let SequentialPolicyNode::Probe { action_id, .. } = result.policy else {
            panic!("inlet step is feasible and decision-sufficient");
        };
        assert_eq!(action_id, "traceable_sensor_inlet_step");
        assert_eq!(
            result.terminal_resource_paths[0].resource_usage,
            resources(1, 1, 1)
        );
    }

    #[test]
    fn reference_only_budget_chooses_the_independent_reference() {
        let mut problem = build_problem(DecisionTask::DecisionSensitive);
        problem.resource_budget = resources(1, 2, 0);
        problem.decision_deadline_step = 1;
        let result = assert_dp_matches_exhaustive(&problem);
        let SequentialPolicyNode::Probe { action_id, .. } = result.policy else {
            panic!("independent reference is feasible and decision-sufficient");
        };
        assert_eq!(action_id, "independent_reference_of_latent_process");
        assert_eq!(
            result.terminal_resource_paths[0].resource_usage,
            resources(1, 2, 0)
        );
    }

    #[test]
    fn an_unaffordable_or_late_disambiguator_cannot_create_a_resolution_claim() {
        let mut problem = build_problem(DecisionTask::DecisionSensitive);
        retain_only(&mut problem, &["traceable_sensor_inlet_frequency_sweep"]);
        problem.resource_budget = resources(1, 1, 1);
        problem.decision_deadline_step = 1;
        let result = assert_dp_matches_exhaustive(&problem);
        assert_eq!(result.worst_case_terminal_loss, 1.0);
        assert!(matches!(result.policy, SequentialPolicyNode::Stop { .. }));
        assert!(!result.policy_terminal_risk_gate_passed);
    }
}
