// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact two-stage oracle distinguishing static model ambiguity from a
//! rectangular, history-dependent adversary.
//!
//! One latent observation-model vertex is selected for an episode. The
//! calibration observation identifies that vertex under the static fixture,
//! but the policy never receives the vertex ID. It chooses the second-stage
//! action only from the observed calibration category.
//!
//! Static evaluation keeps one vertex fixed for both stages. Rectangular
//! evaluation lets the adversary use one vertex for calibration and another
//! at each reached second-stage history. This can create hybrid trajectories
//! that no single persistent model can generate.
//!
//! This is an exact finite benchmark oracle, not a general robust POMDP/RMDP
//! solver. Its fixed dimensions make exhaustive enumeration auditable.

use serde::{Deserialize, Serialize};

const MODEL_VERTEX_COUNT: usize = 2;
const BINARY_OUTCOME_COUNT: usize = 2;
const ACTION_COUNT: usize = 2;

/// Latent observation model, fixed for the whole episode in static evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationModelVertex {
    ModelA,
    ModelB,
}

impl ObservationModelVertex {
    fn index(self) -> usize {
        match self {
            Self::ModelA => 0,
            Self::ModelB => 1,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::ModelA,
            1 => Self::ModelB,
            _ => unreachable!("two model vertices are enumerated"),
        }
    }
}

/// A deployable policy tree expressed only in public observation categories.
///
/// The second-stage action is selected from the calibration category. The
/// terminal decision is selected from the calibration and response categories.
/// No hidden model-vertex field is present in the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwoStageDiagnosticPolicy {
    pub second_action_by_calibration: [ObservationModelVertex; MODEL_VERTEX_COUNT],
    pub terminal_decision_by_history: [[bool; BINARY_OUTCOME_COUNT]; MODEL_VERTEX_COUNT],
}

/// Exact comparison between static, rectangular and clairvoyant risk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticModelRobustFixtureResult {
    pub policies_evaluated: usize,
    pub static_robust_risk: f64,
    pub rectangular_robust_risk: f64,
    pub clairvoyant_lower_bound: f64,
    pub static_optimal_policy: TwoStageDiagnosticPolicy,
    pub rectangular_optimal_policy: TwoStageDiagnosticPolicy,
    /// Risk of the static-optimal policy when the model vertex is allowed to
    /// switch after calibration.
    pub calibrated_policy_rectangular_risk: f64,
    /// Risk of the static-optimal policy under each persistent model vertex.
    pub static_optimal_risk_by_vertex: [f64; MODEL_VERTEX_COUNT],
}

/// Evaluate one policy under a single model vertex that remains fixed.
///
/// The calibration category is CAL-A for MODEL-A and CAL-B for MODEL-B.
/// It is independent of the hidden TRUE/FALSE world. At stage two, the chosen
/// action is truthful only when its identity matches the fixed model vertex.
pub fn static_model_policy_risk(
    policy: &TwoStageDiagnosticPolicy,
    model_vertex: ObservationModelVertex,
) -> f64 {
    let calibration_category = model_vertex.index();
    let selected_action = policy.second_action_by_calibration[calibration_category];
    let mut risk = 0.0;

    for hidden_truth in [false, true] {
        let response_is_positive = if selected_action == model_vertex {
            hidden_truth
        } else {
            !hidden_truth
        };
        let response_category = if response_is_positive { 1 } else { 0 };
        let terminal_decision =
            policy.terminal_decision_by_history[calibration_category][response_category];
        if terminal_decision != hidden_truth {
            risk += 0.5;
        }
    }

    risk
}

/// Evaluate a policy under a rectangular adversary.
///
/// At the root, the adversary chooses the calibration-model vertex. After the
/// public calibration category is observed, it may independently choose a
/// second-stage vertex for that history. The second-stage vertex is shared
/// across the two hidden-world hypotheses at that history; the adversary does
/// not observe TRUE/FALSE.
pub fn rectangular_model_policy_risk(policy: &TwoStageDiagnosticPolicy) -> f64 {
    let mut worst_root_risk: f64 = 0.0;

    for calibration_model_index in 0..MODEL_VERTEX_COUNT {
        let calibration_category = calibration_model_index;
        let selected_action = policy.second_action_by_calibration[calibration_category];
        let mut worst_continuation_risk: f64 = 0.0;

        for second_stage_model_index in 0..MODEL_VERTEX_COUNT {
            let second_stage_vertex =
                ObservationModelVertex::from_index(second_stage_model_index);
            let mut branch_risk: f64 = 0.0;

            for hidden_truth in [false, true] {
                let response_is_positive = if selected_action == second_stage_vertex {
                    hidden_truth
                } else {
                    !hidden_truth
                };
                let response_category = if response_is_positive { 1 } else { 0 };
                let terminal_decision =
                    policy.terminal_decision_by_history[calibration_category][response_category];
                if terminal_decision != hidden_truth {
                    branch_risk += 0.5;
                }
            }
            worst_continuation_risk = worst_continuation_risk.max(branch_risk);
        }

        worst_root_risk = worst_root_risk.max(worst_continuation_risk);
    }

    worst_root_risk
}

/// Exhaustively enumerate every deterministic non-anticipative policy.
///
/// There are two choices for the action after CAL-A, two choices after CAL-B,
/// and 2^4 terminal-decision maps: 2*2*16 = 64 policies. The static objective
/// is min_policy max_vertex static_risk(policy, vertex). The rectangular
/// objective is min_policy rectangular_risk(policy).
pub fn solve_two_stage_static_model_fixture() -> StaticModelRobustFixtureResult {
    let mut policies_evaluated = 0;
    let mut best_static_risk = f64::INFINITY;
    let mut best_rectangular_risk = f64::INFINITY;
    let mut static_optimal_policy = None;
    let mut rectangular_optimal_policy = None;

    // Prefer the calibration-aligned action mapping as the canonical tie
    // break, then enumerate the remaining action mappings deterministically.
    for [action_after_cal_a, action_after_cal_b] in
        [[0, 1], [0, 0], [1, 0], [1, 1]]
    {
        for decision_mask in 0..(1_u8 << (MODEL_VERTEX_COUNT * BINARY_OUTCOME_COUNT)) {
            let policy = TwoStageDiagnosticPolicy {
                second_action_by_calibration: [
                    ObservationModelVertex::from_index(action_after_cal_a),
                    ObservationModelVertex::from_index(action_after_cal_b),
                ],
                    terminal_decision_by_history: [
                        [
                            (decision_mask & 0b0001) != 0,
                            (decision_mask & 0b0010) != 0,
                        ],
                        [
                            (decision_mask & 0b0100) != 0,
                            (decision_mask & 0b1000) != 0,
                        ],
                    ],
                };
                policies_evaluated += 1;

                let risk_a =
                    static_model_policy_risk(&policy, ObservationModelVertex::ModelA);
                let risk_b =
                    static_model_policy_risk(&policy, ObservationModelVertex::ModelB);
                let static_risk = risk_a.max(risk_b);
                if static_risk < best_static_risk {
                    best_static_risk = static_risk;
                    static_optimal_policy = Some(policy);
                }

            let rectangular_risk = rectangular_model_policy_risk(&policy);
            if rectangular_risk < best_rectangular_risk {
                best_rectangular_risk = rectangular_risk;
                rectangular_optimal_policy = Some(policy);
            }
        }
    }

    let static_optimal_policy =
        static_optimal_policy.expect("the finite policy set is non-empty");
    let rectangular_optimal_policy =
        rectangular_optimal_policy.expect("the finite policy set is non-empty");
    let static_optimal_risk_by_vertex = [
        static_model_policy_risk(&static_optimal_policy, ObservationModelVertex::ModelA),
        static_model_policy_risk(&static_optimal_policy, ObservationModelVertex::ModelB),
    ];

    StaticModelRobustFixtureResult {
        policies_evaluated,
        static_robust_risk: best_static_risk,
        rectangular_robust_risk: best_rectangular_risk,
        // Once the vertex is revealed, choose the matching action and decode
        // its truthful response. Both observation-model vertices are noiseless.
        clairvoyant_lower_bound: 0.0,
        calibrated_policy_rectangular_risk: rectangular_model_policy_risk(&static_optimal_policy),
        static_optimal_risk_by_vertex,
        static_optimal_policy,
        rectangular_optimal_policy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_oracle_enumerates_all_sixty_four_non_anticipative_policies() {
        let result = solve_two_stage_static_model_fixture();
        assert_eq!(result.policies_evaluated, 64);
    }

    #[test]
    fn calibration_policy_has_zero_risk_under_each_fixed_model_vertex() {
        let result = solve_two_stage_static_model_fixture();
        assert_eq!(result.static_optimal_risk_by_vertex, [0.0, 0.0]);
        assert_eq!(result.static_robust_risk, 0.0);
        assert_eq!(
            result.static_optimal_policy.second_action_by_calibration,
            [ObservationModelVertex::ModelA, ObservationModelVertex::ModelB]
        );
        assert_eq!(
            result.static_optimal_policy.terminal_decision_by_history,
            [[false, true], [false, true]]
        );
    }

    #[test]
    fn rectangular_relaxation_is_strictly_more_conservative_than_static_model() {
        let result = solve_two_stage_static_model_fixture();
        assert_eq!(result.rectangular_robust_risk, 0.5);
        assert_eq!(result.static_robust_risk, 0.0);
        assert!(
            result.rectangular_robust_risk > result.static_robust_risk,
            "rectangularity should create the impossible hybrid-model penalty"
        );
        assert_eq!(result.calibrated_policy_rectangular_risk, 1.0);
    }

    #[test]
    fn clairvoyant_model_vertex_is_only_a_zero_risk_lower_bound() {
        let result = solve_two_stage_static_model_fixture();
        assert_eq!(result.clairvoyant_lower_bound, 0.0);
        assert!(result.clairvoyant_lower_bound <= result.static_robust_risk);
    }

    #[test]
    fn rectangular_risk_dominates_static_risk_for_every_enumerated_policy() {
        for action_after_cal_a in 0..ACTION_COUNT {
            for action_after_cal_b in 0..ACTION_COUNT {
                for decision_mask in 0..(1_u8 << (MODEL_VERTEX_COUNT * BINARY_OUTCOME_COUNT)) {
                    let policy = TwoStageDiagnosticPolicy {
                        second_action_by_calibration: [
                            ObservationModelVertex::from_index(action_after_cal_a),
                            ObservationModelVertex::from_index(action_after_cal_b),
                        ],
                        terminal_decision_by_history: [
                            [
                                (decision_mask & 0b0001) != 0,
                                (decision_mask & 0b0010) != 0,
                            ],
                            [
                                (decision_mask & 0b0100) != 0,
                                (decision_mask & 0b1000) != 0,
                            ],
                        ],
                    };
                    let static_risk = static_model_policy_risk(
                        &policy,
                        ObservationModelVertex::ModelA,
                    )
                    .max(static_model_policy_risk(
                        &policy,
                        ObservationModelVertex::ModelB,
                    ));
                    let rectangular_risk = rectangular_model_policy_risk(&policy);
                    assert!(rectangular_risk >= static_risk);
                }
            }
        }
    }

    #[test]
    fn policy_tree_is_keyed_by_observed_calibration_not_hidden_vertex() {
        let result = solve_two_stage_static_model_fixture();
        let policy = result.static_optimal_policy;
        assert_eq!(
            policy.second_action_by_calibration[0],
            ObservationModelVertex::ModelA
        );
        assert_eq!(
            policy.second_action_by_calibration[1],
            ObservationModelVertex::ModelB
        );
    }
}
