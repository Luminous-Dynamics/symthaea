// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded counterfactual decision-quality assessment.
//!
//! This module compares one selected vehicle command with a declared finite
//! action set evaluated by a named independent evaluator. It does not run
//! simulation, prove action-set completeness or establish a global optimum.
//! Feasibility and score values are caller-supplied, model-relative assertions.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::types::VehicleCommand;

pub const COUNTERFACTUAL_DECISION_SCHEMA: &str =
    "symthaea.vehicle.counterfactual-decision.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterfactualMetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterfactualFeasibility {
    Feasible,
    Infeasible,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterfactualDecisionOutcome {
    NoBetterFeasibleActionFound,
    BetterFeasibleActionFound,
    SelectedActionInfeasible,
    NoFeasibleAction,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterfactualDecisionReason {
    NoBetterActionInDeclaredSet,
    BetterFeasibleActionAvailable,
    SelectedActionViolatesDeclaredFeasibility,
    NoFeasibleActionInDeclaredSet,
    ActionSetIncomplete,
    CandidateFeasibilityUnknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualActionEvaluation {
    pub action_id: String,
    pub command: VehicleCommand,
    /// BLAKE3 over the exact serialized command payload.
    pub action_digest: String,
    /// Fingerprint of the shared scenario/state/environment/dynamics/protocol context.
    pub execution_context_digest: String,
    pub feasibility: CounterfactualFeasibility,
    /// One named scalar objective only; no implicit weighted composite.
    pub objective_value: f64,
    pub evaluator_id: String,
    pub execution_receipt_id: String,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualDecisionInput {
    pub decision_id: String,
    pub scenario_id: String,
    /// Exact scenario candidate or route artifact identity.
    pub scenario_artifact_digest: String,
    /// Shared decision-time state from which every action candidate is evaluated.
    pub initial_state_digest: String,
    /// Frozen road, weather, actor state, sensor and map context.
    pub environment_profile_digest: String,
    /// Named dynamics/model parameters used by every action evaluation.
    pub dynamics_profile_digest: String,
    /// Execution timestep, horizon, termination and evaluator protocol profile.
    pub execution_protocol_digest: String,
    pub failure_family_id: String,
    /// Frozen controller artifact identity. A digest-format check is not authentication.
    pub controller_artifact_digest: String,
    /// Independent evaluator build/profile identity. Must differ from the controller digest.
    pub evaluator_artifact_digest: String,
    pub evaluator_id: String,
    /// External record ID for the independence argument; not authenticated here.
    pub evaluator_independence_evidence_id: String,
    pub criterion_id: String,
    pub direction: CounterfactualMetricDirection,
    pub tie_tolerance: f64,
    pub selected_action_id: String,
    /// Caller assertion that the finite candidate set is complete for this decision.
    pub candidate_set_complete: bool,
    pub actions: Vec<CounterfactualActionEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterfactualDecisionError {
    EmptyActionSet,
    EmptyIdentity,
    InvalidArtifactDigest,
    ControllerEvaluatorIdentityAliased,
    InvalidTieTolerance,
    DuplicateActionId,
    DuplicateExecutionReceipt,
    InvalidActionCommand,
    InvalidActionDigest,
    InvalidObjectiveValue,
    InvalidExecutionReceipt,
    MixedEvaluatorIdentity,
    SelectedActionMissing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualDecisionFinding {
    pub selected_action_id: String,
    pub selected_action_feasibility: CounterfactualFeasibility,
    pub selected_objective_value: Option<f64>,
    pub best_feasible_action_id: Option<String>,
    pub best_feasible_objective_value: Option<f64>,
    /// Positive values mean the selected action is worse on the named objective.
    pub objective_regret: Option<f64>,
    pub feasible_action_count: usize,
    pub total_action_count: usize,
    pub outcome: CounterfactualDecisionOutcome,
    pub reason: CounterfactualDecisionReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualDecisionReport {
    pub schema_version: String,
    pub input: CounterfactualDecisionInput,
    pub finding: CounterfactualDecisionFinding,
    /// BLAKE3 over exact input and finding with this field empty; not a signature.
    pub report_digest: String,
}

impl CounterfactualDecisionReport {
    /// Re-evaluate the finite action set and verify the retained result and digest.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != COUNTERFACTUAL_DECISION_SCHEMA
            || self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }
        let Ok(expected_finding) = derive_finding(&self.input) else {
            return false;
        };
        if expected_finding != self.finding {
            return false;
        }
        let mut report = self.clone();
        let expected = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

/// Assess whether a better feasible action exists within the caller-declared
/// finite candidate set, under one named metric and model.
///
/// All candidate actions must carry receipts from the same evaluator identity.
/// Feasibility and score semantics are those of that evaluator's declared model.
/// A complete action-set flag is a caller assertion, not an exhaustive-search proof.
pub fn evaluate_counterfactual_decision(
    input: CounterfactualDecisionInput,
) -> Result<CounterfactualDecisionReport, CounterfactualDecisionError> {
    let finding = derive_finding(&input)?;
    let mut report = CounterfactualDecisionReport {
        schema_version: COUNTERFACTUAL_DECISION_SCHEMA.into(),
        input,
        finding,
        report_digest: String::new(),
    };
    let bytes = serde_json::to_vec(&report)
        .map_err(|_| CounterfactualDecisionError::InvalidObjectiveValue)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

fn derive_finding(
    input: &CounterfactualDecisionInput,
) -> Result<CounterfactualDecisionFinding, CounterfactualDecisionError> {
    if input.actions.is_empty() {
        return Err(CounterfactualDecisionError::EmptyActionSet);
    }
    if [
        input.decision_id.as_str(),
        input.scenario_id.as_str(),
        input.failure_family_id.as_str(),
        input.evaluator_id.as_str(),
        input.evaluator_independence_evidence_id.as_str(),
        input.criterion_id.as_str(),
        input.selected_action_id.as_str(),
    ]
    .iter()
    .any(|value| value.trim().is_empty())
    {
        return Err(CounterfactualDecisionError::EmptyIdentity);
    }
    if [
        &input.scenario_artifact_digest,
        &input.initial_state_digest,
        &input.environment_profile_digest,
        &input.dynamics_profile_digest,
        &input.execution_protocol_digest,
        &input.controller_artifact_digest,
        &input.evaluator_artifact_digest,
    ]
    .iter()
    .any(|digest| !digest_is_valid(digest))
    {
        return Err(CounterfactualDecisionError::InvalidArtifactDigest);
    }
    if input
        .controller_artifact_digest
        .eq_ignore_ascii_case(&input.evaluator_artifact_digest)
    {
        return Err(CounterfactualDecisionError::ControllerEvaluatorIdentityAliased);
    }
    if !input.tie_tolerance.is_finite() || input.tie_tolerance < 0.0 {
        return Err(CounterfactualDecisionError::InvalidTieTolerance);
    }

    let mut action_ids = BTreeSet::new();
    let mut receipt_ids = BTreeSet::new();
    for action in &input.actions {
        if action.action_id.trim().is_empty() || !action_ids.insert(action.action_id.as_str()) {
            return Err(CounterfactualDecisionError::DuplicateActionId);
        }
        if !receipt_ids.insert(action.execution_receipt_id.as_str()) {
            return Err(CounterfactualDecisionError::DuplicateExecutionReceipt);
        }
        if action.evaluator_id != input.evaluator_id {
            return Err(CounterfactualDecisionError::MixedEvaluatorIdentity);
        }
        if !action.command.steering.is_finite()
            || !(-1.0..=1.0).contains(&action.command.steering)
            || !action.command.throttle.is_finite()
            || !(0.0..=1.0).contains(&action.command.throttle)
            || !action.command.brake.is_finite()
            || !(0.0..=1.0).contains(&action.command.brake)
        {
            return Err(CounterfactualDecisionError::InvalidActionCommand);
        }
        if action.action_digest != command_digest(&action.command) {
            return Err(CounterfactualDecisionError::InvalidActionDigest);
        }
        if action.execution_context_digest != execution_context_digest(input) {
            return Err(CounterfactualDecisionError::InvalidExecutionReceipt);
        }
        if !action.objective_value.is_finite() {
            return Err(CounterfactualDecisionError::InvalidObjectiveValue);
        }
        if action.execution_receipt_id.trim().is_empty()
            || !digest_is_valid(&action.execution_digest)
        {
            return Err(CounterfactualDecisionError::InvalidExecutionReceipt);
        }
    }

    let selected = input
        .actions
        .iter()
        .find(|action| action.action_id == input.selected_action_id)
        .ok_or(CounterfactualDecisionError::SelectedActionMissing)?;

    let total_action_count = input.actions.len();
    let feasible = input
        .actions
        .iter()
        .filter(|action| action.feasibility == CounterfactualFeasibility::Feasible)
        .collect::<Vec<_>>();
    let feasible_action_count = feasible.len();

    let (outcome, reason, best_action, regret) = if !input.candidate_set_complete {
        (
            CounterfactualDecisionOutcome::Indeterminate,
            CounterfactualDecisionReason::ActionSetIncomplete,
            None,
            None,
        )
    } else if input
        .actions
        .iter()
        .any(|action| action.feasibility == CounterfactualFeasibility::Indeterminate)
    {
        (
            CounterfactualDecisionOutcome::Indeterminate,
            CounterfactualDecisionReason::CandidateFeasibilityUnknown,
            None,
            None,
        )
    } else if feasible.is_empty() {
        (
            CounterfactualDecisionOutcome::NoFeasibleAction,
            CounterfactualDecisionReason::NoFeasibleActionInDeclaredSet,
            None,
            None,
        )
    } else {
        let best_action = feasible
            .iter()
            .copied()
            .reduce(|best, candidate| {
                let candidate_is_better = match input.direction {
                    CounterfactualMetricDirection::LowerIsBetter => {
                        candidate.objective_value < best.objective_value
                    }
                    CounterfactualMetricDirection::HigherIsBetter => {
                        candidate.objective_value > best.objective_value
                    }
                } || (candidate.objective_value == best.objective_value
                    && candidate.action_id < best.action_id);
                if candidate_is_better {
                    candidate
                } else {
                    best
                }
            })
            .expect("non-empty feasible set checked above");
        if selected.feasibility == CounterfactualFeasibility::Infeasible {
            (
                CounterfactualDecisionOutcome::SelectedActionInfeasible,
                CounterfactualDecisionReason::SelectedActionViolatesDeclaredFeasibility,
                Some(best_action),
                None,
            )
        } else {
            let signed_regret = match input.direction {
                CounterfactualMetricDirection::LowerIsBetter => {
                    selected.objective_value - best_action.objective_value
                }
                CounterfactualMetricDirection::HigherIsBetter => {
                    best_action.objective_value - selected.objective_value
                }
            };
            if !signed_regret.is_finite() {
                return Err(CounterfactualDecisionError::InvalidObjectiveValue);
            }
            let regret = signed_regret.max(0.0);
            if regret > input.tie_tolerance {
                (
                    CounterfactualDecisionOutcome::BetterFeasibleActionFound,
                    CounterfactualDecisionReason::BetterFeasibleActionAvailable,
                    Some(best_action),
                    Some(regret),
                )
            } else {
                (
                    CounterfactualDecisionOutcome::NoBetterFeasibleActionFound,
                    CounterfactualDecisionReason::NoBetterActionInDeclaredSet,
                    Some(best_action),
                    Some(0.0),
                )
            }
        }
    };

    Ok(CounterfactualDecisionFinding {
        selected_action_id: input.selected_action_id.clone(),
        selected_action_feasibility: selected.feasibility,
        selected_objective_value: Some(selected.objective_value),
        best_feasible_action_id: best_action.map(|action| action.action_id.clone()),
        best_feasible_objective_value: best_action.map(|action| action.objective_value),
        objective_regret: regret,
        feasible_action_count,
        total_action_count,
        outcome,
        reason,
    })
}

fn execution_context_digest(input: &CounterfactualDecisionInput) -> String {
    let context = (
        &input.decision_id,
        &input.scenario_id,
        &input.scenario_artifact_digest,
        &input.initial_state_digest,
        &input.environment_profile_digest,
        &input.dynamics_profile_digest,
        &input.execution_protocol_digest,
        &input.evaluator_artifact_digest,
        &input.evaluator_id,
        &input.criterion_id,
        input.direction,
        input.tie_tolerance.to_bits(),
    );
    let bytes = serde_json::to_vec(&context).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn command_digest(command: &VehicleCommand) -> String {
    let bytes = serde_json::to_vec(command).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn digest_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(
        id: &str,
        steering: f32,
        feasibility: CounterfactualFeasibility,
        objective: f64,
    ) -> CounterfactualActionEvaluation {
        let command = VehicleCommand {
            steering,
            throttle: 0.2,
            brake: 0.0,
        };
        let mut hasher = blake3::Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(&objective.to_bits().to_le_bytes());
        CounterfactualActionEvaluation {
            action_id: id.into(),
            action_digest: command_digest(&command),
            execution_context_digest: String::new(),
            command,
            feasibility,
            objective_value: objective,
            evaluator_id: "independent-bicycle-model-v1".into(),
            execution_receipt_id: format!("receipt-{id}"),
            execution_digest: hasher.finalize().to_hex().to_string(),
        }
    }

    fn input() -> CounterfactualDecisionInput {
        let mut input = CounterfactualDecisionInput {
            decision_id: "decision-001".into(),
            scenario_id: "scenario-001".into(),
            scenario_artifact_digest: "1".repeat(64),
            initial_state_digest: "2".repeat(64),
            environment_profile_digest: "3".repeat(64),
            dynamics_profile_digest: "4".repeat(64),
            execution_protocol_digest: "5".repeat(64),
            failure_family_id: "pedestrian-crossing".into(),
            controller_artifact_digest: "a".repeat(64),
            evaluator_artifact_digest: "b".repeat(64),
            evaluator_id: "independent-bicycle-model-v1".into(),
            evaluator_independence_evidence_id: "independence-review-record-001".into(),
            criterion_id: "maximum-predicted-lateral-acceleration-v1".into(),
            direction: CounterfactualMetricDirection::LowerIsBetter,
            tie_tolerance: 1e-6,
            selected_action_id: "selected".into(),
            candidate_set_complete: true,
            actions: vec![
                action("selected", 0.0, CounterfactualFeasibility::Feasible, 2.0),
                action("alternative", 0.2, CounterfactualFeasibility::Feasible, 1.0),
                action("infeasible", 0.8, CounterfactualFeasibility::Infeasible, 0.1),
            ],
        };
        let context_digest = execution_context_digest(&input);
        for action in &mut input.actions {
            action.execution_context_digest = context_digest.clone();
        }
        input
    }

    #[test]
    fn identifies_better_feasible_counterfactual_and_positive_regret() {
        let report = evaluate_counterfactual_decision(input()).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::BetterFeasibleActionFound
        );
        assert_eq!(
            report.finding.best_feasible_action_id.as_deref(),
            Some("alternative")
        );
        assert_eq!(report.finding.objective_regret, Some(1.0));
        assert!(report.identity_is_valid());
    }

    #[test]
    fn no_better_action_reports_zero_regret() {
        let mut sample = input();
        sample.actions[1].objective_value = 2.0;
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::NoBetterFeasibleActionFound
        );
        assert_eq!(report.finding.objective_regret, Some(0.0));
    }

    #[test]
    fn higher_is_better_uses_the_opposite_objective_direction() {
        let mut sample = input();
        sample.direction = CounterfactualMetricDirection::HigherIsBetter;
        sample.criterion_id = "predicted-clearance-margin-v1".into();
        sample.actions[0].objective_value = 0.2;
        sample.actions[1].objective_value = 0.8;
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(report.finding.objective_regret, Some(0.6));
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::BetterFeasibleActionFound
        );
    }

    #[test]
    fn tied_best_actions_use_stable_action_id_tiebreaker() {
        let mut sample = input();
        sample.actions[1].objective_value = 1.0;
        sample.actions.push(action(
            "aaa-alternative",
            -0.2,
            CounterfactualFeasibility::Feasible,
            1.0,
        ));
        let context_digest = execution_context_digest(&sample);
        for candidate in &mut sample.actions {
            candidate.execution_context_digest = context_digest.clone();
        }
        let first = evaluate_counterfactual_decision(sample.clone()).unwrap();
        sample.actions.reverse();
        let second = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            first.finding.best_feasible_action_id,
            Some("aaa-alternative".into())
        );
        assert_eq!(
            first.finding.best_feasible_action_id,
            second.finding.best_feasible_action_id
        );
    }

    #[test]
    fn incomplete_action_set_is_indeterminate_not_optimal() {
        let mut sample = input();
        sample.candidate_set_complete = false;
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::Indeterminate
        );
        assert_eq!(
            report.finding.reason,
            CounterfactualDecisionReason::ActionSetIncomplete
        );
        assert_eq!(report.finding.objective_regret, None);
    }

    #[test]
    fn unknown_feasibility_prevents_best_action_claim() {
        let mut sample = input();
        sample.actions.push(action(
            "unknown",
            -0.1,
            CounterfactualFeasibility::Indeterminate,
            0.0,
        ));
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::Indeterminate
        );
        assert_eq!(
            report.finding.reason,
            CounterfactualDecisionReason::CandidateFeasibilityUnknown
        );
    }

    #[test]
    fn selected_infeasible_action_is_reported_separately() {
        let mut sample = input();
        sample.actions[0].feasibility = CounterfactualFeasibility::Infeasible;
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::SelectedActionInfeasible
        );
        assert_eq!(
            report.finding.best_feasible_action_id.as_deref(),
            Some("alternative")
        );
        assert_eq!(report.finding.objective_regret, None);
    }

    #[test]
    fn no_feasible_action_is_not_reported_as_better_alternative() {
        let mut sample = input();
        for action in &mut sample.actions {
            action.feasibility = CounterfactualFeasibility::Infeasible;
        }
        let report = evaluate_counterfactual_decision(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            CounterfactualDecisionOutcome::NoFeasibleAction
        );
    }

    #[test]
    fn rejects_missing_or_malformed_shared_execution_context_digests() {
        let mut sample = input();
        sample.initial_state_digest.clear();
        assert_eq!(
            evaluate_counterfactual_decision(sample).unwrap_err(),
            CounterfactualDecisionError::InvalidArtifactDigest
        );

        let mut sample = input();
        sample.dynamics_profile_digest = "not-a-digest".into();
        assert_eq!(
            evaluate_counterfactual_decision(sample).unwrap_err(),
            CounterfactualDecisionError::InvalidArtifactDigest
        );
    }

    #[test]
    fn rejects_duplicate_ids_mixed_evaluators_and_identity_aliasing() {
        let mut duplicate = input();
        duplicate.actions.push(duplicate.actions[0].clone());
        assert_eq!(
            evaluate_counterfactual_decision(duplicate).unwrap_err(),
            CounterfactualDecisionError::DuplicateActionId
        );

        let mut mixed = input();
        mixed.actions[1].evaluator_id = "other-evaluator".into();
        assert_eq!(
            evaluate_counterfactual_decision(mixed).unwrap_err(),
            CounterfactualDecisionError::MixedEvaluatorIdentity
        );

        let mut aliased = input();
        aliased.evaluator_artifact_digest = aliased.controller_artifact_digest.clone();
        assert_eq!(
            evaluate_counterfactual_decision(aliased).unwrap_err(),
            CounterfactualDecisionError::ControllerEvaluatorIdentityAliased
        );
    }

    #[test]
    fn rejects_action_receipt_from_a_different_shared_execution_context() {
        let mut sample = input();
        sample.actions[0].execution_context_digest = "f".repeat(64);
        assert_eq!(
            evaluate_counterfactual_decision(sample).unwrap_err(),
            CounterfactualDecisionError::InvalidExecutionReceipt
        );
    }

    #[test]
    fn rejects_action_payload_mutation_even_when_action_id_is_unchanged() {
        let mut sample = input();
        sample.actions[0].command.steering = 0.5;
        assert_eq!(
            evaluate_counterfactual_decision(sample).unwrap_err(),
            CounterfactualDecisionError::InvalidActionDigest
        );
    }

    #[test]
    fn report_validator_recomputes_findings_after_digest_recalculation_tampering() {
        let mut report = evaluate_counterfactual_decision(input()).unwrap();
        report.finding.objective_regret = Some(0.0);
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }
}
