// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-model stability analysis for bounded counterfactual action sets.
//!
//! The same finite action set is evaluated under multiple declared dynamics /
//! evaluator profiles. This detects best-action reversals and selected-action
//! infeasibility that a single-model counterfactual audit cannot reveal.
//!
//! This is a model-sensitivity diagnostic, not a model ensemble optimizer,
//! physical validation, common-cause independence proof or global safety case.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::types::VehicleCommand;

pub const VEHICLE_MODEL_STABILITY_SCHEMA: &str =
    "symthaea.vehicle.model-stability.v0.1";
pub const MIN_MODEL_STABILITY_PROFILES: usize = 2;
pub const MAX_MODEL_STABILITY_PROFILES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStabilityDirection {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelActionFeasibility {
    Feasible,
    Infeasible,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStabilityOutcome {
    SelectedActionBestInEveryModel,
    SameAlternativeBestInEveryModel,
    BestActionChangesAcrossModels,
    SelectedActionInfeasibleInAtLeastOneModel,
    NoFeasibleActionInAtLeastOneModel,
    Indeterminate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelActionEvaluation {
    pub model_id: String,
    pub model_artifact_digest: String,
    pub model_profile_digest: String,
    pub evaluator_id: String,
    pub evaluator_artifact_digest: String,
    /// Recomputed fingerprint of the shared experiment context plus model identity.
    pub execution_context_digest: String,
    pub execution_receipt_id: String,
    pub execution_digest: String,
    pub action_id: String,
    pub command: VehicleCommand,
    pub command_digest: String,
    pub feasibility: ModelActionFeasibility,
    /// One declared scalar objective, with identical units and direction across profiles.
    pub objective_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleModelStabilityInput {
    pub decision_id: String,
    pub scenario_id: String,
    pub scenario_artifact_digest: String,
    pub initial_state_digest: String,
    pub environment_profile_digest: String,
    pub horizon_profile_digest: String,
    pub metric_id: String,
    pub metric_unit_id: String,
    pub direction: ModelStabilityDirection,
    pub tie_tolerance: f64,
    pub selected_action_id: String,
    /// Caller assertion; the evaluator cannot prove candidate generation was exhaustive.
    pub action_set_complete: bool,
    /// Caller assertion; this does not prove the model profiles are independent.
    pub model_set_complete: bool,
    pub evaluations: Vec<ModelActionEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStabilityError {
    EmptyEvaluationSet,
    TooFewModels,
    TooManyModels,
    EmptyIdentity,
    InvalidDigest,
    InvalidTieTolerance,
    InvalidCommand,
    InvalidCommandDigest,
    InvalidObjective,
    InvalidReceipt,
    DuplicateModelProfile,
    DuplicateModelAction,
    DuplicateExecutionReceipt,
    MixedModelIdentity,
    MixedEvaluatorIdentity,
    MismatchedActionSets,
    MismatchedActionCommand,
    MismatchedExecutionContext,
    MissingSelectedAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerModelStabilityFinding {
    pub model_id: String,
    pub model_artifact_digest: String,
    pub model_profile_digest: String,
    pub selected_action_feasibility: ModelActionFeasibility,
    pub selected_objective_value: f64,
    pub best_feasible_action_id: Option<String>,
    pub best_feasible_objective_value: Option<f64>,
    pub selected_action_regret: Option<f64>,
    pub feasible_action_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VehicleModelStabilityFinding {
    pub selected_action_id: String,
    pub model_count: usize,
    pub action_count: usize,
    pub selected_action_feasible_model_count: usize,
    pub better_action_model_count: usize,
    pub best_action_agreement_rate: f64,
    pub worst_case_selected_action_regret: Option<f64>,
    pub best_action_ids: Vec<String>,
    pub outcome: ModelStabilityOutcome,
    pub per_model: Vec<PerModelStabilityFinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleModelStabilityReport {
    pub schema_version: String,
    pub input: VehicleModelStabilityInput,
    pub finding: VehicleModelStabilityFinding,
    pub report_digest: String,
}

impl VehicleModelStabilityReport {
    /// Recompute model groupings, shared context, action set and all sensitivity metrics.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != VEHICLE_MODEL_STABILITY_SCHEMA
            || self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }
        let Ok(expected) = derive_finding(&self.input) else {
            return false;
        };
        if expected != self.finding {
            return false;
        }
        let mut report = self.clone();
        let expected_digest = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected_digest)
            .unwrap_or(false)
    }
}

/// Compare one frozen finite action set across two or more named model profiles.
///
/// All model profiles must evaluate the same action IDs and exact command
/// payloads using the same declared event/metric and forecast horizon. A
/// missing action, unknown feasibility, incomplete model set or incomplete
/// action set prevents an invariance claim. Model IDs/digests are structural
/// identities only; they do not prove independent physics or common-cause
/// separation.
pub fn evaluate_vehicle_model_stability(
    mut input: VehicleModelStabilityInput,
) -> Result<VehicleModelStabilityReport, ModelStabilityError> {
    input.evaluations.sort_by(|left, right| {
        left.model_id
            .cmp(&right.model_id)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });
    let finding = derive_finding(&input)?;
    let mut report = VehicleModelStabilityReport {
        schema_version: VEHICLE_MODEL_STABILITY_SCHEMA.into(),
        input,
        finding,
        report_digest: String::new(),
    };
    let bytes = serde_json::to_vec(&report).map_err(|_| ModelStabilityError::InvalidObjective)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

fn derive_finding(
    input: &VehicleModelStabilityInput,
) -> Result<VehicleModelStabilityFinding, ModelStabilityError> {
    if input.evaluations.is_empty() {
        return Err(ModelStabilityError::EmptyEvaluationSet);
    }
    if [
        input.decision_id.as_str(),
        input.scenario_id.as_str(),
        input.metric_id.as_str(),
        input.metric_unit_id.as_str(),
        input.selected_action_id.as_str(),
    ]
    .iter()
    .any(|identity| identity.trim().is_empty())
    {
        return Err(ModelStabilityError::EmptyIdentity);
    }
    if ![
        input.scenario_artifact_digest.as_str(),
        input.initial_state_digest.as_str(),
        input.environment_profile_digest.as_str(),
        input.horizon_profile_digest.as_str(),
    ]
    .iter()
    .all(|digest| digest_is_valid(digest))
    {
        return Err(ModelStabilityError::InvalidDigest);
    }
    if !input.tie_tolerance.is_finite() || input.tie_tolerance < 0.0 {
        return Err(ModelStabilityError::InvalidTieTolerance);
    }

    let mut models: BTreeMap<&str, Vec<&ModelActionEvaluation>> = BTreeMap::new();
    let mut execution_receipt_ids = BTreeSet::new();
    let mut action_commands: BTreeMap<&str, (String, VehicleCommand)> = BTreeMap::new();
    let mut action_ids = BTreeSet::new();
    for evaluation in &input.evaluations {
        if [
            evaluation.model_id.as_str(),
            evaluation.evaluator_id.as_str(),
            evaluation.action_id.as_str(),
            evaluation.execution_receipt_id.as_str(),
        ]
        .iter()
        .any(|identity| identity.trim().is_empty())
        {
            return Err(ModelStabilityError::EmptyIdentity);
        }
        if !digest_is_valid(&evaluation.model_artifact_digest)
            || !digest_is_valid(&evaluation.model_profile_digest)
            || !digest_is_valid(&evaluation.evaluator_artifact_digest)
            || !digest_is_valid(&evaluation.execution_digest)
            || !digest_is_valid(&evaluation.command_digest)
            || !digest_is_valid(&evaluation.execution_context_digest)
        {
            return Err(ModelStabilityError::InvalidDigest);
        }
        if !evaluation.command.steering.is_finite()
            || !(-1.0..=1.0).contains(&evaluation.command.steering)
            || !evaluation.command.throttle.is_finite()
            || !(0.0..=1.0).contains(&evaluation.command.throttle)
            || !evaluation.command.brake.is_finite()
            || !(0.0..=1.0).contains(&evaluation.command.brake)
        {
            return Err(ModelStabilityError::InvalidCommand);
        }
        if evaluation.command_digest != command_digest(&evaluation.command) {
            return Err(ModelStabilityError::InvalidCommandDigest);
        }
        if !evaluation.objective_value.is_finite() {
            return Err(ModelStabilityError::InvalidObjective);
        }
        if evaluation.execution_receipt_id.trim().is_empty() {
            return Err(ModelStabilityError::InvalidReceipt);
        }
        if !execution_receipt_ids.insert(evaluation.execution_receipt_id.as_str()) {
            return Err(ModelStabilityError::DuplicateExecutionReceipt);
        }
        let expected_context = model_context_digest(input, evaluation);
        if evaluation.execution_context_digest != expected_context {
            return Err(ModelStabilityError::MismatchedExecutionContext);
        }

        if let Some((existing_digest, existing_command)) =
            action_commands.get(evaluation.action_id.as_str())
        {
            if existing_digest != &evaluation.command_digest {
                return Err(ModelStabilityError::MismatchedActionCommand);
            }
            if command_digest(existing_command) != command_digest(&evaluation.command) {
                return Err(ModelStabilityError::MismatchedActionCommand);
            }
        } else {
            action_commands.insert(
                evaluation.action_id.as_str(),
                (evaluation.command_digest.clone(), evaluation.command),
            );
            action_ids.insert(evaluation.action_id.as_str());
        }

        models
            .entry(evaluation.model_id.as_str())
            .or_default()
            .push(evaluation);
    }

    if action_ids.is_empty() || !action_ids.contains(input.selected_action_id.as_str()) {
        return Err(ModelStabilityError::MissingSelectedAction);
    }
    if models.len() < MIN_MODEL_STABILITY_PROFILES {
        return Err(ModelStabilityError::TooFewModels);
    }
    if models.len() > MAX_MODEL_STABILITY_PROFILES {
        return Err(ModelStabilityError::TooManyModels);
    }

    let mut per_model = Vec::with_capacity(models.len());
    let mut profile_digests = BTreeSet::new();
    let mut all_best_ids = BTreeSet::new();
    let mut selected_action_feasible_model_count = 0usize;
    let mut better_action_model_count = 0usize;
    let mut no_feasible_action_model_count = 0usize;
    let mut model_has_unknown_feasibility = false;
    let mut worst_case_selected_action_regret: Option<f64> = Some(0.0);

    for (model_id, evaluations) in &models {
        let first = evaluations[0];
        if evaluations.iter().any(|evaluation| {
            evaluation.model_artifact_digest != first.model_artifact_digest
                || evaluation.model_profile_digest != first.model_profile_digest
                || evaluation.evaluator_id != first.evaluator_id
                || evaluation.evaluator_artifact_digest != first.evaluator_artifact_digest
        }) {
            return Err(ModelStabilityError::MixedModelIdentity);
        }
        if !profile_digests.insert(first.model_profile_digest.as_str()) {
            return Err(ModelStabilityError::DuplicateModelProfile);
        }
        if evaluations
            .iter()
            .any(|evaluation| evaluation.evaluator_id != first.evaluator_id
                || evaluation.evaluator_artifact_digest != first.evaluator_artifact_digest)
        {
            return Err(ModelStabilityError::MixedEvaluatorIdentity);
        }

        let mut seen_actions = BTreeSet::new();
        for evaluation in evaluations {
            if !seen_actions.insert(evaluation.action_id.as_str()) {
                return Err(ModelStabilityError::DuplicateModelAction);
            }
        }
        if seen_actions != action_ids {
            return Err(ModelStabilityError::MismatchedActionSets);
        }

        let selected = evaluations
            .iter()
            .find(|evaluation| evaluation.action_id == input.selected_action_id)
            .ok_or(ModelStabilityError::MissingSelectedAction)?;

        let feasible = evaluations
            .iter()
            .filter(|evaluation| evaluation.feasibility == ModelActionFeasibility::Feasible)
            .collect::<Vec<_>>();
        let unknown = evaluations
            .iter()
            .any(|evaluation| evaluation.feasibility == ModelActionFeasibility::Indeterminate);
        model_has_unknown_feasibility |= unknown;

        if feasible.is_empty() {
            no_feasible_action_model_count += 1;
            worst_case_selected_action_regret = None;
            per_model.push(PerModelStabilityFinding {
                model_id: (*model_id).to_owned(),
                model_artifact_digest: first.model_artifact_digest.clone(),
                model_profile_digest: first.model_profile_digest.clone(),
                selected_action_feasibility: selected.feasibility,
                selected_objective_value: selected.objective_value,
                best_feasible_action_id: None,
                best_feasible_objective_value: None,
                selected_action_regret: None,
                feasible_action_count: 0,
            });
            continue;
        }

        let best = feasible
            .iter()
            .copied()
            .reduce(|best, candidate| {
                let candidate_is_better = match input.direction {
                    ModelStabilityDirection::LowerIsBetter => {
                        candidate.objective_value < best.objective_value
                    }
                    ModelStabilityDirection::HigherIsBetter => {
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
            .ok_or(ModelStabilityError::InvalidObjective)?;
        all_best_ids.insert(best.action_id.as_str());

        let selected_regret = if selected.feasibility == ModelActionFeasibility::Feasible {
            selected_action_feasible_model_count += 1;
            let raw_regret = match input.direction {
                ModelStabilityDirection::LowerIsBetter => {
                    selected.objective_value - best.objective_value
                }
                ModelStabilityDirection::HigherIsBetter => {
                    best.objective_value - selected.objective_value
                }
            };
            if !raw_regret.is_finite() {
                return Err(ModelStabilityError::InvalidObjective);
            }
            let regret = raw_regret.max(0.0);
            if regret > input.tie_tolerance {
                better_action_model_count += 1;
            }
            worst_case_selected_action_regret = Some(
                worst_case_selected_action_regret
                    .unwrap_or(0.0)
                    .max(regret),
            );
            Some(regret)
        } else {
            worst_case_selected_action_regret = None;
            None
        };

        per_model.push(PerModelStabilityFinding {
            model_id: (*model_id).to_owned(),
            model_artifact_digest: first.model_artifact_digest.clone(),
            model_profile_digest: first.model_profile_digest.clone(),
            selected_action_feasibility: selected.feasibility,
            selected_objective_value: selected.objective_value,
            best_feasible_action_id: Some(best.action_id.clone()),
            best_feasible_objective_value: Some(best.objective_value),
            selected_action_regret: selected_regret,
            feasible_action_count: feasible.len(),
        });
    }

    let model_count = models.len();
    let best_action_agreement_rate = per_model
        .iter()
        .filter_map(|finding| finding.best_feasible_action_id.as_deref())
        .fold(BTreeMap::<&str, usize>::new(), |mut counts, action_id| {
            *counts.entry(action_id).or_default() += 1;
            counts
        })
        .values()
        .copied()
        .max()
        .unwrap_or(0) as f64
        / model_count as f64;
    let mut best_action_ids = all_best_ids
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    best_action_ids.sort();

    let outcome = if !input.action_set_complete
        || !input.model_set_complete
        || model_has_unknown_feasibility
    {
        ModelStabilityOutcome::Indeterminate
    } else if no_feasible_action_model_count > 0 {
        ModelStabilityOutcome::NoFeasibleActionInAtLeastOneModel
    } else if selected_action_feasible_model_count < model_count {
        ModelStabilityOutcome::SelectedActionInfeasibleInAtLeastOneModel
    } else if best_action_ids.len() > 1 {
        ModelStabilityOutcome::BestActionChangesAcrossModels
    } else if best_action_ids.first().is_some_and(|best| best == &input.selected_action_id) {
        ModelStabilityOutcome::SelectedActionBestInEveryModel
    } else {
        ModelStabilityOutcome::SameAlternativeBestInEveryModel
    };

    Ok(VehicleModelStabilityFinding {
        selected_action_id: input.selected_action_id.clone(),
        model_count,
        action_count: action_ids.len(),
        selected_action_feasible_model_count,
        better_action_model_count,
        best_action_agreement_rate,
        worst_case_selected_action_regret,
        best_action_ids,
        outcome,
        per_model,
    })
}

fn model_context_digest(
    input: &VehicleModelStabilityInput,
    evaluation: &ModelActionEvaluation,
) -> String {
    let context = (
        &input.decision_id,
        &input.scenario_id,
        &input.scenario_artifact_digest,
        &input.initial_state_digest,
        &input.environment_profile_digest,
        &input.horizon_profile_digest,
        &input.metric_id,
        &input.metric_unit_id,
        input.direction,
        input.tie_tolerance.to_bits(),
        &evaluation.model_id,
        &evaluation.model_artifact_digest,
        &evaluation.model_profile_digest,
        &evaluation.evaluator_id,
        &evaluation.evaluator_artifact_digest,
    );
    let bytes = serde_json::to_vec(&context).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn command_digest(command: &VehicleCommand) -> String {
    let bytes = serde_json::to_vec(command).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn digest_is_valid(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(steering: f32) -> VehicleCommand {
        VehicleCommand {
            steering,
            throttle: 0.2,
            brake: 0.0,
        }
    }

    fn profile(
        model_id: &str,
        model_digest: &str,
        profile_digest: &str,
        action_id: &str,
        command: VehicleCommand,
        feasibility: ModelActionFeasibility,
        objective: f64,
    ) -> ModelActionEvaluation {
        let command_digest = command_digest(&command);
        ModelActionEvaluation {
            model_id: model_id.into(),
            model_artifact_digest: model_digest.into(),
            model_profile_digest: profile_digest.into(),
            evaluator_id: "independent-evaluator-v1".into(),
            evaluator_artifact_digest: "e".repeat(64),
            execution_context_digest: String::new(),
            execution_receipt_id: format!("receipt-{model_id}-{action_id}"),
            execution_digest: blake3::hash(format!("{model_id}-{action_id}").as_bytes())
                .to_hex()
                .to_string(),
            action_id: action_id.into(),
            command,
            command_digest,
            feasibility,
            objective_value: objective,
        }
    }

    fn input() -> VehicleModelStabilityInput {
        let mut input = VehicleModelStabilityInput {
            decision_id: "decision-01".into(),
            scenario_id: "crossing-01".into(),
            scenario_artifact_digest: "a".repeat(64),
            initial_state_digest: "b".repeat(64),
            environment_profile_digest: "c".repeat(64),
            horizon_profile_digest: "d".repeat(64),
            metric_id: "minimum-clearance-margin".into(),
            metric_unit_id: "meters".into(),
            direction: ModelStabilityDirection::HigherIsBetter,
            tie_tolerance: 1e-6,
            selected_action_id: "selected".into(),
            action_set_complete: true,
            model_set_complete: true,
            evaluations: vec![
                profile(
                    "model-dry",
                    &"1".repeat(64),
                    &"2".repeat(64),
                    "selected",
                    command(0.0),
                    ModelActionFeasibility::Feasible,
                    1.0,
                ),
                profile(
                    "model-dry",
                    &"1".repeat(64),
                    &"2".repeat(64),
                    "alternative",
                    command(0.2),
                    ModelActionFeasibility::Feasible,
                    0.8,
                ),
                profile(
                    "model-wet",
                    &"3".repeat(64),
                    &"4".repeat(64),
                    "selected",
                    command(0.0),
                    ModelActionFeasibility::Feasible,
                    0.7,
                ),
                profile(
                    "model-wet",
                    &"3".repeat(64),
                    &"4".repeat(64),
                    "alternative",
                    command(0.2),
                    ModelActionFeasibility::Feasible,
                    0.9,
                ),
            ],
        };
        let context_input = input.clone();
        for evaluation in &mut input.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        input
    }

    #[test]
    fn detects_best_action_reversal_across_two_models() {
        let report = evaluate_vehicle_model_stability(input()).unwrap();
        assert_eq!(
            report.finding.outcome,
            ModelStabilityOutcome::BestActionChangesAcrossModels
        );
        assert_eq!(report.finding.best_action_ids, vec!["alternative", "selected"]);
        assert_eq!(report.finding.best_action_agreement_rate, 0.5);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn recognizes_selected_action_best_in_every_model() {
        let mut sample = input();
        sample.evaluations[3].objective_value = 0.5;
        let context_input = sample.clone();
        for evaluation in &mut sample.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        let report = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            ModelStabilityOutcome::SelectedActionBestInEveryModel
        );
        assert_eq!(report.finding.best_action_agreement_rate, 1.0);
    }

    #[test]
    fn incomplete_model_or_action_set_is_indeterminate() {
        let mut sample = input();
        sample.model_set_complete = false;
        let report = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(report.finding.outcome, ModelStabilityOutcome::Indeterminate);

        let mut sample = input();
        sample.action_set_complete = false;
        let report = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(report.finding.outcome, ModelStabilityOutcome::Indeterminate);
    }

    #[test]
    fn selected_action_infeasible_in_one_model_is_not_hidden() {
        let mut sample = input();
        sample.evaluations[2].feasibility = ModelActionFeasibility::Infeasible;
        let context_input = sample.clone();
        for evaluation in &mut sample.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        let report = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(
            report.finding.outcome,
            ModelStabilityOutcome::SelectedActionInfeasibleInAtLeastOneModel
        );
    }

    #[test]
    fn unknown_feasibility_is_indeterminate() {
        let mut sample = input();
        sample.evaluations[2].feasibility = ModelActionFeasibility::Indeterminate;
        let context_input = sample.clone();
        for evaluation in &mut sample.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        let report = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(report.finding.outcome, ModelStabilityOutcome::Indeterminate);
    }

    #[test]
    fn catches_mismatched_action_set_and_command_payload() {
        let mut mismatch = input();
        mismatch.evaluations.pop();
        assert_eq!(
            evaluate_vehicle_model_stability(mismatch).unwrap_err(),
            ModelStabilityError::MismatchedActionSets
        );

        let mut mismatch = input();
        mismatch.evaluations[2].command = command(-0.1);
        mismatch.evaluations[2].command_digest = command_digest(&mismatch.evaluations[2].command);
        let context_digest =
            model_context_digest(&mismatch, &mismatch.evaluations[2]);
        mismatch.evaluations[2].execution_context_digest = context_digest;
        assert_eq!(
            evaluate_vehicle_model_stability(mismatch).unwrap_err(),
            ModelStabilityError::MismatchedActionCommand
        );
    }

    #[test]
    fn input_order_does_not_change_canonical_model_report() {
        let mut sample = input();
        let first = evaluate_vehicle_model_stability(sample.clone()).unwrap();
        sample.evaluations.reverse();
        let context_input = sample.clone();
        for evaluation in &mut sample.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        let second = evaluate_vehicle_model_stability(sample).unwrap();
        assert_eq!(first.report_digest, second.report_digest);
    }

    #[test]
    fn report_validator_detects_tampering_even_after_digest_recomputation() {
        let mut report = evaluate_vehicle_model_stability(input()).unwrap();
        report.finding.best_action_agreement_rate = 1.0;
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }

    #[test]
    fn rejects_duplicate_receipts_and_single_model_inputs() {
        let mut duplicate = input();
        duplicate.evaluations[2].execution_receipt_id =
            duplicate.evaluations[0].execution_receipt_id.clone();
        assert_eq!(
            evaluate_vehicle_model_stability(duplicate).unwrap_err(),
            ModelStabilityError::DuplicateExecutionReceipt
        );

        let mut one_model = input();
        one_model
            .evaluations
            .retain(|evaluation| evaluation.model_id == "model-dry");
        assert_eq!(
            evaluate_vehicle_model_stability(one_model).unwrap_err(),
            ModelStabilityError::TooFewModels
        );
    }

    #[test]
    fn model_profiles_must_be_unique_and_distinct() {
        let mut sample = input();
        sample.evaluations[2].model_profile_digest = sample.evaluations[0].model_profile_digest.clone();
        let context_input = sample.clone();
        for evaluation in &mut sample.evaluations {
            evaluation.execution_context_digest =
                model_context_digest(&context_input, evaluation);
        }
        assert_eq!(
            evaluate_vehicle_model_stability(sample).unwrap_err(),
            ModelStabilityError::DuplicateModelProfile
        );
    }
}
