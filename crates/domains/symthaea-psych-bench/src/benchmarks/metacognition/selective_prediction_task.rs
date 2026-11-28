// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Controlled task adapter for selective prediction.
//!
//! This module evaluates decision policies over a fixed, evaluator-annotated
//! item set. Policies receive only SelectivePredictionObservation; item
//! answerability, counterfactual correctness and clarification outcomes remain
//! evaluator-side. The bundled fixture is synthetic and is not a human norm or
//! a measured Symthaea result.

use super::selective_prediction::{
    DecisionAction, SelectivePredictionMetrics, SelectivePredictionTrial,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SCENARIO_ID: &str = "psych.selective-control.v0";
pub const GENERATOR_VERSION: &str = "selective-control-fixture-0.1.0";

/// Agent-visible evidence state. Evaluator labels are intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectivePredictionObservation {
    /// Synthetic evidence strength in [0, 1].
    pub evidence_strength: f64,
    /// Whether the premise is internally consistent with the supplied facts.
    pub premise_consistent: bool,
    /// Whether the supplied evidence is current for the task's time horizon.
    pub source_is_current: bool,
    /// Whether a clarification or evidence request can obtain relevant context.
    pub clarification_available: bool,
}

/// Which missing fact the policy is requesting. The target is observable to
/// the evaluator as part of the action trace, while the correct target remains
/// evaluator-owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClarificationTarget {
    SupplyMissingConstraint,
    VerifyPremise,
    RefreshSource,
}

fn target_for_observation(observation: &SelectivePredictionObservation) -> ClarificationTarget {
    if !observation.premise_consistent {
        ClarificationTarget::VerifyPremise
    } else if !observation.source_is_current {
        ClarificationTarget::RefreshSource
    } else {
        ClarificationTarget::SupplyMissingConstraint
    }
}

/// One policy decision; confidence is the estimated probability that a direct
/// answer would be correct under the currently available evidence. An
/// information request must identify the uncertainty it is trying to resolve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectivePolicyDecision {
    pub action: DecisionAction,
    pub confidence: f64,
    pub clarification_target: Option<ClarificationTarget>,
}

/// A fixed candidate-answer case. Evaluator-only fields are private so policy
/// implementations cannot read labels through this API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSelectivePredictionItem {
    pub id: String,
    pub observation: SelectivePredictionObservation,
    answerable_now: bool,
    answer_correct_if_given: bool,
    clarification_resolves: bool,
    clarification_target: Option<ClarificationTarget>,
    post_clarification_observation: Option<SelectivePredictionObservation>,
    answer_correct_after_clarification: bool,
}

/// Transparent fixed baselines. These are policy controls, not foundation-model
/// competitors or trained Symthaea policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectivePredictionBaseline {
    AlwaysAnswer,
    AlwaysAbstain,
    EvidenceThreshold,
    PremiseAndFreshnessAware,
}

impl SelectivePredictionBaseline {
    pub fn name(self) -> &'static str {
        match self {
            Self::AlwaysAnswer => "always_answer",
            Self::AlwaysAbstain => "always_abstain",
            Self::EvidenceThreshold => "evidence_threshold",
            Self::PremiseAndFreshnessAware => "premise_freshness_aware",
        }
    }

    /// The policy receives only the observable evidence state.
    pub fn decide(self, observation: &SelectivePredictionObservation) -> SelectivePolicyDecision {
        match self {
            Self::AlwaysAnswer => SelectivePolicyDecision {
                action: DecisionAction::Answer,
                confidence: 0.95,
                clarification_target: None,
            },
            Self::AlwaysAbstain => SelectivePolicyDecision {
                action: DecisionAction::Abstain,
                confidence: 0.05,
                clarification_target: None,
            },
            Self::EvidenceThreshold => {
                let action = if observation.evidence_strength >= 0.70 {
                    DecisionAction::Answer
                } else if observation.clarification_available {
                    DecisionAction::RequestInformation
                } else {
                    DecisionAction::Abstain
                };
                SelectivePolicyDecision {
                    action,
                    confidence: observation.evidence_strength,
                    clarification_target: (action == DecisionAction::RequestInformation)
                        .then(|| target_for_observation(observation)),
                }
            }
            Self::PremiseAndFreshnessAware => {
                let action = if !observation.premise_consistent || !observation.source_is_current {
                    if observation.clarification_available {
                        DecisionAction::RequestInformation
                    } else {
                        DecisionAction::Abstain
                    }
                } else if observation.evidence_strength >= 0.80 {
                    DecisionAction::Answer
                } else if observation.clarification_available {
                    DecisionAction::RequestInformation
                } else {
                    DecisionAction::Abstain
                };

                let confidence = if !observation.premise_consistent {
                    0.05
                } else if !observation.source_is_current {
                    0.15
                } else {
                    (0.25 + 0.70 * observation.evidence_strength).clamp(0.0, 0.95)
                };
                SelectivePolicyDecision {
                    action,
                    confidence,
                    clarification_target: (action == DecisionAction::RequestInformation)
                        .then(|| target_for_observation(observation)),
                }
            }
        }
    }
}

/// Action costs are relative fixture units, not latency, tokens, joules or money.
/// The evaluator charges them; a policy cannot self-report a lower cost.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectivePredictionCostModel {
    pub answer_units: f64,
    pub abstain_units: f64,
    pub information_request_units: f64,
}

impl Default for SelectivePredictionCostModel {
    fn default() -> Self {
        Self {
            answer_units: 1.0,
            abstain_units: 0.25,
            information_request_units: 2.0,
        }
    }
}

impl SelectivePredictionCostModel {
    fn charge(self, action: DecisionAction) -> f64 {
        match action {
            DecisionAction::Answer => self.answer_units,
            DecisionAction::Abstain => self.abstain_units,
            DecisionAction::RequestInformation => self.information_request_units,
        }
    }

    fn validate(self) -> Result<(), SelectivePredictionTaskError> {
        let values = [
            self.answer_units,
            self.abstain_units,
            self.information_request_units,
        ];
        if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(SelectivePredictionTaskError::InvalidCostModel);
        }
        Ok(())
    }
}

/// Task-level report combines the reusable scorer with clarification quality.
/// No scalar composite is used: the report preserves the decision trade-offs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectivePredictionTaskReport {
    pub scenario_id: String,
    pub generator_version: String,
    pub policy: String,
    pub manifest_digest: String,
    pub items: usize,
    pub clarification_opportunities: usize,
    pub useful_information_requests: usize,
    pub total_information_requests: usize,
    /// Useful requests / all information requests.
    pub clarification_precision: Option<f64>,
    /// Useful requests / all available resolvable clarifications.
    pub clarification_recall: Option<f64>,
    pub resolved_clarifications: usize,
    pub final_answer_count: usize,
    pub final_correct_count: usize,
    pub post_clarification_answer_count: usize,
    pub post_clarification_correct_count: usize,
    pub final_answer_accuracy: Option<f64>,
    pub post_clarification_accuracy: Option<f64>,
    /// Includes initial actions and any evaluator-mediated follow-up decision.
    pub total_cost_units: f64,
    /// Relative cost of post-clarification decisions only.
    pub post_clarification_cost_units: f64,
    pub metrics: SelectivePredictionMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectivePredictionTaskError {
    EmptyItems,
    EmptyItemId { index: usize },
    DuplicateItemId { id: String },
    InvalidEvidenceStrength { index: usize },
    UnanswerableItemMarkedCorrect { index: usize },
    ClarificationWithoutAvailableContext { index: usize },
    MissingClarificationOutcome { index: usize },
    UnexpectedClarificationOutcome { index: usize },
    InvalidClarificationObservation { index: usize },
    InvalidPolicyConfidence { index: usize },
    InvalidPolicyDecision { index: usize },
    EmptyPolicyName,
    InvalidCostModel,
    Serialization(String),
    Scoring(String),
}

impl std::fmt::Display for SelectivePredictionTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyItems => write!(f, "selective-prediction task requires at least one item"),
            Self::EmptyItemId { index } => write!(f, "item {index} has an empty ID"),
            Self::DuplicateItemId { id } => write!(f, "duplicate selective-prediction item ID: {id}"),
            Self::InvalidEvidenceStrength { index } => {
                write!(f, "item {index} evidence strength must be finite and in [0, 1]")
            }
            Self::UnanswerableItemMarkedCorrect { index } => {
                write!(f, "unanswerable item {index} cannot have a correct direct answer")
            }
            Self::ClarificationWithoutAvailableContext { index } => write!(
                f,
                "item {index} cannot be resolved by clarification when no context is available"
            ),
            Self::MissingClarificationOutcome { index } => {
                write!(f, "item {index} declares a resolvable clarification without a post-clarification state")
            }
            Self::UnexpectedClarificationOutcome { index } => {
                write!(f, "item {index} supplies a post-clarification state without a resolvable clarification")
            }
            Self::InvalidClarificationObservation { index } => {
                write!(f, "item {index} has an invalid post-clarification observation")
            }
            Self::InvalidPolicyConfidence { index } => {
                write!(f, "policy confidence for item {index} must be finite and in [0, 1]")
            }
            Self::InvalidPolicyDecision { index } => {
                write!(f, "policy returned an information request without a clarification target, or attached a target to a non-request action, for item {index}")
            }
            Self::EmptyPolicyName => write!(f, "custom selective-prediction policy name must not be empty"),
            Self::InvalidCostModel => write!(f, "all action costs must be finite and non-negative"),
            Self::Serialization(message) => write!(f, "could not digest task manifest: {message}"),
            Self::Scoring(message) => write!(f, "selective-prediction scoring failed: {message}"),
        }
    }
}

impl std::error::Error for SelectivePredictionTaskError {}

/// Build the deterministic v0.1 fixture. It deliberately includes answerable
/// difficult items as well as underspecified, false-premise, stale and
/// information-poor cases. It is a smoke fixture, not a broad benchmark sample.
pub fn fixture_items() -> Vec<SyntheticSelectivePredictionItem> {
    fn item(
        id: &str,
        evidence_strength: f64,
        premise_consistent: bool,
        source_is_current: bool,
        clarification_available: bool,
        answerable_now: bool,
        answer_correct_if_given: bool,
        clarification_resolves: bool,
    ) -> SyntheticSelectivePredictionItem {
        SyntheticSelectivePredictionItem {
            id: id.to_owned(),
            observation: SelectivePredictionObservation {
                evidence_strength,
                premise_consistent,
                source_is_current,
                clarification_available,
            },
            answerable_now,
            answer_correct_if_given,
            clarification_resolves,
            clarification_target: clarification_available.then(|| {
                if !premise_consistent {
                    ClarificationTarget::VerifyPremise
                } else if !source_is_current {
                    ClarificationTarget::RefreshSource
                } else {
                    ClarificationTarget::SupplyMissingConstraint
                }
            }),
            post_clarification_observation: clarification_resolves.then_some(
                SelectivePredictionObservation {
                    evidence_strength: 0.94,
                    premise_consistent: true,
                    source_is_current: true,
                    clarification_available: false,
                },
            ),
            answer_correct_after_clarification: clarification_resolves,
        }
    }

    vec![
        item("supported-01", 0.98, true, true, false, true, true, false),
        item("supported-02", 0.92, true, true, false, true, true, false),
        item("supported-03", 0.86, true, true, false, true, true, false),
        item("supported-04", 0.82, true, true, false, true, true, false),
        item("difficult-01", 0.68, true, true, false, true, true, false),
        item("difficult-02", 0.58, true, true, false, true, false, false),
        item("difficult-03", 0.52, true, true, false, true, false, false),
        item("missing-constraint-01", 0.42, true, true, true, false, false, true),
        item("missing-constraint-02", 0.35, true, true, true, false, false, true),
        item("false-premise-01", 0.96, false, true, false, false, false, false),
        item("false-premise-02", 0.89, false, true, false, false, false, false),
        item("stale-evidence-01", 0.84, true, false, true, false, false, true),
        item("stale-evidence-02", 0.63, true, false, true, false, false, true),
        item("unavailable-source-01", 0.12, true, false, false, false, false, false),
    ]
}

fn validate_items(
    items: &[SyntheticSelectivePredictionItem],
) -> Result<(), SelectivePredictionTaskError> {
    if items.is_empty() {
        return Err(SelectivePredictionTaskError::EmptyItems);
    }

    let mut ids = HashSet::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        if item.id.trim().is_empty() {
            return Err(SelectivePredictionTaskError::EmptyItemId { index });
        }
        if !ids.insert(item.id.as_str()) {
            return Err(SelectivePredictionTaskError::DuplicateItemId {
                id: item.id.clone(),
            });
        }
        let evidence = item.observation.evidence_strength;
        if !evidence.is_finite() || !(0.0..=1.0).contains(&evidence) {
            return Err(SelectivePredictionTaskError::InvalidEvidenceStrength { index });
        }
        if !item.answerable_now && item.answer_correct_if_given {
            return Err(SelectivePredictionTaskError::UnanswerableItemMarkedCorrect { index });
        }
        if item.clarification_resolves && !item.observation.clarification_available {
            return Err(SelectivePredictionTaskError::ClarificationWithoutAvailableContext { index });
        }
        match (
            item.clarification_resolves,
            item.post_clarification_observation.as_ref(),
        ) {
            (true, Some(observation)) => {
                let evidence = observation.evidence_strength;
                if !evidence.is_finite()
                    || !(0.0..=1.0).contains(&evidence)
                    || observation.clarification_available
                    || !observation.premise_consistent
                    || !observation.source_is_current
                {
                    return Err(SelectivePredictionTaskError::InvalidClarificationObservation {
                        index,
                    });
                }
            }
            (true, None) => {
                return Err(SelectivePredictionTaskError::MissingClarificationOutcome { index });
            }
            (false, Some(_)) => {
                return Err(SelectivePredictionTaskError::UnexpectedClarificationOutcome { index });
            }
            (false, None) => {}
        }
        if item.observation.clarification_available
            && item.clarification_target != Some(target_for_observation(&item.observation))
        {
            return Err(SelectivePredictionTaskError::InvalidClarificationObservation { index });
        }
    }
    Ok(())
}

/// Evaluate one policy over exactly the supplied evaluator-owned items.
pub fn evaluate_policy(
    items: &[SyntheticSelectivePredictionItem],
    policy: SelectivePredictionBaseline,
    cost_model: SelectivePredictionCostModel,
) -> Result<SelectivePredictionTaskReport, SelectivePredictionTaskError> {
    evaluate_policy_with(items, policy.name(), |observation| policy.decide(observation), cost_model)
}

/// Evaluate an external policy adapter over the same evaluator-owned fixture.
///
/// The closure receives only public observations; answerability, correctness,
/// clarification targets and post-clarification outcomes remain hidden. This
/// is the integration seam for a live Symthaea policy or a black-box model
/// adapter. The benchmark fixture remains synthetic and is not an external SOTA set.
pub fn evaluate_policy_with<F>(
    items: &[SyntheticSelectivePredictionItem],
    policy_name: &str,
    decide: F,
    cost_model: SelectivePredictionCostModel,
) -> Result<SelectivePredictionTaskReport, SelectivePredictionTaskError>
where
    F: Fn(&SelectivePredictionObservation) -> SelectivePolicyDecision,
{
    if policy_name.trim().is_empty() {
        return Err(SelectivePredictionTaskError::EmptyPolicyName);
    }
    validate_items(items)?;
    cost_model.validate()?;

    let manifest = serde_json::to_vec(items)
        .map_err(|error| SelectivePredictionTaskError::Serialization(error.to_string()))?;
    let manifest_digest = blake3::hash(&manifest).to_hex().to_string();

    let mut trials = Vec::with_capacity(items.len());
    let mut total_information_requests = 0usize;
    let mut useful_information_requests = 0usize;
    let mut clarification_opportunities = 0usize;
    let mut resolved_clarifications = 0usize;
    let mut final_answer_count = 0usize;
    let mut final_correct_count = 0usize;
    let mut post_clarification_answer_count = 0usize;
    let mut post_clarification_correct_count = 0usize;
    let mut total_cost_units = 0.0;
    let mut post_clarification_cost_units = 0.0;

    for (index, item) in items.iter().enumerate() {
        let decision = decide(&item.observation);
        if !decision.confidence.is_finite() || !(0.0..=1.0).contains(&decision.confidence) {
            return Err(SelectivePredictionTaskError::InvalidPolicyConfidence { index });
        }
        if (decision.action == DecisionAction::RequestInformation)
            != decision.clarification_target.is_some()
        {
            return Err(SelectivePredictionTaskError::InvalidPolicyDecision { index });
        }

        if item.clarification_resolves {
            clarification_opportunities += 1;
        }

        let initial_cost = cost_model.charge(decision.action);
        total_cost_units += initial_cost;
        trials.push(SelectivePredictionTrial {
            is_answerable: item.answerable_now,
            action: decision.action,
            answer_correct_if_given: item.answer_correct_if_given,
            confidence: decision.confidence,
            cost_units: initial_cost,
        });

        match decision.action {
            DecisionAction::Answer => {
                final_answer_count += 1;
                if item.answerable_now && item.answer_correct_if_given {
                    final_correct_count += 1;
                }
            }
            DecisionAction::Abstain => {}
            DecisionAction::RequestInformation => {
                total_information_requests += 1;
                let correctly_targeted = decision.clarification_target.is_some()
                    && decision.clarification_target == item.clarification_target;
                if correctly_targeted && item.clarification_resolves {
                    useful_information_requests += 1;
                    resolved_clarifications += 1;

                    let updated_observation = item
                        .post_clarification_observation
                        .as_ref()
                        .expect("validated clarification outcome");
                    let follow_up = decide(updated_observation);
                    if !follow_up.confidence.is_finite()
                        || !(0.0..=1.0).contains(&follow_up.confidence)
                    {
                        return Err(SelectivePredictionTaskError::InvalidPolicyConfidence {
                            index,
                        });
                    }
                    if (follow_up.action == DecisionAction::RequestInformation)
                        != follow_up.clarification_target.is_some()
                    {
                        return Err(SelectivePredictionTaskError::InvalidPolicyDecision { index });
                    }

                    let follow_up_cost = cost_model.charge(follow_up.action);
                    total_cost_units += follow_up_cost;
                    post_clarification_cost_units += follow_up_cost;

                    match follow_up.action {
                        DecisionAction::Answer => {
                            final_answer_count += 1;
                            post_clarification_answer_count += 1;
                            if item.answer_correct_after_clarification {
                                final_correct_count += 1;
                                post_clarification_correct_count += 1;
                            }
                        }
                        DecisionAction::Abstain | DecisionAction::RequestInformation => {}
                    }
                }
            }
        }
    }

    let metrics = SelectivePredictionMetrics::from_trials(&trials)
        .map_err(|error| SelectivePredictionTaskError::Scoring(error.to_string()))?;

    Ok(SelectivePredictionTaskReport {
        scenario_id: SCENARIO_ID.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        policy: policy_name.to_owned(),
        manifest_digest,
        items: items.len(),
        clarification_opportunities,
        useful_information_requests,
        total_information_requests,
        clarification_precision: (total_information_requests > 0).then(|| {
            useful_information_requests as f64 / total_information_requests as f64
        }),
        clarification_recall: (clarification_opportunities > 0).then(|| {
            useful_information_requests as f64 / clarification_opportunities as f64
        }),
        resolved_clarifications,
        final_answer_count,
        final_correct_count,
        post_clarification_answer_count,
        post_clarification_correct_count,
        final_answer_accuracy: (final_answer_count > 0).then(|| {
            final_correct_count as f64 / final_answer_count as f64
        }),
        post_clarification_accuracy: (post_clarification_answer_count > 0).then(|| {
            post_clarification_correct_count as f64 / post_clarification_answer_count as f64
        }),
        total_cost_units,
        post_clarification_cost_units,
        metrics,
    })
}

/// Run matched policy controls over the same frozen fixture and manifest.
pub fn run_reference_suite(
) -> Result<Vec<SelectivePredictionTaskReport>, SelectivePredictionTaskError> {
    let items = fixture_items();
    let cost_model = SelectivePredictionCostModel::default();
    [
        SelectivePredictionBaseline::AlwaysAnswer,
        SelectivePredictionBaseline::AlwaysAbstain,
        SelectivePredictionBaseline::EvidenceThreshold,
        SelectivePredictionBaseline::PremiseAndFreshnessAware,
    ]
    .into_iter()
    .map(|policy| evaluate_policy(&items, policy, cost_model))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_controls_share_one_frozen_manifest() {
        let reports = run_reference_suite().expect("fixture should score");
        assert_eq!(reports.len(), 4);
        assert!(reports.windows(2).all(|pair| {
            pair[0].manifest_digest == pair[1].manifest_digest
                && pair[0].items == pair[1].items
        }));
        assert_eq!(reports[0].scenario_id, SCENARIO_ID);
        assert_eq!(reports[0].generator_version, GENERATOR_VERSION);
    }

    #[test]
    fn premise_and_freshness_control_avoids_known_unanswerable_traps() {
        let reports = run_reference_suite().expect("fixture should score");
        let aware = reports
            .iter()
            .find(|report| report.policy == "premise_freshness_aware")
            .expect("aware baseline");
        let always_answer = reports
            .iter()
            .find(|report| report.policy == "always_answer")
            .expect("always-answer baseline");

        assert_eq!(aware.metrics.inappropriate_answer_rate_on_unanswerable, Some(0.0));
        assert_eq!(
            always_answer.metrics.inappropriate_answer_rate_on_unanswerable,
            Some(1.0)
        );
        assert_eq!(aware.clarification_precision, Some(1.0));
        assert_eq!(aware.clarification_recall, Some(1.0));
        assert_eq!(aware.resolved_clarifications, 4);
        assert_eq!(aware.post_clarification_answer_count, 4);
        assert_eq!(aware.post_clarification_correct_count, 4);
        assert_eq!(aware.post_clarification_accuracy, Some(1.0));
        assert_eq!(aware.final_answer_accuracy, Some(1.0));
    }

    #[test]
    fn clarification_requires_the_right_target_and_a_valid_updated_observation() {
        let mut wrong_target = fixture_items();
        wrong_target[7].clarification_target = Some(ClarificationTarget::RefreshSource);
        assert!(matches!(
            validate_items(&wrong_target),
            Err(SelectivePredictionTaskError::InvalidClarificationObservation { .. })
        ));

        let mut invalid_update = fixture_items();
        invalid_update[7]
            .post_clarification_observation
            .as_mut()
            .expect("resolvable fixture item")
            .clarification_available = true;
        assert!(matches!(
            validate_items(&invalid_update),
            Err(SelectivePredictionTaskError::InvalidClarificationObservation { .. })
        ));
    }

    #[test]
    fn always_abstain_is_not_rewarded_as_a_complete_solution() {
        let reports = run_reference_suite().expect("fixture should score");
        let always_abstain = reports
            .iter()
            .find(|report| report.policy == "always_abstain")
            .expect("always-abstain baseline");

        assert_eq!(always_abstain.metrics.answer_rate, 0.0);
        assert_eq!(always_abstain.metrics.false_nonanswer_rate_on_answerable, Some(1.0));
        assert_eq!(always_abstain.metrics.accuracy_when_answered, None);
    }

    #[test]
    fn evidence_threshold_is_vulnerable_to_high_confidence_false_premises() {
        let reports = run_reference_suite().expect("fixture should score");
        let threshold = reports
            .iter()
            .find(|report| report.policy == "evidence_threshold")
            .expect("threshold baseline");
        let aware = reports
            .iter()
            .find(|report| report.policy == "premise_freshness_aware")
            .expect("aware baseline");

        assert!(
            threshold.metrics.inappropriate_answer_rate_on_unanswerable
                > aware.metrics.inappropriate_answer_rate_on_unanswerable
        );
    }

    #[test]
    fn evaluator_rejects_duplicate_ids_and_inconsistent_hidden_labels() {
        let mut items = fixture_items();
        items.push(items[0].clone());
        assert!(matches!(
            validate_items(&items),
            Err(SelectivePredictionTaskError::DuplicateItemId { .. })
        ));

        let mut inconsistent = fixture_items();
        inconsistent[0].answerable_now = false;
        assert!(matches!(
            validate_items(&inconsistent),
            Err(SelectivePredictionTaskError::UnanswerableItemMarkedCorrect { .. })
        ));
    }

    #[test]
    fn evaluator_rejects_non_finite_evidence_and_impossible_clarification_labels() {
        let mut bad_evidence = fixture_items();
        bad_evidence[0].observation.evidence_strength = f64::NAN;
        assert!(matches!(
            validate_items(&bad_evidence),
            Err(SelectivePredictionTaskError::InvalidEvidenceStrength { .. })
        ));

        let mut bad_clarification = fixture_items();
        bad_clarification[0].clarification_resolves = true;
        assert!(matches!(
            validate_items(&bad_clarification),
            Err(SelectivePredictionTaskError::ClarificationWithoutAvailableContext { .. })
        ));
    }
}
