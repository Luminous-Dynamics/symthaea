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

/// One policy decision; confidence is the estimated probability that a direct
/// answer would be correct under the currently available evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectivePolicyDecision {
    pub action: DecisionAction,
    pub confidence: f64,
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
            },
            Self::AlwaysAbstain => SelectivePolicyDecision {
                action: DecisionAction::Abstain,
                confidence: 0.05,
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
                SelectivePolicyDecision { action, confidence }
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
    pub total_cost_units: f64,
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
    InvalidPolicyConfidence { index: usize },
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
            Self::InvalidPolicyConfidence { index } => {
                write!(f, "policy confidence for item {index} must be finite and in [0, 1]")
            }
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
    }
    Ok(())
}

/// Evaluate one policy over exactly the supplied evaluator-owned items.
pub fn evaluate_policy(
    items: &[SyntheticSelectivePredictionItem],
    policy: SelectivePredictionBaseline,
    cost_model: SelectivePredictionCostModel,
) -> Result<SelectivePredictionTaskReport, SelectivePredictionTaskError> {
    validate_items(items)?;
    cost_model.validate()?;

    let manifest = serde_json::to_vec(items)
        .map_err(|error| SelectivePredictionTaskError::Serialization(error.to_string()))?;
    let manifest_digest = blake3::hash(&manifest).to_hex().to_string();

    let mut trials = Vec::with_capacity(items.len());
    let mut total_information_requests = 0usize;
    let mut useful_information_requests = 0usize;
    let mut clarification_opportunities = 0usize;
    let mut total_cost_units = 0.0;

    for (index, item) in items.iter().enumerate() {
        let decision = policy.decide(&item.observation);
        if !decision.confidence.is_finite() || !(0.0..=1.0).contains(&decision.confidence) {
            return Err(SelectivePredictionTaskError::InvalidPolicyConfidence { index });
        }

        if item.clarification_resolves {
            clarification_opportunities += 1;
        }
        if decision.action == DecisionAction::RequestInformation {
            total_information_requests += 1;
            if item.clarification_resolves {
                useful_information_requests += 1;
            }
        }

        let cost_units = cost_model.charge(decision.action);
        total_cost_units += cost_units;
        trials.push(SelectivePredictionTrial {
            is_answerable: item.answerable_now,
            action: decision.action,
            answer_correct_if_given: item.answer_correct_if_given,
            confidence: decision.confidence,
            cost_units,
        });
    }

    let metrics = SelectivePredictionMetrics::from_trials(&trials)
        .map_err(|error| SelectivePredictionTaskError::Scoring(error.to_string()))?;

    Ok(SelectivePredictionTaskReport {
        scenario_id: SCENARIO_ID.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        policy: policy.name().to_owned(),
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
        total_cost_units,
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
