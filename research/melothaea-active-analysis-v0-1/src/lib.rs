#![forbid(unsafe_code)]

//! One-step EVSI policy for choosing the next auditory analysis action.
//!
//! The module consumes an externally validated prior, task loss matrix and
//! calibrated action likelihoods. It chooses the measurement with greatest
//! positive net expected value under an explicit budget. It does not infer or
//! calibrate likelihoods itself, and it is not a globally optimal multi-step
//! sensing planner.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const NUMERICAL_TOLERANCE: f64 = 1.0e-12;
const DIGEST_LENGTH: usize = 64;
const MAX_HYPOTHESES: usize = 256;
const MAX_DECISIONS: usize = 256;
const MAX_ACTIONS: usize = 512;
const MAX_OUTCOMES_PER_ACTION: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvsiError {
    EmptyIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    InvalidDigest,
    InvalidProbability(&'static str),
    ProbabilityDimensionMismatch,
    LikelihoodDimensionMismatch,
    LossDimensionMismatch,
    EmptyHypothesisSet,
    EmptyDecisionSet,
    EmptyActionOutcomeSet,
    TooManyElements(&'static str),
    InvalidCost,
    InvalidBudget,
    InvalidCostWeight,
    InvalidLoss,
    InvalidLikelihoodModel,
    LikelihoodTableDigestMismatch,
    ActionNotAdmissible,
    UnknownOutcome,
    ZeroProbabilityOutcome,
    NumericalOverflow,
    NegativeEvsiBeyondTolerance,
}

impl fmt::Display for EvsiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::DuplicateIdentifier(field) => write!(formatter, "{field} identifiers must be unique"),
            Self::InvalidDigest => write!(formatter, "likelihood model identity must be lowercase SHA-256"),
            Self::InvalidProbability(field) => write!(formatter, "{field} must be finite, normalized and in [0, 1]"),
            Self::ProbabilityDimensionMismatch => write!(formatter, "prior probabilities do not match the hypothesis set"),
            Self::LikelihoodDimensionMismatch => write!(formatter, "outcome likelihoods do not match the hypothesis set"),
            Self::LossDimensionMismatch => write!(formatter, "decision loss vectors do not match the hypothesis set"),
            Self::EmptyHypothesisSet => write!(formatter, "at least one hypothesis is required"),
            Self::EmptyDecisionSet => write!(formatter, "at least one decision is required"),
            Self::EmptyActionOutcomeSet => write!(formatter, "every measurement action requires at least one outcome"),
            Self::TooManyElements(field) => write!(formatter, "{field} exceeds the bounded policy limit"),
            Self::InvalidCost => write!(formatter, "measurement cost must be finite and strictly positive"),
            Self::InvalidBudget => write!(formatter, "resource budget must be finite and non-negative"),
            Self::InvalidCostWeight => write!(formatter, "cost-to-risk weight must be finite and non-negative"),
            Self::InvalidLoss => write!(formatter, "decision losses must be finite and non-negative"),
            Self::InvalidLikelihoodModel => write!(formatter, "measurement likelihoods are not a complete probability model"),
            Self::LikelihoodTableDigestMismatch => write!(formatter, "measurement likelihood table digest does not match the supplied probabilities"),
            Self::ActionNotAdmissible => write!(formatter, "measurement action is not admissible"),
            Self::UnknownOutcome => write!(formatter, "observed outcome is not part of the selected action"),
            Self::ZeroProbabilityOutcome => write!(formatter, "observed outcome has zero probability under the supplied prior"),
            Self::NumericalOverflow => write!(formatter, "EVSI arithmetic overflowed or became non-finite"),
            Self::NegativeEvsiBeyondTolerance => write!(formatter, "expected posterior risk exceeds current Bayes risk beyond numerical tolerance"),
        }
    }
}

impl Error for EvsiError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorBeliefV1 {
    pub prior_id: String,
    pub hypothesis_ids: Vec<String>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionLossV1 {
    pub decision_id: String,
    pub losses_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationOutcomeV1 {
    pub outcome_id: String,
    /// Likelihood P(outcome | hypothesis), ordered like PriorBeliefV1.hypothesis_ids.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementActionV1 {
    pub action_id: String,
    /// Digest of a separately qualified/calibrated likelihood model.
    pub likelihood_table_sha256: String,
    pub cost: f64,
    pub admissible: bool,
    pub outcomes: Vec<ObservationOutcomeV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfigV1 {
    pub resource_budget: f64,
    /// Converts the declared resource unit into the decision-loss unit.
    pub cost_to_risk_weight: f64,
    /// When false, no likelihood-based action is selected outside its declared domain.
    pub in_domain: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveAnalysisRequestV1 {
    pub prior: PriorBeliefV1,
    pub decisions: Vec<DecisionLossV1>,
    pub actions: Vec<MeasurementActionV1>,
    pub policy: PolicyConfigV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NoActionReason {
    OutOfDomain,
    NoCandidates,
    NoAdmissibleActions,
    BudgetExhausted,
    NoPositiveNetValue,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutcomeValueV1 {
    pub outcome_id: String,
    pub marginal_probability: f64,
    /// None is used for a mathematically impossible outcome; no posterior is fabricated.
    pub posterior_bayes_risk: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActionEstimateV1 {
    pub action_id: String,
    pub likelihood_table_sha256: String,
    pub admissible: bool,
    pub within_budget: bool,
    pub current_bayes_risk: f64,
    pub expected_posterior_bayes_risk: f64,
    pub evsi: f64,
    pub cost: f64,
    pub evsi_per_cost: f64,
    pub net_expected_value: f64,
    pub outcomes: Vec<OutcomeValueV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveAnalysisReportV1 {
    pub schema_version: String,
    pub current_bayes_risk: f64,
    pub resource_budget: f64,
    pub cost_to_risk_weight: f64,
    pub chosen_action_id: Option<String>,
    pub no_action_reason: Option<NoActionReason>,
    pub estimates: Vec<ActionEstimateV1>,
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), EvsiError> {
    if value.trim().is_empty() {
        Err(EvsiError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn validate_unique<'a>(
    values: impl IntoIterator<Item = &'a str>,
    field: &'static str,
) -> Result<(), EvsiError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_identifier(value, field)?;
        if !seen.insert(value) {
            return Err(EvsiError::DuplicateIdentifier(field));
        }
    }
    Ok(())
}

fn valid_digest(value: &str) -> bool {
    value.len() == DIGEST_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// SHA-256 over the exact compact JSON encoding of the ordered likelihood table.
pub fn likelihood_table_sha256(
    outcomes: &[ObservationOutcomeV1],
) -> Result<String, EvsiError> {
    let encoded =
        serde_json::to_vec(outcomes).map_err(|_| EvsiError::InvalidLikelihoodModel)?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn validate_distribution(values: &[f64], field: &'static str) -> Result<(), EvsiError> {
    if values.is_empty() {
        return Err(EvsiError::EmptyHypothesisSet);
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(EvsiError::InvalidProbability(field));
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(EvsiError::InvalidProbability(field));
    }
    Ok(())
}

fn validate_prior(prior: &PriorBeliefV1) -> Result<(), EvsiError> {
    validate_identifier(&prior.prior_id, "prior_id")?;
    if prior.hypothesis_ids.is_empty() {
        return Err(EvsiError::EmptyHypothesisSet);
    }
    if prior.hypothesis_ids.len() > MAX_HYPOTHESES {
        return Err(EvsiError::TooManyElements("hypotheses"));
    }
    validate_unique(
        prior.hypothesis_ids.iter().map(String::as_str),
        "hypothesis_id",
    )?;
    if prior.probabilities.len() != prior.hypothesis_ids.len() {
        return Err(EvsiError::ProbabilityDimensionMismatch);
    }
    validate_distribution(&prior.probabilities, "prior probabilities")
}

fn validate_decisions(
    decisions: &[DecisionLossV1],
    hypothesis_count: usize,
) -> Result<(), EvsiError> {
    if decisions.is_empty() {
        return Err(EvsiError::EmptyDecisionSet);
    }
    if decisions.len() > MAX_DECISIONS {
        return Err(EvsiError::TooManyElements("decisions"));
    }
    validate_unique(
        decisions.iter().map(|decision| decision.decision_id.as_str()),
        "decision_id",
    )?;
    for decision in decisions {
        if decision.losses_by_hypothesis.len() != hypothesis_count {
            return Err(EvsiError::LossDimensionMismatch);
        }
        if decision
            .losses_by_hypothesis
            .iter()
            .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(EvsiError::InvalidLoss);
        }
    }
    Ok(())
}

fn validate_action(
    action: &MeasurementActionV1,
    hypothesis_count: usize,
) -> Result<(), EvsiError> {
    validate_identifier(&action.action_id, "action_id")?;
    if !valid_digest(&action.likelihood_table_sha256) {
        return Err(EvsiError::InvalidDigest);
    }
    if !action.cost.is_finite() || action.cost <= 0.0 {
        return Err(EvsiError::InvalidCost);
    }
    if action.outcomes.is_empty() {
        return Err(EvsiError::EmptyActionOutcomeSet);
    }
    if action.outcomes.len() > MAX_OUTCOMES_PER_ACTION {
        return Err(EvsiError::TooManyElements("action outcomes"));
    }
    validate_unique(
        action.outcomes.iter().map(|outcome| outcome.outcome_id.as_str()),
        "outcome_id",
    )?;

    let mut column_totals = vec![0.0f64; hypothesis_count];
    for outcome in &action.outcomes {
        if outcome.likelihood_by_hypothesis.len() != hypothesis_count {
            return Err(EvsiError::LikelihoodDimensionMismatch);
        }
        for (index, likelihood) in outcome.likelihood_by_hypothesis.iter().enumerate() {
            if !likelihood.is_finite() || !(0.0..=1.0).contains(likelihood) {
                return Err(EvsiError::InvalidProbability("outcome likelihoods"));
            }
            column_totals[index] += likelihood;
        }
    }
    if column_totals.iter().any(|total| {
        !total.is_finite() || (*total - 1.0).abs() > PROBABILITY_TOLERANCE
    }) {
        return Err(EvsiError::InvalidLikelihoodModel);
    }
    if action.likelihood_table_sha256 != likelihood_table_sha256(&action.outcomes)? {
        return Err(EvsiError::LikelihoodTableDigestMismatch);
    }
    Ok(())
}

fn bayes_risk(
    probabilities: &[f64],
    decisions: &[DecisionLossV1],
) -> Result<f64, EvsiError> {
    let minimum = decisions
        .iter()
        .map(|decision| {
            decision
                .losses_by_hypothesis
                .iter()
                .zip(probabilities.iter())
                .map(|(loss, probability)| loss * probability)
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    if !minimum.is_finite() {
        return Err(EvsiError::NumericalOverflow);
    }
    Ok(minimum)
}

fn estimate_action(
    prior: &PriorBeliefV1,
    decisions: &[DecisionLossV1],
    action: &MeasurementActionV1,
    current_risk: f64,
    policy: PolicyConfigV1,
) -> Result<ActionEstimateV1, EvsiError> {
    let hypothesis_count = prior.hypothesis_ids.len();
    let mut expected_posterior_risk = 0.0f64;
    let mut marginal_total = 0.0f64;
    let mut outcomes = Vec::with_capacity(action.outcomes.len());

    for outcome in &action.outcomes {
        let marginal_probability: f64 = prior
            .probabilities
            .iter()
            .zip(outcome.likelihood_by_hypothesis.iter())
            .map(|(prior_probability, likelihood)| prior_probability * likelihood)
            .sum();

        if !marginal_probability.is_finite() {
            return Err(EvsiError::NumericalOverflow);
        }
        marginal_total += marginal_probability;

        let posterior_bayes_risk = if marginal_probability <= 0.0 {
            None
        } else {
            let posterior: Vec<f64> = prior
                .probabilities
                .iter()
                .zip(outcome.likelihood_by_hypothesis.iter())
                .map(|(prior_probability, likelihood)| {
                    prior_probability * likelihood / marginal_probability
                })
                .collect();
            let risk = bayes_risk(&posterior, decisions)?;
            expected_posterior_risk += marginal_probability * risk;
            Some(risk)
        };

        outcomes.push(OutcomeValueV1 {
            outcome_id: outcome.outcome_id.clone(),
            marginal_probability,
            posterior_bayes_risk,
        });
    }

    if !marginal_total.is_finite()
        || (marginal_total - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err(EvsiError::InvalidLikelihoodModel);
    }
    if !expected_posterior_risk.is_finite() {
        return Err(EvsiError::NumericalOverflow);
    }

    let raw_evsi = current_risk - expected_posterior_risk;
    if raw_evsi < -NUMERICAL_TOLERANCE {
        return Err(EvsiError::NegativeEvsiBeyondTolerance);
    }
    let evsi = raw_evsi.max(0.0);
    let evsi_per_cost = evsi / action.cost;
    let net_expected_value = evsi - policy.cost_to_risk_weight * action.cost;
    if !evsi_per_cost.is_finite() || !net_expected_value.is_finite() {
        return Err(EvsiError::NumericalOverflow);
    }

    Ok(ActionEstimateV1 {
        action_id: action.action_id.clone(),
        likelihood_table_sha256: action.likelihood_table_sha256.clone(),
        admissible: action.admissible,
        within_budget: action.cost <= policy.resource_budget,
        current_bayes_risk: current_risk,
        expected_posterior_bayes_risk: expected_posterior_risk,
        evsi,
        cost: action.cost,
        evsi_per_cost,
        net_expected_value,
        outcomes,
    })
}

/// Select the next measurement by expected reduction in Bayes decision risk.
///
/// This is a one-step myopic policy. Likelihood models are treated as supplied
/// and externally calibrated; this function validates their probability
/// contract but does not calibrate them. Actions outside the declared domain
/// are never selected.
pub fn select_next_measurement(
    request: &ActiveAnalysisRequestV1,
) -> Result<ActiveAnalysisReportV1, EvsiError> {
    validate_prior(&request.prior)?;
    validate_decisions(&request.decisions, request.prior.hypothesis_ids.len())?;
    if request.actions.len() > MAX_ACTIONS {
        return Err(EvsiError::TooManyElements("measurement actions"));
    }
    validate_unique(
        request.actions.iter().map(|action| action.action_id.as_str()),
        "action_id",
    )?;
    for action in &request.actions {
        validate_action(action, request.prior.hypothesis_ids.len())?;
    }

    if !request.policy.resource_budget.is_finite() || request.policy.resource_budget < 0.0 {
        return Err(EvsiError::InvalidBudget);
    }
    if !request.policy.cost_to_risk_weight.is_finite()
        || request.policy.cost_to_risk_weight < 0.0
    {
        return Err(EvsiError::InvalidCostWeight);
    }

    let current_risk = bayes_risk(&request.prior.probabilities, &request.decisions)?;
    let mut estimates = Vec::with_capacity(request.actions.len());

    for action in &request.actions {
        estimates.push(estimate_action(
            &request.prior,
            &request.decisions,
            action,
            current_risk,
            request.policy,
        )?);
    }

    estimates.sort_by(|left, right| {
        right
            .net_expected_value
            .total_cmp(&left.net_expected_value)
            .then_with(|| right.evsi_per_cost.total_cmp(&left.evsi_per_cost))
            .then_with(|| left.action_id.cmp(&right.action_id))
    });

    let no_action_reason = if !request.policy.in_domain {
        Some(NoActionReason::OutOfDomain)
    } else if request.actions.is_empty() {
        Some(NoActionReason::NoCandidates)
    } else if !request.actions.iter().any(|action| action.admissible) {
        Some(NoActionReason::NoAdmissibleActions)
    } else if !estimates.iter().any(|estimate| {
        estimate.admissible && estimate.within_budget
    }) {
        Some(NoActionReason::BudgetExhausted)
    } else if !estimates.iter().any(|estimate| {
        estimate.admissible
            && estimate.within_budget
            && estimate.net_expected_value > NUMERICAL_TOLERANCE
    }) {
        Some(NoActionReason::NoPositiveNetValue)
    } else {
        None
    };

    let chosen_action_id = if no_action_reason.is_none() {
        estimates
            .iter()
            .find(|estimate| {
                estimate.admissible
                    && estimate.within_budget
                    && estimate.net_expected_value > NUMERICAL_TOLERANCE
            })
            .map(|estimate| estimate.action_id.clone())
    } else {
        None
    };

    Ok(ActiveAnalysisReportV1 {
        schema_version: "mel-hear-014-active-analysis-report-v0.1".to_owned(),
        current_bayes_risk: current_risk,
        resource_budget: request.policy.resource_budget,
        cost_to_risk_weight: request.policy.cost_to_risk_weight,
        chosen_action_id,
        no_action_reason,
        estimates,
    })
}

/// Update a prior after one actually observed outcome.
///
/// Zero-probability observations are rejected; the function never fabricates a
/// posterior or silently applies an epsilon floor.
pub fn posterior_after_observation(
    prior: &PriorBeliefV1,
    action: &MeasurementActionV1,
    outcome_id: &str,
) -> Result<Vec<f64>, EvsiError> {
    validate_prior(prior)?;
    validate_action(action, prior.hypothesis_ids.len())?;
    if !action.admissible {
        return Err(EvsiError::ActionNotAdmissible);
    }
    let outcome = action
        .outcomes
        .iter()
        .find(|outcome| outcome.outcome_id == outcome_id)
        .ok_or(EvsiError::UnknownOutcome)?;
    let marginal_probability: f64 = prior
        .probabilities
        .iter()
        .zip(outcome.likelihood_by_hypothesis.iter())
        .map(|(prior_probability, likelihood)| prior_probability * likelihood)
        .sum();
    if !marginal_probability.is_finite() {
        return Err(EvsiError::NumericalOverflow);
    }
    if marginal_probability <= 0.0 {
        return Err(EvsiError::ZeroProbabilityOutcome);
    }

    let posterior: Vec<f64> = prior
        .probabilities
        .iter()
        .zip(outcome.likelihood_by_hypothesis.iter())
        .map(|(prior_probability, likelihood)| {
            prior_probability * likelihood / marginal_probability
        })
        .collect();
    validate_distribution(&posterior, "posterior probabilities")?;
    Ok(posterior)
}

#[cfg(test)]
mod tests {
    use super::{
        posterior_after_observation, select_next_measurement, ActiveAnalysisRequestV1,
        DecisionLossV1, EvsiError, MeasurementActionV1, NoActionReason,
        ObservationOutcomeV1, PolicyConfigV1, PriorBeliefV1,
    };

    fn prior() -> PriorBeliefV1 {
        PriorBeliefV1 {
            prior_id: "two-cause-equal-prior".to_owned(),
            hypothesis_ids: vec![
                "musical-change".to_owned(),
                "room-change".to_owned(),
            ],
            probabilities: vec![0.5, 0.5],
        }
    }

    fn decisions() -> Vec<DecisionLossV1> {
        vec![
            DecisionLossV1 {
                decision_id: "report-musical-change".to_owned(),
                losses_by_hypothesis: vec![0.0, 1.0],
            },
            DecisionLossV1 {
                decision_id: "report-room-change".to_owned(),
                losses_by_hypothesis: vec![1.0, 0.0],
            },
        ]
    }

    fn action(
        action_id: &str,
        cost: f64,
        admissible: bool,
        likelihoods: [[f64; 2]; 2],
    ) -> MeasurementActionV1 {
        let outcomes = vec![
            ObservationOutcomeV1 {
                outcome_id: "supports-musical-change".to_owned(),
                likelihood_by_hypothesis: likelihoods[0].to_vec(),
            },
            ObservationOutcomeV1 {
                outcome_id: "supports-room-change".to_owned(),
                likelihood_by_hypothesis: likelihoods[1].to_vec(),
            },
        ];
        MeasurementActionV1 {
            action_id: action_id.to_owned(),
            likelihood_table_sha256: super::likelihood_table_sha256(&outcomes).unwrap(),
            cost,
            admissible,
            outcomes,
        }
    }

    fn request(actions: Vec<MeasurementActionV1>) -> ActiveAnalysisRequestV1 {
        ActiveAnalysisRequestV1 {
            prior: prior(),
            decisions: decisions(),
            actions,
            policy: PolicyConfigV1 {
                resource_budget: 1.0,
                cost_to_risk_weight: 0.2,
                in_domain: true,
            },
        }
    }

    #[test]
    fn cost_aware_evsi_prefers_more_efficient_measurement() {
        let weak = action(
            "longer-context-window",
            0.1,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let perfect = action(
            "expensive-stereo-source-probe",
            0.8,
            true,
            [[1.0, 0.0], [0.0, 1.0]],
        );
        let report = select_next_measurement(&request(vec![perfect, weak])).unwrap();

        assert_eq!(
            report.chosen_action_id.as_deref(),
            Some("longer-context-window")
        );
        let selected = report.estimates.iter().find(|item| {
            item.action_id == "longer-context-window"
        }).unwrap();
        assert!((selected.current_bayes_risk - 0.5).abs() < 1.0e-12);
        assert!((selected.expected_posterior_bayes_risk - 0.1).abs() < 1.0e-12);
        assert!((selected.evsi - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn posterior_is_updated_only_from_a_named_observed_outcome() {
        let prior = prior();
        let action = action(
            "context-view",
            0.2,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let posterior = posterior_after_observation(
            &prior,
            &action,
            "supports-musical-change",
        ).unwrap();
        assert!((posterior[0] - 0.9).abs() < 1.0e-12);
        assert!((posterior[1] - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn zero_information_action_has_no_positive_net_value() {
        let no_information = action(
            "redundant-view",
            0.1,
            true,
            [[0.5, 0.5], [0.5, 0.5]],
        );
        let report = select_next_measurement(&request(vec![no_information])).unwrap();
        assert_eq!(report.chosen_action_id, None);
        assert_eq!(
            report.no_action_reason,
            Some(NoActionReason::NoPositiveNetValue)
        );
    }

    #[test]
    fn out_of_domain_context_never_selects_a_likelihood_based_action() {
        let mut request = request(vec![action(
            "high-resolution-cqt",
            0.1,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        )]);
        request.policy.in_domain = false;
        let report = select_next_measurement(&request).unwrap();
        assert_eq!(report.chosen_action_id, None);
        assert_eq!(
            report.no_action_reason,
            Some(NoActionReason::OutOfDomain)
        );
    }

    #[test]
    fn resource_budget_blocks_expensive_measurement() {
        let mut request = request(vec![action(
            "expensive-source-separation",
            1.1,
            true,
            [[1.0, 0.0], [0.0, 1.0]],
        )]);
        request.policy.resource_budget = 1.0;
        let report = select_next_measurement(&request).unwrap();
        assert_eq!(report.chosen_action_id, None);
        assert_eq!(
            report.no_action_reason,
            Some(NoActionReason::BudgetExhausted)
        );
    }

    #[test]
    fn invalid_prior_normalization_rejects() {
        let mut request = request(vec![]);
        request.prior.probabilities = vec![0.4, 0.4];
        assert_eq!(
            select_next_measurement(&request),
            Err(EvsiError::InvalidProbability("prior probabilities"))
        );
    }

    #[test]
    fn incomplete_likelihood_distribution_rejects() {
        let mut invalid = action(
            "bad-calibrator",
            0.2,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        invalid.outcomes[1].likelihood_by_hypothesis[0] = 0.2;
        assert_eq!(
            select_next_measurement(&request(vec![invalid])),
            Err(EvsiError::InvalidLikelihoodModel)
        );
    }

    #[test]
    fn duplicate_outcome_identifiers_reject() {
        let mut invalid = action(
            "duplicate-outcome",
            0.2,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        invalid.outcomes[1].outcome_id = invalid.outcomes[0].outcome_id.clone();
        assert_eq!(
            select_next_measurement(&request(vec![invalid])),
            Err(EvsiError::DuplicateIdentifier("outcome_id"))
        );
    }

    #[test]
    fn probability_table_substitution_rejects_without_a_new_digest() {
        let mut altered = action(
            "altered-likelihood-table",
            0.1,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        altered.outcomes[0].likelihood_by_hypothesis = vec![0.8, 0.2];
        altered.outcomes[1].likelihood_by_hypothesis = vec![0.2, 0.8];
        assert_eq!(
            select_next_measurement(&request(vec![altered])),
            Err(EvsiError::LikelihoodTableDigestMismatch)
        );
    }

    #[test]
    fn zero_probability_outcome_does_not_create_a_posterior() {
        let prior = PriorBeliefV1 {
            prior_id: "certain-hypothesis".to_owned(),
            hypothesis_ids: vec!["a".to_owned(), "b".to_owned()],
            probabilities: vec![1.0, 0.0],
        };
        let action = action(
            "partition-test",
            0.1,
            true,
            [[1.0, 0.0], [0.0, 1.0]],
        );
        assert_eq!(
            posterior_after_observation(
                &prior,
                &action,
                "supports-room-change",
            ),
            Err(EvsiError::ZeroProbabilityOutcome)
        );
    }

    #[test]
    fn committed_active_analysis_fixture_parses_and_selects_cost_efficient_measurement() {
        let fixture: super::ActiveAnalysisRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/active-analysis-request-v0.1.json"
        ))
        .unwrap();
        let report = select_next_measurement(&fixture).unwrap();
        assert_eq!(
            report.chosen_action_id.as_deref(),
            Some("longer-context-window")
        );
    }

    #[test]
    fn ties_are_broken_by_action_identifier_not_input_order() {
        let alpha = action(
            "alpha",
            0.1,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let beta = action(
            "beta",
            0.1,
            true,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let report = select_next_measurement(&request(vec![beta, alpha])).unwrap();
        assert_eq!(report.chosen_action_id.as_deref(), Some("alpha"));
    }
}
