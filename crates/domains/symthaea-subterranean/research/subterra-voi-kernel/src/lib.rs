//! Decision-focused measurement selection for Subterra research.
//!
//! This is an offline/reference planner, not an actuator or mission authority.
//! It computes one-step expected value of sample information (EVSI) over a
//! finite, explicitly open-world hypothesis set. Candidate measurements must
//! already have passed an external admissibility/ODD/preservation gate.
//!
//! Probabilities and losses are supplied by the caller. This crate does not
//! calibrate them, authenticate evidence, model correlated joint measurements,
//! plan robot trajectories, or validate physical safety.

use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const NUMERICAL_TOLERANCE: f64 = 1.0e-8;
const UNKNOWN_HYPOTHESIS_ID: &str = "unknown";

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Stable hypothesis identity. Exactly one hypothesis must use "unknown".
    pub id: String,
    /// Prior probability. The unknown hypothesis must have non-zero mass.
    pub prior_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionDecision {
    pub id: String,
    /// Non-negative loss for each hypothesis, in the same declared utility
    /// units used by the measurement cost shadow prices.
    /// The order must exactly match PlanningProblem::hypotheses.
    pub loss_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationOutcome {
    pub id: String,
    /// P(outcome | hypothesis, action), indexed in hypothesis order.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResourceCost {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    /// A caller-defined exposure budget quantity. It is not a safety score.
    pub exposure_units: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAction {
    pub id: String,
    pub modality: String,
    pub outcomes: Vec<ObservationOutcome>,
    pub cost: ResourceCost,
    /// Must be supplied by a separate reviewed gate. This boolean is not
    /// proof of authority and is never produced by this planner.
    pub external_gate_passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResourceBudget {
    pub energy_j: f64,
    pub duration_s: f64,
    pub data_bytes: f64,
    pub exposure_units: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CostPrices {
    /// Utility loss units per joule.
    pub energy_per_j: f64,
    /// Utility loss units per second.
    pub time_per_s: f64,
    /// Utility loss units per byte.
    pub data_per_byte: f64,
    /// Utility loss units per exposure unit.
    pub exposure_per_unit: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningProblem {
    pub hypotheses: Vec<Hypothesis>,
    pub decisions: Vec<MissionDecision>,
    pub actions: Vec<MeasurementAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    Eligible,
    ExternalGateDenied,
    OverBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionAssessment {
    pub action_id: String,
    pub modality: String,
    pub status: ActionStatus,
    /// Entropy reduction in nats; diagnostic only, not the primary objective.
    pub expected_information_gain_nats: Option<f64>,
    pub expected_post_decision_loss: Option<f64>,
    pub evsi: Option<f64>,
    pub weighted_cost: Option<f64>,
    pub net_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningReport {
    pub baseline_expected_loss: f64,
    /// Selected measurement proposal. None means abstain.
    pub selected_action_id: Option<String>,
    /// Sorted by descending net value; stable lexical tie-break on action id.
    pub assessments: Vec<ActionAssessment>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanningError {
    EmptyHypotheses,
    MissingUnknownHypothesis,
    MultipleUnknownHypotheses,
    UnknownHypothesisHasZeroPrior,
    DuplicateHypothesisId(String),
    DuplicateDecisionId(String),
    DuplicateActionId(String),
    DuplicateOutcomeId { action_id: String, outcome_id: String },
    EmptyIdentifier(&'static str),
    InvalidPrior(String),
    PriorNotNormalized(f64),
    NoMissionDecisions,
    InvalidDecisionLoss(String),
    DecisionDimensionMismatch(String),
    NoObservationOutcomes(String),
    InvalidLikelihood { action_id: String, outcome_id: String },
    LikelihoodDimensionMismatch { action_id: String, outcome_id: String },
    LikelihoodsNotNormalized { action_id: String, hypothesis_id: String, sum: f64 },
    InvalidResourceValue(&'static str),
    InvalidCostPrice(&'static str),
    InvalidMinimumNetValue,
    NumericalOverflow(&'static str),
    NonMonotonicEvsi { baseline: f64, after: f64 },
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlanningError {}

fn require_identifier(value: &str, field: &'static str) -> Result<(), PlanningError> {
    if value.trim().is_empty() {
        Err(PlanningError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn validate_nonnegative_finite(value: f64, field: &'static str) -> Result<(), PlanningError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PlanningError::InvalidResourceValue(field))
    }
}

impl ResourceCost {
    fn validate(self) -> Result<(), PlanningError> {
        validate_nonnegative_finite(self.energy_j, "action.energy_j")?;
        validate_nonnegative_finite(self.duration_s, "action.duration_s")?;
        validate_nonnegative_finite(self.data_bytes, "action.data_bytes")?;
        validate_nonnegative_finite(self.exposure_units, "action.exposure_units")
    }

    fn fits(self, budget: ResourceBudget) -> bool {
        self.energy_j <= budget.energy_j + PROBABILITY_TOLERANCE
            && self.duration_s <= budget.duration_s + PROBABILITY_TOLERANCE
            && self.data_bytes <= budget.data_bytes + PROBABILITY_TOLERANCE
            && self.exposure_units <= budget.exposure_units + PROBABILITY_TOLERANCE
    }

    fn weighted(self, prices: CostPrices) -> Result<f64, PlanningError> {
        let total = self.energy_j * prices.energy_per_j
            + self.duration_s * prices.time_per_s
            + self.data_bytes * prices.data_per_byte
            + self.exposure_units * prices.exposure_per_unit;
        if total.is_finite() {
            Ok(total)
        } else {
            Err(PlanningError::NumericalOverflow("weighted resource cost"))
        }
    }
}

impl ResourceBudget {
    fn validate(self) -> Result<(), PlanningError> {
        validate_nonnegative_finite(self.energy_j, "budget.energy_j")?;
        validate_nonnegative_finite(self.duration_s, "budget.duration_s")?;
        validate_nonnegative_finite(self.data_bytes, "budget.data_bytes")?;
        validate_nonnegative_finite(self.exposure_units, "budget.exposure_units")
    }
}

impl CostPrices {
    fn validate(self) -> Result<(), PlanningError> {
        for (value, field) in [
            (self.energy_per_j, "prices.energy_per_j"),
            (self.time_per_s, "prices.time_per_s"),
            (self.data_per_byte, "prices.data_per_byte"),
            (self.exposure_per_unit, "prices.exposure_per_unit"),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PlanningError::InvalidCostPrice(field));
            }
        }
        Ok(())
    }
}

impl PlanningProblem {
    /// Validate dimensions, the open-world hypothesis, probabilities, IDs,
    /// decision losses, outcome likelihoods and candidate resource costs.
    pub fn validate(&self) -> Result<(), PlanningError> {
        if self.hypotheses.is_empty() {
            return Err(PlanningError::EmptyHypotheses);
        }

        let mut hypothesis_ids = BTreeSet::new();
        let mut prior_sum = 0.0;
        let mut unknown_count = 0usize;
        let mut unknown_prior = 0.0;

        for hypothesis in &self.hypotheses {
            require_identifier(&hypothesis.id, "hypothesis.id")?;
            if !hypothesis_ids.insert(hypothesis.id.as_str()) {
                return Err(PlanningError::DuplicateHypothesisId(hypothesis.id.clone()));
            }
            if !hypothesis.prior_probability.is_finite()
                || !(0.0..=1.0).contains(&hypothesis.prior_probability)
            {
                return Err(PlanningError::InvalidPrior(hypothesis.id.clone()));
            }
            prior_sum += hypothesis.prior_probability;
            if !prior_sum.is_finite() {
                return Err(PlanningError::NumericalOverflow("prior probability sum"));
            }
            if hypothesis.id == UNKNOWN_HYPOTHESIS_ID {
                unknown_count += 1;
                unknown_prior = hypothesis.prior_probability;
            }
        }

        if unknown_count == 0 {
            return Err(PlanningError::MissingUnknownHypothesis);
        }
        if unknown_count > 1 {
            return Err(PlanningError::MultipleUnknownHypotheses);
        }
        if unknown_prior <= 0.0 {
            return Err(PlanningError::UnknownHypothesisHasZeroPrior);
        }
        if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(PlanningError::PriorNotNormalized(prior_sum));
        }
        if self.decisions.is_empty() {
            return Err(PlanningError::NoMissionDecisions);
        }

        let hypothesis_count = self.hypotheses.len();
        let mut decision_ids = BTreeSet::new();
        for decision in &self.decisions {
            require_identifier(&decision.id, "decision.id")?;
            if !decision_ids.insert(decision.id.as_str()) {
                return Err(PlanningError::DuplicateDecisionId(decision.id.clone()));
            }
            if decision.loss_by_hypothesis.len() != hypothesis_count {
                return Err(PlanningError::DecisionDimensionMismatch(decision.id.clone()));
            }
            if decision
                .loss_by_hypothesis
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(PlanningError::InvalidDecisionLoss(decision.id.clone()));
            }
        }

        let mut action_ids = BTreeSet::new();
        for action in &self.actions {
            require_identifier(&action.id, "action.id")?;
            require_identifier(&action.modality, "action.modality")?;
            if !action_ids.insert(action.id.as_str()) {
                return Err(PlanningError::DuplicateActionId(action.id.clone()));
            }
            action.cost.validate()?;
            if action.outcomes.is_empty() {
                return Err(PlanningError::NoObservationOutcomes(action.id.clone()));
            }

            let mut outcome_ids = BTreeSet::new();
            for outcome in &action.outcomes {
                require_identifier(&outcome.id, "outcome.id")?;
                if !outcome_ids.insert(outcome.id.as_str()) {
                    return Err(PlanningError::DuplicateOutcomeId {
                        action_id: action.id.clone(),
                        outcome_id: outcome.id.clone(),
                    });
                }
                if outcome.likelihood_by_hypothesis.len() != hypothesis_count {
                    return Err(PlanningError::LikelihoodDimensionMismatch {
                        action_id: action.id.clone(),
                        outcome_id: outcome.id.clone(),
                    });
                }
                if outcome
                    .likelihood_by_hypothesis
                    .iter()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                {
                    return Err(PlanningError::InvalidLikelihood {
                        action_id: action.id.clone(),
                        outcome_id: outcome.id.clone(),
                    });
                }
            }

            // For every hidden hypothesis, all mutually exclusive outcomes
            // must form a complete conditional distribution.
            for (hypothesis_index, hypothesis) in self.hypotheses.iter().enumerate() {
                let sum = action
                    .outcomes
                    .iter()
                    .map(|outcome| outcome.likelihood_by_hypothesis[hypothesis_index])
                    .sum::<f64>();
                if !sum.is_finite() {
                    return Err(PlanningError::NumericalOverflow("outcome likelihood sum"));
                }
                if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                    return Err(PlanningError::LikelihoodsNotNormalized {
                        action_id: action.id.clone(),
                        hypothesis_id: hypothesis.id.clone(),
                        sum,
                    });
                }
            }
        }

        Ok(())
    }
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn minimum_expected_loss(probabilities: &[f64], decisions: &[MissionDecision]) -> f64 {
    decisions
        .iter()
        .map(|decision| {
            probabilities
                .iter()
                .zip(&decision.loss_by_hypothesis)
                .map(|(probability, loss)| probability * loss)
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min)
}

fn evaluate_action(
    action: &MeasurementAction,
    hypotheses: &[Hypothesis],
    decisions: &[MissionDecision],
) -> Result<(f64, f64), PlanningError> {
    let prior = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();
    let baseline_loss = minimum_expected_loss(&prior, decisions);
    let prior_entropy = entropy(&prior);

    let mut expected_post_loss = 0.0;
    let mut expected_post_entropy = 0.0;

    for outcome in &action.outcomes {
        let outcome_probability = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(prior_probability, likelihood)| prior_probability * likelihood)
            .sum::<f64>();

        if !outcome_probability.is_finite() {
            return Err(PlanningError::NumericalOverflow("outcome probability"));
        }
        if outcome_probability <= 0.0 {
            continue;
        }

        let posterior = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(prior_probability, likelihood)| {
                prior_probability * likelihood / outcome_probability
            })
            .collect::<Vec<_>>();

        let posterior_sum = posterior.iter().sum::<f64>();
        if !posterior_sum.is_finite()
            || (posterior_sum - 1.0).abs() > NUMERICAL_TOLERANCE
        {
            return Err(PlanningError::NumericalOverflow("posterior normalization"));
        }

        expected_post_loss +=
            outcome_probability * minimum_expected_loss(&posterior, decisions);
        expected_post_entropy += outcome_probability * entropy(&posterior);
    }

    if !expected_post_loss.is_finite() || !expected_post_entropy.is_finite() {
        return Err(PlanningError::NumericalOverflow("expected posterior score"));
    }
    if expected_post_loss > baseline_loss + NUMERICAL_TOLERANCE {
        return Err(PlanningError::NonMonotonicEvsi {
            baseline: baseline_loss,
            after: expected_post_loss,
        });
    }

    let evsi = (baseline_loss - expected_post_loss).max(0.0);
    let information_gain = (prior_entropy - expected_post_entropy).max(0.0);

    if !evsi.is_finite() || !information_gain.is_finite() {
        return Err(PlanningError::NumericalOverflow("EVSI or information gain"));
    }

    Ok((evsi, information_gain))
}

/// Rank a finite set of measurement proposals by expected decision value.
///
/// The function will not recommend an action unless:
/// - its separate external gate has passed;
/// - its resource cost fits the declared hard budget; and
/// - net EVSI exceeds the caller's minimum net value.
///
/// Cost prices must use the same utility units as decision losses. Exposure is
/// both budget-constrained and costed, but safety/planetary-protection checks
/// must remain hard external gates and must never be reduced to this price.
pub fn plan_measurements(
    problem: &PlanningProblem,
    budget: ResourceBudget,
    prices: CostPrices,
    minimum_net_value: f64,
) -> Result<PlanningReport, PlanningError> {
    problem.validate()?;
    budget.validate()?;
    prices.validate()?;
    if !minimum_net_value.is_finite() || minimum_net_value < 0.0 {
        return Err(PlanningError::InvalidMinimumNetValue);
    }

    let prior = problem
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect::<Vec<_>>();
    let baseline_expected_loss = minimum_expected_loss(&prior, &problem.decisions);
    if !baseline_expected_loss.is_finite() {
        return Err(PlanningError::NumericalOverflow("baseline expected loss"));
    }

    let mut assessments = Vec::with_capacity(problem.actions.len());
    for action in &problem.actions {
        if !action.external_gate_passed {
            assessments.push(ActionAssessment {
                action_id: action.id.clone(),
                modality: action.modality.clone(),
                status: ActionStatus::ExternalGateDenied,
                expected_information_gain_nats: None,
                expected_post_decision_loss: None,
                evsi: None,
                weighted_cost: None,
                net_value: None,
            });
            continue;
        }

        if !action.cost.fits(budget) {
            assessments.push(ActionAssessment {
                action_id: action.id.clone(),
                modality: action.modality.clone(),
                status: ActionStatus::OverBudget,
                expected_information_gain_nats: None,
                expected_post_decision_loss: None,
                evsi: None,
                weighted_cost: None,
                net_value: None,
            });
            continue;
        }

        let (evsi, information_gain) =
            evaluate_action(action, &problem.hypotheses, &problem.decisions)?;
        let weighted_cost = action.cost.weighted(prices)?;
        let net_value = evsi - weighted_cost;
        if !net_value.is_finite() {
            return Err(PlanningError::NumericalOverflow("net measurement value"));
        }

        assessments.push(ActionAssessment {
            action_id: action.id.clone(),
            modality: action.modality.clone(),
            status: ActionStatus::Eligible,
            expected_information_gain_nats: Some(information_gain),
            expected_post_decision_loss: Some((baseline_expected_loss - evsi).max(0.0)),
            evsi: Some(evsi),
            weighted_cost: Some(weighted_cost),
            net_value: Some(net_value),
        });
    }

    assessments.sort_by(|left, right| match (left.net_value, right.net_value) {
        (Some(left_value), Some(right_value)) => right_value
            .total_cmp(&left_value)
            .then_with(|| left.action_id.cmp(&right.action_id)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.action_id.cmp(&right.action_id),
    });

    let selected_action_id = assessments
        .iter()
        .find(|assessment| {
            assessment.status == ActionStatus::Eligible
                && assessment
                    .net_value
                    .is_some_and(|value| value > minimum_net_value)
        })
        .map(|assessment| assessment.action_id.clone());

    Ok(PlanningReport {
        baseline_expected_loss,
        selected_action_id,
        assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "open_continuation".into(),
                prior_probability: 0.45,
            },
            Hypothesis {
                id: "blocked_dead_end".into(),
                prior_probability: 0.45,
            },
            Hypothesis {
                id: UNKNOWN_HYPOTHESIS_ID.into(),
                prior_probability: 0.10,
            },
        ]
    }

    fn decisions() -> Vec<MissionDecision> {
        vec![
            MissionDecision {
                id: "enter".into(),
                loss_by_hypothesis: vec![0.0, 100.0, 50.0],
            },
            MissionDecision {
                id: "hold".into(),
                loss_by_hypothesis: vec![30.0, 0.0, 10.0],
            },
        ]
    }

    fn informative_scan(id: &str) -> MeasurementAction {
        MeasurementAction {
            id: id.into(),
            modality: "geophysical_scan".into(),
            outcomes: vec![
                ObservationOutcome {
                    id: "continuation_signal".into(),
                    likelihood_by_hypothesis: vec![0.9, 0.1, 0.5],
                },
                ObservationOutcome {
                    id: "no_continuation_signal".into(),
                    likelihood_by_hypothesis: vec![0.1, 0.9, 0.5],
                },
            ],
            cost: ResourceCost {
                energy_j: 1_000.0,
                duration_s: 10.0,
                data_bytes: 2_000.0,
                exposure_units: 0.2,
            },
            external_gate_passed: true,
        }
    }

    fn problem(actions: Vec<MeasurementAction>) -> PlanningProblem {
        PlanningProblem {
            hypotheses: hypotheses(),
            decisions: decisions(),
            actions,
        }
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            energy_j: 2_000.0,
            duration_s: 30.0,
            data_bytes: 10_000.0,
            exposure_units: 1.0,
        }
    }

    fn prices() -> CostPrices {
        CostPrices {
            energy_per_j: 0.001,
            time_per_s: 0.01,
            data_per_byte: 0.00001,
            exposure_per_unit: 0.1,
        }
    }

    #[test]
    fn decision_relevant_scan_has_positive_evsi() {
        let p = problem(vec![informative_scan("gpr")]);
        let report = plan_measurements(&p, budget(), prices(), 0.0).unwrap();
        assert!((report.baseline_expected_loss - 14.5).abs() < 1e-9);
        let assessment = &report.assessments[0];
        assert!((assessment.evsi.unwrap() - 5.65).abs() < 1e-8);
        assert_eq!(report.selected_action_id.as_deref(), Some("gpr"));
    }

    #[test]
    fn entropy_free_measurement_is_not_selected_for_decision_value() {
        let mut uninformative = informative_scan("chemistry");
        uninformative.modality = "spectroscopy".into();
        for outcome in &mut uninformative.outcomes {
            outcome.likelihood_by_hypothesis = vec![0.5, 0.5, 0.5];
        }
        uninformative.cost = ResourceCost::default();

        let report = plan_measurements(
            &problem(vec![uninformative]),
            budget(),
            CostPrices::default(),
            0.0,
        )
        .unwrap();

        assert!(report.assessments[0].evsi.unwrap().abs() < 1e-9);
        assert!(report.selected_action_id.is_none());
    }

    #[test]
    fn unknown_hypothesis_is_mandatory_and_must_have_mass() {
        let mut p = problem(vec![informative_scan("gpr")]);
        p.hypotheses.pop();
        assert_eq!(
            p.validate(),
            Err(PlanningError::MissingUnknownHypothesis)
        );

        let mut p = problem(vec![informative_scan("gpr")]);
        p.hypotheses[2].prior_probability = 0.0;
        assert_eq!(
            p.validate(),
            Err(PlanningError::UnknownHypothesisHasZeroPrior)
        );
    }

    #[test]
    fn malformed_conditional_likelihood_is_rejected() {
        let mut action = informative_scan("gpr");
        action.outcomes[0].likelihood_by_hypothesis[0] = 0.7;
        let p = problem(vec![action]);
        assert!(matches!(
            p.validate(),
            Err(PlanningError::LikelihoodsNotNormalized { .. })
        ));
    }

    #[test]
    fn denied_and_over_budget_actions_are_never_selected() {
        let mut denied = informative_scan("denied");
        denied.external_gate_passed = false;

        let mut over_budget = informative_scan("too-expensive");
        over_budget.cost.energy_j = 10_000.0;

        let report = plan_measurements(
            &problem(vec![denied, over_budget]),
            budget(),
            prices(),
            0.0,
        )
        .unwrap();

        assert_eq!(report.selected_action_id, None);
        assert_eq!(report.assessments[0].status, ActionStatus::ExternalGateDenied);
        assert_eq!(report.assessments[1].status, ActionStatus::OverBudget);
    }

    #[test]
    fn deterministic_tie_break_uses_lexical_action_id() {
        let report = plan_measurements(
            &problem(vec![informative_scan("z-gpr"), informative_scan("a-gpr")]),
            budget(),
            prices(),
            0.0,
        )
        .unwrap();
        assert_eq!(report.selected_action_id.as_deref(), Some("a-gpr"));
        assert_eq!(report.assessments[0].action_id, "a-gpr");
    }

    #[test]
    fn action_order_does_not_change_ranked_result() {
        let left = plan_measurements(
            &problem(vec![informative_scan("z-gpr"), informative_scan("a-gpr")]),
            budget(),
            prices(),
            0.0,
        )
        .unwrap();
        let right = plan_measurements(
            &problem(vec![informative_scan("a-gpr"), informative_scan("z-gpr")]),
            budget(),
            prices(),
            0.0,
        )
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn high_resource_price_causes_abstention() {
        let expensive = CostPrices {
            energy_per_j: 1.0,
            time_per_s: 1.0,
            data_per_byte: 1.0,
            exposure_per_unit: 1.0,
        };
        let report = plan_measurements(
            &problem(vec![informative_scan("gpr")]),
            budget(),
            expensive,
            0.0,
        )
        .unwrap();
        assert_eq!(report.selected_action_id, None);
    }

    #[test]
    fn outcome_information_gain_is_reported_separately() {
        let report = plan_measurements(
            &problem(vec![informative_scan("gpr")]),
            budget(),
            CostPrices::default(),
            0.0,
        )
        .unwrap();
        assert!(report.assessments[0]
            .expected_information_gain_nats
            .unwrap()
            > 0.0);
    }

    #[test]
    fn no_measurement_actions_is_a_valid_abstention() {
        let report = plan_measurements(
            &problem(Vec::new()),
            budget(),
            prices(),
            0.0,
        )
        .unwrap();
        assert!(report.selected_action_id.is_none());
        assert!(report.assessments.is_empty());
    }
}
