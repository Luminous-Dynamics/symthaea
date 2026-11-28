//! Exact finite decision-value-of-intervention reference.
//!
//! Evaluator-side mathematics only. Hidden likelihoods, utility tables, exact
//! posteriors, and oracle probe rankings must never be passed to a candidate.
//! Information gain, downstream decision value, and net intervention value
//! are reported separately.

#![forbid(unsafe_code)]

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const POSITIVE_VALUE_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum DecisionProblemError {
    EmptyHypothesisSet,
    EmptyDecisionSet,
    EmptyId { kind: &'static str },
    DuplicateId { kind: &'static str, id: String },
    PriorLengthMismatch { expected: usize, actual: usize },
    PosteriorLengthMismatch { expected: usize, actual: usize },
    UtilityRowCountMismatch { context: String, expected: usize, actual: usize },
    UtilityColumnCountMismatch { context: String, expected: usize, actual: usize },
    InvalidProbability { context: String, value: f64 },
    ProbabilityMassMismatch { context: String, sum: f64 },
    NonFiniteUtility { context: String },
    InvalidInterventionCost { intervention_id: String, cost: f64 },
    EmptyOutcomeSet { intervention_id: String },
    UnknownIntervention { intervention_id: String },
    UnknownOutcome { intervention_id: String, outcome_id: String },
    ImpossibleOutcome { intervention_id: String, outcome_id: String },
}

impl std::fmt::Display for DecisionProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis set must not be empty"),
            Self::EmptyDecisionSet => write!(f, "decision set must not be empty"),
            Self::EmptyId { kind } => write!(f, "{kind} identifier must not be empty"),
            Self::DuplicateId { kind, id } => write!(f, "duplicate {kind} identifier: {id}"),
            Self::PriorLengthMismatch { expected, actual } => write!(
                f, "prior/likelihood length mismatch: expected {expected}, got {actual}"
            ),
            Self::PosteriorLengthMismatch { expected, actual } => write!(
                f, "posterior length mismatch: expected {expected}, got {actual}"
            ),
            Self::UtilityRowCountMismatch { context, expected, actual } => write!(
                f, "{context}: expected {expected} utility rows, got {actual}"
            ),
            Self::UtilityColumnCountMismatch { context, expected, actual } => write!(
                f, "{context}: expected {expected} utility columns, got {actual}"
            ),
            Self::InvalidProbability { context, value } => {
                write!(f, "{context}: invalid probability {value}")
            }
            Self::ProbabilityMassMismatch { context, sum } => {
                write!(f, "{context}: probability mass is {sum}, expected 1")
            }
            Self::NonFiniteUtility { context } => write!(f, "{context}: utility must be finite"),
            Self::InvalidInterventionCost { intervention_id, cost } => {
                write!(f, "intervention {intervention_id} has invalid cost {cost}")
            }
            Self::EmptyOutcomeSet { intervention_id } => {
                write!(f, "intervention {intervention_id} has no outcomes")
            }
            Self::UnknownIntervention { intervention_id } => {
                write!(f, "unknown intervention: {intervention_id}")
            }
            Self::UnknownOutcome { intervention_id, outcome_id } => {
                write!(f, "unknown outcome {outcome_id} for intervention {intervention_id}")
            }
            Self::ImpossibleOutcome { intervention_id, outcome_id } => {
                write!(f, "outcome {outcome_id} is impossible under intervention {intervention_id}")
            }
        }
    }
}

impl std::error::Error for DecisionProblemError {}

/// An observation that may follow an intervention.
///
/// Likelihoods follow the parent problem's hypothesis order. The utility
/// matrix is downstream utility after this outcome, including side effects.
#[derive(Debug, Clone, PartialEq)]
pub struct InterventionOutcome {
    pub id: String,
    pub likelihood_by_hypothesis: Vec<f64>,
    /// Rows are decisions; columns are hidden hypotheses.
    pub post_intervention_utility_by_decision: Vec<Vec<f64>>,
}

/// A bounded experiment or intervention available to the agent.
///
/// Cost uses the same utility units as the decision matrices. It can aggregate
/// action budget, scarce resources, expected damage, or declared compute cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervention {
    pub id: String,
    pub cost: f64,
    pub outcomes: Vec<InterventionOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionValue {
    pub decision_id: String,
    pub expected_utility: f64,
}

/// Exact diagnostic score for one intervention.
///
/// Gross value delta is expected downstream value minus the best current value.
/// Net value subtracts intervention cost. Information gain is a separate
/// diagnostic and is not used as a proxy for decision value.
#[derive(Debug, Clone, PartialEq)]
pub struct InterventionAssessment {
    pub intervention_id: String,
    pub baseline_value: f64,
    pub expected_downstream_value: f64,
    pub gross_value_delta: f64,
    pub cost: f64,
    pub net_value: f64,
    pub information_gain_bits: f64,
}

/// Evaluator-side finite-hypothesis / finite-decision reference model.
///
/// True outcome likelihoods and utility tables are private to this model.
/// Never serialize the entire model into candidate observations.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionProblem {
    hypothesis_ids: Vec<String>,
    decision_ids: Vec<String>,
    prior: Vec<f64>,
    utility_by_decision: Vec<Vec<f64>>,
    interventions: Vec<Intervention>,
}

impl DecisionProblem {
    /// Build and validate the finite decision problem.
    ///
    /// Baseline utility is indexed as decision, then hidden hypothesis.
    pub fn new(
        hypothesis_ids: Vec<String>,
        decision_ids: Vec<String>,
        prior: Vec<f64>,
        utility_by_decision: Vec<Vec<f64>>,
        interventions: Vec<Intervention>,
    ) -> Result<Self, DecisionProblemError> {
        if hypothesis_ids.is_empty() {
            return Err(DecisionProblemError::EmptyHypothesisSet);
        }
        if decision_ids.is_empty() {
            return Err(DecisionProblemError::EmptyDecisionSet);
        }
        validate_unique_ids(&hypothesis_ids, "hypothesis")?;
        validate_unique_ids(&decision_ids, "decision")?;

        if prior.len() != hypothesis_ids.len() {
            return Err(DecisionProblemError::PriorLengthMismatch {
                expected: hypothesis_ids.len(),
                actual: prior.len(),
            });
        }
        validate_distribution(&prior, "prior")?;
        validate_utility_matrix(
            &utility_by_decision,
            decision_ids.len(),
            hypothesis_ids.len(),
            "baseline utility",
        )?;

        let mut intervention_ids: Vec<String> = Vec::with_capacity(interventions.len());
        for intervention in &interventions {
            if intervention.id.trim().is_empty() {
                return Err(DecisionProblemError::EmptyId { kind: "intervention" });
            }
            if intervention_ids.iter().any(|id| id == &intervention.id) {
                return Err(DecisionProblemError::DuplicateId {
                    kind: "intervention",
                    id: intervention.id.clone(),
                });
            }
            intervention_ids.push(intervention.id.clone());

            if !intervention.cost.is_finite() || intervention.cost < 0.0 {
                return Err(DecisionProblemError::InvalidInterventionCost {
                    intervention_id: intervention.id.clone(),
                    cost: intervention.cost,
                });
            }
            if intervention.outcomes.is_empty() {
                return Err(DecisionProblemError::EmptyOutcomeSet {
                    intervention_id: intervention.id.clone(),
                });
            }

            let mut outcome_ids: Vec<String> = Vec::with_capacity(intervention.outcomes.len());
            for outcome in &intervention.outcomes {
                if outcome.id.trim().is_empty() {
                    return Err(DecisionProblemError::EmptyId { kind: "outcome" });
                }
                if outcome_ids.iter().any(|id| id == &outcome.id) {
                    return Err(DecisionProblemError::DuplicateId {
                        kind: "outcome",
                        id: outcome.id.clone(),
                    });
                }
                outcome_ids.push(outcome.id.clone());

                if outcome.likelihood_by_hypothesis.len() != hypothesis_ids.len() {
                    return Err(DecisionProblemError::PriorLengthMismatch {
                        expected: hypothesis_ids.len(),
                        actual: outcome.likelihood_by_hypothesis.len(),
                    });
                }
                for (index, probability) in outcome.likelihood_by_hypothesis.iter().enumerate() {
                    validate_probability(
                        *probability,
                        format!("intervention {} outcome {} hypothesis {}",
                            intervention.id, outcome.id, hypothesis_ids[index]),
                    )?;
                }
                validate_utility_matrix(
                    &outcome.post_intervention_utility_by_decision,
                    decision_ids.len(),
                    hypothesis_ids.len(),
                    &format!("intervention {} outcome {} utility",
                        intervention.id, outcome.id),
                )?;
            }

            // For each true hypothesis, all possible outcome likelihoods sum to 1.
            for (hypothesis_index, hypothesis_id) in hypothesis_ids.iter().enumerate() {
                let mass: f64 = intervention.outcomes.iter()
                    .map(|outcome| outcome.likelihood_by_hypothesis[hypothesis_index])
                    .sum();
                validate_mass(
                    mass,
                    format!("intervention {} under hypothesis {}",
                        intervention.id, hypothesis_id),
                )?;
            }
        }

        Ok(Self {
            hypothesis_ids,
            decision_ids,
            prior,
            utility_by_decision,
            interventions,
        })
    }

    pub fn hypothesis_ids(&self) -> &[String] {
        &self.hypothesis_ids
    }

    pub fn decision_ids(&self) -> &[String] {
        &self.decision_ids
    }

    pub fn prior(&self) -> &[f64] {
        &self.prior
    }

    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }

    /// Choose the best current decision under a normalized posterior.
    /// Equal-valued decisions preserve declaration order.
    pub fn best_current_decision(
        &self,
        posterior: &[f64],
    ) -> Result<DecisionValue, DecisionProblemError> {
        self.best_decision_with_utilities(posterior, &self.utility_by_decision)
    }

    /// Return the exact posterior after observing an intervention outcome.
    pub fn posterior_after_outcome(
        &self,
        posterior: &[f64],
        intervention_id: &str,
        outcome_id: &str,
    ) -> Result<Vec<f64>, DecisionProblemError> {
        validate_posterior(posterior, self.hypothesis_ids.len())?;
        let outcome = self.find_outcome(intervention_id, outcome_id)?;
        bayes_update(
            posterior,
            &outcome.likelihood_by_hypothesis,
            intervention_id,
            outcome_id,
        )
    }

    /// Choose the best downstream decision after a particular outcome.
    pub fn best_decision_after_outcome(
        &self,
        posterior: &[f64],
        intervention_id: &str,
        outcome_id: &str,
    ) -> Result<DecisionValue, DecisionProblemError> {
        let updated = self.posterior_after_outcome(posterior, intervention_id, outcome_id)?;
        let outcome = self.find_outcome(intervention_id, outcome_id)?;
        self.best_decision_with_utilities(
            &updated,
            &outcome.post_intervention_utility_by_decision,
        )
    }

    /// Score an intervention using exact expected downstream decision value.
    ///
    /// For a pure information probe, all outcome utility matrices equal the
    /// baseline matrix. For a state-changing probe, include side effects in
    /// each outcome's post-intervention matrix.
    pub fn assess_intervention(
        &self,
        posterior: &[f64],
        intervention_id: &str,
    ) -> Result<InterventionAssessment, DecisionProblemError> {
        validate_posterior(posterior, self.hypothesis_ids.len())?;
        let intervention = self.find_intervention(intervention_id)?;
        let baseline = self.best_current_decision(posterior)?.expected_utility;
        let mut expected_downstream_value = 0.0;
        let mut expected_posterior_entropy = 0.0;

        for outcome in &intervention.outcomes {
            let outcome_probability = dot(posterior, &outcome.likelihood_by_hypothesis);
            if outcome_probability <= 0.0 {
                continue;
            }
            let updated = bayes_update(
                posterior,
                &outcome.likelihood_by_hypothesis,
                intervention_id,
                &outcome.id,
            )?;
            let downstream = self.best_decision_with_utilities(
                &updated,
                &outcome.post_intervention_utility_by_decision,
            )?.expected_utility;

            expected_downstream_value += outcome_probability * downstream;
            expected_posterior_entropy += outcome_probability * entropy_bits(&updated);
        }

        let gross_value_delta = expected_downstream_value - baseline;
        let net_value = gross_value_delta - intervention.cost;
        let information_gain_bits =
            (entropy_bits(posterior) - expected_posterior_entropy).max(0.0);

        Ok(InterventionAssessment {
            intervention_id: intervention.id.clone(),
            baseline_value: baseline,
            expected_downstream_value,
            gross_value_delta,
            cost: intervention.cost,
            net_value,
            information_gain_bits,
        })
    }

    pub fn assess_all_interventions(
        &self,
        posterior: &[f64],
    ) -> Result<Vec<InterventionAssessment>, DecisionProblemError> {
        self.interventions.iter()
            .map(|intervention| self.assess_intervention(posterior, &intervention.id))
            .collect()
    }

    /// Select the positive-net-value intervention with greatest net value.
    ///
    /// Returns None when no intervention is available or every intervention
    /// has non-positive net value. Ties preserve declaration order.
    pub fn best_intervention_or_abstain(
        &self,
        posterior: &[f64],
    ) -> Result<Option<InterventionAssessment>, DecisionProblemError> {
        let mut best: Option<InterventionAssessment> = None;
        for assessment in self.assess_all_interventions(posterior)? {
            if assessment.net_value <= POSITIVE_VALUE_EPSILON {
                continue;
            }
            if best.as_ref().is_none_or(|current| assessment.net_value > current.net_value) {
                best = Some(assessment);
            }
        }
        Ok(best)
    }

    fn best_decision_with_utilities(
        &self,
        posterior: &[f64],
        utility_by_decision: &[Vec<f64>],
    ) -> Result<DecisionValue, DecisionProblemError> {
        validate_posterior(posterior, self.hypothesis_ids.len())?;
        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (decision_index, utilities) in utility_by_decision.iter().enumerate() {
            let value = dot(posterior, utilities);
            if value > best_value {
                best_value = value;
                best_index = decision_index;
            }
        }
        Ok(DecisionValue {
            decision_id: self.decision_ids[best_index].clone(),
            expected_utility: best_value,
        })
    }

    fn find_intervention(
        &self,
        intervention_id: &str,
    ) -> Result<&Intervention, DecisionProblemError> {
        self.interventions.iter()
            .find(|intervention| intervention.id == intervention_id)
            .ok_or_else(|| DecisionProblemError::UnknownIntervention {
                intervention_id: intervention_id.to_owned(),
            })
    }

    fn find_outcome(
        &self,
        intervention_id: &str,
        outcome_id: &str,
    ) -> Result<&InterventionOutcome, DecisionProblemError> {
        self.find_intervention(intervention_id)?.outcomes.iter()
            .find(|outcome| outcome.id == outcome_id)
            .ok_or_else(|| DecisionProblemError::UnknownOutcome {
                intervention_id: intervention_id.to_owned(),
                outcome_id: outcome_id.to_owned(),
            })
    }
}

fn validate_unique_ids(ids: &[String], kind: &'static str) -> Result<(), DecisionProblemError> {
    let mut seen: Vec<String> = Vec::with_capacity(ids.len());
    for id in ids {
        if id.trim().is_empty() {
            return Err(DecisionProblemError::EmptyId { kind });
        }
        if seen.iter().any(|prior| prior == id) {
            return Err(DecisionProblemError::DuplicateId { kind, id: id.clone() });
        }
        seen.push(id.clone());
    }
    Ok(())
}

fn validate_distribution(values: &[f64], context: &str) -> Result<(), DecisionProblemError> {
    for value in values {
        validate_probability(*value, context.to_owned())?;
    }
    validate_mass(values.iter().sum(), context.to_owned())
}

fn validate_probability(value: f64, context: String) -> Result<(), DecisionProblemError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DecisionProblemError::InvalidProbability { context, value });
    }
    Ok(())
}

fn validate_mass(sum: f64, context: String) -> Result<(), DecisionProblemError> {
    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DecisionProblemError::ProbabilityMassMismatch { context, sum });
    }
    Ok(())
}

fn validate_posterior(
    posterior: &[f64],
    expected_len: usize,
) -> Result<(), DecisionProblemError> {
    if posterior.len() != expected_len {
        return Err(DecisionProblemError::PosteriorLengthMismatch {
            expected: expected_len,
            actual: posterior.len(),
        });
    }
    validate_distribution(posterior, "posterior")
}

fn validate_utility_matrix(
    matrix: &[Vec<f64>],
    expected_rows: usize,
    expected_columns: usize,
    context: &str,
) -> Result<(), DecisionProblemError> {
    if matrix.len() != expected_rows {
        return Err(DecisionProblemError::UtilityRowCountMismatch {
            context: context.to_owned(),
            expected: expected_rows,
            actual: matrix.len(),
        });
    }
    for row in matrix {
        if row.len() != expected_columns {
            return Err(DecisionProblemError::UtilityColumnCountMismatch {
                context: context.to_owned(),
                expected: expected_columns,
                actual: row.len(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(DecisionProblemError::NonFiniteUtility {
                context: context.to_owned(),
            });
        }
    }
    Ok(())
}

fn bayes_update(
    prior: &[f64],
    likelihood: &[f64],
    intervention_id: &str,
    outcome_id: &str,
) -> Result<Vec<f64>, DecisionProblemError> {
    let evidence = dot(prior, likelihood);
    if evidence <= 0.0 || !evidence.is_finite() {
        return Err(DecisionProblemError::ImpossibleOutcome {
            intervention_id: intervention_id.to_owned(),
            outcome_id: outcome_id.to_owned(),
        });
    }
    Ok(prior.iter().zip(likelihood)
        .map(|(prior_mass, likelihood)| prior_mass * likelihood / evidence)
        .collect())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

fn entropy_bits(distribution: &[f64]) -> f64 {
    distribution.iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, likelihood: Vec<f64>, utilities: Vec<Vec<f64>>) -> InterventionOutcome {
        InterventionOutcome {
            id: id.to_owned(),
            likelihood_by_hypothesis: likelihood,
            post_intervention_utility_by_decision: utilities,
        }
    }

    fn two_hypothesis_problem(interventions: Vec<Intervention>) -> DecisionProblem {
        DecisionProblem::new(
            vec!["left_rule".into(), "right_rule".into()],
            vec!["choose_left".into(), "choose_right".into()],
            vec![0.5, 0.5],
            vec![vec![10.0, 0.0], vec![0.0, 10.0]],
            interventions,
        ).expect("valid two-hypothesis problem")
    }

    #[test]
    fn rejects_prior_that_does_not_sum_to_one() {
        let error = DecisionProblem::new(
            vec!["a".into(), "b".into()],
            vec!["x".into()],
            vec![0.2, 0.2],
            vec![vec![1.0, 0.0]],
            Vec::new(),
        ).expect_err("invalid prior must fail closed");
        assert!(matches!(error, DecisionProblemError::ProbabilityMassMismatch { .. }));
    }

    #[test]
    fn rejects_intervention_outcomes_that_do_not_normalize_per_hypothesis() {
        let intervention = Intervention {
            id: "probe".into(),
            cost: 0.0,
            outcomes: vec![outcome(
                "only_outcome",
                vec![0.4, 0.4],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )],
        };
        let error = DecisionProblem::new(
            vec!["left_rule".into(), "right_rule".into()],
            vec!["choose_left".into(), "choose_right".into()],
            vec![0.5, 0.5],
            vec![vec![10.0, 0.0], vec![0.0, 10.0]],
            vec![intervention],
        )
        .expect_err("constructor must reject likelihood mass below one");
        assert!(matches!(error, DecisionProblemError::ProbabilityMassMismatch { .. }));
    }

    #[test]
    fn bayes_update_uses_observed_likelihood_not_the_most_likely_hypothesis() {
        let utilities = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let intervention = Intervention {
            id: "switch_test".into(),
            cost: 0.0,
            outcomes: vec![
                outcome("left_signal", vec![0.8, 0.2], utilities.clone()),
                outcome("right_signal", vec![0.2, 0.8], utilities),
            ],
        };
        let problem = two_hypothesis_problem(vec![intervention]);
        let posterior = problem.posterior_after_outcome(
            &[0.5, 0.5], "switch_test", "left_signal"
        ).expect("possible observation");
        assert!((posterior[0] - 0.8).abs() < 1.0e-12);
        assert!((posterior[1] - 0.2).abs() < 1.0e-12);
        assert_eq!(
            problem.best_decision_after_outcome(
                &[0.5, 0.5], "switch_test", "left_signal"
            ).expect("downstream decision").decision_id,
            "choose_left"
        );
    }

    #[test]
    fn decision_value_prefers_lower_entropy_probe_when_it_changes_the_best_action() {
        let utilities = vec![
            vec![10.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 10.0],
        ];
        let irrelevant_high_information = Intervention {
            id: "irrelevant".into(),
            cost: 0.1,
            outcomes: vec![
                outcome("even", vec![1.0, 0.0, 1.0, 0.0], utilities.clone()),
                outcome("odd", vec![0.0, 1.0, 0.0, 1.0], utilities.clone()),
            ],
        };
        let useful_lower_information = Intervention {
            id: "decision_relevant".into(),
            cost: 0.1,
            outcomes: vec![
                outcome("signal_a", vec![0.75, 0.75, 0.25, 0.25], utilities.clone()),
                outcome("signal_b", vec![0.25, 0.25, 0.75, 0.75], utilities.clone()),
            ],
        };
        let problem = DecisionProblem::new(
            vec!["a0".into(), "a1".into(), "b0".into(), "b1".into()],
            vec!["choose_a".into(), "choose_b".into()],
            vec![0.25, 0.25, 0.25, 0.25],
            utilities,
            vec![irrelevant_high_information, useful_lower_information],
        ).expect("valid decision problem");

        let irrelevant = problem.assess_intervention(problem.prior(), "irrelevant")
            .expect("irrelevant probe");
        let useful = problem.assess_intervention(problem.prior(), "decision_relevant")
            .expect("useful probe");

        assert!(irrelevant.information_gain_bits > useful.information_gain_bits);
        assert!(irrelevant.gross_value_delta.abs() < 1.0e-10);
        assert!(useful.gross_value_delta > 2.49);
        assert!(problem.best_intervention_or_abstain(problem.prior())
            .expect("probe selection")
            .is_some_and(|selected| selected.intervention_id == "decision_relevant"));
    }

    #[test]
    fn abstains_when_every_intervention_has_non_positive_net_value() {
        let utilities = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let costly = Intervention {
            id: "expensive_perfect_probe".into(),
            cost: 5.1,
            outcomes: vec![
                outcome("left", vec![1.0, 0.0], utilities.clone()),
                outcome("right", vec![0.0, 1.0], utilities),
            ],
        };
        let problem = two_hypothesis_problem(vec![costly]);
        let assessment = problem.assess_intervention(problem.prior(), "expensive_perfect_probe")
            .expect("assessment");
        assert!(assessment.gross_value_delta > 4.99);
        assert!(assessment.net_value < 0.0);
        assert_eq!(problem.best_intervention_or_abstain(problem.prior())
            .expect("selection"), None);
    }

    #[test]
    fn state_changing_intervention_can_be_rejected_despite_perfect_information() {
        let destroyed_utility = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let destructive = Intervention {
            id: "irreversible_switch".into(),
            cost: 0.0,
            outcomes: vec![
                outcome("local_rule", vec![1.0, 0.0], destroyed_utility.clone()),
                outcome("global_rule", vec![0.0, 1.0], destroyed_utility),
            ],
        };
        let problem = two_hypothesis_problem(vec![destructive]);
        let assessment = problem.assess_intervention(problem.prior(), "irreversible_switch")
            .expect("assessment");
        assert!((assessment.information_gain_bits - 1.0).abs() < 1.0e-12);
        assert!(assessment.net_value < 0.0);
        assert_eq!(problem.best_intervention_or_abstain(problem.prior())
            .expect("selection"), None);
    }

    #[test]
    fn impossible_observation_is_rejected() {
        let utilities = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let intervention = Intervention {
            id: "probe".into(),
            cost: 0.0,
            outcomes: vec![
                outcome("possible", vec![1.0, 1.0], utilities.clone()),
                outcome("impossible", vec![0.0, 0.0], utilities),
            ],
        };
        let problem = two_hypothesis_problem(vec![intervention]);
        assert!(matches!(
            problem.posterior_after_outcome(problem.prior(), "probe", "impossible"),
            Err(DecisionProblemError::ImpossibleOutcome { .. })
        ));
    }

    #[test]
    fn deterministic_ties_keep_declared_decision_and_intervention_order() {
        let utilities = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
        let problem = DecisionProblem::new(
            vec!["a".into(), "b".into()],
            vec!["first".into(), "second".into()],
            vec![0.5, 0.5],
            utilities.clone(),
            vec![
                Intervention {
                    id: "first_probe".into(),
                    cost: 0.0,
                    outcomes: vec![outcome("only", vec![1.0, 1.0], utilities.clone())],
                },
                Intervention {
                    id: "second_probe".into(),
                    cost: 0.0,
                    outcomes: vec![outcome("only", vec![1.0, 1.0], utilities)],
                },
            ],
        ).expect("valid tied problem");
        assert_eq!(
            problem.best_current_decision(problem.prior()).expect("decision").decision_id,
            "first"
        );
        assert_eq!(
            problem.best_intervention_or_abstain(problem.prior())
                .expect("selection").expect("positive net intervention").intervention_id,
            "first_probe"
        );
    }

    #[test]
    fn no_interventions_returns_none_without_fabricating_a_probe() {
        let problem = two_hypothesis_problem(Vec::new());
        assert_eq!(problem.best_intervention_or_abstain(problem.prior()).expect("selection"), None);
    }

    #[test]
    fn rejects_non_finite_utility_values() {
        let error = DecisionProblem::new(
            vec!["a".into(), "b".into()],
            vec!["x".into()],
            vec![0.5, 0.5],
            vec![vec![f64::NAN, 0.0]],
            Vec::new(),
        ).expect_err("non-finite utility must fail closed");
        assert!(matches!(error, DecisionProblemError::NonFiniteUtility { .. }));
    }
}
