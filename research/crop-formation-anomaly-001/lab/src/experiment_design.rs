//! Small finite-hypothesis expected-information-gain reference for experiment
//! selection. Probabilities are supplied by a separately qualified predictive
//! model; this module does not fit or validate those models.

/// Invalid inputs to a finite expected-information-gain calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentDesignError {
    EmptyPrior,
    InvalidPrior(&'static str),
    InvalidOutcomeMatrix(&'static str),
    InvalidAction(&'static str),
    DuplicateActionId,
}

/// Tolerance used only to validate caller-supplied probability normalization.
const PROBABILITY_TOLERANCE: f64 = 1.0e-9;

fn validate_probability_distribution(
    values: &[f64],
    name: &'static str,
) -> Result<(), ExperimentDesignError> {
    if values.is_empty() {
        return Err(ExperimentDesignError::InvalidOutcomeMatrix(name));
    }

    let mut total = 0.0;
    for value in values {
        if !value.is_finite() || *value < 0.0 {
            return Err(ExperimentDesignError::InvalidOutcomeMatrix(name));
        }
        total += *value;
    }

    if !total.is_finite() || (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(ExperimentDesignError::InvalidOutcomeMatrix(name));
    }
    Ok(())
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -*probability * probability.log2())
        .sum()
}

/// Expected mutual information I(H;Y|a), in bits, for one candidate action.
///
/// Prior entry i is P(H_i). Row i of the conditional matrix is P(Y_j | H_i, a),
/// with the same outcome ordering for every hypothesis. Every distribution
/// must already sum to one; this function rejects rather than silently
/// normalizes inputs.
///
/// Include an explicit unknown/model-inadequacy hypothesis only when a
/// defensible predictive distribution for its outcomes is available. Otherwise
/// evaluate sensitivity to alternative prediction sets outside this function.
pub fn expected_information_gain_bits(
    prior_probabilities: &[f64],
    conditional_outcome_probabilities: &[Vec<f64>],
) -> Result<f64, ExperimentDesignError> {
    if prior_probabilities.is_empty() {
        return Err(ExperimentDesignError::EmptyPrior);
    }
    validate_probability_distribution(prior_probabilities, "prior probabilities").map_err(
        |_| {
            ExperimentDesignError::InvalidPrior(
                "prior must be finite, nonnegative and sum to one",
            )
        },
    )?;

    if conditional_outcome_probabilities.len() != prior_probabilities.len() {
        return Err(ExperimentDesignError::InvalidOutcomeMatrix(
            "one conditional outcome row is required per hypothesis",
        ));
    }

    let outcome_count = conditional_outcome_probabilities
        .first()
        .map(Vec::len)
        .ok_or(ExperimentDesignError::InvalidOutcomeMatrix(
            "conditional outcome matrix is empty",
        ))?;

    if outcome_count == 0 {
        return Err(ExperimentDesignError::InvalidOutcomeMatrix(
            "at least one outcome is required",
        ));
    }

    for row in conditional_outcome_probabilities {
        if row.len() != outcome_count {
            return Err(ExperimentDesignError::InvalidOutcomeMatrix(
                "all hypotheses must use the same outcome ordering and width",
            ));
        }
        validate_probability_distribution(row, "conditional outcome probabilities")?;
    }

    let prior_entropy = entropy_bits(prior_probabilities);
    let mut expected_posterior_entropy = 0.0;

    for outcome_index in 0..outcome_count {
        let joint: Vec<f64> = prior_probabilities
            .iter()
            .zip(conditional_outcome_probabilities)
            .map(|(prior, row)| *prior * row[outcome_index])
            .collect();
        let outcome_probability: f64 = joint.iter().sum();

        if outcome_probability <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = joint
            .iter()
            .map(|joint_probability| *joint_probability / outcome_probability)
            .collect();
        expected_posterior_entropy += outcome_probability * entropy_bits(&posterior);
    }

    let information_gain = prior_entropy - expected_posterior_entropy;
    if !information_gain.is_finite() {
        return Err(ExperimentDesignError::InvalidOutcomeMatrix(
            "information-gain calculation produced a non-finite result",
        ));
    }

    // Roundoff can produce a tiny negative number for a mathematically
    // uninformative action. Mutual information is never negative.
    Ok(information_gain.max(0.0))
}


/// A proposed measurement action with caller-supplied predictive outcomes.
///
/// Cost units must be comparable across all actions in one ranking (for
/// example, staff-hours or a preregistered normalized budget unit). Hard gates
/// are external authorization/feasibility decisions, not probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentAction {
    pub action_id: String,
    pub cost_units: f64,
    pub hard_gates_passed: bool,
    pub ineligibility_reason: Option<String>,
    pub conditional_outcome_probabilities: Vec<Vec<f64>>,
}

/// Ranked information value for one action. Ineligible actions are retained,
/// but have no information score and are always placed after eligible actions.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentActionScore {
    pub action_id: String,
    pub eligible: bool,
    pub expected_information_gain_bits: Option<f64>,
    pub cost_units: f64,
    pub bits_per_cost_unit: Option<f64>,
    pub ineligibility_reason: Option<String>,
}

/// Score and rank candidate actions by expected information gain per common
/// cost unit. Hard-gate-failing actions are retained unscored. This routine
/// does not itself determine whether permission, safety, ethics or legal gates
/// pass; the caller must supply that disposition from the relevant authority.
pub fn rank_experiment_actions(
    prior_probabilities: &[f64],
    actions: &[ExperimentAction],
) -> Result<Vec<ExperimentActionScore>, ExperimentDesignError> {
    let mut seen_ids = std::collections::HashSet::new();
    let mut scores = Vec::with_capacity(actions.len());

    for action in actions {
        if action.action_id.trim().is_empty() {
            return Err(ExperimentDesignError::InvalidAction(
                "action identifier must not be empty",
            ));
        }
        if !seen_ids.insert(action.action_id.as_str()) {
            return Err(ExperimentDesignError::DuplicateActionId);
        }
        if !action.cost_units.is_finite() || action.cost_units <= 0.0 {
            return Err(ExperimentDesignError::InvalidAction(
                "cost must be finite and positive",
            ));
        }

        if action.hard_gates_passed {
            let gain = expected_information_gain_bits(
                prior_probabilities,
                &action.conditional_outcome_probabilities,
            )?;
            scores.push(ExperimentActionScore {
                action_id: action.action_id.clone(),
                eligible: true,
                expected_information_gain_bits: Some(gain),
                cost_units: action.cost_units,
                bits_per_cost_unit: Some(gain / action.cost_units),
                ineligibility_reason: None,
            });
        } else {
            let reason = action
                .ineligibility_reason
                .as_deref()
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .ok_or(ExperimentDesignError::InvalidAction(
                    "ineligible actions require a reason",
                ))?;
            scores.push(ExperimentActionScore {
                action_id: action.action_id.clone(),
                eligible: false,
                expected_information_gain_bits: None,
                cost_units: action.cost_units,
                bits_per_cost_unit: None,
                ineligibility_reason: Some(reason.to_string()),
            });
        }
    }

    scores.sort_by(|left, right| match (left.bits_per_cost_unit, right.bits_per_cost_unit) {
        (Some(left_value), Some(right_value)) => right_value
            .total_cmp(&left_value)
            .then_with(|| left.action_id.cmp(&right.action_id)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.action_id.cmp(&right.action_id),
    });

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-9;


    #[test]
    fn action_ranking_uses_information_gain_per_common_cost_unit() {
        let actions = vec![
            ExperimentAction {
                action_id: "perfect-expensive".into(),
                cost_units: 2.0,
                hard_gates_passed: true,
                ineligibility_reason: None,
                conditional_outcome_probabilities: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            },
            ExperimentAction {
                action_id: "partial-cheap".into(),
                cost_units: 0.1,
                hard_gates_passed: true,
                ineligibility_reason: None,
                conditional_outcome_probabilities: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            },
        ];
        let ranking = rank_experiment_actions(&[0.5, 0.5], &actions).unwrap();
        assert_eq!(ranking[0].action_id, "partial-cheap");
        assert_eq!(ranking[1].action_id, "perfect-expensive");
    }

    #[test]
    fn hard_gate_failure_is_visible_but_never_scored() {
        let actions = vec![
            ExperimentAction {
                action_id: "not-permitted".into(),
                cost_units: 0.01,
                hard_gates_passed: false,
                ineligibility_reason: Some("landowner permission not granted".into()),
                conditional_outcome_probabilities: Vec::new(),
            },
            ExperimentAction {
                action_id: "eligible".into(),
                cost_units: 1.0,
                hard_gates_passed: true,
                ineligibility_reason: None,
                conditional_outcome_probabilities: vec![vec![1.0], vec![1.0]],
            },
        ];
        let ranking = rank_experiment_actions(&[0.5, 0.5], &actions).unwrap();
        assert_eq!(ranking[0].action_id, "eligible");
        assert_eq!(ranking[1].action_id, "not-permitted");
        assert!(!ranking[1].eligible);
        assert!(ranking[1].expected_information_gain_bits.is_none());
        assert_eq!(
            ranking[1].ineligibility_reason.as_deref(),
            Some("landowner permission not granted")
        );
    }

    #[test]
    fn action_ranking_rejects_duplicate_identifiers() {
        let action = ExperimentAction {
            action_id: "duplicate".into(),
            cost_units: 1.0,
            hard_gates_passed: true,
            ineligibility_reason: None,
            conditional_outcome_probabilities: vec![vec![1.0], vec![1.0]],
        };
        assert_eq!(
            rank_experiment_actions(&[0.5, 0.5], &[action.clone(), action]),
            Err(ExperimentDesignError::DuplicateActionId)
        );
    }

    #[test]
    fn action_ranking_rejects_invalid_costs() {
        let action = ExperimentAction {
            action_id: "bad-cost".into(),
            cost_units: 0.0,
            hard_gates_passed: false,
            ineligibility_reason: Some("not permitted".into()),
            conditional_outcome_probabilities: Vec::new(),
        };
        assert_eq!(
            rank_experiment_actions(&[0.5, 0.5], &[action]),
            Err(ExperimentDesignError::InvalidAction(
                "cost must be finite and positive"
            ))
        );
    }

    #[test]
    fn ineligible_action_requires_explanation() {
        let action = ExperimentAction {
            action_id: "blocked".into(),
            cost_units: 1.0,
            hard_gates_passed: false,
            ineligibility_reason: None,
            conditional_outcome_probabilities: Vec::new(),
        };
        assert_eq!(
            rank_experiment_actions(&[0.5, 0.5], &[action]),
            Err(ExperimentDesignError::InvalidAction(
                "ineligible actions require a reason"
            ))
        );
    }

    #[test]
    fn perfect_binary_discriminator_provides_one_bit() {
        let gain = expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        assert!((gain - 1.0).abs() < EPS);
    }

    #[test]
    fn identical_predictions_provide_zero_information() {
        let gain = expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![0.8, 0.2], vec![0.8, 0.2]],
        )
        .unwrap();
        assert!(gain.abs() < EPS);
    }

    #[test]
    fn overlapping_predictions_match_binary_channel_known_answer() {
        let gain = expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![0.8, 0.2], vec![0.2, 0.8]],
        )
        .unwrap();
        let expected = 1.0 - (-(0.2_f64) * 0.2_f64.log2() - 0.8 * 0.8_f64.log2());
        assert!((gain - expected).abs() < EPS);
    }

    #[test]
    fn perfect_discrimination_is_bounded_by_prior_entropy() {
        let gain = expected_information_gain_bits(
            &[0.8, 0.2],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let prior_entropy = -(0.8_f64 * 0.8_f64.log2() + 0.2 * 0.2_f64.log2());
        assert!((gain - prior_entropy).abs() < EPS);
    }

    #[test]
    fn zero_probability_outcome_is_handled_without_nan() {
        let gain = expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![1.0, 0.0], vec![1.0, 0.0]],
        )
        .unwrap();
        assert!(gain.abs() < EPS);
    }

    #[test]
    fn invalid_prior_normalization_is_rejected() {
        assert_eq!(
            expected_information_gain_bits(&[0.4, 0.4], &[vec![1.0], vec![1.0]]),
            Err(ExperimentDesignError::InvalidPrior(
                "prior must be finite, nonnegative and sum to one"
            ))
        );
    }

    #[test]
    fn conditional_matrix_requires_one_row_per_hypothesis() {
        assert_eq!(
            expected_information_gain_bits(&[0.5, 0.5], &[vec![1.0, 0.0]]),
            Err(ExperimentDesignError::InvalidOutcomeMatrix(
                "one conditional outcome row is required per hypothesis"
            ))
        );
    }

    #[test]
    fn conditional_rows_must_have_equal_width() {
        assert_eq!(
            expected_information_gain_bits(
                &[0.5, 0.5],
                &[vec![1.0, 0.0], vec![1.0]],
            ),
            Err(ExperimentDesignError::InvalidOutcomeMatrix(
                "all hypotheses must use the same outcome ordering and width"
            ))
        );
    }

    #[test]
    fn invalid_conditional_probabilities_are_rejected() {
        assert!(expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![1.2, -0.2], vec![0.5, 0.5]],
        )
        .is_err());
        assert!(expected_information_gain_bits(
            &[0.5, 0.5],
            &[vec![f64::NAN, 0.0], vec![0.5, 0.5]],
        )
        .is_err());
    }
}
