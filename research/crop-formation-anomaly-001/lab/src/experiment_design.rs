//! Small finite-hypothesis expected-information-gain reference for experiment
//! selection. Probabilities are supplied by a separately qualified predictive
//! model; this module does not fit or validate those models.

/// Invalid inputs to a finite expected-information-gain calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentDesignError {
    EmptyPrior,
    InvalidPrior(&'static str),
    InvalidOutcomeMatrix(&'static str),
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
        total += value;
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
        .map(|probability| -probability * probability.log2())
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
    validate_probability_distribution(prior_probabilities, "prior probabilities")
        .map_err(|_| ExperimentDesignError::InvalidPrior("prior must be finite, nonnegative and sum to one"))?;

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
            .map(|(prior, row)| prior * row[outcome_index])
            .collect();
        let outcome_probability: f64 = joint.iter().sum();

        if outcome_probability <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = joint
            .iter()
            .map(|joint_probability| joint_probability / outcome_probability)
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-9;

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
