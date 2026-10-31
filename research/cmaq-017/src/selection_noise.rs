//! Denoised best-of-N selection metrics for CMAQ-017.
//!
//! Rows are candidate actions/chunks; columns are repeated continuation seeds
//! shared across candidates where the environment permits valid common-random
//! number coupling. Selection is cross-fitted: each column is scored only
//! after the candidate was selected using all the other columns.

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionNoiseMetrics {
    pub candidate_count: usize,
    pub continuation_count: usize,
    pub baseline_mean: f64,
    pub best_candidate_mean: f64,
    pub naive_best_of_n_mean: f64,
    pub cross_fitted_selected_mean: f64,
    pub naive_selection_gain: f64,
    pub held_out_selection_gain: f64,
    pub selection_optimism_gap: f64,
    pub candidate_variance_share: f64,
    pub continuation_variance_share: f64,
    pub candidate_continuation_interaction_share: f64,
    pub fold_selected_candidate_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionNoiseError {
    TooFewCandidates,
    EmptyContinuationSet,
    TooFewContinuations,
    UnequalContinuationCount,
    NonFiniteOutcome,
    NonFiniteDerivedMetric,
}

/// Analyze a candidate-by-continuation outcome matrix.
///
/// Every row must contain the same number of repeated continuations. Column j
/// is the same continuation seed/tape for every candidate when common random
/// numbers are valid for the environment. The first row is the predeclared
/// direct-policy/reference candidate, not a candidate selected after seeing
/// qualification outcomes.
///
/// Naive best-of-N selects and scores on the same continuation column;
/// cross-fitted selection chooses from all other columns and scores only on
/// the held-out column.
pub fn evaluate_selection_noise(
    outcomes: &[Vec<f64>],
) -> Result<Option<SelectionNoiseMetrics>, SelectionNoiseError> {
    let Some(first_candidate) = outcomes.first() else {
        return Ok(None);
    };
    if outcomes.len() < 2 {
        return Err(SelectionNoiseError::TooFewCandidates);
    }
    if first_candidate.is_empty() {
        return Err(SelectionNoiseError::EmptyContinuationSet);
    }
    let continuation_count = first_candidate.len();
    if continuation_count < 3 {
        return Err(SelectionNoiseError::TooFewContinuations);
    }

    for candidate in outcomes {
        if candidate.len() != continuation_count {
            return Err(SelectionNoiseError::UnequalContinuationCount);
        }
        if candidate.iter().any(|outcome| !outcome.is_finite()) {
            return Err(SelectionNoiseError::NonFiniteOutcome);
        }
    }

    let candidate_count = outcomes.len();
    let candidate_means: Vec<f64> = outcomes.iter().map(|candidate| mean(candidate)).collect();
    let best_candidate_mean = candidate_means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let baseline_mean = candidate_means[0];

    let mut naive_best_of_n_total = 0.0;
    let mut cross_fitted_selected_total = 0.0;
    let mut fold_selected_candidate_indices = Vec::with_capacity(continuation_count);

    for held_out_column in 0..continuation_count {
        let mut naive_column_best = f64::NEG_INFINITY;
        for candidate in outcomes {
            naive_column_best = naive_column_best.max(candidate[held_out_column]);
        }
        naive_best_of_n_total += naive_column_best;

        let mut selected_index = 0usize;
        let mut selected_training_mean = f64::NEG_INFINITY;
        for candidate_index in 0..candidate_count {
            let mut training_total = 0.0;
            for column in 0..continuation_count {
                if column != held_out_column {
                    training_total += outcomes[candidate_index][column];
                }
            }
            let training_mean = training_total / (continuation_count - 1) as f64;
            if training_mean > selected_training_mean {
                selected_training_mean = training_mean;
                selected_index = candidate_index;
            }
        }

        cross_fitted_selected_total += outcomes[selected_index][held_out_column];
        fold_selected_candidate_indices.push(selected_index);
    }

    let continuation_count_f64 = continuation_count as f64;
    let naive_best_of_n_mean = naive_best_of_n_total / continuation_count_f64;
    let cross_fitted_selected_mean = cross_fitted_selected_total / continuation_count_f64;
    let naive_selection_gain = naive_best_of_n_mean - baseline_mean;
    let held_out_selection_gain = cross_fitted_selected_mean - baseline_mean;
    let selection_optimism_gap = naive_selection_gain - held_out_selection_gain;

    let (candidate_variance_share, continuation_variance_share, interaction_share) =
        variance_component_shares(outcomes, &candidate_means)?;
    let derived = [
        best_candidate_mean,
        baseline_mean,
        naive_best_of_n_mean,
        cross_fitted_selected_mean,
        naive_selection_gain,
        held_out_selection_gain,
        selection_optimism_gap,
        candidate_variance_share,
        continuation_variance_share,
        interaction_share,
    ];
    if derived.iter().any(|value| !value.is_finite()) {
        return Err(SelectionNoiseError::NonFiniteDerivedMetric);
    }

    Ok(Some(SelectionNoiseMetrics {
        candidate_count,
        continuation_count,
        baseline_mean,
        best_candidate_mean,
        naive_best_of_n_mean,
        cross_fitted_selected_mean,
        naive_selection_gain,
        held_out_selection_gain,
        selection_optimism_gap,
        candidate_variance_share,
        continuation_variance_share,
        candidate_continuation_interaction_share: interaction_share,
        fold_selected_candidate_indices,
    }))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Balanced two-way decomposition into candidate, continuation and
/// candidate-by-continuation interaction sums of squares. With one observation
/// per cell, the interaction term also contains cell-level residual noise; it
/// must not be interpreted as a separately identified physical mechanism.
fn variance_component_shares(
    outcomes: &[Vec<f64>],
    candidate_means: &[f64],
) -> Result<(f64, f64, f64), SelectionNoiseError> {
    let candidate_count = outcomes.len();
    let continuation_count = outcomes[0].len();
    let grand_mean = candidate_means.iter().sum::<f64>() / candidate_count as f64;
    let mut continuation_means = vec![0.0; continuation_count];

    for candidate in outcomes {
        for (column, outcome) in candidate.iter().enumerate() {
            continuation_means[column] += *outcome / candidate_count as f64;
        }
    }

    let candidate_ss = continuation_count as f64
        * candidate_means
            .iter()
            .map(|candidate_mean| (candidate_mean - grand_mean).powi(2))
            .sum::<f64>();
    let continuation_ss = candidate_count as f64
        * continuation_means
            .iter()
            .map(|continuation_mean| (continuation_mean - grand_mean).powi(2))
            .sum::<f64>();

    let mut interaction_ss = 0.0;
    for (candidate_index, candidate) in outcomes.iter().enumerate() {
        for (column, outcome) in candidate.iter().enumerate() {
            let residual = outcome - candidate_means[candidate_index]
                - continuation_means[column]
                + grand_mean;
            interaction_ss += residual * residual;
        }
    }

    let total_ss = candidate_ss + continuation_ss + interaction_ss;
    if !candidate_ss.is_finite()
        || !continuation_ss.is_finite()
        || !interaction_ss.is_finite()
        || !total_ss.is_finite()
    {
        return Err(SelectionNoiseError::NonFiniteDerivedMetric);
    }
    if total_ss <= 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    Ok((
        candidate_ss / total_ss,
        continuation_ss / total_ss,
        interaction_ss / total_ss,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_best_of_n_can_report_gain_when_candidates_have_equal_means() {
        let outcomes = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![10.0, -10.0, 0.0, 0.0],
            vec![-10.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, -10.0],
        ];
        let metrics = evaluate_selection_noise(&outcomes).unwrap().unwrap();

        assert!((metrics.best_candidate_mean - 0.0).abs() < 1e-12);
        assert!((metrics.baseline_mean - 0.0).abs() < 1e-12);
        assert!((metrics.naive_best_of_n_mean - 7.5).abs() < 1e-12);
        assert!((metrics.cross_fitted_selected_mean + 7.5).abs() < 1e-12);
        assert!((metrics.naive_selection_gain - 7.5).abs() < 1e-12);
        assert!((metrics.held_out_selection_gain + 7.5).abs() < 1e-12);
        assert!((metrics.selection_optimism_gap - 15.0).abs() < 1e-12);
    }

    #[test]
    fn stable_candidate_advantage_survives_held_out_continuations() {
        let outcomes = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        ];
        let metrics = evaluate_selection_noise(&outcomes).unwrap().unwrap();

        assert_eq!(metrics.naive_best_of_n_mean, 2.0);
        assert_eq!(metrics.cross_fitted_selected_mean, 2.0);
        assert_eq!(metrics.held_out_selection_gain, 2.0);
        assert_eq!(metrics.selection_optimism_gap, 0.0);
        assert!(metrics.fold_selected_candidate_indices.iter().all(|index| *index == 1));
    }

    #[test]
    fn common_continuation_shocks_are_separated_from_candidate_quality() {
        let outcomes = vec![
            vec![-2.0, 0.0, 1.0, 4.0],
            vec![-2.0, 0.0, 1.0, 4.0],
            vec![-2.0, 0.0, 1.0, 4.0],
        ];
        let metrics = evaluate_selection_noise(&outcomes).unwrap().unwrap();

        assert_eq!(metrics.candidate_variance_share, 0.0);
        assert!((metrics.continuation_variance_share - 1.0).abs() < 1e-12);
        assert_eq!(metrics.candidate_continuation_interaction_share, 0.0);
        assert_eq!(metrics.naive_selection_gain, 0.0);
        assert_eq!(metrics.held_out_selection_gain, 0.0);
    }

    #[test]
    fn two_way_decomposition_identifies_candidate_and_interaction_components() {
        let candidate_only = vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 2.0, 2.0],
            vec![4.0, 4.0, 4.0],
        ];
        let candidate_metrics = evaluate_selection_noise(&candidate_only).unwrap().unwrap();
        assert!((candidate_metrics.candidate_variance_share - 1.0).abs() < 1e-12);
        assert_eq!(candidate_metrics.continuation_variance_share, 0.0);
        assert_eq!(candidate_metrics.candidate_continuation_interaction_share, 0.0);

        let interaction_only = vec![
            vec![0.0, 1.0, -1.0],
            vec![1.0, -1.0, 0.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let interaction_metrics = evaluate_selection_noise(&interaction_only).unwrap().unwrap();
        assert!((interaction_metrics.candidate_continuation_interaction_share - 1.0).abs() < 1e-12);
    }

    #[test]
    fn variance_component_shares_sum_to_one_when_outcomes_vary() {
        let outcomes = vec![
            vec![0.0, 2.0, 4.0, 3.0],
            vec![5.0, 7.0, 9.0, 12.0],
            vec![10.0, 12.0, 14.0, 16.0],
        ];
        let metrics = evaluate_selection_noise(&outcomes).unwrap().unwrap();
        let sum = metrics.candidate_variance_share
            + metrics.continuation_variance_share
            + metrics.candidate_continuation_interaction_share;
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn malformed_candidate_continuation_matrices_fail_closed() {
        assert_eq!(evaluate_selection_noise(&[]).unwrap(), None);
        assert_eq!(
            evaluate_selection_noise(&[vec![1.0, 2.0, 3.0]]),
            Err(SelectionNoiseError::TooFewCandidates)
        );
        assert_eq!(
            evaluate_selection_noise(&[vec![1.0, 2.0], vec![1.0, 2.0]]),
            Err(SelectionNoiseError::TooFewContinuations)
        );
        assert_eq!(
            evaluate_selection_noise(&[vec![1.0, 2.0, 3.0], vec![1.0, 2.0]]),
            Err(SelectionNoiseError::UnequalContinuationCount)
        );
        assert_eq!(
            evaluate_selection_noise(&[vec![1.0, f64::NAN, 3.0], vec![1.0, 2.0, 3.0]]),
            Err(SelectionNoiseError::NonFiniteOutcome)
        );
    }

    #[test]
    fn repeated_analysis_is_deterministic() {
        let outcomes = vec![
            vec![1.0, 1.5, 1.2, 0.8],
            vec![1.2, 1.0, 1.6, 0.9],
            vec![0.7, 0.8, 1.1, 1.4],
        ];
        let first = evaluate_selection_noise(&outcomes).unwrap().unwrap();
        let second = evaluate_selection_noise(&outcomes).unwrap().unwrap();
        assert_eq!(first, second);
    }
}
