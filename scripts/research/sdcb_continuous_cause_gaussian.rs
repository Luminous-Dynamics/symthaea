//! Exact small-instance Gaussian common-cause experiment-selection oracle.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_continuous_cause_gaussian.rs -o /tmp/sdcb-continuous-cause
//! Run:
//!   /tmp/sdcb-continuous-cause
//!
//! Model: independent Gaussian priors over one target scientific parameter and
//! continuous nuisance factors. Candidate scalar measurements are linear in
//! that latent vector with independent Gaussian residual noise. The oracle
//! exhaustively evaluates every candidate subset under a hard integer budget.
//!
//! This is a finite design reference, not a physical sensor model, a learned
//! selector, or evidence that any declared reference/intervention is valid.

const VARIANCE_FLOOR: f64 = 1e-12;
const OBJECTIVE_TOL: f64 = 1e-12;
const MAX_ACTIONS: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAction {
    pub id: String,
    pub cost_units: u64,
    /// Linear measurement loading, one coefficient per latent state.
    pub loadings: Vec<f64>,
    /// Independent residual measurement-noise variance (not standard deviation).
    pub noise_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianDesign {
    /// The first/selected target need not be index zero.
    pub latent_ids: Vec<String>,
    /// Independent prior variances for each latent coordinate.
    pub prior_variances: Vec<f64>,
    pub science_index: usize,
    pub budget_units: u64,
    pub actions: Vec<MeasurementAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyLatentRegistry,
    InvalidLatentIdentity,
    DuplicateLatentIdentity,
    InvalidPriorVariance,
    InvalidScienceIndex,
    EmptyActionSet,
    TooManyActions,
    InvalidActionIdentity,
    DuplicateActionIdentity,
    InvalidActionCost,
    InvalidNoiseVariance,
    LoadingDimensionMismatch,
    NonFiniteLoading,
    NumericFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianDesignReport {
    pub selected_action_ids: Vec<String>,
    pub selected_cost_units: u64,
    pub budget_units: u64,
    pub prior_science_variance: f64,
    pub posterior_science_variance: f64,
    pub science_variance_reduction_fraction: f64,
    pub prior_marginal_variances: Vec<f64>,
    pub posterior_marginal_variances: Vec<f64>,
    /// 0.5 * log(det(P_post) / det(P_prior)), in nats.
    /// This is joint information gain across all latent variables, not the
    /// science-target objective used for selection.
    pub joint_information_gain_nats: f64,
    pub subsets_evaluated: u64,
}

#[derive(Debug, Clone)]
struct PosteriorSummary {
    marginal_variances: Vec<f64>,
    log_determinant_precision: f64,
}

fn validate_distribution_variances(values: &[f64]) -> Result<(), Error> {
    if values.is_empty() {
        return Err(Error::EmptyLatentRegistry);
    }
    for value in values {
        if !value.is_finite() || *value <= VARIANCE_FLOOR {
            return Err(Error::InvalidPriorVariance);
        }
    }
    Ok(())
}

fn validate_design(design: &GaussianDesign) -> Result<(), Error> {
    validate_distribution_variances(&design.prior_variances)?;
    if design.latent_ids.len() != design.prior_variances.len() {
        return Err(Error::LoadingDimensionMismatch);
    }
    if design.science_index >= design.latent_ids.len() {
        return Err(Error::InvalidScienceIndex);
    }

    for (i, id) in design.latent_ids.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(Error::InvalidLatentIdentity);
        }
        if design.latent_ids[..i].iter().any(|other| other == id) {
            return Err(Error::DuplicateLatentIdentity);
        }
    }

    if design.actions.is_empty() {
        return Err(Error::EmptyActionSet);
    }
    if design.actions.len() > MAX_ACTIONS {
        return Err(Error::TooManyActions);
    }

    for (i, action) in design.actions.iter().enumerate() {
        if action.id.trim().is_empty() {
            return Err(Error::InvalidActionIdentity);
        }
        if design.actions[..i].iter().any(|other| other.id == action.id) {
            return Err(Error::DuplicateActionIdentity);
        }
        if action.cost_units == 0 {
            return Err(Error::InvalidActionCost);
        }
        if !action.noise_variance.is_finite()
            || action.noise_variance <= VARIANCE_FLOOR
        {
            return Err(Error::InvalidNoiseVariance);
        }
        if action.loadings.len() != design.latent_ids.len() {
            return Err(Error::LoadingDimensionMismatch);
        }
        if action.loadings.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteLoading);
        }
    }
    Ok(())
}

/// Cholesky factorization for a symmetric positive-definite matrix.
fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];

    for i in 0..n {
        if matrix[i].len() != n {
            return Err(Error::NumericFailure);
        }
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(Error::NumericFailure);
                }
                lower[i][j] = sum.sqrt();
            } else {
                let denominator = lower[j][j];
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(Error::NumericFailure);
                }
                lower[i][j] = sum / denominator;
                if !lower[i][j].is_finite() {
                    return Err(Error::NumericFailure);
                }
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, Error> {
    let n = lower.len();
    if rhs.len() != n {
        return Err(Error::NumericFailure);
    }
    let mut y = vec![0.0; n];

    // Forward substitution: L y = rhs.
    for i in 0..n {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[i][j] * y[j];
        }
        y[i] = value / lower[i][i];
        if !y[i].is_finite() {
            return Err(Error::NumericFailure);
        }
    }

    // Back substitution: L^T x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = y[i];
        for j in (i + 1)..n {
            value -= lower[j][i] * x[j];
        }
        x[i] = value / lower[i][i];
        if !x[i].is_finite() {
            return Err(Error::NumericFailure);
        }
    }
    Ok(x)
}

fn invert_from_cholesky(lower: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
    let n = lower.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut basis = vec![0.0; n];
        basis[column] = 1.0;
        let solution = solve_cholesky(lower, &basis)?;
        for row in 0..n {
            inverse[row][column] = solution[row];
        }
    }
    // Remove harmless asymmetric round-off from independently solved columns.
    for row in 0..n {
        for column in 0..row {
            let symmetric = 0.5 * (inverse[row][column] + inverse[column][row]);
            inverse[row][column] = symmetric;
            inverse[column][row] = symmetric;
        }
    }
    Ok(inverse)
}

fn posterior_for_mask(
    design: &GaussianDesign,
    mask: u64,
) -> Result<PosteriorSummary, Error> {
    let dimension = design.latent_ids.len();
    let mut precision = vec![vec![0.0; dimension]; dimension];

    for i in 0..dimension {
        precision[i][i] = 1.0 / design.prior_variances[i];
    }

    for (action_index, action) in design.actions.iter().enumerate() {
        if (mask & (1_u64 << action_index)) == 0 {
            continue;
        }
        let inverse_noise = 1.0 / action.noise_variance;
        for row in 0..dimension {
            for column in 0..dimension {
                precision[row][column] +=
                    action.loadings[row] * action.loadings[column] * inverse_noise;
            }
        }
    }

    if precision.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::NumericFailure);
    }

    let lower = cholesky(&precision)?;
    let covariance = invert_from_cholesky(&lower)?;
    let mut marginal_variances = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let variance = covariance[i][i];
        if !variance.is_finite() || variance <= 0.0 {
            return Err(Error::NumericFailure);
        }
        marginal_variances.push(variance);
    }

    let log_determinant_precision =
        2.0 * (0..dimension).map(|i| lower[i][i].ln()).sum::<f64>();
    if !log_determinant_precision.is_finite() {
        return Err(Error::NumericFailure);
    }

    Ok(PosteriorSummary {
        marginal_variances,
        log_determinant_precision,
    })
}

fn selected_ids(design: &GaussianDesign, mask: u64) -> Vec<String> {
    let mut ids: Vec<String> = design
        .actions
        .iter()
        .enumerate()
        .filter(|(index, _)| (mask & (1_u64 << index)) != 0)
        .map(|(_, action)| action.id.clone())
        .collect();
    ids.sort();
    ids
}

/// Exhaustively minimize the target scientific-parameter posterior variance
/// subject to a hard total cost budget. Residual measurement noises are
/// independent conditional on the explicit latent vector.
pub fn optimize(design: &GaussianDesign) -> Result<GaussianDesignReport, Error> {
    validate_design(design)?;

    let prior = posterior_for_mask(design, 0)?;
    let prior_science_variance = design.prior_variances[design.science_index];
    let action_count = design.actions.len();
    let subset_count = 1_u64 << action_count;

    let mut best_mask = 0_u64;
    let mut best_cost = 0_u64;
    let mut best_posterior = prior.clone();
    let mut best_ids: Vec<String> = Vec::new();

    for mask in 1_u64..subset_count {
        let mut total_cost = 0_u64;
        let mut affordable = true;
        for (index, action) in design.actions.iter().enumerate() {
            if (mask & (1_u64 << index)) != 0 {
                match total_cost.checked_add(action.cost_units) {
                    Some(sum) => total_cost = sum,
                    None => {
                        affordable = false;
                        break;
                    }
                }
                if total_cost > design.budget_units {
                    affordable = false;
                    break;
                }
            }
        }
        if !affordable {
            continue;
        }

        let candidate = posterior_for_mask(design, mask)?;
        let candidate_science_variance =
            candidate.marginal_variances[design.science_index];
        let best_science_variance =
            best_posterior.marginal_variances[design.science_index];
        let ids = selected_ids(design, mask);

        let lower_variance = candidate_science_variance
            < best_science_variance - OBJECTIVE_TOL;
        let same_variance = (candidate_science_variance - best_science_variance).abs()
            <= OBJECTIVE_TOL;
        let deterministic_tie_break = same_variance
            && (total_cost < best_cost
                || (total_cost == best_cost && ids < best_ids));

        if lower_variance || deterministic_tie_break {
            best_mask = mask;
            best_cost = total_cost;
            best_posterior = candidate;
            best_ids = ids;
        }
    }

    let prior_log_determinant_precision = design
        .prior_variances
        .iter()
        .map(|variance| (1.0 / variance).ln())
        .sum::<f64>();
    let joint_information_gain_nats =
        0.5 * (best_posterior.log_determinant_precision - prior_log_determinant_precision);
    if !joint_information_gain_nats.is_finite() || joint_information_gain_nats < -OBJECTIVE_TOL {
        return Err(Error::NumericFailure);
    }

    let posterior_science_variance =
        best_posterior.marginal_variances[design.science_index];
    let science_variance_reduction_fraction =
        1.0 - posterior_science_variance / prior_science_variance;

    Ok(GaussianDesignReport {
        selected_action_ids: selected_ids(design, best_mask),
        selected_cost_units: best_cost,
        budget_units: design.budget_units,
        prior_science_variance,
        posterior_science_variance,
        science_variance_reduction_fraction,
        prior_marginal_variances: design.prior_variances.clone(),
        posterior_marginal_variances: best_posterior.marginal_variances,
        joint_information_gain_nats,
        subsets_evaluated: subset_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common_cause_design(budget_units: u64) -> GaussianDesign {
        GaussianDesign {
            latent_ids: vec!["science".into(), "shared_bias".into()],
            prior_variances: vec![1.0, 1.0],
            science_index: 0,
            budget_units,
            actions: vec![
                MeasurementAction {
                    id: "science-device-a".into(),
                    cost_units: 1,
                    loadings: vec![1.0, 1.0],
                    noise_variance: 0.01,
                },
                MeasurementAction {
                    id: "science-device-b".into(),
                    cost_units: 1,
                    loadings: vec![1.0, 1.0],
                    noise_variance: 0.01,
                },
                MeasurementAction {
                    id: "shared-reference".into(),
                    cost_units: 1,
                    loadings: vec![0.0, 1.0],
                    noise_variance: 0.01,
                },
            ],
        }
    }

    #[test]
    fn duplicate_science_channels_do_not_remove_shared_bias_floor() {
        let mut design = common_cause_design(2);
        design.actions.pop();
        let report = optimize(&design).unwrap();
        assert_eq!(report.selected_action_ids.len(), 2);
        assert!((report.posterior_science_variance - 201.0 / 401.0).abs() < 1e-10);
        assert!(report.posterior_science_variance > 0.49);
    }

    #[test]
    fn exact_budgeted_selection_prefers_reference_plus_science() {
        let report = optimize(&common_cause_design(2)).unwrap();
        assert_eq!(
            report.selected_action_ids,
            vec!["science-device-a".to_string(), "shared-reference".to_string()]
        );
        assert!((report.posterior_science_variance - 201.0 / 10_301.0).abs() < 1e-10);
        assert!(report.posterior_science_variance < 0.02);
        assert_eq!(report.selected_cost_units, 2);
        assert_eq!(report.subsets_evaluated, 8);
    }

    #[test]
    fn one_reference_without_a_science_measurement_does_not_improve_science_variance() {
        let mut design = common_cause_design(1);
        design.actions.truncate(3);
        design.budget_units = 1;
        // The reference alone only observes shared bias, so the optimizer selects
        // one science observation instead of wasting the budget on the reference.
        let report = optimize(&design).unwrap();
        assert!((report.posterior_science_variance - 101.0 / 201.0).abs() < 1e-10);
        assert_ne!(report.selected_action_ids, vec!["shared-reference".to_string()]);
    }

    #[test]
    fn independent_sensor_errors_allow_repetition_to_reduce_target_uncertainty() {
        let design = GaussianDesign {
            latent_ids: vec![
                "science".into(),
                "unused_common_factor".into(),
            ],
            prior_variances: vec![1.0, 1.0],
            science_index: 0,
            budget_units: 2,
            actions: vec![
                MeasurementAction {
                    id: "independent-sensor-a".into(),
                    cost_units: 1,
                    loadings: vec![1.0, 0.0],
                    noise_variance: 0.01,
                },
                MeasurementAction {
                    id: "independent-sensor-b".into(),
                    cost_units: 1,
                    loadings: vec![1.0, 0.0],
                    noise_variance: 0.01,
                },
            ],
        };
        let report = optimize(&design).unwrap();
        assert!((report.posterior_science_variance - 1.0 / 201.0).abs() < 1e-10);
        assert!(report.science_variance_reduction_fraction > 0.99);
    }

    #[test]
    fn multiple_continuous_common_causes_require_distinct_reference_information() {
        let design = GaussianDesign {
            latent_ids: vec![
                "science".into(),
                "environment_bias".into(),
                "reference_chain_bias".into(),
            ],
            prior_variances: vec![1.0, 1.0, 1.0],
            science_index: 0,
            budget_units: 3,
            actions: vec![
                MeasurementAction {
                    id: "science-channel-a".into(),
                    cost_units: 1,
                    loadings: vec![1.0, 1.0, 1.0],
                    noise_variance: 0.01,
                },
                MeasurementAction {
                    id: "environment-check".into(),
                    cost_units: 1,
                    loadings: vec![0.0, 1.0, 0.0],
                    noise_variance: 0.01,
                },
                MeasurementAction {
                    id: "independent-chain-check".into(),
                    cost_units: 1,
                    loadings: vec![0.0, 0.0, 1.0],
                    noise_variance: 0.01,
                },
            ],
        };
        let report = optimize(&design).unwrap();
        assert_eq!(report.selected_action_ids.len(), 3);
        assert!(report.posterior_science_variance < 0.04);
        assert!(report.joint_information_gain_nats > 0.0);
    }

    #[test]
    fn zero_budget_returns_the_prior_without_selecting_actions() {
        let report = optimize(&common_cause_design(0)).unwrap();
        assert!(report.selected_action_ids.is_empty());
        assert_eq!(report.selected_cost_units, 0);
        assert_eq!(report.posterior_science_variance, 1.0);
        assert_eq!(report.joint_information_gain_nats, 0.0);
    }

    #[test]
    fn duplicate_action_ids_reject() {
        let mut design = common_cause_design(2);
        design.actions[1].id = design.actions[0].id.clone();
        assert_eq!(optimize(&design), Err(Error::DuplicateActionIdentity));
    }

    #[test]
    fn malformed_loadings_reject() {
        let mut design = common_cause_design(2);
        design.actions[0].loadings = vec![1.0];
        assert_eq!(optimize(&design), Err(Error::LoadingDimensionMismatch));
    }

    #[test]
    fn non_finite_loadings_reject() {
        let mut design = common_cause_design(2);
        design.actions[0].loadings[0] = f64::NAN;
        assert_eq!(optimize(&design), Err(Error::NonFiniteLoading));
    }

    #[test]
    fn invalid_prior_variance_rejects() {
        let mut design = common_cause_design(2);
        design.prior_variances[0] = 0.0;
        assert_eq!(optimize(&design), Err(Error::InvalidPriorVariance));
    }

    #[test]
    fn zero_noise_variance_rejects() {
        let mut design = common_cause_design(2);
        design.actions[0].noise_variance = 0.0;
        assert_eq!(optimize(&design), Err(Error::InvalidNoiseVariance));
    }

    #[test]
    fn action_space_limit_is_enforced_before_enumeration() {
        let mut design = common_cause_design(u64::MAX);
        while design.actions.len() <= MAX_ACTIONS {
            design.actions.push(MeasurementAction {
                id: format!("extra-{}", design.actions.len()),
                cost_units: 1,
                loadings: vec![1.0, 0.0],
                noise_variance: 1.0,
            });
        }
        assert_eq!(optimize(&design), Err(Error::TooManyActions));
    }
}
