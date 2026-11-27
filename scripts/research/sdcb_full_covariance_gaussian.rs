//! Exact full-covariance Gaussian common-cause design oracle.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_full_covariance_gaussian.rs -o /tmp/sdcb-full-covariance
//! Run:
//!   /tmp/sdcb-full-covariance
//!
//! Each candidate is y_a = h_a^T x + e_a. Both the latent prior covariance
//! and the candidate residual covariance are explicit full SPD matrices.
//! The bounded exact subset objective minimizes scientific marginal variance.

const MAX_LATENTS: usize = 8;
const MAX_ACTIONS: usize = 12;
const MATRIX_TOL: f64 = 1e-10;
const VARIANCE_TOL: f64 = 1e-12;
const OBJECTIVE_TOL: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAction {
    pub id: String,
    pub cost_units: u64,
    pub loadings: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FullCovarianceDesign {
    pub latent_ids: Vec<String>,
    pub prior_covariance: Vec<Vec<f64>>,
    pub science_index: usize,
    pub budget_units: u64,
    pub actions: Vec<MeasurementAction>,
    /// Full residual covariance in action-registry order.
    pub residual_covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyLatentRegistry,
    TooManyLatents,
    InvalidLatentIdentity,
    DuplicateLatentIdentity,
    InvalidScienceIndex,
    InvalidPriorDimensions,
    InvalidResidualDimensions,
    NonFiniteMatrixValue,
    AsymmetricMatrix,
    MatrixNotPositiveDefinite,
    EmptyActionSet,
    TooManyActions,
    InvalidActionIdentity,
    DuplicateActionIdentity,
    InvalidActionCost,
    LoadingDimensionMismatch,
    NonFiniteLoading,
    NumericFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DesignReport {
    pub selected_action_ids: Vec<String>,
    pub selected_cost_units: u64,
    pub budget_units: u64,
    pub prior_science_variance: f64,
    pub posterior_science_variance: f64,
    pub science_variance_reduction_fraction: f64,
    pub prior_marginal_variances: Vec<f64>,
    pub posterior_marginal_variances: Vec<f64>,
    pub joint_information_gain_nats: f64,
    pub subsets_evaluated: u64,
}

#[derive(Debug, Clone)]
struct Posterior {
    covariance: Vec<Vec<f64>>,
    log_det_precision: f64,
}

fn validate_matrix(matrix: &[Vec<f64>], dimension: usize, dimension_error: Error) -> Result<(), Error> {
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return Err(dimension_error);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteMatrixValue);
    }
    for i in 0..dimension {
        for j in 0..i {
            if (matrix[i][j] - matrix[j][i]).abs() > MATRIX_TOL {
                return Err(Error::AsymmetricMatrix);
            }
        }
    }
    Ok(())
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(Error::NumericFailure);
    }
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= VARIANCE_TOL {
                    return Err(Error::MatrixNotPositiveDefinite);
                }
                lower[i][j] = sum.sqrt();
            } else {
                let denominator = lower[j][j];
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(Error::MatrixNotPositiveDefinite);
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
    for row in 0..n {
        for column in 0..row {
            let symmetric = 0.5 * (inverse[row][column] + inverse[column][row]);
            inverse[row][column] = symmetric;
            inverse[column][row] = symmetric;
        }
    }
    Ok(inverse)
}

fn invert_spd(matrix: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64), Error> {
    let lower = cholesky(matrix)?;
    let inverse = invert_from_cholesky(&lower)?;
    let log_det = 2.0 * (0..lower.len()).map(|i| lower[i][i].ln()).sum::<f64>();
    if !log_det.is_finite() {
        return Err(Error::NumericFailure);
    }
    Ok((inverse, log_det))
}

fn validate_design(design: &FullCovarianceDesign) -> Result<(), Error> {
    let n = design.latent_ids.len();
    if n == 0 {
        return Err(Error::EmptyLatentRegistry);
    }
    if n > MAX_LATENTS {
        return Err(Error::TooManyLatents);
    }
    if design.science_index >= n {
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

    validate_matrix(&design.prior_covariance, n, Error::InvalidPriorDimensions)?;
    cholesky(&design.prior_covariance)?;

    let m = design.actions.len();
    if m == 0 {
        return Err(Error::EmptyActionSet);
    }
    if m > MAX_ACTIONS {
        return Err(Error::TooManyActions);
    }
    validate_matrix(
        &design.residual_covariance,
        m,
        Error::InvalidResidualDimensions,
    )?;
    cholesky(&design.residual_covariance)?;

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
        if action.loadings.len() != n {
            return Err(Error::LoadingDimensionMismatch);
        }
        if action.loadings.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteLoading);
        }
    }
    Ok(())
}

fn posterior_for_selection(
    design: &FullCovarianceDesign,
    selected: &[usize],
    prior_precision: &[Vec<f64>],
) -> Result<Posterior, Error> {
    let n = design.latent_ids.len();
    if selected.is_empty() {
        let (_, log_det_precision) = invert_spd(prior_precision)?;
        return Ok(Posterior {
            covariance: design.prior_covariance.clone(),
            log_det_precision,
        });
    }

    let k = selected.len();
    let mut residual = vec![vec![0.0; k]; k];
    let mut h = vec![vec![0.0; n]; k];
    for (row, &action_index) in selected.iter().enumerate() {
        h[row].clone_from_slice(&design.actions[action_index].loadings);
        for (column, &other_index) in selected.iter().enumerate() {
            residual[row][column] = design.residual_covariance[action_index][other_index];
        }
    }

    let (residual_precision, _) = invert_spd(&residual)?;
    let mut precision = prior_precision.to_vec();
    for row in 0..n {
        for column in 0..n {
            let mut information = 0.0;
            for i in 0..k {
                for j in 0..k {
                    information += h[i][row] * residual_precision[i][j] * h[j][column];
                }
            }
            precision[row][column] += information;
        }
    }

    let lower = cholesky(&precision)?;
    let covariance = invert_from_cholesky(&lower)?;
    let log_det_precision = 2.0 * (0..n).map(|i| lower[i][i].ln()).sum::<f64>();
    if !log_det_precision.is_finite() {
        return Err(Error::NumericFailure);
    }
    Ok(Posterior { covariance, log_det_precision })
}

fn ids_for_selection(design: &FullCovarianceDesign, selected: &[usize]) -> Vec<String> {
    let mut ids: Vec<String> = selected
        .iter()
        .map(|&index| design.actions[index].id.clone())
        .collect();
    ids.sort();
    ids
}

/// Exhaustively evaluate all subsets under a hard budget. Cross-action residual
/// covariance is preserved through the selected principal submatrix of R.
pub fn optimize(design: &FullCovarianceDesign) -> Result<DesignReport, Error> {
    validate_design(design)?;
    // invert_spd returns the inverse and log determinant of its input.
    let (prior_precision, log_det_prior_covariance) =
        invert_spd(&design.prior_covariance)?;
    let prior_log_det_precision = -log_det_prior_covariance;
    let prior = posterior_for_selection(design, &[], &prior_precision)?;
    let subset_count = 1_u64 << design.actions.len();

    let mut best_selected = Vec::<usize>::new();
    let mut best_cost = 0_u64;
    let mut best = prior.clone();

    for mask in 1_u64..subset_count {
        let selected: Vec<usize> = (0..design.actions.len())
            .filter(|index| (mask & (1_u64 << *index)) != 0)
            .collect();
        let mut total_cost = 0_u64;
        let mut affordable = true;
        for &index in &selected {
            match total_cost.checked_add(design.actions[index].cost_units) {
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
        if !affordable {
            continue;
        }

        let candidate = posterior_for_selection(design, &selected, &prior_precision)?;
        let candidate_variance = candidate.covariance[design.science_index][design.science_index];
        let best_variance = best.covariance[design.science_index][design.science_index];
        let candidate_ids = ids_for_selection(design, &selected);
        let best_ids = ids_for_selection(design, &best_selected);
        let lower_variance = candidate_variance < best_variance - OBJECTIVE_TOL;
        let same_variance = (candidate_variance - best_variance).abs() <= OBJECTIVE_TOL;
        let deterministic_tie = same_variance
            && (total_cost < best_cost
                || (total_cost == best_cost && candidate_ids < best_ids));
        if lower_variance || deterministic_tie {
            best_selected = selected;
            best_cost = total_cost;
            best = candidate;
        }
    }

    let prior_science_variance =
        design.prior_covariance[design.science_index][design.science_index];
    let posterior_science_variance =
        best.covariance[design.science_index][design.science_index];
    let science_variance_reduction_fraction =
        1.0 - posterior_science_variance / prior_science_variance;
    let joint_information_gain_nats =
        0.5 * (best.log_det_precision - prior_log_det_precision);
    if !joint_information_gain_nats.is_finite() || joint_information_gain_nats < -OBJECTIVE_TOL {
        return Err(Error::NumericFailure);
    }

    Ok(DesignReport {
        selected_action_ids: ids_for_selection(design, &best_selected),
        selected_cost_units: best_cost,
        budget_units: design.budget_units,
        prior_science_variance,
        posterior_science_variance,
        science_variance_reduction_fraction,
        prior_marginal_variances: (0..design.latent_ids.len())
            .map(|i| design.prior_covariance[i][i])
            .collect(),
        posterior_marginal_variances: (0..design.latent_ids.len())
            .map(|i| best.covariance[i][i])
            .collect(),
        joint_information_gain_nats,
        subsets_evaluated: subset_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_design() -> FullCovarianceDesign {
        FullCovarianceDesign {
            latent_ids: vec!["science".into(), "shared_bias".into()],
            prior_covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            science_index: 0,
            budget_units: 2,
            actions: vec![
                MeasurementAction { id: "science-a".into(), cost_units: 1, loadings: vec![1.0, 1.0] },
                MeasurementAction { id: "science-b".into(), cost_units: 1, loadings: vec![1.0, 1.0] },
                MeasurementAction { id: "reference".into(), cost_units: 1, loadings: vec![0.0, 1.0] },
            ],
            residual_covariance: vec![
                vec![0.01, 0.0, 0.0],
                vec![0.0, 0.01, 0.0],
                vec![0.0, 0.0, 0.01],
            ],
        }
    }

    #[test]
    fn full_covariance_reference_reproduces_diagonal_gaussian_fixture() {
        let report = optimize(&base_design()).unwrap();
        assert_eq!(report.selected_action_ids, vec!["reference".to_string(), "science-a".to_string()]);
        assert!((report.posterior_science_variance - 201.0 / 10_301.0).abs() < 1e-10);
    }

    #[test]
    fn correlated_prior_is_used_in_posterior() {
        let mut design = base_design();
        design.prior_covariance = vec![vec![1.0, 0.4], vec![0.4, 1.0]];
        let report = optimize(&design).unwrap();
        assert!(report.posterior_science_variance > 0.0);
        assert!(report.posterior_science_variance < report.prior_science_variance);
    }

    #[test]
    fn correlated_residuals_reduce_false_information_from_duplicate_channels() {
        let mut design = FullCovarianceDesign {
            latent_ids: vec!["science".into()],
            prior_covariance: vec![vec![1.0]],
            science_index: 0,
            budget_units: 2,
            actions: vec![
                MeasurementAction { id: "a".into(), cost_units: 1, loadings: vec![1.0] },
                MeasurementAction { id: "b".into(), cost_units: 1, loadings: vec![1.0] },
            ],
            residual_covariance: vec![vec![0.01, 0.0099], vec![0.0099, 0.01]],
        };
        let correlated = optimize(&design).unwrap();
        design.residual_covariance = vec![vec![0.01, 0.0], vec![0.0, 0.01]];
        let independent = optimize(&design).unwrap();
        assert!(correlated.posterior_science_variance > independent.posterior_science_variance);
        assert!((correlated.posterior_science_variance - 1.0 / (1.0 + 2.0 / 0.0199)).abs() < 1e-10);
    }

    #[test]
    fn residual_correlation_is_preserved_in_selected_principal_submatrix() {
        let mut design = base_design();
        design.residual_covariance[0][1] = 0.009;
        design.residual_covariance[1][0] = 0.009;
        let report = optimize(&design).unwrap();
        assert!(report.posterior_science_variance < report.prior_science_variance);
        assert!(report.selected_cost_units <= design.budget_units);
    }

    #[test]
    fn zero_budget_returns_prior() {
        let mut design = base_design();
        design.budget_units = 0;
        let report = optimize(&design).unwrap();
        assert!(report.selected_action_ids.is_empty());
        assert!((report.posterior_science_variance - 1.0).abs() < 1e-12);
        assert!(report.joint_information_gain_nats.abs() < 1e-12);
    }

    #[test]
    fn non_positive_definite_prior_rejects() {
        let mut design = base_design();
        design.prior_covariance = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(optimize(&design), Err(Error::MatrixNotPositiveDefinite));
    }

    #[test]
    fn non_positive_definite_residual_covariance_rejects() {
        let mut design = base_design();
        design.residual_covariance = vec![
            vec![0.01, 0.02, 0.0],
            vec![0.02, 0.01, 0.0],
            vec![0.0, 0.0, 0.01],
        ];
        assert_eq!(optimize(&design), Err(Error::MatrixNotPositiveDefinite));
    }

    #[test]
    fn asymmetric_matrix_rejects() {
        let mut design = base_design();
        design.prior_covariance[0][1] = 0.2;
        assert_eq!(optimize(&design), Err(Error::AsymmetricMatrix));
    }

    #[test]
    fn duplicate_action_ids_reject() {
        let mut design = base_design();
        design.actions[1].id = design.actions[0].id.clone();
        assert_eq!(optimize(&design), Err(Error::DuplicateActionIdentity));
    }

    #[test]
    fn non_finite_matrix_values_reject() {
        let mut design = base_design();
        design.residual_covariance[0][0] = f64::NAN;
        assert_eq!(optimize(&design), Err(Error::NonFiniteMatrixValue));
    }

    #[test]
    fn malformed_loading_rejects() {
        let mut design = base_design();
        design.actions[0].loadings.pop();
        assert_eq!(optimize(&design), Err(Error::LoadingDimensionMismatch));
    }

    #[test]
    fn action_limit_is_checked_before_enumeration() {
        let mut design = base_design();
        while design.actions.len() <= MAX_ACTIONS {
            design.actions.push(MeasurementAction {
                id: format!("extra-{}", design.actions.len()),
                cost_units: 1,
                loadings: vec![1.0, 0.0],
            });
        }
        let n = design.actions.len();
        design.residual_covariance = vec![vec![0.0; n]; n];
        for i in 0..n {
            design.residual_covariance[i][i] = 1.0;
        }
        assert_eq!(optimize(&design), Err(Error::TooManyActions));
    }
}
