// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Single-observer unequal-variance signal-detection likelihood fitting.
//!
//! This crate consumes frozen ordered category counts. It does not construct
//! observations, pool participants, or implement ROC/AUC a second time.

use std::fmt;

use symthaea_statistics::{normal_log_interval, normal_quantile};

const MU_BOUND: f64 = 12.0;
const LOG_SIGMA_BOUND: f64 = 4.0;
const FIRST_THRESHOLD_BOUND: f64 = 12.0;
const LOG_GAP_MIN: f64 = -6.0;
const LOG_GAP_MAX: f64 = 4.0;

/// One ordered type-1 response category. Include zero-count categories from
/// the frozen observation profile so the category boundaries are explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoryCount {
    pub order: i16,
    pub signal: u64,
    pub noise: u64,
}

/// Errors detected while constructing a complete category table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableError {
    TooFewCategories,
    NonIncreasingCategoryOrder,
    EmptySignalClass,
    EmptyNoiseClass,
    CountOverflow,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::TooFewCategories => "UVSD fitting requires at least three ordered categories",
            Self::NonIncreasingCategoryOrder => "category order values must be strictly increasing",
            Self::EmptySignalClass => "the table contains no signal trials",
            Self::EmptyNoiseClass => "the table contains no noise trials",
            Self::CountOverflow => "category count total overflowed u64",
        };
        f.write_str(message)
    }
}

impl std::error::Error for TableError {}

/// Validated, unpooled category frequencies for one participant/session unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoryTable {
    categories: Vec<CategoryCount>,
    signal_total: u64,
    noise_total: u64,
}

impl CategoryTable {
    pub fn new(categories: Vec<CategoryCount>) -> Result<Self, TableError> {
        if categories.len() < 3 {
            return Err(TableError::TooFewCategories);
        }
        if categories.windows(2).any(|pair| pair[0].order >= pair[1].order) {
            return Err(TableError::NonIncreasingCategoryOrder);
        }
        let mut signal_total = 0u64;
        let mut noise_total = 0u64;
        for row in &categories {
            signal_total = signal_total.checked_add(row.signal).ok_or(TableError::CountOverflow)?;
            noise_total = noise_total.checked_add(row.noise).ok_or(TableError::CountOverflow)?;
        }
        if signal_total == 0 {
            return Err(TableError::EmptySignalClass);
        }
        if noise_total == 0 {
            return Err(TableError::EmptyNoiseClass);
        }
        Ok(Self { categories, signal_total, noise_total })
    }

    pub fn categories(&self) -> &[CategoryCount] {
        &self.categories
    }

    pub fn signal_total(&self) -> u64 {
        self.signal_total
    }

    pub fn noise_total(&self) -> u64 {
        self.noise_total
    }
}

/// Optional count correction used to reproduce an external implementation.
///
/// The primary analysis is unsmoothed. MiyoshiReferencePerCategory adds 1/K
/// to every category in each stimulus class, where K is the number of ordered
/// response categories. This matches the authors' public R implementation
/// (1 / length(nr_s1)) and the independent rt-da Python implementation
/// (1 / (2 * n_bins)); it is not a trial-count-scaled correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountCorrection {
    None,
    MiyoshiReferencePerCategory,
}

impl CountCorrection {
    fn additive_count(self, category_count: usize) -> f64 {
        match self {
            Self::None => 0.0,
            Self::MiyoshiReferencePerCategory => 1.0 / category_count as f64,
        }
    }
}

/// Optimizer used to find the maximum-likelihood estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// Multi-start BFGS with analytic gradients and a finite-difference fallback.
    Bfgs,
    /// Derivative-free Nelder-Mead simplex.
    NelderMead,
}

/// Deterministic fit controls. The default is deliberately modest so that
/// individual participant/session fits remain practical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitOptions {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub starts: usize,
    /// Defaults to no smoothing; select the reference correction only for
    /// explicitly labelled compatibility comparisons.
    pub count_correction: CountCorrection,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_iterations: 800,
            gradient_tolerance: 1.0e-6,
            starts: 6,
            count_correction: CountCorrection::None,
        }
    }
}

/// Model parameters in the noise-reference parameterization.
#[derive(Debug, Clone, PartialEq)]
pub struct UvsdEstimate {
    /// Signal-distribution mean; the noise distribution is N(0, 1).
    pub mu: f64,
    /// Signal-distribution standard deviation (strictly positive).
    pub sigma: f64,
    /// Unequal-variance sensitivity index.
    pub d_a: f64,
    /// Ordered decision criteria, one fewer than the category count.
    pub thresholds: Vec<f64>,
}

/// Gradient of the unsmoothed negative log likelihood in natural parameters.
/// log_sigma is the derivative with respect to ln(sigma), not sigma itself.
#[derive(Debug, Clone, PartialEq)]
pub struct UvsdGradient {
    pub mu: f64,
    pub log_sigma: f64,
    pub thresholds: Vec<f64>,
}


impl UvsdEstimate {
    pub fn new(mu: f64, sigma: f64, thresholds: Vec<f64>) -> Option<Self> {
        if !mu.is_finite()
            || !sigma.is_finite()
            || sigma <= 0.0
            || thresholds.len() < 2
            || thresholds.iter().any(|value| !value.is_finite())
            || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return None;
        }
        let d_a = mu / ((1.0 + sigma * sigma) / 2.0).sqrt();
        if !d_a.is_finite() {
            return None;
        }
        Some(Self { mu, sigma, d_a, thresholds })
    }
}

/// Explicit disposition. Only Converged is a successful interior fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    Converged,
    Boundary,
    NotIdentifiable,
    WeaklyIdentified,
    NotConverged,
    NotEvaluable,
}

/// Expected Fisher information for the fitted category design. Eigenvalues
/// use internal coordinates (mu, log-sigma, first threshold, log-gaps).
#[derive(Debug, Clone, PartialEq)]
pub struct FisherInformationDiagnostics {
    pub eigenvalues_ascending: Vec<f64>,
    pub parameter_dimension: usize,
    pub effective_rank: usize,
    pub condition_number: Option<f64>,
    pub relative_minimum_eigenvalue: Option<f64>,
    pub weakly_identified: bool,
    pub mu_standard_error: Option<f64>,
    pub sigma_standard_error: Option<f64>,
    pub sigma_wald_95: Option<[f64; 2]>,
    pub d_a_standard_error: Option<f64>,
    pub d_a_wald_95: Option<[f64; 2]>,
}

/// Diagnostics are part of the result, not optional logging.
#[derive(Debug, Clone, PartialEq)]
pub struct FitDiagnostics {
    pub optimizer: Optimizer,
    pub negative_log_likelihood: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub iterations: usize,
    pub starts_attempted: usize,
    /// Starts that satisfied the optimizer's numerical stopping criterion.
    pub starts_converged: usize,
    /// Starts whose objective was non-finite and therefore could not be ranked.
    pub starts_non_finite: usize,
    pub solution_objective_spread: Option<f64>,
    /// Per-parameter max-minus-min across converged starts, in internal optimizer coordinates
    /// (mu, log-sigma, first threshold, then log-gaps), when at least two converge.
    pub solution_parameter_spread: Option<Vec<f64>>,
    /// Coordinate-wise observed second derivative at the selected solution; not a full Hessian,
    /// covariance matrix, or confidence interval.
    pub diagonal_curvature: Option<Vec<f64>>,
    pub expected_fisher_information: Option<FisherInformationDiagnostics>,
    pub at_parameter_boundary: bool,
    pub note: Option<String>,
}

/// An estimate is returned for Boundary and NotConverged to support inspection,
/// but callers must not treat either status as a qualified point estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub status: FitStatus,
    pub estimate: Option<UvsdEstimate>,
    pub diagnostics: FitDiagnostics,
}

impl FitResult {
    fn failure(status: FitStatus, optimizer: Optimizer, note: impl Into<String>) -> Self {
        Self::failure_after_starts(status, optimizer, note, 0, 0)
    }

    fn failure_after_starts(
        status: FitStatus,
        optimizer: Optimizer,
        note: impl Into<String>,
        starts_attempted: usize,
        starts_non_finite: usize,
    ) -> Self {
        Self {
            status,
            estimate: None,
            diagnostics: FitDiagnostics {
                optimizer,
                negative_log_likelihood: None,
                gradient_norm: None,
                iterations: 0,
                starts_attempted,
                starts_converged: 0,
                starts_non_finite,
                solution_objective_spread: None,
                solution_parameter_spread: None,
                diagonal_curvature: None,
                expected_fisher_information: None,
                at_parameter_boundary: false,
                note: Some(note.into()),
            },
        }
    }
}

/// Evaluate expected Fisher information for a proposed category design before
/// collecting observations. `thresholds` are strictly ordered evidence cut-points;
/// the model uses noise N(0,1) and signal N(mu,sigma^2). The returned local
/// Wald standard errors are design expectations, not empirical intervals.
pub fn expected_fisher_information_for_design(
    mu: f64,
    sigma: f64,
    thresholds: &[f64],
    signal_trials: u64,
    noise_trials: u64,
) -> Option<FisherInformationDiagnostics> {
    if !mu.is_finite()
        || !sigma.is_finite()
        || sigma <= 0.0
        || thresholds.len() < 2
        || thresholds.iter().any(|value| !value.is_finite())
        || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
        || signal_trials == 0
        || noise_trials == 0
    {
        return None;
    }
    let category_count = thresholds.len() + 1;
    let mut categories = Vec::with_capacity(category_count);
    for index in 0..category_count {
        categories.push(CategoryCount {
            order: index as i16,
            signal: if index == 0 { signal_trials } else { 0 },
            noise: if index == 0 { noise_trials } else { 0 },
        });
    }
    let table = CategoryTable::new(categories).ok()?;
    let mut parameters = Vec::with_capacity(category_count + 1);
    parameters.push(mu);
    parameters.push(sigma.ln());
    parameters.push(thresholds[0]);
    for pair in thresholds.windows(2) {
        parameters.push((pair[1] - pair[0]).ln());
    }
    expected_fisher_information(&table, &parameters, true)
}

/// Fit with multi-start BFGS.
pub fn fit(table: &CategoryTable, options: FitOptions) -> FitResult {
    fit_with_optimizer(table, options, Optimizer::Bfgs)
}

/// Fit with an explicitly selected optimizer. Run BFGS and Nelder-Mead
/// separately on identical frozen input when assessing optimizer agreement.
pub fn fit_with_optimizer(
    table: &CategoryTable,
    options: FitOptions,
    optimizer: Optimizer,
) -> FitResult {
    if !is_identifiable_from_category_support(table) {
        return FitResult::failure(
            FitStatus::NotIdentifiable,
            optimizer,
            "fewer than three categories are populated or one stimulus class occupies only one category",
        );
    }

    let options = sanitized_options(options);
    let mut runs = Vec::with_capacity(options.starts);
    for start_index in 0..options.starts {
        let initial = initial_parameters(table.categories.len(), start_index);
        let run = match optimizer {
            Optimizer::Bfgs => run_bfgs(table, initial, options),
            Optimizer::NelderMead => run_nelder_mead(table, initial, options),
        };
        if run.objective.is_finite() {
            runs.push(run);
        }
    }

    if runs.is_empty() {
        return FitResult::failure_after_starts(
            FitStatus::NotConverged,
            optimizer,
            "all optimizer starts produced non-finite likelihoods",
            options.starts,
            options.starts,
        );
    }

    let starts_converged = runs.iter().filter(|run| run.converged).count();
    let starts_non_finite = options.starts.saturating_sub(runs.len());
    let mut all_objectives: Vec<f64> = runs.iter().map(|run| run.objective).collect();
    all_objectives.sort_by(f64::total_cmp);
    let parameter_spread = converged_parameter_spread(&runs);
    // Select the lowest objective regardless of status. A converged but worse
    // start must not hide a lower-NLL non-converged candidate.
    runs.sort_by(|a, b| a.objective.total_cmp(&b.objective));
    let best = runs.remove(0);
    let curvature = diagonal_curvature_with_correction(table, &best.parameters, options.count_correction);
    let fisher = expected_fisher_information(
        table,
        &best.parameters,
        options.count_correction == CountCorrection::None && best.converged && !best.at_boundary,
    );
    let estimate = decode_parameters(&best.parameters, table.categories.len())
        .and_then(|(mu, sigma, thresholds, _)| UvsdEstimate::new(mu, sigma, thresholds));

    let status = if !best.converged {
        FitStatus::NotConverged
    } else if best.at_boundary {
        FitStatus::Boundary
    } else if fisher.as_ref().map(|value| value.weakly_identified).unwrap_or(true) {
        FitStatus::WeaklyIdentified
    } else {
        FitStatus::Converged
    };

    let spread = match (all_objectives.first(), all_objectives.last()) {
        (Some(minimum), Some(maximum)) => Some((maximum - minimum).max(0.0)),
        _ => None,
    };

    FitResult {
        status,
        estimate,
        diagnostics: FitDiagnostics {
            optimizer,
            negative_log_likelihood: Some(best.objective),
            gradient_norm: Some(best.gradient_norm),
            iterations: best.iterations,
            starts_attempted: options.starts,
            starts_converged,
            starts_non_finite,
            solution_objective_spread: spread,
            solution_parameter_spread: parameter_spread,
            diagonal_curvature: curvature,
            expected_fisher_information: fisher,
            at_parameter_boundary: best.at_boundary,
            note: if status == FitStatus::Converged {
                None
            } else {
                Some("inspect boundary and convergence diagnostics; this result is not a qualified interior estimate".to_owned())
            },
        },
    }
}

/// Validate raw counts and return NotEvaluable rather than panicking.
pub fn fit_ordered_counts(
    categories: Vec<CategoryCount>,
    options: FitOptions,
    optimizer: Optimizer,
) -> FitResult {
    match CategoryTable::new(categories) {
        Ok(table) => fit_with_optimizer(&table, options, optimizer),
        Err(error) => FitResult::failure(FitStatus::NotEvaluable, optimizer, error.to_string()),
    }
}

/// Unsmoothed negative multinomial log likelihood for a supplied parameter set.
/// Returns None for invalid parameters or a non-finite positive-count cell.
/// Unsmoothed negative multinomial log likelihood for a supplied parameter set.
/// Returns None for invalid parameters or a non-finite positive-count cell.
/// Unsmoothed negative multinomial log likelihood for a supplied parameter set.
pub fn negative_log_likelihood(
    table: &CategoryTable,
    estimate: &UvsdEstimate,
) -> Option<f64> {
    negative_log_likelihood_with_correction(table, estimate, CountCorrection::None)
}

/// Negative multinomial log likelihood with a separately named count correction.
/// The primary likelihood is always available through negative_log_likelihood().
pub fn negative_log_likelihood_with_correction(
    table: &CategoryTable,
    estimate: &UvsdEstimate,
    correction: CountCorrection,
) -> Option<f64> {
    if !estimate.mu.is_finite()
        || !estimate.sigma.is_finite()
        || estimate.sigma <= 0.0
        || estimate.thresholds.len() + 1 != table.categories.len()
        || estimate.thresholds.windows(2).any(|pair| pair[0] >= pair[1])
        || estimate.thresholds.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let additive = correction.additive_count(table.categories.len());
    let mut nll = 0.0;
    for (index, row) in table.categories.iter().enumerate() {
        let lower = if index == 0 { f64::NEG_INFINITY } else { estimate.thresholds[index - 1] };
        let upper = if index == estimate.thresholds.len() {
            f64::INFINITY
        } else {
            estimate.thresholds[index]
        };
        let noise_count = row.noise as f64 + additive;
        let signal_count = row.signal as f64 + additive;
        if noise_count > 0.0 {
            let log_probability = normal_log_interval(lower, upper);
            if !log_probability.is_finite() {
                return None;
            }
            nll -= noise_count * log_probability;
        }
        if signal_count > 0.0 {
            let signal_lower = standardize_boundary(lower, estimate.mu, estimate.sigma);
            let signal_upper = standardize_boundary(upper, estimate.mu, estimate.sigma);
            let log_probability = normal_log_interval(signal_lower, signal_upper);
            if !log_probability.is_finite() {
                return None;
            }
            nll -= signal_count * log_probability;
        }
    }
    if nll.is_finite() { Some(nll) } else { None }
}

/// Analytic gradient of the unsmoothed negative multinomial log likelihood.
pub fn negative_log_likelihood_gradient(
    table: &CategoryTable,
    estimate: &UvsdEstimate,
) -> Option<UvsdGradient> {
    negative_log_likelihood_gradient_with_correction(table, estimate, CountCorrection::None)
}

/// Analytic gradient with an explicitly selected count correction.
pub fn negative_log_likelihood_gradient_with_correction(
    table: &CategoryTable,
    estimate: &UvsdEstimate,
    correction: CountCorrection,
) -> Option<UvsdGradient> {
    if !estimate.mu.is_finite()
        || !estimate.sigma.is_finite()
        || estimate.sigma <= 0.0
        || estimate.thresholds.len() + 1 != table.categories.len()
        || estimate.thresholds.windows(2).any(|pair| pair[0] >= pair[1])
        || estimate.thresholds.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let mut gradient = UvsdGradient {
        mu: 0.0,
        log_sigma: 0.0,
        thresholds: vec![0.0; estimate.thresholds.len()],
    };
    let inverse_sigma = 1.0 / estimate.sigma;
    let additive = correction.additive_count(table.categories.len());

    for (index, row) in table.categories.iter().enumerate() {
        let lower = if index == 0 { f64::NEG_INFINITY } else { estimate.thresholds[index - 1] };
        let upper = if index == estimate.thresholds.len() {
            f64::INFINITY
        } else {
            estimate.thresholds[index]
        };

        let noise_count = row.noise as f64 + additive;
        if noise_count > 0.0 {
            let log_probability = normal_log_interval(lower, upper);
            if !log_probability.is_finite() {
                return None;
            }
            let lower_ratio = density_probability_ratio(lower, log_probability);
            let upper_ratio = density_probability_ratio(upper, log_probability);
            if !lower_ratio.is_finite() || !upper_ratio.is_finite() {
                return None;
            }
            if index > 0 {
                gradient.thresholds[index - 1] += noise_count * lower_ratio;
            }
            if index < estimate.thresholds.len() {
                gradient.thresholds[index] -= noise_count * upper_ratio;
            }
        }

        let signal_count = row.signal as f64 + additive;
        if signal_count > 0.0 {
            let signal_lower = standardize_boundary(lower, estimate.mu, estimate.sigma);
            let signal_upper = standardize_boundary(upper, estimate.mu, estimate.sigma);
            let log_probability = normal_log_interval(signal_lower, signal_upper);
            if !log_probability.is_finite() {
                return None;
            }
            let lower_ratio = density_probability_ratio(signal_lower, log_probability);
            let upper_ratio = density_probability_ratio(signal_upper, log_probability);
            if !lower_ratio.is_finite() || !upper_ratio.is_finite() {
                return None;
            }

            gradient.mu -= signal_count * (lower_ratio - upper_ratio) * inverse_sigma;
            gradient.log_sigma -= signal_count
                * (finite_z_ratio_product(signal_lower, lower_ratio)
                    - finite_z_ratio_product(signal_upper, upper_ratio));
            if index > 0 {
                gradient.thresholds[index - 1] += signal_count * lower_ratio * inverse_sigma;
            }
            if index < estimate.thresholds.len() {
                gradient.thresholds[index] -= signal_count * upper_ratio * inverse_sigma;
            }
        }
    }

    if gradient.mu.is_finite()
        && gradient.log_sigma.is_finite()
        && gradient.thresholds.iter().all(|value| value.is_finite())
    {
        Some(gradient)
    } else {
        None
    }
}

fn density_probability_ratio(z: f64, log_probability: f64) -> f64 {
    if !z.is_finite() {
        return 0.0;
    }
    let log_density = -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln();
    (log_density - log_probability).exp()
}

fn finite_z_ratio_product(z: f64, ratio: f64) -> f64 {
    if z.is_finite() { z * ratio } else { 0.0 }
}

fn expected_fisher_information(
    table: &CategoryTable,
    parameters: &[f64],
    allow_wald: bool,
) -> Option<FisherInformationDiagnostics> {
    let (mu, sigma, thresholds, _) = decode_parameters(parameters, table.categories.len())?;
    let _estimate = UvsdEstimate::new(mu, sigma, thresholds.clone())?;
    let dimension = parameters.len();
    let mut information = vec![vec![0.0; dimension]; dimension];
    let inverse_sigma = 1.0 / sigma;

    for index in 0..table.categories.len() {
        let lower = if index == 0 { f64::NEG_INFINITY } else { thresholds[index - 1] };
        let upper = if index == thresholds.len() { f64::INFINITY } else { thresholds[index] };

        let noise_logp = normal_log_interval(lower, upper);
        if !noise_logp.is_finite() { return None; }
        let noise_p = noise_logp.exp();
        let mut noise_score = vec![0.0; 2 + thresholds.len()];
        if index > 0 {
            noise_score[2 + index - 1] =
                -density_probability_ratio(lower, noise_logp);
        }
        if index < thresholds.len() {
            noise_score[2 + index] =
                density_probability_ratio(upper, noise_logp);
        }
        let noise_score = natural_to_internal_score(&noise_score, parameters, thresholds.len());
        add_information_outer(&mut information, table.noise_total as f64 * noise_p, &noise_score)?;

        let signal_lower = standardize_boundary(lower, mu, sigma);
        let signal_upper = standardize_boundary(upper, mu, sigma);
        let signal_logp = normal_log_interval(signal_lower, signal_upper);
        if !signal_logp.is_finite() { return None; }
        let signal_p = signal_logp.exp();
        let lower_ratio = density_probability_ratio(signal_lower, signal_logp);
        let upper_ratio = density_probability_ratio(signal_upper, signal_logp);
        let mut signal_score = vec![0.0; 2 + thresholds.len()];
        // d log P(category | signal) / d mu = (phi(z_lower)-phi(z_upper))/(sigma*P).
        signal_score[0] = (lower_ratio - upper_ratio) * inverse_sigma;
        signal_score[1] =
            finite_z_ratio_product(signal_lower, lower_ratio)
            - finite_z_ratio_product(signal_upper, upper_ratio);
        if index > 0 {
            signal_score[2 + index - 1] = -lower_ratio * inverse_sigma;
        }
        if index < thresholds.len() {
            signal_score[2 + index] = upper_ratio * inverse_sigma;
        }
        let signal_score = natural_to_internal_score(&signal_score, parameters, thresholds.len());
        add_information_outer(&mut information, table.signal_total as f64 * signal_p, &signal_score)?;
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen_decomposition(information)?;
    let maximum = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if !maximum.is_finite() || maximum <= 0.0 { return None; }
    let rank_tolerance = maximum * 1.0e-10;
    let effective_rank = eigenvalues.iter().filter(|value| **value > rank_tolerance).count();
    let minimum_positive = eigenvalues.iter().copied()
        .filter(|value| *value > 0.0).fold(f64::INFINITY, f64::min);
    let minimum_effective = eigenvalues.iter().copied()
        .filter(|value| *value > rank_tolerance).fold(f64::INFINITY, f64::min);
    let condition_number = if effective_rank == dimension && minimum_effective.is_finite() {
        Some(maximum / minimum_effective)
    } else {
        None
    };
    let relative_minimum_eigenvalue = if minimum_positive.is_finite() {
        Some(minimum_positive / maximum)
    } else {
        None
    };
    let weakly_identified = effective_rank < dimension
        || condition_number.map(|value| value > 1.0e10).unwrap_or(true);

    let covariance = if weakly_identified || !allow_wald {
        None
    } else {
        inverse_from_eigen(&eigenvalues, &eigenvectors)
    };
    let (mu_se, sigma_se, sigma_ci, da_se, da_ci) = if let Some(cov) = covariance {
        let mu_se = cov[0][0].max(0.0).sqrt();
        let log_sigma_se = cov[1][1].max(0.0).sqrt();
        let sigma_se = sigma * log_sigma_se;
        let sigma_ci = [
            (parameters[1] - 1.959_963_984_540_054 * log_sigma_se).exp(),
            (parameters[1] + 1.959_963_984_540_054 * log_sigma_se).exp(),
        ];
        let scale = ((1.0 + sigma * sigma) / 2.0).sqrt();
        let mut gradient = vec![0.0; dimension];
        gradient[0] = 1.0 / scale;
        gradient[1] = -mu * sigma * sigma / (2.0 * scale.powi(3));
        let mut variance = 0.0;
        for row in 0..dimension {
            for col in 0..dimension {
                variance += gradient[row] * cov[row][col] * gradient[col];
            }
        }
        let da_se = variance.max(0.0).sqrt();
        let da = mu / scale;
        (
            Some(mu_se),
            Some(sigma_se),
            Some(sigma_ci),
            Some(da_se),
            Some([da - 1.959_963_984_540_054 * da_se, da + 1.959_963_984_540_054 * da_se]),
        )
    } else {
        (None, None, None, None, None)
    };

    Some(FisherInformationDiagnostics {
        eigenvalues_ascending: eigenvalues,
        parameter_dimension: dimension,
        effective_rank,
        condition_number,
        relative_minimum_eigenvalue,
        weakly_identified,
        mu_standard_error: mu_se,
        sigma_standard_error: sigma_se,
        sigma_wald_95: sigma_ci,
        d_a_standard_error: da_se,
        d_a_wald_95: da_ci,
    })
}

fn natural_to_internal_score(
    natural: &[f64],
    parameters: &[f64],
    threshold_count: usize,
) -> Vec<f64> {
    let mut score = vec![0.0; parameters.len()];
    score[0] = natural[0];
    score[1] = natural[1];
    score[2] = natural[2..].iter().sum();
    for gap_index in 0..threshold_count.saturating_sub(1) {
        let gap = parameters[3 + gap_index].exp();
        score[3 + gap_index] =
            gap * natural[2 + gap_index + 1..].iter().sum::<f64>();
    }
    score
}

fn add_information_outer(
    information: &mut [Vec<f64>],
    weight: f64,
    score: &[f64],
) -> Option<()> {
    if !weight.is_finite() || weight < 0.0 || score.len() != information.len() {
        return None;
    }
    if weight == 0.0 {
        return Some(());
    }
    if score.iter().any(|value| !value.is_finite()) {
        return None;
    }
    for row in 0..score.len() {
        for col in 0..score.len() {
            information[row][col] += weight * score[row] * score[col];
            if !information[row][col].is_finite() { return None; }
        }
    }
    Some(())
}

fn symmetric_eigen_decomposition(
    mut matrix: Vec<Vec<f64>>,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let dimension = matrix.len();
    if dimension == 0 || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    let mut vectors = vec![vec![0.0; dimension]; dimension];
    for index in 0..dimension { vectors[index][index] = 1.0; }

    let mut converged = false;
    for _ in 0..(100 * dimension * dimension).max(1) {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut largest = 0.0_f64;
        let mut diagonal_scale = 1.0_f64;
        for row in 0..dimension {
            diagonal_scale = diagonal_scale.max(matrix[row][row].abs());
            for col in (row + 1)..dimension {
                if matrix[row][col].abs() > largest {
                    largest = matrix[row][col].abs();
                    p = row;
                    q = col;
                }
            }
        }
        if largest <= 1.0e-13 * diagonal_scale {
            converged = true;
            break;
        }
        let apq = matrix[p][q];
        if !apq.is_finite() || apq == 0.0 { return None; }
        let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + t * t).sqrt();
        let sine = t * cosine;
        let pp = matrix[p][p];
        let qq = matrix[q][q];
        matrix[p][p] = pp - t * apq;
        matrix[q][q] = qq + t * apq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for index in 0..dimension {
            if index != p && index != q {
                let ip = matrix[index][p];
                let iq = matrix[index][q];
                matrix[index][p] = cosine * ip - sine * iq;
                matrix[p][index] = matrix[index][p];
                matrix[index][q] = sine * ip + cosine * iq;
                matrix[q][index] = matrix[index][q];
            }
            let vp = vectors[index][p];
            let vq = vectors[index][q];
            vectors[index][p] = cosine * vp - sine * vq;
            vectors[index][q] = sine * vp + cosine * vq;
        }
    }

    if !converged {
        return None;
    }

    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|left, right| matrix[*left][*left].total_cmp(&matrix[*right][*right]));
    let eigenvalues = order.iter().map(|index| matrix[*index][*index]).collect();
    let mut sorted_vectors = vec![vec![0.0; dimension]; dimension];
    for (column, source) in order.iter().copied().enumerate() {
        for row in 0..dimension {
            sorted_vectors[row][column] = vectors[row][source];
        }
    }
    Some((eigenvalues, sorted_vectors))
}

fn inverse_from_eigen(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
    let dimension = eigenvalues.len();
    if eigenvectors.len() != dimension || eigenvectors.iter().any(|row| row.len() != dimension) {
        return None;
    }
    let mut inverse = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            for eigen in 0..dimension {
                let value = eigenvalues[eigen];
                if value <= 0.0 || !value.is_finite() { return None; }
                inverse[row][col] +=
                    eigenvectors[row][eigen] * eigenvectors[col][eigen] / value;
            }
        }
    }
    Some(inverse)
}

fn converged_parameter_spread(runs: &[RunResult]) -> Option<Vec<f64>> {
    let mut converged = runs.iter().filter(|run| run.converged);
    let first = converged.next()?;
    let mut minimum = first.parameters.clone();
    let mut maximum = first.parameters.clone();
    let mut count = 1usize;
    for run in converged {
        count += 1;
        for index in 0..minimum.len() {
            minimum[index] = minimum[index].min(run.parameters[index]);
            maximum[index] = maximum[index].max(run.parameters[index]);
        }
    }
    if count < 2 {
        return None;
    }
    Some(
        minimum
            .iter()
            .zip(maximum.iter())
            .map(|(low, high)| high - low)
            .collect(),
    )
}

fn diagonal_curvature(table: &CategoryTable, parameters: &[f64]) -> Option<Vec<f64>> {
    diagonal_curvature_with_correction(table, parameters, CountCorrection::None)
}

fn diagonal_curvature_with_correction(
    table: &CategoryTable,
    parameters: &[f64],
    correction: CountCorrection,
) -> Option<Vec<f64>> {
    let (lower, upper) = parameter_bounds(table.categories.len());
    let center = objective_with_correction(table, parameters, correction);
    if !center.is_finite() {
        return None;
    }
    let mut curvature = Vec::with_capacity(parameters.len());
    for index in 0..parameters.len() {
        let step = 1.0e-3 * parameters[index].abs().max(1.0);
        if parameters[index] - step <= lower[index]
            || parameters[index] + step >= upper[index]
        {
            return None;
        }
        let mut plus = parameters.to_vec();
        let mut minus = parameters.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let high = objective_with_correction(table, &plus, correction);
        let low = objective_with_correction(table, &minus, correction);
        let value = (high - 2.0 * center + low) / (step * step);
        if !value.is_finite() {
            return None;
        }
        curvature.push(value);
    }
    Some(curvature)
}

fn standardize_boundary(value: f64, mu: f64, sigma: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else if value == f64::INFINITY {
        f64::INFINITY
    } else {
        (value - mu) / sigma
    }
}

fn is_identifiable_from_category_support(table: &CategoryTable) -> bool {
    let populated = table.categories.iter().filter(|row| row.signal > 0 || row.noise > 0).count();
    let signal_categories = table.categories.iter().filter(|row| row.signal > 0).count();
    let noise_categories = table.categories.iter().filter(|row| row.noise > 0).count();
    populated >= 3 && signal_categories >= 2 && noise_categories >= 2
}

fn sanitized_options(options: FitOptions) -> FitOptions {
    FitOptions {
        max_iterations: options.max_iterations.clamp(20, 20_000),
        gradient_tolerance: if options.gradient_tolerance.is_finite()
            && options.gradient_tolerance > 0.0
        {
            options.gradient_tolerance.clamp(1.0e-10, 1.0e-2)
        } else {
            FitOptions::default().gradient_tolerance
        },
        starts: options.starts.clamp(1, 16),
        count_correction: options.count_correction,
    }
}

fn parameter_bounds(category_count: usize) -> (Vec<f64>, Vec<f64>) {
    let dimension = category_count + 1;
    let mut lower = vec![-12.0; dimension];
    let mut upper = vec![12.0; dimension];
    lower[0] = -MU_BOUND;
    upper[0] = MU_BOUND;
    lower[1] = -LOG_SIGMA_BOUND;
    upper[1] = LOG_SIGMA_BOUND;
    lower[2] = -FIRST_THRESHOLD_BOUND;
    upper[2] = FIRST_THRESHOLD_BOUND;
    for index in 3..dimension {
        lower[index] = LOG_GAP_MIN;
        upper[index] = LOG_GAP_MAX;
    }
    (lower, upper)
}

fn project(parameters: &mut [f64], lower: &[f64], upper: &[f64]) {
    for index in 0..parameters.len() {
        parameters[index] = parameters[index].clamp(lower[index], upper[index]);
    }
}

fn decode_parameters(
    parameters: &[f64],
    category_count: usize,
) -> Option<(f64, f64, Vec<f64>, bool)> {
    if parameters.len() != category_count + 1 || parameters.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (lower, upper) = parameter_bounds(category_count);
    let mut at_boundary = false;
    for index in 0..parameters.len() {
        if parameters[index] <= lower[index] + 1.0e-5
            || parameters[index] >= upper[index] - 1.0e-5
        {
            at_boundary = true;
        }
    }

    let mu = parameters[0].clamp(-MU_BOUND, MU_BOUND);
    let sigma = parameters[1].clamp(-LOG_SIGMA_BOUND, LOG_SIGMA_BOUND).exp();
    let mut thresholds = Vec::with_capacity(category_count - 1);
    let first = parameters[2].clamp(-FIRST_THRESHOLD_BOUND, FIRST_THRESHOLD_BOUND);
    thresholds.push(first);
    for value in parameters.iter().skip(3) {
        let gap = value.clamp(LOG_GAP_MIN, LOG_GAP_MAX).exp();
        let next = thresholds.last().copied().unwrap_or(first) + gap;
        if !next.is_finite() {
            return None;
        }
        thresholds.push(next);
    }
    Some((mu, sigma, thresholds, at_boundary))
}

fn objective(table: &CategoryTable, parameters: &[f64]) -> f64 {
    objective_with_correction(table, parameters, CountCorrection::None)
}

fn objective_with_correction(
    table: &CategoryTable,
    parameters: &[f64],
    correction: CountCorrection,
) -> f64 {
    match decode_parameters(parameters, table.categories.len()) {
        Some((mu, sigma, thresholds, _)) => {
            match UvsdEstimate::new(mu, sigma, thresholds)
                .and_then(|estimate| negative_log_likelihood_with_correction(table, &estimate, correction))
            {
                Some(value) => value,
                None => f64::INFINITY,
            }
        }
        None => f64::INFINITY,
    }
}

fn initial_parameters(category_count: usize, start_index: usize) -> Vec<f64> {
    let starts = [
        (0.75, 0.0, 0.0),
        (1.50, 0.0, 0.0),
        (2.50, 0.0, 0.0),
        (1.50, -0.45, 0.25),
        (1.50, 0.45, -0.25),
        (-0.75, 0.0, 0.0),
        (3.00, -0.7, 0.5),
        (0.50, 0.7, -0.5),
        (2.00, 0.0, 0.75),
        (1.00, 0.0, -0.75),
        (4.00, 0.3, 0.0),
        (-1.50, 0.0, 0.25),
        (0.25, -0.25, -0.25),
        (2.00, 0.7, 0.5),
        (3.00, -0.4, -0.5),
        (-2.00, 0.4, 0.0),
    ];
    let (mu, log_sigma, shift) = starts[start_index % starts.len()];
    let mut values = Vec::with_capacity(category_count + 1);
    values.push(mu);
    values.push(log_sigma);

    let mut cuts = Vec::with_capacity(category_count - 1);
    for index in 1..category_count {
        let probability = index as f64 / category_count as f64;
        cuts.push(normal_quantile(probability) + shift);
    }
    values.push(cuts[0]);
    for pair in cuts.windows(2) {
        values.push((pair[1] - pair[0]).ln());
    }
    values
}

#[derive(Debug, Clone)]
struct RunResult {
    parameters: Vec<f64>,
    objective: f64,
    gradient_norm: f64,
    iterations: usize,
    converged: bool,
    at_boundary: bool,
}

fn parameter_gradient(table: &CategoryTable, parameters: &[f64]) -> Vec<f64> {
    parameter_gradient_with_correction(table, parameters, CountCorrection::None)
}

fn parameter_gradient_with_correction(
    table: &CategoryTable,
    parameters: &[f64],
    correction: CountCorrection,
) -> Vec<f64> {
    let Some((mu, sigma, thresholds, _)) = decode_parameters(parameters, table.categories.len()) else {
        return vec![f64::NAN; parameters.len()];
    };
    let Some(estimate) = UvsdEstimate::new(mu, sigma, thresholds) else {
        return vec![f64::NAN; parameters.len()];
    };
    let Some(natural) = negative_log_likelihood_gradient_with_correction(table, &estimate, correction) else {
        let (lower, upper) = parameter_bounds(table.categories.len());
        return numeric_gradient_with_correction(table, parameters, &lower, &upper, correction);
    };

    let mut gradient = vec![0.0; parameters.len()];
    gradient[0] = natural.mu;
    gradient[1] = natural.log_sigma;
    gradient[2] = natural.thresholds.iter().sum();
    for gap_index in 0..natural.thresholds.len().saturating_sub(1) {
        let gap = parameters[3 + gap_index].exp();
        let downstream_gradient: f64 = natural.thresholds.iter().skip(gap_index + 1).sum();
        gradient[3 + gap_index] = gap * downstream_gradient;
    }
    gradient
}

fn numeric_gradient(table: &CategoryTable, x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    numeric_gradient_with_correction(table, x, lower, upper, CountCorrection::None)
}

fn numeric_gradient_with_correction(
    table: &CategoryTable,
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    correction: CountCorrection,
) -> Vec<f64> {
    let mut gradient = vec![0.0; x.len()];
    for index in 0..x.len() {
        let step = 1.0e-5 * x[index].abs().max(1.0);
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[index] = (x[index] + step).min(upper[index]);
        minus[index] = (x[index] - step).max(lower[index]);
        let width = plus[index] - minus[index];
        if width > 0.0 {
            gradient[index] = (objective_with_correction(table, &plus, correction)
                - objective_with_correction(table, &minus, correction)) / width;
        }
    }
    gradient
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn run_bfgs(table: &CategoryTable, mut x: Vec<f64>, options: FitOptions) -> RunResult {
    let dimension = x.len();
    let (lower, upper) = parameter_bounds(table.categories.len());
    project(&mut x, &lower, &upper);
    let mut inverse_hessian = vec![vec![0.0; dimension]; dimension];
    for (index, row) in inverse_hessian.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    let mut value = objective_with_correction(table, &x, options.count_correction);
    let mut gradient = parameter_gradient_with_correction(table, &x, options.count_correction);
    let mut converged = false;
    let mut iterations = 0usize;

    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;
        let gradient_norm = vector_norm(&gradient);
        if gradient_norm.is_finite() && gradient_norm <= options.gradient_tolerance {
            converged = true;
            break;
        }
        if !value.is_finite() || !gradient_norm.is_finite() {
            break;
        }

        let mut direction = vec![0.0; dimension];
        for row in 0..dimension {
            direction[row] = -dot(&inverse_hessian[row], &gradient);
        }
        let mut directional_derivative = dot(&gradient, &direction);
        if !directional_derivative.is_finite() || directional_derivative >= -1.0e-12 {
            for row in 0..dimension {
                direction[row] = -gradient[row];
            }
            directional_derivative = -dot(&gradient, &gradient);
            inverse_hessian.fill(vec![0.0; dimension]);
            for index in 0..dimension {
                inverse_hessian[index][index] = 1.0;
            }
        }

        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let mut candidate: Vec<f64> = x
                .iter()
                .zip(direction.iter())
                .map(|(current, delta)| current + step * delta)
                .collect();
            project(&mut candidate, &lower, &upper);
            let candidate_value = objective_with_correction(table, &candidate, options.count_correction);
            if candidate_value.is_finite()
                && candidate_value <= value + 1.0e-4 * step * directional_derivative
            {
                accepted = Some((candidate, candidate_value));
                break;
            }
            step *= 0.5;
        }
        let Some((next_x, next_value)) = accepted else {
            break;
        };

        let next_gradient = parameter_gradient_with_correction(table, &next_x, options.count_correction);
        let s: Vec<f64> = next_x.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = next_gradient.iter().zip(gradient.iter()).map(|(a, b)| a - b).collect();
        let ys = dot(&y, &s);
        if ys > 1.0e-12 * vector_norm(&s) * vector_norm(&y) {
            let rho = 1.0 / ys;
            let hy: Vec<f64> = inverse_hessian
                .iter()
                .map(|row| dot(row, &y))
                .collect();
            let yhy = dot(&y, &hy);
            for row in 0..dimension {
                for col in 0..dimension {
                    inverse_hessian[row][col] +=
                        (1.0 + yhy * rho) * rho * s[row] * s[col]
                            - rho * (s[row] * hy[col] + hy[row] * s[col]);
                }
            }
        }

        let relative_change = (value - next_value).abs() / (1.0 + value.abs());
        x = next_x;
        value = next_value;
        gradient = next_gradient;
        if relative_change <= options.gradient_tolerance * 1.0e-2
            && vector_norm(&gradient) <= options.gradient_tolerance.sqrt()
        {
            converged = true;
            break;
        }
    }

    let gradient_norm = vector_norm(&parameter_gradient_with_correction(table, &x, options.count_correction));
    let at_boundary = decode_parameters(&x, table.categories.len())
        .map(|value| value.3)
        .unwrap_or(true);
    RunResult {
        parameters: x,
        objective: value,
        gradient_norm,
        iterations,
        converged: converged || gradient_norm <= options.gradient_tolerance,
        at_boundary,
    }
}

#[derive(Clone)]
struct Vertex {
    point: Vec<f64>,
    value: f64,
}

fn run_nelder_mead(table: &CategoryTable, initial: Vec<f64>, options: FitOptions) -> RunResult {
    let dimension = initial.len();
    let (lower, upper) = parameter_bounds(table.categories.len());
    let mut simplex = Vec::with_capacity(dimension + 1);
    simplex.push(Vertex {
        value: objective_with_correction(table, &initial, options.count_correction),
        point: initial.clone(),
    });
    for index in 0..dimension {
        let mut point = initial.clone();
        point[index] += 0.15 * (1.0 + point[index].abs());
        project(&mut point, &lower, &upper);
        simplex.push(Vertex {
            value: objective_with_correction(table, &point, options.count_correction),
            point,
        });
    }

    let mut converged = false;
    let mut iterations = 0usize;
    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;
        simplex.sort_by(|left, right| left.value.total_cmp(&right.value));
        let best_value = simplex[0].value;
        let worst_value = simplex[dimension].value;

        let mut max_spread = 0.0f64;
        for vertex in simplex.iter().skip(1) {
            for (left, right) in vertex.point.iter().zip(simplex[0].point.iter()) {
                max_spread = max_spread.max((left - right).abs());
            }
        }
        if max_spread <= options.gradient_tolerance.sqrt()
            && (worst_value - best_value).abs()
                <= options.gradient_tolerance * (1.0 + best_value.abs())
        {
            converged = true;
            break;
        }

        let mut centroid = vec![0.0; dimension];
        for vertex in simplex.iter().take(dimension) {
            for index in 0..dimension {
                centroid[index] += vertex.point[index] / dimension as f64;
            }
        }

        let worst = &simplex[dimension].point;
        let mut reflected: Vec<f64> = centroid
            .iter()
            .zip(worst.iter())
            .map(|(center, bad)| center + (center - bad))
            .collect();
        project(&mut reflected, &lower, &upper);
        let reflected_value = objective_with_correction(table, &reflected, options.count_correction);

        if reflected_value < best_value {
            let mut expanded: Vec<f64> = centroid
                .iter()
                .zip(reflected.iter())
                .map(|(center, reflected)| center + 2.0 * (reflected - center))
                .collect();
            project(&mut expanded, &lower, &upper);
            let expanded_value = objective_with_correction(table, &expanded, options.count_correction);
            if expanded_value < reflected_value {
                simplex[dimension] = Vertex { point: expanded, value: expanded_value };
            } else {
                simplex[dimension] = Vertex { point: reflected, value: reflected_value };
            }
        } else if reflected_value < simplex[dimension - 1].value {
            simplex[dimension] = Vertex { point: reflected, value: reflected_value };
        } else {
            let outside = reflected_value < worst_value;
            let target = if outside { &reflected } else { worst };
            let mut contracted: Vec<f64> = centroid
                .iter()
                .zip(target.iter())
                .map(|(center, target)| center + 0.5 * (target - center))
                .collect();
            project(&mut contracted, &lower, &upper);
            let contracted_value = objective_with_correction(table, &contracted, options.count_correction);
            let should_accept = if outside {
                contracted_value <= reflected_value
            } else {
                contracted_value < worst_value
            };
            if should_accept {
                simplex[dimension] = Vertex { point: contracted, value: contracted_value };
            } else {
                let best = simplex[0].point.clone();
                for vertex in simplex.iter_mut().skip(1) {
                    for index in 0..dimension {
                        vertex.point[index] = best[index] + 0.5 * (vertex.point[index] - best[index]);
                    }
                    project(&mut vertex.point, &lower, &upper);
                    vertex.value = objective_with_correction(table, &vertex.point, options.count_correction);
                }
            }
        }
    }

    simplex.sort_by(|left, right| left.value.total_cmp(&right.value));
    let best = &simplex[0];
    let gradient_norm = vector_norm(&parameter_gradient_with_correction(table, &best.point, options.count_correction));
    let at_boundary = decode_parameters(&best.point, table.categories.len())
        .map(|value| value.3)
        .unwrap_or(true);
    RunResult {
        parameters: best.point.clone(),
        objective: best.value,
        gradient_norm,
        iterations,
        converged,
        at_boundary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_table(mu: f64, sigma: f64, trials_per_class: u64) -> CategoryTable {
        let cuts = [-1.0, 0.0, 1.0];
        let mut categories = Vec::new();
        let mut lower = f64::NEG_INFINITY;
        let mut signal_used = 0u64;
        let mut noise_used = 0u64;
        for index in 0..4 {
            let upper = if index == 3 { f64::INFINITY } else { cuts[index] };
            let noise_probability = normal_log_interval(lower, upper).exp();
            let signal_probability = normal_log_interval(
                if lower.is_infinite() { lower } else { (lower - mu) / sigma },
                if upper.is_infinite() { upper } else { (upper - mu) / sigma },
            )
            .exp();
            let noise = if index == 3 {
                trials_per_class - noise_used
            } else {
                (noise_probability * trials_per_class as f64).round() as u64
            };
            let signal = if index == 3 {
                trials_per_class - signal_used
            } else {
                (signal_probability * trials_per_class as f64).round() as u64
            };
            noise_used += noise;
            signal_used += signal;
            categories.push(CategoryCount { order: index as i16, signal, noise });
            lower = upper;
        }
        CategoryTable::new(categories).unwrap()
    }

    #[test]
    fn table_requires_complete_order_and_both_classes() {
        assert_eq!(
            CategoryTable::new(vec![
                CategoryCount { order: 0, signal: 1, noise: 1 },
                CategoryCount { order: 1, signal: 1, noise: 1 },
            ]),
            Err(TableError::TooFewCategories)
        );
        assert_eq!(
            CategoryTable::new(vec![
                CategoryCount { order: 1, signal: 1, noise: 1 },
                CategoryCount { order: 1, signal: 1, noise: 1 },
                CategoryCount { order: 2, signal: 1, noise: 1 },
            ]),
            Err(TableError::NonIncreasingCategoryOrder)
        );
    }

    #[test]
    fn signal_category_fisher_mu_score_matches_log_probability_derivative() {
        let mu = 0.8;
        let sigma = 1.3;
        let lower = -0.5;
        let upper = 0.7;
        let lower_z = (lower - mu) / sigma;
        let upper_z = (upper - mu) / sigma;
        let log_probability = normal_log_interval(lower_z, upper_z);
        let lower_ratio = density_probability_ratio(lower_z, log_probability);
        let upper_ratio = density_probability_ratio(upper_z, log_probability);
        let analytic = (lower_ratio - upper_ratio) / sigma;

        let epsilon = 1.0e-6;
        let log_probability_at = |candidate_mu: f64| {
            normal_log_interval(
                (lower - candidate_mu) / sigma,
                (upper - candidate_mu) / sigma,
            )
        };
        let numeric = (log_probability_at(mu + epsilon) - log_probability_at(mu - epsilon))
            / (2.0 * epsilon);
        assert!((analytic - numeric).abs() < 1.0e-7, "{analytic} vs {numeric}");
    }

    #[test]
    fn analytic_likelihood_gradient_matches_central_differences() {
        let table = synthetic_table(1.35, 1.45, 120_000);
        let estimate = UvsdEstimate::new(1.2, 1.3, vec![-0.8, 0.1, 1.2]).unwrap();
        let gradient = negative_log_likelihood_gradient(&table, &estimate).unwrap();
        let epsilon = 1.0e-5;

        let mu_plus = UvsdEstimate::new(estimate.mu + epsilon, estimate.sigma, estimate.thresholds.clone()).unwrap();
        let mu_minus = UvsdEstimate::new(estimate.mu - epsilon, estimate.sigma, estimate.thresholds.clone()).unwrap();
        let mu_fd = (negative_log_likelihood(&table, &mu_plus).unwrap()
            - negative_log_likelihood(&table, &mu_minus).unwrap()) / (2.0 * epsilon);
        assert!((gradient.mu - mu_fd).abs() < 1.0e-3 * (1.0 + mu_fd.abs()));

        let sigma_plus = UvsdEstimate::new(estimate.mu, estimate.sigma * epsilon.exp(), estimate.thresholds.clone()).unwrap();
        let sigma_minus = UvsdEstimate::new(estimate.mu, estimate.sigma * (-epsilon).exp(), estimate.thresholds.clone()).unwrap();
        let sigma_fd = (negative_log_likelihood(&table, &sigma_plus).unwrap()
            - negative_log_likelihood(&table, &sigma_minus).unwrap()) / (2.0 * epsilon);
        assert!((gradient.log_sigma - sigma_fd).abs() < 1.0e-3 * (1.0 + sigma_fd.abs()));

        for index in 0..estimate.thresholds.len() {
            let mut plus = estimate.thresholds.clone();
            let mut minus = estimate.thresholds.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus = UvsdEstimate::new(estimate.mu, estimate.sigma, plus).unwrap();
            let minus = UvsdEstimate::new(estimate.mu, estimate.sigma, minus).unwrap();
            let finite_difference = (negative_log_likelihood(&table, &plus).unwrap()
                - negative_log_likelihood(&table, &minus).unwrap()) / (2.0 * epsilon);
            assert!(
                (gradient.thresholds[index] - finite_difference).abs()
                    < 1.0e-3 * (1.0 + finite_difference.abs()),
                "cut {index}: analytic={} finite_difference={finite_difference}",
                gradient.thresholds[index]
            );
        }
    }

    #[test]
    fn transformed_analytic_gradient_matches_parameter_finite_differences() {
        let table = synthetic_table(1.1, 1.4, 80_000);
        let parameters = initial_parameters(table.categories.len(), 3);
        let gradient = parameter_gradient(&table, &parameters);
        let epsilon = 1.0e-5;
        for index in 0..parameters.len() {
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&table, &plus) - objective(&table, &minus)) / (2.0 * epsilon);
            assert!(
                (gradient[index] - finite_difference).abs()
                    < 1.0e-3 * (1.0 + finite_difference.abs()),
                "parameter {index}: analytic={} finite_difference={finite_difference}",
                gradient[index]
            );
        }
    }

    #[test]
    fn recovers_known_unequal_variance_parameters() {
        let table = synthetic_table(1.4, 1.55, 250_000);
        let result = fit(&table, FitOptions::default());
        assert_eq!(result.status, FitStatus::Converged, "{result:?}");
        let estimate = result.estimate.unwrap();
        assert!((estimate.mu - 1.4).abs() < 0.08, "{estimate:?}");
        assert!((estimate.sigma - 1.55).abs() < 0.08, "{estimate:?}");
        let expected_da = 1.4 / ((1.0 + 1.55_f64.powi(2)) / 2.0).sqrt();
        assert!((estimate.d_a - expected_da).abs() < 0.08);
    }

    #[test]
    fn equal_variance_negative_control_is_recoverable() {
        let table = synthetic_table(1.2, 1.0, 250_000);
        let result = fit(&table, FitOptions::default());
        assert!(matches!(result.status, FitStatus::Converged | FitStatus::Boundary), "{result:?}");
        let estimate = result.estimate.unwrap();
        assert!((estimate.sigma - 1.0).abs() < 0.08);
    }

    #[test]
    fn matches_published_miyoshi_reference_example_with_explicit_correction() {
        // Source vectors from kiyomiyoshi/rt_type1_roc README. That reference
        // orders strongest-Yes to strongest-No; this crate orders evidence
        // from strongest-No to strongest-Yes, so reverse both vectors.
        let absent_reference = [10_u64, 7, 16, 27, 29, 10];
        let present_reference = [43_u64, 21, 10, 12, 8, 3];
        let categories = (0..absent_reference.len())
            .map(|index| {
                let reverse = absent_reference.len() - 1 - index;
                CategoryCount {
                    order: index as i16,
                    signal: present_reference[reverse],
                    noise: absent_reference[reverse],
                }
            })
            .collect();
        let table = CategoryTable::new(categories).unwrap();
        let options = FitOptions {
            count_correction: CountCorrection::MiyoshiReferencePerCategory,
            ..FitOptions::default()
        };
        let result = fit(&table, options);
        assert_eq!(result.status, FitStatus::Converged, "{result:?}");
        let estimate = result.estimate.unwrap();
        assert!((estimate.mu - 1.231_417).abs() < 0.005, "{estimate:?}");
        assert!((estimate.sigma - 1.252_275).abs() < 0.005, "{estimate:?}");
        assert!((estimate.d_a - 1.086_691).abs() < 0.005, "{estimate:?}");
        let nll = result.diagnostics.negative_log_likelihood.unwrap();
        assert!((nll - 315.878_1).abs() < 0.01, "NLL={nll}");
    }

    #[test]
    fn reference_count_correction_is_explicit_and_gradient_consistent() {
        let table = synthetic_table(1.2, 1.4, 40_000);
        let estimate = UvsdEstimate::new(1.1, 1.3, vec![-0.8, 0.1, 1.2]).unwrap();
        let raw = negative_log_likelihood(&table, &estimate).unwrap();
        let corrected = negative_log_likelihood_with_correction(
            &table,
            &estimate,
            CountCorrection::MiyoshiReferencePerCategory,
        )
        .unwrap();
        assert_ne!(raw, corrected);
        let gradient = negative_log_likelihood_gradient_with_correction(
            &table,
            &estimate,
            CountCorrection::MiyoshiReferencePerCategory,
        )
        .unwrap();
        let epsilon = 1.0e-5;
        let plus = UvsdEstimate::new(
            estimate.mu + epsilon,
            estimate.sigma,
            estimate.thresholds.clone(),
        )
        .unwrap();
        let minus = UvsdEstimate::new(
            estimate.mu - epsilon,
            estimate.sigma,
            estimate.thresholds.clone(),
        )
        .unwrap();
        let finite_difference = (
            negative_log_likelihood_with_correction(
                &table, &plus, CountCorrection::MiyoshiReferencePerCategory
            ).unwrap()
            - negative_log_likelihood_with_correction(
                &table, &minus, CountCorrection::MiyoshiReferencePerCategory
            ).unwrap()
        ) / (2.0 * epsilon);
        assert!((gradient.mu - finite_difference).abs() < 1.0e-3 * (1.0 + finite_difference.abs()));
    }

    #[test]
    fn design_information_matches_fit_model_parameter_dimension() {
        let information = expected_fisher_information_for_design(
            1.2,
            1.3,
            &[-0.8, 0.1, 1.2],
            10_000,
            10_000,
        )
        .unwrap();
        assert_eq!(information.parameter_dimension, 5);
        assert_eq!(information.effective_rank, 5);
        assert!(!information.weakly_identified);
        assert!(information.d_a_standard_error.unwrap().is_finite());
        assert!(expected_fisher_information_for_design(
            1.2, 1.3, &[-0.8], 10_000, 10_000
        ).is_none());
        assert!(expected_fisher_information_for_design(
            1.2, 1.3, &[-0.8, 0.1, 1.2], 0, 10_000
        ).is_none());
    }

    #[test]
    fn expected_fisher_information_reports_rank_condition_and_wald_uncertainty() {
        let table = synthetic_table(1.4, 1.55, 250_000);
        let result = fit(&table, FitOptions::default());
        let information = result.diagnostics.expected_fisher_information.unwrap();
        assert_eq!(information.parameter_dimension, 5);
        assert_eq!(information.effective_rank, 5, "{information:?}");
        assert!(information.condition_number.unwrap().is_finite());
        assert!(information.condition_number.unwrap() >= 1.0);
        assert!(!information.weakly_identified);
        assert!(information.mu_standard_error.unwrap().is_finite());
        assert!(information.sigma_standard_error.unwrap().is_finite());
        assert!(information.sigma_wald_95.unwrap()[0] < information.sigma_wald_95.unwrap()[1]);
        assert!(information.d_a_wald_95.unwrap()[0] < information.d_a_wald_95.unwrap()[1]);
    }

    #[test]
    fn symmetric_eigen_decomposition_reconstructs_small_matrix() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.25],
            vec![0.5, 0.25, 2.0],
        ];
        let (eigenvalues, eigenvectors) = symmetric_eigen_decomposition(matrix.clone()).unwrap();
        assert!(eigenvalues.windows(2).all(|pair| pair[0] <= pair[1]));
        for row in 0..3 {
            for col in 0..3 {
                let reconstructed: f64 = (0..3)
                    .map(|index| {
                        eigenvectors[row][index]
                            * eigenvalues[index]
                            * eigenvectors[col][index]
                    })
                    .sum();
                assert!((reconstructed - matrix[row][col]).abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn derivative_free_optimizer_produces_inspectable_fit() {
        let table = synthetic_table(1.0, 1.35, 100_000);
        let options = FitOptions {
            max_iterations: 2_000,
            gradient_tolerance: 1.0e-5,
            starts: 3,
            count_correction: CountCorrection::None,
        };
        let result = fit_with_optimizer(&table, options, Optimizer::NelderMead);
        assert!(matches!(result.status, FitStatus::Converged | FitStatus::Boundary), "{result:?}");
        assert!(result.estimate.is_some());
        assert!(result.diagnostics.negative_log_likelihood.unwrap().is_finite());
    }

    #[test]
    fn sparse_support_is_not_identifiable() {
        let table = CategoryTable::new(vec![
            CategoryCount { order: 0, signal: 0, noise: 10 },
            CategoryCount { order: 1, signal: 10, noise: 0 },
            CategoryCount { order: 2, signal: 0, noise: 0 },
            CategoryCount { order: 3, signal: 0, noise: 0 },
        ])
        .unwrap();
        let result = fit(&table, FitOptions::default());
        assert_eq!(result.status, FitStatus::NotIdentifiable);
        assert!(result.estimate.is_none());
    }

    #[test]
    fn zero_cells_are_not_smoothed_or_discarded() {
        let table = synthetic_table(1.5, 1.8, 50_000);
        let estimate = UvsdEstimate::new(1.5, 1.8, vec![-1.0, 0.0, 1.0]).unwrap();
        let nll = negative_log_likelihood(&table, &estimate).unwrap();
        assert!(nll.is_finite());
        let sparse = CategoryTable::new(vec![
            CategoryCount { order: 0, signal: 0, noise: 10 },
            CategoryCount { order: 1, signal: 1, noise: 9 },
            CategoryCount { order: 2, signal: 9, noise: 1 },
            CategoryCount { order: 3, signal: 10, noise: 0 },
        ])
        .unwrap();
        let nll_sparse = negative_log_likelihood(&sparse, &estimate).unwrap();
        assert!(nll_sparse.is_finite());
    }

    #[test]
    fn invalid_raw_input_is_not_evaluable() {
        let result = fit_ordered_counts(
            vec![CategoryCount { order: 0, signal: 0, noise: 0 }],
            FitOptions::default(),
            Optimizer::Bfgs,
        );
        assert_eq!(result.status, FitStatus::NotEvaluable);
        assert!(result.diagnostics.note.is_some());
    }
}

/// Operational sequential action-selection primitives for SPECTRA.
pub mod sequential;
/// Failure-preserving scenario aggregation for adaptive candidate assessments.
pub mod scenario;

/// Known-truth observer and measurement-channel generation for SPECTRA.
pub mod scenario_dgp;

/// Simultaneous finite-sample confidence bounds for candidate score Monte Carlo.
pub mod score_bounds;

/// Robust indifference-zone certificates for Monte Carlo design selection.
pub mod ranking;

/// Time-uniform score bounds and adaptive robust simulation allocation.
pub mod allocation;

/// Robust threshold-feasibility classification over action utility intervals.
pub mod threshold;
