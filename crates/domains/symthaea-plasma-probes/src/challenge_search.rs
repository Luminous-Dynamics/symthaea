// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conditional one-dimensional separation oracle for adversarial pulse search.
//!
//! The oracle adaptively searches a closed scalar parameter interval using an
//! externally supplied, certificate-pinned Lipschitz upper bound. It reports
//! a conditional upper bound only when the sampled objective gap is within the
//! requested tolerance. If the evaluation budget or floating-point resolution
//! is exhausted first, the result remains explicitly unqualified.
//!
//! This is a one-dimensional building block. It does not establish that the
//! caller's Lipschitz bound is physically valid, does not prove a response
//! branch is stable, and does not solve a multidimensional pulse optimization
//! problem. Those facts must be supplied / qualified independently.

/// Errors from scalar adversarial response search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChallengeSearchError {
    /// The interval, objective tolerance, resource budget, branch ID, or
    /// provenance digest is invalid.
    InvalidConfiguration,
    /// The caller's objective returned a negative or non-finite value.
    InvalidObjective,
    /// The supplied Lipschitz envelope overflowed floating-point arithmetic.
    NonFiniteUpperBound,
    /// An objective sample reports a different response branch.
    ResponseBranchChanged,
    /// A full-order response sample did not pass its convergence contract.
    FullOrderResponseUnqualified,
    /// A full-order response sample supplied an invalid absolute uncertainty.
    InvalidObjectiveUncertainty,
}

/// Search termination. Only ConditionalGapSatisfied establishes the requested
/// gap conditional on the supplied Lipschitz bound being valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChallengeSearchTermination {
    /// The maximum sampled value and global Lipschitz upper envelope differ by
    /// no more than the requested absolute tolerance.
    ConditionalGapSatisfied,
    /// The evaluation budget was exhausted before the requested gap was met.
    EvaluationBudgetExhausted,
    /// No distinct floating-point midpoint remains in the worst interval.
    FloatingPointResolutionExhausted,
}

/// Shared evaluation budget for a scalar or multi-dimensional challenge search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChallengeSearchBudget {
    pub absolute_gap_tolerance: f64,
    pub maximum_evaluations: usize,
}

/// Provenance binding for the response branch and its global Lipschitz evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ChallengeSearchProvenance {
    pub branch_id: String,
    pub lipschitz_certificate_sha256: String,
}

/// Scalar coordinate domain and parameter identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSearchDomain {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub lipschitz_constant: f64,
    pub parameter_id: String,
}

/// Ordered continuous-coordinate box and its anisotropic Lipschitz envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxSearchDomain {
    pub axis_ids: Vec<String>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub lipschitz_constants: Vec<f64>,
}

/// Controls for a scalar pulse-domain separation search.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChallengeSearchConfig {
    /// Inclusive lower bound of the scalar pulse parameter.
    pub lower_bound: f64,
    /// Inclusive upper bound of the scalar pulse parameter.
    pub upper_bound: f64,
    /// A globally valid Lipschitz constant for the response-residual objective
    /// over the entire interval and the registered response branch.
    pub lipschitz_constant: f64,
    /// Requested absolute difference between best sampled value and global
    /// upper envelope.
    pub absolute_gap_tolerance: f64,
    /// Maximum number of objective evaluations, including the initial sample.
    pub maximum_evaluations: usize,
    /// Name of the parameter being searched, e.g. second_pulse_phase_rad.
    pub parameter_id: String,
    /// Stable identifier for the response branch over this search interval.
    pub branch_id: String,
    /// SHA-256 digest of the independently checked evidence supporting the
    /// supplied Lipschitz constant and branch-stability domain.
    pub lipschitz_certificate_sha256: String,
}

impl ScalarChallengeSearchConfig {
    /// Construct a scalar search configuration from typed domain, budget
    /// and evidence-provenance objects.
    pub fn new(
        domain: ScalarSearchDomain,
        budget: ChallengeSearchBudget,
        provenance: ChallengeSearchProvenance,
    ) -> Self {
        Self {
            lower_bound: domain.lower_bound,
            upper_bound: domain.upper_bound,
            lipschitz_constant: domain.lipschitz_constant,
            absolute_gap_tolerance: budget.absolute_gap_tolerance,
            maximum_evaluations: budget.maximum_evaluations,
            parameter_id: domain.parameter_id,
            branch_id: provenance.branch_id,
            lipschitz_certificate_sha256: provenance.lipschitz_certificate_sha256,
        }
    }

    fn validate(&self) -> Result<(), ChallengeSearchError> {
        if !self.lower_bound.is_finite()
            || !self.upper_bound.is_finite()
            || self.lower_bound >= self.upper_bound
            || !self.lipschitz_constant.is_finite()
            || self.lipschitz_constant < 0.0
            || !self.absolute_gap_tolerance.is_finite()
            || self.absolute_gap_tolerance <= 0.0
            || self.maximum_evaluations == 0
            || self.parameter_id.trim().is_empty()
            || self.branch_id.trim().is_empty()
            || !is_sha256_hex(&self.lipschitz_certificate_sha256)
        {
            return Err(ChallengeSearchError::InvalidConfiguration);
        }
        Ok(())
    }
}

/// One objective evaluation returned by the registered response oracle.
///
/// Every sample must identify the same response branch and confirm that its
/// full-order reference trajectory passed the independently registered
/// convergence contract. Unqualified samples cannot contribute to a bound.
#[derive(Clone, Debug, PartialEq)]
pub struct ChallengeObjectiveSample {
    pub objective_value: f64,
    /// Absolute error bound for the sampled objective after the registered
    /// full-order convergence and QoI-normalization checks.
    pub objective_absolute_error_bound: f64,
    pub branch_id: String,
    pub full_order_reference_converged: bool,
}

impl ChallengeObjectiveSample {
    /// Construct an exact/manufactured objective sample with zero error bound.
    pub fn new(
        objective_value: f64,
        branch_id: impl Into<String>,
        full_order_reference_converged: bool,
    ) -> Self {
        Self::new_with_error_bound(
            objective_value,
            0.0,
            branch_id,
            full_order_reference_converged,
        )
    }

    /// Construct a sample with an independently justified absolute objective
    /// error bound. The bound must include reference-solver, QoI extraction,
    /// and normalization uncertainty; a convergence boolean alone is not an
    /// error estimate.
    pub fn new_with_error_bound(
        objective_value: f64,
        objective_absolute_error_bound: f64,
        branch_id: impl Into<String>,
        full_order_reference_converged: bool,
    ) -> Self {
        Self {
            objective_value,
            objective_absolute_error_bound,
            branch_id: branch_id.into(),
            full_order_reference_converged,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EvaluatedObjective {
    estimate: f64,
    lower_bound: f64,
    upper_bound: f64,
    absolute_error_bound: f64,
}

/// Result of the scalar separation oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChallengeSearchReport {
    pub parameter_id: String,
    pub branch_id: String,
    pub lipschitz_certificate_sha256: String,
    pub best_parameter: f64,
    /// Sampled objective estimate at best_parameter.
    pub best_sampled_objective: f64,
    /// Conservative lower bound for that sampled objective.
    pub best_sample_lower_bound: f64,
    pub best_sample_absolute_error_bound: f64,
    pub global_upper_bound: f64,
    pub absolute_gap: f64,
    pub evaluations: usize,
    pub termination: ChallengeSearchTermination,
}

#[derive(Clone, Copy, Debug)]
struct SearchCell {
    left: f64,
    right: f64,
    upper_bound: f64,
}

/// Search for the maximum of a non-negative scalar response-risk objective.
///
/// The objective should be the full-order-validated or otherwise qualified
/// response-risk metric for a single registered pulse parameter, with every
/// other parameter fixed. The supplied Lipschitz certificate must apply to
/// the entire closed interval and one stable response branch.
///
/// The algorithm repeatedly subdivides the interval with the largest current
/// Lipschitz upper bound. It never infers a global result from endpoint checks.
/// The returned bound is conditional on the caller's certificate; if no
/// independently defensible bound exists, the result must be treated as a
/// search heuristic rather than a robust guarantee.
pub fn search_scalar_response_challenge<F>(
    config: &ScalarChallengeSearchConfig,
    mut evaluate: F,
) -> Result<ScalarChallengeSearchReport, ChallengeSearchError>
where
    F: FnMut(f64) -> Result<ChallengeObjectiveSample, ChallengeSearchError>,
{
    config.validate()?;

    let initial_midpoint = midpoint(config.lower_bound, config.upper_bound);
    let has_interior_midpoint =
        initial_midpoint > config.lower_bound && initial_midpoint < config.upper_bound;
    let initial_parameter = if has_interior_midpoint {
        initial_midpoint
    } else {
        config.lower_bound
    };
    let initial_radius = if has_interior_midpoint {
        half_width(config.lower_bound, config.upper_bound)
    } else {
        config.upper_bound - config.lower_bound
    };
    let initial_value = evaluate_checked(&mut evaluate, initial_parameter, &config.branch_id)?;
    let initial_upper = upper_envelope(
        initial_value.upper_bound,
        config.lipschitz_constant,
        outward_radius(initial_radius),
    )?;

    let mut best_parameter = initial_parameter;
    let mut best_value = initial_value.estimate;
    let mut best_lower_bound = initial_value.lower_bound;
    let mut best_error_bound = initial_value.absolute_error_bound;
    let mut evaluations = 1_usize;
    let mut cells = vec![SearchCell {
        left: config.lower_bound,
        right: config.upper_bound,
        upper_bound: initial_upper,
    }];

    let termination = loop {
        let (worst_index, global_upper_bound) = worst_cell(&cells);
        let absolute_gap = (global_upper_bound - best_lower_bound).max(0.0);
        if absolute_gap <= config.absolute_gap_tolerance {
            break ChallengeSearchTermination::ConditionalGapSatisfied;
        }
        if evaluations
            .checked_add(2)
            .is_none_or(|next_count| next_count > config.maximum_evaluations)
        {
            break ChallengeSearchTermination::EvaluationBudgetExhausted;
        }

        let cell = cells.remove(worst_index);
        let split = midpoint(cell.left, cell.right);
        if split <= cell.left || split >= cell.right {
            cells.push(cell);
            break ChallengeSearchTermination::FloatingPointResolutionExhausted;
        }

        for (left, right) in [(cell.left, split), (split, cell.right)] {
            let sample_parameter = midpoint(left, right);
            if sample_parameter <= left || sample_parameter >= right {
                cells.push(SearchCell {
                    left,
                    right,
                    upper_bound: cell.upper_bound,
                });
                continue;
            }

            let sample_value =
                evaluate_checked(&mut evaluate, sample_parameter, &config.branch_id)?;
            evaluations = evaluations
                .checked_add(1)
                .ok_or(ChallengeSearchError::InvalidConfiguration)?;
            let cell_upper = upper_envelope(
                sample_value.upper_bound,
                config.lipschitz_constant,
                outward_radius(half_width(left, right)),
            )?;
            if sample_value.lower_bound > best_lower_bound
                || (sample_value.lower_bound == best_lower_bound
                    && sample_parameter < best_parameter)
            {
                best_value = sample_value.estimate;
                best_lower_bound = sample_value.lower_bound;
                best_error_bound = sample_value.absolute_error_bound;
                best_parameter = sample_parameter;
            }
            cells.push(SearchCell {
                left,
                right,
                upper_bound: cell_upper,
            });
        }
    };

    let (_, global_upper_bound) = worst_cell(&cells);
    let absolute_gap = (global_upper_bound - best_lower_bound).max(0.0);
    Ok(ScalarChallengeSearchReport {
        parameter_id: config.parameter_id.clone(),
        branch_id: config.branch_id.clone(),
        lipschitz_certificate_sha256: config.lipschitz_certificate_sha256.clone(),
        best_parameter,
        best_sampled_objective: best_value,
        best_sample_lower_bound: best_lower_bound,
        best_sample_absolute_error_bound: best_error_bound,
        global_upper_bound,
        absolute_gap,
        evaluations,
        termination,
    })
}

fn evaluate_checked<F>(
    evaluate: &mut F,
    parameter: f64,
    expected_branch_id: &str,
) -> Result<EvaluatedObjective, ChallengeSearchError>
where
    F: FnMut(f64) -> Result<ChallengeObjectiveSample, ChallengeSearchError>,
{
    let sample = evaluate(parameter)?;
    if sample.branch_id != expected_branch_id {
        return Err(ChallengeSearchError::ResponseBranchChanged);
    }
    if !sample.full_order_reference_converged {
        return Err(ChallengeSearchError::FullOrderResponseUnqualified);
    }
    if !sample.objective_value.is_finite() || sample.objective_value < 0.0 {
        return Err(ChallengeSearchError::InvalidObjective);
    }
    if !sample.objective_absolute_error_bound.is_finite()
        || sample.objective_absolute_error_bound < 0.0
    {
        return Err(ChallengeSearchError::InvalidObjectiveUncertainty);
    }

    let lower_raw = sample.objective_value - sample.objective_absolute_error_bound;
    let lower_bound = if lower_raw > 0.0 {
        lower_raw.next_down().max(0.0)
    } else {
        0.0
    };
    let upper_raw = sample.objective_value + sample.objective_absolute_error_bound;
    if !upper_raw.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let upper_bound = if sample.objective_absolute_error_bound > 0.0 {
        upper_raw.next_up()
    } else {
        upper_raw
    };
    if !upper_bound.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    Ok(EvaluatedObjective {
        estimate: sample.objective_value,
        lower_bound,
        upper_bound,
        absolute_error_bound: sample.objective_absolute_error_bound,
    })
}

fn outward_radius(radius: f64) -> f64 {
    if radius > 0.0 {
        radius.next_up()
    } else {
        radius
    }
}

fn upper_envelope(
    sample_upper: f64,
    lipschitz_constant: f64,
    radius: f64,
) -> Result<f64, ChallengeSearchError> {
    if !sample_upper.is_finite()
        || sample_upper < 0.0
        || !lipschitz_constant.is_finite()
        || lipschitz_constant < 0.0
        || !radius.is_finite()
        || radius < 0.0
    {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    if lipschitz_constant == 0.0 || radius == 0.0 {
        return Ok(sample_upper);
    }
    let product = lipschitz_constant * radius;
    if !product.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let product_upper = product.next_up();
    let sum = sample_upper + product_upper;
    if !sum.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let upper = sum.next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(ChallengeSearchError::NonFiniteUpperBound)
    }
}

fn midpoint(left: f64, right: f64) -> f64 {
    left * 0.5 + right * 0.5
}

fn half_width(left: f64, right: f64) -> f64 {
    right * 0.5 - left * 0.5
}

fn worst_cell(cells: &[SearchCell]) -> (usize, f64) {
    let mut worst_index = 0_usize;
    for index in 1..cells.len() {
        let candidate = cells[index];
        let incumbent = cells[worst_index];
        if candidate.upper_bound > incumbent.upper_bound
            || (candidate.upper_bound == incumbent.upper_bound && candidate.left < incumbent.left)
        {
            worst_index = index;
        }
    }
    (worst_index, cells[worst_index].upper_bound)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Configuration for a joint intervention-box response search.
///
/// The coordinate-wise Lipschitz constants certify
/// |J(x)-J(y)| <= sum_i L_i * |x_i-y_i|.
/// This is an anisotropic L1-domain envelope. The certificate digest must
/// bind the entire closed box, objective definition, QoI scaling, branch
/// identity, and every L_i; per-axis bounds from unrelated boxes cannot be
/// combined into a global certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxChallengeSearchConfig {
    pub axis_ids: Vec<String>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub lipschitz_constants: Vec<f64>,
    pub absolute_gap_tolerance: f64,
    pub maximum_evaluations: usize,
    pub branch_id: String,
    pub lipschitz_certificate_sha256: String,
}

impl BoxChallengeSearchConfig {
    /// Construct a multi-dimensional search configuration from typed domain,
    /// budget and evidence-provenance objects.
    pub fn new(
        domain: BoxSearchDomain,
        budget: ChallengeSearchBudget,
        provenance: ChallengeSearchProvenance,
    ) -> Self {
        Self {
            axis_ids: domain.axis_ids,
            lower_bounds: domain.lower_bounds,
            upper_bounds: domain.upper_bounds,
            lipschitz_constants: domain.lipschitz_constants,
            absolute_gap_tolerance: budget.absolute_gap_tolerance,
            maximum_evaluations: budget.maximum_evaluations,
            branch_id: provenance.branch_id,
            lipschitz_certificate_sha256: provenance.lipschitz_certificate_sha256,
        }
    }

    fn validate(&self) -> Result<(), ChallengeSearchError> {
        let dimension = self.axis_ids.len();
        if dimension < 2
            || self.lower_bounds.len() != dimension
            || self.upper_bounds.len() != dimension
            || self.lipschitz_constants.len() != dimension
            || !self.absolute_gap_tolerance.is_finite()
            || self.absolute_gap_tolerance <= 0.0
            || self.maximum_evaluations == 0
            || self.branch_id.trim().is_empty()
            || !is_sha256_hex(&self.lipschitz_certificate_sha256)
        {
            return Err(ChallengeSearchError::InvalidConfiguration);
        }

        let mut seen = std::collections::HashSet::with_capacity(dimension);
        for (((axis, &lower), &upper), &lipschitz) in self
            .axis_ids
            .iter()
            .zip(&self.lower_bounds)
            .zip(&self.upper_bounds)
            .zip(&self.lipschitz_constants)
        {
            if axis.trim().is_empty()
                || !seen.insert(axis)
                || !lower.is_finite()
                || !upper.is_finite()
                || lower >= upper
                || !lipschitz.is_finite()
                || lipschitz < 0.0
            {
                return Err(ChallengeSearchError::InvalidConfiguration);
            }
        }
        Ok(())
    }
}

/// Result of a joint, branch-conditioned intervention-box search.
///
/// ConditionalGapSatisfied is conditional on the supplied coordinate-wise
/// Lipschitz certificate being valid over the complete closed box and on the
/// declared response branch remaining stable throughout that box.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxChallengeSearchReport {
    pub axis_ids: Vec<String>,
    pub branch_id: String,
    pub lipschitz_certificate_sha256: String,
    pub best_parameters: Vec<f64>,
    /// Sampled objective estimate at best_parameters.
    pub best_sampled_objective: f64,
    /// Conservative lower bound for that sampled objective.
    pub best_sample_lower_bound: f64,
    pub best_sample_absolute_error_bound: f64,
    pub global_upper_bound: f64,
    pub absolute_gap: f64,
    pub evaluations: usize,
    pub termination: ChallengeSearchTermination,
}

#[derive(Clone, Debug)]
struct BoxSearchCell {
    lower: Vec<f64>,
    upper: Vec<f64>,
    upper_bound: f64,
}

/// Search jointly over a closed multi-parameter intervention box.
///
/// The objective must be non-negative and every sample must be confirmed by
/// the registered full-order convergence contract on one stable response
/// branch. The global upper envelope for a cell is formed from a sample point
/// and the coordinate-wise Lipschitz radii. The cell with the largest upper
/// envelope is subdivided along the axis with the largest L_i * half_width_i.
///
/// The oracle explores interior points; it does not infer a global maximum
/// from boundary checks. Budget exhaustion, floating-point resolution
/// exhaustion, branch changes, or unqualified full-order samples never yield
/// a qualified global bound.
pub fn search_box_response_challenge<F>(
    config: &BoxChallengeSearchConfig,
    mut evaluate: F,
) -> Result<BoxChallengeSearchReport, ChallengeSearchError>
where
    F: FnMut(&[f64]) -> Result<ChallengeObjectiveSample, ChallengeSearchError>,
{
    config.validate()?;
    let (initial_center, initial_radii) =
        box_center_and_radii(&config.lower_bounds, &config.upper_bounds)?;
    let initial_value =
        evaluate_box_checked(&mut evaluate, &initial_center, &config.branch_id)?;
    let outward_radii = initial_radii.iter().copied().map(outward_radius).collect::<Vec<_>>();
    let initial_upper = box_upper_envelope(
        initial_value.upper_bound,
        &config.lipschitz_constants,
        &outward_radii,
    )?;

    let mut best_parameters = initial_center;
    let mut best_value = initial_value.estimate;
    let mut best_lower_bound = initial_value.lower_bound;
    let mut best_error_bound = initial_value.absolute_error_bound;
    let mut evaluations = 1_usize;
    let mut cells = vec![BoxSearchCell {
        lower: config.lower_bounds.clone(),
        upper: config.upper_bounds.clone(),
        upper_bound: initial_upper,
    }];

    let termination = loop {
        let (worst_index, global_upper_bound) = worst_box_cell(&cells);
        let absolute_gap = (global_upper_bound - best_lower_bound).max(0.0);
        if absolute_gap <= config.absolute_gap_tolerance {
            break ChallengeSearchTermination::ConditionalGapSatisfied;
        }
        if evaluations
            .checked_add(2)
            .is_none_or(|next_count| next_count > config.maximum_evaluations)
        {
            break ChallengeSearchTermination::EvaluationBudgetExhausted;
        }

        let cell = cells.remove(worst_index);
        let Some(split_axis) = choose_box_split_axis(&cell, &config.lipschitz_constants) else {
            cells.push(cell);
            break ChallengeSearchTermination::FloatingPointResolutionExhausted;
        };
        let split = midpoint(cell.lower[split_axis], cell.upper[split_axis]);
        if split <= cell.lower[split_axis] || split >= cell.upper[split_axis] {
            cells.push(cell);
            break ChallengeSearchTermination::FloatingPointResolutionExhausted;
        }

        for choose_left in [true, false] {
            let mut lower = cell.lower.clone();
            let mut upper = cell.upper.clone();
            if choose_left {
                upper[split_axis] = split;
            } else {
                lower[split_axis] = split;
            }
            let (center, radii) = box_center_and_radii(&lower, &upper)?;
            let sample_value =
                evaluate_box_checked(&mut evaluate, &center, &config.branch_id)?;
            evaluations = evaluations
                .checked_add(1)
                .ok_or(ChallengeSearchError::InvalidConfiguration)?;
            let outward_radii = radii.iter().copied().map(outward_radius).collect::<Vec<_>>();
            let cell_upper = box_upper_envelope(
                sample_value.upper_bound,
                &config.lipschitz_constants,
                &outward_radii,
            )?;
            if sample_value.lower_bound > best_lower_bound
                || (sample_value.lower_bound == best_lower_bound
                    && lexicographically_less(&center, &best_parameters))
            {
                best_value = sample_value.estimate;
                best_lower_bound = sample_value.lower_bound;
                best_error_bound = sample_value.absolute_error_bound;
                best_parameters = center;
            }
            cells.push(BoxSearchCell {
                lower,
                upper,
                upper_bound: cell_upper,
            });
        }
    };

    let (_, global_upper_bound) = worst_box_cell(&cells);
    let absolute_gap = (global_upper_bound - best_lower_bound).max(0.0);
    Ok(BoxChallengeSearchReport {
        axis_ids: config.axis_ids.clone(),
        branch_id: config.branch_id.clone(),
        lipschitz_certificate_sha256: config.lipschitz_certificate_sha256.clone(),
        best_parameters,
        best_sampled_objective: best_value,
        best_sample_lower_bound: best_lower_bound,
        best_sample_absolute_error_bound: best_error_bound,
        global_upper_bound,
        absolute_gap,
        evaluations,
        termination,
    })
}

fn evaluate_box_checked<F>(
    evaluate: &mut F,
    parameters: &[f64],
    expected_branch_id: &str,
) -> Result<EvaluatedObjective, ChallengeSearchError>
where
    F: FnMut(&[f64]) -> Result<ChallengeObjectiveSample, ChallengeSearchError>,
{
    let sample = evaluate(parameters)?;
    if sample.branch_id != expected_branch_id {
        return Err(ChallengeSearchError::ResponseBranchChanged);
    }
    if !sample.full_order_reference_converged {
        return Err(ChallengeSearchError::FullOrderResponseUnqualified);
    }
    if !sample.objective_value.is_finite() || sample.objective_value < 0.0 {
        return Err(ChallengeSearchError::InvalidObjective);
    }
    if !sample.objective_absolute_error_bound.is_finite()
        || sample.objective_absolute_error_bound < 0.0
    {
        return Err(ChallengeSearchError::InvalidObjectiveUncertainty);
    }

    let lower_raw = sample.objective_value - sample.objective_absolute_error_bound;
    let lower_bound = if lower_raw > 0.0 {
        lower_raw.next_down().max(0.0)
    } else {
        0.0
    };
    let upper_raw = sample.objective_value + sample.objective_absolute_error_bound;
    if !upper_raw.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let upper_bound = if sample.objective_absolute_error_bound > 0.0 {
        upper_raw.next_up()
    } else {
        upper_raw
    };
    if !upper_bound.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    Ok(EvaluatedObjective {
        estimate: sample.objective_value,
        lower_bound,
        upper_bound,
        absolute_error_bound: sample.objective_absolute_error_bound,
    })
}

fn outward_radius(radius: f64) -> f64 {
    if radius > 0.0 {
        radius.next_up()
    } else {
        radius
    }
}

fn upper_envelope(
    sample_upper: f64,
    lipschitz_constant: f64,
    radius: f64,
) -> Result<f64, ChallengeSearchError> {
    if !sample_upper.is_finite()
        || sample_upper < 0.0
        || !lipschitz_constant.is_finite()
        || lipschitz_constant < 0.0
        || !radius.is_finite()
        || radius < 0.0
    {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    if lipschitz_constant == 0.0 || radius == 0.0 {
        return Ok(sample_upper);
    }
    let product = lipschitz_constant * radius;
    if !product.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let product_upper = product.next_up();
    let sum = sample_upper + product_upper;
    if !sum.is_finite() {
        return Err(ChallengeSearchError::NonFiniteUpperBound);
    }
    let upper = sum.next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(ChallengeSearchError::NonFiniteUpperBound)
    }
}

fn box_center_and_radii(
    lower: &[f64],
    upper: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), ChallengeSearchError> {
    if lower.is_empty() || lower.len() != upper.len() {
        return Err(ChallengeSearchError::InvalidConfiguration);
    }
    let mut center = Vec::with_capacity(lower.len());
    let mut radii = Vec::with_capacity(lower.len());
    for (&left, &right) in lower.iter().zip(upper) {
        let middle = midpoint(left, right);
        let half = half_width(left, right);
        let (sample, radius) = if middle > left && middle < right {
            (middle, half)
        } else {
            // Adjacent representable endpoints have no interior midpoint.
            // Use the full endpoint difference: half-width arithmetic can
            // underflow to zero for subnormal intervals near zero.
            (left, right - left)
        };
        if !radius.is_finite() || radius < 0.0 {
            return Err(ChallengeSearchError::NonFiniteUpperBound);
        }
        center.push(sample);
        radii.push(radius);
    }
    Ok((center, radii))
}

fn box_upper_envelope(
    sample_upper: f64,
    lipschitz_constants: &[f64],
    radii: &[f64],
) -> Result<f64, ChallengeSearchError> {
    if lipschitz_constants.len() != radii.len() {
        return Err(ChallengeSearchError::InvalidConfiguration);
    }
    let mut upper = sample_upper;
    for (&lipschitz, &radius) in lipschitz_constants.iter().zip(radii) {
        if !lipschitz.is_finite() || lipschitz < 0.0 || !radius.is_finite() || radius < 0.0 {
            return Err(ChallengeSearchError::NonFiniteUpperBound);
        }
        if lipschitz == 0.0 || radius == 0.0 {
            continue;
        }
        let product = lipschitz * radius;
        if !product.is_finite() {
            return Err(ChallengeSearchError::NonFiniteUpperBound);
        }
        upper += product.next_up();
        if !upper.is_finite() {
            return Err(ChallengeSearchError::NonFiniteUpperBound);
        }
        upper = upper.next_up();
        if !upper.is_finite() {
            return Err(ChallengeSearchError::NonFiniteUpperBound);
        }
    }
    Ok(upper)
}

fn choose_box_split_axis(
    cell: &BoxSearchCell,
    lipschitz_constants: &[f64],
) -> Option<usize> {
    let mut selected: Option<(usize, f64)> = None;
    for (axis, (&lower, &upper)) in cell.lower.iter().zip(&cell.upper).enumerate() {
        let lipschitz = lipschitz_constants[axis];
        if lipschitz == 0.0 {
            continue;
        }
        let middle = midpoint(lower, upper);
        if middle <= lower || middle >= upper {
            continue;
        }
        let contribution = lipschitz * half_width(lower, upper);
        match selected {
            None => selected = Some((axis, contribution)),
            Some((best_axis, best_contribution))
                if contribution > best_contribution
                    || (contribution == best_contribution && axis < best_axis) =>
            {
                selected = Some((axis, contribution));
            }
            _ => {}
        }
    }
    selected.map(|(axis, _)| axis)
}

fn worst_box_cell(cells: &[BoxSearchCell]) -> (usize, f64) {
    let mut worst_index = 0_usize;
    for index in 1..cells.len() {
        let candidate = &cells[index];
        let incumbent = &cells[worst_index];
        if candidate.upper_bound > incumbent.upper_bound
            || (candidate.upper_bound == incumbent.upper_bound
                && lexicographically_less(&candidate.lower, &incumbent.lower))
        {
            worst_index = index;
        }
    }
    (worst_index, cells[worst_index].upper_bound)
}

fn lexicographically_less(left: &[f64], right: &[f64]) -> bool {
    for (&left_value, &right_value) in left.iter().zip(right) {
        let ordering = left_value.total_cmp(&right_value);
        if ordering != std::cmp::Ordering::Equal {
            return ordering == std::cmp::Ordering::Less;
        }
    }
    left.len() < right.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certificate() -> String {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned()
    }

    #[test]
    fn box_search_finds_joint_interior_response_peak_and_bounds_the_full_box() {
        let config = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec![
                    "second_pulse_phase".to_owned(),
                    "second_pulse_amplitude".to_owned(),
                    "pulse_separation".to_owned(),
                ],
                lower_bounds: vec![0.0, 0.0, 0.0],
                upper_bounds: vec![1.0, 1.0, 1.0],
                lipschitz_constants: vec![1.0, 2.0, 0.5],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 5.0e-2,
                maximum_evaluations: 1_024,
            },
            ChallengeSearchProvenance {
                branch_id: "echo-branch-7".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let objective = |parameters: &[f64]| {
            let value = 3.0
                - (parameters[0] - 0.37).abs()
                - 2.0 * (parameters[1] - 0.61).abs()
                - 0.5 * (parameters[2] - 0.22).abs();
            Ok(ChallengeObjectiveSample::new(
                value,
                "echo-branch-7",
                true,
            ))
        };
        let report = search_box_response_challenge(&config, objective).unwrap();

        assert!((report.best_parameters[0] - 0.37).abs() < 0.05);
        assert!((report.best_parameters[1] - 0.61).abs() < 0.05);
        assert!((report.best_parameters[2] - 0.22).abs() < 0.1);
        assert!(report.global_upper_bound >= 3.0);
        assert!(report.absolute_gap <= config.absolute_gap_tolerance);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::ConditionalGapSatisfied
        );
        assert!(report.evaluations <= config.maximum_evaluations);
    }

    #[test]
    fn box_search_budget_exhaustion_is_explicitly_unqualified() {
        let config = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["phase".to_owned(), "amplitude".to_owned()],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![1.0, 1.0],
                lipschitz_constants: vec![1.0, 1.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-6,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_box_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new(0.5, "stable-branch", true))
        })
        .unwrap();
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::EvaluationBudgetExhausted
        );
        assert!(report.absolute_gap > config.absolute_gap_tolerance);
        assert_eq!(report.evaluations, 1);
    }

    #[test]
    fn box_search_rejects_duplicate_axes_branch_changes_and_unconverged_samples() {
        let invalid_axes = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["phase".to_owned(), "phase".to_owned()],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![1.0, 1.0],
                lipschitz_constants: vec![1.0, 1.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        assert_eq!(
            search_box_response_challenge(&invalid_axes, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "stable-branch", true))
            }),
            Err(ChallengeSearchError::InvalidConfiguration)
        );

        let valid = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["phase".to_owned(), "amplitude".to_owned()],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![1.0, 1.0],
                lipschitz_constants: vec![1.0, 1.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        assert_eq!(
            search_box_response_challenge(&valid, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "different-branch", true))
            }),
            Err(ChallengeSearchError::ResponseBranchChanged)
        );
        assert_eq!(
            search_box_response_challenge(&valid, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "stable-branch", false))
            }),
            Err(ChallengeSearchError::FullOrderResponseUnqualified)
        );
    }

    #[test]
    fn box_search_subnormal_adjacent_endpoints_do_not_collapse_to_zero_radius() {
        let smallest_subnormal = f64::from_bits(1);
        let config = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["subnormal_axis".to_owned(), "regular_axis".to_owned()],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![smallest_subnormal, 1.0],
                lipschitz_constants: vec![1.0, 0.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 0.0,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_box_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new(0.0, "stable-branch", true))
        })
        .unwrap();
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::EvaluationBudgetExhausted
        );
        assert!(report.global_upper_bound > 0.0);
        assert!(report.absolute_gap > 0.0);
    }

    #[test]
    fn box_search_zero_lipschitz_axes_do_not_consume_refinement_budget() {
        let config = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["phase".to_owned(), "irrelevant_axis".to_owned()],
                lower_bounds: vec![0.0, -1.0],
                upper_bounds: vec![1.0, 1.0],
                lipschitz_constants: vec![0.0, 0.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-10,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "constant-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_box_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new(0.125, "constant-branch", true))
        })
        .unwrap();
        assert_eq!(report.best_sampled_objective, 0.125);
        assert_eq!(report.global_upper_bound, 0.125);
        assert_eq!(report.evaluations, 1);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::ConditionalGapSatisfied
        );
    }

    #[test]
    fn interior_worst_case_is_found_and_conditionally_bounded() {
        // The objective has an interior maximum at 0.37; its boundary values
        // are strictly smaller. L=2.52 bounds |f'(x)| on [0, 1].
        let objective = |parameter: f64| {
            Ok(ChallengeObjectiveSample::new(
                3.0 - 2.0 * (parameter - 0.37).powi(2),
                "echo-branch-2",
                true,
            ))
        };
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 2.52,
                parameter_id: "second_pulse_phase_rad".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 128,
            },
            ChallengeSearchProvenance {
                branch_id: "echo-branch-2".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, objective).unwrap();
        let left_boundary = 3.0 - 2.0 * (0.0_f64 - 0.37).powi(2);
        let right_boundary = 3.0 - 2.0 * (1.0_f64 - 0.37).powi(2);
        assert!(report.best_sampled_objective > left_boundary.max(right_boundary));
        assert!((report.best_parameter - 0.37).abs() < 0.02);
        assert!(report.global_upper_bound >= 3.0 - 1.0e-10);
        assert!(report.absolute_gap <= config.absolute_gap_tolerance);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::ConditionalGapSatisfied
        );
        assert_eq!(report.branch_id, "echo-branch-2");
        assert!(report.evaluations <= config.maximum_evaluations);
    }

    #[test]
    fn exhausted_search_budget_never_returns_a_qualified_bound() {
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 2.52,
                parameter_id: "second_pulse_amplitude".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-6,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "echo-branch-2".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, |parameter| {
            Ok(ChallengeObjectiveSample::new(
                3.0 - 2.0 * (parameter - 0.37).powi(2),
                "echo-branch-2",
                true,
            ))
        })
        .unwrap();
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::EvaluationBudgetExhausted
        );
        assert!(report.absolute_gap > config.absolute_gap_tolerance);
        assert_eq!(report.evaluations, 1);
    }

    #[test]
    fn scalar_search_adjacent_float_interval_is_unqualified_not_invalid() {
        let smallest_subnormal = f64::from_bits(1);
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: smallest_subnormal,
                lipschitz_constant: 1.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 0.0,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new(0.0, "stable-branch", true))
        })
        .unwrap();
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::FloatingPointResolutionExhausted
        );
        assert!(report.global_upper_bound > 0.0);
        assert!(report.absolute_gap > 0.0);
        assert_eq!(report.best_parameter, 0.0);
    }

    #[test]
    fn scalar_search_keeps_reference_objective_uncertainty_in_the_gap() {
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 0.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-3,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new_with_error_bound(
                1.0,
                0.025,
                "stable-branch",
                true,
            ))
        })
        .unwrap();
        assert!((report.best_sample_lower_bound - 0.975).abs() < 1.0e-12);
        assert!((report.best_sample_absolute_error_bound - 0.025).abs() < 1.0e-12);
        assert!(report.global_upper_bound >= 1.025);
        assert!(report.absolute_gap > config.absolute_gap_tolerance);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::EvaluationBudgetExhausted
        );
    }

    #[test]
    fn box_search_does_not_call_uncertain_full_order_samples_exact() {
        let config = BoxChallengeSearchConfig::new(
            BoxSearchDomain {
                axis_ids: vec!["phase".to_owned(), "amplitude".to_owned()],
                lower_bounds: vec![0.0, 0.0],
                upper_bounds: vec![1.0, 1.0],
                lipschitz_constants: vec![0.0, 0.0],
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-3,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_box_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new_with_error_bound(
                0.5,
                0.01,
                "stable-branch",
                true,
            ))
        })
        .unwrap();
        assert!((report.best_sample_lower_bound - 0.49).abs() < 1.0e-12);
        assert!(report.global_upper_bound >= 0.51);
        assert!(report.absolute_gap > config.absolute_gap_tolerance);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::EvaluationBudgetExhausted
        );
    }

    #[test]
    fn negative_or_nonfinite_objective_error_bounds_are_rejected() {
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 1.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "branch-a".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        assert_eq!(
            search_scalar_response_challenge(&config, |_| {
                Ok(ChallengeObjectiveSample::new_with_error_bound(
                    0.5,
                    -0.01,
                    "branch-a",
                    true,
                ))
            }),
            Err(ChallengeSearchError::InvalidObjectiveUncertainty)
        );
        assert_eq!(
            search_scalar_response_challenge(&config, |_| {
                Ok(ChallengeObjectiveSample::new_with_error_bound(
                    0.5,
                    f64::INFINITY,
                    "branch-a",
                    true,
                ))
            }),
            Err(ChallengeSearchError::InvalidObjectiveUncertainty)
        );
    }

    #[test]
    fn invalid_provenance_or_objective_is_rejected() {
        let invalid_hash = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 1.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "branch-a".into(),
                lipschitz_certificate_sha256: "not-a-digest".into(),
            },
        );
        assert_eq!(
            search_scalar_response_challenge(&invalid_hash, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "branch-a", true))
            }),
            Err(ChallengeSearchError::InvalidConfiguration)
        );

        let valid = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 1.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "branch-a".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        assert_eq!(
            search_scalar_response_challenge(&valid, |_| {
                Ok(ChallengeObjectiveSample::new(f64::NAN, "branch-a", true))
            }),
            Err(ChallengeSearchError::InvalidObjective)
        );
    }

    #[test]
    fn branch_changes_and_unconverged_full_order_samples_are_rejected() {
        let branch_config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: 1.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-2,
                maximum_evaluations: 8,
            },
            ChallengeSearchProvenance {
                branch_id: "branch-a".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        assert_eq!(
            search_scalar_response_challenge(&branch_config, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "branch-b", true))
            }),
            Err(ChallengeSearchError::ResponseBranchChanged)
        );
        assert_eq!(
            search_scalar_response_challenge(&branch_config, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "branch-a", false))
            }),
            Err(ChallengeSearchError::FullOrderResponseUnqualified)
        );
    }

    #[test]
    fn conditional_upper_envelope_covers_multiple_interior_peaks() {
        // A max of L-Lipschitz cones remains L-Lipschitz. The peaks are all
        // interior, so a boundary-only search is not a valid substitute.
        let centers = [0.07_f64, 0.19, 0.37, 0.61, 0.88];
        let heights = [0.8_f64, 2.4, 1.7, 2.9, 1.2];
        let lipschitz = 4.0;
        let objective = |parameter: f64| {
            let value = centers
                .iter()
                .zip(heights)
                .map(|(center, height)| height - lipschitz * (parameter - center).abs())
                .fold(0.0_f64, f64::max);
            Ok(ChallengeObjectiveSample::new(
                value,
                "stable-multi-peak",
                true,
            ))
        };
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: 0.0,
                upper_bound: 1.0,
                lipschitz_constant: lipschitz,
                parameter_id: "second_pulse_phase_rad".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 5.0e-3,
                maximum_evaluations: 512,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-multi-peak".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, objective).unwrap();
        let dense_grid_max = (0..=20_000)
            .map(|index| {
                let parameter = index as f64 / 20_000.0;
                centers
                    .iter()
                    .zip(heights)
                    .map(|(center, height)| height - lipschitz * (parameter - center).abs())
                    .fold(0.0_f64, f64::max)
            })
            .fold(0.0_f64, f64::max);

        assert!(report.global_upper_bound + 1.0e-12 >= dense_grid_max);
        assert!(report.absolute_gap <= config.absolute_gap_tolerance);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::ConditionalGapSatisfied
        );
    }

    #[test]
    fn zero_lipschitz_objective_is_bounded_after_one_evaluation() {
        let config = ScalarChallengeSearchConfig::new(
            ScalarSearchDomain {
                lower_bound: -1.0,
                upper_bound: 1.0,
                lipschitz_constant: 0.0,
                parameter_id: "pulse_phase".into(),
            },
            ChallengeSearchBudget {
                absolute_gap_tolerance: 1.0e-8,
                maximum_evaluations: 1,
            },
            ChallengeSearchProvenance {
                branch_id: "stable-branch".into(),
                lipschitz_certificate_sha256: certificate().into(),
            },
        );
        let report = search_scalar_response_challenge(&config, |_| {
            Ok(ChallengeObjectiveSample::new(0.125, "stable-branch", true))
        })
        .unwrap();
        assert_eq!(report.best_sampled_objective, 0.125);
        assert_eq!(report.global_upper_bound, 0.125);
        assert_eq!(report.evaluations, 1);
        assert_eq!(
            report.termination,
            ChallengeSearchTermination::ConditionalGapSatisfied
        );
    }
}
