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
    /// Construct a scalar search configuration.
    pub fn new(
        lower_bound: f64,
        upper_bound: f64,
        lipschitz_constant: f64,
        absolute_gap_tolerance: f64,
        maximum_evaluations: usize,
        parameter_id: impl Into<String>,
        branch_id: impl Into<String>,
        lipschitz_certificate_sha256: impl Into<String>,
    ) -> Self {
        Self {
            lower_bound,
            upper_bound,
            lipschitz_constant,
            absolute_gap_tolerance,
            maximum_evaluations,
            parameter_id: parameter_id.into(),
            branch_id: branch_id.into(),
            lipschitz_certificate_sha256: lipschitz_certificate_sha256.into(),
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
    pub branch_id: String,
    pub full_order_reference_converged: bool,
}

impl ChallengeObjectiveSample {
    pub fn new(
        objective_value: f64,
        branch_id: impl Into<String>,
        full_order_reference_converged: bool,
    ) -> Self {
        Self {
            objective_value,
            branch_id: branch_id.into(),
            full_order_reference_converged,
        }
    }
}

/// Result of the scalar separation oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChallengeSearchReport {
    pub parameter_id: String,
    pub branch_id: String,
    pub lipschitz_certificate_sha256: String,
    pub best_parameter: f64,
    pub best_sampled_objective: f64,
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
    if initial_midpoint <= config.lower_bound || initial_midpoint >= config.upper_bound {
        return Err(ChallengeSearchError::InvalidConfiguration);
    }
    let initial_value = evaluate_checked(&mut evaluate, initial_midpoint, &config.branch_id)?;
    let initial_upper = upper_envelope(
        initial_value,
        config.lipschitz_constant,
        half_width(config.lower_bound, config.upper_bound),
    )?;

    let mut best_parameter = initial_midpoint;
    let mut best_value = initial_value;
    let mut evaluations = 1_usize;
    let mut cells = vec![SearchCell {
        left: config.lower_bound,
        right: config.upper_bound,
        upper_bound: initial_upper,
    }];

    let termination = loop {
        let (worst_index, global_upper_bound) = worst_cell(&cells);
        let absolute_gap = (global_upper_bound - best_value).max(0.0);
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
                sample_value,
                config.lipschitz_constant,
                half_width(left, right),
            )?;
            if sample_value > best_value
                || (sample_value == best_value && sample_parameter < best_parameter)
            {
                best_value = sample_value;
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
    let absolute_gap = (global_upper_bound - best_value).max(0.0);
    Ok(ScalarChallengeSearchReport {
        parameter_id: config.parameter_id.clone(),
        branch_id: config.branch_id.clone(),
        lipschitz_certificate_sha256: config.lipschitz_certificate_sha256.clone(),
        best_parameter,
        best_sampled_objective: best_value,
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
) -> Result<f64, ChallengeSearchError>
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
    Ok(sample.objective_value)
}

fn upper_envelope(
    sample_value: f64,
    lipschitz_constant: f64,
    radius: f64,
) -> Result<f64, ChallengeSearchError> {
    let upper = sample_value + lipschitz_constant * radius;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn certificate() -> String {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned()
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
            0.0,
            1.0,
            2.52,
            1.0e-2,
            128,
            "second_pulse_phase_rad",
            "echo-branch-2",
            certificate(),
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
            0.0,
            1.0,
            2.52,
            1.0e-6,
            1,
            "second_pulse_amplitude",
            "echo-branch-2",
            certificate(),
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
    fn invalid_provenance_or_objective_is_rejected() {
        let invalid_hash = ScalarChallengeSearchConfig::new(
            0.0,
            1.0,
            1.0,
            1.0e-2,
            8,
            "pulse_phase",
            "branch-a",
            "not-a-digest",
        );
        assert_eq!(
            search_scalar_response_challenge(&invalid_hash, |_| {
                Ok(ChallengeObjectiveSample::new(0.5, "branch-a", true))
            }),
            Err(ChallengeSearchError::InvalidConfiguration)
        );

        let valid = ScalarChallengeSearchConfig::new(
            0.0,
            1.0,
            1.0,
            1.0e-2,
            8,
            "pulse_phase",
            "branch-a",
            certificate(),
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
            0.0,
            1.0,
            1.0,
            1.0e-2,
            8,
            "pulse_phase",
            "branch-a",
            certificate(),
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
            0.0,
            1.0,
            lipschitz,
            5.0e-3,
            512,
            "second_pulse_phase_rad",
            "stable-multi-peak",
            certificate(),
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
            -1.0,
            1.0,
            0.0,
            1.0e-8,
            1,
            "pulse_phase",
            "stable-branch",
            certificate(),
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
