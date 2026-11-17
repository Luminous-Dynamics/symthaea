//! A2 safeguarded static hydrostatic equilibrium solver.
//!
//! This solver reuses the A0/A1 trial-attitude evaluator. It solves for
//! waterplane offset, pitch, and roll with a scaled central-difference
//! Jacobian, pivoted 3x3 solve, bounded step, and backtracking line search.
//! Multiple declared initial seeds are retained so distinct converged roots
//! are surfaced rather than silently selecting the first one.
//!
//! A converged equilibrium is not a stability, GZ/GM, damage-survival, or
//! seaworthiness assessment.

use serde::{Deserialize, Serialize};

use crate::static_attitude::{
    MAX_STATIC_ATTITUDE_ABS_RAD, StaticAttitude, StaticAttitudeEvaluationReceipt,
    StaticAttitudeEvaluationRequest, StaticAttitudeIssue, StaticAttitudeResiduals,
    compute_static_attitude_receipt_digest, evaluate_static_attitude,
};

pub const STATIC_EQUILIBRIUM_SCHEMA_VERSION: u32 = 1;
pub const STATIC_EQUILIBRIUM_ALGORITHM: &str =
    "bounded-scaled-central-difference-newton-backtracking-v1";
pub const MAX_EQUILIBRIUM_STARTS: usize = 8;
pub const MAX_EQUILIBRIUM_ITERATIONS: u32 = 100;
pub const MAX_BACKTRACKING_STEPS: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAttitudeDomain {
    pub minimum_waterplane_offset_m: f64,
    pub maximum_waterplane_offset_m: f64,
    pub maximum_abs_pitch_rad: f64,
    pub maximum_abs_roll_rad: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticEquilibriumSolverConfig {
    pub maximum_iterations: u32,
    pub maximum_backtracking_steps: u32,
    pub finite_difference_offset_step_m: f64,
    pub finite_difference_angle_step_rad: f64,
    pub mass_residual_tolerance_tonnes: f64,
    pub moment_residual_tolerance_nm: f64,
    pub offset_variable_scale_m: f64,
    pub angle_variable_scale_rad: f64,
    pub maximum_scaled_step_norm: f64,
    pub minimum_pivot_ratio: f64,
    pub maximum_condition_number: f64,
    pub sufficient_decrease_fraction: f64,
    pub root_merge_offset_tolerance_m: f64,
    pub root_merge_angle_tolerance_rad: f64,
}

impl Default for StaticEquilibriumSolverConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 40,
            maximum_backtracking_steps: 12,
            finite_difference_offset_step_m: 1.0e-3,
            finite_difference_angle_step_rad: 1.0e-5,
            mass_residual_tolerance_tonnes: 1.0e-6,
            moment_residual_tolerance_nm: 1.0,
            offset_variable_scale_m: 1.0,
            angle_variable_scale_rad: 0.01,
            maximum_scaled_step_norm: 1.0,
            minimum_pivot_ratio: 1.0e-10,
            maximum_condition_number: 1.0e12,
            sufficient_decrease_fraction: 1.0e-4,
            root_merge_offset_tolerance_m: 1.0e-5,
            root_merge_angle_tolerance_rad: 1.0e-7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAttitudeEquilibriumRequest {
    pub schema_version: u32,
    /// The embedded evaluation's attitude is the first initial guess.
    pub evaluation: StaticAttitudeEvaluationRequest,
    /// Additional starts are independently solved and recorded. Distinct
    /// converged roots are reported as multiple-roots-observed.
    #[serde(default)]
    pub additional_initial_guesses: Vec<StaticAttitude>,
    pub domain: StaticAttitudeDomain,
    pub solver: StaticEquilibriumSolverConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EquilibriumDisposition {
    ConvergedCandidate,
    MultipleRootsObserved,
    NoConvergedRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverAttemptDisposition {
    Converged,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverStopReason {
    Converged,
    InitialEvaluationRejected,
    MaximumIterations,
    CentralDifferenceOutsideDomain,
    JacobianEvaluationRejected,
    IllConditionedJacobian,
    BacktrackingFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquilibriumIterationRecord {
    pub iteration: u32,
    pub attitude: StaticAttitude,
    pub residuals: StaticAttitudeResiduals,
    pub scaled_residual_norm: f64,
    pub accepted_step_scale: Option<f64>,
    pub estimated_condition_number: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquilibriumSolverAttempt {
    pub initial_attitude: StaticAttitude,
    pub final_attitude: Option<StaticAttitude>,
    pub final_evaluation: Option<StaticAttitudeEvaluationReceipt>,
    pub disposition: SolverAttemptDisposition,
    pub stop_reason: SolverStopReason,
    pub trace: Vec<EquilibriumIterationRecord>,
    pub issue: Option<StaticAttitudeIssue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergedRootCandidate {
    pub attempt_index: usize,
    pub attitude: StaticAttitude,
    pub evaluation_receipt_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticAttitudeEquilibriumReceipt {
    pub receipt_schema_version: u32,
    pub algorithm: String,
    pub request_digest_blake3: String,
    pub disposition: EquilibriumDisposition,
    pub attempts: Vec<EquilibriumSolverAttempt>,
    pub converged_roots: Vec<ConvergedRootCandidate>,
    pub receipt_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct EquilibriumReceiptDigestIdentity<'a> {
    receipt_schema_version: u32,
    algorithm: &'a str,
    request_digest_blake3: &'a str,
    disposition: EquilibriumDisposition,
    attempts: &'a [EquilibriumSolverAttempt],
    converged_roots: &'a [ConvergedRootCandidate],
}

fn issue(code: &str, detail: impl Into<String>) -> StaticAttitudeIssue {
    StaticAttitudeIssue {
        code: code.to_owned(),
        detail: detail.into(),
    }
}

fn finite_attitude(attitude: StaticAttitude) -> bool {
    attitude.waterplane_offset_m.is_finite()
        && attitude.pitch_rad.is_finite()
        && attitude.roll_rad.is_finite()
}

fn in_domain(attitude: StaticAttitude, domain: StaticAttitudeDomain) -> bool {
    finite_attitude(attitude)
        && (domain.minimum_waterplane_offset_m..=domain.maximum_waterplane_offset_m)
            .contains(&attitude.waterplane_offset_m)
        && attitude.pitch_rad.abs() <= domain.maximum_abs_pitch_rad
        && attitude.roll_rad.abs() <= domain.maximum_abs_roll_rad
}

fn validate_config(config: StaticEquilibriumSolverConfig) -> Vec<StaticAttitudeIssue> {
    let mut issues = Vec::new();
    if !(1..=MAX_EQUILIBRIUM_ITERATIONS).contains(&config.maximum_iterations) {
        issues.push(issue(
            "maximum_iterations",
            format!(
                "maximum iterations must be within 1..={MAX_EQUILIBRIUM_ITERATIONS}"
            ),
        ));
    }
    if !(1..=MAX_BACKTRACKING_STEPS).contains(&config.maximum_backtracking_steps) {
        issues.push(issue(
            "maximum_backtracking_steps",
            format!(
                "backtracking steps must be within 1..={MAX_BACKTRACKING_STEPS}"
            ),
        ));
    }
    let positive_finite = [
        config.finite_difference_offset_step_m,
        config.finite_difference_angle_step_rad,
        config.mass_residual_tolerance_tonnes,
        config.moment_residual_tolerance_nm,
        config.offset_variable_scale_m,
        config.angle_variable_scale_rad,
        config.maximum_scaled_step_norm,
        config.minimum_pivot_ratio,
        config.maximum_condition_number,
        config.root_merge_offset_tolerance_m,
        config.root_merge_angle_tolerance_rad,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value > 0.0);
    if !positive_finite {
        issues.push(issue(
            "solver_positive_parameters",
            "solver step sizes, tolerances, scales, bounds and condition limits must be finite and positive",
        ));
    }
    if !config.sufficient_decrease_fraction.is_finite()
        || !(0.0..1.0).contains(&config.sufficient_decrease_fraction)
    {
        issues.push(issue(
            "sufficient_decrease_fraction",
            "sufficient-decrease fraction must be finite and in [0,1)",
        ));
    }
    if config.minimum_pivot_ratio >= 1.0 {
        issues.push(issue(
            "minimum_pivot_ratio",
            "minimum pivot ratio must be below one",
        ));
    }
    if config.maximum_condition_number < 1.0 {
        issues.push(issue(
            "maximum_condition_number",
            "maximum condition number must be at least one",
        ));
    }
    issues
}

fn validate_request(request: &StaticAttitudeEquilibriumRequest) -> Vec<StaticAttitudeIssue> {
    let mut issues = validate_config(request.solver);
    if request.schema_version != STATIC_EQUILIBRIUM_SCHEMA_VERSION {
        issues.push(issue(
            "schema_version",
            format!(
                "unsupported equilibrium request schema {}; expected {}",
                request.schema_version, STATIC_EQUILIBRIUM_SCHEMA_VERSION
            ),
        ));
    }
    let domain = request.domain;
    if !domain.minimum_waterplane_offset_m.is_finite()
        || !domain.maximum_waterplane_offset_m.is_finite()
        || domain.minimum_waterplane_offset_m >= domain.maximum_waterplane_offset_m
        || !domain.maximum_abs_pitch_rad.is_finite()
        || !domain.maximum_abs_roll_rad.is_finite()
        || domain.maximum_abs_pitch_rad <= 0.0
        || domain.maximum_abs_roll_rad <= 0.0
        || domain.maximum_abs_pitch_rad > MAX_STATIC_ATTITUDE_ABS_RAD
        || domain.maximum_abs_roll_rad > MAX_STATIC_ATTITUDE_ABS_RAD
    {
        issues.push(issue(
            "attitude_domain",
            "offset bounds must be finite/increasing and positive pitch/roll bounds must not exceed the A0/A1 attitude domain",
        ));
    }
    if request.additional_initial_guesses.len() + 1 > MAX_EQUILIBRIUM_STARTS {
        issues.push(issue(
            "initial_guess_count",
            format!(
                "at most {MAX_EQUILIBRIUM_STARTS} initial guesses are supported"
            ),
        ));
    }
    let mut guesses = Vec::with_capacity(request.additional_initial_guesses.len() + 1);
    guesses.push(request.evaluation.attitude);
    guesses.extend(request.additional_initial_guesses.iter().copied());
    for (index, guess) in guesses.iter().copied().enumerate() {
        if !in_domain(guess, domain) {
            issues.push(issue(
                "initial_guess_outside_domain",
                format!("initial guess {index} lies outside the declared offset/pitch/roll domain"),
            ));
        }
        if guesses[..index].contains(&guess) {
            issues.push(issue(
                "duplicate_initial_guess",
                format!("initial guess {index} exactly duplicates an earlier seed"),
            ));
        }
    }
    issues
}

fn residual_vector(
    residuals: StaticAttitudeResiduals,
    config: StaticEquilibriumSolverConfig,
) -> [f64; 3] {
    [
        residuals.mass_residual_tonnes / config.mass_residual_tolerance_tonnes,
        residuals.moment_residual_1_nm / config.moment_residual_tolerance_nm,
        residuals.moment_residual_2_nm / config.moment_residual_tolerance_nm,
    ]
}

fn residual_norm(value: [f64; 3]) -> f64 {
    value[0].hypot(value[1]).hypot(value[2])
}

fn is_converged(
    residuals: StaticAttitudeResiduals,
    config: StaticEquilibriumSolverConfig,
) -> bool {
    residuals.mass_residual_tonnes.abs() <= config.mass_residual_tolerance_tonnes
        && residuals.moment_residual_1_nm.abs() <= config.moment_residual_tolerance_nm
        && residuals.moment_residual_2_nm.abs() <= config.moment_residual_tolerance_nm
}

fn set_coordinate(attitude: &mut StaticAttitude, index: usize, value: f64) {
    match index {
        0 => attitude.waterplane_offset_m = value,
        1 => attitude.pitch_rad = value,
        2 => attitude.roll_rad = value,
        _ => unreachable!("three-coordinate attitude vector"),
    }
}

fn coordinate(attitude: StaticAttitude, index: usize) -> f64 {
    match index {
        0 => attitude.waterplane_offset_m,
        1 => attitude.pitch_rad,
        2 => attitude.roll_rad,
        _ => unreachable!("three-coordinate attitude vector"),
    }
}

fn variable_scale(config: StaticEquilibriumSolverConfig, index: usize) -> f64 {
    match index {
        0 => config.offset_variable_scale_m,
        1 | 2 => config.angle_variable_scale_rad,
        _ => unreachable!("three-coordinate attitude vector"),
    }
}

fn evaluate_at(
    template: &StaticAttitudeEvaluationRequest,
    attitude: StaticAttitude,
) -> Result<StaticAttitudeEvaluationReceipt, Vec<StaticAttitudeIssue>> {
    let mut request = template.clone();
    request.attitude = attitude;
    evaluate_static_attitude(&request)
}

fn central_difference_jacobian(
    template: &StaticAttitudeEvaluationRequest,
    attitude: StaticAttitude,
    domain: StaticAttitudeDomain,
    config: StaticEquilibriumSolverConfig,
) -> Result<([[f64; 3]; 3], SolverStopReason), StaticAttitudeIssue> {
    let mut jacobian = [[0.0; 3]; 3];
    for column in 0..3 {
        let step = if column == 0 {
            config.finite_difference_offset_step_m
        } else {
            config.finite_difference_angle_step_rad
        };
        let mut plus = attitude;
        let mut minus = attitude;
        set_coordinate(&mut plus, column, coordinate(attitude, column) + step);
        set_coordinate(&mut minus, column, coordinate(attitude, column) - step);
        if !in_domain(plus, domain) || !in_domain(minus, domain) {
            return Err(issue(
                "central_difference_outside_domain",
                format!("central difference for coordinate {column} leaves the declared solver domain"),
            ));
        }
        let plus_eval = evaluate_at(template, plus).map_err(|issues| {
            issue(
                "jacobian_plus_evaluation",
                format!("positive central-difference evaluation failed: {}", issues[0].detail),
            )
        })?;
        let minus_eval = evaluate_at(template, minus).map_err(|issues| {
            issue(
                "jacobian_minus_evaluation",
                format!("negative central-difference evaluation failed: {}", issues[0].detail),
            )
        })?;
        let plus_residual = residual_vector(plus_eval.residuals, config);
        let minus_residual = residual_vector(minus_eval.residuals, config);
        for row in 0..3 {
            jacobian[row][column] = (plus_residual[row] - minus_residual[row]) / (2.0 * step)
                * variable_scale(config, column);
            if !jacobian[row][column].is_finite() {
                return Err(issue(
                    "jacobian_non_finite",
                    format!("central-difference Jacobian entry [{row},{column}] is non-finite"),
                ));
            }
        }
    }
    Ok((jacobian, SolverStopReason::Converged))
}

/// Pivoted Gauss-Jordan solve with an infinity-norm condition estimate.
fn solve_3x3(
    matrix: [[f64; 3]; 3],
    right_hand_side: [f64; 3],
    minimum_pivot_ratio: f64,
    maximum_condition_number: f64,
) -> Option<([f64; 3], f64)> {
    let mut augmented = [[0.0; 6]; 3];
    let matrix_norm = matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if !matrix_norm.is_finite() || matrix_norm <= f64::EPSILON {
        return None;
    }
    for row in 0..3 {
        for column in 0..3 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][3 + row] = 1.0;
        augmented[row][5] = right_hand_side[row];
    }

    let mut maximum_pivot = 0.0_f64;
    let mut minimum_pivot = f64::INFINITY;
    for pivot_column in 0..3 {
        let pivot_row = (pivot_column..3).max_by(|left, right| {
            augmented[*left][pivot_column]
                .abs()
                .total_cmp(&augmented[*right][pivot_column].abs())
        })?;
        let pivot_value = augmented[pivot_row][pivot_column].abs();
        if !pivot_value.is_finite() || pivot_value <= f64::EPSILON * matrix_norm {
            return None;
        }
        maximum_pivot = maximum_pivot.max(pivot_value);
        minimum_pivot = minimum_pivot.min(pivot_value);
        augmented.swap(pivot_column, pivot_row);

        let divisor = augmented[pivot_column][pivot_column];
        for column in pivot_column..6 {
            augmented[pivot_column][column] /= divisor;
        }
        for row in 0..3 {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[row][pivot_column];
            for column in pivot_column..6 {
                augmented[row][column] -= factor * augmented[pivot_column][column];
            }
        }
    }

    let pivot_ratio = minimum_pivot / maximum_pivot;
    let inverse_norm = (0..3)
        .map(|row| {
            (0..3)
                .map(|column| augmented[row][3 + column].abs())
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let condition_number = matrix_norm * inverse_norm;
    if !pivot_ratio.is_finite()
        || pivot_ratio < minimum_pivot_ratio
        || !condition_number.is_finite()
        || condition_number > maximum_condition_number
    {
        return None;
    }
    let solution = [augmented[0][5], augmented[1][5], augmented[2][5]];
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some((solution, condition_number))
}

fn attempt_one(
    template: &StaticAttitudeEvaluationRequest,
    initial: StaticAttitude,
    domain: StaticAttitudeDomain,
    config: StaticEquilibriumSolverConfig,
) -> EquilibriumSolverAttempt {
    let mut current_attitude = initial;
    let mut current_evaluation = match evaluate_at(template, initial) {
        Ok(receipt) => receipt,
        Err(errors) => {
            return EquilibriumSolverAttempt {
                initial_attitude: initial,
                final_attitude: None,
                final_evaluation: None,
                disposition: SolverAttemptDisposition::Incomplete,
                stop_reason: SolverStopReason::InitialEvaluationRejected,
                trace: Vec::new(),
                issue: errors.into_iter().next(),
            };
        }
    };
    let initial_residual = residual_vector(current_evaluation.residuals, config);
    let mut trace = vec![EquilibriumIterationRecord {
        iteration: 0,
        attitude: current_attitude,
        residuals: current_evaluation.residuals,
        scaled_residual_norm: residual_norm(initial_residual),
        accepted_step_scale: None,
        estimated_condition_number: None,
    }];

    if is_converged(current_evaluation.residuals, config) {
        return EquilibriumSolverAttempt {
            initial_attitude: initial,
            final_attitude: Some(current_attitude),
            final_evaluation: Some(current_evaluation),
            disposition: SolverAttemptDisposition::Converged,
            stop_reason: SolverStopReason::Converged,
            trace,
            issue: None,
        };
    }

    let mut stop_reason = SolverStopReason::MaximumIterations;
    let mut stop_issue = None;
    for iteration in 1..=config.maximum_iterations {
        let (jacobian, _) =
            match central_difference_jacobian(template, current_attitude, domain, config) {
                Ok(value) => value,
                Err(error) => {
                    stop_reason = if error.code == "central_difference_outside_domain" {
                        SolverStopReason::CentralDifferenceOutsideDomain
                    } else {
                        SolverStopReason::JacobianEvaluationRejected
                    };
                    stop_issue = Some(error);
                    break;
                }
            };
        let scaled_residual = residual_vector(current_evaluation.residuals, config);
        let right_hand_side = [
            -scaled_residual[0],
            -scaled_residual[1],
            -scaled_residual[2],
        ];
        let Some((mut scaled_step, condition_number)) = solve_3x3(
            jacobian,
            right_hand_side,
            config.minimum_pivot_ratio,
            config.maximum_condition_number,
        ) else {
            stop_reason = SolverStopReason::IllConditionedJacobian;
            stop_issue = Some(issue(
                "ill_conditioned_jacobian",
                "scaled Newton Jacobian is singular or exceeds the configured condition-number limit",
            ));
            break;
        };

        let step_norm = residual_norm(scaled_step);
        if step_norm > config.maximum_scaled_step_norm {
            let factor = config.maximum_scaled_step_norm / step_norm;
            for component in &mut scaled_step {
                *component *= factor;
            }
        }

        let current_norm = residual_norm(scaled_residual);
        let mut accepted: Option<(StaticAttitude, StaticAttitudeEvaluationReceipt, f64)> = None;
        for backtrack in 0..config.maximum_backtracking_steps {
            let alpha = 0.5_f64.powi(backtrack as i32);
            let mut candidate = current_attitude;
            for coordinate_index in 0..3 {
                let delta = alpha
                    * scaled_step[coordinate_index]
                    * variable_scale(config, coordinate_index);
                set_coordinate(
                    &mut candidate,
                    coordinate_index,
                    coordinate(current_attitude, coordinate_index) + delta,
                );
            }
            if !in_domain(candidate, domain) {
                continue;
            }
            let Ok(candidate_evaluation) = evaluate_at(template, candidate) else {
                continue;
            };
            let candidate_norm =
                residual_norm(residual_vector(candidate_evaluation.residuals, config));
            let sufficient_reduction =
                candidate_norm < current_norm * (1.0 - config.sufficient_decrease_fraction * alpha);
            if sufficient_reduction || is_converged(candidate_evaluation.residuals, config) {
                accepted = Some((candidate, candidate_evaluation, alpha));
                break;
            }
        }

        let Some((next_attitude, next_evaluation, alpha)) = accepted else {
            stop_reason = SolverStopReason::BacktrackingFailed;
            stop_issue = Some(issue(
                "backtracking_failed",
                "no in-domain trial step reduced the scaled force/moment residual",
            ));
            break;
        };

        current_attitude = next_attitude;
        current_evaluation = next_evaluation;
        let norm = residual_norm(residual_vector(current_evaluation.residuals, config));
        trace.push(EquilibriumIterationRecord {
            iteration,
            attitude: current_attitude,
            residuals: current_evaluation.residuals,
            scaled_residual_norm: norm,
            accepted_step_scale: Some(alpha),
            estimated_condition_number: Some(condition_number),
        });
        if is_converged(current_evaluation.residuals, config) {
            stop_reason = SolverStopReason::Converged;
            break;
        }
    }

    let converged = is_converged(current_evaluation.residuals, config);
    EquilibriumSolverAttempt {
        initial_attitude: initial,
        final_attitude: Some(current_attitude),
        final_evaluation: Some(current_evaluation),
        disposition: if converged {
            SolverAttemptDisposition::Converged
        } else {
            SolverAttemptDisposition::Incomplete
        },
        stop_reason,
        trace,
        issue: stop_issue,
    }
}

fn roots_are_distinct(
    left: StaticAttitude,
    right: StaticAttitude,
    config: StaticEquilibriumSolverConfig,
) -> bool {
    (left.waterplane_offset_m - right.waterplane_offset_m).abs()
        > config.root_merge_offset_tolerance_m
        || (left.pitch_rad - right.pitch_rad).abs()
            > config.root_merge_angle_tolerance_rad
        || (left.roll_rad - right.roll_rad).abs()
            > config.root_merge_angle_tolerance_rad
}

pub fn compute_static_equilibrium_receipt_digest(
    receipt: &StaticAttitudeEquilibriumReceipt,
) -> Result<String, serde_json::Error> {
    let identity = EquilibriumReceiptDigestIdentity {
        receipt_schema_version: receipt.receipt_schema_version,
        algorithm: &receipt.algorithm,
        request_digest_blake3: &receipt.request_digest_blake3,
        disposition: receipt.disposition,
        attempts: &receipt.attempts,
        converged_roots: &receipt.converged_roots,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?)
        .to_hex()
        .to_string())
}

/// Solve the static force/moment equilibrium from one or more declared starts.
///
/// A single converged root is reported as a candidate, not as a proof of
/// uniqueness. If distinct roots are found from the supplied starts, the
/// disposition is MultipleRootsObserved and no root is silently selected.
pub fn solve_static_attitude_equilibrium(
    request: &StaticAttitudeEquilibriumRequest,
) -> Result<StaticAttitudeEquilibriumReceipt, Vec<StaticAttitudeIssue>> {
    let issues = validate_request(request);
    if !issues.is_empty() {
        return Err(issues);
    }

    let request_digest = blake3::hash(&serde_json::to_vec(request).map_err(|error| {
        vec![issue(
            "request_digest",
            format!("could not serialize exact equilibrium request: {error}"),
        )]
    })?)
    .to_hex()
    .to_string();

    let mut starts = Vec::with_capacity(request.additional_initial_guesses.len() + 1);
    starts.push(request.evaluation.attitude);
    starts.extend(request.additional_initial_guesses.iter().copied());

    let mut attempts = Vec::with_capacity(starts.len());
    let mut converged_roots = Vec::<ConvergedRootCandidate>::new();
    for initial in starts {
        let attempt = attempt_one(
            &request.evaluation,
            initial,
            request.domain,
            request.solver,
        );
        if attempt.disposition == SolverAttemptDisposition::Converged
            && let (Some(attitude), Some(evaluation)) =
                (attempt.final_attitude, attempt.final_evaluation.as_ref())
            && !converged_roots.iter().any(|existing| {
                !roots_are_distinct(existing.attitude, attitude, request.solver)
            })
        {
            converged_roots.push(ConvergedRootCandidate {
                attempt_index: attempts.len(),
                attitude,
                evaluation_receipt_digest_blake3: evaluation.receipt_digest_blake3.clone(),
            });
        }
        attempts.push(attempt);
    }

    let disposition = match converged_roots.len() {
        0 => EquilibriumDisposition::NoConvergedRoot,
        1 => EquilibriumDisposition::ConvergedCandidate,
        _ => EquilibriumDisposition::MultipleRootsObserved,
    };
    let mut receipt = StaticAttitudeEquilibriumReceipt {
        receipt_schema_version: 1,
        algorithm: STATIC_EQUILIBRIUM_ALGORITHM.into(),
        request_digest_blake3: request_digest,
        disposition,
        attempts,
        converged_roots,
        receipt_digest_blake3: String::new(),
    };
    receipt.receipt_digest_blake3 =
        compute_static_equilibrium_receipt_digest(&receipt).map_err(|error| {
            vec![issue(
                "receipt_digest",
                format!("could not serialize equilibrium receipt identity: {error}"),
            )]
        })?;
    Ok(receipt)
}

pub fn verify_static_attitude_equilibrium_receipt(
    request: &StaticAttitudeEquilibriumRequest,
    receipt: &StaticAttitudeEquilibriumReceipt,
) -> Result<(), Vec<StaticAttitudeIssue>> {
    let expected = solve_static_attitude_equilibrium(request)?;
    if &expected != receipt {
        return Err(vec![issue(
            "equilibrium_receipt_mismatch",
            "receipt does not reproduce the exact bounded equilibrium solve and iteration trace",
        )]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_attitude::BodyPoint3;

    fn box_request() -> StaticAttitudeEquilibriumRequest {
        StaticAttitudeEquilibriumRequest {
            schema_version: 1,
            evaluation: StaticAttitudeEvaluationRequest {
                schema_version: 1,
                source_spec: serde_json::from_str(include_str!(
                    "../../../../docs/research/marine/fixtures/marine-full-envelope-box-001.json"
                ))
                .unwrap(),
                load_case_id: "A2-BOX-CENTRAL-CG".into(),
                total_mass_tonnes: 12_300.0,
                water_density_kg_m3: 1025.0,
                gravity_m_s2: 9.80665,
                center_of_gravity_body_m: BodyPoint3 {
                    x_m: 50.0,
                    y_m: 0.0,
                    z_m: 3.0,
                },
                attitude: StaticAttitude {
                    waterplane_offset_m: 0.0,
                    pitch_rad: 0.0,
                    roll_rad: 0.0,
                },
                geometry_tolerance_m: 1.0e-9,
            },
            additional_initial_guesses: Vec::new(),
            domain: StaticAttitudeDomain {
                minimum_waterplane_offset_m: -2.0,
                maximum_waterplane_offset_m: 2.0,
                maximum_abs_pitch_rad: 0.2,
                maximum_abs_roll_rad: 0.2,
            },
            solver: StaticEquilibriumSolverConfig::default(),
        }
    }

    #[test]
    fn exact_upright_balance_converges_and_binds_full_solver_configuration() {
        let request = box_request();
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::ConvergedCandidate);
        assert_eq!(receipt.converged_roots.len(), 1);
        assert_eq!(
            receipt.attempts[0].stop_reason,
            SolverStopReason::Converged
        );
        assert!((receipt.attempts[0].final_evaluation.as_ref().unwrap().residuals.mass_residual_tonnes).abs() < 1.0e-8);
        verify_static_attitude_equilibrium_receipt(&request, &receipt).unwrap();
        let mut changed_config = request.clone();
        changed_config.solver.maximum_iterations -= 1;
        let changed = solve_static_attitude_equilibrium(&changed_config).unwrap();
        assert_ne!(receipt.request_digest_blake3, changed.request_digest_blake3);
    }

    #[test]
    fn off_waterplane_initial_guess_returns_to_analytic_upright_balance() {
        let mut request = box_request();
        request.evaluation.attitude.waterplane_offset_m = 0.4;
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::ConvergedCandidate);
        let solution = receipt.converged_roots[0].attitude;
        assert!(solution.waterplane_offset_m.abs() < 1.0e-6);
        assert!(solution.pitch_rad.abs() < 1.0e-7);
        assert!(solution.roll_rad.abs() < 1.0e-7);
    }

    #[test]
    fn longitudinal_cg_offset_requires_nonzero_pitch_and_balanced_moments() {
        let mut request = box_request();
        request.evaluation.center_of_gravity_body_m.x_m = 52.0;
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::ConvergedCandidate);
        let solution = receipt.converged_roots[0].attitude;
        assert!(solution.pitch_rad.abs() > 1.0e-6);
        let residuals = receipt.attempts[0]
            .final_evaluation
            .as_ref()
            .unwrap()
            .residuals;
        assert!(residuals.mass_residual_tonnes.abs() <= request.solver.mass_residual_tolerance_tonnes);
        assert!(residuals.moment_residual_1_nm.abs() <= request.solver.moment_residual_tolerance_nm);
        assert!(residuals.moment_residual_2_nm.abs() <= request.solver.moment_residual_tolerance_nm);
    }

    #[test]
    fn roll_cg_offset_requires_nonzero_roll_and_balanced_moments() {
        let mut request = box_request();
        request.evaluation.center_of_gravity_body_m.y_m = 1.0;
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::ConvergedCandidate);
        assert!(receipt.converged_roots[0].attitude.roll_rad.abs() > 1.0e-6);
    }

    #[test]
    fn multiple_start_solutions_are_deduplicated_but_each_attempt_is_retained() {
        let mut request = box_request();
        let mut second = request.evaluation.attitude;
        second.waterplane_offset_m = 0.1;
        request.additional_initial_guesses.push(second);
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::ConvergedCandidate);
        assert_eq!(receipt.attempts.len(), 2);
        assert_eq!(receipt.converged_roots.len(), 1);
    }

    #[test]
    fn out_of_domain_seed_is_rejected_before_any_geometry_evaluation() {
        let mut request = box_request();
        request.evaluation.attitude.waterplane_offset_m = 2.5;
        let errors = solve_static_attitude_equilibrium(&request).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "initial_guess_outside_domain"));
    }

    #[test]
    fn duplicate_initial_seeds_are_rejected() {
        let mut request = box_request();
        request.additional_initial_guesses.push(request.evaluation.attitude);
        let errors = solve_static_attitude_equilibrium(&request).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "duplicate_initial_guess"));
    }

    #[test]
    fn non_convergence_is_a_typed_receipt_not_a_false_success() {
        let mut request = box_request();
        request.evaluation.attitude.waterplane_offset_m = 0.4;
        request.solver.maximum_iterations = 1;
        request.solver.maximum_scaled_step_norm = 1.0e-6;
        let receipt = solve_static_attitude_equilibrium(&request).unwrap();
        assert_eq!(receipt.disposition, EquilibriumDisposition::NoConvergedRoot);
        assert_eq!(receipt.attempts[0].disposition, SolverAttemptDisposition::Incomplete);
    }

    #[test]
    fn digest_rebinding_does_not_make_a_tampered_solver_receipt_valid() {
        let request = box_request();
        let mut receipt = solve_static_attitude_equilibrium(&request).unwrap();
        receipt.attempts[0].trace[0].scaled_residual_norm += 1.0;
        receipt.receipt_digest_blake3 =
            compute_static_equilibrium_receipt_digest(&receipt).unwrap();
        assert!(verify_static_attitude_equilibrium_receipt(&request, &receipt).is_err());
    }

    #[test]
    fn zero_tolerance_and_ill_conditioning_limits_are_rejected() {
        let mut request = box_request();
        request.solver.moment_residual_tolerance_nm = 0.0;
        assert!(solve_static_attitude_equilibrium(&request).is_err());
        let mut request = box_request();
        request.solver.maximum_condition_number = 0.5;
        assert!(solve_static_attitude_equilibrium(&request).is_err());
    }
}
